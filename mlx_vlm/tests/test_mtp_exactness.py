"""P369 E1a / E1b: MTP-ON output is distributed exactly as MTP-OFF output.

Fork-only file. Both campaign picks ship ``draft_kind: mtp`` at T > 0, so the claim
under test is distributional identity, not "no detectable shift":

- E1a (exact enumeration): a test sampler reuses ``_PositionedTargetSampler._filter``,
  forces each candidate token by emitted position and records the exact conditional
  mass ``softmax(filtered_logprobs / temperature)`` of every emitted position. Every
  prefix of the finite support is replayed from a clean cache, ON and OFF; the
  conditionals (hence the joint law of the first ``k`` tokens) must agree on every
  sequence. Only the categorical selection is replaced: ``generate_step``, the
  verifier, the acceptance walk and the cache transaction are the production ones.
- E1b (paired RNG): with ``seed=`` the production ``_PositionedTargetSampler`` keys
  each draw by (seed, row, position) on both paths, so ON and OFF must emit
  bitwise-identical token ids at T > 0.

Fixtures: a genuinely HYBRID tiny qwen3_5 target (GatedDeltaNet + full attention,
``ArraysCache`` + ``KVCache``) with a real ``Qwen3_5MTPDraftModel`` head whose
proposals are scripted (high acceptance = the forced path, the M3/M4 worst case;
low acceptance = never the forced token) while its own cache still evolves; dense
float32, and bfloat16 affine 4-bit (uniform pick) / mixed 4+8-bit (mixed pick,
some modules left dense) quantized targets and drafters, sized so the exact affine
verifier kernels are eligible, with spies proving both the ``quantized_verifier``
and the dense ``exact_speculative_verify`` kernels ran. Sampling transforms are
the two picks' deployed ``generation_defaults`` from the pinned certified registry
(P387: the last MTP-ON stack registry; the live registry never changes these tests).
"""

import copy
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from test_mtp_verify_scan import certified_registry

from mlx_vlm.generate import ar
from mlx_vlm.models.qwen3_5.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel
from mlx_vlm.speculative.drafters.qwen3_5_mtp import ModelConfig as MTPConfig
from mlx_vlm.speculative.drafters.qwen3_5_mtp import Qwen3_5MTPDraftModel
from mlx_vlm.speculative.ops import linear as spec_linear

PICKS = (
    "Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed",
    "Qwen3.8-27B-mlx-uniform-4bit",
)
PROMPT = [3, 5, 7, 9, 11, 13, 2, 4]
BLOCK = 3  # two drafts + the verifier bonus per round


def deployed_sampling(name):
    """The pick's deployed temperature / top_p / top_k / min_p (the pinned certified registry)."""
    entry = next(m for m in certified_registry()["models"] if m.get("name") == name)
    assert entry.get("draft_kind") == "mtp", f"{name} no longer ships MTP"
    defaults = entry["generation_defaults"]
    return {key: defaults[key] for key in ("temperature", "top_p", "top_k", "min_p")}


# --------------------------------------------------------------------------- fixtures
def hybrid_lm(
    *,
    hidden=32,
    layers=4,
    vocab=64,
    dtype=mx.float32,
    logit_scale=12.0,
    seed=0,
):
    """Tiny qwen3_5 target: layers alternate GatedDeltaNet and full attention."""
    mx.random.seed(seed)
    heads = max(2, hidden // 64)
    cfg = TextConfig(
        model_type="qwen3_5_text",
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=max(8, hidden // 16),
        linear_value_head_dim=max(8, hidden // 16),
        linear_conv_kernel_dim=4,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        rms_norm_eps=1e-6,
        vocab_size=vocab,
        num_key_value_heads=1,
        max_position_embeddings=512,
        full_attention_interval=2,
        head_dim=16 if hidden <= 64 else 64,
        tie_word_embeddings=False,
    )
    vision = VisionConfig(
        model_type="qwen3_5",
        depth=1,
        hidden_size=8,
        intermediate_size=16,
        out_hidden_size=hidden,
        num_heads=1,
        in_channels=3,
        patch_size=4,
        temporal_patch_size=2,
        spatial_merge_size=1,
        num_position_embeddings=4,
    )
    lm = LanguageModel(
        cfg,
        config=ModelConfig(text_config=cfg, vision_config=vision, model_type="qwen3_5"),
    )
    # Sharpen the head so the deployed top_p keeps a small, enumerable support.
    lm.model.norm.weight = lm.model.norm.weight * logit_scale
    lm.set_dtype(dtype)
    mx.eval(lm.parameters())
    kinds = {type(c).__name__ for c in lm.make_cache()}
    assert kinds == {"ArraysCache", "KVCache"}, kinds  # genuinely hybrid
    return lm


def mtp_drafter(lm, *, block=BLOCK, seed=1):
    text_config = copy.deepcopy(lm.config.text_config)
    text_config.mtp_num_hidden_layers = 1
    mx.random.seed(seed)
    drafter = Qwen3_5MTPDraftModel(MTPConfig(text_config=text_config, block_size=block))
    drafter.set_dtype(lm.model.norm.weight.dtype)
    mx.eval(drafter.parameters())
    return drafter


def quantize_uniform(module, bits=4):
    nn.quantize(module, group_size=64, bits=bits, mode="affine")
    mx.eval(module.parameters())
    return module


def quantize_mixed(module):
    """Mixed affine recipe (8-bit attention / GDN projections, 4-bit elsewhere)
    leaving a few small projections dense, so both exact verifier kernel families
    are on the verify path."""

    def predicate(path, layer):
        if not isinstance(layer, (nn.Linear, nn.Embedding)):
            return False
        if path.endswith(("o_proj", "in_proj_a", "in_proj_b")):
            return False  # stays dense
        if layer.weight.shape[-1] % 64:
            return False
        if "self_attn" in path or "linear_attn" in path:
            return {"group_size": 64, "bits": 8, "mode": "affine"}
        return {"group_size": 64, "bits": 4, "mode": "affine"}

    nn.quantize(module, class_predicate=predicate)
    mx.eval(module.parameters())
    return module


def fake_model(lm):
    """The ``model`` generate_step expects around a bare language model."""

    def get_input_embeddings(input_ids, pixel_values, mask=None, **kwargs):
        return SimpleNamespace(
            inputs_embeds=lm.model.embed_tokens(input_ids), to_dict=lambda: {}
        )

    config = copy.copy(lm.config)
    config.eos_token_id = None
    return SimpleNamespace(
        language_model=lm, get_input_embeddings=get_input_embeddings, config=config
    )


class ScriptedDrafter:
    """Replace the real drafter's proposals while its own cache still evolves.

    ``propose(position) -> token`` gives the draft for an emitted position. The
    wrapper tracks the emitted count from the accepted rounds it is told about.
    """

    def __init__(self, drafter, propose):
        self.drafter = drafter
        self.propose = propose
        self.position = 0
        self.rounds = []
        cls = type(drafter)

        def reset(*args, **kwargs):
            self.position = 1  # the prefill's first token is already emitted
            self.rounds = []
            return cls.reset(drafter, *args, **kwargs)

        def draft_block(last_bonus, hidden, cache, block_size, sampler, *a, **kw):
            cls.draft_block(
                drafter, last_bonus, hidden, cache, block_size, sampler, *a, **kw
            )
            drafts = [self.propose(self.position + j) for j in range(block_size - 1)]
            return mx.array([drafts], dtype=mx.int32)

        def accept(verify_hidden, draft_tokens, accepted, new_tokens, *a, **kw):
            self.rounds.append((int(accepted), int(draft_tokens.shape[1])))
            self.position += len(new_tokens)
            return cls.accept_verified_tokens(
                drafter, verify_hidden, draft_tokens, accepted, new_tokens, *a, **kw
            )

        drafter.reset = reset
        drafter.draft_block = draft_block
        drafter.accept_verified_tokens = accept


def scripted(drafter, path_fn, *, high, vocab):
    def propose(position):
        token = path_fn(position)
        if token is None:
            return 0
        return token if high else (token + 1) % vocab

    return ScriptedDrafter(drafter, propose)


class ForcingSampler(ar._PositionedTargetSampler):
    """E1a: production filter, forced categorical selection, exact mass recorded."""

    def __init__(self, path, **params):
        super().__init__(seed=0, **params)
        self.path = list(path)
        self.law = {}

    def sample_target(self, logprobs, *, row_ids, positions, seeds=None):
        filtered = self._filter(logprobs).astype(mx.float32)
        probs = mx.softmax(filtered * (1.0 / self.temperature), axis=-1)
        mx.eval(probs)
        tokens = []
        for row, position in enumerate(positions):
            # Last write wins: a rejected tail is re-sampled by the next round.
            self.law[position] = probs[row]
            tokens.append(
                self.path[position]
                if position < len(self.path)
                else int(mx.argmax(probs[row]).item())
            )
        return mx.array(tokens, dtype=mx.int32)

    def __call__(self, logprobs):
        raise AssertionError("every target draw must be positioned")


def generate(lm, drafter=None, *, n, prompt=PROMPT, block=BLOCK, **kwargs):
    extra = {}
    if drafter is not None:
        extra = dict(draft_model=drafter, draft_kind="mtp", draft_block_size=block)
    stream = ar.generate_step(
        mx.array([prompt], dtype=mx.int32),
        fake_model(lm),
        None,
        None,
        max_tokens=n,
        prompt_cache=lm.make_cache(),
        prefill_step_size=None,
        **extra,
        **kwargs,
    )
    return [int(token) for token, _ in stream]


# --------------------------------------------------------------------------- E1a
def conditional(lm, drafter, prefix, params):
    sampler = ForcingSampler(prefix, **params)
    tokens = generate(lm, drafter, n=len(prefix) + 1, sampler=sampler)
    assert tokens[: len(prefix)] == list(prefix)
    return sampler.law[len(prefix)]


def enumerate_law(lm, drafter, k, params, support_cap=64):
    """Exact joint law of the first ``k`` tokens: {sequence: probability}."""
    law = {}

    def walk(prefix, mass):
        if len(prefix) == k:
            law[prefix] = mass
            return
        probs = conditional(lm, drafter, prefix, params)
        support = [t for t, p in enumerate(probs.tolist()) if p > 0.0]
        assert 0 < len(support) <= support_cap
        for token in support:
            walk(prefix + (token,), mass * float(probs[token].item()))

    walk((), 1.0)
    return law


def assert_equal_laws(law_on, law_off, tol):
    assert set(law_on) == set(law_off), "ON and OFF supports differ"
    total = sum(law_off.values())
    assert abs(total - 1.0) < 1e-3
    worst = max(abs(law_on[s] - law_off[s]) for s in law_off)
    tv = 0.5 * sum(abs(law_on[s] - law_off[s]) for s in law_off)
    assert worst <= tol, f"max |P_on - P_off| = {worst:.3g} (TV {tv:.3g})"
    return worst, tv


def _assert_rounds(script, high):
    rounds = script.rounds
    assert rounds, "no speculative round ran"
    if high:
        assert any(accepted == drafts for accepted, drafts in rounds), rounds
    else:
        assert all(accepted == 0 for accepted, _ in rounds), rounds


SYNTHETIC_TOPK = {"temperature": 0.5, "top_p": 1.0, "top_k": 2, "min_p": 0.05}


@pytest.mark.parametrize("high", [True, False], ids=["high-accept", "low-accept"])
@pytest.mark.parametrize("pick", PICKS + ("top_k-binding",))
def test_e1a_exact_joint_law_dense_hybrid(pick, high):
    params = SYNTHETIC_TOPK if pick == "top_k-binding" else deployed_sampling(pick)
    k = 5
    lm = hybrid_lm()
    law_off = enumerate_law(lm, None, k, params)

    path = {}
    drafter = mtp_drafter(lm)
    script = scripted(drafter, lambda pos: path.get(pos), high=high, vocab=64)

    law_on = {}
    rounds = []

    def walk(prefix, mass):
        if len(prefix) == k:
            law_on[prefix] = mass
            return
        path.clear()
        path.update(enumerate(prefix))
        probs = conditional(lm, drafter, prefix, params)
        rounds.extend(script.rounds)
        for token in [t for t, p in enumerate(probs.tolist()) if p > 0.0]:
            walk(prefix + (token,), mass * float(probs[token].item()))

    walk((), 1.0)
    _assert_rounds(SimpleNamespace(rounds=rounds), high)
    assert len(law_off) > 4, "support too small to exercise branching"
    assert_equal_laws(law_on, law_off, tol=1e-5)


@pytest.fixture
def verifier_spies(monkeypatch):
    counts = {"quantized": 0, "quantized_fused": 0, "dense": 0}

    def spy(name, key):
        real = getattr(spec_linear, name)

        def wrapped(*args, **kwargs):
            out = real(*args, **kwargs)
            if out is not None:
                counts[key] += 1
            return out

        monkeypatch.setattr(spec_linear, name, wrapped)

    spy("_target_verify_optimized_affine_linear", "quantized")
    spy("_target_verify_quantized_linears", "quantized_fused")
    spy("_target_verify_weight", "dense")
    return counts


def _quantized_pair(pick):
    lm = hybrid_lm(hidden=512, layers=2, dtype=mx.bfloat16, logit_scale=10.0)
    drafter = mtp_drafter(lm)
    if pick == PICKS[0]:  # mixed affine 4/8-bit, some modules dense
        quantize_mixed(lm)
        quantize_mixed(drafter)
    else:  # uniform affine 4-bit
        quantize_uniform(lm, 4)
        quantize_uniform(drafter, 4)
    return lm, drafter


@pytest.mark.skipif(not mx.metal.is_available(), reason="exact kernels are Metal")
@pytest.mark.parametrize("pick", PICKS)
def test_e1a_exact_joint_law_quantized_hybrid(pick, verifier_spies):
    params = deployed_sampling(pick)
    k = 4
    lm, drafter = _quantized_pair(pick)
    law_off = enumerate_law(lm, None, k, params)

    path = {}
    script = scripted(drafter, lambda pos: path.get(pos), high=True, vocab=64)
    law_on = {}
    rounds = []

    def walk(prefix, mass):
        if len(prefix) == k:
            law_on[prefix] = mass
            return
        path.clear()
        path.update(enumerate(prefix))
        probs = conditional(lm, drafter, prefix, params)
        rounds.extend(script.rounds)
        for token in [t for t, p in enumerate(probs.tolist()) if p > 0.0]:
            walk(prefix + (token,), mass * float(probs[token].item()))

    walk((), 1.0)
    _assert_rounds(SimpleNamespace(rounds=rounds), True)
    assert verifier_spies["quantized"] + verifier_spies["quantized_fused"] > 0
    if pick == PICKS[0]:
        assert verifier_spies["dense"] > 0, verifier_spies
    assert_equal_laws(law_on, law_off, tol=1e-5)


# --------------------------------------------------------------------------- E1b
def _paired(lm, drafter_factory, params, seeds, n=24):
    mismatches = []
    for seed in seeds:
        off = generate(lm, None, n=n, seed=seed, **params)
        drafter = drafter_factory(off)
        on = generate(lm, drafter, n=n, seed=seed, **params)
        if on != off:
            mismatches.append((seed, off, on))
    return mismatches


@pytest.mark.parametrize("high", [True, False], ids=["high-accept", "low-accept"])
@pytest.mark.parametrize("pick", PICKS)
def test_e1b_paired_rng_bitwise_identity_dense(pick, high):
    params = deployed_sampling(pick)
    lm = hybrid_lm(logit_scale=4.0)  # top_k = 20 binds at this scale
    drafter = mtp_drafter(lm)
    scripts = []

    def factory(reference):
        script = scripted(
            drafter,
            lambda pos: reference[pos] if pos < len(reference) else None,
            high=high,
            vocab=64,
        )
        scripts.append(script)
        return drafter

    assert _paired(lm, factory, params, seeds=range(12)) == []
    for script in scripts:
        _assert_rounds(script, high)


@pytest.mark.parametrize("pick", PICKS)
def test_e1b_paired_rng_bitwise_identity_real_drafter(pick):
    """The unmodified drafter's own proposals (whatever acceptance they get)."""
    params = deployed_sampling(pick)
    lm = hybrid_lm(logit_scale=4.0)
    drafter = mtp_drafter(lm)
    assert _paired(lm, lambda ref: drafter, params, seeds=range(8)) == []


@pytest.mark.skipif(not mx.metal.is_available(), reason="exact kernels are Metal")
@pytest.mark.parametrize("pick", PICKS)
def test_e1b_paired_rng_bitwise_identity_quantized(pick, verifier_spies):
    params = deployed_sampling(pick)
    lm, drafter = _quantized_pair(pick)
    scripts = []

    def factory(reference):
        script = scripted(
            drafter,
            lambda pos: reference[pos] if pos < len(reference) else None,
            high=True,
            vocab=64,
        )
        scripts.append(script)
        return drafter

    assert _paired(lm, factory, params, seeds=range(6)) == []
    assert any(a == d for s in scripts for a, d in s.rounds)
    assert verifier_spies["quantized"] + verifier_spies["quantized_fused"] > 0
    if pick == PICKS[0]:
        assert verifier_spies["dense"] > 0, verifier_spies


def test_deployed_sampling_reads_both_picks():
    fable, uniform = (deployed_sampling(name) for name in PICKS)
    assert fable == {"temperature": 0.5, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
    assert uniform == {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


def test_forcing_sampler_reuses_the_production_filter():
    """The enumerator's mass is softmax(_filter(logprobs) / T), nothing else."""
    params = deployed_sampling(PICKS[1])
    sampler = ForcingSampler([], **params)
    logits = mx.array([[4.0, 3.0, 2.5, 0.0, -1.0] + [-9.0] * 59])
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    sampler.sample_target(logprobs, row_ids=[0], positions=[0])
    reference = ar._PositionedTargetSampler(seed=0, **params)
    expected = mx.softmax(reference._filter(logprobs) / params["temperature"], -1)
    assert mx.allclose(sampler.law[0], expected[0]).item()
    assert int((sampler.law[0] > 0).sum().item()) < 64  # top_p truncated
