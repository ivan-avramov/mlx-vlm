"""M58 AC8: same-instance generation identity on a CPU micro-model with MTP ON.

One seeded prompt runs under the default (`per_query`); the target cache, drafter
cache and RNG are rebuilt on the SAME model instance; the run repeats with
`joint_v1` (AB off, the production path). Emitted tokens, the per-round acceptance
sequence, the hidden states handed to the drafter and the post-rollback cache
state must agree. CPU limits: the CPU attention fallback is not bitwise between
the joint and the per-query call (STEP 1 / Claude S3), so numeric agreement is
asserted at a stated tolerance; control flow (tokens, acceptance, offsets) is
asserted EXACTLY. Bit identity is the GPU gate (self-test, AB, parity replay).
"""

import copy

import mlx.core as mx
import pytest

import mlx_vlm.speculative.mtp as mtp_utils
from mlx_vlm import mtp_verify_scan as mv
from mlx_vlm.models.qwen3_5 import language as qwen_language
from mlx_vlm.speculative.drafters.qwen3_5_mtp import ModelConfig as MTPConfig
from mlx_vlm.speculative.drafters.qwen3_5_mtp import Qwen3_5MTPDraftModel
from test_attention_policy import _attn_modules, _cpu_device, _tiny_lm  # noqa: F401

PROMPT = [3, 5, 7, 9, 11, 13, 2, 4]
BLOCK = 4
STEPS = 40
TOL = dict(atol=1e-4, rtol=1e-4)


def _plan(n_keys, q_heads, kv_heads):
    # a threshold inside the generated range so straddling blocks occur too
    return ("two_pass", 64) if n_keys >= 20 else ("one_pass", 0)


@pytest.fixture
def micro(monkeypatch):
    monkeypatch.setattr(mv, "_device_class", lambda: "applegpu_g17s")
    monkeypatch.setattr(qwen_language, "_qwen3_5_sdpa_vector_plan", _plan)
    monkeypatch.setattr(
        mv, "QUALIFIED_DOMAIN",
        mv.Domain(dtypes=(mx.float32,), head_dim=16, gqa=(2,),
                  device_classes=("applegpu_g17s",)),
    )
    lm = _tiny_lm(dtype=mx.float32, layers=4)
    text_config = copy.deepcopy(lm.config.text_config)
    text_config.mtp_num_hidden_layers = 1
    drafter = Qwen3_5MTPDraftModel(MTPConfig(text_config=text_config, block_size=BLOCK))
    mx.eval(drafter.parameters())
    return lm, drafter


def _reference(lm):
    """The target's own greedy continuation (plain decode, one token per step)."""
    cache = lm.make_cache()
    out = lm(mx.array([PROMPT], dtype=mx.int32), cache=cache)
    token = int(mx.argmax(out.logits[0, -1]).item())
    ref = [token]
    for _ in range(STEPS + BLOCK):
        out = lm(mx.array([[token]], dtype=mx.int32), cache=cache)
        token = int(mx.argmax(out.logits[0, -1]).item())
        ref.append(token)
    return ref


def _oracle(drafter, ref, record):
    """Wrap the real drafter so drafts follow `ref` with deterministic errors:
    acceptance varies per round, and the real drafter's cache still evolves."""
    state = {"pos": 0, "round": 0}
    cls = type(drafter)  # always the class methods: re-wrapping never stacks wrappers
    real_draft = lambda *a, **kw: cls.draft_block(drafter, *a, **kw)  # noqa: E731
    real_accept = lambda *a, **kw: cls.accept_verified_tokens(drafter, *a, **kw)  # noqa: E731

    def draft_block(last_bonus, hidden, cache, block_size, sampler, *a, **kw):
        real_draft(last_bonus, hidden, cache, block_size, sampler, *a, **kw)
        drafts = list(ref[state["pos"] + 1 : state["pos"] + block_size])
        if state["round"] % 3 == 1:
            drafts[-1] = (drafts[-1] + 1) % 60
        elif state["round"] % 3 == 2:
            drafts[0] = (drafts[0] + 1) % 60
        state["round"] += 1
        return mx.array([drafts], dtype=mx.int32)

    def accept(verify_hidden, draft_tokens, accepted, new_tokens, *a, **kw):
        record["hidden"].append(verify_hidden)
        record["accepted"].append(int(accepted))
        state["pos"] += len(new_tokens)
        return real_accept(verify_hidden, draft_tokens, accepted, new_tokens, *a, **kw)

    drafter.draft_block, drafter.accept_verified_tokens = draft_block, accept


def _run(lm, drafter, ref):
    mx.random.seed(7)  # RNG state restored before each run
    record = {"hidden": [], "accepted": []}
    _oracle(drafter, ref, record)
    cache = lm.make_cache()  # target cache rebuilt (drafter cache: reset by the rounds)
    ids = mx.array([PROMPT], dtype=mx.int32)
    out = lm(ids, cache=cache, return_hidden=True, return_shared_kv=True)
    mx.eval(out.logits)
    first = int(mx.argmax(out.logits[0, -1]).item())
    tokens = [first]
    model = type("M", (), {"language_model": lm})()
    for token, _ in mtp_utils._mtp_rounds(
        model, drafter, cache, out.hidden_states[-1], out.shared_kv_states,
        prompt_tokens=ids, first_bonus=first, max_tokens=STEPS,
        sampler=lambda logits: mx.argmax(logits, axis=-1), draft_block_size=BLOCK,
        token_dtype=mx.int32, greedy_sampling=True,
    ):
        tokens.append(token)
    states = []
    for layer in cache:
        arrays = layer.state
        arrays = arrays if isinstance(arrays, (list, tuple)) else [arrays]
        states.append([a for a in arrays if a is not None])
    mx.eval([a for layer in states for a in layer])
    offsets = [int(getattr(layer, "offset", 0)) for layer in cache]
    return tokens, record, states, offsets


class TestAC8GenerationIdentity:
    def test_ac8_same_instance_per_query_then_joint_v1(self, micro):
        lm, drafter = micro
        ref = _reference(lm)
        base = _run(lm, drafter, ref)
        assert any(a > 0 for a in base[1]["accepted"])  # partial acceptance exercised
        assert not all(a > 0 for a in base[1]["accepted"])

        model = type("M", (), {"config": lm.config, "language_model": lm})()
        policy = mv.JointV1Policy()
        mv.apply_to_model(model, policy)
        again = _run(lm, drafter, ref)
        counters = policy.counters()

        # the joint path really ran, and straddling blocks took the per-query path
        assert counters["verify_blocks_joint_v1"] > 0
        assert counters["verify_blocks_straddle"] > 0
        assert counters["verify_fallback_reasons"] == {}

        assert again[0] == base[0]  # emitted tokens, exactly
        assert again[1]["accepted"] == base[1]["accepted"]  # per-round acceptance
        assert again[3] == base[3]  # post-rollback cache offsets
        assert len(again[1]["hidden"]) == len(base[1]["hidden"])
        for got, want in zip(again[1]["hidden"], base[1]["hidden"]):  # to the drafter
            assert got.shape == want.shape
            assert mx.allclose(got, want, **TOL).item()
        for got_layer, want_layer in zip(again[2], base[2]):  # post-rollback state
            assert len(got_layer) == len(want_layer)
            for got, want in zip(got_layer, want_layer):
                assert got.shape == want.shape
                assert mx.allclose(got, want, **TOL).item()

    def test_ac8_reload_control_per_query_twice_is_exactly_identical(self, micro):
        lm, drafter = micro
        ref = _reference(lm)
        one, two = _run(lm, drafter, ref), _run(lm, drafter, ref)
        assert one[0] == two[0] and one[1]["accepted"] == two[1]["accepted"]
        for got, want in zip(two[1]["hidden"], one[1]["hidden"]):
            assert mx.array_equal(got, want).item()
