"""P400: MTP exactness on each pick's DEPLOYED KV codec, incl. the width-2 verify block (fork-only).

`Qwen3.8-27B-mlx-uniform-4bit`'s registry entry asks TurboQuant 4-bit KV, but the server currently builds AFFINE 4-bit KV
for it (P405: ``--kv-quant-scheme turboquant`` is ignored), so both quantized codecs are covered — TurboQuant (the registry's
intent) and affine (what is served today); `Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed` serves native16. The quantized
exactness fixtures never enabled quantized KV, which hid P400: a length-2 verify block went through ONE batched
SDPA call (on a TurboQuant cache: ``quantized_attention`` with ``L = 2``, the multi-query path), not plain decoding's
singleton calls. Every case asserts the cache codec it ran on and the verifier widths it saw, so a plumbing fallback to native
KV cannot pass. Metal only (the exact verifier kernels).
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from test_mtp_exactness import (
    PICKS,
    PROMPT,
    ForcingSampler,
    _quantized_pair,
    assert_equal_laws,
    deployed_sampling,
    fake_model,
    scripted,
)

from mlx_vlm import turboquant as tq
from mlx_vlm.generate import ar
from mlx_vlm.models import base
from mlx_vlm.models import cache as kv_cache
from mlx_vlm.models.qwen3_5 import speculative_verifier as sv
from mlx_vlm.speculative.cache_state import iter_leaf_caches

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="exact kernels are Metal"
)

TQ4 = dict(kv_bits=4, kv_quant_scheme="turboquant", quantized_kv_start=0)
PRODUCTION_NUMERICS = os.environ.get("MLX_VLM_TEST_TF32") == "1"  # conftest (P403)
AFFINE4 = dict(kv_bits=4, kv_quant_scheme="uniform", quantized_kv_start=0)
# Each pick's KV codecs: `Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed` native16; `Qwen3.8-27B-mlx-uniform-4bit` the
# registry's TurboQuant 4-bit AND the affine 4-bit KV the server actually builds today (P405: --kv-quant-scheme ignored).
CASES = [
    (PICKS[0], "native16", {}),
    (PICKS[1], "turboquant4", TQ4),
    (PICKS[1], "affine4", AFFINE4),
]


@pytest.fixture
def observed(monkeypatch):
    """Verifier block widths, and the query length of every TurboQuant attention call made INSIDE the verifier —
    through any of its entry points (``decode_attention`` for L == 1; ``prefill_attention`` / ``quantized_attention``
    for L > 1)."""
    rec = SimpleNamespace(widths=[], tq_lengths=[], affine_lengths=[])
    depth = [0]
    real_attention = sv.Qwen3_5BatchInvariantForward._attention
    real_affine = base.quantized_scaled_dot_product_attention

    def attention(self, attn, x, *args, **kwargs):
        rec.widths.append(int(x.shape[1]))
        depth[0] += 1
        try:
            return real_attention(self, attn, x, *args, **kwargs)
        finally:
            depth[0] -= 1

    def recording(real):
        def wrapped(self, queries, *args, **kwargs):
            if depth[0]:
                rec.tq_lengths.append(int(queries.shape[2]))
            return real(self, queries, *args, **kwargs)

        return wrapped

    def affine(queries, *args, **kwargs):  # affine-quantized KV (QuantizedKVCache) SDPA
        if depth[0]:
            rec.affine_lengths.append(int(queries.shape[-2]))
        return real_affine(queries, *args, **kwargs)

    monkeypatch.setattr(sv.Qwen3_5BatchInvariantForward, "_attention", attention)
    monkeypatch.setattr(base, "quantized_scaled_dot_product_attention", affine)
    for name in ("decode_attention", "prefill_attention", "quantized_attention"):
        real = getattr(tq._TurboQuantAttentionMixin, name)
        monkeypatch.setattr(tq._TurboQuantAttentionMixin, name, recording(real))
    return rec


def _run(lm, drafter, *, n, kv, block, **kwargs):
    cache = lm.make_cache()
    extra = {}
    if drafter is not None:
        extra = dict(draft_model=drafter, draft_kind="mtp", draft_block_size=block)
    stream = ar.generate_step(
        mx.array([PROMPT], dtype=mx.int32),
        fake_model(lm),
        None,
        None,
        max_tokens=n,
        prompt_cache=cache,
        prefill_step_size=None,
        **kv,
        **extra,
        **kwargs,
    )
    return [int(t) for t, _ in stream], cache


def _assert_codec(cache, codec):
    leaves = list(iter_leaf_caches(cache))
    names = [type(c).__name__ for c in leaves]
    turbo = [c for c in leaves if isinstance(c, tq.TurboQuantKVCache)]
    affine = [c for c in leaves if isinstance(c, kv_cache.QuantizedKVCache)]
    if codec == "turboquant4":
        assert turbo and not affine, names
        assert all(float(c.bits) == 4.0 for c in turbo), [c.bits for c in turbo]
    elif codec == "affine4":
        assert affine and not turbo, names
        assert all(int(c.bits) == 4 for c in affine), [c.bits for c in affine]
    else:
        assert not turbo and not affine, names


def _assert_singleton_quantized_attention(observed, codec):
    """Every quantized-KV attention call INSIDE the verifier is a singleton (plain decoding's call)."""
    lengths = {
        "turboquant4": observed.tq_lengths,
        "affine4": observed.affine_lengths,
    }.get(codec)
    if lengths is not None:
        assert lengths and set(lengths) == {1}, sorted(set(lengths))


@pytest.mark.parametrize("block", [2, 3, 4])
@pytest.mark.parametrize(
    ("pick", "codec", "kv"), CASES, ids=[f"{p}-{c}" for p, c, _ in CASES]
)
def test_e1b_token_identity_on_the_deployed_kv_codec(pick, codec, kv, block, observed):
    """ON emits plain decoding's tokens at every block size, incl. the terminal round with one token of budget (n with
    (n - 1) % block == 1) — the round that served a width-2 block through the batched call.
    """
    params = deployed_sampling(pick)
    lm, drafter = _quantized_pair(pick)
    mismatches = []
    for n in (13, 14, 17):
        for seed in range(3):
            off, _ = _run(lm, None, n=n, kv=kv, block=block, seed=seed, **params)
            scripted(
                drafter,
                lambda p: off[p] if p < len(off) else None,
                high=True,
                vocab=64,
            )
            on, cache = _run(lm, drafter, n=n, kv=kv, block=block, seed=seed, **params)
            _assert_codec(cache, codec)
            if on != off:
                mismatches.append((n, seed, off, on))
    assert 2 in observed.widths, sorted(
        set(observed.widths)
    )  # a width-2 verify block ran
    _assert_singleton_quantized_attention(observed, codec)
    assert mismatches == []


def _law(lm, drafter, k, params, kv, block, path=None, script=None):
    law, rounds = {}, []

    def walk(prefix, mass):
        if len(prefix) == k:
            law[prefix] = mass
            return
        if path is not None:
            path.clear()
            path.update(enumerate(prefix))
        sampler = ForcingSampler(prefix, **params)
        tokens, _ = _run(
            lm, drafter, n=len(prefix) + 1, kv=kv, block=block, sampler=sampler
        )
        assert tokens[: len(prefix)] == list(prefix)
        if script is not None:
            rounds.extend(script.rounds)
        probs = sampler.law[len(prefix)]
        for token in [t for t, p in enumerate(probs.tolist()) if p > 0.0]:
            walk(prefix + (token,), mass * float(probs[token].item()))

    walk((), 1.0)
    return law, rounds


@pytest.mark.parametrize(
    ("pick", "codec", "kv"), CASES, ids=[f"{p}-{c}" for p, c, _ in CASES]
)
def test_e1a_exact_joint_law_on_the_deployed_kv_codec(pick, codec, kv, observed):
    """The exact joint law of the first 3 tokens with a full-accept drafter: every conditional of length >= 1 ends with
    one token of budget, so width-2 verify blocks are asserted to have run."""
    params = deployed_sampling(pick)
    lm, drafter = _quantized_pair(pick)
    law_off, _ = _law(lm, None, 3, params, kv, 3)
    path = {}
    script = scripted(drafter, lambda pos: path.get(pos), high=True, vocab=64)
    law_on, rounds = _law(lm, drafter, 3, params, kv, 3, path=path, script=script)
    assert rounds and 2 in observed.widths, sorted(set(observed.widths))
    _assert_singleton_quantized_attention(observed, codec)
    assert_equal_laws(law_on, law_off, tol=1e-5)


def _prefilled(lm, kv):
    """A cache holding the prompt, built by the production prefill (and its KV quantization)."""
    _, cache = _run(lm, None, n=1, kv=kv, block=3)
    return cache


@pytest.mark.parametrize(
    ("pick", "codec", "kv"), CASES, ids=[f"{p}-{c}" for p, c, _ in CASES]
)
def test_a_width_2_verify_block_equals_two_singleton_calls(pick, codec, kv, observed):
    """Route level: one width-2 ``speculative_verify`` call vs two plain singleton calls on an INDEPENDENT, identically
    prefilled cache — the logits rows and every full-attention layer's KV bitwise."""
    lm, _ = _quantized_pair(pick)
    block_cache, plain_cache = _prefilled(lm, kv), _prefilled(lm, kv)
    _assert_codec(block_cache, codec)
    tokens = mx.array([[5, 9]], dtype=mx.int32)
    verify = lm(tokens, cache=block_cache, speculative_verify=True)
    rows = [
        lm(tokens[:, i : i + 1], cache=plain_cache).logits[:, -1, :] for i in range(2)
    ]
    mx.eval(verify.logits, rows)
    assert 2 in observed.widths, observed.widths
    for i in range(2):
        assert mx.array_equal(verify.logits[:, i, :], rows[i]).item(), f"row {i}"
    _assert_singleton_quantized_attention(observed, codec)
    compared = 0
    for a, b in zip(iter_leaf_caches(block_cache), iter_leaf_caches(plain_cache)):
        if isinstance(a, (tq.TurboQuantKVCache, kv_cache.QuantizedKVCache)) or (
            type(a).__name__ == "KVCache"
        ):
            assert a.offset == b.offset
            fa, fb = tree_flatten(a.state), tree_flatten(b.state)
            assert [k for k, _ in fa] == [k for k, _ in fb]
            for (key, x), (_, y) in zip(fa, fb):
                if isinstance(x, mx.array):
                    assert x.shape == y.shape and mx.array_equal(x, y).item(), key
                    compared += 1
    assert compared > 0


# --------------------------------------------------------------------------- P403: production numerics (TF32 on)
QUANTIZED_EXACTNESS = (  # the deployed kernel path (bf16 affine-quantized); dense float32 fixtures stay TF32-off (P390)
    "test_mtp_exactness.py::test_e1a_exact_joint_law_quantized_hybrid",
    "test_mtp_exactness.py::test_e1b_paired_rng_bitwise_identity_quantized",
    "test_p369_phase2_observe.py::test_quantized_e1c_on_equals_off_rows_states_and_probes",
    "test_p369_phase2_observe.py::test_quantized_e1c_detects_an_old_prefix_byte_flip_at_one_offset",
    "test_p369_phase2_observe.py::test_quantized_e1c_detects_an_in_support_change_of_one_verifier_row",
    "test_p369_phase2_observe.py::test_quantized_e1c_on_the_deployed_kv_codec",
    "test_p369_phase2_observe.py::test_quantized_e1c_detects_a_turboquant_packed_state_change",
    "test_p369_phase2_moe.py::test_v3d_m8_a_served_moe_verify_block_equals_sequential_singletons",  # P369 v13 V3 (d) / M8
)


def test_the_exactness_checks_hold_under_production_numerics():
    """The suite runs with TF32 OFF (conftest); this box serves with it ON (MLX's default), and P400 showed only then
    (TurboQuant's multi-query attention path runs TF32 matmuls the singleton decode path does not). Rerun this module and
    every quantized exactness test in a fresh interpreter under production numerics (P403).
    """
    if PRODUCTION_NUMERICS:
        assert os.environ["MLX_ENABLE_TF32"] == "1"
        pytest.skip("this IS the production-numerics run")
    here = Path(__file__).resolve()
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(here)]
        + [
            str(here.with_name(node.split("::")[0])) + "::" + node.split("::")[1]
            for node in QUANTIZED_EXACTNESS
        ],
        cwd=str(here.parent),
        env=dict(os.environ, MLX_VLM_TEST_TF32="1"),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert out.returncode == 0, out.stdout[-4000:] + out.stderr[-2000:]
