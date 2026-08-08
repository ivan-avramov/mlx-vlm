"""Fused MSE prefill (Phase 2) — TDD against a full-fp32 reference.

Reference of record (spec §8): full fp32 dequant of the KV ->
mx.fast.scaled_dot_product_attention in fp32. We assert a tight max-abs-diff,
not just parity with the old K-tile loop.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.turboquant import TurboQuantKVCache


def build_cache(B, n_kv, T, D, bits, seed=0):
    """Populate a TurboQuantKVCache with T tokens of small random K/V."""
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, n_kv, T, D)).astype(np.float32) * 0.1)
    v = mx.array(rng.standard_normal((B, n_kv, T, D)).astype(np.float32) * 0.1)
    cache = TurboQuantKVCache(bits=bits, seed=seed)
    ks, vs = cache.update_and_fetch(k, v)
    mx.eval(cache.keys, cache.values)
    return cache, ks, vs


def reference_attention(q, cache, ks, vs, scale, causal):
    """Full fp32 dequant -> mx.fast.scaled_dot_product_attention (fp32)."""
    kd, vd = cache.dequantize(ks, vs)  # fp32 [B,n_kv,T,D]
    B, n_q, L, D = q.shape
    n_kv = kd.shape[1]
    r = n_q // n_kv
    kd = mx.repeat(kd, r, axis=1)
    vd = mx.repeat(vd, r, axis=1)
    mask = "causal" if causal else None
    return mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), kd, vd, scale=scale, mask=mask
    )


def mad(a, b):
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()


def test_reference_matches_bruteforce_numpy():
    B, n_kv, T, D, r = 1, 1, 8, 16, 1
    cache, ks, vs = build_cache(B, n_kv, T, D, bits=3, seed=1)
    q = mx.array(
        np.random.default_rng(2).standard_normal((B, n_kv * r, 1, D)).astype(np.float32)
        * 0.1
    )
    scale = 1.0 / math.sqrt(D)
    out = reference_attention(q, cache, ks, vs, scale, causal=False)
    kd, vd = cache.dequantize(ks, vs)
    qn = np.array(q[0, 0, 0])
    kn = np.array(kd[0, 0])
    vn = np.array(vd[0, 0])
    s = (qn @ kn.T) * scale
    s -= s.max()
    w = np.exp(s)
    w /= w.sum()
    ref = w @ vn
    assert mad(out[0, 0, 0], mx.array(ref)) < 1e-4


def test_fused_prefill_off_by_default():
    # Fused prefill is opt-in (OFF) by default after the 16K validation showed it
    # ties the loop on speed while costing memory; gate must report ineligible.
    cache, ks, vs = build_cache(1, 1, 64, 256, bits=3)
    q = mx.zeros((1, 1, 4, 256))
    assert (
        cache._fused_prefill_eligible(q, cache._unwrap(ks), cache._unwrap(vs)) is False
    )


def test_gate_on_for_mse_supported_dims():
    cache, ks, vs = build_cache(1, 1, 64, 256, bits=3)
    cache._fused_prefill_enabled = True  # opt in (default is off)
    q = mx.zeros((1, 1, 4, 256))
    assert (
        cache._fused_prefill_eligible(q, cache._unwrap(ks), cache._unwrap(vs)) is True
    )


def test_gate_off_via_killswitch():
    cache = TurboQuantKVCache(bits=3, seed=0, fused_prefill=False)
    k = mx.zeros((1, 1, 64, 256))
    v = mx.zeros((1, 1, 64, 256))
    ks, vs = cache.update_and_fetch(k, v)
    q = mx.zeros((1, 1, 4, 256))
    assert (
        cache._fused_prefill_eligible(q, cache._unwrap(ks), cache._unwrap(vs)) is False
    )


def test_mse_prefill_matches_reference_small_noncausal():
    B, n_kv, r, T, D = 1, 2, 6, 512, 256
    cache, ks, vs = build_cache(B, n_kv, T, D, bits=3, seed=3)
    q = mx.array(
        np.random.default_rng(4)
        .standard_normal((B, n_kv * r, 64, D))
        .astype(np.float32)
        * 0.1
    )
    scale = 1.0 / math.sqrt(D)
    ref = reference_attention(q, cache, ks, vs, scale, causal=False)
    out = cache.mse_prefill(
        q, cache._unwrap(ks), cache._unwrap(vs), scale=scale, mask=None
    )
    assert out.shape == (B, n_kv * r, 64, D)
    assert mad(out, ref) < 3e-2


def test_mse_prefill_causal_matches_reference():
    B, n_kv, r, T, D = 1, 2, 6, 1024, 256
    cache, ks, vs = build_cache(B, n_kv, T, D, bits=3, seed=5)
    L = 256
    q = mx.array(
        np.random.default_rng(6).standard_normal((B, n_kv * r, L, D)).astype(np.float32)
        * 0.1
    )
    scale = 1.0 / math.sqrt(D)
    ref = reference_attention(q, cache, ks, vs, scale, causal=True)
    out = cache.mse_prefill(
        q, cache._unwrap(ks), cache._unwrap(vs), scale=scale, mask="causal"
    )
    assert mad(out, ref) < 3e-2


def test_mse_prefill_causal_oddshape():
    B, n_kv, r, T, D = 1, 1, 6, 600, 256
    cache, ks, vs = build_cache(B, n_kv, T, D, bits=3, seed=7)
    L = 128
    q = mx.array(
        np.random.default_rng(8).standard_normal((B, n_kv * r, L, D)).astype(np.float32)
        * 0.1
    )
    scale = 1.0 / math.sqrt(D)
    ref = reference_attention(q, cache, ks, vs, scale, causal=True)
    out = cache.mse_prefill(q, cache._unwrap(ks), cache._unwrap(vs), scale, "causal")
    assert mad(out, ref) < 3e-2


@pytest.mark.parametrize("D", [256, 128, 96])  # 96 exercises RHT padding (non-pow2)
@pytest.mark.parametrize("n_kv,r", [(1, 1), (2, 4), (4, 6)])  # MHA, GQA-4, GQA-6
@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("L,T", [(1, 512), (64, 2048), (256, 9000)])  # span diagonal
def test_mse_prefill_matrix(D, n_kv, r, bits, L, T):
    cache, ks, vs = build_cache(1, n_kv, T, D, bits=bits, seed=D + n_kv + bits + L)
    q = mx.array(
        np.random.default_rng(99)
        .standard_normal((1, n_kv * r, L, D))
        .astype(np.float32)
        * 0.1
    )
    scale = 1.0 / math.sqrt(D)
    ref = reference_attention(q, cache, ks, vs, scale, causal=True)
    out = cache.mse_prefill(q, cache._unwrap(ks), cache._unwrap(vs), scale, "causal")
    assert mad(out, ref) < 3e-2


def test_prefill_attention_uses_fused_for_mse_when_eligible():
    cache, ks, vs = build_cache(1, 1, 256, 256, bits=3, seed=9)
    cache._fused_prefill_enabled = True  # opt in (default is off)
    q = mx.array(
        np.random.default_rng(10).standard_normal((1, 6, 32, 256)).astype(np.float32)
        * 0.1
    )
    out = cache.prefill_attention(
        q, keys_state=ks, values_state=vs, scale=1.0 / 16, mask="causal"
    )
    assert out is not None
    assert out.shape == (1, 6, 32, 256)


def test_prefill_attention_none_when_killswitch_off():
    cache = TurboQuantKVCache(bits=3, seed=0, fused_prefill=False)
    k = mx.zeros((1, 1, 256, 256))
    v = mx.zeros((1, 1, 256, 256))
    ks, vs = cache.update_and_fetch(k, v)
    q = mx.zeros((1, 6, 32, 256))
    assert (
        cache.prefill_attention(
            q, keys_state=ks, values_state=vs, scale=1.0 / 16, mask="causal"
        )
        is None
    )
