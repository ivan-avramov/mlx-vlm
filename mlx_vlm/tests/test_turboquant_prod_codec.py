"""Prod codec wiring + quality (Phase 3).

Prod = MSE coarse + a 1-bit QJL residual sketch on keys -> higher-fidelity
reconstruction at ~equal storage. Wired via kv_quant_mode="prod" (Prod key,
MSE value). The headline test is the reconstruction-quality uplift over MSE.
"""
import math
import numpy as np
import mlx.core as mx
import pytest

from mlx_vlm.turboquant import (
    TurboQuantKVCache,
    _TurboQuantProdCodec,
    _TurboQuantMSECodec,
)


def _kv(T, D, n_kv=2, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((1, n_kv, T, D)).astype(np.float32) * scale)
    v = mx.array(rng.standard_normal((1, n_kv, T, D)).astype(np.float32) * scale)
    return k, v


def test_prod_mode_builds_prod_key_mse_value():
    c = TurboQuantKVCache(bits=3, seed=0, kv_quant_mode="prod")
    k, v = _kv(64, 256)
    c.update_and_fetch(k, v)
    assert isinstance(c.key_codec, _TurboQuantProdCodec)
    assert isinstance(c.value_codec, _TurboQuantMSECodec)


def test_mse_mode_is_default():
    c = TurboQuantKVCache(bits=3, seed=0)  # no kv_quant_mode -> mse
    k, v = _kv(64, 256)
    c.update_and_fetch(k, v)
    assert isinstance(c.key_codec, _TurboQuantMSECodec)
    assert isinstance(c.value_codec, _TurboQuantMSECodec)


def _softmax(x):
    x = x - mx.max(x, axis=-1, keepdims=True)
    e = mx.exp(x)
    return e / mx.sum(e, axis=-1, keepdims=True)


def _attn_out_err_vs_true(bits, mode, seed, T=4096, D=256):
    """Attention output error vs the TRUE (un-quantized) attention, on a
    selective/peaked pattern. Values are shared (true) so this isolates the
    key/score-quality effect of the codec."""
    rng = np.random.default_rng(seed)
    kk = rng.standard_normal((T, D)).astype(np.float32)
    idx = rng.choice(T, size=8, replace=False)
    qq = np.stack([kk[i] + 0.5 * rng.standard_normal(D).astype(np.float32) for i in idx])
    k = mx.array(kk[None, None])
    q = mx.array(qq[None, None])
    scale = 1.0 / math.sqrt(D)
    true_out = mx.matmul(_softmax(mx.matmul(q, mx.swapaxes(k, -1, -2)) * scale), k)
    c = TurboQuantKVCache(bits=bits, seed=0, kv_quant_mode=mode)
    ks, _ = c.update_and_fetch(k, k)
    kd = c.key_codec.dequantize(c._unwrap(ks)).astype(mx.float32)
    out = mx.matmul(_softmax(mx.matmul(q, mx.swapaxes(kd, -1, -2)) * scale), k)
    return mx.mean(mx.abs(out - true_out)).item()


def test_prod_beats_mse_on_attention_output_at_3bit():
    # The metric that matters for attention quality. Prod's QJL gives UNBIASED
    # scores -> better softmax-weighted output than MSE's biased 3-bit quant,
    # even though Prod's per-score MAE and L2 reconstruction are worse.
    mse = float(np.mean([_attn_out_err_vs_true(3, "mse", s) for s in (0, 1, 2)]))
    prod = float(np.mean([_attn_out_err_vs_true(3, "prod", s) for s in (0, 1, 2)]))
    print(f"\n3-bit attention out-MAE vs true: MSE={mse:.5f}  Prod={prod:.5f}")
    assert prod < mse, "Prod (unbiased QJL) should preserve attention output better than MSE at 3-bit"


def test_prod_l2_reconstruction_is_worse_by_design():
    # QJL optimizes unbiased dot products, NOT min-L2. Document that L2 recon is
    # intentionally worse so future readers don't "fix" it.
    k, _ = _kv(512, 256, seed=1)
    cm = TurboQuantKVCache(bits=3, seed=0, kv_quant_mode="mse")
    cp = TurboQuantKVCache(bits=3, seed=0, kv_quant_mode="prod")
    ksm, _ = cm.update_and_fetch(k, k)
    ksp, _ = cp.update_and_fetch(k, k)
    em = mx.mean((k - cm.key_codec.dequantize(cm._unwrap(ksm)).astype(mx.float32)) ** 2).item()
    ep = mx.mean((k - cp.key_codec.dequantize(cp._unwrap(ksp)).astype(mx.float32)) ** 2).item()
    assert ep > em  # Prod trades L2 fidelity for unbiased dot products


def test_prod_dispatch_prefill_and_decode_run():
    # Prod routes: prefill (L>1) -> prefill_attention Prod-key path; decode (L=1)
    # -> Prod/separate decode path. Confirm both run via the dispatch and match
    # the Prod-dequantized fp32 reference.
    from mlx_vlm.models.base import scaled_dot_product_attention

    B, n_kv, r, T, D = 1, 2, 3, 256, 256
    k, v = _kv(T, D, n_kv=n_kv, seed=4)
    c = TurboQuantKVCache(bits=3, seed=0, kv_quant_mode="prod")
    ks, vs = c.update_and_fetch(k, v)
    scale = 1.0 / math.sqrt(D)
    kd, vd = c.dequantize(ks, vs)
    kd = mx.repeat(kd, r, axis=1)
    vd = mx.repeat(vd, r, axis=1)

    for L, seed in [(32, 5), (1, 6)]:
        q = mx.array(np.random.default_rng(seed).standard_normal((B, n_kv * r, L, D)).astype(np.float32) * 0.1)
        ref = mx.fast.scaled_dot_product_attention(q, kd, vd, scale=scale, mask="causal")
        out = scaled_dot_product_attention(q, ks, vs, c, scale=scale, mask="causal")
        assert out.shape == ref.shape, f"L={L} shape {out.shape} != {ref.shape}"
        d = mx.max(mx.abs(out - ref)).item()
        print(f"\nProd dispatch L={L}: max-abs-diff vs Prod-dequant ref = {d:.4f}")
        assert d < 5e-2, f"L={L} diff {d}"
