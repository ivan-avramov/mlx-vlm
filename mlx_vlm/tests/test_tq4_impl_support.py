"""M72 1.3 shared fixtures for the ``test_tq4_impl_*`` files (no tests here).

Independent references: K / V dequantized through the codec (``codec.dequantize`` on the state sliced to T, fp32), then
exact softmax attention in numpy (float32 matmuls, float64 softmax). A1 judge as the prototype specs: rel error <=
2 * max(production rel error, 2^-12) and ``allclose(atol = rtol = 3e-2)``, finite; rel error = max|out - ref| / max|ref|.
"""

from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx
import numpy as np

from mlx_vlm import turboquant as tq
from mlx_vlm.models.base import scaled_dot_product_attention as sdpa

NQ, NKV, D = 24, 4, 256
R = NQ // NKV
SCALE = 1.0 / math.sqrt(D)
BF16, FP16 = mx.bfloat16, mx.float16
FLOOR = 2.0**-12
CAP = 262144  # the production preallocation (kv_prealloc_tokens = max_kv_cache_size)
BIG_T = 260096
HEADS = (0, 6, 12, 18)  # one query head per KV head (A1 rows at T > 40,960)
FILL_CHUNK = 8192


def normal(shape, seed, dtype=BF16):
    return mx.random.normal(shape, key=mx.random.key(seed)).astype(dtype)


def queries(L, seed):
    return normal((1, NQ, L, D), seed)


def fill_cache(T, cap=CAP, seed=1, kind="plain", K=None, V=None):
    """A TQ4 cache (MSE codec, 4 bits per dimension, keys and values) holding T tokens in a ``cap``-token
    preallocated state, written in chunks (the prefill's update pattern). kinds: plain (per-token scale
    e**U(-1.5, 1.5)), heads (V of KV head h scaled 1, 4, 16, 64 over 64)."""
    cache = tq.TurboQuantKVCache(bits=4, max_kv_size=cap, prealloc_tokens=cap)
    for c0 in range(0, T, FILL_CHUNK):
        n = min(FILL_CHUNK, T - c0)
        if K is not None:
            k, v = K[:, :, c0 : c0 + n], V[:, :, c0 : c0 + n]
        else:
            s = seed * 1000 + c0 // FILL_CHUNK
            ks = mx.exp(
                mx.random.uniform(-1.5, 1.5, (1, NKV, n, 1), key=mx.random.key(s + 7))
            )
            vs = mx.exp(
                mx.random.uniform(-1.5, 1.5, (1, NKV, n, 1), key=mx.random.key(s + 8))
            )
            k = normal((1, NKV, n, D), s + 1, mx.float32) * ks
            v = normal((1, NKV, n, D), s + 2, mx.float32) * vs
            if kind == "heads":
                v = normal((1, NKV, n, D), s + 2, mx.float32) * (
                    mx.array([1.0, 4.0, 16.0, 64.0]) / 64.0
                ).reshape(1, NKV, 1, 1)
            k, v = k.astype(BF16), v.astype(BF16)
        cache.update_and_fetch(k, v)
        mx.eval(cache.keys, cache.values)
    assert cache.offset == T
    assert cache.keys.norms.shape == (1, NKV, cap)
    assert cache.keys.indices.shape == (1, NKV, cap, 32)
    return cache


@lru_cache(maxsize=1)
def big():
    """Module-lifetime 260,096-token cache in a 262,144-token buffer; tests never mutate it."""
    return fill_cache(BIG_T, CAP, seed=1)


def dequant_np(cache, T, block=16384):
    """(Kd, Vd) fp32 numpy (1, NKV, T, D): the codec's own dequantize of the state sliced to T."""
    Ks, Vs = [], []
    for t0 in range(0, T, block):
        t1 = min(T, t0 + block)
        ks = tq._slice_state_range(cache.keys, t0, t1)
        vs = tq._slice_state_range(cache.values, t0, t1)
        k = cache.key_codec.dequantize(ks).astype(mx.float32)
        v = cache.value_codec.dequantize(vs).astype(mx.float32)
        mx.eval(k, v)
        Ks.append(np.array(k))
        Vs.append(np.array(v))
    return np.concatenate(Ks, axis=2), np.concatenate(Vs, axis=2)


@lru_cache(maxsize=1)
def big_np():
    return dequant_np(big(), BIG_T)


def np_for(cache, T):
    if cache is big():
        Kd, Vd = big_np()
        return Kd[:, :, :T], Vd[:, :, :T]
    return dequant_np(cache, T)


def ref_prefill(Kd, Vd, q, T, heads=None, align="last"):
    """fp32 causal reference: the query at row i sees keys j <= T - L + i ('last') or j <= i ('first')."""
    qn = np.array(q.astype(mx.float32)) * np.float32(SCALE)
    L = qn.shape[2]
    heads = range(NQ) if heads is None else heads
    out = np.zeros(qn.shape, np.float32)
    for hq in heads:
        h = hq // R
        for i0 in range(0, L, 64):
            i1 = min(L, i0 + 64)
            s = (qn[0, hq, i0:i1, :] @ Kd[0, h, :T].T).astype(np.float64)
            last = (T - L if align == "last" else 0) + np.arange(i0, i1)
            s = np.where(np.arange(T)[None, :] <= last[:, None], s, -np.inf)
            s -= s.max(axis=-1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(axis=-1, keepdims=True)
            out[0, hq, i0:i1, :] = p.astype(np.float32) @ Vd[0, h, :T]
    return out


def ref_lengths(Kd, Vd, q, lengths, heads=None):
    """fp32 reference: query j attends to the first lengths[j] tokens."""
    qn = np.array(q.astype(mx.float32)) * np.float32(SCALE)
    heads = range(NQ) if heads is None else heads
    out = np.zeros(qn.shape, np.float32)
    for hq in heads:
        h = hq // R
        for j, L in enumerate(lengths):
            s = (qn[0, hq, j, :] @ Kd[0, h, :L].T).astype(np.float64)
            s -= s.max()
            p = np.exp(s)
            p /= p.sum()
            out[0, hq, j, :] = p.astype(np.float32) @ Vd[0, h, :L]
    return out


def proxies(cache, T):
    ks, vs = tq._slice_state(cache.keys, T), tq._slice_state(cache.values, T)
    return tq._QuantizedStateProxy(ks, T, NKV), tq._QuantizedStateProxy(vs, T, NKV)


def production_prefill(cache, q, T):
    """Today's TQ4 prefill on the state sliced to T (a cache whose offset were T)."""
    ks, vs = proxies(cache, T)
    out = sdpa(q, ks, vs, cache=cache, scale=SCALE, mask="causal")
    mx.eval(out)
    return out


def singleton(cache, q1, T):
    """Today's TQ4 singleton decode on the state sliced to T (base.py's call)."""
    ks, vs = proxies(cache, T)
    out = sdpa(q1, ks, vs, cache=cache, scale=SCALE, mask=None)
    mx.eval(out)
    return out


def production_lengths(cache, q, lengths):
    out = mx.concatenate(
        [singleton(cache, q[:, :, j : j + 1, :], L) for j, L in enumerate(lengths)],
        axis=2,
    )
    mx.eval(out)
    return out


def sel(x, heads):
    a = np.array(x.astype(mx.float32)) if isinstance(x, mx.array) else x
    return a if heads is None else a[:, list(heads)]


def rel_err(out, ref):
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))


def judge(out, ref, prod, heads=None):
    """A1 on a kernel output against the fp32 reference, floor = the production path's error."""
    o, r, p = sel(out, heads), sel(ref, heads), sel(prod, heads)
    prod_err = rel_err(p, r)
    if not np.isfinite(o).all():
        return {"finite": False, "rel_err": math.inf, "a1": False, "prod_err": prod_err}
    err = rel_err(o, r)
    ok = err <= 2 * max(prod_err, FLOOR) and bool(
        np.allclose(o, r, atol=3e-2, rtol=3e-2)
    )
    return {"finite": True, "rel_err": err, "a1": ok, "prod_err": prod_err}


def assert_a1(res, what=""):
    assert res["finite"], f"non-finite output {what}"
    assert res[
        "a1"
    ], f"A1 violated {what}: rel_err {res['rel_err']:.4g} > 2*max({res['prod_err']:.4g}, 2^-12) or allclose failed"


def bits16(x):
    return mx.contiguous(x).view(mx.uint16)


def bit_equal(a, b) -> bool:
    mx.eval(a, b)
    return a.shape == b.shape and bool(mx.all(bits16(a) == bits16(b)).item())


# ----------------------------------------------------------------------------- synthetic qwen3_5 target (24 / 4 / 256)
def lm24(layers=4, seed=0, hidden=128, logit_scale=1.0):
    """Tiny hybrid qwen3_5 target with the kernels' attention shape (24 query heads, 4 KV heads, D 256), bf16.
    ``full_attention_interval`` 2: layers 1 and 3 are full attention; the last (3) stays native16 under
    ``maybe_quantize_kv_cache`` (its last-layer skip), as layer 63 of the 64-layer pick.
    """
    from mlx_vlm.models.qwen3_5.config import ModelConfig, TextConfig, VisionConfig
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    mx.random.seed(seed)
    cfg = TextConfig(
        model_type="qwen3_5_text",
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_hidden_layers=layers,
        num_attention_heads=NQ,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=NKV,
        max_position_embeddings=1 << 20,
        full_attention_interval=2,
        head_dim=D,
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
    if logit_scale != 1.0:
        lm.model.norm.weight = lm.model.norm.weight * logit_scale
    lm.set_dtype(BF16)
    mx.eval(lm.parameters())
    return lm


def tq4_cache(lm, cap=8192):
    """The served cache: ``make_cache`` + ``maybe_quantize_kv_cache`` (TQ4, last layer native16)."""
    from mlx_vlm.generate.common import maybe_quantize_kv_cache

    cache = lm.make_cache()
    maybe_quantize_kv_cache(
        cache,
        quantized_kv_start=0,
        kv_group_size=64,
        kv_bits=4,
        kv_quant_scheme="turboquant",
        max_kv_size=cap,
        kv_prealloc_tokens=cap,
    )
    return cache


def prompt(n, seed=3):
    rng = np.random.default_rng(seed)
    return mx.array(rng.integers(1, 60, size=(1, n)), dtype=mx.int32)


def prefill(lm, cache, ids, step=512):
    out = None
    for s in range(0, ids.shape[1], step):
        out = lm(ids[:, s : s + step], cache=cache)
        mx.eval(out.logits)
    return out


def cache_arrays(cache):
    arrays = []
    for c in cache:
        st = c.state
        for x in st if isinstance(st, (list, tuple)) else [st]:
            if x is None:
                continue
            if isinstance(x, mx.array):
                arrays.append(x)
            else:  # TurboQuant state NamedTuple
                arrays.extend(a for a in x if isinstance(a, mx.array))
    mx.eval(arrays)
    return arrays


def caches_bit_equal(a, b) -> bool:
    xa, xb = cache_arrays(a), cache_arrays(b)
    if len(xa) != len(xb):
        return False
    for x, y in zip(xa, xb):
        if x.shape != y.shape or x.dtype != y.dtype:
            return False
        if x.size and not bool(mx.array_equal(x, y).item()):
            return False
    return True
