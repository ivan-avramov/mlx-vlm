"""Shared helpers for the M72 1.3 TQ4 production-mode tests (``test_tq4_m72_*.py``). No tests live here except the
instrument checks at the bottom (class [invariant]: they hold on today's code and prove the references below are sound).

Spec: ``docs/specs/m72-1.3-tq4-production.md`` v1.1 + "Modes (C319)". Every other file in this family imports from here.

The references are independent of the code under test:
  * accuracy: the cache's own codec ``dequantize`` (unrotated, fp32), chunked over tokens, then exact softmax attention in numpy
    (float32 matmuls, float64 softmax) per (query head, query) over the first ``lengths[j]`` keys; query head h reads KV head
    h // R. A prefill chunk of L queries over T keys is the same thing with ``lengths[j] = T - L + 1 + j`` (causal, aligned to
    the last L keys).
  * floor: today's production path (``scaled_dot_product_attention`` on the state sliced to the length) supplies the error
    floor A1 is defined against: rel error <= 2 * max(production rel error, 2**-12) and ``allclose(atol = rtol = 3e-2)``.
  * bit identity: ``bits_equal`` compares fp16 / bf16 bit patterns (not ``==``: -0 / NaN).
"""

from __future__ import annotations

import contextlib
import hashlib
import math
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_vlm import turboquant as tq  # noqa: E402
from mlx_vlm.generate import ar  # noqa: E402
from mlx_vlm.generate.common import maybe_quantize_kv_cache  # noqa: E402
from mlx_vlm.models.base import scaled_dot_product_attention as sdpa  # noqa: E402
from mlx_vlm.models.qwen3_5.config import (  # noqa: E402
    ModelConfig,
    TextConfig,
    VisionConfig,
)
from mlx_vlm.models.qwen3_5.language import LanguageModel  # noqa: E402

METAL = mx.metal.is_available()
needs_metal = pytest.mark.skipif(
    not METAL, reason="Metal GPU required for the TQ4 fast kernels"
)

NQ, NKV, D = 24, 4, 256
R = NQ // NKV
SCALE = 1.0 / math.sqrt(D)
BF16, FP16 = mx.bfloat16, mx.float16
FLOOR = 2.0**-12
CAP_FULL = 262144  # the pick's capacity (kv_prealloc_tokens = max_kv_cache_size)
HEADS = (0, 6, 12, 18)  # query heads the long-context rows check (one per KV head)
MODES = ("m72_v1", "m72_exact")
ENV_MODE = "MLX_VLM_TQ4_ATTENTION"


# ------------------------------------------------------------------ module handles (fail loudly when the change is absent)
def fast_module():
    import importlib

    return importlib.import_module("mlx_vlm.tq4_fast")


def mode_module():
    import importlib

    return importlib.import_module("mlx_vlm.tq4_attention")


# ------------------------------------------------------------------ data
def normal(shape, seed):
    return mx.random.normal(shape, key=mx.random.key(seed)).astype(BF16)


def queries_for(k, seed):
    return normal((1, NQ, k, D), seed)


def verify_lengths(T, k):
    return [T - k + 1 + j for j in range(k)]


def tq4_cache(T, cap=CAP_FULL, seed=1, chunk=8192):
    """A TQ4 cache (MSE codec, 4 bits keys and values, as served) with offset = T in a ``cap``-token preallocated state.
    Per-token scale e**U(-1.5, 1.5) so norms differ; filled in chunks (no giant fp32 intermediate).
    """
    cache = tq.TurboQuantKVCache(bits=4, max_kv_size=cap, prealloc_tokens=cap)
    done = 0
    while done < T:
        n = min(chunk, T - done)
        ks = mx.exp(
            mx.random.uniform(-1.5, 1.5, (1, NKV, n, 1), key=mx.random.key(seed + done))
        )
        vs = mx.exp(
            mx.random.uniform(
                -1.5, 1.5, (1, NKV, n, 1), key=mx.random.key(seed + 7 + done)
            )
        )
        K = (normal((1, NKV, n, D), seed * 3 + done).astype(mx.float32) * ks).astype(
            BF16
        )
        V = (normal((1, NKV, n, D), seed * 5 + done).astype(mx.float32) * vs).astype(
            BF16
        )
        cache.update_and_fetch(K, V)
        mx.eval(cache.keys, cache.values)
        done += n
    assert cache.offset == T
    assert cache.keys.norms.shape == (1, NKV, max(cap, T))
    assert cache.keys.indices.shape == (1, NKV, max(cap, T), 32)
    return cache


@lru_cache(maxsize=None)
def shared_cache(T, cap=CAP_FULL, seed=1):
    """Module-lifetime caches; tests never mutate them."""
    return tq4_cache(T, cap, seed)


def planted_cache(positions, T, cap, amp=30.0, seed=11):
    """A TQ4 cache holding T tokens in which every position in ``positions`` carries key ``amp * u`` (unit u), so a query
    ``16 * u`` (scale 1/16) attends to those keys with score ~30 vs ~N(0, 1) elsewhere. Returns (cache, u).
    """
    u = np.random.default_rng(1234).standard_normal(D).astype(np.float32)
    u /= np.linalg.norm(u)
    cache = tq.TurboQuantKVCache(bits=4, max_kv_size=cap, prealloc_tokens=cap)
    done, chunk = 0, 8192
    pos = sorted(positions)
    while done < T:
        n = min(chunk, T - done)
        Kn = np.array(normal((1, NKV, n, D), seed + done).astype(mx.float32))
        Vn = np.array(normal((1, NKV, n, D), seed + 1 + done).astype(mx.float32))
        for p in pos:
            if done <= p < done + n:
                Kn[:, :, p - done, :] = amp * u
        cache.update_and_fetch(mx.array(Kn).astype(BF16), mx.array(Vn).astype(BF16))
        mx.eval(cache.keys, cache.values)
        done += n
    return cache, u


# ------------------------------------------------------------------ references
def _dequant_chunks(cache, T, which, chunk=16384):
    state, codec = (
        (cache.keys, cache.key_codec)
        if which == "k"
        else (cache.values, cache.value_codec)
    )
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        x = codec.dequantize(tq._slice_state_range(state, t0, t1)).astype(mx.float32)
        mx.eval(x)
        yield t0, t1, np.array(x)


def ref_attention(cache, q, lengths, heads=None, rows=None):
    """fp32 reference for ``q`` (1, NQ, k, D) (queries j sees keys [0, lengths[j])) restricted to query ``heads`` (default all)
    and query ``rows`` (default all): returns {(head, row): (D,) float32}. Chunked over tokens (260K-token caches).
    """
    heads = list(range(NQ)) if heads is None else list(heads)
    k = q.shape[2]
    rows = list(range(k)) if rows is None else list(rows)
    qn = np.array(q.astype(mx.float32)) * np.float32(SCALE)
    T = max(lengths[j] for j in rows)
    sel = [(h, j) for h in heads for j in rows]
    scores = {}
    for t0, t1, Kd in _dequant_chunks(cache, T, "k"):
        for h, j in sel:
            s = (qn[0, h, j, :] @ Kd[0, h // R, : t1 - t0].T).astype(np.float64)
            scores.setdefault((h, j), []).append(s)
    probs = {}
    for key, parts in scores.items():
        s = np.concatenate(parts)
        L = lengths[key[1]]
        s = s[:L]
        s -= s.max()
        p = np.exp(s)
        probs[key] = p / p.sum()
    out = {key: np.zeros(D, np.float64) for key in sel}
    for t0, t1, Vd in _dequant_chunks(cache, T, "v"):
        for h, j in sel:
            p = probs[(h, j)]
            hi = min(t1, len(p))
            if hi > t0:
                out[(h, j)] += p[t0:hi].astype(np.float32) @ Vd[0, h // R, : hi - t0]
    return {key: v.astype(np.float32) for key, v in out.items()}


def pick(out, heads, rows):
    """Select the checked (head, row) vectors from a kernel output (1, NQ, k, D) as {(h, j): (D,) float32}."""
    o = np.array(out.astype(mx.float32))
    return {(h, j): o[0, h, j, :] for h in heads for j in rows}


def rel_err(got, ref):
    keys = list(ref)
    a = np.stack([got[k] for k in keys])
    b = np.stack([ref[k] for k in keys])
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def judge(got, ref, prod_err):
    keys = list(ref)
    a = np.stack([got[k] for k in keys])
    b = np.stack([ref[k] for k in keys])
    if not np.isfinite(a).all():
        return {"finite": False, "rel_err": math.inf, "a1": False, "prod_err": prod_err}
    err = rel_err(got, ref)
    ok = err <= 2 * max(prod_err, FLOOR) and bool(
        mx.allclose(mx.array(a), mx.array(b), atol=3e-2, rtol=3e-2).item()
    )
    return {"finite": True, "rel_err": err, "a1": ok, "prod_err": prod_err}


def assert_a1(res, what=""):
    assert res["finite"], f"non-finite output {what}"
    assert res[
        "a1"
    ], f"A1 violated {what}: rel_err {res['rel_err']:.4g} > 2*max({res['prod_err']:.4g}, 2^-12) or allclose failed"


def state_proxies(cache):
    ksr, vsr = cache.state
    return (
        tq._QuantizedStateProxy(ksr, cache.offset, NKV),
        tq._QuantizedStateProxy(vsr, cache.offset, NKV),
    )


def singleton(cache, q1, L):
    """Today's TQ4 decode for ONE query (1, NQ, 1, D) at key length L (the state sliced to [0, L), mask None): the output the
    per-query fallback and plain decoding produce, and what D2 must reproduce bit for bit.
    """
    ks, vs = state_proxies(cache)
    return sdpa(
        q1, ks[:, :, :L, :], vs[:, :, :L, :], cache=cache, scale=SCALE, mask=None
    )


def production_block(cache, q, lengths):
    outs = [singleton(cache, q[:, :, j : j + 1, :], L) for j, L in enumerate(lengths)]
    out = mx.concatenate(outs, axis=2)
    mx.eval(out)
    return out


def production_prefill(cache, q, T):
    """Today's prefill for the chunk at positions [T - L, T) of a cache holding exactly T tokens."""
    assert cache.offset == T
    ks, vs = state_proxies(cache)
    out = sdpa(q, ks, vs, cache=cache, scale=SCALE, mask="causal")
    mx.eval(out)
    return out


def bits_equal(a, b) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return bool(
        mx.array_equal(mx.contiguous(a).view(view), mx.contiguous(b).view(view)).item()
    )


def n_bit_mismatch(a, b) -> int:
    view = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return int(
        mx.sum(mx.contiguous(a).view(view) != mx.contiguous(b).view(view)).item()
    )


# ------------------------------------------------------------------ route counters
def counts():
    return dict(fast_module().route_counts())


def delta(before, after):
    keys = set(before) | set(after)
    return {
        k: after.get(k, 0) - before.get(k, 0)
        for k in keys
        if after.get(k, 0) != before.get(k, 0)
    }


@contextlib.contextmanager
def counted():
    box = SimpleNamespace(delta={})
    before = counts()
    try:
        yield box
    finally:
        box.delta = delta(before, counts())


# ------------------------------------------------------------------ the synthetic model (Qwen3.5 family, the pick's attention shape)
def tiny_model(layers=4, hidden=64, seed=0, dtype=BF16, interval=2):
    """Hybrid GatedDeltaNet + full attention, attention shape 24 query heads / 4 KV heads / D = 256, bf16 (inside
    ``tq4_fast.supports``). Layer ``layers - 1`` is full attention: ``maybe_quantize_kv_cache`` never converts it, so it plays
    the role of layer 63 (native16); the earlier full-attention layers are TQ4."""
    mx.random.seed(seed)
    cfg = TextConfig(
        model_type="qwen3_5_text",
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_hidden_layers=layers,
        num_attention_heads=NQ,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=NKV,
        max_position_embeddings=CAP_FULL,
        full_attention_interval=interval,
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
    lm.model.norm.weight = lm.model.norm.weight * 12.0
    lm.set_dtype(dtype)
    mx.eval(lm.parameters())
    return lm


def model_cache(lm, kv="tq4", cap=4096):
    """The served cache list: ``maybe_quantize_kv_cache`` over ``lm.make_cache()`` (TQ4) or the native list."""
    cache = lm.make_cache()
    if kv == "tq4":
        maybe_quantize_kv_cache(
            cache,
            0,
            64,
            4,
            "turboquant",
            max_kv_size=cap,
            kv_prealloc_tokens=cap,
        )
    return cache


def full_attention_caches(cache):
    return [c for c in cache if hasattr(c, "keys") and not isinstance(c, list)]


def tq4_layers(cache):
    return [c for c in cache if isinstance(c, tq.TurboQuantKVCache)]


def prompt_ids(n, seed=3):
    return mx.array([[(i * 7 + seed) % 64 for i in range(n)]], dtype=mx.int32)


def prefill(lm, cache, n, chunk=512, seed=3):
    ids = prompt_ids(n, seed)
    out = None
    for s in range(0, n, chunk):
        out = lm(ids[:, s : s + chunk], cache=cache)
        mx.eval(out.logits)
    return out.logits[0, -1]


def greedy(lm, cache, last_logits, n):
    """Plain (MTP OFF) greedy decode: (tokens, per-step fp32 logits as numpy)."""
    tok = int(mx.argmax(last_logits).item())
    toks, logits = [tok], []
    for _ in range(n):
        out = lm(mx.array([[tok]], dtype=mx.int32), cache=cache)
        lg = out.logits[0, -1]
        mx.eval(lg)
        logits.append(np.array(lg.astype(mx.float32)))
        tok = int(mx.argmax(lg).item())
        toks.append(tok)
    return toks, logits


def digest(tokens, logits):
    h = hashlib.sha256()
    h.update(np.array(tokens, np.int64).tobytes())
    for lg in logits:
        h.update(np.array(lg, np.float32).tobytes())
    return h.hexdigest()


def device_arch() -> str:
    try:
        info = (
            mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        )
        return str(info.get("architecture", ""))
    except Exception:  # noqa: BLE001
        return ""


def fake_model(lm):
    from test_mtp_exactness import fake_model as _fake

    return _fake(lm)


def mtp_run(lm, drafter, *, n, kv, block, prompt, **kwargs):
    """``generate_step`` as the server's cached-session path drives it: (tokens, cache)."""
    cache = lm.make_cache()
    extra = {}
    if drafter is not None:
        extra = dict(draft_model=drafter, draft_kind="mtp", draft_block_size=block)
    stream = ar.generate_step(
        prompt,
        fake_model(lm),
        None,
        None,
        max_tokens=n,
        prompt_cache=cache,
        prefill_step_size=512,
        **kv,
        **extra,
        **kwargs,
    )
    return [int(t) for t, _ in stream], cache


TQ4_KV = dict(
    kv_bits=4,
    kv_quant_scheme="turboquant",
    quantized_kv_start=0,
    max_kv_size=4096,
    kv_prealloc_tokens=4096,
)


# ================================================================== instrument validation (hold on today's code)
@needs_metal
def test_instrument_reference_agrees_with_the_production_path_on_a_decode_block():
    # [invariant] the chunked numpy reference and today's singleton path agree to the floor on a block
    cache = shared_cache(5000, cap=5000)
    q = queries_for(3, 5)
    lengths = verify_lengths(5000, 3)
    ref = ref_attention(cache, q, lengths, heads=HEADS)
    got = pick(production_block(cache, q, lengths), HEADS, range(3))
    assert rel_err(got, ref) < 2**-6


@needs_metal
def test_instrument_reference_agrees_with_the_production_prefill():
    # [invariant] the same for a prefill chunk (causal, aligned to the last L keys)
    cache = shared_cache(3000, cap=3000)
    L = 37
    q = queries_for(L, 6)
    lengths = [3000 - L + 1 + j for j in range(L)]
    ref = ref_attention(cache, q, lengths, heads=HEADS, rows=[0, 5, 36])
    got = pick(production_prefill(cache, q, 3000), HEADS, [0, 5, 36])
    assert rel_err(got, ref) < 2**-6


@needs_metal
def test_instrument_judge_rejects_a_one_token_leak_and_a_one_token_drop_on_planted_keys():
    # [invariant] A1's criterion tells the leak / drop controls from the truth (planted keys at lengths[j] - 1)
    lengths = verify_lengths(2600, 3)
    positions = sorted({L - 1 for L in lengths} | {lengths[0]})
    cache, u = planted_cache(positions, T=2603, cap=2700)
    q = mx.array(np.broadcast_to(16.0 * u, (1, NQ, 3, D)).copy()).astype(BF16)
    ref = ref_attention(cache, q, lengths, heads=HEADS)
    prod_err = rel_err(pick(production_block(cache, q, lengths), HEADS, range(3)), ref)
    assert judge(ref, ref, prod_err)["a1"]
    leak = ref_attention(cache, q, [L + 1 for L in lengths], heads=HEADS)
    drop = ref_attention(cache, q, [L - 1 for L in lengths], heads=HEADS)
    assert not judge(leak, ref, prod_err)["a1"]
    assert not judge(drop, ref, prod_err)["a1"]


@needs_metal
def test_instrument_tiny_model_has_the_supported_shape_and_the_served_cache_mix():
    # [invariant] the synthetic model is inside the scope predicate's shape and maybe_quantize_kv_cache leaves its last
    # (full-attention) layer native, like layer 63
    lm = tiny_model(layers=4)
    cache = model_cache(lm)
    kinds = [type(c).__name__ for c in cache]
    assert kinds == [
        "ArraysCache",
        "TurboQuantKVCache",
        "ArraysCache",
        "KVCache",
    ], kinds
    prefill(lm, cache, 600)
    tq_cache = tq4_layers(cache)[0]
    assert tq_cache.keys.norms.shape[1] == NKV and tq_cache.offset == 600
    assert tq_cache.keys.indices.shape[-1] == 32
    assert cache[-1].keys.dtype == BF16 and cache[-1].keys.shape[1] == NKV


@needs_metal
def test_instrument_greedy_decode_is_deterministic_in_one_process():
    # [invariant] two identical mode-off runs agree bitwise (so a later difference is the change's, not noise)
    runs = []
    for _ in range(2):
        lm = tiny_model()
        cache = model_cache(lm)
        last = prefill(lm, cache, 700)
        toks, logits = greedy(lm, cache, last, 8)
        runs.append(digest(toks, logits))
    assert runs[0] == runs[1]
