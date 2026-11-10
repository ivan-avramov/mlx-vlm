"""M72 P2' stage 1: GPU tests of ``dev/m72_scratch_prefill.py`` (fused TQ4 dequantize into a per-layer fp16 / bf16 scratch in the
codec's rotated space + MLX's fused SDPA), through its specified interface only: ``dequant_rotated``, ``scratch_prefill_tq4``,
``reference_dequant_rotated``, ``reference_prefill`` (spec ``docs/specs/m72-p2-stage1-scratch.md`` v1.1, criteria A1, A2, A3, A5).

The references are independent of the kernel under test:
  * fidelity: the packed words are unpacked with numpy shifts (dim d at bits [4d, 4d + 4) of a token's words, LSB first),
    the fp32 codebook is gathered in numpy, multiplied by the fp16 norm in fp32 and rounded ONCE to the dtype (MLX ``astype``);
  * accuracy: ``codec.dequantize`` of the state sliced to T (unrotated, fp32), then exact causal softmax attention in numpy
    (float32 matmuls, float64 softmax) with an explicit mask: the query at position T - L + i sees keys j <= T - L + i.
The production path (``scaled_dot_product_attention`` on the sliced state, mask="causal") supplies the error floor A1 is defined
against: rel error <= 2 * max(production rel error, 2**-12) and ``allclose(atol = rtol = 3e-2)``, all finite;
rel error = max|out - ref| / max|ref|.

Every test is classed in its comment: [new] exercises new behaviour, [invariant] preserves an invariant of the codec / the
production path / the instrument, [fix] reproduces a failure the specification fixes, [threat] strict-mode threat the change
touches (memory blow-up, reading past T or past the state, wrong KV head, dtype / rounding drift, NaN at edges).
Caches are at most 40,960 tokens.
"""

from __future__ import annotations

import gc
import importlib.util
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(),
    reason="Metal GPU required for the fused dequantize kernel",
)

from mlx_vlm import turboquant as tq  # noqa: E402
from mlx_vlm.models.base import scaled_dot_product_attention as sdpa  # noqa: E402

DEV = Path(__file__).resolve().parents[1]
NQ, NKV, D = 24, 4, 256
R = NQ // NKV
SCALE = 1.0 / math.sqrt(D)
BF16, FP16 = mx.bfloat16, mx.float16
FLOOR = 2.0**-12
CAP = 40960  # the probe's exactness-sweep cap
SWEEP_T = (1, 511, 512, 4096, 8193, 32768, 40000)
MB = 1 << 20


# ------------------------------------------------------------------ module under test
def load_module(name="m72_scratch_prefill"):
    path = DEV / "m72_scratch_prefill.py"
    assert path.exists(), f"{path} does not exist yet (module under test)"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sp():
    return load_module()


@pytest.fixture()
def spied(monkeypatch):
    """A fresh copy of the module loaded while ``mx.fast.metal_kernel`` is wrapped: every kernel launch records its ``inputs``."""
    calls = []
    real = mx.fast.metal_kernel

    def spy(*a, **kw):
        kernel = real(*a, **kw)

        def launch(*ca, **ckw):
            calls.append(list(ckw["inputs"] if "inputs" in ckw else ca[0]))
            return kernel(*ca, **ckw)

        return launch

    monkeypatch.setattr(mx.fast, "metal_kernel", spy)
    return load_module("m72_scratch_prefill_spied"), calls


# ------------------------------------------------------------------ data
def normal(shape, seed):
    return mx.random.normal(shape, key=mx.random.key(seed))


def queries(L, seed):
    return normal((1, NQ, L, D), seed).astype(BF16)


def fresh(T, cap=CAP, seed=1, kind="plain"):
    """A TQ4 cache (TQ4: MSE codec, 4 bits, keys and values) filled to offset = T in a cap-token preallocated state; synthetic bf16 K / V.
    kinds: plain (per-token scale e**U(-1.5, 1.5)), extreme (e**U(ln 1e-6, ln 1e3): fp16 subnormal to 1.6e4 norms),
    zero3 (tokens 0..2 all zero), heads (V of KV head h scaled by 1, 4, 16, 64 over 64: the planted data of N1).
    """
    lo, hi = (math.log(1e-6), math.log(1e3)) if kind == "extreme" else (-1.5, 1.5)
    ks = mx.exp(mx.random.uniform(lo, hi, (1, NKV, T, 1), key=mx.random.key(seed + 10)))
    vs = mx.exp(mx.random.uniform(lo, hi, (1, NKV, T, 1), key=mx.random.key(seed + 11)))
    K = normal((1, NKV, T, D), seed) * ks
    V = normal((1, NKV, T, D), seed + 1) * vs
    if (
        kind == "heads"
    ):  # outputs stay O(1) (allclose's atol is absolute); the heads differ by up to 64x
        V = normal((1, NKV, T, D), seed + 1) * (
            mx.array([1.0, 4.0, 16.0, 64.0]) / 64.0
        ).reshape(1, NKV, 1, 1)
    if kind == "zero3":
        K = mx.concatenate([mx.zeros((1, NKV, 3, D)), K[:, :, 3:]], axis=2)
        V = mx.concatenate([mx.zeros((1, NKV, 3, D)), V[:, :, 3:]], axis=2)
    cache = tq.TurboQuantKVCache(bits=4, max_kv_size=cap, prealloc_tokens=cap)
    cache.update_and_fetch(K.astype(BF16), V.astype(BF16))
    mx.eval(cache.keys, cache.values)
    assert cache.offset == T
    assert cache.keys.norms.shape == (1, NKV, cap) and cache.keys.indices.shape == (
        1,
        NKV,
        cap,
        32,
    )
    assert cache.values.norms.shape == (1, NKV, cap) and cache.values.indices.shape == (
        1,
        NKV,
        cap,
        32,
    )
    return cache


@lru_cache(maxsize=None)
def shared(T, cap=CAP, seed=1, kind="plain"):
    """Module-lifetime caches; tests never mutate them (mutating tests use ``fresh``)."""
    return fresh(T, cap, seed, kind)


# ------------------------------------------------------------------ independent references
def np_unpack(words):
    """(..., 32) uint32 -> (..., 256): dim d = (word d // 8 >> 4 * (d % 8)) & 15."""
    shifts = (4 * np.arange(8)).astype(np.uint32)
    return ((words[..., :, None] >> shifts) & np.uint32(15)).reshape(
        *words.shape[:-1], D
    )


def np_rot_dequant(state, codec, t0, t1):
    """fp32 product of the fp32 codebook entry and the fp16 norm for tokens [t0, t1): (1, NKV, t1 - t0, D)."""
    cb = np.array(codec.codebook).astype(np.float32)
    norms = np.array(state.norms[:, :, t0:t1]).astype(np.float32)
    idx = np_unpack(np.array(state.indices[:, :, t0:t1, :]))
    return cb[idx] * norms[..., None]


def bits16(x):
    return mx.contiguous(x).view(mx.uint16)


def n_mismatch(out, state, codec, T, dtype, block=4096):
    """Elements of out[:, :, :T] whose 16 bits differ from the independent rounding of the fp32 product (0 = bit-identical)."""
    bad = 0
    for t0 in range(0, T, block):
        t1 = min(T, t0 + block)
        want = mx.array(np_rot_dequant(state, codec, t0, t1)).astype(dtype)
        bad += int(mx.sum(bits16(want) != bits16(out[:, :, t0:t1])).item())
    return bad


def assert_fidelity(cache, T, dtype, got):
    k_rot, v_rot = got
    cap = cache.keys.norms.shape[2]
    for name, x, st, codec in (
        ("K", k_rot, cache.keys, cache.key_codec),
        ("V", v_rot, cache.values, cache.value_codec),
    ):
        assert x.shape == (1, NKV, cap, D) and x.dtype == dtype, (
            name,
            x.shape,
            x.dtype,
        )
        bad = n_mismatch(x, st, codec, T, dtype)
        assert (
            bad == 0
        ), f"{name}_rot: {bad} of {NKV * T * D} elements differ from the fp32 product rounded once to {dtype}"


def unrotated(cache, T):
    ks, vs = tq._slice_state(cache.keys, T), tq._slice_state(cache.values, T)
    Kd, Vd = cache.key_codec.dequantize(ks).astype(
        mx.float32
    ), cache.value_codec.dequantize(vs).astype(mx.float32)
    mx.eval(Kd, Vd)
    return np.array(Kd), np.array(Vd)


@lru_cache(maxsize=2)
def unrotated_shared(T, cap=CAP, seed=1, kind="plain"):
    return unrotated(shared(T, cap, seed, kind), T)


def ref_attention(Kd, Vd, q, T, align="last"):
    """fp32 reference: exact causal softmax; the query at row i sees keys j <= (T - L + i) ('last', correct) or j <= i ('first',
    the misaligned instrument-positive). Query head h reads KV head h // R."""
    qn = np.array(q.astype(mx.float32)) * np.float32(SCALE)
    L = qn.shape[2]
    out = np.zeros(qn.shape, np.float32)
    for h in range(NKV):
        for i0 in range(0, L, 128):
            i1 = min(L, i0 + 128)
            nb = i1 - i0
            s = (
                qn[0, h * R : (h + 1) * R, i0:i1, :].reshape(-1, D) @ Kd[0, h, :T].T
            ).astype(np.float64)
            s = s.reshape(R, nb, T)
            last = (T - L if align == "last" else 0) + np.arange(i0, i1)
            s = np.where(np.arange(T)[None, None, :] <= last[None, :, None], s, -np.inf)
            s -= s.max(axis=-1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(axis=-1, keepdims=True)
            out[0, h * R : (h + 1) * R, i0:i1, :] = (
                p.astype(np.float32).reshape(-1, T) @ Vd[0, h, :T]
            ).reshape(R, nb, D)
    return out


def production(cache, q):
    """Today's TQ4 prefill on the state at T = cache.offset (the cache holds exactly T)."""
    T = cache.offset
    ksr, vsr = cache.state
    ks, vs = tq._QuantizedStateProxy(ksr, T, NKV), tq._QuantizedStateProxy(vsr, T, NKV)
    out = sdpa(q, ks, vs, cache=cache, scale=SCALE, mask="causal")
    mx.eval(out)
    return np.array(out.astype(mx.float32))


def rel_err(out, ref):
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))


def judge(out, ref, prod_err):
    o = np.array(out.astype(mx.float32))
    if not np.isfinite(o).all():
        return {"finite": False, "rel_err": math.inf, "a1": False, "prod_err": prod_err}
    err = rel_err(o, ref)
    ok = err <= 2 * max(prod_err, FLOOR) and bool(
        mx.allclose(mx.array(o), mx.array(ref), atol=3e-2, rtol=3e-2).item()
    )
    return {"finite": True, "rel_err": err, "a1": ok, "prod_err": prod_err}


def assert_a1(res, what=""):
    assert res["finite"], f"non-finite output {what}"
    assert res[
        "a1"
    ], f"A1 violated {what}: rel_err {res['rel_err']:.4g} > 2*max({res['prod_err']:.4g}, 2^-12) or allclose failed"


def a1_check(sp, cache, L, seed=7, dtype=FP16, **kw):
    T = cache.offset
    q = queries(L, seed)
    Kd, Vd = unrotated_shared(T) if cache is shared(T) else unrotated(cache, T)
    ref = ref_attention(Kd, Vd, q, T)
    out = sp.scratch_prefill_tq4(cache, q, SCALE, dtype=dtype, **kw)
    mx.eval(out)
    assert out.shape == (1, NQ, L, D) and out.dtype == q.dtype
    return judge(out, ref, rel_err(production(cache, q), ref)), ref, q


# ================================================================== instrument validation (no module under test)
# [invariant] the numpy unpack / codebook gather / norm product, inverse-rotated, equals the codec's own dequantize: the
# fidelity reference reads the packed layout the spec states (a wrong bit layout here would make every fidelity test lie)
@pytest.mark.parametrize("which", ("keys", "values"))
def test_instrument_numpy_rotated_dequant_matches_the_codec(which):
    cache = shared(511)
    st, codec = (
        (cache.keys, cache.key_codec)
        if which == "keys"
        else (cache.values, cache.value_codec)
    )
    rot = mx.array(np_rot_dequant(st, codec, 0, 511))
    got = codec._rotate_inverse(rot)
    want = codec.dequantize(tq._slice_state(st, 511)).astype(mx.float32)
    mx.eval(got, want)
    assert rel_err(np.array(got), np.array(want)) < 1e-5


# [invariant] the fp32 attention reference agrees with today's production TQ4 prefill to a few bf16 ulps, L = T and L < T
@pytest.mark.parametrize("T,L", [(512, 512), (4096, 512), (4096, 37), (511, 1), (1, 1)])
def test_instrument_production_agrees_with_the_reference(T, L):
    cache = shared(T)
    q = queries(L, 100 + T + L)
    Kd, Vd = unrotated_shared(T)
    ref = ref_attention(Kd, Vd, q, T)
    err = rel_err(production(cache, q), ref)
    assert (
        err < 2**-6
    ), f"production rel error {err:.3g} vs the fp32 reference (T={T}, L={L})"


# [invariant] the reference can tell a misaligned causal mask (queries at positions [0, L)) from the aligned one by a wide
# margin, so the alignment test below can fail
def test_instrument_misaligned_causal_mask_is_visible():
    T, L = 4096, 37
    Kd, Vd = unrotated_shared(T)
    q = queries(L, 5)
    good, bad = ref_attention(Kd, Vd, q, T), ref_attention(Kd, Vd, q, T, align="first")
    assert rel_err(bad, good) > 0.1


# [invariant] the judge itself: a rel-error just inside / outside 2 * 2^-12 on an exact production path
@pytest.mark.parametrize("c,ok", [(1.5, True), (2.6, False)])
def test_instrument_judge_floor(c, ok):
    ref = np.random.default_rng(0).uniform(-0.9, 0.9, (1, NQ, 1, D)).astype(np.float32)
    ref[0, 0, 0, 0] = 1.0
    out = ref.copy()
    out[0, 1, 0, 3] += c * FLOOR
    assert judge(mx.array(out), ref, 0.0)["a1"] is ok


# ================================================================== fidelity (A1 / A3): dequantize bit-identical to the generic-op form
# [new] A1 fidelity: K_rot / V_rot [0, T) bit-identical to the independent fp32-product-rounded-once, fp16 and bf16, shapes
# cap-sized, at the probe's sweep lengths (T not a multiple of any tile for 1, 511, 8193, 40000)
@pytest.mark.parametrize("dtype", (FP16, BF16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("T", SWEEP_T)
def test_dequant_rotated_is_bit_identical_to_the_generic_form(sp, T, dtype):
    cache = shared(T) if T <= 8193 else shared(40000)
    got = sp.dequant_rotated(cache, T, dtype=dtype)
    assert isinstance(got, tuple) and len(got) == 2
    mx.eval(*got)
    assert_fidelity(cache, T, dtype, got=got)


# [new] the default dtype is fp16 (the preflight: a bf16 scratch is outside A1)
def test_dequant_rotated_default_dtype_is_fp16(sp):
    k, v = sp.dequant_rotated(shared(511), 511)
    mx.eval(k, v)
    assert k.dtype == FP16 and v.dtype == FP16


# [new] T < cache.offset: reads exactly [0, T) of a longer state (T at every sweep value on one cache holding 40,000 tokens
# is covered above; here a sub-prefix of a small odd-capacity cache)
@pytest.mark.parametrize("T", (1, 17, 255, 700))
def test_dequant_rotated_prefix_of_a_longer_state(sp, T):
    cache = shared(700, cap=1100)
    got = sp.dequant_rotated(cache, T)
    mx.eval(*got)
    assert_fidelity(cache, T, FP16, got=got)


# [threat] T == cap (a full state) and a capacity that is not a multiple of any tile / thread mapping: no out-of-range access
@pytest.mark.parametrize("T,cap", [(1100, 1100), (1099, 1100), (700, 1100), (37, 37)])
def test_dequant_rotated_capacity_edges(sp, T, cap):
    cache = shared(T, cap=cap, seed=3)
    got = sp.dequant_rotated(cache, T)
    mx.eval(*got)
    assert_fidelity(cache, T, FP16, got=got)


# [threat] dtype / rounding drift: single rounding of the fp32 product also at the ends of the fp16 range (norms from fp16
# subnormal to 1.6e4), where a half * half double rounding or a downcast codebook would differ in some elements
@pytest.mark.parametrize("dtype", (FP16, BF16), ids=("fp16", "bf16"))
def test_dequant_rotated_extreme_norms_single_rounding(sp, dtype):
    cache = shared(2048, cap=2048, seed=4, kind="extreme")
    got = sp.dequant_rotated(cache, 2048, dtype=dtype)
    mx.eval(*got)
    assert_fidelity(cache, 2048, dtype, got=got)


# [threat] every packed bit pattern: random 32-bit words (all 16 index values at every position, top bit set), random fp16
# norms including 0, a subnormal and 65504; wrong shifts / signed shifts / word order show up here
@pytest.mark.parametrize("dtype", (FP16, BF16), ids=("fp16", "bf16"))
def test_dequant_rotated_arbitrary_bit_patterns(sp, dtype):
    T, cap = 600, 777
    cache = fresh(T, cap=cap, seed=6)
    for i, st in enumerate((cache.keys, cache.values)):
        hi = mx.random.randint(
            0, 1 << 16, (1, NKV, cap, 32), key=mx.random.key(50 + i)
        ).astype(mx.uint32)
        lo = mx.random.randint(
            0, 1 << 16, (1, NKV, cap, 32), key=mx.random.key(60 + i)
        ).astype(mx.uint32)
        st.indices[...] = (hi << 16) | lo
        st.indices[:, :, 0, :] = mx.array(0xFFFFFFFF, dtype=mx.uint32)
        st.indices[:, :, 1, :] = mx.array(0, dtype=mx.uint32)
        nrm = (
            mx.random.uniform(0, 100, (1, NKV, cap), key=mx.random.key(70 + i))
        ).astype(mx.float16)
        nrm[:, :, 0] = mx.array(65504.0, dtype=mx.float16)
        nrm[:, :, 1] = mx.array(0.0, dtype=mx.float16)
        nrm[:, :, 2] = mx.array(6e-8, dtype=mx.float16)
        st.norms[...] = nrm
    mx.eval(cache.keys, cache.values)
    got = sp.dequant_rotated(cache, T, dtype=dtype)
    mx.eval(*got)
    assert_fidelity(cache, T, dtype, got=got)


# [threat] the independent reference helpers of the module: reference_dequant_rotated reads exactly [0, T) and equals the
# independent computation (it is P1's deq_rot with an explicit T), fp16 and bf16
@pytest.mark.parametrize("dtype", (FP16, BF16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("T", (1, 255, 700))
def test_reference_dequant_rotated_matches_the_independent_form_and_ignores_past_t(
    sp, T, dtype
):
    cache = fresh(700, cap=1100, seed=8)
    for name, st, codec in (
        ("K", cache.keys, cache.key_codec),
        ("V", cache.values, cache.value_codec),
    ):
        ref = sp.reference_dequant_rotated(st, T, codec, dtype)
        mx.eval(ref)
        assert (
            ref.dtype == dtype
            and ref.shape[:2] == (1, NKV)
            and ref.shape[2] >= T
            and ref.shape[3] == D
        )
        assert n_mismatch(ref, st, codec, T, dtype) == 0, name
    before = np.array(
        bits16(
            sp.reference_dequant_rotated(cache.keys, T, cache.key_codec, dtype)[
                :, :, :T
            ]
        )
    )
    poison_past(cache, T)
    after = np.array(
        bits16(
            sp.reference_dequant_rotated(cache.keys, T, cache.key_codec, dtype)[
                :, :, :T
            ]
        )
    )
    assert np.array_equal(before, after)


# ================================================================== accuracy (A1): scratch_prefill_tq4 vs the fp32 reference
# [new] A1 on the sweep's L = 512 rows (the 512-token prefill chunk), all 24 query heads (every KV head), finite, shape, dtype
@pytest.mark.parametrize("T", (512, 4096, 8193))
def test_a1_chunk_of_512(sp, T):
    res, _, _ = a1_check(sp, shared(T), 512, seed=T)
    assert_a1(res, f"(T={T}, L=512)")


# [new] A1 / A3 edges: L = 1, L = 37 and T = 1, 511, 8193, 40000 (not multiples of any power-of-two mapping width <= 32)
@pytest.mark.parametrize(
    "T,L",
    [
        (1, 1),
        (511, 511),
        (511, 37),
        (511, 1),
        (8193, 37),
        (8193, 1),
        (40000, 37),
        (40000, 1),
    ],
)
def test_a1_a3_edges_small_chunks(sp, T, L):
    res, _, _ = a1_check(sp, shared(T), L, seed=T + L)
    assert_a1(res, f"(T={T}, L={L})")


# [new] A3: a full chunk at the largest sweep length is finite and of the right shape (no reference: a 512 x 40,000 numpy
# softmax would dominate the run time)
def test_a3_full_chunk_at_the_largest_sweep_length_is_finite(sp):
    out = sp.scratch_prefill_tq4(shared(40000), queries(512, 77), SCALE)
    mx.eval(out)
    assert out.shape == (1, NQ, 512, D) and out.dtype == BF16
    assert bool(mx.all(mx.isfinite(out.astype(mx.float32))).item())


# [new] bf16 scratch is an option: runs, finite, right shape / dtype (its accuracy is outside A1 and not asserted)
def test_bf16_option_runs_and_is_finite(sp):
    q = queries(37, 3)
    out = sp.scratch_prefill_tq4(shared(4096), q, SCALE, dtype=BF16)
    mx.eval(out)
    assert out.shape == q.shape and out.dtype == BF16
    assert bool(mx.all(mx.isfinite(out.astype(mx.float32))).item())


# [new] the default is the fp16 scratch: the default call equals the explicit fp16 call bit for bit
def test_default_dtype_equals_explicit_fp16(sp):
    cache, q = shared(4096), queries(37, 4)
    a = sp.scratch_prefill_tq4(cache, q, SCALE)
    b = sp.scratch_prefill_tq4(cache, q, SCALE, dtype=FP16)
    mx.eval(a, b)
    assert bool(mx.all(bits16(a) == bits16(b)).item())


# [new] the output equals what P1's generic path gives (dequantize identical, SDPA deterministic): within bf16 rounding of it
@pytest.mark.parametrize("T,L", [(4096, 512), (4096, 37), (511, 1)])
def test_matches_reference_prefill_closely(sp, T, L):
    cache, q = shared(T), queries(L, 11 + T)
    a = sp.scratch_prefill_tq4(cache, q, SCALE)
    b = sp.reference_prefill(cache, q, SCALE, T, FP16)
    mx.eval(a, b)
    assert a.shape == b.shape and a.dtype == b.dtype
    an, bn = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    assert rel_err(an, bn) < 2**-8


# [diagnostic, spec: bit-identity of the full path is recorded, not gated] equal inputs to a deterministic SDPA give equal bits
@pytest.mark.xfail(
    reason="diagnostic per spec v1.1: full-path bit-identity is recorded, not a criterion",
    strict=False,
)
@pytest.mark.parametrize("T,L", [(4096, 512), (4096, 37), (511, 1)])
def test_full_path_bit_identity_with_reference_prefill_diagnostic(sp, T, L):
    cache, q = shared(T), queries(L, 11 + T)
    a = sp.scratch_prefill_tq4(cache, q, SCALE)
    b = sp.reference_prefill(cache, q, SCALE, T, FP16)
    mx.eval(a, b)
    assert bool(mx.all(bits16(a) == bits16(b)).item())


# [new] reference_prefill with an explicit T < cache.offset reads exactly [0, T): A1 against the fp32 reference over the prefix
def test_reference_prefill_explicit_t_reads_exactly_the_prefix(sp):
    cache = shared(700, cap=1100)
    T, L = 600, 37
    q = queries(L, 12)
    Kd, Vd = unrotated(cache, 600)  # the state sliced to 600 tokens
    ref = ref_attention(Kd, Vd, q, T)
    out = sp.reference_prefill(cache, q, SCALE, T, FP16)
    mx.eval(out)
    assert (
        rel_err(np.array(out.astype(mx.float32)), ref) < 2 * 2**-7
    )  # the P1 fp16 error is ~4e-3 at worst


# ================================================================== causal alignment (A3)
# [new] L < T: query i sees key j iff j <= i + T - L, against an explicit mask (numpy), several (T, L) incl. L = T
@pytest.mark.parametrize(
    "T,L", [(700, 37), (700, 700), (700, 1), (1100, 513), (1100, 256)]
)
def test_causal_alignment_matches_an_explicit_mask(sp, T, L):
    cache = shared(T, cap=1100, seed=21)
    q = queries(L, 30 + L)
    Kd, Vd = unrotated_shared(T, 1100, 21)
    ref = ref_attention(Kd, Vd, q, T)
    out = sp.scratch_prefill_tq4(cache, q, SCALE)
    mx.eval(out)
    res = judge(out, ref, rel_err(production(cache, q), ref))
    assert_a1(res, f"(T={T}, L={L})")
    wrong = ref_attention(Kd, Vd, q, T, align="first")
    if (
        L < T
    ):  # the output must be closer to the aligned reference than to the misaligned one by a wide margin
        assert rel_err(np.array(out.astype(mx.float32)), wrong) > 10 * rel_err(
            np.array(out.astype(mx.float32)), ref
        )


# ================================================================== N1 / wrong KV head
def planted():
    T = 8193
    cache = fresh(T, cap=8448, seed=31, kind="heads")
    Kd, Vd = unrotated(cache, T)
    q = queries(37, 32)
    ref = ref_attention(Kd, Vd, q, T)
    return cache, q, ref, rel_err(production(cache, q), ref)


# [invariant] N1's instrument positive: with the correct KV heads, the planted data passes A1; the heads' V scales differ by
# up to 64x so any head mix-up is visible
def test_n1_instrument_positive_unshifted_passes_a1(sp):
    cache, q, ref, prod = planted()
    out = sp.scratch_prefill_tq4(cache, q, SCALE)
    mx.eval(out)
    assert_a1(judge(out, ref, prod), "(planted, shift 0)")


# [new] A2 / N1: a one-head roll of the KV heads (any direction or distance) must FAIL A1 on the planted data
@pytest.mark.parametrize("shift", (1, -1, 2, 3))
def test_n1_kv_head_shift_fails_a1(sp, shift):
    cache, q, ref, prod = planted()
    out = sp.scratch_prefill_tq4(cache, q, SCALE, _kv_head_shift=shift)
    mx.eval(out)
    res = judge(out, ref, prod)
    assert (
        res["a1"] is False
    ), f"shift {shift} passed A1 (rel_err {res['rel_err']:.3g}): the control cannot detect a wrong KV head"
    assert res["rel_err"] > 0.1


# [new] the control knob defaults to the identity: shift 0 equals the default call bit for bit
def test_kv_head_shift_zero_is_the_default(sp):
    cache, q = shared(4096), queries(37, 5)
    a = sp.scratch_prefill_tq4(cache, q, SCALE)
    b = sp.scratch_prefill_tq4(cache, q, SCALE, _kv_head_shift=0)
    mx.eval(a, b)
    assert bool(mx.all(bits16(a) == bits16(b)).item())


# [threat] the KV head is right in the dequantize itself: head h of K_rot / V_rot equals head h of the independent form even
# when the heads' data differ by 64x (fidelity on the planted cache)
def test_dequant_keeps_heads_apart_on_planted_data(sp):
    cache = fresh(900, cap=1000, seed=33, kind="heads")
    got = sp.dequant_rotated(cache, 900)
    mx.eval(*got)
    assert_fidelity(cache, 900, FP16, got=got)
    v = np.array(got[1][:, :, :900].astype(mx.float32))
    scale_by_head = np.abs(v).mean(axis=(0, 2, 3))
    assert (
        scale_by_head[3] / scale_by_head[0] > 20
    )  # the heads really differ: 64x data, same-order-of-magnitude codebook


# ================================================================== NaN / edges
# [threat] zero vectors (norm 0) in the state: finite scratch and output, bit-identical dequantize
def test_zero_tokens_stay_finite_and_exact(sp):
    cache = shared(64, cap=100, seed=40, kind="zero3")
    got = sp.dequant_rotated(cache, 64)
    mx.eval(*got)
    assert_fidelity(cache, 64, FP16, got=got)
    assert bool(mx.all(mx.isfinite(got[0][:, :, :64].astype(mx.float32))).item())
    for L in (1, 3, 64):
        out = sp.scratch_prefill_tq4(cache, queries(L, 41), SCALE)
        mx.eval(out)
        assert bool(mx.all(mx.isfinite(out.astype(mx.float32))).item()), L


# ================================================================== in place (A5) and reading past T
def poison_past(cache, start):
    """Garbage in the state past ``start``: all-ones words and a large finite fp16 norm, for keys and values."""
    for st in (cache.keys, cache.values):
        st.norms[:, :, start:] = mx.array(60000.0, dtype=mx.float16)
        st.indices[:, :, start:, :] = mx.array(0xFFFFFFFF, dtype=mx.uint32)
    mx.eval(cache.keys, cache.values)


# [new] A5: the kernel's inputs are the cache's full-cap state arrays themselves, passed unsliced: the arrays handed to the
# kernel include cache.{keys,values}.{norms,indices} by identity, and no array input has been sliced to T
@pytest.mark.parametrize("entry", ("dequant_rotated", "scratch_prefill_tq4"))
def test_a5_kernel_inputs_are_the_unsliced_state_arrays(spied, entry):
    sp, calls = spied
    cache = shared(700, cap=1100, seed=5)
    if entry == "dequant_rotated":
        out = sp.dequant_rotated(cache, 300)
        mx.eval(*out)
    else:
        out = sp.scratch_prefill_tq4(cache, queries(37, 6), SCALE)
        mx.eval(out)
    assert (
        calls
    ), "no Metal kernel launch recorded through mx.fast.metal_kernel (the specification asks for one fused kernel)"
    passed = [x for c in calls for x in c]
    for name, arr in (
        ("keys.norms", cache.keys.norms),
        ("keys.indices", cache.keys.indices),
        ("values.norms", cache.values.norms),
        ("values.indices", cache.values.indices),
    ):
        assert any(
            x is arr for x in passed
        ), f"{name} (the state's own array) was not passed to the kernel"
    for x in passed:
        if hasattr(x, "ndim") and x.ndim >= 3:
            assert (
                x.shape[2] == 1100
            ), f"an input of shape {x.shape} is sliced / copied, not the full-cap state"


# [new] A5: one fused launch writes both outputs (K and V in one launch) per dequantize
def test_one_kernel_launch_per_dequantize(spied):
    sp, calls = spied
    out = sp.dequant_rotated(shared(700, cap=1100, seed=5), 700)
    mx.eval(*out)
    assert (
        len(calls) == 1
    ), f"{len(calls)} kernel launches for one dequantize_rotated (spec: one fused kernel)"


# [threat] A5 sentinel: garbage written in the state past T (norm 60000, all-ones words) does not change the [0, T) output of
# the dequantize or of the whole prefill
def test_a5_sentinel_past_t_does_not_change_the_output(sp):
    T = 700
    cache = fresh(T, cap=1100, seed=5)
    q = queries(37, 6)
    k0, v0 = sp.dequant_rotated(cache, T)
    o0 = sp.scratch_prefill_tq4(cache, q, SCALE)
    mx.eval(k0, v0, o0)
    k0n, v0n, o0n = (np.array(bits16(x)) for x in (k0[:, :, :T], v0[:, :, :T], o0))
    poison_past(cache, T)
    k1, v1 = sp.dequant_rotated(cache, T)
    o1 = sp.scratch_prefill_tq4(cache, q, SCALE)
    mx.eval(k1, v1, o1)
    assert np.array_equal(k0n, np.array(bits16(k1[:, :, :T])))
    assert np.array_equal(v0n, np.array(bits16(v1[:, :, :T])))
    assert np.array_equal(o0n, np.array(bits16(o1)))


# [threat] the same for T below the cache's offset: garbage inside the state but past T does not change the [0, T) dequantize
def test_sentinel_between_t_and_offset_does_not_change_the_dequantize(sp):
    cache = fresh(700, cap=1100, seed=5)
    k0, v0 = sp.dequant_rotated(cache, 300)
    mx.eval(k0, v0)
    k0n, v0n = np.array(bits16(k0[:, :, :300])), np.array(bits16(v0[:, :, :300]))
    poison_past(cache, 300)
    k1, v1 = sp.dequant_rotated(cache, 300)
    mx.eval(k1, v1)
    assert np.array_equal(k0n, np.array(bits16(k1[:, :, :300])))
    assert np.array_equal(v0n, np.array(bits16(v1[:, :, :300])))


# [threat] stale scratch contents (a recycled buffer holding another layer's / chunk's data past T) never leak into the output:
# the result for a cache at T < cap after a call on a different cache of the same cap equals the result from a cleared buffer cache
def test_stale_recycled_scratch_does_not_leak(sp):
    a = shared(700, cap=1100, seed=5)
    b = shared(1100, cap=1100, seed=9)  # full, different data, large tokens past 700
    q = queries(37, 6)
    mx.clear_cache()
    first = sp.scratch_prefill_tq4(a, q, SCALE)
    mx.eval(first)
    first_np = np.array(bits16(first))
    del first
    other = sp.scratch_prefill_tq4(b, queries(37, 7), SCALE)
    mx.eval(other)
    del other
    second = sp.scratch_prefill_tq4(a, q, SCALE)
    mx.eval(second)
    assert np.array_equal(first_np, np.array(bits16(second)))


# [invariant] the call does not modify the cache (offset, state bits) nor the queries, and is deterministic
def test_call_does_not_modify_its_inputs_and_is_deterministic(sp):
    cache = fresh(700, cap=1100, seed=5)
    q = queries(37, 6)
    q_before = np.array(q.astype(mx.float32))
    snap = [
        np.array(x) for st in (cache.keys, cache.values) for x in (st.norms, st.indices)
    ]
    a = sp.scratch_prefill_tq4(cache, q, SCALE)
    b = sp.scratch_prefill_tq4(cache, q, SCALE)
    mx.eval(a, b)
    assert cache.offset == 700
    assert np.array_equal(q_before, np.array(q.astype(mx.float32)))
    after = [
        np.array(x) for st in (cache.keys, cache.values) for x in (st.norms, st.indices)
    ]
    assert all(np.array_equal(x, y) for x, y in zip(snap, after))
    assert bool(mx.all(bits16(a) == bits16(b)).item())


# ================================================================== memory (the scratch is one layer's K / V, never kept or duplicated)
def scratch_bytes(cap=CAP, itemsize=2):
    return 2 * NKV * cap * D * itemsize  # K + V, cap-sized


def peak_above(fn):
    """Peak allocation during fn (evaluated inside) above the resident bytes at its start; the result is dropped."""
    gc.collect()
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    out = fn()
    mx.eval(out)
    peak = mx.get_peak_memory() - base
    del out
    return peak


def resident_rotated(cache, T):
    """SDPA-only baseline inputs: the pre-dequantized fp16 copy, resident (counted in the baseline's base)."""
    ks, vs = [], []
    for st, codec, out in (
        (cache.keys, cache.key_codec, ks),
        (cache.values, cache.value_codec, vs),
    ):
        parts = [
            mx.array(np_rot_dequant(st, codec, t0, min(T, t0 + 4096))).astype(FP16)
            for t0 in range(0, T, 4096)
        ]
        out.append(mx.contiguous(mx.concatenate(parts, axis=2)))
    mx.eval(ks[0], vs[0])
    return ks[0], vs[0]


def sdpa_only(cache, q, k_res, v_res):
    qr = cache.key_codec.prepare_queries(q).astype(FP16)
    o = mx.fast.scaled_dot_product_attention(
        qr, k_res, v_res, scale=SCALE, mask="causal", force_fused=True
    )
    return cache.value_codec._rotate_inverse(o.astype(mx.float32)).astype(q.dtype)


# [threat] A4-shaped, at 40,000 tokens: one call peaks <= 1.25 x one scratch (K + V, cap-sized, fp16) above the SDPA-only
# baseline (the same forward on a resident pre-dequantized copy): no K / V-sized copy, no fp32 intermediate, no kept scratch
def test_memory_one_call_peaks_within_one_scratch_above_sdpa_only(sp):
    cache = shared(40000)
    q = queries(512, 50)
    k_res, v_res = resident_rotated(cache, 40000)
    for _ in range(2):  # warm: kernel compile, RHT constants
        mx.eval(
            sdpa_only(cache, q, k_res, v_res), sp.scratch_prefill_tq4(cache, q, SCALE)
        )
    base = peak_above(lambda: sdpa_only(cache, q, k_res, v_res))
    cand = peak_above(lambda: sp.scratch_prefill_tq4(cache, q, SCALE))
    sb = scratch_bytes()
    assert (
        cand - base <= 1.25 * sb
    ), f"peak above the SDPA-only baseline: {(cand - base) / MB:.0f} MB > 1.25 x {sb / MB:.0f} MB"
    assert cand - base >= 0.5 * sb, (
        f"peak above the baseline {(cand - base) / MB:.0f} MB is below half a scratch ({sb / MB:.0f} MB): the instrument "
        "does not see the scratch"
    )


# [threat] two chained layers evaluated per layer (each call's queries depend on the previous output) peak at one scratch above
# the baseline, not two: the scratch is reused (buffer cache) or freed, never kept
def test_memory_chained_layers_evaluated_per_layer_reuse_one_scratch(sp):
    cache = shared(40000)
    q = queries(512, 51)
    k_res, v_res = resident_rotated(cache, 40000)

    def chain(layer):
        x = q
        for _ in range(3):
            x = layer(x)
            mx.eval(x)
        return x

    cand_layer = lambda x: sp.scratch_prefill_tq4(cache, x, SCALE)  # noqa: E731
    base_layer = lambda x: sdpa_only(cache, x, k_res, v_res)  # noqa: E731
    mx.eval(chain(base_layer), chain(cand_layer))  # warm
    base = peak_above(lambda: chain(base_layer))
    cand = peak_above(lambda: chain(cand_layer))
    sb = scratch_bytes()
    assert (
        cand - base <= 1.25 * sb
    ), f"chained peak above baseline {(cand - base) / MB:.0f} MB > 1.25 x {sb / MB:.0f} MB"


# [threat] nothing is kept after the call: the active bytes return to their starting level (a module-level / cached scratch
# would stay allocated)
def test_memory_active_bytes_return_to_baseline_after_a_call():
    mod = load_module(
        "m72_scratch_prefill_memory"
    )  # a fresh copy: module-level state starts empty, so a kept scratch shows
    cache, small = shared(40000), shared(511, cap=1100)
    q = queries(512, 52)
    mx.eval(q)
    mx.eval(
        mod.scratch_prefill_tq4(small, queries(37, 53), SCALE)
    )  # warm (kernel compile) on a small-capacity cache
    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()
    out = mod.scratch_prefill_tq4(cache, q, SCALE)
    mx.eval(out)
    del out
    gc.collect()
    mx.clear_cache()
    assert (
        mx.get_active_memory() - before <= 2 * MB
    ), f"{(mx.get_active_memory() - before) / MB:.1f} MB still allocated"


# [threat] the dequantize alone allocates the two cap-sized outputs and nothing larger (no fp32 K / V staging)
def test_memory_dequantize_alone_allocates_only_its_outputs(sp):
    cache = shared(40000)
    mx.eval(*sp.dequant_rotated(cache, 40000))  # warm
    peak = peak_above(lambda: sp.dequant_rotated(cache, 40000))
    sb = scratch_bytes()
    assert (
        peak <= 1.05 * sb + 4 * MB
    ), f"dequantize peak {peak / MB:.0f} MB for outputs of {sb / MB:.0f} MB"
    assert peak >= 0.9 * sb


# ================================================================== preconditions (assert)
# [new] 1 <= L <= T <= cache.offset; TQ4 (MSE, 4 bits) codec on both; batch 1
@pytest.mark.parametrize("T", (0, -1, 701))
def test_dequant_rotated_precondition_t_in_range(sp, T):
    with pytest.raises(AssertionError):
        sp.dequant_rotated(shared(700, cap=1100, seed=5), T)


def test_scratch_prefill_precondition_l_at_least_one(sp):
    with pytest.raises(AssertionError):
        sp.scratch_prefill_tq4(
            shared(700, cap=1100, seed=5), mx.zeros((1, NQ, 0, D), dtype=BF16), SCALE
        )


def test_scratch_prefill_precondition_l_at_most_t(sp):
    with pytest.raises(AssertionError):
        sp.scratch_prefill_tq4(shared(100, cap=1100, seed=5), queries(101, 1), SCALE)


def test_scratch_prefill_precondition_batch_one(sp):
    q2 = mx.concatenate([queries(8, 1), queries(8, 2)], axis=0)
    assert q2.shape[0] == 2
    with pytest.raises(AssertionError):
        sp.scratch_prefill_tq4(shared(700, cap=1100, seed=5), q2, SCALE)


# [fix] defect 5 (fix round): an unsupported scratch dtype is an AssertionError (a precondition), not a bare KeyError
@pytest.mark.parametrize(
    "dtype", (mx.float32, mx.int32, mx.uint8), ids=("fp32", "int32", "uint8")
)
def test_dequant_rotated_unsupported_dtype_is_an_assertion_error(sp, dtype):
    with pytest.raises(AssertionError):
        sp.dequant_rotated(shared(511), 511, dtype=dtype)


@pytest.mark.parametrize("dtype", (mx.float32, mx.int32), ids=("fp32", "int32"))
def test_scratch_prefill_unsupported_dtype_is_an_assertion_error(sp, dtype):
    with pytest.raises(AssertionError):
        sp.scratch_prefill_tq4(shared(511), queries(37, 3), SCALE, dtype=dtype)


# [invariant] the supported dtypes are unaffected by the dtype check
@pytest.mark.parametrize("dtype", (FP16, BF16), ids=("fp16", "bf16"))
def test_supported_dtypes_still_dequantize(sp, dtype):
    got = sp.dequant_rotated(shared(511), 511, dtype=dtype)
    mx.eval(*got)
    assert_fidelity(shared(511), 511, dtype, got=got)


def test_precondition_four_bit_codec(sp):
    cache = tq.TurboQuantKVCache(bits=3, max_kv_size=1100, prealloc_tokens=1100)
    cache.update_and_fetch(
        normal((1, NKV, 64, D), 1).astype(BF16), normal((1, NKV, 64, D), 2).astype(BF16)
    )
    mx.eval(cache.keys, cache.values)
    with pytest.raises(AssertionError):
        sp.dequant_rotated(cache, 64)
    with pytest.raises(AssertionError):
        sp.scratch_prefill_tq4(cache, queries(8, 3), SCALE)
