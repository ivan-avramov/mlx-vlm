"""M72 1.3 A1 (kernel accuracy) and R2 on ``mlx_vlm.tq4_fast`` (not ``dev/``), caches preallocated to 262,144 tokens.

- p2s and the matrix-unit verify meet their prototype specs' A1 (rel <= 2 * max(production rel, 2^-12), allclose 3e-2
  vs the fp32 reference) on the prototypes' sweeps (<= 40,960) and at T in {65,537, 131,072, 260,096} (heads 0 / 6 / 12
  / 18; k = 2, 3, 5; L = 512 and 37), with each spec's negative controls failing.
- D2 bit-identical to today's singleton at T in {2,049 ... 260,096} and on a state sliced below the offset; D2
  multi-query per query bit-identical for k = 2..5 at the same T and across the 65,536 tier; no NaN / Inf, incl. fully
  masked blocks.
Long-context rows run in the full fork suite.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Metal GPU required for the TQ4 kernels"
)

from test_tq4_impl_support import (  # noqa: E402
    BIG_T,
    CAP,
    HEADS,
    NKV,
    NQ,
    SCALE,
    D,
    assert_a1,
    big,
    bit_equal,
    fill_cache,
    judge,
    np_for,
    production_lengths,
    production_prefill,
    queries,
    ref_lengths,
    ref_prefill,
    rel_err,
    singleton,
)

from mlx_vlm import tq4_fast as f  # noqa: E402
from mlx_vlm import turboquant as tq  # noqa: E402

LONG_T = (65537, 131072, 260096)


def consecutive(T, k):
    return [T - k + 1 + j for j in range(k)]


# ================================================================== p2s
def _p2s_case(T, L, seed, heads=None):
    cache = big()
    q = queries(L, seed)
    Kd, Vd = np_for(cache, T)
    ref = ref_prefill(Kd, Vd, q, T, heads=heads)
    prod = production_prefill(cache, q, T)
    out = f.prefill_attention(cache, q, SCALE, T)
    mx.eval(out)
    assert out.shape == (1, NQ, L, D) and out.dtype == mx.bfloat16
    return judge(out, ref, prod, heads)


# [new] A1 p2s on the prototype sweep (all query heads), the 512 chunk and the 37 edge chunk
@pytest.mark.parametrize(
    "T,L",
    [
        (1, 1),
        (511, 37),
        (512, 512),
        (4096, 512),
        (4096, 37),
        (8193, 512),
        (8193, 37),
        (32768, 37),
        (40000, 512),
        (40000, 37),
    ],
)
def test_a1_p2s_prototype_sweep(T, L):
    assert_a1(_p2s_case(T, L, seed=T + L), f"(p2s T={T}, L={L})")


# [new] A1 p2s at target context, heads 0 / 6 / 12 / 18
@pytest.mark.parametrize("L", (512, 37))
@pytest.mark.parametrize("T", LONG_T)
def test_a1_p2s_long_context(T, L):
    assert_a1(_p2s_case(T, L, seed=T + L, heads=HEADS), f"(p2s T={T}, L={L})")


# [new] p2s negative control (N1, the p2s spec): a KV-head roll fails A1 on planted per-head data; unshifted passes
@pytest.fixture(scope="module")
def planted_heads():
    T = 8193
    cache = fill_cache(T, CAP, seed=31, kind="heads")
    Kd, Vd = np_for(cache, T)
    q = queries(37, 32)
    ref = ref_prefill(Kd, Vd, q, T)
    prod = production_prefill(cache, q, T)
    return cache, q, ref, prod, T


def test_p2s_n1_instrument_positive(planted_heads):
    cache, q, ref, prod, T = planted_heads
    out = f.prefill_attention(cache, q, SCALE, T)
    assert_a1(judge(out, ref, prod), "(planted, shift 0)")


@pytest.mark.parametrize("shift", (1, -1, 2, 3))
def test_p2s_n1_kv_head_shift_fails_a1(planted_heads, shift):
    cache, q, ref, prod, T = planted_heads
    out = f.prefill_attention(cache, q, SCALE, T, _kv_head_shift=shift)
    res = judge(out, ref, prod)
    assert res["a1"] is False and res["rel_err"] > 0.1, res


# [new] p2s causal alignment: the output is far closer to the last-aligned reference than to a first-aligned one
def test_p2s_causal_alignment():
    cache, T, L = big(), 4096, 37
    q = queries(L, 5)
    Kd, Vd = np_for(cache, T)
    good = ref_prefill(Kd, Vd, q, T)
    bad = ref_prefill(Kd, Vd, q, T, align="first")
    out = np.array(f.prefill_attention(cache, q, SCALE, T).astype(mx.float32))
    assert rel_err(out, bad) > 10 * rel_err(out, good)


# [fix] MLX's fused SDPA reads masked rows past the sliced key length (up to its key-block boundary); a recycled
# scratch buffer holding NaN / Inf there made p2s non-finite (found in 1.3: T = 4,100 failed after other tests ran;
# the prototype's stale-scratch test used finite garbage only). The dequantize zeroes [T, round_up(T, 128)).
def _poison_buffer_cache(value, cap, n=6):
    rows = f.scratch_capacity(cap)
    bufs = [mx.full((1, NKV, rows, D), value, dtype=mx.float16) for _ in range(n)]
    mx.eval(bufs)
    del bufs


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
@pytest.mark.parametrize(
    "T,L",
    [
        (4100, 512),
        (4100, 37),
        (4100, 9),
        (4097, 7),
        (1500, 3),
        (1500, 5),
        (4127, 512),
        (4096, 512),
    ],
)
def test_p2s_is_immune_to_poisoned_recycled_scratch(value, T, L):
    cache, q = big(), queries(L, T + L)
    mx.clear_cache()
    clean = f.prefill_attention(cache, q, SCALE, T)
    mx.eval(clean)
    _poison_buffer_cache(value, CAP)
    dirty = f.prefill_attention(cache, q, SCALE, T)
    mx.eval(dirty)
    assert bool(mx.all(mx.isfinite(dirty.astype(mx.float32))).item()), (value, T, L)
    assert bit_equal(dirty, clean)


@pytest.mark.parametrize("T", (1100, 1099, 1029, 1))
def test_p2s_poisoned_scratch_with_a_capacity_off_the_pad(T):
    """Capacity 1,100 (not a multiple of 128): the scratch rounds up to 1,152 rows, the tail is zeroed in range."""
    cache = fill_cache(1100, 1100, seed=13)
    assert f.scratch_capacity(1100) == 1152
    q = queries(min(T, 37), 3)
    mx.clear_cache()
    clean = f.prefill_attention(cache, q, SCALE, T)
    mx.eval(clean)
    _poison_buffer_cache(float("nan"), 1100)
    dirty = f.prefill_attention(cache, q, SCALE, T)
    assert bit_equal(dirty, clean)
    k, v = f.dequant_rotated(cache, T)
    mx.eval(k, v)
    t_pad = min(-(-T // 128) * 128, 1152)
    assert k.shape == (1, NKV, 1152, D)
    assert bool(mx.all(k[:, :, T:t_pad] == 0).item()) and bool(
        mx.all(v[:, :, T:t_pad] == 0).item()
    )


# [new] the scratch is capacity-sized (one size for every chunk) and fp16
def test_scratch_is_capacity_sized_fp16():
    k, v = f.dequant_rotated(big(), 700)
    mx.eval(k, v)
    assert k.shape == v.shape == (1, NKV, CAP, D) and k.dtype == mx.float16


# ================================================================== matrix-unit verify
def _mma_case(T, k, seed, heads=None, lengths=None, **kw):
    cache = big()
    lengths = lengths or consecutive(T, k)
    q = queries(len(lengths), seed)
    Kd, Vd = np_for(cache, lengths[-1])
    ref = ref_lengths(Kd, Vd, q, lengths, heads=heads)
    prod = production_lengths(cache, q, lengths)
    out = f.mma_attention(cache, q, SCALE, lengths, **kw)
    mx.eval(out)
    assert out.shape == q.shape and out.dtype == mx.bfloat16
    return judge(out, ref, prod, heads)


MMA_SWEEP_T = (4096, 8190, 8193, 8320, 9000, 16385, 32766, 32770, 33025, 40000)


# [new] A1 matrix-unit verify on the 512-token block plan, prototype sweep, k = 1..5 (all heads)
@pytest.mark.parametrize("k", (1, 2, 3, 4, 5))
@pytest.mark.parametrize("T", MMA_SWEEP_T)
def test_a1_mma_prototype_sweep(T, k):
    assert_a1(_mma_case(T, k, seed=1000 * k + T), f"(mma T={T}, k={k})")


# [new] verify_attention (the routed entry) equals the instrument entry on consecutive lengths
def test_verify_attention_is_mma_on_consecutive_lengths():
    cache, q = big(), queries(3, 9)
    a = f.verify_attention(cache, q, SCALE, consecutive(9000, 3))
    b = f.mma_attention(cache, q, SCALE, consecutive(9000, 3))
    assert bit_equal(a, b)


# [new] A1 matrix-unit verify at target context, heads 0 / 6 / 12 / 18, k = 2, 3, 5
@pytest.mark.parametrize("k", (2, 3, 5))
@pytest.mark.parametrize("T", LONG_T)
def test_a1_mma_long_context(T, k):
    assert_a1(_mma_case(T, k, seed=1000 * k + T, heads=HEADS), f"(mma T={T}, k={k})")


# [new] A1 on tile / block boundaries of the 512 plan and gapped lengths (instrument entry)
@pytest.mark.parametrize(
    "lengths",
    [
        [1],
        [1, 2],
        [15, 16, 17],
        [511, 512, 513],
        [512, 513],
        [1023, 1024, 1025],
        [100, 8191, 8192],
        [4095, 4097, 20000, 33024, 33025],
    ],
)
def test_a1_mma_boundaries_and_gaps(lengths):
    assert_a1(_mma_case(0, 0, seed=7 + sum(lengths), lengths=lengths), f"({lengths})")


# [new] R2: fully masked (row, block) pairs (row 0 of [512, 513] sees nothing in block 1) stay finite and exact enough
def test_r2_fully_masked_block_no_nan():
    cache = big()
    q = queries(2, 11)
    out = f.verify_attention(cache, q, SCALE, [512, 513])
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out.astype(mx.float32))).item())
    assert_a1(_mma_case(0, 0, seed=11, lengths=[512, 513]), "(R2)")


# ------------------------------------------------------------------ negative controls (the mma spec's N1 / N2 / N3)
GAPPED = (-4000, -1717, -500, -33, 0)


def _planted(k, pattern, T=9000, amp=30.0):
    lengths = (
        consecutive(T, k) if pattern == "consecutive" else [T + d for d in GAPPED[-k:]]
    )
    rng = np.random.default_rng(1234 + k)
    u = rng.standard_normal(D).astype(np.float32)
    u /= np.linalg.norm(u)
    positions = sorted({L - 1 for L in lengths} | {lengths[0]})
    offset = lengths[-1] + 3
    Kn = np.array(mx.random.normal((1, NKV, offset, D), key=mx.random.key(11)))
    Vn = np.array(mx.random.normal((1, NKV, offset, D), key=mx.random.key(12)))
    Kn[:, :, positions, :] = amp * u
    K, V = mx.array(Kn).astype(mx.bfloat16), mx.array(Vn).astype(mx.bfloat16)
    cache = fill_cache(offset, CAP, K=K, V=V)
    q = mx.array(np.broadcast_to(16.0 * u, (1, NQ, k, D)).copy()).astype(mx.bfloat16)
    return cache, q, lengths


@pytest.fixture(scope="module")
def planted_cases():
    return {}


def _planted_ref(planted_cases, k, pattern):
    key = (k, pattern)
    if key not in planted_cases:
        cache, q, lengths = _planted(k, pattern)
        Kd, Vd = np_for(cache, cache.offset)
        ref = ref_lengths(Kd, Vd, q, lengths)
        prod = production_lengths(cache, q, lengths)
        planted_cases[key] = (cache, q, lengths, ref, prod)
    return planted_cases[key]


def _control(name, lengths):
    if name == "N1":
        return {"_kv_head_shift": 1}, lengths
    if name == "N2":
        return {}, [L + 1 for L in lengths]  # leak: one token too many
    return {}, [L - 1 for L in lengths]  # N3, drop: one token too few


@pytest.mark.parametrize("pattern", ("consecutive", "gapped"))
@pytest.mark.parametrize("control", ("N1", "N2", "N3"))
@pytest.mark.parametrize("k", (1, 3, 5))
def test_mma_negative_controls_fail_a1(planted_cases, k, control, pattern):
    cache, q, lengths, ref, prod = _planted_ref(planted_cases, k, pattern)
    ok = f.mma_attention(cache, q, SCALE, lengths)
    assert_a1(judge(ok, ref, prod), "(unperturbed planted call)")
    kw, ctl = _control(control, lengths)
    bad = f.mma_attention(cache, q, SCALE, ctl, **kw)
    res = judge(bad, ref, prod)
    assert not res["a1"], f"{control} passed A1 (k={k}, {pattern}): {res}"


def test_fragment_mapping_ok():
    assert f.fragment_mapping_ok() is True


# ================================================================== D2 (bit identity)
D2_T = (2049, 4097, 8193, 40000, 65537, 131072, 260096)
D2_T_FAST = (2049, 4097, 8193, 40000)


# [new] A1 D2: copy-free decode bit-identical to today's singleton on the same state at T
@pytest.mark.parametrize("T", D2_T)
def test_a1_d2_decode_bit_identical(T):
    cache, q = big(), queries(1, T)
    got = f.decode_attention(cache, q, SCALE, T)
    want = singleton(cache, q, T)
    assert bit_equal(got, want), f"T={T}"
    assert bool(mx.all(mx.isfinite(got.astype(mx.float32))).item())


# [new] D2 on a state sliced below the offset (the per-query fallback shape): the cache holds more tokens than T_eff
@pytest.mark.parametrize("T", (2049, 9000, 70000))
def test_a1_d2_on_a_state_sliced_below_the_offset(T):
    cache = big()
    assert cache.offset > T + 1000
    q = queries(1, 5 + T)
    assert bit_equal(f.decode_attention(cache, q, SCALE, T), singleton(cache, q, T))


# [new] D2 multi-query (m72_exact verify): each query bit-identical to the singleton at its own T_eff, k = 2..5
@pytest.mark.parametrize("k", (2, 3, 4, 5))
@pytest.mark.parametrize("T", D2_T)
def test_a1_d2_multi_query_bit_identical(T, k):
    if T - k + 1 <= f.SINGLE_PASS_MAX:
        T = (
            f.SINGLE_PASS_MAX + k
        )  # lengths[0] > 2048 (shorter blocks take the per-query loop)
    cache, q = big(), queries(k, 7 * T + k)
    lengths = consecutive(T, k)
    got = f.verify_attention(cache, q, SCALE, lengths, exact=True)
    want = production_lengths(cache, q, lengths)
    assert bit_equal(got, want), f"T={T} k={k}"


# [new] across the 65,536 block-count tier (each query keeps the singleton's own partition: two classes)
@pytest.mark.parametrize(
    "lengths",
    [
        [65535, 65536],
        [65535, 65536, 65537],
        [65534, 65535, 65536, 65537, 65538],
        [65536, 65537],
        [8191, 8192, 8193],
        [32767, 32768, 32769, 32770],
    ],
)
def test_a1_d2_multi_query_across_tiers(lengths):
    cache, q = big(), queries(len(lengths), sum(lengths) % 9973)
    got = f.verify_attention(cache, q, SCALE, lengths, exact=True)
    want = production_lengths(cache, q, lengths)
    assert bit_equal(got, want), lengths


# [new] R2 for D2: blocks past T (empty in the singleton's partition, e.g. T = 2049 with 128 blocks) stay finite
def test_r2_d2_empty_blocks_finite():
    cache = big()
    nb = f.production_num_blocks(2049, NQ // NKV)
    assert (nb - 1) * -(-2049 // nb) >= 2049  # the last block is empty
    out = f.decode_attention(cache, queries(1, 3), SCALE, 2049)
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out.astype(mx.float32))).item())


# [invariant] the block count mirror equals the shipped dispatch at every tier edge
@pytest.mark.parametrize("T", (2049, 8192, 8193, 32768, 32769, 65536, 65537, 262144))
def test_production_num_blocks_mirror(T):
    want = 64 if T <= 8192 else 128 if T <= 32768 else 256 if T <= 65536 else 512
    assert f.production_num_blocks(T, 6) == 2 * want
    assert f.production_num_blocks(T, 5) == want


# [new] single-pass row (T <= 2048): decode_attention is today's call
@pytest.mark.parametrize("T", (1, 37, 2048))
def test_decode_single_pass_is_todays_call(T):
    cache, q = big(), queries(1, 40 + T)
    assert bit_equal(f.decode_attention(cache, q, SCALE, T), singleton(cache, q, T))


# ================================================================== preconditions / scope
def test_preconditions_are_asserted():
    cache = fill_cache(1024, 1100)
    with pytest.raises(AssertionError):
        f.verify_attention(
            cache, queries(2, 1), SCALE, [1024, 1025]
        )  # beyond the offset
    with pytest.raises(AssertionError):
        f.verify_attention(cache, queries(3, 1), SCALE, [10, 12, 13])  # gapped
    with pytest.raises(AssertionError):
        f.verify_attention(cache, queries(6, 1), SCALE, list(range(10, 16)))  # k = 6
    with pytest.raises(AssertionError):
        f.verify_attention(
            cache, queries(2, 1), SCALE, [100, 101], exact=True
        )  # <= 2048
    with pytest.raises(AssertionError):
        f.prefill_attention(cache, queries(101, 1), SCALE, 100)  # L > T
    with pytest.raises(AssertionError):
        f.dequant_rotated(cache, 1025)


def test_supports_scope():
    cache = fill_cache(64, 256)
    q = queries(1, 1)
    assert f.supports(cache, q) == (True, None)
    assert f.supports(cache, q, "causal") == (True, None)
    assert f.supports(cache, q, mx.ones((1, 1, 1, 64), dtype=mx.bool_)) == (
        False,
        "mask",
    )
    assert f.supports(cache, q, None, mx.zeros((NQ,))) == (False, "sinks")
    assert f.supports(cache, q.astype(mx.float16)) == (False, "dtype")
    assert f.supports(cache, mx.zeros((1, 8, 1, D), dtype=mx.bfloat16)) == (
        False,
        "shape",
    )
    assert f.supports(cache, mx.zeros((2, NQ, 1, D), dtype=mx.bfloat16)) == (
        False,
        "batch",
    )
    legacy = fill_cache(64, 256)
    legacy._decode_2pass_use_legacy = True
    assert f.supports(legacy, q) == (False, "legacy")
    override = fill_cache(64, 256)
    override._decode_2pass_num_blocks = 32
    assert f.supports(override, q) == (False, "block_override")
    three = tq.TurboQuantKVCache(bits=3, max_kv_size=256, prealloc_tokens=256)
    three.update_and_fetch(queries(4, 2)[:, :NKV], queries(4, 3)[:, :NKV])
    assert f.supports(three, q) == (False, "bits")
    assert f.supports(object(), q) == (False, "cache_class")


def test_supports_refuses_batch_caches():
    batch = tq.BatchTurboQuantKVCache([0], bits=4)
    assert f.supports(batch, queries(1, 1)) == (False, "batch_cache")
    from mlx_vlm.models import cache as kv_cache

    assert f.supports(kv_cache.BatchKVCache([0]), queries(1, 1)) == (
        False,
        "batch_cache",
    )


def _entry_point_calls(cache):
    T = 2500
    return {
        "dequant_rotated": lambda: f.dequant_rotated(cache, T),
        "prefill_attention": lambda: f.prefill_attention(
            cache, queries(4, 1), SCALE, T
        ),
        "decode_attention": lambda: f.decode_attention(cache, queries(1, 1), SCALE, T),
        "verify_attention": lambda: f.verify_attention(
            cache, queries(3, 1), SCALE, [T - 2, T - 1, T]
        ),
        "verify_attention_exact": lambda: f.verify_attention(
            cache, queries(3, 1), SCALE, [T - 2, T - 1, T], exact=True
        ),
        "mma_attention": lambda: f.mma_attention(
            cache, queries(2, 1), SCALE, [T - 1, T]
        ),
        "mq_decode_attention": lambda: f.mq_decode_attention(
            cache, queries(2, 1), SCALE, [T - 1, T]
        ),
    }


@pytest.mark.parametrize(
    "entry",
    [
        "dequant_rotated",
        "prefill_attention",
        "decode_attention",
        "verify_attention",
        "verify_attention_exact",
        "mma_attention",
        "mq_decode_attention",
    ],
)
def test_kernel_entry_points_assert_the_exact_cache_class(entry):
    batch = tq.BatchTurboQuantKVCache([0], bits=4)
    batch.update_and_fetch(queries(3000, 2)[:, :NKV], queries(3000, 3)[:, :NKV])

    class Sub(tq.TurboQuantKVCache):
        pass

    sub = Sub(bits=4, max_kv_size=3072, prealloc_tokens=3072)
    sub.update_and_fetch(queries(3000, 2)[:, :NKV], queries(3000, 3)[:, :NKV])
    for cache, name in ((batch, "BatchTurboQuantKVCache"), (sub, "Sub")):
        with pytest.raises(AssertionError, match=f"exactly TurboQuantKVCache.*{name}"):
            mx.eval(_entry_point_calls(cache)[entry]())


def test_self_test_passes():
    out = f.self_test()
    assert out["cells"] >= 7


# [threat] the full-cap state arrays are the kernels' inputs (copy-free), never a T-sliced copy
def test_kernels_read_the_full_cap_state(monkeypatch):
    seen = []
    real = mx.fast.metal_kernel

    def spy(*a, **kw):
        kern = real(*a, **kw)

        def launch(*ca, **ckw):
            seen.append([x for x in ckw["inputs"] if hasattr(x, "shape")])
            return kern(*ca, **ckw)

        return launch

    monkeypatch.setattr(mx.fast, "metal_kernel", spy)
    for fn in (f._dq_kernel, f._mma_kernel, f._d2_pass1_kernel):
        fn.cache_clear()
    try:
        cache = fill_cache(3000, 4096)
        q = queries(3, 1)
        mx.eval(
            f.decode_attention(cache, q[:, :, :1], SCALE, 2500),
            f.verify_attention(cache, q, SCALE, [2900, 2901, 2902]),
            f.verify_attention(cache, q, SCALE, [2900, 2901, 2902], exact=True),
            f.prefill_attention(cache, q, SCALE, 3000),
        )
    finally:
        for fn in (f._dq_kernel, f._mma_kernel, f._d2_pass1_kernel):
            fn.cache_clear()
    pass1 = [c for c in seen if any(x is cache.keys.norms for x in c)]
    assert len(pass1) >= 4
    for call in pass1:
        assert any(x is cache.keys.indices for x in call)
        assert any(x is cache.values.norms for x in call)
        assert any(x is cache.values.indices for x in call)
        for x in call:
            if x.ndim >= 3:
                assert x.shape[2] == 4096, x.shape
