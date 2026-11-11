"""M72 1.3 A1: kernel accuracy of ``mlx_vlm.tq4_fast`` (the promoted p2s dequantize + prefill, matrix-unit verify, D2 decode /
multi-query), through its specified interface only (spec ``docs/specs/m72-1.3-tq4-production.md`` v1.1, "Interfaces", A1, R2, R4).

Caches are preallocated to 262,144 tokens (the pick's capacity). The references and the A1 criterion are in
``test_tq4_m72_support``. Every test is classed in its comment: [new] exercises new behaviour, [invariant] preserves an
invariant of today's code or of the instrument, [fix] reproduces a failure the specification fixes (the Python-tiled TQ4
prefill, the per-token singleton verify, the strided-state copy per decode call, D2 without tests / below T = 2,049).
"""

from __future__ import annotations

import numpy as np
import pytest
from test_tq4_m72_support import (
    BF16,
    CAP_FULL,
    FLOOR,
    FP16,
    HEADS,
    NKV,
    NQ,
    SCALE,
    D,
    assert_a1,
    bits_equal,
    fast_module,
    judge,
    mx,
    n_bit_mismatch,
    needs_metal,
    normal,
    pick,
    planted_cache,
    production_block,
    queries_for,
    ref_attention,
    rel_err,
    shared_cache,
    singleton,
    tq,
    tq4_cache,
    verify_lengths,
)

pytestmark = needs_metal


@pytest.fixture(scope="module")
def fast():
    return fast_module()


# --------------------------------------------------------------------------------------------- helpers
def prefill_check(fast, cache, T, L, *, seed=7, rows=None, heads=HEADS):
    """A1 for the p2s prefill of the chunk at [T - L, T) of ``cache`` (T may be below cache.offset)."""
    q = queries_for(L, seed + T + L)
    lengths = [T - L + 1 + j for j in range(L)]
    rows = list(range(L)) if rows is None else rows
    ref = ref_attention(cache, q, lengths, heads=heads, rows=rows)
    out = fast.prefill_attention(cache, q, SCALE, T)
    mx.eval(out)
    assert out.shape == (1, NQ, L, D) and out.dtype == q.dtype
    return (
        judge(pick(out, heads, rows), ref, floor_of(cache, q, T, L, ref, heads, rows)),
        ref,
    )


def floor_of(cache, q, T, L, ref, heads, rows):
    """Production error floor on the same (heads, rows): today's path on the state sliced to T."""
    ks, vs = (
        tq._QuantizedStateProxy(cache.state[0], cache.offset, NKV),
        tq._QuantizedStateProxy(cache.state[1], cache.offset, NKV),
    )
    from test_tq4_m72_support import sdpa

    out = sdpa(
        q, ks[:, :, :T, :], vs[:, :, :T, :], cache=cache, scale=SCALE, mask="causal"
    )
    mx.eval(out)
    return rel_err(pick(out, heads, rows), ref)


def verify_check(fast, cache, lengths, *, seed=9, heads=HEADS, **kw):
    k = len(lengths)
    q = queries_for(k, seed + lengths[-1] + k)
    ref = ref_attention(cache, q, lengths, heads=heads)
    prod = rel_err(pick(production_block(cache, q, lengths), heads, range(k)), ref)
    out = fast.verify_attention(cache, q, SCALE, lengths, **kw)
    mx.eval(out)
    assert out.shape == (1, NQ, k, D) and out.dtype == q.dtype
    return judge(pick(out, heads, range(k)), ref, prod)


# =============================================================================== dequant_rotated (p2s step 1)
def generic_dequant_rotated(state, T, codec, dtype):
    st = tq._slice_state(state, int(T))
    idx = tq._unpack_lowbit(st.indices, int(codec.bits), int(codec.dim)).astype(
        mx.int32
    )
    return (mx.take(codec.codebook, idx, axis=0) * st.norms[..., None]).astype(dtype)


# [new] the scratch is capacity-sized (one size, so MLX's buffer cache reuses it), fp16 by default, only [0, T) written
@pytest.mark.parametrize("T", (1, 511, 2049, 40000))
def test_dequant_rotated_is_capacity_sized_fp16_and_bit_identical_to_the_generic_form(
    fast, T
):
    cache = shared_cache(40000)
    k_rot, v_rot = fast.dequant_rotated(cache, T)
    for x, st, codec in (
        (k_rot, cache.keys, cache.key_codec),
        (v_rot, cache.values, cache.value_codec),
    ):
        assert x.shape == (1, NKV, CAP_FULL, D) and x.dtype == FP16
        want = generic_dequant_rotated(st, T, codec, FP16)
        assert n_bit_mismatch(x[:, :, :T, :], want) == 0


# [new] bf16 is the one other dtype; unsupported dtypes are refused (preconditions asserted)
def test_dequant_rotated_dtype_option_and_refusal(fast):
    cache = shared_cache(40000)
    k_rot, _ = fast.dequant_rotated(cache, 1000, dtype=BF16)
    assert k_rot.dtype == BF16
    with pytest.raises((AssertionError, ValueError, TypeError)):
        fast.dequant_rotated(cache, 1000, dtype=mx.float32)


# [new] T outside [1, offset] is refused
def test_dequant_rotated_refuses_t_outside_the_cache(fast):
    cache = shared_cache(40000)
    for T in (0, 40001):
        with pytest.raises((AssertionError, ValueError)):
            fast.dequant_rotated(cache, T)


# =============================================================================== p2s prefill A1
P2S_SWEEP = (
    (512, 512),
    (511, 37),
    (4096, 512),
    (8193, 512),
    (8193, 37),
    (32768, 512),
    (40000, 512),
    (40000, 37),
    (2049, 2),
    (1000, 1),
)


# [new] A1 at the prototype's sweeps (<= 40,960), cap 262,144
@pytest.mark.parametrize("T,L", P2S_SWEEP)
def test_a1_p2s_prototype_sweep(fast, T, L):
    cache = shared_cache(40000)
    rows = sorted({0, L // 2, L - 1})
    res, _ = prefill_check(fast, cache, T, L, rows=rows)
    assert_a1(res, f"(p2s T={T}, L={L})")


# [new] A1 at target context, heads 0 / 6 / 12 / 18, L = 512 and L = 37 (queries at positions [T - L, T))
@pytest.mark.parametrize("T", (65537, 131072, 260096))
@pytest.mark.parametrize("L", (512, 37))
def test_a1_p2s_at_target_context(fast, T, L):
    cache = shared_cache(T)
    res, _ = prefill_check(fast, cache, T, L, rows=sorted({0, 1, L // 2, L - 2, L - 1}))
    assert_a1(res, f"(p2s T={T}, L={L})")


# [new] T is T_eff, not cache.offset: the chunk at [T - L, T) of a longer cache reads only keys < T
@pytest.mark.parametrize("T", (2049, 8193, 65537))
def test_p2s_reads_t_not_the_cache_offset(fast, T):
    cache = shared_cache(131072)
    assert cache.offset > T
    res, _ = prefill_check(fast, cache, T, 37, rows=[0, 18, 36])
    assert_a1(res, f"(p2s T={T} < offset {cache.offset})")


# [fix] non-finite outputs never appear at the largest chunk and context (the Python-tiled path is the only one today)
def test_p2s_largest_chunk_at_the_largest_context_is_finite(fast):
    cache = shared_cache(260096)
    out = fast.prefill_attention(cache, queries_for(512, 3), SCALE, 260096)
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out.astype(mx.float32))).item())


# [new] negative controls, public interface: a leak (T + 1) and a drop (T - 1) on planted keys fail A1, the true call passes
def test_p2s_negative_controls_fail_a1_on_planted_keys(fast):
    T, L = 9000, 5
    positions = [
        T - 1,
        T - 3,
        T,
    ]  # the key at index T is the bait: visible only to the leak control
    cache, u = planted_cache(positions, T=T + 3, cap=CAP_FULL)
    # queries sit at positions [T - L, T): the cache holds 3 more tokens than T (the leak control is legal)
    q = mx.array(np.broadcast_to(16.0 * u, (1, NQ, L, D)).copy()).astype(BF16)
    lengths = [T - L + 1 + j for j in range(L)]
    ref = ref_attention(cache, q, lengths, heads=HEADS)
    prod = rel_err(pick(production_prefill_at(cache, q, T), HEADS, range(L)), ref)
    ok = fast.prefill_attention(cache, q, SCALE, T)
    mx.eval(ok)
    assert_a1(judge(pick(ok, HEADS, range(L)), ref, prod), "(unperturbed planted call)")
    for ctl_T in (T + 1, T - 1):
        bad = fast.prefill_attention(cache, q, SCALE, ctl_T)
        mx.eval(bad)
        res = judge(pick(bad, HEADS, range(L)), ref, prod)
        assert not res[
            "a1"
        ], f"control T={ctl_T} did not fail A1 (rel_err {res['rel_err']:.4g})"


def production_prefill_at(cache, q, T):
    from test_tq4_m72_support import sdpa, state_proxies

    ks, vs = state_proxies(cache)
    out = sdpa(
        q, ks[:, :, :T, :], vs[:, :, :T, :], cache=cache, scale=SCALE, mask="causal"
    )
    mx.eval(out)
    return out


# [new] bytes beyond T inside the capacity are never read for effect
def test_p2s_ignores_garbage_beyond_t(fast):
    T, cap = 5000, 6000
    cache = tq4_cache(T + 400, cap=cap, seed=21)
    q = queries_for(37, 5)
    lengths = [T - 37 + 1 + j for j in range(37)]
    ref = ref_attention(cache, q, lengths, heads=HEADS, rows=[0, 36])
    for st in (cache.keys, cache.values):
        st.indices[:, :, T:, :] = mx.random.randint(
            -(2**31),
            2**31 - 1,
            (1, NKV, cap - T, 32),
            dtype=mx.int32,
            key=mx.random.key(31),
        ).view(mx.uint32)
        st.norms[:, :, T:] = mx.full((1, NKV, cap - T), 1000.0, dtype=FP16)
    mx.eval(cache.keys, cache.values)
    out = fast.prefill_attention(cache, q, SCALE, T)
    mx.eval(out)
    assert_a1(judge(pick(out, HEADS, [0, 36]), ref, 2**-8), "(garbage beyond T)")


# [invariant] the call does not modify the cache and is deterministic
def test_p2s_does_not_modify_the_cache_and_is_deterministic(fast):
    cache = shared_cache(8193)
    before = (
        mx.array(cache.keys.norms[:, :, :8193]),
        mx.array(cache.keys.indices[:, :, :8193]),
    )
    q = queries_for(64, 4)
    a = fast.prefill_attention(cache, q, SCALE, 8193)
    b = fast.prefill_attention(cache, q, SCALE, 8193)
    mx.eval(a, b)
    assert bits_equal(a, b)
    assert bits_equal(before[0], cache.keys.norms[:, :, :8193])
    assert bits_equal(before[1], cache.keys.indices[:, :, :8193])
    assert cache.offset == 8193


# =============================================================================== mma verify A1 (m72_v1)
MMA_SWEEP_T = (4096, 8190, 8193, 8320, 9000, 16385, 32766, 32770, 33025, 40000)


# [new] A1 at the prototype's sweep lengths (straddling the 8,192 / 32,768 production thresholds), k = 2, 3, 5
@pytest.mark.parametrize("k", (2, 3, 5))
@pytest.mark.parametrize("T", MMA_SWEEP_T)
def test_a1_mma_prototype_sweep(fast, T, k):
    res = verify_check(fast, shared_cache(40000), verify_lengths(T, k))
    assert_a1(res, f"(mma T={T}, k={k})")


# [new] A1 at target context, heads 0 / 6 / 12 / 18
@pytest.mark.parametrize("k", (2, 3, 5))
@pytest.mark.parametrize("T", (65537, 131072, 260096))
def test_a1_mma_at_target_context(fast, T, k):
    res = verify_check(fast, shared_cache(T), verify_lengths(T, k))
    assert_a1(res, f"(mma T={T}, k={k})")


# [new] lengths straddling a 512-token block edge / the 65,536 tier: block-plan boundaries must not change the answer
@pytest.mark.parametrize(
    "lengths",
    [
        [512, 513],
        [511, 512, 513],
        [1023, 1024, 1025, 1026],
        [4095, 4096, 4097, 4098, 4099],
        [65535, 65536, 65537],
        [65534, 65535, 65536, 65537, 65538],
        [131071, 131072],
    ],
)
def test_a1_mma_block_and_tier_edges(fast, lengths):
    res = verify_check(fast, shared_cache(131072), lengths)
    assert_a1(res, f"(mma {lengths})")


# [new] R2: fully masked blocks (an earlier query's range ends before a later block starts) leave no NaN / Inf, k = 2..5
@pytest.mark.parametrize(
    "lengths", [[512, 513], [1024, 1025, 1026], [2048, 2049, 2050, 2051, 2052]]
)
def test_mma_fully_masked_blocks_stay_finite(fast, lengths):
    cache = shared_cache(40000)
    q = queries_for(len(lengths), 77)
    out = fast.verify_attention(cache, q, SCALE, lengths)
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out.astype(mx.float32))).item())


# [new] a kernel reading cache.offset instead of the passed length would attend to later tokens: planted bait at lengths[0]
# (visible only to queries j >= 1) and at every lengths[j] - 1, cache holds 3 more tokens than the last length
@pytest.mark.parametrize("k", (2, 3, 5))
def test_mma_reads_the_passed_lengths_not_the_offset_planted(fast, k):
    T = 9000
    lengths = verify_lengths(T, k)
    positions = sorted({L - 1 for L in lengths} | {lengths[0]})
    cache, u = planted_cache(positions, T=T + 3, cap=CAP_FULL)
    assert cache.offset == T + 3
    q = mx.array(np.broadcast_to(16.0 * u, (1, NQ, k, D)).copy()).astype(BF16)
    ref = ref_attention(cache, q, lengths, heads=HEADS)
    prod = rel_err(pick(production_block(cache, q, lengths), HEADS, range(k)), ref)
    ok = fast.verify_attention(cache, q, SCALE, lengths)
    mx.eval(ok)
    assert_a1(judge(pick(ok, HEADS, range(k)), ref, prod), f"(planted, k={k})")
    # negative controls through the public interface: leak (lengths + 1) and drop (lengths - 1) fail A1
    for name, ctl in (
        ("leak", [L + 1 for L in lengths]),
        ("drop", [L - 1 for L in lengths]),
    ):
        bad = fast.verify_attention(cache, q, SCALE, ctl)
        mx.eval(bad)
        res = judge(pick(bad, HEADS, range(k)), ref, prod)
        assert not res[
            "a1"
        ], f"{name} control did not fail A1 (k={k}, rel_err {res['rel_err']:.4g})"


# [new] bytes beyond lengths[-1] inside the capacity are never read for effect
def test_mma_ignores_garbage_beyond_the_last_length(fast):
    T, cap, Lmax = 9000, 9300, 8500
    cache = tq4_cache(T, cap=cap, seed=23)
    lengths = [Lmax - 2, Lmax - 1, Lmax]
    q = queries_for(3, 24)
    ref = ref_attention(cache, q, lengths, heads=HEADS)
    prod = rel_err(pick(production_block(cache, q, lengths), HEADS, range(3)), ref)
    for st in (cache.keys, cache.values):
        st.indices[:, :, Lmax:, :] = mx.random.randint(
            -(2**31),
            2**31 - 1,
            (1, NKV, cap - Lmax, 32),
            dtype=mx.int32,
            key=mx.random.key(32),
        ).view(mx.uint32)
        st.norms[:, :, Lmax:] = mx.full((1, NKV, cap - Lmax), 1000.0, dtype=FP16)
    mx.eval(cache.keys, cache.values)
    out = fast.verify_attention(cache, q, SCALE, lengths)
    mx.eval(out)
    assert_a1(
        judge(pick(out, HEADS, range(3)), ref, prod), "(garbage beyond the last length)"
    )


# [new] preconditions are asserted: lengths[-1] <= cache.offset, lengths ascending and >= 1, k in 2..5
def test_verify_attention_refuses_out_of_contract_arguments(fast):
    cache = shared_cache(40000)
    q2, q6 = queries_for(2, 1), queries_for(6, 1)
    with pytest.raises((AssertionError, ValueError)):
        fast.verify_attention(cache, q2, SCALE, [40000, 40001])
    with pytest.raises((AssertionError, ValueError)):
        fast.verify_attention(cache, q2, SCALE, [0, 5])
    with pytest.raises((AssertionError, ValueError)):
        fast.verify_attention(cache, q2, SCALE, [5000, 4999])
    with pytest.raises((AssertionError, ValueError)):
        fast.verify_attention(cache, q6, SCALE, verify_lengths(5000, 6))


# [invariant] output contract and determinism
def test_mma_output_contract_and_determinism(fast):
    cache = shared_cache(40000)
    q, lengths = queries_for(4, 5), verify_lengths(33025, 4)
    a, b = fast.verify_attention(cache, q, SCALE, lengths), fast.verify_attention(
        cache, q, SCALE, lengths
    )
    mx.eval(a, b)
    assert a.shape == (1, NQ, 4, D) and a.dtype == BF16 and bits_equal(a, b)


# =============================================================================== D2: bit identity to today's singleton
D2_T = (2049, 4097, 8193, 40000, 65537, 131072, 260096)


def decode_equals_singleton(fast, cache, T, seed):
    q = queries_for(1, seed + T)
    got = fast.decode_attention(cache, q, SCALE, T)
    want = singleton(cache, q, T)
    mx.eval(got, want)
    assert got.shape == want.shape and got.dtype == want.dtype
    return n_bit_mismatch(got, want)


# [new] [fix] R4 D2 decode: bit-identical to today's singleton on the same state at the same T_eff (cache holds exactly T)
@pytest.mark.parametrize("T", D2_T)
def test_a1_d2_decode_is_bit_identical_to_the_singleton(fast, T):
    assert decode_equals_singleton(fast, shared_cache(T), T, 1) == 0


# [new] a state sliced below the offset (the per-query fallback shape), incl. the 8,192 / 32,768 / 65,536 block-count tiers
@pytest.mark.parametrize("T", (2049, 8192, 8193, 32768, 32769, 65536, 65537, 100000))
def test_d2_decode_on_a_state_sliced_below_the_offset(fast, T):
    cache = shared_cache(131072)
    assert cache.offset > T
    assert decode_equals_singleton(fast, cache, T, 2) == 0


# [new] T <= 2,048 is today's single-pass kernel: still bit-identical (the entry point serves the whole range)
@pytest.mark.parametrize("T", (1, 17, 1024, 2047, 2048))
def test_decode_attention_at_or_below_2048_is_bit_identical_to_the_singleton(fast, T):
    assert decode_equals_singleton(fast, shared_cache(40000), T, 3) == 0


# [new] D2 reads T, not the offset: planted bait one token past T must not leak
def test_d2_decode_reads_the_passed_length(fast):
    T = 9000
    cache, u = planted_cache([T - 1, T], T=T + 3, cap=CAP_FULL)
    q = mx.array(np.broadcast_to(16.0 * u, (1, NQ, 1, D)).copy()).astype(BF16)
    got, want = fast.decode_attention(cache, q, SCALE, T), singleton(cache, q, T)
    mx.eval(got, want)
    assert n_bit_mismatch(got, want) == 0
    leaked = fast.decode_attention(cache, q, SCALE, T + 1)
    mx.eval(leaked)
    assert (
        n_bit_mismatch(leaked, want) > 0
    )  # the bait changes the answer: the control can fail


# =============================================================================== D2 multi-query (m72_exact verify)
EXACT_T = (2053, 4097, 8193, 40000, 65537, 131072)


def exact_equals_singletons(fast, cache, lengths, seed=4):
    k = len(lengths)
    q = queries_for(k, seed + lengths[-1])
    got = fast.verify_attention(cache, q, SCALE, lengths, exact=True)
    mx.eval(got)
    assert got.shape == (1, NQ, k, D) and got.dtype == q.dtype
    bad = []
    for j, L in enumerate(lengths):
        want = singleton(cache, q[:, :, j : j + 1, :], L)
        mx.eval(want)
        if n_bit_mismatch(got[:, :, j : j + 1, :], want):
            bad.append((j, L))
    return bad, got


# [new] each query bit-identical to the singleton at its own T_eff, k = 2..5, the tier-spanning T values
@pytest.mark.parametrize("k", (2, 3, 4, 5))
@pytest.mark.parametrize("T", EXACT_T)
def test_a1_d2_multi_query_is_per_query_bit_identical(fast, T, k):
    bad, _ = exact_equals_singletons(fast, shared_cache(T), verify_lengths(T, k))
    assert bad == []


# [new] the largest context
@pytest.mark.parametrize("k", (2, 5))
def test_a1_d2_multi_query_at_260096(fast, k):
    bad, _ = exact_equals_singletons(
        fast, shared_cache(260096), verify_lengths(260096, k)
    )
    assert bad == []


# [new] lengths straddling the production block-count tiers: each query uses the singleton's own block count
@pytest.mark.parametrize(
    "lengths",
    [
        [65535, 65536, 65537, 65538],
        [65534, 65535, 65536, 65537, 65538],
        [8191, 8192, 8193, 8194],
        [32767, 32768, 32769, 32770, 32771],
        [2049, 2050],
    ],
)
def test_d2_multi_query_across_tier_straddles(fast, lengths):
    bad, _ = exact_equals_singletons(fast, shared_cache(131072), lengths)
    assert bad == []


# [new] on a state sliced below the offset (the per-query fallback shape)
def test_d2_multi_query_on_a_state_below_the_offset(fast):
    cache = shared_cache(131072)
    bad, _ = exact_equals_singletons(fast, cache, verify_lengths(50000, 4))
    assert bad == []


# [new] no NaN / Inf, including fully masked blocks
@pytest.mark.parametrize("lengths", [[2049, 2050, 2051], [4096, 4097], [8192, 8193]])
def test_d2_multi_query_is_finite_on_block_edges(fast, lengths):
    cache = shared_cache(40000)
    q = queries_for(len(lengths), 11)
    out = fast.verify_attention(cache, q, SCALE, lengths, exact=True)
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out.astype(mx.float32))).item())


# [new] the exact and the fast verify agree within A1 (they are different summation orders of the same attention)
@pytest.mark.parametrize("T", (9000, 65537))
def test_exact_and_fast_verify_agree_within_a1(fast, T):
    cache = shared_cache(131072)
    lengths = verify_lengths(T, 3)
    q = queries_for(3, 6)
    ex = fast.verify_attention(cache, q, SCALE, lengths, exact=True)
    fa = fast.verify_attention(cache, q, SCALE, lengths)
    mx.eval(ex, fa)
    a, b = np.array(ex.astype(mx.float32)), np.array(fa.astype(mx.float32))
    assert np.max(np.abs(a - b)) / np.max(np.abs(a)) <= 2 * max(2**-8, FLOOR) * 2


# =============================================================================== copy-free (settled constraint, A5 shape)
def peak_extra(fn):
    fn()  # compile / warm outside the measurement
    mx.synchronize()
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    out = fn()
    mx.eval(out)
    mx.synchronize()
    return mx.get_peak_memory() - base


def used_kv_bytes(cache, T):
    return sum(a.nbytes for st in (cache.keys, cache.values) for a in st) * T / CAP_FULL


# [invariant] instrument: today's singleton on a sub-cap state copies the strided slice (peak >= the used K/V bytes)
def test_instrument_todays_decode_copies_the_strided_state():
    T = 260096
    cache = shared_cache(T)
    q = queries_for(1, 8)
    extra = peak_extra(lambda: singleton(cache, q, T))
    assert extra >= 0.8 * used_kv_bytes(cache, T)


# [fix] D2 decode takes the full-cap arrays: no layout copy (peak far below the used K/V bytes)
def test_d2_decode_makes_no_layout_copy(fast):
    T = 260096
    cache = shared_cache(T)
    q = queries_for(1, 8)
    extra = peak_extra(lambda: fast.decode_attention(cache, q, SCALE, T))
    assert extra < 0.5 * used_kv_bytes(cache, T), f"{extra / 1e6:.1f} MB"


# [fix] the verify kernels (fast and exact) take the full-cap arrays too
@pytest.mark.parametrize("exact", (False, True))
def test_verify_makes_no_layout_copy_and_no_dequantized_materialization(fast, exact):
    T = 260096
    cache = shared_cache(T)
    q, lengths = queries_for(3, 8), verify_lengths(T, 3)
    extra = peak_extra(
        lambda: fast.verify_attention(cache, q, SCALE, lengths, exact=exact)
    )
    assert extra < 0.5 * used_kv_bytes(
        cache, T
    ), f"{extra / 1e6:.1f} MB (exact={exact})"


# =============================================================================== scope predicate
def supports(fast, cache, q=None, mask=None, sinks=None):
    q = queries_for(1, 1) if q is None else q
    return fast.supports(cache, q, mask, sinks)


@pytest.fixture(scope="module")
def small_cache():
    return tq4_cache(600, cap=1024, seed=5)


# [new] in scope: the pick's shape, causal / no mask
@pytest.mark.parametrize("mask", (None, "causal"))
@pytest.mark.parametrize("L", (1, 2, 37, 512))
def test_supports_in_scope(fast, small_cache, mask, L):
    assert supports(fast, small_cache, queries_for(L, 2), mask) == (True, None)


# [new] out of scope: a (False, reason) pair, never an exception (the reason is a non-empty string)
def check_out(res):
    ok, reason = res
    assert ok is False and isinstance(reason, str) and reason


def test_supports_refuses_array_mask_and_sinks(fast, small_cache):
    q = queries_for(2, 1)
    check_out(
        supports(fast, small_cache, q, mask=mx.ones((1, 1, 2, 600), dtype=mx.bool_))
    )
    check_out(supports(fast, small_cache, q, sinks=mx.zeros((NQ,), dtype=BF16)))


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1, 16, 1, D), BF16),
        ((1, NQ, 1, 128), BF16),
        ((2, NQ, 1, D), BF16),
        ((1, NQ, 1, D), FP16),
        ((1, NQ, 1, D), mx.float32),
    ],
    ids=["n_q16", "D128", "batch2", "fp16", "fp32"],
)
def test_supports_refuses_other_query_shapes_and_dtypes(
    fast, small_cache, shape, dtype
):
    check_out(supports(fast, small_cache, mx.zeros(shape, dtype=dtype)))


# [new] the legacy flag and a block override leave scope (the D2 bit-identity contract is stated for the non-legacy default)
def test_supports_refuses_the_legacy_flag_and_a_block_override():
    fast = fast_module()
    legacy = tq4_cache(300, cap=512, seed=6)
    legacy._decode_2pass_use_legacy = True
    check_out(supports(fast, legacy))
    override = tq4_cache(300, cap=512, seed=6)
    override._decode_2pass_num_blocks = 64
    check_out(supports(fast, override))
    legacy_ctor = tq.TurboQuantKVCache(
        bits=4, max_kv_size=512, prealloc_tokens=512, decode_2pass_use_legacy=True
    )
    legacy_ctor.update_and_fetch(
        normal((1, NKV, 300, D), 1), normal((1, NKV, 300, D), 2)
    )
    check_out(supports(fast, legacy_ctor))


# [new] other codecs / widths / classes: only the exact class with MSE 4-bit keys and values qualifies
def test_supports_refuses_other_cache_classes_and_codecs(fast):
    K, V = normal((1, NKV, 300, D), 1), normal((1, NKV, 300, D), 2)
    bits3 = tq.TurboQuantKVCache(bits=3, max_kv_size=512, prealloc_tokens=512)
    bits3.update_and_fetch(K, V)
    check_out(supports(fast, bits3))
    mixed = tq.TurboQuantKVCache(
        bits=4, key_bits=4, value_bits=3, max_kv_size=512, prealloc_tokens=512
    )
    mixed.update_and_fetch(K, V)
    check_out(supports(fast, mixed))
    prod = tq.TurboQuantKVCache(
        bits=4, kv_quant_mode="prod", max_kv_size=512, prealloc_tokens=512
    )
    prod.update_and_fetch(K, V)
    check_out(supports(fast, prod))

    class Sub(tq.TurboQuantKVCache):
        pass

    sub = Sub(bits=4, max_kv_size=512, prealloc_tokens=512)
    sub.update_and_fetch(K, V)
    check_out(
        supports(fast, sub)
    )  # type(cache) is TurboQuantKVCache, not an instance check

    two_heads = tq.TurboQuantKVCache(bits=4, max_kv_size=512, prealloc_tokens=512)
    two_heads.update_and_fetch(normal((1, 2, 300, D), 1), normal((1, 2, 300, D), 2))
    check_out(supports(fast, two_heads))


# [new] the batch class is out of scope with the reason the specification names
def test_supports_batch_cache_reason_is_batch_cache(fast):
    batch = tq.BatchTurboQuantKVCache(left_padding=[0], bits=4)
    batch.update_and_fetch(normal((1, NKV, 300, D), 1), normal((1, NKV, 300, D), 2))
    ok, reason = supports(fast, batch)
    assert ok is False and reason == "batch_cache"


# [new] a native cache, an empty cache and None never raise (per call, never an error at request time)
def test_supports_never_raises_on_foreign_or_empty_caches(fast):
    from mlx_vlm.models.cache import KVCache

    check_out(supports(fast, KVCache()))
    check_out(
        supports(
            fast, tq.TurboQuantKVCache(bits=4, max_kv_size=512, prealloc_tokens=512)
        )
    )
    check_out(supports(fast, None))


# =============================================================================== counters, self-test
# [new] route_counts: a flat str -> int mapping, never decreasing, reset never during a process
def test_route_counts_is_a_monotone_str_to_int_mapping(fast):
    before = dict(fast.route_counts())
    assert all(isinstance(k, str) and isinstance(v, int) for k, v in before.items())
    fast.self_test()
    after = fast.route_counts()
    assert all(after.get(k, 0) >= v for k, v in before.items())


# [new] self_test on small synthetic caches vs references: returns a dict and does not raise on a working build
def test_self_test_returns_a_dict(fast):
    res = fast.self_test()
    assert isinstance(res, dict) and res


# [invariant] entry points reject the batch class instead of reading it (preconditions asserted: batch-1 single-sequence cache)
def test_kernels_refuse_a_batch_cache(fast):
    batch = tq.BatchTurboQuantKVCache(left_padding=[0], bits=4)
    batch.update_and_fetch(normal((1, NKV, 3000, D), 1), normal((1, NKV, 3000, D), 2))
    with pytest.raises(Exception):
        out = fast.decode_attention(batch, queries_for(1, 1), SCALE, 2500)
        mx.eval(out)
