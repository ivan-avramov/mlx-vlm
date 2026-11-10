"""M72 1.2: GPU tests of ``dev/m72_mma_verify.py`` (matrix-unit MTP verify kernel prototype), through its specified interface only:
``mma_verify_tq4``, ``mma_verify_native16``, ``plan_blocks`` (spec ``docs/specs/m72-mma-verify.md`` v1.1, criteria A1-A5).

The reference is independent of the kernel under test: K / V dequantized through the codec (TQ4) or taken as bf16 (native16), then
exact softmax attention per query in numpy (float32 matmuls, float64 softmax). The production path (TQ4 singleton / native16 SDPA, one
query at a time) supplies the error floor A1 is defined against: rel error <= 2 * max(production rel error, 2**-12) and
``mx.allclose(out_fp32, ref, atol=3e-2, rtol=3e-2)``; rel error = max|out - ref| / max|ref| over the block.

Every test is classed in its comment: [new] exercises new behaviour, [invariant] preserves an invariant of the production paths
or of the instrument, [fix] reproduces a failure the specification fixes.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(),
    reason="Metal GPU required for the matrix-unit verify kernel",
)

from mlx_vlm import turboquant as tq  # noqa: E402
from mlx_vlm.models.base import scaled_dot_product_attention as sdpa  # noqa: E402

DEV = Path(__file__).resolve().parents[1]
NQ, NKV, D = 24, 4, 256
R = NQ // NKV
SCALE = 1.0 / math.sqrt(D)
DT = mx.bfloat16
FLOOR = (
    2.0**-12
)  # spec v1.2: A1 bound = 2 x max(production rel error, 2**-12) (v1.1 had 2**-8)
BIG_CAP, BIG_T = 40960, 40000  # the largest cache any test builds (<= 40,960 tokens)
SWEEP_T = (
    4096,
    8190,
    8193,
    8320,
    9000,
    16385,
    32766,
    32770,
    33025,
    40000,
)  # the probe's MQ_SWEEP_T
KS = (1, 2, 3, 4, 5)


# ------------------------------------------------------------------ module under test
@pytest.fixture(scope="module")
def mma():
    path = DEV / "m72_mma_verify.py"
    assert path.exists(), f"{path} does not exist yet (module under test)"
    spec = importlib.util.spec_from_file_location("m72_mma_verify", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m72_mma_verify"] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ data
def normal(shape, seed):
    return mx.random.normal(shape, key=mx.random.key(seed)).astype(DT)


def queries_for(k, seed):
    return normal((1, NQ, k, D), seed)


def verify_lengths(T, k):
    return [T - k + 1 + j for j in range(k)]


def build_tq4(K, V, cap):
    cache = tq.TurboQuantKVCache(bits=4, max_kv_size=cap, prealloc_tokens=cap)
    cache.update_and_fetch(K, V)
    mx.eval(cache.keys, cache.values)
    assert cache.keys.norms.shape[2] == cap and cache.keys.indices.shape == (
        1,
        NKV,
        cap,
        32,
    )
    assert cache.values.norms.shape[2] == cap
    return cache


def full_cap(X, cap):
    out = mx.concatenate([X, mx.zeros((1, NKV, cap - X.shape[2], D), dtype=DT)], axis=2)
    mx.eval(out)
    assert out.shape == (1, NKV, cap, D)
    return out


def dequant_np(cache):
    ks = tq._slice_state(cache.keys, cache.offset)
    vs = tq._slice_state(cache.values, cache.offset)
    Kd = cache.key_codec.dequantize(ks).astype(mx.float32)
    Vd = cache.value_codec.dequantize(vs).astype(mx.float32)
    mx.eval(Kd, Vd)
    return np.array(Kd), np.array(Vd)


class Arm:
    """One attention arm: how to run the kernel, the production path and the reference inputs."""

    def __init__(self, name, *, cache=None, K=None, V=None, Kn=None, Vn=None):
        self.name, self.cache, self.K, self.V, self.Kn, self.Vn = (
            name,
            cache,
            K,
            V,
            Kn,
            Vn,
        )

    @property
    def valid(self):  # tokens that hold real data
        return self.Kn.shape[2]

    def run(self, mma, q, lengths, **kw):
        if self.name == "tq4":
            return mma.mma_verify_tq4(self.cache, q, SCALE, lengths, **kw)
        return mma.mma_verify_native16(self.K, self.V, q, SCALE, lengths, **kw)

    def prod(self, q, lengths):
        outs = []
        if self.name == "tq4":
            ksr, vsr = self.cache.state
            ks, vs = tq._QuantizedStateProxy(
                ksr, self.cache.offset, NKV
            ), tq._QuantizedStateProxy(vsr, self.cache.offset, NKV)
            for j, L in enumerate(lengths):
                outs.append(
                    sdpa(
                        q[:, :, j : j + 1, :],
                        ks[:, :, :L, :],
                        vs[:, :, :L, :],
                        cache=self.cache,
                        scale=SCALE,
                        mask=None,
                    )
                )
        else:
            for j, L in enumerate(lengths):
                outs.append(
                    sdpa(
                        q[:, :, j : j + 1, :],
                        self.K[:, :, :L, :],
                        self.V[:, :, :L, :],
                        cache=None,
                        scale=SCALE,
                        mask=None,
                    )
                )
        out = mx.concatenate(outs, axis=2)
        mx.eval(out)
        return np.array(out.astype(mx.float32))


def make_tq4_arm(K, V, cap):
    cache = build_tq4(K, V, cap)
    Kn, Vn = dequant_np(cache)
    return Arm("tq4", cache=cache, Kn=Kn, Vn=Vn)


def make_n16_arm(K, V, cap):
    return Arm(
        "n16",
        K=full_cap(K, cap),
        V=full_cap(V, cap),
        Kn=np.array(K.astype(mx.float32)),
        Vn=np.array(V.astype(mx.float32)),
    )


ARM_MAKERS = {"tq4": make_tq4_arm, "n16": make_n16_arm}
ARMS = tuple(ARM_MAKERS)


def ref_attention(Kn, Vn, queries, lengths):
    """fp32 reference: exact softmax attention per query over the first lengths[j] tokens (query head h reads KV head h // R)."""
    q = np.array(queries.astype(mx.float32)) * np.float32(SCALE)
    k = q.shape[2]
    out = np.zeros(q.shape, np.float32)
    for h in range(NKV):
        for j in range(k):
            L = lengths[j]
            s = (q[0, h * R : (h + 1) * R, j, :] @ Kn[0, h, :L].T).astype(np.float64)
            s -= s.max(axis=-1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(axis=-1, keepdims=True)
            out[0, h * R : (h + 1) * R, j, :] = p.astype(np.float32) @ Vn[0, h, :L]
    return out


def rel_err(out, ref):
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))


def reference(arm, q, lengths):
    """(ref, production rel error) for one block."""
    ref = ref_attention(arm.Kn, arm.Vn, q, lengths)
    return ref, rel_err(arm.prod(q, lengths), ref)


def judge(out, ref, prod_err):
    """A1 (+ finiteness) on a kernel output -> dict(finite, rel_err, a1)."""
    o = np.array(out.astype(mx.float32))
    finite = bool(np.isfinite(o).all())
    if not finite:
        return {"finite": False, "rel_err": math.inf, "a1": False, "prod_err": prod_err}
    err = rel_err(o, ref)
    ok = err <= 2 * max(prod_err, FLOOR) and bool(
        mx.allclose(mx.array(o), mx.array(ref), atol=3e-2, rtol=3e-2).item()
    )
    return {"finite": True, "rel_err": err, "a1": ok, "prod_err": prod_err}


def check(mma, arm, q, lengths, **kw):
    ref, prod_err = reference(arm, q, lengths)
    out = arm.run(mma, q, lengths, **kw)
    mx.eval(out)
    assert out.shape == (1, NQ, q.shape[2], D) and out.dtype == q.dtype
    return judge(out, ref, prod_err)


def assert_a1(res, what=""):
    assert res["finite"], f"non-finite output {what}"
    assert res[
        "a1"
    ], f"A1 violated {what}: rel_err {res['rel_err']:.4g} > 2*max({res['prod_err']:.4g}, 2^-12) or allclose failed"


# ------------------------------------------------------------------ shared caches (module scope; never mutated by tests)
@pytest.fixture(scope="module")
def big():
    """Random K/V, BIG_T tokens in a BIG_CAP-token full-cap buffer; both arms. stride = cap > offset everywhere."""
    K, V = normal((1, NKV, BIG_T, D), 1), normal((1, NKV, BIG_T, D), 2)
    mx.eval(K, V)
    return types.SimpleNamespace(
        tq4=make_tq4_arm(K, V, BIG_CAP), n16=make_n16_arm(K, V, BIG_CAP)
    )


@pytest.fixture(scope="module")
def small():
    """1,024 random tokens in a 1,100-token buffer (capacity not a multiple of any tile / block): tile / block boundary cases."""
    K, V = normal((1, NKV, 1024, D), 3), normal((1, NKV, 1024, D), 4)
    mx.eval(K, V)
    return types.SimpleNamespace(
        tq4=make_tq4_arm(K, V, 1100), n16=make_n16_arm(K, V, 1100)
    )


# ================================================================== instrument validation (no kernel under test)
# [invariant] the reference and the production path agree to the floor on a random block: if this fails the A1 instrument is wrong
@pytest.mark.parametrize("arm_name", ARMS)
@pytest.mark.parametrize("T", (4096, 40000))
def test_instrument_production_path_agrees_with_reference(big, arm_name, T):
    arm = getattr(big, arm_name)
    q = queries_for(3, 10 + T)
    ref, prod_err = reference(arm, q, verify_lengths(T, 3))
    assert (
        prod_err < 2**-6
    ), f"production {arm_name} rel err {prod_err:.3g} vs the fp32 reference"


# [invariant] the same on tile / block boundary lengths and tiny T (production path usable as the A1 floor there)
@pytest.mark.parametrize("arm_name", ARMS)
@pytest.mark.parametrize(
    "lengths",
    [
        [1],
        [1, 2],
        [15, 16, 17],
        [31, 32, 33, 34, 35],
        [255, 256, 257],
        [500, 512, 513],
        [1, 1000],
    ],
)
def test_instrument_production_path_agrees_with_reference_on_boundaries(
    small, arm_name, lengths
):
    arm = getattr(small, arm_name)
    q = queries_for(len(lengths), 20 + sum(lengths))
    ref, prod_err = reference(arm, q, lengths)
    assert (
        prod_err < 2**-6
    ), f"production {arm_name} rel err {prod_err:.3g} at lengths {lengths}"


# ================================================================== plan_blocks
# [new] contract of the block planner: every token covered, every block non-empty and a whole number of tiles (>= 16 tokens)
@pytest.mark.parametrize(
    "max_length", (1, 15, 16, 17, 100, 4096, 8193, 33025, 40000, 262144)
)
def test_plan_blocks_covers_all_tokens_with_nonempty_blocks(mma, max_length):
    blocks, per_block = mma.plan_blocks(max_length)
    assert isinstance(blocks, int) and isinstance(per_block, int)
    assert blocks >= 1 and per_block >= 16 and per_block % 16 == 0
    assert blocks * per_block >= max_length  # covers [0, max_length)
    assert (
        blocks - 1
    ) * per_block < max_length  # the last block holds at least one token


# [new] block_tokens overrides the tokens per block
@pytest.mark.parametrize(
    "max_length,bt,blocks",
    [(40000, 256, 157), (100, 64, 2), (64, 64, 1), (65, 64, 2), (8193, 128, 65)],
)
def test_plan_blocks_honours_block_tokens(mma, max_length, bt, blocks):
    assert mma.plan_blocks(max_length, block_tokens=bt) == (blocks, bt)


# ================================================================== A1 + A4: random sweep, both arms, verify blocks k = 1..5
# [new] A1 at the probe's sweep lengths (straddling the 8192 / 32768 production block thresholds, T not a multiple of the tile),
# every k; A4 finiteness; output shape / dtype
@pytest.mark.parametrize("arm_name", ARMS)
@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("T", SWEEP_T)
def test_a1_a4_random_verify_block_sweep(mma, big, arm_name, T, k):
    arm = getattr(big, arm_name)
    res = check(mma, arm, queries_for(k, 1000 * k + T), verify_lengths(T, k))
    assert_a1(res, f"({arm_name}, T={T}, k={k})")


# [new] queries need not be a verify block ending at the last token: gapped lengths, a prefix of the cache (lengths[-1] < offset)
@pytest.mark.parametrize("arm_name", ARMS)
@pytest.mark.parametrize(
    "lengths",
    [
        [9000],
        [100, 8191, 8192],
        [4095, 4097, 20000, 33024, 33025],
        [1, 17, 33, 4096, 16385],
    ],
)
def test_a1_gapped_lengths_below_the_cache_offset(mma, big, arm_name, lengths):
    arm = getattr(big, arm_name)
    assert lengths[-1] < arm.valid
    assert_a1(
        check(mma, arm, queries_for(len(lengths), 7 + sum(lengths)), lengths),
        f"({arm_name}, {lengths})",
    )


# [new] A4: lengths at tile (16 / 32) and block (32 / 64 / 256) boundaries and k = 1, T not a multiple of the tile, T = 1
@pytest.mark.parametrize("arm_name", ARMS)
@pytest.mark.parametrize(
    "lengths",
    [
        [1],
        [1, 2],
        [1, 2, 3, 4, 5],
        [15],
        [16],
        [17],
        [15, 16, 17],
        [16, 17],
        [31, 32, 33, 34, 35],
        [32],
        [32, 33],
        [33],
        [63, 64, 65],
        [64, 65, 66, 67, 68],
        [255, 256, 257],
        [500, 512, 513],
        [1023, 1024],
        [1, 16, 17, 33, 1000],
    ],
)
@pytest.mark.parametrize("block_tokens", (None, 32, 64, 256))
def test_a1_a4_tile_and_block_boundaries(mma, small, arm_name, lengths, block_tokens):
    arm = getattr(small, arm_name)
    q = queries_for(len(lengths), 30 + sum(lengths))
    res = check(mma, arm, q, lengths, block_tokens=block_tokens)
    assert_a1(res, f"({arm_name}, {lengths}, block_tokens={block_tokens})")


# [new] more blocks than the production pass 2 strides in one go (32 simdgroups), and a single huge block
@pytest.mark.parametrize("arm_name", ARMS)
@pytest.mark.parametrize(
    "block_tokens,lengths",
    [
        (128, [8191, 8192, 8193]),
        (32, [4094, 4095, 4096, 4097, 4098]),
        (65536, [39998, 39999, 40000]),
    ],
)
def test_a1_many_blocks_and_one_block(mma, big, arm_name, block_tokens, lengths):
    arm = getattr(big, arm_name)
    res = check(
        mma, arm, queries_for(len(lengths), 55), lengths, block_tokens=block_tokens
    )
    assert_a1(res, f"({arm_name}, block_tokens={block_tokens}, {lengths})")


# [new] the block split is a free policy: the result must not depend on it (beyond rounding)
@pytest.mark.parametrize("arm_name", ARMS)
def test_result_independent_of_block_split(mma, big, arm_name):
    arm = getattr(big, arm_name)
    lengths = verify_lengths(33025, 3)
    q = queries_for(3, 77)
    ref, prod_err = reference(arm, q, lengths)
    for bt in (None, 64, 1024, 8192):
        out = arm.run(mma, q, lengths, block_tokens=bt)
        mx.eval(out)
        assert_a1(judge(out, ref, prod_err), f"(block_tokens={bt})")


# [new] output contract: shape, dtype, batch of one, all k
@pytest.mark.parametrize("arm_name", ARMS)
@pytest.mark.parametrize("k", KS)
def test_output_shape_and_dtype(mma, small, arm_name, k):
    arm = getattr(small, arm_name)
    out = arm.run(mma, queries_for(k, 5), verify_lengths(1024, k))
    mx.eval(out)
    assert out.shape == (1, NQ, k, D)
    assert out.dtype == mx.bfloat16


# [new] query j depends only on its own prefix: a block's result for query j matches the same query run alone as k = 1
@pytest.mark.parametrize("arm_name", ARMS)
def test_block_rows_match_single_query_calls(mma, big, arm_name):
    arm = getattr(big, arm_name)
    lengths = verify_lengths(9000, 5)
    q = queries_for(5, 99)
    ref, prod_err = reference(arm, q, lengths)
    block = np.array(arm.run(mma, q, lengths).astype(mx.float32))
    for j in range(5):
        single = np.array(
            arm.run(mma, q[:, :, j : j + 1, :], [lengths[j]]).astype(mx.float32)
        )
        assert rel_err(single[:, :, 0, :], ref[:, :, j, :]) <= 2 * max(prod_err, FLOOR)
        assert rel_err(block[:, :, j, :], ref[:, :, j, :]) <= 2 * max(prod_err, FLOOR)


# ================================================================== A2 planted tokens, A3 negative controls
GAPPED = (-4000, -1717, -500, -33, 0)


def planted_lengths(T, k, pattern):
    if pattern == "consecutive":
        return verify_lengths(T, k)
    return [T + d for d in GAPPED[-k:]]


@lru_cache(maxsize=None)
def planted_case(arm_name, k, pattern, T=9000, amp=30.0):
    """A cache in which one key dominates attention: at p_j = lengths[j]-1 for each query and at e = lengths[0] (visible only to
    queries j >= 1; the bait a leaking / off-by-one mask would read). Every planted key = amp * u, every query = 16 * u (scale
    1/16 -> planted score amp * cos(codec error) ~ 30 vs ~N(0, 1) for random keys). The cache holds 3 more tokens than the last length,
    so the leak control (lengths + 1) is legal."""
    lengths = planted_lengths(T, k, pattern)
    u = np.random.default_rng(1234 + k).standard_normal(D).astype(np.float32)
    u /= np.linalg.norm(u)
    positions = sorted({L - 1 for L in lengths} | {lengths[0]})
    offset = lengths[-1] + 3
    assert offset > max(positions) + 1
    Kn, Vn = np.array(normal((1, NKV, offset, D), 11).astype(mx.float32)), np.array(
        normal((1, NKV, offset, D), 12).astype(mx.float32)
    )
    Kn[:, :, positions, :] = amp * u
    K, V = mx.array(Kn).astype(DT), mx.array(Vn).astype(DT)
    arm = ARM_MAKERS[arm_name](K, V, offset + 61)
    q = mx.array(np.broadcast_to(16.0 * u, (1, NQ, k, D)).copy()).astype(DT)
    return arm, q, lengths, positions


# [invariant] the planted reference is the closed form: query j = mean of the visible planted values (equal weights)
@pytest.mark.parametrize("arm_name", ARMS)
@pytest.mark.parametrize("pattern", ("consecutive", "gapped"))
@pytest.mark.parametrize("k", (1, 2, 3, 5))
def test_instrument_planted_reference_is_the_closed_form(arm_name, k, pattern):
    arm, q, lengths, positions = planted_case(arm_name, k, pattern)
    ref = ref_attention(arm.Kn, arm.Vn, q, lengths)
    for j, L in enumerate(lengths):
        vis = [p for p in positions if p < L]
        assert vis, "the planted case must have a visible planted key for every query"
        for h in range(NKV):
            want = arm.Vn[0, h, vis, :].mean(axis=0)
            assert np.allclose(ref[0, h * R : (h + 1) * R, j, :], want, atol=1e-4), (
                j,
                h,
            )


# [invariant] the A1 criterion can tell every negative control from the truth: the control applied to the *reference* itself fails A1
# (so a kernel that implements the control cannot pass by insensitivity of the instrument)
@pytest.mark.parametrize("arm_name", ARMS)
@pytest.mark.parametrize("control", ("N1", "N2", "N3"))
@pytest.mark.parametrize("pattern", ("consecutive", "gapped"))
@pytest.mark.parametrize("k", (1, 3, 5))
def test_instrument_a1_detects_each_control_on_the_planted_case(
    arm_name, control, k, pattern
):
    arm, q, lengths, _ = planted_case(arm_name, k, pattern)
    ref, prod_err = reference(arm, q, lengths)
    assert judge(mx.array(ref).astype(mx.bfloat16), ref, prod_err)[
        "a1"
    ]  # the true answer, rounded to bf16, passes
    kw, ctl_lengths = control_kwargs_lengths(control, lengths)
    Kn, Vn = (
        (
            np.roll(arm.Kn, -kw["_kv_head_shift"], axis=1),
            np.roll(arm.Vn, -kw["_kv_head_shift"], axis=1),
        )
        if kw
        else (arm.Kn, arm.Vn)
    )
    perturbed = ref_attention(Kn, Vn, q, ctl_lengths)
    assert not judge(mx.array(perturbed).astype(mx.bfloat16), ref, prod_err)["a1"]


# [new] A2: planted keys at lengths[j]-1 and the bait at lengths[0] (visible only to j >= 1): A1 holds on both arms
@pytest.mark.parametrize("arm_name", ARMS)
@pytest.mark.parametrize("pattern", ("consecutive", "gapped"))
@pytest.mark.parametrize("k", (1, 2, 3, 5))
def test_a2_planted_tokens(mma, arm_name, k, pattern):
    arm, q, lengths, _ = planted_case(arm_name, k, pattern)
    assert_a1(check(mma, arm, q, lengths), f"(planted {arm_name}, k={k}, {pattern})")


def control_kwargs_lengths(name, lengths):
    if name == "N1":
        return {"_kv_head_shift": 1}, lengths
    if name == "N2":
        return {}, [L + 1 for L in lengths]  # leak: one token too many
    return {}, [L - 1 for L in lengths]  # N3, drop: one token too few


# [new] A3: each negative control fails A1 on the planted case, while the unperturbed call passes (so it fails for the right reason)
@pytest.mark.parametrize("arm_name", ARMS)
@pytest.mark.parametrize("control", ("N1", "N2", "N3"))
@pytest.mark.parametrize("pattern", ("consecutive", "gapped"))
@pytest.mark.parametrize("k", (1, 3, 5))
def test_a3_negative_controls_fail_a1(mma, arm_name, control, k, pattern):
    arm, q, lengths, _ = planted_case(arm_name, k, pattern)
    ref, prod_err = reference(arm, q, lengths)
    ok = arm.run(mma, q, lengths)
    mx.eval(ok)
    assert_a1(judge(ok, ref, prod_err), "(unperturbed planted call)")
    kw, ctl_lengths = control_kwargs_lengths(control, lengths)
    bad = arm.run(mma, q, ctl_lengths, **kw)
    mx.eval(bad)
    res = judge(bad, ref, prod_err)
    assert not res[
        "a1"
    ], f"negative control {control} did not fail A1 ({arm_name}, k={k}, {pattern}): rel_err {res['rel_err']:.4g}"


# [new] N1 also on random data (wrong KV head for every query group): fails A1; and the unshifted call passes
@pytest.mark.parametrize("arm_name", ARMS)
@pytest.mark.parametrize("shift", (1, 2, 3))
def test_a3_head_shift_fails_a1_on_random_data(mma, big, arm_name, shift):
    arm = getattr(big, arm_name)
    lengths = verify_lengths(9000, 3)
    q = queries_for(3, 123)
    ref, prod_err = reference(arm, q, lengths)
    bad = arm.run(mma, q, lengths, _kv_head_shift=shift)
    mx.eval(bad)
    assert not judge(bad, ref, prod_err)["a1"]


# [new] _kv_head_shift=0 is the identity (the default)
@pytest.mark.parametrize("arm_name", ARMS)
def test_head_shift_zero_is_the_default(mma, small, arm_name):
    arm = getattr(small, arm_name)
    lengths = verify_lengths(1024, 3)
    q = queries_for(3, 8)
    a, b = arm.run(mma, q, lengths), arm.run(mma, q, lengths, _kv_head_shift=0)
    mx.eval(a, b)
    assert bool(mx.all(a == b).item())


# ================================================================== A5 no layout copy; preconditions
# [new] the kernel's inputs are the full-cap buffers themselves: capacity > valid length, per-head stride = capacity, and a call with
# exactly those arrays (no slice, no contiguous copy made by the caller) satisfies A1 for lengths below the capacity
@pytest.mark.parametrize("arm_name", ARMS)
def test_a5_full_cap_buffers_are_the_inputs(mma, big, arm_name):
    arm = getattr(big, arm_name)
    if arm_name == "tq4":
        c = arm.cache
        assert c.keys.norms.shape == (1, NKV, BIG_CAP) and c.keys.indices.shape == (
            1,
            NKV,
            BIG_CAP,
            32,
        )
        assert c.values.norms.shape == (1, NKV, BIG_CAP) and c.values.indices.shape == (
            1,
            NKV,
            BIG_CAP,
            32,
        )
        assert c.offset == BIG_T < BIG_CAP
        before = (c.keys, c.values)
    else:
        assert (
            arm.K.shape == arm.V.shape == (1, NKV, BIG_CAP, D)
            and arm.valid == BIG_T < BIG_CAP
        )
        before = (arm.K, arm.V)
    assert_a1(
        check(mma, arm, queries_for(3, 5), verify_lengths(BIG_T, 3)),
        f"({arm_name} on the full-cap buffers)",
    )
    after = (arm.cache.keys, arm.cache.values) if arm_name == "tq4" else (arm.K, arm.V)
    assert all(
        a is b for a, b in zip(before, after)
    ), "the kernel replaced / shrank the caller's buffers"


# [new] A5, in place: bytes beyond lengths[-1] (inside the preallocated capacity) are never read for effect - large finite garbage
# there must not change the answer
@pytest.mark.parametrize("arm_name", ARMS)
def test_a5_reads_nothing_beyond_the_last_length(mma, arm_name):
    T, cap, Lmax = 9000, 9300, 8500
    K, V = normal((1, NKV, T, D), 21), normal((1, NKV, T, D), 22)
    arm = ARM_MAKERS[arm_name](K, V, cap)
    lengths = [Lmax - 4, Lmax - 2, Lmax]
    q = queries_for(3, 24)
    ref, prod_err = reference(arm, q, lengths)
    if arm_name == "tq4":
        for st in (arm.cache.keys, arm.cache.values):
            # uint32 garbage: random int32 bit patterns viewed as uint32 (mx.random.bits does not exist in MLX 0.32.4)
            st.indices[:, :, Lmax:, :] = mx.random.randint(
                -(2**31),
                2**31 - 1,
                (1, NKV, cap - Lmax, 32),
                dtype=mx.int32,
                key=mx.random.key(31),
            ).view(mx.uint32)
            st.norms[:, :, Lmax:] = mx.full(
                (1, NKV, cap - Lmax), 1000.0, dtype=mx.float16
            )
        mx.eval(arm.cache.keys, arm.cache.values)
    else:
        for X in (arm.K, arm.V):
            X[:, :, Lmax:, :] = mx.full((1, NKV, cap - Lmax, D), 1e4, dtype=DT)
        mx.eval(arm.K, arm.V)
    out = arm.run(mma, q, lengths)
    mx.eval(out)
    assert_a1(judge(out, ref, prod_err), f"({arm_name}, garbage beyond length {Lmax})")


# [new] A5, no materialization: the kernel's peak memory above the live arrays stays well below a copy of the K/V bytes it reads
@pytest.mark.parametrize("arm_name", ARMS)
def test_a5_no_layout_copy_or_dequantized_materialization(mma, big, arm_name):
    arm = getattr(big, arm_name)
    lengths = verify_lengths(BIG_T, 3)
    q = queries_for(3, 6)
    mx.eval(
        arm.run(mma, q, lengths, block_tokens=2048)
    )  # compile / warm outside the measurement
    mx.synchronize()
    if arm_name == "tq4":
        kv_bytes = sum(
            a.nbytes for st in (arm.cache.keys, arm.cache.values) for a in st
        )
    else:
        kv_bytes = arm.K.nbytes + arm.V.nbytes
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    out = arm.run(mma, q, lengths, block_tokens=2048)
    mx.eval(out)
    mx.synchronize()
    extra = mx.get_peak_memory() - base
    assert (
        extra < 0.4 * kv_bytes
    ), f"peak {extra / 1e6:.1f} MB above the live arrays vs K/V bytes {kv_bytes / 1e6:.1f} MB"


# [new] preconditions are asserts: lengths[-1] <= cache.offset / keys.shape[2]; lengths[0] >= 1
@pytest.mark.parametrize("arm_name", ARMS)
def test_preconditions_are_asserted(mma, small, arm_name):
    arm = getattr(small, arm_name)
    q = queries_for(2, 9)
    too_long = (
        [1024, 1025] if arm_name == "tq4" else [1100, 1101]
    )  # TQ4: cache.offset == 1024; native16: keys.shape[2] == 1100
    with pytest.raises(AssertionError):
        arm.run(mma, q, too_long)
    with pytest.raises(AssertionError):
        arm.run(mma, q, [0, 5])  # lengths[0] >= 1


# [new] native16 precondition is the buffer's capacity, not the valid-token count: lengths may reach keys.shape[2]
def test_native16_precondition_is_the_buffer_capacity(mma, small):
    arm = small.n16
    q = queries_for(2, 9)
    out = arm.run(
        mma, q, [1099, 1100]
    )  # capacity 1100 (zeros beyond 1024): allowed, finite
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out.astype(mx.float32))).item())
    with pytest.raises(AssertionError):
        arm.run(mma, q, [1100, 1101])


# [new] the fragment element mapping the kernel relies on holds on this GPU (spec: verify the mapping with a unit test)
def test_fragment_mapping_ok_on_this_gpu(mma):
    assert mma.fragment_mapping_ok() is True
