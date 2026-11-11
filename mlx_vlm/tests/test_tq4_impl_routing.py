"""M72 1.3 A2 (routing): one test per routing row of the specification — the path taken (spies on the ``tq4_fast``
kernels), the route counter and the output standard (A1 tolerance, or bit identity to today's call).

Rows: prefill chunk -> p2s; short prefill chunk (L 2..5, T_eff > 2048) -> mma_prefill; plain decode -> d2 (T_eff > 2048)
/ single_pass; width-1 verify (commit-pending) -> d2 / single_pass; verify k 2..5 -> mma (m72_v1) / d2_verify
(m72_exact) / per_query:short (m72_exact, lengths[0] <= 2048); k > 5 and gapped -> per_query:{k_gt_5, gapped}; native
last layer -> native:{prefill, verify, decode} with fused_v1 (force_fused) and joint_v1; out of scope ->
fallback:<reason> with today's output.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Metal GPU required for the TQ4 kernels"
)

from test_tq4_impl_support import (  # noqa: E402
    NKV,
    NQ,
    SCALE,
    D,
    assert_a1,
    bit_equal,
    caches_bit_equal,
    fill_cache,
    judge,
    lm24,
    np_for,
    prefill,
    production_lengths,
    production_prefill,
    prompt,
    proxies,
    queries,
    ref_lengths,
    ref_prefill,
    singleton,
    tq4_cache,
)

from mlx_vlm import attention_policy as ap  # noqa: E402
from mlx_vlm import mtp_verify_scan as mv  # noqa: E402
from mlx_vlm import tq4_attention as t4  # noqa: E402
from mlx_vlm import tq4_fast as f  # noqa: E402
from mlx_vlm import turboquant as tq  # noqa: E402
from mlx_vlm.models import base  # noqa: E402

CAP = 8192
T_LONG = 4100  # > 2048: the D2 / mma_prefill / d2_verify side
T_SHORT = 1500  # <= 2048


@pytest.fixture(scope="module")
def cache():
    return fill_cache(T_LONG + 300, CAP, seed=5)


class Delta:
    """Route-counter delta over a block."""

    def __enter__(self):
        self.before = f.route_counts()
        return self

    def __exit__(self, *exc):
        after = f.route_counts()
        self.d = {
            k: after.get(k, 0) - self.before.get(k, 0)
            for k in after
            if after.get(k, 0) != self.before.get(k, 0)
        }
        return False


@pytest.fixture
def spies(monkeypatch):
    calls = []
    for name in ("prefill_attention", "decode_attention", "verify_attention"):
        real = getattr(f, name)

        def wrap(*a, _real=real, _name=name, **kw):
            calls.append((_name, kw.get("exact", False)))
            return _real(*a, **kw)

        monkeypatch.setattr(f, name, wrap)
    return calls


def mode(name="m72_v1"):
    return t4.Tq4Mode(name)


def attend(m, cache, q, T, mask="causal", sinks=None):
    ks, vs = proxies(cache, T)
    return base.tq4_scaled_dot_product_attention(
        q, ks, vs, cache, SCALE, mask, sinks, tq4=m
    )


def today(cache, q, T, mask="causal", sinks=None):
    ks, vs = proxies(cache, T)
    return base.scaled_dot_product_attention(
        q, ks, vs, cache=cache, scale=SCALE, mask=mask, sinks=sinks
    )


# ================================================================== prefill
@pytest.mark.parametrize(
    "T,L",
    [
        (T_LONG, 512),
        (T_SHORT, 37),
        (T_LONG, 6),
        (T_LONG, 7),
        (T_LONG, 8),
        (T_LONG, 9),
        (T_SHORT, 3),
        (T_SHORT, 2),
        (T_SHORT, 8),
    ],
)
def test_prefill_chunk_takes_p2s(cache, spies, T, L):
    q = queries(L, T + L)
    with Delta() as d:
        out = attend(mode(), cache, q, T)
        mx.eval(out)
    assert spies == [("prefill_attention", False)]
    assert d.d == {"p2s": 1}
    Kd, Vd = np_for(cache, T)
    assert_a1(
        judge(out, ref_prefill(Kd, Vd, q, T), production_prefill(cache, q, T)),
        f"(p2s T={T} L={L})",
    )


@pytest.mark.parametrize("L", (2, 3, 5))
def test_short_prefill_chunk_takes_mma_prefill(cache, spies, L):
    T = T_LONG
    q = queries(L, 77 + L)
    with Delta() as d:
        out = attend(mode(), cache, q, T)
        mx.eval(out)
    assert spies == [("verify_attention", False)]
    assert d.d == {"mma_prefill": 1}
    lengths = list(
        range(T - L + 1, T + 1)
    )  # same causal semantics: query i sees [0, T - L + i]
    Kd, Vd = np_for(cache, T)
    assert_a1(
        judge(
            out,
            ref_lengths(Kd, Vd, q, lengths),
            production_lengths(cache, q, lengths),
        ),
        f"(mma_prefill L={L})",
    )
    # also within A1 of the causal prefill reference
    assert_a1(judge(out, ref_prefill(Kd, Vd, q, T), production_prefill(cache, q, T)))


@pytest.mark.parametrize("name", ("m72_v1", "m72_exact"))
def test_short_prefill_is_mode_independent(cache, name):
    q = queries(3, 5)
    a = attend(mode(name), cache, q, T_LONG)
    b = attend(mode("m72_v1"), cache, q, T_LONG)
    assert bit_equal(a, b)


# ================================================================== decode
@pytest.mark.parametrize(
    "T,route",
    [(T_LONG, "d2"), (2049, "d2"), (2048, "single_pass"), (T_SHORT, "single_pass")],
)
def test_plain_decode_routes_and_is_bit_identical(cache, spies, T, route):
    q = queries(1, T)
    with Delta() as d:
        out = attend(mode(), cache, q, T, mask=None)
    assert spies == [("decode_attention", False)]
    assert d.d == {route: 1}
    assert bit_equal(out, today(cache, q, T, mask=None))


def test_decode_uses_t_eff_of_the_passed_state_not_the_offset(cache):
    """The per-query fallback slices the state below the offset: the kernel must attend to T_eff tokens."""
    T = T_LONG - 50
    assert cache.offset > T
    q = queries(1, 9)
    with Delta() as d:
        out = attend(mode(), cache, q, T, mask=None)
    assert d.d == {"d2": 1}
    assert bit_equal(out, singleton(cache, q, T))
    assert not bit_equal(out, singleton(cache, q, cache.offset))


# ================================================================== verify (direct, on a TQ4 cache)
def verify(m, cache, q, T):
    ks, vs = proxies(cache, T)
    return m.verify(q, ks, vs, cache=cache, scale=SCALE, mask="causal")


@pytest.mark.parametrize("k", (2, 3, 4, 5))
@pytest.mark.parametrize("T", (T_LONG, T_SHORT))
def test_verify_m72_v1_takes_mma(cache, spies, k, T):
    q = queries(k, 13 * k + T)
    with Delta() as d:
        out = verify(mode("m72_v1"), cache, q, T)
        mx.eval(out)
    assert spies == [("verify_attention", False)]
    assert d.d == {"mma": 1}
    lengths = list(range(T - k + 1, T + 1))
    Kd, Vd = np_for(cache, T)
    assert_a1(
        judge(
            out, ref_lengths(Kd, Vd, q, lengths), production_lengths(cache, q, lengths)
        ),
        f"(mma k={k} T={T})",
    )


@pytest.mark.parametrize("k", (2, 3, 4, 5))
def test_verify_m72_exact_takes_d2_verify_bit_identical(cache, spies, k):
    q = queries(k, 31 * k)
    with Delta() as d:
        out = verify(mode("m72_exact"), cache, q, T_LONG)
    assert spies == [("verify_attention", True)]
    assert d.d == {"d2_verify": 1}
    assert bit_equal(
        out, production_lengths(cache, q, list(range(T_LONG - k + 1, T_LONG + 1)))
    )


@pytest.mark.parametrize("T", (2050, T_SHORT))
def test_verify_m72_exact_short_takes_the_per_query_loop(cache, spies, T):
    """lengths[0] <= 2048 under m72_exact: today's per-query loop (exact); the hook returns None."""
    k = 3
    q = queries(k, T)
    lengths = list(range(T - k + 1, T + 1))
    assert lengths[0] <= 2048
    with Delta() as d:
        assert verify(mode("m72_exact"), cache, q, T) is None
    assert d.d == {"per_query:short": 1} and spies == []
    ks, vs = proxies(cache, T)
    with Delta() as d:
        out = mode("m72_exact").verify_block(cache, q, ks, vs, SCALE, lengths)
    singles = [("single_pass" if L <= 2048 else "d2") for L in lengths]
    want = {"per_query:short": 1}
    for s in singles:
        want[s] = want.get(s, 0) + 1
    assert d.d == want
    assert bit_equal(out, production_lengths(cache, q, lengths))


@pytest.mark.parametrize("name", ("m72_v1", "m72_exact"))
def test_verify_k_gt_5_takes_the_per_query_loop(cache, name):
    k, T = 6, T_LONG
    q = queries(k, 66)
    lengths = list(range(T - k + 1, T + 1))
    with Delta() as d:
        assert verify(mode(name), cache, q, T) is None
    assert d.d == {"per_query:k_gt_5": 1}
    ks, vs = proxies(cache, T)
    with Delta() as d:
        out = mode(name).verify_block(cache, q, ks, vs, SCALE, lengths)
    assert d.d == {"per_query:k_gt_5": 1, "d2": k}
    for j, L in enumerate(
        lengths
    ):  # each query bit-identical to a singleton at its own T_eff above 2048
        assert bit_equal(out[:, :, j : j + 1], singleton(cache, q[:, :, j : j + 1], L))


@pytest.mark.parametrize("name", ("m72_v1", "m72_exact"))
def test_verify_gapped_lengths_take_the_per_query_loop(cache, spies, name):
    lengths = [T_LONG - 400, T_LONG - 33, T_LONG]
    q = queries(3, 4)
    ks, vs = proxies(cache, T_LONG)
    with Delta() as d:
        out = mode(name).verify_block(cache, q, ks, vs, SCALE, lengths)
    assert d.d == {"per_query:gapped": 1, "d2": 3}
    assert [s for s in spies if s[0] == "verify_attention"] == []
    for j, L in enumerate(lengths):
        assert bit_equal(out[:, :, j : j + 1], singleton(cache, q[:, :, j : j + 1], L))


# ================================================================== out of scope -> fallback with today's output
def test_fallback_array_mask(cache):
    T, L = T_LONG, 8
    q = queries(L, 1)
    mask = mx.ones((1, 1, L, T), dtype=mx.bool_)
    with Delta() as d:
        out = attend(mode(), cache, q, T, mask=mask)
    assert d.d == {"fallback:mask": 1}
    assert bit_equal(out, today(cache, q, T, mask=mask))


def test_fallback_sinks(cache):
    T, L = 600, 4
    q = queries(L, 2)
    sinks = mx.zeros((NQ,), dtype=mx.bfloat16)
    with Delta() as d:
        out = attend(mode(), cache, q, T, sinks=sinks)
    assert d.d == {"fallback:sinks": 1}
    assert bit_equal(out, today(cache, q, T, sinks=sinks))


@pytest.mark.parametrize(
    "attr,value,reason",
    [
        ("_decode_2pass_use_legacy", True, "legacy"),
        ("_decode_2pass_num_blocks", 32, "block_override"),
    ],
)
def test_fallback_legacy_flag_and_block_override(attr, value, reason):
    c = fill_cache(3000, 4096, seed=9)
    setattr(c, attr, value)
    q = queries(1, 3)
    with Delta() as d:
        out = attend(mode(), c, q, 3000, mask=None)
    assert d.d == {f"fallback:{reason}": 1}
    assert bit_equal(out, today(c, q, 3000, mask=None))


def test_fallback_batch_cache():
    keys = queries(300, 1)[:, :NKV]
    values = queries(300, 2)[:, :NKV]
    c = tq.BatchTurboQuantKVCache([0], bits=4)
    ks, vs = c.update_and_fetch(keys, values)
    q = queries(1, 3)
    m = mode()
    with Delta() as d:
        assert m.attend(q, ks, vs, cache=c, scale=SCALE, mask=None) is None
        assert m.verify(queries(3, 3), ks, vs, cache=c, scale=SCALE, mask=None) is None
    assert d.d == {"fallback:batch_cache": 2}
    got = base.tq4_scaled_dot_product_attention(q, ks, vs, c, SCALE, None, tq4=m)
    want = base.scaled_dot_product_attention(q, ks, vs, cache=c, scale=SCALE, mask=None)
    assert bit_equal(got, want)


@pytest.mark.parametrize("dtype,reason", [(mx.float16, "dtype")])
def test_fallback_dtype(cache, dtype, reason):
    q = queries(1, 1).astype(dtype)
    with Delta() as d:
        out = attend(mode(), cache, q, T_LONG, mask=None)
    assert d.d == {f"fallback:{reason}": 1}
    assert bit_equal(out, today(cache, q, T_LONG, mask=None))


def test_non_causal_multi_query_call_falls_back(cache):
    q = queries(4, 1)
    with Delta() as d:
        out = attend(mode(), cache, q, 900, mask=None)
    assert d.d == {"fallback:mask": 1}
    assert bit_equal(out, today(cache, q, 900, mask=None))


# ================================================================== through the model: native last layer, width-1 verify
@pytest.fixture(scope="module")
def stamped():
    """lm24 with the shipped native16 policies stamped (fused_v1, joint_v1) — the arm's load order."""
    lm = lm24()
    model = type("M", (), {"config": lm.config, "language_model": lm})()
    ap.apply_to_model(model, ap.FusedV1Policy())
    vp = mv.JointV1Policy()
    mv.apply_to_model(model, vp)
    return lm, model, vp


@pytest.fixture
def twins(stamped):
    """Two identical mode-off prefills (T = 2600) of the served cache layout (TQ4 + native last layer)."""
    lm, model, _ = stamped
    t4.apply_to_model(model, None)
    ids = prompt(2600)
    a, b = tq4_cache(lm), tq4_cache(lm)
    prefill(lm, a, ids)
    prefill(lm, b, ids)
    assert type(a[1]) is tq.TurboQuantKVCache and type(a[3]).__name__ == "KVCache"
    assert caches_bit_equal(a, b)
    yield a, b
    t4.apply_to_model(model, None)


def test_native_layer_prefill_is_fused_v1_and_counted(stamped, monkeypatch):
    lm, model, _ = stamped
    t4.apply_to_model(model, "m72_v1")
    calls = []
    real = mx.fast.scaled_dot_product_attention

    def rec(q, k, v, **kw):
        calls.append((q.shape[-2], k.shape[-2], k.dtype, kw.get("force_fused", False)))
        return real(q, k, v, **kw)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", rec)
    try:
        c = tq4_cache(lm)
        policy = lm.model.layers[3].self_attn.attention_policy
        forced0 = policy.forced
        with Delta() as d:
            prefill(lm, c, prompt(1536))
    finally:
        t4.apply_to_model(model, None)
    assert d.d == {"p2s": 3, "native:prefill": 3}
    native = [x for x in calls if x[2] == mx.bfloat16 and x[0] == 512]
    p2s = [x for x in calls if x[2] == mx.float16]
    assert len(native) == 3 and all(
        x[3] for x in native
    )  # fused_v1: force_fused on the native layer
    assert len(p2s) == 3 and all(x[3] for x in p2s)
    assert policy.forced - forced0 == 3


def test_native_layer_verify_is_joint_v1_and_tq4_layer_is_mma(stamped, twins):
    lm, model, vp = stamped
    a, _ = twins
    t4.apply_to_model(model, "m72_v1")
    before = vp.counters()
    with Delta() as d:
        out = lm(mx.array([[5, 9, 11]]), cache=a, speculative_verify=True)
        mx.eval(out.logits)
    after = vp.since(before)
    assert d.d == {"mma": 1, "native:verify": 1}
    assert after["verify_blocks_joint_v1"] + after["verify_blocks_straddle"] == 1
    assert after["verify_blocks_per_query"] == 0  # the TQ4 layer never reaches joint_v1


def test_m72_exact_verify_through_the_model_is_bit_identical_to_mode_off(
    stamped, twins
):
    lm, model, _ = stamped
    a, b = twins
    tokens = mx.array([[5, 9, 11, 2]])
    t4.apply_to_model(model, "m72_exact")
    with Delta() as d:
        on = lm(tokens, cache=a, speculative_verify=True)
        mx.eval(on.logits)
    assert d.d == {"d2_verify": 1, "native:verify": 1}
    t4.apply_to_model(model, None)
    off = lm(tokens, cache=b, speculative_verify=True)
    mx.eval(off.logits)
    assert bit_equal(on.logits, off.logits)


def test_width_1_commit_pending_verify_takes_d2(stamped, twins):
    from mlx_vlm.speculative.mtp import _mtp_commit_pending_token

    lm, model, _ = stamped
    a, b = twins
    t4.apply_to_model(model, "m72_v1")
    with Delta() as d:
        _mtp_commit_pending_token(lm, a, 7, lambda x: mx.argmax(x, axis=-1), mx.int32)
    assert d.d.get("d2") == 1 and d.d.get("native:decode") == 1
    assert set(d.d) == {"d2", "native:decode"}
    t4.apply_to_model(model, None)
    _mtp_commit_pending_token(lm, b, 7, lambda x: mx.argmax(x, axis=-1), mx.int32)
    assert caches_bit_equal(a, b)


def test_plain_decode_through_the_model_is_bit_identical(stamped, twins):
    lm, model, _ = stamped
    a, b = twins
    t4.apply_to_model(model, "m72_v1")
    with Delta() as d:
        on = lm(mx.array([[4]]), cache=a)
        mx.eval(on.logits)
    assert d.d == {"d2": 1, "native:decode": 1}
    t4.apply_to_model(model, None)
    off = lm(mx.array([[4]]), cache=b)
    assert bit_equal(on.logits, off.logits)


def test_mode_absent_counts_nothing(stamped, twins):
    lm, model, _ = stamped
    a, _ = twins
    t4.apply_to_model(model, None)
    with Delta() as d:
        mx.eval(lm(mx.array([[4]]), cache=a).logits)
        prefill(lm, a, prompt(40))
        out = lm(mx.array([[4, 5, 6]]), cache=a, speculative_verify=True)
        mx.eval(out.logits)
        out.gdn_states.abort()
    assert d.d == {}


def test_route_counter_names_are_the_specified_set(cache):
    allowed = {
        "p2s",
        "mma_prefill",
        "d2",
        "single_pass",
        "mma",
        "d2_verify",
        "per_query:short",
        "per_query:k_gt_5",
        "per_query:gapped",
        "native:prefill",
        "native:verify",
        "native:decode",
    }
    for name in f.route_counts():
        assert name in allowed or name.startswith("fallback:"), name
    assert D == 256
