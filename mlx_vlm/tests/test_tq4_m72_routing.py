"""M72 1.3 A2: routing, one test per row of the specification's "Regimes and routing" table, driven through the model the way the
server drives it (prefill chunks, plain decode, ``speculative_verify`` blocks) on a synthetic Qwen3.5-family model whose attention
shape is the pick's (24 / 4 / 256, bf16): TQ4 full-attention layers + a last full-attention layer that ``maybe_quantize_kv_cache``
leaves native (the analogue of layer 63). Each row asserts the counter (``tq4_fast.route_counts()``), that no other route counter
moved, and the output standard (bit-identical to the unmodified path where the table says so, near it otherwise).

Two models with identical weights are prefilled mode-off to identical states; the mode (and the shipped native policies, as the
loader stamps them) is applied to one, so every comparison is mode-on vs today's path on the same state.

Every test is classed in its comment: [new] exercises new behaviour, [invariant] preserves an invariant of today's code,
[fix] reproduces a failure the specification fixes.
"""

from __future__ import annotations

import builtins
import contextlib
from types import SimpleNamespace

import pytest
from test_tq4_m72_support import (
    BF16,
    MODES,
    counted,
    counts,
    mode_module,
    model_cache,
    mx,
    needs_metal,
    prefill,
    tiny_model,
    tq,
    tq4_layers,
)

from mlx_vlm import attention_policy as ap
from mlx_vlm import mtp_verify_scan as mv

pytestmark = needs_metal

NATIVE = {"native:prefill", "native:verify", "native:decode"}
CAP = 4096


# ------------------------------------------------------------------ builders
def stamp_policies(lm):
    """What the loader stamps before the mode: fused_v1 + joint_v1 on every attention module."""
    ap.apply_to_model(lm, ap.FusedV1Policy())
    mv.apply_to_model(lm, mv.JointV1Policy())


def build(
    T,
    *,
    mode=None,
    layers=4,
    dtype=BF16,
    cap=CAP,
    chunk=512,
    apply_after_prefill=True,
    kv="tq4",
):
    lm = tiny_model(layers=layers, dtype=dtype)
    cache = model_cache(lm, kv, cap)
    last = None
    if apply_after_prefill:
        last = prefill(lm, cache, T, chunk=chunk)
    stamp_policies(lm)
    if mode:
        mode_module().apply_to_model(lm, mode)
    return SimpleNamespace(lm=lm, cache=cache, last=last, T=T)


def pair(T, mode, **kw):
    """(mode-on, mode-off) models on identical mode-off-prefilled states."""
    a, b = build(T, mode=mode, **kw), build(T, mode=None, **kw)
    for ca, cb in zip(a.cache, b.cache):
        if isinstance(ca, tq.TurboQuantKVCache):
            assert ca.offset == cb.offset == T
            assert mx.array_equal(
                ca.keys.indices[:, :, :T], cb.keys.indices[:, :, :T]
            ).item()
    return a, b


def step(m, n, *, verify=False, seed=5):
    ids = mx.array([[(seed + 3 * i) % 64 for i in range(n)]], dtype=mx.int32)
    kw = {"speculative_verify": True} if verify else {}
    out = m.lm(ids, cache=m.cache, **kw)
    mx.eval(out.logits)
    if (
        verify
    ):  # close the speculative transaction (every token accepted) so the next call starts clean
        out.gdn_states.commit(n)
    return out.logits


def same(a, b):
    return a.shape == b.shape and bool(mx.array_equal(a, b).item())


def near(a, b, tol=0.05):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (
        float(mx.max(mx.abs(a - b)).item() / max(float(mx.max(mx.abs(b)).item()), 1e-6))
        <= tol
    )


def matches(key, patterns):
    return any(
        key == p or (p.endswith("*") and key.startswith(p[:-1])) for p in patterns
    )


def assert_routes(delta, expect, allow=()):
    """Every expected counter moved; no counter outside expect + allow + the native-layer ones moved."""
    for p in expect:
        assert any(matches(k, [p]) and v > 0 for k, v in delta.items()), (p, delta)
    extra = {
        k: v for k, v in delta.items() if not matches(k, [*expect, *allow, *NATIVE])
    }
    assert (
        not extra
    ), f"unexpected route counters {extra} (expected {sorted(expect)}), delta {delta}"


# ================================================================== prefill rows
# [new] prefill chunk (L > 1, not a verify, not the short row): p2s; the output is near today's prefill
def max_rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float(
        mx.max(mx.abs(a - b)).item() / max(float(mx.max(mx.abs(b)).item()), 1e-30)
    )


# amended (delivery 3): the specification's 2x rule at model level. The chunk's last-token logits: mode-on TQ4 vs mode-off TQ4
# (d_on) must be <= 2 x the mode-off TQ4 vs the same model with native16 caches (d_floor); all three are in the message
@pytest.mark.parametrize("mode", MODES)
def test_prefill_chunk_takes_p2s(mode, capsys):
    a, b = pair(2100, mode)
    n = build(2100, mode=None, kv="native")
    with counted() as c:
        la = step(a, 512)
    lb, ln = step(b, 512), step(n, 512)
    assert_routes(c.delta, {"p2s"})
    d_on, d_floor = max_rel(la[:, -1], lb[:, -1]), max_rel(lb[:, -1], ln[:, -1])
    with capsys.disabled():
        print(
            f"\n[p2s 2x rule {mode}] on-vs-off {d_on:.5f}  off-vs-native16 {d_floor:.5f}  bound 2x = {2 * d_floor:.5f}"
        )
    assert d_on <= 2 * d_floor, (d_on, d_floor)


# [new] the first chunks of a prompt (T_eff <= 2048) are p2s too: the short-chunk row needs T_eff > 2048
@pytest.mark.parametrize("mode", MODES)
def test_prefill_below_2048_is_p2s_not_mma_prefill(mode):
    a = build(0, mode=mode, apply_after_prefill=False)
    with counted() as c:
        prefill(a.lm, a.cache, 1024)
    assert_routes(c.delta, {"p2s"})


# [new] short prefill chunk row: 2 <= L <= 5 and T_eff > 2048, not a verify: the verify kernel, causal within the chunk
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("L", (2, 3, 4, 5))
def test_short_prefill_chunk_takes_mma_prefill(mode, L):
    a, b = pair(2100, mode)
    with counted() as c:
        la = step(a, L)
    lb = step(b, L)
    assert_routes(c.delta, {"mma_prefill"})
    assert near(la, lb)


# [new] L = 6 is a prefill chunk again (p2s); L = 1 is a plain decode (d2)
@pytest.mark.parametrize("mode", MODES)
def test_chunk_of_six_is_p2s_and_chunk_of_one_is_d2(mode):
    a, _ = pair(2100, mode)
    with counted() as c6:
        step(a, 6)
    assert_routes(c6.delta, {"p2s"})
    with counted() as c1:
        step(a, 1)
    assert_routes(c1.delta, {"d2"})


# [new] boundary T_eff = 2,048 / 2,049 for the short row: the chunk's key length after the write decides
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("prefix,expect", [(2045, "p2s"), (2046, "mma_prefill")])
def test_short_prefill_boundary_at_t_eff_2048(mode, prefix, expect):
    a = build(prefix, mode=mode)
    with counted() as c:
        step(a, 3)  # T_eff = prefix + 3 = 2048 / 2049
    assert_routes(c.delta, {expect})


# [new] a verify block of the same width is NOT the short-prefill row: it takes the verify route
def test_verify_block_is_not_a_short_prefill_chunk():
    a = build(2100, mode="m72_v1")
    with counted() as c:
        step(a, 3, verify=True)
    assert_routes(c.delta, {"mma"})
    assert "mma_prefill" not in c.delta


# ================================================================== decode rows
# [new] plain decode, T_eff > 2048: D2, bit-identical to today's singleton on the same state
@pytest.mark.parametrize("mode", MODES)
def test_plain_decode_above_2048_takes_d2_bit_identical(mode):
    a, b = pair(2100, mode)
    with counted() as c:
        la = step(a, 1)
    assert_routes(c.delta, {"d2"})
    assert same(la, step(b, 1))


# [new] plain decode, T_eff <= 2048: today's single-pass kernel, unchanged
@pytest.mark.parametrize("mode", MODES)
def test_plain_decode_at_or_below_2048_takes_the_single_pass_kernel(mode):
    a, b = pair(1000, mode)
    with counted() as c:
        la = step(a, 1)
    assert_routes(c.delta, {"single_pass"})
    assert same(la, step(b, 1))


# [new] the boundary: T_eff = 2,048 is single_pass, 2,049 is d2
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("prefix,expect", [(2047, "single_pass"), (2048, "d2")])
def test_decode_boundary_at_t_eff_2048(mode, prefix, expect):
    a, b = pair(prefix, mode)
    with counted() as c:
        la = step(a, 1)
    assert_routes(c.delta, {expect})
    assert same(la, step(b, 1))


# [new] width-1 verify (the commit-pending token): as plain decode, through the verifier
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("T,expect", [(2100, "d2"), (1000, "single_pass")])
def test_width_one_verify_routes_as_decode(mode, T, expect):
    a, b = pair(T, mode)
    with counted() as c:
        la = step(a, 1, verify=True)
    assert_routes(c.delta, {expect})
    assert same(la, step(b, 1, verify=True))


# ================================================================== verify rows
# [new] m72_v1: k = 2..5 consecutive: the matrix-unit kernel; logits near the per-query path
@pytest.mark.parametrize("k", (2, 3, 4, 5))
def test_v1_verify_block_takes_mma(k):
    a, b = pair(2100, "m72_v1")
    with counted() as c:
        la = step(a, k, verify=True)
    assert_routes(c.delta, {"mma"})
    assert near(la, step(b, k, verify=True))


# [new] m72_v1 has no length floor for the matrix-unit kernel (the table's row carries no T condition)
def test_v1_verify_block_below_2048_takes_mma():
    a, b = pair(600, "m72_v1")
    with counted() as c:
        la = step(a, 3, verify=True)
    assert_routes(c.delta, {"mma"})
    assert near(la, step(b, 3, verify=True))


# [new] k > 5: the per-query loop, each query bit-identical to a singleton at its own T_eff (> 2048), counted with its reason
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("k", (6, 7))
def test_verify_wider_than_five_takes_the_per_query_loop(mode, k):
    a, b = pair(2100, mode)
    with counted() as c:
        la = step(a, k, verify=True)
    assert_routes(c.delta, {"per_query:k_gt_5", "d2"})
    assert "mma" not in c.delta and "d2_verify" not in c.delta
    assert same(la, step(b, k, verify=True))


# [new] m72_exact: k = 2..5 with lengths[0] > 2048: D2 multi-query, each row bit-identical to today's per-query loop
@pytest.mark.parametrize("k", (2, 3, 4, 5))
def test_exact_verify_block_takes_d2_verify_bit_identical(k):
    a, b = pair(2100, "m72_exact")
    with counted() as c:
        la = step(a, k, verify=True)
    assert_routes(c.delta, {"d2_verify"})
    assert "mma" not in c.delta
    assert same(la, step(b, k, verify=True))


# [new] m72_exact: lengths[0] <= 2048 takes today's per-query loop (exact), counted per_query:short; boundary 2,048 / 2,049
@pytest.mark.parametrize(
    "prefix,k,expect",
    [
        (1000, 3, "per_query:short"),
        (2047, 3, "per_query:short"),
        (2048, 3, "d2_verify"),
    ],
)
def test_exact_verify_short_boundary(prefix, k, expect):
    # lengths = prefix + 1 .. prefix + k: lengths[0] = prefix + 1 (2,048 -> short; 2,049 -> d2_verify)
    a, b = pair(prefix, "m72_exact")
    with counted() as c:
        la = step(a, k, verify=True)
    assert_routes(c.delta, {expect}, allow={"single_pass", "d2"})
    assert same(la, step(b, k, verify=True))


# [new] m72_v1 is not exact, but m72_exact must never touch the matrix-unit kernel
def test_exact_mode_never_counts_mma():
    a, _ = pair(2100, "m72_exact")
    with counted() as c:
        for k in (2, 3, 5):
            step(a, k, verify=True)
    assert "d2_verify" in c.delta and "mma" not in c.delta, c.delta


# ================================================================== native layer (layer 63 analogue)
class Spy:
    def __init__(self):
        self.calls = []


@contextlib.contextmanager
def sdpa_spy():
    spy, real = Spy(), mx.fast.scaled_dot_product_attention

    def wrapped(q, k, v, **kw):
        spy.calls.append(
            SimpleNamespace(
                qlen=q.shape[2], kdtype=k.dtype, klen=k.shape[2], kw=dict(kw)
            )
        )
        return real(q, k, v, **kw)

    mx.fast.scaled_dot_product_attention = wrapped
    try:
        yield spy
    finally:
        mx.fast.scaled_dot_product_attention = real


# [new] the last (native) layer is counted for every regime and keeps the shipped paths
@pytest.mark.parametrize(
    "n,verify,name",
    [
        (512, False, "native:prefill"),
        (3, True, "native:verify"),
        (1, False, "native:decode"),
    ],
)
def test_native_layer_counters(n, verify, name):
    a = build(2100, mode="m72_v1")
    with counted() as c:
        step(a, n, verify=verify)
    assert c.delta.get(name, 0) > 0, c.delta


# [new] native prefill chunk: fused_v1 forces the fused kernel on bf16 keys
def test_native_layer_prefill_is_forced_fused():
    a = build(2100, mode="m72_v1")
    with sdpa_spy() as spy:
        step(a, 512)
    native = [x for x in spy.calls if x.kdtype == BF16]
    assert native and any(x.kw.get("force_fused") is True for x in native), [
        x.kw for x in native
    ]


# [new] native verify block: joint_v1 serves it with one call over the full keys (kv_bits 4 + m72_v1 accepts joint_v1)
def test_native_layer_verify_is_one_joint_call():
    if mv._device_class() not in mv.QUALIFIED_DOMAIN.device_classes:
        pytest.skip("joint_v1 domain is applegpu_g17s only")
    a = build(2100, mode="m72_v1")
    with sdpa_spy() as spy:
        step(a, 3, verify=True)
    native = [x for x in spy.calls if x.kdtype == BF16]
    assert [x.qlen for x in native] == [3], [x.qlen for x in native]
    assert native[0].kw.get("mask") == "causal"


# [new] and the native decode stays the plain SDPA call (one query)
def test_native_layer_decode_is_one_query_call():
    a = build(2100, mode="m72_v1")
    with sdpa_spy() as spy:
        step(a, 1)
    native = [x for x in spy.calls if x.kdtype == BF16]
    assert [x.qlen for x in native] == [1]


# ================================================================== out of scope: today's path, counted, never an error
def stamp_cache(a, **attrs):
    for c in tq4_layers(a.cache):
        for name, value in attrs.items():
            setattr(c, name, value)


# [new] a cache with the legacy decode flag / a block override falls back per call, byte-identical to today
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "attrs",
    [dict(_decode_2pass_use_legacy=True), dict(_decode_2pass_num_blocks=64)],
    ids=["legacy", "block_override"],
)
@pytest.mark.parametrize(
    "n,verify",
    [(1, False), (3, True), (512, False)],
    ids=["decode", "verify3", "prefill512"],
)
def test_cache_flags_fall_back_with_a_counted_reason(mode, attrs, n, verify):
    a, b = pair(2100, mode)
    stamp_cache(a, **attrs)
    stamp_cache(b, **attrs)
    with counted() as c:
        la = step(a, n, verify=verify)
    lb = step(b, n, verify=verify)
    fallbacks = {k: v for k, v in c.delta.items() if k.startswith("fallback:")}
    assert fallbacks, c.delta
    for fast_route in ("p2s", "mma", "mma_prefill", "d2", "d2_verify", "single_pass"):
        assert fast_route not in c.delta, c.delta
    assert same(la, lb)


# [new] queries that are not bf16 (an fp32 model) are out of scope: counted fallback, bit-identical to today, no error
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n,verify", [(1, False), (3, True), (37, False)])
def test_non_bf16_queries_fall_back(mode, n, verify):
    a = build(300, mode=mode, dtype=mx.float32, cap=512)
    b = build(300, mode=None, dtype=mx.float32, cap=512)
    with counted() as c:
        la = step(a, n, verify=verify)
    lb = step(b, n, verify=verify)
    assert any(k.startswith("fallback:") for k in c.delta), c.delta
    for fast_route in ("p2s", "mma", "mma_prefill", "d2", "d2_verify", "single_pass"):
        assert fast_route not in c.delta, c.delta
    assert same(la, lb)


# ================================================================== counters reach the worker's /health (A8, "Route counters")
def find_dicts_with(obj, key):
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], int):
            yield obj
        for v in obj.values():
            yield from find_dicts_with(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from find_dicts_with(v, key)


# [new] after mode-on traffic the worker's /health details carry the per-process route counters (the same numbers
# ``route_counts()`` reports; the field name is free)
class StubMode:
    """What a worker's generator carries when the mode is on: the mode's name and its counters."""

    def __init__(self, name):
        self.name = name

    def counters(self):
        return counts()


def health_body(monkeypatch, generator):
    from fastapi.testclient import TestClient

    import mlx_vlm.server as server
    from mlx_vlm.server.runtime import runtime

    monkeypatch.setattr(runtime, "response_generator", generator, raising=False)
    r = TestClient(server.app).get("/health")
    assert r.status_code == 200, r.text
    return r.json()


# amended (delivery 3): through the app's own health path with a runtime whose generator carries the mode (a stub, no model loaded)
def test_health_details_carry_the_route_counters(monkeypatch):
    a = build(2100, mode="m72_v1")
    step(a, 1)
    step(a, 3, verify=True)
    snapshot = counts()
    assert snapshot.get("d2", 0) > 0 and snapshot.get("mma", 0) > 0
    body = health_body(
        monkeypatch,
        SimpleNamespace(tq4_attention=StubMode("m72_v1"), mtp_verify_policy=None),
    )
    found = list(find_dicts_with(body, "d2"))
    assert found, body
    assert any(all(d.get(k, 0) >= v for k, v in snapshot.items()) for d in found), (
        found,
        snapshot,
    )
    assert "m72_v1" in repr(body)


# [new] with the mode off the details carry neither the mode nor counters
def test_health_details_are_absent_with_the_mode_off(monkeypatch):
    counts()  # import the module so a stale counter dict could exist in the process
    body = health_body(
        monkeypatch, SimpleNamespace(tq4_attention=None, mtp_verify_policy=None)
    )
    assert (
        not list(find_dicts_with(body, "d2"))
        and "m72" not in repr(body)
        and "tq4" not in repr(body)
    ), body


# ================================================================== mode absent: today, byte for byte, nothing imported
# [invariant] with the mode absent no route counter moves and the request path imports neither module
@pytest.mark.parametrize("kv", ("tq4", "native"))
def test_mode_absent_moves_no_counter_and_imports_nothing(monkeypatch, kv):
    before = counts()  # the counters module may be imported here, before the guard
    real_import = builtins.__import__

    def guard(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            "tq4_fast" in (name or "")
            or "tq4_attention" in (name or "")
            or "tq4_fast" in (fromlist or ())
        ):
            raise AssertionError(f"{name} imported on the mode-absent request path")
        return real_import(name, globals, locals, fromlist, level)

    lm = tiny_model(layers=4)
    cache = model_cache(lm, kv, CAP)
    monkeypatch.setattr(builtins, "__import__", guard)
    try:
        prefill(lm, cache, 2100)
        step(SimpleNamespace(lm=lm, cache=cache), 1)
        step(SimpleNamespace(lm=lm, cache=cache), 3, verify=True)
        step(SimpleNamespace(lm=lm, cache=cache), 6, verify=True)
        step(SimpleNamespace(lm=lm, cache=cache), 3)
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
    assert counts() == before


# [invariant] with the mode absent and the shipped native policies stamped, native16 verify/prefill/decode run as today
def test_mode_absent_native16_runs_with_the_shipped_policies():
    lm = tiny_model(layers=4)
    cache = model_cache(lm, "native", CAP)
    stamp_policies(lm)
    prefill(lm, cache, 2100)
    step(SimpleNamespace(lm=lm, cache=cache), 3, verify=True)
    assert not tq4_layers(cache)
