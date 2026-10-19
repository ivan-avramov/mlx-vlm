"""P369 phase 2 (Q18 / Q29 / Q22): runtime verifier-kernel outcome telemetry.

Fork-only file. Every consuming alias of the qwen3_5 verifier (its by-value imports from
``speculative.ops.linear``) is an ENTRY; the implementations those dispatch among are LEAVES, wrapped in
``speculative.ops.linear`` where the dispatchers look them up. A leaf counts only when it returned a
non-``None`` result; every entry call is classified exactly once (entry, family, width, dtype, modules);
the fused ``_decode_quantized_linears_fused`` branch is its own family, apart from the tuple fallback.
"""

import sys
from pathlib import Path

import mlx.core as mx
import pytest
from test_mtp_exactness import (
    PICKS,
    PROMPT,
    _quantized_pair,
    deployed_sampling,
    fake_model,
    scripted,
)
from test_mtp_stop_state import turn

from mlx_vlm.generate.common import PromptCacheState

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))

from p369_phase2 import telemetry  # noqa: E402

metal = pytest.mark.skipif(
    not mx.metal.is_available(), reason="exact kernels are Metal"
)
_PAIRS = {}


def _pair(pick):
    if pick not in _PAIRS:
        _PAIRS[pick] = _quantized_pair(pick)
    return _PAIRS[pick]


def _on_run(pick, n=9, seed=5, temperature=None):
    lm, drafter = _pair(pick)
    model, params = fake_model(lm), dict(deployed_sampling(pick))
    if temperature is not None:
        params["temperature"] = temperature
    plain, _ = turn(model, PROMPT, PromptCacheState(), n=n, seed=seed, params=params)
    plain = list(plain)
    scripted(
        drafter, lambda p: plain[p] if p < len(plain) else None, high=True, vocab=64
    )
    with telemetry.KernelTelemetry(lm) as tel:
        emitted, _ = turn(
            model,
            PROMPT,
            PromptCacheState(),
            drafter=drafter,
            n=n,
            seed=seed,
            params=params,
        )
    return lm, model, tel, list(emitted), plain


@metal
@pytest.mark.parametrize("pick", PICKS)
def test_every_wide_verifier_call_is_classified_once(pick):
    _, _, tel, emitted, plain = _on_run(pick)
    assert emitted == plain
    wide = [c for c in tel.calls if c["width"] > 1]
    assert wide, "no verify-block projection observed"
    assert all(c["family"] != "unclassified" for c in wide), tel.summary()
    assert sum(tel.summary()["by_family"].values()) == len(tel.calls)
    families = {c["family"] for c in wide}
    assert families & {"quantized_affine_optimized", "quantized_linears_optimized"}
    if pick == PICKS[0]:  # the mixed recipe keeps some projections dense
        assert "dense_exact" in families, families


@metal
def test_a_leaf_returning_none_moves_the_classification():
    lm, drafter = _pair(PICKS[1])
    from mlx_vlm.speculative.ops import linear as ops_linear

    real = ops_linear._target_verify_optimized_affine_linear
    ops_linear._target_verify_optimized_affine_linear = lambda *a, **k: None
    try:
        _, _, tel, _, _ = _on_run(PICKS[1])
    finally:
        ops_linear._target_verify_optimized_affine_linear = real
    families = {c["family"] for c in tel.calls if c["width"] > 1}
    assert "quantized_affine_optimized" not in families
    assert families & {"quantized_singleton", "quantized_linears_optimized"}


@metal
@pytest.mark.parametrize("pick", PICKS)
def test_greedy_runs_the_argmax_readout(pick):
    _, _, tel, emitted, plain = _on_run(pick, temperature=0.0)
    assert emitted == plain
    s = tel.summary()
    assert s["speculative_argmax_from_hidden"] > 0
    assert any(f.startswith("argmax_") for f in s["by_family"]), s


def test_probe_calls_are_tagged_and_excluded(monkeypatch):
    class Lm:
        def named_modules(self):
            return []

    tel = telemetry.KernelTelemetry(Lm())
    with tel:
        with tel.probe():
            tel._record("_target_verify_linear", "plain", 1, "float32", ["x"])
        tel._record("_target_verify_linear", "plain", 1, "float32", ["x"])
    assert len(tel.calls) == 1 and len(tel.probe_calls) == 1
    assert len(tel.attributions) == 1  # probe calls attribute nothing (code round 1 H2)


def test_uninstall_restores_every_attribute():
    from mlx_vlm.models.qwen3_5 import speculative_verifier as sv
    from mlx_vlm.speculative.ops import linear as ops_linear

    before = {n: getattr(ops_linear, n) for n in telemetry.LEAVES} | {
        n: getattr(sv, n) for n in telemetry.ENTRIES
    }

    class Lm:
        def named_modules(self):
            return []

    with telemetry.KernelTelemetry(Lm()):
        pass
    after = {n: getattr(ops_linear, n) for n in telemetry.LEAVES} | {
        n: getattr(sv, n) for n in telemetry.ENTRIES
    }
    assert before == after


# ------------------------------------------------------------------ scope + verdict
@metal
@pytest.mark.parametrize("pick", PICKS)
def test_verifier_scope_is_tagged_and_the_verdict_passes(pick):
    lm, _, tel, _, _ = _on_run(pick)
    scopes = {c["scope"] for c in tel.calls}
    assert "verify" in scopes
    assert all(c["width"] == 1 for c in tel.calls if c["scope"] != "verify")
    verdict = telemetry.verdict(tel, telemetry.manifest_from_model(lm))
    assert verdict["pass"], verdict
    assert verdict["verify_calls"] > 0


def _fake(calls, manifest):
    class Lm:
        def named_modules(self):
            return []

    tel = telemetry.KernelTelemetry(Lm())
    tel.attributions = [
        dict(c, module=m, kind=c.get("kind", "linear"))
        for c in calls
        for m in c["modules"]
    ]
    tel.calls = calls
    return telemetry.verdict(tel, manifest)


def _call(family, module="model.layers.0.mlp", width=3, scope="verify"):
    return dict(
        entry="_target_verify_linear",
        family=family,
        width=width,
        dtype="bfloat16",
        modules=[module],
        scope=scope,
    )


OPTIMIZED = ["quantized_affine_optimized", "quantized_linears_optimized"]
QUANT = {
    "model.layers.0.mlp": {
        "kind": "quantized",
        "bits": 4,
        "group_size": 64,
        "expected": {"linear": OPTIMIZED, "argmax": ["argmax_affine_optimized"]},
    }
}
DENSE = {
    "model.layers.0.mlp": {"kind": "dense", "expected": {"linear": ["dense_exact"]}}
}


@pytest.mark.parametrize("family", ["plain_wide", "declined", "error"])
def test_the_verdict_refuses_a_forbidden_verify_family(family):
    assert not _fake([_call(family)], QUANT)["pass"]


def test_the_verdict_refuses_a_family_inconsistent_with_the_manifest():
    assert not _fake([_call("dense_exact")], QUANT)["pass"]
    assert not _fake([_call("quantized_affine_optimized")], DENSE)["pass"]
    assert _fake([_call("dense_exact")], DENSE)["pass"]
    assert _fake([_call("quantized_affine_optimized")], QUANT)["pass"]


def test_the_verdict_refuses_an_unknown_module_and_no_verify_calls():
    assert not _fake([_call("quantized_affine_optimized", module="?")], QUANT)["pass"]
    assert not _fake([_call("plain", width=1, scope="other")], QUANT)["pass"]


# ------------------------------------------------------------------ code round 1 H2: expected families + coverage
def test_the_verdict_refuses_a_fallback_where_the_manifest_expects_an_exact_kernel():
    for family in ("fallback_singletons", "fallback_timewise", "quantized_singleton"):
        got = _fake([_call(family)], QUANT)
        assert not got["pass"] and any(
            "unexpected family" in p for p in got["problems"]
        ), (family, got)
    expected_fallback = {
        "model.layers.0.mlp": {
            "kind": "dense",
            "expected": {"linear": ["fallback_singletons"]},
        }
    }
    assert _fake([_call("fallback_singletons")], expected_fallback)["pass"]
    assert not _fake([_call("dense_exact")], expected_fallback)["pass"]


def test_the_verdict_requires_every_on_path_projection_in_a_verify_block():
    two = dict(QUANT, **{"model.layers.1.mlp": QUANT["model.layers.0.mlp"]})
    got = _fake([_call("quantized_affine_optimized")], two)
    assert not got["pass"] and any("never observed" in p for p in got["problems"])
    assert _fake(
        [
            _call("quantized_affine_optimized"),
            _call("quantized_linears_optimized", module="model.layers.1.mlp"),
        ],
        two,
    )["pass"]
    # a width-1 call does not cover a projection; an off-path module is reported, not required
    got = _fake(
        [
            _call("quantized_affine_optimized"),
            _call("plain", module="model.layers.1.mlp", width=1),
        ],
        two,
    )
    assert not got["pass"]
    off_path = dict(QUANT, **{"model.embed_tokens": {"kind": "quantized", "bits": 4}})
    got = _fake([_call("quantized_affine_optimized")], off_path)
    assert got["pass"] and "model.embed_tokens" in got["not_on_path"]


@metal
@pytest.mark.parametrize("pick", PICKS)
def test_tuple_fallbacks_are_attributed_per_module_and_reconcile_with_the_manifest(
    pick,
):
    lm, _, tel, _, _ = _on_run(pick)
    wide = [c for c in tel.calls if c["width"] > 1]
    assert not [c for c in wide if "+" in c["family"]], [
        c for c in wide if "+" in c["family"]
    ]
    manifest = telemetry.manifest_from_model(lm)
    seen = {}
    for a in tel.attributions:
        if a["scope"] == "verify" and a["width"] > 1:
            seen.setdefault(a["module"], set()).add(a["family"])
    for module, families in seen.items():
        kinds = manifest[module]["expected"]
        allowed = set().union(*kinds.values())
        assert families <= allowed, (module, families, kinds)
    if (
        pick == PICKS[0]
    ):  # the mixed recipe: dense in_proj_a / b decline the dense kernel by shape
        assert seen["model.layers.0.linear_attn.in_proj_a"] == {"fallback_singletons"}
        assert seen["model.layers.0.linear_attn.in_proj_qkv"] == {
            "quantized_affine_optimized"
        }
        assert seen["model.layers.1.self_attn.o_proj"] == {"dense_exact"}
    verdict = telemetry.verdict(tel, manifest)
    assert verdict["pass"], verdict
    assert not verdict["missing_coverage"]
    # code round 2 H2: the tuple check is not vacuous — eligible groups occur and used the tuple kernel,
    # ineligible ones (e.g. the GDN projections with small heads) fell back per module
    tuples = [c for c in tel.calls if "tuple_expected" in c and c["scope"] == "verify"]
    assert any(c["tuple_expected"] for c in tuples), tuples
    assert any(not c["tuple_expected"] for c in tuples), tuples


@metal
def test_the_verdict_fails_when_the_optimized_kernels_decline():
    from mlx_vlm.speculative.ops import linear as ops_linear

    real = (
        ops_linear._target_verify_optimized_affine_linear,
        ops_linear._target_verify_quantized_linears,
    )
    ops_linear._target_verify_optimized_affine_linear = lambda *a, **k: None
    ops_linear._target_verify_quantized_linears = lambda *a, **k: None
    try:
        lm, _, tel, emitted, plain = _on_run(PICKS[1])
    finally:
        (
            ops_linear._target_verify_optimized_affine_linear,
            ops_linear._target_verify_quantized_linears,
        ) = real
    assert emitted == plain  # the exact-by-construction fallbacks keep the tokens
    verdict = telemetry.verdict(tel, telemetry.manifest_from_model(lm))
    assert not verdict["pass"]
    assert any("unexpected family" in p for p in verdict["problems"]), verdict


# ------------------------------------------------------------------ code round 2 H2: the tuple kernel is expected
@metal
def test_the_verdict_fails_when_only_the_tuple_kernel_declines():
    from mlx_vlm.speculative.ops import linear as ops_linear

    real = ops_linear._target_verify_quantized_linears
    ops_linear._target_verify_quantized_linears = lambda *a, **k: None
    try:
        lm, _, tel, emitted, plain = _on_run(PICKS[1])
    finally:
        ops_linear._target_verify_quantized_linears = real
    assert emitted == plain  # the per-module optimized kernels keep the tokens
    verdict = telemetry.verdict(tel, telemetry.manifest_from_model(lm))
    assert not verdict["pass"], verdict
    assert any("tuple kernel" in p for p in verdict["problems"]), verdict
