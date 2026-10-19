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
    calls = []

    class Lm:
        def named_modules(self):
            return []

    tel = telemetry.KernelTelemetry(Lm())
    with tel:
        with tel.probe():
            tel._record("_target_verify_linear", "plain", 1, "float32", ["x"])
        tel._record("_target_verify_linear", "plain", 1, "float32", ["x"])
    assert len(tel.calls) == 1 and len(tel.probe_calls) == 1


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
