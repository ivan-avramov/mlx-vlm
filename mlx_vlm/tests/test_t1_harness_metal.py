"""T1 harness behaviour with real criteria/comparator; guarded before imports."""

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="T1 harness imports require Metal"
)


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [{"<think>": 61, "</think>": 62, "\n": 63}[text]]


def _criteria(budget):
    from mlx_vlm.utils import ThinkingBudgetCriteria

    return ThinkingBudgetCriteria(
        _Tokenizer(),
        thinking_budget=budget,
        thinking_start_token="<think>",
        enable_thinking=True,
        prompt_preopens_thinking=True,
    )


@pytest.mark.parametrize(
    "where,p", [("first_draft", 4), ("second_draft", 5), ("bonus", 6)]
)
@pytest.mark.parametrize("trajectory", ["ordinary", "leading-starts", "skip-round"])
def test_derived_budget_trips_real_criteria_at_p(where, p, trajectory):
    """Exercises new behaviour: real counting first forces newline at p + 1."""
    from p369_phase2 import oracle_cases

    free = [7] * 16
    if trajectory == "leading-starts":
        free[:2] = [61, 61]
    elif trajectory == "skip-round":
        free[p] = 61
        p += 3
    derived = oracle_cases.budget_trip(free, where)
    assert derived is not None
    budget, n = derived
    assert n == p + 4
    criteria = _criteria(budget)
    for token in free[:p]:
        assert criteria(token) is None
        assert criteria.pop_forced_token_id() is None
    assert criteria.thinking_token_count == budget
    assert criteria(free[p]) == 63
    assert criteria.thinking_token_count == budget + 1
    assert criteria.pop_forced_token_id() == 63
    assert criteria.pop_forced_token_id() is None
    # The scheduled token is consumed at the next emitted position, p + 1.
    assert criteria(63) == 62
    assert criteria.pop_forced_token_id() == 62
    assert criteria(62) is None
    assert not criteria.in_thinking


def test_thinking_criteria_start_close_and_cumulative_count():
    """Preserves an invariant: starts are uncounted and close does not reset count."""
    criteria = _criteria(3)
    for token in [61, 7, 61, 8]:
        assert criteria(token) is None
    assert criteria.thinking_token_count == 2
    assert criteria(62) is None
    assert criteria(9) is None
    assert criteria.thinking_token_count == 2
    assert criteria(61) is None
    assert criteria(10) is None
    assert criteria(11) == 63
    assert criteria.pop_forced_token_id() == 63


@pytest.mark.parametrize(
    "where,p,overshoot",
    [("first_draft", 4, 2), ("second_draft", 5, 1), ("bonus", 6, 0)],
)
def test_live_budget_case_has_one_intended_m3_event(where, p, overshoot):
    """Reproduces a failure: trajectory-derived cases retain the 2/1/0 ladder."""
    from p369_phase2 import classes, oracle, oracle_cases

    name = f"budget_{where}"
    case = next((c for c in oracle.CASES if c["name"] == name), None)
    assert case is not None, f"missing trajectory-derived case {name}"
    observer = classes.Observer()
    with observer:
        (result,) = oracle_cases.run_cases([case], observer=observer)
    assert "error" not in result, result
    budget, n = oracle_cases.budget_trip(result["free"], where)
    trip_position = result["trip_position"]
    assert trip_position >= p and (trip_position - p) % 3 == 0
    assert result["budget"] == budget
    assert result["n"] == n == trip_position + 4
    for arm in ("on", "off"):
        emitted = result[arm]["emitted"]
        assert emitted[: trip_position + 1] == result["free"][: trip_position + 1]
        assert emitted[trip_position + 1 : trip_position + 3] == [63, 62]
    defects = oracle.defects(result)
    assert defects["overshoot"] == 0
    assert not defects["forced_sequence_cut"]
    assert not defects["emission_differs"]
    (request,) = observer.requests
    events = [event for event in request["events"] if event["cls"] == "M3"]
    assert len(events) == 1, request
    assert events[0]["overshoot"] == overshoot


def _controlled_observe(monkeypatch, first, missing_one=False, row_overrides=None):
    import test_p369_phase2_observe as harness

    params = dict(temperature=0.5, top_p=0.95, top_k=0, min_p=0.0)
    single = harness.compare.as_record(mx.array([0.0, -100.0, -100.0, -100.0]))
    multiple = harness.compare.as_record(mx.array([0.0, 0.0, -100.0, -100.0]))
    assert np.count_nonzero(harness.compare.canonical(single, params)) == 1
    assert np.count_nonzero(harness.compare.canonical(multiple, params)) == 2
    eligible = {0} if first is None else {0, first, first + 1}
    row_overrides = row_overrides or {}
    rows = [
        (p, row_overrides.get(p, multiple if p in eligible else single))
        for p in range(7)
        if not (missing_one and p == 1)
    ]
    # Neither an un-emitted row nor the prefill row is a valid selection.
    rows += [(7, multiple)]
    on, off = SimpleNamespace(rows=list(rows)), SimpleNamespace(rows=list(rows))
    emitted = [0] * 7
    monkeypatch.setattr(
        harness, "_e1c", lambda *a, **k: (None, None, params, on, emitted, [])
    )
    monkeypatch.setattr(harness.observe, "Recorder", lambda **k: nullcontext(off))
    monkeypatch.setattr(harness, "turn", lambda *a, **k: None)
    reports = []
    compare_streams = harness.compare.compare_streams

    def compare(*args, **kwargs):
        report = compare_streams(*args, **kwargs)
        reports.append(report)
        return report

    monkeypatch.setattr(harness.compare, "compare_streams", compare)
    return harness, reports


@pytest.mark.parametrize(
    "first,missing_one", [(1, False), (2, False), (4, False), (5, False), (2, True)]
)
def test_observe_selects_first_emitted_multi_support_row(
    monkeypatch, first, missing_one
):
    """Reproduces a failure: select by canonical support rather than position 3."""
    harness, reports = _controlled_observe(monkeypatch, first, missing_one)
    harness.test_quantized_e1c_detects_an_in_support_change_of_one_verifier_row()
    (report,) = reports
    assert not report["pass"]
    assert [m["position"] for m in report["mismatches"]] == [first]


def test_observe_rejects_trajectory_without_eligible_row(monkeypatch):
    """Exercises new behaviour: fail before comparison when no row qualifies."""
    harness, reports = _controlled_observe(monkeypatch, None)
    with pytest.raises(AssertionError):
        harness.test_quantized_e1c_detects_an_in_support_change_of_one_verifier_row()
    assert not reports, "no eligible row must be asserted before the corruption check"


def test_observe_skips_bf16_multi_support_row_when_bump_rounds_away(monkeypatch):
    """Reproduces D: multiple supported tokens do not guarantee an effective bump."""
    from p369_phase2 import compare

    params = dict(temperature=0.5, top_p=0.95, top_k=0, min_p=0.0)
    uniform = mx.full((64,), -4.15625, dtype=mx.bfloat16)
    effective = mx.array([0.0, 0.0] + [-100.0] * 62, dtype=mx.bfloat16)
    single = mx.array([0.0] + [-100.0] * 63, dtype=mx.bfloat16)
    for row, changes in ((uniform, False), (effective, True)):
        top = int(mx.argmax(row).item())
        bumped = mx.where(mx.arange(row.size) == top, row + 0.01, row).astype(row.dtype)
        before = compare.canonical(row, params)
        after = compare.canonical(bumped, params)
        assert np.count_nonzero(before) >= 2
        assert (not np.array_equal(before, after)) is changes
        if not changes:
            np.testing.assert_array_equal(
                np.array(row.view(mx.uint16)), np.array(bumped.view(mx.uint16))
            )

    # Controlled rows: 1 and the old hard-coded position 3 have ineffective bumps;
    # 4 is the first emitted position >= 1 with an effective in-support change.
    records = {
        p: compare.as_record(
            uniform if p in (1, 3) else effective if p in (0, 4, 5) else single
        )
        for p in range(7)
    }
    harness, reports = _controlled_observe(monkeypatch, 4, row_overrides=records)
    harness.test_quantized_e1c_detects_an_in_support_change_of_one_verifier_row()
    (report,) = reports
    assert not report["pass"]
    assert [m["position"] for m in report["mismatches"]] == [4]
