"""T1 pure-Python contracts; runnable without the MLX-seeding conftest."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))

from p369_phase2 import classes, oracle, oracle_cases  # noqa: E402

KINDS = [("first_draft", 4), ("second_draft", 5), ("bonus", 6)]


@pytest.fixture
def trip():
    def call(free, where):
        function = getattr(oracle_cases, "budget_trip", None)
        assert callable(function), "T1 requires oracle_cases.budget_trip(free, where)"
        return function(free, where)

    return call


@pytest.mark.parametrize("where,p", KINDS)
@pytest.mark.parametrize("leading_thinks", [0, 2])
def test_budget_counts_tokens_not_trajectory_positions(trip, where, p, leading_thinks):
    """Exercises new behaviour: both documented trajectories trip at round 1."""
    free = [61] * leading_thinks + [4, 57, 9, 8, 7, 6, 5, 3, 2, 1, 0, 10, 11, 12]
    original = list(free)
    assert trip(free, where) == (p - leading_thinks, p + 4)
    assert free == original


@pytest.mark.parametrize("where,p", KINDS)
def test_think_at_candidate_advances_one_round(trip, where, p):
    """Exercises new behaviour: a start token cannot itself exceed the budget."""
    free = [7] * (p + 4)
    free[p] = 61
    assert trip(free, where) == (p + 2, p + 7)


@pytest.mark.parametrize("where,p", KINDS)
def test_fewer_than_two_counted_tokens_advances_one_round(trip, where, p):
    """Exercises new behaviour: reject a zero budget and count internal starts."""
    free = [61] * p + [7, 61, 8, 9]
    assert trip(free, where) == (2, p + 7)


@pytest.mark.parametrize("where,p", KINDS)
def test_last_available_token_is_a_usable_trip(trip, where, p):
    """Exercises new behaviour: p + 1 == len(free) is included, not skipped."""
    assert trip([7] * (p + 1), where) == (p, p + 4)
    assert trip([7] * p, where) is None


@pytest.mark.parametrize("where,p", KINDS)
@pytest.mark.parametrize("token", [62, 63])
@pytest.mark.parametrize(
    "location", ["before", "at", "after", "round_end", "after_round"]
)
def test_close_or_newline_invalidates_candidate_through_round_end(
    trip, where, p, token, location
):
    """Exercises new behaviour: reject specials through the bonus, not beyond it."""
    free = [7] * 16
    round_end = oracle_cases.WHERE["bonus"](1)
    position = {
        "before": 1,
        "at": p,
        "after": p + 1,
        "round_end": round_end,
        "after_round": round_end + 1,
    }[location]
    free[position] = token
    expected = (p, p + 4) if position > round_end else None
    assert trip(free, where) == expected


def test_budget_trip_rejects_seed11_newline_in_round_remainder(trip):
    """Reproduces C: a natural newline in the rest of the round spoils the oracle."""
    # Supplied seed-11 prefix, with an explicitly synthetic ordinary-token suffix.
    # This does not assume that the installed MLX samples this trajectory.
    free = [4, 23, 37, 24, 15, 63, 50, 2] + [7] * 8
    assert trip(free, "first_draft") is None


@pytest.mark.parametrize("cut", [False, True], ids=["intact-window", "cut-window"])
def test_forced_sequence_cut_ignores_natural_newline_after_window(cut):
    """Reproduces C: later natural newlines neither hide nor invent a cut."""
    off = [7] * 5 + [63, 62]
    on = [7] * 7 + ([62, 17] if cut else [63, 62])
    for add_natural_newline in (False, True):
        result = dict(
            name="controlled_budget_trip",
            off=dict(
                emitted=off + ([63] if add_natural_newline and not cut else []),
                offset=8,
                next=[9],
            ),
            on=dict(
                emitted=on + ([63] if add_natural_newline and cut else []),
                offset=8,
                next=[9],
            ),
        )
        defects = oracle.defects(result)
        assert defects["overshoot"] == 2
        assert defects["forced_sequence_cut"] is cut, result


@pytest.mark.parametrize("where,p", KINDS)
@pytest.mark.parametrize("free", [[], [7], [61] * 16, [61] * 15 + [7]])
def test_no_usable_round_returns_none(trip, where, p, free):
    """Exercises new behaviour: empty, short and uncountable trajectories stop."""
    assert trip(free, where) is None


def test_oracle_cases_and_expected_budget_signatures():
    """Reproduces a failure: fixed budgets do not encode the intended trip slots."""
    from test_p369_phase2_oracle import EXPECT_PRE_FIX

    cases = {case["name"]: case for case in oracle.CASES}
    names = {f"budget_{where}" for where, _ in KINDS}
    assert len(cases) == len(oracle.CASES) == 8
    assert set(cases) == names | {
        "first_draft",
        "second_draft",
        "bonus",
        "first_token",
        "length",
    }
    assert set(EXPECT_PRE_FIX) == set(cases)
    for (where, _), overshoot in zip(KINDS, [2, 1, 0]):
        name = f"budget_{where}"
        assert cases[name]["budget_at"] == where
        assert "budget" not in cases[name]
        assert EXPECT_PRE_FIX[name] == dict(
            overshoot=overshoot, forced_sequence_cut=True
        )


def test_stop_and_length_cases_are_unchanged():
    """Preserves an invariant: budget work leaves the five other cases intact."""
    cases = {case["name"]: case for case in oracle.CASES}
    for where in ["first_draft", "second_draft", "bonus", "first_token"]:
        assert cases[where] == dict(name=where, where=where, n=12)
    assert cases["length"] == dict(name="length", n=7)


@pytest.mark.parametrize("where,p", KINDS)
def test_defect_and_classifier_budget_ladder(where, p):
    """Preserves an invariant: the observable pre-fix overshoot ladder is 2/1/0."""
    overshoot = 6 - p
    off = [7] * (p + 1) + [63, 62]
    on = [7] * 7 + [62]
    result = dict(
        name=f"budget_{where}",
        off=dict(emitted=off, offset=8, next=[9]),
        on=dict(emitted=on, offset=8, next=[9]),
    )
    defects = oracle.defects(result)
    assert defects["overshoot"] == overshoot
    assert defects["forced_sequence_cut"] is True
    (event,) = classes.Observer.classify(2, [7, 7, 7], p - 3, False, 63, 99)
    assert event["cls"] == "M3"
    assert event["overshoot"] == overshoot


def _stub_generation(monkeypatch, free):
    """Replace only model execution; exercise the real run_cases orchestration."""
    fixtures = ModuleType("test_mtp_exactness")
    fixtures.PROMPT = [11, 12]
    fixtures.hybrid_lm = lambda **kwargs: object()
    fixtures.fake_model = lambda lm: lm
    fixtures.mtp_drafter = lambda lm: object()
    fixtures.scripted = lambda *args, **kwargs: None
    common = ModuleType("mlx_vlm.generate.common")
    common.PromptCacheState = lambda: SimpleNamespace(cache=None)
    monkeypatch.setitem(sys.modules, "test_mtp_exactness", fixtures)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate.common", common)
    calls = []

    def run(model, ids, state, **kwargs):
        calls.append(dict(ids=list(ids), **kwargs))
        return list(free[: kwargs["n"]]), "length"

    monkeypatch.setattr(oracle_cases, "_run", run)
    return calls


@pytest.mark.parametrize("where,p", KINDS)
@pytest.mark.parametrize("leading_thinks", [0, 2])
def test_run_cases_derives_and_reports_budget(monkeypatch, where, p, leading_thinks):
    """Exercises new behaviour: prompt, 16-token probe, arm limits and metadata."""
    free = [61] * leading_thinks + [7] * (16 - leading_thinks)
    calls = _stub_generation(monkeypatch, free)
    name = f"budget_{where}"
    (result,) = oracle_cases.run_cases([dict(name=name, budget_at=where, n=12)])
    assert calls[0]["ids"] == [11, 12, 61]
    assert calls[0]["n"] == 16
    assert calls[0].get("budget") is None
    assert "error" not in result
    assert result["budget"] == p - leading_thinks
    assert result["n"] == p + 4
    assert result["trip_position"] == p
    assert result["free"] == free
    assert len(calls) == 5  # free, OFF, OFF follow-up, ON, ON follow-up
    for index in (1, 3):
        assert calls[index]["ids"] == [11, 12, 61]
        assert calls[index]["budget"] == result["budget"]
        assert calls[index]["n"] == result["n"]
        assert not calls[index].get("stop_ids")


@pytest.mark.parametrize("free", [[61] * 16, [7, 62] + [7] * 14])
def test_run_cases_reports_unusable_budget_without_running_arms(monkeypatch, free):
    """Exercises new behaviour: an unusable probe yields the specified error."""
    calls = _stub_generation(monkeypatch, free)
    case = dict(name="budget_first_draft", budget_at="first_draft", n=12)
    assert oracle_cases.run_cases([case]) == [
        dict(name=case["name"], error="no usable budget trip")
    ]
    assert len(calls) == 1
