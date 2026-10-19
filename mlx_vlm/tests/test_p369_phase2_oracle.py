"""P369 phase 2 E7 (Q23 / Q30): the M3 / M4 classes, validated against genuinely pre-fix execution.

Fork-only file. The same scripted cases (tiny dense hybrid target, full-accept scripted MTP drafter, a stop
placed on the first draft / second draft / bonus / first token, a length cut, thinking budgets that trip
inside a block) run in two ISOLATED checkouts, each in a fresh ``python -I`` interpreter with only its own
``mlx_vlm`` on ``sys.path``: ``88724023`` (pre-fix, = ``0b43dae4^``) must show each precise defect,
``0b43dae4`` (the M4 + M3 fix) none. The live ``Observer`` on the CURRENT code must then classify exactly
the pre-fix defects of each case (class, M4a's committed-past count, M3's overshoot).
"""

import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dev"))

from p369_phase2 import classes, oracle, oracle_cases  # noqa: E402


def _has(commit):
    return (
        subprocess.run(
            ["git", "-C", str(REPO), "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
        ).returncode
        == 0
    )


def _oracle_gate(available, *, required):
    """Only the tests that run the two pre-fix trees depend on git history: skip them in a shallow clone,
    or FAIL when ``MLX_REQUIRE_P369_ORACLE=1`` (a gating job must prove the M3 / M4 binding).
    """
    if available:
        return
    reason = "git history lacks the P369 commits (shallow clone): the oracle needs both trees"
    if required:
        pytest.fail(f"MLX_REQUIRE_P369_ORACLE=1 but {reason}")
    pytest.skip(reason)


EXPECT_PRE_FIX = {
    "first_draft": dict(
        committed_past_stop=1, terminal_missing=0, next_turn_diverged=True
    ),
    "second_draft": dict(
        committed_past_stop=0, terminal_missing=0, next_turn_diverged=False
    ),
    "bonus": dict(committed_past_stop=0, terminal_missing=1, next_turn_diverged=True),
    "first_token": dict(
        committed_past_stop=0, terminal_missing=1, next_turn_diverged=True
    ),
    "length": dict(committed_past_stop=0, terminal_missing=1, next_turn_diverged=True),
    "budget2": dict(overshoot=2, forced_sequence_cut=True),
    "budget3": dict(overshoot=1, forced_sequence_cut=True),
    "budget4": dict(overshoot=0, forced_sequence_cut=True),
}


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    _oracle_gate(
        _has(oracle.PRE_FIX) and _has(oracle.FIX),
        required=os.environ.get("MLX_REQUIRE_P369_ORACLE") == "1",
    )
    work = tmp_path_factory.mktemp("p369-oracle")
    return {
        commit: {
            name: oracle.defects(result)
            for name, result in oracle.run(REPO, commit, oracle.CASES, work).items()
        }
        for commit in (oracle.PRE_FIX, oracle.FIX)
    }


def test_the_pre_fix_tree_shows_each_precise_defect(measured):
    got = measured[oracle.PRE_FIX]
    assert set(got) == set(EXPECT_PRE_FIX)
    for name, want in EXPECT_PRE_FIX.items():
        for key, value in want.items():
            assert got[name][key] == value, (name, key, got[name])


def test_the_fixed_tree_shows_none(measured):
    for name, d in measured[oracle.FIX].items():
        assert d["committed_past_stop"] == 0 and d["terminal_missing"] == 0, name
        assert not d["next_turn_diverged"] and not d["emission_differs"], name
        assert d["overshoot"] in (None, 0) and not d["forced_sequence_cut"], name


def _expected_classes(name, d):
    out = []
    if d["committed_past_stop"]:
        out.append("M4a")
    if d["terminal_missing"]:
        out.append("M4b_first" if name == "first_token" else "M4b")
    if d["overshoot"] is not None and (d["overshoot"] > 0 or d["forced_sequence_cut"]):
        out.append("M3")
    return Counter(out)


def test_the_live_classifier_flags_exactly_the_pre_fix_defects(measured):
    observer = classes.Observer()
    with observer:
        oracle_cases.run_cases(oracle.CASES, observer=observer)
    events = {r["key"]: r["events"] for r in observer.requests}
    for name, d in measured[oracle.PRE_FIX].items():
        got = events[name]
        assert Counter(e["cls"] for e in got) == _expected_classes(name, d), (name, got)
        for e in got:
            if e["cls"] == "M4a":
                assert e["committed_past"] == d["committed_past_stop"], (name, e)
            if e["cls"] == "M3":
                assert e["overshoot"] == d["overshoot"], (name, e)


def test_the_classifier_unit_cases():
    c = classes.Observer.classify
    assert [e["cls"] for e in c(2, [5, 6, 7], 1, True, None, 9)] == ["M4a"]
    assert c(2, [5, 6, 7], 2, True, None, 9) == []  # stop on the LAST accepted draft
    assert [e["cls"] for e in c(2, [5, 6, 7], 3, True, None, 9)] == ["M4b"]
    assert c(2, [5, 6, 7], 3, False, None, 9) == []
    (m3,) = c(2, [5, 6, 7], 1, False, 63, 9)
    assert m3["cls"] == "M3" and m3["overshoot"] == 2 and m3["trip_index"] == 0
    (m4b,) = c(1, [5, 6], 2, True, None, 2)
    assert m4b["cause"] == "budget"


# ------------------------------------------------------------------ code round 1 H12: the gate's scope
def test_the_unit_cases_do_not_depend_on_git_history():
    # only the tests that need both trees may skip in a shallow clone; the classifier cases always run
    assert not getattr(sys.modules[__name__], "pytestmark", None)


def test_the_oracle_gate_skips_or_fails_closed():
    _oracle_gate(True, required=True)
    _oracle_gate(True, required=False)
    with pytest.raises(pytest.skip.Exception):
        _oracle_gate(False, required=False)
    with pytest.raises(pytest.fail.Exception, match="MLX_REQUIRE_P369_ORACLE"):
        _oracle_gate(False, required=True)
