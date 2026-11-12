"""M76 B2 delivery 3: the fix round's defects (engagement band, rejected E1c attempts, joint_v1's ``cache`` reason, the tuple's
full-run projection and operator approval, the engagement helper's mode) (fork-only, dev; CPU only; no GPU).

Each test carries a ``# class:`` comment: (a) reproduces a failure the fix round fixes (fails on the candidate WITHOUT the fix
round), (b) preserves an invariant, (c) new behaviour.  The production pin, the record binding and the refusal message are in
``test_m76_b2_fe_qualify.py``.
"""

# isort: skip_file

import copy

import pytest

from mlx_vlm.tests.test_m76_b2_fe_common import (
    PICK,
    registry_states,
)  # noqa: F401  (also puts dev/ on sys.path)
from mlx_vlm.tests.test_m76_b2_fe_orchestrate import (  # noqa: F401  (fixtures and helpers)
    plan,
    roots,
    tuple_run,
)
from p369_phase2 import orchestrate, worker  # noqa: E402

LIMIT, BLOCK = (
    2048,
    5,
)  # tq4_fast.SINGLE_PASS_MAX / MAX_VERIFY: m72_exact's per_query:short / d2_verify split; a draft block


def routes(d2=0, short=0, **fallbacks):
    before = {"d2_verify": 10, "per_query:short": 10, "fallback:mask": 0}
    after = {"d2_verify": 10 + d2, "per_query:short": 10 + short, "fallback:mask": 0}
    for key, n in fallbacks.items():
        after[f"fallback:{key}"] = n
    return before, after


def engaged(prompt, generated, **moved):
    before, after = routes(**moved)
    return worker.tq4_engagement(
        before, after, prompt_tokens=prompt, generated=generated, mode="m72_exact"
    )


# class: (b)
def test_the_engagement_limits_are_the_specified_ones():
    from mlx_vlm import tq4_fast

    assert (tq4_fast.SINGLE_PASS_MAX, tq4_fast.MAX_VERIFY) == (LIMIT, BLOCK)


# ================================================================== the engagement rule by key-length span (Errata v1.1)
# class: (c) (entirely above 2,048 -> d2_verify; entirely at or below -> per_query:short)
@pytest.mark.parametrize(
    "prompt, generated, need",
    (
        (5000, 100, "d2_verify"),
        (65000, 300, "d2_verify"),
        (LIMIT + 1, 50, "d2_verify"),
        (500, 100, "per_query:short"),
        (1024, 200, "per_query:short"),
        (
            LIMIT - 60,
            60,
            "per_query:short",
        ),  # ends exactly at the limit: entirely at or below
    ),
)
def test_a_span_on_one_side_of_2048_needs_that_sides_route(prompt, generated, need):
    right, wrong = (
        ({"d2": 4}, {"short": 4}) if need == "d2_verify" else ({"short": 4}, {"d2": 4})
    )
    assert engaged(prompt, generated, **right)["ok"] is True
    none = engaged(prompt, generated)
    assert none["ok"] is False and any(need in p for p in none["problems"]), none
    assert engaged(prompt, generated, **wrong)["ok"] is False


# class: (a) (fix 5: a crossing of more than one block needs BOTH routes)
@pytest.mark.parametrize(
    "prompt, generated",
    (
        (1000, 2000),
        (LIMIT - 400, 800),
        (LIMIT - 30, 70),
        (
            LIMIT - BLOCK,
            2 * BLOCK + 2 * BLOCK,
        ),  # [2043, 2086]: starts exactly one block below, ends beyond one block above
        (1000, LIMIT + 7 - 1000),  # [1000, 2055]: ends 7 (> one block) above the limit
        (LIMIT - BLOCK, 100),  # [2043, 2148]
    ),
)
def test_a_crossing_of_two_blocks_or_more_needs_both_routes(prompt, generated):
    assert engaged(prompt, generated, d2=3, short=3)["ok"] is True
    only_d2 = engaged(prompt, generated, d2=3)
    only_short = engaged(prompt, generated, short=3)
    assert only_d2["ok"] is False and any(
        "per_query:short" in p for p in only_d2["problems"]
    ), only_d2
    assert only_short["ok"] is False and any(
        "d2_verify" in p for p in only_short["problems"]
    ), only_short
    assert engaged(prompt, generated)["ok"] is False


# class: (a) (fix 5: within ONE draft block of 2,048 either route suffices; neither fails)
@pytest.mark.parametrize(
    "prompt, generated",
    ((LIMIT - 2, 6), (LIMIT, 4), (LIMIT - BLOCK + 1, 2 * BLOCK - 2)),
)
def test_a_crossing_within_one_block_needs_either_route(prompt, generated):
    assert engaged(prompt, generated, d2=2)["ok"] is True
    assert engaged(prompt, generated, short=2)["ok"] is True
    assert engaged(prompt, generated, d2=1, short=1)["ok"] is True
    none = engaged(prompt, generated)
    assert none["ok"] is False and "neither" in " ".join(none["problems"])


# class: (c) (every fallback:* counter unchanged across the trajectory, whatever the span)
@pytest.mark.parametrize(
    "prompt, generated", ((5000, 100), (500, 100), (1000, 2000), (LIMIT - 2, 6))
)
def test_a_moved_fallback_counter_fails_engagement(prompt, generated):
    got = engaged(prompt, generated, d2=3, short=3, mask=2)
    assert got["ok"] is False and any("fallback" in p for p in got["problems"]), got
    assert got["fallbacks"] == {"fallback:mask": 2}


# class: (c) (a counter standing at a non-zero value before the trajectory is a delta of 0: no fault)
def test_a_fallback_counter_held_at_a_nonzero_value_is_no_fault():
    before, after = {"d2_verify": 1, "fallback:mask": 7}, {
        "d2_verify": 5,
        "fallback:mask": 7,
    }
    assert (
        worker.tq4_engagement(
            before, after, prompt_tokens=5000, generated=100, mode="m72_exact"
        )["ok"]
        is True
    )


# class: (a) (fix 7: the engagement helper supports m72_exact only; m72_v1 and anything else is refused, never mis-judged)
@pytest.mark.parametrize("mode", ("m72_v1", "m72_v9", "", None, "exact"))
def test_the_engagement_helper_refuses_every_mode_but_m72_exact(mode):
    before, after = routes(d2=3)
    # the routes ANY mode could be judged by all moved: a helper that judged by route would call this engaged
    before, after = dict(before, mma=0), dict(after, mma=3)
    try:
        res = worker.tq4_engagement(
            before, after, prompt_tokens=5000, generated=100, mode=mode
        )
    except ValueError:
        return  # refused explicitly
    assert res["ok"] is False, res  # or refused as a verdict; never judged engaged


# ================================================================== E1c: rejected attempts (fix 4) and joint_v1's `cache` reason (fix 2)
def verdict_cell(rejected, **over):
    cell = {
        "c1": {"identical": True},
        "c2": {"identical": True},
        "neutral": True,
        "evidence_ok": True,
        "pass_": True,
        "rejected": rejected,
    }
    cell.update(over)
    return {"T1024": cell}


# class: (b)
def test_pick_outcome_passes_with_no_rejected_attempt():
    assert orchestrate.pick_outcome(verdict_cell([]))["e1c"] == "PASS"


# class: (a) (fix 4: a rejected attempt that was not engaged makes E1c INCOMPLETE; E2 / E7 are skipped)
def test_a_rejected_attempt_that_was_not_engaged_makes_e1c_incomplete():
    got = orchestrate.pick_outcome(
        verdict_cell([dict(neutral=True, c1=True, tq4_engaged=False)])
    )
    assert got["e1c"] == "INCOMPLETE" and got["run_e2_e7"] is False


# class: (b) (an engaged rejected attempt, or a non-tuple attempt with no engagement flag, keeps PASS)
@pytest.mark.parametrize(
    "attempt",
    (dict(neutral=True, c1=True, tq4_engaged=True), dict(neutral=True, c1=True)),
)
def test_an_engaged_or_unflagged_rejected_attempt_keeps_pass(attempt):
    assert orchestrate.pick_outcome(verdict_cell([attempt]))["e1c"] == "PASS"


# class: (c) (precedence: after C1 and neutrality; a C1 failure ABORTs, a non-neutral attempt FAILs, whatever the engagement)
def test_engagement_comes_after_c1_and_neutrality_in_the_precedence():
    bad_c1 = dict(neutral=True, c1=False, tq4_engaged=False)
    bad_neutral = dict(neutral=False, c1=True, tq4_engaged=False)
    assert orchestrate.pick_outcome(verdict_cell([bad_c1]))["e1c"] == "ABORT"
    assert orchestrate.pick_outcome(verdict_cell([bad_neutral]))["e1c"] == "FAIL"
    assert (
        orchestrate.pick_outcome(verdict_cell([bad_neutral], c1={"identical": False}))[
            "e1c"
        ]
        == "ABORT"
    )


# class: (a) (one unengaged attempt among several decides)
def test_any_one_unengaged_rejected_attempt_decides():
    attempts = [
        dict(neutral=True, c1=True, tq4_engaged=True),
        dict(neutral=True, c1=True, tq4_engaged=False),
    ]
    assert orchestrate.pick_outcome(verdict_cell(attempts))["e1c"] == "INCOMPLETE"


def e1c_entry(reasons, short, *, ok=True):
    return {
        "verdict": {"pass": True},
        "draft_rounds": 7,
        "prompt_end": orchestrate.straddle_target(1024),
        "straddle": True,
        "response": {
            "timings": {"verify_blocks_straddle": 3, "verify_fallback_reasons": reasons}
        },
        "tq4_engagement": {"ok": ok, "problems": [], "per_query_short": short},
    }


OFF_ENTRY = {"rows": {"pass": True}, "states": {"pass": True}, "forced": {"pass": True}}


def verdict(reasons, short, *, tq4=True):
    return orchestrate.e1c_verdict(
        e1c_entry(reasons, short),
        OFF_ENTRY,
        threshold=1024,
        verify_policy="joint_v1",
        tq4=tq4,
    )


# class: (a) (fix 2: below 2,048 keys m72_exact counts per_query:short and joint_v1 records reason `cache`: equal counts pass)
@pytest.mark.parametrize("n", (1, 4, 37))
def test_cache_reasons_equal_to_the_per_query_short_delta_are_accepted(n):
    got = verdict({"cache": n}, n)
    assert got["pass_"] is True and got["evidence_problems"] == [], got


# class: (b) (no fallback reason at all is the clean case)
def test_no_fallback_reason_is_clean():
    assert verdict({}, 0)["pass_"] is True and verdict(None, 3)["pass_"] is True


# class: (c) (an unequal `cache` count, or any other reason, still fails the evidence, never the neutrality)
@pytest.mark.parametrize(
    "reasons, short",
    (
        ({"cache": 3}, 4),
        ({"cache": 5}, 4),
        ({"cache": 1}, 0),
        ({"mask": 2}, 2),
        ({"gapped": 1}, 0),
        ({"cache": 4, "mask": 1}, 4),
        ({"cache": 2, "k_gt_5": 2}, 4),
    ),
)
def test_other_reasons_or_an_unequal_cache_count_still_fail(reasons, short):
    got = verdict(reasons, short)
    assert (
        got["pass_"] is False and got["evidence_ok"] is False and got["neutral"] is True
    ), got
    assert any("fallback reasons" in p for p in got["evidence_problems"])


# class: (b) (non-tuple runs unchanged: ANY reason, `cache` included, still fails)
def test_a_non_tuple_run_still_fails_on_cache_reasons():
    entry = e1c_entry({"cache": 2}, 2)
    del entry["tq4_engagement"]
    got = orchestrate.e1c_verdict(
        entry, OFF_ENTRY, threshold=1024, verify_policy="joint_v1"
    )
    assert got["pass_"] is False and any(
        "fallback reasons" in p for p in got["evidence_problems"]
    )


# class: (c) (a failed engagement fails the tuple's verdict whatever the reasons)
def test_a_failed_engagement_fails_the_tuple_verdict():
    got = orchestrate.e1c_verdict(
        e1c_entry({"cache": 2}, 2, ok=False),
        OFF_ENTRY,
        threshold=1024,
        verify_policy="joint_v1",
        tq4=True,
    )
    assert got["pass_"] is False and got["evidence_ok"] is False


# class: (c) (a missing per_query:short count is malformed evidence: a harness defect, it raises)
def test_a_missing_per_query_short_count_is_malformed_evidence():
    entry = e1c_entry({"cache": 2}, 2)
    entry["tq4_engagement"].pop("per_query_short")
    with pytest.raises(orchestrate.MalformedEvidence):
        orchestrate.e1c_verdict(
            entry, OFF_ENTRY, threshold=1024, verify_policy="joint_v1", tq4=True
        )


# ================================================================== the tuple's full-run projection and the operator approval (fix 6)
STEP_KEYS = ("prep", "e1c", "e2", "e7b")


def steps(k):
    return {name: k for name in STEP_KEYS}


def tuple_gate(k):
    projected = orchestrate.tuple_phase_projection(steps(k), "m76-x")["nominal_s"]
    limit = orchestrate.TUPLE_APPROVAL_LIMIT_S
    return dict(
        problems=[],
        projected_full_s=projected,
        projected_with_tail_s=projected,
        fits=projected <= limit,
        pass_=True,
        e7b={},
        tuple="m76-x",
        limit_s=limit,
    )


def k_for_hours(hours):
    unit = orchestrate.tuple_phase_projection(steps(1.0), "m76-x")["nominal_s"]
    return hours * 3600 / unit


# class: (a) (the projection is taken from the tuple's FULL cells: more than the four-threshold native16 projection)
def test_the_tuple_projection_includes_the_extra_cells():
    s = steps(100.0)
    assert (
        orchestrate.tuple_phase_projection(s, "m76-x")["nominal_s"]
        > orchestrate.phase_projection(s)["nominal_s"]
    )


# class: (c) (the projection is linear in the measured steps)
def test_the_tuple_projection_scales_with_the_measured_steps():
    a = orchestrate.tuple_phase_projection(steps(10.0), "m76-x")["nominal_s"]
    b = orchestrate.tuple_phase_projection(steps(20.0), "m76-x")["nominal_s"]
    assert b == pytest.approx(2 * a)


# class: (b) (the existing picks' projection and limit are untouched)
def test_the_native16_projection_is_unchanged():
    got = orchestrate.phase_projection(steps(10.0))
    e1c = 10.0 * sum(orchestrate.THRESHOLDS) / sum(orchestrate.SMOKE_THRESHOLDS)
    assert got["nominal_s"] == pytest.approx(15 + e1c + 100 + e1c + 10.0 * (20 / 3 + 5))
    assert orchestrate.ESTIMATE_LIMIT_S == 3.5 * 3600


# class: (c) (M76 §7 stage 2: a tuple full run projected over 8 h goes to the operator)
def test_the_tuple_approval_limit_is_8_hours():
    assert orchestrate.TUPLE_APPROVAL_LIMIT_S == 8 * 3600


def smoke_summary(monkeypatch, k):
    """A tuple smoke summary whose stored gate equals its recomputation.  The evidence side of ``smoke_gate`` is a pure
    stand-in (only the projection and the approval are under test); ``tuple_phase_projection`` is the real one.
    """
    gate = tuple_gate(k)
    monkeypatch.setattr(
        orchestrate, "smoke_gate", lambda picks, tuple_name=None: copy.deepcopy(gate)
    )
    fits = gate["fits"]
    return dict(
        phase="smoke",
        tuple="m76-x",
        picks={PICK: {}},
        status="SMOKE_PASS" if fits else "SMOKE_NOT_PASS",
        exit_code=0 if fits else 2,
        smoke_gate=gate,
    )


# class: (b) (a tuple smoke projecting at most 8 h authorizes the full phase without the flag)
def test_a_tuple_projection_within_8h_authorizes_without_approval(monkeypatch):
    summary = smoke_summary(monkeypatch, k_for_hours(7.0))
    assert summary["smoke_gate"]["fits"] is True
    assert orchestrate.smoke_authorization_problems(summary) == []


# class: (a) (fix 6: over 8 h the full phase refuses without --operator-approved-hours)
def test_a_tuple_projection_over_8h_refuses_without_approval(monkeypatch):
    summary = smoke_summary(monkeypatch, k_for_hours(9.0))
    assert summary["smoke_gate"]["fits"] is False
    assert orchestrate.smoke_authorization_problems(summary)


# class: (a) (the approval must cover the projection: hours below it refuse, at or above authorize)
def test_the_operator_approval_must_cover_the_projection(monkeypatch):
    summary = smoke_summary(monkeypatch, k_for_hours(9.0))
    assert orchestrate.smoke_authorization_problems(
        summary, operator_approved_hours=8.5
    )
    assert orchestrate.smoke_authorization_problems(
        summary, operator_approved_hours=0.5
    )
    assert (
        orchestrate.smoke_authorization_problems(summary, operator_approved_hours=9.5)
        == []
    )


# class: (c) (the approval lifts the projection only: a failed gate, or a smoke that is no tuple's, still refuses)
def test_the_approval_lifts_only_the_projection(monkeypatch):
    summary = smoke_summary(monkeypatch, k_for_hours(9.0))
    bad = copy.deepcopy(summary)
    bad["smoke_gate"]["pass_"] = False
    bad["smoke_gate"]["problems"] = ["x"]
    monkeypatch.setattr(
        orchestrate,
        "smoke_gate",
        lambda picks, tuple_name=None: copy.deepcopy(bad["smoke_gate"]),
    )
    assert orchestrate.smoke_authorization_problems(bad, operator_approved_hours=20)
    native = {k: v for k, v in summary.items() if k != "tuple"}
    assert orchestrate.smoke_authorization_problems(native, operator_approved_hours=20)


# class: (c) (a smoke that already fits is authorized with or without an approval)
def test_an_approval_on_a_fitting_tuple_smoke_is_inert(monkeypatch):
    summary = smoke_summary(monkeypatch, k_for_hours(2.0))
    assert (
        orchestrate.smoke_authorization_problems(summary, operator_approved_hours=12)
        == []
    )


# class: (c) (the CLI: --operator-approved-hours belongs to a tuple full phase; refused elsewhere, before any plan)
@pytest.mark.parametrize("use_tuple", (True, False), ids=("tuple-smoke", "pick-smoke"))
def test_operator_approved_hours_is_refused_outside_a_tuple_full_phase(
    plan, roots, use_tuple
):
    on, off = roots
    assert tuple_run(plan, on, off).ok  # control
    which = (
        ("--tuple", "m76-x", "--off-stack-root", off) if use_tuple else ("--pick", PICK)
    )
    res = plan(*which, "--stack-root", on, "--operator-approved-hours", "9")
    assert not res.ok and "--model" not in res.out


@pytest.mark.parametrize("hours", ("0", "-1", "nan", "inf", "abc"))
def test_operator_approved_hours_must_be_finite_and_positive(plan, roots, hours):
    on, off = roots
    assert tuple_run(plan, on, off).ok  # control
    res = plan(
        "--tuple",
        "m76-x",
        "--stack-root",
        on,
        "--off-stack-root",
        off,
        "--operator-approved-hours",
        hours,
    )
    assert not res.ok and "--model" not in res.out
