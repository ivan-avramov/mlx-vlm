"""P475a v4.1 fix round (delivery 3): tests for dispositions fixes 1-3 and GLM F3 (fork-only, dev; CPU only).

Fix 1 (Sol P1): with the expected manifest known, ``sdpa_forced`` must be a present finite integer >= 0 — ``> 0`` required iff
fused_v1 with a native leaf, exactly 0 otherwise; missing / None / negative / NaN is a problem.
Fix 2 (Sol P2): no attention verdict at all when the expected manifest is unknown, under ANY policy.
Fix 3 (GLM F1): ``--kv-arm --predictor on`` with a NON-EMPTY KV delta refuses as uncertified; an empty delta is today's
``--predictor on`` (the C199 argv binding is never waived).
GLM F3: ``--kv-arm`` without ``--predictor`` refuses; ``--kv-arm --predictor on`` without a P369 summary refuses.

``# class:`` as in the other two files; ``reproduces-failure`` = must FAIL on the unfixed d2 candidate.
"""

import math

import pytest
from p369_phase2 import qualify, registry_states  # noqa: E402

from mlx_vlm.tests import test_p475a_kv_arm as kv  # noqa: E402  (puts dev/ on sys.path)
from mlx_vlm.tests.test_p475a_qualify_driver import (  # noqa: E402,F401  (rig is the pytest fixture)
    KV,
    kv_run,
    mlx_serve_argv,
    refused,
    rig,
    serve,
)

NO_MANIFEST = "no leaf manifest / template of the measured cache"
BAD_COUNTERS = {
    "missing": "missing",
    "none": None,
    "negative": -1,
    "nan": math.nan,
    "inf": math.inf,
    "string": "0",
    "bool": True,
}
# C282 (fix round 2): under a policy other than fused_v1 an ABSENT / None counter is 0 (the server omits it under `auto`)
ABSENT_LABELS = ("missing", "none")


def _record(pick, arm, forced, *, drop=False, **changes):
    expected = qualify.expected_config(kv.entry_for(arm, pick))
    rec = kv.valid_record(expected, kv.arm_manifest(arm), sdpa_forced=forced)
    if drop:
        del rec["sdpa_forced"]
    rec.update(changes)
    return expected, rec


def _sdpa(problems):
    return [p for p in problems if "sdpa_forced" in p]


# ================================================================== fix 1: the counter must be present, finite, >= 0
# cells: zero required under fused_v1 (pick A AF8), forced required (pick A N16), zero required under auto (uniform TQ4)
CELLS1 = [(kv.PICK_A, "AF8", 0), (kv.PICK_A, "N16", 640), (kv.UNIFORM, "TQ4", 0)]
IDS1 = [
    "fable-AF8-zero-required",
    "fable-N16-forced-required",
    "uniform-TQ4-zero-required",
]


# class: reproduces-failure for the zero-required cells (the unfixed rule reads a missing / None / NaN / negative counter as
#        0 or "not > 0", the right value there) and, in every cell, for inf (> 0) and a string (TypeError); invariant for
#        fable-N16 with missing / None / negative / NaN (the unfixed rule already flags a counter that is not > 0)
@pytest.mark.parametrize("pick, arm, right", CELLS1, ids=IDS1)
@pytest.mark.parametrize("label", sorted(BAD_COUNTERS))
def test_fix1_a_missing_none_negative_or_nonfinite_counter_is_a_problem(
    pick, arm, right, label
):
    value = BAD_COUNTERS[label]
    expected, rec = _record(pick, arm, right, drop=value == "missing")
    if value != "missing":
        rec["sdpa_forced"] = value
    problems = qualify.request_problems(rec, expected)
    if expected["attention_policy"] != "fused_v1" and label in ABSENT_LABELS:
        assert _sdpa(problems) == [], (
            label,
            problems,
        )  # C282: absent / None = 0 under `auto`
    else:
        assert _sdpa(problems), (label, problems)


# class: invariant (the right counter still passes: the fix must not over-reject)
@pytest.mark.parametrize("pick, arm, right", CELLS1, ids=IDS1)
def test_fix1_the_right_counter_still_passes(pick, arm, right):
    expected, rec = _record(pick, arm, right)
    assert qualify.request_problems(rec, expected) == []


# class: invariant (a numeric counter of the wrong sign for the arm is the existing problem, whatever the value)
@pytest.mark.parametrize("pick, arm, right", CELLS1, ids=IDS1)
def test_fix1_the_wrong_but_valid_counter_is_still_a_problem(pick, arm, right):
    expected, rec = _record(pick, arm, 640 if right == 0 else 0)
    assert _sdpa(qualify.request_problems(rec, expected))


# class: reproduces-failure (the driver, whole path: a worker record with no ``sdpa_forced`` must not complete)
@pytest.mark.parametrize(
    "pick, arm", ((kv.PICK_A, "AF8"), (kv.PICK_A, "N16"))
)  # fused_v1 only (C282)
def test_fix1_driver_a_record_without_the_counter_fails_the_entry(rig, pick, arm):
    rig.registry(kv.arm_text(arm, pick))
    serve(rig, pick, arm)
    plain = rig.worker

    def worker(job):
        rec = plain(job)
        del rec["sdpa_forced"]
        return rec

    rig.worker = worker
    got = kv_run(rig, pick)
    state = got.state(pick)
    assert got.code == 1 and state["status"] == "FAILED", got.text()
    assert _sdpa(state["problems"]), state["problems"]


# ================================================================== fix round 2 (C282): `auto` — an absent counter is 0
# class: reproduces-failure (round-1 candidate: `sdpa_forced None is not an integer >= 0` for every `auto` record; the real
#        worker reports None because the server omits the key under `auto`)
@pytest.mark.parametrize("arm", tuple(kv.ARMS))
@pytest.mark.parametrize("label", ABSENT_LABELS)
def test_c282_an_auto_record_with_the_counter_absent_passes(arm, label):
    expected, rec = _record(kv.UNIFORM, arm, 0, drop=label == "missing")
    if label == "none":
        rec["sdpa_forced"] = None
    assert expected["attention_policy"] == "auto"
    assert qualify.request_problems(rec, expected) == []


# class: invariant (a PRESENT value under `auto` must still be an int >= 0 and exactly 0)
@pytest.mark.parametrize("arm", ("TQ4", "AF8"))
@pytest.mark.parametrize(
    "value", (640, 1, -1, math.nan, math.inf, "0", True, False, 0.0)
)
def test_c282_a_present_auto_counter_must_be_exactly_integer_zero(arm, value):
    expected, rec = _record(kv.UNIFORM, arm, 0)
    rec["sdpa_forced"] = value
    assert _sdpa(qualify.request_problems(rec, expected)), value


# class: invariant (the present, right value passes)
@pytest.mark.parametrize("arm", tuple(kv.ARMS))
def test_c282_a_present_zero_passes_under_auto(arm):
    expected, rec = _record(kv.UNIFORM, arm, 0)
    assert rec["sdpa_forced"] == 0 and qualify.request_problems(rec, expected) == []


# class: invariant (fused_v1 keeps the strict reading: an absent / None counter fails in every arm of the pick A pick)
@pytest.mark.parametrize("arm", tuple(kv.ARMS))
@pytest.mark.parametrize("label", ABSENT_LABELS)
def test_c282_a_fused_v1_record_with_the_counter_absent_still_fails(arm, label):
    expected, rec = _record(
        kv.PICK_A, arm, kv.right_forced(kv.PICK_A, arm), drop=label == "missing"
    )
    if label == "none":
        rec["sdpa_forced"] = None
    assert expected["attention_policy"] == "fused_v1"
    assert _sdpa(qualify.request_problems(rec, expected)), qualify.request_problems(
        rec, expected
    )


# class: reproduces-failure (the driver, whole path, the first P475 pick: the record the real worker emits under `auto` —
#        no sdpa_forced key — must complete, in every arm)
@pytest.mark.parametrize("arm", tuple(kv.ARMS))
@pytest.mark.parametrize("label", ABSENT_LABELS)
def test_c282_driver_the_uniform_pick_completes_with_the_counter_absent(
    rig, arm, label
):
    rig.registry(kv.arm_text(arm, kv.UNIFORM))
    serve(rig, kv.UNIFORM, arm)
    plain = rig.worker

    def worker(job):
        rec = plain(job)
        if label == "missing":
            del rec["sdpa_forced"]
        else:
            rec["sdpa_forced"] = None
        return rec

    rig.worker = worker
    got = kv_run(rig, kv.UNIFORM)
    assert got.completed(kv.UNIFORM), got.text()
    assert got.state(kv.UNIFORM)["problems"] == []


# class: reproduces-failure (without the flag too: the committed uniform pick, today's path, with the key absent)
def test_c282_driver_without_the_flag_the_committed_uniform_pick_completes_with_the_counter_absent(
    rig,
):
    serve(rig, kv.UNIFORM, "TQ4")
    plain = rig.worker

    def worker(job):
        rec = plain(job)
        del rec["sdpa_forced"]
        return rec

    rig.worker = worker
    got = rig.run("--predictor", "off", "--entry", kv.UNIFORM)
    assert got.completed(kv.UNIFORM), got.text()


# class: invariant (the driver, fused_v1: the same absent key still fails the pick A entry — also in the AF8 arm that needs 0)
@pytest.mark.parametrize("arm", ("AF8", "N16"))
def test_c282_driver_the_fable_pick_fails_with_the_counter_absent(rig, arm):
    rig.registry(kv.arm_text(arm, kv.PICK_A))
    serve(rig, kv.PICK_A, arm)
    plain = rig.worker

    def worker(job):
        rec = plain(job)
        del rec["sdpa_forced"]
        return rec

    rig.worker = worker
    got = kv_run(rig, kv.PICK_A)
    assert got.code == 1 and got.state(kv.PICK_A)["status"] == "FAILED", got.text()
    assert _sdpa(got.state(kv.PICK_A)["problems"])


# ================================================================== fix 2: no attention verdict without the expected manifest
# class: reproduces-failure for ``auto`` with forced > 0 (the unfixed rule adds a spurious sdpa_forced problem); invariant for
#        the rest
@pytest.mark.parametrize("missing", ("manifest", "template", "both"))
@pytest.mark.parametrize(
    "pick, arm, forced",
    [
        (kv.UNIFORM, "TQ4", 640),
        (kv.UNIFORM, "TQ4", 0),
        (kv.UNIFORM, "AF8", 640),
        (kv.PICK_A, "AF8", 0),
        (kv.PICK_A, "AF8", 640),
        (kv.PICK_A, "N16", 0),
        (kv.PICK_A, "N16", 640),
    ],
)
def test_fix2_a_missing_manifest_yields_only_the_manifest_problem(
    pick, arm, forced, missing
):
    expected, rec = _record(pick, arm, forced)
    for key in ("manifest", "template"):
        if missing in (key, "both"):
            rec[key] = None
    assert qualify.request_problems(rec, expected) == [NO_MANIFEST]


# class: reproduces-failure for ``auto`` (the counter itself is not judged either when there is nothing to judge it against)
@pytest.mark.parametrize("pick, arm", ((kv.UNIFORM, "TQ4"), (kv.PICK_A, "AF8")))
@pytest.mark.parametrize("label", sorted(BAD_COUNTERS))
def test_fix2_an_unreadable_counter_is_not_judged_without_a_manifest(pick, arm, label):
    expected, rec = _record(
        pick, arm, 0, drop=BAD_COUNTERS[label] == "missing", manifest=None
    )
    if BAD_COUNTERS[label] != "missing":
        rec["sdpa_forced"] = BAD_COUNTERS[label]
    assert _sdpa(qualify.request_problems(rec, expected)) == []


# class: reproduces-failure (the driver, whole path: a no-manifest record under ``auto`` with forced > 0 fails with the manifest
#        problem ONLY; under fused_v1 it already did)
@pytest.mark.parametrize(
    "pick, arm, forced",
    ((kv.UNIFORM, "TQ4", 640), (kv.UNIFORM, "AF8", 640), (kv.PICK_A, "AF8", 0)),
)
def test_fix2_driver_a_no_manifest_record_fails_with_only_the_manifest_problem(
    rig, pick, arm, forced
):
    rig.registry(kv.arm_text(arm, pick))
    serve(rig, pick, arm, forced=forced, manifest=None, template=None)
    got = kv_run(rig, pick)
    state = got.state(pick)
    assert got.code == 1 and state["status"] == "FAILED", got.text()
    assert state["problems"] == [NO_MANIFEST]


# ================================================================== fix 3: --kv-arm --predictor on
ON_CELLS = [
    (kv.PICK_A, "AF8"),
    (kv.PICK_A, "TQ8"),
    (kv.PICK_A, "AF4"),
    (kv.UNIFORM, "AF8"),
    (kv.UNIFORM, "AF4"),
    (kv.UNIFORM, "TQ8"),
    (kv.UNIFORM, "N16"),  # delta {kv_bits: (4, 0)}: non-empty
]
ON_IDS = [f"{'pickA' if p == kv.PICK_A else 'uniform'}-{a}" for p, a in ON_CELLS]


def _on(rig, pick, arm, *, certified="arm"):
    """Register the committed MTP-ON arm registry and run ``--kv-arm --predictor on`` with a P369 summary whose certified argv
    is the arm's OWN argv (``certified="arm"``) — so the C199 binding itself would pass and only the uncertified-arm rule is
    left to refuse — or the committed one."""
    text = kv.arm_text(arm, pick)
    rig.registry(text)
    base = text if certified == "arm" else kv.FIXTURE
    argv_on = mlx_serve_argv(registry_states.settings(base)[pick], rig.stack)[1:]
    summary = rig.summary(pick, argv_on=argv_on)
    return rig.run(
        "--kv-arm", "--predictor", "on", "--entry", pick, "--p369-summary", summary
    )


# class: reproduces-failure (the unfixed candidate completes these: it serves the MTP-ON arm as committed)
@pytest.mark.parametrize("pick, arm", ON_CELLS, ids=ON_IDS)
def test_fix3_predictor_on_with_a_nonempty_delta_refuses_as_uncertified(rig, pick, arm):
    assert kv.arm_delta(arm, pick)  # premise: non-empty
    serve(rig, pick, arm, mtp_on=True)
    got = _on(rig, pick, arm)
    refused(rig, got)
    assert "uncertified" in got.text().lower(), got.text()
    assert not list(got.out.glob("overlay-*")) if got.out.exists() else True


# class: reproduces-failure (same, with the summary certifying the COMMITTED argv: still the uncertified-arm reason first)
@pytest.mark.parametrize(
    "pick, arm", (ON_CELLS[0], ON_CELLS[3]), ids=(ON_IDS[0], ON_IDS[3])
)
def test_fix3_the_refusal_is_the_uncertified_arm_not_the_argv_binding(rig, pick, arm):
    serve(rig, pick, arm, mtp_on=True)
    got = _on(rig, pick, arm, certified="committed")
    refused(rig, got)
    assert "uncertified" in got.text().lower(), got.text()


# class: invariant (an empty delta is today's --predictor on: served as committed, bound to the certified argv)
@pytest.mark.parametrize("pick, arm", ((kv.PICK_A, "N16"), (kv.UNIFORM, "TQ4")))
def test_fix3_predictor_on_with_an_empty_delta_behaves_as_today(rig, pick, arm):
    assert kv.arm_delta(arm, pick) == {}
    serve(rig, pick, arm, mtp_on=True)
    got = _on(rig, pick, arm)
    assert got.completed(pick), got.text()
    assert got.state(pick)["arm"] == "served_on"
    assert [s["path"] for s in rig.served] == [rig.committed]
    assert got.report["kv_arm"] == dict(entry=pick, delta={})
    assert "uncertified" not in got.text().lower()


# class: invariant (C199 is never waived for an empty delta: a certified argv that differs refuses, as without the flag)
@pytest.mark.parametrize("pick, arm", ((kv.PICK_A, "N16"), (kv.UNIFORM, "TQ4")))
def test_fix3_an_empty_delta_still_needs_the_certified_argv(rig, pick, arm):
    serve(rig, pick, arm, mtp_on=True)
    rig.registry(kv.FIXTURE)
    argv_on = mlx_serve_argv(registry_states.settings(kv.FIXTURE)[pick], rig.stack)[1:]
    argv_on = [a if a != "--prefill-step-size" else a for a in argv_on] + [
        "--extra-flag",
        "x",
    ]
    summary = rig.summary(pick, argv_on=argv_on)
    got = rig.run(
        "--kv-arm", "--predictor", "on", "--entry", pick, "--p369-summary", summary
    )
    refused(rig, got)
    assert "certified" in got.text()


# class: invariant (the same refusal holds without the flag: the baseline behaviour the empty-delta case must match)
@pytest.mark.parametrize("pick", kv.PICKS)
def test_fix3_the_unflagged_binding_refusal_is_the_same_for_the_same_certified_argv(
    rig, pick
):
    arm = "N16" if pick == kv.PICK_A else "TQ4"
    serve(rig, pick, arm, mtp_on=True)
    rig.registry(kv.FIXTURE)
    argv_on = mlx_serve_argv(registry_states.settings(kv.FIXTURE)[pick], rig.stack)[
        1:
    ] + ["--extra-flag", "x"]
    summary = rig.summary(pick, argv_on=argv_on)
    plain = rig.run("--predictor", "on", "--entry", pick, "--p369-summary", summary)
    refused(rig, plain)
    flagged = rig.run(
        "--kv-arm", "--predictor", "on", "--entry", pick, "--p369-summary", summary
    )
    refused(rig, flagged)
    assert (plain.refused or "") == (flagged.refused or "")


# ================================================================== GLM F3
# class: invariant (passes on the unfixed candidate: the gap was a missing test, not a missing behaviour)
@pytest.mark.parametrize("pick, arm", ((kv.UNIFORM, "AF8"), (kv.PICK_A, "N16")))
def test_f3_kv_arm_without_a_predictor_refuses(rig, pick, arm):
    rig.registry(kv.arm_text(arm, pick))
    serve(rig, pick, arm)
    assert kv_run(rig, pick).completed(
        pick
    )  # control: with --predictor off the same registry runs
    got = rig.run("--kv-arm", "--entry", pick)
    refused(rig, got)
    assert "predictor" in got.text().lower()


# class: invariant
@pytest.mark.parametrize(
    "pick, arm", ((kv.PICK_A, "N16"), (kv.UNIFORM, "TQ4"), (kv.UNIFORM, "AF8"))
)
def test_f3_kv_arm_predictor_on_without_a_summary_refuses(rig, pick, arm):
    rig.registry(kv.arm_text(arm, pick))
    serve(rig, pick, arm, mtp_on=True)
    got = rig.run("--kv-arm", "--predictor", "on", "--entry", pick)
    refused(rig, got)
    assert not list(got.out.glob("overlay-*")) if got.out.exists() else True
    # control: the empty-delta arm with its summary runs (the refusal above was the missing summary)
    if not kv.arm_delta(arm, pick):
        argv_on = mlx_serve_argv(registry_states.settings(kv.FIXTURE)[pick], rig.stack)[
            1:
        ]
        assert rig.run(
            "--kv-arm",
            "--predictor",
            "on",
            "--entry",
            pick,
            "--p369-summary",
            rig.summary(pick, argv_on=argv_on),
        ).completed(pick)
