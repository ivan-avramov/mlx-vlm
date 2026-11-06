"""P658 (a): ``qualify --kv-arm --predictor on --engagement-only`` through the REAL ``qualify.main`` -> ``drive`` path
(fork-only, dev; CPU only, temp git fork).

Written from ``.p658-material/spec.md`` (SPEC v1) before the implementation, by a test writer who did not write it.  The first
line of each test body is a ``# class:`` comment: "reproduces a failure the spec fixes", "preserves an invariant" or
"exercises new behaviour".

What is real: the argument parser, the gate (``classify_kv_arm`` is wrapped in a pass-through spy), ``entry_problems``,
``command_problems``, ``request_problems``, the drafter judgement, the report.  What is mocked is the host and the worker boundary
(the P475a rig: lock, preflight, power, git bindings, ``run_job``, weights).  The worker record is hand-built from the P475a
per-arm manifests plus a hand-written ``drafter_manifest`` in the exact shape ``kvmanifest.manifest`` emits.
"""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))

from p369_phase2 import kvmanifest, qualify, registry_states  # noqa: E402

from mlx_vlm.tests import test_p415_qualify as t415  # noqa: E402
from mlx_vlm.tests import test_p475a_kv_arm as kv  # noqa: E402
from mlx_vlm.tests.test_p475a_qualify_driver import (  # noqa: E402,F401  (rig is the pytest fixture)
    flags_of,
    kv_run,
    mlx_serve_argv,
    rig,
    serve,
)

ENGAGE = ("--kv-arm", "--predictor", "on", "--engagement-only")
NO_DRAFTER_MANIFEST = "no drafter cache manifest"

# the drafter's own template: two native KV layers; what the served path builds is the pre-allocating variant of each
DRAFTER_TEMPLATE = [
    dict(path="0", kind="native", cls="KVCache", batch=False),
    dict(path="1", kind="native", cls="KVCache", batch=False),
]
DRAFTER_OBSERVED = [dict(t, cls="PreallocKVCache") for t in DRAFTER_TEMPLATE]
GOOD = dict(observed=DRAFTER_OBSERVED, template=DRAFTER_TEMPLATE)

CELLS_ON = kv.CELLS  # all five arms of both picks
IDS_ON = kv.CELL_IDS


def drafter(**changes):
    return dict(copy.deepcopy(GOOD), **changes)


def serve_on(rig, pick, arm, *, record=GOOD, honour_flag=True, always=False):
    """The MTP-ON worker for ``arm`` of ``pick``.  It records ``drafter_manifest`` only when the job asks for it
    (``honour_flag``), as the real worker must; ``always`` records it whatever the job says (for the no-flag judgement tests);
    ``record`` None records nothing at all."""
    rig.registry(kv.arm_text(arm, pick))
    expected = serve(rig, pick, arm, mtp_on=True)
    plain = rig.worker

    def worker(job):
        rec = plain(job)
        asked = job.get("drafter_manifest") is True
        if record is not None and (always or (honour_flag and asked)):
            rec["drafter_manifest"] = copy.deepcopy(record)
        return rec

    rig.worker = worker
    return expected


def engage(rig, pick, *extra):
    return rig.run(*ENGAGE, "--entry", pick, *extra)


def refusal_is(rig, got, message):
    assert got.refused == message, (got.code, got.text())
    assert rig.jobs == [], rig.jobs


def failed_with(got, pick, needle):
    state = got.state(pick)
    assert got.code not in (0, None), got.text()
    assert state.get("status") != "COMPLETED", got.text()
    assert any(needle in p for p in state.get("problems", [])), state.get("problems")
    return state


# ================================================================== acceptance 1: the CLI refusals
# every refusal happens before the lock is taken and before any job runs
def _lock_spy(rig):
    taken = []
    rig.hooks["acquire_lock"] = lambda: taken.append(1)
    return taken


@pytest.mark.parametrize(
    "argv",
    [
        ("--engagement-only", "--predictor", "on", "--entry", kv.UNIFORM),
        ("--engagement-only", "--entry", kv.UNIFORM),
        ("--engagement-only",),
    ],
    ids=("no-kv-arm", "no-kv-arm-no-predictor", "bare"),
)
def test_engagement_only_without_kv_arm_refuses_before_the_lock(rig, argv):
    # class: exercises new behaviour
    taken = _lock_spy(rig)
    got = rig.run(*argv)
    refusal_is(rig, got, "refused: ['--engagement-only needs --kv-arm --predictor on']")
    assert taken == []


@pytest.mark.parametrize(
    "predictor",
    [("--predictor", "off"), ()],
    ids=("predictor-off", "predictor-unset"),
)
def test_engagement_only_without_predictor_on_refuses_before_the_lock(rig, predictor):
    # class: exercises new behaviour
    rig.registry(kv.arm_text("TQ8", kv.UNIFORM))
    taken = _lock_spy(rig)
    got = rig.run("--kv-arm", *predictor, "--engagement-only", "--entry", kv.UNIFORM)
    refusal_is(rig, got, "refused: ['--engagement-only needs --kv-arm --predictor on']")
    assert taken == []


@pytest.mark.parametrize("arm", ("TQ8", "TQ4"))
def test_engagement_only_with_a_p369_summary_refuses_before_the_lock(rig, arm):
    # class: exercises new behaviour
    rig.registry(kv.arm_text(arm, kv.UNIFORM))
    summary = rig.summary(kv.UNIFORM, argv_on=None)
    taken = _lock_spy(rig)
    got = rig.run(*ENGAGE, "--entry", kv.UNIFORM, "--p369-summary", summary)
    refusal_is(rig, got, "refused: ['--engagement-only takes no --p369-summary']")
    assert taken == []


def test_engagement_only_on_an_arm_registry_serving_the_entry_mtp_off_refuses(rig):
    # class: exercises new behaviour
    base = registry_states.apply(kv.FIXTURE, {kv.UNIFORM})  # TQ4-OFF: committed MTP-OFF
    rig.registry(kv.arm_text("TQ4", kv.UNIFORM, base))
    assert registry_states.classify_kv_arm(
        rig.committed.read_text(), kv.FIXTURE, kv.UNIFORM
    )[0] == {
        kv.UNIFORM
    }  # premise
    got = engage(rig, kv.UNIFORM)
    assert got.code not in (0, None) and rig.jobs == []
    assert "MTP-OFF (no overlay is built)" in (got.refused or ""), got.text()


@pytest.mark.parametrize("arm", ("AF8", "TQ8"))
def test_engagement_only_refuses_a_non_empty_delta_arm_whose_entry_is_mtp_off(rig, arm):
    # class: exercises new behaviour
    base = registry_states.apply(kv.FIXTURE, {kv.PICK_A})
    rig.registry(kv.arm_text(arm, kv.PICK_A, base))
    got = engage(rig, kv.PICK_A)
    assert got.code not in (0, None) and rig.jobs == []
    assert "MTP-OFF (no overlay is built)" in (got.refused or ""), got.text()


# everything --kv-arm checks today is kept under --engagement-only
def test_engagement_only_keeps_the_one_entry_rule(rig):
    # class: exercises new behaviour
    serve_on(rig, kv.UNIFORM, "TQ8")
    got = rig.run(*ENGAGE, "--entry", kv.UNIFORM, "--entry", kv.PICK_A)
    assert got.code not in (0, None) and rig.jobs == [], got.text()
    got = rig.run(*ENGAGE)
    assert got.code not in (0, None) and rig.jobs == [], got.text()
    assert engage(rig, kv.UNIFORM).completed(kv.UNIFORM)  # control: one entry runs


@pytest.mark.parametrize("other", kv.OTHERS)
def test_engagement_only_keeps_the_p475_pick_rule(rig, other):
    # class: exercises new behaviour
    serve_on(rig, kv.UNIFORM, "TQ8")
    got = rig.run(*ENGAGE, "--entry", other)
    assert got.code not in (0, None) and rig.jobs == [], got.text()
    assert "not one of" in (
        got.refused or ""
    ), got.text()  # the pick rule's own refusal, not a parser error


@pytest.mark.parametrize(
    "key, value",
    [
        ("prefill_step_size", 1024),
        ("kv_prealloc_tokens", 131072),
        ("on_demand", "false"),
    ],
)
def test_engagement_only_keeps_the_classify_kv_arm_rule(rig, key, value):
    # class: exercises new behaviour
    serve_on(rig, kv.UNIFORM, "TQ8")
    rig.registry(kv.set_kv(kv.arm_text("TQ8", kv.UNIFORM), kv.UNIFORM, **{key: value}))
    got = engage(rig, kv.UNIFORM)
    assert got.code not in (0, None) and rig.jobs == [], got.text()
    assert "arm row" in (got.refused or ""), got.text()
    assert rig.kv_arm_calls  # the gate asked classify_kv_arm


# ================================================================== acceptance 2: every arm passes the gate with no summary
@pytest.mark.parametrize("pick, arm", CELLS_ON, ids=IDS_ON)
def test_each_arm_serves_mtp_on_exactly_as_committed_with_no_summary(rig, pick, arm):
    # class: exercises new behaviour
    serve_on(rig, pick, arm)
    got = engage(rig, pick)  # no --p369-summary
    assert got.completed(pick), got.text()
    assert got.code == 0
    state = got.state(pick)
    assert state["arm"] == "served_on"
    assert [s["path"] for s in rig.served] == [
        rig.committed
    ]  # no overlay, no derived registry
    assert not list(got.out.glob("overlay-*")) and not list(got.out.glob("mtp-off-*"))
    assert got.report["kv_arm"] == dict(
        entry=pick, delta=kv.as_json(kv.arm_delta(arm, pick))
    )
    assert rig.kv_arm_calls, "qualify never asked classify_kv_arm"
    assert state["problems"] == []


@pytest.mark.parametrize("pick, arm", CELLS_ON, ids=IDS_ON)
def test_the_served_argv_carries_the_mtp_flags_and_the_arms_kv_flags(rig, pick, arm):
    # class: exercises new behaviour
    serve_on(rig, pick, arm)
    got = engage(rig, pick)
    assert got.completed(pick), got.text()
    (job,) = rig.specs
    flags = flags_of(job["argv"])
    bits, scheme = kv.ARMS[arm]
    assert flags["--draft-kind"] == "mtp" and flags["--draft-model"]
    assert ("--kv-bits" in flags) == bool(bits)
    assert not bits or flags["--kv-bits"] == str(bits)
    assert flags["--kv-quant-scheme"] == (
        scheme or kv.FIXTURE_KV[pick]["kv_quant_scheme"]
    )
    assert flags["--quantized-kv-start"] == "0"
    # the argv is mlx-serve's own spelling of the committed arm registry, and passes today's two-way check
    entry = registry_states.settings(rig.committed.read_text())[pick]
    assert job["argv"] == mlx_serve_argv(entry, rig.stack)[1:]
    expected = qualify.expected_config(entry)
    assert expected["drafter"] is True
    assert (
        qualify.command_problems(
            expected, mlx_serve_argv(entry, rig.stack), rig.stack / ".venv" / "bin"
        )
        == []
    )


@pytest.mark.parametrize(
    "tamper",
    [
        lambda c: [a if a != "mtp" else "suffix" for a in c],
        lambda c: [
            x
            for i, x in enumerate(c)
            if c[max(i - 1, 0)] != "--draft-model" and x != "--draft-model"
        ],
        lambda c: c + ["--extra-flag", "x"],
        lambda c: [a if a != "--prefill-step-size" else "--prefill-chunk" for a in c],
    ],
    ids=("draft-kind", "no-draft-model", "extra-flag", "renamed-flag"),
)
def test_command_problems_still_refuses_a_tampered_mtp_on_argv(tamper):
    # class: preserves an invariant
    entry = kv.entry_for("TQ8", kv.UNIFORM)
    expected = qualify.expected_config(entry)
    cmd = mlx_serve_argv(entry, "/s")
    venv = "/s/.venv/bin"
    assert qualify.command_problems(expected, cmd, venv) == []
    assert qualify.command_problems(expected, tamper(cmd), venv) != []


def test_command_problems_refuses_a_wrong_kv_width_for_the_arm():
    # class: preserves an invariant
    expected = qualify.expected_config(kv.entry_for("TQ8", kv.UNIFORM))
    cmd = mlx_serve_argv(
        kv.entry_for("TQ4", kv.UNIFORM), "/s"
    )  # the TQ4 argv against the TQ8 arm
    assert qualify.command_problems(expected, cmd, "/s/.venv/bin") != []


def test_a_tampered_served_argv_fails_the_entry_and_runs_no_job(rig, monkeypatch):
    # class: exercises new behaviour
    serve_on(rig, kv.UNIFORM, "TQ8")
    real = qualify.full_command

    def tampered(stack, path, name):
        cmd, module = real(stack, path, name)
        return cmd + ["--extra-flag", "x"], module

    monkeypatch.setattr(qualify, "full_command", tampered)
    got = engage(rig, kv.UNIFORM)
    assert got.code not in (0, None), got.text()
    assert (
        got.state(kv.UNIFORM).get("status") == "FAILED"
    )  # a per-entry failure, past the parser and the gate
    assert got.report["engagement_only"] is True
    assert rig.jobs == []


def test_the_registry_changing_during_the_job_still_fails_the_entry(rig):
    # class: exercises new behaviour
    serve_on(rig, kv.UNIFORM, "TQ8")
    plain = rig.worker

    def worker(job):
        rec = plain(job)
        rig.committed.write_text(
            rig.committed.read_text() + "\n# edited during the job\n"
        )
        return rec

    rig.worker = worker
    got = engage(rig, kv.UNIFORM)
    assert (
        got.code not in (0, None) and got.state(kv.UNIFORM)["status"] != "COMPLETED"
    ), got.text()
    assert any("main_models.yaml" in p for p in got.state(kv.UNIFORM)["problems"])


def test_the_target_manifest_is_still_judged_against_the_arms_expected_manifest(rig):
    # class: exercises new behaviour
    serve_on(rig, kv.UNIFORM, "TQ8")
    plain = rig.worker

    def worker(job):  # a good drafter, but the TQ4 target cache while the arm asks TQ8
        return dict(plain(job), manifest=t415._tq_manifest())

    rig.worker = worker
    got = engage(rig, kv.UNIFORM)
    state = failed_with(got, kv.UNIFORM, "layer")
    assert state["status"] == "FAILED"


# ================================================================== the job: today's one request, plus the drafter flag
@pytest.mark.parametrize("pick", kv.PICKS)
def test_the_engagement_job_is_todays_one_request_plus_the_drafter_flag(rig, pick):
    # class: exercises new behaviour
    serve_on(rig, pick, "TQ8")
    got = engage(rig, pick)
    assert got.completed(pick), got.text()
    (job,) = rig.specs
    assert job["drafter_manifest"] is True
    assert job["mode"] == "qualify" and len(job["requests"]) == 1
    (request,) = job["requests"]
    expected = qualify.expected_config(kv.entry_for("TQ8", pick, mtp_on=True))
    assert (
        request["name"] == "max-context"
        and request["target"] == expected["prompt_tokens"]
    )
    assert request["generation_defaults"] == expected["generation_defaults"]
    assert set(request) == {
        "name",
        "model",
        "target",
        "prefix",
        "suffix",
        "seed",
        "generation_defaults",
        "timeout_s",
    }
    assert job["expect"]["drafter"] is True


def test_the_no_flag_kv_arm_job_does_not_ask_for_the_drafter_manifest(rig):
    # class: preserves an invariant
    rig.registry(kv.arm_text("AF8", kv.UNIFORM))
    serve(rig, kv.UNIFORM, "AF8")
    got = kv_run(rig, kv.UNIFORM)
    assert got.completed(kv.UNIFORM), got.text()
    assert not rig.specs[0].get("drafter_manifest")


def test_the_plain_committed_run_does_not_ask_for_the_drafter_manifest(rig):
    # class: preserves an invariant
    serve(rig, kv.UNIFORM, "TQ4")
    got = rig.run("--predictor", "off", "--entry", kv.UNIFORM)
    assert got.completed(kv.UNIFORM), got.text()
    assert not rig.specs[0].get("drafter_manifest")


# ================================================================== acceptance 3: the drafter judgement through the driver
def test_a_k3_conforming_drafter_manifest_gives_no_problem_and_is_carried_in_the_state(
    rig,
):
    # class: exercises new behaviour
    serve_on(rig, kv.UNIFORM, "TQ8")
    got = engage(rig, kv.UNIFORM)
    assert got.completed(kv.UNIFORM), got.text()
    state = got.state(kv.UNIFORM)
    assert state["problems"] == []
    assert state["drafter_manifest"] == GOOD
    assert (
        kvmanifest.drafter_problems(GOOD["observed"], GOOD["template"]) == []
    )  # premise


def test_a_quantized_drafter_leaf_fails_the_entry_naming_the_drafter_manifest(rig):
    # class: exercises new behaviour
    quantized = [
        dict(
            path="0",
            kind="affine",
            cls="PreallocQuantizedKVCache",
            batch=False,
            bits=8,
            group_size=64,
        ),
        DRAFTER_OBSERVED[1],
    ]
    serve_on(rig, kv.UNIFORM, "TQ8", record=drafter(observed=quantized))
    got = engage(rig, kv.UNIFORM)
    state = failed_with(got, kv.UNIFORM, "drafter cache")
    assert state["status"] == "FAILED" and got.code == 1
    assert state["drafter_manifest"]["observed"] == quantized  # the evidence is kept


def test_a_turboquant_drafter_leaf_fails_the_entry(rig):
    # class: exercises new behaviour
    tq = [
        dict(
            path="0",
            kind="tq",
            cls="TurboQuantKVCache",
            batch=False,
            bits=8.0,
            key_bits=8.0,
            value_bits=8.0,
            mode="mse",
        ),
        DRAFTER_OBSERVED[1],
    ]
    serve_on(rig, kv.PICK_A, "TQ8", record=drafter(observed=tq))
    failed_with(engage(rig, kv.PICK_A), kv.PICK_A, "drafter cache")


def test_a_continuous_batching_drafter_leaf_fails_the_entry(rig):
    # class: exercises new behaviour
    batch = [
        dict(DRAFTER_OBSERVED[0], cls="BatchKVCache", batch=True),
        DRAFTER_OBSERVED[1],
    ]
    serve_on(rig, kv.UNIFORM, "TQ8", record=drafter(observed=batch))
    failed_with(engage(rig, kv.UNIFORM), kv.UNIFORM, "continuous-batching")


@pytest.mark.parametrize(
    "observed",
    [
        DRAFTER_OBSERVED[:1],
        DRAFTER_OBSERVED
        + [dict(path="2", kind="native", cls="PreallocKVCache", batch=False)],
    ],
    ids=("a-layer-missing", "a-layer-extra"),
)
def test_a_drafter_cache_that_does_not_match_its_template_fails_the_entry(
    rig, observed
):
    # class: exercises new behaviour
    serve_on(rig, kv.UNIFORM, "TQ8", record=drafter(observed=observed))
    failed_with(engage(rig, kv.UNIFORM), kv.UNIFORM, "drafter cache")


def test_no_drafter_manifest_recorded_fails_the_entry(rig):
    # class: exercises new behaviour
    serve_on(rig, kv.UNIFORM, "TQ8", record=None)
    got = engage(rig, kv.UNIFORM)
    state = failed_with(got, kv.UNIFORM, NO_DRAFTER_MANIFEST)
    assert state["status"] == "FAILED" and got.code == 1
    assert "drafter_manifest" not in state  # nothing recorded, nothing carried


def test_a_worker_that_ignores_the_job_field_fails_the_entry(rig):
    # class: exercises new behaviour
    serve_on(rig, kv.UNIFORM, "TQ8", honour_flag=False)
    failed_with(engage(rig, kv.UNIFORM), kv.UNIFORM, NO_DRAFTER_MANIFEST)


@pytest.mark.parametrize(
    "record",
    [
        drafter(observed=[]),
        drafter(template=[]),
        dict(observed=[], template=[]),
        dict(observed=DRAFTER_OBSERVED),
        dict(template=DRAFTER_TEMPLATE),
        dict(observed=None, template=DRAFTER_TEMPLATE),
        dict(observed=DRAFTER_OBSERVED, template=None),
        dict(observed="x", template=DRAFTER_TEMPLATE),
        dict(observed=DRAFTER_OBSERVED, template={"0": "native"}),
        dict(observed=7, template=7),
        {},
        None,
        "manifest",
        [GOOD],
    ],
    ids=(
        "observed-empty",
        "template-empty",
        "both-empty",
        "template-absent",
        "observed-absent",
        "observed-none",
        "template-none",
        "observed-string",
        "template-mapping",
        "ints",
        "empty-dict",
        "none",
        "string",
        "list",
    ),
)
def test_an_empty_absent_or_malformed_drafter_manifest_fails_without_raising(
    rig, record
):
    # class: exercises new behaviour
    serve_on(rig, kv.UNIFORM, "TQ8", record=None)
    plain = rig.worker

    def worker(job):
        rec = plain(job)
        rec["drafter_manifest"] = copy.deepcopy(record)
        return rec

    rig.worker = worker
    got = engage(rig, kv.UNIFORM)
    state = got.state(kv.UNIFORM)
    assert got.code == 1 and state["status"] == "FAILED", got.text()
    assert NO_DRAFTER_MANIFEST in state["problems"], state["problems"]
    assert state.get("error") is None  # a problem, not an escaped exception


def test_without_the_flag_the_drafter_manifest_is_not_judged(rig):
    # class: preserves an invariant
    bad = drafter(observed=[], template=[])
    rig.registry(kv.arm_text("AF8", kv.UNIFORM))
    serve(rig, kv.UNIFORM, "AF8", drafter_manifest=bad)  # extra record key, flag absent
    got = kv_run(rig, kv.UNIFORM)
    assert got.completed(kv.UNIFORM), got.text()
    assert got.state(kv.UNIFORM)["problems"] == []
    assert "engagement_only" not in got.report


@pytest.mark.parametrize("arm", ("N16", "AF8", "TQ8", "TQ4"))
def test_request_problems_does_not_judge_the_drafter_manifest(arm):
    # class: preserves an invariant
    expected = qualify.expected_config(kv.entry_for(arm, kv.UNIFORM, mtp_on=True))
    good = kv.valid_record(expected, kv.arm_manifest(arm), sdpa_forced=0)
    assert qualify.request_problems(good, expected) == []
    for record in (
        drafter(observed=[]),
        drafter(
            observed=[
                dict(
                    path="0",
                    kind="affine",
                    cls="QuantizedKVCache",
                    batch=False,
                    bits=8,
                    group_size=64,
                )
            ]
        ),
        None,
    ):
        rec = dict(good, drafter_manifest=record)
        assert qualify.request_problems(rec, expected) == []


# ================================================================== acceptance 5: the report
@pytest.mark.parametrize(
    "pick, arm", ((kv.UNIFORM, "TQ8"), (kv.PICK_A, "AF4"), (kv.UNIFORM, "TQ4"))
)
def test_the_report_declares_engagement_only_and_certifies_false(
    rig, monkeypatch, pick, arm
):
    # class: exercises new behaviour
    from mlx_vlm.tests import test_p475a_qualify_driver as drv

    drv._references(
        monkeypatch, pick
    )  # a reference exists: a capacity comparison would be possible
    serve_on(rig, pick, arm)
    got = engage(rig, pick)
    assert got.completed(pick), got.text()
    assert got.report["engagement_only"] is True
    assert got.report["certifies"] is False
    state = got.state(pick)
    assert (
        "reference" in state and state["reference"] is None
    )  # no capacity comparison is made
    assert got.report["kv_arm"] == dict(
        entry=pick, delta=kv.as_json(kv.arm_delta(arm, pick))
    )
    assert got.report["predictor"] == "on"
    assert state["drafter_manifest"] == GOOD
    assert (
        state["capacity_gb"] > 0
    )  # capacity is still recorded; only the comparison is dropped


def test_a_failed_engagement_run_still_declares_itself_uncertified(rig):
    # class: exercises new behaviour
    serve_on(rig, kv.UNIFORM, "TQ8", record=None)
    got = engage(rig, kv.UNIFORM)
    assert got.code == 1
    assert got.report["engagement_only"] is True and got.report["certifies"] is False


def test_the_no_flag_kv_arm_report_is_unchanged(rig, monkeypatch):
    # class: preserves an invariant
    from mlx_vlm.tests import test_p475a_qualify_driver as drv

    drv._references(monkeypatch, kv.UNIFORM)
    rig.registry(kv.arm_text("AF8", kv.UNIFORM))
    serve(rig, kv.UNIFORM, "AF8")
    got = kv_run(rig, kv.UNIFORM)
    assert got.completed(kv.UNIFORM), got.text()
    assert "engagement_only" not in got.report and "certifies" not in got.report
    assert "drafter_manifest" not in got.state(kv.UNIFORM)
    ref = got.state(kv.UNIFORM)["reference"]
    assert ref["label"] == "non-comparable: KV arm" and ref["comparable"] is False


def test_the_plain_report_is_unchanged(rig):
    # class: preserves an invariant
    serve(rig, kv.UNIFORM, "TQ4")
    got = rig.run("--predictor", "off", "--entry", kv.UNIFORM)
    assert got.completed(kv.UNIFORM), got.text()
    assert "engagement_only" not in got.report and "certifies" not in got.report
    assert "kv_arm" not in got.report


# ================================================================== C199 for runs WITHOUT the flag is unchanged
@pytest.mark.parametrize(
    "pick, arm", ((kv.UNIFORM, "AF8"), (kv.UNIFORM, "TQ8"), (kv.PICK_A, "AF4"))
)
def test_without_the_flag_a_non_empty_delta_predictor_on_still_refuses_as_uncertified(
    rig, pick, arm
):
    # class: preserves an invariant
    serve_on(rig, pick, arm)
    argv_on = mlx_serve_argv(
        registry_states.settings(rig.committed.read_text())[pick], rig.stack
    )[1:]
    got = rig.run(
        "--kv-arm",
        "--predictor",
        "on",
        "--entry",
        pick,
        "--p369-summary",
        rig.summary(pick, argv_on=argv_on),
    )
    assert got.code not in (0, None) and rig.jobs == []
    assert "uncertified" in got.text().lower()


@pytest.mark.parametrize("extra", ((), ("--kv-arm",)), ids=("plain", "kv-arm"))
def test_without_the_flag_predictor_on_still_needs_a_p369_summary(rig, extra):
    # class: preserves an invariant
    serve_on(rig, kv.UNIFORM, "TQ4")
    got = rig.run(*extra, "--predictor", "on", "--entry", kv.UNIFORM)
    assert got.code not in (0, None) and rig.jobs == [], got.text()
    assert "P369 full-run PASS" in got.text()
