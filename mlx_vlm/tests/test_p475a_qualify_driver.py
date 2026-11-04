"""P475a v4: the REAL ``qualify.main`` -> ``drive`` path with ``--kv-arm`` (fork-only, dev; CPU only, temp git fork).

Written from SPEC v4 before the implementation, by a test writer who did not write it.  Each test carries a ``# class:``
comment: ``reproduces-failure`` (fails on baseline fork ``fd67082a`` for the named defect), ``invariant`` (passes on the baseline
and must keep passing), ``new-behaviour`` (needs ``--kv-arm`` / ``classify_kv_arm``).

What is real here (v4 acceptance 4-6; GLM r1-1 r1-2 r1-3 SR1, Sol r1-4): the classifier (``registry_states.classify`` and
``classify_kv_arm`` are only wrapped in pass-through spies, never replaced), ``request_problems``, ``command_problems``, the
argument parser, the registry parse / MTP-off derivation, the report.  What is mocked is the host and the worker boundary only
(lock, preflight, power, console script, the git / submodule bindings, ``run_job``, the model weights).  The served argv is
spelled by ``mlx_serve_argv`` below — a by-hand copy of mlx-serve's ``_build_command`` flag spelling taken from the committed
``test_p415_qualify._cmd`` fixture, NOT derived from ``qualify.expected_config`` (the circular ``_argv`` of the P542 rig is not
used).  The worker records come from ``test_p475a_kv_arm``'s hand-written per-arm manifests.
"""

import json
import sys
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import kvmanifest, orchestrate, qualify, registry_states  # noqa: E402

from mlx_vlm.tests import test_p415_qualify as t415  # noqa: E402
from mlx_vlm.tests import test_p475a_kv_arm as kv  # noqa: E402
from mlx_vlm.tests import test_p542_qualify_any as p542  # noqa: E402

# captured at import: the real functions, before any rig patches the module
REAL_REQUEST_PROBLEMS = qualify.request_problems
REAL_CLASSIFY = registry_states.classify
REAL_CLASSIFY_KV_ARM = getattr(registry_states, "classify_kv_arm", None)
FP = {f: f"fp-{f}" for f in qualify.FINGERPRINT}
VENV_BIN = "/.venv/bin"


# ------------------------------------------------------------------ mlx-serve's argv spelling, by hand
def mlx_serve_argv(entry, stack):
    """``process_manager._build_command`` for a registry entry (settings dict): the executable, then the flags in mlx-serve's
    order and spelling (``--kv-bits`` only when > 0, ``--cache-session-shrink on``, valueless ``--lazy-prompt-embeddings``,
    ``--quantized-kv-start`` always, the default ``--memory-limit-frac 0.85``)."""
    cmd = [str(Path(stack) / ".venv" / "bin" / "mlx_vlm.server")]
    cmd += ["--model", entry["hf_path"], "--host", "127.0.0.1", "--port", "8091"]
    cmd += ["--max-kv-size", str(entry["max_kv_cache_size"])]
    cmd += ["--kv-prealloc-tokens", str(entry["kv_prealloc_tokens"])]
    if entry.get("cache_session_shrink"):
        cmd += ["--cache-session-shrink", "on"]
    if entry.get("attention_policy") not in (None, "", "auto"):
        cmd += ["--attention-policy", entry["attention_policy"]]
    if entry.get("lazy_prompt_embeddings") is True:
        cmd += ["--lazy-prompt-embeddings"]
    if entry.get("mtp_verify_scan") not in (None, "", "per_query"):
        cmd += ["--mtp-verify-scan", entry["mtp_verify_scan"]]
    if entry.get("kv_bits"):
        cmd += ["--kv-bits", str(entry["kv_bits"])]
    if entry.get("kv_quant_scheme"):
        cmd += ["--kv-quant-scheme", entry["kv_quant_scheme"]]
    cmd += ["--prefill-step-size", str(entry["prefill_step_size"])]
    cmd += ["--generation-defaults", json.dumps(entry["generation_defaults"])]
    if entry.get("draft_kind"):
        cmd += ["--draft-kind", entry["draft_kind"]]
    if entry.get("draft_model"):
        cmd += ["--draft-model", entry["draft_model"]]
    cmd += ["--quantized-kv-start", str(entry.get("quantized_kv_start", 0))]
    cmd += ["--memory-limit-frac", "0.85"]
    return cmd


def flags_of(argv):
    """``{flag: value}`` of a worker argv (``cmd[1:]``)."""
    out, i = {}, 0
    while i < len(argv):
        if argv[i] in qualify.VALUELESS:
            out[argv[i]], i = "", i + 1
        else:
            out[argv[i]], i = argv[i + 1], i + 2
    return out


def served_flags(rig):
    return flags_of(rig.specs[-1]["argv"])


# ------------------------------------------------------------------ the rig: the P542 host mocks, with the REAL gate, argv check and request check
@pytest.fixture
def rig(tmp_path, monkeypatch):
    r = p542._Rig(tmp_path, monkeypatch)
    r.specs, r.expecteds, r.gate_calls, r.kv_arm_calls = [], [], [], []
    r.worker = None  # (job) -> the worker record; set by the test through ``serve``

    def classify(text, fixture):
        r.gate_calls.append(sys._getframe(1).f_globals.get("__name__"))
        return REAL_CLASSIFY(text, fixture)

    monkeypatch.setattr(registry_states, "classify", classify)
    if REAL_CLASSIFY_KV_ARM is not None:

        def classify_kv_arm(text, fixture, entry):
            r.kv_arm_calls.append(sys._getframe(1).f_globals.get("__name__"))
            return REAL_CLASSIFY_KV_ARM(text, fixture, entry)

        monkeypatch.setattr(registry_states, "classify_kv_arm", classify_kv_arm)

    def full_command(stack, path, name):
        text = Path(path).read_bytes().decode("utf-8")
        entry = registry_states.settings(text)[name]
        r.served.append(dict(name=name, path=Path(path), text=text))
        return mlx_serve_argv(entry, stack), "m"

    def request_problems(rec, expected):
        r.expecteds.append(expected)
        return REAL_REQUEST_PROBLEMS(rec, expected)

    def run_job(job, *a, **k):
        r.jobs.append(job["name"].rsplit("/", 1)[0])
        r.job_dirs.append(job["out_dir"])
        r.specs.append(job)
        return dict(requests=[dict(qualify=r.worker(job))])

    monkeypatch.setattr(qualify, "full_command", full_command)
    monkeypatch.setattr(qualify, "request_problems", request_problems)
    monkeypatch.setattr(orchestrate, "run_job", run_job)
    monkeypatch.setattr(qualify, "fingerprint_of", lambda rec, expected: dict(FP))
    plain_run = r.run

    def run(*extra):
        r.specs, r.expecteds, r.gate_calls, r.kv_arm_calls = [], [], [], []
        return plain_run(*extra)

    r.run = run
    return r


def serve(rig, pick, arm, *, mtp_on=False, forced=None, **changes):
    """The worker reports what a correct worker serving ``arm`` of ``pick`` reports (by-hand manifest, the arm's attention
    counter unless ``forced`` overrides it)."""
    entry = kv.entry_for(arm, pick, mtp_on=mtp_on)
    expected = qualify.expected_config(entry)
    manifest = kv.arm_manifest(arm)
    n = kv.right_forced(pick, arm) if forced is None else forced

    def worker(job):
        return dict(kv.valid_record(expected, manifest, sdpa_forced=n), **changes)

    rig.worker = worker
    return expected


KV = ("--kv-arm", "--predictor", "off")


def kv_run(rig, pick, *extra):
    return rig.run(*KV, "--entry", pick, *extra)


def refused(rig, got):
    assert got.code not in (0, None), got.text()
    assert rig.jobs == [], rig.jobs
    for name in (got.report or {}).get("entries", {}):
        assert got.state(name).get("status") != "COMPLETED", name


def control(rig, registry=None, arm="AF8"):
    """A valid arm that MUST complete in the same rig — a refusal below is then about the refused input, not a dead flag.
    ``registry`` and ``arm`` must match (the worker serves ``arm``'s manifest); the default is the AF8 arm of the default-attention pick.
    """
    rig.registry(kv.arm_text("AF8", kv.UNIFORM) if registry is None else registry)
    serve(rig, kv.UNIFORM, arm)
    got = kv_run(rig, kv.UNIFORM)
    assert got.completed(kv.UNIFORM), got.text()


# ================================================================== acceptance 4: every arm x pick, real classifier, real argv, real attention rule
# class: new-behaviour
@pytest.mark.parametrize("pick, arm", kv.CELLS, ids=kv.CELL_IDS)
def test_acceptance_4_each_arm_of_each_pick_completes_through_the_real_driver(
    rig, pick, arm
):
    rig.registry(kv.arm_text(arm, pick))
    expected = serve(rig, pick, arm)
    got = kv_run(rig, pick)
    assert got.completed(pick), got.text()
    state = got.state(pick)
    assert state["arm"] == "mtp_off"
    # the report records the arm: entry and delta (JSON lists)
    assert got.report["kv_arm"] == dict(
        entry=pick, delta=kv.as_json(kv.arm_delta(arm, pick))
    )
    # the served argv is mlx-serve's own spelling of the ARM registry's entry and passes the real two-way check
    assert (
        len(rig.served) == 1 and rig.served[0]["path"] != rig.committed
    )  # the derived MTP-off registry of the arm
    cmd = mlx_serve_argv(
        registry_states.settings(rig.served[0]["text"])[pick], rig.stack
    )
    assert (
        qualify.command_problems(
            qualify.expected_config(
                registry_states.settings(rig.served[0]["text"])[pick]
            ),
            cmd,
            rig.stack / ".venv" / "bin",
        )
        == []
    )
    flags = served_flags(rig)
    bits, scheme = kv.ARMS[arm]
    assert ("--kv-bits" in flags) == bool(bits) and (
        not bits or flags["--kv-bits"] == str(bits)
    )
    assert flags["--kv-quant-scheme"] == (
        scheme or kv.FIXTURE_KV[pick]["kv_quant_scheme"]
    )
    assert flags["--quantized-kv-start"] == "0"
    assert (
        "--draft-kind" not in flags and "--draft-model" not in flags
    )  # the MTP-OFF arm
    # what the worker is told to verify follows the argv, and the expected manifest is the arm's
    assert rig.specs[0]["expect"]["kv"] == kvmanifest.kv_from_argv(rig.specs[0]["argv"])
    assert rig.specs[0]["expect"]["kv"]["kv_bits"] == (float(bits) if bits else None)
    (seen,) = rig.expecteds
    assert seen["kv_bits"] == (bits or None) and seen["quantized_kv_start"] == 0
    assert (
        kvmanifest.manifest_problems(
            kv.arm_manifest(arm), t415._template(), qualify.expected_kv(seen)
        )
        == []
    )
    assert state["record"]["sdpa_forced"] == kv.right_forced(pick, arm)
    assert expected["attention_policy"] == ("fused_v1" if pick == kv.PICK_A else "auto")


# class: reproduces-failure (pick A AF8 / AF4 under fused_v1 complete with 0 forced calls; the baseline's rule refuses them)
@pytest.mark.parametrize("arm", ("AF8", "AF4"))
def test_fable_affine_arms_complete_with_zero_forced_sdpa_calls(rig, arm):
    rig.registry(kv.arm_text(arm, kv.PICK_A))
    serve(rig, kv.PICK_A, arm, forced=0)
    got = kv_run(rig, kv.PICK_A)
    assert got.completed(kv.PICK_A), got.text()
    assert got.state(kv.PICK_A)["record"]["sdpa_forced"] == 0
    assert not any("sdpa_forced" in p for p in got.state(kv.PICK_A)["problems"])


# class: new-behaviour (no separate ``classify`` call remains under --kv-arm: the gate and the mtp_off check read classify_kv_arm)
@pytest.mark.parametrize("pick, arm", (("pickA", "AF8"), ("uniform", "TQ8")))
def test_the_gate_is_classify_kv_arm_and_qualify_never_calls_classify(rig, pick, arm):
    pick = kv.PICK_A if pick == "pickA" else kv.UNIFORM
    rig.registry(kv.arm_text(arm, pick))
    serve(rig, pick, arm)
    assert kv_run(rig, pick).completed(pick)
    assert rig.kv_arm_calls, "qualify never asked classify_kv_arm"
    assert "p369_phase2.qualify" not in rig.gate_calls, rig.gate_calls


# class: new-behaviour (the arm is built on any accepted base state, driven through the real gate)
@pytest.mark.parametrize(
    "pick, base_off",
    [
        (kv.PICK_A, {kv.PICK_D}),
        (kv.PICK_A, {kv.UNIFORM, kv.PICK_C, kv.PICK_D}),
        (kv.UNIFORM, {kv.PICK_A}),
        (kv.UNIFORM, {kv.PICK_C}),
    ],
    ids=(
        "fable-ornith-off",
        "fable-three-off",
        "uniform-fable-off",
        "uniform-opus-off",
    ),
)
def test_an_arm_on_a_base_where_another_entry_is_mtp_off_runs(rig, pick, base_off):
    rig.registry(
        kv.arm_text("AF8", pick, registry_states.apply(kv.FIXTURE, set(base_off)))
    )
    serve(rig, pick, "AF8")
    got = kv_run(rig, pick)
    assert got.completed(pick), got.text()
    assert got.state(pick)["arm"] == "mtp_off"
    assert got.report["kv_arm"]["entry"] == pick


# class: new-behaviour (the entry under test is itself MTP-OFF in the committed arm registry: served as committed, predictor off)
@pytest.mark.parametrize("pick, arm", ((kv.PICK_A, "TQ8"), (kv.UNIFORM, "AF4")))
def test_an_arm_whose_entry_is_committed_mtp_off_is_served_as_committed(rig, pick, arm):
    base = registry_states.apply(kv.FIXTURE, {pick})
    committed = rig.registry(kv.arm_text(arm, pick, base))
    serve(rig, pick, arm)
    got = kv_run(rig, pick)
    assert got.completed(pick), got.text()
    assert got.state(pick)["arm"] == "served_off"
    assert [s["path"] for s in rig.served] == [committed]
    assert got.report["kv_arm"] == dict(
        entry=pick, delta=kv.as_json(kv.arm_delta(arm, pick))
    )


# class: new-behaviour (the committed registry itself is an arm with an empty delta)
@pytest.mark.parametrize("pick", kv.PICKS)
def test_the_committed_registry_is_an_arm_with_an_empty_delta(rig, pick):
    committed_arm = "N16" if pick == kv.PICK_A else "TQ4"
    serve(rig, pick, committed_arm)
    got = kv_run(rig, pick)
    assert got.completed(pick), got.text()
    assert got.report["kv_arm"] == dict(entry=pick, delta={})


# ================================================================== --predictor on (C233): served as committed, no overlay
# class: new-behaviour
@pytest.mark.parametrize(
    "pick, arm",
    (
        (kv.PICK_A, "N16"),
        (kv.UNIFORM, "TQ4"),
    ),  # v4.1: only an EMPTY delta; the other cells refuse (test_p475a_fixround)
    ids=("fable-N16", "uniform-TQ4"),
)
def test_predictor_on_serves_the_committed_mtp_on_arm_registry_without_an_overlay(
    rig, pick, arm
):
    committed = rig.registry(kv.arm_text(arm, pick))
    argv_on = mlx_serve_argv(
        registry_states.settings(kv.arm_text(arm, pick))[pick], rig.stack
    )[1:]
    summary = rig.summary(pick, argv_on=argv_on)
    expected = serve(rig, pick, arm, mtp_on=True)
    got = rig.run(
        "--kv-arm", "--predictor", "on", "--entry", pick, "--p369-summary", summary
    )
    assert got.completed(pick), got.text()
    state = got.state(pick)
    assert state["arm"] == "served_on" and "overlay_sha256" not in state
    assert {s["path"] for s in rig.served} == {
        committed
    }  # the committed arm registry as is: no overlay, no derived file
    assert not list(got.out.glob("overlay-*")) and not list(got.out.glob("mtp-off-*"))
    flags = served_flags(rig)
    assert flags["--draft-kind"] == "mtp" and flags["--draft-model"].endswith(
        "-mtp-drafter"
    )
    bits, scheme = kv.ARMS[arm]
    assert ("--kv-bits" in flags) == bool(bits)
    # the expected manifest includes the drafter's cache: the job expects a drafter and the record carries its live checks
    assert (
        rig.specs[0]["expect"]["drafter"] is True
        and rig.expecteds[0]["drafter"] is True
    )
    assert state["record"]["drafter_live_checks"] == [[]]
    assert expected["drafter"] is True
    assert got.report["kv_arm"] == dict(
        entry=pick, delta=kv.as_json(kv.arm_delta(arm, pick))
    )


# class: new-behaviour (an arm whose entry is MTP-OFF is never restored under --kv-arm: it refuses, builds no overlay)
@pytest.mark.parametrize("pick, arm", ((kv.PICK_A, "AF8"), (kv.UNIFORM, "TQ8")))
def test_predictor_on_refuses_an_arm_whose_entry_is_mtp_off(rig, pick, arm):
    base = registry_states.apply(kv.FIXTURE, {pick})
    rig.registry(kv.arm_text(arm, pick, base))
    serve(rig, pick, arm)
    assert kv_run(rig, pick).completed(
        pick
    )  # control: the same registry runs predictor-off
    # a valid P369 pass for exactly this arm's MTP-ON argv, so the ONLY reason left to refuse is the MTP-OFF entry
    on_text = kv.arm_text(arm, pick)
    summary = rig.summary(
        pick,
        argv_on=mlx_serve_argv(registry_states.settings(on_text)[pick], rig.stack)[1:],
    )
    got = rig.run(
        "--kv-arm", "--predictor", "on", "--entry", pick, "--p369-summary", summary
    )
    refused(rig, got)
    assert not list(got.out.glob("overlay-*")) if got.out.exists() else True


# ================================================================== acceptance 6: refusals (each with a completing control in the same rig)
# class: new-behaviour
def test_refuses_without_an_entry_and_with_two_entries(rig):
    control(rig)
    refused(rig, rig.run(*KV))
    refused(rig, rig.run(*KV, "--entry", kv.PICK_A, "--entry", kv.UNIFORM))


# class: new-behaviour (Sol r1-1)
@pytest.mark.parametrize("other", kv.OTHERS)
def test_refuses_a_third_entry_even_when_the_registry_is_an_accepted_state(rig, other):
    control(rig, kv.FIXTURE, "TQ4")
    refused(rig, kv_run(rig, other))


# class: new-behaviour (Sol r1-1: a valid KV change on a third entry)
@pytest.mark.parametrize("other", kv.OTHERS)
def test_refuses_a_third_entry_whose_kv_keys_were_changed(rig, other):
    control(rig, kv.FIXTURE, "TQ4")
    rig.registry(kv.set_kv(kv.FIXTURE, other, kv_bits=8, kv_quant_scheme="uniform"))
    refused(rig, kv_run(rig, other))


# class: new-behaviour
def test_refuses_an_unknown_entry_listing_the_registry(rig):
    control(rig, kv.FIXTURE, "TQ4")
    got = kv_run(rig, "No-Such-Model")
    refused(rig, got)
    assert "No-Such-Model" in got.text()


# class: new-behaviour (Sol r1-2 / r1-3: a fractional width was classified valid and truncated to TQ4)
@pytest.mark.parametrize("pick", kv.PICKS)
@pytest.mark.parametrize("bits", ("4.5", "8.5", "0.5"))
def test_refuses_a_fractional_width(rig, pick, bits):
    control(rig)
    rig.registry(
        kv.set_kv(kv.FIXTURE, pick, kv_bits=bits, kv_quant_scheme="turboquant")
    )
    refused(rig, kv_run(rig, pick))


# class: new-behaviour (Sol r1-3: `.inf` raised OverflowError)
@pytest.mark.parametrize("pick", kv.PICKS)
@pytest.mark.parametrize("bits", (".inf", ".nan", "-.inf"))
def test_refuses_a_non_finite_width(rig, pick, bits):
    control(rig)
    rig.registry(
        kv.set_kv(kv.FIXTURE, pick, kv_bits=bits, kv_quant_scheme="turboquant")
    )
    refused(rig, kv_run(rig, pick))


# class: new-behaviour (Sol r1-3: `models: [null]` raised AttributeError — a refusal, not a crash; a raised exception fails here)
@pytest.mark.parametrize("item", ("null", "just-a-string", "[1, 2]"))
def test_refuses_a_registry_with_a_malformed_models_item(rig, item):
    control(rig)
    text = kv.arm_text("AF8", kv.UNIFORM).rstrip("\n") + f"\n  - {item}\n"
    rig.registry(text)
    refused(rig, kv_run(rig, kv.UNIFORM))


# class: new-behaviour (`models: [null]` as the whole registry)
def test_refuses_a_registry_that_is_only_a_null_models_list(rig):
    control(rig)
    rig.registry("models: [null]\n")
    refused(rig, kv_run(rig, kv.UNIFORM))


# class: new-behaviour (KV-arm acceptance is KV keys only: one non-KV difference refuses before any job)
@pytest.mark.parametrize("pick", kv.PICKS)
@pytest.mark.parametrize(
    "key, value",
    (
        ("prefill_step_size", 1024),
        ("kv_prealloc_tokens", 131072),
        ("max_kv_cache_size", 131072),
    ),
)
def test_refuses_a_non_kv_key_changed_alongside_the_arm(rig, pick, key, value):
    control(rig)
    rig.registry(kv.set_kv(kv.arm_text("AF8", pick), pick, **{key: value}))
    refused(rig, kv_run(rig, pick))


# class: new-behaviour
@pytest.mark.parametrize("pick", kv.PICKS)
def test_refuses_a_second_entry_changed(rig, pick):
    control(rig)
    other = kv.UNIFORM if pick == kv.PICK_A else kv.PICK_A
    rig.registry(kv.arm_text("TQ8", other, kv.arm_text("AF8", pick)))
    refused(rig, kv_run(rig, pick))


# class: new-behaviour
@pytest.mark.parametrize("pick", kv.PICKS)
def test_refuses_a_nonzero_quantized_kv_start(rig, pick):
    control(rig)
    rig.registry(kv.set_kv(kv.arm_text("AF8", pick), pick, quantized_kv_start=5000))
    refused(rig, kv_run(rig, pick))


# class: new-behaviour (an MTP-key change plus a KV change, outside the 16 states)
def test_refuses_a_partial_mtp_change_with_a_kv_arm(rig):
    control(rig)
    lines = kv.arm_text("AF8", kv.UNIFORM).split("\n")
    start, end = registry_states._span(lines, kv.UNIFORM)
    del lines[
        next(i for i in range(start, end) if lines[i].startswith("    draft_kind: mtp"))
    ]
    rig.registry("\n".join(lines))
    refused(rig, kv_run(rig, kv.UNIFORM))


# class: new-behaviour (a scheme / width outside the five rows)
@pytest.mark.parametrize("pick", kv.PICKS)
@pytest.mark.parametrize(
    "bits, scheme", ((6, "turboquant"), (4, "affine"), (8, "mse"), (2, "uniform"))
)
def test_refuses_a_width_or_scheme_outside_the_five_rows(rig, pick, bits, scheme):
    control(rig)
    rig.registry(kv.set_kv(kv.FIXTURE, pick, kv_bits=bits, kv_quant_scheme=scheme))
    refused(rig, kv_run(rig, pick))


# class: new-behaviour (the parser: --kv-arm is a flag, off by default)
def test_the_flag_is_a_plain_switch_off_by_default():
    base = [
        "--stack-root",
        "s",
        "--fork-root",
        "f",
        "--out",
        "o",
        "--reviewed-head",
        "h",
    ]
    assert qualify.build_parser().parse_args(base).kv_arm is False
    assert qualify.build_parser().parse_args(base + ["--kv-arm"]).kv_arm is True


# ================================================================== acceptance 5: a wrong attention counter fails the entry
WRONG_FORCED = [
    (kv.PICK_A, "AF8", 640),
    (kv.PICK_A, "AF4", 640),
    (kv.PICK_A, "N16", 0),
    (kv.PICK_A, "TQ8", 0),
    (kv.PICK_A, "TQ4", 0),
    (kv.UNIFORM, "AF8", 640),
    (kv.UNIFORM, "TQ4", 640),
    (kv.UNIFORM, "N16", 640),
]


# class: new-behaviour (GLM r1-3: the real request_problems through the real driver; the entry is NOT completed)
@pytest.mark.parametrize(
    "pick, arm, forced",
    WRONG_FORCED,
    ids=[
        f"{'pickA' if p == kv.PICK_A else 'uniform'}-{a}-{n}"
        for p, a, n in WRONG_FORCED
    ],
)
def test_acceptance_5_a_wrong_sdpa_forced_for_the_arm_fails_the_entry(
    rig, pick, arm, forced
):
    control(rig)
    rig.registry(kv.arm_text(arm, pick))
    serve(rig, pick, arm, forced=forced)
    got = kv_run(rig, pick)
    state = got.state(pick)
    assert got.code == 1 and state["status"] == "FAILED", got.text()
    assert any("sdpa_forced" in p for p in state["problems"]), state["problems"]
    assert [
        p for p in state["problems"] if "sdpa_forced" not in p
    ] == []  # the only wrong thing is the counter
    assert (
        got.report["kv_arm"]["entry"] == pick
    )  # recorded before the entry runs: it survives the entry's failure
    assert "capacity_gb" not in state  # a failed entry reports no capacity


# class: new-behaviour (the same record is accepted for the arm it is right for: the counter, not the record, is the difference)
@pytest.mark.parametrize(
    "pick, arm, forced",
    WRONG_FORCED,
    ids=[
        f"{'pickA' if p == kv.PICK_A else 'uniform'}-{a}-{n}"
        for p, a, n in WRONG_FORCED
    ],
)
def test_acceptance_5_the_right_counter_for_the_same_arm_completes(
    rig, pick, arm, forced
):
    rig.registry(kv.arm_text(arm, pick))
    serve(rig, pick, arm, forced=kv.right_forced(pick, arm))
    assert kv_run(rig, pick).completed(pick)


# class: new-behaviour (a worker that reports the wrong manifest for the arm — N16's native leaves for an AF8 registry — fails)
@pytest.mark.parametrize("pick", kv.PICKS)
def test_a_manifest_of_another_arm_fails_the_entry(rig, pick):
    control(rig)
    rig.registry(kv.arm_text("AF8", pick))
    expected = qualify.expected_config(kv.entry_for("AF8", pick))
    wrong = kv.valid_record(
        expected, kv.arm_manifest("N16"), sdpa_forced=kv.right_forced(pick, "AF8")
    )
    rig.worker = lambda job: wrong
    got = kv_run(rig, pick)
    assert got.code == 1 and got.state(pick)["status"] == "FAILED", got.text()
    assert any("layer" in p for p in got.state(pick)["problems"])


# ================================================================== the reference comparison is labelled, never a refusal (C199)
def _references(monkeypatch, name):
    ref = dict(peak_gb=40.0, source="synthetic", source_path="p", fingerprint=dict(FP))
    monkeypatch.setattr(qualify, "load_references", lambda path, **k: {name: ref})


# class: new-behaviour
@pytest.mark.parametrize(
    "pick, arm",
    ((kv.PICK_A, "AF8"), (kv.PICK_A, "TQ8"), (kv.UNIFORM, "AF4"), (kv.UNIFORM, "N16")),
)
def test_the_reference_block_is_labelled_non_comparable_for_a_kv_arm(
    rig, monkeypatch, pick, arm
):
    _references(monkeypatch, pick)
    rig.registry(kv.arm_text(arm, pick))
    serve(rig, pick, arm)
    got = kv_run(rig, pick)
    assert got.completed(pick), got.text()  # never a refusal
    ref = got.state(pick)["reference"]
    assert ref["label"] == "non-comparable: KV arm" and ref["comparable"] is False
    assert (
        ref["reference_gb"] == 40.0
        and isinstance(ref["delta_gb"], float)
        and ref["source"] == "synthetic"
    )


# ================================================================== the no-flag path: byte-for-byte today's behaviour
# class: invariant
@pytest.mark.parametrize("pick, arm", ((kv.PICK_A, "N16"), (kv.UNIFORM, "TQ4")))
def test_without_the_flag_the_committed_registry_runs_as_today(
    rig, monkeypatch, pick, arm
):
    _references(monkeypatch, pick)
    serve(rig, pick, arm)
    got = rig.run("--predictor", "off", "--entry", pick)
    assert got.completed(pick), got.text()
    assert "kv_arm" not in got.report
    ref = got.state(pick)["reference"]
    assert (
        ref["comparable"] is True and "label" not in ref
    )  # the same fingerprint stays comparable; no KV-arm label
    assert set(ref) == {"reference_gb", "delta_gb", "differs", "comparable", "source"}
    assert rig.kv_arm_calls == []
    assert got.state(pick)["arm"] == "mtp_off"
    assert {s["path"] for s in rig.served} != {
        rig.committed
    }  # the derived MTP-off registry, as today


# class: invariant (the committed served-ON path: today's argv binding and the P369 gate, no kv_arm anywhere)
@pytest.mark.parametrize("pick, arm", ((kv.PICK_A, "N16"), (kv.UNIFORM, "TQ4")))
def test_without_the_flag_predictor_on_serves_the_committed_registry_as_today(
    rig, pick, arm
):
    argv_on = mlx_serve_argv(registry_states.settings(kv.FIXTURE)[pick], rig.stack)[1:]
    serve(rig, pick, arm, mtp_on=True)
    got = rig.run(
        "--predictor",
        "on",
        "--entry",
        pick,
        "--p369-summary",
        rig.summary(pick, argv_on=argv_on),
    )
    assert got.completed(pick), got.text()
    assert "kv_arm" not in got.report and got.state(pick)["arm"] == "served_on"
    assert [s["path"] for s in rig.served] == [rig.committed]
    assert rig.kv_arm_calls == []


# class: invariant (a refusal that exists today without the flag still exists: a bad name, a duplicate --entry)
def test_without_the_flag_todays_refusals_stay(rig):
    serve(rig, kv.UNIFORM, "TQ4")
    got = rig.run("--predictor", "off", "--entry", "No-Such-Model")
    assert (
        got.code not in (0, None) and rig.jobs == [] and "No-Such-Model" in got.text()
    )
    assert rig.run("--predictor", "off", "--entry", kv.UNIFORM).completed(kv.UNIFORM)


# class: invariant (the real attention rule through the no-flag driver: the committed pick A entry needs forced calls)
def test_without_the_flag_the_committed_fable_entry_still_fails_a_zero_forced_record(
    rig,
):
    serve(rig, kv.PICK_A, "N16", forced=0)
    got = rig.run("--predictor", "off", "--entry", kv.PICK_A)
    state = got.state(kv.PICK_A)
    assert state["status"] == "FAILED" and any(
        "sdpa_forced" in p for p in state["problems"]
    )
    assert "kv_arm" not in got.report
