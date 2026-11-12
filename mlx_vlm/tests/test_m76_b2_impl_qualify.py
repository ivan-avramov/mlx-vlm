"""M76 B2 acceptance 2: ``qualify`` on the X / V / XO arm registries of ``Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed``
through the REAL ``qualify.main`` -> ``drive`` path (the P475a driver rig: real classifier, argv check, request check; the
host and worker mocked). The served argv is mlx-serve's ``_build_command`` spelling, pinned below as captured from mlx-serve
``3b02e4c`` on these arm registries (``REAL_ARGV``) and reproduced by ``serve_argv``. CPU only.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import orchestrate, qualify, registry_states  # noqa: E402

from mlx_vlm.tests import test_m76_b2_impl_arms as arms  # noqa: E402
from mlx_vlm.tests import test_p475a_kv_arm as kv  # noqa: E402
from mlx_vlm.tests import test_p475a_qualify_driver as drv  # noqa: E402
from mlx_vlm.tests import test_p542_qualify_any as p542  # noqa: E402
from mlx_vlm.tests.test_p475a_qualify_driver import (  # noqa: E402,F401  (the fixture)
    rig,
)

PICK = arms.PICK
FIXTURE = arms.FIXTURE
_GD = (
    '{"temperature": 0.5, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "max_tokens": 102400, '
    '"thinking_budget": 81920, "enable_thinking": true, "reasoning_effort": "medium"}'
)
_HEAD = [
    "--model",
    "caslca/Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed",
    "--host",
    "127.0.0.1",
    "--port",
    "8091",
    "--max-kv-size",
    "262144",
    "--kv-prealloc-tokens",
    "262144",
    "--cache-session-shrink",
    "on",
    "--attention-policy",
    "fused_v1",
    "--lazy-prompt-embeddings",
]
_DRAFT = [
    "--draft-kind",
    "mtp",
    "--draft-model",
    "caslca/Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed-mtp-drafter",
]
_TAIL = ["--quantized-kv-start", "0", "--memory-limit-frac", "0.85"]
_KV4 = [
    "--kv-bits",
    "4",
    "--kv-quant-scheme",
    "turboquant",
    "--prefill-step-size",
    "512",
    "--generation-defaults",
    _GD,
]
# mlx-serve 3b02e4c ``process_manager._build_command(...)[1:]`` for the pick in each arm registry (captured, by hand)
REAL_ARGV = {
    "N16": _HEAD
    + [
        "--mtp-verify-scan",
        "joint_v1",
        "--kv-quant-scheme",
        "turboquant",
        "--prefill-step-size",
        "512",
        "--generation-defaults",
        _GD,
    ]
    + _DRAFT
    + _TAIL,
    "X": _HEAD
    + ["--mtp-verify-scan", "joint_v1", "--tq4-attention", "m72_exact"]
    + _KV4
    + _DRAFT
    + _TAIL,
    "V": _HEAD
    + ["--mtp-verify-scan", "joint_v1", "--tq4-attention", "m72_v1"]
    + _KV4
    + _DRAFT
    + _TAIL,
    "XO": _HEAD + ["--tq4-attention", "m72_exact"] + _KV4 + _TAIL,
}


_p475a = drv.mlx_serve_argv  # captured before any test patches the driver module


def serve_argv(entry, stack):
    """mlx-serve's ``_build_command`` spelling: the P475a by-hand copy plus ``--tq4-attention`` (``type: vision`` with a
    mode), placed after ``--mtp-verify-scan`` and before ``--kv-bits`` as mlx-serve 3b02e4c places it.
    """
    cmd = _p475a(entry, stack)
    mode = entry.get("tq4_attention")
    if entry.get("type") == "vision" and mode:
        at = (
            cmd.index("--kv-bits")
            if "--kv-bits" in cmd
            else cmd.index("--kv-quant-scheme")
        )
        cmd[at:at] = ["--tq4-attention", mode]
    return cmd


def entry_of(text):
    return registry_states.settings(text)[PICK]


def arm_registry(arm):
    return FIXTURE if arm == "N16" else arms.arm_text(arm)


@pytest.mark.parametrize("arm", ("N16", "X", "V", "XO"))
def test_serve_argv_is_mlx_serves_real_spelling(arm):
    assert serve_argv(entry_of(arm_registry(arm)), "/s")[1:] == REAL_ARGV[arm]


@pytest.fixture
def real_argv(rig, monkeypatch):
    """The rig serves mlx-serve's real spelling (with ``--tq4-attention``). M77's production (the tuple's) is the rig's
    P0 — a real strict ancestor of the reviewed head, distinct from ``orchestrate.PRODUCTION`` (the rig's P).
    """
    monkeypatch.setattr(drv, "mlx_serve_argv", serve_argv)
    m77 = rig.P0[:8]
    monkeypatch.setattr(orchestrate, "M77_PRODUCTION", m77)
    monkeypatch.setitem(orchestrate.TUPLES["m76-x"], "production", m77)
    return rig


def worker_for(rig, text, *, mtp_on):
    """A correct worker for the pick in ``text``: the TQ4 manifest (the mode changes no cache class), forced calls."""
    entry = dict(entry_of(text))
    if not mtp_on:
        for key in qualify.MTP_KEYS:
            entry.pop(key, None)
    expected = qualify.expected_config(entry)
    rec = kv.valid_record(
        expected, kv.arm_manifest("TQ4"), sdpa_forced=kv.right_forced(PICK, "TQ4")
    )
    if mtp_on:
        template = [dict(path="0", kind="native", cls="KVCache", batch=False)]
        rec["drafter_manifest"] = (
            dict(  # P658's shape: the pre-allocating variant of each drafter layer
                observed=[dict(t, cls="PreallocKVCache") for t in template],
                template=template,
            )
        )
    rig.worker = lambda job: rec
    return expected


def flags(argv):
    return drv.flags_of(argv)


def json_delta(mode):
    return {"kv_bits": [0, 4], "tq4_attention": [None, mode]}


# ================================================================== --predictor off on X / V / XO
@pytest.mark.parametrize("arm", ("X", "V", "XO"))
def test_predictor_off_completes_on_each_arm_commit(real_argv, arm):
    rig = real_argv
    committed = rig.registry(arms.arm_text(arm))
    worker_for(rig, arms.arm_text(arm), mtp_on=False)
    got = drv.kv_run(rig, PICK)
    assert got.completed(PICK), got.text()
    state = got.state(PICK)
    assert state["arm"] == ("served_off" if arm == "XO" else "mtp_off")
    assert got.report["kv_arm"] == dict(
        entry=PICK, delta=json_delta(arms.MODES[arm]), tq4_attention=arms.MODES[arm]
    )
    served = rig.specs[-1]["argv"]
    f = flags(served)
    assert f["--tq4-attention"] == arms.MODES[arm]
    assert not set(f) & {"--draft-kind", "--draft-model", "--mtp-verify-scan"}
    assert f["--kv-bits"] == "4" and f["--kv-quant-scheme"] == "turboquant"
    if arm == "XO":
        assert [s["path"] for s in rig.served] == [committed]
    # the OFF argv of X is exactly XO's deployed argv
    if arm in ("X", "XO"):
        assert served == REAL_ARGV["XO"]
    assert state["record"]["sdpa_forced"] == 640
    # the worker is told to check the served hook's mode
    assert rig.specs[-1]["expect"]["tq4_attention"] == arms.MODES[arm]


def test_a_job_without_a_mode_carries_no_mode_expectation(real_argv):
    rig = real_argv
    rig.registry(kv.arm_text("TQ4", PICK))
    worker_for(rig, kv.arm_text("TQ4", PICK), mtp_on=False)
    assert drv.kv_run(rig, PICK).completed(PICK)
    assert "tq4_attention" not in rig.specs[-1]["expect"]


def test_the_off_derivation_of_x_strips_the_mtp_keys_and_keeps_the_mode():
    derived = qualify.mtp_off_registry(arms.arm_text("X"), PICK)
    entry = entry_of(derived)
    assert entry["tq4_attention"] == "m72_exact" and entry["kv_bits"] == 4
    assert not set(entry) & set(qualify.MTP_KEYS)
    assert serve_argv(entry, "/s")[1:] == REAL_ARGV["XO"]
    # every other byte of the registry unchanged: the derived text is the arm minus the three MTP lines
    lines = arms.arm_text("X").split("\n")
    start, end = registry_states._span(lines, PICK)
    drop = [
        i
        for i in range(start, end)
        if lines[i].split(":")[0].strip() in qualify.MTP_KEYS
    ]
    assert len(drop) == 3
    assert derived.split("\n") == [l for i, l in enumerate(lines) if i not in drop]
    assert registry_states.classify_kv_arm(derived, FIXTURE, PICK)[0] == {PICK}


@pytest.mark.parametrize("arm", ("X", "V", "XO"))
def test_the_expected_config_and_the_real_argv_agree(arm):
    entry = entry_of(arms.arm_text(arm))
    expected = qualify.expected_config(entry)
    assert expected["flags"]["--tq4-attention"] == arms.MODES[arm]
    cmd = ["/s/.venv/bin/mlx_vlm.server"] + REAL_ARGV[arm]
    assert qualify.command_problems(expected, cmd, "/s/.venv/bin") == []
    # without the flag the two-way check names it
    stripped = [a for a in cmd if a not in ("--tq4-attention", arms.MODES[arm])]
    assert any(
        "--tq4-attention" in p
        for p in qualify.command_problems(expected, stripped, "/s/.venv/bin")
    )


def test_the_expected_manifest_of_the_mode_arms_is_todays_tq4_manifest():
    from p369_phase2 import kvmanifest

    from mlx_vlm.tests import test_p415_qualify as t415

    want = kvmanifest.expected_manifest(
        t415._template(),
        qualify.expected_kv(qualify.expected_config(kv.entry_for("TQ4", PICK))),
    )
    for arm in ("X", "V", "XO"):
        got = kvmanifest.expected_manifest(
            t415._template(),
            qualify.expected_kv(qualify.expected_config(entry_of(arms.arm_text(arm)))),
        )
        assert got == want


@pytest.mark.parametrize("forced", (0,))
def test_a_mode_arm_without_forced_sdpa_calls_fails(real_argv, forced):
    """The native leaf under fused_v1 requires forced calls, mode or not (the sdpa_forced rule unchanged)."""
    rig = real_argv
    rig.registry(arms.arm_text("X"))
    expected = worker_for(rig, arms.arm_text("X"), mtp_on=False)
    rig.worker = lambda job: kv.valid_record(
        expected, kv.arm_manifest("TQ4"), sdpa_forced=forced
    )
    got = drv.kv_run(rig, PICK)
    assert got.state(PICK)["status"] == "FAILED"
    assert all("sdpa_forced" in p for p in got.state(PICK)["problems"])


# ================================================================== --predictor on --engagement-only (unchanged rule)
@pytest.mark.parametrize("arm", ("X", "V"))
def test_engagement_only_completes_on_the_mtp_on_arms(real_argv, arm):
    rig = real_argv
    committed = rig.registry(arms.arm_text(arm))
    worker_for(rig, arms.arm_text(arm), mtp_on=True)
    got = rig.run("--kv-arm", "--predictor", "on", "--engagement-only", "--entry", PICK)
    assert got.completed(PICK), got.text()
    assert got.state(PICK)["arm"] == "served_on"
    assert [s["path"] for s in rig.served] == [committed]
    assert rig.specs[-1]["argv"] == REAL_ARGV[arm]
    assert got.report["engagement_only"] is True and "certified_by" not in got.report
    assert got.report["kv_arm"]["tq4_attention"] == arms.MODES[arm]


def test_engagement_only_on_xo_refuses(real_argv):
    rig = real_argv
    rig.registry(arms.arm_text("XO"))
    worker_for(rig, arms.arm_text("XO"), mtp_on=True)
    drv.refused(
        rig,
        rig.run("--kv-arm", "--predictor", "on", "--engagement-only", "--entry", PICK),
    )


# ================================================================== --predictor on with a delta: --certified-by
def record(
    rig,
    *,
    tuple_name="m76-x",
    e1c="PASS",
    e2="PASS",
    stack_head=None,
    fork_head=None,
    argv_on=None,
    manifest_tuple=None,
    picks=None,
    status="PASS",
    production=None,
):
    """A P369 phase-2 tuple FULL-run record as ``orchestrate`` writes it: ``manifest.json`` (every binding key) and
    ``summary.json`` beside it whose ``binding`` is ``orchestrate.binding(manifest)`` and whose ``manifest`` block is
    ``Campaign._execute``'s; under the campaign root, bound to the tuple's production, this rig's fork head and stack root
    commit. ``manifest_tuple`` replaces the manifest's tuple AFTER the summary was bound.
    """
    rig.runs += 1
    run = rig.workdir / "p369" / "phase2" / f"m76-x-{rig.runs}"
    run.mkdir(parents=True)
    prod = (
        orchestrate.TUPLES["m76-x"]["production"] if production is None else production
    )
    fork = rig.R if fork_head is None else fork_head
    stack = "s" * 40 if stack_head is None else stack_head
    heads = dict(stack_head=stack, off_stack_head="o" * 40)
    if tuple_name is not None:
        heads["tuple"] = tuple_name
    manifest = dict(
        production=prod,
        fork_head=fork,
        harness_tree="h" * 40,
        reviewed_head=fork,
        versions=dict(mlx="0.32.4"),
        env_digest="e" * 64,
        stack_bench=dict(tree="b" * 40),
        **heads,
        picks={
            PICK: dict(
                overlay_sha256=None,
                argv_on=REAL_ARGV["X"] if argv_on is None else argv_on,
                argv_off=REAL_ARGV["XO"],
                params=dict(temperature=0.0),
                target=dict(revision="t", files={}),
                drafter=dict(revision="d", files={}),
                registry_on_sha256="1" * 64,
                registry_off_sha256="2" * 64,
            )
        },
    )
    summary = dict(
        phase="full",
        status=status,
        binding=orchestrate.binding(manifest),
        manifest=dict(production=prod, fork_head=fork, **heads),
        picks=(
            picks
            if picks is not None
            else {PICK: dict(outcome=dict(e1c=e1c), e2=dict(verdict=e2))}
        ),
    )
    if tuple_name is not None:
        summary["tuple"] = tuple_name
    (run / "summary.json").write_text(json.dumps(summary))
    if manifest_tuple is not None:
        manifest["tuple"] = manifest_tuple
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run / "summary.json"


def on_run(rig, *extra):
    return rig.run("--kv-arm", "--predictor", "on", "--entry", PICK, *extra)


def x_rig(rig):
    rig.registry(arms.arm_text("X"))
    worker_for(rig, arms.arm_text("X"), mtp_on=True)
    return rig


def test_predictor_on_with_a_delta_is_refused_without_certified_by(real_argv):
    rig = x_rig(real_argv)
    got = on_run(rig)
    drv.refused(rig, got)
    assert "uncertified" in got.text()


def test_predictor_on_with_a_matching_record_is_a_certifying_run(real_argv):
    rig = x_rig(real_argv)
    path = record(rig)
    got = on_run(rig, "--certified-by", path)
    assert got.completed(PICK), got.text()
    state = got.state(PICK)
    assert state["arm"] == "served_on" and "overlay_sha256" not in state
    assert rig.specs[-1]["argv"] == REAL_ARGV["X"]
    beside = Path(path).parent / "manifest.json"
    assert got.report["certified_by"] == dict(  # both files of the record, by sha256
        path=str(Path(path).resolve()),
        sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        manifest_path=str(beside.resolve()),
        manifest_sha256=hashlib.sha256(beside.read_bytes()).hexdigest(),
    )
    assert got.report["production"] == rig.P0  # the tuple's production, resolved
    assert "engagement_only" not in got.report and "certifies" not in got.report
    assert got.report["kv_arm"] == dict(
        entry=PICK, delta=json_delta("m72_exact"), tq4_attention="m72_exact"
    )
    assert rig.specs[-1]["expect"]["drafter"] is True


NOT_TUPLE = "not a P369 campaign-tuple run"
NOT_PASS = "E1c / E2 not both PASS"
C199 = "MTP-ON only after its P369 full-run PASS (C199)"


@pytest.mark.parametrize(
    "bad, why",
    (
        (dict(tuple_name=None), NOT_TUPLE),
        (dict(tuple_name="m76-v"), NOT_TUPLE),
        (dict(manifest_tuple="other"), NOT_TUPLE),
        (dict(e1c="FAIL"), NOT_PASS),
        (dict(e1c="INCOMPLETE"), NOT_PASS),
        (dict(e2="FAIL"), NOT_PASS),
        (dict(e2="INCOMPLETE"), NOT_PASS),
        (dict(status="NOT_PASS"), C199),
        (dict(picks={}), NOT_PASS),
    ),
    ids=(
        "no-tuple",
        "unknown-tuple",
        "manifest-tuple",
        "e1c-fail",
        "e1c-incomplete",
        "e2-fail",
        "e2-incomplete",
        "not-pass",
        "no-pick",
    ),
)
def test_a_record_failing_a_is_refused(real_argv, bad, why):
    rig = x_rig(real_argv)
    got = on_run(rig, "--certified-by", record(rig, **bad))
    drv.refused(rig, got)
    assert why in got.text()
    assert on_run(rig, "--certified-by", record(rig)).completed(PICK)  # control


@pytest.mark.parametrize(
    "bad, why",
    (
        (dict(stack_head="t" * 40), "stack_head"),
        (dict(fork_head="e" * 40), "fork_head"),
    ),
    ids=("another-stack-commit", "another-fork-head"),
)
def test_a_record_failing_b_is_refused(real_argv, bad, why):
    rig = x_rig(real_argv)
    got = on_run(rig, "--certified-by", record(rig, **bad))
    drv.refused(rig, got)
    assert f"{why} [" in got.text()  # certification_problems names the key
    assert on_run(rig, "--certified-by", record(rig)).completed(PICK)  # control


def test_a_record_whose_stack_head_differs_only_in_the_manifest_is_refused(real_argv):
    rig = x_rig(real_argv)
    path = record(rig)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    manifest["stack_head"] = "t" * 40
    (path.parent / "manifest.json").write_text(json.dumps(manifest))
    got = on_run(rig, "--certified-by", path)
    drv.refused(rig, got)
    assert "stack_head" in got.text()


@pytest.mark.parametrize("other", ("V", "XO", "N16"))
def test_a_record_failing_c_is_refused(real_argv, other):
    rig = x_rig(real_argv)
    got = on_run(rig, "--certified-by", record(rig, argv_on=REAL_ARGV[other]))
    drv.refused(rig, got)
    assert qualify.CERTIFIED_DIFFERS in got.text()


def test_a_record_without_an_argv_is_refused(real_argv):
    rig = x_rig(real_argv)
    path = record(rig)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    del manifest["picks"]
    (path.parent / "manifest.json").write_text(json.dumps(manifest))
    drv.refused(rig, on_run(rig, "--certified-by", path))


def test_v_is_never_certified_by_xs_record(real_argv):
    rig = real_argv
    rig.registry(arms.arm_text("V"))
    worker_for(rig, arms.arm_text("V"), mtp_on=True)
    got = on_run(rig, "--certified-by", record(rig))
    drv.refused(rig, got)
    assert "m72_v1" in got.text()


def test_a_record_outside_the_campaign_root_is_refused(real_argv, tmp_path):
    rig = x_rig(real_argv)
    path = record(rig)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    for name in ("summary.json", "manifest.json"):
        (elsewhere / name).write_bytes((path.parent / name).read_bytes())
    drv.refused(rig, on_run(rig, "--certified-by", elsewhere / "summary.json"))


def test_a_missing_or_malformed_record_is_refused(real_argv, tmp_path):
    rig = x_rig(real_argv)
    drv.refused(rig, on_run(rig, "--certified-by", tmp_path / "nope.json"))
    bad = rig.workdir / "p369" / "phase2" / "bad"
    bad.mkdir(parents=True)
    (bad / "summary.json").write_text("[1, 2")
    drv.refused(rig, on_run(rig, "--certified-by", bad / "summary.json"))


@pytest.mark.parametrize(
    "argv",
    (
        ("--kv-arm", "--predictor", "off"),
        ("--kv-arm", "--predictor", "on", "--engagement-only"),
        ("--predictor", "on"),
    ),
    ids=("predictor-off", "engagement-only", "no-kv-arm"),
)
def test_certified_by_outside_a_certifying_kv_arm_run_is_refused(real_argv, argv):
    rig = x_rig(real_argv)
    drv.refused(rig, rig.run(*argv, "--entry", PICK, "--certified-by", record(rig)))


def test_certified_by_with_a_p369_summary_is_refused(real_argv):
    rig = x_rig(real_argv)
    path = record(rig)
    drv.refused(rig, on_run(rig, "--certified-by", path, "--p369-summary", path))


def test_certified_by_on_an_empty_delta_is_refused(real_argv):
    """The committed N16 entry is today's --predictor on (P369 summary binding); a tuple record never stands in for it."""
    rig = real_argv
    rig.registry(FIXTURE)
    worker_for(rig, FIXTURE, mtp_on=True)
    drv.refused(
        rig, on_run(rig, "--certified-by", record(rig, argv_on=REAL_ARGV["N16"]))
    )


REPLACED = "its binding is not the binding of the manifest.json beside it"


@pytest.mark.parametrize(
    "key, value",
    (
        ("harness_tree", "x" * 40),
        ("env_digest", "f" * 64),
        ("versions", dict(mlx="0.32.3")),
        ("off_stack_head", "p" * 40),
    ),
)
def test_a_pass_summary_paired_with_a_replaced_manifest_is_refused(
    real_argv, key, value
):
    """Sol P3: the summary's binding is that of ITS manifest; a manifest.json replaced beside it (same argv, same heads)
    no longer binds it."""
    rig = x_rig(real_argv)
    path = record(rig)
    beside = path.parent / "manifest.json"
    manifest = json.loads(beside.read_text())
    manifest[key] = value
    beside.write_text(json.dumps(manifest))
    got = on_run(rig, "--certified-by", path)
    drv.refused(rig, got)
    assert REPLACED in got.text()
    assert on_run(rig, "--certified-by", record(rig)).completed(PICK)  # control


def test_a_replaced_pick_entry_in_the_manifest_is_refused(real_argv):
    rig = x_rig(real_argv)
    path = record(rig)
    beside = path.parent / "manifest.json"
    manifest = json.loads(beside.read_text())
    manifest["picks"][PICK]["registry_off_sha256"] = "3" * 64
    beside.write_text(json.dumps(manifest))
    got = on_run(rig, "--certified-by", path)
    drv.refused(rig, got)
    assert REPLACED in got.text()


@pytest.mark.parametrize("key", ("off_stack_head", "production", "tuple"))
def test_the_summary_manifest_block_must_agree_with_the_manifest(real_argv, key):
    rig = x_rig(real_argv)
    path = record(rig)
    summary = json.loads(path.read_text())
    summary["manifest"][key] = "z" * 8
    path.write_text(json.dumps(summary))
    got = on_run(rig, "--certified-by", path)
    drv.refused(rig, got)
    assert f"the summary's manifest ['{key}'] differs" in got.text()


def test_a_record_on_another_production_than_the_tuples_is_refused(real_argv):
    """The record's production must be TUPLES[tuple]["production"], even when this run attests that other production."""
    rig = x_rig(real_argv)
    other = (
        orchestrate.PRODUCTION
    )  # the rig's P: also a strict ancestor of the reviewed head
    path = record(rig, production=other)
    got = on_run(rig, "--certified-by", path, "--production-ref", other)
    drv.refused(rig, got)
    assert f"production {other!r} is not tuple m76-x's" in got.text()


# ================================================================== the production pin (Sol P1 / GLM F3), unmocked diff
@pytest.fixture
def diffed(real_argv, monkeypatch):
    """The rig with the preflight's production-diff check REAL (``orchestrate.production_diff_problem`` on the rig's fork
    against the ref the run resolved); the rest of the host preflight stays mocked. Returns ``(rig, refs)``.
    """
    refs = []

    def preflight(args, out):
        refs.append(args.production_ref)
        diff = orchestrate.production_diff_problem(args.fork_root, args.production_ref)
        return dict(problems=[diff] if diff else [])

    monkeypatch.setattr(orchestrate, "preflight", preflight)
    return real_argv, refs


@pytest.mark.parametrize("arm", ("X", "V", "XO"))
def test_a_tq4_attention_arm_defaults_to_the_tuples_production(diffed, arm):
    rig, refs = diffed
    rig.registry(arms.arm_text(arm))
    worker_for(rig, arms.arm_text(arm), mtp_on=False)
    got = drv.kv_run(rig, PICK)
    assert got.completed(PICK), got.text()
    assert (
        refs == [rig.P0] and got.report["production"] == rig.P0
    )  # M77's, not PRODUCTION's


def test_the_production_diff_runs_against_the_tuples_production(diffed):
    """The real diff names the M77 sha: production code changed relative to it refuses."""
    rig, refs = diffed
    rig.registry(arms.arm_text("X"))
    worker_for(rig, arms.arm_text("X"), mtp_on=False)
    (rig.fork / "mlx_vlm").mkdir()
    (rig.fork / "mlx_vlm" / "changed.py").write_text("x = 1\n")
    p542._git(rig.fork, "add", "mlx_vlm/changed.py")
    got = drv.kv_run(rig, PICK)
    drv.refused(rig, got)
    assert refs == [rig.P0]
    assert f"production code differs from {rig.P0}" in got.text()


def test_a_certifying_run_defaults_to_the_tuples_production(diffed):
    rig, refs = diffed
    x_rig(rig)
    got = on_run(rig, "--certified-by", record(rig))
    assert got.completed(PICK), got.text()
    assert refs == [rig.P0] and got.report["production"] == rig.P0


def test_an_explicit_production_ref_wins_on_a_tq4_attention_arm(diffed):
    rig, refs = diffed
    rig.registry(arms.arm_text("X"))
    worker_for(rig, arms.arm_text("X"), mtp_on=False)
    got = drv.kv_run(rig, PICK, "--production-ref", rig.P[:8])
    assert got.completed(PICK), got.text()
    assert refs == [rig.P]


@pytest.mark.parametrize(
    "text, pick, arm",
    (
        (lambda: kv.arm_text("TQ4", PICK), PICK, "TQ4"),
        (lambda: kv.arm_text("AF8", kv.UNIFORM), kv.UNIFORM, "AF8"),
    ),
    ids=("tq4-without-mode", "af8-uniform"),
)
def test_every_other_arm_keeps_todays_production(diffed, text, pick, arm):
    rig, refs = diffed
    rig.registry(text())
    if pick == PICK:
        worker_for(rig, text(), mtp_on=False)
    else:
        drv.serve(rig, pick, arm)
    got = drv.kv_run(rig, pick)
    assert got.completed(pick), got.text()
    assert refs == [rig.P]  # orchestrate.PRODUCTION (the rig's P), unchanged


def test_the_default_production_rule():
    o = orchestrate
    assert qualify.default_production(None, tq4=False, orchestrate=o) == o.PRODUCTION
    assert qualify.default_production(None, tq4=True, orchestrate=o) == o.M77_PRODUCTION
    default = qualify.DefaultRef(o.PRODUCTION)
    assert (
        default == o.PRODUCTION
    )  # the parser default still reads as PRODUCTION (test_p452)
    assert (
        qualify.default_production(default, tq4=True, orchestrate=o) == o.M77_PRODUCTION
    )
    assert qualify.default_production(default, tq4=False, orchestrate=o) == o.PRODUCTION
    assert qualify.default_production("abc", tq4=True, orchestrate=o) == "abc"
    assert (
        qualify.default_production(o.PRODUCTION, tq4=True, orchestrate=o)
        == o.PRODUCTION
    )
    assert o.M77_PRODUCTION == o.TUPLES["m76-x"]["production"]
    args = qualify.build_parser().parse_args(
        ["--stack-root", "s", "--fork-root", "f", "--out", "o", "--reviewed-head", "r"]
    )
    assert isinstance(args.production_ref, qualify.DefaultRef)


# ================================================================== the KV-arm refusal names the delta keys (GLM F1)
def test_the_kv_arm_refusal_names_the_delta_keys(real_argv):
    rig = real_argv
    text = arms.arm_text("X").replace(
        "prefill_step_size: 512", "prefill_step_size: 1024", 1
    )
    assert text != arms.arm_text("X")
    rig.registry(text)
    got = drv.kv_run(rig, PICK)
    drv.refused(rig, got)
    assert str(list(registry_states.KV_ARM_DELTA_KEYS)) in got.text()
    assert "tq4_attention" in got.text()


# ================================================================== fingerprints
REC = dict(
    versions=dict(mlx="0.32.4"),
    generation_tokens=5,
    prompt=dict(sized=dict(sha256="p" * 64)),
)


def fp(text, *, mtp_on=True):
    entry = dict(entry_of(text))
    if not mtp_on:
        entry = {k: v for k, v in entry.items() if k not in qualify.MTP_KEYS}
    return qualify.fingerprint_of(REC, qualify.expected_config(entry))


def test_fingerprints_differ_between_x_and_v_only_in_the_mode():
    x, v = fp(arms.arm_text("X")), fp(arms.arm_text("V"))
    assert x != v
    assert {k for k in x if x[k] != v.get(k)} == {"tq4_attention"}
    assert x["tq4_attention"] == "m72_exact" and v["tq4_attention"] == "m72_v1"


def test_the_mode_field_is_emitted_only_with_a_mode():
    tq4 = fp(kv.arm_text("TQ4", PICK))
    x = fp(arms.arm_text("X"))
    assert "tq4_attention" not in tq4
    assert {k: v for k, v in x.items() if k != "tq4_attention"} == tq4
    assert list(x)[:-1] == list(qualify.FINGERPRINT) and list(x)[-1] == "tq4_attention"
    assert fp(arms.arm_text("XO"), mtp_on=False)["tq4_attention"] == "m72_exact"


def test_the_reference_comparison_counts_the_mode_only_where_one_side_has_it():
    x, tq4 = fp(arms.arm_text("X")), fp(kv.arm_text("TQ4", PICK))
    ref = lambda f: dict(peak_gb=40.0, source="s", fingerprint=dict(f))  # noqa: E731
    assert qualify.reference_comparison(ref(x), 40.0, x)["differs"] == []
    assert qualify.reference_comparison(ref(tq4), 40.0, tq4)["differs"] == []
    assert qualify.reference_comparison(ref(tq4), 40.0, x)["differs"] == [
        "tq4_attention"
    ]
    assert qualify.reference_comparison(ref(x), 40.0, tq4)["differs"] == [
        "tq4_attention"
    ]
    v = fp(arms.arm_text("V"))
    assert qualify.reference_comparison(ref(x), 40.0, v)["differs"] == ["tq4_attention"]


def test_the_serving_table_entry():
    assert qualify.SERVING["tq4_attention"] == ("", "--tq4-attention", "nonempty")
    # absent / empty: no flag (every existing entry's expected flags unchanged)
    for entry in registry_states.settings(FIXTURE).values():
        assert "--tq4-attention" not in qualify.expected_config(entry)["flags"]
