"""P658 (a): ``qualify --kv-arm --predictor on --engagement-only`` and P656 item 6 (fork-only, dev; CPU only).

Implementer's own tests (the acceptance tests are written separately). The driver tests reuse the P475a rig: the real gate,
``classify_kv_arm``, ``command_problems``, ``request_problems`` and report; the host and the worker boundary mocked.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import (  # noqa: E402
    kvmanifest,
    orchestrate,
    qualify,
    registry_states,
    worker,
)

from mlx_vlm.tests import test_p475a_kv_arm as kv  # noqa: E402
from mlx_vlm.tests.test_p475a_qualify_driver import (  # noqa: E402,F401  (rig is the pytest fixture)
    flags_of,
    mlx_serve_argv,
    refused,
    rig,
    serve,
)

ENGAGE = ("--kv-arm", "--predictor", "on", "--engagement-only")


def _leaf(path, kind="native", cls="KVCache", **extra):
    return dict(path=path, kind=kind, cls=cls, batch=False, **extra)


DRAFTER_TEMPLATE = [_leaf("0")]
GOOD = dict(observed=[_leaf("0", cls="PreallocKVCache")], template=DRAFTER_TEMPLATE)
QUANTIZED = dict(
    observed=[_leaf("0", kind="affine", cls="QuantizedKVCache", bits=8, group_size=64)],
    template=DRAFTER_TEMPLATE,
)
BATCHED = dict(
    observed=[dict(_leaf("0", cls="BatchKVCache"), batch=True)],
    template=DRAFTER_TEMPLATE,
)
MISMATCH = dict(observed=[_leaf("0"), _leaf("1")], template=DRAFTER_TEMPLATE)


def _drafter(problems):
    return [p for p in problems if "drafter cache" in p]


def engage(rig, pick, *extra):
    return rig.run(*ENGAGE, "--entry", pick, *extra)


def _lock_spy(rig, monkeypatch):
    calls = []
    real = orchestrate.acquire_lock

    def acquire(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(orchestrate, "acquire_lock", acquire)
    return calls


# ================================================================== acceptance 1: CLI refusals
NEEDS = "--engagement-only needs --kv-arm --predictor on"


@pytest.mark.parametrize(
    "argv",
    (
        ("--predictor", "on", "--engagement-only"),
        ("--kv-arm", "--predictor", "off", "--engagement-only"),
        ("--kv-arm", "--engagement-only"),
        ("--engagement-only",),
    ),
    ids=("no-kv-arm", "predictor-off", "predictor-unset", "alone"),
)
def test_engagement_only_needs_kv_arm_and_predictor_on_before_the_lock(
    rig, monkeypatch, argv
):
    rig.registry(kv.arm_text("TQ8", kv.UNIFORM))
    serve(rig, kv.UNIFORM, "TQ8", mtp_on=True, drafter_manifest=GOOD)
    assert engage(rig, kv.UNIFORM).completed(kv.UNIFORM)  # control
    locks = _lock_spy(rig, monkeypatch)
    got = rig.run(*argv, "--entry", kv.UNIFORM)
    refused(rig, got)
    assert got.refused == f"refused: [{NEEDS!r}]"
    assert locks == [] and not got.out.exists()


def test_engagement_only_refuses_a_p369_summary_before_the_lock(rig, monkeypatch):
    rig.registry(kv.arm_text("TQ8", kv.UNIFORM))
    serve(rig, kv.UNIFORM, "TQ8", mtp_on=True, drafter_manifest=GOOD)
    summary = rig.summary(kv.UNIFORM, argv_on=["--x"])
    locks = _lock_spy(rig, monkeypatch)
    got = engage(rig, kv.UNIFORM, "--p369-summary", summary)
    refused(rig, got)
    assert got.refused == "refused: ['--engagement-only takes no --p369-summary']"
    assert locks == []


@pytest.mark.parametrize("pick", kv.PICKS)
def test_engagement_only_refuses_an_entry_the_registry_serves_mtp_off(rig, pick):
    rig.registry(kv.arm_text("TQ4", pick, registry_states.apply(kv.FIXTURE, {pick})))
    serve(rig, pick, "TQ4", drafter_manifest=GOOD)
    got = engage(rig, pick)
    refused(rig, got)
    assert "MTP-OFF (no overlay is built)" in got.text()


def test_the_flag_is_off_by_default():
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
    assert qualify.build_parser().parse_args(base).engagement_only is False
    on = qualify.build_parser().parse_args(base + ["--engagement-only"])
    assert on.engagement_only is True


# ================================================================== acceptance 2: every arm, no summary, the served argv
ARM_CELLS = [(kv.UNIFORM, a) for a in kv.ARMS] + [(kv.PICK_A, "AF8")]
ARM_IDS = [f"{'pickA' if p == kv.PICK_A else 'uniform'}-{a}" for p, a in ARM_CELLS]


@pytest.mark.parametrize("pick, arm", ARM_CELLS, ids=ARM_IDS)
def test_gate_and_entry_problems_pass_with_no_summary(pick, arm):
    text = kv.arm_text(arm, pick)
    gate = qualify.kv_arm_gate([pick], "on", text, DEV.parent, engagement_only=True)
    assert gate["entry"] == pick and pick not in gate["off"]
    assert {k: tuple(v) for k, v in gate["delta"].items()} == kv.arm_delta(arm, pick)
    common = dict(
        registry_text=text, predictor="on", production="p" * 40, resolve=lambda r: r
    )
    assert qualify.entry_problems([pick], engagement_only=True, **common) == []
    gated = qualify.entry_problems([pick], **common)  # control: C199 without the flag
    assert any("C199" in p for p in gated)


NONEMPTY = [(p, a) for p, a in ARM_CELLS if kv.arm_delta(a, p)]


@pytest.mark.parametrize("pick, arm", NONEMPTY)
def test_without_the_flag_a_nonempty_delta_still_refuses_as_uncertified(pick, arm):
    with pytest.raises(SystemExit, match="uncertified"):
        qualify.kv_arm_gate([pick], "on", kv.arm_text(arm, pick), DEV.parent)


@pytest.mark.parametrize("pick, arm", ARM_CELLS, ids=ARM_IDS)
def test_each_arm_completes_mtp_on_as_committed(rig, monkeypatch, pick, arm):
    ref = dict(peak_gb=40.0, source="s", source_path="p", fingerprint={})
    monkeypatch.setattr(qualify, "load_references", lambda path, **k: {pick: ref})
    committed = rig.registry(kv.arm_text(arm, pick))
    serve(rig, pick, arm, mtp_on=True, drafter_manifest=GOOD)
    got = engage(rig, pick)
    assert got.completed(pick), got.text()
    state = got.state(pick)
    assert state["arm"] == "served_on" and "overlay_sha256" not in state
    assert [s["path"] for s in rig.served] == [committed]
    assert not list(got.out.glob("overlay-*")) and not list(got.out.glob("mtp-off-*"))
    assert got.report["engagement_only"] is True and got.report["certifies"] is False
    assert got.report["kv_arm"] == dict(
        entry=pick, delta=kv.as_json(kv.arm_delta(arm, pick))
    )
    assert state["reference"] is None
    assert state["drafter_manifest"] == GOOD
    (job,) = rig.specs
    assert job["drafter_manifest"] is True
    flags = flags_of(job["argv"])
    assert flags["--draft-kind"] == "mtp" and flags["--draft-model"].endswith(
        "-mtp-drafter"
    )
    bits, scheme = kv.ARMS[arm]
    assert ("--kv-bits" in flags) == bool(bits)
    if bits:
        assert flags["--kv-bits"] == str(bits)
    assert flags["--kv-quant-scheme"] == (
        scheme or kv.FIXTURE_KV[pick]["kv_quant_scheme"]
    )
    assert (
        job["argv"]
        == mlx_serve_argv(
            registry_states.settings(kv.arm_text(arm, pick))[pick], rig.stack
        )[1:]
    )


def test_a_tampered_served_argv_fails_the_entry(rig, monkeypatch):
    rig.registry(kv.arm_text("TQ8", kv.UNIFORM))
    serve(rig, kv.UNIFORM, "TQ8", mtp_on=True, drafter_manifest=GOOD)
    real = qualify.full_command

    def tampered(stack, path, name):
        cmd, mod = real(stack, path, name)
        return cmd + ["--extra-flag", "x"], mod

    monkeypatch.setattr(qualify, "full_command", tampered)
    got = engage(rig, kv.UNIFORM)
    assert got.code == 1 and rig.jobs == [], got.text()
    state = got.state(kv.UNIFORM)
    assert state["status"] == "FAILED"
    assert any("--extra-flag" in p for p in state["problems"])


def test_the_served_argv_fails_command_problems_when_tampered():
    entry = registry_states.settings(kv.arm_text("TQ8", kv.UNIFORM))[kv.UNIFORM]
    cmd = mlx_serve_argv(entry, "/s")
    expected = qualify.expected_config(entry)
    assert qualify.command_problems(expected, cmd, "/s/.venv/bin") == []
    i = cmd.index("--kv-bits")
    bad = cmd[: i + 1] + ["4"] + cmd[i + 2 :]
    assert qualify.command_problems(expected, bad, "/s/.venv/bin") != []


# ================================================================== acceptance 3: the drafter judgment
def test_drafter_manifest_problems_pure():
    assert qualify.drafter_manifest_problems(dict(drafter_manifest=GOOD)) == []
    for bad in (QUANTIZED, BATCHED, MISMATCH):
        assert _drafter(qualify.drafter_manifest_problems(dict(drafter_manifest=bad)))
    for rec in (
        {},
        dict(drafter_manifest=None),
        dict(drafter_manifest=dict(observed=[], template=DRAFTER_TEMPLATE)),
        dict(drafter_manifest=dict(observed=GOOD["observed"], template=[])),
        dict(drafter_manifest=dict(observed="x", template=DRAFTER_TEMPLATE)),
        dict(drafter_manifest=dict(template=DRAFTER_TEMPLATE)),
    ):
        assert qualify.drafter_manifest_problems(rec) == ["no drafter cache manifest"]


@pytest.mark.parametrize(
    "label, manifest",
    (
        ("quantized", QUANTIZED),
        ("batched", BATCHED),
        ("mismatch", MISMATCH),
        ("empty", dict(observed=[], template=DRAFTER_TEMPLATE)),
        ("absent", None),
    ),
)
def test_a_bad_drafter_manifest_fails_the_entry_under_the_flag(rig, label, manifest):
    rig.registry(kv.arm_text("TQ8", kv.UNIFORM))
    changes = {} if manifest is None else dict(drafter_manifest=manifest)
    serve(rig, kv.UNIFORM, "TQ8", mtp_on=True, **changes)
    got = engage(rig, kv.UNIFORM)
    state = got.state(kv.UNIFORM)
    assert got.code == 1 and state["status"] == "FAILED", got.text()
    assert state["problems"] and all("drafter" in p for p in state["problems"])
    assert ("drafter_manifest" in state) == (manifest is not None)


def test_without_the_flag_the_drafter_manifest_is_not_judged(rig):
    # today's --kv-arm --predictor on with the empty delta and its summary: a bad drafter manifest is not judged
    pick, arm = kv.UNIFORM, "TQ4"
    rig.registry(kv.arm_text(arm, pick))
    serve(rig, pick, arm, mtp_on=True, drafter_manifest=QUANTIZED)
    argv_on = mlx_serve_argv(registry_states.settings(kv.FIXTURE)[pick], rig.stack)[1:]
    got = rig.run(
        "--kv-arm",
        "--predictor",
        "on",
        "--entry",
        pick,
        "--p369-summary",
        rig.summary(pick, argv_on=argv_on),
    )
    assert got.completed(pick), got.text()
    assert "engagement_only" not in got.report and "certifies" not in got.report
    assert "drafter_manifest" not in got.state(pick)
    assert "drafter_manifest" not in rig.specs[0]


def test_without_the_flag_a_kv_arm_off_report_has_no_engagement_fields(rig):
    rig.registry(kv.arm_text("AF8", kv.UNIFORM))
    serve(rig, kv.UNIFORM, "AF8")
    got = rig.run("--kv-arm", "--predictor", "off", "--entry", kv.UNIFORM)
    assert got.completed(kv.UNIFORM), got.text()
    assert "engagement_only" not in got.report and "certifies" not in got.report
    assert "drafter_manifest" not in rig.specs[0]


# ================================================================== acceptance 4: the worker records the drafter manifest
class KVCache:  # kvmanifest classifies by class name
    pass


class QuantizedKVCache:
    bits, group_size = 8, 64


class _LM:
    args = SimpleNamespace(num_attention_heads=24)

    def __call__(self, *a, **k):
        return None

    def make_cache(self):
        return [KVCache(), KVCache()]


class _Model:
    def __init__(self):
        self.language_model = _LM()


class _Drafter:
    def __init__(self, events, cache):
        self.events, self.cache = events, cache

    @property
    def _cache(self):
        self.events.append("drafter._cache")
        return self.cache

    def make_cache(self):
        self.events.append("drafter.make_cache")
        return [KVCache()]


def _run_qualify_request(monkeypatch, *, drafter_cache, **kw):
    events = []
    monkeypatch.setattr(worker, "prompt_tokens", lambda *a, **k: 10)
    monkeypatch.setattr(worker, "size_prompt", lambda count, target, **k: ("t", target))
    monkeypatch.setattr(worker, "prompt_ids", lambda *a, **k: [1, 2, 3])
    monkeypatch.setattr(worker, "check_server_main", lambda: True)

    def post(client, payload, chat_id, capture, timeout_s):
        events.append("post")
        return dict(
            chat_id="c1",
            final=dict(
                cached_tokens=0,
                finish_reason="stop",
                generation_tokens=3,
                prompt_tps=1.0,
                generation_tps=2.0,
            ),
            prompt=dict(tokens=3, sha256="x"),
            response={},
            wall_s=0.1,
        )

    monkeypatch.setattr(worker, "post", post)
    monkeypatch.setattr(worker, "release_session", lambda cid: events.append("release"))
    rg = SimpleNamespace(model=_Model(), draft_model=_Drafter(events, drafter_cache))
    spec = dict(
        model="m",
        target=3,
        prefix="p",
        suffix="s",
        seed=7,
        timeout_s=1,
        generation_defaults={},
    )
    rec = worker.qualify_request(
        object(), SimpleNamespace(backend="cached_session"), rg, spec, {}, **kw
    )
    return rec, events


def test_the_worker_records_the_drafter_manifest_after_the_response_before_release(
    monkeypatch,
):
    rec, events = _run_qualify_request(
        monkeypatch, drafter_cache=[KVCache()], drafter_manifest=True
    )
    assert rec["drafter_manifest"] == dict(
        observed=kvmanifest.manifest([KVCache()]),
        template=kvmanifest.manifest([KVCache()]),
    )
    assert (
        events.index("post") < events.index("drafter._cache") < events.index("release")
    )
    assert events.index("drafter.make_cache") < events.index("release")
    assert qualify.drafter_manifest_problems(rec) == []


def test_the_worker_records_a_quantized_drafter_leaf_as_is(monkeypatch):
    rec, _ = _run_qualify_request(
        monkeypatch, drafter_cache=[QuantizedKVCache()], drafter_manifest=True
    )
    assert rec["drafter_manifest"]["observed"][0]["kind"] == "affine"
    assert _drafter(qualify.drafter_manifest_problems(rec))


def test_a_none_drafter_cache_is_recorded_empty(monkeypatch):
    rec, _ = _run_qualify_request(
        monkeypatch, drafter_cache=None, drafter_manifest=True
    )
    assert rec["drafter_manifest"]["observed"] == []
    assert qualify.drafter_manifest_problems(rec) == ["no drafter cache manifest"]


def test_without_the_field_the_record_keys_are_todays(monkeypatch):
    rec, events = _run_qualify_request(monkeypatch, drafter_cache=[KVCache()])
    assert "drafter_manifest" not in rec
    assert "drafter._cache" not in events and "drafter.make_cache" not in events
    assert set(rec) == {
        "prompt",
        "chunks",
        "prompt_step_tokens",
        "boundary_seen",
        "peaks",
        "lazy_calls",
        "shrink_calls",
        "memory",
        "template",
        "delegates_to_cli_main",
        "cached_tokens",
        "backend",
        "finish_reason",
        "generation_tokens",
        "rates",
        "sdpa_forced",
        "server_defaults",
        "wall_s",
    }


def _run_job_qualify(monkeypatch, job_extra):
    calls = []

    def fake_request(client, capture, rg, spec, memory, **kw):
        calls.append(kw)
        return dict(x=1)

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class NullCapture:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import mlx_vlm.server as server

    rg = SimpleNamespace(model=_Model(), draft_model=None)
    monkeypatch.setattr(server.runtime, "response_generator", rg, raising=False)
    monkeypatch.setattr(worker, "qualify_request", fake_request)
    monkeypatch.setattr(worker, "record_memory_limits", lambda: ({}, lambda: None))
    monkeypatch.setattr(worker, "boot", lambda argv: Client())
    monkeypatch.setattr(worker, "served_state", lambda: {})
    monkeypatch.setattr(worker, "assert_served", lambda s, e: None)
    monkeypatch.setattr(worker, "check_sources", lambda r, d: None)
    monkeypatch.setattr(worker, "check_snapshots", lambda a, r: {})
    monkeypatch.setattr(worker, "write_progress", lambda *a, **k: None)
    monkeypatch.setattr(worker, "Capture", NullCapture)
    from p369_phase2 import telemetry

    monkeypatch.setattr(telemetry, "manifest_from_model", lambda lm: {})
    job = dict(
        mode="qualify",
        argv=["--model", "m"],
        expect={},
        source_root="r",
        dev_dir="d",
        requests=[dict(name="max-context", timeout_s=1)],
        **job_extra,
    )
    result = dict(requests=[])
    worker.run_job(job, Path("/nonexistent"), result)
    return calls


def test_run_job_forwards_the_job_field_only_when_true(monkeypatch):
    assert _run_job_qualify(monkeypatch, dict(drafter_manifest=True)) == [
        dict(drafter_manifest=True)
    ]
    assert _run_job_qualify(monkeypatch, {}) == [{}]


# ================================================================== acceptance 6: the P475 orchestrator is a launcher
def test_launcher_modules():
    assert orchestrate.LAUNCHER_MODULES == (
        "bench.p426_leg",
        "bench.p426_wave1",
        "bench.p475_kv",
    )


def test_a_p475_kv_ancestor_is_exempt_a_non_ancestor_is_not():
    p475 = ["/venv/bin/python", "-m", "bench.p475_kv", "--arm", "TQ8"]
    other = ["/venv/bin/python", "-m", "bench.other"]
    procs = [(100, p475), (300, p475), (200, other)]
    problems = orchestrate.foreign_process_problems(procs, me=1, ancestors={100, 200})
    pids = sorted(int(p.split()[1]) for p in problems)
    assert pids == [200, 300]


CHILD = """
import argparse, json, os, sys, tempfile
sys.path.insert(0, {dev!r})
from p369_phase2 import orchestrate
args = argparse.Namespace(fork_root={fork!r}, production_ref="HEAD")
pre = orchestrate.preflight(args, tempfile.mkdtemp())
from p369_phase2 import qualify
print(json.dumps(dict(problems=[p for p in pre["problems"] if p.startswith("process ")], parent=os.getppid(),
                      interference=qualify.interference_problems(orchestrate))))
"""
LAUNCH = """
import subprocess, sys, time
if "--sibling" in sys.argv:
    time.sleep(30)
    sys.exit(0)
r = subprocess.run([sys.executable, "-c", {child!r}], capture_output=True, text=True, check=True)
print(r.stdout, end="")
"""


def _tree(tmp_path, module):
    """``python -m bench.<module>`` -> the preflight child, beside a non-ancestor ``python -m bench.p475_kv``."""
    pytest.importorskip("psutil", reason="the process census needs psutil")
    pkg = tmp_path / "bench"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for name in {module, "p475_kv"}:
        (pkg / f"{name}.py").write_text(
            LAUNCH.format(child=CHILD.format(dev=str(DEV), fork=str(DEV.parent)))
        )
    env = {"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    sibling = subprocess.Popen(
        [sys.executable, "-m", "bench.p475_kv", "--sibling"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.DEVNULL,
    )
    time.sleep(0.5)
    try:
        out = subprocess.run(
            [sys.executable, "-m", f"bench.{module}"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
    finally:
        sibling.kill()
        sibling.wait()
    return json.loads(out), sibling.pid


def test_real_topology_a_p475_kv_parent_is_exempt_a_sibling_is_not(tmp_path):
    doc, sibling = _tree(tmp_path, "p475_kv")
    for key in ("problems", "interference"):
        pids = [int(p.split()[1]) for p in doc[key]]
        assert doc["parent"] not in pids, (key, doc)
        assert sibling in pids, (key, doc)


def test_real_topology_another_bench_parent_still_refuses(tmp_path):
    doc, sibling = _tree(tmp_path, "other")
    for key in ("problems", "interference"):
        pids = [int(p.split()[1]) for p in doc[key]]
        assert doc["parent"] in pids and sibling in pids, (key, doc)
