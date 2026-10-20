"""P369 phase 2: the orchestrator's pure parts (the overlay, the worker env, the bounds, the E1c rule).

Fork-only file. The orchestrator runs under the STACK venv (mlx-serve, psutil); its pure functions are
tested here without either.
"""

import json
import sys
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import orchestrate  # noqa: E402

FABLE, UNIFORM = list(orchestrate.PICKS)
MARK = "# P369 INTERIM (C177 2026-10-17): MTP OFF on the daily driver"

REGISTRY = f"""models:
  - name: {UNIFORM}
    type: vision
    kv_bits: 4
    # draft_kind: mtp  {MARK}
    # draft_model: caslca/{UNIFORM}-mtp-drafter  # UPLOADED 2026-08-26 (public)
    presentation:
      role: main

  - name: Other-Model
    type: vision
    # draft_kind: mtp  {MARK}
    # draft_model: caslca/other-drafter

  - name: {FABLE}
    type: vision
    kv_bits: 0
    # mtp_verify_scan: joint_v1  {MARK} (joint_v1 requires draft_kind mtp)
    generation_defaults:
      temperature: 0.5
    # draft_kind: mtp  {MARK}
    # draft_model: caslca/{FABLE}-mtp-drafter
    presentation:
      role: main
"""


@pytest.mark.parametrize(
    ("pick", "want"),
    [
        (
            FABLE,
            [
                "mtp_verify_scan: joint_v1",
                "draft_kind: mtp",
                f"draft_model: caslca/{FABLE}-mtp-drafter",
            ],
        ),
        (UNIFORM, ["draft_kind: mtp", f"draft_model: caslca/{UNIFORM}-mtp-drafter"]),
    ],
)
def test_overlay_restores_exactly_the_c177_lines_of_the_pick(pick, want):
    text, restored = orchestrate.overlay_registry(REGISTRY, pick)
    assert restored == want
    before, after = REGISTRY.split("\n"), text.split("\n")
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(before) == len(after) and len(changed) == len(want)
    assert all(b.strip() in want for _, b in changed)
    other = [b for b in after if "other-drafter" in b]
    assert other == [
        "    # draft_model: caslca/other-drafter"
    ]  # another pick untouched


def test_overlay_refuses_a_registry_without_the_marked_lines():
    restored_text, _ = orchestrate.overlay_registry(REGISTRY, FABLE)
    with pytest.raises(ValueError, match="expected 3 C177 lines"):
        orchestrate.overlay_registry(restored_text, FABLE)
    unmarked = REGISTRY.replace(MARK, "# some other note")
    with pytest.raises(ValueError, match="expected 2 C177 lines"):
        orchestrate.overlay_registry(unmarked, UNIFORM)
    with pytest.raises(ValueError, match="no registry block"):
        orchestrate.overlay_registry(REGISTRY, "Missing-Model")


@pytest.mark.requires_stack_registry
def test_overlay_on_the_live_stack_registry():
    # P387: opt-in (MLX_REQUIRE_STACK_REGISTRY=1), the same resolution as the golden live check (K1)
    import yaml

    sys.path.insert(0, str(DEV.parent / "mlx_vlm" / "tests"))
    from test_mtp_verify_scan import _fixture_verbatim, live_registry

    path, state = live_registry()
    text = path.read_text()
    want = {e["name"]: e for e in yaml.safe_load(_fixture_verbatim())["models"]}
    for pick in orchestrate.PICKS:
        if (
            state == "mtp_on"
        ):  # nothing to restore: the live registry IS the certified one
            with pytest.raises(ValueError, match="C177 lines"):
                orchestrate.overlay_registry(text, pick)
            continue
        new, _ = orchestrate.overlay_registry(text, pick)
        got = {e["name"]: e for e in yaml.safe_load(new)["models"]}
        assert got[pick] == want[pick], pick


def test_worker_env_drops_conditional_knobs_and_sets_the_session_max():
    base = {
        "PATH": "/bin",
        "HF_HUB_OFFLINE": "1",
        "APC_ENABLED": "1",
        "MLX_VLM_DRAFT_KIND": "suffix",
        "MLX_VLM_PRELOAD_MODEL": "x",
        "MLX_VLM_MTP_VERIFY_SCAN": "joint_v1",
        "MLX_VLM_CACHE_SESSION_MAX": "2",
    }
    env = orchestrate.worker_env(base)
    assert env == {
        "PATH": "/bin",
        "HF_HUB_OFFLINE": "1",
        "MLX_VLM_CACHE_SESSION_MAX": "1",
    }


# ------------------------------------------------------------------ code round 1 H3 H9 H11 H14 + power
def test_worker_env_is_an_allowlist():
    base = {
        "PATH": "/bin",
        "HOME": "/home/u",
        "HF_TOKEN": "t",
        "HF_HUB_OFFLINE": "1",
        "LC_ALL": "C",
        "MLX_EPICACHE_BUDGET": "4096",
        "MIN_OUTPUT_TOKENS": "64",
        "MLX_VLM_SPEC_BATCH_COALESCE_MS": "5",
        "KV_BITS": "4",
        "TQ_FUSED_PREFILL": "1",
        "MLX_SDPA_BLOCKS": "1",
        "SOME_FUTURE_KNOB": "x",
        "PYTHONPATH": "/elsewhere",
    }
    env = orchestrate.worker_env(base)
    assert env == {
        "PATH": "/bin",
        "HOME": "/home/u",
        "HF_TOKEN": "t",
        "HF_HUB_OFFLINE": "1",
        "LC_ALL": "C",
        "MLX_VLM_CACHE_SESSION_MAX": "1",
    }
    assert orchestrate.dropped_env(base) == sorted(k for k in base if k not in env)


@pytest.mark.parametrize(
    ("argv", "refused"),
    [
        (["/v/bin/python", "-m", "mlx_vlm.generate", "--model", "m"], True),
        (["python3", "-I", "-m", "mlx_vlm.server"], True),
        (["/v/bin/python", "-m", "pytest", "-q"], True),
        (["/v/bin/python", "-m", "mlx_lm.server"], True),
        (["uv", "run", "mlx-serve", "start"], True),
        (["/v/bin/mlx-serve", "start"], True),
        (["opencode", "run", "x"], True),
        (["/v/bin/python", "/s/benchmark/bench/run_tg1_probe.py"], True),
        (["/v/bin/python", "-m", "bench.runner"], True),
        (["/v/bin/python", "-I", "/f/dev/p369_phase2/worker.py", "job.json"], True),
        (["zsh", "-c", "cd x && pytest -q && mlx-serve start"], False),
        (["/usr/bin/vim", "notes_mlx_vlm.txt"], False),
        (["/v/bin/python", "-c", "import mlx_vlm"], False),
        (["cursor-agent", "-p", "review mlx_vlm"], False),
        ([], False),
    ],
)
def test_refused_process_inspects_argv_structurally(argv, refused):
    assert bool(orchestrate.refused_process(argv)) is refused, argv


def test_tree_status_refuses_any_tracked_or_untracked_change():
    assert orchestrate.tree_status_problems("") == []
    assert orchestrate.tree_status_problems(" M dev/p369_phase2/worker.py\n")
    assert orchestrate.tree_status_problems("?? dev/p369_phase2/new.py\n")


PMSET_AC = """ Wattage = 140W
 Current = 4990mA
 Voltage = 28000mV
 AdapterID = 28697
 Adapter Name = 140W USB-C Power Adapter
 Hardware Version = 1.0
"""


def test_parse_power_reads_watts_volts_and_battery():
    batt = " -InternalBattery-0 (id=1)\t64%; charging; (no estimate) present: true\n"
    got = orchestrate.parse_power(PMSET_AC, batt)
    assert (got["watts"], got["volts"], got["battery"]) == (140, 28.0, 64)
    assert got["ac_ok"] and got["battery_ok"]
    low_volts = orchestrate.parse_power(PMSET_AC.replace("28000mV", "20000mV"), batt)
    assert not low_volts["ac_ok"]
    assert not orchestrate.parse_power("No adapter attached.\n", batt)["ac_ok"]
    assert not orchestrate.parse_power(PMSET_AC, batt.replace("64%", "19%"))[
        "battery_ok"
    ]


def _binding(**changes):
    pick = dict(
        overlay_sha256="o",
        argv_on=["--model", "m", "--draft-model", "d"],
        argv_off=["--model", "m"],
        params=dict(temperature=0.5),
        target=dict(revision="r", files={"a.safetensors": "1"}),
        drafter=dict(revision="r2", files={"b.safetensors": "2"}),
    )
    pick.update(changes)
    return dict(
        production=orchestrate.PRODUCTION,
        fork_head="h",
        harness_tree="t",
        reviewed_head="h",
        versions={"mlx": "0.32.3", "mlx-metal": "0.32.3"},
        env_digest=dict(names=["HOME", "PATH"], sha256="e"),
        stack_bench=dict(tree="b", bundles={"x.jsonl": "1"}),
        picks={FABLE: pick},
    )


# ------------------------------------------------------------------ code round 2 H9: reviewed head, tree, binding
def test_reviewed_head_rules():
    rh = orchestrate.reviewed_head_problems
    assert (
        rh("smoke", None, "h") == []
    )  # smoke iterations may run unreviewed (P394 refined)
    assert rh("smoke", "h", "h") == [] and rh("full", "h", "h") == []
    assert rh("smoke", "x", "h") and rh("full", "x", "h")
    assert rh("full", None, "h")  # the full run needs a reviewed head


def test_binding_extends_to_versions_env_bench_and_a_reviewed_smoke():
    manifest = _binding()
    smoke = dict(binding=orchestrate.binding(manifest))
    assert orchestrate.binding_problems(smoke, manifest) == []
    for key, value in [
        ("versions", {"mlx": "0.32.4", "mlx-metal": "0.32.3"}),
        ("env_digest", dict(names=["HOME", "PATH"], sha256="other")),
        ("stack_bench", dict(tree="b2", bundles={"x.jsonl": "1"})),
        ("harness_tree", "t2"),
    ]:
        assert orchestrate.binding_problems(smoke, dict(manifest, **{key: value})), key
    unreviewed = dict(binding=orchestrate.binding(dict(manifest, reviewed_head=None)))
    problems = orchestrate.binding_problems(unreviewed, manifest)
    assert any("unreviewed" in p for p in problems), problems


def test_env_digest_hashes_values_without_recording_them():
    a = orchestrate.env_digest({"HF_TOKEN": "s3cret", "PATH": "/b"})
    assert a["names"] == ["HF_TOKEN", "PATH"] and "s3cret" not in json.dumps(a)
    assert a != orchestrate.env_digest({"HF_TOKEN": "other", "PATH": "/b"})
    assert a == orchestrate.env_digest({"PATH": "/b", "HF_TOKEN": "s3cret"})


def test_hf_hub_cache_follows_huggingface_hub_precedence(tmp_path):
    from pathlib import Path

    home = Path("/h")
    assert orchestrate.hf_hub_cache(
        {"HF_HUB_CACHE": "/a", "HF_HOME": "/x"}, home
    ) == Path("/a")
    assert orchestrate.hf_hub_cache(
        {"HUGGINGFACE_HUB_CACHE": "/b", "HF_HOME": "/x"}, home
    ) == Path("/b")
    assert orchestrate.hf_hub_cache({"HF_HOME": "/x"}, home) == Path("/x/hub")
    assert orchestrate.hf_hub_cache({}, home) == Path("/h/.cache/huggingface/hub")


def test_tree_guard_refuses_a_moved_head_or_any_change(tmp_path):
    import subprocess

    def git(*args):
        return subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@example.invalid",
                *args,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    git("init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n")
    git("add", "a.py")
    git("commit", "-q", "-m", "one")
    head = git("rev-parse", "HEAD")
    orchestrate.tree_guard(tmp_path, head)
    (tmp_path / "a.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="not clean"):
        orchestrate.tree_guard(tmp_path, head)
    git("checkout", "-q", "a.py")
    (tmp_path / "new.py").write_text("")
    with pytest.raises(RuntimeError, match="not clean"):
        orchestrate.tree_guard(tmp_path, head)
    (tmp_path / "new.py").unlink()
    (tmp_path / "a.py").write_text("x = 3\n")
    git("commit", "-q", "-am", "two")
    with pytest.raises(RuntimeError, match="HEAD moved"):
        orchestrate.tree_guard(tmp_path, head)


def test_binding_mismatch_refuses_the_full_run():
    manifest = _binding()
    smoke = dict(binding=orchestrate.binding(manifest))
    assert orchestrate.binding_problems(smoke, manifest) == []
    for key, value in [
        ("argv_on", ["--model", "other"]),
        ("target", dict(revision="r", files={"a.safetensors": "9"})),
        ("overlay_sha256", "x"),
        ("params", dict(temperature=0.6)),
    ]:
        assert orchestrate.binding_problems(smoke, _binding(**{key: value})), key
    head = dict(_binding(), fork_head="other")
    assert orchestrate.binding_problems(smoke, head)
    picks = dict(_binding(), picks={})
    assert orchestrate.binding_problems(smoke, picks)
    assert orchestrate.binding_problems(
        {}, manifest
    )  # a smoke summary without a binding


@pytest.mark.parametrize(
    ("k", "n", "lo", "hi"),
    [(0, 20, 0.0, 0.1684), (20, 20, 0.8316, 1.0), (5, 20, 0.0866, 0.4910)],
)
def test_clopper_pearson_matches_known_values(k, n, lo, hi):
    got = orchestrate.clopper_pearson(k, n)
    assert got[0] == pytest.approx(lo, abs=1e-4) and got[1] == pytest.approx(
        hi, abs=1e-4
    )


def _entries(**changes):
    on = dict(
        verdict=dict(**{"pass": True}),
        draft_rounds=12,
        prompt_end=orchestrate.straddle_target(8192),
        straddle=True,
        response=dict(
            timings=dict(verify_blocks_straddle=3, verify_fallback_reasons={})
        ),
    )
    off = dict(rows=dict(**{"pass": True}), states=dict(**{"pass": True}))
    for key, value in changes.items():
        (off if key in ("rows", "states") else on)[key] = value
    return on, off


@pytest.mark.parametrize("policy", ["joint_v1", None])
def test_e1c_verdict_passes_a_conforming_request(policy):
    on, off = _entries()
    assert orchestrate.e1c_verdict(on, off, threshold=8192, verify_policy=policy)[
        "pass_"
    ]


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (dict(rows=dict(**{"pass": False})), "rows"),
        (dict(states=dict(**{"pass": False})), "states"),
        (dict(verdict=dict(**{"pass": False})), "kernels"),
        (dict(draft_rounds=0), "no MTP rounds"),
        (dict(prompt_end=8000), "prompt_end"),
    ],
)
def test_e1c_verdict_flags_each_failure(change, problem):
    on, off = _entries(**change)
    got = orchestrate.e1c_verdict(on, off, threshold=8192, verify_policy="joint_v1")
    assert not got["pass_"] and any(p.startswith(problem) for p in got["problems"])


def test_e1c_verdict_needs_the_policy_specific_straddle_evidence():
    on, off = _entries(response=dict(timings=dict(verify_blocks_straddle=0)))
    assert not orchestrate.e1c_verdict(
        on, off, threshold=8192, verify_policy="joint_v1"
    )["pass_"]
    on, off = _entries(straddle=False)
    assert not orchestrate.e1c_verdict(on, off, threshold=8192, verify_policy=None)[
        "pass_"
    ]
    on, off = _entries(
        response=dict(
            timings=dict(verify_blocks_straddle=2, verify_fallback_reasons={"x": 1})
        )
    )
    assert not orchestrate.e1c_verdict(
        on, off, threshold=8192, verify_policy="joint_v1"
    )["pass_"]


# ------------------------------------------------------------------ round 5 precedence (Q37)
def _v(pass_=True, c1=True, c2=True, evidence=True):
    return dict(
        pass_=pass_ and evidence,
        neutral=pass_,
        evidence_ok=evidence,
        c1=dict(identical=c1),
        c2=dict(identical=c2),
        problems=[],
    )


def test_pick_outcome_precedence():
    assert orchestrate.pick_outcome({"a": _v(), "b": _v()}) == dict(
        e1c="PASS", run_e2_e7=True, cross_load="identical"
    )
    assert orchestrate.pick_outcome({"a": _v(), "b": _v(c2=False)}) == dict(
        e1c="PASS", run_e2_e7=True, cross_load="unestablished"
    )
    assert orchestrate.pick_outcome({"a": _v(pass_=False), "b": _v()}) == dict(
        e1c="FAIL", run_e2_e7=False, cross_load="identical"
    )
    assert orchestrate.pick_outcome(
        {"a": _v(pass_=False, c1=False), "b": _v()}
    ) == dict(e1c="ABORT", run_e2_e7=False, cross_load="identical")


# ------------------------------------------------------------------ code round 2 H15: evidence vs neutrality
@pytest.mark.parametrize(
    ("change", "kind"),
    [
        (dict(rows=dict(**{"pass": False})), "neutrality"),
        (dict(states=dict(**{"pass": False})), "neutrality"),
        (dict(verdict=dict(**{"pass": False})), "evidence"),
        (dict(draft_rounds=0), "evidence"),
        (dict(prompt_end=8000), "evidence"),
        (dict(straddle=False), "evidence"),
    ],
)
def test_e1c_verdict_separates_evidence_from_neutrality(change, kind):
    on, off = _entries(**change)
    got = orchestrate.e1c_verdict(on, off, threshold=8192, verify_policy=None)
    assert not got["pass_"]
    assert got["neutral"] is (kind != "neutrality"), got
    assert got["evidence_ok"] is (kind != "evidence"), got


def test_pick_outcome_never_reports_invalid_evidence_as_non_neutrality():
    po = orchestrate.pick_outcome
    assert po({"a": _v(evidence=False), "b": _v()})["e1c"] == "INCOMPLETE"
    assert not po({"a": _v(evidence=False), "b": _v()})["run_e2_e7"]
    # an established difference outranks invalid evidence elsewhere; C1 outranks both
    assert po({"a": _v(evidence=False), "b": _v(pass_=False)})["e1c"] == "FAIL"
    assert po({"a": _v(pass_=False), "b": _v(c1=False)})["e1c"] == "ABORT"
    assert po({"a": _v(), "b": _v()})["e1c"] == "PASS"


# ------------------------------------------------------------------ code round 1 H8: watchdogs
SLEEPER = """
import json, os, signal, sys, time
from pathlib import Path
out = Path(sys.argv[1])
if sys.argv[2] == "ignore-term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if sys.argv[3] != "no-progress":
    tmp = out / ".progress.json.tmp"
    tmp.write_text(json.dumps(dict(index=1, name="req-1", started=time.time(), deadline_s=0.3)))
    os.replace(tmp, out / "progress.json")
time.sleep(60)
"""


@pytest.fixture
def sleeper(tmp_path):
    """Start fake workers; every one still alive at teardown is killed (a failing test leaves no orphan)."""
    import subprocess

    procs = []

    def start(*flags):
        proc = subprocess.Popen(
            [sys.executable, "-c", SLEEPER, str(tmp_path), *flags],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(proc)
        return proc

    yield start
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


@pytest.mark.parametrize("term", ["obey-term", "ignore-term"])
def test_supervise_kills_a_request_past_its_deadline_and_records_it(
    tmp_path, sleeper, term
):
    import time

    proc = sleeper(term, "progress")
    started = time.monotonic()
    got = orchestrate.supervise(
        proc,
        tmp_path,
        boot_timeout_s=30,
        deadline=time.monotonic() + 30,
        poll_s=0.05,
        grace_s=0.5,
    )
    assert got["outcome"] == "request_timeout" and got["index"] == 1, got
    assert (
        proc.poll() is not None
    )  # the worker is gone (SIGKILL when SIGTERM is ignored)
    assert time.monotonic() - started < 20
    record = json.loads((tmp_path / "timeout.json").read_text())
    assert record["index"] == 1 and record["name"] == "req-1"


def test_a_term_ignoring_worker_is_reaped_when_its_grace_ends(
    tmp_path, sleeper, monkeypatch
):
    """P396: the grace loop's last ps scan keeps PS_MARGIN_S (as reap's do). A scan bounded by the grace end
    alone runs ps with the few ms left, times out, and turned the SIGKILL reap of a TERM-ignoring worker into
    a harness error (the ignore-term flake). Modelled deterministically: ps needs at least 0.3 s.
    """
    import subprocess
    import time

    real = subprocess.run

    def slow_ps(cmd, *a, **k):
        if cmd[0] == "ps" and k.get("timeout") is not None and k["timeout"] < 0.3:
            raise subprocess.TimeoutExpired(cmd, k["timeout"])
        return real(cmd, *a, **k)

    proc = sleeper("ignore-term", "progress")
    while not (tmp_path / "progress.json").exists():  # written after SIGTERM is ignored
        time.sleep(0.01)
    monkeypatch.setattr(orchestrate.subprocess, "run", slow_ps)
    orchestrate._stop(proc, 0.5)
    assert proc.returncode is not None


def test_supervise_times_out_a_silent_boot(tmp_path, sleeper):
    import time

    proc = sleeper("obey-term", "no-progress")
    got = orchestrate.supervise(
        proc,
        tmp_path,
        boot_timeout_s=0.3,
        deadline=time.monotonic() + 30,
        poll_s=0.05,
        grace_s=0.5,
    )
    assert got["outcome"] == "boot_timeout" and proc.poll() is not None


def test_supervise_enforces_the_campaign_deadline(tmp_path, sleeper):
    import time

    proc = sleeper("obey-term", "no-progress")
    got = orchestrate.supervise(
        proc,
        tmp_path,
        boot_timeout_s=30,
        deadline=time.monotonic() + 0.3,
        poll_s=0.05,
        grace_s=0.5,
    )
    assert got["outcome"] == "wall_limit" and proc.poll() is not None


def test_supervise_returns_the_exit_of_a_finished_worker(tmp_path):
    import subprocess
    import time

    proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    got = orchestrate.supervise(
        proc,
        tmp_path,
        boot_timeout_s=30,
        deadline=time.monotonic() + 30,
        poll_s=0.05,
        grace_s=0.5,
    )
    # code round 5 H30: the exit is observed without reaping; the code comes once reap proved extinction
    assert got == dict(outcome="exited") and proc.returncode is None
    orchestrate.reap(proc, proc.p369_tracker)
    assert proc.returncode == 3


def test_run_with_censoring_relaunches_the_rest_after_a_request_timeout():
    requests = [dict(name=f"r{i}") for i in range(5)]
    launches = []

    def run(batch):
        launches.append([r["name"] for r in batch])
        if len(launches) == 1:  # the third request of the first worker hangs
            raise orchestrate.JobTimeout(
                dict(outcome="request_timeout", index=2), {0: "e0", 1: "e1"}
            )
        return {i: f"late{i}" for i in range(len(batch))}

    done, censored = orchestrate.run_with_censoring(run, requests)
    assert launches == [["r0", "r1", "r2", "r3", "r4"], ["r3", "r4"]]
    assert censored == ["r2"]
    assert done == {"r0": "e0", "r1": "e1", "r3": "late0", "r4": "late1"}


def test_run_with_censoring_never_absorbs_other_failures():
    def run(batch):
        raise orchestrate.JobTimeout(dict(outcome="boot_timeout"), {})

    with pytest.raises(orchestrate.JobTimeout):
        orchestrate.run_with_censoring(run, [dict(name="r0")])


# ------------------------------------------------------------------ code round 1 H5 H6 H7 H10
def _e7_entry(rounds, events=(), n=40, finish="stop", verdict=True, sha="d"):
    return dict(
        verdict={"pass": verdict},
        turns=[
            dict(
                mtp_rounds=rounds,
                events=[dict(cls=c) for c in events],
                prompt=dict(sha256=sha),
                final=dict(generation_tokens=n, finish_reason=finish),
            )
        ],
    )


def test_e7a_stats_count_only_engaged_uncensored_requests():
    done = {
        "r0": _e7_entry(3),
        "r1": _e7_entry(5, ["M4a"]),
        "r2": _e7_entry(0),  # MTP never engaged: not an event-free observation
        "r3": _e7_entry(0, ["M4b_first"], n=1),  # a terminal first token is an event
    }
    got = orchestrate.e7a_stats(done, censored=["r4"])
    assert got["requests"] == 3 and got["not_engaged"] == ["r2"]
    assert got["censored"] == ["r4"]
    assert got["classes"]["M4a"]["requests_with_event"] == 1
    assert got["classes"]["M4b_first"]["requests_with_event"] == 1
    assert got["classes"]["M3"]["requests_with_event"] == 0
    assert got["classes"]["M3"]["ci95"] == orchestrate.clopper_pearson(0, 3)


def _e7b_turn(ok=True, why="", rounds=2, first=False):
    return dict(
        events=[],
        mtp_rounds=rounds,
        prompt=dict(sha256="d"),
        continuity=None if first else dict(continuity=ok, reason=why),
    )


def _e7b_entry(*turns, verdict=True):
    return dict(turns=list(turns), verdict={"pass": verdict})


def test_e7b_session_is_censored_unless_every_follow_up_is_continuous():
    first = _e7b_turn(first=True)
    good = orchestrate.e7b_session(
        "s", _e7b_entry(first, _e7b_turn(), _e7b_turn()), digest="d"
    )
    assert good["continuity"] and not good["censored"]
    bad = orchestrate.e7b_session(
        "s",
        _e7b_entry(first, _e7b_turn(), _e7b_turn(False, "reuse stops")),
        digest="d",
    )
    assert bad["censored"] and "reuse stops" in bad["reason"]


def test_e7b_session_needs_mtp_rounds_the_kernel_verdict_and_the_digest():
    # code round 2 H18 / H6
    first = _e7b_turn(first=True)
    ok = _e7b_entry(first, _e7b_turn(), _e7b_turn())
    assert not orchestrate.e7b_session("s", ok, digest="d")["censored"]
    plain = _e7b_entry(first, _e7b_turn(rounds=0), _e7b_turn())
    assert orchestrate.e7b_session("s", plain, digest="d")["censored"]
    kernel = _e7b_entry(first, _e7b_turn(), _e7b_turn(), verdict=False)
    assert orchestrate.e7b_session("s", kernel, digest="d")["censored"]
    assert orchestrate.e7b_session("s", ok, digest="x")["censored"]
    assert (
        orchestrate.e7b_status([dict(censored=True), dict(censored=False)])
        == "OBSERVED"
    )
    assert orchestrate.e7b_status([dict(censored=True)]) == "INCOMPLETE"


def test_e7a_observation_needs_engagement_a_kernel_verdict_and_the_prompt_digest():
    # code round 2 H6 / H10
    done = {
        "ok": _e7_entry(3),
        "kernel": _e7_entry(3, verdict=False),
        "digest": _e7_entry(3, sha="x"),
        "first": _e7_entry(0, ["M4b_first"], n=1, verdict=False),  # no verify block ran
        "plain": _e7_entry(0),
    }
    got = orchestrate.e7a_stats(done, censored=["late"], digests={k: "d" for k in done})
    assert got["requests"] == 2 and got["status"] == "OBSERVED"
    assert got["invalid"] == ["digest", "kernel"] and got["not_engaged"] == ["plain"]
    none = orchestrate.e7a_stats(
        {"plain": _e7_entry(0)}, censored=["x"], digests={"plain": "d"}
    )
    assert none["status"] == "INCOMPLETE"


def test_prompt_digest_problems_bind_every_request_to_the_reconstruction():
    want = {"math500/1": "a", "cjudge/2": "b"}
    entries = {
        "math500/1": dict(turns=[dict(prompt=dict(sha256="a"))]),
        "cjudge/2": dict(turns=[dict(prompt=dict(sha256="b"))]),
    }
    assert orchestrate.prompt_digest_problems(want, entries) == []
    entries["cjudge/2"]["turns"][0]["prompt"]["sha256"] = "x"
    assert orchestrate.prompt_digest_problems(want, entries)
    del entries["cjudge/2"]["turns"][0]["prompt"]
    assert orchestrate.prompt_digest_problems(want, entries)


OBSERVED = dict(e7a=dict(status="OBSERVED"), e7b=dict(status="OBSERVED"))


def test_overall_status_and_exit_code():
    passing = dict(
        phase="full",
        picks={
            FABLE: dict(outcome=dict(e1c="PASS"), e2=dict(verdict="PASS"), **OBSERVED),
            UNIFORM: dict(
                outcome=dict(e1c="PASS"), e2=dict(verdict="PASS"), **OBSERVED
            ),
        },
    )
    assert orchestrate.overall_status(passing) == ("PASS", 0)
    # code round 2 H10 / P394 (2) refined: E7a (the incidence measure) must be observed for a full PASS;
    # E7b (scripted chat continuity, descriptive by design) is reported and never gates
    for key, want in (("e7a", ("NOT_PASS", 2)), ("e7b", ("PASS", 0))):
        thin = dict(
            passing,
            picks=dict(
                passing["picks"],
                **{
                    FABLE: dict(
                        passing["picks"][FABLE], **{key: dict(status="INCOMPLETE")}
                    )
                },
            ),
        )
        assert orchestrate.overall_status(thin) == want, key
    failing = dict(
        passing,
        picks=dict(passing["picks"], **{UNIFORM: dict(outcome=dict(e1c="FAIL"))}),
    )
    status, code = orchestrate.overall_status(failing)
    assert status == "NOT_PASS" and code == 2
    informative = dict(
        passing,
        picks=dict(
            passing["picks"],
            **{UNIFORM: dict(outcome=dict(e1c="PASS"), e2=dict(verdict="ABORT"))},
        ),
    )
    assert orchestrate.overall_status(informative) == ("NOT_PASS", 2)
    smoke = dict(phase="smoke", picks={}, smoke_gate=dict(pass_=True, fits=True))
    assert orchestrate.overall_status(smoke) == ("SMOKE_PASS", 0)
    smoke_bad = dict(smoke, smoke_gate=dict(pass_=True, fits=False))
    assert orchestrate.overall_status(smoke_bad) == ("SMOKE_NOT_PASS", 2)


# ------------------------------------------------------------------ C184: no default hard stop, results saved per step
def test_the_campaign_has_no_wall_limit_unless_the_operator_sets_one():
    import math

    assert orchestrate.campaign_deadline(None, now=100.0) == math.inf
    assert orchestrate.campaign_deadline(60.0, now=100.0) == 160.0


def test_run_pick_saves_each_step_as_it_finishes(monkeypatch):
    saved = []
    camp = object.__new__(orchestrate.Campaign)
    camp.stack, camp.bench_paths = None, []
    monkeypatch.setattr(orchestrate, "c84_items", lambda stack, pick: [])
    camp.check_bench = lambda pick: None
    camp.size = lambda pick, thresholds: {}
    camp.check_reconstruction = lambda pick, items: {}
    camp.e1c = lambda pick, prompts, smoke: {"T1024": _v()}
    camp.e2 = lambda pick, items, prompts, smoke: dict(verdict="PASS")
    camp.e7a = lambda pick, items: dict(status="OBSERVED")

    def e7b(pick, items):
        raise RuntimeError("worker crashed")

    camp.e7b = e7b
    camp.summary = dict(picks={})
    camp.write_summary = lambda: saved.append(json.loads(json.dumps(camp.summary)))
    with pytest.raises(RuntimeError, match="worker crashed"):
        camp.run_pick(FABLE, smoke=False)
    last = saved[-1]["picks"][FABLE]
    assert last["outcome"]["e1c"] == "PASS" and last["e2"]["verdict"] == "PASS"
    assert (
        last["e7a"]["status"] == "OBSERVED"
    )  # code round 2 H16: E7a saved before E7b runs


def test_a_harness_error_is_recorded_before_it_propagates(tmp_path):
    saved = []
    camp = object.__new__(orchestrate.Campaign)
    camp.args = type("A", (), dict(smoke_summary=None, phase="smoke"))()
    camp.out, camp.picks = tmp_path, [FABLE]
    camp._out_created = (
        True  # what the real prepare() marks once it created the run dir
    )
    camp.prepare = lambda: _binding()
    camp.summary = dict(picks={})
    camp.write_summary = lambda: saved.append(json.loads(json.dumps(camp.summary)))

    def boom(pick, smoke):
        raise RuntimeError("size job failed")

    camp.run_pick = boom
    with pytest.raises(RuntimeError, match="size job failed"):
        camp.execute()
    assert saved[-1]["status"] == "HARNESS_ERROR"
    assert "size job failed" in saved[-1]["error"]


# ------------------------------------------------------------------ P387 (K1): the ON overlay restores the certified tuple
def test_overlay_restores_the_fixture_entries_from_the_expected_interim():
    import yaml

    sys.path.insert(0, str(DEV.parent / "mlx_vlm" / "tests"))
    from test_mtp_verify_scan import _fixture_verbatim, expected_interim

    fixture = _fixture_verbatim()
    want = {e["name"]: e for e in yaml.safe_load(fixture)["models"]}
    for pick in orchestrate.PICKS:
        text, _ = orchestrate.overlay_registry(expected_interim(fixture), pick)
        got = {e["name"]: e for e in yaml.safe_load(text)["models"]}
        assert got[pick] == want[pick], pick


# ------------------------------------------------------------------ code round 2 H14: an flock run lock
def test_the_run_lock_is_exclusive_and_released_with_its_descriptor(tmp_path):
    import os

    lock = tmp_path / ".lock"
    fd = orchestrate.acquire_lock(lock)
    try:
        with pytest.raises(SystemExit, match="held"):
            orchestrate.acquire_lock(lock)
        assert json.loads(lock.read_text())["pid"] == os.getpid()
    finally:
        orchestrate.release_lock(fd)
    assert lock.exists()  # never unlinked: no pathname to race on
    orchestrate.release_lock(orchestrate.acquire_lock(lock))


def test_a_killed_owners_lock_is_free_without_any_reclaim(tmp_path):
    import subprocess

    lock = tmp_path / ".lock"
    code = (
        "import fcntl, os, sys, time\n"
        "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT)\n"
        "fcntl.flock(fd, fcntl.LOCK_EX)\n"
        "print('locked', flush=True)\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(lock)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert proc.stdout.readline().strip() == "locked"
        with pytest.raises(SystemExit, match="held"):
            orchestrate.acquire_lock(lock)
        proc.kill()
        proc.wait()
        orchestrate.release_lock(orchestrate.acquire_lock(lock))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# ------------------------------------------------------------------ code round 2 H8 / H17: stop ordering, process groups
GROUP_WORKER = """
import json, os, signal, subprocess, sys, time
from pathlib import Path
out = Path(sys.argv[1])
child = subprocess.Popen([sys.executable, "-c",  # the child records ITSELF (code round 5 H24)
    "import os, signal, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)", str(out / "child.pid")])
while not (out / "child.pid").exists():
    time.sleep(0.01)
def on_term(*_):
    (out / "saw_timeout.txt").write_text(str((out / "timeout.json").exists()))
    sys.exit(0)
signal.signal(signal.SIGTERM, on_term)
tmp = out / ".progress.json.tmp"
tmp.write_text(json.dumps(dict(index=0, name="r0", phase="request", started=time.time(), deadline_s=0.3)))
os.replace(tmp, out / "progress.json")
time.sleep(60)
"""


def _alive(pid):
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_supervise_records_the_timeout_before_signalling_and_kills_the_group(tmp_path):
    import os
    import signal
    import subprocess
    import time

    proc = (
        subprocess.Popen(  # as launch_worker starts a worker: its own session / group
            [sys.executable, "-c", GROUP_WORKER, str(tmp_path)],
            stdout=subprocess.DEVNULL,
            start_new_session=True,
        )
    )
    child = None
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "child.pid").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        child = int((tmp_path / "child.pid").read_text())
        got = orchestrate.supervise(
            proc,
            tmp_path,
            boot_timeout_s=30,
            deadline=time.monotonic() + 30,
            poll_s=0.05,
            grace_s=0.5,
        )
        assert got["outcome"] == "request_timeout"
        assert (
            tmp_path / "saw_timeout.txt"
        ).read_text() == "True"  # durable before the signal (H8)
        deadline = time.monotonic() + 10
        while _alive(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(child), "the worker's descendant survived the stop (H17)"
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # the whole test group (H24)
        except (ProcessLookupError, PermissionError):
            pass
        recorded = tmp_path / "child.pid"
        for pid in [proc.pid] + (
            [int(recorded.read_text())] if recorded.exists() else []
        ):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait()


def test_launch_worker_starts_its_own_process_group(tmp_path):
    import os
    import subprocess

    proc = orchestrate.launch_worker(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        env=dict(os.environ),
    )
    try:
        assert os.getpgid(proc.pid) == proc.pid != os.getpgid(0)
    finally:
        proc.kill()
        proc.wait()


# ------------------------------------------------------------------ code round 3 H9 / H21
def _git_repo(path):
    import subprocess

    def git(*args):
        return subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@example.invalid",
                *args,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    git("init", "-q")
    return git


def test_the_reviewed_head_is_a_resolved_full_commit_id(tmp_path):
    git = _git_repo(tmp_path)
    (tmp_path / "a").write_text("1")
    git("add", "a")
    git("commit", "-q", "-m", "one")
    head = git("rev-parse", "HEAD")
    assert orchestrate.resolve_commit(tmp_path, head[:8]) == head
    with pytest.raises(SystemExit, match="not a commit"):
        orchestrate.resolve_commit(tmp_path, "0000dead")
    rh = orchestrate.reviewed_head_problems
    assert rh("smoke", head[:1], head)  # a prefix never stands for the reviewed commit
    assert rh("full", head, head) == []


def test_hf_hub_cache_honours_xdg_cache_home():
    from pathlib import Path

    got = orchestrate.hf_hub_cache({"XDG_CACHE_HOME": "/x"}, Path("/h"))
    assert got == Path("/x/huggingface/hub")
    assert orchestrate.hf_hub_cache(
        {"XDG_CACHE_HOME": "/x", "HF_HOME": "/y"}, Path("/h")
    ) == Path("/y/hub")


def test_stack_bench_problems_refuse_dirty_or_untracked_bench_inputs(tmp_path):
    git = _git_repo(tmp_path)
    bench = tmp_path / "benchmark" / "bench"
    bench.mkdir(parents=True)
    (bench / "benchmarks.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-q", "-m", "one")
    paths = ["benchmark/bench"]
    assert orchestrate.stack_bench_problems(tmp_path, paths) == []
    (bench / "benchmarks.py").write_text("x = 2\n")
    assert orchestrate.stack_bench_problems(tmp_path, paths)
    git("checkout", "-q", ".")
    (bench / "new.py").write_text("")
    assert orchestrate.stack_bench_problems(tmp_path, paths)


def test_the_run_lock_and_outputs_live_under_the_stack_workdir(tmp_path):
    from pathlib import Path

    env = {"STACK_WORKDIR": str(tmp_path)}
    assert orchestrate.lock_path(env) == tmp_path / "p369" / "phase2" / ".lock"
    with pytest.raises(SystemExit, match="STACK_WORKDIR"):
        orchestrate.lock_path({})
    assert orchestrate.out_problems(tmp_path / "p369" / "phase2" / "run-1", env) == []
    assert orchestrate.out_problems(Path("/elsewhere/run-1"), env)


# ------------------------------------------------------------------ code round 3 H20 / H22 / H23 / H24
COMPLETED_THEN_HUNG = """
import json, os, sys, time
from pathlib import Path
out = Path(sys.argv[1])
started = time.time()
(out / "req-0.json").write_text(json.dumps(dict(name="r0", done=True)))  # the entry landed...
tmp = out / ".progress.json.tmp"  # ...but the heartbeat still says "request" (the H20 window)
tmp.write_text(json.dumps(dict(index=0, name="r0", phase="request", started=started, deadline_s=0.3)))
os.replace(tmp, out / "progress.json")
time.sleep(60)
"""

ESCAPING_WORKER = """
import json, os, subprocess, sys, time
from pathlib import Path
out = Path(sys.argv[1])
# the child records ITSELF, then leaves the group (the handshake: recorded before it escapes; H24)
code = ("import os, signal, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); os.setsid(); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)")
child = subprocess.Popen([sys.executable, "-c", code, str(out / "child.pid")])
while not (out / "child.pid").exists():
    time.sleep(0.01)
time.sleep(float(sys.argv[2]))
if sys.argv[3] == "hang":
    tmp = out / ".progress.json.tmp"
    tmp.write_text(json.dumps(dict(index=0, name="r0", phase="request", started=time.time(), deadline_s=0.3)))
    os.replace(tmp, out / "progress.json")
    time.sleep(60)
"""


def _wait_dead(pid, seconds=10):
    import time

    deadline = time.monotonic() + seconds
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not _alive(pid)


@pytest.fixture
def owned(tmp_path):
    """Processes a test starts: every one (and its group) is SIGKILLed at teardown (code round 3 H24)."""
    import os
    import signal

    procs, pids = [], []
    yield procs, pids
    for proc in procs:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    recorded = (
        tmp_path / "child.pid"
    )  # a child that escaped the group before the test read its pid
    if recorded.exists() and recorded.read_text().strip().isdigit():
        pids.append(int(recorded.read_text()))
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _start(owned, code, *args):
    import subprocess

    proc = subprocess.Popen(
        [sys.executable, "-c", code, *map(str, args)],
        stdout=subprocess.DEVNULL,
        start_new_session=True,
    )
    owned[0].append(proc)
    return proc


def test_a_completed_request_is_never_censored(tmp_path, owned):
    import time

    proc = _start(owned, COMPLETED_THEN_HUNG, tmp_path)
    got = orchestrate.supervise(
        proc,
        tmp_path,
        boot_timeout_s=30,
        deadline=time.monotonic() + 30,
        poll_s=0.05,
        grace_s=0.5,
    )
    assert got["outcome"] == "hang_after" and got["index"] == 0, got


def test_a_stop_reaches_a_descendant_that_left_the_group(tmp_path, owned):
    import time

    proc = _start(owned, ESCAPING_WORKER, tmp_path, 0.5, "hang")
    deadline = time.monotonic() + 10
    while not (tmp_path / "child.pid").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    child = int((tmp_path / "child.pid").read_text())
    owned[1].append(child)
    tracker = orchestrate.Descendants(proc.pid)
    got = orchestrate.supervise(
        proc,
        tmp_path,
        boot_timeout_s=30,
        deadline=time.monotonic() + 30,
        poll_s=0.05,
        grace_s=0.5,
        tracker=tracker,
    )
    assert got["outcome"] == "request_timeout"
    assert _wait_dead(child), "an escaped (setsid) descendant survived the stop (H22)"


def test_a_normal_exit_leaves_no_descendant(tmp_path, owned):
    import time

    proc = _start(owned, ESCAPING_WORKER, tmp_path, 0.5, "exit")
    tracker = orchestrate.Descendants(proc.pid)
    got = orchestrate.supervise(
        proc,
        tmp_path,
        boot_timeout_s=30,
        deadline=time.monotonic() + 30,
        poll_s=0.05,
        grace_s=0.5,
        tracker=tracker,
    )
    assert got["outcome"] == "exited"
    child = int((tmp_path / "child.pid").read_text())
    owned[1].append(child)
    orchestrate.reap(proc, tracker)
    assert _wait_dead(child), "a descendant outlived a worker's normal exit (H24)"


def _lifecycle_camp(tmp_path, run_pick):
    saved = []
    camp = object.__new__(orchestrate.Campaign)
    camp.args = type("A", (), dict(smoke_summary=None, phase="smoke"))()
    camp.out, camp.picks, camp.fork, camp.head = tmp_path, [FABLE], tmp_path, "h"
    camp._out_created = (
        True  # what the real prepare() marks once it created the run dir
    )
    camp.prepare = lambda: _binding()
    camp.summary = dict(picks={})
    camp.write_summary = lambda: saved.append(json.loads(json.dumps(camp.summary)))
    camp.run_pick = run_pick
    camp.smoke_gate = lambda: dict(pass_=True, fits=True)
    return camp, saved


def test_the_whole_lifecycle_is_one_persistence_boundary(tmp_path, monkeypatch):
    camp, saved = _lifecycle_camp(tmp_path, lambda pick, smoke: None)

    def moved(*a):
        raise RuntimeError("fork HEAD moved during the campaign")

    monkeypatch.setattr(orchestrate, "tree_guard", moved)
    with pytest.raises(RuntimeError, match="HEAD moved"):
        camp.execute()
    assert saved[-1]["status"] == "HARNESS_ERROR" and "HEAD moved" in saved[-1]["error"]

    def interrupted(pick, smoke):
        raise SystemExit(143)

    camp, saved = _lifecycle_camp(tmp_path, interrupted)
    with pytest.raises(SystemExit):
        camp.execute()
    assert saved[-1]["status"] == "INTERRUPTED"


# ------------------------------------------------------------------ P394 (2) refined: E7b in the smoke, never a gate
def _smoke_camp(monkeypatch, calls):
    camp = object.__new__(orchestrate.Campaign)
    camp.stack, camp.bench_paths = None, []
    monkeypatch.setattr(orchestrate, "c84_items", lambda stack, pick: ["i0", "i1"])
    camp.check_bench = lambda pick: None
    camp.size = lambda pick, thresholds: {}
    camp.check_reconstruction = lambda pick, items: {}
    camp.e1c = lambda pick, prompts, smoke: {"T1024": _v()}
    camp.e2 = lambda pick, items, prompts, smoke: dict(verdict="PASS")
    camp.e7a = lambda pick, items: calls.append(("e7a", items)) or dict(
        status="OBSERVED"
    )
    camp.e7b = lambda pick, items, sessions=5: calls.append(("e7b", sessions)) or dict(
        sessions=[dict(name="i0", censored=True)], status="INCOMPLETE"
    )
    camp.summary = dict(picks={})
    camp.write_summary = lambda: None
    return camp


def test_the_smoke_runs_one_e7b_session_and_no_e7a(monkeypatch):
    calls = []
    record = _smoke_camp(monkeypatch, calls).run_pick(FABLE, smoke=True)
    assert calls == [("e7b", 1)] and record["e7b"]["status"] == "INCOMPLETE"
    calls.clear()
    _smoke_camp(monkeypatch, calls).run_pick(FABLE, smoke=False)
    assert [c[0] for c in calls] == ["e7a", "e7b"] and calls[1] == ("e7b", 5)


def test_the_smoke_gate_needs_e7b_to_have_run_not_to_be_continuous():
    camp = object.__new__(orchestrate.Campaign)
    record = dict(
        reconstruction={"math500/1": {}},
        outcome=dict(e1c="PASS"),
        e1c={},
        e2=dict(verdict="PASS"),
        wall_s=60,
    )
    camp.summary = dict(picks={FABLE: record})
    assert any("E7b" in p for p in camp.smoke_gate()["problems"])  # never ran
    record["e7b"] = dict(
        sessions=[dict(name="math500/1", censored=True, reason="timeout")],
        status="INCOMPLETE",
    )  # code round 6 H38: the expected session, a valid report
    gate = camp.smoke_gate()
    assert gate["pass_"], gate  # ran and reported: a censored session does not block
    assert gate["e7b"][FABLE] == "INCOMPLETE"


# ------------------------------------------------------------------ code round 6 H34 H36 H38 (C186): exact-boolean evidence
BAD = ("false", "true", 1, 0, None)


def _bad_paths(record, *paths):
    import copy

    for path in paths:
        for value in BAD:
            changed = copy.deepcopy(record)
            node = changed
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            yield path, value, changed


@pytest.mark.parametrize(
    "path",
    [
        ("pass_",),
        ("neutral",),
        ("evidence_ok",),
        ("c1", "identical"),
        ("c2", "identical"),
    ],
)
def test_pick_outcome_refuses_non_boolean_evidence(path):
    for _, value, bad in _bad_paths(_v(), path):
        with pytest.raises(orchestrate.MalformedEvidence, match="malformed evidence"):
            orchestrate.pick_outcome({"a": bad, "b": _v()})


def test_e1c_verdict_refuses_a_non_boolean_per_query_straddle():
    for value in BAD:
        on, off = _entries(straddle=value)
        with pytest.raises(orchestrate.MalformedEvidence, match="straddle"):
            orchestrate.e1c_verdict(on, off, threshold=8192, verify_policy=None)


def test_e7a_stats_refuses_a_non_boolean_kernel_verdict():
    for value in BAD:
        for entry in (
            _e7_entry(3, verdict=value),
            _e7_entry(0, ["M4b_first"], n=1, verdict=value),  # first-terminal too
        ):
            with pytest.raises(orchestrate.MalformedEvidence, match="verdict"):
                orchestrate.e7a_stats({"r": entry}, censored=[])
    with pytest.raises(orchestrate.MalformedEvidence, match="verdict"):
        orchestrate.e7a_stats({"r": dict(_e7_entry(3), verdict=None)}, censored=[])


def test_e7b_session_refuses_non_boolean_verdict_or_continuity():
    first = _e7b_turn(first=True)
    for value in BAD:
        with pytest.raises(orchestrate.MalformedEvidence, match="verdict"):
            orchestrate.e7b_session(
                "s", _e7b_entry(first, _e7b_turn(), verdict=value), digest="d"
            )
        turn = dict(_e7b_turn(), continuity=dict(continuity=value, reason=None))
        with pytest.raises(orchestrate.MalformedEvidence, match="continuity"):
            orchestrate.e7b_session("s", _e7b_entry(first, turn), digest="d")
    # a follow-up turn always carries a continuity record (the worker writes one for every e7b turn > 0)
    missing = dict(_e7b_turn(), continuity=None)
    with pytest.raises(orchestrate.MalformedEvidence, match="continuity"):
        orchestrate.e7b_session("s", _e7b_entry(first, missing), digest="d")


def test_overall_status_refuses_a_non_boolean_smoke_gate():
    for key in ("pass_", "fits"):
        for value in BAD:
            gate = {**dict(pass_=True, fits=True), key: value}
            with pytest.raises(orchestrate.MalformedEvidence, match=key):
                orchestrate.overall_status(dict(phase="smoke", smoke_gate=gate))


def _e1c_camp(tmp_path, *, acceptance=True, c2_rows=True, c2_states=True):
    camp = object.__new__(orchestrate.Campaign)
    camp.out = tmp_path
    camp.job = lambda pick, name, mode, arm, requests: name
    camp.run = lambda name: None
    on = dict(
        final=dict(token_ids=[1] * 70),
        straddle=True,
        verdict={"pass": True},
        draft_rounds=5,
        prompt_end=orchestrate.straddle_target(1024),
        response=dict(
            timings=dict(verify_blocks_straddle=1, verify_fallback_reasons={})
        ),
        pair=dict(rows={"pass": True}, states={"pass": True}),
        c1=dict(acceptance={"pass": acceptance, "problems": []}),
    )
    c2 = dict(rows={"pass": c2_rows}, states={"pass": c2_states})
    for name, entry in (("e1c-pair-a0", on), ("e1c-c2", c2)):
        d = tmp_path / FABLE / name
        d.mkdir(parents=True)
        (d / "req-0.json").write_text(json.dumps(entry))
    return camp


def test_e1c_refuses_non_boolean_c1_and_c2_evidence(tmp_path):
    good = _e1c_camp(tmp_path / "good").e1c(FABLE, {1024: "x"}, smoke=True)["T1024"]
    assert good["pass_"] and good["c1"]["identical"] is True
    assert good["c2"]["identical"] is True
    for k, value in enumerate(BAD):
        for field in ("acceptance", "c2_rows", "c2_states"):
            camp = _e1c_camp(tmp_path / f"{field}-{k}", **{field: value})
            with pytest.raises(orchestrate.MalformedEvidence, match="malformed"):
                camp.e1c(FABLE, {1024: "x"}, smoke=True)


def _e7b_report(**changes):
    session = dict(
        name="math500/1",
        continuity=False,
        censored=True,
        reason="cache reuse stops",
        events=["M4a"],
        mtp_rounds=[3, 2, 2],
    )
    session.update(changes)
    status = "OBSERVED" if session["censored"] is False else "INCOMPLETE"
    return dict(sessions=[session], status=status)


def _smoke_record(**changes):
    record = dict(
        reconstruction={"math500/1": {}, "math500/2": {}},
        outcome=dict(e1c="PASS"),
        e1c={"T1024": dict(pass_=True)},
        e2=dict(verdict="PASS"),
        e7b=_e7b_report(),
        wall_s=60,
    )
    record.update(changes)
    return record


def test_the_smoke_gate_validates_the_one_expected_e7b_session():
    gate = orchestrate.smoke_gate({FABLE: _smoke_record()})
    assert gate["pass_"] and gate["fits"] and gate["e7b"][FABLE] == "INCOMPLETE", gate
    timeout = dict(sessions=[dict(name="math500/1", censored=True, reason="timeout")])
    timeout["status"] = "INCOMPLETE"
    assert orchestrate.smoke_gate({FABLE: _smoke_record(e7b=timeout)})["pass_"]
    observed = _e7b_report(continuity=True, censored=False, reason=None)
    assert orchestrate.smoke_gate({FABLE: _smoke_record(e7b=observed)})["pass_"]
    # not the expected session: a different item, two sessions, none
    for e7b in (
        _e7b_report(name="cjudge/9"),
        dict(_e7b_report(), sessions=_e7b_report()["sessions"] * 2),
        dict(sessions=[], status="INCOMPLETE"),
    ):
        got = orchestrate.smoke_gate({FABLE: _smoke_record(e7b=e7b)})
        assert not got["pass_"] and any("E7b" in p for p in got["problems"]), e7b
    assert not orchestrate.smoke_gate({FABLE: _smoke_record(e7b=None)})["pass_"]
    assert not orchestrate.smoke_gate({})["pass_"]  # no pick ran


@pytest.mark.parametrize(
    "e7b",
    [
        dict(sessions=[{}], status="INCOMPLETE"),  # the review's minimal shape
        _e7b_report(continuity="false"),
        _e7b_report(censored="true"),
        _e7b_report(censored=False),  # censored must agree with the reason
        _e7b_report(reason=3),
        _e7b_report(events="M4a"),
        _e7b_report(mtp_rounds=[True]),
        _e7b_report(name=None),
        dict(_e7b_report(), status="OBSERVED"),  # the status is re-derived
        dict(_e7b_report(), sessions="s"),
        dict(
            sessions=[dict(name="math500/1", censored=True, reason="late")],
            status="INCOMPLETE",
        ),
    ],
)
def test_the_smoke_gate_refuses_a_malformed_e7b_report(e7b):
    with pytest.raises(orchestrate.MalformedEvidence, match="malformed evidence"):
        orchestrate.smoke_gate({FABLE: _smoke_record(e7b=e7b)})


@pytest.mark.parametrize(
    "record",
    [
        _smoke_record(e1c={"T1024": dict(pass_="true")}),
        _smoke_record(wall_s="60"),
        _smoke_record(
            e1c={
                "T1024": dict(
                    pass_=True, smoke_positives=dict(baseline_pass="true", fired={})
                )
            }
        ),
    ],
)
def test_the_smoke_gate_refuses_malformed_e1c_evidence(record):
    with pytest.raises(orchestrate.MalformedEvidence, match="malformed evidence"):
        orchestrate.smoke_gate({FABLE: record})


def _smoke_summary(**changes):
    picks = {FABLE: _smoke_record()}
    summary = dict(
        phase="smoke",
        status="SMOKE_PASS",
        exit_code=0,
        picks=picks,
        smoke_gate=orchestrate.smoke_gate(picks),
    )
    summary.update(changes)
    return json.loads(json.dumps(summary))  # as loaded from summary.json


def test_a_full_run_is_authorized_only_by_a_validated_smoke_summary():
    sap = orchestrate.smoke_authorization_problems
    assert sap(_smoke_summary()) == []
    gate = _smoke_summary()["smoke_gate"]
    for bad in (
        _smoke_summary(
            smoke_gate=dict(gate, pass_="false", fits="false")
        ),  # H36's scenario
        _smoke_summary(smoke_gate=dict(gate, fits=False)),
        _smoke_summary(status="SMOKE_NOT_PASS"),
        _smoke_summary(status="PASS"),
        _smoke_summary(exit_code=2),
        _smoke_summary(phase="full"),
        _smoke_summary(smoke_gate=None),
        # a stored gate that its own evidence does not reproduce
        _smoke_summary(picks={FABLE: _smoke_record(e2=dict(verdict="FAIL"))}),
        _smoke_summary(
            picks={FABLE: _smoke_record(e7b=dict(sessions=[{}], status="X"))}
        ),
        _smoke_summary(picks={}),
    ):
        assert sap(bad), bad


def test_main_refuses_a_malformed_smoke_summary_before_any_preflight(
    tmp_path, monkeypatch
):
    (tmp_path / "fork").mkdir()
    git = _git_repo(tmp_path / "fork")
    (tmp_path / "fork" / "a").write_text("1")
    git("add", "a")
    git("commit", "-q", "-m", "one")
    head = git("rev-parse", "HEAD")
    monkeypatch.setenv("STACK_WORKDIR", str(tmp_path / "work"))

    def reached(*a, **k):
        raise AssertionError("reached preflight")

    monkeypatch.setattr(orchestrate, "preflight", reached)
    gate = _smoke_summary()["smoke_gate"]
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(_smoke_summary(smoke_gate=dict(gate, pass_="false", fits="false")))
    )
    argv = [
        "--stack-root",
        str(tmp_path),
        "--fork-root",
        str(tmp_path / "fork"),
        "--out",
        str(tmp_path / "work" / "p369" / "phase2" / "run-full"),
        "--phase",
        "full",
        "--reviewed-head",
        head,
        "--smoke-summary",
        str(summary),
    ]
    with pytest.raises(SystemExit, match="smoke"):
        orchestrate.main(argv)
    summary.write_text(json.dumps(_smoke_summary()))
    with pytest.raises(
        AssertionError, match="reached preflight"
    ):  # a valid one gets past
        orchestrate.main(argv)


# ------------------------------------------------------------------ C185 (plan v8 §3 E2): the same-instance triple
D64, E64, F64 = "d" * 64, "e" * 64, "f" * 64


def _turn(tokens, *, rounds=3, argmax=4, sha=D64, finish="length"):
    return dict(
        final=dict(token_ids=list(tokens), finish_reason=finish),
        mtp_rounds=rounds,
        prompt=dict(sha256=sha),
        telemetry=dict(speculative_argmax_from_hidden=argmax),
    )


def _pair(on=(1, 2), off1=(1, 2), off2=(1, 2), *, kernel=True, **on_kw):
    return dict(
        turns=[_turn(on, **on_kw)],
        off=[_turn(off1, rounds=0, argmax=0), _turn(off2, rounds=0, argmax=0)],
        verdict={"pass": kernel},
    )


def _cross(tokens=(1, 2)):
    return dict(turns=[_turn(tokens, rounds=0, argmax=0)])


@pytest.mark.parametrize(
    ("c1", "engaged", "identical", "want"),
    [
        (True, True, True, "PASS"),
        (True, True, False, "FAIL"),
        (True, False, True, "INCOMPLETE"),
        (True, False, False, "INCOMPLETE"),
        (False, True, True, "ABORT"),
        (False, False, False, "ABORT"),
    ],
)
def test_e2_verdict_same_instance_precedence(c1, engaged, identical, want):
    got = orchestrate.e2_verdict(
        c1_identical=c1, engaged=engaged, all_identical=identical
    )
    assert got == want


def test_e2_verdict_refuses_non_boolean_flags():
    for key in ("c1_identical", "engaged", "all_identical"):
        flags = dict(c1_identical=True, engaged=True, all_identical=True)
        for value in BAD:
            with pytest.raises(orchestrate.MalformedEvidence, match=key):
                orchestrate.e2_verdict(**{**flags, key: value})


def test_e2_summary_decides_on_the_same_instance_pair_only():
    got = orchestrate.e2_summary({"a": _pair()}, {"a": _cross()}, digests={"a": D64})
    assert got["verdict"] == "PASS" and got["identity"] == {"a": True}
    assert got["c1_identical"] is True and got["cross_load"]["summary"] == "identical"
    # a cross-load difference is reported and never changes the verdict (C185, like C2)
    far = orchestrate.e2_summary(
        {"a": _pair()}, {"a": _cross((1, 3))}, digests={"a": D64}
    )
    assert far["verdict"] == "PASS"
    assert far["cross_load"] == dict(per_request={"a": False}, summary="unestablished")
    # ON vs OFF1 decides
    fail = orchestrate.e2_summary(
        {"a": _pair(on=(1, 3))}, {"a": _cross()}, digests={"a": D64}
    )
    assert fail["verdict"] == "FAIL" and fail["identity"] == {"a": False}
    # OFF1 vs OFF2 (tokens or finish) differing is ABORT, whatever ON did
    for pair in (_pair(off2=(1, 3)), _pair(on=(1, 3), off2=(1, 3))):
        got = orchestrate.e2_summary({"a": pair}, {"a": _cross()}, digests={"a": D64})
        assert got["verdict"] == "ABORT" and got["c1_identical"] is False, got
    finish = _pair()
    finish["off"][1]["final"]["finish_reason"] = "stop"
    got = orchestrate.e2_summary({"a": finish}, {"a": _cross()}, digests={"a": D64})
    assert got["verdict"] == "ABORT"


def test_e2_summary_needs_engagement_and_equal_digests():
    for pair, why in (
        (_pair(rounds=0), "no MTP rounds"),
        (_pair(argmax=0), "no argmax readout"),
        (_pair(kernel=False), "kernel verdict"),
        (_pair(sha=E64), "digest"),
    ):
        got = orchestrate.e2_summary({"a": pair}, {"a": _cross()}, digests={"a": D64})
        assert got["verdict"] == "INCOMPLETE", why
    # the three trajectories share one templated prompt (straddles have no reconstruction digest)
    split = _pair()
    split["off"][0]["prompt"]["sha256"] = F64
    got = orchestrate.e2_summary({"T1024": split}, {"T1024": _cross()}, digests={})
    assert got["verdict"] == "INCOMPLETE" and got["prompt_digest_problems"]
    # the cross-load arm's prompt is bound to the reconstruction too
    cross = _cross()
    cross["turns"][0]["prompt"]["sha256"] = E64
    got = orchestrate.e2_summary({"a": _pair()}, {"a": cross}, digests={"a": D64})
    assert got["prompt_digest_problems"] and got["verdict"] == "INCOMPLETE"


def test_e2_summary_refuses_malformed_evidence():
    for value in BAD:
        with pytest.raises(orchestrate.MalformedEvidence, match="verdict"):
            orchestrate.e2_summary(
                {"a": _pair(kernel=value)}, {"a": _cross()}, digests={"a": D64}
            )
    two = _pair()
    two["off"] = two["off"][:1]
    with pytest.raises(orchestrate.MalformedEvidence, match="off"):
        orchestrate.e2_summary({"a": two}, {"a": _cross()}, digests={"a": D64})
    with pytest.raises(orchestrate.MalformedEvidence, match="cross"):
        orchestrate.e2_summary({"a": _pair()}, {}, digests={"a": D64})


def _e2_pair_camp(tmp_path, pair, cross):
    camp = object.__new__(orchestrate.Campaign)
    camp.out, camp.digests, camp.jobs = tmp_path, {FABLE: {"a": D64}}, []
    camp.job = lambda pick, name, mode, arm, requests: (name, mode, arm, requests)
    camp.run = lambda job: camp.jobs.append(job)
    camp.c84_payloads = lambda pick, items, greedy, limit=None: [
        dict(name="a", timeout_s=10)
    ]
    for name, entry in (("e2-pair", pair), ("e2-off", cross)):
        d = tmp_path / FABLE / name
        d.mkdir(parents=True)
        (d / "req-0.json").write_text(json.dumps(entry))
    return camp


def test_e2_runs_the_triple_in_the_on_worker_and_one_cross_load_off_arm(tmp_path):
    camp = _e2_pair_camp(tmp_path, _pair(), _cross())
    got = camp.e2(FABLE, [], {}, smoke=True)
    assert got["verdict"] == "PASS", got
    (n1, m1, a1, r1), (n2, m2, a2, r2) = camp.jobs
    assert (n1, m1, a1) == ("e2-pair", "e2_pair", "on")
    assert (n2, m2, a2) == ("e2-off", "e2", "off")
    # ON + OFF1 + OFF2 in one request's budget; the cross-load arm runs it once
    assert r1[0]["timeout_s"] == 3 * r2[0]["timeout_s"]


# ------------------------------------------------------------------ C185 review round 1 E1 E2
@pytest.mark.parametrize(
    ("where", "key", "value"),
    [
        ("on", "token_ids", None),
        ("on", "token_ids", [1, True]),
        ("on", "token_ids", (1, 2)),
        ("off1", "token_ids", None),
        ("off2", "finish_reason", None),
        ("cross", "token_ids", [1.0, 2.0]),
        ("on", "mtp_rounds", True),
        ("on", "mtp_rounds", "3"),
        ("on", "argmax", True),
        ("off1", "mtp_rounds", None),
    ],
)
def test_e2_summary_refuses_malformed_turn_evidence(where, key, value):
    pair, cross = _pair(), _cross()
    turn = dict(
        on=pair["turns"][0],
        off1=pair["off"][0],
        off2=pair["off"][1],
        cross=cross["turns"][0],
    )[where]
    if key == "argmax":
        turn["telemetry"]["speculative_argmax_from_hidden"] = value
    elif key == "mtp_rounds":
        turn["mtp_rounds"] = value
    else:
        turn["final"][key] = value
    with pytest.raises(orchestrate.MalformedEvidence, match="malformed evidence"):
        orchestrate.e2_summary({"a": pair}, {"a": cross}, digests={"a": D64})


@pytest.mark.parametrize("sha", [None, "", "d", "D" * 64, 64])
def test_e2_summary_needs_a_valid_digest_on_every_trajectory(sha):
    # straddles have no reconstruction digest: four equal missing digests must not bind the prompt
    pair, cross = _pair(sha=sha), _cross()
    for turn in pair["off"] + cross["turns"]:
        turn["prompt"]["sha256"] = sha
    got = orchestrate.e2_summary({"T1024": pair}, {"T1024": cross}, digests={})
    assert got["verdict"] == "INCOMPLETE" and got["prompt_digest_problems"], got
    for turn in pair["turns"] + pair["off"] + cross["turns"]:
        del turn["prompt"]["sha256"]
    got = orchestrate.e2_summary({"T1024": pair}, {"T1024": cross}, digests={})
    assert got["verdict"] == "INCOMPLETE" and got["prompt_digest_problems"], got


# ------------------------------------------------------------------ C185 review round 2 E5 E1
def test_e2_output_identity_includes_the_finish_reason():
    # E5: equal token ids with a different finish reason are a different generation outcome
    pair = _pair()
    pair["turns"][0]["final"]["finish_reason"] = "stop"
    got = orchestrate.e2_summary({"a": pair}, {"a": _cross()}, digests={"a": D64})
    assert got["verdict"] == "FAIL" and got["identity"] == {"a": False}, got
    cross = _cross()
    cross["turns"][0]["final"]["finish_reason"] = "stop"
    got = orchestrate.e2_summary({"a": _pair()}, {"a": cross}, digests={"a": D64})
    assert got["verdict"] == "PASS"
    assert got["cross_load"]["per_request"] == {"a": False}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda pair, cross: pair.update(turns=[]),
        lambda pair, cross: pair.update(turns=None),
        lambda pair, cross: pair["turns"][0].update(telemetry=["bad"]),
        lambda pair, cross: pair["turns"][0].update(final=["bad"]),
        lambda pair, cross: pair["off"].__setitem__(0, "bad"),
        lambda pair, cross: cross.update(turns=[]),
        lambda pair, cross: cross.update(turns="x"),
        lambda pair, cross: pair.update(verdict=["bad"]),
    ],
)
def test_e2_summary_refuses_malformed_containers(mutate):
    # E1 (round 2): every malformed shape is MalformedEvidence, never an AttributeError / IndexError
    pair, cross = _pair(), _cross()
    mutate(pair, cross)
    with pytest.raises(orchestrate.MalformedEvidence, match="malformed evidence"):
        orchestrate.e2_summary({"a": pair}, {"a": cross}, digests={"a": D64})
    with pytest.raises(orchestrate.MalformedEvidence, match="malformed evidence"):
        orchestrate.e2_summary({"a": "entry"}, {"a": _cross()}, digests={"a": D64})
