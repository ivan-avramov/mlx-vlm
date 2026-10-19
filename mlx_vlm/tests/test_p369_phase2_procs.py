"""P369 phase 2: worker process supervision (code round 4 H27–H30, H9 residual, H23, H24).

Fork-only file. The orchestrator owns every worker it starts: interruptions during the spawn are deferred
and replayed (never a blocked mask the worker inherits), process state comes from a strict ``ps`` table
(a failure is an error, never "no descendants"), every tracked identity (pid + start time, validated on each
traversal) is killed AND awaited, and a cached process-group id is used only while it is still ours.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import orchestrate  # noqa: E402


def _table(rows):
    """A ``ps_table``-shaped dict from ``{pid: (ppid, pgid, stat, start)}``."""
    return {
        pid: dict(ppid=ppid, pgid=pgid, stat=stat, start=start)
        for pid, (ppid, pgid, stat, start) in rows.items()
    }


def _gone(pid):
    row = orchestrate.ps_table().get(pid)
    return row is None or row["stat"].startswith("Z")


@pytest.fixture
def owned(tmp_path):
    """Test processes: every started process group and every pid a child recorded itself under (before it
    could escape) is SIGKILLed at teardown (code round 4 H24)."""
    procs = []
    yield procs
    for proc in procs:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    for path in tmp_path.glob("*.pid"):
        text = path.read_text().strip()
        if text.isdigit():
            try:
                os.kill(int(text), signal.SIGKILL)
            except ProcessLookupError:
                pass


# A child that records ITSELF, then leaves the session (the handshake: recorded before it escapes).
ESCAPER = (
    "import os, signal, sys, time\n"
    "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
    "os.setsid()\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "time.sleep(60)\n"
)
PARENT = (
    "import subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])\n"
    "time.sleep(float(sys.argv[3]))\n"
)


def _parent(owned, tmp_path, linger):
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            PARENT,
            ESCAPER,
            str(tmp_path / "child.pid"),
            str(linger),
        ],
        start_new_session=True,
    )
    owned.append(proc)
    deadline = time.monotonic() + 10
    while not (tmp_path / "child.pid").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.2)  # the child has called setsid
    return proc, int((tmp_path / "child.pid").read_text())


# ------------------------------------------------------------------ H28: a strict ps table
def test_ps_table_is_strict_and_sees_this_process():
    table = orchestrate.ps_table()
    me = table[os.getpid()]
    assert me["ppid"] == os.getppid() and me["pgid"] == os.getpgid(0)
    assert len(me["start"].split()) == 5  # LC_ALL=C lstart: "Mon Oct 19 14:03:12 2026"


def test_ps_table_raises_instead_of_failing_open(monkeypatch):
    calls = []

    def failing(*args, **kwargs):
        calls.append(1)
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(orchestrate.subprocess, "run", failing)
    with pytest.raises(RuntimeError, match="ps"):
        orchestrate.ps_table(retries=3, backoff_s=0)
    assert len(calls) == 3

    def garbage(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="  12 x y\n", stderr="")

    monkeypatch.setattr(orchestrate.subprocess, "run", garbage)
    with pytest.raises(RuntimeError, match="unparseable"):
        orchestrate.ps_table(retries=1, backoff_s=0)

    def empty(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    monkeypatch.setattr(orchestrate.subprocess, "run", empty)
    with pytest.raises(RuntimeError, match="lacks"):
        orchestrate.ps_table(retries=1, backoff_s=0)


# ------------------------------------------------------------------ H30: validated identities
def test_a_stale_identity_is_not_a_traversal_frontier():
    root = _table({100: (1, 100, "S", "t0")})
    tracker = orchestrate.Descendants(100, table=root)
    tracker.seen = {500: "old"}
    reused = _table(
        {
            100: (1, 100, "S", "t0"),
            500: (1, 500, "S", "new"),  # pid 500 now belongs to an unrelated process
            501: (500, 500, "S", "new2"),  # ...whose child must not be enrolled
            600: (100, 100, "S", "t1"),  # a genuine descendant
        }
    )
    tracker.scan(table=reused)
    assert 501 not in tracker.seen and tracker.seen[600] == "t1"
    assert tracker.alive(reused) == [600]  # the stale 500 is not ours to kill


def test_a_reused_root_pid_makes_the_cached_group_foreign():
    tracker = orchestrate.Descendants(100, table=_table({100: (1, 100, "S", "t0")}))
    foreign = _table({100: (1, 100, "S", "other"), 101: (100, 100, "S", "x")})
    with pytest.raises(RuntimeError, match="no longer ours"):
        tracker.group_members(foreign)
    ours = _table(
        {101: (1, 100, "S", "t2")}
    )  # the leader exited; a member of its group remains
    assert tracker.group_members(ours) == [101]


# ------------------------------------------------------------------ H29: tracked identities are awaited
def test_reap_kills_and_awaits_an_escaped_descendant(tmp_path, owned):
    proc, child = _parent(owned, tmp_path, linger=30)
    tracker = orchestrate.Descendants(proc.pid)
    tracker.scan()
    assert child in tracker.seen
    orchestrate.reap(proc, tracker, wait_s=10)
    assert _gone(child)  # no wait of its own: reap returned only after extinction


def test_reap_fails_closed_when_a_tracked_identity_survives(
    tmp_path, owned, monkeypatch
):
    proc, child = _parent(owned, tmp_path, linger=30)
    tracker = orchestrate.Descendants(proc.pid)
    tracker.scan()
    real_kill = os.kill
    monkeypatch.setattr(
        orchestrate.os,
        "kill",
        lambda pid, sig: None if pid == child else real_kill(pid, sig),
    )
    with pytest.raises(RuntimeError, match="survived"):
        orchestrate.reap(proc, tracker, wait_s=0.5)


# ------------------------------------------------------------------ H27: deferred interruptions
def test_interruptions_during_the_spawn_are_deferred_and_replayed():
    before = signal.getsignal(signal.SIGTERM)
    with orchestrate.deferred_signals() as pending:
        os.kill(os.getpid(), signal.SIGTERM)
        deadline = time.monotonic() + 5
        while not pending and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pending == [signal.SIGTERM]  # recorded, nothing raised
    assert signal.getsignal(signal.SIGTERM) is before


def test_a_worker_never_inherits_a_blocked_sigterm():
    probe = (
        "import signal\n"
        "print(int(signal.SIGTERM in signal.pthread_sigmask(signal.SIG_BLOCK, [])))\n"
    )
    with orchestrate.deferred_signals():
        proc = orchestrate.launch_worker(
            [sys.executable, "-c", probe], stdout=subprocess.PIPE, env=dict(os.environ)
        )
        out, _ = proc.communicate(timeout=30)
    assert out.strip() == b"0"


# ------------------------------------------------------------------ H9 residual: stack inputs + offline workers
def _git_repo(path):
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


def test_the_stack_bench_state_must_still_equal_the_manifest(tmp_path):
    git = _git_repo(tmp_path)
    bench = tmp_path / "benchmark" / "bench"
    bench.mkdir(parents=True)
    (bench / "benchmarks.py").write_text("x = 1\n")
    (tmp_path / "bundle.jsonl").write_text("{}\n")
    git("add", ".")
    git("commit", "-q", "-m", "one")
    want = orchestrate.stack_bench_state(tmp_path, ["bundle.jsonl"])
    assert orchestrate.stack_bench_drift(tmp_path, want) == []
    (bench / "benchmarks.py").write_text("x = 2\n")
    git("commit", "-q", "-am", "moved cleanly")  # a clean tree, but not the bound one
    assert orchestrate.stack_bench_drift(tmp_path, want)


def test_workers_resolve_models_offline():
    env = orchestrate.worker_env({"PATH": "/bin"})
    assert (
        env["HF_HUB_OFFLINE"] == "1"
    )  # no mid-campaign download can change the loaded weights


# ------------------------------------------------------------------ H23: preparation inside the boundary
def test_a_preparation_refusal_is_recorded(tmp_path):
    saved = []
    camp = object.__new__(orchestrate.Campaign)
    camp.args = type("A", (), dict(smoke_summary=None, phase="smoke"))()
    camp.out, camp.summary = tmp_path, dict(picks={})
    camp.write_summary = lambda: saved.append(json.loads(json.dumps(camp.summary)))

    def refuse():
        camp._out_created = True
        raise orchestrate.Refused("stack bench inputs not clean")

    camp.prepare = refuse
    with pytest.raises(SystemExit, match="not clean"):
        camp.execute()
    assert saved[-1]["status"] == "REFUSED"
