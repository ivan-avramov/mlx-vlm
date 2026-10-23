"""P452: `qualify.py` inside the P426 leg wrapper — the M1 memory leg runs as a CHILD of the P426 leg wrapper (itself a child
of the sequencer), on stack S whose production code carries P426 B1's log-only draft counters.

(1) the quiet-box preflight never refuses the caller's own ancestor chain (it refused the wrapper and the sequencer:
Wave-1 M1 smoke NOT_RUN 2026-10-22), while any other refused process still refuses; (2) `--production-ref` names the
production tree the run attests (default: the pin every earlier qualification ran on), the preflight diffs against it and
the report records it. Pure / temp-git (CPU only).
"""

import subprocess
import sys
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import orchestrate, qualify  # noqa: E402

LEG = ["/venv/bin/python", "-m", "bench.p426_leg", "--axis", "m1"]
SEQ = ["/venv/bin/python", "-m", "bench.p426_wave1", "--stack-root", "/s"]


def test_ancestors_are_not_foreign_but_others_are():
    procs = [(101, LEG), (100, SEQ), (300, LEG), (400, ["/bin/zsh"])]
    problems = orchestrate.foreign_process_problems(procs, me=102, ancestors={101, 100})
    assert len(problems) == 1 and problems[0].startswith("process 300 ")


def test_the_caller_itself_is_never_foreign():
    assert (
        orchestrate.foreign_process_problems([(102, LEG)], me=102, ancestors=set())
        == []
    )


def _git(cwd, *a):
    return subprocess.run(
        ["git", "-C", str(cwd), *a], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def fork(tmp_path):
    root = tmp_path / "fork"
    (root / "mlx_vlm" / "tests").mkdir(parents=True)
    _git(tmp_path, "init", "-q", str(root))
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "t")
    (root / "mlx_vlm" / "server.py").write_text("x = 1\n")
    (root / "mlx_vlm" / "tests" / "test_x.py").write_text("pass\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "prod")
    base = _git(root, "rev-parse", "HEAD")
    (root / "mlx_vlm" / "server.py").write_text("x = 2  # B1 log-only\n")
    _git(root, "commit", "-qam", "b1")
    s = _git(root, "rev-parse", "HEAD")
    (root / "mlx_vlm" / "tests" / "test_x.py").write_text("pass  # tests never count\n")
    _git(root, "commit", "-qam", "test only")
    return root, base, s


def test_production_diff_is_against_the_named_ref(fork):
    root, base, s = fork
    assert "production code differs from" in orchestrate.production_diff_problem(
        root, base
    )
    assert (
        orchestrate.production_diff_problem(root, s) is None
    )  # only a tests change since S


def test_the_production_ref_defaults_to_the_pin_and_is_parsed():
    args = qualify.build_parser().parse_args(
        ["--stack-root", "s", "--fork-root", "f", "--out", "o", "--reviewed-head", "r"]
    )
    assert args.production_ref == orchestrate.PRODUCTION
    args = qualify.build_parser().parse_args(
        [
            "--stack-root",
            "s",
            "--fork-root",
            "f",
            "--out",
            "o",
            "--reviewed-head",
            "r",
            "--production-ref",
            "abc123",
        ]
    )
    assert args.production_ref == "abc123"


# ------------------------------------------------------------------ code round 1 F2 / F3 / F4
CHILD = """
import argparse, json, os, sys, tempfile
sys.path.insert(0, {dev!r})
from p369_phase2 import orchestrate
# code round 2 F4: the REAL preflight (its other checks may add problems on a busy box; only its process census is asserted)
args = argparse.Namespace(fork_root={fork!r}, production_ref="HEAD")
pre = orchestrate.preflight(args, tempfile.mkdtemp())
import psutil
from p369_phase2 import qualify
parent = os.getppid()
print(json.dumps(dict(problems=[p for p in pre["problems"] if p.startswith("process ")], parent=parent,
                      grandparent=psutil.Process(parent).ppid(),
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
# P453 F4: the sequencer level — `python -m bench.p426_wave1` -> `python -m bench.p426_leg` -> the preflight child
SEQUENCER = """
import subprocess, sys
r = subprocess.run([sys.executable, "-m", "bench.p426_leg"], capture_output=True, text=True, check=True)
print(r.stdout, end="")
"""


def _launcher_tree(tmp_path, module):
    """A real process chain: `python -m bench.<module>` -> `python -c <preflight check>`, beside a SIBLING
    `python -m bench.p426_leg` that is not an ancestor."""
    import json
    import time

    pytest.importorskip(
        "psutil",
        reason="the preflight's process census needs psutil (fork-suite extra, P452)",
    )
    pkg = tmp_path / "bench"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for name in {module, "p426_leg"} - {"p426_wave1"}:
        (pkg / f"{name}.py").write_text(
            LAUNCH.format(child=CHILD.format(dev=str(DEV), fork=str(DEV.parent)))
        )
    (pkg / "p426_wave1.py").write_text(SEQUENCER)
    sibling = subprocess.Popen(
        [sys.executable, "-m", "bench.p426_leg", "--sibling"],
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        stdout=subprocess.DEVNULL,
    )
    time.sleep(0.5)
    try:
        out = subprocess.run(
            [sys.executable, "-m", f"bench.{module}"],
            cwd=tmp_path,
            env={"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
    finally:
        sibling.kill()
        sibling.wait()
    doc = json.loads(out)
    return doc, sibling.pid


def test_real_topology_the_launcher_parent_is_exempt_a_sibling_is_not(tmp_path):
    doc, sibling = _launcher_tree(tmp_path, "p426_leg")
    pids = [int(p.split()[1]) for p in doc["problems"]]
    assert doc["parent"] not in pids  # the P426 leg wrapper that launched it
    assert sibling in pids  # the same argv, but not an ancestor


def test_real_topology_another_refused_parent_still_refuses(tmp_path):
    doc, _ = _launcher_tree(tmp_path, "p426_other")
    pids = [int(p.split()[1]) for p in doc["problems"]]
    assert doc["parent"] in pids  # a bench.* parent that is not a P426 launcher


def test_the_production_ref_resolves_to_a_strict_ancestor_sha(fork):
    root, base, s = fork
    head = _git(root, "rev-parse", "HEAD")
    assert (
        orchestrate.resolve_production_ref(root, s[:8], head) == s
    )  # full sha, recorded as diffed
    with pytest.raises(ValueError, match="strict ancestor"):
        orchestrate.resolve_production_ref(root, "HEAD", head)  # tautological
    with pytest.raises(ValueError, match="not a commit"):
        orchestrate.resolve_production_ref(root, "no-such-ref", head)


def test_real_topology_sequencer_and_wrapper_are_both_exempt(tmp_path):
    """P453 F4: sequencer -> leg wrapper -> qualify preflight, as in production: neither launcher is foreign."""
    doc, sibling = _launcher_tree(tmp_path, "p426_wave1")
    pids = [int(p.split()[1]) for p in doc["problems"]]
    assert doc["parent"] not in pids and doc["grandparent"] not in pids
    assert sibling in pids


def test_the_script_entry_point_parses_its_arguments():
    """P452: the leg runs `python dev/p369_phase2/qualify.py …` as a SCRIPT (no package on sys.path); build_parser's
    import must work there too (M1 smoke 2026-10-23: ModuleNotFoundError: p369_phase2 at --help time).
    """
    script = DEV / "p369_phase2" / "qualify.py"
    r = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        cwd=DEV.parent,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr[-800:]
    assert "--production-ref" in r.stdout and "--predictor" in r.stdout


def test_real_topology_the_per_entry_interference_check_exempts_the_launchers_too(tmp_path):
    """P452: qualify's per-entry before / after check (`interference`) is the SECOND process census; it refused the
    sequencer on M1 smoke attempt 4 (2026-10-23) after the preflight passed. It must apply the same launcher rule."""
    doc, sibling = _launcher_tree(tmp_path, "p426_wave1")
    pids = [int(p.split()[1]) for p in doc["interference"]]
    assert doc["parent"] not in pids and doc["grandparent"] not in pids
    assert sibling in pids
