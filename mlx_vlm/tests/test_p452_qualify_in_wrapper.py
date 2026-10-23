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
