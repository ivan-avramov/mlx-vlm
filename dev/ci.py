#!/usr/bin/env python
"""Local CI: run here, on demand, what the fork once delegated to GitHub Actions.

The fork runs no hosted CI. This script is the gate. A stage exits non-zero only
for the gating stages; `drift` and `leads` are information.

    python dev/ci.py                    # style, audits, suite (the full gate)
    python dev/ci.py audits             # the eight audits vs the last-merged ref
    python dev/ci.py audits suite       # any subset, in the order given
    python dev/ci.py drift              # the same audits vs the moving upstream/main
    python dev/ci.py leads              # find_dropped_hunks (wide) + find_untested_fork_code
    python dev/ci.py suite -- -n 3      # everything after `--` goes to pytest

Stages:
  style   pre-commit run --all-files (pre-commit on PATH, else `uvx pre-commit`)
  audits  the eight gating dev/check_*.py audits against .upstream-sync-ref
  suite   the full pytest suite, run from mlx_vlm/ with this interpreter
  drift   the audits against upstream/main; reports, never fails
  leads   the two lead generators; reports, never fails

.upstream-sync-ref holds the upstream ref the fork last merged. Bump it in the
same commit as every sync (AGENTS.md "Fork & branches"), so the audits answer "did
the fork lose parity with what it merged", not "what did upstream add today".
Full output of each step goes to a temporary log directory, printed at the end.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SYNC_REF_FILE = ROOT / ".upstream-sync-ref"

# (script, extra arguments). Every gating dev/check_*.py audit; test_local_ci.py
# fails if a check_*.py is added to dev/ without being listed here.
AUDITS: List[tuple] = [
    ("check_upstream_parity.py", ()),
    ("check_upstream_symbols.py", ()),
    ("check_upstream_deletions.py", ("--strict",)),
    ("check_fork_markers.py", ()),
    ("check_dead_helpers.py", ()),
    ("check_body_divergence.py", ()),
    ("check_upstream_registries.py", ()),
    ("check_call_arguments.py", ()),
]

# (script, the flag that takes the sync ref, extra arguments).
LEADS: List[tuple] = [
    (
        "find_dropped_hunks.py",
        "--ref",
        ("--min-lines", "1", "--min-share", "0.05", "--max-commits", "400"),
    ),
    ("find_untested_fork_code.py", "--upstream-ref", ()),
]

GATING_STAGES = ("style", "audits", "suite")
INFO_STAGES = ("drift", "leads")
DEFAULT_STAGES = GATING_STAGES


@dataclass
class StepResult:
    stage: str
    name: str
    returncode: int
    seconds: float
    log: Optional[Path] = None

    @property
    def gating(self) -> bool:
        return self.stage in GATING_STAGES

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def load_sync_ref(path: Path = SYNC_REF_FILE) -> str:
    """The first non-blank, non-comment line of the sync-ref file."""
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            return line
    raise ValueError(f"{path} names no upstream ref")


def exit_code(results: Sequence[StepResult]) -> int:
    """1 if any gating step failed. Information-only stages never fail the run."""
    return 1 if any(r.gating and not r.ok for r in results) else 0


def _pre_commit_command() -> List[str]:
    if shutil.which("pre-commit"):
        return ["pre-commit"]
    if shutil.which("uvx"):
        return ["uvx", "pre-commit"]
    raise SystemExit("style: neither `pre-commit` nor `uvx` is on PATH")


def _run(stage, name, cmd, logdir, cwd=ROOT) -> StepResult:
    log = logdir / f"{stage}-{name.replace('/', '_')}.log"
    start = time.monotonic()
    with open(log, "w") as out:
        proc = subprocess.run(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT)
    result = StepResult(stage, name, proc.returncode, time.monotonic() - start, log)
    mark = "ok  " if result.ok else ("FAIL" if result.gating else "note")
    print(f"  [{mark}] {stage:6} {name:34} {result.seconds:6.1f}s", flush=True)
    if not result.ok:
        tail = log.read_text().splitlines()[-12:]
        print("\n".join("         | " + t for t in tail), flush=True)
    return result


def _resolve(ref: str) -> None:
    probe = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
    )
    if probe.returncode != 0:
        raise SystemExit(
            f"cannot resolve {ref!r}: run `git fetch upstream` (with tags) first"
        )


def run_stage(stage, logdir, sync_ref, pytest_args) -> List[StepResult]:
    py = sys.executable
    if stage == "style":
        return [
            _run(
                stage,
                "pre-commit",
                _pre_commit_command() + ["run", "--all-files"],
                logdir,
            )
        ]
    if stage in ("audits", "drift"):
        ref = sync_ref if stage == "audits" else "upstream/main"
        _resolve(ref)
        return [
            _run(
                stage,
                script,
                [py, f"dev/{script}", "--upstream-ref", ref, *extra],
                logdir,
            )
            for script, extra in AUDITS
        ]
    if stage == "leads":
        _resolve(sync_ref)
        return [
            _run(stage, script, [py, f"dev/{script}", flag, sync_ref, *extra], logdir)
            for script, flag, extra in LEADS
        ]
    if stage == "suite":
        cmd = [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "./tests", "-rf"]
        return [
            _run(stage, "pytest", cmd + list(pytest_args), logdir, cwd=ROOT / "mlx_vlm")
        ]
    raise SystemExit(f"unknown stage {stage!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pytest_args: List[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, pytest_args = argv[:split], argv[split + 1 :]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "stages",
        nargs="*",
        choices=GATING_STAGES + INFO_STAGES,
        help="default: " + " ".join(DEFAULT_STAGES),
    )
    parser.add_argument("--sync-ref", help="override .upstream-sync-ref")
    args = parser.parse_args(argv)
    stages = args.stages or list(DEFAULT_STAGES)
    sync_ref = args.sync_ref or load_sync_ref()
    logdir = Path(tempfile.mkdtemp(prefix="mlx-vlm-ci-"))
    print(f"local CI: stages={' '.join(stages)} sync-ref={sync_ref} logs={logdir}")
    results: List[StepResult] = []
    for stage in stages:
        results += run_stage(stage, logdir, sync_ref, pytest_args)
    failed = [r for r in results if r.gating and not r.ok]
    noted = [r for r in results if not r.gating and not r.ok]
    print(
        f"local CI: {len(results) - len(failed) - len(noted)} ok, "
        f"{len(failed)} failed, {len(noted)} informational findings; logs in {logdir}"
    )
    return exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
