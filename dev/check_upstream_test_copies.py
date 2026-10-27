#!/usr/bin/env python3
"""Upstream's copies of the three test files the fork owns, run against this tree.

The fork owns `test_speculative.py`, `test_server.py` and `test_generate.py` (C104),
which upstream also has. Upstream's own assertions in those files therefore never run
against the fork's library, and a fork change that breaks one of them -- or a merge
that drops the library half of a commit whose tests the fork kept -- fails nothing.
During the 2026-10-25 sync, running upstream's copies and judging only the *new*
failures found a standalone `glm_moe_dsa_mtp` drafter that loaded with every weight
silently dropped.

This audit makes that check permanent:

    git show <ref>:<path>  ->  mlx_vlm/tests/test_zz_up_<stem>.py  ->  pytest  ->  rm

and gates on **new** failures only. A failing test that `.upstream-test-copy-failures`
excuses (with a written reason) is reported as a count; a baseline entry that matches no
failing test is a stale-entry warning, never a failure.

    python dev/check_upstream_test_copies.py --upstream-ref <ref>   # the gate, ~15s
    python dev/check_upstream_test_copies.py --list                 # every failing id

Rules it keeps, each of them a way an audit of this shape goes quietly green:

* Outcomes come from a pytest plugin that writes JSON, never from terminal output
  (which truncates, and node ids may contain ` - `).
* Anything that stops the run from being a run -- an unknown ref, a pytest exit code
  other than 0/1, no result file, a copy that collected nothing -- is exit 2, not 0.
* A run is complete or it is exit 2: collect-only, a deselected test, a collected test
  with no final outcome (`-x`, `--maxfail`) or a copy whose every test was skipped all
  refuse to report.
* The copies, their bytecode, the plugin and the result file are always removed,
  including on Ctrl-C, SIGTERM and SIGHUP. A tracked file, a symlink or a directory at
  a destination is refused, never written through. One run at a time: a lock in the
  git dir.
* `MLX_VLM_*`, `PYTEST_ADDOPTS` and `PYTEST_PLUGINS` are removed from the child's
  environment: upstream's copies assume upstream's defaults, so the result must not
  depend on the operator's shell.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
BASELINE_FILE = ROOT / ".upstream-test-copy-failures"

COPIES = (
    "mlx_vlm/tests/test_speculative.py",
    "mlx_vlm/tests/test_server.py",
    "mlx_vlm/tests/test_generate.py",
)

COPY_PREFIX = "test_zz_up_"
PLUGIN_NAME = "upstream_copy_plugin"
RESULT_ENV = "UPSTREAM_TEST_COPY_RESULT"

# Runs inside the child pytest. Records outcomes as JSON; nothing is parsed from output.
PLUGIN_SOURCE = """\
import json
import os

_failed = {}
_outcomes = {}
_collect_errors = []
_deselected = []
_items = []


def _message(report):
    longrepr = getattr(report, "longrepr", None)
    crash = getattr(longrepr, "reprcrash", None)
    message = getattr(crash, "message", None)
    if message:
        lines = [line.strip() for line in message.splitlines() if line.strip()]
        return lines[0][:300] if lines else ""
    lines = [line for line in str(longrepr or "").splitlines() if line.strip()]
    errors = [line for line in lines if line.startswith("E ")]
    if errors:
        return errors[-1][2:].strip()[:300]
    return lines[-1].strip()[:300] if lines else ""


def pytest_collection_finish(session):
    _items.extend(item.nodeid for item in session.items)


def pytest_deselected(items):
    _deselected.extend(item.nodeid for item in items)


def pytest_collectreport(report):
    if report.failed:
        _failed.setdefault(report.nodeid, _message(report))
        _collect_errors.append(report.nodeid)


def pytest_runtest_logreport(report):
    if report.failed:
        _failed.setdefault(report.nodeid, _message(report))
        _outcomes[report.nodeid] = "failed"
    elif _outcomes.get(report.nodeid) != "failed" and (
        report.when == "call" or (report.when == "setup" and report.skipped)
    ):
        if report.skipped and hasattr(report, "wasxfail"):
            _outcomes[report.nodeid] = "xfailed"
        else:
            _outcomes[report.nodeid] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    with open(os.environ["@RESULT_ENV@"], "w", encoding="utf-8") as out:
        json.dump(
            {
                "failed": _failed,
                "items": _items,
                "outcomes": _outcomes,
                "deselected": _deselected,
                "collect_only": bool(session.config.getoption("collectonly")),
                "collect_errors": _collect_errors,
            },
            out,
        )
""".replace("@RESULT_ENV@", RESULT_ENV)


class CopyRunError(RuntimeError):
    """The copies could not be run to a trustworthy result."""


class CopyResult(Dict[str, str]):
    """{failing original id: message}, plus `counts`: per copy, its outcome counts."""

    counts: Dict[str, Dict[str, int]]


def load_baseline(path: Path = BASELINE_FILE) -> Dict[str, str]:
    """{pattern: reason} from the baseline file; every malformation is a hard error."""
    path = Path(path)
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern, _, reason = line.partition("#")
        pattern, reason = pattern.strip(), reason.strip()
        where = f"{path}:{lineno}"
        if not reason:
            sys.exit(
                f"{where}: entry {pattern!r} has no '# reason' comment. "
                f"Every entry must say why."
            )
        if not any(
            pattern == copy or pattern.startswith(copy + "::") for copy in COPIES
        ):
            sys.exit(
                f"{where}: {pattern!r} does not start with one of "
                f"{', '.join(COPIES)}"
            )
        if pattern in out:
            sys.exit(f"{where}: duplicate pattern {pattern!r}")
        out[pattern] = reason
    return out


def _regex(pattern: str) -> "re.Pattern[str]":
    return re.compile(".*".join(re.escape(part) for part in pattern.split("*")), re.S)


def matches(pattern: str, nodeid: str) -> bool:
    """`*` matches any run (even empty), all else literal; the whole id must match.

    A pattern with no `[` also matches `nodeid` with its trailing `[...]` removed, but
    only when the id contains a `[` and ends in `]`; otherwise the match is literal.
    """
    rx = _regex(pattern)
    if rx.fullmatch(nodeid):
        return True
    if "[" in pattern or "[" not in nodeid or not nodeid.endswith("]"):
        return False
    return bool(rx.fullmatch(nodeid[: nodeid.index("[")]))


def compare(
    failing: "set[str]", baseline: Dict[str, str]
) -> Tuple[List[str], List[str]]:
    """(new, stale): failing ids no pattern matches; patterns no failing id matches."""
    new = sorted(n for n in failing if not any(matches(p, n) for p in baseline))
    stale = sorted(p for p in baseline if not any(matches(p, n) for n in failing))
    return new, stale


def _destination(path: str) -> str:
    parent, _, name = path.rpartition("/")
    return f"{parent}/{COPY_PREFIX}{name}" if parent else f"{COPY_PREFIX}{name}"


def _tail(text: str, lines: int = 30) -> str:
    return "\n".join(text.splitlines()[-lines:])


@contextlib.contextmanager
def _single_run(repo_root: Path) -> Iterator[None]:
    """Hold an exclusive lock in the git dir; a second concurrent run is refused."""
    git = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--absolute-git-dir"],
        capture_output=True,
    )
    if git.returncode != 0:
        raise CopyRunError(
            f"{repo_root} is not a git checkout: "
            f"{git.stderr.decode(errors='replace').strip()}"
        )
    lock = Path(git.stdout.decode().strip()) / "upstream-test-copies.lock"
    try:
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        raise CopyRunError(f"cannot open {lock}: {exc}") from exc
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise CopyRunError(
                f"another run holds {lock}; one run at a time ({exc})"
            ) from exc
        yield
    finally:
        os.close(fd)


@contextlib.contextmanager
def _terminate_as_interrupt() -> Iterator[None]:
    """SIGTERM and SIGHUP raise KeyboardInterrupt, so the cleanup runs."""

    def raise_interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = {
        sig: signal.signal(sig, raise_interrupt)
        for sig in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextlib.contextmanager
def _signals_deferred() -> Iterator[None]:
    """Record SIGINT, SIGTERM and SIGHUP and raise KeyboardInterrupt once the block ends.

    A thread mask would not do: a process-directed signal can reach another thread,
    and Python still runs the handler in the main thread, mid-section.
    """
    received: List[int] = []

    def record(signum, frame):
        received.append(signum)

    previous = {
        sig: signal.signal(sig, record)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if received:
            raise KeyboardInterrupt


def _stale_bytecode(dest: Path) -> List[Path]:
    return list((dest.parent / "__pycache__").glob(f"{dest.stem}*"))


def _cleanup(
    created: Sequence[Path], targets: Sequence[Path], plugin_dir: Path
) -> List[str]:
    """Remove every artifact independently; returns one problem per artifact left."""
    problems: List[str] = []

    def remove(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            problems.append(f"could not remove {path}: {exc}")

    def rmtree_problem(func, path, exc) -> None:
        if isinstance(exc, tuple):  # `onerror` passes exc_info
            exc = exc[1]
        if not isinstance(exc, FileNotFoundError):
            problems.append(f"could not remove {path}: {exc}")

    with _signals_deferred():
        for path in created:
            remove(path)
        for target in targets:
            # Every destination that exists as a regular file is the audit's own: the
            # lock and the pre-checks in `_run_locked` guarantee it.
            try:
                if stat.S_ISREG(target.lstat().st_mode):
                    remove(target)
            except FileNotFoundError:
                pass
            except OSError as exc:
                problems.append(f"could not inspect {target}: {exc}")
            try:
                leftovers = _stale_bytecode(target)
            except OSError as exc:
                problems.append(f"could not list the bytecode of {target}: {exc}")
                continue
            for path in leftovers:
                remove(path)
        if sys.version_info >= (3, 12):
            shutil.rmtree(plugin_dir, onexc=rmtree_problem)
        else:
            shutil.rmtree(plugin_dir, onerror=rmtree_problem)
    return problems


def run_copies(
    ref: str, *, repo_root: Path, copies: Sequence[str], python: str
) -> Dict[str, str]:
    """Run upstream's `copies` at `ref` against the tree; {failing original id: message}."""
    repo_root = Path(repo_root)
    with _single_run(repo_root), _terminate_as_interrupt():
        return _run_locked(ref, repo_root, copies, python)


def _run_locked(
    ref: str, repo_root: Path, copies: Sequence[str], python: str
) -> Dict[str, str]:
    sources: Dict[str, bytes] = {}
    for path in copies:
        shown = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{ref}:{path}"],
            capture_output=True,
        )
        if shown.returncode != 0:
            raise CopyRunError(
                f"git show {ref}:{path} failed: "
                f"{shown.stderr.decode(errors='replace').strip()}"
            )
        try:
            shown.stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CopyRunError(f"{ref}:{path} is not UTF-8: {exc}") from exc
        sources[path] = shown.stdout

    dests = {path: _destination(path) for path in copies}
    for dest in dests.values():
        tracked = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", dest],
            capture_output=True,
        )
        if tracked.returncode == 0:
            raise CopyRunError(f"{dest} is tracked by git; refusing to overwrite it")
        try:
            mode = (repo_root / dest).lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise CopyRunError(f"cannot inspect {dest}: {exc}") from exc
        if not stat.S_ISREG(mode):
            raise CopyRunError(f"{dest} exists and is not a regular file; refusing it")

    run_dir = Path(os.path.commonpath([str(Path(d).parent) for d in dests.values()]))
    cwd = repo_root / run_dir.parent
    targets = [repo_root / d for d in dests.values()]

    try:
        plugin_dir = Path(tempfile.mkdtemp(prefix="upstream-copy-plugin-"))
    except OSError as exc:
        raise CopyRunError(f"cannot create a plugin directory: {exc}") from exc
    created: List[Path] = []
    try:
        result = _execute(python, cwd, plugin_dir, sources, dests, targets, created)
    except BaseException:
        for problem in _cleanup(created, targets, plugin_dir):
            print(f"error: {problem}", file=sys.stderr)
        raise
    problems = _cleanup(created, targets, plugin_dir)
    if problems:
        raise CopyRunError("cleanup failed:\n" + "\n".join(problems))
    return result


def _execute(
    python: str,
    cwd: Path,
    plugin_dir: Path,
    sources: Dict[str, bytes],
    dests: Dict[str, str],
    targets: List[Path],
    created: List[Path],
) -> Dict[str, str]:
    by_basename = {Path(d).name: path for path, d in dests.items()}
    result_file = plugin_dir / "result.json"
    try:
        # Deferred, so a signal cannot land between creating a copy and recording it;
        # it is raised when the block ends, with `created` complete.
        with _signals_deferred():
            (plugin_dir / f"{PLUGIN_NAME}.py").write_text(
                PLUGIN_SOURCE, encoding="utf-8"
            )
            # PYTHONDONTWRITEBYTECODE stops writes, not reads: drop stale bytecode first.
            for target in targets:
                for stale in _stale_bytecode(target):
                    stale.unlink(missing_ok=True)
                target.unlink(missing_ok=True)
            for path, target in zip(dests, targets):
                fd = os.open(
                    target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644
                )
                created.append(target)
                with os.fdopen(fd, "wb") as out:
                    out.write(sources[path])
    except OSError as exc:
        raise CopyRunError(f"cannot write the copies: {exc}") from exc

    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("MLX_VLM_")
        and k not in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS")
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env[RESULT_ENV] = str(result_file)
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(plugin_dir) + (os.pathsep + inherited if inherited else "")
    cmd = [
        python,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--continue-on-collection-errors",
        "-p",
        PLUGIN_NAME,
        *[os.path.relpath(t, cwd) for t in targets],
    ]
    proc = subprocess.run(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    output = proc.stdout.decode(errors="replace")
    if proc.returncode not in (0, 1):
        raise CopyRunError(
            f"pytest exited {proc.returncode}, not 0 or 1:\n{_tail(output)}"
        )
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
        raw_failed: Dict[str, str] = data["failed"]
        items: List[str] = data["items"]
        outcomes: Dict[str, str] = data["outcomes"]
        deselected: List[str] = data["deselected"]
        collect_only: bool = data["collect_only"]
        collect_errors: List[str] = data["collect_errors"]
    except (OSError, ValueError, KeyError) as exc:
        raise CopyRunError(
            f"no usable result file ({exc!r}):\n{_tail(output)}"
        ) from exc

    if collect_only:
        raise CopyRunError(f"pytest ran with --collect-only:\n{_tail(output)}")
    if deselected:
        raise CopyRunError(
            f"{len(deselected)} test(s) were deselected, e.g. {deselected[0]}:\n"
            f"{_tail(output)}"
        )
    unfinished = [n for n in items if n not in outcomes]
    if unfinished:
        raise CopyRunError(
            f"{len(unfinished)} collected test(s) have no outcome (-x, --maxfail or "
            f"a crash?), e.g. {unfinished[0]}:\n{_tail(output)}"
        )

    def original(nodeid: str) -> Optional[str]:
        file_part, sep, rest = nodeid.partition("::")
        orig = by_basename.get(Path(file_part).name)
        return None if orig is None else orig + sep + rest

    failing = CopyResult()
    for nodeid, message in raw_failed.items():
        mapped = original(nodeid)
        if mapped is None:
            raise CopyRunError(f"unexpected failing id {nodeid!r}:\n{_tail(output)}")
        failing[mapped] = re.sub(rf"(?<!\w){COPY_PREFIX}(\w+\.py)", r"\1", message)
    if proc.returncode == 1 and not failing:
        raise CopyRunError(
            f"pytest reported failures but none were recorded:\n{_tail(output)}"
        )

    failing.counts = {}
    for path, dest in dests.items():
        name = Path(dest).name
        mine = [n for n in items if Path(n.partition("::")[0]).name == name]
        errored = any(Path(f).name == name for f in collect_errors)
        if not mine and not errored:
            raise CopyRunError(
                f"{path}: no tests collected from its copy:\n{_tail(output)}"
            )
        states = [outcomes[n] for n in mine]
        if mine and all(state == "skipped" for state in states):
            raise CopyRunError(
                f"{path}: all {len(mine)} collected tests were skipped:\n"
                f"{_tail(output)}"
            )
        failing.counts[path] = {
            "collected": len(mine),
            "passed": states.count("passed"),
            "failed": sum(1 for f in failing if f == path or f.startswith(path + "::")),
            "skipped": states.count("skipped"),
            "xfailed": states.count("xfailed"),
        }
    return failing


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--upstream-ref", default="upstream/main")
    ap.add_argument(
        "--list",
        action="store_true",
        help="print every failing id, ignoring the baseline, and exit 0",
    )
    args = ap.parse_args(argv)
    ref = args.upstream_ref

    baseline: Dict[str, str] = {}
    if not args.list:
        baseline = load_baseline(BASELINE_FILE)
    try:
        failing = run_copies(ref, repo_root=ROOT, copies=COPIES, python=sys.executable)
    except CopyRunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.list:
        for nodeid in sorted(failing):
            print(nodeid)
        return 0

    new, stale = compare(set(failing), baseline)
    excused = len(failing) - len(new)
    used = len(baseline) - len(stale)
    print(
        f"{len(failing)} failing, {excused} excused by {used} baseline entries, "
        f"{len(new)} new vs {ref}"
    )
    for path, count in getattr(failing, "counts", {}).items():
        print(
            f"  {path}: {count['collected']} collected, {count['passed']} passed, "
            f"{count['failed']} failed, {count['skipped']} skipped, "
            f"{count['xfailed']} xfailed"
        )
    for nodeid in new:
        message = failing[nodeid]
        print(f"  NEW {nodeid}" + (f"\n        {message}" if message else ""))
    if stale:
        print(
            f"\nwarning: {len(stale)} entr{'y' if len(stale) == 1 else 'ies'} in "
            f"{Path(BASELINE_FILE).name} no longer excuse a failing test and should "
            f"be pruned:"
        )
        for pattern in stale:
            print(f"      - {pattern}")
    return 1 if new else 0


if __name__ == "__main__":
    raise SystemExit(main())
