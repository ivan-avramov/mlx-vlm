#!/usr/bin/env python3
"""Real line coverage for fork-only definitions. The strong form of the seventh direction.

`find_untested_fork_code.py` counts textual references and says outright that this is
weaker: a zero there is a real signal, a non-zero proves nothing, and a zero may still
be exercised through a caller. This script answers the question that one only
approximates -- **were these lines actually executed by the test suite?**

It needs a coverage data file, so it is not cheap and it is not a gate; it is the
confirmation step the other script tells you to run.

    uv pip install --python .venv/bin/python coverage
    cd mlx_vlm && ../.venv/bin/python -m coverage run \\
        --source=<repo>/mlx_vlm --data-file=/tmp/cov.data \\
        -m pytest -q ./tests --ignore=tests/test_smoke.py
    .venv/bin/python dev/fork_coverage_report.py --data-file=/tmp/cov.data

WHAT "FORK-ONLY" MEANS HERE

Same definition as the rest of `dev/`: a module-level `def`/`class` in a file we share
with `upstream/main` that upstream does not define, plus everything in files upstream
does not have at all. `dev/` itself and test files are excluded -- `dev/` is this
tooling and tests are the thing doing the covering.

WHY PER-DEFINITION AND NOT PER-FILE

A file-level percentage is useless here. `server/openai.py` is thousands of lines of
mostly-upstream code; the fork's contribution to it is a dozen definitions, and their
coverage is invisible in the file's total. The unit that matters is the definition,
which is also the unit `find_untested_fork_code.py` reports and the unit a `# Fork:`
marker annotates.

CAVEATS, in the same spirit as the sibling script

  * Line coverage, not branch. A fully-covered definition can still have an
    unexercised branch; `--branch` on the coverage run plus `analysis2`'s arc data
    would be needed for that, and is a bigger job than this answers.
  * "Executed" is not "asserted". A line run as a side effect of an unrelated test is
    covered here and tested by nobody. This narrows the search; it does not close it.
  * A definition with no executable lines (a bare dataclass body, a Protocol) reports
    as fully covered because there is nothing to miss. Those are listed separately
    rather than silently counted as wins.

Usage:
    python dev/fork_coverage_report.py [--data-file PATH] [--upstream-ref REF] [--all]
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout


def read_blobs(ref: str, paths: list[str]) -> dict[str, str]:
    if not paths:
        return {}
    proc = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=REPO_ROOT,
        input="".join(f"{ref}:{p}\n" for p in paths).encode(),
        stdout=subprocess.PIPE,
        check=True,
    )
    out = proc.stdout
    result: dict[str, str] = {}
    pos = 0
    for path in paths:
        newline = out.find(b"\n", pos)
        if newline == -1:
            break
        header = out[pos:newline].split()
        if len(header) != 3:
            pos = newline + 1
            continue
        size = int(header[2])
        result[path] = out[newline + 1 : newline + 1 + size].decode(
            "utf-8", errors="replace"
        )
        pos = newline + 1 + size + 1
    return result


def top_level_defs(source: str) -> dict[str, tuple[int, int]]:
    """{name: (start_line, end_line)}, decorators included in the span."""
    out: dict[str, tuple[int, int]] = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            out[node.name] = (start, node.end_lineno or node.lineno)
    return out


def fork_only_spans(ref: str) -> dict[str, dict[str, tuple[int, int]]]:
    """{path: {name: span}} for every fork-only top-level definition."""
    upstream_files = {
        p
        for p in git("ls-tree", "-r", "--name-only", ref).splitlines()
        if p.endswith(".py")
    }
    our_files = [p for p in git("ls-files").splitlines() if p.endswith(".py")]
    upstream_sources = read_blobs(ref, sorted(set(our_files) & upstream_files))

    out: dict[str, dict[str, tuple[int, int]]] = {}
    for path in our_files:
        if path.startswith("dev/") or "/tests/" in path:
            continue
        local = REPO_ROOT / path
        if not local.exists():
            continue
        try:
            ours = top_level_defs(local.read_text())
            upstream = (
                set(top_level_defs(upstream_sources[path]))
                if path in upstream_sources
                else set()
            )
        except SyntaxError:
            continue
        fork = {
            name: span
            for name, span in ours.items()
            if name not in upstream and not name.startswith("__")
        }
        if fork:
            out[path] = fork
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default="/tmp/cov.data")
    parser.add_argument("--upstream-ref", default="upstream/main")
    parser.add_argument(
        "--all", action="store_true", help="list every definition, not just gaps"
    )
    args = parser.parse_args()

    try:
        from coverage import Coverage
    except ImportError:
        print(
            "error: `coverage` is not installed in this interpreter.\n"
            "    uv pip install --python .venv/bin/python coverage",
            file=sys.stderr,
        )
        return 1

    if not Path(args.data_file).exists():
        print(
            f"error: no coverage data at {args.data_file}. Run the suite under "
            "coverage first — see this module's docstring.",
            file=sys.stderr,
        )
        return 1

    cov = Coverage(data_file=args.data_file)
    cov.load()

    spans = fork_only_spans(args.upstream_ref)
    rows = []
    no_executable = []
    unmeasured = []

    for path in sorted(spans):
        absolute = str(REPO_ROOT / path)
        try:
            _fname, statements, _excluded, missing, _fmt = cov.analysis2(absolute)
        except Exception:
            # A file coverage never measured (never imported by any test).
            unmeasured.append(path)
            continue
        stmt_set, miss_set = set(statements), set(missing)
        for name, (start, end) in sorted(spans[path].items()):
            in_span = {ln for ln in stmt_set if start <= ln <= end}
            if not in_span:
                no_executable.append((name, path))
                continue
            missed = len(in_span & miss_set)
            rows.append((len(in_span) - missed, len(in_span), missed, name, path))

    total_stmts = sum(r[1] for r in rows)
    total_hit = sum(r[0] for r in rows)
    fully_missed = [r for r in rows if r[0] == 0]
    partial = [r for r in rows if r[0] and r[2]]

    print(
        f"Fork-only definitions with executable lines: {len(rows)}\n"
        f"Statements: {total_stmts}, covered {total_hit} "
        f"({100.0 * total_hit / total_stmts:.1f}%)\n"
        f"ENTIRELY UNCOVERED: {len(fully_missed)}   partially covered: {len(partial)}"
    )
    if unmeasured:
        print(
            f"\n{len(unmeasured)} file(s) never imported by any test (no coverage data):"
        )
        for path in unmeasured:
            print(f"      - {path}")
    if no_executable:
        print(
            f"\n{len(no_executable)} definition(s) have no executable lines "
            "(bare class bodies etc.) — nothing to miss, not a win:"
        )
        for name, path in no_executable:
            print(f"      - {name} @ {path}")

    def show(label, subset):
        if not subset:
            return
        print(f"\n{label}")
        print(f"  {'hit':>4} {'stmts':>5} {'miss':>4}  name @ file")
        for hit, stmts, missed, name, path in sorted(
            subset, key=lambda r: (r[0], -r[2])
        ):
            print(f"  {hit:>4} {stmts:>5} {missed:>4}  {name} @ {path}")

    show("ENTIRELY UNCOVERED — no line of these ran during the suite:", fully_missed)
    show("Partially covered:", partial if args.all else partial[:0])
    if args.all:
        show("Fully covered:", [r for r in rows if r[2] == 0])

    print(
        "\n  Line coverage, not branch: a fully-covered definition can still have an\n"
        "  unexercised branch. And 'executed' is not 'asserted' — a line run as a side\n"
        "  effect of an unrelated test counts as covered here. This narrows the search."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
