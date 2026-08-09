#!/usr/bin/env python3
"""Fail when a fork change to a file upstream also has is not marked `# Fork:`.

The fourth audit direction. The other three ask what content is *missing* or
*wrongly kept* relative to `upstream/main`; none of them can tell you, for a file
both trees have, **which side wrote a given hunk**. That question is asked on
every merge conflict and answered by archaeology (`git log -S`, `git log --merges`,
reading two versions side by side), which is slow, easy to get wrong under time
pressure, and the direct cause of most entries in `docs/upstream-gaps.md`.

`models/cache.py` already carries the answer as a convention: a boundary comment
plus `# Fork:` on the hunks that deviate from vendored upstream code. This script
makes that convention checkable, so "is this fork work?" becomes a grep.

Granularity: the **enclosing top-level definition**, not the raw hunk.

That choice is deliberate and was calibrated against `cache.py`, the one file that
already follows the convention. It diverges in 7 hunks but carries only 3 markers:
the `prealloc_tokens` feature touches a signature, a constructor body and a growth
calculation, and one `# Fork:` on the body explains all three. A per-hunk rule
fails the file that defines the convention, which would mean the rule is wrong. So
a hunk is covered when the top-level `def`/`class` containing it carries a marker
anywhere inside it -- which is also the unit a reader actually asks about.

Three ways a hunk is covered:

  1. a `# Fork:` marker inside the enclosing top-level definition;
  2. the hunk starts after the file's fork-boundary comment (see BOUNDARY_RE) --
     everything below that line is fork territory by declaration;
  3. for changes outside any definition (imports, module constants), a marker
     inside the hunk itself, since there is no enclosing span to attach to.

Whole files still being brought under the convention go in
`.fork-marker-allowlist` with a reason, and that list is meant to shrink. A
permissive check that passes vacuously is worse than no check, so the allowlist is
per-file and explicit rather than a global threshold, and an entry that no longer
excuses anything is reported as stale exactly like `.symbol-exclusions`.

Reads the git **index**, like the other three audits, so it can gate a commit.

Usage:
    python dev/check_fork_markers.py [--upstream-ref upstream/main] [--summary]
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST_FILE = REPO_ROOT / ".fork-marker-allowlist"

# `# Fork:` is the established spelling in models/cache.py. `# Fork-only:` and
# `# Fork (…):` are accepted so the marker can carry a short qualifier.
MARKER_RE = re.compile(r"#\s*Fork\b[-\s]*(only)?\s*[:(]")

# The boundary convention: everything below this line in the file is fork work.
# Matched loosely on purpose -- the exact wording in cache.py is a paragraph.
BOUNDARY_RE = re.compile(r"#\s*Fork-only\b.*\.\s*$|#\s*Fork additions below\b")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout


def load_allowlist() -> list[tuple[str, str]]:
    """Return [(path_glob, reason)] from .fork-marker-allowlist."""
    if not ALLOWLIST_FILE.exists():
        return []

    out: list[tuple[str, str]] = []
    for lineno, raw in enumerate(ALLOWLIST_FILE.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rule, _, reason = line.partition("#")
        rule, reason = rule.strip(), reason.strip()
        if not rule:
            continue
        if not reason:
            sys.exit(
                f"{ALLOWLIST_FILE.name}:{lineno}: entry {rule!r} has no "
                f"'# reason' comment. Every entry must say why."
            )
        out.append((rule, reason))
    return out


def top_level_spans(source: str) -> list[tuple[int, int]]:
    """(start, end) 1-indexed line spans of every module-level statement.

    Not just defs and classes: a multi-line statement is an enclosing construct
    too, and for some of them a marker *cannot* live inside the hunk. Adding one
    name to a parenthesized `from x import (...)` block is the case that forced
    this -- isort hoists any standalone comment in the block up to the `import (`
    header line, so a comment can never sit adjacent to the added name. Treating
    the whole statement as the span is the same granularity trade-off already made
    for functions, applied consistently.

    Single-line statements yield a 1-line span, so they still require the marker on
    the line itself (`import logging  # Fork: ...`).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    spans = []
    for node in tree.body:
        start = node.lineno
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([start] + [d.lineno for d in node.decorator_list])
        spans.append((start, node.end_lineno or node.lineno))
    return spans


def boundary_line(lines: list[str]) -> int | None:
    """1-indexed first line of the comment block declaring the fork boundary.

    The sentinel sits inside a banner comment, and the hunk that introduces the
    banner starts at the banner's first `# ---` rule, not at the sentence. Walk
    back over the contiguous comment block so that hunk counts as below the
    boundary -- otherwise adding the boundary itself reports as unmarked.
    """
    for i, line in enumerate(lines, start=1):
        if BOUNDARY_RE.search(line):
            start = i
            # Comments, and the blank lines PEP 8 requires before a top-level
            # block: both are part of the hunk that introduces the banner.
            while start > 1 and (
                lines[start - 2].lstrip().startswith("#")
                or not lines[start - 2].strip()
            ):
                start -= 1
            return start
    return None


HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def hunks(diff: str) -> list[tuple[int, int, list[str]]]:
    """[(new_start, new_count, body_lines)] for each hunk of a -U0 diff."""
    out: list[tuple[int, int, list[str]]] = []
    current: list[str] | None = None
    start = count = 0
    for line in diff.splitlines():
        m = HUNK_RE.match(line)
        if m:
            if current is not None:
                out.append((start, count, current))
            start = int(m.group(2))
            count = 1 if m.group(3) is None else int(m.group(3))
            current = []
        elif current is not None and line[:1] in "+-":
            if not line.startswith(("+++", "---")):
                current.append(line)
    if current is not None:
        out.append((start, count, current))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream-ref", default="upstream/main")
    ap.add_argument(
        "--summary",
        action="store_true",
        help="print per-file uncovered counts instead of every hunk; use this to "
        "pick the next file to bring under the convention",
    )
    args = ap.parse_args()
    ref = args.upstream_ref

    upstream_files = {
        p
        for p in git("ls-tree", "-r", "--name-only", ref).splitlines()
        if p.endswith(".py")
    }
    our_files = {p for p in git("ls-files").splitlines() if p.endswith(".py")}

    # Only files that actually differ. Asking git for a per-file diff across all
    # ~1100 shared files takes minutes; one --numstat pass narrows it to ~60.
    diverged = set()
    for line in git("diff", "--cached", "--numstat", ref).splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[2].endswith(".py"):
            diverged.add(parts[2])
    shared = sorted(upstream_files & our_files & diverged)

    allowlist = load_allowlist()
    used: set[str] = set()

    # Keyed by the *site* that needs one marker -- an enclosing top-level
    # definition, or a module-scope hunk -- not by raw hunk. Raw -U0 hunks are a
    # useless unit for a rewritten file: `apc.py` reports 361 of them inside just
    # 4 default-context regions, because diff finds many small common substrings
    # in a rewrite. Sites are what a person actually has to go and annotate.
    uncovered: dict[str, dict[tuple[int, int] | None, int]] = {}
    allowed_files: list[str] = []
    covered_total = 0

    # Coverage is computed for EVERY diverged file, allowlisted or not, and the
    # allowlist is consulted only afterwards. Checking it first looks equivalent
    # and is not: an entry naming a file that is already fully marked would be
    # silently accepted as if it were doing work, so the "no longer excuses
    # anything" warning below could never fire for it. Found by adding an entry
    # for models/base.py -- already marked -- and watching it pass unremarked.
    for path in shared:
        diff = git("diff", "--cached", "-U0", ref, "--", path)
        if not diff.strip():
            continue

        try:
            source = git("show", f":{path}")
        except subprocess.CalledProcessError:
            continue
        lines = source.splitlines()
        spans = top_level_spans(source)
        bound = boundary_line(lines)

        def span_of(line_no: int) -> tuple[int, int] | None:
            for start, end in spans:
                if start <= line_no <= end:
                    return (start, end)
            return None

        def marked(start: int, end: int) -> bool:
            return any(MARKER_RE.search(l) for l in lines[start - 1 : end])

        for new_start, new_count, body in hunks(diff):
            # A pure deletion reports the line it would have followed; look at
            # that line's definition, which is where a marker would explain it.
            probe = max(1, new_start if new_count else new_start)
            if bound is not None and probe >= bound:
                covered_total += 1
                continue
            span = span_of(probe)
            if span is not None and marked(*span):
                covered_total += 1
                continue
            if span is None and any(
                MARKER_RE.search(l[1:]) for l in body if l.startswith("+")
            ):
                covered_total += 1
                continue
            # Module-scope hunks each need their own marker, so key them by
            # line; hunks inside one definition collapse to a single site.
            key = span if span is not None else (probe, probe)
            sites = uncovered.setdefault(path, {})
            sites[key] = sites.get(key, 0) + 1

    # Now suppress the files the allowlist covers, recording which entries did
    # real work so the rest can be reported as stale.
    for path in list(uncovered):
        for glob, _reason in allowlist:
            if fnmatch.fnmatch(path, glob):
                used.add(glob)
                allowed_files.append(path)
                del uncovered[path]
                break

    stale = [glob for glob, _ in allowlist if glob not in used]

    if args.summary or uncovered:
        if allowed_files:
            print(
                f"{len(allowed_files)} file(s) allowlisted in "
                f"{ALLOWLIST_FILE.name} (not yet under the convention)."
            )
        print(f"{covered_total} fork hunk(s) covered by a marker or the boundary.")

    if args.summary:
        if not uncovered:
            print("No unmarked fork sites outside the allowlist.")
        else:
            print("\nUnmarked fork sites, by file (worst first):")
            print(f"  {'sites':>5} {'hunks':>6}  path")
            for path, sites in sorted(uncovered.items(), key=lambda kv: -len(kv[1])):
                print(f"  {len(sites):>5} {sum(sites.values()):>6}  {path}")
            print(
                f"\n  {sum(len(s) for s in uncovered.values()):>5} "
                f"{sum(sum(s.values()) for s in uncovered.values()):>6}  TOTAL "
                f"across {len(uncovered)} file(s)"
            )

    if stale:
        print(
            f"\nwarning: {len(stale)} entr(ies) in {ALLOWLIST_FILE.name} no longer "
            f"excuse anything and should be pruned:"
        )
        for glob in stale:
            print(f"      - {glob}")

    if uncovered and not args.summary:
        total = sum(len(v) for v in uncovered.values())
        print(
            f"\nerror: {total} site(s) in {len(uncovered)} shared file(s) differ "
            f"from {ref} without a `# Fork:` marker:",
            file=sys.stderr,
        )
        for path, sites in sorted(uncovered.items()):
            print(f"  {path}", file=sys.stderr)
            for (start, end), n in sorted(sites.items())[:20]:
                where = f"lines {start}-{end}" if end > start else f"line {start}"
                print(f"      - {where} ({n} hunk(s))", file=sys.stderr)
            if len(sites) > 20:
                print(f"      ... and {len(sites) - 20} more", file=sys.stderr)
        print(
            "\nEither mark the enclosing definition with a `# Fork:` comment "
            f"saying what deviates and why, or add the file to "
            f"{ALLOWLIST_FILE.name} with a reason.",
            file=sys.stderr,
        )
        return 1

    if not args.summary:
        print(f"OK: every fork hunk outside the allowlist is marked, vs {ref}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
