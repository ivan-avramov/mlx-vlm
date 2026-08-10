#!/usr/bin/env python3
"""Fail when a file's divergence from upstream/main is only ALIGNMENT, not CONTENT.

The fifth audit direction, and the only one that is a *measurement* rather than a
presence test. The other four ask what exists where:

    check_upstream_parity.py     what FILE does upstream have that we lack?
    check_upstream_symbols.py    what SYMBOL does upstream have that we lack?
    check_upstream_deletions.py  what did upstream DELETE that we kept?
    check_fork_markers.py        for a file both trees have, WHICH SIDE wrote this?

None of them, and no line-based measure, can answer the question you need before
sizing a diverged file: **how much of it actually differs.** A file whose
definitions have merely been *reordered* reports as maximally diverged by
`--numstat`, by "missing lines", by "sites" and by "hunks" alike -- those are all
the same measure wearing different hats. `apc.py` read as 554 missing lines / 36
fork-marker sites / 75 hunks and turned out to be four block moves: 74 of its 76
shared definition bodies were byte-identical. Mis-sizing it by two orders of
magnitude cost a cycle, and `docs/upstream-gaps.md` lists five more wrong calls of
the same shape.

This script compares the *bodies* of definitions both trees define, which is the
only measure that distinguishes the two. It exists because that comparison used to
be prose plus a copy-paste snippet in `AGENTS.md`, so it only ran when someone
remembered -- and in the one session where it did run it settled four files and
surfaced a dropped hunk that three prior report-based passes had missed.

Granularity is the **module-level** `def`/`class`, matching `check_fork_markers.py`.
A change to one method therefore shows as its class's body differing. That is a real
limitation and deliberately not fixed: the coarse unit is the one a reader asks
about, and refining it would fragment the ranking that is this script's main output.

WHAT IT GATES ON

Only the three findings that are *alignment masquerading as content*. Each is
mechanically fixable with no archaeology, which is what makes gating on them honest:

  1. `<file>`     -- the file differs from upstream but its content divergence is
                     ZERO: every shared body byte-identical, no names on either
                     side alone, module statements the same multiset. The whole
                     diff is ordering or whitespace. CONVERGE the file.
  2. ALIGNMENT    -- a shared definition whose bodies differ but are equal after
                     dropping blank lines and trailing whitespace. CONVERGE the
                     definition. (Leading whitespace is NOT normalised: in Python
                     indentation is content.)
  3. RELOCATED    -- a shared definition whose body is byte-identical to upstream's
                     but which sits outside the longest common subsequence of the
                     two files' definition order. It is upstream's code in the wrong
                     place, so it is a permanent conflict site for no benefit. Move
                     it back.

It deliberately does NOT gate on "every differing body must be reviewed".
`check_fork_markers.py` already requires a `# Fork:` marker on each diverging site,
and a second per-definition ledger would only add ~50 entries nobody had looked at
-- recreating the "baseline: pre-existing divergence, unreviewed" anti-pattern that
`.symbol-exclusions` took months to drain. What content divergence needs is a human
reading `--summary` and picking the next file, not another list.

TWO CAVEATS, BOTH PAID FOR

  * **Comments are not AST nodes.** A node-based reorder lands a relocated
    definition on the wrong side of a `# ====` banner, and nothing here can see
    that. After acting on a RELOCATED finding, check banner positions by eye. For a
    *test* file also compare `pytest --collect-only` counts on both sides -- that is
    the only thing that catches a shadowed definition, which is how the doubly
    defined `TestLagunaProcessor` hid a missing symbol.
  * **`ours >= upstream` occurrence counts mean "rewrite"; `ours < upstream` is the
    only signal worth chasing.** The cheaper identifier-count technique is strictly
    weaker than this one -- it answers "is this symbol present in both?", not "is
    this body upstream's?" Use it on whatever `--summary` flags, not instead of it.

Reads the git **index**, like the other four audits, so it can gate a commit.

Usage:
    python dev/check_body_divergence.py [--upstream-ref upstream/main]
    python dev/check_body_divergence.py --summary          # rank files by CONTENT
    python dev/check_body_divergence.py --file <path>      # per-definition report
"""

from __future__ import annotations

import argparse
import ast
import difflib
import fnmatch
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUSIONS_FILE = REPO_ROOT / ".body-divergence-exclusions"

# The pseudo-symbol an exclusion uses to excuse finding 1, which is about the whole
# file and so has no symbol of its own. Not a legal Python identifier, so it cannot
# collide with a real name; `path::*` still covers it, which is intended.
FILE_SYMBOL = "<file>"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout


def definition_bodies(source: str) -> tuple[dict[str, str], list[str]]:
    """({name: source text}, [module-level statement texts]) for one file.

    Module-level `def`/`class` only, decorators included in the span -- a decorator
    change is a body change. Raises SyntaxError, which the caller must report rather
    than swallow: a check that goes quiet on an unparseable file is the failure mode
    these scripts exist to prevent.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    defs: dict[str, str] = {}
    module_stmts: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            defs[node.name] = "\n".join(lines[start - 1 : node.end_lineno])
        else:
            module_stmts.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
    return defs, module_stmts


def normalise(text: str) -> str:
    """Strip blank lines and trailing whitespace -- and nothing else.

    Leading whitespace stays: in Python an indentation change is a semantic change,
    so normalising it would let a real content difference read as alignment. So do
    comments, because a dropped comment is one of the ten dropped hunks the marker
    rollout found.
    """
    return "\n".join(l.rstrip() for l in text.splitlines() if l.strip())


def relocated_definitions(up_order: list[str], our_order: list[str]) -> list[str]:
    """Shared definitions outside the longest common subsequence of the two orders.

    The minimum set of definitions to move to align the ordering. Names present on
    only one side are dropped first, so inserting a fork-only helper mid-file does
    not make everything after it look moved -- which is what makes this signal quiet
    enough to gate on (2 hits across 31 diverged files when it was written).
    """
    shared = set(up_order) & set(our_order)
    a = [n for n in up_order if n in shared]
    b = [n for n in our_order if n in shared]
    kept: set[str] = set()
    for block in difflib.SequenceMatcher(
        a=a, b=b, autojunk=False
    ).get_matching_blocks():
        for i in range(block.size):
            kept.add(a[block.a + i])
    return sorted(shared - kept)


class FileComparison:
    """The body-level comparison of one file against upstream."""

    def __init__(self, path: str, upstream_source: str, our_source: str):
        self.path = path
        up_defs, up_mods = definition_bodies(upstream_source)
        our_defs, our_mods = definition_bodies(our_source)
        self.up_defs, self.our_defs = up_defs, our_defs
        self.up_mods, self.our_mods = up_mods, our_mods

        self.shared = sorted(set(up_defs) & set(our_defs))
        self.ours_only = sorted(set(our_defs) - set(up_defs))
        self.up_only = sorted(set(up_defs) - set(our_defs))
        self.identical = [n for n in self.shared if up_defs[n] == our_defs[n]]
        differing = [n for n in self.shared if up_defs[n] != our_defs[n]]
        # An alignment-only body is not content, so it is excluded from the ranking
        # and reported as its own finding instead.
        self.alignment_only = [
            n for n in differing if normalise(up_defs[n]) == normalise(our_defs[n])
        ]
        self.content_differing = [n for n in differing if n not in self.alignment_only]
        self.relocated = [
            n
            for n in relocated_definitions(list(up_defs), list(our_defs))
            if up_defs[n] == our_defs[n]
        ]

        self.module_stmts_ordered_equal = up_mods == our_mods
        self.module_stmts_same_multiset = Counter(
            normalise(s) for s in up_mods
        ) == Counter(normalise(s) for s in our_mods)

    @property
    def content_score(self) -> int:
        """How much of this file genuinely differs. NOT a line count."""
        return (
            len(self.content_differing)
            + len(self.ours_only)
            + len(self.up_only)
            + (0 if self.module_stmts_same_multiset else 1)
        )

    @property
    def alignment_only_file(self) -> bool:
        """The file differs from upstream, but nothing in it does. Finding 1."""
        return self.content_score == 0

    def findings(self) -> list[tuple[str, str]]:
        """[(symbol, kind)] for every gated finding in this file."""
        out: list[tuple[str, str]] = []
        if self.alignment_only_file:
            out.append((FILE_SYMBOL, "the whole diff is ordering or whitespace"))
        for name in self.alignment_only:
            out.append((name, "bodies differ only in blank lines / trailing space"))
        for name in self.relocated:
            out.append((name, "body is byte-identical to upstream but out of order"))
        return out


def load_exclusions() -> list[tuple[str, str, str]]:
    """Return [(path_glob, symbol_glob, reason)] from .body-divergence-exclusions.

    Format: `path_glob::symbol_glob  # reason`, the same as `.symbol-exclusions`.
    Validated here (this is the only reader), so a malformed or reasonless line is a
    hard error rather than a silently ignored one.
    """
    if not EXCLUSIONS_FILE.exists():
        return []

    out: list[tuple[str, str, str]] = []
    for lineno, raw in enumerate(EXCLUSIONS_FILE.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rule, _, reason = line.partition("#")
        rule, reason = rule.strip(), reason.strip()
        if not rule:
            continue
        if "::" not in rule:
            sys.exit(
                f"{EXCLUSIONS_FILE.name}:{lineno}: expected "
                f"'path_glob::symbol_glob', got {rule!r}"
            )
        if not reason:
            sys.exit(
                f"{EXCLUSIONS_FILE.name}:{lineno}: exclusion {rule!r} has no "
                f"'# reason' comment. Every exclusion must say why."
            )
        path_glob, _, symbol_glob = rule.partition("::")
        out.append((path_glob.strip(), symbol_glob.strip(), reason))
    return out


def matching_exclusion(
    path: str, symbol: str, exclusions: list[tuple[str, str, str]]
) -> tuple[str, str] | None:
    """The first exclusion excusing `path::symbol`, or None. Empty list excuses none."""
    for path_glob, symbol_glob, _reason in exclusions:
        if (path == path_glob or fnmatch.fnmatch(path, path_glob)) and (
            symbol == symbol_glob or fnmatch.fnmatch(symbol, symbol_glob)
        ):
            return (path_glob, symbol_glob)
    return None


def diverged_files(ref: str) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Shared .py files whose index copy differs from `ref`, plus their line deltas.

    The line deltas are read only so `--summary` can print them *next to* the content
    score. Showing the two measures side by side is the point of this script.
    """
    upstream_files = {
        p
        for p in git("ls-tree", "-r", "--name-only", ref).splitlines()
        if p.endswith(".py")
    }
    our_files = {p for p in git("ls-files").splitlines() if p.endswith(".py")}

    numstat: dict[str, tuple[int, int]] = {}
    for line in git("diff", "--cached", "--numstat", ref).splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[2].endswith(".py"):
            added, removed, path = parts
            if added.isdigit() and removed.isdigit():  # not binary
                numstat[path] = (int(added), int(removed))

    shared = sorted(upstream_files & our_files & set(numstat))
    return shared, numstat


def compare(path: str, ref: str) -> FileComparison | None:
    """Compare one path, or None if either side is unparseable (reported by caller)."""
    try:
        upstream_source = git("show", f"{ref}:{path}")
        our_source = git("show", f":{path}")
    except subprocess.CalledProcessError:
        return None
    try:
        return FileComparison(path, upstream_source, our_source)
    except SyntaxError:
        return None


def report_one_file(cmp: FileComparison) -> None:
    """The per-definition report the AGENTS.md snippet used to print by hand."""
    print(f"{cmp.path}")
    print(
        f"  shared={len(cmp.shared)} identical={len(cmp.identical)} "
        f"content-differing={len(cmp.content_differing)} "
        f"alignment-only={len(cmp.alignment_only)}"
    )
    print(f"  ours-only={cmp.ours_only}")
    print(f"  up-only={cmp.up_only}")
    print(
        f"  module stmts: up={len(cmp.up_mods)} ours={len(cmp.our_mods)} "
        f"same-order={cmp.module_stmts_ordered_equal} "
        f"same-multiset={cmp.module_stmts_same_multiset}"
    )
    if cmp.relocated:
        print(f"  relocated (identical body, wrong place)={cmp.relocated}")
    for name in cmp.content_differing + cmp.alignment_only:
        up, ours = cmp.up_defs[name], cmp.our_defs[name]
        changed = [
            l
            for l in difflib.unified_diff(
                up.splitlines(), ours.splitlines(), lineterm="", n=0
            )
            if l[:1] in "+-" and not l.startswith(("---", "+++"))
        ]
        tag = "ALIGN" if name in cmp.alignment_only else "DIFF "
        print(
            f"  {tag} {name:48s} up={len(up.splitlines()):4d} "
            f"ours={len(ours.splitlines()):4d} changed={len(changed)}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream-ref", default="upstream/main")
    ap.add_argument(
        "--summary",
        action="store_true",
        help="rank diverged files by CONTENT divergence beside their line delta; "
        "use this to pick the next file to converge or review",
    )
    ap.add_argument(
        "--file",
        metavar="PATH",
        help="print the full per-definition report for one path and exit",
    )
    args = ap.parse_args()
    ref = args.upstream_ref

    if args.file:
        cmp = compare(args.file, ref)
        if cmp is None:
            print(
                f"error: cannot compare {args.file} against {ref} "
                "(absent on one side, or unparseable)",
                file=sys.stderr,
            )
            return 1
        report_one_file(cmp)
        return 0

    shared, numstat = diverged_files(ref)
    exclusions = load_exclusions()

    comparisons: list[FileComparison] = []
    unparseable: list[str] = []
    for path in shared:
        cmp = compare(path, ref)
        if cmp is None:
            unparseable.append(path)
            continue
        comparisons.append(cmp)

    findings: list[tuple[str, str, str]] = []  # (path, symbol, why)
    excused = 0
    used: set[tuple[str, str]] = set()
    for cmp in comparisons:
        for symbol, why in cmp.findings():
            hit = matching_exclusion(cmp.path, symbol, exclusions)
            if hit is not None:
                excused += 1
                used.add(hit)
            else:
                findings.append((cmp.path, symbol, why))

    if excused:
        print(f"{excused} known alignment exclusion(s) vs {ref}.")

    if args.summary:
        print(
            f"\n{len(comparisons)} diverged shared .py file(s) vs {ref}, ranked by "
            "CONTENT divergence.\nThe `lines` column is the SAME file's `--numstat` "
            "delta, shown only to be distrusted:\nit measures alignment. `content` "
            "counts definitions that genuinely differ."
        )
        print(
            f"\n  {'content':>7} {'lines':>11} {'shared':>6} {'ident':>5} "
            f"{'ours':>4} {'up':>3} mod  path"
        )
        for cmp in sorted(comparisons, key=lambda c: (-c.content_score, c.path)):
            added, removed = numstat.get(cmp.path, (0, 0))
            mod = (
                "="
                if cmp.module_stmts_ordered_equal
                else ("~" if cmp.module_stmts_same_multiset else "X")
            )
            print(
                f"  {cmp.content_score:>7} {f'+{added}/-{removed}':>11} "
                f"{len(cmp.shared):>6} {len(cmp.identical):>5} "
                f"{len(cmp.ours_only):>4} {len(cmp.up_only):>3} {mod:>3}  {cmp.path}"
            )
        print(
            "\n  mod: '=' module statements identical in order, '~' same multiset "
            "reordered, 'X' differ.\n  A file with content=0 is a pure reordering — "
            "converge it rather than reading its diff."
        )

    # An exclusion that no longer excuses anything means the alignment was fixed --
    # the same staleness rule the other audits use, and for the same reason: a stale
    # entry would silently excuse the finding a second time if it came back.
    stale = [f"{p}::{s}" for p, s, _r in exclusions if (p, s) not in used]
    if stale:
        print(
            f"\nwarning: {len(stale)} exclusion(s) in {EXCLUSIONS_FILE.name} no "
            f"longer excuse a finding and should be pruned:"
        )
        for entry in stale:
            print(f"      - {entry}")

    # Loud, and fatal. A file this script cannot parse is a file it is not checking,
    # and an audit that quietly covers less than it claims is the exact bug
    # `tests/test_fork_marker_check.py` exists to prevent.
    if unparseable:
        print(
            f"\nerror: {len(unparseable)} diverged file(s) could not be compared "
            f"against {ref} (unparseable, or absent at one side):",
            file=sys.stderr,
        )
        for path in unparseable:
            print(f"      - {path}", file=sys.stderr)
        return 1

    if findings:
        print(
            f"\nerror: {len(findings)} divergence(s) from {ref} are ALIGNMENT, not "
            "content, and should be converged:",
            file=sys.stderr,
        )
        current = None
        for path, symbol, why in findings:
            if path != current:
                print(f"  {path}", file=sys.stderr)
                current = path
            print(f"      - {symbol}: {why}", file=sys.stderr)
        print(
            "\nConverge them — take upstream's text/ordering — or add to "
            f"{EXCLUSIONS_FILE.name} as\n"
            "    path/glob.py::symbol_name  # why the misalignment is deliberate\n"
            f"using {FILE_SYMBOL} as the symbol for a whole-file finding.\n"
            "After moving a definition, check `# ====` banner positions by eye: "
            "comments are not AST\nnodes. For a test file, compare "
            "`pytest --collect-only` counts on both sides too.",
            file=sys.stderr,
        )
        return 1

    if not args.summary:
        print(
            f"OK: no file's divergence from {ref} is alignment-only, and no "
            "upstream definition is merely out of place."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
