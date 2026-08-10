#!/usr/bin/env python3
"""List fork-only definitions that no test mentions. A LEAD GENERATOR, not a gate.

The seventh direction, and the first one that is not about upstream at all.

Every other check in `dev/` compares against `upstream/main`, which means **fork-only
code is invisible to all of them by construction.** There is no upstream copy to diff,
no upstream symbol to miss, no upstream hunk to drop, no marker to demand. The whole
apparatus protects the upstream content this fork carries; nothing protects the fork's
own ~800 commits of work.

That is not a hypothetical gap. Running this found that the fork's legacy
`/v1/completions` endpoint -- `completions_endpoint` (409 lines), five schemas and four
helpers, with both `/completions` and `/v1/completions` as live registered routes -- had
**zero test references.** Writing the tests it lacked immediately turned up a real
user-visible bug: a stop sequence split across token boundaries had its prefix streamed
to the client (`stop="END"` arriving as "EN" + "D" emitted "EN"). Fixed in `b75c18b7`.

WHY THIS IS NOT A GATE, AND SHOULD NOT BECOME ONE

Two reasons, and the first is the same argument that kept
`.body-divergence-exclusions` from becoming a per-definition ledger:

  1. A gate needs an exclusions file, and most of what this reports is legitimately
     fine -- a three-line predicate exercised through its caller's tests does not need
     its own entry. Demanding ~18 written reasons would produce ~18 unreviewed ones,
     which is the "baseline: pre-existing divergence, unreviewed" anti-pattern
     `.symbol-exclusions` took months to drain.
  2. Unlike every other check here, a hit is not a correctness claim. A dropped
     upstream hunk is *wrong*. Untested fork code is *risk*, and risk is ranked and
     worked down, not gated.

WHAT IT ACTUALLY MEASURES, AND WHAT IT DOES NOT

Textual reference counting, not coverage. A name that appears in no test file is
certainly untested; a name that appears in one may be mentioned in an unrelated
assertion, and a name that appears in none may still be fully exercised *through its
caller*. So:

  * a zero in the `tests` column is a real signal, and worth acting on;
  * a non-zero is NOT evidence of coverage.

That asymmetry is why this prints only the zeros. Running an actual coverage pass would
be stronger and is the right follow-up for anything this flags -- it is deliberately
not done here because it needs the full suite (~4 minutes) and this is meant to be
cheap enough to run while deciding what to look at.

The `lib` column counts references from library files OTHER than the defining one. A
definition with 0 tests and 0 external library references is either used only within its
own file or used nowhere -- and "used nowhere" in fork-only code is the shape
`check_dead_helpers.py` deliberately does not report (a fork-only helper nothing calls
yet is our business, one upstream calls is a dropped hunk).

Reads the git **index** (`git ls-files`), like the seven gating audits. That means a
brand-new test file is invisible until it is `git add`ed -- write the tests, stage them,
then re-run, or the count will not move and it looks like the tests did not help.

Usage:
    python dev/find_untested_fork_code.py [--upstream-ref upstream/main] [--all]
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout


def read_blobs(ref: str, paths: list[str]) -> dict[str, str]:
    """One `git cat-file --batch` pass; see check_upstream_symbols.py for why."""
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


def top_level_names(source: str) -> set[str]:
    """Module-level `def`/`class` names. Raises SyntaxError to the caller."""
    return {
        node.name
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def is_test_path(path: str) -> bool:
    return "/tests/" in path


def reference_count(
    name: str, corpus: dict[str, str], exclude: str | None = None
) -> int:
    """Whole-word occurrences of `name` across `corpus`, skipping `exclude`.

    Whole-word on purpose: a substring match would count `_trim_cache` inside
    `_trim_cache_for_apc` and report coverage that does not exist.
    """
    pattern = re.compile(r"\b" + re.escape(name) + r"\b")
    return sum(
        len(pattern.findall(text)) for path, text in corpus.items() if path != exclude
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-ref", default="upstream/main")
    parser.add_argument(
        "--all",
        action="store_true",
        help="list every fork-only definition with its counts, not just the untested "
        "ones. Remember a non-zero `tests` count is NOT evidence of coverage",
    )
    args = parser.parse_args()
    ref = args.upstream_ref

    upstream_files = {
        p
        for p in git("ls-tree", "-r", "--name-only", ref).splitlines()
        if p.endswith(".py")
    }
    our_files = [p for p in git("ls-files").splitlines() if p.endswith(".py")]
    upstream_sources = read_blobs(ref, sorted(set(our_files) & upstream_files))

    # dev/ is this tooling itself, and it has its own tests; test files are the corpus,
    # not the subject.
    fork_defs: dict[str, str] = {}
    unparseable: list[str] = []
    for path in our_files:
        if path.startswith("dev/") or is_test_path(path):
            continue
        local = REPO_ROOT / path
        if not local.exists():
            continue
        try:
            ours = top_level_names(local.read_text())
            upstream = (
                top_level_names(upstream_sources[path])
                if path in upstream_sources
                else set()
            )
        except SyntaxError:
            unparseable.append(path)
            continue
        for name in sorted(ours - upstream):
            if name.startswith("__"):  # dunders are protocol, not features
                continue
            fork_defs.setdefault(name, path)

    library: dict[str, str] = {}
    tests: dict[str, str] = {}
    for path in our_files:
        if path.startswith("dev/"):
            continue
        local = REPO_ROOT / path
        if not local.exists():
            continue
        # Explicit if/else, NOT `cond and a or b`: with an empty dict on the left that
        # idiom silently selects the fallback, which is how the first run of this
        # analysis reported all 95 definitions as untested.
        if is_test_path(path):
            tests[path] = local.read_text()
        else:
            library[path] = local.read_text()

    rows = []
    for name, home in fork_defs.items():
        rows.append(
            (
                reference_count(name, tests),
                reference_count(name, library, exclude=home),
                name,
                home,
            )
        )
    rows.sort()

    untested = [r for r in rows if r[0] == 0]
    print(
        f"{len(fork_defs)} fork-only top-level definitions in library code vs {ref}\n"
        f"{len(untested)} are mentioned by NO test file.\n"
        f"(dev/ excluded — it is this tooling; test files are the corpus.)"
    )
    if unparseable:
        print(f"\nwarning: {len(unparseable)} file(s) could not be parsed:")
        for path in unparseable:
            print(f"      - {path}")

    shown = rows if args.all else untested
    print(f"\n  {'tests':>5} {'lib':>4}  name @ defining file")
    for test_refs, lib_refs, name, home in shown:
        print(f"  {test_refs:>5} {lib_refs:>4}  {name} @ {home}")
    print(
        "\n  `tests`=0 is a real signal. A NON-ZERO count is not evidence of coverage —\n"
        "  it may be an unrelated mention, and a zero may still be exercised through a\n"
        "  caller. Confirm anything here with a real coverage run before concluding.\n"
        "  This is a lead generator; it does not gate, and should not (see the module\n"
        "  docstring for why)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
