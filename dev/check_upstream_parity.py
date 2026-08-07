#!/usr/bin/env python3
"""Fail if any file present in upstream/main is missing from this tree.

Why this exists
---------------
A fork accumulates *invisible* upstream losses. When a merge resolution drops a
file that upstream added, the upstream commit that added it still becomes an
ancestor of `main` (via the merge's second parent). Git therefore treats the
file as "merged, then deliberately deleted", and every subsequent 3-way merge
sees `base has it / ours deleted it / theirs unchanged` and silently keeps the
deletion. The loss is permanent and self-perpetuating: no future `git merge`
will ever surface it again, and `git log --diff-filter=D` finds nothing because
the deletion never happened in a real commit.

This check is the tripwire. Anything upstream has that we don't must either be
restored or listed -- with a reason -- in .merge-exclusions.

Usage:
    python dev/check_upstream_parity.py [--upstream-ref upstream/main]
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUSIONS_FILE = REPO_ROOT / ".merge-exclusions"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def load_exclusions() -> list[tuple[str, str]]:
    """Return [(glob, reason)] from .merge-exclusions.

    Format: one `glob  # reason` per line. A reason is mandatory -- an
    unexplained exclusion is how silent loss creeps back in.
    """
    if not EXCLUSIONS_FILE.exists():
        return []

    out: list[tuple[str, str]] = []
    for lineno, raw in enumerate(EXCLUSIONS_FILE.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern, _, reason = line.partition("#")
        pattern, reason = pattern.strip(), reason.strip()
        if not pattern:
            continue
        if not reason:
            sys.exit(
                f"{EXCLUSIONS_FILE.name}:{lineno}: exclusion '{pattern}' has "
                f"no '# reason' comment. Every exclusion must say why."
            )
        out.append((pattern, reason))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-ref", default="upstream/main")
    args = parser.parse_args()

    ref = args.upstream_ref
    try:
        git("rev-parse", "--verify", ref)
    except subprocess.CalledProcessError:
        print(
            f"error: {ref} not found. Add the upstream remote first:\n"
            "    git remote add upstream git@github.com:Blaizzy/mlx-vlm.git\n"
            "    git remote set-url --push upstream no_push\n"
            "    git fetch upstream",
            file=sys.stderr,
        )
        return 2

    upstream_paths = set(git("ls-tree", "-r", "--name-only", ref).splitlines())
    # Index, not HEAD: a restore that is staged but not yet committed should
    # count as present, so this can gate a commit rather than only report on one.
    our_paths = set(git("ls-files").splitlines())

    missing = sorted(upstream_paths - our_paths)
    exclusions = load_exclusions()

    unexplained: list[str] = []
    excused: list[tuple[str, str]] = []
    for path in missing:
        for pattern, reason in exclusions:
            if fnmatch.fnmatch(path, pattern):
                excused.append((path, reason))
                break
        else:
            unexplained.append(path)

    if excused:
        print(f"{len(excused)} known exclusion(s) vs {ref}:")
        for path, reason in excused:
            print(f"  - {path}  ({reason})")

    if unexplained:
        print(
            f"\nerror: {len(unexplained)} file(s) exist in {ref} but are "
            f"missing here, with no entry in {EXCLUSIONS_FILE.name}:",
            file=sys.stderr,
        )
        for path in unexplained:
            print(f"  - {path}", file=sys.stderr)
        print(
            "\nEither restore them:\n"
            f"    git checkout {ref} -- <path>...\n"
            f"or add each to {EXCLUSIONS_FILE.name} with a '# reason'.",
            file=sys.stderr,
        )
        return 1

    print(f"\nOK: no unexplained files missing vs {ref}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
