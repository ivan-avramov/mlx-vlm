#!/usr/bin/env python3
"""Fail if a top-level/class-level symbol in upstream/main is absent from our tree.

Companion to check_upstream_parity.py. That script catches whole *files* lost in
a merge; this one catches lost *hunks* -- the subtler half of the same failure.

Same root cause: when a merge resolution drops a function upstream added, the
commit that added it is still an ancestor of `main`, so git considers it merged
and no future merge re-offers it. Unlike a missing file, a missing function is
invisible to tree-level diffing -- the file is present, just quietly behind.

Verified instances of exactly this at the time of writing:
  * generate/dispatch.py    -- _prefix_cache_trim_amount / _cache_fully_retained
  * server/generation.py    -- _log_prefill_started / _log_prefill_progress /
                               _log_decode_progress / _request_log_id
  * tests/test_generate.py  -- TestPrefixCacheReuseTrim

Comparison is per-file and name-only: for each .py file upstream and ours both
have, every `def`/`class` name upstream defines must also be defined somewhere
in our copy of that file. It deliberately ignores signatures and bodies -- the
fork legitimately rewrites those. It only asks: did a name silently vanish?

[correction 2026-08-10] This list used to open with `deepseek_v4/config.py --
ModelConfig.index_block / .index_keep`. **This script cannot see those.** It collects
`FunctionDef`/`AsyncFunctionDef`/`ClassDef` names, and a dataclass field is an
`AnnAssign` that `ast.walk` never yields as a name. Both fields are present today, so
they were restored -- but not by this check, and claiming them here made a whole
category look covered when nothing covered it. `dev/check_upstream_registries.py` is
that category (registry entries, re-exports, class attributes); it was built after
this correction, and dropping either field is one of the losses its tests reproduce.

Usage:
    python dev/check_upstream_symbols.py [--upstream-ref upstream/main]
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUSIONS_FILE = REPO_ROOT / ".symbol-exclusions"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout


def read_blobs(ref: str, paths: list[str]) -> dict[str, str]:
    """Read many blobs from ``ref`` in one pass.

    One ``git show`` per file made this take minutes over ~1000 files, which is
    too slow to run in CI; ``git cat-file --batch`` does it in a single process.
    """
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
        if len(header) != 3:  # "<oid> missing" -- path absent at this ref
            pos = newline + 1
            continue
        size = int(header[2])
        body = out[newline + 1 : newline + 1 + size]
        result[path] = body.decode("utf-8", errors="replace")
        pos = newline + 1 + size + 1  # trailing newline after the blob
    return result


def defined_names(source: str) -> set[str]:
    """All function/class names defined anywhere in `source`."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def load_exclusions() -> list[tuple[str, str, str]]:
    """Return [(path_glob, symbol_glob, reason)] from .symbol-exclusions.

    Format: `path_glob::symbol_glob  # reason`
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-ref", default="upstream/main")
    args = parser.parse_args()
    ref = args.upstream_ref

    upstream_files = {
        p
        for p in git("ls-tree", "-r", "--name-only", ref).splitlines()
        if p.endswith(".py")
    }
    # Index, not HEAD, for the same reason as check_upstream_parity.py.
    our_files = {p for p in git("ls-files").splitlines() if p.endswith(".py")}

    exclusions = load_exclusions()
    missing: list[tuple[str, str]] = []
    excused = 0
    used: set[tuple[str, str]] = set()

    shared = sorted(upstream_files & our_files)
    upstream_sources = read_blobs(ref, shared)

    for path in shared:
        up_names = defined_names(upstream_sources.get(path, ""))
        if not up_names:
            continue
        local = REPO_ROOT / path
        if not local.exists():  # staged deletion
            continue
        our_names = defined_names(local.read_text())
        for name in sorted(up_names - our_names):
            for path_glob, sym_glob, _reason in exclusions:
                if fnmatch.fnmatch(path, path_glob) and fnmatch.fnmatch(name, sym_glob):
                    excused += 1
                    used.add((path_glob, sym_glob))
                    break
            else:
                missing.append((path, name))

    if excused:
        print(f"{excused} known symbol exclusion(s) vs {ref}.")

    # An exclusion that no longer excuses anything means the symbol came back —
    # usually because a restore landed. Those entries go stale silently
    # otherwise, and a stale exclusion is a hole in the next audit: it would
    # excuse the symbol again if it were dropped a second time.
    stale = [
        f"{path_glob}::{sym_glob}"
        for path_glob, sym_glob, _reason in exclusions
        if (path_glob, sym_glob) not in used
    ]
    if stale:
        print(
            f"\nwarning: {len(stale)} exclusion(s) in {EXCLUSIONS_FILE.name} no "
            f"longer excuse a missing symbol and should be pruned:"
        )
        for entry in stale:
            print(f"      - {entry}")

    if missing:
        print(
            f"\nerror: {len(missing)} symbol(s) defined in {ref} are missing "
            f"from our copy of the same file:",
            file=sys.stderr,
        )
        current = None
        for path, name in missing:
            if path != current:
                print(f"  {path}", file=sys.stderr)
                current = path
            print(f"      - {name}", file=sys.stderr)
        print(
            f"\nEither port them, or add to {EXCLUSIONS_FILE.name} as\n"
            "    path/glob.py::symbol_name  # why it is intentionally absent",
            file=sys.stderr,
        )
        return 1

    print(f"OK: no upstream symbols silently missing vs {ref}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
