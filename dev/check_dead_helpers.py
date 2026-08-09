#!/usr/bin/env python3
"""Fail when upstream calls a helper from library code and we only call it from tests.

The sixth audit, and it exists because one failure shape kept slipping past the
other five. Call it **helper landed, call site dropped**:

    upstream adds a helper AND its call site in one commit -> our merge applies the
    helper's file but drops the call site's hunk -> the symbol exists, is importable,
    is often even unit-tested, and is reachable from nothing.

Every existing check is blind to it *by construction*:

  * `check_upstream_parity.py`    — the file is present.
  * `check_upstream_symbols.py`   — the `def` is present.
  * `check_upstream_deletions.py` — nothing was deleted.
  * `check_fork_markers.py`       — the missing hunk is in a *different* file, and
                                    that file is usually already allowlisted.
  * `find_dropped_hunks.py`       — reports the owning commit, but ranked among
                                    dozens and attributed by line content, so the
                                    signal is diluted (`8422ece8` was reported for
                                    two files, and 17 of its 49 "missing" lines were
                                    coincidental matches in a third).

Four instances were found by hand in a single session, each a real defect:

  * `apc.apc_disk_namespace`      — the on-disk APC namespace stopped fingerprinting
                                    the KV-quant config, so two runs with different
                                    `--kv-bits` shared a prefix cache.
  * `embedding_loader.load_embedding_model` + `models.pooling.read_pooling_config`
                                  — every embedding model silently used the default
                                    pooling mode and ignored its `1_Pooling/config.json`.
  * `apc.self_check_model_apc`    — the APC layout dry-run never ran.
  * `apc.apc_lookup_plan` + `apc.semantic_extra_hash`
                                  — still dropped; `8422ece8`'s dispatch.py refactor.

Three of the four were unit-tested, which is what makes the shape so durable: the
tests keep passing, so the suite says the feature is fine.

The test: for each symbol defined in a *library* module (not tests, not dev), is it
referenced from library code in `upstream/main` but, here, referenced only from
tests or not at all? That difference is the signature. Comparing against upstream
rather than flagging all unused helpers is what keeps the noise down — a fork-only
helper that nothing calls yet is our business; one upstream *does* call is a
dropped hunk.

Reference counting is by name over the AST (`ast.Name` / `ast.Attribute`), not by
import graph: a call site reached through `_apc.foo(...)`, `getattr(mod, "foo")` or
a re-export still counts. That over-counts rather than under-counts, which is the
right direction for a check whose false negatives are silent bugs.

Reviewed hits go in `.dead-helper-exclusions` with a real reason. Reads the git
**index**, like the other gating audits, so it can gate a commit.

Usage:
    python dev/check_dead_helpers.py [--upstream-ref upstream/main] [--summary]
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUSIONS_FILE = REPO_ROOT / ".dead-helper-exclusions"

PACKAGE = "mlx_vlm/"
# Anything under these is not library code: a reference from here does not make a
# helper reachable in production.
NON_LIBRARY = ("mlx_vlm/tests/",)

# Dunders, and the conventional names of things called by frameworks rather than by
# name (pytest fixtures, pydantic validators, nn.Module hooks). Flagging these would
# be pure noise.
SKIP_NAMES = {"main", "setUp", "tearDown"}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout


def read_blobs(ref: str, paths: list[str]) -> dict[str, str]:
    """Read many blobs from ``ref`` in one ``git cat-file --batch`` pass."""
    if not paths:
        return {}
    proc = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=REPO_ROOT,
        input="".join(f"{ref}:{p}\n" for p in paths).encode(),
        stdout=subprocess.PIPE,
        check=True,
    )
    out, result, pos = proc.stdout, {}, 0
    for path in paths:
        newline = out.find(b"\n", pos)
        if newline == -1:
            break
        header = out[pos:newline].split()
        if len(header) != 3:  # "<oid> missing"
            pos = newline + 1
            continue
        size = int(header[2])
        result[path] = out[newline + 1 : newline + 1 + size].decode(
            "utf-8", errors="replace"
        )
        pos = newline + 1 + size + 1
    return result


def is_library(path: str) -> bool:
    return path.startswith(PACKAGE) and not path.startswith(NON_LIBRARY)


def definitions(source: str) -> dict[str, int]:
    """Top-level function/class names -> line number. Methods are excluded.

    Methods are reached through an instance, so "is this name referenced" says
    nothing useful about them; only module-level helpers have call sites that a
    merge can drop independently of the definition.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("__") or node.name in SKIP_NAMES:
                continue
            out[node.name] = node.lineno
    return out


def referenced_names(source: str) -> set[str]:
    """Every bare name and attribute name mentioned anywhere in `source`.

    Deliberately coarse: `foo(...)`, `mod.foo(...)`, `getattr(mod, "foo")` via the
    string constant, and a re-export in `__all__` all count as a reference.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # `getattr(m, "foo")` / __all__ entries.
            if node.value.isidentifier():
                names.add(node.value)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[-1])
    return names


def load_exclusions() -> list[tuple[str, str, str]]:
    if not EXCLUSIONS_FILE.exists():
        return []
    out = []
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
                f"{EXCLUSIONS_FILE.name}:{lineno}: expected 'path_glob::symbol_glob', "
                f"got {rule!r}"
            )
        if not reason:
            sys.exit(
                f"{EXCLUSIONS_FILE.name}:{lineno}: entry {rule!r} has no '# reason' "
                f"comment. Every entry must say why."
            )
        path_glob, _, sym_glob = rule.partition("::")
        out.append((path_glob.strip(), sym_glob.strip(), reason))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream-ref", default="upstream/main")
    ap.add_argument("--summary", action="store_true", help="group hits by file")
    args = ap.parse_args()
    ref = args.upstream_ref

    ours_paths = [p for p in git("ls-files").splitlines() if p.endswith(".py")]
    theirs_paths = [
        p
        for p in git("ls-tree", "-r", "--name-only", ref).splitlines()
        if p.endswith(".py")
    ]

    ours = {
        p: (REPO_ROOT / p).read_text(errors="replace")
        for p in ours_paths
        if (REPO_ROOT / p).exists()
    }
    theirs = read_blobs(ref, theirs_paths)

    # Reference index per side, built in ONE pass over library files. A per-symbol
    # scan would be O(symbols x files) and take minutes on ~1000 files.
    #
    # No special case is needed for the defining file: `def foo` stores its name as
    # a plain string on the FunctionDef node, not as an `ast.Name`, so a definition
    # never counts as a reference to itself. Any hit inside the defining module is
    # therefore a genuine sibling call.
    def reference_index(sources: dict[str, str]) -> set[str]:
        acc: set[str] = set()
        for path, src in sources.items():
            if is_library(path):
                acc |= referenced_names(src)
        return acc

    our_lib_refs = reference_index(ours)
    their_lib_refs = reference_index(theirs)

    exclusions = load_exclusions()
    used: set[tuple[str, str]] = set()
    hits: list[tuple[str, int, str]] = []
    excused = 0

    for path, src in sorted(ours.items()):
        if not is_library(path) or path not in theirs:
            continue
        for name, lineno in sorted(definitions(src).items()):
            if name in our_lib_refs:
                continue  # reachable from library code here
            if name not in their_lib_refs:
                continue  # upstream does not call it either -> not our problem

            for path_glob, sym_glob, _reason in exclusions:
                if fnmatch.fnmatch(path, path_glob) and fnmatch.fnmatch(name, sym_glob):
                    excused += 1
                    used.add((path_glob, sym_glob))
                    break
            else:
                hits.append((path, lineno, name))

    if excused:
        print(f"{excused} known dead-helper exclusion(s) vs {ref}.")

    stale = [f"{pg}::{sg}" for pg, sg, _ in exclusions if (pg, sg) not in used]
    if stale:
        print(
            f"\nwarning: {len(stale)} entr(ies) in {EXCLUSIONS_FILE.name} no longer "
            f"excuse anything and should be pruned:"
        )
        for entry in stale:
            print(f"      - {entry}")

    if hits:
        print(
            f"\nerror: {len(hits)} helper(s) are called from library code in {ref} "
            f"but from nothing but tests here — the dropped-call-site shape:",
            file=sys.stderr,
        )
        current = None
        for path, lineno, name in hits:
            if path != current:
                print(f"  {path}", file=sys.stderr)
                current = path
            print(f"      - {name}  (line {lineno})", file=sys.stderr)
        print(
            "\nFind the commit that added both the helper and its call site "
            "(`git log -S`), restore the call site, or record why it is "
            f"intentionally unreachable in {EXCLUSIONS_FILE.name}.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: no upstream-called helper is unreachable here, vs {ref}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
