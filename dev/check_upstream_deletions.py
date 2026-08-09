#!/usr/bin/env python
"""Audit the *reverse* direction: content upstream deleted that we still carry.

`check_upstream_parity.py`, `check_upstream_symbols.py` and
`find_dropped_hunks.py` all ask the same question — "what does `upstream/main`
have that we lack?" — and none of them asks the mirror image. That leaves a whole
class of divergence invisible:

    upstream deletes a file or a symbol -> the deleting commit becomes an ancestor
    of `main` through a merge -> our resolution keeps our side -> the content
    lives on forever as *stale upstream code* that reads exactly like fork work.

`mlx_vlm/video_generate.py` was 645 lines of this. Upstream removed the command
in `e48ed11b` (#1454); 7 of that commit's 8 files applied, so every README, doc
and example reference was stripped while the module and its `__main__.py`
registration survived. A working, undocumented command that no check could see.

AGENTS.md warns that "a 'fork-only' symbol is often stale *upstream* code". This
script is that warning, automated. It reports two directions:

  FILES    a file we have that `upstream/main` lacks, where some commit that
           *deleted* it is an ancestor of `upstream/main` -> upstream removed it
           on purpose. If our copy is byte-identical to the last pre-deletion
           copy, that is conclusive: STALE, not FORK.

  SYMBOLS  a `def`/`class` in a file we share with upstream that exists in our
           copy and not theirs, where the name appears somewhere in that file's
           *upstream-reachable history* -> upstream once had it and removed it.

Both are **lead generators, not oracles**, exactly like `find_dropped_hunks.py`.
A symbol can legitimately be reintroduced as fork work under a name upstream once
used, and a file upstream deleted can be one we deliberately keep. Every hit
still needs `git log -S` and a read before acting. Reviewed hits go in
`.deletion-exclusions` with a reason.

Exit status is 0 unless --strict is passed; this is a report, and the backlog it
describes should not break CI on day one.

    python dev/check_upstream_deletions.py
    python dev/check_upstream_deletions.py --files-only
    python dev/check_upstream_deletions.py --strict
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUSIONS_FILE = REPO_ROOT / ".deletion-exclusions"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def git_ok(*args: str) -> bool:
    return (
        subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
        ).returncode
        == 0
    )


def defined_names(source: str) -> set[str]:
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
    """Parse `.deletion-exclusions`. Same format and rules as `.symbol-exclusions`.

    `path/glob::symbol_glob  # reason`, or `path/glob  # reason` for a whole file.
    Every entry must carry a reason — an unexplained exclusion is how a real gap
    becomes permanent.
    """
    if not EXCLUSIONS_FILE.exists():
        return []
    out = []
    for raw in EXCLUSIONS_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rule, _, reason = line.partition("#")
        rule, reason = rule.strip(), reason.strip()
        if not reason:
            raise SystemExit(
                f"error: {EXCLUSIONS_FILE.name}: entry {rule!r} has no "
                f"'# reason' comment. Every exclusion must say why."
            )
        path_glob, _, symbol_glob = rule.partition("::")
        out.append((path_glob.strip(), symbol_glob.strip() or "*", reason))
    return out


def excused(exclusions, path: str, symbol: str) -> bool:
    return any(
        fnmatch.fnmatch(path, pg) and fnmatch.fnmatch(symbol, sg)
        for pg, sg, _ in exclusions
    )


def history_defines(ref: str, path: str, candidates: set[str]) -> set[str]:
    """Which of `candidates` this file ever defined, anywhere in `ref`'s history.

    Deliberately *not* "AST-parse every historical revision and union the names".
    That was the obvious implementation and it cost ~10 minutes on this repo,
    because files like `tests/test_server.py` have long histories and parsing each
    revision is expensive. Three things make it fast instead:

      1. one batched `git cat-file` for all revisions, not a `git log -S` per name
      2. a cheap `def <name>`/`class <name>` substring scan to *reject* blobs,
         falling back to a real AST parse only for the ones that might match —
         so a nested `def` inside a string still can't produce a false positive
      3. early exit once every candidate has been accounted for

    Revisions are deduplicated by blob OID first, which alone removes most of the
    work: a file's content is unchanged across most commits that "touch" it via
    merges.
    """
    if not candidates:
        return set()
    revs = git("log", "--format=%H", ref, "--", path).split()
    if not revs:
        return set()

    # Resolve revisions to blob OIDs and dedupe before reading any content.
    specs = "\n".join(f"{rev}:{path}" for rev in revs) + "\n"
    oid_proc = subprocess.run(
        ["git", "cat-file", "--batch-check"],
        cwd=REPO_ROOT,
        input=specs.encode(),
        capture_output=True,
    )
    oids = []
    seen_oids = set()
    for line in oid_proc.stdout.decode(errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] == "blob" and parts[0] not in seen_oids:
            seen_oids.add(parts[0])
            oids.append(parts[0])
    if not oids:
        return set()

    needles = {name: (f"def {name}", f"class {name}") for name in candidates}
    found: set[str] = set()

    proc = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=REPO_ROOT,
        input=("\n".join(oids) + "\n").encode(),
        capture_output=True,
    )
    buf = proc.stdout
    pos = 0
    while pos < len(buf) and len(found) < len(candidates):
        nl = buf.find(b"\n", pos)
        if nl == -1:
            break
        parts = buf[pos:nl].decode(errors="replace").split()
        if len(parts) != 3:  # "<oid> missing" and friends
            pos = nl + 1
            continue
        size = int(parts[2])
        body = buf[nl + 1 : nl + 1 + size]
        pos = nl + 1 + size + 1  # trailing newline after the object

        text = body.decode("utf-8", errors="replace")
        maybe = {
            name
            for name in candidates - found
            if any(needle in text for needle in needles[name])
        }
        if maybe:
            found |= maybe & defined_names(text)
    return found


def audit_files(ref: str, exclusions) -> list[tuple[str, str, bool]]:
    """Files we have and `ref` lacks, which some ancestor commit of `ref` deleted."""
    ours = set(git("ls-files").splitlines())
    theirs = set(git("ls-tree", "-r", "--name-only", ref).splitlines())
    findings = []
    for path in sorted(ours - theirs):
        if excused(exclusions, path, "*"):
            continue
        deleters = git(
            "log", "--all", "--diff-filter=D", "--format=%H", "--", path
        ).split()
        for commit in deleters:
            # Only upstream's own deletions count. Ours (e.g. a file we removed
            # and later reinstated) are not evidence of upstream intent.
            if not git_ok("merge-base", "--is-ancestor", commit, ref):
                continue
            # Byte-identical to the last pre-deletion copy => we never touched it,
            # so it is upstream's code, kept by accident. If it differs, we may
            # have edited it since and it needs a human.
            unchanged = git_ok("diff", "--quiet", f"{commit}^:{path}", "--", path)
            findings.append((path, commit, unchanged))
            break
    return findings


def audit_symbols(ref: str, exclusions) -> list[tuple[str, str]]:
    """Symbols in shared files that we define, `ref` does not, and `ref` once did."""
    ours = {p for p in git("ls-files").splitlines() if p.endswith(".py")}
    theirs = {
        p
        for p in git("ls-tree", "-r", "--name-only", ref).splitlines()
        if p.endswith(".py")
    }
    shared = sorted(ours & theirs)
    # One batched read of upstream's copies rather than a `git show` per file.
    specs = "\n".join(f"{ref}:{path}" for path in shared) + "\n"
    proc = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=REPO_ROOT,
        input=specs.encode(),
        capture_output=True,
    )
    their_sources: dict[str, str] = {}
    buf, pos, idx = proc.stdout, 0, 0
    while pos < len(buf) and idx < len(shared):
        nl = buf.find(b"\n", pos)
        if nl == -1:
            break
        parts = buf[pos:nl].decode(errors="replace").split()
        if len(parts) != 3:
            pos = nl + 1
            idx += 1
            continue
        size = int(parts[2])
        their_sources[shared[idx]] = buf[nl + 1 : nl + 1 + size].decode(
            "utf-8", errors="replace"
        )
        pos = nl + 1 + size + 1
        idx += 1

    findings = []
    for path in shared:
        local = REPO_ROOT / path
        if not local.exists():
            continue
        try:
            our_names = defined_names(local.read_text())
        except UnicodeDecodeError:
            continue
        their_names = defined_names(their_sources.get(path, ""))
        candidates = {
            name
            for name in our_names - their_names
            if not excused(exclusions, path, name)
        }
        for name in sorted(history_defines(ref, path, candidates)):
            findings.append((path, name))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-ref", default="upstream/main")
    parser.add_argument("--files-only", action="store_true")
    parser.add_argument("--symbols-only", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when anything is reported (for CI once the backlog is clear)",
    )
    args = parser.parse_args()
    ref = args.upstream_ref
    exclusions = load_exclusions()

    files = [] if args.symbols_only else audit_files(ref, exclusions)
    symbols = [] if args.files_only else audit_symbols(ref, exclusions)

    if files:
        print(f"FILES: {len(files)} file(s) we carry that {ref} deleted on purpose:\n")
        for path, commit, unchanged in files:
            verdict = (
                "byte-identical to the pre-deletion copy => STALE"
                if unchanged
                else "modified since the deletion => needs a read"
            )
            subject = git("log", "-1", "--format=%s", commit).strip()
            print(f"  {path}")
            print(f"      deleted by {commit[:10]}  {subject}")
            print(f"      {verdict}")
        print()
    elif not args.symbols_only:
        print(f"OK: no file we carry was deleted on purpose by {ref}.")

    if symbols:
        print(
            f"SYMBOLS: {len(symbols)} symbol(s) we define that {ref} once "
            f"defined in the same file and no longer does:\n"
        )
        current = None
        for path, name in symbols:
            if path != current:
                print(f"  {path}")
                current = path
            print(f"      - {name}")
        print(
            "\n  Lead generator, not an oracle: a name upstream once used can be\n"
            "  legitimate fork work today. Confirm each with\n"
            "      git log --oneline -S'def <name>' --all -- <path>\n"
            "  then classify FORK (keep, and add to .deletion-exclusions with a\n"
            "  reason) or STALE (delete ours)."
        )
    elif not args.files_only:
        print(f"OK: no symbol we define was previously defined and removed by {ref}.")

    if exclusions:
        print(f"\n{len(exclusions)} known deletion exclusion(s) vs {ref}.")

    return 1 if args.strict and (files or symbols) else 0


if __name__ == "__main__":
    sys.exit(main())
