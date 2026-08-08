#!/usr/bin/env python3
"""Find upstream commits whose content is MISSING despite being merged.

The failure mode this repo keeps hitting (see docs/upstream-gaps.md) is not
"we are behind upstream" -- `git log HEAD..upstream/main` is empty. It is that a
merge resolution dropped a hunk from a commit that IS an ancestor of `main`, so
git will never re-offer it and no audit notices.

File-level divergence (`git diff --numstat upstream/main`) cannot distinguish
that from legitimate fork work, which is why triaging by line-count ratio keeps
producing wrong calls. This works at commit granularity instead:

  for each file that diverges from upstream:
      missing := upstream lines absent from our copy
      for each upstream commit that touched the file:
          added := lines that commit introduced
          if a large share of `added` is in `missing`: that commit is dropped

Attribution by line content rather than by blame is deliberate: blame follows
our side of history, which is exactly the side that lost the content.

Output is ranked by how much of a commit is missing. It is a *lead generator* --
every hit still needs `git log -S` and a read, because a commit whose content
upstream itself later replaced will also show up here (the "stale upstream code
retained" shape), and that wants deleting, not restoring.
"""

import argparse
import collections
import subprocess
import sys

TRIVIAL = {
    "",
    "{",
    "}",
    "(",
    ")",
    "):",
    "else:",
    "try:",
    "return",
    "pass",
    "continue",
    "break",
    '"""',
}


def sh(*args: str) -> str:
    r = subprocess.run(args, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def significant(line: str) -> bool:
    s = line.strip()
    if s in TRIVIAL or len(s) < 6:
        return False
    # Comment-only and decorator lines move around too much to attribute.
    return not s.startswith(("#", "@"))


def added_lines(commit: str, path: str) -> set[str]:
    out = sh("git", "show", "--format=", "--unified=0", commit, "--", path)
    return {
        l[1:].strip()
        for l in out.splitlines()
        if l.startswith("+") and not l.startswith("+++") and significant(l[1:])
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="upstream/main")
    ap.add_argument("--min-lines", type=int, default=3,
                    help="ignore commits with fewer than N missing lines")
    ap.add_argument("--min-share", type=float, default=0.5,
                    help="flag when this share of a commit's added lines is missing")
    ap.add_argument("--max-commits", type=int, default=80,
                    help="most recent N upstream commits per file")
    args = ap.parse_args()

    numstat = sh("git", "diff", "--numstat", args.ref, "--", "*.py")
    files = []
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            if int(parts[1]) > 0:  # upstream has lines we lack
                files.append(parts[2])
    print(f"scanning {len(files)} diverged file(s) against {args.ref}\n", file=sys.stderr)

    hits = collections.defaultdict(list)  # commit -> [(path, n_missing, share)]
    subjects: dict[str, str] = {}

    for path in files:
        theirs = sh("git", "show", f"{args.ref}:{path}")
        if not theirs:
            continue
        try:
            ours = open(path, encoding="utf-8").read()
        except OSError:
            ours = ""
        ours_set = {l.strip() for l in ours.splitlines()}
        missing = {l.strip() for l in theirs.splitlines() if significant(l)} - ours_set
        if not missing:
            continue

        commits = sh(
            "git", "log", f"--max-count={args.max_commits}", "--format=%H",
            args.ref, "--", path,
        ).split()
        for c in commits:
            added = added_lines(c, path)
            if not added:
                continue
            gone = added & missing
            share = len(gone) / len(added)
            if len(gone) >= args.min_lines and share >= args.min_share:
                hits[c].append((path, len(gone), share))
                if c not in subjects:
                    subjects[c] = sh("git", "log", "-1", "--format=%s", c).strip()

    if not hits:
        print("No dropped upstream content detected.")
        return 0

    ranked = sorted(hits.items(), key=lambda kv: -sum(n for _, n, _ in kv[1]))
    print(f"{len(ranked)} upstream commit(s) with content missing here:\n")
    for c, entries in ranked:
        total = sum(n for _, n, _ in entries)
        anc = sh("git", "merge-base", "--is-ancestor", c, "HEAD")
        merged = subprocess.run(
            ["git", "merge-base", "--is-ancestor", c, "HEAD"]
        ).returncode == 0
        tag = "MERGED-THEN-DROPPED" if merged else "never merged"
        print(f"{c[:10]}  {total:>5} lines missing  [{tag}]  {subjects.get(c,'')[:70]}")
        for path, n, share in sorted(entries, key=lambda e: -e[1]):
            print(f"{'':12}  {n:>5} ({share:.0%} of its hunk)  {path}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
