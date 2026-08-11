#!/usr/bin/env python3
"""The eighth direction: is the call passing everything upstream passes?

Every other audit in `dev/` keys on the *existence* of something -- a file
(`check_upstream_parity.py`), a `def`/`class` (`check_upstream_symbols.py`), a
deletion (`check_upstream_deletions.py`), a hunk (`check_fork_markers.py`), a
module-level assignment or re-export (`check_upstream_registries.py`), a body
(`check_body_divergence.py`), or a caller (`check_dead_helpers.py`).

None of them can see a **dropped keyword argument**, and that shape is a live-bug
generator:

    upstream adds a parameter AND passes it at N call sites -> our merge applies
    the callee's file byte-identically and applies M < N of the call sites ->
    the parameter exists, is keyword-only with a default, is documented, is
    unit-tested through the callee, and is silently never supplied

The instance this script was written for: `generate/dispatch.py::stream_generate`
called `stream_diffusion_generate_from_kwargs(...)` without upstream's
`skip_special_tokens=` and `verbose=`. Both are keyword-only with `= False`
defaults, and `stream_generate` *pops* `skip_special_tokens` out of `kwargs`
first, so `**kwargs` did not rescue it either. Effect: `--skip-special-tokens`
was a no-op on every diffusion model, since that flag reaches the
`decode_generated()` used for every streamed token batch. All seven gating audits
were green, the full suite was green (3001 passed), and the site carried a
`# Fork:` marker asserting "everything else upstream does here is still done".

    `check_dead_helpers.py` is the near miss and worth understanding: it asks
    whether an upstream-called helper is reachable here. This helper WAS
    reachable -- it had a caller. "Called" and "called correctly" are different
    questions, and only this script asks the second.

## What it compares, and the asymmetry that makes it usable

For each `def`/`class` **both trees define** in a **shared file**, plus the file's
module scope (filed under `<module>`), it collects every call inside it, keyed by
callee simple name (`f(...)` -> `f`, `a.b.f(...)` -> `f`), and unions the keyword
names passed to that callee across all its call sites in that definition. It then
reports where **ours is a strict subset of upstream's**.

Reading it asymmetrically is the whole design, and it is the same rule
`find_untested_fork_code.py` states for its `tests` column:

* `ours < upstream` -- the only direction worth chasing. Upstream passes a name
  we never pass anywhere in that definition.
* `ours > upstream` -- a fork addition. Silent by construction; `_make_cache`
  passing `prealloc_tokens=` to `cache.BatchKVCache()` must not report.
* names, never **values** -- exactly `check_upstream_registries.py`'s rule.
  `make_cache=_make_cache` vs `make_cache=functools.partial(_make_cache, ...)`
  is a fork enhancement, not a drop, and comparing values would report it. A
  wrong value is `check_body_divergence.py --file`'s job.

Positional arity is deliberately **not** compared. The fork reorders and rewrites
call sites freely, `*args` forwarding is common, and every experiment produced
noise without finding anything a keyword comparison missed.

## `**kwargs` forwarding, and why those hits are tagged rather than dropped

If our call forwards `**something`, a name we do not pass explicitly may still
arrive through the dict, so the hit is genuinely ambiguous -- it is tagged `[**]`
in the report. They are still reported and still gate. Dropping them would have
been the wrong call: the bug above is *adjacent* to that shape (the callee takes a
`kwargs` dict positionally), and a check that goes quiet where the answer is hard
is the failure mode this directory exists to prevent. Tag, review, exclude with a
reason -- do not filter.

Reviewed hits go in `.call-argument-exclusions`, format
`path_glob::definition_glob::callee_glob  # reason`, three fields rather than the
usual two because one definition can call many helpers and excusing a whole
definition would be too coarse.

Reads the git **index**, like the other gating audits, so it can gate a commit.

    python dev/check_call_arguments.py                  # the gate
    python dev/check_call_arguments.py --file <path>     # one file, verbose
    python dev/check_call_arguments.py --summary         # rank files by hits
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUSIONS_FILE = REPO_ROOT / ".call-argument-exclusions"

# The pseudo-definition module-scope calls are filed under, so an exclusion can name
# them. Not a legal Python identifier, so it cannot collide with a real definition;
# `path::*::callee` still covers it, which is intended. Same trick as
# check_body_divergence.py's `<file>`.
MODULE_SCOPE = "<module>"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout


def callee_name(node: ast.expr) -> str | None:
    """Simple name of a call target: `f` -> 'f', `a.b.f` -> 'f', else None.

    Matching on the simple name is intentional. The fork routinely changes how a
    helper is *reached* -- `from .common import maybe_quantize_kv_cache` here
    against `common.maybe_quantize_kv_cache` upstream, or a module alias like
    `_apc.` -- while calling the same function. Keying on the dotted path would
    make every one of those a spurious "callee missing" and bury the real hits.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


class CallSites:
    """Per-definition map {callee: (kwarg names, uses ** forwarding)} for one file.

    Module-scope calls are collected too, under the pseudo-definition
    `MODULE_SCOPE` -- `check_body_divergence.py`'s `<file>` convention, and not a legal
    Python identifier so it cannot collide with a real name. They were omitted in the
    first draft, which would have left a real hole: a registration or middleware call
    at module scope (`app.add_middleware(...)`) is exactly the shape that loses a
    keyword in a merge, and `check_body_divergence.py`'s `gone` cannot see module scope
    either. Measured before adding: it reports ZERO hits across the tree, so the
    coverage was free.
    """

    def __init__(self, source: str) -> None:
        self.tree = ast.parse(source)
        self.by_definition: dict[str, dict[str, tuple[set[str], bool]]] = {}
        module_level: list[ast.stmt] = []
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.by_definition[node.name] = self._collect(node)
            else:
                module_level.append(node)
        if module_level:
            merged: dict[str, tuple[set[str], bool]] = {}
            for stmt in module_level:
                for callee, (names, star) in self._collect(stmt).items():
                    have, had_star = merged.get(callee, (set(), False))
                    merged[callee] = (have | names, had_star or star)
            self.by_definition[MODULE_SCOPE] = merged

    @staticmethod
    def _collect(defn: ast.AST) -> dict[str, tuple[set[str], bool]]:
        calls: dict[str, tuple[set[str], bool]] = {}
        for node in ast.walk(defn):
            if not isinstance(node, ast.Call):
                continue
            name = callee_name(node.func)
            if name is None:
                continue
            names, star = calls.get(name, (set(), False))
            for kw in node.keywords:
                if kw.arg is None:  # **something
                    star = True
                else:
                    names.add(kw.arg)
            calls[name] = (names, star)
        return calls


class Finding:
    __slots__ = ("path", "definition", "callee", "missing", "star")

    def __init__(
        self,
        path: str,
        definition: str,
        callee: str,
        missing: set[str],
        star: bool,
    ) -> None:
        self.path = path
        self.definition = definition
        self.callee = callee
        self.missing = missing
        self.star = star

    @property
    def key(self) -> str:
        return f"{self.path}::{self.definition}::{self.callee}"

    def describe(self) -> str:
        tag = " [**]" if self.star else ""
        names = ", ".join(sorted(self.missing))
        return f"{self.definition} -> {self.callee}({names}=...){tag}"


def load_exclusions() -> list[tuple[str, str, str, str]]:
    """[(path_glob, definition_glob, callee_glob, reason)] from the exclusions file.

    Validated here (the only reader), so a malformed or reasonless line is a hard
    error rather than a silently ignored one -- the same contract as every other
    audit's exclusions file.
    """
    if not EXCLUSIONS_FILE.exists():
        return []

    out: list[tuple[str, str, str, str]] = []
    for lineno, raw in enumerate(EXCLUSIONS_FILE.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rule, _, reason = line.partition("#")
        rule, reason = rule.strip(), reason.strip()
        if not rule:
            continue
        parts = rule.split("::")
        if len(parts) != 3:
            sys.exit(
                f"{EXCLUSIONS_FILE.name}:{lineno}: expected "
                f"'path_glob::definition_glob::callee_glob', got {rule!r}"
            )
        if not reason:
            sys.exit(
                f"{EXCLUSIONS_FILE.name}:{lineno}: exclusion {rule!r} has no "
                f"'# reason' comment. Every exclusion must say why."
            )
        out.append((parts[0].strip(), parts[1].strip(), parts[2].strip(), reason))
    return out


def is_excused(finding: Finding, exclusions: list[tuple[str, str, str, str]]) -> bool:
    for path_glob, def_glob, callee_glob, _reason in exclusions:
        if (
            fnmatch.fnmatch(finding.path, path_glob)
            and fnmatch.fnmatch(finding.definition, def_glob)
            and fnmatch.fnmatch(finding.callee, callee_glob)
        ):
            return True
    return False


def shared_python_files(ref: str) -> list[str]:
    """Shared .py paths whose index copy DIFFERS from `ref`.

    Scoped to diverged files on purpose, and it is not a coverage compromise: a
    file byte-identical to upstream cannot have a call passing fewer keyword
    arguments than upstream's. It is also the difference between a ~13s check and
    a >5min one, since each file costs two `git show` invocations.
    """
    upstream = {
        p
        for p in git("ls-tree", "-r", "--name-only", ref).splitlines()
        if p.endswith(".py")
    }
    ours = {p for p in git("ls-files").splitlines() if p.endswith(".py")}
    diverged = {
        parts[2]
        for line in git("diff", "--cached", "--numstat", ref).splitlines()
        if len(parts := line.split("\t")) == 3 and parts[2].endswith(".py")
    }
    return sorted(upstream & ours & diverged)


def compare_file(path: str, ref: str) -> tuple[list[Finding], str | None]:
    """(findings, error). A read or parse failure is returned, never swallowed."""
    try:
        upstream_source = git("show", f"{ref}:{path}")
        our_source = git("show", f":{path}")
    except subprocess.CalledProcessError:
        return [], f"{path}: unreadable from index or {ref}"
    if upstream_source == our_source:
        return [], None
    try:
        up = CallSites(upstream_source)
        ours = CallSites(our_source)
    except SyntaxError as exc:
        return [], f"{path}: unparseable ({exc})"

    findings: list[Finding] = []
    for definition, up_calls in up.by_definition.items():
        our_calls = ours.by_definition.get(definition)
        if our_calls is None:
            continue  # not a shared definition -- check_upstream_symbols.py's job
        for callee, (up_names, _up_star) in up_calls.items():
            if callee not in our_calls:
                continue  # a dropped CALL, not a dropped argument
            our_names, our_star = our_calls[callee]
            missing = up_names - our_names
            if missing:
                findings.append(Finding(path, definition, callee, missing, our_star))
    return findings, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream-ref", default="upstream/main")
    ap.add_argument("--file", help="report one path in detail, and do not gate")
    ap.add_argument(
        "--summary", action="store_true", help="rank files by hit count, do not gate"
    )
    args = ap.parse_args()

    ref = args.upstream_ref
    exclusions = load_exclusions()
    paths = [args.file] if args.file else shared_python_files(ref)

    all_findings: list[Finding] = []
    excused = 0
    errors: list[str] = []
    for path in paths:
        findings, error = compare_file(path, ref)
        if error:
            errors.append(error)
            continue
        for finding in findings:
            if is_excused(finding, exclusions):
                excused += 1
            else:
                all_findings.append(finding)

    if args.file:
        print(f"{args.file}  vs {ref}")
        if not all_findings and not errors:
            print("  no keyword argument upstream passes is missing here.")
        for finding in all_findings:
            print(f"  {finding.describe()}")
        for error in errors:
            print(f"  ERROR: {error}")
        return 1 if errors else 0

    if args.summary:
        counts: dict[str, int] = {}
        for finding in all_findings:
            counts[finding.path] = counts.get(finding.path, 0) + 1
        print(
            f"{len(all_findings)} unexcused hit(s) across {len(counts)} file(s), "
            f"vs {ref}."
        )
        print("  A hit is a keyword name upstream passes and we never do.")
        for path, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {count:5d}  {path}")
        for error in errors:
            print(f"  ERROR: {error}")
        return 0

    if excused:
        print(f"{excused} known call-argument exclusion(s) vs {ref}.")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if not all_findings:
        print(f"OK: no call passes fewer keyword arguments than upstream's, vs {ref}.")
        return 0

    print(
        f"{len(all_findings)} call(s) pass fewer keyword arguments than "
        f"upstream's, vs {ref}:",
        file=sys.stderr,
    )
    by_path: dict[str, list[Finding]] = {}
    for finding in all_findings:
        by_path.setdefault(finding.path, []).append(finding)
    for path, findings in sorted(by_path.items()):
        print(f"\n  {path}", file=sys.stderr)
        for finding in sorted(findings, key=lambda f: (f.definition, f.callee)):
            print(f"    {finding.describe()}", file=sys.stderr)
    print(
        "\n  Each is either content a merge resolution dropped (restore it) or a\n"
        "  deliberate fork divergence (add it to .call-argument-exclusions with a\n"
        "  reason). `[**]` means our call forwards **kwargs, so the name may still\n"
        "  arrive through the dict -- confirm before excusing, and say which.\n"
        "  Establish provenance first: git log -S'<name>=' -- <path>",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
