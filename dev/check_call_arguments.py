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

## The second measure: arity, for the call that has no names to miss

A keyword check is blind to a purely positional call -- `f(a, b, c)` losing `c` has
no name to be missing. So each (definition, callee) pair also compares **total
supplied argument count**: positional (excluding `*x`) plus distinct keyword names,
maxed over the definition's call sites.

**Total, not positional.** Moving an argument between positional and keyword form
changes nothing and the fork does it freely, so `f(a, b)` against `f(a, b=x)` must
not report -- and a positional-only count reports it every time.

Two rules keep it honest, both derived from measuring the tree rather than guessed:

* **The deferred-call idiom is skipped.** When our call passes a `lambda`, the real
  arguments live inside the lambda body where no count can see them:
  `asyncio.to_thread(lambda: gen(a, b, c))` against upstream's
  `asyncio.to_thread(gen, a, b, c)` is the same call, and `chat_completions_endpoint`
  really does this. Without the rule that site reports forever; with it, it does not.
* **Arity is reported only when the keyword check found nothing** for that pair. A
  missing name already implies a lower count, so emitting both would double-count one
  defect and inflate the arity baseline.

An arity hit is a **pointer to a diff, not a conclusion** -- being a count, it cannot
say *which* argument went. The fork extracting upstream's inline expression into a
fork-only helper produces one legitimately (the `arange` baseline entry), which is why
this measure needs its reasons written down rather than being treated as proof.

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


class CalleeUse:
    """How one definition calls one callee, unioned over all its call sites.

    `kwargs`  -- every keyword NAME passed at any site.
    `arity`   -- the largest total supplied-argument count at any site: positional
                 (excluding `*x`) plus distinct keyword names. **Total, not positional**,
                 so moving an argument between positional and keyword form -- which the
                 fork does freely and which changes nothing -- cannot report.
    `star`    -- some site forwards `*x` or `**x`, so both measures are lower bounds.
    `defers`  -- some site passes a `lambda`, i.e. the deferred-call idiom, where the
                 real arguments live inside the lambda body and no count can see them.
    """

    __slots__ = ("kwargs", "arity", "star", "defers")

    def __init__(self) -> None:
        self.kwargs: set[str] = set()
        self.arity = 0
        self.star = False
        self.defers = False

    def observe(self, node: ast.Call) -> None:
        positional = sum(0 if isinstance(a, ast.Starred) else 1 for a in node.args)
        named = {kw.arg for kw in node.keywords if kw.arg is not None}
        self.kwargs |= named
        self.arity = max(self.arity, positional + len(named))
        if any(isinstance(a, ast.Starred) for a in node.args) or any(
            kw.arg is None for kw in node.keywords
        ):
            self.star = True
        if any(isinstance(a, ast.Lambda) for a in node.args):
            self.defers = True

    def merge(self, other: "CalleeUse") -> None:
        self.kwargs |= other.kwargs
        self.arity = max(self.arity, other.arity)
        self.star = self.star or other.star
        self.defers = self.defers or other.defers


class CallSites:
    """Per-definition map {callee: CalleeUse} for one file.

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
        self.by_definition: dict[str, dict[str, CalleeUse]] = {}
        module_level: list[ast.stmt] = []
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.by_definition[node.name] = self._collect(node)
            else:
                module_level.append(node)
        if module_level:
            merged: dict[str, CalleeUse] = {}
            for stmt in module_level:
                for callee, use in self._collect(stmt).items():
                    merged.setdefault(callee, CalleeUse()).merge(use)
            self.by_definition[MODULE_SCOPE] = merged

    @staticmethod
    def _collect(defn: ast.AST) -> dict[str, CalleeUse]:
        calls: dict[str, CalleeUse] = {}
        for node in ast.walk(defn):
            if not isinstance(node, ast.Call):
                continue
            name = callee_name(node.func)
            if name is None:
                continue
            calls.setdefault(name, CalleeUse()).observe(node)
        return calls


class Finding:
    """One call whose arguments are a strict subset of upstream's, in one of two ways.

    `kind="kwargs"` -- a keyword NAME upstream passes and we never do. The founding
    instance, and the sharper of the two: a name is unambiguous.

    `kind="arity"` -- we supply FEWER TOTAL ARGUMENTS than upstream at every site. It
    catches what a name cannot -- an argument dropped from a purely positional call,
    where there is no name to be missing -- at the cost of being a count, so it cannot
    see *which* argument went. Treat it as a pointer to a diff, not as a conclusion.
    """

    __slots__ = ("path", "definition", "callee", "kind", "missing", "counts", "star")

    def __init__(
        self,
        path: str,
        definition: str,
        callee: str,
        kind: str,
        star: bool,
        missing: set[str] | None = None,
        counts: tuple[int, int] | None = None,
    ) -> None:
        self.path = path
        self.definition = definition
        self.callee = callee
        self.kind = kind
        self.star = star
        self.missing = missing or set()
        self.counts = counts

    @property
    def key(self) -> str:
        return f"{self.path}::{self.definition}::{self.callee}"

    def describe(self) -> str:
        tag = " [**]" if self.star else ""
        if self.kind == "arity":
            up, ours = self.counts or (0, 0)
            return (
                f"{self.definition} -> {self.callee}(...) "
                f"[arity] upstream passes {up} args, we pass {ours}{tag}"
            )
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


def compare_sites(path: str, up: CallSites, ours: CallSites) -> list[Finding]:
    """The whole comparison, with no git or filesystem in it.

    Split out from `compare_file` so the tests exercise THIS rather than a copy of it.
    `test_call_argument_check.py` originally reimplemented this loop, which meant a bug
    in the real one could pass every test -- the exact "a check that lies is worse than
    no check" failure the test file exists to prevent, one level up.
    """
    findings: list[Finding] = []
    for definition, up_calls in up.by_definition.items():
        our_calls = ours.by_definition.get(definition)
        if our_calls is None:
            continue  # not a shared definition -- check_upstream_symbols.py's job
        for callee, up_use in up_calls.items():
            our_use = our_calls.get(callee)
            if our_use is None:
                continue  # a dropped CALL, not a dropped argument
            missing = up_use.kwargs - our_use.kwargs
            if missing:
                findings.append(
                    Finding(
                        path,
                        definition,
                        callee,
                        "kwargs",
                        our_use.star,
                        missing=missing,
                    )
                )
            # Arity is reported only when the keyword check found nothing: a missing
            # NAME already implies a lower count, and emitting both would double-count
            # one defect and make the arity baseline look worse than it is.
            elif our_use.arity < up_use.arity and not our_use.defers:
                findings.append(
                    Finding(
                        path,
                        definition,
                        callee,
                        "arity",
                        our_use.star,
                        counts=(up_use.arity, our_use.arity),
                    )
                )
    return findings


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
    return compare_sites(path, up, ours), None


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
            print("  no argument upstream passes is missing here.")
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
        print(
            "  A hit is a keyword name upstream passes and we never do, or a\n"
            "  call where we supply fewer total arguments than upstream does."
        )
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
        print(f"OK: no call passes fewer arguments than upstream's, vs {ref}.")
        return 0

    print(
        f"{len(all_findings)} call(s) pass fewer arguments than upstream's, "
        f"vs {ref}:",
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
        "  `[arity]` means we supply fewer TOTAL arguments; being a count it cannot\n"
        "  say WHICH one went, so read the diff rather than the number.\n"
        "  Establish provenance first: git log -S'<name>=' -- <path>",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
