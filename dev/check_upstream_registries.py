#!/usr/bin/env python3
"""Fail when a registry entry, re-export or class attribute in upstream/main is absent.

The sixth audit direction, and it closes the blind spot `AGENTS.md` has named for
months without a check behind it:

    Registries are a known blind spot. MODEL_REMAPPING, prompt-format maps, drafter
    registries, tool parsers and __init__.py re-exports are all invisible to both
    audits -- parity only sees missing files, the symbol check only sees missing
    def/class names. A dropped dict entry or re-export passes both.

That is exactly right, and it applies to every one of the other five checks. Walk
through what each sees and a dropped `MODEL_REMAPPING["unlimited-ocr"]` survives all
of them: the file is present (parity), no `def`/`class` name vanished (symbols),
nothing was deleted (deletions), the hunk lives in a file whose fork sites are all
marked (fork markers), and the entry is not a helper anyone calls (dead helpers). The
model implementation sits in the tree byte-identical to upstream and the family just
fails with "Model type X not supported".

Four shapes, all of them `ast.Assign` / `ast.AnnAssign` / `ast.Import*` rather than a
definition, and all four are known to have cost real losses here:

  1. REGISTRY ENTRY  -- a key or element missing from a module-level container
                        literal both trees assign to the same name. Four
                        `MODEL_REMAPPING` entries were lost this way, taking
                        "unlimited-ocr" and "inkling_mm_model" with them.
  2. REGISTRY        -- a whole module-level container upstream assigns and we do
                        not. Not a `def`, so `check_upstream_symbols.py` is blind.
  3. RE-EXPORT       -- a name an upstream import statement binds that ours does not.
                        `models/gemma4/__init__.py`'s `Gemma4VideoProcessor` was lost
                        exactly here; a package `__init__.py` is nothing but this.
  4. CLASS ATTRIBUTE -- a class-level assignment upstream has and we lack. Dataclass
                        fields are the case that matters: every `ModelConfig` in
                        `models/*/config.py` is one, and a dropped field silently
                        changes model behaviour rather than raising.

Shape 4 is worth spelling out because `check_upstream_symbols.py`'s own docstring
claims to catch it -- it lists `deepseek_v4/config.py -- ModelConfig.index_block /
.index_keep` among its verified instances. It cannot. That script collects
`FunctionDef`/`AsyncFunctionDef`/`ClassDef` names, and a dataclass field is an
`AnnAssign`; `ast.walk` never yields it as a name. Those two fields are present today,
so they were restored -- but not by that check, and its docstring has been corrected.

WHAT IT DOES NOT DO

Only *presence* of a key, never its value. `{"a": 1}` versus `{"a": 2}` passes, and
that is deliberate: values legitimately diverge all over this fork (every tuned
constant), while a missing key is almost always a loss. `#1402`'s test-certified
cache-key bug is the shape this cannot see, and `check_body_divergence.py --file` is
what shows it.

Comparison is per-file and per-name: a registry has to exist under the same name in
our copy of the same file to have its entries compared. A registry the fork *renamed*
therefore reports as shape 2, which is correct -- a rename is exactly the event that
loses entries in a merge.

Reads the git **index**, like the other five audits, so it can gate a commit. Uses
`git cat-file --batch` for the same reason `check_upstream_symbols.py` does: one
`git show` per file over ~1150 files takes minutes and this has to run in CI.

Usage:
    python dev/check_upstream_registries.py [--upstream-ref upstream/main]
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUSIONS_FILE = REPO_ROOT / ".registry-exclusions"

# How a finding is spelled in .registry-exclusions. The symbol field is
# `<container>::<entry>` for shape 1 and `<Class>::<attr>` for shape 4, so one glob
# can excuse a whole container (`MODEL_REMAPPING::*`) or one entry.
KIND_REGISTRY_ENTRY = "registry-entry"
KIND_REGISTRY = "registry"
KIND_REEXPORT = "re-export"
KIND_CLASS_ATTR = "class-attribute"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout


def read_blobs(ref: str, paths: list[str]) -> dict[str, str]:
    """Read many blobs from `ref` in one `git cat-file --batch` pass.

    Lifted from check_upstream_symbols.py deliberately rather than imported: these
    scripts are standalone by convention so any one of them can be run, copied or
    rewritten without dragging the others along.
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
        result[path] = out[newline + 1 : newline + 1 + size].decode(
            "utf-8", errors="replace"
        )
        pos = newline + 1 + size + 1  # trailing newline after the blob
    return result


def container_entries(node: ast.AST) -> set[str] | None:
    """Identity set for a container literal, or None if `node` is not one.

    Constants are keyed by `repr` so `"4"` and `4` cannot collide -- a string key and
    an int key are different entries and a registry keyed by both is not unusual.
    Non-constant keys/elements fall back to their unparsed source, which is what makes
    `__all__`-style lists and name-keyed dicts comparable at all.

    Unwraps a single-argument call so `frozenset([...])`, `dict(...)`, `tuple(...)`
    and `OrderedDict([...])` are all read as their literal argument. Without that,
    every registry built through a constructor is silently skipped -- and skipping is
    the failure mode these scripts exist to prevent.
    """
    if isinstance(node, ast.Dict):
        return {
            # `**other` in a dict literal contributes unknown keys. Recorded as a
            # sentinel rather than ignored, so a registry that MERGES another is
            # visibly not fully comparable instead of looking exhaustively checked.
            "**" if key is None else _entry_key(key)
            for key in node.keys
        }
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        return {_entry_key(element) for element in node.elts}
    if isinstance(node, ast.Call) and len(node.args) == 1:
        return container_entries(node.args[0])
    return None


def _entry_key(node: ast.AST) -> str:
    if isinstance(node, ast.Constant):
        return repr(node.value)
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover -- ast.unparse is total for parsed trees
        return "<unparseable>"


def module_registries(tree: ast.Module) -> dict[str, set[str]]:
    """{name: entries} for every module-level container-literal assignment.

    Module level only. A container built inside a function is local state, not a
    registry, and including them turned the signal to noise in the prototype.
    """
    out: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        entries = container_entries(value)
        if entries is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                out[target.id] = entries
    return out


def bound_import_names(tree: ast.Module) -> set[str]:
    """Names any import statement binds into the module namespace.

    `import a.b` binds `a`, `import a.b as c` binds `c`, `from x import y as z` binds
    `z`. `from x import *` binds an unknowable set and is skipped -- there is nothing
    to compare, and pretending otherwise would report every starred module.
    """
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    continue
                out.add(alias.asname or alias.name.split(".")[0])
    return out


def class_attributes(tree: ast.Module) -> dict[str, set[str]]:
    """{class name: attribute names} for class-level assignments, at any nesting.

    Keyed by bare class name, and same-named classes merge. That is a deliberate
    loosening: nested and conditionally defined classes are common here, and the
    alternative -- a dotted path -- makes an exclusion unwritable for a class whose
    enclosing scope the fork restructured.
    """
    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        names: set[str] = set()
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign):
                if isinstance(statement.target, ast.Name):
                    names.add(statement.target.id)
            elif isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
        if names:
            out.setdefault(node.name, set()).update(names)
    return out


def findings_for_file(
    path: str, upstream_source: str, our_source: str
) -> list[tuple[str, str, str]]:
    """[(kind, symbol, detail)] for one file. Raises SyntaxError to the caller."""
    upstream, ours = ast.parse(upstream_source), ast.parse(our_source)
    out: list[tuple[str, str, str]] = []

    up_registries, our_registries = module_registries(upstream), module_registries(ours)
    for name in sorted(set(up_registries) - set(our_registries)):
        out.append((KIND_REGISTRY, name, "upstream assigns this container, we do not"))
    for name in sorted(set(up_registries) & set(our_registries)):
        for entry in sorted(up_registries[name] - our_registries[name]):
            out.append((KIND_REGISTRY_ENTRY, f"{name}::{entry}", "entry missing"))

    for name in sorted(bound_import_names(upstream) - bound_import_names(ours)):
        out.append((KIND_REEXPORT, name, "upstream imports this name, we do not"))

    up_attrs, our_attrs = class_attributes(upstream), class_attributes(ours)
    for cls in sorted(set(up_attrs) & set(our_attrs)):
        for attr in sorted(up_attrs[cls] - our_attrs[cls]):
            out.append((KIND_CLASS_ATTR, f"{cls}::{attr}", "class attribute missing"))
    # A class upstream has and we do not is check_upstream_symbols.py's job, so it is
    # deliberately not reported here -- two checks reporting one event is how the
    # symbol/deletion pair used to leave both halves of a rename unconnected.

    return out


def load_exclusions() -> list[tuple[str, str, str]]:
    """Return [(path_glob, symbol_glob, reason)] from .registry-exclusions.

    Format: `path_glob::symbol_glob  # reason`, matching `.symbol-exclusions`. Note
    the symbol field for shapes 1 and 4 itself contains `::`, so the split is on the
    FIRST `::` only -- `mlx_vlm/utils.py::MODEL_REMAPPING::'foo'` is one path and one
    symbol, not three fields.
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
    # Index, not HEAD, for the same reason as the other five audits.
    our_files = {p for p in git("ls-files").splitlines() if p.endswith(".py")}
    shared = sorted(upstream_files & our_files)
    upstream_sources = read_blobs(ref, shared)

    exclusions = load_exclusions()
    findings: list[tuple[str, str, str, str]] = []  # (path, kind, symbol, detail)
    excused = 0
    used: set[tuple[str, str]] = set()
    unparseable: list[str] = []

    for path in shared:
        upstream_source = upstream_sources.get(path)
        local = REPO_ROOT / path
        if upstream_source is None or not local.exists():  # staged deletion
            continue
        try:
            file_findings = findings_for_file(path, upstream_source, local.read_text())
        except SyntaxError:
            unparseable.append(path)
            continue
        for kind, symbol, detail in file_findings:
            hit = matching_exclusion(path, symbol, exclusions)
            if hit is not None:
                excused += 1
                used.add(hit)
            else:
                findings.append((path, kind, symbol, detail))

    if excused:
        print(f"{excused} known registry exclusion(s) vs {ref}.")

    stale = [f"{p}::{s}" for p, s, _r in exclusions if (p, s) not in used]
    if stale:
        print(
            f"\nwarning: {len(stale)} exclusion(s) in {EXCLUSIONS_FILE.name} no "
            f"longer excuse a finding and should be pruned:"
        )
        for entry in stale:
            print(f"      - {entry}")

    # Loud and fatal: a file this cannot parse is a file it is not checking.
    if unparseable:
        print(
            f"\nerror: {len(unparseable)} file(s) could not be parsed on one side:",
            file=sys.stderr,
        )
        for path in unparseable:
            print(f"      - {path}", file=sys.stderr)
        return 1

    if findings:
        print(
            f"\nerror: {len(findings)} registry entr(ies), re-export(s) or class "
            f"attribute(s) in {ref} are missing from our copy of the same file:",
            file=sys.stderr,
        )
        current = None
        for path, kind, symbol, detail in findings:
            if path != current:
                print(f"  {path}", file=sys.stderr)
                current = path
            print(f"      - [{kind}] {symbol} — {detail}", file=sys.stderr)
        print(
            f"\nEither port them, or add to {EXCLUSIONS_FILE.name} as\n"
            "    path/glob.py::symbol_glob  # why it is intentionally absent\n"
            "Establish provenance first (`git log -S`): a missing registry entry is "
            "usually a\ndropped hunk, but it can also be an entry upstream itself "
            "later removed.",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: no upstream registry entry, re-export or class attribute silently "
        f"missing vs {ref}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
