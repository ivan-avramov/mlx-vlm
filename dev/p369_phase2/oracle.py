"""P369 phase 2 E7 oracle, host side (Q30): run the scripted cases in isolated checkouts and derive defects.

``run(repo, commit, cases, workdir)`` extracts ``mlx_vlm`` of ``commit`` (``git archive``, no git metadata)
and runs ``oracle_cases.py`` on it in a fresh ``python -I`` interpreter (only that tree on ``sys.path``).
``defects(result)`` turns one case's ON / OFF observables into the pre-registered defect record:

- ``committed_past_stop``: ON cache offset − OFF cache offset when positive (M4a);
- ``terminal_missing``: OFF − ON when positive (M4b: the terminal token is not in ON's cache);
- ``overshoot``: index of ON's first forced token − OFF's (M3), and ``forced_sequence_cut`` when ON's
  window of ``len(FORCED)`` tokens from its first forced token differs from OFF's (or ON has none);
- ``next_turn_diverged``, ``emission_differs``.

Digests are informational: on the dense float32 fixture MTP and plain state are not bit-identical (P390).
"""

import json
import subprocess
import sys
from pathlib import Path

CASES = [
    dict(name="first_draft", where="first_draft", n=12),
    dict(name="second_draft", where="second_draft", n=12),
    dict(name="bonus", where="bonus", n=12),
    dict(name="first_token", where="first_token", n=12),
    dict(name="length", n=7),
    dict(name="budget_first_draft", budget_at="first_draft"),
    dict(name="budget_second_draft", budget_at="second_draft"),
    dict(name="budget_bonus", budget_at="bonus"),
]
FORCED = (63, 62)  # the scripted fixtures' newline and closer ids (oracle_cases.py)
PRE_FIX = "88724023"  # = 0b43dae4^ (tests only on top of the base)
FIX = "0b43dae4"  # P369 M4 + M3


def extract(repo, commit, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", commit, "mlx_vlm"],
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
    return dest


def run(repo, commit, cases, workdir, *, timeout=900):
    workdir = Path(workdir)
    tree = extract(repo, commit, workdir / f"tree-{commit}")
    spec = workdir / f"cases-{commit}.json"
    spec.write_text(json.dumps(cases))
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            str(Path(__file__).with_name("oracle_cases.py")),
            str(tree),
            str(spec),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(tree),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"oracle run on {commit} failed:\n{proc.stderr[-4000:]}")
    return {r["name"]: r for r in json.loads(proc.stdout)}


def _first_forced(emitted):
    hits = [i for i, t in enumerate(emitted) if t in FORCED]
    return hits[0] if hits else None


def defects(result):
    if "error" in result:
        raise ValueError(f"{result['name']}: {result['error']}")
    on, off = result["on"], result["off"]
    delta = (on["offset"] or 0) - (off["offset"] or 0)
    out = dict(
        committed_past_stop=max(delta, 0),
        terminal_missing=max(-delta, 0),
        next_turn_diverged=on["next"] != off["next"],
        emission_differs=on["emitted"] != off["emitted"],
        overshoot=None,
        forced_sequence_cut=False,
    )
    a, b = _first_forced(on["emitted"]), _first_forced(off["emitted"])
    if b is not None:
        out["overshoot"] = None if a is None else a - b
        width = len(FORCED)
        out["forced_sequence_cut"] = (
            a is None or on["emitted"][a : a + width] != off["emitted"][b : b + width]
        )
    return out
