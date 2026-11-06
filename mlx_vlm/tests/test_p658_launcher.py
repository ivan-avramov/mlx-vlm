"""P658 / P656 item 6: the quiet-box scan stops refusing the P475 orchestrator that launched it (fork-only, dev; CPU only).

Written from ``.p658-material/spec.md`` (Interface 7, acceptance 6) before the implementation.  The first line of each test body
is a ``# class:`` comment.  The pure tests drive ``orchestrate.foreign_process_problems`` with the ``(pid, argv)`` lists the
census builds; the topology tests build a REAL process chain (as ``test_p452_qualify_in_wrapper`` does) and run the real
preflight / ``qualify.interference_problems`` census in the grandchild.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import orchestrate  # noqa: E402

from mlx_vlm.tests.test_p452_qualify_in_wrapper import CHILD, LAUNCH  # noqa: E402

P475 = [
    "/venv/bin/python",
    "-m",
    "bench.p475_kv",
    "--arm",
    "TQ8",
    "--entry",
    "Qwen3.8-27B-mlx-uniform-4bit",
]
LEG = ["/venv/bin/python", "-m", "bench.p426_leg", "--axis", "m1"]
SEQ = ["/venv/bin/python", "-m", "bench.p426_wave1", "--stack-root", "/s"]


def test_the_launcher_modules_are_exactly_the_three_named():
    # class: exercises new behaviour
    assert tuple(orchestrate.LAUNCHER_MODULES) == (
        "bench.p426_leg",
        "bench.p426_wave1",
        "bench.p475_kv",
    )


def test_a_p475_kv_ancestor_is_not_foreign():
    # class: reproduces a failure the spec fixes
    assert (
        orchestrate.foreign_process_problems([(101, P475)], me=102, ancestors={101})
        == []
    )


def test_the_whole_launcher_chain_is_exempt_together():
    # class: reproduces a failure the spec fixes
    procs = [(100, P475), (101, SEQ), (103, LEG)]
    assert (
        orchestrate.foreign_process_problems(procs, me=102, ancestors={100, 101, 103})
        == []
    )


def test_a_p475_kv_non_ancestor_still_refuses():
    # class: preserves an invariant
    problems = orchestrate.foreign_process_problems(
        [(300, P475)], me=102, ancestors={101}
    )
    assert len(problems) == 1 and problems[0].startswith("process 300 ")
    assert "bench.p475_kv" in problems[0]


def test_an_ancestor_p475_kv_exempts_only_itself_not_a_same_argv_sibling():
    # class: reproduces a failure the spec fixes
    problems = orchestrate.foreign_process_problems(
        [(101, P475), (300, P475)], me=102, ancestors={101}
    )
    assert [p.split()[1] for p in problems] == ["300"]


@pytest.mark.parametrize(
    "module",
    [
        "bench.other",
        "bench.p475_other",
        "bench.p475",
        "bench.p475_kv_extra",
        "bench.p426_other",
        "bench",
    ],
)
def test_any_other_bench_ancestor_still_refuses(module):
    # class: preserves an invariant
    argv = ["/venv/bin/python", "-m", module]
    problems = orchestrate.foreign_process_problems(
        [(101, argv)], me=102, ancestors={101}
    )
    assert len(problems) == 1 and problems[0].startswith("process 101 "), module


def test_the_caller_itself_is_never_foreign_even_as_p475_kv():
    # class: preserves an invariant
    assert (
        orchestrate.foreign_process_problems([(102, P475)], me=102, ancestors=set())
        == []
    )


def test_unrelated_refused_processes_still_refuse_beside_a_p475_kv_ancestor():
    # class: reproduces a failure the spec fixes
    other = ["/venv/bin/python", "-m", "bench.something_else"]
    problems = orchestrate.foreign_process_problems(
        [(101, P475), (400, other)], me=102, ancestors={101}
    )
    assert [p.split()[1] for p in problems] == ["400"]


# ------------------------------------------------------------------ the real topology (acceptance 6, as P452's)
def _tree(tmp_path, parent_module, sibling_module):
    """``python -m bench.<parent_module>`` -> a grandchild running the REAL preflight census and ``interference_problems``,
    beside a SIBLING ``python -m bench.<sibling_module>`` that is not an ancestor."""
    pytest.importorskip(
        "psutil",
        reason="the preflight's process census needs psutil (fork-suite extra, P452)",
    )
    pkg = tmp_path / "bench"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("")
    launch = LAUNCH.format(child=CHILD.format(dev=str(DEV), fork=str(DEV.parent)))
    for name in {parent_module, sibling_module}:
        (pkg / f"{name}.py").write_text(launch)
    env = {"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    sibling = subprocess.Popen(
        [sys.executable, "-m", f"bench.{sibling_module}", "--sibling"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.DEVNULL,
    )
    time.sleep(0.5)
    try:
        out = subprocess.run(
            [sys.executable, "-m", f"bench.{parent_module}"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
    finally:
        sibling.kill()
        sibling.wait()
    return json.loads(out), sibling.pid


def _pids(problems):
    return [int(p.split()[1]) for p in problems]


def test_real_topology_a_p475_kv_parent_is_exempt_a_p475_kv_sibling_is_not(tmp_path):
    # class: reproduces a failure the spec fixes
    doc, sibling = _tree(tmp_path, "p475_kv", "p475_kv")
    assert doc["parent"] not in _pids(doc["problems"])  # the preflight census
    assert doc["parent"] not in _pids(
        doc["interference"]
    )  # the per-entry census: the same rule
    assert sibling in _pids(doc["problems"])  # same argv, not an ancestor
    assert sibling in _pids(doc["interference"])


def test_real_topology_another_bench_parent_still_refuses_beside_a_p475_kv_sibling(
    tmp_path,
):
    # class: preserves an invariant
    doc, sibling = _tree(tmp_path, "p475_other", "p475_kv")
    assert doc["parent"] in _pids(doc["problems"])
    assert doc["parent"] in _pids(doc["interference"])
    assert sibling in _pids(doc["problems"])


def test_real_topology_the_p426_launcher_is_still_exempt(tmp_path):
    # class: preserves an invariant
    doc, sibling = _tree(tmp_path, "p426_leg", "p475_kv")
    assert doc["parent"] not in _pids(doc["problems"])
    assert sibling in _pids(doc["problems"])
