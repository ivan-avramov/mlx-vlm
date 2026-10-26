"""Fork: dev/ci.py, the local stand-in for hosted CI.

A permissive bug in a gate fails nothing and still prints OK, so pin the parts
that decide what runs and what fails.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("local_ci", ROOT / "dev" / "ci.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass resolves its module by name
    spec.loader.exec_module(module)
    return module


ci = _load()


def test_every_dev_audit_is_wired_into_the_runner():
    on_disk = {p.name for p in (ROOT / "dev").glob("check_*.py")}
    assert {script for script, _ in ci.AUDITS} == on_disk


def test_every_dev_lead_generator_is_wired_into_the_runner():
    on_disk = {p.name for p in (ROOT / "dev").glob("find_*.py")}
    assert {script for script, _flag, _extra in ci.LEADS} == on_disk


def test_the_deletions_audit_runs_strict():
    extra = dict(ci.AUDITS)["check_upstream_deletions.py"]
    assert "--strict" in extra


def test_the_default_run_is_every_gating_stage():
    assert (
        set(ci.DEFAULT_STAGES) == set(ci.GATING_STAGES) == {"style", "audits", "suite"}
    )


@pytest.mark.parametrize(
    "results, code",
    [
        ([("audits", 0), ("suite", 0)], 0),
        ([("audits", 0), ("suite", 1)], 1),
        ([("style", 1)], 1),
        ([("audits", 0), ("drift", 1), ("leads", 1)], 0),
        ([("drift", 1), ("audits", 1)], 1),
        ([], 0),
    ],
)
def test_only_a_gating_failure_fails_the_run(results, code):
    steps = [ci.StepResult(stage, "x", rc, 0.0) for stage, rc in results]
    assert ci.exit_code(steps) == code


def test_sync_ref_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "ref"
    path.write_text("# comment\n\n  abc123  # trailing\nignored\n")
    assert ci.load_sync_ref(path) == "abc123"


def test_an_empty_sync_ref_is_an_error(tmp_path):
    path = tmp_path / "ref"
    path.write_text("# only a comment\n\n")
    with pytest.raises(ValueError):
        ci.load_sync_ref(path)


def test_the_recorded_sync_ref_is_merged_into_head():
    ref = ci.load_sync_ref()
    merged = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ref, "HEAD"], cwd=ROOT
    ).returncode
    assert merged == 0, f".upstream-sync-ref {ref} is not an ancestor of HEAD"
