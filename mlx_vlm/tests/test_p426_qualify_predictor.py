"""P426 B7: `qualify.py --predictor {on,off}` — the M1 memory axis measures an MTP-certified entry in BOTH predictor states.

Without the flag the tool serves each entry exactly as the committed registry does. P542 (C254) re-derives the arms from the
committed block: `off` removes an MTP-ON entry's MTP keys in a derived registry (`mtp_off`) and serves an OFF entry as
committed (`served_off`); `on` serves an MTP-ON entry as committed (`served_on`) or restores an OFF one through the 16-state
generator (`mtp_on_restored`), and an entry with no drafter to restore refuses. Pure / mocked (CPU only).
"""

import sys
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import qualify  # noqa: E402

from mlx_vlm.tests import test_p415_qualify as t415  # noqa: E402

PICK_A, SYNTH_S, PICK_C = t415.PICK_A, t415.SYNTH_S, t415.PICK_C
PICK_A_OFF_ENTRY = {
    k: v for k, v in t415.PICK_A_ENTRY.items() if k not in qualify.MTP_KEYS
}


@pytest.mark.parametrize(
    "entry, restorable, predictor, arm",
    [
        (t415.PICK_A_ENTRY, None, None, "served_on"),
        (t415.PICK_A_ENTRY, None, "on", "served_on"),
        (t415.PICK_A_ENTRY, None, "off", "mtp_off"),
        (PICK_A_OFF_ENTRY, frozenset({PICK_A}), None, "served_off"),
        (PICK_A_OFF_ENTRY, frozenset({PICK_A}), "on", "mtp_on_restored"),
        (PICK_A_OFF_ENTRY, frozenset({PICK_A}), "off", "served_off"),
        (t415.SYNTH_S_ENTRY, None, None, "served_off"),
        (t415.SYNTH_S_ENTRY, None, "off", "served_off"),
    ],
)
def test_the_arm_follows_the_predictor(entry, restorable, predictor, arm):
    """P542 (was: a lookup of the removed ENTRIES table with arms ``mtp_on`` / ``mtp_on_gated`` / ``served``): the arm is
    derived from the committed block; ``registry_state`` only decides a restore."""
    got = qualify.arm_for(
        entry["name"], predictor, registry_entry=entry, registry_state=restorable
    )
    assert got == arm


def test_a_served_entry_without_a_drafter_to_restore_has_no_on_arm():
    """P542 (was ``no MTP-ON arm (not MTP-certified)``): the refusal text is ``no MTP drafter to restore``."""
    with pytest.raises(ValueError, match="no MTP drafter to restore"):
        qualify.arm_for(
            SYNTH_S, "on", registry_entry=t415.SYNTH_S_ENTRY, registry_state=None
        )


def test_the_off_arm_needs_no_p369_gate(tmp_path):
    assert t415._problems([PICK_C], tmp_path, predictor="off") == []
    assert t415._problems([PICK_C], tmp_path, predictor="on")


def test_the_cli_takes_on_or_off_only(monkeypatch):
    seen = {}
    monkeypatch.setattr(qualify, "drive", lambda args: seen.setdefault("a", args) and 0)
    base = [
        "--stack-root",
        "s",
        "--fork-root",
        "f",
        "--out",
        "o",
        "--reviewed-head",
        "h",
    ]
    with pytest.raises(SystemExit):
        qualify.main(base + ["--predictor", "off"])
    assert seen["a"].predictor == "off"
    with pytest.raises(SystemExit):
        qualify.main(base + ["--predictor", "maybe"])


def _run_job(job, *a, **k):
    return dict(requests=[dict(qualify=t415._record(drafter_live_checks=None))])


def test_the_off_arm_serves_a_derived_registry_without_the_mtp_keys(
    tmp_path, monkeypatch
):
    """P542 (was: ``--predictor off`` serves the committed registry and refuses one that serves the entry MTP-ON): an
    entry committed MTP-ON gets a derived registry, its sha256 recorded; no 16-state overlay is built.
    """
    from p369_phase2 import orchestrate

    overlays = []
    monkeypatch.setattr(
        orchestrate,
        "overlay_registry",
        lambda text, name, fixture: overlays.append(name) or (text, True),
    )
    code, report = t415._drive(
        tmp_path, monkeypatch, _run_job, names=(PICK_A,), predictor="off"
    )
    state = report["entries"][PICK_A]
    assert state["arm"] == "mtp_off" and state["status"] == "COMPLETED", state
    assert len(state["overlay_sha256"]) == 64
    assert overlays == [] and report.get("predictor") == "off"
    derived = (
        tmp_path / "run" / f"mtp-off-{qualify.safe_id(PICK_A)}.yaml"
    )  # v1.2 safe id
    assert derived.exists() and "draft_kind" not in derived.read_text()


def test_the_off_arm_serves_an_entry_committed_off_as_committed(tmp_path, monkeypatch):
    """P542 (replaces: ``--predictor off`` refuses a registry serving the entry MTP-ON — the guard is removed)."""
    code, report = t415._drive(
        tmp_path, monkeypatch, _run_job, names=(SYNTH_S,), predictor="off"
    )
    assert report["entries"][SYNTH_S]["arm"] == "served_off"
    assert "overlay_sha256" not in report["entries"][SYNTH_S]
