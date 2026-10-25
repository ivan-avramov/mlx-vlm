"""P426 B7: `qualify.py --predictor {on,off}` — the M1 memory axis measures an MTP-certified entry in BOTH predictor states.

Without the flag the tool keeps C199's shipped arm (MTP-ON for the certified entries, the others as served). `off` serves the
COMMITTED registry and requires it to hold the entry MTP-OFF (one of the 16 states with the entry OFF); `on` is the C199 overlay.
A served (non-MTP) entry has no ON arm. Pure / mocked (CPU only).
"""

import sys
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import qualify  # noqa: E402

from mlx_vlm.tests import test_p415_qualify as t415  # noqa: E402

FABLE, STATIC, QWEN36 = t415.FABLE, t415.STATIC, t415.QWEN36
FABLE_OFF_ENTRY = {
    k: v
    for k, v in t415.FABLE_ENTRY.items()
    if k not in ("draft_kind", "draft_model", "mtp_verify_scan")
}


@pytest.mark.parametrize(
    "name, predictor, arm",
    [
        (FABLE, None, "mtp_on"),
        (FABLE, "on", "mtp_on"),
        (FABLE, "off", "mtp_off"),
        (QWEN36, None, "mtp_on_gated"),
        (QWEN36, "on", "mtp_on_gated"),
        (QWEN36, "off", "mtp_off"),
        (STATIC, None, "served"),
        (STATIC, "off", "served"),
    ],
)
def test_the_arm_follows_the_predictor(monkeypatch, name, predictor, arm):
    # P476: STATIC is the scoped SYNTHETIC served entry (as in test_p415_qualify), not a registry name.  # allow-shorthand
    t415._with_served(monkeypatch)
    assert qualify.arm_for(name, predictor) == arm


def test_a_served_entry_has_no_on_arm(monkeypatch):
    t415._with_served(monkeypatch)
    with pytest.raises(ValueError, match="no MTP-ON arm"):
        qualify.arm_for(STATIC, "on")


def test_the_off_arm_needs_no_p369_gate():
    assert qualify.entry_problems([QWEN36], p369_summary=None, predictor="off") == []
    assert qualify.entry_problems([QWEN36], p369_summary=None, predictor="on")


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


def _drive_off(tmp_path, monkeypatch, off_set):
    """t415._drive with Fable MTP-OFF in the committed registry (classify → ``off_set``) and ``--predictor off``."""
    from p369_phase2 import orchestrate, registry_states

    overlays = []
    real_drive = qualify.drive

    def drive(args):
        args.predictor = "off"
        monkeypatch.setattr(registry_states, "classify", lambda text, fx: off_set)
        monkeypatch.setattr(
            orchestrate,
            "overlay_registry",
            lambda text, name, fixture: overlays.append(name) or (text, True),
        )
        return real_drive(args)

    monkeypatch.setattr(qualify, "drive", drive)
    on_cmd = t415._cmd(t415.FABLE_ENTRY)
    drafter_flags = {"--mtp-verify-scan", "--draft-kind", "--draft-model"}
    off_cmd = [
        a
        for i, a in enumerate(on_cmd)
        if a not in drafter_flags and (i == 0 or on_cmd[i - 1] not in drafter_flags)
    ]
    real_cmd = t415._cmd
    monkeypatch.setattr(
        t415, "_cmd", lambda e: off_cmd if e is FABLE_OFF_ENTRY else real_cmd(e)
    )
    monkeypatch.setattr(t415, "FABLE_ENTRY", FABLE_OFF_ENTRY)

    def run_job(job, *a, **k):
        rec = t415._record(drafter_live_checks=None)
        return dict(requests=[dict(qualify=rec)])

    code, report = t415._drive(tmp_path, monkeypatch, run_job, names=(FABLE,))
    return code, report, overlays


def test_the_off_arm_serves_the_committed_registry(tmp_path, monkeypatch):
    code, report, overlays = _drive_off(tmp_path, monkeypatch, {FABLE})
    assert report["entries"][FABLE]["arm"] == "mtp_off"
    assert overlays == []  # no MTP-ON overlay was built
    assert report.get("predictor") == "off"


def test_the_off_arm_refuses_a_registry_serving_the_entry_mtp_on(tmp_path, monkeypatch):
    code, report, overlays = _drive_off(tmp_path, monkeypatch, set())
    assert code != 0
    assert report["entries"][FABLE]["status"] != "COMPLETED"
    assert any(
        "serves" in p and "MTP-ON" in p
        for p in report["entries"][FABLE].get("problems", [])
    ), report["entries"][FABLE]
