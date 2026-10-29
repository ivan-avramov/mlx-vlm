"""P542 (C254): the implementer's own tests for the spec v1.1 rulings the independent tests do not pin. Pure (CPU only)."""

import sys
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import orchestrate, qualify, registry_states  # noqa: E402

FIXTURE = registry_states.fixture_text(Path(__file__).resolve().parent / "fixtures")
PICK_A = "Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed"
UNIFORM = "Qwen3.8-27B-mlx-uniform-4bit"


def test_the_derived_off_registry_changes_only_the_entrys_mtp_key_lines():
    got = qualify.mtp_off_registry(FIXTURE, PICK_A)
    import difflib

    old, new = FIXTURE.split("\n"), got.split("\n")
    diff = [
        l
        for l in difflib.ndiff(old, new)
        if l[0] in "+-"  # ndiff: a pure removal is "- ", nothing is added or edited
    ]
    assert all(l.startswith("- ") for l in diff)
    assert sorted(l[2:].split(":")[0].strip() for l in diff) == [
        "draft_kind",
        "draft_model",
        "mtp_verify_scan",
    ]


def test_an_entry_without_mtp_keys_derives_to_the_same_bytes():
    off = registry_states.apply(FIXTURE, {PICK_A})
    assert qualify.mtp_off_registry(off, PICK_A) == off


def test_summary_arguments_rules():
    names = [PICK_A, UNIFORM]
    assert qualify.summary_arguments([f"{PICK_A}=/a=b/s.json"], names) == (
        {PICK_A: "/a=b/s.json"},  # split on the FIRST "="
        [],
    )
    assert qualify.summary_arguments(["/p/s.json"], [PICK_A]) == (
        {PICK_A: "/p/s.json"},
        [],
    )
    assert qualify.summary_arguments(["/p/s.json"], names)[
        1
    ]  # a bare path, two entries
    assert qualify.summary_arguments(["Other=/p/s.json"], names)[1]  # not an --entry
    assert qualify.summary_arguments([f"{PICK_A}=/a", f"{PICK_A}=/b"], names)[1]


@pytest.mark.parametrize("value", ["b26fa7", "", None, 7, "B26FA7C9", "zzzzzzzz"])
def test_the_summarys_production_must_be_a_lowercase_hex_ref_of_at_least_seven(value):
    full = "b26fa7c9" + "0" * 32
    resolve = (
        lambda ref: full
    )  # noqa: E731  (every ref "resolves": only the shape is under test)
    assert not qualify._production_matches(value, full, resolve)
    assert qualify._production_matches("b26fa7c", full, resolve)


def test_an_unresolvable_or_other_production_never_matches():
    full = "b26fa7c9" + "0" * 32

    def refuse(ref):
        raise ValueError(ref)

    assert not qualify._production_matches("b26fa7c9", full, refuse)
    assert not qualify._production_matches("b26fa7c9", full, lambda ref: "1" * 40)


def test_safe_ids_are_one_clean_component_with_the_name_digest():
    import hashlib

    for name in ("org/Model", "/abs/M", "../M", "..", "a b"):
        got = qualify.safe_id(name)
        assert "/" not in got and got.endswith(
            hashlib.sha256(name.encode()).hexdigest()[:8]
        )
    assert qualify.safe_id("a/b") != qualify.safe_id("a_b")


def test_a_restore_refusal_names_the_unclassifiable_registry():
    with pytest.raises(ValueError, match="not one of the 16 accepted states"):
        qualify.arm_for(
            PICK_A,
            "on",
            registry_entry=registry_states.settings(
                registry_states.apply(FIXTURE, {PICK_A})
            )[PICK_A],
            registry_state=None,
        )


def test_default_entries_follow_registry_order():
    assert qualify.default_entries(["x", UNIFORM, "y", PICK_A]) == [UNIFORM, PICK_A]
    assert set(qualify.default_entries(list(registry_states.settings(FIXTURE)))) == set(
        orchestrate.PICKS
    )


def test_an_mtp_off_failure_names_the_key_and_its_one_based_line():
    flow = FIXTURE.rstrip("\n") + (
        "\n  - {name: Flow-9B, type: vision, draft_kind: mtp, draft_model: x/y}\n"
    )
    with pytest.raises(ValueError, match=r"Flow-9B: draft_kind \(line \d+\)"):
        qualify.mtp_off_registry(flow, "Flow-9B")
    line = flow.split("\n").index(flow.split("\n")[-2]) + 1
    with pytest.raises(ValueError, match=rf"line {line}\b"):
        qualify.mtp_off_registry(flow, "Flow-9B")


# ================================================================== spec v1.4 (the verify pass)
from mlx_vlm.tests import test_p542_qualify_any as ind  # noqa: E402


@pytest.fixture
def rig(tmp_path, monkeypatch):
    return ind._Rig(tmp_path, monkeypatch)


def test_v14_a_summary_name_may_contain_a_slash():
    names = ["org/Model-X", UNIFORM]
    got = qualify.summary_arguments(["org/Model-X=/p/s.json"], names)
    assert got == ({"org/Model-X": "/p/s.json"}, [])
    # not an --entry name: a path with a "/" before the "=" is a bare path (one entry only)
    assert qualify.summary_arguments(["/p/a=b.json"], ["only"]) == (
        {"only": "/p/a=b.json"},
        [],
    )
    assert qualify.summary_arguments(["/p/a=b.json"], names)[1]


def test_v14_the_argv_binding_refuses_before_the_run(rig, monkeypatch):
    made, preflighted = [], []

    class Monitor(ind.t415._NullMonitor):
        def __init__(self, *a, **k):
            made.append(1)
            super().__init__(*a, **k)

    monkeypatch.setattr(orchestrate, "Monitor", Monitor)
    rig.hooks["preflight"] = lambda: preflighted.append(1)
    argv = ind._set_flag(rig.certified_argv(PICK_A), "--kv-bits", "4")
    got = rig.run(
        "--entry", PICK_A, "--p369-summary", rig.summary(PICK_A, argv_on=argv)
    )
    assert isinstance(got.refused, str) and got.refused.startswith("refused: [")
    assert f"{PICK_A}: {ind.CERTIFIED} (--kv-bits)" in got.refused
    assert got.report is None and rig.jobs == [] and made == [] and preflighted == []


def test_v14_the_certified_argv_is_read_once_and_a_missing_one_has_its_message(
    rig, monkeypatch
):
    real, calls = qualify.certified_argv, []
    monkeypatch.setattr(
        qualify, "certified_argv", lambda *a: calls.append(a) or real(*a)
    )
    got = rig.run("--entry", PICK_A, "--p369-summary", rig.summary(PICK_A))
    assert got.completed(PICK_A), got.text()
    assert len(calls) == 1
    got = rig.run(
        "--entry", PICK_A, "--p369-summary", rig.summary(PICK_A, argv_on=None)
    )
    assert f"{PICK_A}: its P369 manifest.json has no picks[{PICK_A}].argv_on" in str(
        got.refused
    )
    assert got.report is None and rig.jobs == []


def test_v14_merge_inherited_mtp_keys_refuse_naming_key_and_source_line():
    text = (
        "x-mtp: &mtp\n  draft_kind: mtp\n  draft_model: example/d\n"
        "models:\n  - name: M\n    type: vision\n    <<: *mtp\n"
    )
    with pytest.raises(ValueError, match=r"M: draft_kind \(line 2\).*merge"):
        qualify.mtp_off_registry(text, "M")


def test_v14_a_null_integer_value_refuses_the_entry_without_a_traceback(rig):
    block = ind.CANDIDATE_BLOCK.replace(
        "max_kv_cache_size: 131072", "max_kv_cache_size: null"
    )
    rig.registry(ind._with(FIXTURE, block))
    got = rig.run("--entry", ind.CANDIDATE)  # a TypeError would propagate out of run()
    assert isinstance(got.refused, str) and f"{ind.CANDIDATE}:" in got.refused
    assert rig.jobs == []
