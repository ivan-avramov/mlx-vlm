"""P369 v13 (V5 / V10) + C203: the ONE state generator / classifier for the stack registry (fork-only, CPU).

Each MTP-certified entry's MTP-OFF form is an ATOMIC substitution of the pinned golden fixture (C177 for the two Qwen3.8-27B
picks, P406 / C191 for `Ornith-1.0-35B-mlx-uniform-4bit` and `Qwen3.6-27B-Opus-Distill-OptiQ-4bit`). C203: the live registry is
classified on its SETTINGS — equal to the fixture except that each of the four entries has all or none of its MTP keys (the 16
states); comments never refuse; any settings change, duplicate key, CR line ending or unparseable text does. `Campaign.prepare`, the overlays and the
fork's opt-in live-registry test all use this module.
"""

import itertools
import sys
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import registry_states as rs  # noqa: E402

FIXTURE = rs.fixture_text(Path(__file__).resolve().parent / "fixtures")
ENTRIES = tuple(rs.ENTRY_OFF)


def test_the_four_entries_and_the_pinned_fixture():
    assert ENTRIES == (
        "Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed",
        "Qwen3.8-27B-mlx-uniform-4bit",
        "Ornith-1.0-35B-mlx-uniform-4bit",
        "Qwen3.6-27B-Opus-Distill-OptiQ-4bit",
    )
    assert [len(rs.ENTRY_OFF[e]) for e in ENTRIES] == [3, 2, 2, 2]


@pytest.mark.parametrize(
    "off",
    [frozenset(c) for k in range(5) for c in itertools.combinations(ENTRIES, k)],
    ids=lambda off: "+".join(sorted(e.split("-")[0] + e[-6:] for e in off)) or "on",
)
def test_every_one_of_the_16_states_round_trips(off):
    text = rs.apply(FIXTURE, off)
    assert rs.classify(text, FIXTURE) == off
    for (
        entry
    ) in ENTRIES:  # each entry can be restored independently, whatever the others are
        on, restored = rs.overlay_on(text, FIXTURE, entry)
        assert rs.classify(on, FIXTURE) == off - {entry}
        assert rs.block(on, entry) == rs.block(FIXTURE, entry)
        want = [line for line, _ in rs.ENTRY_OFF[entry]] if entry in off else []
        assert restored == want


def test_todays_live_registry_shape_is_accepted():
    """C177 for both Qwen3.8-27B picks + C191 for the other two: all four OFF."""
    assert rs.classify(rs.apply(FIXTURE, frozenset(ENTRIES)), FIXTURE) == frozenset(
        ENTRIES
    )


def _half(text):
    entry = "Qwen3.6-27B-Opus-Distill-OptiQ-4bit"
    old, new = rs.ENTRY_OFF[entry][1]  # draft_model back ON, draft_kind still OFF
    return text.replace(new, old, 1)


ORNITH_OFF = rs.ENTRY_OFF["Ornith-1.0-35B-mlx-uniform-4bit"][0][1]
COMMENT_ONLY = [  # C203: comments never refuse — the state is read from the SETTINGS
    (
        "a changed P406 marker",
        lambda t: t.replace("(C191 2026-10-20)", "(C191 2026-10-21)", 1),
    ),
    (
        "a changed commented drafter path",
        lambda t: t.replace(
            "# draft_model: caslca/Ornith-1.0-35B-mlx-uniform-4bit-mtp-drafter",
            "# draft_model: caslca/x",
            1,
        ),
    ),
    ("another comment line", lambda t: t + "# edited\n"),
    ("trailing whitespace", lambda t: t.replace(ORNITH_OFF, ORNITH_OFF + " ", 1)),
    (
        "a CERTIFIED note in generation_defaults",
        lambda t: t.replace(
            "  - name: Qwen3.8-27B-mlx-uniform-4bit\n",
            "  - name: Qwen3.8-27B-mlx-uniform-4bit\n    # CERTIFIED P426 2026-10-30 (example)\n",
            1,
        ),
    ),
    ("no trailing newline", lambda t: t.rstrip("\n")),
]


@pytest.mark.parametrize(("label", "change"), COMMENT_ONLY)
def test_c203_a_comment_only_edit_keeps_the_state(label, change):
    off = frozenset(ENTRIES)
    text = change(rs.apply(FIXTURE, off))
    assert rs.classify(text, FIXTURE) == off, label
    for (
        entry
    ) in (
        ENTRIES
    ):  # the overlay still restores exactly that entry, to the fixture's settings
        on, restored = rs.overlay_on(text, FIXTURE, entry)
        assert rs.classify(on, FIXTURE) == off - {entry}
        assert restored == [line for line, _ in rs.ENTRY_OFF[entry]]
        assert rs.settings(on)[entry] == rs.settings(FIXTURE)[entry]


def _setting(old, new):
    return lambda t: t.replace(old, new, 1)


SETTINGS = [
    ("half an atomic block", _half),
    (
        "a changed sampling setting",
        _setting("      temperature: 0.4", "      temperature: 0.41"),
    ),
    (
        "an uncommented foreign drafter",
        _setting(
            "    # draft_model: caslca/Ornith-1.0-35B-mlx-uniform-4bit-mtp-drafter",
            "    draft_kind: mtp\n    draft_model: caslca/x",
        ),
    ),
    (
        "a duplicate key",
        _setting(
            "  - name: Qwen3.8-27B-mlx-uniform-4bit\n",
            "  - name: Qwen3.8-27B-mlx-uniform-4bit\n    hf_path: caslca/x\n",
        ),
    ),
    ("CRLF", lambda t: t.replace("\n", "\r\n")),
    ("not YAML", lambda t: t + "\n  : : [\n"),
]


@pytest.mark.parametrize(("label", "change"), SETTINGS)
def test_c203_a_settings_edit_refuses(label, change):
    text = change(rs.apply(FIXTURE, frozenset(ENTRIES)))
    assert text != rs.apply(
        FIXTURE, frozenset(ENTRIES)
    ), f"{label}: the edit did not apply"
    assert rs.classify(text, FIXTURE) is None, label
    with pytest.raises(ValueError, match="not one of the 16"):
        rs.overlay_on(text, FIXTURE, "Qwen3.8-27B-mlx-uniform-4bit")


def test_c203_duplicate_identical_marker_comments_never_block_the_overlay():
    """B4 code round 2 F3: comments are inert — two identical commented ``draft_kind: mtp`` markers: the first is restored,
    the other stays a comment, and the settings are the fixture's."""
    off = frozenset(ENTRIES)
    lines = rs.apply(FIXTURE, off).split("\n")
    start, end = rs._span(lines, "Ornith-1.0-35B-mlx-uniform-4bit")
    i = next(i for i in range(start, end) if lines[i] == ORNITH_OFF)
    lines.insert(i + 1, ORNITH_OFF)
    text = "\n".join(lines)
    assert rs.classify(text, FIXTURE) == off  # comments only
    on, _ = rs.overlay_on(text, FIXTURE, "Ornith-1.0-35B-mlx-uniform-4bit")
    assert (
        rs.settings(on)["Ornith-1.0-35B-mlx-uniform-4bit"]
        == rs.settings(FIXTURE)["Ornith-1.0-35B-mlx-uniform-4bit"]
    )
    assert ORNITH_OFF in rs.block(on, "Ornith-1.0-35B-mlx-uniform-4bit")


def test_the_fixture_is_pinned_by_sha():
    with pytest.raises(ValueError, match="sha256"):
        rs.check_fixture(FIXTURE + "# edited\n")
    rs.check_fixture(FIXTURE)


# ------------------------------------------------------------------ B4 code round 1 (F2–F4)
ALL_OFF = rs.apply(FIXTURE, frozenset(ENTRIES))
ORNITH = "Ornith-1.0-35B-mlx-uniform-4bit"


@pytest.mark.parametrize(
    "old, new",
    [
        (
            "    kv_bits: 0\n",
            "    kv_bits: false\n",
        ),  # F2: 0 == False in Python, not in YAML
        ("    lazy_prompt_embeddings: true\n", "    lazy_prompt_embeddings: 1\n"),
    ],
)
def test_f2_a_type_change_refuses(old, new):
    assert ALL_OFF.count(old) >= 1
    assert rs.classify(ALL_OFF.replace(old, new, 1), FIXTURE) is None


def test_f2_a_lexically_different_equal_float_is_accepted():
    text = ALL_OFF.replace("      temperature: 0.4\n", "      temperature: 0.40\n", 1)
    assert text != ALL_OFF and rs.classify(text, FIXTURE) == frozenset(ENTRIES)


@pytest.mark.parametrize(
    "name_line",
    [f"  - name: {ORNITH}  # a note", f'  - name: "{ORNITH}"', f"  - name: '{ORNITH}'"],
)
def test_f3_a_commented_or_quoted_name_line_keeps_the_overlay_working(name_line):
    text = ALL_OFF.replace(f"  - name: {ORNITH}\n", name_line + "\n", 1)
    assert rs.classify(text, FIXTURE) == frozenset(ENTRIES)
    on, _ = rs.overlay_on(text, FIXTURE, ORNITH)
    assert rs.classify(on, FIXTURE) == frozenset(ENTRIES) - {ORNITH}
    assert rs.settings(on)[ORNITH] == rs.settings(FIXTURE)[ORNITH]


def test_f3_a_removed_marker_is_inserted_and_a_historical_note_is_never_touched():
    marker = rs.ENTRY_OFF[ORNITH][0][
        1
    ]  # the commented draft_kind: mtp line (shared verbatim with Qwen3.6's block)
    lines = ALL_OFF.split("\n")
    start, end = rs._span(lines, ORNITH)
    hit = [i for i in range(start, end) if lines[i] == marker]
    assert len(hit) == 1
    del lines[hit[0]]
    text = "\n".join(lines)
    assert rs.classify(text, FIXTURE) == frozenset(ENTRIES)  # a comment removal
    on, _ = rs.overlay_on(text, FIXTURE, ORNITH)
    assert "    # draft_kind: suffix  # suffix WITHDRAWN 2026-08-16" in rs.block(
        on, ORNITH
    )
    assert rs.settings(on)[ORNITH] == rs.settings(FIXTURE)[ORNITH]
    assert rs.classify(on, FIXTURE) == frozenset(ENTRIES) - {ORNITH}


def test_f4_merge_keys_resolve_and_only_explicit_duplicates_refuse():
    doc = rs.parse("a: &x {k: 1}\nb:\n  <<: *x\n  j: 2\nc:\n  <<: *x\n  k: 3\n")
    assert doc == {"a": {"k": 1}, "b": {"k": 1, "j": 2}, "c": {"k": 3}}
    with pytest.raises(ValueError, match="duplicate"):
        rs.parse("a:\n  k: 1\n  k: 2\n")


def test_f3_round2_a_reordered_name_key_keeps_the_overlay_working():
    """B4 code round 2 F3: the entry's span comes from the YAML structure, not from ``name`` being the first key."""
    lines = ALL_OFF.split("\n")
    start, _ = rs._span(lines, ORNITH)
    assert lines[start] == f"  - name: {ORNITH}" and lines[start + 1].startswith(
        "    type:"
    )
    lines[start], lines[start + 1] = (
        "  - " + lines[start + 1].strip(),
        f"    name: {ORNITH}",
    )
    text = "\n".join(lines)
    assert rs.classify(text, FIXTURE) == frozenset(ENTRIES)
    on, _ = rs.overlay_on(text, FIXTURE, ORNITH)
    assert rs.classify(on, FIXTURE) == frozenset(ENTRIES) - {ORNITH}
    assert rs.settings(on)[ORNITH] == rs.settings(FIXTURE)[ORNITH]


def test_i1_a_merge_only_model_item_keeps_the_overlay_working():
    """B4 code round 3 I1: a models item written as ``- <<:`` + its own mapping is settings-identical, so it classifies — and
    the overlay must find it (the name comes from the merged mapping) and restore inside that mapping.
    """
    lines = ALL_OFF.split("\n")
    start, end = rs._span(lines, ORNITH)
    item = [f"      {lines[start][4:]}"] + [
        ("  " + line) if line.startswith("    ") else line
        for line in lines[start + 1 : end]
    ]
    text = "\n".join(lines[:start] + ["  - <<:"] + item + lines[end:])
    assert rs.classify(text, FIXTURE) == frozenset(ENTRIES)
    on, _ = rs.overlay_on(text, FIXTURE, ORNITH)
    assert rs.classify(on, FIXTURE) == frozenset(ENTRIES) - {ORNITH}
    assert rs.settings(on)[ORNITH] == rs.settings(FIXTURE)[ORNITH]
