"""P369 v13 (V5 / V10): the ONE byte-state generator for the stack registry (fork-only, CPU).

Each MTP-certified entry's MTP-OFF form is an ATOMIC raw-byte substitution of the pinned golden fixture (C177 for the two
Qwen3.8-27B picks, P406 / C191 for `Ornith-1.0-35B-mlx-uniform-4bit` and `Qwen3.6-27B-Opus-Distill-OptiQ-4bit`). The accepted
live registries are exactly the 2**4 = 16 combinations; anything else refuses. `Campaign.prepare`, the overlays and the
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


@pytest.mark.parametrize(
    ("label", "change"),
    [
        ("half an atomic block", _half),
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
        (
            "a duplicate marker line",
            lambda t: t.replace(
                rs.ENTRY_OFF["Ornith-1.0-35B-mlx-uniform-4bit"][0][1],
                rs.ENTRY_OFF["Ornith-1.0-35B-mlx-uniform-4bit"][0][1]
                + "\n"
                + rs.ENTRY_OFF["Ornith-1.0-35B-mlx-uniform-4bit"][0][1],
                1,
            ),
        ),
        ("another edit", lambda t: t + "# edited\n"),
        ("CRLF", lambda t: t.replace("\n", "\r\n")),
        (
            "trailing whitespace",
            lambda t: t.replace(
                rs.ENTRY_OFF["Ornith-1.0-35B-mlx-uniform-4bit"][0][1],
                rs.ENTRY_OFF["Ornith-1.0-35B-mlx-uniform-4bit"][0][1] + " ",
                1,
            ),
        ),
    ],
)
def test_anything_else_refuses(label, change):
    text = change(rs.apply(FIXTURE, frozenset(ENTRIES)))
    assert rs.classify(text, FIXTURE) is None, label
    with pytest.raises(ValueError, match="not one of the 16"):
        rs.overlay_on(text, FIXTURE, "Qwen3.8-27B-mlx-uniform-4bit")


def test_the_fixture_is_pinned_by_sha():
    with pytest.raises(ValueError, match="sha256"):
        rs.check_fixture(FIXTURE + "# edited\n")
    rs.check_fixture(FIXTURE)
