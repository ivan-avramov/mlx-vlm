"""P369 v13 (V5 / V10): the ONE byte-state generator for the stack registry (pure; no MLX import).

The pinned golden fixture (`mlx_vlm/tests/fixtures/stack_registry_golden.yaml`, the shipped-when-ON registry: stack `59d7802f`
with every substitution below reversed, P422) is the reference. Each MTP-certified entry's MTP-OFF form is an ATOMIC raw-byte substitution of its block: the C177 lines
for the two Qwen3.8-27B picks, the P406 / C191 lines for `Ornith-1.0-35B-mlx-uniform-4bit` and
`Qwen3.6-27B-Opus-Distill-OptiQ-4bit`. The accepted live registries are exactly the 2**4 = 16 combinations; `classify`
returns the set of OFF entries or None, `overlay_on` turns one entry ON whatever the others are. `Campaign.prepare`, the
overlays and the fork's opt-in live-registry test use this module.
"""

import hashlib
import itertools
from pathlib import Path

FIXTURE_NAME = "stack_registry_golden.yaml"
FIXTURE_MARKER = "# ---- BEGIN VERBATIM main_models.yaml ----\n"
FIXTURE_SHA256 = "96067c3d019375a19f28bd9546f45da9907da62ea75c2f8649deffd81b6ebfdb"
_C177 = (
    "  # P369 INTERIM (C177 2026-10-17): MTP OFF on the daily driver until the MTP EOS/budget fixes ship and are "
    "re-certified"
)
_P406 = (
    "  # P406 INTERIM (C191 2026-10-20): MTP OFF on the daily driver until re-qualified on the P400 + P405 fork "
    "(P369 phase 2)"
)


def _commented(line, suffix=""):
    return line.replace("    ", "    # ", 1) + suffix


# entry -> ((fixture line, its OFF line), ...), verbatim, applied together or not at all
ENTRY_OFF = {
    "Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed": (
        (
            "    mtp_verify_scan: joint_v1",
            _commented(
                "    mtp_verify_scan: joint_v1",
                _C177 + " (joint_v1 requires draft_kind mtp)",
            ),
        ),
        ("    draft_kind: mtp", _commented("    draft_kind: mtp", _C177)),
        (
            "    draft_model: caslca/Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed-mtp-drafter",
            _commented(
                "    draft_model: caslca/Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed-mtp-drafter"
            ),
        ),
    ),
    "Qwen3.8-27B-mlx-uniform-4bit": (
        ("    draft_kind: mtp", _commented("    draft_kind: mtp", _C177)),
        (
            "    draft_model: caslca/Qwen3.8-27B-mlx-uniform-4bit-mtp-drafter  # UPLOADED 2026-08-26 (public; card still says "
            "PROBE-ONLY — needs the certified update)",
            _commented(
                "    draft_model: caslca/Qwen3.8-27B-mlx-uniform-4bit-mtp-drafter  # UPLOADED 2026-08-26 (public; card still "
                "says PROBE-ONLY — needs the certified update)"
            ),
        ),
    ),
    "Ornith-1.0-35B-mlx-uniform-4bit": (
        ("    draft_kind: mtp", _commented("    draft_kind: mtp", _P406)),
        (
            "    draft_model: caslca/Ornith-1.0-35B-mlx-uniform-4bit-mtp-drafter  # UPLOADED 2026-08-31 (public)",
            _commented(
                "    draft_model: caslca/Ornith-1.0-35B-mlx-uniform-4bit-mtp-drafter  # UPLOADED 2026-08-31 (public)"
            ),
        ),
    ),
    "Qwen3.6-27B-Opus-Distill-OptiQ-4bit": (
        ("    draft_kind: mtp", _commented("    draft_kind: mtp", _P406)),
        (
            "    draft_model: caslca/Qwen3.6-27B-Opus-Distill-OptiQ-4bit-mtp-drafter  # UPLOADED 2026-08-25 (public, anonymous "
            "download verified): extracted native sidecar drafter",
            _commented(
                "    draft_model: caslca/Qwen3.6-27B-Opus-Distill-OptiQ-4bit-mtp-drafter  # UPLOADED 2026-08-25 (public, "
                "anonymous download verified): extracted native sidecar drafter"
            ),
        ),
    ),
}


def fixture_text(fixtures_dir):
    """The verbatim registry after the fixture's marker, read as bytes (never universal-newline), sha-checked."""
    raw = (Path(fixtures_dir) / FIXTURE_NAME).read_bytes().decode("utf-8")
    text = raw.split(FIXTURE_MARKER, 1)[1]
    check_fixture(text)
    return text


def check_fixture(text):
    got = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if got != FIXTURE_SHA256:
        raise ValueError(f"golden fixture sha256 {got} != pinned {FIXTURE_SHA256}")


def _span(lines, entry):
    starts = [i for i, line in enumerate(lines) if line.startswith("  - name: ")]
    hits = [i for i in starts if lines[i] == f"  - name: {entry}"]
    if len(hits) != 1:
        raise ValueError(f"{entry}: {len(hits)} registry blocks")
    end = next((i for i in starts if i > hits[0]), len(lines))
    return hits[0], end


def block(text, entry):
    lines = text.split("\n")
    start, end = _span(lines, entry)
    return "\n".join(lines[start:end])


def apply(fixture, off):
    """The fixture with exactly the entries in ``off`` turned MTP-OFF (each fixture line exactly once in its block)."""
    lines = fixture.split("\n")
    for entry in off:
        start, end = _span(lines, entry)
        for old, new in ENTRY_OFF[entry]:
            hits = [i for i in range(start, end) if lines[i] == old]
            if len(hits) != 1:
                raise ValueError(f"{entry}: {old!r} occurs {len(hits)} times")
            lines[hits[0]] = new
    return "\n".join(lines)


def classify(text, fixture):
    """The frozenset of MTP-OFF entries if ``text`` is byte-identical to one of the 16 states, else None."""
    for k in range(len(ENTRY_OFF) + 1):
        for off in itertools.combinations(ENTRY_OFF, k):
            if text == apply(fixture, off):
                return frozenset(off)
    return None


def overlay_on(text, fixture, entry):
    """``(text with ``entry`` MTP-ON, the restored fixture lines)``; the other entries keep their state. Refuses a
    ``text`` outside the 16 states, and asserts the entry's block equals the fixture's.
    """
    off = classify(text, fixture)
    if off is None:
        raise ValueError("the live registry is not one of the 16 accepted states")
    on = apply(fixture, off - {entry})
    if block(on, entry) != block(fixture, entry):
        raise ValueError(f"{entry}: restored block differs from the fixture")
    restored = [old for old, _ in ENTRY_OFF[entry]] if entry in off else []
    return on, restored
