"""M76 B2 acceptance 1: ``registry_states.classify_kv_arm`` accepts M76's arms X / V / XO of
``Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed`` and refuses every other mode use (CPU only, no GPU).

The arm registries are built BY HAND with the stack's B1 edit model (``bench.p475_kv`` campaign ``f45``: ``set kv_bits 4``, ``add
tq4_attention <mode>`` after it, XO also comments ``draft_kind`` / ``draft_model`` / ``mtp_verify_scan``, one
``# PROVISIONAL M76 arm <ARM>`` line after the name), applied to the pinned golden fixture (the verbatim stack registry).
"""

import re
import sys
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import registry_states  # noqa: E402

FIXTURE = registry_states.fixture_text(Path(__file__).resolve().parent / "fixtures")
PICK = "Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed"
UNIFORM = "Qwen3.8-27B-mlx-uniform-4bit"
THIRD_ENTRY = "Ornith-1.0-35B-mlx-uniform-4bit"
FOURTH_ENTRY = "Qwen3.6-27B-Opus-Distill-OptiQ-4bit"
MTP = ("draft_kind", "draft_model", "mtp_verify_scan")
MODES = {"X": "m72_exact", "V": "m72_v1", "XO": "m72_exact"}


# ------------------------------------------------------------------ the B1 edit model, by hand
def edit(text, entry, *, sets=(), adds=(), comments=(), label=None):
    """``text`` with ``entry``'s 4-space key lines edited: ``sets`` ``(key, value)`` rewrite the line, ``adds``
    ``(anchor, key, value)`` insert ``key: value`` after the anchor key's line, ``comments`` turn ``key: v`` into
    ``# key: v``; ``label`` adds ``# <label>`` after the name line."""
    lines = text.split("\n")
    start, end = registry_states._span(lines, entry)
    out = []
    for i, line in enumerate(lines):
        m = re.match(r"^    ([a-z_]+):(.*)$", line) if start < i < end else None
        key = m.group(1) if m else None
        if key in dict(sets):
            out.append(f"    {key}: {dict(sets)[key]}")
        elif key in comments:
            out.append(f"    # {key}:{m.group(2)}")
        else:
            out.append(line)
        for anchor, new, value in adds:
            if key == anchor:
                out.append(f"    {new}: {value}")
        if i == start and label:
            out.append(f"    # {label}")
    return "\n".join(out)


def arm_text(arm, base=FIXTURE, entry=PICK, mode=None, bits="4"):
    """M76 §6 arm ``arm`` (X / V / XO) of ``entry`` on ``base``; ``mode`` / ``bits`` override the arm's values."""
    mode = MODES[arm] if mode is None else mode
    return edit(
        base,
        entry,
        sets=(("kv_bits", bits),),
        adds=(("kv_bits", "tq4_attention", mode),),
        comments=MTP if arm == "XO" else (),
        label=f"PROVISIONAL M76 arm {arm}",
    )


def delta(mode):
    return {"kv_bits": (0, 4), "tq4_attention": (None, mode)}


def base_states():
    """Every one of the 16 accepted base states (each a set of MTP-OFF entries)."""
    entries = list(registry_states.ENTRY_OFF)
    return [
        frozenset(e for i, e in enumerate(entries) if mask >> i & 1)
        for mask in range(16)
    ]


# ================================================================== acceptance 1: X / V / XO accepted
@pytest.mark.parametrize("arm", ("X", "V", "XO"))
def test_each_m76_arm_is_accepted_with_its_delta(arm):
    got = registry_states.classify_kv_arm(arm_text(arm), FIXTURE, PICK)
    assert got is not None
    off, kv_delta = got
    assert off == (frozenset({PICK}) if arm == "XO" else frozenset())
    assert kv_delta == delta(MODES[arm])


@pytest.mark.parametrize("arm", ("X", "V", "XO"))
@pytest.mark.parametrize(
    "base_off", base_states(), ids=lambda s: "+".join(sorted(s)) or "none"
)
def test_each_m76_arm_on_every_accepted_base_state(arm, base_off):
    base = registry_states.apply(FIXTURE, set(base_off) - {PICK})
    off, kv_delta = registry_states.classify_kv_arm(arm_text(arm, base), FIXTURE, PICK)
    want = (set(base_off) - {PICK}) | ({PICK} if arm == "XO" else set())
    assert off == frozenset(want) and kv_delta == delta(MODES[arm])


def test_the_new_rows_and_keys():
    assert registry_states.KV_ARM_ROWS["TQ4X"] == (4, "turboquant", "m72_exact")
    assert registry_states.KV_ARM_ROWS["TQ4V"] == (4, "turboquant", "m72_v1")
    assert registry_states.TQ4_MODE_KEY == "tq4_attention"
    assert registry_states.KV_ARM_DELTA_KEYS == (
        "kv_bits",
        "kv_quant_scheme",
        "tq4_attention",
    )
    assert registry_states.KV_ARM_MODE_ENTRIES == (PICK,)


@pytest.mark.parametrize("arm", ("X", "V", "XO"))
def test_classify_itself_refuses_every_m76_arm(arm):
    """The 16-state rule is unchanged: a KV arm is outside it."""
    assert registry_states.classify(arm_text(arm), FIXTURE) is None


def test_the_mode_line_may_sit_anywhere_in_the_block_and_carry_a_comment():
    text = edit(
        FIXTURE,
        PICK,
        sets=(("kv_bits", "4  # M76 arm"),),
        adds=(("prefill_step_size", "tq4_attention", "m72_exact  # 1.3 mode"),),
    )
    assert registry_states.classify_kv_arm(text, FIXTURE, PICK) == (
        frozenset(),
        delta("m72_exact"),
    )


# ================================================================== acceptance 1: refusals
@pytest.mark.parametrize("mode", ("m72_exact", "m72_v1"))
def test_a_mode_on_the_uniform_pick_is_refused(mode):
    """``Qwen3.8-27B-mlx-uniform-4bit`` already serves TQ4 4-bit; adding a mode is still refused (not a mode entry)."""
    text = edit(FIXTURE, UNIFORM, adds=(("kv_bits", "tq4_attention", mode),))
    assert registry_states.classify_kv_arm(text, FIXTURE, UNIFORM) is None
    assert registry_states.classify_kv_arm(text, FIXTURE, PICK) is None


@pytest.mark.parametrize("entry", (THIRD_ENTRY, FOURTH_ENTRY))
def test_a_mode_on_a_third_entry_is_refused(entry):
    text = edit(FIXTURE, entry, adds=(("hf_path", "tq4_attention", "m72_exact"),))
    for asked in (entry, PICK, UNIFORM):
        assert registry_states.classify_kv_arm(text, FIXTURE, asked) is None


@pytest.mark.parametrize(
    "bits, scheme",
    ((4, "uniform"), (8, "turboquant"), (8, "uniform"), (0, None), (2, "turboquant")),
    ids=("AF4", "TQ8", "AF8", "N16", "TQ2"),
)
@pytest.mark.parametrize("mode", ("m72_exact", "m72_v1"))
def test_a_mode_with_another_rows_bits_or_scheme_is_refused(bits, scheme, mode):
    sets = [("kv_bits", str(bits))]
    if scheme is not None:
        sets.append(("kv_quant_scheme", scheme))
    text = edit(FIXTURE, PICK, sets=sets, adds=(("kv_bits", "tq4_attention", mode),))
    assert registry_states.classify_kv_arm(text, FIXTURE, PICK) is None


@pytest.mark.parametrize(
    "mode", ("m72_v2", "M72_EXACT", "m72", "exact", '""', "'m72_exact '", "[m72_exact]")
)
def test_an_unknown_mode_is_refused(mode):
    assert (
        registry_states.classify_kv_arm(arm_text("X", mode=mode), FIXTURE, PICK) is None
    )


@pytest.mark.parametrize("mode", ("false", "0", "null", "~", "off", "no", "1"))
def test_false_zero_or_null_as_a_mode_is_refused(mode):
    """Also on the N16 (committed) KV keys: the key's presence alone is outside every existing row."""
    assert (
        registry_states.classify_kv_arm(arm_text("X", mode=mode), FIXTURE, PICK) is None
    )
    n16 = edit(FIXTURE, PICK, adds=(("kv_bits", "tq4_attention", mode),))
    assert registry_states.classify_kv_arm(n16, FIXTURE, PICK) is None


@pytest.mark.parametrize("mode", ("m72_exact", "m72_v1"))
def test_a_mode_with_the_committed_native16_kv_is_refused(mode):
    text = edit(FIXTURE, PICK, adds=(("kv_bits", "tq4_attention", mode),))
    assert registry_states.classify_kv_arm(text, FIXTURE, PICK) is None


def test_a_fractional_or_string_width_with_a_mode_is_refused():
    for bits in ("4.0", "'4'", "4.5", ".inf"):
        assert (
            registry_states.classify_kv_arm(arm_text("X", bits=bits), FIXTURE, PICK)
            is None
        ), bits


@pytest.mark.parametrize("keep", MTP)
def test_xo_with_only_some_mtp_keys_commented_is_refused(keep):
    text = edit(
        FIXTURE,
        PICK,
        sets=(("kv_bits", "4"),),
        adds=(("kv_bits", "tq4_attention", "m72_exact"),),
        comments=tuple(k for k in MTP if k != keep),
    )
    assert registry_states.classify_kv_arm(text, FIXTURE, PICK) is None


def test_an_arm_plus_a_non_kv_change_is_refused():
    text = edit(arm_text("X"), PICK, sets=(("prefill_step_size", "1024"),))
    assert registry_states.classify_kv_arm(text, FIXTURE, PICK) is None


def test_an_arm_plus_a_second_entry_changed_is_refused():
    text = edit(
        arm_text("X"), UNIFORM, sets=(("kv_bits", "8"), ("kv_quant_scheme", "uniform"))
    )
    assert registry_states.classify_kv_arm(text, FIXTURE, PICK) is None
    assert registry_states.classify_kv_arm(text, FIXTURE, UNIFORM) is None


def test_a_duplicate_mode_key_is_refused():
    text = edit(
        arm_text("X"), PICK, adds=(("prefill_step_size", "tq4_attention", "m72_exact"),)
    )
    assert registry_states.classify_kv_arm(text, FIXTURE, PICK) is None
