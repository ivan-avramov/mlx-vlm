"""M76 B2 acceptance 1: ``registry_states.classify_kv_arm`` and the TQ4 ``tq4_attention`` arms (fork-only, dev; pure, CPU).

Written from SPEC v1 (``m76-b2-fork-dev-tools.md``) before the implementation, by a test writer who did not write it.  Each test
carries a ``# class:`` comment: (a) reproduces a failure the specification fixes, (b) preserves an invariant (passes on the
baseline and must keep passing), (c) new behaviour.  Registries are the fixture with B1's edits (``test_m76_b2_fe_common``).
"""

# isort: skip_file

import pytest

from mlx_vlm.tests import test_p475a_kv_arm as kv
from mlx_vlm.tests.test_m76_b2_fe_common import (  # also puts dev/ on sys.path
    ARM_TEXTS,
    registry_states,
    DELTAS,
    FIXTURE,
    PICK,
    PICK_C,
    PICK_D,
    SERVED_X,
    UNIFORM,
    X_TEXT,
    XO_ENTRY_OFF_TEXT,
    XO_TEXT,
    add_mode,
    comment_mtp,
    kv_text,
)

classify_kv_arm = registry_states.classify_kv_arm


def delta_of(got):
    return {k: tuple(v) for k, v in got[1].items()}


# ================================================================== the three accepted arms
# class: (c) new behaviour
def test_kv_arm_delta_keys_gain_the_mode_key_and_kv_arm_keys_stay():
    # Errata v1.1: KV_ARM_KEYS is pinned by test_p475a_kv_arm.py; the three keys an arm may change are KV_ARM_DELTA_KEYS
    assert registry_states.KV_ARM_KEYS == ("kv_bits", "kv_quant_scheme")
    assert registry_states.KV_ARM_DELTA_KEYS == (
        "kv_bits",
        "kv_quant_scheme",
        "tq4_attention",
    )


# class: (c)
def test_the_arm_rows_gain_tq4x_and_tq4v_and_keep_the_five_p475_rows():
    rows = registry_states.KV_ARM_ROWS
    assert rows["TQ4X"] == (4, "turboquant", "m72_exact")
    assert rows["TQ4V"] == (4, "turboquant", "m72_v1")
    # the five P475 rows keep their (bits, scheme) prefix whatever the row tuples' length becomes
    assert {k: tuple(rows[k])[:2] for k in ("N16", "AF8", "TQ8", "AF4", "TQ4")} == {
        "N16": (0, None),
        "AF8": (8, "uniform"),
        "TQ8": (8, "turboquant"),
        "AF4": (4, "uniform"),
        "TQ4": (4, "turboquant"),
    }
    assert set(rows) == {"N16", "AF8", "TQ8", "AF4", "TQ4", "TQ4X", "TQ4V"}


# class: (c) (a on the baseline too: X / V / XO are refused today)
@pytest.mark.parametrize("arm", ("X", "V", "XO"))
def test_x_v_xo_are_accepted_with_the_right_off_set_and_delta(arm):
    got = classify_kv_arm(ARM_TEXTS[arm], FIXTURE, PICK)
    assert got is not None
    off, _ = got
    assert off == (frozenset({PICK}) if arm == "XO" else frozenset())
    assert delta_of(got) == DELTAS[arm]


# class: (c) (the OFF state exactly as ENTRY_OFF spells it, notes and all, with an X row: XO as today's OFF states are)
def test_xo_spelled_as_entry_off_is_accepted():
    got = classify_kv_arm(XO_ENTRY_OFF_TEXT, FIXTURE, PICK)
    assert got is not None and got[0] == frozenset({PICK})
    assert delta_of(got) == DELTAS["XO"]


# class: (c) (the arm is built on any accepted base state, as the P475 arms are)
@pytest.mark.parametrize("base_off", kv.SIXTEEN, ids=kv.SIXTEEN_IDS)
@pytest.mark.parametrize("mode", ("m72_exact", "m72_v1"))
def test_a_mode_arm_on_every_accepted_base_state(base_off, mode):
    base = registry_states.apply(FIXTURE, set(base_off))
    got = classify_kv_arm(kv_text(PICK, 4, mode=mode, base=base), FIXTURE, PICK)
    assert got is not None, base_off
    assert got[0] == frozenset(base_off)
    assert delta_of(got) == {"kv_bits": (0, 4), "tq4_attention": (None, mode)}


# class: (c) (a quoted string is the same value)
def test_a_quoted_mode_string_is_the_same_mode():
    got = classify_kv_arm(kv_text(PICK, 4, mode='"m72_exact"'), FIXTURE, PICK)
    assert got is not None and delta_of(got) == DELTAS["X"]


# class: (c) (the mode line's position / the comment text does not matter: settings, not bytes)
def test_the_mode_line_may_sit_anywhere_in_the_block_and_carry_a_comment():
    lines = X_TEXT.split("\n")
    start, end = registry_states._span(lines, PICK)
    i = next(i for i in range(start, end) if lines[i].startswith("    tq4_attention:"))
    mode_line = lines.pop(i) + "  # M76 arm X"
    lines.insert(start + 1, mode_line)
    got = classify_kv_arm("\n".join(lines), FIXTURE, PICK)
    assert got is not None and delta_of(got) == DELTAS["X"]


# ================================================================== refusals: a mode only where the spec admits it
# class: (b) (refused on the baseline too — a mode on another entry)
@pytest.mark.parametrize("mode", ("m72_exact", "m72_v1"))
def test_a_mode_on_the_uniform_pick_is_refused(mode):
    text = kv_text(UNIFORM, 4, "turboquant", mode=mode)
    assert classify_kv_arm(text, FIXTURE, UNIFORM) is None


# class: (b) (a mode on `Qwen3.8-27B-mlx-uniform-4bit` is refused even when the registry also holds a valid X for `Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed`: two entries changed)
def test_a_mode_on_the_uniform_pick_beside_a_valid_x_is_refused_for_both():
    text = kv_text(UNIFORM, 4, "turboquant", mode="m72_exact", base=X_TEXT)
    assert classify_kv_arm(text, FIXTURE, UNIFORM) is None
    assert classify_kv_arm(text, FIXTURE, PICK) is None


# class: (b)
@pytest.mark.parametrize("other", (PICK_D, PICK_C, SERVED_X))
def test_a_mode_on_a_third_entry_is_refused_whatever_its_kv_keys(other):
    text = kv_text(other, 4, "turboquant", mode="m72_exact")
    assert classify_kv_arm(text, FIXTURE, other) is None


# class: (b) (the mode with another row's bits / scheme)
@pytest.mark.parametrize(
    "bits, scheme",
    (
        (8, "turboquant"),  # TQ8
        (4, "uniform"),  # AF4
        (8, "uniform"),  # AF8
        (0, "turboquant"),  # N16
        (0, None),  # N16, scheme as committed
        (2, "turboquant"),
        (6, "turboquant"),
        (4, "affine"),
    ),
    ids=("TQ8", "AF4", "AF8", "N16-tq", "N16", "w2", "w6", "affine"),
)
@pytest.mark.parametrize("mode", ("m72_exact", "m72_v1"))
def test_a_mode_with_another_rows_bits_or_scheme_is_refused(bits, scheme, mode):
    assert (
        classify_kv_arm(kv_text(PICK, bits, scheme, mode=mode), FIXTURE, PICK) is None
    )


# class: (b) (4-bit turboquant stays the scheme only when the key is present: an absent scheme is not the row)
def test_a_mode_without_the_scheme_key_is_refused():
    lines = X_TEXT.split("\n")
    start, end = registry_states._span(lines, PICK)
    lines = [
        l
        for i, l in enumerate(lines)
        if not (start <= i < end and l.startswith("    kv_quant_scheme:"))
    ]
    assert classify_kv_arm("\n".join(lines), FIXTURE, PICK) is None


# class: (b) (any other mode string; the match is exact)
@pytest.mark.parametrize(
    "mode",
    (
        "m72",
        "exact",
        "v1",
        "m72_v2",
        "M72_EXACT",
        "M72_V1",
        '"m72_exact "',
        '" m72_exact"',
        "m72_exact_v1",
        '""',
        "off",
        "auto",
        "none",
    ),
)
def test_an_unknown_mode_string_is_refused(mode):
    assert classify_kv_arm(kv_text(PICK, 4, mode=mode), FIXTURE, PICK) is None


# class: (b) (YAML false / 0 / null / empty / non-string values are values, never "absent")
@pytest.mark.parametrize(
    "literal",
    (
        "false",
        "False",
        "0",
        "null",
        "~",
        "",
        "true",
        "1",
        "4",
        "[m72_exact]",
        "{a: b}",
        "1.0",
        "[]",
    ),
)
def test_a_non_string_or_falsy_mode_is_refused(literal):
    assert classify_kv_arm(kv_text(PICK, 4, mode=literal), FIXTURE, PICK) is None


# class: (b) (a falsy mode on an otherwise unchanged registry is not "absent" either)
@pytest.mark.parametrize("literal", ("false", "0", "null"))
def test_a_falsy_mode_on_the_committed_registry_is_refused(literal):
    assert classify_kv_arm(add_mode(FIXTURE, PICK, literal), FIXTURE, PICK) is None


# class: (b) (a mode with the committed native16 width: the mode is no row of its own)
@pytest.mark.parametrize("mode", ("m72_exact", "m72_v1"))
def test_a_mode_alone_on_the_committed_registry_is_refused(mode):
    assert classify_kv_arm(add_mode(FIXTURE, PICK, mode), FIXTURE, PICK) is None


# class: (b) (a duplicate mode key is a refused registry, never "the last one")
def test_a_duplicate_mode_key_is_refused():
    lines = X_TEXT.split("\n")
    start, end = registry_states._span(lines, PICK)
    i = next(i for i in range(start, end) if lines[i].startswith("    tq4_attention:"))
    lines.insert(i + 1, "    tq4_attention: m72_v1")
    assert classify_kv_arm("\n".join(lines), FIXTURE, PICK) is None


# class: (b) (a mode plus one non-KV change is refused: X is a KV arm only)
@pytest.mark.parametrize(
    "key, value",
    (
        ("prefill_step_size", 1024),
        ("kv_prealloc_tokens", 131072),
        ("max_kv_cache_size", 131072),
        ("quantized_kv_start", 5000),
        ("attention_policy", "auto"),
        ("kv_quant_mode", "prod"),
    ),
)
def test_a_mode_with_a_non_kv_change_is_refused(key, value):
    assert (
        classify_kv_arm(kv.set_kv(X_TEXT, PICK, **{key: value}), FIXTURE, PICK) is None
    )


# class: (b) (XO is MTP-OFF as a unit: a partial MTP removal under a mode is refused)
@pytest.mark.parametrize("key", ("draft_kind", "draft_model", "mtp_verify_scan"))
def test_x_with_one_mtp_line_commented_is_refused(key):
    lines = X_TEXT.split("\n")
    start, end = registry_states._span(lines, PICK)
    i = next(i for i in range(start, end) if lines[i].startswith(f"    {key}:"))
    lines[i] = "    # " + lines[i][4:]
    assert classify_kv_arm("\n".join(lines), FIXTURE, PICK) is None


# class: (b) (the mode is the pick's only: asked for another entry name it is refused)
@pytest.mark.parametrize(
    "name", (UNIFORM, PICK_D, PICK_C, SERVED_X, "No-Such-Model", "")
)
def test_x_asked_for_another_entry_is_refused(name):
    assert classify_kv_arm(X_TEXT, FIXTURE, name) is None


# class: (b) (never an exception on malformed input)
@pytest.mark.parametrize("literal", ("[", "!!python/object:object {}", "*alias"))
def test_a_malformed_mode_never_raises(literal):
    assert classify_kv_arm(kv_text(PICK, 4, mode=literal), FIXTURE, PICK) is None


# ================================================================== the existing rows, states and functions are unchanged
# class: (b) (every existing P475 row of both picks, mode absent: accepted with the P475 delta, exactly as today)
@pytest.mark.parametrize("pick, arm", kv.CELLS, ids=kv.CELL_IDS)
def test_every_existing_row_is_accepted_with_no_mode_in_its_delta(pick, arm):
    got = classify_kv_arm(kv.arm_text(arm, pick), FIXTURE, pick)
    assert got is not None
    assert got[0] == frozenset()
    assert delta_of(got) == kv.arm_delta(arm, pick)
    assert "tq4_attention" not in got[1]


# class: (b) (`Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed`'s turboquant at 4 bits WITHOUT a mode is still the plain P475 TQ4 row)
def test_the_fable_pick_tq4_without_a_mode_is_still_the_plain_tq4_row():
    got = classify_kv_arm(kv_text(PICK, 4, "turboquant"), FIXTURE, PICK)
    assert got is not None and delta_of(got) == {"kv_bits": (0, 4)}


# class: (b) (the committed state and the 16 states: an empty delta, no mode)
@pytest.mark.parametrize("pick", kv.PICKS)
@pytest.mark.parametrize("base_off", kv.SIXTEEN, ids=kv.SIXTEEN_IDS)
def test_the_sixteen_states_are_an_empty_delta(pick, base_off):
    text = registry_states.apply(FIXTURE, set(base_off))
    assert classify_kv_arm(text, FIXTURE, pick) == (
        registry_states.classify(text, FIXTURE),
        {},
    )


# class: (b) (classify / overlay_on / ENTRY_OFF are unchanged: an arm registry is NOT one of the 16 states)
@pytest.mark.parametrize("arm", ("X", "V", "XO"))
def test_classify_does_not_accept_a_mode_arm_and_overlay_on_refuses_it(arm):
    assert registry_states.classify(ARM_TEXTS[arm], FIXTURE) is None
    with pytest.raises(ValueError):
        registry_states.overlay_on(ARM_TEXTS[arm], FIXTURE, PICK)


# class: (b)
def test_entry_off_keeps_the_three_mtp_lines_of_the_fable_pick_and_no_mode_key():
    keys = [old.strip().split(":")[0] for old, _ in registry_states.ENTRY_OFF[PICK]]
    assert keys == ["mtp_verify_scan", "draft_kind", "draft_model"]
    assert all(
        "tq4_attention" not in old + new for old, new in registry_states.ENTRY_OFF[PICK]
    )
    assert tuple(registry_states.KV_ARM_ENTRIES) == (UNIFORM, PICK)


# class: (b) (the plain-commented XO and the ENTRY_OFF XO are the same settings)
def test_the_two_spellings_of_xo_classify_alike():
    assert classify_kv_arm(XO_TEXT, FIXTURE, PICK) == classify_kv_arm(
        XO_ENTRY_OFF_TEXT, FIXTURE, PICK
    )
    assert comment_mtp(X_TEXT, PICK) == XO_TEXT
