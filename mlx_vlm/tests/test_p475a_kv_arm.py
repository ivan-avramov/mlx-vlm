"""P475a v4: `registry_states.classify_kv_arm` and the arm-aware pure parts of `qualify` (fork-only, dev; CPU only).

Written from SPEC v4 (`p475a-qualify-kv-arm.md`) and the arm table of `p475-kv-matrix.md` BEFORE the implementation, by a test
writer who did not write it.  Every test carries a ``# class:`` comment:

* ``reproduces-failure`` — fails on the baseline (fork ``fd67082a``) for the defect the spec names, passes once fixed;
* ``invariant``          — passes on the baseline AND must keep passing (the no-flag behaviour, the accepted states);
* ``new-behaviour``      — exercises an interface the baseline lacks (`classify_kv_arm`, ``--kv-arm``).

The arm table, the per-arm cache manifests and the argv spelling below are written BY HAND from the matrix, never derived from
the code under test.  The driver tests are in ``test_p475a_qualify_driver.py`` (they import this module's helpers).
"""

import math
import sys
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import kvmanifest, qualify, registry_states  # noqa: E402

from mlx_vlm.tests import test_p415_qualify as t415  # noqa: E402

FIXTURE = registry_states.fixture_text(Path(__file__).resolve().parent / "fixtures")
PICK_A = "Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed"  # shipped: native16 KV, attention_policy fused_v1, shrink, lazy
UNIFORM = "Qwen3.8-27B-mlx-uniform-4bit"  # shipped: TurboQuant 4-bit, default attention
PICKS = (PICK_A, UNIFORM)
PICK_D = "Ornith-1.0-35B-mlx-uniform-4bit"
PICK_C = "Qwen3.6-27B-Opus-Distill-OptiQ-4bit"
SERVED_X = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"
OTHERS = (PICK_D, PICK_C, SERVED_X)
MTP_ENTRIES = (PICK_A, UNIFORM, PICK_D, PICK_C)
# the committed (fixture) KV keys of each pick
FIXTURE_KV = {
    PICK_A: dict(kv_bits=0, kv_quant_scheme="turboquant"),
    UNIFORM: dict(kv_bits=4, kv_quant_scheme="turboquant"),
}

# the five arm rows of p475-kv-matrix.md (Arms), by hand: (kv_bits, kv_quant_scheme); None = the key is left as committed
ARMS = {
    "N16": (0, None),
    "AF8": (8, "uniform"),
    "TQ8": (8, "turboquant"),
    "AF4": (4, "uniform"),
    "TQ4": (4, "turboquant"),
}
CELLS = [(pick, arm) for pick in PICKS for arm in ARMS]
CELL_IDS = [f"{'pickA' if p == PICK_A else 'uniform'}-{a}" for p, a in CELLS]
SIXTEEN = [
    frozenset(n for i, n in enumerate(MTP_ENTRIES) if mask >> i & 1)
    for mask in range(16)
]
# id: one char per entry of MTP_ENTRIES (pick A, uniform, pick D, pick C), "1" = that entry is MTP-OFF in the base state
SIXTEEN_IDS = ["off-" + format(mask, "04b")[::-1] for mask in range(16)]


# ------------------------------------------------------------------ helpers shared with the driver tests
def set_kv(text, entry, **keys):
    """``text`` with the registry key lines of ``entry``'s block (4-space indent, exactly one each) replaced by ``key: value``."""
    import re

    lines = text.split("\n")
    start, end = registry_states._span(lines, entry)
    for key, value in keys.items():
        hits = [
            i for i in range(start, end) if re.match(rf"^    {key}:( |$)", lines[i])
        ]
        assert len(hits) <= 1, (entry, key, hits)
        if hits:
            lines[hits[0]] = f"    {key}: {value}"
        else:  # a key the block does not carry (the third entries have no kv_quant_scheme): added after the name line
            lines.insert(start + 1, f"    {key}: {value}")
            end += 1
    return "\n".join(lines)


def arm_text(arm, entry, base=None):
    """A committed arm registry: ``base`` (default the golden fixture) with ``entry``'s KV keys set to the arm's row."""
    bits, scheme = ARMS[arm]
    keys = dict(kv_bits=bits)
    if scheme is not None:
        keys["kv_quant_scheme"] = scheme
    return set_kv(FIXTURE if base is None else base, entry, **keys)


def arm_delta(arm, entry):
    """The expected ``kv_delta``: ``{key: (fixture value, arm value)}`` for the keys that differ from the fixture."""
    bits, scheme = ARMS[arm]
    fx, delta = FIXTURE_KV[entry], {}
    if bits != fx["kv_bits"]:
        delta["kv_bits"] = (fx["kv_bits"], bits)
    if scheme is not None and scheme != fx["kv_quant_scheme"]:
        delta["kv_quant_scheme"] = (fx["kv_quant_scheme"], scheme)
    return delta


def as_json(delta):
    return {k: list(v) for k, v in delta.items()}


def arm_manifest(arm):
    """The live per-layer manifest the served path builds for ``t415._template()`` under the arm (4 top-level entries: two
    linear-attention, two KV): affine converts every KV layer, TurboQuant every one but the last top-level entry.
    """

    def affine(path, bits):
        return dict(
            path=path,
            kind="affine",
            cls="PreallocQuantizedKVCache",
            batch=False,
            bits=bits,
            group_size=64,
        )

    def tq(path, bits):
        return dict(
            path=path,
            kind="tq",
            cls="TurboQuantKVCache",
            batch=False,
            bits=float(bits),
            key_bits=float(bits),
            value_bits=float(bits),
            mode="mse",
        )

    native = t415._native_manifest()
    bits = ARMS[arm][0]
    if arm == "N16":
        return native
    if arm.startswith("AF"):
        return [affine(m["path"], bits) if m["kind"] == "native" else m for m in native]
    return [
        tq("1", bits) if m["path"] == "1" else m for m in native
    ]  # the last KV layer ("3") stays native


def has_native_leaf(manifest):
    return any(m["kind"] == "native" for m in manifest)


def entry_for(arm, pick, *, mtp_on=True):
    """The registry entry dict of ``pick`` in the arm registry (its MTP keys dropped when ``mtp_on`` is False)."""
    entry = dict(registry_states.settings(arm_text(arm, pick))[pick])
    if not mtp_on:
        for key in qualify.MTP_KEYS:
            entry.pop(key, None)
    return entry


def valid_record(expected, manifest, *, sdpa_forced):
    """A worker record that passes every ``request_problems`` rule for ``expected`` (the arm's own hand-built manifest and the
    given ``sdpa_forced``); fields not under test are derived from ``expected`` the way the worker reports them.
    """
    gd = expected["generation_defaults"]
    resolved = dict(gd, max_tokens=2048, seed=7)
    if "thinking_budget" in gd:
        resolved["thinking_budget"] = min(gd["thinking_budget"], int(2048 * 0.8))
    caps = {
        m["path"]: expected["prealloc"]
        for m in manifest
        if m["kind"] in ("tq", "affine", "native")
    }
    caps.update({m["path"]: None for m in manifest if m["path"] not in caps})
    return t415._record(
        resolved=resolved,
        server_defaults=gd,
        chunks=qualify.chunk_sequence(
            expected["prompt_tokens"], expected["prefill_step"]
        ),
        manifest=manifest,
        template=t415._template(),
        capacities_by_path=caps,
        sdpa_forced=sdpa_forced,
        lazy_calls=1 if expected["lazy_embeddings"] else 0,
        shrink_calls=1 if expected["shrink"] else 0,
        peaks=dict(
            prefill=36.5e9,
            decode=36.9e9,
            retire=37.1e9 if expected["shrink"] else None,
        ),
        drafter_live_checks=[[]] if expected["drafter"] else None,
        memory=dict(
            phys=128e9,
            rec_wss=110e9,
            heads=24,
            set_memory_limit=[int(min(0.85 * 128e9, 110e9))],
            set_cache_limit=[qualify.expected_cache_limit(24, expected)],
        ),
    )


def right_forced(pick, arm):
    """The attention counter the arm must show: forced calls only under ``fused_v1`` (pick A) AND with a native-KV leaf."""
    return 640 if pick == PICK_A and has_native_leaf(arm_manifest(arm)) else 0


# ================================================================== the baseline premises the tests lean on
# class: invariant
def test_the_fixture_premises():
    assert registry_states.classify(FIXTURE, FIXTURE) == frozenset()
    entries = registry_states.settings(FIXTURE)
    for pick, kv in FIXTURE_KV.items():
        assert {k: entries[pick][k] for k in kv} == kv
        assert entries[pick]["quantized_kv_start"] == 0
    assert entries[PICK_A]["attention_policy"] == "fused_v1"
    assert "attention_policy" not in entries[UNIFORM]


# ================================================================== acceptance 1: classify_kv_arm
# class: new-behaviour
def test_kv_arm_keys_are_exactly_the_two_kv_keys():
    assert registry_states.KV_ARM_KEYS == ("kv_bits", "kv_quant_scheme")


# class: new-behaviour
@pytest.mark.parametrize("pick, arm", CELLS, ids=CELL_IDS)
def test_each_arm_of_each_pick_is_accepted_with_the_right_delta(pick, arm):
    got = registry_states.classify_kv_arm(arm_text(arm, pick), FIXTURE, pick)
    assert got is not None
    off, delta = got
    assert off == frozenset()
    assert {k: tuple(v) for k, v in delta.items()} == arm_delta(arm, pick)


# class: new-behaviour
@pytest.mark.parametrize("pick", PICKS)
def test_the_committed_state_is_an_empty_delta_equal_to_classify(pick):
    off, delta = registry_states.classify_kv_arm(FIXTURE, FIXTURE, pick)
    assert delta == {}
    assert off == registry_states.classify(FIXTURE, FIXTURE) == frozenset()


# class: new-behaviour
@pytest.mark.parametrize("pick", PICKS)
@pytest.mark.parametrize("base_off", SIXTEEN, ids=SIXTEEN_IDS)
def test_an_empty_delta_equals_classify_on_every_accepted_state(pick, base_off):
    # an accepted state with the pick's KV keys untouched: (classify, {}) — each of the 16 states, for both picks
    text = registry_states.apply(FIXTURE, base_off)
    assert registry_states.classify_kv_arm(text, FIXTURE, pick) == (
        registry_states.classify(text, FIXTURE),
        {},
    )


# class: new-behaviour
@pytest.mark.parametrize("pick", PICKS)
@pytest.mark.parametrize("base_off", SIXTEEN, ids=SIXTEEN_IDS)
def test_an_arm_built_on_any_of_the_16_accepted_states_is_accepted(pick, base_off):
    # GLM SR2: includes bases where the pick is MTP-OFF and bases where a non-tested entry is MTP-ON / MTP-OFF
    base = registry_states.apply(FIXTURE, base_off)
    arm = "AF8"
    text = arm_text(arm, pick, base)
    got = registry_states.classify_kv_arm(text, FIXTURE, pick)
    assert got is not None, (pick, base_off)
    off, delta = got
    assert (
        off == base_off
    )  # the OFF set of the BASE state: what the mtp_off check reads
    assert {k: tuple(v) for k, v in delta.items()} == arm_delta(arm, pick)


# class: new-behaviour
def test_the_off_set_ignores_the_kv_change_of_the_pick_when_the_pick_is_itself_off():
    base = registry_states.apply(FIXTURE, {UNIFORM, PICK_D})
    off, _ = registry_states.classify_kv_arm(
        arm_text("AF4", UNIFORM, base), FIXTURE, UNIFORM
    )
    assert off == frozenset({UNIFORM, PICK_D})


# ---- entries restricted to the two picks (Sol r1-1)
# class: new-behaviour
@pytest.mark.parametrize("other", OTHERS)
def test_only_the_two_picks_are_accepted_even_with_an_unchanged_registry(other):
    assert registry_states.classify_kv_arm(FIXTURE, FIXTURE, other) is None


# class: new-behaviour
@pytest.mark.parametrize("other", (PICK_D, PICK_C, SERVED_X))
@pytest.mark.parametrize("arm", ("AF8", "TQ8", "AF4"))
def test_a_valid_kv_change_on_a_third_entry_is_refused(other, arm):
    text = set_kv(
        FIXTURE, other, **dict(kv_bits=ARMS[arm][0], kv_quant_scheme=ARMS[arm][1])
    )
    assert registry_states.classify_kv_arm(text, FIXTURE, other) is None


# class: new-behaviour
@pytest.mark.parametrize(
    "name",
    (
        "No-Such-Model",
        "",
        "qwen3.8-27b-mlx-uniform-4bit",
        "Qwen3.8-27B-mlx-uniform",  # allow-shorthand: a deliberately wrong entry name
    ),
)
def test_an_unknown_entry_is_refused(name):
    assert (
        registry_states.classify_kv_arm(arm_text("AF8", UNIFORM), FIXTURE, name) is None
    )


# ---- a second entry changed, a non-KV key changed
# class: new-behaviour
def test_a_second_entry_changed_is_refused_even_when_both_changes_are_valid_arms():
    text = arm_text("AF8", PICK_A, arm_text("TQ8", UNIFORM))
    assert registry_states.classify_kv_arm(text, FIXTURE, PICK_A) is None
    assert registry_states.classify_kv_arm(text, FIXTURE, UNIFORM) is None


# class: new-behaviour
@pytest.mark.parametrize("other", OTHERS + (UNIFORM,))
def test_a_kv_change_on_another_entry_than_the_one_under_test_is_refused(other):
    text = set_kv(FIXTURE, other, kv_bits=8, kv_quant_scheme="uniform")
    assert registry_states.classify_kv_arm(text, FIXTURE, PICK_A) is None


# class: new-behaviour
@pytest.mark.parametrize("pick", PICKS)
@pytest.mark.parametrize(
    "key, value",
    [
        ("prefill_step_size", 1024),
        ("max_kv_cache_size", 131072),
        ("kv_prealloc_tokens", 131072),
        ("hf_path", "caslca/other"),
        ("on_demand", "false"),
        ("type", "text"),
    ],
)
def test_a_non_kv_key_changed_alongside_a_valid_arm_is_refused(pick, key, value):
    text = set_kv(arm_text("AF8", pick), pick, **{key: value})
    assert registry_states.classify_kv_arm(text, FIXTURE, pick) is None


# class: new-behaviour
def test_a_non_kv_key_changed_alone_is_refused():
    assert (
        registry_states.classify_kv_arm(
            set_kv(FIXTURE, UNIFORM, prefill_step_size=1024), FIXTURE, UNIFORM
        )
        is None
    )


# class: new-behaviour
def test_a_top_level_setting_changed_is_refused():
    text = FIXTURE.replace("manager_port: 8000", "manager_port: 8001", 1)
    assert text != FIXTURE
    assert registry_states.classify_kv_arm(text, FIXTURE, UNIFORM) is None
    assert (
        registry_states.classify_kv_arm(
            arm_text("AF8", UNIFORM, text), FIXTURE, UNIFORM
        )
        is None
    )


# class: new-behaviour
@pytest.mark.parametrize("pick", PICKS)
def test_an_added_kv_group_size_or_split_width_key_is_refused(pick):
    # KV keys outside KV_ARM_KEYS (split key / value widths, group size, TurboQuant mode) are non-goals: any added key refuses
    lines = arm_text("AF8", pick).split("\n")
    start, _ = registry_states._span(lines, pick)
    for extra in (
        "kv_group_size: 32",
        "kv_key_bits: 8",
        "kv_value_bits: 4",
        "kv_quant_mode: mse",
    ):
        edited = list(lines)
        edited.insert(start + 1, f"    {extra}")
        assert (
            registry_states.classify_kv_arm("\n".join(edited), FIXTURE, pick) is None
        ), extra


# ---- quantized_kv_start stays 0, enforced by the settings comparison
# class: new-behaviour
@pytest.mark.parametrize("pick", PICKS)
@pytest.mark.parametrize("start", (1, 5000, -1))
def test_a_nonzero_quantized_kv_start_is_refused(pick, start):
    text = set_kv(arm_text("AF8", pick), pick, quantized_kv_start=start)
    assert registry_states.classify_kv_arm(text, FIXTURE, pick) is None


# class: new-behaviour. The d1 AST test ("no quantized_kv_start literal in classify_kv_arm") is DELETED (d3: tests constrain
# behaviour, not source). The behaviour — a non-zero start refuses, for both picks, through the classifier and the driver — is
# test_a_nonzero_quantized_kv_start_is_refused above and test_refuses_a_nonzero_quantized_kv_start in the driver file.
def test_quantized_kv_start_is_not_an_arm_key():
    assert "quantized_kv_start" not in registry_states.KV_ARM_KEYS


# ---- an MTP-key change + a KV change outside the 16 states
# class: new-behaviour
@pytest.mark.parametrize("pick", PICKS)
def test_a_partial_mtp_change_with_a_kv_change_is_refused(pick):
    # one of the entry's MTP keys removed (not all-or-none: outside the 16 states) AND a valid KV arm
    text = arm_text("AF8", pick)
    lines = text.split("\n")
    start, end = registry_states._span(lines, pick)
    hit = next(
        i for i in range(start, end) if lines[i].startswith("    draft_kind: mtp")
    )
    del lines[hit]
    assert (
        registry_states.classify("\n".join(lines), FIXTURE) is None
    )  # premise: outside the 16 states on its own
    assert registry_states.classify_kv_arm("\n".join(lines), FIXTURE, pick) is None


# class: new-behaviour
def test_another_entrys_partial_mtp_change_with_a_kv_arm_is_refused():
    text = arm_text("AF8", UNIFORM)
    lines = text.split("\n")
    start, end = registry_states._span(lines, PICK_D)
    hit = next(
        i for i in range(start, end) if lines[i].startswith("    draft_kind: mtp")
    )
    del lines[hit]
    assert registry_states.classify_kv_arm("\n".join(lines), FIXTURE, UNIFORM) is None


# class: new-behaviour
@pytest.mark.parametrize("pick", PICKS)
def test_a_foreign_mtp_value_with_a_kv_arm_is_refused(pick):
    text = set_kv(arm_text("AF8", pick), pick, draft_kind="suffix")
    assert registry_states.classify_kv_arm(text, FIXTURE, pick) is None


# ---- arm values: one of the five rows EXACTLY, integer widths only (Sol r1-2, r1-3)
# class: new-behaviour
@pytest.mark.parametrize("pick", PICKS)
@pytest.mark.parametrize(
    "bits, scheme",
    [
        (4.5, "turboquant"),
        (4.5, "uniform"),
        (8.5, "turboquant"),
        (3.5, "turboquant"),
        (0.5, "turboquant"),
        (6, "turboquant"),
        (6, "uniform"),
        (2, "turboquant"),
        (3, "uniform"),
        (16, "uniform"),
        (-4, "turboquant"),
        (8, "affine"),
        (4, "Uniform"),
        (4, "TURBOQUANT"),
        (4, "mse"),
        (8, "''"),
        (8, "null"),
    ],
)
def test_widths_and_schemes_outside_the_five_rows_are_refused(pick, bits, scheme):
    text = set_kv(FIXTURE, pick, kv_bits=bits, kv_quant_scheme=scheme)
    assert registry_states.classify_kv_arm(text, FIXTURE, pick) is None


# class: new-behaviour
@pytest.mark.parametrize("pick", PICKS)
@pytest.mark.parametrize(
    "bits", (".inf", "-.inf", ".nan", "+.inf", "'8'", "true", "[8]", "{a: 8}", "null")
)
def test_non_integer_non_finite_or_mistyped_widths_are_refused_without_raising(
    pick, bits
):
    for scheme in ("uniform", "turboquant"):
        text = set_kv(FIXTURE, pick, kv_bits=bits, kv_quant_scheme=scheme)
        assert registry_states.classify_kv_arm(text, FIXTURE, pick) is None, (
            bits,
            scheme,
        )


# class: new-behaviour
@pytest.mark.parametrize("pick", PICKS)
@pytest.mark.parametrize("scheme", ("[uniform]", "{a: b}", "7", ".inf", "true"))
def test_mistyped_schemes_are_refused_without_raising(pick, scheme):
    text = set_kv(FIXTURE, pick, kv_bits=8, kv_quant_scheme=scheme)
    assert registry_states.classify_kv_arm(text, FIXTURE, pick) is None


# class: new-behaviour
@pytest.mark.parametrize("pick", PICKS)
def test_a_non_finite_quantized_kv_start_is_refused_without_raising(pick):
    for start in (".inf", ".nan", "-.inf"):
        text = set_kv(arm_text("AF8", pick), pick, quantized_kv_start=start)
        assert registry_states.classify_kv_arm(text, FIXTURE, pick) is None


# class: new-behaviour
def test_a_non_finite_value_in_an_unrelated_key_is_refused_without_raising():
    for value in (".inf", ".nan"):
        text = set_kv(arm_text("AF8", PICK_A), PICK_A, prefill_step_size=value)
        assert registry_states.classify_kv_arm(text, FIXTURE, PICK_A) is None


# ---- robust None on malformed structure (Sol r1-3): never raises
MALFORMED = {
    "empty": "",
    "scalar": "just text",
    "top-level-list": "- a\n- b\n",
    "models-null-only": "models: [null]\n",
    "models-scalar": "models: 5\n",
    "models-mapping": "models: {a: b}\n",
    "models-string-items": "models: [a, b]\n",
    "no-models-key": "manager_port: 8000\n",
    "tab-garbage": "models:\n\t- name: x\n",
    "unclosed": "models: [\n",
    "cr-line-endings": FIXTURE.replace("\n", "\r\n", 3),
    "duplicate-top-key": FIXTURE + "\nmanager_port: 9\n",
    "appended-null-item": FIXTURE.rstrip("\n") + "\n  - null\n",
    "appended-string-item": FIXTURE.rstrip("\n") + "\n  - just-a-string\n",
    "appended-list-item": FIXTURE.rstrip("\n") + "\n  - [1, 2]\n",
    "appended-nameless-item": FIXTURE.rstrip("\n") + "\n  - {type: vision}\n",
    "appended-int-name": FIXTURE.rstrip("\n") + "\n  - name: 7\n",
}


# class: new-behaviour
@pytest.mark.parametrize("pick", PICKS)
@pytest.mark.parametrize("label", sorted(MALFORMED))
def test_malformed_input_returns_none_and_never_raises(pick, label):
    assert registry_states.classify_kv_arm(MALFORMED[label], FIXTURE, pick) is None


# class: new-behaviour
def test_a_registry_whose_models_list_has_a_null_item_in_the_middle_is_refused():
    lines = arm_text("AF8", UNIFORM).split("\n")
    start, _ = registry_states._span(lines, UNIFORM)
    lines.insert(start, "  - null")
    assert registry_states.classify_kv_arm("\n".join(lines), FIXTURE, UNIFORM) is None


# class: new-behaviour
def test_a_duplicate_entry_name_or_key_is_refused():
    dup_key = FIXTURE.replace("    kv_bits: 4\n", "    kv_bits: 4\n    kv_bits: 8\n", 1)
    assert dup_key != FIXTURE
    assert registry_states.classify_kv_arm(dup_key, FIXTURE, UNIFORM) is None
    lines = arm_text("AF8", UNIFORM).split("\n")
    start, end = registry_states._span(lines, UNIFORM)
    doubled = "\n".join(lines + lines[start:end])
    assert registry_states.classify_kv_arm(doubled, FIXTURE, UNIFORM) is None


# class: new-behaviour
def test_classify_itself_is_unchanged_a_kv_arm_is_outside_the_16_states():
    assert registry_states.classify(arm_text("AF8", UNIFORM), FIXTURE) is None
    assert registry_states.classify(arm_text("TQ8", PICK_A), FIXTURE) is None


# ================================================================== acceptance 0: the attention rule, pure
# class: reproduces-failure for pick A AF8 / AF4 (the baseline demands forced calls under fused_v1 whatever the cache);
#        invariant for every other cell
@pytest.mark.parametrize("pick, arm", CELLS, ids=CELL_IDS)
@pytest.mark.parametrize("mtp_on", (True, False), ids=("mtp-on", "mtp-off"))
def test_a_right_record_passes_the_full_request_path_for_every_arm_and_pick(
    pick, arm, mtp_on
):
    expected = qualify.expected_config(entry_for(arm, pick, mtp_on=mtp_on))
    rec = valid_record(expected, arm_manifest(arm), sdpa_forced=right_forced(pick, arm))
    assert qualify.request_problems(rec, expected) == []


# class: reproduces-failure for pick A AF8 / AF4 (they must pass with 0 forced calls); invariant for the rest
@pytest.mark.parametrize("arm", ("AF8", "AF4"))
def test_acceptance_0_affine_arms_on_fable_pass_with_zero_forced_sdpa_calls(arm):
    expected = qualify.expected_config(entry_for(arm, PICK_A))
    assert expected["attention_policy"] == "fused_v1"
    rec = valid_record(expected, arm_manifest(arm), sdpa_forced=0)
    assert qualify.request_problems(rec, expected) == []


# class: new-behaviour (a forced count on an affine arm is wrong: the quantized-attention path forces no SDPA)
@pytest.mark.parametrize("arm", ("AF8", "AF4"))
def test_affine_arms_on_fable_fail_with_forced_sdpa_calls(arm):
    expected = qualify.expected_config(entry_for(arm, PICK_A))
    rec = valid_record(expected, arm_manifest(arm), sdpa_forced=640)
    assert any("sdpa_forced" in p for p in qualify.request_problems(rec, expected))


# class: invariant
@pytest.mark.parametrize("arm", ("N16", "TQ8", "TQ4"))
def test_arms_with_a_native_leaf_on_fable_still_require_forced_calls(arm):
    expected = qualify.expected_config(entry_for(arm, PICK_A))
    assert has_native_leaf(arm_manifest(arm))
    for forced in (0, None):
        rec = valid_record(expected, arm_manifest(arm), sdpa_forced=forced)
        assert any(
            "sdpa_forced" in p for p in qualify.request_problems(rec, expected)
        ), forced
    good = valid_record(expected, arm_manifest(arm), sdpa_forced=640)
    assert qualify.request_problems(good, expected) == []


# class: invariant (the default attention policy never forces: today's rule for the committed TQ4 uniform pick, every arm)
@pytest.mark.parametrize("arm", tuple(ARMS))
def test_the_default_attention_pick_requires_no_forced_calls_in_every_arm(arm):
    expected = qualify.expected_config(entry_for(arm, UNIFORM))
    assert expected["attention_policy"] == "auto"
    rec = valid_record(expected, arm_manifest(arm), sdpa_forced=640)
    assert any("sdpa_forced" in p for p in qualify.request_problems(rec, expected))
    assert (
        qualify.request_problems(
            valid_record(expected, arm_manifest(arm), sdpa_forced=0), expected
        )
        == []
    )


# class: new-behaviour (iff: a TurboQuant arm whose template has NO native leaf — <= 2 top-level entries convert every KV layer)
def test_turboquant_without_a_native_leaf_requires_zero_forced_calls_under_fused_v1():
    template = [
        dict(path="0", kind="native", cls="KVCache", batch=False),
        dict(path="1", kind="native", cls="KVCache", batch=False),
    ]
    tq = [
        dict(
            path=t["path"],
            kind="tq",
            cls="TurboQuantKVCache",
            batch=False,
            bits=8.0,
            key_bits=8.0,
            value_bits=8.0,
            mode="mse",
        )
        for t in template
    ]
    expected = qualify.expected_config(entry_for("TQ8", PICK_A))
    assert (
        kvmanifest.manifest_problems(tq, template, qualify.expected_kv(expected)) == []
    )  # premise: no native leaf expected

    def rec(forced):
        r = valid_record(expected, tq, sdpa_forced=forced)
        r["template"] = template
        r["capacities_by_path"] = {"0": expected["prealloc"], "1": expected["prealloc"]}
        return r

    assert qualify.request_problems(rec(0), expected) == []
    assert any("sdpa_forced" in p for p in qualify.request_problems(rec(640), expected))


# ---- GLM r1-4: a missing manifest / template yields only the manifest problem
# class: reproduces-failure for forced=0 (fused_v1 + an affine arm: the baseline adds a spurious sdpa_forced problem next to the
#        manifest one); invariant for forced=640 (no problem to add either way)
@pytest.mark.parametrize("missing", ("manifest", "template", "both"))
@pytest.mark.parametrize("forced", (0, 640))
def test_a_missing_manifest_or_template_yields_only_the_manifest_problem(
    missing, forced
):
    expected = qualify.expected_config(entry_for("AF8", PICK_A))
    rec = valid_record(expected, arm_manifest("AF8"), sdpa_forced=forced)
    if missing in ("manifest", "both"):
        rec["manifest"] = None
    if missing in ("template", "both"):
        rec["template"] = None
    problems = qualify.request_problems(rec, expected)
    assert any("no leaf manifest" in p for p in problems), problems
    assert not any("sdpa_forced" in p for p in problems), problems


# class: reproduces-failure (same defect on an arm with a native leaf: the attention check must not run without a manifest)
def test_a_missing_manifest_on_an_n16_arm_does_not_add_a_forced_calls_problem_either():
    expected = qualify.expected_config(entry_for("N16", PICK_A))
    rec = valid_record(expected, arm_manifest("N16"), sdpa_forced=0)
    rec["manifest"] = None
    problems = qualify.request_problems(rec, expected)
    assert [p for p in problems if "sdpa_forced" in p] == []
    assert any("no leaf manifest" in p for p in problems)


# class: invariant (today's t9: with the manifest present the committed pick A record still needs forced calls)
def test_a_present_manifest_keeps_todays_no_flag_attention_rule_for_the_committed_fable_entry():
    expected = qualify.expected_config(t415.PICK_A_ENTRY)
    assert qualify.request_problems(t415._record(), expected) == []
    problems = qualify.request_problems(t415._record(sdpa_forced=0), expected)
    assert any("sdpa_forced" in p for p in problems)


# ================================================================== the expected manifest follows the ARM registry
# class: invariant
@pytest.mark.parametrize("pick, arm", CELLS, ids=CELL_IDS)
def test_the_expected_manifest_is_derived_from_the_arm_registry(pick, arm):
    expected = qualify.expected_config(entry_for(arm, pick))
    kv = qualify.expected_kv(expected)
    template = t415._template()
    assert kvmanifest.manifest_problems(arm_manifest(arm), template, kv) == []
    for other in ARMS:
        if other != arm:
            assert (
                kvmanifest.manifest_problems(arm_manifest(other), template, kv) != []
            ), (arm, other)


# class: invariant
@pytest.mark.parametrize("pick, arm", CELLS, ids=CELL_IDS)
def test_the_arm_widths_and_scheme_reach_expected_config(pick, arm):
    expected = qualify.expected_config(entry_for(arm, pick))
    bits, scheme = ARMS[arm]
    assert expected["kv_bits"] == (bits or None)
    assert expected["kv_quant_scheme"] == (
        scheme or FIXTURE_KV[pick]["kv_quant_scheme"]
    )
    assert expected["quantized_kv_start"] == 0


# ---- Sol r1-2: expected_config keeps the exact numeric width
# class: reproduces-failure
def test_expected_config_keeps_a_fractional_width_exactly():
    entry = dict(t415.SYNTH_S_ENTRY, kv_bits=4.5)
    assert qualify.expected_config(entry)["kv_bits"] == 4.5
    assert qualify.expected_kv(qualify.expected_config(entry))["kv_bits"] == 4.5


# class: invariant (integer widths keep their spelling: the reference fingerprint says "turboquant4", never "turboquant4.0")
@pytest.mark.parametrize("bits", (4, 8))
def test_integer_widths_keep_their_integer_spelling(bits):
    expected = qualify.expected_config(dict(t415.SYNTH_S_ENTRY, kv_bits=bits))
    assert expected["kv_bits"] == bits
    assert str(expected["kv_bits"]) == str(bits)
    assert qualify.fingerprint_of({}, expected)["kv_codec"] == f"turboquant{bits}"
    assert (
        qualify.expected_config(dict(t415.SYNTH_S_ENTRY, kv_bits=0))["kv_bits"] is None
    )


# class: reproduces-failure (served as TQ4.5, the live manifest must be judged against TQ4.5, not TQ4)
def test_a_fractional_turboquant_width_is_judged_against_the_width_it_serves():
    expected = qualify.expected_config(dict(t415.SYNTH_S_ENTRY, kv_bits=4.5))
    manifest = t415._native_manifest()
    manifest[1] = dict(
        path="1",
        kind="tq",
        cls="TurboQuantKVCache",
        batch=False,
        bits=4.5,
        key_bits=4.0,
        value_bits=5.0,
        mode="mse",
    )
    rec = valid_record(expected, manifest, sdpa_forced=0)
    assert qualify.request_problems(rec, expected) == []
    manifest4 = (
        t415._tq_manifest()
    )  # the TQ4 manifest the int() truncation would expect
    assert (
        qualify.request_problems(
            valid_record(expected, manifest4, sdpa_forced=0), expected
        )
        != []
    )


# ================================================================== the spec's own premises about the classifier's inputs
# class: invariant
def test_the_five_arm_rows_are_all_distinct_manifests_on_the_test_template():
    seen = [str(arm_manifest(a)) for a in ARMS]
    assert len(set(seen)) == 5
    assert all(math.isfinite(ARMS[a][0]) for a in ARMS)
