"""Registry-integrity tests.

These exist because a dropped *dict entry* is invisible to both upstream audits:
`dev/check_upstream_parity.py` only sees missing files and
`dev/check_upstream_symbols.py` only sees missing `def`/`class` names, while a
registry is a module-level `ast.Assign`. Four `MODEL_REMAPPING` entries were lost
in a merge and two model families ("unlimited-ocr", "inkling_mm_model") failed
outright with "Model type X not supported" while their implementations sat in the
tree byte-identical to upstream. The only way to catch that shape is to exercise
the entries.

Deliberately a fork-only file rather than an addition to `tests/test_utils.py`,
which at the time was excluded from the suite (`--ignore=tests/test_utils.py`, 5
pre-existing failures) so a guard placed there would never actually run.

**[correction 2026-08-10]** That exclusion is gone — `test_utils.py` has been
collected and green since 2026-08-09, and its 5 "pre-existing failures" turned out
to be stale test code rather than product bugs. The sentence above stated it as a
present fact, which would have sent the next reader looking for an `--ignore` that
no longer exists. Keeping this file separate is still right, but for the reason in
the first paragraph (registries need exercising, and a fork-only file is where
fork-only guards belong) and not because of a suite exclusion.
"""

import importlib


def test_model_remapping_targets_are_all_importable():
    from mlx_vlm.utils import MODEL_REMAPPING

    broken = {}
    for alias, target in MODEL_REMAPPING.items():
        try:
            importlib.import_module(f"mlx_vlm.models.{target}")
        except ImportError as e:  # pragma: no cover - only on a real regression
            broken[alias] = f"{target}: {e}"

    assert not broken, f"MODEL_REMAPPING targets not importable: {broken}"


def test_model_remapping_covers_the_four_entries_lost_in_a_merge():
    """Pin the specific aliases that were dropped, so a re-drop names itself."""
    from mlx_vlm.utils import MODEL_REMAPPING

    for alias, target in (
        ("unlimited-ocr", "unlimited_ocr"),
        ("mistral", "llama"),
        ("nemotron-nas", "nemotron_nas"),
        ("inkling_mm_model", "inkling"),
    ):
        assert MODEL_REMAPPING.get(alias) == target, f"{alias} lost again"


def test_drafter_registry_kinds_are_all_known():
    """Every model_type in the drafter registry must map to a known kind.

    Same blind spot as MODEL_REMAPPING -- `glm4_moe_lite_mtp` and `inkling_mtp`
    were previously found missing from this registry, which silently disables
    drafter auto-detection rather than failing.

    Note this deliberately does NOT import a module per key: the registry maps
    *target model types* to kinds, and the drafter packages are named separately
    (`laguna` -> `laguna_dflash`), so key-to-module import checking would assert a
    convention that does not hold.
    """
    from mlx_vlm.speculative.drafters import (
        DRAFTER_KIND_BY_MODEL_TYPE,
        KNOWN_DRAFTER_KINDS,
    )

    bad = {
        mt: kind
        for mt, kind in DRAFTER_KIND_BY_MODEL_TYPE.items()
        if kind not in KNOWN_DRAFTER_KINDS
    }
    assert not bad, f"registry maps to unknown drafter kinds: {bad}"
