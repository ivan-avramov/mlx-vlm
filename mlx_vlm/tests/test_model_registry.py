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


def test_generate_package_all_entries_all_resolve():
    """Every name `mlx_vlm.generate.__all__` promises must actually resolve.

    Same blind spot as `MODEL_REMAPPING` above, one level out: `__all__` is a
    module-level `ast.Assign`, so parity, symbols, deletions, fork-markers and
    dead-helpers are all blind to a name it lists but nothing binds. Nothing fails at
    import time either — the break surfaces only in a downstream
    `from mlx_vlm.generate import <name>`. A `Gemma4VideoProcessor` re-export was lost
    exactly this way.

    Note `importlib.import_module`: `mlx_vlm.generate` is the re-exported *function*,
    not the module, so `import mlx_vlm.generate as g` gives the function and
    `g.__all__` raises AttributeError. That is AGENTS.md trap 5, and writing this test
    the obvious way hits it.
    """
    package = importlib.import_module("mlx_vlm.generate")

    unresolvable = [n for n in package.__all__ if not hasattr(package, n)]

    assert unresolvable == [], unresolvable


def test_generation_stream_resolves_lazily_through_common():
    """`generation_stream` is in `__all__` but deliberately NOT bound at module level.

    Upstream binds it eagerly (`generation_stream = mx.new_thread_local_stream(...)` at
    `common.py` module scope). The fork cannot: a stream object created on one thread is
    not the one `mx.async_eval` resolves on another, which produced "no Stream(gpu, N)
    in current thread" on every worker-thread generation path (asyncio executors, the
    server's BatchGenerator GPU thread). So `common` exposes
    `_get_generation_stream()` and a module-level `__getattr__` maps the old public
    name onto it, deferring Metal stream creation to first use.

    **`common.__getattr__` is the mechanism, and establishing that took an experiment
    rather than a reading.** `generate/__init__.py` also carried an explicit
    `if name == "generation_stream"` branch whose comment claimed it was what kept the
    name working. Deleting that branch changed nothing — the `hasattr(common, name)`
    probe on the next line resolves it through `common.__getattr__` identically. The
    branch was redundant and its comment credited the wrong code, so it is gone. Two
    layers doing the same job are not defence in depth; they are duplication where one
    of the two comments is wrong.

    **What this test pins is the negative, and only the negative.** Deleting
    `common.__getattr__` cannot happen quietly — `generate/diffusion.py` does
    `from .common import generation_stream`, so the package fails at import and every
    test in the suite errors. That needs no guard. What nothing else catches is the
    *opposite* regression: someone "converging" `common.py` toward upstream by binding
    `generation_stream = mx.new_thread_local_stream(...)` eagerly at module scope. The
    import keeps working, the name resolves, the suite stays green, and the
    cross-thread bug is back on every worker-thread generation path. Verified by making
    exactly that edit: this assertion is the only thing in the suite that fails.
    """
    package = importlib.import_module("mlx_vlm.generate")
    common = importlib.import_module("mlx_vlm.generate.common")

    assert "generation_stream" in package.__all__
    assert "generation_stream" not in vars(package)
    assert "generation_stream" not in vars(common), (
        "generation_stream is bound eagerly again — that is upstream's form and it "
        "reintroduces the cross-thread stream bug; see mlx_vlm/generate/common.py"
    )
    # Resolution works from both, and both go through the same lazy path.
    assert common.generation_stream is not None
    assert package.generation_stream is not None
