"""Test collection config for the fork.

Why this file exists
--------------------
Sixteen upstream test files were silently dropped by an earlier merge (see
docs/upstream-gaps.md). They have now been restored **byte-identical to
upstream**, deliberately: keeping them unmodified means future `git merge
upstream/main` applies upstream's own edits to them cleanly, and
`dev/check_upstream_parity.py` can prove none has gone missing again.

Six of them import symbols this fork has never ported. Rather than editing those
files (which would create merge friction forever) or porting the symbols as dead
code (which would make the tests pass without testing anything real -- a false
green), collection is skipped here, in one reviewable place, with a reason each.

Removing an entry is the definition of done for porting that feature: delete the
line, and the upstream tests for it run as-is.
"""

from __future__ import annotations

import pathlib

# path -> (missing symbol, what porting it actually requires)
UNPORTED_UPSTREAM_TESTS: dict[str, tuple[str, str]] = {
    "test_apc_semantic_key.py": (
        "mlx_vlm.apc._hash_payload",
        "upstream's semantic-key hashing; this fork hashes via _hash_tokens/"
        "_hash_use_sha256 instead, so the key derivation differs by design",
    ),
    "test_apc_observability.py": (
        "mlx_vlm.apc.APCSelfCheckResult",
        "upstream's APC self-check subsystem, not ported",
    ),
    "test_apc_quantized.py": (
        "mlx_vlm.models.cache.should_quantize_kv_layer",
        "upstream's shared kv-quantization layer policy. Porting the helper "
        "alone would not test anything: upstream requires _make_cache, stream "
        "quantize and APC warm restore to all route through it, and this fork "
        "decides per-layer quantization inline",
    ),
    "test_quant_sdpa_mask.py": (
        "mlx_vlm.models.cache.dynamic_roll",
        "upstream's vendored rotating-cache roll helper; this fork delegates "
        "the base cache classes to mlx_lm",
    ),
    "test_quant_sdpa_mask_adversarial.py": (
        "mlx_vlm.models.cache.dynamic_roll",
        "same as test_quant_sdpa_mask.py",
    ),
    "test_minimax_m3.py": (
        "mlx_vlm.models.base.align_attention_mask_to_scores",
        "upstream's mask/score alignment helper, not ported",
    ),
}

_HERE = pathlib.Path(__file__).parent

collect_ignore = [name for name in UNPORTED_UPSTREAM_TESTS if (_HERE / name).exists()]


def pytest_report_collectionfinish() -> list[str]:
    """Say out loud what is being skipped, so it cannot rot silently."""
    if not collect_ignore:
        return []
    lines = [
        f"Skipping {len(collect_ignore)} restored upstream test file(s) that "
        "need unported symbols (docs/upstream-gaps.md):"
    ]
    for name in sorted(collect_ignore):
        symbol, _reason = UNPORTED_UPSTREAM_TESTS[name]
        lines.append(f"  - {name}  (needs {symbol})")
    return lines
