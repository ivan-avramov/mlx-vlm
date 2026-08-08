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
    # These two are a deliberate, permanent divergence rather than a backlog
    # item -- see docs/upstream-gaps.md. Upstream's quantized attention calls
    # mx.quantized_matmul and reshapes scores to 5D (B, n_kv_heads, n_repeats,
    # L, K), which is what makes right-aligning a 4D (B, 1, L, K) mask alias B
    # with n_kv_heads (upstream #1567) and require
    # align_attention_mask_to_scores. This fork instead dequantizes via
    # cache.dequantize() and runs dense scaled_dot_product_attention
    # (models/base.py:251), so scores are never 5D and the hazard these tests
    # cover is structurally impossible here. Porting the helper would be dead
    # code; porting the whole quantized-matmul path would be a
    # performance-motivated rewrite of our attention, not a bug fix.
    "test_quant_sdpa_mask.py": (
        "mlx_vlm.models.base.quantized_scaled_dot_product_attention",
        "fork dequantizes instead of using quantized_matmul, so the 5D "
        "GQA mask-aliasing bug these tests cover cannot occur -- divergence",
    ),
    "test_quant_sdpa_mask_adversarial.py": (
        "mlx_vlm.models.base.quantized_scaled_dot_product_attention",
        "same as test_quant_sdpa_mask.py -- divergence, not a backlog item",
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
