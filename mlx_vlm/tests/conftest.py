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
#
# Empty as of the mlx-lm vendoring: adopting upstream's models/base.py brought
# `quantized_scaled_dot_product_attention` and `align_attention_mask_to_scores`
# with it, so the two quant-SDPA test files that lived here now run. They were
# previously described as a "deliberate, permanent divergence" on the grounds
# that this fork never builds 5D scores -- which was wrong twice over (see the
# [correction] in docs/upstream-gaps.md).
#
# Keep the mechanism: it is the definition of done for porting an upstream
# feature, and it fails loudly rather than silently skipping.
UNPORTED_UPSTREAM_TESTS: dict[str, tuple[str, str]] = {}


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
