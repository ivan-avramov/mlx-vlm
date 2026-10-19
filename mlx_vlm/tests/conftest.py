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

# Fork: this docstring is fork-only -- upstream's conftest.py (#1846) is just the
TF32 pin below it, with no docstring. This file is the union of the two.
"""

from __future__ import annotations

import os
import pathlib  # Fork: needed by the collection config below; upstream imports only os.

# float32 matmul runs at TF32 precision on hardware with matrix units, which is
# looser than the float32 references these tests compare against. Set rather than
# setdefault, so an inherited MLX_ENABLE_TF32=1 doesn't leak into the test run.
os.environ["MLX_ENABLE_TF32"] = "0"


def pytest_sessionfinish(session, exitstatus):
    """Release thread-local MLX resources before Python finalization."""
    del session, exitstatus

    import mlx.core as mx

    clear_streams = getattr(mx, "clear_streams", None)
    if clear_streams is not None:
        clear_streams()


# Fork additions below this line

import pytest  # noqa: E402  Fork: autouse determinism fixture below


def seed_for(nodeid: str) -> int:
    """A fixed 32-bit seed per test node id (identical across runs and orders)."""
    import hashlib

    return int.from_bytes(hashlib.sha256(nodeid.encode()).digest()[:4], "big")


def _seed_all(seed: int) -> None:
    import random

    import mlx.core as mx
    import numpy as np

    mx.random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def fixture_seed(scope: str, nodeid: str, argname: str) -> int:
    """The seed a non-function-scoped fixture's setup runs under."""
    return seed_for(f"{scope}:{nodeid}:{argname}")


class _FixtureSeeding:
    """Seed the mx / numpy / python streams before every class-, module-, package-
    or session-scoped fixture's setup, from (scope, node id, fixture name), so a
    shared fixture's draws do not depend on which test happened to request it first.

    Registered as a plugin (``pytest_configure``): hooks defined in a conftest are
    path-scoped and a session-scoped fixture's hook proxy is the rootdir's, which
    would not include this directory's conftest. The scope is the EFFECTIVE one
    (``request.scope``): a parametrize ``scope=`` overrides the declared one.
    """

    @pytest.hookimpl(tryfirst=True)
    def pytest_fixture_setup(self, fixturedef, request):
        if request.scope != "function":
            _seed_all(
                fixture_seed(request.scope, request.node.nodeid, fixturedef.argname)
            )


def _register_fixture_seeding(config):
    if not config.pluginmanager.has_plugin("mlx_vlm_fixture_seeding"):
        config.pluginmanager.register(_FixtureSeeding(), "mlx_vlm_fixture_seeding")


@pytest.fixture(autouse=True)
def _deterministic_random_state(request):
    """Every test (and parametrization) starts from its own fixed random stream.

    Seeded, all from a fixed hash of the node id: the start of each test function
    (after any higher-scoped fixtures) and the setup of every class-, module-,
    package- and session-scoped fixture (``pytest_fixture_setup``). Unseeded draws
    made results depend on suite order (a tolerance-edge test flaked in ~25% of
    runs); the operator requires zero nondeterministic tests. NOT seeded: draws at
    import / collection time, and anything a test spawns on other threads or
    processes. A test that wants another stream seeds itself.
    """
    _seed_all(seed_for(request.node.nodeid))
    yield


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


def pytest_configure(config):  # M58 D5
    _register_fixture_seeding(config)  # P369 round-6 F25
    config.addinivalue_line(
        "markers",
        "requires_stack_registry: opt-in check of the stack's LIVE main_models.yaml against the pinned "
        "fixture (P387): skipped unless MLX_REQUIRE_STACK_REGISTRY=1, which makes a missing registry or "
        "any state other than the fixture / its C177 interim a failure",
    )
