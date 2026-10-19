"""Every test starts from a fixed random stream derived from its node id.

Fork-only file (conftest autouse fixture): each test/param gets its own stream, and
the stream is identical across runs and independent of suite order.
"""

import random

import mlx.core as mx
import numpy as np
import pytest
from conftest import fixture_seed, seed_for


def _draws(seed):
    mx.random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return (
        mx.random.normal((3,)).tolist(),
        np.random.normal(size=3).tolist(),
        random.random(),
    )


def test_first_draws_come_from_the_nodeid_seed(request):
    first = (
        mx.random.normal((3,)).tolist(),
        np.random.normal(size=3).tolist(),
        random.random(),
    )
    assert first == _draws(seed_for(request.node.nodeid))


@pytest.mark.parametrize("case", [0, 1])
def test_parametrizations_get_distinct_streams(request, case):
    mine = mx.random.normal((3,)).tolist()
    other = request.node.nodeid.replace(f"[{case}]", f"[{1 - case}]")
    assert mine == _draws(seed_for(request.node.nodeid))[0]
    assert mine != _draws(seed_for(other))[0]


def test_seed_for_is_stable():
    assert seed_for("a::b") == seed_for("a::b") != seed_for("a::c")
    assert 0 <= seed_for("a::b") < 2**32


@pytest.fixture(scope="module")
def _module_scoped_draw(request):
    # Drawn at MODULE-scoped setup (before any test body's per-test reseed).
    return mx.random.normal((3,)).tolist(), request.node.nodeid


def test_module_scoped_setup_is_seeded_from_scope_nodeid_and_name(_module_scoped_draw):
    draw, module_id = _module_scoped_draw
    want = _draws(fixture_seed("module", module_id, "_module_scoped_draw"))[0]
    assert draw == want


@pytest.fixture(scope="session")
def _session_scoped_draw(request):
    return mx.random.normal((3,)).tolist(), request.node.nodeid


def test_session_scoped_setup_is_seeded_even_when_requested_late(_session_scoped_draw):
    draw, session_id = _session_scoped_draw
    want = _draws(fixture_seed("session", session_id, "_session_scoped_draw"))[0]
    assert draw == want


@pytest.fixture
def _scope_overridden_draw(request):
    # Declared function-scoped; the parametrize below makes it MODULE-scoped.
    return mx.random.normal((3,)).tolist(), request.node.nodeid, request.scope


@pytest.mark.parametrize("_scope_overridden_draw", [0], indirect=True, scope="module")
def test_a_parametrize_scope_override_is_seeded_from_the_effective_scope(
    _scope_overridden_draw,
):
    """Round 7 F30: parametrization can override a fixture's declared scope; the
    setup is seeded from the scope it really runs under (``request.scope``)."""
    draw, node_id, scope = _scope_overridden_draw
    assert scope == "module"
    want = _draws(fixture_seed("module", node_id, "_scope_overridden_draw"))[0]
    assert draw == want
