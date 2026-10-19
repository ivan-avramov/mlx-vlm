"""Every test starts from a fixed random stream derived from its node id.

Fork-only file (conftest autouse fixture): each test/param gets its own stream, and
the stream is identical across runs and independent of suite order.
"""

import random

import mlx.core as mx
import numpy as np
import pytest
from conftest import seed_for


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
