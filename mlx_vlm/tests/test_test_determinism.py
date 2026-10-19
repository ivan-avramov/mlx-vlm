"""Every test starts from a fixed random state (conftest autouse fixture).

Fork-only file. The two tests draw first in an arbitrary order and must see the
same values from the seed-0 stream, so no draw depends on suite order.
"""

import random

import mlx.core as mx
import numpy as np

from conftest import TEST_RANDOM_SEED


def _reference():
    mx.random.seed(TEST_RANDOM_SEED)
    np.random.seed(TEST_RANDOM_SEED)
    random.seed(TEST_RANDOM_SEED)
    return (
        mx.random.normal((3,)).tolist(),
        np.random.normal(size=3).tolist(),
        random.random(),
    )


def test_first_draws_equal_the_seed_stream_first():
    mx_first = mx.random.normal((3,)).tolist()
    np_first = np.random.normal(size=3).tolist()
    py_first = random.random()
    assert (mx_first, np_first, py_first) == _reference()


def test_first_draws_equal_the_seed_stream_second():
    # burn draws, then require the NEXT test's start state to be reset again
    mx.random.normal((7,))
    np.random.normal(size=7)
    random.random()


def test_first_draws_equal_the_seed_stream_third():
    assert mx.random.normal((3,)).tolist() == _reference()[0]
