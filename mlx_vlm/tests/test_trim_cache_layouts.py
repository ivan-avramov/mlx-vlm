"""Tests for `_trim_cache`'s uncovered half and `_kv_seq_axis`'s layout disambiguation.

Fork-only. Found by a real coverage run (`dev/fork_coverage_report.py`), not by
reference counting: `_trim_cache` was the worst-covered fork-only definition in the tree
at **30 of 58 statements**, and `_kv_seq_axis` never returned 1 anywhere in 2955 tests.
Between them that is the entire `[B, L, H, D]` cache layout convention and the whole
chunked-list trimming branch — 24 consecutive statements that no test executed.

Why this is the one worth closing rather than the biggest number: `_trim_cache` runs on
the prompt-cache prefix-reuse path, and `common.py`'s own comment on it says a physical
slice of the wrong thing is *"silent wrong output, or a broadcast crash on the next
update."* Untested code whose failure mode is silent wrong output is the worst
combination available.

Nothing here is a bug fix. The branches turned out correct; these pin them, including
one documented heuristic (`_kv_seq_axis`'s equal-dims tie-break) that is a guess by
construction and should not be changed accidentally.
"""

import mlx.core as mx
import pytest

from mlx_vlm.generate.common import _kv_seq_axis, _trim_cache


class TestKvSeqAxis:
    """MLX's convention is `[B, H, L, D]` (axis 2); some models use `[B, L, H, D]`
    (axis 1). Everything downstream slices on whichever this returns."""

    def test_fewer_than_three_dims_is_axis_one(self):
        assert _kv_seq_axis((4, 8)) == 1

    def test_a_longer_second_dim_means_the_sequence_is_axis_one(self):
        # [B=1, L=128, H=8, D=64]
        assert _kv_seq_axis((1, 128, 8, 64)) == 1

    def test_a_longer_third_dim_means_the_sequence_is_axis_two(self):
        # [B=1, H=8, L=128, D=64] — the standard MLX layout
        assert _kv_seq_axis((1, 8, 128, 64)) == 2

    def test_equal_dims_default_to_the_standard_layout(self):
        """A documented guess, pinned because it IS a guess.

        With `shape[1] == shape[2]` the layout is genuinely ambiguous — a model with
        as many heads as cached tokens is indistinguishable either way — and the helper
        picks axis 2 because that is MLX's convention. Changing this silently would
        transpose every slice for such a cache.
        """
        assert _kv_seq_axis((1, 32, 32, 64)) == 2


class _Chunked:
    """A cache whose state is a LIST of per-chunk arrays.

    Deliberately not a real `ChunkedKVCache`: `_trim_cache` dispatches on cache class
    NAMES through the MRO (`_cache_kind_names`) and skips several, so a fake with a
    neutral name is what reaches the chunked-list branch. It also has no `trim` or
    `truncate`, which is what exercises the bare-`offset` fallback on the way in.
    """

    def __init__(self, chunks, offset, seq_axis=2, heads=2, dim=4):
        self.keys = [self._arr(n, seq_axis, heads, dim) for n in chunks]
        self.values = [self._arr(n, seq_axis, heads, dim) for n in chunks]
        self.offset = offset

    @staticmethod
    def _arr(length, seq_axis, heads, dim):
        shape = (1, heads, length, dim) if seq_axis == 2 else (1, length, heads, dim)
        return mx.zeros(shape)

    def lengths(self, seq_axis=2):
        return [k.shape[seq_axis] for k in self.keys]


class TestTrimCacheChunkedList:
    """The 24 statements no test executed."""

    def test_whole_chunks_that_fit_are_kept(self):
        cache = _Chunked([4, 4, 4], offset=12)

        _trim_cache(cache, 8)

        assert cache.lengths() == [4, 4]
        assert cache.offset == 8

    def test_a_straddling_chunk_is_sliced(self):
        """target 6 lands mid-way through the second chunk."""
        cache = _Chunked([4, 4, 4], offset=12)

        _trim_cache(cache, 6)

        assert cache.lengths() == [4, 2]

    def test_chunks_past_the_target_are_dropped_entirely(self):
        cache = _Chunked([4, 4, 4], offset=12)

        _trim_cache(cache, 4)

        assert cache.lengths() == [4]

    def test_keys_and_values_stay_in_lockstep(self):
        """They are sliced in the same loop, so a divergence here would desync K from
        V — the shape of failure that produces wrong attention rather than a crash."""
        cache = _Chunked([4, 4, 4], offset=12)

        _trim_cache(cache, 6)

        assert [k.shape for k in cache.keys] == [v.shape for v in cache.values]

    def test_the_axis_one_layout_slices_on_axis_one(self):
        """`[B, L, H, D]`. Slicing this on axis 2 would cut the HEAD dimension and
        leave the sequence length untouched — wrong output, no crash."""
        cache = _Chunked([8, 8], offset=16, seq_axis=1, heads=2, dim=4)

        _trim_cache(cache, 12)

        assert cache.lengths(seq_axis=1) == [8, 4]
        # heads and dim must be untouched
        assert all(k.shape[2] == 2 and k.shape[3] == 4 for k in cache.keys)

    def test_a_target_at_the_exact_chunk_boundary_keeps_no_partial(self):
        cache = _Chunked([5, 5], offset=10)

        _trim_cache(cache, 5)

        assert cache.lengths() == [5]

    def test_an_offset_already_at_the_target_is_left_alone(self):
        cache = _Chunked([4, 4], offset=8)

        _trim_cache(cache, 8)

        assert cache.lengths() == [4, 4]

    def test_an_offset_below_the_target_is_left_alone(self):
        cache = _Chunked([4], offset=4)

        _trim_cache(cache, 99)

        assert cache.lengths() == [4]
        assert cache.offset == 4


class TestTrimCacheFlatArray:
    class _Flat:
        def __init__(self, length, seq_axis=2, heads=2, dim=4):
            shape = (
                (1, heads, length, dim) if seq_axis == 2 else (1, length, heads, dim)
            )
            self.keys = mx.zeros(shape)
            self.values = mx.zeros(shape)
            self.offset = length

    def test_the_axis_two_layout_slices_the_sequence(self):
        cache = self._Flat(16)

        _trim_cache(cache, 6)

        assert cache.keys.shape == (1, 2, 6, 4)

    def test_the_axis_one_layout_slices_the_sequence(self):
        """The `seq_axis == 1` physical-slice branch, previously unexecuted."""
        cache = self._Flat(16, seq_axis=1)

        _trim_cache(cache, 6)

        assert cache.keys.shape == (1, 6, 2, 4)


class TestTrimCacheOffsetFallbacks:
    """How the offset gets moved when a cache does not expose `trim()`."""

    def test_truncate_is_used_when_trim_is_absent(self):
        class _Truncatable:
            def __init__(self):
                self.offset = 20
                self.truncated_to = None

            def truncate(self, target):
                self.truncated_to = target
                self.offset = target

        cache = _Truncatable()

        _trim_cache(cache, 8)

        assert cache.truncated_to == 8
        assert cache.offset == 8

    def test_trim_is_preferred_over_truncate(self):
        """`trim` takes an AMOUNT, `truncate` takes a TARGET — calling the wrong one
        with the other's argument silently trims to the wrong length."""

        class _Both:
            def __init__(self):
                self.offset = 20
                self.trimmed_by = None
                self.truncated_to = None

            def trim(self, amount):
                self.trimmed_by = amount
                self.offset -= amount

            def truncate(self, target):
                self.truncated_to = target

        cache = _Both()

        _trim_cache(cache, 8)

        assert cache.trimmed_by == 12  # 20 - 8, an amount and not a target
        assert cache.truncated_to is None

    def test_a_bare_offset_is_assigned_when_neither_exists(self):
        class _Bare:
            def __init__(self):
                self.offset = 20

        cache = _Bare()

        _trim_cache(cache, 8)

        assert cache.offset == 8

    def test_an_mx_array_offset_is_read_through_item(self):
        """Batch caches carry `offset` as a 0-d array; comparing that to an int
        directly would raise rather than trim."""

        class _ArrayOffset:
            def __init__(self):
                self.offset = mx.array(20)
                self.trimmed_by = None

            def trim(self, amount):
                self.trimmed_by = amount

        cache = _ArrayOffset()

        _trim_cache(cache, 8)

        assert cache.trimmed_by == 12


class TestTrimCacheSkipsAndGuards:
    def test_a_cache_without_key_value_attributes_returns_quietly(self):
        class _NoState:
            def __init__(self):
                self.offset = 20

            def trim(self, amount):
                self.offset -= amount

        cache = _NoState()

        _trim_cache(cache, 8)  # must not raise

        assert cache.offset == 8

    def test_none_state_returns_before_slicing(self):
        class _NoneState:
            def __init__(self):
                self.offset = 20
                self.keys = None
                self.values = None

            def trim(self, amount):
                self.offset -= amount

        cache = _NoneState()

        _trim_cache(cache, 8)  # must not raise on None

        assert cache.offset == 8
        assert cache.keys is None

    @pytest.mark.parametrize("container", [list, tuple])
    def test_containers_are_walked(self, container):
        a, b = _Chunked([4, 4], offset=8), _Chunked([4, 4], offset=8)

        _trim_cache(container([a, b]), 4)

        assert a.lengths() == [4]
        assert b.lengths() == [4]

    def test_a_wrapper_exposing_dot_caches_is_walked(self):
        class _Wrapper:
            def __init__(self, inner):
                self.caches = inner

        inner = [_Chunked([4, 4], offset=8)]

        _trim_cache(_Wrapper(inner), 4)

        assert inner[0].lengths() == [4]

    def test_a_quantized_cache_keeps_its_three_element_state(self):
        """The guard `common.py`'s comment is about: `QuantizedKVCache.keys` is
        `[packed, scales, biases]`, and the chunked-list branch would read those as
        three sequence chunks and strip scales+biases — breaking the next
        `mx.quantized_matmul` with a missing-argument TypeError rather than any visible
        cache error.
        """
        from mlx_vlm.models.cache import QuantizedKVCache

        cache = QuantizedKVCache()
        cache.update_and_fetch(mx.zeros((1, 2, 16, 64)), mx.zeros((1, 2, 16, 64)))
        before = len(cache.keys)

        _trim_cache(cache, 4)

        assert len(cache.keys) == before == 3
