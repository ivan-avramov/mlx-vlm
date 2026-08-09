"""Tests for BatchQuantizedKVCache — batch-aware quantized KV cache."""

import mlx.core as mx
import pytest

from mlx_vlm.models.cache import (  # Fork: ArraysCache, BatchRotatingKVCache and QuantizedKVCache are imported for; the fork-only contract classes below the boundary at the bottom of this file.
    ArraysCache,
    BatchKVCache,
    BatchQuantizedKVCache,
    BatchRotatingKVCache,
    QuantizedKVCache,
    StaticPrefixKVCache,
    should_quantize_kv_layer,
)

B, H, D = 2, 4, 64  # batch, heads, head_dim
GROUP_SIZE = 32
BITS = 8


def _rand_kv(batch, seq_len):
    """Return random (keys, values) tensors."""
    k = mx.random.normal((batch, H, seq_len, D))
    v = mx.random.normal((batch, H, seq_len, D))
    return k, v


class TestUpdateAndFetch:
    def test_basic_insert(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(B, 5)
        qk, qv = cache.update_and_fetch(k, v)
        # Quantized state is a tuple of 3 arrays
        assert len(qk) == 3
        assert len(qv) == 3
        # Sequence dimension should match
        assert qk[0].shape[2] == 5
        assert cache._idx == 5

    def test_incremental_insert(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        k1, v1 = _rand_kv(B, 3)
        cache.update_and_fetch(k1, v1)
        k2, v2 = _rand_kv(B, 1)
        qk, qv = cache.update_and_fetch(k2, v2)
        assert qk[0].shape[2] == 4
        assert cache._idx == 4

    def test_offset_tracks_per_sequence(self):
        cache = BatchQuantizedKVCache([2, 0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(B, 5)
        cache.update_and_fetch(k, v)
        offsets = cache.offset.tolist()
        # offset starts at [-2, 0] and adds 5
        assert offsets == [3, 5]


class TestFilter:
    def test_filter_keeps_correct_sequences(self):
        cache = BatchQuantizedKVCache([0, 0, 0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(3, 4)
        cache.update_and_fetch(k, v)
        mx.eval(cache.keys)

        cache.filter(mx.array([0, 2], mx.int32))
        # Should have 2 sequences left
        assert cache.keys[0].shape[0] == 2
        assert cache.offset.shape[0] == 2

    def test_filter_removes_common_left_padding(self):
        cache = BatchQuantizedKVCache([3, 1], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(B, 6)
        cache.update_and_fetch(k, v)
        mx.eval(cache.keys)

        # Keep only second sequence (left_padding=1)
        cache.filter(mx.array([1], mx.int32))
        # min left_padding=1, so it should shift left by 1
        assert cache.left_padding.tolist() == [0]
        assert cache._idx == 5  # 6 - 1

    def test_filter_single_sequence(self):
        cache = BatchQuantizedKVCache([0, 0, 0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(3, 2)
        cache.update_and_fetch(k, v)
        mx.eval(cache.keys)

        cache.filter(mx.array([1], mx.int32))
        assert cache.keys[0].shape[0] == 1


class TestExtend:
    def test_extend_concatenates_batches(self):
        c1 = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        k1, v1 = _rand_kv(2, 4)
        c1.update_and_fetch(k1, v1)
        mx.eval(c1.keys)

        c2 = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        k2, v2 = _rand_kv(1, 4)
        c2.update_and_fetch(k2, v2)
        mx.eval(c2.keys)

        c1.extend(c2)
        assert c1.keys[0].shape[0] == 3
        assert c1.offset.shape[0] == 3
        assert c1.left_padding.shape[0] == 3

    def test_extend_handles_different_lengths(self):
        c1 = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        k1, v1 = _rand_kv(1, 8)
        c1.update_and_fetch(k1, v1)
        mx.eval(c1.keys)

        c2 = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        k2, v2 = _rand_kv(1, 3)
        c2.update_and_fetch(k2, v2)
        mx.eval(c2.keys)

        c1.extend(c2)
        # max_idx should be 8, the shorter one gets right-padded
        assert c1._idx == 8
        assert c1.keys[0].shape[0] == 2
        assert c1.left_padding.shape[0] == 2

    def test_extend_handles_filtered_non_step_aligned_capacity(self):
        c1 = BatchQuantizedKVCache([7, 7], group_size=GROUP_SIZE, bits=BITS)
        k1, v1 = _rand_kv(2, 512)
        c1.update_and_fetch(k1, v1)
        mx.eval(c1.keys)

        # Filtering rows can trim common left padding and leave a backing
        # sequence length that is no longer aligned to the allocation step.
        c1.filter(mx.array([0], mx.int32))
        mx.eval(c1.keys)
        assert c1.keys[0].shape[-2] == 505
        assert c1._idx == 505

        c2 = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        k2, v2 = _rand_kv(1, 500)
        c2.update_and_fetch(k2, v2)
        mx.eval(c2.keys)
        assert c2.keys[0].shape[-2] == 512
        assert c2._idx == 500

        c1.extend(c2)
        mx.eval(c1.keys)

        assert c1.keys[0].shape[0] == 2
        assert c1.keys[0].shape[-2] == 512
        assert c1._idx == 505
        assert c1.left_padding.tolist() == [0, 5]

    def test_extend_empty_into_populated(self):
        c1 = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        k1, v1 = _rand_kv(1, 4)
        c1.update_and_fetch(k1, v1)
        mx.eval(c1.keys)

        c2 = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        c1.extend(c2)
        # Should still have 2 entries (1 populated + 1 empty offset)
        assert c1.offset.shape[0] == 2

    def test_extend_into_empty(self):
        c1 = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        c2 = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        k2, v2 = _rand_kv(1, 4)
        c2.update_and_fetch(k2, v2)
        mx.eval(c2.keys)

        c1.extend(c2)
        assert c1._idx == 4
        assert c1.keys is not None
        assert c1.keys[0].shape[0] == 2
        assert c1.offset.shape[0] == 2


class TestState:
    def test_state_roundtrip(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(B, 4)
        cache.update_and_fetch(k, v)
        mx.eval(cache.keys)

        state = cache.state
        assert len(state) == 4  # keys, values, offset, left_padding

        cache2 = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        cache2.state = state
        assert cache2._idx == 4

    def test_empty_state(self):
        cache = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        state = cache.state
        assert state[0] is None
        assert state[1] is None


class TestMakeMask:
    def test_make_mask_matches_batch_kv_cache_with_left_padding(self):
        left_padding = [2, 0]
        cache = BatchQuantizedKVCache(left_padding, group_size=GROUP_SIZE, bits=BITS)
        reference = BatchKVCache(left_padding)
        k, v = _rand_kv(B, 5)

        cache.update_and_fetch(k, v)
        reference.update_and_fetch(k, v)

        mask = cache.make_mask(2, return_array=True, window_size=None)
        reference_mask = reference.make_mask(2, return_array=True, window_size=None)

        assert mask.shape == reference_mask.shape
        assert mx.all(mask == reference_mask).item()


class TestPrepareFinalize:
    """Multi-row right-pad lifecycle parity with BatchKVCache (#1567 / #1562)."""

    def test_prepare_finalize_methods_exist(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        assert callable(getattr(cache, "prepare", None))
        assert callable(getattr(cache, "finalize", None))

    def test_prepare_stores_right_padding(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        cache.prepare(right_padding=[3, 0])
        assert cache._right_padding is not None
        assert cache._right_padding.tolist() == [3, 0]

    def test_finalize_updates_left_padding_like_batch_kv(self):
        right_padding = [2, 0]
        quant = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        ref = BatchKVCache([0, 0])

        quant.prepare(right_padding=right_padding)
        ref.prepare(right_padding=right_padding)

        k, v = _rand_kv(B, 6)
        quant.update_and_fetch(k, v)
        ref.update_and_fetch(k, v)

        quant.finalize()
        ref.finalize()

        assert quant._right_padding is None
        assert quant.left_padding.tolist() == ref.left_padding.tolist()
        assert quant.offset.tolist() == ref.offset.tolist()

    def test_finalize_noop_without_prepare(self):
        cache = BatchQuantizedKVCache([1, 0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(B, 4)
        cache.update_and_fetch(k, v)
        before = cache.left_padding.tolist()
        cache.finalize()
        assert cache.left_padding.tolist() == before


class TestShouldQuantizeKvLayerPolicy:
    """Shared last-layer quant policy used by _make_cache / APC warm / stream."""

    def test_shallow_stack_quantizes_all(self):
        assert should_quantize_kv_layer(0, 1) is True
        assert should_quantize_kv_layer(0, 2) is True
        assert should_quantize_kv_layer(1, 2) is True

    def test_deep_stack_skips_last(self):
        n = 4
        assert [should_quantize_kv_layer(i, n) for i in range(n)] == [
            True,
            True,
            True,
            False,
        ]

    def test_deep_stack_boundary(self):
        assert should_quantize_kv_layer(26, 28) is True
        assert should_quantize_kv_layer(27, 28) is False


class TestMakeCache:
    """Test that _make_cache creates BatchQuantizedKVCache when kv_bits is set."""

    def test_make_cache_with_kv_bits(self):
        from mlx_vlm.generate import _make_cache
        from mlx_vlm.models.cache import BatchQuantizedKVCache as BQKV

        class FakeLayer:
            pass

        class FakeModel:
            layers = [FakeLayer() for _ in range(4)]

        caches = _make_cache(FakeModel(), [0, 0], kv_bits=8, kv_group_size=64)
        # All but last should be quantized (model has >2 layers)
        assert isinstance(caches[0], BQKV)
        assert isinstance(caches[1], BQKV)
        assert isinstance(caches[2], BQKV)
        # Last layer should be unquantized
        assert isinstance(caches[3], BatchKVCache)

    def test_make_cache_without_kv_bits(self):
        from mlx_vlm.generate import _make_cache

        class FakeLayer:
            pass

        class FakeModel:
            layers = [FakeLayer() for _ in range(4)]

        caches = _make_cache(FakeModel(), [0, 0])
        for c in caches:
            assert isinstance(c, BatchKVCache)

    def test_make_cache_uses_should_quantize_kv_layer_policy(self):
        from mlx_vlm.generate import _make_cache

        class FakeLayer:
            pass

        class FakeModelDeep:
            layers = [FakeLayer() for _ in range(4)]

        class FakeModelShallow:
            layers = [FakeLayer() for _ in range(2)]

        deep = _make_cache(FakeModelDeep(), [0], kv_bits=8, kv_group_size=64)
        assert [type(c).__name__ for c in deep] == [
            "BatchQuantizedKVCache",
            "BatchQuantizedKVCache",
            "BatchQuantizedKVCache",
            "BatchKVCache",
        ]
        shallow = _make_cache(FakeModelShallow(), [0], kv_bits=8, kv_group_size=64)
        assert all(type(c).__name__ == "BatchQuantizedKVCache" for c in shallow)


class TestStaticPrefixKVCache:
    def test_read_only_view_does_not_mutate_prefix(self):
        prefix_cache = StaticPrefixKVCache(max_size=8)
        prefix_keys = mx.ones((1, 1, 2, 2), dtype=mx.float32)
        prefix_values = mx.full((1, 1, 2, 2), 2.0, dtype=mx.float32)
        prefix_cache.update_and_fetch(prefix_keys, prefix_values)

        read_only_cache = StaticPrefixKVCache.from_prefix(prefix_cache)
        current_keys = mx.full((1, 1, 1, 2), 3.0, dtype=mx.float32)
        current_values = mx.full((1, 1, 1, 2), 4.0, dtype=mx.float32)
        keys, values = read_only_cache.update_and_fetch(current_keys, current_values)

        assert prefix_cache.offset == 2
        assert read_only_cache.offset == 2
        assert keys.shape == (1, 1, 3, 2)
        assert values.shape == (1, 1, 3, 2)
        assert bool(mx.all(keys[..., :2, :] == 1.0).item())
        assert bool(mx.all(keys[..., 2:, :] == 3.0).item())
        assert bool(mx.all(values[..., :2, :] == 2.0).item())
        assert bool(mx.all(values[..., 2:, :] == 4.0).item())


class TestBatchGeneratorIntegration:
    """Test that BatchGenerator accepts and propagates kv_bits."""

    def test_batch_generator_accepts_kv_params(self):
        from unittest.mock import Mock

        from mlx_vlm.generate import BatchGenerator

        model = Mock()
        model.layers = [Mock() for _ in range(2)]
        proc = Mock()
        proc.tokenizer = Mock()
        proc.tokenizer.stopping_criteria = Mock()
        proc.tokenizer.stopping_criteria.add_eos_token_ids = Mock()

        gen = BatchGenerator(
            model, proc, kv_bits=4, kv_group_size=64, quantized_kv_start=100
        )
        assert gen.kv_bits == 4
        assert gen.kv_group_size == 64
        assert gen.quantized_kv_start == 100

    def test_batch_generator_default_no_quantization(self):
        from unittest.mock import Mock

        from mlx_vlm.generate import BatchGenerator

        model = Mock()
        model.layers = [Mock() for _ in range(2)]
        proc = Mock()
        proc.tokenizer = Mock()
        proc.tokenizer.stopping_criteria = Mock()
        proc.tokenizer.stopping_criteria.add_eos_token_ids = Mock()

        gen = BatchGenerator(model, proc)
        assert gen.kv_bits is None


# ---------------------------------------------------------------------------
# Fork additions below this line.
#
# Everything above is vendored from upstream mlx-vlm and should stay
# byte-identical so `git merge upstream/main` applies cleanly. The two classes
# below cover fork-only cache contracts (`dequantize_for_apc`, and the
# batch_size / is_single_row asymmetry `apc_adapters` relies on). Add fork tests
# here, not above.
# ---------------------------------------------------------------------------


class TestDequantizeForApcContract:
    """Behavioural contract for `dequantize_for_apc`, whoever provides it.

    `models/cache.py` used to graft this method onto mlx_lm's QuantizedKVCache
    behind a `hasattr` guard. Since the mlx-lm vendoring it is a plain method on
    our own vendored class, so these assertions are now a straightforward
    own. That guard cannot tell a compatible implementation from an incompatible
    one -- these tests can. They assert the contract APC relies on rather than
    our specific implementation, so they pass against either provider and fail
    behavioural contract rather than a defence against a dependency.

    Kept because the semantics matter to APC's block harvest; formerly this also
    to a different implementation; check its slicing/emptiness behaviour before
    assuming APC still stores correct K/V.
    """

    def test_method_is_available_on_quantized_kv_cache(self):
        assert hasattr(QuantizedKVCache, "dequantize_for_apc")

    def test_empty_cache_returns_none_pair(self):
        c = QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS)
        assert c.dequantize_for_apc() == (None, None)

    def test_returns_float_arrays_sliced_to_offset(self):
        c = QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(1, 8)
        c.update_and_fetch(k, v)
        dk, dv = c.dequantize_for_apc()
        assert dk is not None and dv is not None
        # Dense float, not the packed (uint32, scales, biases) triple.
        assert isinstance(dk, mx.array) and isinstance(dv, mx.array)
        assert dk.dtype != mx.uint32 and dv.dtype != mx.uint32
        # Sliced to the logical length, not the padded step capacity.
        assert dk.shape == (1, H, 8, D)
        assert dv.shape == (1, H, 8, D)

    def test_roundtrip_is_close_to_original(self):
        c = QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(1, 8)
        c.update_and_fetch(k, v)
        dk, dv = c.dequantize_for_apc()
        # 8-bit uniform quantization: loose tolerance, but must track the input.
        assert mx.allclose(dk, k, atol=0.1).item()
        assert mx.allclose(dv, v, atol=0.1).item()

    def test_batch_variant_slices_to_idx(self):
        c = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(B, 8)
        c.update_and_fetch(k, v)
        dk, dv = c.dequantize_for_apc()
        assert dk.shape == (B, H, 8, D)
        assert dv.shape == (B, H, 8, D)

    def test_batch_extract_yields_single_row_quantized_cache(self):
        c = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(B, 8)
        c.update_and_fetch(k, v)
        row = c.extract(1)
        assert isinstance(row, QuantizedKVCache)
        assert int(row.offset) == 8
        dk, _ = row.dequantize_for_apc()
        assert dk.shape == (1, H, 8, D)

    def test_batch_extract_honours_left_padding(self):
        c = BatchQuantizedKVCache([3, 0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(B, 8)
        c.update_and_fetch(k, v)
        # Row 0 has 3 pad tokens, so only 5 real positions survive extraction.
        assert int(c.extract(0).offset) == 5
        assert int(c.extract(1).offset) == 8

    def test_batch_extract_on_empty_cache_returns_empty(self):
        c = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        row = c.extract(0)
        assert isinstance(row, QuantizedKVCache)
        assert row.keys is None


class TestBatchSizeContract:
    """Contract for `batch_size` / `is_single_row`, whoever provides them.

    Same self-retiring graft arrangement as TestDequantizeForApcContract:
    BatchRotatingKVCache and ArraysCache are vendored now, so models/cache.py
    attaches these behind `hasattr` guards. These tests assert the behaviour APC
    row-normalization relies on, so they hold for either provider and fail if a
    declares these directly -- no graft, no version guard.
    """

    def _filled(self, cache, batch):
        k, v = _rand_kv(batch, 8)
        cache.update_and_fetch(k, v)
        return cache

    def test_dense_batch_caches_report_row_count(self):
        for cache in (
            self._filled(BatchKVCache(left_padding=[0] * B), B),
            self._filled(BatchRotatingKVCache(max_size=512, left_padding=[0] * B), B),
        ):
            assert cache.batch_size == B
            assert cache.is_single_row() is False

    def test_quantized_batch_cache_reports_row_count(self):
        # keys are a (packed, scales, biases) triple here, so batch_size must
        # index into keys[0] rather than keys.
        c = self._filled(
            BatchQuantizedKVCache([0] * B, group_size=GROUP_SIZE, bits=BITS), B
        )
        assert c.batch_size == B
        assert c.is_single_row() is False

    def test_single_row_is_detected(self):
        c = self._filled(BatchKVCache(left_padding=[0]), 1)
        assert c.batch_size == 1
        assert c.is_single_row() is True

    def test_empty_cache_falls_back_to_left_padding(self):
        # No keys yet: row count must still come from left_padding.
        assert BatchKVCache(left_padding=[0, 0, 0]).batch_size == 3
        assert (
            BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS).batch_size
            == 2
        )

    def test_arrays_cache_reports_batch_size(self):
        a = ArraysCache(4)
        assert a.batch_size == 1

    def test_arrays_cache_must_not_gain_is_single_row(self):
        # ArraysCache already has `extract`, and clone_cache_entry treats
        # "extract + is_single_row" as batch-shaped and recurses on extract(0).
        # Extracting an ArraysCache yields another ArraysCache, so granting it
        # is_single_row makes that recursion infinite. Upstream declares
        # batch_size on ArraysCache but deliberately not is_single_row.
        assert not hasattr(ArraysCache(4), "is_single_row")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
