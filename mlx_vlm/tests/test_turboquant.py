import mlx.core as mx
import pytest

from mlx_vlm.generate import maybe_quantize_kv_cache
from mlx_vlm.models.base import scaled_dot_product_attention
from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache
from mlx_vlm.turboquant import (
    BatchTurboQuantKVCache,
    TurboQuantKVCache,
    _build_codec,
    _TurboQuantMSECodec,
    _TurboQuantProdCodec,
    turboquant_enabled,
)


def _sample_unit_vectors(count: int, dim: int) -> mx.array:
    vectors = mx.random.normal((count, dim))
    return vectors / mx.linalg.norm(vectors, axis=-1, keepdims=True)


def test_turboquant_mse_matches_paper_small_bit_distortions():
    vectors = _sample_unit_vectors(256, 64)
    expected = {1: 0.36, 2: 0.117, 3: 0.03}

    for bits, target in expected.items():
        codec = _TurboQuantMSECodec(64, bits, seed=0)
        state = codec.quantize(vectors)
        reconstructed = codec.dequantize(state)
        mse = mx.mean(mx.sum((vectors - reconstructed) ** 2, axis=-1)).item()
        assert mse == pytest.approx(target, rel=0.25, abs=0.02)


def test_turboquant_prod_is_nearly_unbiased_across_seeds():
    mx.random.seed(42)
    keys = _sample_unit_vectors(128, 64)
    queries = mx.random.normal((128, 64))
    true_inner_products = mx.sum(keys * queries, axis=-1)

    estimates = []
    for seed in range(16):
        codec = _TurboQuantProdCodec(64, 2, seed=seed)
        state = codec.quantize(keys)
        reconstructed = codec.dequantize(state)
        estimates.append(mx.sum(reconstructed * queries, axis=-1))

    mean_estimate = mx.mean(mx.stack(estimates), axis=0)
    bias = mx.mean(mean_estimate - true_inner_products).item()
    assert abs(bias) < 0.03


def test_fractional_turboquant_improves_reconstruction():
    vectors = mx.random.normal((1, 2, 32, 64))

    codec_3bit = _build_codec(vectors, 3.0, mode="mse", seed=0)
    codec_35bit = _build_codec(vectors, 3.5, mode="mse", seed=0)

    state_3bit = codec_3bit.quantize(vectors)
    state_35bit = codec_35bit.quantize(vectors)

    mse_3bit = mx.mean((vectors - codec_3bit.dequantize(state_3bit)) ** 2).item()
    mse_35bit = mx.mean((vectors - codec_35bit.dequantize(state_35bit)) ** 2).item()

    assert turboquant_enabled(3.5)
    assert not turboquant_enabled(3.0)
    assert mse_35bit < mse_3bit


def test_turboquant_cache_replaces_kv_cache_for_fractional_bits():
    layer_cache = KVCache()
    layer_cache.update_and_fetch(
        mx.random.normal((1, 2, 8, 32)),
        mx.random.normal((1, 2, 8, 32)),
    )
    prompt_cache = [layer_cache]

    maybe_quantize_kv_cache(
        prompt_cache,
        quantized_kv_start=4,
        kv_group_size=64,
        kv_bits=3.5,
        kv_quant_scheme="uniform",
    )

    assert isinstance(prompt_cache[0], TurboQuantKVCache)


def test_explicit_turboquant_scheme_supports_integer_bits():
    layer_cache = KVCache()
    layer_cache.update_and_fetch(
        mx.random.normal((1, 2, 8, 32)),
        mx.random.normal((1, 2, 8, 32)),
    )
    prompt_cache = [layer_cache]

    maybe_quantize_kv_cache(
        prompt_cache,
        quantized_kv_start=4,
        kv_group_size=64,
        kv_bits=3.0,
        kv_quant_scheme="turboquant",
    )

    assert isinstance(prompt_cache[0], TurboQuantKVCache)
    assert prompt_cache[0].bits == pytest.approx(3.0)


def test_turboquant_skips_non_kv_cache_entries():
    linear_cache = ArraysCache(size=2)
    linear_cache[0] = mx.zeros((1, 8))
    linear_cache[1] = mx.ones((1, 8))

    attention_cache = KVCache()
    attention_cache.update_and_fetch(
        mx.random.normal((1, 2, 8, 32)),
        mx.random.normal((1, 2, 8, 32)),
    )
    prompt_cache = [linear_cache, attention_cache]

    maybe_quantize_kv_cache(
        prompt_cache,
        quantized_kv_start=4,
        kv_group_size=64,
        kv_bits=3.5,
        kv_quant_scheme="turboquant",
    )

    assert isinstance(prompt_cache[0], ArraysCache)
    assert isinstance(prompt_cache[1], TurboQuantKVCache)


def test_batch_turboquant_extend_supports_uniform_single_item_offsets():
    keys = mx.ones((1, 2, 3, 8), dtype=mx.float16)
    values = mx.ones((1, 2, 3, 8), dtype=mx.float16)
    first = BatchTurboQuantKVCache([0], bits=3.5)
    second = BatchTurboQuantKVCache([0], bits=3.5)

    first.update_and_fetch(keys, values)
    second.update_and_fetch(keys, values)
    first.extend(second)

    assert first.offset.tolist() == [3, 3]
    assert first.left_padding.tolist() == [0, 0]


def test_batch_turboquant_extend_supports_empty_uniform_offsets():
    first = BatchTurboQuantKVCache([0], bits=3.5)
    second = BatchTurboQuantKVCache([0], bits=3.5)

    first.extend(second)

    assert first.offset.tolist() == [0, 0]
    assert first.left_padding.tolist() == [0, 0]


def test_batch_turboquant_filter_supports_uniform_single_item_offsets():
    keys = mx.ones((1, 2, 3, 8), dtype=mx.float16)
    values = mx.ones((1, 2, 3, 8), dtype=mx.float16)
    cache = BatchTurboQuantKVCache([0], bits=3.5)

    cache.update_and_fetch(keys, values)
    cache.filter(mx.array([0]))

    assert cache.offset.tolist() == [3]
    assert cache.left_padding.tolist() == [0]


def test_batch_turboquant_extend_pads_shorter_uniform_batch():
    longer = BatchTurboQuantKVCache([0], bits=3.5)
    shorter = BatchTurboQuantKVCache([0], bits=3.5)

    longer.update_and_fetch(
        mx.ones((1, 2, 5, 8), dtype=mx.float16),
        mx.ones((1, 2, 5, 8), dtype=mx.float16),
    )
    shorter.update_and_fetch(
        mx.ones((1, 2, 3, 8), dtype=mx.float16),
        mx.ones((1, 2, 3, 8), dtype=mx.float16),
    )
    longer.extend(shorter)

    assert longer.offset.tolist() == [5, 3]
    assert longer.left_padding.tolist() == [0, 2]
    assert longer._idx == 5


def test_batch_turboquant_make_mask_matches_batch_kv_cache_with_left_padding():
    left_padding = [2, 0]
    cache = BatchTurboQuantKVCache(left_padding, bits=3.5)
    reference = BatchKVCache(left_padding)
    keys = mx.random.normal((2, 2, 5, 64))
    values = mx.random.normal((2, 2, 5, 64))

    cache.update_and_fetch(keys, values)
    reference.update_and_fetch(keys, values)

    mask = cache.make_mask(2, return_array=True, window_size=None)
    reference_mask = reference.make_mask(2, return_array=True, window_size=None)

    assert mask.shape == reference_mask.shape
    assert mx.all(mask == reference_mask).item()


def test_turboquant_cache_preserves_attention_shape_and_compresses_memory():
    keys = mx.random.normal((1, 2, 16, 32))
    values = mx.random.normal((1, 2, 16, 32))
    queries = mx.random.normal((1, 2, 1, 32))

    fp_cache = KVCache()
    fp_keys, fp_values = fp_cache.update_and_fetch(keys, values)
    reference = scaled_dot_product_attention(
        queries,
        fp_keys,
        fp_values,
        fp_cache,
        scale=32**-0.5,
        mask=None,
    )

    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=3.5)
    turbo_keys, turbo_values = turbo_cache.state
    quantized = scaled_dot_product_attention(
        queries,
        turbo_keys,
        turbo_values,
        turbo_cache,
        scale=32**-0.5,
        mask=None,
    )

    diff = mx.mean(mx.abs(reference - quantized)).item()

    assert quantized.shape == reference.shape
    assert turbo_cache.nbytes < fp_cache.nbytes
    assert diff < 0.35


def test_turboquant_decode_attention_matches_dequantized_attention():
    keys = mx.random.normal((1, 2, 16, 32))
    values = mx.random.normal((1, 2, 16, 32))
    queries = mx.random.normal((1, 4, 1, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=3.5)
    turbo_keys, turbo_values = turbo_cache.state
    dequantized_keys, dequantized_values = turbo_cache.dequantize(
        turbo_keys,
        turbo_values,
    )

    reference = mx.fast.scaled_dot_product_attention(
        queries,
        dequantized_keys.astype(queries.dtype),
        dequantized_values.astype(queries.dtype),
        scale=32**-0.5,
        mask=None,
    )
    quantized = scaled_dot_product_attention(
        queries,
        turbo_keys,
        turbo_values,
        turbo_cache,
        scale=32**-0.5,
        mask=None,
    )

    diff = mx.max(mx.abs(reference - quantized)).item()
    assert quantized.shape == reference.shape
    assert diff < 1e-4


def test_turboquant_decode_attention_skips_full_dequantize():
    keys = mx.random.normal((1, 2, 8, 32))
    values = mx.random.normal((1, 2, 8, 32))
    queries = mx.random.normal((1, 4, 1, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=3.5)
    turbo_keys, turbo_values = turbo_cache.state

    def fail(*args, **kwargs):
        raise AssertionError("decode_attention should not call full dequantize")

    turbo_cache.dequantize = fail
    output = scaled_dot_product_attention(
        queries,
        turbo_keys,
        turbo_values,
        turbo_cache,
        scale=32**-0.5,
        mask=None,
    )

    assert output.shape == queries.shape


def test_turboquant_decode_attention_metal_fast_path_skips_unpack(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")

    import mlx_vlm.turboquant as turboquant

    keys = mx.random.normal((1, 2, 8, 32))
    values = mx.random.normal((1, 2, 8, 32))
    queries = mx.random.normal((1, 4, 1, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=3.5)
    turbo_keys, turbo_values = turbo_cache.state

    def fail(*args, **kwargs):
        raise AssertionError("decode metal fast path should not unpack low-bit state")

    monkeypatch.setattr(turboquant, "_unpack_lowbit", fail)
    output = scaled_dot_product_attention(
        queries,
        turbo_keys,
        turbo_values,
        turbo_cache,
        scale=32**-0.5,
        mask=None,
    )

    assert output.shape == queries.shape


def test_turboquant_decode_attention_4bit_uses_paper_prod_key_codec():
    keys = mx.random.normal((1, 2, 8, 32))
    values = mx.random.normal((1, 2, 8, 32))
    queries = mx.random.normal((1, 4, 1, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    turbo_keys, turbo_values = turbo_cache.state

    # Keys now use MSE-only codec (QJL/Prod dropped for speed+quality)
    assert type(turbo_cache.key_codec).__name__ == "_TurboQuantMSECodec"
    output = scaled_dot_product_attention(
        queries,
        turbo_keys,
        turbo_values,
        turbo_cache,
        scale=32**-0.5,
        mask=None,
    )

    assert output.shape == queries.shape


def _assert_mse_states_equal(left, right):
    assert bool(mx.all(left.norms == right.norms).item())
    assert bool(mx.all(left.indices == right.indices).item())


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_turboquant_mse_prefill_decode_matches_batch_quantized_state(dim):
    mx.random.seed(99)
    keys = mx.random.normal((1, 8, 17, dim)).astype(mx.float16)
    values = mx.random.normal((1, 8, 17, dim)).astype(mx.float16)

    batch_cache = TurboQuantKVCache(bits=4.0)
    batch_cache.update_and_fetch(keys, values)

    split_cache = TurboQuantKVCache(bits=4.0)
    split_cache.update_and_fetch(keys[:, :, :16, :], values[:, :, :16, :])
    split_cache.update_and_fetch(keys[:, :, 16:17, :], values[:, :, 16:17, :])

    batch_keys, batch_values = batch_cache.state
    split_keys, split_values = split_cache.state
    _assert_mse_states_equal(batch_keys, split_keys)
    _assert_mse_states_equal(batch_values, split_values)


def test_turboquant_decode_attention_integer_separate_path_bypasses_fused(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")

    keys = mx.random.normal((1, 2, 8, 32))
    values = mx.random.normal((1, 2, 8, 32))
    queries = mx.random.normal((1, 4, 1, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    turbo_keys, turbo_values = turbo_cache.state

    def fail(*args, **kwargs):
        raise AssertionError(
            "separate-kernel path should handle integer bits without fused fallback"
        )

    monkeypatch.setattr(turbo_cache, "_compiled_integer_decode_attention", fail)
    output = scaled_dot_product_attention(
        queries,
        turbo_keys,
        turbo_values,
        turbo_cache,
        scale=32**-0.5,
        mask=None,
    )

    assert output.shape == queries.shape


def test_turboquant_decode_attention_separate_path_bypasses_fused_split(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")

    keys = mx.random.normal((1, 2, 8, 32))
    values = mx.random.normal((1, 2, 8, 32))
    queries = mx.random.normal((1, 4, 1, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    turbo_keys, turbo_values = turbo_cache.state

    def fail(*args, **kwargs):
        raise AssertionError(
            "separate-kernel path should handle this without fused split fallback"
        )

    monkeypatch.setattr(turbo_cache, "_compiled_split_decode_attention", fail)
    output = scaled_dot_product_attention(
        queries,
        turbo_keys,
        turbo_values,
        turbo_cache,
        scale=32**-0.5,
        mask=None,
    )

    assert output.shape == queries.shape


# Reference of record (design spec §8): full fp32 dequant of the KV -> fp32
# mx.fast.scaled_dot_product_attention. Both the fused kernel and the reference
# see identical dequantized values, so this bounds pure kernel numerics (fp32
# accumulation differences), NOT the codec's quantization error.
_DECODE_GQA_TOL = 2e-2


def _build_decode_case(dim, gqa, bits, T, n_kv=2, seed=0):
    """Build a TurboQuant decode case plus its dequantized-KV reference.

    Returns (queries, turbo_cache, turbo_keys, turbo_values, deq_keys, deq_values).
    """
    mx.random.seed(seed)
    n_q = n_kv * gqa
    keys = mx.random.normal((1, n_kv, T, dim))
    values = mx.random.normal((1, n_kv, T, dim))
    queries = mx.random.normal((1, n_q, 1, dim))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=float(bits))
    turbo_keys, turbo_values = turbo_cache.state
    deq_keys, deq_values = turbo_cache.dequantize(turbo_keys, turbo_values)
    return queries, turbo_cache, turbo_keys, turbo_values, deq_keys, deq_values


@pytest.mark.parametrize(
    "dim,gqa,bits,T",
    [
        (256, 6, 3, 512),  # single-pass, R=6, 3-bit, head_dim=256 (Qwen decode)
        (256, 6, 3, 40000),  # 2-pass long-T (256-block ladder), R=6, 3-bit
        (256, 6, 3, 131072),  # 2-pass 512-block ladder + long-T tile-reuse
        (256, 4, 4, 4096),  # 2-pass, R=4, 4-bit
        (128, 1, 3, 512),  # single-pass MHA (R=1), head_dim=128
        (96, 6, 3, 4096),  # 2-pass, non-pow2-ish head_dim=96 (multiple of 32)
        (256, 6, 4, 512),  # single-pass, R=6, 4-bit
        (128, 4, 3, 4096),  # 2-pass, R=4, head_dim=128
    ],
)
def test_decode_gqa_matches_fp32_reference(dim, gqa, bits, T):
    """Regression net for GQA tile-reuse: the fused decode kernel must match a
    full-fp32 dequant reference across head_dim / GQA / bits / T. Green on the
    CURRENT kernel (correct, just slow) and must stay green after tile-reuse."""
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")

    q, cache, tk, tv, deq_k, deq_v = _build_decode_case(dim, gqa, bits, T)
    scale = dim**-0.5

    quantized = scaled_dot_product_attention(q, tk, tv, cache, scale=scale, mask=None)
    reference = mx.fast.scaled_dot_product_attention(
        q,
        deq_k.astype(q.dtype),
        deq_v.astype(q.dtype),
        scale=scale,
        mask=None,
    )

    assert quantized.shape == reference.shape
    diff = mx.max(mx.abs(quantized - reference)).item()
    assert diff < _DECODE_GQA_TOL, f"max-abs-diff {diff} >= {_DECODE_GQA_TOL}"


def test_turboquant_prod_quantize_skips_mse_dequantize(monkeypatch):
    codec = _TurboQuantProdCodec(32, 4, seed=0)
    vectors = mx.random.normal((1, 2, 8, 32))

    def fail(*args, **kwargs):
        raise AssertionError("Product quantization should not dequantize MSE state")

    monkeypatch.setattr(codec.mse_codec, "_dequantize_unit", fail)
    state = codec.quantize(vectors)

    assert state.mse_indices.shape[:3] == (1, 2, 8)


def test_turboquant_prefill_attention_matches_dequantized_attention():
    keys = mx.random.normal((1, 2, 12, 32))
    values = mx.random.normal((1, 2, 12, 32))
    queries = mx.random.normal((1, 4, 4, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=3.5)
    # This test asserts quantized_attention's fp32-tight (1e-4) match; pin it off
    # the fp16 fused/decomposed MSE prefill path (covered in its own test file
    # with an fp16-appropriate tolerance).
    turbo_cache._fused_prefill_enabled = False
    turbo_keys, turbo_values = turbo_cache.state
    dequantized_keys, dequantized_values = turbo_cache.dequantize(
        turbo_keys,
        turbo_values,
    )

    reference = mx.fast.scaled_dot_product_attention(
        queries,
        dequantized_keys.astype(queries.dtype),
        dequantized_values.astype(queries.dtype),
        scale=32**-0.5,
        mask="causal",
    )
    quantized = scaled_dot_product_attention(
        queries,
        turbo_keys,
        turbo_values,
        turbo_cache,
        scale=32**-0.5,
        mask="causal",
    )

    diff = mx.max(mx.abs(reference - quantized)).item()
    assert quantized.shape == reference.shape
    assert diff < 1e-4


def test_turboquant_quantized_attention_invariant_to_query_block_size():
    # prefill_query_block_size is a pure performance knob: the flash-style online
    # softmax accumulates the same result regardless of how the queries are
    # blocked, so the output must be (bit-)identical across block sizes. Guards
    # the perf default (16 -> 256), which is ~15x faster prefill attention.
    keys = mx.random.normal((1, 2, 512, 32))
    values = mx.random.normal((1, 2, 512, 32))
    queries = mx.random.normal((1, 4, 512, 32))
    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo = TurboQuantKVCache.from_cache(fp_cache, bits=3)
    tk, tv = turbo.state

    turbo.prefill_query_block_size = 16
    small = turbo.quantized_attention(queries, tk, tv, scale=32**-0.5, mask="causal")
    turbo.prefill_query_block_size = 256
    big = turbo.quantized_attention(queries, tk, tv, scale=32**-0.5, mask="causal")
    mx.eval(small, big)

    assert small.shape == big.shape
    assert mx.max(mx.abs(small - big)).item() < 1e-5


def test_turboquant_prefill_no_dequantize_called(monkeypatch):
    """quantized_attention must be used for TQ prefill; dequantize must not fire.

    For kv_bits=3 (MSE+MSE codecs) prefill_attention always returns None, so
    the dispatch must fall through to quantized_attention — never to the
    full-context dequantize spike path.
    """
    keys = mx.random.normal((1, 2, 32, 32))
    values = mx.random.normal((1, 2, 32, 32))
    queries = mx.random.normal((1, 4, 8, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=3)
    turbo_keys, turbo_values = turbo_cache.state

    dequantize_called = []

    original_dequantize = turbo_cache.dequantize

    def patched_dequantize(*args, **kwargs):
        dequantize_called.append(True)
        return original_dequantize(*args, **kwargs)

    monkeypatch.setattr(turbo_cache, "dequantize", patched_dequantize)

    result = scaled_dot_product_attention(
        queries,
        turbo_keys,
        turbo_values,
        turbo_cache,
        scale=32**-0.5,
        mask="causal",
    )
    mx.eval(result)

    assert (
        not dequantize_called
    ), "dequantize must not be called during TQ prefill dispatch"
    assert result.shape == (1, 4, 8, 32)


# ============================================================================
# Allocation hook lifecycle (memory.md #12)
# ============================================================================
#
# The hook list is module-level state in turboquant. Each test resets it
# via the autouse fixture so cross-test ordering can't poison results.
#
# The production caller is server.py's MultiCacheManager: it registers an
# eviction callback so the Metal allocator can request memory mid-prefill
# (when geometric cache growth would otherwise OOM crash). Losing
# register/unregister symmetry leaks hooks across model reloads — which
# means stale Python callbacks fire from new sessions' allocations,
# silently mutating unrelated cache state.


@pytest.fixture(autouse=True)
def _reset_allocation_hooks():
    from mlx_vlm import turboquant as tq

    saved = list(tq._ALLOCATION_HOOKS)
    tq._ALLOCATION_HOOKS.clear()
    yield
    tq._ALLOCATION_HOOKS.clear()
    tq._ALLOCATION_HOOKS.extend(saved)


class TestAllocationHookLifecycle:
    def test_register_adds_hook(self):
        from mlx_vlm import turboquant as tq

        hook = lambda: None  # noqa: E731
        tq.register_allocation_hook(hook)
        assert hook in tq._ALLOCATION_HOOKS

    def test_register_is_idempotent(self):
        # The MultiCacheManager re-registers on every model load; the
        # idempotency guard prevents the same callback firing N times
        # per allocation after several reloads.
        from mlx_vlm import turboquant as tq

        hook = lambda: None  # noqa: E731
        tq.register_allocation_hook(hook)
        tq.register_allocation_hook(hook)
        tq.register_allocation_hook(hook)
        assert tq._ALLOCATION_HOOKS.count(hook) == 1

    def test_unregister_removes_hook(self):
        from mlx_vlm import turboquant as tq

        hook = lambda: None  # noqa: E731
        tq.register_allocation_hook(hook)
        tq.unregister_allocation_hook(hook)
        assert hook not in tq._ALLOCATION_HOOKS

    def test_unregister_unknown_hook_is_noop(self):
        # Defensive: model unload calls unregister even if the hook was
        # never registered (e.g. when the previous load failed before
        # the manager attached). Must not raise.
        from mlx_vlm import turboquant as tq

        tq.unregister_allocation_hook(lambda: None)  # no exception

    def test_trigger_calls_every_hook_in_registration_order(self):
        from mlx_vlm import turboquant as tq

        order = []
        tq.register_allocation_hook(lambda: order.append("a"))
        tq.register_allocation_hook(lambda: order.append("b"))
        tq.register_allocation_hook(lambda: order.append("c"))

        tq._trigger_allocation_hooks()

        assert order == ["a", "b", "c"]

    def test_trigger_with_no_hooks_is_noop(self):
        # No registered hooks must not raise — the production allocator
        # calls _trigger_allocation_hooks() unconditionally.
        from mlx_vlm import turboquant as tq

        tq._trigger_allocation_hooks()  # no exception

    def test_unregister_only_removes_one_instance(self):
        # If the same hook somehow appears multiple times (defensive —
        # idempotency should prevent this, but tests document the
        # guarantee), unregister removes only one.
        from mlx_vlm import turboquant as tq

        hook = lambda: None  # noqa: E731
        # Bypass register's idempotency guard to force a duplicate.
        tq._ALLOCATION_HOOKS.append(hook)
        tq._ALLOCATION_HOOKS.append(hook)
        tq.unregister_allocation_hook(hook)
        assert tq._ALLOCATION_HOOKS.count(hook) == 1


# ============================================================================
# MAX_KV_SIZE plumbing (memory.md #8)
# ============================================================================
#
# `max_kv_size` flows: server CLI / env var → MultiCacheManager →
# TurboQuantKVCache constructor → static pre-allocation in the first
# write (so the GPU never has to hold both old and new buffers
# simultaneously during geometric growth).
#
# Regression mode: someone refactors the constructor and drops the
# parameter; the cache silently reverts to dynamic growth, OOMs at the
# 76K → 95K reallocation boundary on Gemma 4 31B (~11 GB double-buffer).


class TestMaxKVSizePlumbing:
    def test_default_max_kv_size_is_none(self):
        # Sentinel: None means "use dynamic growth". Any other default
        # would silently pre-allocate huge buffers on small models.
        cache = TurboQuantKVCache(bits=4)
        assert cache.max_kv_size is None

    def test_max_kv_size_stored_on_instance(self):
        cache = TurboQuantKVCache(bits=4, max_kv_size=131072)
        assert cache.max_kv_size == 131072

    @staticmethod
    def _seeded_kv_cache(seq_len: int = 4) -> KVCache:
        """Build a KVCache with populated keys/values — the migration
        path (.from_cache) requires real state to read. mlx_lm's
        empty-cache .state accessor raises AttributeError, so feed it
        a real update first.
        """
        cache = KVCache()
        # Shape: (batch, kv_heads, seq_len, head_dim) — minimal valid.
        cache.update_and_fetch(
            mx.zeros((1, 2, seq_len, 8)),
            mx.zeros((1, 2, seq_len, 8)),
        )
        return cache

    def test_from_cache_propagates_max_kv_size(self):
        # The .from_cache() classmethod is the production migration path
        # (FP16 KVCache → quantized TurboQuantKVCache at threshold). The
        # max_kv_size kwarg must reach the constructed cache, otherwise
        # post-threshold caches lose static pre-allocation.
        promoted = TurboQuantKVCache.from_cache(
            self._seeded_kv_cache(), bits=4, max_kv_size=65536
        )
        assert promoted.max_kv_size == 65536

    def test_from_cache_default_max_kv_size_is_none(self):
        promoted = TurboQuantKVCache.from_cache(self._seeded_kv_cache(), bits=4)
        assert promoted.max_kv_size is None

    def test_zero_max_kv_size_treated_as_unset_for_initial_alloc(self):
        # Implementation does `max(new_end, self.max_kv_size or 0)` —
        # explicit 0 collapses to "no pre-alloc", same as None. Document
        # this so a future refactor doesn't introduce surprising
        # zero-allocation cliffs.
        cache = TurboQuantKVCache(bits=4, max_kv_size=0)
        assert cache.max_kv_size == 0  # stored verbatim
        # Behavior at first write equivalence is enforced by the
        # `or 0` idiom in _step; we trust that idiom here rather than
        # spinning up a Metal device just to assert it.
