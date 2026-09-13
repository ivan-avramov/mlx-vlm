"""Live cache storage contracts; tiny heads exercise the real full-cap floor."""

from functools import partial
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.generate.ar import _make_cache
from mlx_vlm.speculative.utils import make_speculative_prompt_cache
from mlx_vlm.turboquant import (
    BatchTurboQuantKVCache,
    TurboQuantKVCache,
    _state_length,
    _TurboQuantAttentionMixin,
)

FLOOR = 262144


def _tokens(batch=1, count=3):
    return mx.ones((batch, 1, count, 8), dtype=mx.float16)


def _assert_capacity(cache, size=FLOOR):
    mx.eval(cache.keys, cache.values)
    assert _state_length(cache.keys) == size
    assert _state_length(cache.values) == size


def test_attention_mixin_does_not_own_storage_protocols():
    forbidden = {
        "__init__",
        "from_cache",
        "_ensure_codecs",
        "update_and_fetch",
        "prefix_cache_snapshot",
        "prefix_cache_restore",
        "prefix_cache_merge",
        "prefix_cache_reserve",
    }
    assert forbidden.isdisjoint(vars(_TurboQuantAttentionMixin))
    assert "_try_fused_kv_quantize" not in vars(TurboQuantKVCache)
    for cls in (TurboQuantKVCache, BatchTurboQuantKVCache):
        assert cls.__mro__[1] is _TurboQuantAttentionMixin
        assert "update_and_fetch" in vars(cls)


@pytest.mark.parametrize("draft_kind", ["mtp", "eagle3", "dflash"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_speculative_factory_uses_live_batch_storage(draft_kind, batch_size):
    model = SimpleNamespace(layers=[object() for _ in range(4)])
    factory = partial(
        _make_cache, kv_bits=4, kv_quant_scheme="turboquant", kv_prealloc_tokens=FLOOR
    )
    caches = make_speculative_prompt_cache(
        model,
        draft_kind=draft_kind,
        batch_size=batch_size,
        left_padding=[0] * batch_size,
        make_cache=factory,
    )
    cache = caches[0]
    assert type(cache) is BatchTurboQuantKVCache
    assert cache.prealloc_tokens == FLOOR
    assert cache._fused_prefill_enabled is False
    cache.update_and_fetch(_tokens(batch_size), _tokens(batch_size))
    _assert_capacity(cache)
    assert cache.trim(2) == 2
    cache.update_and_fetch(_tokens(batch_size, 1), _tokens(batch_size, 1))
    _assert_capacity(cache)
    assert cache.offset.tolist() == [2] * batch_size
    extracted = cache.extract(0)
    assert type(extracted) is TurboQuantKVCache
    assert extracted.prealloc_tokens == FLOOR
    _assert_capacity(extracted)
    assert extracted.offset == 2
    assert bool(mx.allclose(extracted.dequantize()[0], cache.dequantize()[0][:1]))


def test_single_shrink_and_packed_restore_preserve_floor_and_values():
    cache = TurboQuantKVCache(
        4,
        max_kv_size=FLOOR,
        prealloc_tokens=FLOOR,
        fused_prefill=False,
        decode_2pass_use_legacy=True,
    )
    cache.update_and_fetch(_tokens(), _tokens())
    _assert_capacity(cache)
    expected = cache.dequantize()[0]
    cache.shrink_to_offset()
    _assert_capacity(cache, 256)
    cache.update_and_fetch(_tokens(count=1), _tokens(count=1))
    _assert_capacity(cache)
    snapshot = cache.prefix_cache_snapshot()
    restored = TurboQuantKVCache(4)
    restored.prefix_cache_restore(snapshot)
    assert restored.prealloc_tokens == restored.max_kv_size == FLOOR
    assert restored._decode_2pass_use_legacy is True
    _assert_capacity(restored)
    assert bool(mx.allclose(restored.dequantize()[0][:, :, :3], expected))
    restored.update_and_fetch(_tokens(count=1), _tokens(count=1))
    _assert_capacity(restored)


def test_packed_merge_and_extract_preserve_floor_and_padding():
    rows = [TurboQuantKVCache(4, prealloc_tokens=FLOOR) for _ in range(2)]
    for count, row in enumerate(rows, 2):
        row.update_and_fetch(_tokens(count=count), _tokens(count=count))
    merged = rows[0].prefix_cache_merge(rows, [2, 3])
    assert type(merged) is BatchTurboQuantKVCache
    assert merged.prealloc_tokens == FLOOR
    _assert_capacity(merged)
    assert merged.offset.tolist() == [2, 3]
    assert merged.left_padding.tolist() == [1, 0]
    for index, row in enumerate(rows):
        extracted = merged.extract(index)
        assert extracted.prealloc_tokens == FLOOR
        _assert_capacity(extracted)
        assert extracted.offset == row.offset
        assert bool(mx.allclose(extracted.dequantize()[0], row.dequantize()[0]))


def test_batch_filter_extend_and_state_restore_keep_full_floor():
    cache = BatchTurboQuantKVCache([1, 0], 4, prealloc_tokens=FLOOR)
    cache.update_and_fetch(_tokens(2), _tokens(2))
    expected = cache.extract(0).dequantize()[0]
    cache.filter(mx.array([0]))
    assert cache.offset.tolist() == [2]
    assert cache.left_padding.tolist() == [0]
    _assert_capacity(cache)
    assert bool(mx.allclose(cache.dequantize()[0], expected))
    other = BatchTurboQuantKVCache([0], 4, prealloc_tokens=FLOOR)
    other.update_and_fetch(_tokens(), _tokens())
    cache.extend(other)
    _assert_capacity(cache)
    assert cache.offset.tolist() == [2, 3]
    restored = BatchTurboQuantKVCache([0, 0], 4)
    restored.state = cache.state
    restored.meta_state = cache.meta_state
    assert restored.prealloc_tokens == FLOOR
    _assert_capacity(restored)


@pytest.mark.parametrize("legacy", [False, True])
def test_packed_snapshot_restores_into_configured_full_floor(legacy):
    source = TurboQuantKVCache(4)
    source.update_and_fetch(_tokens(), _tokens())
    snapshot = source.prefix_cache_snapshot()
    if legacy:
        snapshot["meta_state"] = snapshot["meta_state"][:5]
        snapshot.pop("options")
        snapshot.pop("capacity")
    target = TurboQuantKVCache(4, prealloc_tokens=FLOOR, max_kv_size=FLOOR)
    target.prefix_cache_restore(snapshot)
    assert target.prealloc_tokens == target.max_kv_size == FLOOR
    _assert_capacity(target)


@pytest.mark.parametrize("legacy", [False, True])
def test_metadata_restore_cannot_reduce_destination_batch_floor(legacy):
    source = BatchTurboQuantKVCache(bits=4, left_padding=[0])
    source.update_and_fetch(_tokens(), _tokens())
    destination = BatchTurboQuantKVCache(
        bits=4, left_padding=[0], prealloc_tokens=FLOOR
    )
    destination.state = source.state
    destination.meta_state = source.meta_state[:5] if legacy else source.meta_state
    assert destination.prealloc_tokens == FLOOR
    other = BatchTurboQuantKVCache(bits=4, left_padding=[0])
    other.update_and_fetch(_tokens(), _tokens())
    destination.extend(other)
    _assert_capacity(destination)
    destination.filter(mx.array([0]))
    _assert_capacity(destination)
    destination.update_and_fetch(_tokens(count=1), _tokens(count=1))
    _assert_capacity(destination)


@pytest.mark.parametrize("legacy", [False, True])
def test_generic_metadata_restore_cannot_reduce_destination_single_floor(legacy):
    source = TurboQuantKVCache(4)
    source.update_and_fetch(_tokens(), _tokens())
    destination = TurboQuantKVCache(4, max_kv_size=FLOOR, prealloc_tokens=FLOOR)
    destination.state = source.state
    destination.meta_state = source.meta_state[:5] if legacy else source.meta_state
    assert destination.prealloc_tokens == destination.max_kv_size == FLOOR
    destination.shrink_to_offset()
    destination.update_and_fetch(_tokens(count=1), _tokens(count=1))
    _assert_capacity(destination)
