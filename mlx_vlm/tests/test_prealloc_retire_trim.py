"""Native preallocated cache retirement must release storage, not retain views."""

import gc

import mlx.core as mx
import pytest

from mlx_vlm.generate import common
from mlx_vlm.generate.common import PromptCacheState, _trim_cache
from mlx_vlm.models.cache import ArraysCache, KVCache, PreallocKVCache


class _DerivedPreallocKVCache(PreallocKVCache):
    pass


def _materialize(cache):
    mx.eval(cache.keys, cache.values)
    mx.synchronize()
    gc.collect()
    mx.clear_cache()


@pytest.mark.parametrize("cache_type", [PreallocKVCache, _DerivedPreallocKVCache])
@pytest.mark.parametrize("shrink", [False, True])
def test_trim_retire_and_resume_preserve_floor_and_release_storage(
    monkeypatch, cache_type, shrink
):
    monkeypatch.setattr(common, "_SESSION_SHRINK_ON_RETIRE", shrink)
    floor = 262144
    prefix_len = 263
    cache = cache_type(prealloc_tokens=floor)
    keys = mx.arange(400 * 8).reshape(1, 1, 400, 8).astype(mx.float16)
    values = -keys
    cache.update_and_fetch(keys, values)
    recurrent = ArraysCache(2)
    recurrent.state = [mx.ones((1, 4)), mx.full((1, 4), 2.0)]
    recurrent_state = list(recurrent.state)
    mx.eval(recurrent.state)
    _materialize(cache)
    full_bytes = cache.keys.nbytes + cache.values.nbytes
    active_before = mx.get_active_memory()
    key_id, value_id = id(cache.keys), id(cache.values)

    # This is the asymmetric retirement path: trim first, publish/shrink next.
    _trim_cache([cache, recurrent], prefix_len)
    assert cache.offset == prefix_len
    assert cache.keys.shape[2] == cache.values.shape[2] == floor
    assert (id(cache.keys), id(cache.values)) == (key_id, value_id)
    assert cache._needs_refloor is False

    state = PromptCacheState()
    state.update(list(range(prefix_len)), [cache, recurrent])
    _materialize(cache)
    if shrink:
        assert cache.keys.shape[2] == cache.values.shape[2] == 512
        assert cache._needs_refloor is True
        released_bytes = full_bytes - cache.keys.nbytes - cache.values.nbytes
        # A short view can report tiny nbytes while retaining the full buffer.
        # Check real MLX ownership after evaluation; allow allocator bookkeeping.
        assert active_before - mx.get_active_memory() >= released_bytes * 0.9
    else:
        assert cache.keys.shape[2] == cache.values.shape[2] == floor
        assert cache._needs_refloor is False
    assert mx.array_equal(cache.keys[:, :, :prefix_len], keys[:, :, :prefix_len]).item()
    assert mx.array_equal(
        cache.values[:, :, :prefix_len], values[:, :, :prefix_len]
    ).item()
    assert all(a is b for a, b in zip(recurrent.state, recurrent_state))
    snapshot = state.snapshot_ring.find_nearest(prefix_len)
    assert snapshot.states[0] is None
    assert all(a is b for a, b in zip(snapshot.states[1], recurrent_state))

    # Reuse resumes with the full configured floor and unmodified cached prefix.
    new_keys = mx.full((1, 1, 1, 8), 17.0, mx.float16)
    new_values = -new_keys
    cache.update_and_fetch(new_keys, new_values)
    _materialize(cache)
    assert cache.offset == prefix_len + 1
    assert cache.keys.shape[2] == cache.values.shape[2] == floor
    assert cache._needs_refloor is False
    assert mx.array_equal(cache.keys[:, :, :prefix_len], keys[:, :, :prefix_len]).item()
    assert mx.array_equal(
        cache.values[:, :, :prefix_len], values[:, :, :prefix_len]
    ).item()
    assert mx.array_equal(
        cache.keys[:, :, prefix_len : prefix_len + 1], new_keys
    ).item()
    assert mx.array_equal(
        cache.values[:, :, prefix_len : prefix_len + 1], new_values
    ).item()
    assert all(a is b for a, b in zip(recurrent.state, recurrent_state))


def test_ordinary_kv_still_physically_trims():
    cache = KVCache()
    cache.update_and_fetch(mx.ones((1, 1, 20, 8)), mx.ones((1, 1, 20, 8)))
    _trim_cache(cache, 12)
    assert cache.offset == 12
    assert cache.keys.shape[2] == cache.values.shape[2] == 12
