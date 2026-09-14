"""Native preallocated cache retirement must release storage, not retain views."""

import gc

import mlx.core as mx
import pytest

from mlx_vlm.generate import common
from mlx_vlm.generate.common import PromptCacheState, _trim_cache
from mlx_vlm.models.cache import (
    ArraysCache,
    KVCache,
    PreallocKVCache,
    PreallocQuantizedKVCache,
)
from mlx_vlm.turboquant import TurboQuantKVCache


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
    if shrink:
        assert cache.keys.shape[2] == cache.values.shape[2] == 512
        assert cache._needs_refloor is True
        released_bytes = full_bytes - cache.keys.nbytes - cache.values.nbytes
        # A short view can report tiny nbytes while retaining the full buffer.
        # Retirement must release storage before returning, without caller eval.
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


def _leaves(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (tuple, list)):
        for part in value:
            yield from _leaves(part)


def _storage_bytes(cache):
    return sum(a.nbytes for a in _leaves((cache.keys, cache.values)))


def _state_values(cache):
    # Convert to Python data; retaining MLX state views would itself pin storage.
    return [a.tolist() for a in _leaves(cache.state)]


@pytest.mark.parametrize("family", ["native", "uniform", "turboquant"])
@pytest.mark.parametrize("via_session", [False, True])
def test_shrink_releases_storage_before_return(monkeypatch, family, via_session):
    monkeypatch.setattr(common, "_SESSION_SHRINK_ON_RETIRE", True)
    floor = 262144
    if family == "native":
        cache = PreallocKVCache(prealloc_tokens=floor)
        dim = 8
    elif family == "uniform":
        cache = PreallocQuantizedKVCache(group_size=32, bits=4, prealloc_tokens=floor)
        dim = 64
    else:
        cache = TurboQuantKVCache(4, prealloc_tokens=floor, max_kv_size=floor)
        dim = 8
    keys = mx.arange(400 * dim).reshape(1, 1, 400, dim).astype(mx.float16)
    cache.update_and_fetch(keys, -keys)
    _trim_cache(cache, 263)
    # Also exercise a populated TurboQuant memoized state view before shrinking.
    expected = _state_values(cache)
    metadata = cache.meta_state
    _materialize(cache)
    full_bytes = _storage_bytes(cache)
    active_before = mx.get_active_memory()

    if via_session:
        state = PromptCacheState()
        state.update(list(range(263)), [cache])
        assert state.cache[0] is cache
        assert state.token_ids == list(range(263))
    else:
        before, after = cache.shrink_to_offset()
        assert before == full_bytes
        assert after == _storage_bytes(cache)

    # NO eval/synchronize/clear_cache or value reads may precede this assertion.
    compact_bytes = _storage_bytes(cache)
    assert compact_bytes == full_bytes * 512 // floor
    assert active_before - mx.get_active_memory() >= (full_bytes - compact_bytes) * 0.9
    assert cache._needs_refloor is True
    assert cache.meta_state == metadata
    assert _state_values(cache) == expected

    cache.update_and_fetch(
        mx.ones((1, 1, 1, dim), mx.float16), -mx.ones((1, 1, 1, dim), mx.float16)
    )
    _materialize(cache)
    assert _storage_bytes(cache) == full_bytes
    assert cache.offset == 264
    assert cache.prealloc_tokens == floor
    assert cache._needs_refloor is False
    cache.trim(1)
    assert _state_values(cache) == expected


def _small_retiring_cache(family):
    if family == "native":
        cache = PreallocKVCache(prealloc_tokens=262144)
        dim = 8
    elif family == "uniform":
        cache = PreallocQuantizedKVCache(group_size=32, bits=4, prealloc_tokens=262144)
        dim = 64
    else:
        cache = TurboQuantKVCache(4, prealloc_tokens=262144, max_kv_size=262144)
        dim = 8
    cache.update_and_fetch(mx.ones((1, 1, 400, dim)), -mx.ones((1, 1, 400, dim)))
    _trim_cache(cache, 263)
    _materialize(cache)
    return cache


@pytest.mark.parametrize("family", ["native", "uniform", "turboquant"])
def test_shrink_keeps_external_views_valid_until_caller_releases_them(family):
    cache = _small_retiring_cache(family)
    expected = _state_values(cache)
    aliases = cache.state
    mx.eval(aliases)
    full_bytes = _storage_bytes(cache)
    active_before = mx.get_active_memory()
    cache.shrink_to_offset()
    active_with_aliases = mx.get_active_memory()
    assert active_with_aliases >= active_before * 0.9
    assert [a.tolist() for a in _leaves(aliases)] == expected
    assert _state_values(cache) == expected
    del aliases
    # Only external ownership delayed the release; no cache eval is needed.
    assert (
        active_with_aliases - mx.get_active_memory()
        >= (full_bytes - _storage_bytes(cache)) * 0.9
    )


@pytest.mark.parametrize("family", ["native", "uniform", "turboquant"])
@pytest.mark.parametrize("operation", ["eval", "synchronize"])
def test_shrink_evaluation_failure_preserves_original_cache(
    monkeypatch, family, operation
):
    cache = _small_retiring_cache(family)
    keys, values = cache.keys, cache.values
    expected = _state_values(cache)
    metadata = cache.meta_state

    def fail(*args, **kwargs):
        raise RuntimeError("copy evaluation failed")

    with monkeypatch.context() as scope:
        scope.setattr(mx, operation, fail)
        with pytest.raises(RuntimeError, match="copy evaluation failed"):
            cache.shrink_to_offset()
    assert cache.keys is keys and cache.values is values
    assert cache._needs_refloor is False
    assert cache.meta_state == metadata
    assert _state_values(cache) == expected
