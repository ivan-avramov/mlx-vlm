import mlx.core as mx
from mlx_lm.models import cache as lmcache
from mlx_vlm.models.cache import PreallocKVCache

H, D = 8, 128


def _fake(T):
    return mx.zeros((1, H, T, D), mx.float16), mx.zeros((1, H, T, D), mx.float16)


def test_prealloc_kvcache_allocs_floor_and_never_reallocs():
    c = PreallocKVCache(prealloc_tokens=262144)
    k, v = _fake(1000)
    c.update_and_fetch(k, v)
    assert c.keys.shape[2] == 262144          # allocated to the floor, not 1024
    for _ in range(400):                       # decode
        c.update_and_fetch(*_fake(1))
    assert c.keys.shape[2] == 262144          # zero reallocs
    assert c.offset == 1400


def test_prealloc_kvcache_floor_below_prefill_uses_prefill():
    c = PreallocKVCache(prealloc_tokens=512)
    c.update_and_fetch(*_fake(1000))
    assert c.keys.shape[2] == 1024            # ceil(1000/256)*256; floor 512 < prefill


def test_prealloc_kvcache_zero_is_backward_compatible():
    c = PreallocKVCache(prealloc_tokens=0)
    c.update_and_fetch(*_fake(1000))
    assert c.keys.shape[2] == 1024            # identical to stock KVCache
    c.update_and_fetch(*_fake(400))
    assert c.keys.shape[2] > 1024             # grows (fallback)


def test_prealloc_kvcache_state_and_trim_inherited():
    c = PreallocKVCache(prealloc_tokens=4096)
    c.update_and_fetch(*_fake(1000))
    ks, vs = c.state
    assert ks.shape[2] == 1000                # state slices to offset, not capacity
    assert c.trim(200) == 200
    assert c.offset == 800


def test_prealloc_kvcache_from_kvcache_copies_nonempty():
    from mlx_lm.models.cache import KVCache
    src = KVCache()
    src.update_and_fetch(*_fake(512))          # a mid-prefill plain cache
    c = PreallocKVCache.from_kvcache(src, prealloc_tokens=262144)
    assert c.keys.shape[2] == 262144           # pre-allocated to the floor
    assert c.offset == 512                      # content preserved
    for _ in range(400):
        c.update_and_fetch(*_fake(1))
    assert c.keys.shape[2] == 262144           # zero reallocs after copy


from mlx_vlm.models.cache import PreallocQuantizedKVCache


def test_prealloc_quantized_allocs_floor_and_never_reallocs():
    c = PreallocQuantizedKVCache(group_size=64, bits=4, prealloc_tokens=262144)
    c.update_and_fetch(*_fake(1000))
    assert c.keys[0].shape[2] == 262144       # packed dim pre-sized to floor
    for _ in range(400):
        c.update_and_fetch(*_fake(1))
    assert c.keys[0].shape[2] == 262144       # zero reallocs
    assert c.offset == 1400


def test_prealloc_quantized_zero_is_backward_compatible():
    c = PreallocQuantizedKVCache(group_size=64, bits=4, prealloc_tokens=0)
    c.update_and_fetch(*_fake(1000))
    assert c.keys[0].shape[2] == 1024         # step-256, like stock QuantizedKVCache


def test_prealloc_quantized_from_quantized_copies_nonempty():
    src = lmcache.QuantizedKVCache(group_size=64, bits=4)
    src.update_and_fetch(*_fake(512))          # a mid-prefill quantized cache
    c = PreallocQuantizedKVCache.from_quantized(src, prealloc_tokens=262144)
    assert c.keys[0].shape[2] == 262144        # pre-allocated triple
    assert c.offset == 512                      # content preserved


from mlx_vlm.turboquant import TurboQuantKVCache, _state_length


def test_turboquant_prealloc_floor_and_never_reallocs():
    c = TurboQuantKVCache(bits=4, prealloc_tokens=262144)
    c.update_and_fetch(*_fake(1000))
    assert _state_length(c.keys) == 262144
    for _ in range(400):
        c.update_and_fetch(*_fake(1))
    assert _state_length(c.keys) == 262144    # zero reallocs
    assert c.offset == 1400


def test_turboquant_none_is_backward_compatible():
    c = TurboQuantKVCache(bits=4, prealloc_tokens=None)
    c.update_and_fetch(*_fake(1000))
    assert _state_length(c.keys) == 1000      # grows from new_end (today's behavior)


from mlx_vlm.turboquant import BatchTurboQuantKVCache
from mlx_vlm.models.cache import BatchKVCache, BatchQuantizedKVCache


def test_batch_turboquant_prealloc():
    c = BatchTurboQuantKVCache(left_padding=[0], bits=4, prealloc_tokens=262144)
    c.update_and_fetch(*_fake(1000))
    assert _state_length(c.keys) == 262144


def test_batch_kvcache_prealloc():
    c = BatchKVCache([0], prealloc_tokens=262144)
    c.update_and_fetch(*_fake(1000))
    assert c.keys.shape[2] == 262144


def test_batch_quantized_prealloc():
    c = BatchQuantizedKVCache([0], group_size=64, bits=4, prealloc_tokens=262144)
    c.update_and_fetch(*_fake(1000))
    assert c.keys[0].shape[2] == 262144


def test_batch_kvcache_allocs_floor_and_never_reallocs():
    c = BatchKVCache([0], prealloc_tokens=262144)
    c.update_and_fetch(*_fake(1000))
    assert c.keys.shape[2] == 262144           # allocated to the floor, not 1024
    for _ in range(400):                        # decode
        c.update_and_fetch(*_fake(1))
    assert c.keys.shape[2] == 262144           # zero reallocs
    assert c.offset.item() == 1400


def test_batch_kvcache_zero_is_backward_compatible():
    c = BatchKVCache([0], prealloc_tokens=0)
    c.update_and_fetch(*_fake(1000))
    assert c.keys.shape[2] == 1024              # ceil(1000/256)*256, like stock BatchKVCache
    assert c.keys.shape[2] < 262144             # grew from the fill, not the floor


def test_batch_quantized_allocs_floor_and_never_reallocs():
    c = BatchQuantizedKVCache([0], group_size=64, bits=4, prealloc_tokens=262144)
    c.update_and_fetch(*_fake(1000))
    assert c.keys[0].shape[2] == 262144        # packed dim pre-sized to floor
    for _ in range(400):                        # decode
        c.update_and_fetch(*_fake(1))
    assert c.keys[0].shape[2] == 262144        # zero reallocs
    assert c.offset.item() == 1400


def test_batch_quantized_zero_is_backward_compatible():
    c = BatchQuantizedKVCache([0], group_size=64, bits=4, prealloc_tokens=0)
    c.update_and_fetch(*_fake(1000))
    assert c.keys[0].shape[2] == 1024          # step-256, like stock BatchQuantizedKVCache
    assert c.keys[0].shape[2] < 262144         # grew from the fill, not the floor


def test_batch_turboquant_allocs_floor_and_never_reallocs():
    c = BatchTurboQuantKVCache(left_padding=[0], bits=4, prealloc_tokens=262144)
    c.update_and_fetch(*_fake(1000))
    assert _state_length(c.keys) == 262144
    for _ in range(400):                        # decode
        c.update_and_fetch(*_fake(1))
    assert _state_length(c.keys) == 262144     # zero reallocs
    assert c.offset.item() == 1400


def test_batch_turboquant_zero_is_backward_compatible():
    c = BatchTurboQuantKVCache(left_padding=[0], bits=4, prealloc_tokens=0)
    c.update_and_fetch(*_fake(1000))
    assert _state_length(c.keys) == 1000       # grows from new_end (today's behavior)
    assert _state_length(c.keys) < 262144      # grew from the fill, not the floor
