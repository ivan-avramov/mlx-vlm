import sys
from unittest.mock import patch

import mlx.core as mx
import pytest

from mlx_vlm.apc import _resolve_turn_capacity
from mlx_vlm.generate import generate_step
from mlx_vlm.generate.ar import _make_cache
from mlx_vlm.generate.common import maybe_preallocate_kv_cache
from mlx_vlm.models import cache as kvc
from mlx_vlm.models import cache as lmcache
from mlx_vlm.models.cache import (
    BatchKVCache,
    BatchQuantizedKVCache,
    PreallocKVCache,
    PreallocQuantizedKVCache,
)
from mlx_vlm.server import generation as G
from mlx_vlm.server.cli import validate_kv_prealloc_tokens
from mlx_vlm.tests.test_kv_cache_quantization import MockModel
from mlx_vlm.turboquant import BatchTurboQuantKVCache, TurboQuantKVCache, _state_length

H, D = 8, 128


def _fake(T):
    return mx.zeros((1, H, T, D), mx.float16), mx.zeros((1, H, T, D), mx.float16)


def test_prealloc_kvcache_allocs_floor_and_never_reallocs():
    c = PreallocKVCache(prealloc_tokens=262144)
    k, v = _fake(1000)
    c.update_and_fetch(k, v)
    assert c.keys.shape[2] == 262144  # allocated to the floor, not 1024
    for _ in range(400):  # decode
        c.update_and_fetch(*_fake(1))
    assert c.keys.shape[2] == 262144  # zero reallocs
    assert c.offset == 1400


def test_prealloc_kvcache_floor_below_prefill_uses_prefill():
    c = PreallocKVCache(prealloc_tokens=512)
    c.update_and_fetch(*_fake(1000))
    assert c.keys.shape[2] == 1024  # ceil(1000/256)*256; floor 512 < prefill


def test_prealloc_kvcache_zero_is_backward_compatible():
    c = PreallocKVCache(prealloc_tokens=0)
    c.update_and_fetch(*_fake(1000))
    assert c.keys.shape[2] == 1024  # identical to stock KVCache
    c.update_and_fetch(*_fake(400))
    assert c.keys.shape[2] > 1024  # grows (fallback)


def test_prealloc_kvcache_state_and_trim_inherited():
    c = PreallocKVCache(prealloc_tokens=4096)
    c.update_and_fetch(*_fake(1000))
    ks, vs = c.state
    assert ks.shape[2] == 1000  # state slices to offset, not capacity
    assert c.trim(200) == 200
    assert c.offset == 800


def test_prealloc_kvcache_from_kvcache_copies_nonempty():
    from mlx_vlm.models.cache import KVCache

    src = KVCache()
    src.update_and_fetch(*_fake(512))  # a mid-prefill plain cache
    c = PreallocKVCache.from_kvcache(src, prealloc_tokens=262144)
    assert c.keys.shape[2] == 262144  # pre-allocated to the floor
    assert c.offset == 512  # content preserved
    for _ in range(400):
        c.update_and_fetch(*_fake(1))
    assert c.keys.shape[2] == 262144  # zero reallocs after copy


def test_prealloc_quantized_allocs_floor_and_never_reallocs():
    c = PreallocQuantizedKVCache(group_size=64, bits=4, prealloc_tokens=262144)
    c.update_and_fetch(*_fake(1000))
    assert c.keys[0].shape[2] == 262144  # packed dim pre-sized to floor
    for _ in range(400):
        c.update_and_fetch(*_fake(1))
    assert c.keys[0].shape[2] == 262144  # zero reallocs
    assert c.offset == 1400


def test_prealloc_quantized_floor_below_prefill_uses_prefill():
    c = PreallocQuantizedKVCache(group_size=64, bits=4, prealloc_tokens=512)
    c.update_and_fetch(*_fake(1000))
    assert c.keys[0].shape[2] == 1024  # ceil(1000/256)*256; floor 512 < prefill


def test_prealloc_quantized_zero_is_backward_compatible():
    c = PreallocQuantizedKVCache(group_size=64, bits=4, prealloc_tokens=0)
    c.update_and_fetch(*_fake(1000))
    assert c.keys[0].shape[2] == 1024  # step-256, like stock QuantizedKVCache


def test_prealloc_quantized_from_quantized_copies_nonempty():
    src = lmcache.QuantizedKVCache(group_size=64, bits=4)
    src.update_and_fetch(*_fake(512))  # a mid-prefill quantized cache
    c = PreallocQuantizedKVCache.from_quantized(src, prealloc_tokens=262144)
    assert c.keys[0].shape[2] == 262144  # pre-allocated triple
    assert c.offset == 512  # content preserved


def test_turboquant_prealloc_floor_and_never_reallocs():
    c = TurboQuantKVCache(bits=4, prealloc_tokens=262144)
    c.update_and_fetch(*_fake(1000))
    assert _state_length(c.keys) == 262144
    for _ in range(400):
        c.update_and_fetch(*_fake(1))
    assert _state_length(c.keys) == 262144  # zero reallocs
    assert c.offset == 1400


def test_turboquant_none_is_backward_compatible():
    c = TurboQuantKVCache(bits=4, prealloc_tokens=None)
    c.update_and_fetch(*_fake(1000))
    assert _state_length(c.keys) == 1000  # grows from new_end (today's behavior)


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
    assert c.keys.shape[2] == 262144  # allocated to the floor, not 1024
    for _ in range(400):  # decode
        c.update_and_fetch(*_fake(1))
    assert c.keys.shape[2] == 262144  # zero reallocs
    assert c.offset.item() == 1400


def test_batch_kvcache_zero_is_backward_compatible():
    c = BatchKVCache([0], prealloc_tokens=0)
    c.update_and_fetch(*_fake(1000))
    assert c.keys.shape[2] == 1024  # ceil(1000/256)*256, like stock BatchKVCache
    assert c.keys.shape[2] < 262144  # grew from the fill, not the floor


def test_batch_quantized_allocs_floor_and_never_reallocs():
    c = BatchQuantizedKVCache([0], group_size=64, bits=4, prealloc_tokens=262144)
    c.update_and_fetch(*_fake(1000))
    assert c.keys[0].shape[2] == 262144  # packed dim pre-sized to floor
    for _ in range(400):  # decode
        c.update_and_fetch(*_fake(1))
    assert c.keys[0].shape[2] == 262144  # zero reallocs
    assert c.offset.item() == 1400


def test_batch_quantized_zero_is_backward_compatible():
    c = BatchQuantizedKVCache([0], group_size=64, bits=4, prealloc_tokens=0)
    c.update_and_fetch(*_fake(1000))
    assert c.keys[0].shape[2] == 1024  # step-256, like stock BatchQuantizedKVCache
    assert c.keys[0].shape[2] < 262144  # grew from the fill, not the floor


def test_batch_turboquant_allocs_floor_and_never_reallocs():
    c = BatchTurboQuantKVCache(left_padding=[0], bits=4, prealloc_tokens=262144)
    c.update_and_fetch(*_fake(1000))
    assert _state_length(c.keys) == 262144
    for _ in range(400):  # decode
        c.update_and_fetch(*_fake(1))
    assert _state_length(c.keys) == 262144  # zero reallocs
    assert c.offset.item() == 1400


def test_batch_turboquant_zero_is_backward_compatible():
    c = BatchTurboQuantKVCache(left_padding=[0], bits=4, prealloc_tokens=0)
    c.update_and_fetch(*_fake(1000))
    assert _state_length(c.keys) == 1000  # grows from new_end (today's behavior)
    assert _state_length(c.keys) < 262144  # grew from the fill, not the floor


def test_maybe_preallocate_converts_empty_plain_and_quantized():
    pc = [lmcache.KVCache(), lmcache.QuantizedKVCache(group_size=64, bits=4)]
    maybe_preallocate_kv_cache(pc, 262144)
    assert isinstance(pc[0], PreallocKVCache) and pc[0].prealloc_tokens == 262144
    assert (
        isinstance(pc[1], PreallocQuantizedKVCache) and pc[1].prealloc_tokens == 262144
    )
    assert pc[1].group_size == 64 and pc[1].bits == 4


def test_maybe_preallocate_converts_nonempty_by_copy():
    pc = [lmcache.KVCache()]
    pc[0].update_and_fetch(*_fake(512))  # mid-prefill (non-empty)
    maybe_preallocate_kv_cache(pc, 262144)
    assert isinstance(pc[0], PreallocKVCache)
    assert pc[0].keys.shape[2] == 262144  # pre-allocated
    assert pc[0].offset == 512  # content copied


def test_maybe_preallocate_converts_nonempty_quantized_by_copy():
    pc = [lmcache.QuantizedKVCache(group_size=64, bits=4)]
    pc[0].update_and_fetch(*_fake(512))  # mid-prefill (non-empty)
    maybe_preallocate_kv_cache(pc, 262144)
    assert isinstance(pc[0], PreallocQuantizedKVCache)
    assert pc[0].keys[0].shape[2] == 262144  # pre-allocated triple
    assert pc[0].offset == 512  # content copied


def test_maybe_preallocate_zero_and_idempotent():
    pc = [lmcache.KVCache()]
    maybe_preallocate_kv_cache(pc, 0)
    assert type(pc[0]) is lmcache.KVCache  # untouched when floor is 0
    maybe_preallocate_kv_cache(pc, 262144)
    first = pc[0]
    maybe_preallocate_kv_cache(pc, 262144)  # second call is a no-op
    assert pc[0] is first


def test_generate_step_forwards_kv_prealloc_tokens():
    seen = {"quantize": [], "prealloc": []}

    def spy_quant(cache, **kw):
        seen["quantize"].append(kw.get("kv_prealloc_tokens", "ABSENT"))

    def spy_prealloc(cache, kv_prealloc_tokens):
        seen["prealloc"].append(kv_prealloc_tokens)

    def spy_make(model, *a, **kw):
        return [kvc.KVCache() for _ in range(2)]

    gen_mod = sys.modules["mlx_vlm.generate"]
    with (
        patch("mlx_vlm.models.cache.make_prompt_cache", spy_make),
        patch.object(gen_mod, "maybe_quantize_kv_cache", spy_quant),
        patch.object(gen_mod, "maybe_preallocate_kv_cache", spy_prealloc),
    ):
        gen = generate_step(
            input_ids=mx.array([[1, 2, 3, 4, 5]]),
            model=MockModel(),
            pixel_values=mx.random.normal((1, 3, 336, 336)),
            mask=mx.ones((1, 5)),
            kv_bits=4,
            kv_group_size=64,
            quantized_kv_start=0,
            max_tokens=3,
            kv_prealloc_tokens=262144,
        )
        for _ in gen:
            pass
    assert seen["quantize"] and all(x == 262144 for x in seen["quantize"])
    assert seen["prealloc"] and all(x == 262144 for x in seen["prealloc"])


def test_make_cache_forwards_kv_prealloc_tokens():
    class _FakeModelWithMakeCache:
        def make_cache(self):
            return [kvc.KVCache()]

    caches = _make_cache(
        _FakeModelWithMakeCache(), left_padding=[0], kv_prealloc_tokens=262144
    )
    assert isinstance(caches[0], kvc.BatchKVCache)
    assert caches[0].prealloc_tokens == 262144


def test_get_kv_prealloc_tokens_env(monkeypatch):
    monkeypatch.setenv("KV_PREALLOC_TOKENS", "262144")
    assert G.get_kv_prealloc_tokens() == 262144
    monkeypatch.delenv("KV_PREALLOC_TOKENS", raising=False)
    assert G.get_kv_prealloc_tokens() is None


def test_validate_kv_prealloc_rejects_over_cap():
    with pytest.raises(ValueError, match="kv_prealloc_tokens"):
        validate_kv_prealloc_tokens(300000, max_kv_size=262144)


def test_validate_kv_prealloc_allows_equal_and_none():
    validate_kv_prealloc_tokens(262144, max_kv_size=262144)  # equal OK
    validate_kv_prealloc_tokens(None, max_kv_size=262144)  # unset OK
    validate_kv_prealloc_tokens(1000, max_kv_size=None)  # no cap configured OK


def test_prealloc_kvcache_to_quantized_slices_to_offset():
    from mlx_vlm.models.cache import QuantizedKVCache

    c = PreallocKVCache(prealloc_tokens=262144)
    c.update_and_fetch(*_fake(512))
    q = c.to_quantized(group_size=64, bits=4)
    assert isinstance(q, QuantizedKVCache)
    assert q.offset == 512
    # quantized ONLY the valid prefix, not the full 262144-row pre-allocated buffer:
    assert q.keys[0].shape[2] == 512


def test_turn_capacity_adaptive_min():
    # content 5000 + output 102400, cap 262144 -> 107400 (right-sized between turns)
    assert (
        _resolve_turn_capacity(
            content=5000, effective_max_tokens=102400, cap=262144, fixed_floor=0
        )
        == 107400
    )


def test_turn_capacity_fixed_floor_wins():
    # fixed floor at cap -> stays at cap (no per-turn shrink)
    assert (
        _resolve_turn_capacity(
            content=5000, effective_max_tokens=102400, cap=262144, fixed_floor=262144
        )
        == 262144
    )


# --- D6: shrink_to_offset() on the three prealloc-floor cache classes ---
# (PreallocKVCache, PreallocQuantizedKVCache, TurboQuantKVCache). Used by
# the session-cache shrink-on-retire path (generate/common.py) to release
# a retired session's prealloc floor back to the allocator; the existing
# first-fill floor logic re-applies the floor lazily on the session's next
# use (re-floor path, tested separately below via update_and_fetch after
# shrink).


def test_prealloc_kvcache_shrink_releases_padding():
    c = PreallocKVCache(prealloc_tokens=262144)
    c.update_and_fetch(*_fake(1000))
    assert c.keys.shape[2] == 262144
    before, after = c.shrink_to_offset()
    assert before == 262144 * H * D * 2 * 2  # K+V, fp16
    assert after < before
    assert c.keys.shape[2] == 1024  # ceil(1000/256)*256
    assert c.values.shape[2] == 1024
    assert c.offset == 1000  # offset unchanged by shrink


def test_prealloc_kvcache_shrink_preserves_prefix_exactly():
    c = PreallocKVCache(prealloc_tokens=262144)
    k, v = mx.random.normal((1, H, 1000, D)), mx.random.normal((1, H, 1000, D))
    c.update_and_fetch(k, v)
    c.shrink_to_offset()
    assert mx.array_equal(c.keys[..., :1000, :], k).item()
    assert mx.array_equal(c.values[..., :1000, :], v).item()


def test_prealloc_kvcache_shrink_noop_when_already_tight():
    c = PreallocKVCache(prealloc_tokens=0)
    c.update_and_fetch(*_fake(1000))
    assert c.keys.shape[2] == 1024
    before, after = c.shrink_to_offset()
    assert before == after
    assert c.keys.shape[2] == 1024


def test_prealloc_kvcache_shrink_noop_on_empty_cache():
    c = PreallocKVCache(prealloc_tokens=262144)
    assert c.shrink_to_offset() == (0, 0)
    assert c.keys is None


def test_prealloc_kvcache_shrink_then_refloor_matches_never_shrunk():
    """The re-floor path: shrink an idle session's cache, then resume it
    (as the server would on the session's next turn) and confirm decode
    continues identically to a cache that was never shrunk — i.e. shrink
    is purely a footprint optimization, not a behavior change."""
    k, v = mx.random.normal((1, H, 1000, D)).astype(mx.float16), mx.random.normal(
        (1, H, 1000, D)
    ).astype(mx.float16)

    reference = PreallocKVCache(prealloc_tokens=262144)
    reference.update_and_fetch(k, v)
    for _ in range(50):
        reference.update_and_fetch(*_fake(1))

    shrunk = PreallocKVCache(prealloc_tokens=262144)
    shrunk.update_and_fetch(k, v)
    shrunk.shrink_to_offset()
    assert shrunk.keys.shape[2] == 1024  # floor released
    # Re-floor is NOT lazy: it must re-establish the full 262144 floor on
    # the very next update_and_fetch() (matching first-fill behavior), not
    # wait for growth past the shrunk 1024 capacity -- waiting would mean a
    # realloc-during-generation transient, exactly what prealloc exists to
    # avoid. Assert this on the very first post-shrink call.
    shrunk.update_and_fetch(*_fake(1))
    assert shrunk.keys.shape[2] == 262144
    for _ in range(49):
        shrunk.update_and_fetch(*_fake(1))
    assert shrunk.keys.shape[2] == 262144  # zero further reallocs
    assert shrunk.offset == reference.offset == 1050
    assert mx.array_equal(
        shrunk.keys[..., : shrunk.offset, :], reference.keys[..., : reference.offset, :]
    ).item()
    assert mx.array_equal(
        shrunk.values[..., : shrunk.offset, :],
        reference.values[..., : reference.offset, :],
    ).item()


def test_prealloc_quantized_shrink_releases_padding():
    c = PreallocQuantizedKVCache(group_size=64, bits=4, prealloc_tokens=262144)
    c.update_and_fetch(*_fake(1000))
    assert c.keys[0].shape[2] == 262144
    before, after = c.shrink_to_offset()
    assert after < before
    assert c.keys[0].shape[2] == 1024
    assert c.offset == 1000


def test_prealloc_quantized_shrink_preserves_prefix_exactly():
    c = PreallocQuantizedKVCache(group_size=64, bits=4, prealloc_tokens=262144)
    k, v = mx.random.normal((1, H, 1000, D)), mx.random.normal((1, H, 1000, D))
    c.update_and_fetch(k, v)
    before_triple = tuple(a[..., :1000, :] for a in c.keys)
    c.shrink_to_offset()
    for before_a, after_a in zip(before_triple, c.keys):
        assert mx.array_equal(before_a, after_a[..., :1000, :]).item()


def test_prealloc_quantized_shrink_noop_when_already_tight():
    c = PreallocQuantizedKVCache(group_size=64, bits=4, prealloc_tokens=0)
    c.update_and_fetch(*_fake(1000))
    before, after = c.shrink_to_offset()
    assert before == after


def test_prealloc_quantized_shrink_noop_on_empty_cache():
    c = PreallocQuantizedKVCache(group_size=64, bits=4, prealloc_tokens=262144)
    assert c.shrink_to_offset() == (0, 0)


def test_prealloc_quantized_shrink_then_refloor_matches_never_shrunk():
    k, v = mx.random.normal((1, H, 1000, D)).astype(mx.float16), mx.random.normal(
        (1, H, 1000, D)
    ).astype(mx.float16)

    reference = PreallocQuantizedKVCache(group_size=64, bits=4, prealloc_tokens=262144)
    reference.update_and_fetch(k, v)
    for _ in range(50):
        reference.update_and_fetch(*_fake(1))

    shrunk = PreallocQuantizedKVCache(group_size=64, bits=4, prealloc_tokens=262144)
    shrunk.update_and_fetch(k, v)
    shrunk.shrink_to_offset()
    assert shrunk.keys[0].shape[2] == 1024
    # Re-floor is immediate (not lazy) -- the very next update_and_fetch()
    # re-establishes the full floor, matching first-fill behavior.
    shrunk.update_and_fetch(*_fake(1))
    assert shrunk.keys[0].shape[2] == 262144
    for _ in range(49):
        shrunk.update_and_fetch(*_fake(1))
    assert shrunk.keys[0].shape[2] == 262144  # zero further reallocs
    assert shrunk.offset == reference.offset == 1050
    for a, b in zip(shrunk.keys, reference.keys):
        assert mx.array_equal(
            a[..., : shrunk.offset, :], b[..., : reference.offset, :]
        ).item()


def test_turboquant_shrink_releases_padding():
    c = TurboQuantKVCache(bits=4, prealloc_tokens=262144)
    c.update_and_fetch(*_fake(1000))
    assert _state_length(c.keys) == 262144
    before, after = c.shrink_to_offset()
    assert after < before
    assert _state_length(c.keys) == 1024
    assert c.offset == 1000


def test_turboquant_shrink_preserves_offset_and_state_shape():
    c = TurboQuantKVCache(bits=4, prealloc_tokens=262144)
    k, v = mx.random.normal((1, H, 1000, D)), mx.random.normal((1, H, 1000, D))
    c.update_and_fetch(k, v)
    ks_before, vs_before = c.state
    mx.eval(ks_before, vs_before)
    c.shrink_to_offset()
    ks_after, vs_after = c.state
    assert c.offset == 1000
    # .state is offset-sliced, so post-shrink content must match pre-shrink
    # content exactly (quantized representation is deterministic given the
    # same codec/seed -- no re-quantization happens during shrink, only a
    # copy of the already-quantized bytes).
    assert mx.array_equal(ks_after.norms, ks_before.norms).item()
    assert mx.array_equal(ks_after.indices, ks_before.indices).item()


def test_turboquant_shrink_noop_when_already_tight():
    c = TurboQuantKVCache(bits=4, prealloc_tokens=None)
    c.update_and_fetch(*_fake(1000))
    before, after = c.shrink_to_offset()
    assert before == after


def test_turboquant_shrink_noop_on_empty_cache():
    c = TurboQuantKVCache(bits=4, prealloc_tokens=262144)
    assert c.shrink_to_offset() == (0, 0)


def test_turboquant_shrink_then_refloor_re_establishes_floor_immediately():
    c = TurboQuantKVCache(bits=4, prealloc_tokens=262144)
    c.update_and_fetch(*_fake(1000))
    assert _state_length(c.keys) == 262144
    c.shrink_to_offset()
    assert _state_length(c.keys) == 1024
    # Re-floor is immediate (not lazy): the very next update_and_fetch()
    # must re-establish the full 262144 floor, matching first-fill
    # behavior, rather than growing incrementally from 1024.
    c.update_and_fetch(*_fake(1))
    assert _state_length(c.keys) == 262144
    assert c.offset == 1001
    for _ in range(400):
        c.update_and_fetch(*_fake(1))
    assert _state_length(c.keys) == 262144  # zero further reallocs
    assert c.offset == 1401


def test_turboquant_shrink_then_refloor_preserves_prefix_bytes():
    c = TurboQuantKVCache(bits=4, prealloc_tokens=262144)
    k, v = mx.random.normal((1, H, 1000, D)), mx.random.normal((1, H, 1000, D))
    c.update_and_fetch(k, v)
    ks_before, vs_before = c.state
    mx.eval(ks_before, vs_before)
    c.shrink_to_offset()
    c.update_and_fetch(*_fake(1))  # triggers re-floor
    ks_after, vs_after = c.state
    # .state is offset-sliced to the ORIGINAL 1000 tokens' worth (the new
    # decode token is token 1001) -- compare the shared prefix.
    assert mx.array_equal(ks_after.norms[..., :1000], ks_before.norms).item()
    assert mx.array_equal(ks_after.indices[..., :1000, :], ks_before.indices).item()
