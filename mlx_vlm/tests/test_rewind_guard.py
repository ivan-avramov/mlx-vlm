"""Tests for the rewind-guard helpers in mlx_vlm.generate.

Covers memory.md #13 (SWA RotatingKVCache rewind safety), #13a (hybrid-cache
rewind guard + DeltaNet state restore), and the PromptCacheState divergence
walker that invalidates stale snapshots when the cached token sequence is
overwritten mid-prefix.

These guard silent-correctness invariants — the failure modes are wrong
generation outputs (no exception), so unit tests are the only way to catch
regressions before they reach inference.
"""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from mlx_vlm.generate import (
    PromptCacheState,
    _has_non_trimmable,
    _restore_deltanet_state,
    _rotating_rewind_safe,
)
from mlx_vlm.snapshot import DeltaNetSnapshotRing


def _arrays_cache(num_arrays: int = 2, fill: float = 0.0) -> ArraysCache:
    c = ArraysCache(num_arrays)
    c.state = [mx.full((1, 4), fill) for _ in range(num_arrays)]
    return c


def _rotating(offset: int, max_size: int) -> RotatingKVCache:
    """Construct a RotatingKVCache with explicit offset/max_size for the
    rewind-safety check. The real cache populates offset via update_and_fetch;
    setting it directly is sufficient because the guard only reads the two
    attributes.
    """
    r = RotatingKVCache(max_size=max_size)
    r.offset = offset
    return r


class _Container:
    """Fake nested cache container exposing a ``caches`` attribute, mimicking
    mlx_lm composite caches (e.g. ChunkedKVCache wrappers) that the rewind
    helpers must recurse into.
    """

    def __init__(self, *children):
        self.caches = list(children)


class TestRotatingRewindSafe:
    def test_empty_entries_is_safe(self):
        assert _rotating_rewind_safe([], target_len=0) is True

    def test_plain_kv_cache_is_skipped(self):
        # Standard KVCache is not a ring buffer, so the guard ignores it.
        assert _rotating_rewind_safe([KVCache()], target_len=10) is True

    def test_unwrapped_buffer_is_safe_at_any_target(self):
        # offset (50) <= max_size (100) → buffer hasn't wrapped, every
        # position is still on disk. Any target_len is safe.
        rkv = _rotating(offset=50, max_size=100)
        assert _rotating_rewind_safe([rkv], target_len=0) is True
        assert _rotating_rewind_safe([rkv], target_len=49) is True

    def test_wrapped_buffer_safe_when_target_in_window(self):
        # offset=150, max_size=100 → tokens 0..49 are overwritten, 50..149
        # are still alive. Targets >= 50 are safe.
        rkv = _rotating(offset=150, max_size=100)
        assert _rotating_rewind_safe([rkv], target_len=50) is True
        assert _rotating_rewind_safe([rkv], target_len=100) is True

    def test_wrapped_buffer_unsafe_when_target_overwritten(self):
        # Same wrapped buffer; target 49 falls in the discarded region.
        rkv = _rotating(offset=150, max_size=100)
        assert _rotating_rewind_safe([rkv], target_len=49) is False
        assert _rotating_rewind_safe([rkv], target_len=0) is False

    def test_offset_as_mx_array_with_item(self):
        # The guard handles mx.array offsets by calling .item() — some
        # cache implementations store offset as a 0-dim mx.array rather
        # than a Python int.
        rkv = RotatingKVCache(max_size=100)
        rkv.offset = mx.array(150)
        assert _rotating_rewind_safe([rkv], target_len=49) is False
        assert _rotating_rewind_safe([rkv], target_len=80) is True

    def test_recurses_into_caches_attribute(self):
        # Composite cache wrapper exposes nested layers via .caches.
        unsafe_inner = _rotating(offset=200, max_size=100)
        wrapper = _Container(unsafe_inner)
        assert _rotating_rewind_safe([wrapper], target_len=50) is False
        assert _rotating_rewind_safe([wrapper], target_len=150) is True

    def test_recurses_into_lists_and_tuples(self):
        unsafe = _rotating(offset=200, max_size=100)
        nested_list = [[KVCache(), unsafe]]
        nested_tuple = ((KVCache(), unsafe),)
        assert _rotating_rewind_safe(nested_list, target_len=50) is False
        assert _rotating_rewind_safe(nested_tuple, target_len=50) is False

    def test_mixed_safe_and_unsafe_returns_unsafe(self):
        # Single unsafe layer poisons the whole batch — full re-prefill.
        safe = _rotating(offset=50, max_size=100)
        unsafe = _rotating(offset=200, max_size=100)
        assert _rotating_rewind_safe([safe, unsafe], target_len=49) is False


class TestHasNonTrimmable:
    def test_pure_attention_returns_false(self):
        assert _has_non_trimmable([KVCache(), KVCache()]) is False

    def test_arrays_cache_returns_true(self):
        assert _has_non_trimmable([_arrays_cache()]) is True

    def test_mixed_returns_true(self):
        assert _has_non_trimmable([KVCache(), _arrays_cache(), KVCache()]) is True

    def test_recurses_into_caches_attribute(self):
        wrapper = _Container(KVCache(), _arrays_cache())
        assert _has_non_trimmable([wrapper]) is True

    def test_recurses_into_lists_and_tuples(self):
        nested = [[KVCache(), _arrays_cache()]]
        assert _has_non_trimmable(nested) is True

    def test_empty_returns_false(self):
        assert _has_non_trimmable([]) is False


class TestRestoreDeltanetState:
    def test_restores_only_arrays_cache_layers(self):
        ac = _arrays_cache(fill=0.0)
        kv = KVCache()
        # Snapshot states aligned to entries: [None for KV, list for ArraysCache]
        new_state = [mx.full((1, 4), 7.0), mx.full((1, 4), 7.0)]
        snapshot_states = [None, new_state]

        _restore_deltanet_state([kv, ac], snapshot_states)

        # ArraysCache.state replaced.
        for arr in ac.state:
            assert mx.array_equal(arr, mx.full((1, 4), 7.0)).item()

    def test_skips_none_snapshot_entries(self):
        # If snapshot says None for a layer (KV layer), the corresponding
        # cache entry is left untouched even if it happens to be ArraysCache.
        ac = _arrays_cache(fill=1.0)
        original = list(ac.state)
        _restore_deltanet_state([ac], [None])
        for before, after in zip(original, ac.state):
            assert before is after

    def test_skips_non_arrays_cache_when_state_present(self):
        # Defensive: if snapshot has a state list at a position that turns
        # out to map to a non-ArraysCache (cache topology changed across
        # turns), the helper must NOT crash and must NOT clobber.
        kv = KVCache()
        snapshot_states = [[mx.zeros((1, 4))]]
        # Should not raise.
        _restore_deltanet_state([kv], snapshot_states)


class TestPromptCacheStateUpdate:
    def test_first_update_captures_snapshot(self):
        ring = DeltaNetSnapshotRing(max_size=3)
        state = PromptCacheState(snapshot_ring=ring)
        ac = _arrays_cache()

        state.update(token_ids=[1, 2, 3, 4, 5], kv_cache=[ac])

        assert state.token_ids == [1, 2, 3, 4, 5]
        assert state.cache == [ac]
        assert len(ring) == 1

    def test_pure_extension_keeps_snapshots(self):
        # New token sequence is a strict extension — divergence equals
        # the old length, so drop_after is a no-op (snapshot offset
        # equals divergence and drop_after keeps equal-offset entries).
        ring = DeltaNetSnapshotRing(max_size=4)
        state = PromptCacheState(snapshot_ring=ring)
        ac = _arrays_cache()

        state.update(token_ids=[1, 2, 3], kv_cache=[ac])
        state.update(token_ids=[1, 2, 3, 4, 5], kv_cache=[ac])

        # Both snapshots survived, latest captured.
        assert {s.offset for s in ring._snapshots} == {3, 5}

    def test_mid_prefix_divergence_drops_stale_snapshots(self):
        # Concrete reproduction of the silent-correctness bug fixed by
        # drop_after: turn N captures snap@offset=20 from a long cached
        # sequence; turn N+1 arrives with a shorter sequence that
        # diverges at offset=10. The snap@20 references DeltaNet state
        # conditioned on tokens that no longer exist in the live cache.
        # If left in the ring, a later rewind to ~20 would restore stale
        # state into a fresh cache → silently wrong outputs.
        ring = DeltaNetSnapshotRing(max_size=4)
        state = PromptCacheState(snapshot_ring=ring)
        ac = _arrays_cache()

        state.update(token_ids=list(range(20)), kv_cache=[ac])
        assert {s.offset for s in ring._snapshots} == {20}

        # New sequence diverges at index 10.
        new_ids = list(range(10)) + [99, 99, 99]
        state.update(token_ids=new_ids, kv_cache=[ac])

        offsets = {s.offset for s in ring._snapshots}
        # Stale snap@20 dropped; new snap@13 captured.
        assert 20 not in offsets
        assert 13 in offsets

    def test_full_replacement_drops_all_old_snapshots(self):
        # Divergence at offset 0 → every prior snapshot is stale.
        ring = DeltaNetSnapshotRing(max_size=4)
        state = PromptCacheState(snapshot_ring=ring)
        ac = _arrays_cache()

        state.update(token_ids=[1, 2, 3, 4, 5], kv_cache=[ac])
        state.update(token_ids=[9, 9, 9], kv_cache=[ac])

        # Only the new snap@3 remains; the old snap@5 was conditioned
        # on tokens [1..5] which no longer exist in the live cache.
        offsets = {s.offset for s in ring._snapshots}
        assert offsets == {3}

    def test_disabled_ring_skips_capture_and_drop(self):
        ring = DeltaNetSnapshotRing(max_size=0)
        state = PromptCacheState(snapshot_ring=ring)
        ac = _arrays_cache()

        # Should not raise even though ring is disabled.
        state.update(token_ids=[1, 2, 3], kv_cache=[ac])
        state.update(token_ids=[1, 2, 9], kv_cache=[ac])

        assert len(ring) == 0
        assert state.token_ids == [1, 2, 9]

    def test_pure_attention_cache_skips_capture(self):
        # Pure-attention model: no ArraysCache → capture is a no-op even
        # though the ring is enabled.
        ring = DeltaNetSnapshotRing(max_size=3)
        state = PromptCacheState(snapshot_ring=ring)

        state.update(token_ids=[1, 2, 3], kv_cache=[KVCache()])
        assert len(ring) == 0


class TestPromptCacheStateFindPrefixLength:
    def test_returns_zero_when_no_prior_state(self):
        state = PromptCacheState()
        assert state.find_prefix_length([1, 2, 3]) == 0

    def test_full_match_returns_full_length(self):
        state = PromptCacheState()
        state.token_ids = [1, 2, 3]
        assert state.find_prefix_length([1, 2, 3]) == 3

    def test_partial_match(self):
        state = PromptCacheState()
        state.token_ids = [1, 2, 3, 4, 5]
        assert state.find_prefix_length([1, 2, 3, 99, 99]) == 3

    def test_handles_new_shorter_than_cached(self):
        state = PromptCacheState()
        state.token_ids = [1, 2, 3, 4, 5]
        assert state.find_prefix_length([1, 2]) == 2

    def test_complete_divergence(self):
        state = PromptCacheState()
        state.token_ids = [1, 2, 3]
        assert state.find_prefix_length([99, 99, 99]) == 0


class TestPromptCacheStateRewindEnabled:
    def test_default_rewind_enabled(self):
        assert PromptCacheState().rewind_enabled is True

    def test_explicit_disable(self):
        state = PromptCacheState(rewind_enabled=False)
        assert state.rewind_enabled is False

    def test_attaches_default_ring_when_omitted(self):
        # PromptCacheState() with no args still gets a usable ring —
        # chat.py / chat_ui.py / tests rely on this.
        state = PromptCacheState()
        assert state.snapshot_ring is not None
        assert state.snapshot_ring.enabled
