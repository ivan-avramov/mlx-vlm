"""Tests for mlx_vlm.snapshot — DeltaNetSnapshotRing capture/restore semantics.

These guard the silent-correctness invariants documented in memory.md #13a:
turn-boundary capture, FIFO eviction, find_nearest semantics, divergence-driven
drop_after, and graceful no-ops on disabled rings or pure-attention models.
"""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from mlx_lm.models.cache import RotatingKVCache

from mlx_vlm.snapshot import (
    DEFAULT_RING_SIZE,
    DeltaNetSnapshot,
    DeltaNetSnapshotRing,
    RotatingKVSnapshot,
    capture_rotating,
    restore_rotating,
)


def _make_arrays_cache(num_arrays: int = 2, fill: float = 0.0) -> ArraysCache:
    """ArraysCache pre-loaded with deterministic mx.array state.

    Production code goes through model forward passes; tests just need
    settable state with stable identity for restore-equality checks.
    """
    c = ArraysCache(num_arrays)
    c.state = [mx.full((1, 4), fill) for _ in range(num_arrays)]
    return c


class TestRingBasics:
    def test_default_size_matches_module_constant(self):
        ring = DeltaNetSnapshotRing()
        assert ring.max_size == DEFAULT_RING_SIZE

    def test_zero_disables_ring(self):
        ring = DeltaNetSnapshotRing(max_size=0)
        assert not ring.enabled
        assert len(ring) == 0

    def test_positive_size_enables_ring(self):
        ring = DeltaNetSnapshotRing(max_size=2)
        assert ring.enabled


class TestCapture:
    def test_disabled_ring_capture_returns_none(self):
        ring = DeltaNetSnapshotRing(max_size=0)
        cache = [_make_arrays_cache()]
        assert ring.capture(offset=10, cache=cache) is None
        assert len(ring) == 0

    def test_pure_attention_cache_returns_none(self):
        # KVCache only — no ArraysCache layers — capture is a no-op.
        ring = DeltaNetSnapshotRing(max_size=3)
        cache = [KVCache(), KVCache()]
        assert ring.capture(offset=10, cache=cache) is None
        assert len(ring) == 0

    def test_mixed_cache_captures_only_arrays_layers(self):
        ring = DeltaNetSnapshotRing(max_size=3)
        ac = _make_arrays_cache()
        cache = [KVCache(), ac, KVCache()]
        snap = ring.capture(offset=10, cache=cache)
        assert snap is not None
        # Positional alignment: KV layers map to None, ArraysCache to its state.
        assert snap.states[0] is None
        assert snap.states[2] is None
        assert snap.states[1] is not None
        assert len(snap.states[1]) == 2  # num_arrays from _make_arrays_cache

    def test_state_refs_survive_subsequent_cache_mutation(self):
        # The "refcount-cheap" invariant: snapshotting captures refs to
        # the current mx.arrays. Mutating cache.state after capture must
        # not retroactively change the snapshot.
        ring = DeltaNetSnapshotRing(max_size=3)
        ac = _make_arrays_cache(fill=1.0)
        original_refs = list(ac.state)
        snap = ring.capture(offset=5, cache=[ac])

        # Replace cache state with new arrays — snapshot should still
        # reference the originals.
        ac.state = [mx.full((1, 4), 99.0) for _ in range(2)]
        for snap_arr, orig_arr in zip(snap.states[0], original_refs):
            assert snap_arr is orig_arr

    def test_duplicate_offset_rejected(self):
        ring = DeltaNetSnapshotRing(max_size=3)
        cache = [_make_arrays_cache()]
        first = ring.capture(offset=10, cache=cache)
        second = ring.capture(offset=10, cache=cache)
        assert first is not None
        assert second is None
        assert len(ring) == 1

    def test_non_monotonic_offset_rejected(self):
        # Capture is strictly monotonic — anything <= last is rejected.
        ring = DeltaNetSnapshotRing(max_size=3)
        cache = [_make_arrays_cache()]
        ring.capture(offset=20, cache=cache)
        assert ring.capture(offset=15, cache=cache) is None
        assert len(ring) == 1


class TestFIFOEviction:
    def test_eviction_at_capacity(self):
        ring = DeltaNetSnapshotRing(max_size=2)
        cache = [_make_arrays_cache()]
        ring.capture(offset=10, cache=cache)
        ring.capture(offset=20, cache=cache)
        ring.capture(offset=30, cache=cache)  # should evict offset=10
        assert len(ring) == 2
        # Oldest dropped, two newest retained.
        assert ring.find_nearest(10) is None
        assert ring.find_nearest(20).offset == 20
        assert ring.find_nearest(30).offset == 30

    def test_eviction_preserves_chronological_order(self):
        ring = DeltaNetSnapshotRing(max_size=3)
        cache = [_make_arrays_cache()]
        for off in (5, 10, 15, 20, 25):
            ring.capture(offset=off, cache=cache)
        # Last 3 should remain.
        assert len(ring) == 3
        assert ring.find_nearest(5) is None
        assert ring.find_nearest(15).offset == 15
        assert ring.find_nearest(25).offset == 25


class TestFindNearest:
    def test_empty_ring_returns_none(self):
        ring = DeltaNetSnapshotRing(max_size=3)
        assert ring.find_nearest(100) is None

    def test_target_below_oldest_returns_none(self):
        # If the rewind target predates the oldest snapshot, the caller
        # MUST fall back to full re-prefill.
        ring = DeltaNetSnapshotRing(max_size=3)
        cache = [_make_arrays_cache()]
        ring.capture(offset=20, cache=cache)
        ring.capture(offset=40, cache=cache)
        assert ring.find_nearest(10) is None

    def test_returns_largest_offset_at_or_below_target(self):
        ring = DeltaNetSnapshotRing(max_size=4)
        cache = [_make_arrays_cache()]
        for off in (10, 20, 30, 40):
            ring.capture(offset=off, cache=cache)
        # Exact match.
        assert ring.find_nearest(30).offset == 30
        # Between snapshots: pick the immediately-preceding one.
        assert ring.find_nearest(35).offset == 30
        assert ring.find_nearest(25).offset == 20
        # Above newest: use newest.
        assert ring.find_nearest(100).offset == 40


class TestDropAfter:
    def test_drop_after_removes_strictly_greater(self):
        ring = DeltaNetSnapshotRing(max_size=4)
        cache = [_make_arrays_cache()]
        for off in (10, 20, 30, 40):
            ring.capture(offset=off, cache=cache)

        dropped = ring.drop_after(25)
        assert dropped == 2
        assert len(ring) == 2
        assert ring.find_nearest(20).offset == 20
        # 30 and 40 dropped — find_nearest(40) now falls back to the
        # newest remaining (20), not None.
        assert ring.find_nearest(40).offset == 20
        assert {s.offset for s in ring._snapshots} == {10, 20}

    def test_drop_after_at_existing_offset_is_inclusive_of_equal(self):
        # drop_after(N) keeps snapshots with offset == N (the wording
        # "strictly greater" in the docstring is the spec).
        ring = DeltaNetSnapshotRing(max_size=4)
        cache = [_make_arrays_cache()]
        ring.capture(offset=10, cache=cache)
        ring.capture(offset=20, cache=cache)
        ring.capture(offset=30, cache=cache)

        dropped = ring.drop_after(20)
        assert dropped == 1  # only offset=30 dropped
        assert ring.find_nearest(20).offset == 20

    def test_drop_after_no_op_when_target_above_all(self):
        ring = DeltaNetSnapshotRing(max_size=3)
        cache = [_make_arrays_cache()]
        ring.capture(offset=10, cache=cache)
        ring.capture(offset=20, cache=cache)
        assert ring.drop_after(100) == 0
        assert len(ring) == 2

    def test_drop_after_zero_clears_all_when_no_zero_offset(self):
        ring = DeltaNetSnapshotRing(max_size=3)
        cache = [_make_arrays_cache()]
        ring.capture(offset=10, cache=cache)
        ring.capture(offset=20, cache=cache)
        ring.drop_after(0)
        assert len(ring) == 0


class TestClear:
    def test_clear_empties_ring(self):
        ring = DeltaNetSnapshotRing(max_size=3)
        cache = [_make_arrays_cache()]
        ring.capture(offset=10, cache=cache)
        ring.capture(offset=20, cache=cache)
        ring.clear()
        assert len(ring) == 0
        assert ring.find_nearest(15) is None
        # Ring stays usable after clear.
        ring.capture(offset=5, cache=cache)
        assert len(ring) == 1


class TestSnapshotDataclass:
    def test_snapshot_is_frozen(self):
        snap = DeltaNetSnapshot(offset=10, states=[None], captured_at=0.0)
        with pytest.raises(Exception):
            snap.offset = 99  # frozen=True


def _make_rotating(offset: int, max_size: int = 1024, fill: float = 0.0):
    """RotatingKVCache pre-populated with K/V tensors at the given offset.

    Mirrors the layer state right after prefill on a real model: K/V are
    sized to max_size, offset reflects how many tokens have been written,
    `_idx` is the ring's write-pointer.
    """
    r = RotatingKVCache(max_size=max_size)
    r.offset = offset
    r._idx = offset % max_size
    r.keys = mx.full((1, 4, max_size, 32), fill)
    r.values = mx.full((1, 4, max_size, 32), fill)
    return r


class TestRotatingKVSnapshotCaptureRestore:
    """Round-trip semantics of the snapshot/restore pair used to anchor
    SWA layers at end-of-user when the chat template renders prior
    asst turns asymmetrically (memory.md #30)."""

    def test_capture_records_all_state_fields(self):
        layer = _make_rotating(offset=600, max_size=1024, fill=3.0)
        snap = capture_rotating(layer, layer_index=7)
        assert snap.layer_index == 7
        assert snap.offset == 600
        assert snap.idx == 600  # 600 < 1024 → no wrap yet
        assert snap.max_size == 1024
        assert snap.keys.shape == (1, 4, 1024, 32)

    def test_capture_after_wrap_records_modular_idx(self):
        layer = _make_rotating(offset=4898, max_size=4096)
        snap = capture_rotating(layer, layer_index=0)
        assert snap.offset == 4898
        # _idx wraps modulo max_size when the ring overflows.
        assert snap.idx == 4898 % 4096

    def test_capture_is_independent_of_subsequent_writes(self):
        # Headline invariant: in-place writes to the live layer's K/V
        # tensors after capture must NOT bleed into the snapshot. This
        # is what makes the snapshot+restore cycle correct in the face
        # of generation's slice-assignments to layer.keys.
        layer = _make_rotating(offset=600, max_size=1024, fill=3.0)
        snap = capture_rotating(layer, layer_index=0)

        # Mutate the live tensor in-place — exactly what
        # RotatingKVCache.update_and_fetch does during gen.
        layer.keys[..., 100:110, :] = mx.full((1, 4, 10, 32), 99.0)
        mx.eval(layer.keys)

        # Snapshot's K/V are unchanged.
        sample = snap.keys[0, 0, 105, 0].item()
        assert sample == 3.0, (
            "Snapshot bled through to mutated source — defensive copy broken"
        )

    def test_restore_resets_layer_state(self):
        layer = _make_rotating(offset=600, max_size=1024, fill=3.0)
        snap = capture_rotating(layer, layer_index=0)

        # Generation mutates the live layer.
        layer.offset = 750
        layer._idx = 750
        layer.keys = mx.full((1, 4, 1024, 32), 99.0)
        layer.values = mx.full((1, 4, 1024, 32), 99.0)

        restore_rotating(layer, snap)

        assert int(layer.offset) == 600
        assert int(layer._idx) == 600
        assert layer.max_size == 1024
        # K/V tensors are the snapshotted ones (3.0 fill, not 99.0).
        sample = layer.keys[0, 0, 0, 0].item()
        assert sample == 3.0

    def test_round_trip_works_when_ring_wrapped(self):
        # The bug we're solving: post-gen the ring has wrapped past
        # end-of-prefill. Restore from the pre-gen snapshot must
        # reset everything — offset, _idx, AND ring contents — so the
        # next request sees end-of-prefill state.
        layer = _make_rotating(offset=4613, max_size=4096, fill=1.0)
        layer._idx = 4613 % 4096  # 517
        snap = capture_rotating(layer, layer_index=0)

        # Simulate generation writing 285 tokens past end-of-prefill,
        # wrapping further into the ring.
        layer.offset = 4898
        layer._idx = 4898 % 4096  # 802
        layer.keys[..., 517:802, :] = mx.full((1, 4, 285, 32), 7.0)
        mx.eval(layer.keys)

        restore_rotating(layer, snap)

        assert int(layer.offset) == 4613
        assert int(layer._idx) == 517
        # Ring contents are pre-gen — the gen writes are gone.
        sample = layer.keys[0, 0, 600, 0].item()
        assert sample == 1.0

    def test_snapshot_is_frozen(self):
        layer = _make_rotating(offset=10, max_size=1024)
        snap = capture_rotating(layer, layer_index=0)
        with pytest.raises(Exception):
            snap.offset = 99  # frozen=True
