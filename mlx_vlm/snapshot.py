"""DeltaNet state snapshots for hybrid-cache rewind support.

Hybrid models (Qwen 3.5 / 3.6 GatedDeltaNet, Mamba-style hybrids, etc.)
mix trimmable KV caches with non-trimmable recurrent state. Standard
prompt-cache reuse can't rewind these layers — ``_trim_cache`` only
handles caches with an ``offset`` attribute, leaving recurrent state
advanced past the rewind target. The result is silent generation drift
(no crash, just subtly wrong outputs).

This module provides ``DeltaNetSnapshotRing``, a per-session FIFO ring
of state snapshots captured at turn boundaries. On rewind, the nearest
snapshot is restored and the model forward replays tokens from the
snapshot offset to the rewind target — KV and recurrent state both end
up coherent.

Capture is essentially free: ``ArraysCache.state`` returns a list of
mx.array references, and mx arrays are immutable, so retaining a list
of refs IS the snapshot. No deep-copy needed; mlx tracks lifetime via
refcount.

This module is pure data — no environment lookups, no defaults beyond
the constant below. Configuration (CLI args, env-var fallbacks) is
resolved at the entry-point layer (e.g. ``mlx_vlm.server`` for the
streaming server) and passed down via constructor args.
"""

import time
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx

DEFAULT_RING_SIZE = 3


@dataclass(frozen=True)
class DeltaNetSnapshot:
    """One captured state of the non-trimmable cache layers in a session."""

    offset: int
    """Token offset at the time of capture (inclusive of all preceding tokens)."""

    states: List[Optional[List[mx.array]]]
    """Per-cache-layer state. None entries correspond to trimmable layers
    (KV caches) which are not snapshotted because they're handled by the
    standard ``_trim_cache`` path. List entries hold the raw ``state``
    list captured from each ArraysCache (or equivalent non-trimmable cache).
    """

    captured_at: float
    """time.time() — telemetry only."""


class DeltaNetSnapshotRing:
    """FIFO ring of DeltaNet state snapshots for one session.

    Each PromptCacheState owns one ring. Capture happens at turn boundaries
    via PromptCacheState.update(). Restore happens in the rewind guard at
    generate.py when prefix-cache reuse is requested for a hybrid model.

    ``max_size=0`` disables the ring entirely (capture is a no-op,
    ``enabled`` returns False). The constructor default is left as
    ``DEFAULT_RING_SIZE`` so direct instantiation in tests / chat.py /
    chat_ui.py gets sensible behavior; CLI entry points override it
    explicitly with the resolved configured value.
    """

    def __init__(self, max_size: int = DEFAULT_RING_SIZE):
        self.max_size = max_size
        self._snapshots: List[DeltaNetSnapshot] = []

    def __len__(self) -> int:
        return len(self._snapshots)

    @property
    def enabled(self) -> bool:
        return self.max_size > 0

    def capture(self, offset: int, cache: list) -> Optional[DeltaNetSnapshot]:
        """Snapshot DeltaNet state. Returns None if ring is disabled, the
        offset is not strictly newer than the latest snapshot, or the
        cache contains no non-trimmable layers (purely-attention models).
        """
        if not self.enabled:
            return None
        if self._snapshots and self._snapshots[-1].offset >= offset:
            return None
        # Local import to keep mlx_lm dependency at the leaves.
        from mlx_lm.models.cache import ArraysCache

        states: List[Optional[List[mx.array]]] = []
        any_arrays = False
        for c in cache:
            if isinstance(c, ArraysCache):
                # `state` returns the list of mx.array references. mx arrays
                # are immutable, so list(c.state) is a refcount-cheap snapshot.
                states.append(list(c.state))
                any_arrays = True
            else:
                states.append(None)
        if not any_arrays:
            return None
        snap = DeltaNetSnapshot(
            offset=offset,
            states=states,
            captured_at=time.time(),
        )
        self._snapshots.append(snap)
        while len(self._snapshots) > self.max_size:
            self._snapshots.pop(0)
        return snap

    def find_nearest(self, target_offset: int) -> Optional[DeltaNetSnapshot]:
        """Return the latest snapshot with offset <= target_offset, or None
        if no usable snapshot exists. The caller falls back to full
        re-prefill when this returns None.
        """
        candidate = None
        for snap in self._snapshots:
            if snap.offset <= target_offset:
                if candidate is None or snap.offset > candidate.offset:
                    candidate = snap
        return candidate

    def drop_after(self, offset: int) -> int:
        """Drop snapshots whose offset is strictly greater than ``offset``.

        Called by ``PromptCacheState.update()`` when the token sequence
        diverges from what was cached at capture time — those snapshots
        reference states conditioned on tokens that are no longer in the
        live cache. Returns the number of snapshots dropped.
        """
        before = len(self._snapshots)
        self._snapshots = [s for s in self._snapshots if s.offset <= offset]
        return before - len(self._snapshots)

    def clear(self) -> None:
        self._snapshots.clear()
