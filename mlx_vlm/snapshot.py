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


# ---------------------------------------------------------------------------
# RotatingKVCache snapshots
#
# Sliding-window attention layers (mlx_lm's RotatingKVCache) hold their
# K/V state in a ring buffer that's mutated in-place during generation.
# Once the ring wraps (offset > max_size), `RotatingKVCache.trim(n)`
# can't undo the writes — it only decrements offset/_idx, leaving the
# physical slots filled with whatever was written last. Asymmetric
# chat templates (Gemma 4: thinking content present in cache after
# generation but stripped from prior-asst rendering on the next request)
# need the cache to anchor at end-of-user, but we can't trim back to it
# safely on a wrapped ring.
#
# Solution mirrors the DeltaNet path above: capture a snapshot right
# after prefill completes (while the end-of-prefill state still exists),
# generate as usual, then restore the snapshot. The persisted cache state
# always reflects end-of-user, regardless of how generation mutated the
# ring.
#
# Memory cost: ~1 MB per SWA layer at kv_bits=4 + max_size=1024 (Gemma 4
# 26B 8-bit), held only between end-of-prefill and end-of-generation.
# Released after restore. Capture is forced to materialize via
# ``mx.array(x)`` + ``mx.eval()`` — verified empirically that this
# produces an independent buffer (subsequent in-place writes to the
# source don't affect the snapshot).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RotatingKVSnapshot:
    """Frozen snapshot of one ``RotatingKVCache`` layer's state.

    Holds defensive copies of ``keys`` / ``values`` plus the integer
    state attributes (``offset``, ``_idx``, ``max_size``, ``keep``).
    Restoring overwrites the live layer's K/V tensors and pointer
    state, returning it to the captured state byte-for-byte.

    ``layer_index`` records this layer's position in the
    ``prompt_cache`` list so the caller can match snapshots to layers
    when restoring (the cache list ordering is stable across a session).
    """

    layer_index: int
    keys: Optional[mx.array]
    values: Optional[mx.array]
    offset: int
    idx: int
    max_size: int
    keep: int


def capture_rotating(layer, layer_index: int) -> RotatingKVSnapshot:
    """Capture a defensive snapshot of a RotatingKVCache layer.

    The K/V tensors are materialized via ``mx.array(...)`` + ``mx.eval()``
    so that subsequent in-place writes from generation
    (``layer.keys[..., a:b, :] = new_kv``) don't leak into the
    snapshot. Verified empirically — see the comment block above.
    """
    keys_copy: Optional[mx.array] = None
    values_copy: Optional[mx.array] = None
    if layer.keys is not None:
        keys_copy = mx.array(layer.keys)
    if layer.values is not None:
        values_copy = mx.array(layer.values)
    if keys_copy is not None or values_copy is not None:
        # Force materialization in MLX's lazy graph so the buffers are
        # definitely independent before we hand back the snapshot.
        mx.eval(*(a for a in (keys_copy, values_copy) if a is not None))
    return RotatingKVSnapshot(
        layer_index=layer_index,
        keys=keys_copy,
        values=values_copy,
        offset=int(
            layer.offset.item() if hasattr(layer.offset, "item") else layer.offset
        ),
        idx=int(
            layer._idx.item() if hasattr(layer._idx, "item") else layer._idx
        ),
        max_size=int(layer.max_size),
        keep=int(getattr(layer, "keep", 0)),
    )


def restore_rotating(layer, snapshot: RotatingKVSnapshot) -> None:
    """Write a captured snapshot back into a live RotatingKVCache layer.

    After this call, ``layer`` is byte-for-byte the layer that existed
    at capture time — same K/V contents, same offset, same write
    pointer, same ring metadata. Any writes generation made are
    discarded.
    """
    layer.keys = snapshot.keys
    layer.values = snapshot.values
    layer.offset = snapshot.offset
    layer._idx = snapshot.idx
    # max_size and keep are layer-construction-time parameters and don't
    # change during a session; we restore them defensively in case some
    # hybrid wrapper rewrote them.
    layer.max_size = snapshot.max_size
    if hasattr(layer, "keep"):
        layer.keep = snapshot.keep
