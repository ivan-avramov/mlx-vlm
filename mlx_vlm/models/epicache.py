"""EpiCache (Phase A): budget-bounded KV cache via block-wise eviction.

Implements the single-episode core of EpiCache (Kim et al., Apple, arXiv:2509.17396)
for long-context inference: as the prompt is prefilled block-by-block, the KV cache
is evicted back down to a per-layer token budget M, so peak KV never exceeds
``M + block`` regardless of total context length. This is the lossy long-context
speed/memory lever for the 256K OptiQ-4bit variants (target <=5% quality drop).

Phase A scope (this file):
  - Wraps a plain ``KVCache`` (full-attention layers only; sliding-window /
    RotatingKVCache layers are already window-bounded and must NOT be wrapped).
  - Eviction policy: always keep the first ``sink`` tokens (attention sinks) and
    the last ``recent`` tokens (recency window); fill the remaining budget with
    the highest-importance middle tokens by a caller-supplied per-token score.
  - Gather-based physical eviction (frees the dropped tokens), preserving causal
    order of the kept positions.

Deliberately NOT in Phase A (documented for the integration/validation step):
  - Score *source*: this class consumes a per-token importance vector; computing
    it (block-local SnapKV-style attention mass) is an attention-layer hook.
  - RoPE positions after eviction: kept keys retain their original (store-time)
    RoPE; the query must continue to be positioned at its TRUE absolute index, not
    the shrunken ``offset`` — handle at the attention/rope call site, not here.
  - TurboQuant inner cache (wrap the quantized state) and measured per-layer
    sensitivity budgets — both follow once Phase A validates on the needle harness.
"""

from __future__ import annotations

import mlx.core as mx


class EpiCacheKVCache:
    """Budget-bounded KV cache. Delegates the read/update contract to an inner
    ``KVCache`` so attention + masking are unchanged; adds ``evict_to_budget``."""

    def __init__(self, inner, *, budget: int, block_size: int = 1024,
                 sink: int = 4, recent: int = 256):
        if budget <= 0:
            raise ValueError("budget must be > 0")
        self.inner = inner
        self.budget = int(budget)
        self.block_size = int(block_size)
        self.sink = int(sink)
        self.recent = int(recent)
        # Count of tokens physically dropped so far (telemetry / position bookkeeping).
        self.evicted = 0

    # -- delegation: keep base.py SDPA dispatch + attention unchanged ---------- #
    def update_and_fetch(self, keys, values):
        return self.inner.update_and_fetch(keys, values)

    @property
    def offset(self):
        return self.inner.offset

    @property
    def state(self):
        return self.inner.state

    @state.setter
    def state(self, v):
        self.inner.state = v

    def make_mask(self, *args, **kwargs):
        return self.inner.make_mask(*args, **kwargs)

    def is_trimmable(self):
        return self.inner.is_trimmable()

    def trim(self, n):
        return self.inner.trim(n)

    def size(self):
        return self.inner.size()

    def empty(self):
        return self.inner.empty()

    @property
    def nbytes(self):
        return self.inner.nbytes

    # -- eviction -------------------------------------------------------------- #
    @staticmethod
    def _select_keep_indices(offset: int, token_scores, budget: int,
                             sink: int, recent: int) -> mx.array:
        """Indices in [0, offset) to KEEP, sorted ascending (causal order).

        Keeps ``sink`` head + ``recent`` tail unconditionally, fills the remaining
        budget with the top-scoring middle tokens. ``token_scores`` is a 1-D array
        of length ``offset`` (higher = more important). Returns <= ``budget`` indices.
        """
        if offset <= budget:
            return mx.arange(offset, dtype=mx.int32)

        sink = max(0, min(int(sink), offset))
        recent = max(0, min(int(recent), offset - sink))
        head = mx.arange(0, sink, dtype=mx.int32)
        tail = mx.arange(offset - recent, offset, dtype=mx.int32)

        n_middle_keep = budget - sink - recent
        if n_middle_keep <= 0:
            # Budget too small for any middle token: keep head + tail, truncated to
            # budget (drop oldest of the protected set first, preserving recency).
            keep = mx.concatenate([head, tail])
            if keep.shape[0] > budget:
                keep = keep[keep.shape[0] - budget:]
            return keep

        mid_lo, mid_hi = sink, offset - recent
        mid_scores = token_scores[mid_lo:mid_hi]
        n_mid = mid_hi - mid_lo
        k = min(n_middle_keep, n_mid)
        if k >= n_mid:
            mid_keep = mx.arange(mid_lo, mid_hi, dtype=mx.int32)
        else:
            # unsorted top-k within the middle, then shift to absolute indices
            top_local = mx.argpartition(mid_scores, kth=n_mid - k)[n_mid - k:]
            mid_keep = (top_local.astype(mx.int32) + mid_lo)

        keep = mx.concatenate([head, mid_keep, tail])
        return mx.sort(keep)

    def evict_to_budget(self, token_scores) -> int:
        """Evict the inner KVCache down to ``budget`` using ``token_scores``
        (1-D, length == current offset). Physically gathers the kept K/V and
        resets the inner offset. Returns the new offset. No-op if already <= budget."""
        inner = self.inner
        off = int(inner.offset)
        if off <= self.budget or inner.keys is None:
            return off
        keep = self._select_keep_indices(off, token_scores, self.budget,
                                         self.sink, self.recent)
        inner.keys = mx.take(inner.keys[..., :off, :], keep, axis=2)
        inner.values = mx.take(inner.values[..., :off, :], keep, axis=2)
        new_off = int(keep.shape[0])
        self.evicted += off - new_off
        inner.offset = new_off
        return new_off
