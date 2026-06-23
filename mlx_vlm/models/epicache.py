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
        # Latest per-key attention-mass importance ([offset]-length), set by observe();
        # consumed (and cleared) by the next evict_to_budget. None -> fall back to key-norm.
        self._scores = None

    # -- delegation: keep base.py SDPA dispatch + attention unchanged ---------- #
    def update_and_fetch(self, keys, values):
        return self.inner.update_and_fetch(keys, values)

    @property
    def offset(self):
        # Inner (physical) offset — governs storage indexing + the causal mask / KV length.
        return self.inner.offset

    @property
    def rope_offset(self):
        """TRUE absolute sequence position for RoPE / position-id derivation.

        Keys are RoPE'd at store time and kept keys retain that store-time RoPE (correct —
        it encodes their original absolute position). But after eviction ``inner.offset`` is
        the SHRUNK physical count, so the next query (and any newly-added keys) must be
        positioned at the real sequence index = ``evicted + inner.offset``, NOT the shrunk
        offset — otherwise the query's RoPE phase no longer matches the kept keys' and
        attention breaks. The attention call site reads this for RoPE while ``offset`` (inner)
        still drives the mask + storage. Physical storage index ≠ RoPE position is fine: RoPE
        is baked into the cached values, so only the relative (query_pos − key_pos) matters."""
        return self.evicted + self.inner.offset

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

    def _keynorm_scores(self, off: int):
        """Attention-free importance proxy: per-token mean key L2 norm over heads+batch.
        High-norm keys tend to attract more attention — a cheap, well-known eviction signal.
        Used when no external score is supplied; the SnapKV-style attention-mass score
        (better, from the attention hook) takes precedence when passed to evict_to_budget."""
        k = self.inner.keys[..., :off, :].astype(mx.float32)  # [B, H, off, D]
        return mx.sqrt((k * k).sum(axis=-1)).mean(axis=(0, 1))  # [off]

    def observe(self, queries, scale, obs_window: int = 32):
        """SnapKV-style attention-mass importance: how much each cached key is attended by the
        most-recent (observation-window) queries. ``queries`` is [B, n_heads, L, D] (RoPE'd, as
        fed to SDPA). Stores a per-key [offset] score (replacing any prior) for the next
        evict_to_budget. Call only when eviction is imminent (it costs an extra obs×offset
        attention per layer).

        Causal masking is intentionally omitted: the observation queries are the most recent, so
        every MIDDLE key (the only region eviction actually chooses among) lies strictly before
        them and is fully attended; only recency-window keys are partially masked, and those are
        protected unconditionally anyway. GQA handled by repeating K up to the query-head count."""
        inner = self.inner
        off = int(inner.offset)
        if inner.keys is None or off == 0:
            return
        k = inner.keys[..., :off, :].astype(mx.float32)       # [B, n_kv, off, D]
        n_heads, n_kv = queries.shape[1], k.shape[1]
        if n_heads != n_kv:
            k = mx.repeat(k, n_heads // n_kv, axis=1)          # GQA -> [B, n_heads, off, D]
        obs = min(int(obs_window), queries.shape[2])
        q = queries[:, :, -obs:, :].astype(mx.float32)         # [B, n_heads, obs, D]
        attn = mx.softmax((q @ k.swapaxes(-1, -2)) * scale, axis=-1)  # [B, n_heads, obs, off]
        self._scores = attn.sum(axis=(0, 1, 2))                # [off]

    def evict_to_budget(self, token_scores=None) -> int:
        """Evict the inner KVCache down to ``budget``. ``token_scores``: 1-D importance vector
        (length == current offset, higher = keep). If None, use the SnapKV attention-mass score
        from the last observe() when present, else the key-norm proxy. Physically gathers the kept
        K/V and resets the inner offset. Returns the new offset. No-op if already <= budget."""
        inner = self.inner
        off = int(inner.offset)
        if off <= self.budget or inner.keys is None:
            self._scores = None
            return off
        if token_scores is None:
            token_scores = self._scores if self._scores is not None else self._keynorm_scores(off)
        keep = self._select_keep_indices(off, token_scores, self.budget,
                                         self.sink, self.recent)
        inner.keys = mx.take(inner.keys[..., :off, :], keep, axis=2)
        inner.values = mx.take(inner.values[..., :off, :], keep, axis=2)
        new_off = int(keep.shape[0])
        self.evicted += off - new_off
        inner.offset = new_off
        self._scores = None
        return new_off
