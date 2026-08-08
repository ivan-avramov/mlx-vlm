"""Unit tests for EpiCache support on the batched (/v1/completions) generation path.

The single-sequence /v1/chat/completions path wraps full-attn layers in EpiCacheKVCache
and evicts per prefill chunk (generate_step's chunked-prefill hook). The batched
BatchGenerator path builds its caches via ``ar._make_cache`` -> ``to_batch_cache``, which
historically raised "does not yet support batching" for an EpiCacheKVCache, so any request
to /v1/completions crashed when MLX_EPICACHE_BUDGET was set.

This teaches ``to_batch_cache`` to wrap a batch-aware inner cache in EpiCacheKVCache for
batch size 1 (preserving the EpiCache config + the rope_offset/observe/evict surface), and
to raise a clear, EpiCache-specific error for B>1 — where eviction by GLOBAL token index is
invalid because each sequence is independently left-padded (and B>1 long-ctx is
memory-prohibitive anyway).

Run directly (no pytest):
  PYTHONPATH=. <venv>/bin/python mlx_vlm/tests/test_epicache_batch.py
"""

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from mlx_vlm.generate import ar
from mlx_vlm.models import cache
from mlx_vlm.models.epicache import EpiCacheKVCache


class _EpiModel:
    """Minimal stand-in: make_cache returns one EpiCache-wrapped full-attn KVCache,
    exactly as gemma4 / qwen3_5 make_cache do when MLX_EPICACHE_BUDGET is set."""

    def __init__(self, budget=20, sink=4, recent=8, block_size=512):
        self._b, self._s, self._r, self._bs = budget, sink, recent, block_size

    def make_cache(self):
        return [
            EpiCacheKVCache(
                KVCache(),
                budget=self._b,
                block_size=self._bs,
                sink=self._s,
                recent=self._r,
            )
        ]


def test_batch_cache_wraps_epicache_for_b1():
    # B=1: to_batch_cache must wrap a batch-aware inner in EpiCacheKVCache (not raise).
    caches = ar._make_cache(
        _EpiModel(budget=20, sink=4, recent=8, block_size=512), left_padding=[0]
    )
    assert len(caches) == 1
    epi = caches[0]
    assert isinstance(epi, EpiCacheKVCache), type(epi)
    # the inner must be the batch-aware cache (left_padding => batch contract)
    assert isinstance(epi.inner, cache.BatchKVCache), type(epi.inner)
    # EpiCache config preserved through the wrap
    assert (epi.budget, epi.sink, epi.recent, epi.block_size) == (20, 4, 8, 512)
    # and the wrapped cache is usable on the batched path (B=1 update_and_fetch)
    B, H, L, D = 1, 2, 5, 4
    keys = mx.broadcast_to(
        mx.arange(L).reshape(1, 1, L, 1).astype(mx.float32), (B, H, L, D)
    )
    rk, rv = epi.update_and_fetch(keys, keys + 0.5)
    assert rk.shape[2] == L and rv.shape[2] == L, (rk.shape, rv.shape)
    # BatchKVCache offset is a per-sequence array; B=1 with left_padding 0 => [L]
    assert int(epi.offset[0]) == L, epi.offset
    # transparent wrapper on the batch path: no eviction fired, so rope_offset == inner offset
    assert epi.evicted == 0
    assert int(epi.rope_offset[0]) == L, epi.rope_offset


def test_batch_cache_rejects_epicache_for_b_gt_1():
    # B>1: EpiCache eviction is by GLOBAL token index — invalid with per-seq left padding.
    # Must raise a clear, EpiCache-specific error naming the B=1 restriction (not the
    # generic "type ... does not yet support batching" fallthrough).
    try:
        ar._make_cache(_EpiModel(), left_padding=[0, 0])
    except ValueError as e:
        msg = str(e).lower()
        assert "epicache" in msg and "batch size 1" in msg, msg
        return
    raise AssertionError("expected ValueError for EpiCache batch size > 1")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nALL {len(tests)} EpiCache batch tests PASS")
