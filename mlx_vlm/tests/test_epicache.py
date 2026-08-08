"""Unit tests for EpiCache Phase-A eviction core (no model required).

Run directly (no pytest):
  PYTHONPATH=. \
    <venv>/bin/python mlx_vlm/tests/test_epicache.py
"""

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from mlx_vlm.models.epicache import EpiCacheKVCache


def test_select_keep_no_eviction():
    # offset <= budget -> keep everything, in order
    keep = EpiCacheKVCache._select_keep_indices(
        10, mx.zeros(10), budget=20, sink=4, recent=8
    )
    assert keep.tolist() == list(range(10)), keep.tolist()


def test_select_keep_sink_recent_and_topk():
    offset, budget, sink, recent = 100, 20, 4, 8
    # middle scores: make positions 50,51,52,...,57 the highest (8 of them)
    scores = mx.zeros(offset)
    hot = list(range(50, 58))  # 8 hottest middle tokens
    for i, p in enumerate(hot):
        scores[p] = 100.0 + i
    keep = EpiCacheKVCache._select_keep_indices(
        offset, scores, budget, sink, recent
    ).tolist()
    assert len(keep) == budget, (len(keep), budget)
    assert keep == sorted(keep), "must be causal order"
    # sink + recent always kept
    assert set(range(sink)).issubset(keep), keep
    assert set(range(offset - recent, offset)).issubset(keep), keep
    # the 8 hottest middle tokens are kept (budget-sink-recent = 8 middle slots)
    assert set(hot).issubset(keep), (hot, keep)


def test_select_keep_budget_smaller_than_protected():
    # budget < sink+recent -> keep the most-recent `budget` of the protected set
    keep = EpiCacheKVCache._select_keep_indices(
        100, mx.zeros(100), budget=5, sink=4, recent=8
    ).tolist()
    assert len(keep) == 5, keep
    assert keep == [95, 96, 97, 98, 99], keep  # recency preserved


def test_evict_to_budget_gather_correctness():
    # Build a KVCache where token t's K and V are filled with the constant t,
    # so we can verify the gather kept exactly the selected positions.
    N, B, H, D = 100, 1, 2, 4
    c = KVCache()
    # push one block of N tokens
    keys = mx.broadcast_to(
        mx.arange(N).reshape(1, 1, N, 1).astype(mx.float32), (B, H, N, D)
    )
    vals = keys + 0.5
    c.update_and_fetch(keys, vals)
    assert c.offset == N

    epi = EpiCacheKVCache(c, budget=20, sink=4, recent=8)
    scores = mx.zeros(N)
    hot = list(range(40, 48))
    for i, p in enumerate(hot):
        scores[p] = 100.0 + i
    expected = EpiCacheKVCache._select_keep_indices(N, scores, 20, 4, 8).tolist()

    new_off = epi.evict_to_budget(scores)
    assert new_off == 20 == c.offset, (new_off, c.offset)
    assert epi.evicted == N - 20

    # the kept keys must equal the expected original token indices, in order
    kept_ids = c.keys[0, 0, :new_off, 0].tolist()
    assert [int(x) for x in kept_ids] == expected, (kept_ids, expected)
    # values track keys (t + 0.5)
    kept_vals = c.values[0, 0, :new_off, 0].tolist()
    assert [round(x - 0.5) for x in kept_vals] == expected
    # a subsequent decode step still works (cache re-grows from the evicted offset)
    nxt_k = mx.broadcast_to(mx.array([999.0]).reshape(1, 1, 1, 1), (B, H, 1, D))
    rk, rv = c.update_and_fetch(nxt_k, nxt_k)
    assert c.offset == 21 and rk.shape[2] == 21


def test_evict_noop_when_under_budget():
    N = 10
    c = KVCache()
    keys = mx.zeros((1, 2, N, 4))
    c.update_and_fetch(keys, keys)
    epi = EpiCacheKVCache(c, budget=50)
    assert epi.evict_to_budget(mx.zeros(N)) == N
    assert epi.evicted == 0


def test_rope_offset_tracks_true_position():
    # rope_offset must stay at the TRUE absolute position across eviction (evicted+inner.offset),
    # while offset (inner) shrinks — this is what keeps post-eviction RoPE correct.
    N = 50
    c = KVCache()
    keys = mx.zeros((1, 2, N, 4))
    c.update_and_fetch(keys, keys)
    epi = EpiCacheKVCache(c, budget=20, sink=4, recent=8)
    assert epi.rope_offset == 50 and epi.offset == 50
    epi.evict_to_budget(
        mx.zeros(N)
    )  # key-norm fallback (all-zero) still evicts to budget
    assert epi.offset == 20 and epi.evicted == 30
    assert epi.rope_offset == 50  # evicted(30) + inner.offset(20) == true position 50


def test_observe_attention_mass_drives_eviction():
    # A GQA cache (4 query heads, 2 kv heads). Give middle key 40 a distinctive direction and
    # send observation queries aligned to it -> attention-mass concentrates on 40 -> evict (with
    # no explicit scores) must use the observed mass and KEEP key 40.
    N, B, nkv, nq, D = 80, 1, 2, 4, 8
    c = KVCache()
    keys = mx.zeros((B, nkv, N, D))
    keys[:, :, 40, 0] = 5.0  # key 40: spike on dim 0
    c.update_and_fetch(keys, keys)
    epi = EpiCacheKVCache(c, budget=20, sink=4, recent=8)

    queries = mx.zeros((B, nq, 6, D))
    queries[:, :, :, 0] = 5.0  # queries aligned to key 40's direction
    epi.observe(queries, scale=1.0, obs_window=6)
    assert epi._scores is not None and epi._scores.shape == (N,)
    assert int(mx.argmax(epi._scores).item()) == 40, int(mx.argmax(epi._scores).item())

    epi.evict_to_budget()  # no explicit scores -> uses observed attention mass
    assert c.offset == 20
    kept_dim0 = c.keys[0, 0, :, 0].tolist()
    assert any(abs(x - 5.0) < 0.1 for x in kept_dim0), kept_dim0  # key 40 survived
    assert epi._scores is None  # cleared after eviction


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nALL {len(tests)} EpiCache eviction tests PASS")
