"""P369 phase 2 (E1c): the emitted-row selection and the post-transform comparator.

Fork-only file. The harness records every target row the production sampler sees, keyed by
emitted position (``_PositionedTargetSampler.sample_target`` positions): MTP verifier rows and
the prefill's first row on the ON arm, the plain decoding rows on the OFF replay. A position's
emitted-history row is the LAST one recorded for it (a later round re-records the positions a
rejection made counterfactual). Rows are compared after the production filter and temperature
(``_PositionedTargetSampler._filter``, ``* (1 / temperature)``, softmax) bit for bit; the raw
logprob rows are compared too, so a change the filter discards is still visible.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))

from p369_phase2 import compare  # noqa: E402

PARAMS = dict(temperature=0.5, top_p=0.95, top_k=20, min_p=0.0)
V = 64


def _row(seed, dtype=np.float32):
    rng = np.random.default_rng(seed)
    logits = (rng.standard_normal(V) * 3.0).astype(np.float32)
    lp = logits - np.log(np.sum(np.exp(logits.astype(np.float64)))).astype(np.float32)
    return lp.astype(dtype)


def _ulp_up(value):
    return np.nextafter(np.float32(value), np.float32(np.inf), dtype=np.float32)


def test_canonical_is_the_production_filter_and_temperature():
    lp = _row(1)
    p = compare.canonical(lp, PARAMS)
    assert p.dtype == np.float32 and p.shape == (V,)
    assert abs(float(p.sum()) - 1.0) < 1e-5
    # Support: at most top_k tokens, all among the highest logprobs.
    support = np.flatnonzero(p > 0)
    assert 0 < len(support) <= PARAMS["top_k"]
    assert set(support) <= set(np.argsort(-lp)[: PARAMS["top_k"]])


def test_latest_row_per_position_wins_and_tail_rows_are_excluded():
    a, b, c, d = (_row(s) for s in range(4))
    records = [(0, a), (1, b), (2, c), (2, d), (5, a)]  # pos 2 re-recorded; 5 >= n
    rows, missing = compare.latest_rows(records, 4)
    assert sorted(rows) == [0, 1, 2]
    assert rows[2] is d
    assert missing == [3]


def test_identical_streams_pass():
    rows = [(i, _row(i)) for i in range(6)]
    rep = compare.compare_streams(rows, [(i, r.copy()) for i, r in rows], 6, PARAMS)
    assert rep["pass"] and rep["compared"] == 6 and rep["canonical_equal"] == 6
    assert rep["raw_equal"] == 6 and rep["mismatches"] == []


def test_an_in_support_one_ulp_change_fails_with_metrics():
    rows = [(i, _row(i)) for i in range(4)]
    off = [(i, r.copy()) for i, r in rows]
    top = int(np.argmax(off[2][1]))
    off[2][1][top] = _ulp_up(off[2][1][top])
    rep = compare.compare_streams(rows, off, 4, PARAMS)
    assert not rep["pass"]
    (m,) = rep["mismatches"]
    assert m["position"] == 2 and m["tv"] > 0 and m["max_abs_dp"] > 0
    assert rep["raw_equal"] == 3


def test_an_out_of_support_change_is_raw_only():
    rows = [(i, _row(i)) for i in range(3)]
    off = [(i, r.copy()) for i, r in rows]
    low = int(np.argmin(off[1][1]))  # far outside top_k
    off[1][1][low] = _ulp_up(off[1][1][low])
    rep = compare.compare_streams(rows, off, 3, PARAMS)
    assert rep["canonical_equal"] == 3 and rep["raw_equal"] == 2
    assert rep["pass"]  # output-relevant criterion: the filtered distribution
    assert rep["raw_only_differences"] == [1]


def test_a_support_change_is_counted():
    rows = [(0, _row(7))]
    off = [(0, rows[0][1].copy())]
    order = np.argsort(-off[0][1])
    k_in, k_out = int(order[PARAMS["top_k"] - 1]), int(order[PARAMS["top_k"]])
    off[0][1][k_out] = off[0][1][k_in] + np.float32(1e-3)  # swaps the k-th token
    rep = compare.compare_streams(rows, off, 1, PARAMS)
    assert not rep["pass"] and rep["support_changes"] == 1
    assert (
        rep["mismatches"][0]["support_on_only"]
        or rep["mismatches"][0]["support_off_only"]
    )


def test_missing_rows_fail():
    rows = [(i, _row(i)) for i in range(3)]
    rep = compare.compare_streams(rows, rows[:2], 3, PARAMS)
    assert not rep["pass"] and rep["missing_off"] == [2]


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_rows_keep_their_production_dtype(dtype):
    import mlx.core as mx

    lp = mx.array(_row(3)).astype(getattr(mx, dtype))
    rec = compare.as_record(lp)
    assert rec["dtype"] == dtype
    back = compare.from_record(rec)
    assert back.dtype == getattr(mx, dtype)
    assert mx.array_equal(back, lp).item()
