"""E1c: emitted-row selection and the post-transform comparator.

Rows are what the production sampler saw: ``logits - logsumexp(logits)`` in the model's dtype, keyed by
emitted position. ``canonical`` applies exactly the production transform of a seeded deployed request
(``_PositionedTargetSampler._filter`` then ``* (1 / temperature)``) and the softmax the categorical draw
samples from. A position passes when its canonical distributions are bit-identical (support and values);
raw-row differences that the filter discards are reported separately.

A row is a 1-D mlx array, a float32 numpy array, or a record from ``as_record`` (dtype + exact bits).
"""

import numpy as np

_MAX_LISTED = 50
_VIEW = {"bfloat16": "uint16", "float16": "uint16", "float32": "uint32"}


def _mx():
    import mlx.core as mx

    return mx


def as_record(logprobs):
    """A storable row: the dtype name and the exact bits of a 1-D mlx row."""
    mx = _mx()
    row = logprobs.reshape(-1)
    dtype = str(row.dtype).split(".")[-1]
    if dtype not in _VIEW:
        raise TypeError(f"unsupported logprob dtype {dtype}")
    return {"dtype": dtype, "bits": np.array(row.view(getattr(mx, _VIEW[dtype])))}


def from_record(row):
    """The 1-D mlx row, in its original dtype."""
    mx = _mx()
    if isinstance(row, dict):
        return mx.array(row["bits"]).view(getattr(mx, row["dtype"]))
    if isinstance(row, np.ndarray):
        return mx.array(row.astype(np.float32, copy=False))
    return row.reshape(-1)


def _bits(row):
    if isinstance(row, dict):
        return row["dtype"], np.asarray(row["bits"])
    if isinstance(row, np.ndarray):
        return "float32", np.ascontiguousarray(row, dtype=np.float32).view(np.uint32)
    rec = as_record(row)
    return rec["dtype"], rec["bits"]


def canonical(row, params):
    """The production post-transform distribution of one row (float32 numpy)."""
    mx = _mx()
    from mlx_vlm.generate import ar

    sampler = ar._PositionedTargetSampler(
        temperature=float(params["temperature"]),
        top_p=float(params["top_p"]),
        seed=0,
        top_k=int(params["top_k"]),
        min_p=float(params["min_p"]),
    )
    filtered = sampler._filter(from_record(row).reshape(1, -1))
    probs = mx.softmax(filtered * (1 / sampler.temperature), axis=-1)
    return np.array(probs.astype(mx.float32)).reshape(-1)


def latest_rows(records, n):
    """``{position: row}`` for positions ``0..n-1`` from ordered ``(position, row)`` records (the last
    record of a position wins), plus the positions that have none."""
    rows = {}
    for position, row in records:
        if 0 <= int(position) < n:
            rows[int(position)] = row
    return rows, [p for p in range(n) if p not in rows]


def _margin(p):
    """Top-1 minus top-2 probability (near-tie diagnosis only, never an excuse)."""
    top = np.sort(p.astype(np.float64))[::-1]
    return float(top[0] - top[1]) if top.size > 1 else float(top[0])


def compare_streams(on_records, off_records, n, params):
    """Compare the emitted-position rows ``0..n-1`` of the ON and OFF streams."""
    on, missing_on = latest_rows(on_records, n)
    off, missing_off = latest_rows(off_records, n)
    report = dict(
        positions=n,
        compared=0,
        missing_on=missing_on,
        missing_off=missing_off,
        raw_equal=0,
        canonical_equal=0,
        raw_only_differences=[],
        support_changes=0,
        max_tv=0.0,
        max_abs_dp=0.0,
        mismatches=[],
    )
    for position in sorted(set(on) & set(off)):
        report["compared"] += 1
        a_dtype, a_bits = _bits(on[position])
        b_dtype, b_bits = _bits(off[position])
        raw_same = a_dtype == b_dtype and np.array_equal(a_bits, b_bits)
        report["raw_equal"] += int(raw_same)
        pa, pb = canonical(on[position], params), canonical(off[position], params)
        if np.array_equal(pa.view(np.uint32), pb.view(np.uint32)):
            report["canonical_equal"] += 1
            if not raw_same:
                report["raw_only_differences"].append(position)
            continue
        sa = set(np.flatnonzero(pa > 0).tolist())
        sb = set(np.flatnonzero(pb > 0).tolist())
        dp = np.abs(pa.astype(np.float64) - pb.astype(np.float64))
        tv = float(0.5 * dp.sum())
        report["support_changes"] += int(sa != sb)
        report["max_tv"] = max(report["max_tv"], tv)
        report["max_abs_dp"] = max(report["max_abs_dp"], float(dp.max()))
        if len(report["mismatches"]) < _MAX_LISTED:
            report["mismatches"].append(
                dict(
                    position=position,
                    dtype=[a_dtype, b_dtype],
                    support_on_only=sorted(sa - sb),
                    support_off_only=sorted(sb - sa),
                    max_abs_dp=float(dp.max()),
                    tv=tv,
                    top_margin_on=_margin(pa),
                    top_margin_off=_margin(pb),
                )
            )
    report["mismatch_count"] = report["compared"] - report["canonical_equal"]
    report["pass"] = bool(
        n > 0 and not missing_on and not missing_off and report["mismatch_count"] == 0
    )
    return report
