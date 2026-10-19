"""P369 phase 2: the worker's CPU-testable parts and the verifier key-length recorder.

Fork-only file. The worker boot (the real CLI with ``uvicorn.run`` intercepted, a real model) is exercised by
the GPU smoke; here: exact prompt sizing by dry-run counts, the row / snapshot artifacts, frozen verify-scan
counters around probes, the served-state and source checks, atomic results, and ``KeyLengths`` (every
verifier block's key-length span, from which straddles of each MLX threshold are proven).
"""

import json
import sys
import types
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from test_mtp_exactness import (
    PICKS,
    PROMPT,
    _quantized_pair,
    deployed_sampling,
    fake_model,
    scripted,
)
from test_mtp_stop_state import turn

from mlx_vlm.generate.common import PromptCacheState

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import compare, observe, worker  # noqa: E402

metal = pytest.mark.skipif(
    not mx.metal.is_available(), reason="exact kernels are Metal"
)


# ------------------------------------------------------------------ prompt sizing
def test_size_prompt_hits_an_exact_count_with_nudges():
    def count(text):  # two tokens per word, one per trailing nudge char
        words = text.split()
        return sum(1 if len(w) == 1 else 2 for w in words) + 7

    text, tokens = worker.size_prompt(count, 1001, prefix="P", suffix="S")
    assert tokens == 1001 and count(text) == 1001


def test_size_prompt_reports_an_unreachable_target():
    count = lambda text: 3 * len(text.split())  # noqa: E731 only multiples of 3
    text, closest = worker.size_prompt(count, 1000, prefix="P", suffix="S")
    assert text is None and abs(closest - 1000) <= 2


# ------------------------------------------------------------------ artifacts
def test_rows_round_trip_with_their_exact_bits(tmp_path):
    rows = [
        (p, compare.as_record(mx.random.normal((7,)).astype(mx.bfloat16)))
        for p in (0, 1, 1)
    ]
    worker.save_rows(tmp_path / "r.npz", rows)
    back = worker.load_rows(tmp_path / "r.npz")
    assert [p for p, _ in back] == [0, 1, 1]
    for (_, a), (_, b) in zip(rows, back):
        assert a["dtype"] == b["dtype"] == "bfloat16"
        assert np.array_equal(a["bits"], b["bits"])


def test_rows_refuse_mixed_dtypes(tmp_path):
    rows = [
        (0, compare.as_record(mx.zeros((3,), dtype=mx.bfloat16))),
        (1, compare.as_record(mx.zeros((3,), dtype=mx.float32))),
    ]
    with pytest.raises(RuntimeError, match="mixed row dtypes"):
        worker.save_rows(tmp_path / "r.npz", rows)


def test_snapshots_round_trip(tmp_path):
    probe = compare.as_record(mx.arange(5).astype(mx.float32))
    snaps = [
        dict(offset=8, sha="a" * 64, neutral=True, kind="prompt_end", probe=probe),
        dict(offset=11, sha="b" * 64, neutral=False, kind="block", probe=probe),
    ]
    worker.save_snapshots(tmp_path / "s.npz", snaps)
    back = worker.load_snapshots(tmp_path / "s.npz")
    for a, b in zip(snaps, back):
        assert (a["offset"], a["sha"], a["neutral"], a["kind"]) == (
            b["offset"],
            b["sha"],
            b["neutral"],
            b["kind"],
        )
        assert np.array_equal(a["probe"]["bits"], b["probe"]["bits"])


def test_probe_counters_frozen_restores_production_counts():
    policy = types.SimpleNamespace(
        _counts={"verify_blocks_len1": 4, "verify_blocks_straddle": 1}
    )
    with worker.probe_counters_frozen(policy):
        policy._counts["verify_blocks_len1"] += 1
    assert policy._counts == {"verify_blocks_len1": 4, "verify_blocks_straddle": 1}
    with worker.probe_counters_frozen(None):
        pass


# ------------------------------------------------------------------ served state / sources / atomic
GOOD = dict(
    drafter_loaded=True,
    draft_kind="mtp",
    verify_policy="joint_v1",
    apc_manager=False,
    apc_env=None,
    session_max=1,
    session_max_env="1",
)
EXPECT = dict(drafter=True, verify_policy="joint_v1")


def test_assert_served_accepts_the_expected_state():
    worker.assert_served(GOOD, EXPECT)


@pytest.mark.parametrize(
    "change",
    [
        dict(drafter_loaded=False),
        dict(draft_kind="suffix"),
        dict(verify_policy=None),
        dict(apc_manager=True),
        dict(apc_env="1"),
        dict(session_max=2),
        dict(session_max_env="2"),
    ],
)
def test_assert_served_refuses_any_mismatch(change):
    with pytest.raises(RuntimeError, match="served state mismatch"):
        worker.assert_served(dict(GOOD, **change), EXPECT)


def test_check_sources_refuses_a_module_outside_the_reviewed_tree(
    tmp_path, monkeypatch
):
    root = Path(worker.__file__).resolve().parents[2]
    worker.check_sources(root, DEV)
    stray = types.ModuleType("mlx_vlm._p369_stray")
    stray.__file__ = str(tmp_path / "elsewhere.py")
    monkeypatch.setitem(sys.modules, "mlx_vlm._p369_stray", stray)
    with pytest.raises(RuntimeError, match="outside the reviewed sources"):
        worker.check_sources(root, DEV)


def test_atomic_write_leaves_only_the_final_file(tmp_path):
    worker.atomic_write(tmp_path / "result.json", {"a": 1})
    assert json.loads((tmp_path / "result.json").read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


# ------------------------------------------------------------------ KeyLengths
@metal
def test_key_lengths_records_every_verifier_block_span():
    lm, drafter = _quantized_pair(PICKS[1])
    model, params = fake_model(lm), deployed_sampling(PICKS[1])
    plain, _ = turn(model, PROMPT, PromptCacheState(), n=9, seed=5, params=params)
    plain = list(plain)
    scripted(
        drafter, lambda p: plain[p] if p < len(plain) else None, high=True, vocab=64
    )
    with observe.KeyLengths() as kl:
        turn(
            model,
            PROMPT,
            PromptCacheState(),
            drafter=drafter,
            n=9,
            seed=5,
            params=params,
        )
    assert kl.blocks and all(b["width"] > 1 for b in kl.blocks)
    first = kl.blocks[0]
    assert (
        first["first_key"] == len(PROMPT) + 1
        and first["last_key"] == len(PROMPT) + first["width"]
    )
    t = first["first_key"]  # a threshold the first block spans
    assert observe.straddles(kl.blocks, t) or first["width"] == 1
    assert not observe.straddles(kl.blocks, 10_000)
