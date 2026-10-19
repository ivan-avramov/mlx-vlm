"""P369 phase 2 (E1c): the in-process observers on the tiny hybrid fixture.

Fork-only file. ``state_hash`` is ONE SHA-256 over the complete logical valid state of a cache list
(per leaf: type, metadata, every valid array incl. recurrent state; preallocated tails never; no side
effects). ``probe`` is a transactional width-one verify forward followed by ``abort()``. ``Recorder``
records every row the production sampler sees (ON: prefill + verifier rows; OFF: plain rows, with the
ON tokens forced), and the state at every ON commit / at the matching OFF offsets.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from test_mtp_exactness import PROMPT, fake_model, hybrid_lm, mtp_drafter, scripted
from test_mtp_stop_state import turn

from mlx_vlm.generate.common import PromptCacheState
from mlx_vlm.speculative.cache_state import iter_leaf_caches

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))

from p369_phase2 import compare, observe  # noqa: E402

N = 9
PROBE = 5
PARAMS = dict(temperature=0.5, top_p=0.95, top_k=20, min_p=0.0)
LM = hybrid_lm(logit_scale=4.0)
MODEL = fake_model(LM)


def _warm(n=N, seed=5):
    state = PromptCacheState()
    emitted, _ = turn(MODEL, PROMPT, state, n=n, seed=seed)
    return state, list(emitted)


def _kv_leaf(caches):
    return next(c for c in iter_leaf_caches(caches) if type(c).__name__ == "KVCache")


def _recurrent_leaf(caches):
    return next(c for c in iter_leaf_caches(caches) if type(c).__name__ != "KVCache")


def _drafter(path):
    drafter = mtp_drafter(LM)
    scripted(drafter, lambda p: path[p] if p < len(path) else None, high=True, vocab=64)
    return drafter


# ------------------------------------------------------------------ state_hash
def test_state_hash_is_deterministic_across_identical_runs():
    a, _ = _warm()
    b, _ = _warm()
    assert observe.state_hash(a.cache) == observe.state_hash(b.cache)


def test_state_hash_ignores_the_preallocated_tail_but_sees_an_old_prefix_byte():
    state, _ = _warm()
    kv = _kv_leaf(state.cache)
    assert kv.keys.shape[2] > kv.offset, "fixture needs a tail beyond the offset"
    before = observe.state_hash(state.cache)
    kv.keys[..., kv.offset :, :] = 7.0
    assert observe.state_hash(state.cache) == before
    kv.keys[..., 0:1, :] = kv.keys[..., 0:1, :] + 1.0
    assert observe.state_hash(state.cache) != before


def test_state_hash_sees_recurrent_state_and_offsets():
    state, _ = _warm()
    before = observe.state_hash(state.cache)
    leaf = _recurrent_leaf(state.cache)
    arrays = [a for a in leaf.state if a is not None]
    arrays[-1][..., 0] = arrays[-1][..., 0] + 1.0
    assert observe.state_hash(state.cache) != before
    other, _ = _warm()
    _kv_leaf(other.cache).offset -= 1
    assert observe.state_hash(other.cache) != observe.state_hash(_warm()[0].cache)


def test_state_hash_has_no_side_effects():
    state, _ = _warm()
    snapshot = [
        {k: id(v) for k, v in vars(leaf).items()}
        for leaf in iter_leaf_caches(state.cache)
    ]
    observe.state_hash(state.cache)
    assert snapshot == [
        {k: id(v) for k, v in vars(leaf).items()}
        for leaf in iter_leaf_caches(state.cache)
    ]


def test_state_hash_refuses_an_unknown_state_type():
    class Odd:
        offset = 1
        state = (object(),)
        meta_state = ""

    with pytest.raises(TypeError):
        observe.state_hash([Odd()])


# ------------------------------------------------------------------ probe
def test_probe_is_neutral_and_repeatable():
    state, _ = _warm()
    before = observe.state_hash(state.cache)
    first = observe.probe(LM, state.cache, PROBE)
    assert observe.state_hash(state.cache) == before
    second = observe.probe(LM, state.cache, PROBE)
    assert compare._bits(first) == compare._bits(second) or np.array_equal(
        compare._bits(first)[1], compare._bits(second)[1]
    )


# ------------------------------------------------------------------ Recorder
def test_off_recorder_records_every_plain_row_and_forcing_reproduces_a_path():
    _, emitted = _warm()
    with observe.Recorder(mode="off", force_tokens=emitted) as rec:
        forced, _ = turn(MODEL, PROMPT, PromptCacheState(), n=N, seed=999)
    assert list(forced) == emitted  # a different seed still walks the forced path
    rows, missing = compare.latest_rows(rec.rows, N)
    assert not missing and sorted(rows) == list(range(N))


def test_on_recorder_sees_verifier_rows_and_hashes_every_commit():
    _, path = _warm(n=N)
    with observe.Recorder(mode="on", lm=LM, probe_token=PROBE) as rec:
        emitted, _ = turn(
            MODEL, PROMPT, PromptCacheState(), drafter=_drafter(path), n=N, seed=5
        )
    assert list(emitted) == path
    rows, missing = compare.latest_rows(rec.rows, N)
    assert not missing
    assert rec.commits, "no commit observed"
    offsets = [c["offset"] for c in rec.commits]
    assert offsets == sorted(offsets) and len(set(offsets)) == len(offsets)
    assert all(c["neutral"] for c in rec.commits)
    assert rec.prompt_end["offset"] == len(PROMPT)
    assert rec.draft_rounds > 0


def _e1c():
    """ON with recording, then the OFF replay forced along ON's tokens, hashing at ON's offsets."""
    _, path = _warm(n=N)
    with observe.Recorder(mode="on", lm=LM, probe_token=PROBE) as on:
        emitted, _ = turn(
            MODEL, PROMPT, PromptCacheState(), drafter=_drafter(path), n=N, seed=5
        )
    offsets = [on.prompt_end["offset"]] + [c["offset"] for c in on.commits]
    return on, emitted, offsets


def test_tiny_e1c_on_equals_off_rows_states_and_probes():
    on, emitted, offsets = _e1c()
    with observe.Recorder(
        mode="off",
        force_tokens=list(emitted),
        hash_offsets=offsets,
        lm=LM,
        probe_token=PROBE,
    ) as off:
        turn(MODEL, PROMPT, PromptCacheState(), n=N, seed=5)
    rep = compare.compare_streams(on.rows, off.rows, N, PARAMS)
    assert rep["pass"], rep
    states = observe.compare_states(on, off)
    assert states["pass"], states
    assert states["compared"] == len(offsets)


def test_tiny_e1c_detects_an_old_prefix_byte_flip_at_one_offset():
    on, emitted, offsets = _e1c()
    target = offsets[len(offsets) // 2]

    def corrupt(caches, offset):
        if offset == target:
            kv = _kv_leaf(caches)
            kv.keys[..., 0:1, :] = kv.keys[..., 0:1, :] * 2.0

    with observe.Recorder(
        mode="off",
        force_tokens=list(emitted),
        hash_offsets=offsets,
        lm=LM,
        probe_token=PROBE,
        before_hash=corrupt,
    ) as off:
        turn(MODEL, PROMPT, PromptCacheState(), n=N, seed=5)
    states = observe.compare_states(on, off)
    assert not states["pass"] and target in states["mismatched_offsets"]


def test_recorder_uninstalls_everything():
    from mlx_vlm.generate import ar, dispatch
    from mlx_vlm.speculative import mtp

    originals = (
        ar._PositionedTargetSampler.sample_target,
        dispatch.generate_step,
        mtp._MTPVerifyResult.commit,
        mtp._mtp_commit_pending_token,
    )
    with observe.Recorder(mode="on", lm=LM, probe_token=PROBE):
        pass
    assert originals == (
        ar._PositionedTargetSampler.sample_target,
        dispatch.generate_step,
        mtp._MTPVerifyResult.commit,
        mtp._mtp_commit_pending_token,
    )
