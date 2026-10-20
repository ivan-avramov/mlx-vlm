"""P369 phase 2 (E1c): the in-process observers on the tiny hybrid fixture.

Fork-only file. ``state_hash`` is ONE SHA-256 over the complete logical valid state of a cache list
(per leaf: type, metadata, every valid array incl. recurrent state; preallocated tails never; no side
effects). ``probe`` is a transactional width-one verify forward followed by ``abort()``. ``Recorder``
records every row the production sampler sees (ON: prefill + verifier rows; OFF: plain rows, with the
ON tokens forced), and the state at every ON commit / at the matching OFF offsets.

The end-to-end E1c checks run on the bf16 affine-quantized pairs of ``test_mtp_exactness`` (the deployed
kernel path: exact verifier kernels, Metal only). On the dense float32 fixture verifier rows and singleton
rows are NOT bit-identical (measured 2026-10-19: |dlogprob| up to ~1e-2 at logit scale 4; E1a's 1e-5 law
tolerance at scale 12 does not see it), which is why the exactness claim is about the deployed path.
"""

import sys
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
    hybrid_lm,
    mtp_drafter,
    scripted,
)
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
    """The first full-attention KV leaf — a native ``KVCache`` or a ``TurboQuantKVCache`` (P400)."""
    return next(
        c
        for c in iter_leaf_caches(caches)
        if type(c).__name__ in ("KVCache", "TurboQuantKVCache")
    )


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


# ------------------------------------------------------------------ code round 1 H1: type-specific metadata
def test_state_hash_sees_the_arrays_cache_mask_metadata():
    from mlx_vlm.models.cache import ArraysCache

    leaf = ArraysCache(2)
    leaf[0], leaf[1] = mx.zeros((1, 3, 4)), mx.ones((1, 2, 2))
    before = observe.state_hash([leaf])
    leaf.left_padding = mx.array([0])
    assert observe.state_hash([leaf]) != before
    leaf.left_padding = None
    assert observe.state_hash([leaf]) == before
    leaf.lengths = mx.array([3])
    assert observe.state_hash([leaf]) != before
    leaf.lengths = None
    leaf.advance(0)
    assert observe.state_hash([leaf]) == before


def test_state_hash_ignores_the_derived_metadata_revision_counter():
    # A cache-invalidation counter, bumped by every speculative commit / abort (so by every ON block and
    # every probe): not logical state, and hashing it would make ON and OFF differ by construction.
    from mlx_vlm.models.cache import ArraysCache

    leaf = ArraysCache(1)
    leaf[0] = mx.zeros((1, 2, 2))
    before = observe.state_hash([leaf])
    leaf.metadata_revision += 3
    assert observe.state_hash([leaf]) == before


def test_state_hash_refuses_an_active_speculative_transaction():
    from mlx_vlm.models.cache import ArraysCache

    leaf = ArraysCache(1)
    leaf[0] = mx.zeros((1, 2, 2))
    leaf.start_speculation(2)
    with pytest.raises(RuntimeError, match="active speculative transaction"):
        observe.state_hash([leaf])


def _turboquant(**bits):
    from mlx_vlm import turboquant as tq

    mx.random.seed(0)
    cache = tq.TurboQuantKVCache(**bits)
    keys = mx.random.normal((1, 2, 8, 64)).astype(mx.bfloat16)
    cache.update_and_fetch(keys, keys)
    return cache


def test_state_hash_sees_the_turboquant_split_codec_coordinates():
    cache = _turboquant(bits=3.5, key_bits=3.5, value_bits=3.5)
    codec = cache.key_codec
    assert type(codec).__name__ == "_SplitCodec"
    before = observe.state_hash([cache])
    low, high = codec.low_idx, codec.high_idx
    codec.low_idx = mx.concatenate([low[:-1], high[:1]])
    codec.high_idx = mx.concatenate([low[-1:], high[1:]])
    assert observe.state_hash([cache]) != before
    codec.low_idx, codec.high_idx = low, high
    assert observe.state_hash([cache]) == before


def test_state_hash_sees_the_turboquant_codec_kind_and_mode():
    a, b = _turboquant(bits=4), _turboquant(bits=4)
    assert observe.state_hash([a]) == observe.state_hash([b])
    b._kv_quant_mode = "prod"
    assert observe.state_hash([a]) != observe.state_hash([b])
    b._kv_quant_mode = a._kv_quant_mode
    b.key_codec = _turboquant(bits=3.5, key_bits=3.5, value_bits=3.5).key_codec
    assert observe.state_hash([a]) != observe.state_hash([b])


# ------------------------------------------------------------------ probe
def test_probe_is_neutral_and_repeatable():
    state, _ = _warm()
    before = observe.state_hash(state.cache)
    first = observe.probe(LM, state.cache, PROBE)
    assert observe.state_hash(state.cache) == before
    second = observe.probe(LM, state.cache, PROBE)
    assert first["dtype"] == second["dtype"]
    assert np.array_equal(first["bits"], second["bits"])


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


_QUANTIZED = {}
metal = pytest.mark.skipif(
    not mx.metal.is_available(), reason="exact kernels are Metal"
)


def _quantized(pick):
    if pick not in _QUANTIZED:
        _QUANTIZED[pick] = _quantized_pair(pick)
    return _QUANTIZED[pick]


def _e1c(pick, seed=5, n=12, kv=None):
    """ON with recording, then the OFF replay forced along ON's tokens, hashing at ON's offsets (``kv``: the
    KV-cache codec kwargs, e.g. the pick's deployed TurboQuant; P400)."""
    kv = kv or {}
    lm, drafter = _quantized(pick)
    model, params = fake_model(lm), deployed_sampling(pick)
    plain, _ = turn(
        model, PROMPT, PromptCacheState(), n=n, seed=seed, params=params, **kv
    )
    plain = list(plain)
    scripted(
        drafter, lambda p: plain[p] if p < len(plain) else None, high=True, vocab=64
    )
    with observe.Recorder(mode="on", lm=lm, probe_token=PROBE) as on:
        emitted, _ = turn(
            model,
            PROMPT,
            PromptCacheState(),
            drafter=drafter,
            n=n,
            seed=seed,
            params=params,
            **kv,
        )
    assert list(emitted) == plain and on.draft_rounds > 0
    offsets = [on.prompt_end["offset"]] + [c["offset"] for c in on.commits]
    return lm, model, params, on, list(emitted), offsets


@metal
@pytest.mark.parametrize("pick", PICKS)
def test_quantized_e1c_on_equals_off_rows_states_and_probes(pick):
    lm, model, params, on, emitted, offsets = _e1c(pick)
    n = len(emitted)
    with observe.Recorder(
        mode="off", force_tokens=emitted, hash_offsets=offsets, lm=lm, probe_token=PROBE
    ) as off:
        turn(model, PROMPT, PromptCacheState(), n=n, seed=5, params=params)
    rep = compare.compare_streams(on.rows, off.rows, n, params)
    assert rep["pass"] and rep["raw_equal"] == n, rep
    states = observe.compare_states(on, off)
    assert states["pass"] and states["compared"] == len(offsets), states


@metal
@pytest.mark.parametrize("pick", PICKS)
def test_quantized_e1c_detects_an_old_prefix_byte_flip_at_one_offset(pick):
    lm, model, params, on, emitted, offsets = _e1c(pick)
    target = offsets[len(offsets) // 2]

    def corrupt(caches, offset):
        if offset == target:
            kv = _kv_leaf(caches)
            kv.keys[..., 0:1, :] = kv.keys[..., 0:1, :] * 2.0

    with observe.Recorder(
        mode="off",
        force_tokens=emitted,
        hash_offsets=offsets,
        lm=lm,
        probe_token=PROBE,
        before_hash=corrupt,
    ) as off:
        turn(model, PROMPT, PromptCacheState(), n=len(emitted), seed=5, params=params)
    states = observe.compare_states(on, off)
    assert not states["pass"] and target in states["mismatched_offsets"]


@metal
def test_quantized_e1c_detects_an_in_support_change_of_one_verifier_row():
    lm, model, params, on, emitted, offsets = _e1c(PICKS[0])
    with observe.Recorder(mode="off", force_tokens=emitted) as off:
        turn(model, PROMPT, PromptCacheState(), n=len(emitted), seed=5, params=params)
    position = 3
    index = max(i for i, (p, _) in enumerate(on.rows) if p == position)
    row = compare.from_record(on.rows[index][1])
    top = int(mx.argmax(row).item())
    bumped = mx.where(mx.arange(row.size) == top, row + 0.01, row)
    on.rows[index] = (position, compare.as_record(bumped.astype(row.dtype)))
    rep = compare.compare_streams(on.rows, off.rows, len(emitted), params)
    assert not rep["pass"] and [m["position"] for m in rep["mismatches"]] == [position]


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


# ------------------------------------------------------------------ smoke 3 (P398): rows after an emission cut
def _budget_setup():
    from test_mtp_thinking_budget import THINKING_PROMPT, ThinkingTokenizer

    lm = hybrid_lm(logit_scale=4.0)
    model = fake_model(lm)
    free, _ = turn(model, THINKING_PROMPT, PromptCacheState(), n=16)
    return lm, model, list(free), THINKING_PROMPT, ThinkingTokenizer


def _budget_on(budget, n=12):
    """MTP-ON with a full-accept drafter proposing the UNBUDGETED path, so a block runs across the trip."""
    lm, model, free, prompt, tokenizer = _budget_setup()
    drafter = mtp_drafter(lm)
    scripted(drafter, lambda p: free[p] if p < len(free) else None, high=True, vocab=64)
    kw = dict(tokenizer=tokenizer(), thinking_budget=budget, enable_thinking=True)
    with observe.Recorder(mode="on") as rec:
        emitted, _ = turn(
            model, prompt, PromptCacheState(), drafter=drafter, n=n, seed=5, **kw
        )
    plain, _ = turn(model, prompt, PromptCacheState(), n=n, seed=5, **kw)
    assert list(emitted) == list(plain)  # production: MTP emits plain's sequence
    return rec, list(emitted)


@pytest.mark.parametrize("budget", range(0, 8))
def test_on_rows_stop_at_each_blocks_emission_cut_and_forced_tokens_are_recorded(
    budget,
):
    """A thinking-budget trip cuts a verified block; its rows past the cut are conditioned on drafts that
    were replaced (unreachable history) and forced tokens are emitted with no sampler call — so the ON
    recorder must keep only each block's first n_emit rows and list the forced tokens (smoke 3: a stale
    row at the forced ``</think>`` position read as an E1c row mismatch with TV 1)."""
    from test_mtp_thinking_budget import CLOSE, NEWLINE

    rec, emitted = _budget_on(budget)
    rows, missing = compare.latest_rows(rec.rows, len(emitted))
    trip = next(
        i for i in range(len(emitted) - 1) if emitted[i : i + 2] == [NEWLINE, CLOSE]
    )
    assert trip in missing and trip + 1 in missing, (
        trip,
        missing,
    )  # no stale row survives
    assert [t for _, t in rec.forced[:2]] == [NEWLINE, CLOSE], rec.forced
    # round 3 E6: each forced pop carries its intended position; the emitted ones are exactly the row-less
    # positions, an unemitted one sits exactly at the end
    inside = [(p, t) for p, t in rec.forced if p < len(emitted)]
    assert [p for p, _ in inside] == missing, (missing, rec.forced)
    assert all(emitted[p] == t for p, t in inside)
    assert [p for p, _ in rec.forced if p >= len(emitted)] in ([], [len(emitted)])
    got = compare.forced_positions(rec, emitted, finish_reason="length")
    assert got == missing


def test_forced_positions_refuse_inconsistent_bookkeeping():
    rec = type("R", (), {})()
    rec.rows = [(0, None), (1, None), (3, None)]
    tokens = [5, 6, 63, 7]
    rec.forced = [(2, 63)]
    assert compare.forced_positions(rec, tokens, finish_reason="stop") == [2]
    rec.forced = [
        (2, 63),
        (4, 62),
    ]  # the next forced token was popped but the length budget ended first
    assert compare.forced_positions(rec, tokens, finish_reason="length") == [2]
    for forced, finish in (
        ([(2, 62)], "stop"),  # the row-less position's token is not the forced one
        ([], "stop"),  # a row-less position with no forced emission
        ([(2, 63), (4, 62)], "stop"),  # an unemitted pop needs a length finish
        ([(2, 63), (4, 62), (5, 61)], "length"),
        (
            [(2, 63), (3, 7)],
            "length",
        ),  # round 3 E6: an EMITTED forced position kept a (stale) row
        ([(1, 6)], "stop"),  # a forced position that is not row-less
    ):
        rec.forced = forced
        with pytest.raises(RuntimeError, match="forced"):
            compare.forced_positions(rec, tokens, finish_reason=finish)


def test_compare_streams_excludes_only_the_given_positions():
    def row(hot):
        return compare.as_record(
            mx.array([4.0 if j == hot else 0.0 for j in range(32)])
        )

    rows = [(i, row(i)) for i in range(4)]
    off = list(rows)
    off[2] = (2, row(9))
    assert not compare.compare_streams(rows, off, 4, PARAMS)["pass"]
    rep = compare.compare_streams(
        [r for r in rows if r[0] != 2], off, 4, PARAMS, exclude=[2]
    )
    assert rep["pass"] and rep["excluded"] == [2] and rep["compared"] == 3, rep
    # an excluded position never hides a missing row elsewhere
    rep = compare.compare_streams(rows[:2], off, 4, PARAMS, exclude=[2])
    assert not rep["pass"] and rep["missing_on"] == [3]


# ------------------------------------------------------------------ round 4 E8: plain decoding's forced schedule
def _replay(emitted, budget, n=12):
    """The OFF recorder replaying ON's tokens under a plain criteria with thinking budget ``budget``."""
    _, model, _, prompt, tokenizer = _budget_setup()
    kw = dict(tokenizer=tokenizer(), thinking_budget=budget, enable_thinking=True)
    with observe.Recorder(mode="off", force_tokens=emitted) as off:
        replayed, _ = turn(model, prompt, PromptCacheState(), n=n, seed=5, **kw)
    return off, list(replayed)


@pytest.mark.parametrize("budget", [2, 5])
def test_the_off_recorder_observes_plain_forcing_without_letting_it_override(budget):
    """C185 round 4 E8: a forced replay must expose where PLAIN decoding would force: its pops are recorded
    (position, token) and consumed but never applied, so the replay stays on ON's tokens while a schedule
    difference (ON forcing where plain would not) becomes visible instead of hiding behind the replay.
    """
    on, emitted = _budget_on(budget)
    for other in (
        budget - 1,
        budget + 1,
    ):  # plain would force elsewhere: its pops never override
        assert _replay(emitted, other)[1] == emitted, other
    same, replayed = _replay(emitted, budget)
    assert replayed == emitted
    assert compare.forced_schedule(on.forced, same.forced, len(emitted))["pass"]
    for other in (budget - 1, budget + 1):  # plain would force elsewhere
        off, replayed = _replay(emitted, other)
        assert replayed == emitted  # the plain pops never override the replay
        got = compare.forced_schedule(on.forced, off.forced, len(emitted))
        assert not got["pass"], (budget, other, got)


def test_forced_schedule_compares_only_emitted_positions():
    fs = compare.forced_schedule
    assert fs([(3, 63), (4, 62)], [(3, 63), (4, 62)], 5)["pass"]
    assert fs([(3, 63), (5, 62)], [(3, 63)], 5)[
        "pass"
    ]  # a pop past the output emits nothing
    assert not fs([(3, 63), (4, 62)], [(3, 63)], 5)["pass"]
    assert not fs([(3, 63)], [(2, 63)], 5)["pass"]
    assert not fs([(3, 63)], [(3, 62)], 5)["pass"]
    got = fs([(3, 63)], [], 5)
    assert got == dict(on=[[3, 63]], off=[], **{"pass": False})


# ------------------------------------------------------------------ P400: the deployed KV codec (TurboQuant)
TQ4 = dict(kv_bits=4, kv_quant_scheme="turboquant", quantized_kv_start=0)
AFFINE4 = dict(
    kv_bits=4, kv_quant_scheme="uniform", quantized_kv_start=0
)  # what the server builds today (P405)
DEPLOYED_KV = {PICKS[0]: {}, PICKS[1]: TQ4}
KV_CASES = [
    (PICKS[0], {}, "KVCache"),
    (PICKS[1], TQ4, "TurboQuantKVCache"),
    (PICKS[1], AFFINE4, "QuantizedKVCache"),
]


@metal
@pytest.mark.parametrize(
    "n", [14, 17]
)  # (n - 1) % 3 == 1: the terminal round has one token of budget (width 2)
@pytest.mark.parametrize(
    ("pick", "kv", "want"), KV_CASES, ids=[f"{p}-{w}" for p, _, w in KV_CASES]
)
def test_quantized_e1c_on_the_deployed_kv_codec(pick, kv, want, n):
    lm, model, params, on, emitted, offsets = _e1c(pick, n=n, kv=kv)
    leaf_kinds = {type(c).__name__ for c in iter_leaf_caches(on._cache or [])}
    with observe.Recorder(
        mode="off", force_tokens=emitted, hash_offsets=offsets, lm=lm, probe_token=PROBE
    ) as off:
        turn(
            model,
            PROMPT,
            PromptCacheState(),
            n=len(emitted),
            seed=5,
            params=params,
            **kv,
        )
        off_kinds = {type(c).__name__ for c in iter_leaf_caches(off._cache or [])}
    kinds = leaf_kinds | off_kinds  # a Prealloc* subclass name counts as its codec
    assert any(k.endswith(want) for k in kinds), (want, leaf_kinds, off_kinds)
    forced = compare.forced_positions(on, emitted, finish_reason="length")
    rows = compare.compare_streams(
        on.rows, off.rows, len(emitted), params, exclude=forced
    )
    assert rows["pass"] and rows["raw_equal"] == len(emitted) - len(forced), rows
    assert compare.forced_schedule(on.forced, off.forced, len(emitted))["pass"]
    states = observe.compare_states(on, off)
    assert states["pass"] and states["compared"] == len(offsets), states


@metal
def test_quantized_e1c_detects_a_turboquant_packed_state_change():
    """The state check sees a changed TurboQuant norm in an OLD prefix position (the corruption positive, on TQ)."""
    kv = DEPLOYED_KV[PICKS[1]]
    lm, model, params, on, emitted, offsets = _e1c(PICKS[1], kv=kv)
    target = offsets[len(offsets) // 2]

    def corrupt(caches, offset):
        if offset == target:
            leaf = _kv_leaf(caches)
            assert type(leaf).__name__ == "TurboQuantKVCache", type(leaf).__name__
            norms = leaf.keys[0]
            norms[..., 0:1] = norms[..., 0:1] * 2.0

    with observe.Recorder(
        mode="off",
        force_tokens=emitted,
        hash_offsets=offsets,
        lm=lm,
        probe_token=PROBE,
        before_hash=corrupt,
    ) as off:
        turn(
            model,
            PROMPT,
            PromptCacheState(),
            n=len(emitted),
            seed=5,
            params=params,
            **kv,
        )
    states = observe.compare_states(on, off)
    assert not states["pass"] and target in states["mismatched_offsets"]


# ------------------------------------------------------------------ smoke 4: either TurboQuant class (the live leaf census)
def _batch_turboquant(**bits):
    from mlx_vlm import turboquant as tq

    mx.random.seed(0)
    cache = tq.BatchTurboQuantKVCache([0], **bits)
    keys = mx.random.normal((1, 2, 8, 64)).astype(mx.bfloat16)
    cache.update_and_fetch(keys, keys)
    return cache


def test_state_hash_sees_the_codec_of_a_batch_turboquant_cache_too():
    """Smoke 4: the live `Qwen3.8-27B-mlx-uniform-4bit` leaf matched neither positive's branch — the hash's codec
    metadata must not depend on which TurboQuant class (single or batch) the server built.
    """
    a, b = _batch_turboquant(bits=4), _batch_turboquant(bits=4)
    assert observe.state_hash([a]) == observe.state_hash([b])
    b.key_codec = _turboquant(bits=3.5, key_bits=3.5, value_bits=3.5).key_codec
    assert observe.state_hash([a]) != observe.state_hash([b])
