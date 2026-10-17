"""thinking_budget in the MTP round loops (O40, 2026-08-24; P369 M3, 2026-10-17).

Fork-only file. Contract (P369/M3, replacing O40's "<= one draft block of overshoot"):
MTP must emit plain decoding's exact sequence. Plain decoding (``generate_step``,
``GenerationBatch``) lets the caller drive ``ThinkingBudgetCriteria`` with every
emitted token and then pops ``pop_forced_token_id()``; a forced token REPLACES the
next token, so the trip is followed immediately by the whole forced sequence
(newline, then the closer). The MTP loops decide that cut on a copy of the criteria
BEFORE the drafter and the target commit anything, commit through the token that
precedes each forced token, and emit the forced tokens one by one. The E4 tests
place the trip at every block position (an accepted draft, the verifier bonus, the
prefill's first token) with a high-acceptance drafter (full accepts across the trip)
and a low-acceptance one (mismatch every round), on the inline cached-session path
and on the continuous-batching B == 1 path, and compare token ids, the session
cache and the next turn with plain decoding.
"""

from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest
from test_mtp_exactness import (
    BLOCK,
    PICKS,
    PROMPT,
    deployed_sampling,
    fake_model,
    hybrid_lm,
    mtp_drafter,
    scripted,
)
from test_mtp_stop_state import Tokenizer, assert_same_cache, cache_snapshot, turn

import mlx_vlm.speculative.mtp as mtp_utils
from mlx_vlm.generate import ar
from mlx_vlm.generate.common import PromptCacheState
from mlx_vlm.speculative import utils as speculative_utils
from mlx_vlm.speculative.utils import _mtp_rounds, _mtp_rounds_batch
from mlx_vlm.utils import ThinkingBudgetCriteria

END_ID = 42
THINK, CLOSE, NEWLINE = 61, 62, 63
SPECIAL = {THINK: "<think>", CLOSE: "</think>", NEWLINE: "\n"}
N = 12


class ThinkingTokenizer(Tokenizer):
    """Ids as text, with the three thinking delimiters as single tokens."""

    def decode(self, ids):
        return " ".join(SPECIAL.get(int(i), str(int(i))) for i in ids)

    def encode(self, text, add_special_tokens=False):
        lookup = {v: k for k, v in SPECIAL.items()}
        return [lookup[text]]


THINKING_PROMPT = PROMPT + [THINK]


# --------------------------------------------------------------------------- E4
@pytest.fixture(scope="module")
def setup():
    lm = hybrid_lm(logit_scale=4.0)
    model = fake_model(lm)
    # The drafter proposes the UNBUDGETED continuation, so blocks are fully
    # accepted across the trip (the M3 worst case).
    free, _ = turn(model, THINKING_PROMPT, PromptCacheState(), n=N + 4)
    return SimpleNamespace(lm=lm, model=model, free=list(free))


def _budget_turn(setup, budget, state, drafter=None, ids=None, n=N, seed=5):
    return turn(
        setup.model,
        ids or THINKING_PROMPT,
        state,
        drafter=drafter,
        n=n,
        seed=seed,
        tokenizer=ThinkingTokenizer(),
        thinking_budget=budget,
        enable_thinking=True,
    )


def _trip(off):
    """Index of the token after which plain decoding forced newline + closer."""
    pairs = [i for i in range(len(off) - 1) if off[i : i + 2] == [NEWLINE, CLOSE]]
    assert pairs, off  # the budget did trip inside the generated window
    return pairs[0] - 1


def _scripted_free(setup, high):
    drafter = mtp_drafter(setup.lm)
    free = setup.free
    script = scripted(
        drafter,
        lambda pos: free[pos] if pos < len(free) else None,
        high=high,
        vocab=64,
    )
    return drafter, script


@pytest.mark.parametrize("high", [True, False], ids=["full-accept", "mismatch"])
def test_e4_budget_trip_at_every_block_position_emits_plain_sequence(setup, high):
    failures = []
    rounds = []
    trips = set()
    for budget in range(0, 8):  # the trip lands on every block position
        off_state, on_state = PromptCacheState(), PromptCacheState()
        off, _ = _budget_turn(setup, budget, off_state)
        trips.add(_trip(off) % BLOCK)
        drafter, script = _scripted_free(setup, high)
        on, _ = _budget_turn(setup, budget, on_state, drafter=drafter)
        rounds.extend(script.rounds)
        if on != off:
            failures.append((budget, "tokens", off, on))
            continue
        try:
            assert on_state.token_ids == off_state.token_ids
            assert_same_cache(cache_snapshot(on_state), cache_snapshot(off_state))
        except AssertionError as e:
            failures.append((budget, "state", str(e).splitlines()[0]))
            continue
        follow = off_state.token_ids + [21, 22, 23]
        off2, _ = turn(setup.model, follow, off_state, n=6, seed=9)
        on2, _ = turn(setup.model, follow, on_state, drafter=drafter, n=6, seed=9)
        if on2 != off2:
            failures.append((budget, "next-turn", off2, on2))
    if high:
        assert any(a == d for a, d in rounds), rounds
    else:  # the scripted drafter never proposes the free path; it rarely hits
        assert sum(a == 0 for a, _ in rounds) >= 0.8 * len(rounds), rounds
    assert trips == set(range(BLOCK)), trips
    assert failures == []


def _batched(setup, budget, drafter):
    """One request through BatchGenerator (continuous batching, B == 1)."""
    params = deployed_sampling(PICKS[0])
    sampler = ar._PositionedTargetSampler(seed=5, **params)
    tokenizer = ThinkingTokenizer()
    generator = ar.BatchGenerator(
        setup.lm,
        SimpleNamespace(tokenizer=tokenizer),
        sampler=sampler,
        prefill_step_size=None,
        draft_model=drafter,
        draft_kind="mtp" if drafter is not None else None,
        draft_block_size=BLOCK,
    )
    criteria = ThinkingBudgetCriteria(
        tokenizer,
        thinking_budget=budget,
        thinking_end_token="</think>",
        thinking_start_token="<think>",
        enable_thinking=True,
        prompt_preopens_thinking=True,
    )
    generator.insert(
        [THINKING_PROMPT],
        max_tokens=N,
        prompt_kwargs=[
            {"inputs_embeds": setup.lm.model.embed_tokens(mx.array([THINKING_PROMPT]))}
        ],
        thinking_budget_criteria=[criteria],
        seeds=[5],
    )
    tokens = []
    try:
        for _ in range(200):
            _, responses = generator.next()
            tokens += [int(r.token) for r in responses if r.token is not None]
            if not generator.has_work:
                break
    finally:
        generator.close()
    return tokens


@pytest.mark.parametrize("high", [True, False], ids=["full-accept", "mismatch"])
def test_e4_continuous_batching_b1_budget_matches_plain(setup, high):
    failures = []
    for budget in range(0, 8):
        off = _batched(setup, budget, None)
        _trip(off)
        drafter, _ = _scripted_free(setup, high)
        on = _batched(setup, budget, drafter)
        if on != off:
            failures.append((budget, off, on))
    assert failures == []


# --------------------------------------------------------------------------- units
class _Draft:
    def __init__(self):
        self.config = SimpleNamespace(block_size=3)
        self.accept_lens = []
        self.draft_lens = []
        self.draft_block_calls = 0

    def set_shared_kv(self, *args, **kwargs):
        pass

    def reset(self, model):
        pass

    def draft_block(self, *args, **kwargs):
        self.draft_block_calls += 1
        return mx.array([[7, 8]], dtype=mx.int32)


class _LM:
    def rollback_speculative_cache(self, *args):
        pass


class _ForcedCriteria:
    """Plain decoding's contract: ``pop_forced_token_id`` hands out the token the
    caller's last ``__call__`` scheduled. Armed: the first bonus already tripped."""

    thinking_end_token_id = END_ID

    def __init__(self, armed):
        self.forced_token_id = END_ID if armed else None

    def __call__(self, token_id):
        return None

    def pop_forced_token_id(self):
        forced, self.forced_token_id = self.forced_token_id, None
        return forced


def test_run_speculative_rounds_forwards_criteria_to_the_mtp_branch():
    """The dispatch gap itself: B==1 mtp must receive the criteria."""
    seen = {}

    def fake_mtp_rounds(*args, **kwargs):
        seen["criteria"] = kwargs.get("thinking_budget_criteria")
        return iter(())

    criteria = _ForcedCriteria(armed=False)
    last_outputs = SimpleNamespace(
        shared_kv_states={}, hidden_states=[mx.zeros((1, 1, 2), dtype=mx.float32)]
    )
    with patch.object(speculative_utils, "_mtp_rounds", side_effect=fake_mtp_rounds):
        drafter = SimpleNamespace(config=SimpleNamespace(block_size=3))
        list(
            speculative_utils.run_speculative_rounds(
                SimpleNamespace(language_model=_LM()),
                drafter,
                [SimpleNamespace(offset=0)],
                mx.array([[1, 2]], dtype=mx.int32),
                mx.array([3], dtype=mx.int32),
                mx.zeros((1,)),
                last_outputs,
                draft_kind="mtp",
                max_tokens=4,
                sampler=lambda logits: mx.argmax(logits, axis=-1),
                sampler_is_greedy=True,
                thinking_budget_criteria=criteria,
            )
        )
    assert seen.get("criteria") is criteria


def test_batch_criteria_with_b_gt_1_refuses():
    with pytest.raises(ValueError, match="batch"):
        list(
            _mtp_rounds_batch(
                SimpleNamespace(language_model=_LM()),
                _Draft(),
                [SimpleNamespace(offset=0)],
                mx.zeros((2, 1, 2), dtype=mx.float32),
                {},
                first_bonus=mx.array([1, 1], dtype=mx.int32),
                max_tokens=4,
                sampler=lambda logits: mx.argmax(logits, axis=-1),
                draft_block_size=3,
                greedy_sampling=True,
                thinking_budget_criteria=[
                    _ForcedCriteria(armed=False),
                    _ForcedCriteria(armed=False),
                ],
            )
        )


def test_server_rounds_forwards_criteria_to_the_batch_loop():
    seen = {}

    def fake_batch(*args, **kwargs):
        seen["criteria"] = kwargs.get("thinking_budget_criteria")
        return iter(())

    crit = [_ForcedCriteria(armed=False)]
    with patch.object(speculative_utils, "_mtp_rounds_batch", side_effect=fake_batch):
        list(
            speculative_utils.run_speculative_server_rounds(
                SimpleNamespace(language_model=_LM()),
                SimpleNamespace(config=SimpleNamespace(block_size=3)),
                [SimpleNamespace(offset=0)],
                mx.zeros((1, 1, 2), dtype=mx.float32),
                draft_kind="mtp",
                first_bonus=mx.array([1], dtype=mx.int32),
                max_tokens=4,
                sampler=lambda logits: mx.argmax(logits, axis=-1),
                thinking_budget_criteria=crit,
            )
        )
    assert seen.get("criteria") is crit


def test_speculative_generation_batch_drives_and_forwards_criteria():
    """The wrapper must (a) drive the criteria's __call__ on every emitted token —
    the round loop pops the scheduled forced token after each yield, as plain
    decoding does — and (b) hand the criteria list to run_speculative_server_rounds."""
    calls = []

    class _RecordingCriteria:
        thinking_end_token_id = END_ID

        def __call__(self, token_id):
            calls.append(token_id)
            return None

        def pop_forced_token_id(self):
            return None

    seen = {}

    def fake_server_rounds(*args, **kwargs):
        seen["criteria"] = kwargs.get("thinking_budget_criteria")
        yield [7], {"round_pos": 0, "round_len": 1}

    crit = _RecordingCriteria()
    batch = ar.SpeculativeGenerationBatch(
        model=SimpleNamespace(language_model=_LM()),
        draft_model=SimpleNamespace(config=SimpleNamespace(block_size=3)),
        draft_kind="mtp",
        uids=[11],
        first_tokens=mx.array([5], dtype=mx.int32),
        prompt_cache=[SimpleNamespace(offset=0)],
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        stop_criteria=lambda token: False,
        max_tokens=[8],
        hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
        shared_kv_states={},
        prompt_tokens=mx.array([[1, 2]], dtype=mx.int32),
        thinking_budget_criteria=[crit],
    )
    with patch.object(
        ar, "run_speculative_server_rounds", side_effect=fake_server_rounds
    ):
        first = batch.next()  # first-bonus send
        second = batch.next()  # one round
    assert [r.token for r in first] == [5]
    assert [r.token for r in second] == [7]
    assert calls == [5, 7], "every emitted token must drive the criteria"
    assert seen.get("criteria") == [crit]


@pytest.mark.parametrize("batched", [False, True])
def test_forced_token_after_first_bonus_commits_bonus_before_yield(batched):
    """A forced token pending after the first bonus: the bonus is committed with a
    width-1 forward BEFORE the forced token is yielded, so a consumer that stops
    there leaves no pending state, and no drafting happened."""
    from mlx_vlm.speculative.cache_state import start_speculative_cache

    class PositionCache:
        def __init__(self):
            self.offset = 0

        def trim(self, count):
            self.offset -= count
            return count

    entry = PositionCache()
    caches = [entry]
    transactions = []
    widths = []

    def fake_verify(lm, inputs, prompt_cache, sampler, **kwargs):
        width = int(inputs.shape[1])
        widths.append(width)
        assert inputs.tolist() == [[1]]
        assert kwargs["sample_target_tokens"] is False
        transaction = start_speculative_cache(prompt_cache, width)
        entry.offset += width
        transactions.append(transaction)
        return mtp_utils._MTPVerifyResult(
            hidden=mx.zeros((1, width, 2)),
            shared_kv_states={},
            rollback_state=transaction,
        )

    draft = _Draft()
    criteria = _ForcedCriteria(armed=True)
    kwargs = dict(
        max_tokens=4,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        draft_block_size=3,
        greedy_sampling=True,
    )
    loop = _mtp_rounds_batch if batched else _mtp_rounds
    with patch.object(mtp_utils, "_mtp_verify_target", side_effect=fake_verify):
        stream = loop(
            SimpleNamespace(language_model=_LM()),
            draft,
            caches,
            mx.zeros((1, 1, 2)),
            {},
            first_bonus=mx.array([1]) if batched else 1,
            thinking_budget_criteria=[criteria] if batched else criteria,
            **kwargs,
        )
        try:
            token, _ = next(stream)
            assert token == ([END_ID] if batched else END_ID)
            assert widths == [1]
            assert entry.offset == 1
            assert len(transactions) == 1 and not transactions[0].active
            assert draft.draft_block_calls == 0
        finally:
            stream.close()
    assert entry.offset == 1, "closing the consumer must preserve committed bonus"


def test_no_pending_forced_token_changes_nothing():
    def run(criteria):
        with (
            patch.object(
                mtp_utils,
                "_mtp_verify_target",
                return_value=speculative_utils._MTPVerifyResult(
                    hidden=mx.zeros((1, 3, 2)), shared_kv_states={}, rollback_state=None
                ),
            ),
            patch.object(mtp_utils, "_mtp_acceptance_walk", return_value=(1, [9, 10])),
        ):
            return [
                t
                for t, _ in _mtp_rounds(
                    SimpleNamespace(language_model=_LM()),
                    _Draft(),
                    [SimpleNamespace(offset=0)],
                    mx.zeros((1, 1, 2), dtype=mx.float32),
                    {},
                    first_bonus=1,
                    max_tokens=4,
                    sampler=lambda logits: mx.argmax(logits, axis=-1),
                    draft_block_size=3,
                    token_dtype=mx.int32,
                    greedy_sampling=True,
                    thinking_budget_criteria=criteria,
                )
            ]

    assert run(_ForcedCriteria(armed=False)) == run(None)
