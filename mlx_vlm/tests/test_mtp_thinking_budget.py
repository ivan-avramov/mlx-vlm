"""thinking_budget support in the MTP round loop (campaign O40, 2026-08-24).

Fork-only file. The gap these pin: `run_speculative_rounds` forwards
`thinking_budget_criteria` ONLY to the suffix branch; the mtp branch drops it, which is
half of why the server hard-rejects thinking_budget + a loaded drafter (HTTP 500 at
`server/generation.py`) — and our deployed profile always sets a budget, so native MTP
was undeployable on this stack.

Contract (ported verbatim from `run_suffix_decoding_rounds`, the established design):
the CALLER drives the criteria's `__call__` per yielded token, keeping
`budget_exceeded` current; the round loop checks it at each round top and, when
tripped, commits the pending bonus into the target cache with a single-token forward,
then force-yields the criteria's `thinking_end_token_id` so generation leaves the
thinking block and answers. Block granularity: <= one draft block of overshoot is
accepted, exactly as suffix documents. Never fires without a criteria.

The fakes mirror test_speculative.py's `_mtp_rounds` fixtures (patched
`_mtp_verify_target` / `_mtp_acceptance_walk`; no real models). The criteria stub
exposes only the two attributes the suffix loop reads (`budget_exceeded`,
`thinking_end_token_id`) — the same duck-type contract.
"""

from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

import mlx_vlm.speculative.mtp as mtp_utils
from mlx_vlm.speculative import utils as speculative_utils
from mlx_vlm.speculative.utils import _mtp_rounds

END_ID = 42


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


class _Criteria:
    """Duck-type stub: trips for the first `trip_reads` round-top checks, then clears
    (the real ThinkingBudgetCriteria clears `budget_exceeded` when its __call__ sees
    the end token — driven by the caller, which this unit test stands in for)."""

    def __init__(self, trip_reads=1):
        self._trips_left = trip_reads
        self.thinking_end_token_id = END_ID

    @property
    def budget_exceeded(self):
        if self._trips_left > 0:
            self._trips_left -= 1
            return True
        return False


def _verify(width):
    return speculative_utils._MTPVerifyResult(
        hidden=mx.zeros((1, width, 2), dtype=mx.float32),
        shared_kv_states={},
        gdn_states=None,
    )


def _run(criteria, max_tokens=4):
    verify_widths = []

    def fake_verify(lm, verify_input, *args, **kwargs):
        verify_widths.append(int(verify_input.shape[1]))
        return _verify(int(verify_input.shape[1]))

    draft = _Draft()
    model = SimpleNamespace(language_model=_LM())
    with (
        patch.object(mtp_utils, "_mtp_verify_target", side_effect=fake_verify),
        patch.object(mtp_utils, "_mtp_acceptance_walk", return_value=(1, [9, 10])),
    ):
        toks = [
            t
            for t, _ in _mtp_rounds(
                model,
                draft,
                [SimpleNamespace(offset=0)],
                mx.zeros((1, 1, 2), dtype=mx.float32),
                {},
                first_bonus=1,
                max_tokens=max_tokens,
                sampler=lambda logits: mx.argmax(logits, axis=-1),
                draft_block_size=3,
                token_dtype=mx.int32,
                greedy_sampling=True,
                thinking_budget_criteria=criteria,
            )
        ]
    return toks, verify_widths, draft


def test_tripped_budget_forces_the_end_token_first():
    toks, verify_widths, draft = _run(_Criteria(trip_reads=1))
    assert toks[0] == END_ID, (
        "with the budget already exceeded at the round top, the end-of-thinking token "
        "must be forced before any further drafting")


def test_tripped_budget_commits_the_pending_bonus_with_a_single_token_forward():
    _toks, verify_widths, _draft = _run(_Criteria(trip_reads=1))
    assert verify_widths[0] == 1, (
        "the pending bonus must be committed into the target cache (width-1 forward) "
        "before the forced end token, or the cache desyncs from the emitted stream")


def test_drafting_resumes_after_the_forced_close():
    toks, _verify_widths, draft = _run(_Criteria(trip_reads=1))
    assert draft.draft_block_calls > 0, "after leaving thinking, normal rounds resume"
    assert len(toks) > 1


def test_untripped_criteria_changes_nothing():
    toks_with, _, _ = _run(_Criteria(trip_reads=0))
    with (
        patch.object(mtp_utils, "_mtp_verify_target", return_value=_verify(3)),
        patch.object(mtp_utils, "_mtp_acceptance_walk", return_value=(1, [9, 10])),
    ):
        toks_without = [
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
            )
        ]
    assert toks_with == toks_without


def test_run_speculative_rounds_forwards_criteria_to_the_mtp_branch():
    """The dispatch gap itself: B==1 mtp must receive the criteria."""
    seen = {}

    def fake_mtp_rounds(*args, **kwargs):
        seen["criteria"] = kwargs.get("thinking_budget_criteria")
        return iter(())

    criteria = _Criteria(trip_reads=0)
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


# --------------------------------------------------------------------------- batched loop
# The continuous-batching path (BatchGenerator -> SpeculativeGenerationBatch ->
# run_speculative_server_rounds -> _mtp_rounds_batch) is where ALL anonymous server
# traffic decodes — including benchmark generation, whose deployed profile always sets a
# thinking budget. v1 scope mirrors suffix's: budget enforcement at B==1 (the campaign
# runs --num-threads 1); criteria with B>1 refuses loudly rather than silently ignoring
# the budget.

from mlx_vlm.speculative.utils import _mtp_rounds_batch  # noqa: E402


def _run_batch(criteria, B=1, max_tokens=4):
    verify_widths = []

    def fake_verify(lm, verify_input, *args, **kwargs):
        verify_widths.append(int(verify_input.shape[1]))
        w = int(verify_input.shape[1])
        return speculative_utils._MTPVerifyResult(
            hidden=mx.zeros((verify_input.shape[0], w, 2), dtype=mx.float32),
            shared_kv_states={},
            target_tokens=mx.zeros((verify_input.shape[0], w), dtype=mx.int32) + 9,
            gdn_states=None,
        )

    def fake_draft_block_active(draft_model, b_active, hidden, bs, sampler, dtype,
                                positions, **kw):
        return mx.array([[7, 8]] * len(b_active), dtype=mx.int32)

    def fake_walk(draft_tokens, target_tokens, budgets):
        n = draft_tokens.shape[0]
        return [1] * n, [[9, 10]] * n

    draft = _Draft()
    model = SimpleNamespace(language_model=_LM())
    with (
        patch.object(mtp_utils, "_mtp_verify_target", side_effect=fake_verify),
        patch.object(mtp_utils, "_mtp_draft_block_active", fake_draft_block_active),
        patch.object(mtp_utils, "_speculative_walk_batch", side_effect=fake_walk),
    ):
        rounds = []
        for tok_list, _meta in _mtp_rounds_batch(
            model,
            draft,
            [SimpleNamespace(offset=0)],
            mx.zeros((B, 1, 2), dtype=mx.float32),
            {},
            first_bonus=mx.array([1] * B, dtype=mx.int32),
            max_tokens=max_tokens,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
            draft_block_size=3,
            token_dtype=mx.int32,
            greedy_sampling=True,
            thinking_budget_criteria=criteria,
        ):
            rounds.append(tok_list)
    return rounds, verify_widths


def test_batch_tripped_budget_forces_end_token_at_b1():
    rounds, verify_widths = _run_batch([_Criteria(trip_reads=1)])
    assert rounds[0] == [END_ID], (
        "with the budget exceeded at the round top, the forced end token must be the "
        "next emission for the row")
    assert verify_widths[0] == 1, "pending bonus committed with a width-1 forward first"


def test_batch_untripped_criteria_changes_nothing():
    with_c, _ = _run_batch([_Criteria(trip_reads=0)])
    without_c, _ = _run_batch(None)
    assert with_c == without_c


def test_batch_criteria_with_b_gt_1_refuses():
    import pytest

    with pytest.raises(ValueError, match="batch"):
        _run_batch([_Criteria(trip_reads=0), _Criteria(trip_reads=0)], B=2)


# --------------------------------------------------------------------------- plumbing
def test_server_rounds_forwards_criteria_to_the_batch_loop():
    seen = {}

    def fake_batch(*args, **kwargs):
        seen["criteria"] = kwargs.get("thinking_budget_criteria")
        return iter(())

    crit = [_Criteria(trip_reads=0)]
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
    that is what keeps `budget_exceeded` current, the round loop only reads it —
    and (b) hand the criteria list to run_speculative_server_rounds."""
    from mlx_vlm.generate import ar

    calls = []

    class _RecordingCriteria:
        thinking_end_token_id = END_ID
        budget_exceeded = False

        def __call__(self, token_id):
            calls.append(token_id)
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
    with patch.object(ar, "run_speculative_server_rounds", side_effect=fake_server_rounds):
        first = batch.next()   # first-bonus send
        second = batch.next()  # one round
    assert [r.token for r in first] == [5]
    assert [r.token for r in second] == [7]
    assert calls == [5, 7], "every emitted token must drive the criteria"
    assert seen.get("criteria") == [crit]
