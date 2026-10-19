"""P369 round-4 F10: ``max_tokens=0`` emits nothing for every draft kind.

Fork-only file. Plain decoding emitted no token but left a warm session's cache
mutated under its old label; MTP emitted (and committed) one token. Now a zero budget
yields nothing on every path and a warm session is RETIRED (the exact choice: the
prompt-only state would have to be re-derived per cache layout), so the next turn
re-prefills identically for MTP-ON and OFF.
"""

import mlx.core as mx
import pytest
from test_mtp_exactness import PICKS, PROMPT, deployed_sampling, fake_model, hybrid_lm
from test_mtp_exactness import generate as ar_generate
from test_mtp_exactness import mtp_drafter
from test_mtp_stop_state import BLOCK, N, Tokenizer, assert_same_cache, cache_snapshot, turn

from mlx_vlm.generate.common import PromptCacheState
from mlx_vlm.generate.dispatch import stream_generate


@pytest.fixture(scope="module")
def lm():
    return hybrid_lm(logit_scale=4.0)


@pytest.mark.parametrize("arm", ["off", "on"])
def test_f10_generate_step_zero_budget_yields_nothing(lm, arm):
    drafter = mtp_drafter(lm) if arm == "on" else None
    assert list(ar_generate(lm, drafter, n=0)) == []


@pytest.mark.parametrize("arm", ["off", "on"])
def test_f10_warm_session_zero_budget_retires_and_next_turn_matches_fresh(lm, arm):
    model = fake_model(lm)
    drafter = mtp_drafter(lm) if arm == "on" else None
    state = PromptCacheState()
    first, _ = turn(model, PROMPT, state, drafter=drafter, n=N, seed=5)
    follow = PROMPT + list(first) + [21, 22]
    kwargs = dict(
        input_ids=mx.array([follow]),
        prompt_cache_state=state,
        max_tokens=0,
        prefill_step_size=None,
        seed=7,
        **deployed_sampling(PICKS[0]),
    )
    if drafter is not None:
        kwargs.update(draft_model=drafter, draft_kind="mtp", draft_block_size=BLOCK)
    chunks = list(stream_generate(model, Tokenizer(()), "", **kwargs))
    assert [c.token for c in chunks if c.token is not None] == []
    assert chunks[-1].finish_reason == "length"
    assert state.token_ids is None and state.cache is None  # retired, not stale
    # Next turn from the retired session == a fresh plain session.
    nxt = follow + [31, 32]
    fresh_state = PromptCacheState()
    want, _ = turn(model, nxt, fresh_state, n=6, seed=9)
    got, _ = turn(model, nxt, state, drafter=drafter, n=6, seed=9)
    assert got == want
    assert_same_cache(cache_snapshot(state), cache_snapshot(fresh_state))
