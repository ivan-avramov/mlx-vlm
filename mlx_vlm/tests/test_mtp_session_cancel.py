"""P369 round-2 F5: an abnormal exit of a cached request retires its session.

Fork-only file. MTP commits a verified block into the session cache BEFORE yielding
its tokens, and the session label (``token_ids``) is only published when a request
runs to completion. A warm request that is cancelled / disconnected / fails therefore
used to leave the OLD ``token_ids`` over an in-place cache already holding the prompt
and (for MTP) un-emitted tokens, and plain decoding left a different ahead-state: the
next turn then differed between MTP-ON and MTP-OFF (worst on hybrid recurrent state).
Every abnormal exit now clears the ``PromptCacheState``, for every draft kind, so the
next turn re-prefills from scratch and ON == OFF == a fresh session.

Cancellation is injected after every number of delivered tokens (first token,
mid-block, verifier bonus, ... up to the one before the terminal chunk), through the
server's own ``_process_cached_request`` cooperative-cancel path, plus a consumer
closing the ``stream_generate`` generator directly, plus a raised exception.
"""

from types import SimpleNamespace

import mlx.core as mx

import pytest
from test_mtp_exactness import PICKS, PROMPT, deployed_sampling, mtp_drafter, scripted
from test_mtp_stop_state import (
    BLOCK,
    N,
    Tokenizer,
    _bare_response_generator,
    _serve_turn,
    assert_same_cache,
    cache_snapshot,
    fake_model,
    hybrid_lm,
    turn,
)

from mlx_vlm.generate.common import PromptCacheState
from mlx_vlm.generate.dispatch import stream_generate

# Warm follow-up: the session holds PROMPT + the plain turn-1 emission, and TURN2
# extends that whole history (what a real next chat turn does), so it is a cache HIT.
_LM = hybrid_lm(logit_scale=4.0)
_TURN1, _ = turn(fake_model(_LM), PROMPT, PromptCacheState(), n=N, seed=5)
TURN2 = PROMPT + list(_TURN1) + [21, 22, 23, 24]
TURN3 = TURN2 + [31, 32, 33]


@pytest.fixture(scope="module")
def setup():
    lm = _LM
    model = fake_model(lm)
    ref2, _ = turn(model, TURN2, PromptCacheState(), n=N, seed=7)
    fresh_state = PromptCacheState()
    fresh3, _ = turn(model, TURN3, fresh_state, n=6, seed=9)
    return SimpleNamespace(
        lm=lm,
        model=model,
        ref2=list(ref2),
        fresh3=list(fresh3),
        fresh_snapshot=cache_snapshot(fresh_state),
    )


def _drafter(setup, arm):
    if arm == "off":
        return None
    drafter = mtp_drafter(setup.lm)
    ref = setup.ref2
    scripted(
        drafter, lambda pos: ref[pos] if pos < len(ref) else None, high=True, vocab=64
    )
    return drafter


def _serve_cancelled(rg, ids, chat_id, n, seed, cancel_after, monkeypatch):
    """Run one cached request, cancelling it once ``cancel_after`` chunks went out."""
    from queue import Queue

    import importlib

    generate_pkg = importlib.import_module("mlx_vlm.generate")
    from mlx_vlm.server import generation as server
    from mlx_vlm.server import session_manager

    rg.processor = Tokenizer(())
    state = session_manager.get_or_create_prompt_cache_state(chat_id)
    params = deployed_sampling(PICKS[0])
    args = server.GenerationArguments(
        max_tokens=n,
        temperature=params["temperature"],
        top_p=params["top_p"],
        top_k=params["top_k"],
        min_p=params["min_p"],
        seed=seed,
    )
    rqueue = Queue()
    real = generate_pkg.stream_generate

    def cancelling(*a, **kw):
        for count, chunk in enumerate(real(*a, **kw), start=1):
            yield chunk
            if count == cancel_after:
                with rg._cancel_lock:
                    rg._cancelled.add(id(rqueue))

    monkeypatch.setattr(generate_pkg, "stream_generate", cancelling)
    rg._process_cached_request(
        rqueue=rqueue,
        prompt=" ".join(map(str, ids)),
        images=None,
        args=args,
        prompt_tokens=len(ids),
        prompt_cache_state=state,
    )
    monkeypatch.setattr(generate_pkg, "stream_generate", real)
    return state


@pytest.mark.parametrize("arm", ["off", "on"])
def test_f5_cancel_a_warm_session_at_every_position_matches_a_fresh_session(
    setup, arm, monkeypatch
):
    from mlx_vlm.server import generation as server
    from mlx_vlm.server import session_manager

    monkeypatch.setattr(session_manager, "_session_cache_max", 8)
    monkeypatch.setattr(server, "_get_draft_block_size_from_env", lambda: BLOCK)
    session_manager.clear_session_caches()
    failures = []
    for k in range(1, N + 1):
        drafter = _drafter(setup, arm)
        rg = _bare_response_generator(setup.model, drafter)
        chat_id = f"p369-cancel-{arm}-{k}"
        _serve_turn(rg, PROMPT, chat_id, [], N, 5)  # warm the session
        state = _cancelled_state = _serve_cancelled(
            rg, TURN2, chat_id, N, 7, k, monkeypatch
        )
        if k < N:
            if state.token_ids is not None or state.cache is not None:
                failures.append((k, "session not retired", state.token_ids))
        elif state.token_ids is None:  # round 3 F8: cancelled AFTER the terminal chunk
            failures.append((k, "completed session lost"))
        _, third, state3 = _serve_turn(rg, TURN3, chat_id, [], 6, 9)
        try:
            assert third == setup.fresh3
            assert_same_cache(cache_snapshot(state3), setup.fresh_snapshot)
        except AssertionError as e:
            failures.append((k, str(e).splitlines()[0]))
    session_manager.clear_session_caches()
    assert failures == []


@pytest.mark.parametrize("arm", ["off", "on"])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_f5_closing_stream_generate_midway_clears_the_session(setup, arm, k):
    drafter = _drafter(setup, arm)
    state = PromptCacheState()
    turn(setup.model, PROMPT, state, drafter=drafter, n=N, seed=5)
    assert state.token_ids is not None  # warm
    assert list(state.token_ids) == PROMPT + list(_TURN1)
    kwargs = dict(
        input_ids=mx.array([TURN2]),
        prompt_cache_state=state,
        max_tokens=N,
        prefill_step_size=None,
        seed=7,
        **deployed_sampling(PICKS[0]),
    )
    if drafter is not None:
        kwargs.update(draft_model=drafter, draft_kind="mtp", draft_block_size=BLOCK)
    stream = stream_generate(setup.model, Tokenizer(()), "", **kwargs)
    for _ in range(k):
        next(stream)
    stream.close()
    assert state.token_ids is None and state.cache is None


def test_f5_a_completed_stream_still_publishes_its_session(setup):
    state = PromptCacheState()
    turn(setup.model, PROMPT, state, n=N, seed=5)
    assert state.token_ids is not None and state.cache is not None


@pytest.mark.parametrize("arm", ["off", "on"])
def test_f8_closing_right_after_the_terminal_chunk_keeps_the_session(setup, arm):
    """Round 3 F8: a consumer that closes on ``finish_reason`` must not lose a valid
    session; the published state and the next turn equal a fully drained stream's."""

    def run(close_after_terminal):
        drafter = _drafter(setup, arm)
        state = PromptCacheState()
        turn(setup.model, PROMPT, state, drafter=drafter, n=N, seed=5)
        kwargs = dict(
            input_ids=mx.array([TURN2]),
            prompt_cache_state=state,
            max_tokens=N,
            prefill_step_size=None,
            seed=7,
            **deployed_sampling(PICKS[0]),
        )
        if drafter is not None:
            kwargs.update(
                draft_model=drafter, draft_kind="mtp", draft_block_size=BLOCK
            )
        stream = stream_generate(setup.model, Tokenizer(()), "", **kwargs)
        if close_after_terminal:
            for chunk in stream:
                if chunk.finish_reason is not None:
                    break
            stream.close()
        else:
            list(stream)
        follow = list(state.token_ids or []) + [41, 42]
        second, _ = turn(
            setup.model, follow, state, drafter=drafter, n=6, seed=9
        )
        return state, second

    drained_state, drained_next = run(False)
    closed_state, closed_next = run(True)
    assert drained_state.token_ids is not None
    assert closed_next == drained_next
    assert list(closed_state.token_ids) == list(drained_state.token_ids)
    assert_same_cache(cache_snapshot(closed_state), cache_snapshot(drained_state))


@pytest.mark.parametrize("arm", ["off", "on"])
def test_f8_closed_after_terminal_session_is_published_before_next_turn(setup, arm):
    drafter = _drafter(setup, arm)
    state = PromptCacheState()
    turn(setup.model, PROMPT, state, drafter=drafter, n=N, seed=5)
    kwargs = dict(
        input_ids=mx.array([TURN2]),
        prompt_cache_state=state,
        max_tokens=N,
        prefill_step_size=None,
        seed=7,
        **deployed_sampling(PICKS[0]),
    )
    stream = stream_generate(setup.model, Tokenizer(()), "", **kwargs)
    for chunk in stream:
        if chunk.finish_reason is not None:
            break
    stream.close()
    assert state.token_ids is not None and state.cache is not None
    assert list(state.token_ids)[: len(TURN2)] == TURN2


# ------------------------------------------------------------------ round 5 / F18
def _start(setup, state, drafter, ids=None, n=N, seed=7):
    kwargs = dict(
        input_ids=mx.array([ids or TURN2]),
        prompt_cache_state=state,
        max_tokens=n,
        prefill_step_size=None,
        seed=seed,
        **deployed_sampling(PICKS[0]),
    )
    if drafter is not None:
        kwargs.update(draft_model=drafter, draft_kind="mtp", draft_block_size=BLOCK)
    return stream_generate(setup.model, Tokenizer(()), "", **kwargs)


@pytest.mark.parametrize("arm", ["off", "on"])
def test_f18_abandoned_unclosed_stream_cannot_corrupt_or_clear_a_newer_one(
    setup, arm
):
    """Round 5 F18: stream A is abandoned WITHOUT closing; stream B starts on the same
    session (it must retire the stale in-flight state and decode from scratch), and
    closing / resuming A afterwards must neither clear nor publish over B."""
    drafter = _drafter(setup, arm)
    state = PromptCacheState()
    turn(setup.model, PROMPT, state, drafter=drafter, n=N, seed=5)  # warm
    stream_a = _start(setup, state, drafter)
    next(stream_a)
    next(stream_a)  # A in flight; its prefill extended the shared cache
    chunks_b = list(_start(setup, state, drafter))  # B on the same session
    tokens_b = [c.token for c in chunks_b if c.token is not None]
    assert tokens_b == setup.ref2[: len(tokens_b)] and len(tokens_b) == N
    published = (list(state.token_ids), cache_snapshot(state))
    # A comes back to life: it must be a no-op for the session.
    with pytest.raises(StopIteration):
        for _ in range(N + 2):
            next(stream_a)
    stream_a.close()
    assert list(state.token_ids) == published[0]
    assert_same_cache(cache_snapshot(state), published[1])


@pytest.mark.parametrize("arm", ["off", "on"])
def test_f18_closing_the_stale_stream_alone_keeps_the_newer_publication(setup, arm):
    drafter = _drafter(setup, arm)
    state = PromptCacheState()
    turn(setup.model, PROMPT, state, drafter=drafter, n=N, seed=5)
    stream_a = _start(setup, state, drafter)
    next(stream_a)
    list(_start(setup, state, drafter))
    ids = list(state.token_ids)
    stream_a.close()
    assert state.token_ids is not None and list(state.token_ids) == ids
