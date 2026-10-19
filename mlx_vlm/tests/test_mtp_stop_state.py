"""P369 E3 (M4): a stop inside an MTP block leaves exactly plain decoding's state.

Fork-only file. Plain decoding writes every emitted token, the stop token included,
into the target cache before yielding it; the session cache is then labelled with
``prompt + emitted``. MTP must leave the same state on the deployed cached-session
path: no accepted draft beyond the stop may reach the KV / GatedDeltaNet state, and
a stop (or the length budget) landing on the verifier bonus must still write that
token. Checked at EVERY block position (stop on each accepted draft, on the bonus,
on the prefill's first token, and the ``max_tokens`` cut), for a high-acceptance
drafter (the worst case) and a low-acceptance one, through ``stream_generate`` with a
``PromptCacheState`` (what ``_process_cached_request`` runs) and through the server's
own ``_process_cached_request`` with a session id: token ids, the session label, KV
offsets and valid prefixes, recurrent snapshots, the drafter's paired-position count
and the NEXT turn's output must all equal plain decoding.
"""

from queue import Queue
from types import SimpleNamespace

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

from mlx_vlm.generate.common import PromptCacheState
from mlx_vlm.generate.dispatch import stream_generate
from mlx_vlm.utils import StoppingCriteria

N = 9


class Detokenizer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.text = ""
        self.last_segment = ""

    def add_token(self, token, skip_special_token_ids=()):
        self.last_segment = f"<{int(token)}>"
        self.text += self.last_segment

    def finalize(self):
        self.last_segment = ""


class Tokenizer:
    """Just enough tokenizer for stream_generate and prepare_inputs (ids as text)."""

    pad_token = "<pad>"
    eos_token = "<eos>"

    def __init__(self, stop_ids=()):
        self.stopping_criteria = StoppingCriteria(
            [], None, additional_eos_token_ids=list(stop_ids)
        )
        self.detokenizer = Detokenizer()

    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)

    def __call__(self, prompts, **kwargs):
        text = prompts[0] if isinstance(prompts, list) else prompts
        ids = [int(t) for t in text.split()]
        return SimpleNamespace(
            input_ids=mx.array([ids]), attention_mask=mx.ones((1, len(ids)))
        )


def turn(
    model,
    ids,
    state,
    *,
    stop_ids=(),
    drafter=None,
    n=N,
    seed=5,
    params=None,
    tokenizer=None,
    **extra,
):
    kwargs = dict(
        input_ids=mx.array([ids], dtype=mx.int32),
        prompt_cache_state=state,
        max_tokens=n,
        prefill_step_size=None,
        seed=seed,
        **(params or deployed_sampling(PICKS[0])),
    )
    kwargs.update(extra)  # an extra may override a default (e.g. prefill_step_size)
    if drafter is not None:
        kwargs.update(draft_model=drafter, draft_kind="mtp", draft_block_size=BLOCK)
    tokenizer = tokenizer or Tokenizer(stop_ids)
    final = list(stream_generate(model, tokenizer, "", **kwargs))[-1]
    return final.token_ids, final.finish_reason


def cache_snapshot(state):
    """Offsets, valid KV prefixes and recurrent arrays of a session cache."""
    snap = []
    for c in state.cache:
        if type(c).__name__ == "KVCache":
            offset = int(c.offset)
            snap.append(
                ("kv", offset, c.keys[..., :offset, :], c.values[..., :offset, :])
            )
        else:
            snap.append(("recurrent", None, *[mx.array(a) for a in c.state]))
    mx.eval([a for entry in snap for a in entry[2:]])
    return snap


def assert_same_cache(got, want):
    assert [(e[0], e[1]) for e in got] == [(e[0], e[1]) for e in want]
    for g, w in zip(got, want):
        for a, b in zip(g[2:], w[2:]):
            assert a.shape == b.shape
            assert mx.allclose(a, b, atol=1e-5, rtol=1e-5).item(), g[0]


@pytest.fixture(scope="module")
def setup():
    lm = hybrid_lm(logit_scale=4.0)
    model = fake_model(lm)
    reference, _ = turn(model, PROMPT, PromptCacheState(), n=N)
    assert len(reference) == N  # token_ids carries the generated tokens only
    return SimpleNamespace(lm=lm, model=model, ref=list(reference))


def _cases(ref):
    """(label, stop_ids, max_tokens): a stop at every emitted position whose token is
    new, plus the length cut at every position."""
    cases = []
    for p, token in enumerate(ref):
        if token not in ref[:p]:
            cases.append((f"stop@{p}", [token], N))
    for p in range(1, N):
        cases.append((f"length@{p}", [], p))
    return cases


def _run_pair(setup, stop_ids, n, high):
    model = setup.model
    off_state, on_state = PromptCacheState(), PromptCacheState()
    off, off_fin = turn(model, PROMPT, off_state, stop_ids=stop_ids, n=n)
    drafter = mtp_drafter(setup.lm)
    ref = setup.ref
    script = scripted(
        drafter, lambda pos: ref[pos] if pos < len(ref) else None, high=high, vocab=64
    )
    on, on_fin = turn(model, PROMPT, on_state, stop_ids=stop_ids, drafter=drafter, n=n)
    return SimpleNamespace(
        off=off,
        on=on,
        off_fin=off_fin,
        on_fin=on_fin,
        off_state=off_state,
        on_state=on_state,
        drafter=drafter,
        script=script,
    )


@pytest.mark.parametrize("high", [True, False], ids=["high-accept", "low-accept"])
def test_e3_stop_at_every_block_position_matches_plain_state(setup, high):
    failures = []
    for label, stop_ids, n in _cases(setup.ref):
        r = _run_pair(setup, stop_ids, n, high)
        assert r.on == r.off and r.on_fin == r.off_fin, label  # emission is exact
        try:
            assert r.on_state.token_ids == r.off_state.token_ids
            assert_same_cache(cache_snapshot(r.on_state), cache_snapshot(r.off_state))
            # The drafter has paired every emitted token but the last (whose
            # successor was never verified): its committed history is the
            # emitted prefix, nothing beyond the stop. (A stop on the prefill's
            # first token never starts a round.)
            if r.script.rounds:
                assert int(r.drafter._next_position) == len(PROMPT) + len(r.on) - 1
        except AssertionError as e:
            failures.append(f"{label}: {e}".splitlines()[0])
    if high:
        assert any(a == d for a, d in r.script.rounds)
    assert failures == []


@pytest.mark.parametrize("high", [True, False], ids=["high-accept", "low-accept"])
def test_e3_next_turn_after_a_stop_matches_plain(setup, high):
    failures = []
    for label, stop_ids, n in _cases(setup.ref):
        r = _run_pair(setup, stop_ids, n, high)
        follow = r.off_state.token_ids + [21, 22, 23]
        off2, _ = turn(setup.model, follow, r.off_state, n=6, seed=9)
        on2, _ = turn(setup.model, follow, r.on_state, drafter=r.drafter, n=6, seed=9)
        if on2 != off2:
            failures.append((label, off2, on2))
    assert failures == []


# ----------------------------------------------------------------- server cached path
def _bare_response_generator(model, drafter):
    from threading import Event, Lock

    from mlx_vlm.server import generation as server

    rg = server.ResponseGenerator.__new__(server.ResponseGenerator)
    rg.model = model
    rg.processor = None  # set per request (carries the stop ids)
    rg.vision_cache = None
    rg.stop_tokens = set()
    rg.kv_bits = None
    rg.kv_group_size = None
    rg.kv_quant_scheme = None
    rg.quantized_kv_start = None
    rg.top_logprobs_k = 0
    rg.apc_manager = None
    rg.draft_model = drafter
    rg.draft_kind = "mtp" if drafter is not None else None
    rg._cancelled = set()
    rg._cancel_lock = Lock()
    rg._stop = False
    rg._load_error = None
    rg._ready = Event()
    rg.requests = Queue()
    return rg


def _serve_turn(rg, ids, chat_id, stop_ids, n, seed):
    from mlx_vlm.server import generation as server
    from mlx_vlm.server import session_manager

    rg.processor = Tokenizer(stop_ids)
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
    rg._process_cached_request(
        rqueue=rqueue,
        prompt=" ".join(map(str, ids)),
        images=None,
        args=args,
        prompt_tokens=len(ids),
        prompt_cache_state=state,
    )
    ctx = rqueue.get_nowait()
    tokens = []
    while True:
        item = rqueue.get_nowait()
        if item is None:
            break
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, server.KeepAlive):
            continue
        if item.finish_reason is None or item.finish_reason == "stop":
            tokens.append(int(item.token))
        if item.finish_reason is not None:
            break
    return ctx, tokens, state


def test_e3_two_turns_through_the_server_cached_session_path(setup, monkeypatch):
    from mlx_vlm.server import generation as server
    from mlx_vlm.server import session_manager

    monkeypatch.setattr(session_manager, "_session_cache_max", 8)
    monkeypatch.setattr(server, "_get_draft_block_size_from_env", lambda: BLOCK)
    session_manager.clear_session_caches()
    ref = setup.ref
    failures = []
    stops = [p for p, t in enumerate(ref) if 0 < p < 6 and t not in ref[:p]]
    for p in stops:
        outputs = {}
        for arm in ("off", "on"):
            drafter = None
            if arm == "on":
                drafter = mtp_drafter(setup.lm)
                scripted(
                    drafter,
                    lambda pos: ref[pos] if pos < len(ref) else None,
                    high=True,
                    vocab=64,
                )
            rg = _bare_response_generator(setup.model, drafter)
            chat_id = f"p369-{arm}-{p}"
            ctx, first, state = _serve_turn(rg, PROMPT, chat_id, [ref[p]], N, 5)
            assert ctx.backend == server.BACKEND_CACHED_SESSION
            follow = list(state.token_ids) + [21, 22, 23]
            _, second, state2 = _serve_turn(rg, follow, chat_id, [], 6, 9)
            assert state2 is state  # the forwarded session id resolved the same state
            outputs[arm] = (first, second, cache_snapshot(state), list(state.token_ids))
        if outputs["on"][:2] != outputs["off"][:2]:
            failures.append((p, outputs["off"][:2], outputs["on"][:2]))
        # P369 round 2: the recorded session state must match too, not only tokens.
        if outputs["on"][3] != outputs["off"][3]:
            failures.append((p, "session token_ids", outputs["off"][3]))
        try:
            assert_same_cache(outputs["on"][2], outputs["off"][2])
        except AssertionError as e:
            failures.append((p, "cache", str(e).splitlines()[0]))
    session_manager.clear_session_caches()
    assert stops and failures == []
