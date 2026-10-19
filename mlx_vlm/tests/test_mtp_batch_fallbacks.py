"""P369 E5 / E5b (M2) and E6 (M5): MTP falls back to plain decoding where it
cannot reproduce plain decoding's law. (Round 1: batched MTP now ALWAYS decodes plainly,
see test_mtp_batch_plain.py; the processor / seed cases below stay as regression pins
for the any-kind reasons, and their controls assert the batched-MTP reason.)

Fork-only file. Continuous batching built ``SpeculativeGenerationBatch`` without
the requests' logits processors (repetition / presence / frequency penalties,
``logit_bias``, grammars), so they applied to the first token only (M2); and the
speculative path keyed every row's draws by the batch sampler's seed instead of
each request's own ``seed`` (M5). The fix is fail-closed, no new state carry: a
prompt batch with any processor, or whose per-request seeds the speculative path
would not honour, is decoded by the plain ``GenerationBatch`` — and so is any
batch admitted while a plain batch is active. The cached (inline) path already
falls back to plain decoding for processors (E5b, pinned here).
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from test_mtp_exactness import (
    BLOCK,
    PICKS,
    PROMPT,
    deployed_sampling,
    hybrid_lm,
    mtp_drafter,
)
from test_mtp_stop_state import turn

from mlx_vlm.generate import ar
from mlx_vlm.generate.common import PromptCacheState
from mlx_vlm.sample_utils import make_logits_processors
from mlx_vlm.speculative.common import (
    speculative_stats_since,
    speculative_stats_snapshot,
)
from mlx_vlm.utils import StoppingCriteria

N = 10
OTHER = [6, 1, 8, 2, 12, 9]


@pytest.fixture(scope="module")
def lm():
    return hybrid_lm(logit_scale=4.0)


def _run(lm, drafter, requests, *, sampler_seed=0):
    """Drive one BatchGenerator; ``requests`` = [(prompt, kwargs, admit_at_step)]."""
    params = deployed_sampling(PICKS[1])
    sampler = ar._PositionedTargetSampler(seed=sampler_seed, **params)
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(stopping_criteria=StoppingCriteria([], None))
    )
    generator = ar.BatchGenerator(
        lm,
        processor,
        sampler=sampler,
        prefill_step_size=None,
        draft_model=drafter,
        draft_kind="mtp" if drafter is not None else None,
        draft_block_size=BLOCK,
    )
    snapshot = speculative_stats_snapshot(drafter) if drafter is not None else None
    out, order = {}, {}
    pending = sorted(enumerate(requests), key=lambda item: item[1][2])
    try:
        for step in range(400):
            while pending and pending[0][1][2] <= step:
                index, (prompt, kwargs, _) = pending.pop(0)
                (uid,) = generator.insert(
                    [prompt],
                    max_tokens=N,
                    prompt_kwargs=[
                        {"inputs_embeds": lm.model.embed_tokens(mx.array([prompt]))}
                    ],
                    **kwargs,
                )
                order[uid] = index
                out[index] = []
            _, responses = generator.next()
            for r in responses:
                if r.token is not None:
                    out[order[r.uid]].append(int(r.token))
            if not pending and not generator.has_work:
                break
    finally:
        generator.close()
    rounds = None
    if drafter is not None:
        rounds = speculative_stats_since(drafter, snapshot)[0] or 0
    return [out[i] for i in range(len(requests))], rounds


def _processors():
    # A strong bias plus a repetition penalty: both change tokens after the first.
    return make_logits_processors({11: 6.0, 7: -50.0}, 1.5, 20)


# --------------------------------------------------------------------------- E5 (M2)
def test_e5_continuous_batching_with_logits_processors_decodes_plain(lm):
    request = [(PROMPT, {"logits_processors": [_processors()]}, 0)]
    plain, _ = _run(lm, None, request)
    on, rounds = _run(lm, mtp_drafter(lm), request)
    assert on == plain
    assert rounds == 0, "no speculative round may run with a logits processor"


def test_e5_processor_row_admitted_with_a_plain_row(lm):
    requests = [
        (PROMPT, {"logits_processors": [_processors()]}, 0),
        (OTHER, {"logits_processors": [None]}, 0),
    ]
    plain, _ = _run(lm, None, requests)
    on, rounds = _run(lm, mtp_drafter(lm), requests)
    assert on == plain and rounds == 0


def test_e5_control_without_processors_decodes_plain_too(lm):
    # P369 round 1: batched MTP never speculates (F1-F3), so the old "control still
    # speculates" arm is gone; the reason recorded is the batched-MTP one.
    ar.SPECULATION_FALLBACKS.clear()
    request = [(PROMPT, {}, 0)]
    plain, _ = _run(lm, None, request)
    on, rounds = _run(lm, mtp_drafter(lm), request)
    assert on == plain and rounds == 0
    assert ar.SPECULATION_FALLBACKS["mtp_batched"] == 1


# --------------------------------------------------------------------------- E5b
@pytest.mark.parametrize(
    "penalty",
    [
        {"repetition_penalty": 1.3},
        {"presence_penalty": 0.8},
        {"frequency_penalty": 0.6},
        {"logit_bias": {11: 5.0}},
    ],
    ids=["repetition", "presence", "frequency", "logit_bias"],
)
def test_e5b_cached_request_with_processors_runs_zero_draft_rounds(lm, penalty):
    from test_mtp_exactness import fake_model

    model = fake_model(lm)
    plain, _ = turn(model, PROMPT, PromptCacheState(), n=N, **penalty)
    drafter = mtp_drafter(lm)
    snapshot = speculative_stats_snapshot(drafter)
    on, _ = turn(model, PROMPT, PromptCacheState(), drafter=drafter, n=N, **penalty)
    assert on == plain
    assert speculative_stats_since(drafter, snapshot)[0] is None


# --------------------------------------------------------------------------- E6 (M5)
def test_e6_a_request_seed_other_than_the_sampler_seed_decodes_plain(lm):
    request = [(PROMPT, {"seeds": [123]}, 0)]
    plain, _ = _run(lm, None, request)
    on, rounds = _run(lm, mtp_drafter(lm), request)
    assert on == plain and rounds == 0


@pytest.mark.parametrize("order", [(123, 456), (456, 123)], ids=["a-b", "b-a"])
def test_e6_distinct_seeds_and_admission_order(lm, order):
    requests = [
        (PROMPT, {"seeds": [order[0]]}, 0),
        (OTHER, {"seeds": [order[1]]}, 0),
    ]
    plain, _ = _run(lm, None, requests)
    on, rounds = _run(lm, mtp_drafter(lm), requests)
    assert on == plain and rounds == 0


def test_e6_late_second_row_with_its_own_seed(lm):
    requests = [(PROMPT, {"seeds": [123]}, 0), (OTHER, {"seeds": [456]}, 3)]
    plain, _ = _run(lm, None, requests)
    on, rounds = _run(lm, mtp_drafter(lm), requests)
    assert on == plain and rounds == 0


def test_e6_late_speculation_eligible_row_joins_the_active_plain_batch(lm):
    """The first row falls back (own seed); a later default-seed row must not
    build a speculative batch on top of the active plain one."""
    requests = [(PROMPT, {"seeds": [123]}, 0), (OTHER, {"seeds": [0]}, 3)]
    plain, _ = _run(lm, None, requests)
    on, rounds = _run(lm, mtp_drafter(lm), requests)
    assert on == plain and rounds == 0


def test_e6_control_default_seed_decodes_plain_for_the_batched_mtp_reason(lm):
    """Unseeded server traffic carries DEFAULT_SEED == the sampler's seed; batched
    MTP no longer speculates at all (P369 round 1), whatever the seed."""
    ar.SPECULATION_FALLBACKS.clear()
    request = [(PROMPT, {"seeds": [0]}, 0)]
    plain, _ = _run(lm, None, request)
    on, rounds = _run(lm, mtp_drafter(lm), request)
    assert on == plain and rounds == 0
    assert ar.SPECULATION_FALLBACKS["mtp_batched"] == 1


# ------------------------------------------------------------ round 4 / F15
def test_f15_inline_penalty_fallback_presents_plain_to_prefill_policies(lm, monkeypatch):
    """A cached MTP request that falls back for a penalty must not request the
    drafter's prefill captures nor present MTP to the chunked-prefill / offload
    policies (an explicit unchunked MoE-offload prefill would bypass its default)."""
    from test_mtp_exactness import fake_model

    seen = {"capture": [], "offload": [], "chunked": []}
    real_capture = ar.speculative_prefill_kwargs
    real_offload = ar._default_prefill_step_size_for_offload
    real_chunked = ar._chunked_prefill_enabled

    monkeypatch.setattr(
        ar,
        "speculative_prefill_kwargs",
        lambda k, d: (seen["capture"].append(k), real_capture(k, d))[1],
    )
    monkeypatch.setattr(
        ar,
        "_default_prefill_step_size_for_offload",
        lambda m, step, draft, default: (
            seen["offload"].append(draft),
            real_offload(m, step, draft, default),
        )[1],
    )
    monkeypatch.setattr(
        ar,
        "_chunked_prefill_enabled",
        lambda m, **kw: (
            seen["chunked"].append(kw.get("draft_model")),
            real_chunked(m, **kw),
        )[1],
    )
    model = fake_model(lm)
    plain, _ = turn(
        model, PROMPT, PromptCacheState(), n=N, prefill_step_size=4, repetition_penalty=1.3
    )
    seen.update(capture=[], offload=[], chunked=[])
    drafter = mtp_drafter(lm)
    on, _ = turn(
        model,
        PROMPT,
        PromptCacheState(),
        drafter=drafter,
        n=N,
        prefill_step_size=4,
        repetition_penalty=1.3,
    )
    assert on == plain
    assert seen["capture"] == []
    assert seen["offload"] and all(d is None for d in seen["offload"])
    assert all(d is None for d in seen["chunked"])
