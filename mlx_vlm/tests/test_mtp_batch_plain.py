"""P369 round-1 review F1-F4: batched MTP decodes exactly as plain batching.

Fork-only file. Cursor round 1 found that the batched speculative loop
(``SpeculativeGenerationBatch`` -> ``_mtp_rounds_batch``) truncates what it emits at a
stop / length boundary but still commits the whole accepted block: a row stopping
mid-block keeps un-emitted KV / recurrent / drafter state (F1), a request stopping on
its first token never writes it to the cache (F2), and a shorter row's own length
budget is invisible to a loop that is given ``max(max_tokens)`` (F3). Per the plan's
M2/M5 rule (no new state-carry machinery) MTP now decodes continuous batches with the
plain ``GenerationBatch`` and refuses the lockstep B > 1 ``generate_step``; the
fallback is counted in ``ar.SPECULATION_FALLBACKS`` so a performance cliff is visible.

These tests drive the real batch generator with a real tiny hybrid qwen3_5 target and
MTP head (drafter proposals unscripted: any acceptance pattern must be harmless) and
require, at every stop position (first token, every later token) and for unequal
per-row ``max_tokens``: tokens identical to MTP-OFF, zero speculative rounds, and no
``SpeculativeGenerationBatch`` ever built, so the decoding state is by construction
the plain batch's.
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

from mlx_vlm.generate import ar
from mlx_vlm.speculative.common import (
    speculative_stats_since,
    speculative_stats_snapshot,
)
from mlx_vlm.utils import StoppingCriteria

N = 9
OTHER = [6, 1, 8, 2, 12, 9]


@pytest.fixture(scope="module")
def lm():
    return hybrid_lm(logit_scale=4.0)


def _drive(lm, drafter, prompts, budgets, stop_ids=()):
    """One BatchGenerator, all prompts inserted together (one prompt batch)."""
    sampler = ar._PositionedTargetSampler(seed=0, **deployed_sampling(PICKS[1]))
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(
            stopping_criteria=StoppingCriteria(
                [], None, additional_eos_token_ids=list(stop_ids)
            )
        )
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
    uids = generator.insert(
        prompts,
        max_tokens=list(budgets),
        prompt_kwargs=[
            {"inputs_embeds": lm.model.embed_tokens(mx.array([p]))} for p in prompts
        ],
    )
    out = {uid: [] for uid in uids}
    reasons = {}
    try:
        for _ in range(400):
            _, responses = generator.next()
            for r in responses:
                if r.token is not None:
                    out[r.uid].append(int(r.token))
                if r.finish_reason is not None:
                    reasons[r.uid] = r.finish_reason
            if not generator.has_work:
                break
    finally:
        generator.close()
    rounds = None
    if drafter is not None:
        rounds = speculative_stats_since(drafter, snapshot)[0] or 0
    return [out[u] for u in uids], [reasons.get(u) for u in uids], rounds


@pytest.fixture
def no_speculative_batch(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("MTP built a SpeculativeGenerationBatch")

    monkeypatch.setattr(ar, "SpeculativeGenerationBatch", refuse)


def _cases(refs):
    """(label, stop_ids, budgets): a stop at every position of row 0 (including the
    first token) and of row 1, plus unequal per-row budgets."""
    cases = []
    for row in (0, 1):
        for p, token in enumerate(refs[row]):
            if token not in refs[row][:p]:
                cases.append((f"row{row}-stop@{p}", [token], (N, N)))
    for budgets in [(1, N), (N, 1), (2, N), (N, 4), (3, 5), (5, 3), (4, 4)]:
        cases.append((f"budgets{budgets}", [], budgets))
    return cases


def test_f1_f2_f3_batched_mtp_equals_plain_at_every_stop_and_budget(
    lm, no_speculative_batch
):
    prompts = [PROMPT, OTHER]
    refs, _, _ = _drive(lm, None, prompts, (N, N))
    assert all(len(r) == N for r in refs), refs
    failures = []
    for label, stop_ids, budgets in _cases(refs):
        plain = _drive(lm, None, prompts, budgets, stop_ids)
        on = _drive(lm, mtp_drafter(lm), prompts, budgets, stop_ids)
        if on[:2] != plain[:2] or on[2] != 0:
            failures.append((label, plain[:2], on))
    assert failures == []


def test_f2_first_token_stop_single_row_matches_plain(lm, no_speculative_batch):
    (ref,), _, _ = _drive(lm, None, [PROMPT], (N,))
    plain = _drive(lm, None, [PROMPT], (N,), [ref[0]])
    on = _drive(lm, mtp_drafter(lm), [PROMPT], (N,), [ref[0]])
    assert on[:2] == plain[:2] and on[2] == 0
    assert len(on[0][0]) == 1


def test_batched_mtp_fallback_is_counted_with_its_reason(lm, caplog):
    before = ar.SPECULATION_FALLBACKS["mtp_batched"]
    with caplog.at_level("WARNING"):
        _, _, rounds = _drive(lm, mtp_drafter(lm), [PROMPT, OTHER], (N, N))
    assert rounds == 0
    assert ar.SPECULATION_FALLBACKS["mtp_batched"] > before
    assert any("mtp_batched" in rec.getMessage() for rec in caplog.records)


def test_f1_lockstep_b_gt_1_generate_step_refuses_mtp(lm):
    from test_mtp_exactness import fake_model

    drafter = mtp_drafter(lm)
    before = ar.SPECULATION_FALLBACKS["mtp_lockstep_refused"]
    with pytest.raises(NotImplementedError, match="lockstep"):
        list(
            ar.generate_step(
                mx.array([PROMPT, PROMPT], dtype=mx.int32),
                fake_model(lm),
                None,
                None,
                max_tokens=N,
                prompt_cache=lm.make_cache(),
                prefill_step_size=None,
                draft_model=drafter,
                draft_kind="mtp",
                draft_block_size=BLOCK,
            )
        )
    assert ar.SPECULATION_FALLBACKS["mtp_lockstep_refused"] == before + 1
