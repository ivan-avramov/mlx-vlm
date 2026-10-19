"""P369 round-5 F16: a zero token budget in continuous batching emits nothing.

Fork-only file. ``insert(max_tokens=0)`` was replaced by the generator's default
(``0 or default``) and a list-form 0 still emitted one token before the budget check.
A zero-budget row is now retired before any prefill / sampling, with no emitted token
and ``finish_reason == "length"`` (plain and MTP-configured batches alike); other rows
in the same batch are unaffected.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from test_mtp_batch_plain import OTHER, N, _drive
from test_mtp_exactness import PICKS, PROMPT, deployed_sampling, hybrid_lm, mtp_drafter

from mlx_vlm.generate import ar
from mlx_vlm.utils import StoppingCriteria


@pytest.fixture(scope="module")
def lm():
    return hybrid_lm(logit_scale=4.0)


@pytest.mark.parametrize("arm", ["off", "on"])
def test_f16_list_form_zero_budget_single_row(lm, arm):
    drafter = mtp_drafter(lm) if arm == "on" else None
    tokens, reasons, _ = _drive(lm, drafter, [PROMPT], (0,))
    assert tokens == [[]] and reasons == ["length"]


@pytest.mark.parametrize("arm", ["off", "on"])
def test_f16_mixed_zero_and_positive_budgets(lm, arm):
    drafter = mtp_drafter(lm) if arm == "on" else None
    plain = _drive(lm, None, [PROMPT, OTHER], (0, N))
    got = _drive(lm, drafter, [PROMPT, OTHER], (0, N))
    assert got[0][0] == [] and len(got[0][1]) == N
    assert got[1] == ["length", "length"]
    assert got[:2] == plain[:2]


@pytest.mark.parametrize("arm", ["off", "on"])
def test_f16_integer_zero_is_not_the_default(lm, arm):
    """``insert(max_tokens=0)`` must mean zero, not the generator's default."""
    drafter = mtp_drafter(lm) if arm == "on" else None
    sampler = ar._PositionedTargetSampler(seed=0, **deployed_sampling(PICKS[1]))
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(stopping_criteria=StoppingCriteria([], None))
    )
    generator = ar.BatchGenerator(
        lm,
        processor,
        sampler=sampler,
        prefill_step_size=None,
        max_tokens=5,
        draft_model=drafter,
        draft_kind="mtp" if drafter is not None else None,
        draft_block_size=3,
    )
    (uid0,) = generator.insert(
        [PROMPT],
        max_tokens=0,
        prompt_kwargs=[{"inputs_embeds": lm.model.embed_tokens(mx.array([PROMPT]))}],
    )
    (uid1,) = generator.insert(
        [OTHER],
        max_tokens=None,  # None -> the generator default (5)
        prompt_kwargs=[{"inputs_embeds": lm.model.embed_tokens(mx.array([OTHER]))}],
    )
    out = {uid0: [], uid1: []}
    try:
        for _ in range(100):
            _, responses = generator.next()
            for r in responses:
                if r.token is not None:
                    out[r.uid].append(int(r.token))
            if not generator.has_work:
                break
    finally:
        generator.close()
    assert out[uid0] == []
    assert len(out[uid1]) == 5


def test_f16_server_budget_helpers_keep_zero_and_default_none():
    from mlx_vlm.server import generation as server

    args = SimpleNamespace(max_tokens=0, thinking_budget=None)
    server._apply_generation_budget(args, 10)
    assert args.max_tokens == 0
    args = SimpleNamespace(max_tokens=None, thinking_budget=None)
    server._apply_generation_budget(args, 10)
    assert args.max_tokens == server.get_server_max_tokens()
