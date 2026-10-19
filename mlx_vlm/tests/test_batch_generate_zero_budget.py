"""P369 round-6 F20: ``batch_generate`` with zero-budget rows detokenizes only tokens.

Fork-only file. A zero-budget row finishes with ``token=None``; the collector used to
append that sentinel and hand it to ``detokenizer.add_token``. Real tiny hybrid
qwen3_5 target; ``prepare_inputs`` / chat templating are patched to token ids.
"""

from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest
from test_mtp_exactness import (
    PICKS,
    deployed_sampling,
    fake_model,
    hybrid_lm,
    mtp_drafter,
)

from mlx_vlm.generate import ar
from mlx_vlm.utils import StoppingCriteria

ROWS = [[3, 5, 7, 9, 11, 13], [6, 1, 8, 2, 12, 9]]


class _Detok:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tokens, self.text = [], ""

    def add_token(self, token, **kw):
        assert token is not None, "the zero-budget sentinel reached the detokenizer"
        self.tokens.append(int(token))
        self.text += f"<{int(token)}>"

    def finalize(self):
        pass


@pytest.fixture(scope="module")
def lm():
    return hybrid_lm(logit_scale=4.0)


def _run(lm, drafter, budgets):
    model = fake_model(lm)
    model.config.model_type = "qwen3_5"
    embeds = lm.model.embed_tokens(mx.array(ROWS))
    model.get_input_embeddings = lambda ids, pv, mask=None, **kw: SimpleNamespace(
        inputs_embeds=embeds, to_dict=lambda: {"inputs_embeds": embeds}
    )
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(stopping_criteria=StoppingCriteria([], None)),
        detokenizer=_Detok(),
    )
    extra = {}
    if drafter is not None:
        extra = dict(draft_model=drafter, draft_kind="mtp", draft_block_size=3)
    params = deployed_sampling(PICKS[1])
    with (
        patch.object(ar, "apply_chat_template", side_effect=lambda p, c, t, **k: t),
        patch.object(ar, "should_add_special_tokens", return_value=False),
        patch.object(
            ar,
            "prepare_inputs",
            return_value={
                "input_ids": mx.array(ROWS),
                "attention_mask": mx.ones((2, 6), dtype=mx.int32),
            },
        ),
    ):
        texts, _ = ar._generate_batch(
            model,
            processor,
            prompts=["a", "b"],
            max_tokens=budgets,
            prefill_step_size=None,
            sampler=ar._PositionedTargetSampler(seed=0, **params),
            **extra,
        )
    return texts


@pytest.mark.parametrize("arm", ["off", "on"])
@pytest.mark.parametrize("budgets", [[0, 0], [0, 4], [4, 0]])
def test_f20_zero_budget_rows_detokenize_cleanly(lm, arm, budgets):
    drafter = mtp_drafter(lm) if arm == "on" else None
    texts = _run(lm, drafter, budgets)
    for text, budget in zip(texts, budgets):
        assert (text == "") == (budget == 0)
        assert text.count("<") == budget
