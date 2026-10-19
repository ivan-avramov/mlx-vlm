"""P369 round-4 F11: a batch that falls back to plain decoding discards its captures.

Fork-only file. Chunked DFlash / Eagle3 prefill retains per-chunk hidden-state
captures; when the batch then falls back (a logits processor, a per-request seed, a
plain batch already decoding) the final forward asks for none, and combining the
chunks with that capture-less output crashed. The fallback now discards the chunks.
The target is a tiny hybrid qwen3_5 with a stub drafter exposing only the layer ids
the capture kwargs need; the run stops once the prefill produced its decode batch.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from test_mtp_exactness import BLOCK, PICKS, PROMPT, deployed_sampling, hybrid_lm
from test_mtp_batch_fallbacks import _processors

from mlx_vlm.generate import ar
from mlx_vlm.utils import StoppingCriteria

LONG = [(i * 7 + 3) % 60 + 1 for i in range(40)]


@pytest.fixture(scope="module")
def lm():
    return hybrid_lm(logit_scale=4.0)


def _drafter(kind):
    config = SimpleNamespace(target_layer_ids=[0, 1], block_size=BLOCK)
    if kind == "eagle3":
        config = SimpleNamespace(target_layer_ids=[0, 1], block_size=BLOCK)
    return SimpleNamespace(config=config)


def _prefill_to_decode_batch(lm, kind, insert_kwargs, monkeypatch):
    monkeypatch.setattr(ar, "validate_drafter_compatibility", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(
        ar, "speculative_prefill_kwargs", lambda k, d: {"capture_layer_ids": [0, 1]}
    )
    from mlx_vlm.speculative import utils as su

    monkeypatch.setattr(
        su, "speculative_prefill_kwargs", lambda k, d: {"capture_layer_ids": [0, 1]}
    )
    sampler = ar._PositionedTargetSampler(seed=0, **deployed_sampling(PICKS[1]))
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(stopping_criteria=StoppingCriteria([], None))
    )
    generator = ar.BatchGenerator(
        lm,
        processor,
        sampler=sampler,
        prefill_step_size=8,
        draft_model=_drafter(kind),
        draft_kind=kind,
        draft_block_size=BLOCK,
    )
    generator.insert(
        [LONG, LONG[:30]],
        max_tokens=[2, 2],
        prompt_kwargs=[
            {"inputs_embeds": lm.model.embed_tokens(mx.array([p]))}
            for p in (LONG, LONG[:30])
        ],
        **insert_kwargs,
    )
    batch = None
    try:
        for _ in range(30):
            generator.next()
            batch = generator._generation_batch
            if len(batch):
                break
    finally:
        generator.close()
    return batch


REASONS = {
    "logits_processors": lambda: {"logits_processors": [_processors(), _processors()]},
    "request_seed": lambda: {"seeds": [123, 456]},
}


@pytest.mark.parametrize("kind", ["dflash", "eagle3"])
@pytest.mark.parametrize("reason", sorted(REASONS))
def test_f11_chunked_capture_then_fallback_does_not_crash(lm, kind, reason, monkeypatch):
    batch = _prefill_to_decode_batch(lm, kind, REASONS[reason](), monkeypatch)
    assert batch is not None and len(batch) > 0
    assert not getattr(batch, "is_speculative", False)
