"""ModernBERT attention masks must stay finite in the compute dtype.

Fork-only test. Upstream builds the additive mask from ``-1e9``, which casts to
``-inf`` in float16. A padding query position whose entire sliding window is
padding then has a row of ``-inf`` scores, and softmax turns it into NaN. The NaN
sits in a padding position, but mean pooling multiplies NaN by the zero mask and
NaN * 0 is NaN, so every short text batched with a long one comes back as NaN.
Reproduced with gte-modernbert-base on sixteen retrieval chunks (six NaN rows);
see docs/upstream-bugs.md item 6.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.models.modernbert import Model, ModelConfig

LOCAL_ATTENTION = 8  # window half-width 4, so 5+ trailing pads fully mask a row


def _config() -> ModelConfig:
    return ModelConfig(
        model_type="modernbert",
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=3,  # layer 0 global, layers 1-2 local
        intermediate_size=48,
        num_attention_heads=2,
        local_attention=LOCAL_ATTENTION,
    )


def _model(dtype) -> Model:
    mx.random.seed(7)
    model = Model(_config())
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    return model


def _batch():
    """One long row and one short row padded far past the sliding window."""
    long_ids = mx.arange(1, 25)[None, :]  # 24 real tokens
    short_len = 5
    short_ids = mx.concatenate(
        [mx.arange(1, short_len + 1), mx.zeros((24 - short_len,), dtype=mx.int32)]
    )[None, :]
    ids = mx.concatenate([long_ids, short_ids], axis=0)
    mask = mx.concatenate(
        [
            mx.ones((1, 24), dtype=mx.int32),
            mx.concatenate(
                [
                    mx.ones((short_len,), dtype=mx.int32),
                    mx.zeros((24 - short_len,), dtype=mx.int32),
                ]
            )[None, :],
        ],
        axis=0,
    )
    return ids, mask, short_ids[:, :short_len]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_masks_are_finite_in_half_precision(dtype):
    model = _model(dtype)
    _, mask, _ = _batch()
    global_mask, sliding_mask = model._masks(mask, dtype)
    assert global_mask.dtype == dtype and sliding_mask.dtype == dtype
    assert bool(mx.isfinite(global_mask).all())
    assert bool(mx.isfinite(sliding_mask).all())


def test_short_text_batched_with_long_text_is_finite_in_float16():
    model = _model(mx.float16)
    ids, mask, _ = _batch()
    out = model(ids, attention_mask=mask)
    mx.eval(out.text_embeds)
    assert bool(mx.isfinite(out.text_embeds).all()), "NaN embedding for the padded row"


def test_batched_float16_embedding_matches_float32_and_single_row():
    ids, mask, short_ids = _batch()
    half = _model(mx.float16)
    full = _model(mx.float32)

    batched = np.asarray(half(ids, attention_mask=mask).text_embeds.astype(mx.float32))
    alone = np.asarray(
        half(
            short_ids, attention_mask=mx.ones(short_ids.shape, dtype=mx.int32)
        ).text_embeds.astype(mx.float32)
    )
    reference = np.asarray(full(ids, attention_mask=mask).text_embeds)

    assert np.all(np.isfinite(batched))
    # Padding must not change the short row: batched == alone (half precision slack).
    np.testing.assert_allclose(batched[1], alone[0], atol=2e-2)
    # And the float16 path must agree with float32 compute on both rows.
    np.testing.assert_allclose(batched, reference, atol=2e-2)
