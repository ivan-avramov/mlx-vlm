"""C26: per-request seeds must reach the cached-path sampler.

`generate_step`'s seeded branch was gated on ``top_k == 0`` / ``min_p == 0`` while
every deployed profile sets ``top_k: 20``, so the declared seed silently fell into
unseeded ``make_sampler`` — byte-identical "seeded" draws never existed on the
cached path. The fix widens ``generate/ar.py``'s ``_PositionedTargetSampler`` to
the server twin's filter semantics (top_p / min_p / top_k, `sample_utils` chain
order) and DE-DUPLICATES the two drifted twins, which were the root cause.
"""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx

from mlx_vlm.generate import ar as ar_module
from mlx_vlm.server import generation as server_generation

generate_module = sys.modules["mlx_vlm.generate"]


def _fixed_logit_model(vocab: int = 4096):
    """Model mock whose logits are FIXED per step, so the only randomness left
    is the sampler's draw — the quantity under test."""
    model = MagicMock()
    logits = (mx.arange(vocab, dtype=mx.float32) * 1e-3)[None, None, :]
    model.language_model.return_value = SimpleNamespace(
        logits=logits, cross_attention_states=None, encoder_outputs=None
    )
    embedding_output = MagicMock()
    embedding_output.inputs_embeds = mx.zeros((1, 1, 8))
    embedding_output.to_dict.return_value = {}
    model.get_input_embeddings.return_value = embedding_output
    return model


def _stream(seed: int, n: int = 8) -> list:
    gen = generate_module.generate_step(
        input_ids=mx.array([[1]], dtype=mx.int32),
        model=_fixed_logit_model(),
        pixel_values=None,
        mask=None,
        max_tokens=n,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        seed=seed,
    )
    out = []
    for step in gen:
        out.append(int(step[0]) if isinstance(step, tuple) else int(step))
        if len(out) >= n:
            break
    return out


@patch.object(generate_module.cache, "make_prompt_cache", return_value=[])
@patch.object(generate_module, "make_logits_processors", return_value=[])
@patch.object(generate_module, "make_sampler")
def test_seeded_branch_covers_top_k(mock_make_sampler, _mlp, _mpc):
    """THE C26 repro: a seeded request at the deployed profile (top_k=20) must
    take the positioned/seeded sampler, never unseeded make_sampler."""
    mock_make_sampler.return_value = lambda logprobs: mx.array([0])
    gen = generate_module.generate_step(
        input_ids=mx.array([[1]], dtype=mx.int32),
        model=_fixed_logit_model(vocab=4096),
        pixel_values=None,
        mask=None,
        max_tokens=1,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        seed=7,
    )
    next(gen)
    mock_make_sampler.assert_not_called()


@patch.object(generate_module.cache, "make_prompt_cache", return_value=[])
@patch.object(generate_module, "make_logits_processors", return_value=[])
def test_same_seed_reproduces_at_deployed_profile(_mlp, _mpc):
    """Two generations with the SAME declared seed and top_k=20 must be
    byte-identical; a different seed must diverge."""
    a, b = _stream(seed=42), _stream(seed=42)
    c = _stream(seed=43)
    assert a == b
    assert a != c


def test_twins_are_deduplicated():
    """The server sampler IS the ar sampler — drift between two copies is what
    produced C26; identity locks it out."""
    assert (
        server_generation._PositionedTargetSampler
        is ar_module._PositionedTargetSampler
    )


def test_sample_target_applies_top_k():
    """top_k=1 collapses the keyed draw to argmax regardless of seed."""
    logprobs = mx.log(mx.softmax(mx.array([[0.1, 3.0, 0.2, 0.5]]), axis=-1))
    for seed in (0, 1, 999):
        s = ar_module._PositionedTargetSampler(
            temperature=1.0, top_p=1.0, seed=seed, top_k=1
        )
        tok = s.sample_target(logprobs, row_ids=[0], positions=[0])
        assert int(tok[0]) == 1
