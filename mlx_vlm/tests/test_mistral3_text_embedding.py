"""Text-only embedding load of `mistral3` retrieval checkpoints (fork-only).

Synthetic tiny models in temporary directories; nothing here reads the Hugging Face
cache.
"""

import json
from pathlib import Path
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_vlm import embedding_loader
from mlx_vlm.embedding_loader import load_embedding_model, load_embedding_processor
from mlx_vlm.models.ministral3_embedding import Model, ModelConfig

TEXT_CONFIG = {
    "model_type": "ministral3",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "intermediate_size": 128,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 32,
    "rms_norm_eps": 1e-5,
    "vocab_size": 96,
    "max_position_embeddings": 256,
    "tie_word_embeddings": True,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "llama_4_scaling_beta": 0.1,
        "original_max_position_embeddings": 128,
    },
}
VISION_CONFIG = {
    "model_type": "pixtral",
    "hidden_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "image_size": 16,
    "patch_size": 4,
}
QUANT = {"group_size": 64, "bits": 8, "mode": "affine"}
IDS = mx.array([[3, 5, 7, 11, 13, 2], [4, 6, 8, 10, 0, 0]])
MASK = mx.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])


def _text_model(seed=0):
    mx.random.seed(seed)
    model = Model(ModelConfig.from_dict(dict(TEXT_CONFIG)))
    params = tree_flatten(model.parameters())
    model.load_weights(
        [
            (k, mx.random.normal(v.shape) * 0.1 + (1.0 if "norm" in k else 0.0))
            for k, v in params
        ]
    )
    mx.eval(model.parameters())
    return model


def _text_weights(model):
    return dict(tree_flatten(model.parameters()))  # keys: model.*


def _mistral3_config(**extra):
    return {
        "model_type": "mistral3",
        "text_config": dict(TEXT_CONFIG),
        "vision_config": dict(VISION_CONFIG),
        **extra,
    }


def _write(directory: Path, config: dict, weights: dict):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(directory / "model.safetensors"), weights)
    return directory


def _vision_junk():
    mx.random.seed(123)
    return {
        "vision_tower.transformer.layers.0.attention.q_proj.weight": mx.random.normal(
            (16, 16)
        ),
        "vision_tower.patch_conv.weight": mx.random.normal((16, 4, 4, 3)),
        "multi_modal_projector.linear_1.weight": mx.random.normal((64, 16)),
        "multi_modal_projector.norm.weight": mx.random.normal((16,)),
    }


def _relayout(weights, layout):
    out = {}
    for k, v in weights.items():
        assert k.startswith("model.")
        bare = k[len("model.") :]
        if layout == "language_model":
            out["language_model." + bare] = v
        elif layout == "model.language_model":
            out["model.language_model." + bare] = v
        elif layout == "language_model.model":
            out["language_model.model." + bare] = v
        elif layout == "language_model+lm_head":
            out["language_model." + bare] = v
        else:
            out[k] = v
    if layout == "language_model+lm_head":
        out["language_model.lm_head.weight"] = mx.random.normal((96, 64))
    return out


def _embed(model, ids=IDS, mask=MASK):
    out = model(ids, attention_mask=mask)
    mx.eval(out.text_embeds)
    return np.array(out.text_embeds)


def _param_paths(model):
    return [k for k, _ in tree_flatten(model.parameters())]


LAYOUTS = [
    "language_model",
    "model.language_model",
    "language_model+lm_head",
    "model",
    "language_model.model",
]


# ---------------------------------------------------------------- criterion 1
@pytest.mark.parametrize("layout", LAYOUTS)
def test_mistral3_text_only_load_matches_plain_ministral3(tmp_path, layout):
    text = _text_model()
    weights = _text_weights(text)

    plain_dir = _write(tmp_path / "plain", dict(TEXT_CONFIG), weights)
    plain = load_embedding_model(plain_dir)

    mm_weights = {**_relayout(weights, layout), **_vision_junk()}
    mm_dir = _write(tmp_path / "mm", _mistral3_config(), mm_weights)
    model = load_embedding_model(mm_dir)

    assert isinstance(model, Model)
    assert len(model.layers) == TEXT_CONFIG["num_hidden_layers"]
    paths = _param_paths(model)
    assert paths
    assert not any("vision_tower" in p or "multi_modal_projector" in p for p in paths)
    assert not any("lm_head" in p for p in paths)
    np.testing.assert_allclose(_embed(model), _embed(plain), atol=1e-5)


def test_mistral3_model_prefixed_vision_keys_are_dropped(tmp_path):
    weights = _text_weights(_text_model())
    junk = {"model." + k: v for k, v in _vision_junk().items()}
    mm = {**_relayout(weights, "model.language_model"), **junk}
    model = load_embedding_model(_write(tmp_path / "mm", _mistral3_config(), mm))
    assert not any("vision" in p or "projector" in p for p in _param_paths(model))


# ---------------------------------------------------------------- criterion 2
def test_quantized_round_trip(tmp_path):
    bf16 = _text_model()
    reference = _embed(bf16)

    quantized = _text_model()
    nn.quantize(quantized, group_size=64, bits=8, mode="affine")
    mx.eval(quantized.parameters())
    expected = _embed(quantized)

    qdir = _write(
        tmp_path / "q",
        _mistral3_config(quantization=dict(QUANT)),
        {**_text_weights(quantized), **_vision_junk()},
    )
    reloaded = load_embedding_model(qdir)
    assert any(isinstance(m, nn.QuantizedLinear) for _, m in reloaded.named_modules())
    got = _embed(reloaded)

    np.testing.assert_allclose(got, expected, atol=1e-5)
    cos = (got * reference).sum(-1) / (
        np.linalg.norm(got, axis=-1) * np.linalg.norm(reference, axis=-1)
    )
    assert (cos >= 0.99).all(), cos


# ---------------------------------------------------------------- criterion 3
def test_mistral3_without_ministral3_text_config_falls_through(tmp_path):
    config = _mistral3_config()
    config["text_config"]["model_type"] = "llama"
    weights = {**_relayout(_text_weights(_text_model()), "language_model")}
    directory = _write(tmp_path / "other", config, weights)
    # The unchanged path builds the full mistral3 model and rejects the tensors;
    # nothing is silently loaded as a text-only embedder.
    with pytest.raises(Exception):
        load_embedding_model(directory)


def test_causal_mistral3_text_config_falls_through(tmp_path):
    """A mistral3 checkpoint whose text model is causal (an instruct model, not an
    embedder) is not loaded text-only; it takes the unchanged path and fails there."""
    config = _mistral3_config()
    config["text_config"]["is_causal"] = True
    weights = {**_relayout(_text_weights(_text_model()), "language_model")}
    directory = _write(tmp_path / "causal", config, weights)
    with pytest.raises(Exception):
        load_embedding_model(directory)


def test_mistral3_without_text_config_raises(tmp_path):
    directory = _write(
        tmp_path / "novision",
        {"model_type": "mistral3", "vision_config": dict(VISION_CONFIG)},
        _vision_junk(),
    )
    with pytest.raises(Exception):
        load_embedding_model(directory)


def test_alias_conflict_raises_naming_both_keys(tmp_path):
    weights = _text_weights(_text_model())
    both = {
        **_relayout(weights, "language_model"),
        "model.norm.weight": weights["model.norm.weight"],
    }
    directory = _write(tmp_path / "alias", _mistral3_config(), both)
    with pytest.raises(ValueError) as err:
        load_embedding_model(directory)
    msg = str(err.value)
    assert "language_model.norm.weight" in msg and "'model.norm.weight'" in msg


def test_sanitize_alias_conflict_directly():
    model = _text_model()
    w = mx.zeros((64,))
    with pytest.raises(ValueError, match="language_model.norm.weight"):
        model.sanitize({"language_model.norm.weight": w, "model.norm.weight": w})
    with pytest.raises(ValueError, match="model.norm.weight"):
        model.sanitize({"model.norm.weight": w, "norm.weight": w})


def test_sanitize_never_keeps_last_silently():
    model = _text_model()
    w = mx.zeros((64,))
    for order in (
        [("model.norm.weight", w), ("language_model.norm.weight", w)],
        [("language_model.norm.weight", w), ("model.norm.weight", w)],
    ):
        with pytest.raises(ValueError):
            model.sanitize(dict(order))


def test_missing_text_tensor_raises(tmp_path):
    weights = _relayout(_text_weights(_text_model()), "language_model")
    del weights["language_model.layers.1.mlp.down_proj.weight"]
    directory = _write(tmp_path / "missing", _mistral3_config(), weights)
    with pytest.raises(ValueError):
        load_embedding_model(directory)


def test_shape_mismatched_text_tensor_raises(tmp_path):
    weights = _relayout(_text_weights(_text_model()), "language_model")
    weights["language_model.layers.0.self_attn.q_proj.weight"] = mx.zeros((3, 5))
    directory = _write(tmp_path / "shape", _mistral3_config(), weights)
    with pytest.raises(ValueError):
        load_embedding_model(directory)


def test_vision_only_weights_raise(tmp_path):
    directory = _write(tmp_path / "vo", _mistral3_config(), _vision_junk())
    with pytest.raises(ValueError):
        load_embedding_model(directory)


# ---------------------------------------------------------------- criterion 4
def test_processor_returned_when_load_processor_succeeds(tmp_path):
    sentinel = object()
    with (
        mock.patch.object(
            embedding_loader, "load_processor", return_value=sentinel
        ) as lp,
        mock.patch.object(embedding_loader, "load_tokenizer") as lt,
    ):
        assert load_embedding_processor(tmp_path) is sentinel
    lp.assert_called_once()
    assert lp.call_args.kwargs["add_detokenizer"] is False
    lt.assert_not_called()


def test_tokenizer_returned_when_load_processor_raises(tmp_path, caplog):
    tok = mock.MagicMock()
    with (
        mock.patch.object(
            embedding_loader, "load_processor", side_effect=ImportError("needs torch")
        ),
        mock.patch.object(embedding_loader, "load_tokenizer", return_value=tok) as lt,
    ):
        with caplog.at_level("INFO", logger=embedding_loader.logger.name):
            assert load_embedding_processor(tmp_path) is tok
    lt.assert_called_once_with(tmp_path)
    records = [r for r in caplog.records if "ImportError" in r.getMessage()]
    assert len(records) == 1


def test_tokenizer_fallback_is_callable_like_a_tokenizer(tmp_path):
    from mlx_vlm.tokenizer_utils import TokenizerWrapper

    inner = mock.MagicMock(return_value={"input_ids": [[1]]})
    wrapper = TokenizerWrapper(inner, detokenizer_class=lambda t: None)
    with (
        mock.patch.object(
            embedding_loader, "load_processor", side_effect=ImportError("x")
        ),
        mock.patch.object(embedding_loader, "load_tokenizer", return_value=wrapper),
    ):
        got = load_embedding_processor(tmp_path)
    assert got is inner
    assert got(["a"]) == {"input_ids": [[1]]}


def test_config_none_takes_the_same_path_as_omitted_config(tmp_path):
    weights = _relayout(_text_weights(_text_model()), "language_model")
    directory = _write(
        tmp_path / "mm", _mistral3_config(), {**weights, **_vision_junk()}
    )
    omitted = load_embedding_model(directory)
    explicit_none = load_embedding_model(directory, config=None)
    assert isinstance(explicit_none, Model)
    np.testing.assert_allclose(_embed(explicit_none), _embed(omitted), atol=1e-5)
