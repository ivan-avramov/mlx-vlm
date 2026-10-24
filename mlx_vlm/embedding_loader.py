import json  # Fork: mistral3 text-only detection reads the raw config.json
import logging  # Fork: load_embedding_processor fallback log
from pathlib import Path

import mlx.nn as nn

from .encoder_loader import load_encoder_model
from .tokenizer_utils import load_tokenizer  # Fork: load_embedding_processor fallback
from .utils import load_config, load_processor  # Fork: load_embedding_processor

logger = logging.getLogger(__name__)  # Fork: load_embedding_processor fallback log

EMBEDDING_MODEL_REMAPPING = {
    "qwen3": "qwen3_embedding",
    "gemma3_text": "gemma3_embedding",
    "lfm2": "lfm2_embedding",
    "ministral3": "ministral3_embedding",
    "xlm-roberta": "xlm_roberta",
}


def load_embedding_model(model_path: Path, lazy: bool = False, **kwargs) -> nn.Module:
    # Fork: a `mistral3` checkpoint whose text model is `ministral3` (Nemotron 3.5
    # Embed) loads text-only through `ministral3_embedding` from its `text_config`;
    # the vision tower and projector tensors are dropped by that model's `sanitize`.
    # Any other `mistral3` config falls through to the unchanged path.
    model_remapping = EMBEDDING_MODEL_REMAPPING
    dense_config = model_path / "1_Dense" / "config.json"
    config_overrides = None
    if dense_config.exists():
        config_overrides = {
            "embedding_dim": load_config(dense_config.parent)["out_features"]
        }
        model_remapping = {**model_remapping, "lfm2": "lfm2_colbert"}

    # A caller-supplied dict keeps its existing meaning and is never rewritten;
    # `config=None` counts as omitted. Detection reads the raw config.json.
    if kwargs.get("config") is None:
        kwargs.pop("config", None)
        raw_path = model_path / "config.json"
        if raw_path.exists():
            with open(raw_path) as f:
                raw = json.load(f)
            text_config = raw.get("text_config")
            if (
                raw.get("model_type") == "mistral3"
                and isinstance(text_config, dict)
                and text_config.get("model_type") == "ministral3"
                # a causal text model (an instruct checkpoint) is not an embedder;
                # the embedding model runs bidirectionally (change review 2)
                and text_config.get("is_causal") is not True
            ):
                derived = dict(text_config, model_type="ministral3")
                if "quantization" in raw:
                    derived.setdefault("quantization", raw["quantization"])
                kwargs["config"] = derived

    return load_encoder_model(
        model_path,
        model_remapping=model_remapping,
        config_overrides=config_overrides,
        lazy=lazy,
        **kwargs,
    )


# Fork: new. Processor for the embedding route; the embedding path only needs the
# tokenizer, so an image processor that cannot be built (Pixtral without PyTorch)
# must not make the load fail.
def load_embedding_processor(model_dir: Path, **kwargs):
    try:
        return load_processor(model_dir, add_detokenizer=False, **kwargs)
    except Exception as e:
        logger.warning(
            "load_processor failed (%s: %s); using the tokenizer only",
            type(e).__name__,
            e,
        )
        tokenizer = load_tokenizer(Path(model_dir))
        # `TokenizerWrapper` forwards attributes but is not callable, and `_embed`
        # calls the tokenizer; hand it the wrapped Hugging Face tokenizer.
        if callable(tokenizer) or not hasattr(tokenizer, "_tokenizer"):
            return tokenizer
        return tokenizer._tokenizer
