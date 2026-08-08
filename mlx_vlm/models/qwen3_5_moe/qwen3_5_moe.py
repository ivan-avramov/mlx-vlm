import mlx.core as mx
import mlx.nn as nn

from ..qwen3_5 import Model as Qwen3_5Model
from ..qwen3_5.qwen3_5 import (
    NORM_WEIGHT_SUFFIXES,
    sanitize_key,
    should_offset_norm_weight,
    should_shift_norm_weights,
)
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(Qwen3_5Model):

    def __init__(self, config: ModelConfig):
        # only initialize nn.Module, skip the initialization of vision_tower and language_model in the parent class
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def sanitize(self, weights):
        # The MTP draft shard is separate from the base model. Its presence must
        # not select the base model's RMSNorm loading convention, so drop it
        # before deciding whether to shift.
        weights = {key: value for key, value in weights.items() if "mtp." not in key}

        # The norm-weight shift is NOT idempotent (it unconditionally adds 1.0),
        # so an already-converted mlx-native checkpoint must not be shifted
        # again -- that corrupted every norm layer in production. This used to be
        # a blanket `return weights` when any key was "language_model."-prefixed,
        # which also skipped mtp filtering, lm_head popping, expert fusion and
        # key renaming for those checkpoints. Gate per key instead: identical
        # protection, and the rest of sanitize still runs.
        shift_norm_weights = should_shift_norm_weights(weights)

        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for l in range(self.config.text_config.num_hidden_layers):
            prefix = f"model.language_model.layers.{l}.mlp"
            if f"{prefix}.experts.gate_up_proj" in weights:
                # FUSED layout (Qwen3.6-VL): gate_up_proj is a single stacked
                # [num_experts, 2 * intermediate_size, hidden_size] tensor.
                gate_up_weight = weights.pop(f"{prefix}.experts.gate_up_proj")
                gate_weight, up_weights = mx.split(gate_up_weight, 2, axis=-2)
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_weight
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = up_weights
                weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                    f"{prefix}.experts.down_proj"
                )
            elif f"{prefix}.experts.0.gate_proj.weight" in weights:
                # UNFUSED layout (Ornith / Qwen3-Next style): one [out, in] tensor
                # per expert. Stack into [num_experts, out, in] for SwitchGLU.
                num_experts = self.config.text_config.num_experts
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    weights[f"{prefix}.switch_mlp.{proj}.weight"] = mx.stack(
                        [
                            weights.pop(f"{prefix}.experts.{e}.{proj}.weight")
                            for e in range(num_experts)
                        ],
                        axis=0,
                    )
            # else: dense (non-MoE) layer — no expert weights to fuse.

        sanitized_weights = {}
        for key, value in weights.items():
            original_key = key
            key = sanitize_key(key)

            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            if any(key.endswith(sfx) for sfx in NORM_WEIGHT_SUFFIXES):
                if value.ndim == 1 and should_offset_norm_weight(
                    original_key, shift_norm_weights
                ):
                    value += 1.0

            sanitized_weights[key] = value

        return sanitized_weights
