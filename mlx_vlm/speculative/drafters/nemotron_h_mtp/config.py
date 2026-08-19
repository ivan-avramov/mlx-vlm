import inspect
from dataclasses import dataclass, field
from typing import List, Optional

from ....models.base import BaseModelConfig
from ....models.nemotron_h.config import ModelConfig as NemotronHConfig


class TextConfig:
    @classmethod
    def from_dict(cls, params: dict):
        return NemotronHConfig.from_dict(params)


@dataclass
class NemotronHMTPConfig(BaseModelConfig):
    model_type: str = "nemotron_h_mtp"
    text_config: Optional[TextConfig] = None
    block_size: int = 2
    runtime_block_size: Optional[int] = None
    tie_word_embeddings: bool = False

    # Nemotron-H's native MTP head is not a single homogeneous decoder layer:
    # the checkpoint stores it as `mtp.layers.<i>.*` where each index carries
    # its own block type (observed on the BF16 base: ["attention", "moe"]).
    # These are HF block-type words (matching `layers_block_type`), not the
    # single-char codes `NemotronHBlock` expects -- `_block_type_chars()`
    # below does that translation.
    mtp_block_types: List[str] = field(default_factory=lambda: ["attention", "moe"])

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if self.runtime_block_size is None:
            nextn_depth = 1
            if self.text_config is not None:
                nextn_depth = getattr(self.text_config, "num_nextn_predict_layers", 1)
            self.runtime_block_size = min(self.block_size, int(nextn_depth) + 1)

    def block_type_chars(self) -> List[str]:
        """Translate HF block-type words to `NemotronHBlock`'s single-char codes."""
        mapping = NemotronHConfig._block_type_to_char
        return [mapping[block_type] for block_type in self.mtp_block_types]

    @classmethod
    def from_dict(cls, params: dict) -> "NemotronHMTPConfig":
        flat = dict(params)
        text_config = flat.get("text_config") or {}
        nextn_depth = text_config.get("num_nextn_predict_layers", 1)
        flat.setdefault("block_size", int(nextn_depth) + 1)
        block_types = flat.get("mtp_block_types") or text_config.get(
            "mtp_layers_block_type"
        )
        if block_types:
            flat["mtp_block_types"] = list(block_types)
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
