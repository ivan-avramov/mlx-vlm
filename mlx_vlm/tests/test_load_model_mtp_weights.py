"""Fork: `utils.load_model`'s packaged-`mtp.*` strip.

The strip drops a target checkpoint's bundled MTP head (the drafter is loaded
separately). It must not fire for a model whose own parameters live under
`mtp.` -- the standalone `glm_moe_dsa_mtp` drafter -- or the drafter loads
with every module dropped and no error.
"""

import importlib
import json
from copy import deepcopy
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_vlm.tests.test_models import TINY_DEFAULTS
from mlx_vlm.utils import load_model

DATA = json.loads(Path(__file__).with_name("model_cases.json").read_text())[
    "speculative"
]


def _glm_moe_dsa_mtp_text_config():
    return deepcopy(TINY_DEFAULTS | DATA["mla_mtp"] | {"model_type": "glm_moe_dsa"})


def test_standalone_drafter_keeps_its_own_mtp_parameters(tmp_path):
    arch = importlib.import_module("mlx_vlm.speculative.drafters.glm_moe_dsa_mtp")
    text = _glm_moe_dsa_mtp_text_config()
    reference = arch.Model(arch.ModelConfig.from_dict({"text_config": text}))
    expected = dict(tree_flatten(reference.parameters()))
    assert expected and all(k.startswith("mtp.") for k in expected)

    config = {"model_type": "glm_moe_dsa_mtp", "text_config": text}
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), expected, metadata={"format": "mlx"}
    )

    loaded = load_model(tmp_path)
    actual = dict(tree_flatten(loaded.parameters()))
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        assert mx.array_equal(value, expected[key]), key
