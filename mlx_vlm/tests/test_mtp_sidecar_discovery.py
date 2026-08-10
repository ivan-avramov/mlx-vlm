"""Tests for the fork's `config.json` fallback that locates an MTP sidecar.

Fork-only file, because `tests/test_qwen3_5_mtp_sanitize.py` is shared with upstream and
restored upstream tests stay byte-identical so future merges apply.

`_config_mtp_file` is fork-only (upstream's `_iter_mtp_keys` iterates
`_safetensor_files` only) and was the last entry on
`dev/find_untested_fork_code.py`'s list with no tested caller. It exists because some
converters — mlx-optiq among them — relocate the sidecar out of the repo root, where a
non-recursive `*.safetensors` glob will not find it, and record its real path in
`config.json`. Without the fallback the drafter loads with zero `mtp.` keys and
speculative decoding silently degrades to no drafter at all rather than failing.

No bug found reading it. These pin the contract, including the two shapes real
converters emit (`mtp_file` at the top level, and nested under
`mlx_lm_extra_tensors`).
"""

import json

import pytest

from mlx_vlm.speculative.drafters.qwen3_5_mtp.split import _config_mtp_file


def _write(tmp_path, config, *, sidecar=None):
    (tmp_path / "config.json").write_text(json.dumps(config))
    if sidecar is not None:
        target = tmp_path / sidecar
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
    return tmp_path


class TestConfigMtpFile:
    def test_a_top_level_mtp_file_resolves(self, tmp_path):
        root = _write(
            tmp_path, {"mtp_file": "mtp.safetensors"}, sidecar="mtp.safetensors"
        )

        assert _config_mtp_file(root) == root / "mtp.safetensors"

    def test_the_nested_extra_tensors_form_resolves(self, tmp_path):
        """The shape mlx-optiq writes."""
        root = _write(
            tmp_path,
            {"mlx_lm_extra_tensors": {"mtp_file": "extra/mtp.safetensors"}},
            sidecar="extra/mtp.safetensors",
        )

        assert _config_mtp_file(root) == root / "extra/mtp.safetensors"

    def test_a_relocated_sidecar_outside_the_root_resolves(self, tmp_path):
        """The entire point of the fallback: a path a non-recursive glob would miss."""
        root = _write(
            tmp_path,
            {"mtp_file": "shards/deep/mtp.safetensors"},
            sidecar="shards/deep/mtp.safetensors",
        )

        resolved = _config_mtp_file(root)
        assert resolved == root / "shards/deep/mtp.safetensors"
        assert resolved.exists()

    def test_the_top_level_key_wins_over_the_nested_one(self, tmp_path):
        root = _write(tmp_path, {"mtp_file": "a.safetensors"}, sidecar="a.safetensors")
        (root / "b.safetensors").write_bytes(b"")
        config = json.loads((root / "config.json").read_text())
        config["mlx_lm_extra_tensors"] = {"mtp_file": "b.safetensors"}
        (root / "config.json").write_text(json.dumps(config))

        assert _config_mtp_file(root) == root / "a.safetensors"

    def test_no_config_returns_none(self, tmp_path):
        """The caller then falls back to globbing, which is the common case — most
        repos keep the sidecar at the root and have no `mtp_file` key at all."""
        assert _config_mtp_file(tmp_path) is None

    def test_a_config_without_the_key_returns_none(self, tmp_path):
        root = _write(tmp_path, {"model_type": "qwen3_5"})

        assert _config_mtp_file(root) is None

    def test_a_null_extra_tensors_block_returns_none(self, tmp_path):
        """`config.get("mlx_lm_extra_tensors") or {}` is what makes this safe; without
        the `or {}` a null block would raise instead of falling through to globbing."""
        root = _write(tmp_path, {"mlx_lm_extra_tensors": None})

        assert _config_mtp_file(root) is None

    def test_an_empty_mtp_file_value_returns_none(self, tmp_path):
        root = _write(tmp_path, {"mtp_file": ""})

        assert _config_mtp_file(root) is None

    def test_a_named_but_absent_sidecar_returns_none(self, tmp_path):
        """Existence is checked, so a stale config entry degrades to globbing rather
        than handing the loader a path that will fail to open."""
        root = _write(tmp_path, {"mtp_file": "gone.safetensors"})

        assert _config_mtp_file(root) is None

    def test_malformed_json_raises_rather_than_degrading(self, tmp_path):
        """Deliberately unlike `cli._model_num_attention_heads`, which swallows
        everything. This runs while loading weights, where a corrupt `config.json`
        means the model is unusable — failing loudly beats silently loading a drafter
        with no `mtp.` keys, which is indistinguishable from having no drafter.

        Pinned so the difference reads as a decision rather than an oversight.
        """
        (tmp_path / "config.json").write_text("{not json")

        with pytest.raises(json.JSONDecodeError):
            _config_mtp_file(tmp_path)
