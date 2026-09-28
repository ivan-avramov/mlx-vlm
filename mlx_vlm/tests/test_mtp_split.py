"""Fork: MTP sidecar split tests (moved out of test_models.py at the 2026-09-27
upstream sync — upstream #2276 rewrote test_models.py as JSON-case-driven and this
class has no upstream counterpart)."""

import unittest

import mlx.core as mx


class TestMTPSplit(unittest.TestCase):
    def _write_source(self, tmp, config, tensors):
        import json
        from pathlib import Path

        path = Path(tmp)
        (path / "config.json").write_text(json.dumps(config))
        # no mlx metadata -> splitter treats it as an HF source (sanitize path)
        mx.save_safetensors(str(path / "model.safetensors"), tensors)
        return str(path)

    def _write_mlx_source(self, tmp, config, tensors):
        import json
        from pathlib import Path

        path = Path(tmp)
        (path / "config.json").write_text(json.dumps(config))
        # ``format: mlx`` metadata -> splitter takes the on_mlx_source branch
        mx.save_safetensors(
            str(path / "model.safetensors"), tensors, metadata={"format": "mlx"}
        )
        return str(path)

    def test_registry_resolves_all_families(self):
        from mlx_vlm.speculative.drafters.mtp_split import get_mtp_splitter
        from mlx_vlm.utils import get_model_and_args

        expected = {
            "qwen3_5": "qwen3_5_mtp",
            "qwen3_5_moe": "qwen3_5_mtp",
            "qwen3_next": "qwen3_5_mtp",
            "deepseek_v4": "deepseek_v4_mtp",
            "glm4_moe_lite": "glm4_moe_lite_mtp",
            "glm5_next": "glm5_next_mtp",
            "glm5_next_text": "glm5_next_mtp",
            "glm_moe_dsa": "glm_moe_dsa_mtp",
            "inkling_mm_model": "inkling_mtp",
        }
        for base, out_type in expected.items():
            splitter = get_mtp_splitter(base)
            self.assertIsNotNone(splitter)
            self.assertEqual(splitter.output_model_type, out_type)
        drafter_module, model_type = get_model_and_args(
            {"model_type": "glm_moe_dsa_mtp"}
        )
        self.assertEqual(model_type, "glm_moe_dsa_mtp")
        self.assertTrue(hasattr(drafter_module, "GlmMoeDsaMTPDraftModel"))
        self.assertIsNone(get_mtp_splitter("not_a_model"))

    def test_qwen_split_strips_prefix_and_shifts_norm(self):
        import json
        import tempfile
        from pathlib import Path

        from mlx_vlm.speculative.drafters.qwen3_5_mtp.split import split_qwen3_5_mtp

        norm = mx.random.normal((8,))
        qproj = mx.random.normal((8, 8))
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            src = self._write_source(
                tmp,
                {
                    "model_type": "qwen3_5",
                    "text_config": {
                        "model_type": "qwen3_5",
                        "mtp_num_hidden_layers": 1,
                        "tie_word_embeddings": True,
                    },
                },
                {
                    "mtp.norm.weight": norm,
                    "mtp.layers.0.self_attn.q_proj.weight": qproj,
                },
            )
            split_qwen3_5_mtp(src, out)
            weights = mx.load(str(Path(out) / "model.safetensors"))
            config = json.loads((Path(out) / "config.json").read_text())

        self.assertEqual(
            set(weights), {"norm.weight", "layers.0.self_attn.q_proj.weight"}
        )
        # HF-layout norm weights get the +1.0 shift; the projection passes through
        self.assertTrue(mx.allclose(weights["norm.weight"], norm + 1.0).item())
        self.assertTrue(
            mx.array_equal(weights["layers.0.self_attn.q_proj.weight"], qproj).item()
        )
        self.assertEqual(config["model_type"], "qwen3_5_mtp")
        self.assertEqual(config["block_size"], 3)  # mtp_num_hidden_layers(1) + 2
        self.assertTrue(config["tie_word_embeddings"])

    def test_glm_split_flattens_splits_mla_and_stacks_experts(self):
        import json
        import tempfile
        from pathlib import Path

        from mlx_vlm.speculative.drafters.glm4_moe_lite_mtp.split import (
            split_glm4_moe_lite_mtp,
        )

        N = 2
        p = f"model.layers.{N}."
        tensors = {
            p + "enorm.weight": mx.random.normal((8,)),
            p + "hnorm.weight": mx.random.normal((8,)),
            p + "eh_proj.weight": mx.random.normal((8, 16)),
            p + "embed_tokens.weight": mx.random.normal((10, 8)),
            p + "shared_head.norm.weight": mx.random.normal((8,)),
            p + "shared_head.head.weight": mx.random.normal((10, 8)),
            p + "input_layernorm.weight": mx.random.normal((8,)),
            p + "self_attn.kv_b_proj.weight": mx.random.normal((8, 3)),
            p + "self_attn.o_proj.weight": mx.random.normal((8, 8)),
            p + "mlp.gate.weight": mx.random.normal((2, 8)),
        }
        for e in range(2):
            for proj in ("gate_proj", "down_proj", "up_proj"):
                tensors[p + f"mlp.experts.{e}.{proj}.weight"] = mx.random.normal((8, 8))
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            src = self._write_source(
                tmp,
                {
                    "text_config": {
                        "model_type": "glm4_moe_lite",
                        "num_hidden_layers": N,
                        "num_attention_heads": 2,
                        "qk_nope_head_dim": 2,
                        "v_head_dim": 2,
                        "n_routed_experts": 2,
                        "num_nextn_predict_layers": 1,
                    }
                },
                tensors,
            )
            split_glm4_moe_lite_mtp(src, out)
            weights = mx.load(str(Path(out) / "model.safetensors"))
            config = json.loads((Path(out) / "config.json").read_text())

        self.assertIn("model.embed_tokens.weight", weights)
        self.assertIn("lm_head.weight", weights)
        self.assertIn("model.mtp_block.self_attn.embed_q.weight", weights)
        self.assertIn("model.mtp_block.self_attn.unembed_out.weight", weights)
        self.assertNotIn("model.mtp_block.self_attn.kv_b_proj.weight", weights)
        stacked = weights["model.mtp_block.mlp.switch_mlp.gate_proj.weight"]
        self.assertEqual(stacked.shape, (2, 8, 8))  # 2 experts stacked
        self.assertEqual(config["model_type"], "glm4_moe_lite_mtp")
        self.assertEqual(config["block_size"], 2)  # num_nextn_predict_layers(1) + 1

    def test_glm_moe_dsa_split_extracts_layer_local_mtp(self):
        import json
        import tempfile
        from pathlib import Path

        from mlx_vlm.speculative.drafters.glm_moe_dsa_mtp.split import (
            split_glm_moe_dsa_mtp,
        )

        layer = 2
        prefix = f"model.layers.{layer}."
        tensors = {
            prefix + "enorm.weight": mx.random.normal((8,)),
            prefix + "hnorm.weight": mx.random.normal((8,)),
            prefix + "eh_proj.weight": mx.random.normal((8, 16)),
            prefix + "input_layernorm.weight": mx.random.normal((8,)),
            prefix + "post_attention_layernorm.weight": mx.random.normal((8,)),
            prefix + "shared_head.norm.weight": mx.random.normal((8,)),
            prefix + "self_attn.kv_b_proj.weight": mx.random.normal((8, 4)),
            prefix + "mlp.gate.weight": mx.random.normal((2, 8)),
        }
        for expert in range(2):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                tensors[prefix + f"mlp.experts.{expert}.{projection}.weight"] = (
                    mx.random.normal((8, 8))
                )
        config = {
            "model_type": "glm_moe_dsa",
            "num_hidden_layers": layer,
            "num_attention_heads": 2,
            "qk_nope_head_dim": 2,
            "v_head_dim": 2,
            "n_routed_experts": 2,
            "num_nextn_predict_layers": 1,
            "index_share_for_mtp_iteration": True,
        }

        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            source = self._write_source(tmp, config, tensors)
            split_glm_moe_dsa_mtp(source, out)
            weights = mx.load(str(Path(out) / "model.safetensors"))
            sidecar = json.loads((Path(out) / "config.json").read_text())

        self.assertEqual(sidecar["model_type"], "glm_moe_dsa_mtp")
        self.assertEqual(sidecar["block_size"], 2)
        self.assertTrue(sidecar["text_config"]["index_share_for_mtp_iteration"])
        self.assertTrue(all(key.startswith("mtp.") for key in weights))
        self.assertIn("mtp.shared_head_norm.weight", weights)
        self.assertIn("mtp.self_attn.embed_q.weight", weights)
        self.assertIn("mtp.self_attn.unembed_out.weight", weights)
        self.assertNotIn("mtp.self_attn.kv_b_proj.weight", weights)
        self.assertEqual(
            weights["mtp.mlp.switch_mlp.gate_proj.weight"].shape,
            (2, 8, 8),
        )

    def test_detect_and_split_mtp_dispatch(self):
        import tempfile
        from pathlib import Path

        from mlx_vlm.speculative.drafters.mtp_split import detect_mtp_splitter
        from mlx_vlm.split_mtp import split_mtp

        cfg = {
            "model_type": "qwen3_5",
            "text_config": {"model_type": "qwen3_5", "mtp_num_hidden_layers": 1},
        }
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            src = self._write_source(
                tmp, cfg, {"mtp.norm.weight": mx.random.normal((8,))}
            )
            splitter = detect_mtp_splitter(Path(src))
            self.assertIsNotNone(splitter)
            self.assertEqual(splitter.output_model_type, "qwen3_5_mtp")
            split_mtp(src, out)  # auto-detect dispatch
            self.assertTrue((Path(out) / "model.safetensors").exists())

    def test_detect_returns_none_when_flag_but_no_tensors(self):
        import tempfile
        from pathlib import Path

        from mlx_vlm.speculative.drafters.mtp_split import detect_mtp_splitter

        # config declares MTP but ships no mtp.* tensors (the MiniMax-M2 trap)
        cfg = {
            "model_type": "qwen3_5",
            "text_config": {"model_type": "qwen3_5", "mtp_num_hidden_layers": 1},
        }
        with tempfile.TemporaryDirectory() as tmp:
            src = self._write_source(
                tmp, cfg, {"model.embed_tokens.weight": mx.random.normal((4, 8))}
            )
            self.assertIsNone(detect_mtp_splitter(Path(src)))

    def test_qwen3_next_split_stacks_separate_experts(self):
        import json
        import tempfile
        from pathlib import Path

        from mlx_vlm.speculative.drafters.mtp_split import detect_mtp_splitter
        from mlx_vlm.split_mtp import split_mtp

        norm = mx.random.normal((8,))
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            tensors = {
                "mtp.pre_fc_norm_embedding.weight": norm,
                "mtp.fc.weight": mx.random.normal((8, 16)),
                "mtp.norm.weight": mx.random.normal((8,)),
                "mtp.layers.0.input_layernorm.weight": mx.random.normal((8,)),
            }
            for e in range(2):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    tensors[f"mtp.layers.0.mlp.experts.{e}.{proj}.weight"] = (
                        mx.random.normal((8, 8))
                    )
            src = self._write_source(
                tmp,
                {"text_config": {"model_type": "qwen3_next", "num_experts": 2}},
                tensors,
            )
            splitter = detect_mtp_splitter(Path(src))
            self.assertIsNotNone(splitter)
            self.assertEqual(splitter.output_model_type, "qwen3_5_mtp")
            split_mtp(src, out)
            weights = mx.load(str(Path(out) / "model.safetensors"))
            config = json.loads((Path(out) / "config.json").read_text())

        self.assertTrue(all(not k.startswith("mtp.") for k in weights))
        # zero-centered-norm +1.0 shift applies to Qwen3-Next norms too
        self.assertTrue(
            mx.allclose(weights["pre_fc_norm_embedding.weight"], norm + 1.0).item()
        )
        # separate up/down/gate experts collapse into a stacked switch_mlp
        self.assertEqual(
            weights["layers.0.mlp.switch_mlp.gate_proj.weight"].shape, (2, 8, 8)
        )
        # expert order is preserved: stacked[e] is expert e (C45 N3b)
        self.assertTrue(
            mx.array_equal(
                weights["layers.0.mlp.switch_mlp.gate_proj.weight"][1],
                tensors["mtp.layers.0.mlp.experts.1.gate_proj.weight"],
            ).item()
        )
        self.assertNotIn("layers.0.mlp.experts.0.gate_proj.weight", weights)
        self.assertEqual(config["model_type"], "qwen3_5_mtp")
        self.assertEqual(config["block_size"], 3)  # depth defaults to 1 (+2)

    def test_requested_quantization_quantizes_fp_drafter(self):
        import json
        import tempfile
        from pathlib import Path

        from mlx_vlm.split_mtp import split_mtp

        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            src = self._write_source(
                tmp,
                {
                    "model_type": "qwen3_5",
                    "text_config": {
                        "model_type": "qwen3_5",
                        "mtp_num_hidden_layers": 1,
                    },
                },
                {
                    "mtp.norm.weight": mx.random.normal((64,)),
                    "mtp.layers.0.self_attn.q_proj.weight": mx.random.normal((64, 64)),
                },
            )
            split_mtp(src, out, q_bits=4, q_group_size=64)
            weights = mx.load(str(Path(out) / "model.safetensors"))
            config = json.loads((Path(out) / "config.json").read_text())

        # the 2D projection got affine-quantized; the 1D norm did not
        self.assertIn("layers.0.self_attn.q_proj.scales", weights)
        self.assertIn("layers.0.self_attn.q_proj.biases", weights)
        self.assertNotIn("norm.scales", weights)
        self.assertEqual(config["quantization"]["mode"], "affine")
        self.assertEqual(config["quantization"]["bits"], 4)

    def test_qwen3_5_moe_fused_layout_unchanged(self):
        # Fork (C41 regression pin): the fused ``experts.gate_up_proj`` /
        # ``experts.down_proj`` layout that Qwen3.5-MoE HF checkpoints ship must
        # keep splitting exactly as before the separate-expert stacking landed.
        import tempfile
        from pathlib import Path

        from mlx_vlm.split_mtp import split_mtp

        norm = mx.random.normal((8,))
        gate_up = mx.random.normal((2, 16, 8))
        down = mx.random.normal((2, 8, 8))
        gate = mx.random.normal((2, 8))
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            src = self._write_source(
                tmp,
                {
                    "model_type": "qwen3_5_moe",
                    "text_config": {
                        "model_type": "qwen3_5_moe",
                        "num_experts": 2,
                        "mtp_num_hidden_layers": 1,
                    },
                },
                {
                    "mtp.norm.weight": norm,
                    "mtp.layers.0.mlp.experts.gate_up_proj": gate_up,
                    "mtp.layers.0.mlp.experts.down_proj": down,
                    "mtp.layers.0.mlp.gate.weight": gate,
                },
            )
            split_mtp(src, out, model_type="qwen3_5_moe")
            weights = mx.load(str(Path(out) / "model.safetensors"))

        self.assertEqual(
            set(weights),
            {
                "norm.weight",
                "layers.0.mlp.switch_mlp.gate_proj.weight",
                "layers.0.mlp.switch_mlp.up_proj.weight",
                "layers.0.mlp.switch_mlp.down_proj.weight",
                "layers.0.mlp.gate.weight",
            },
        )
        self.assertTrue(
            mx.array_equal(
                weights["layers.0.mlp.switch_mlp.gate_proj.weight"], gate_up[:, :8]
            ).item()
        )
        self.assertTrue(
            mx.array_equal(
                weights["layers.0.mlp.switch_mlp.up_proj.weight"], gate_up[:, 8:]
            ).item()
        )
        self.assertTrue(
            mx.array_equal(
                weights["layers.0.mlp.switch_mlp.down_proj.weight"], down
            ).item()
        )
        self.assertTrue(mx.allclose(weights["norm.weight"], norm + 1.0).item())

    def test_qwen3_5_moe_split_stacks_separate_experts(self):
        # Fork (C41 defect 1): Qwen3.5-MoE sources that ship SEPARATE per-expert
        # gate/up/down tensors (the Qwen3-Next layout) must collapse into the
        # stacked switch_mlp the qwen3_5_mtp drafter loads, not pass through loose.
        import re
        import tempfile
        from pathlib import Path

        from mlx_vlm.split_mtp import split_mtp

        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            tensors = {
                "mtp.norm.weight": mx.random.normal((8,)),
                "mtp.layers.0.input_layernorm.weight": mx.random.normal((8,)),
                "mtp.layers.0.mlp.gate.weight": mx.random.normal((2, 8)),
            }
            for e in range(2):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    tensors[f"mtp.layers.0.mlp.experts.{e}.{proj}.weight"] = (
                        mx.random.normal((8, 8))
                    )
            src = self._write_source(
                tmp,
                {
                    "model_type": "qwen3_5_moe",
                    "text_config": {
                        "model_type": "qwen3_5_moe",
                        "num_experts": 2,
                        "mtp_num_hidden_layers": 1,
                    },
                },
                tensors,
            )
            split_mtp(src, out, model_type="qwen3_5_moe")
            weights = mx.load(str(Path(out) / "model.safetensors"))

        stacked = weights["layers.0.mlp.switch_mlp.gate_proj.weight"]
        self.assertEqual(stacked.shape, (2, 8, 8))
        # expert order is preserved: stacked[e] is expert e
        self.assertTrue(
            mx.array_equal(
                stacked[1], tensors["mtp.layers.0.mlp.experts.1.gate_proj.weight"]
            ).item()
        )
        self.assertIn("layers.0.mlp.switch_mlp.up_proj.weight", weights)
        self.assertIn("layers.0.mlp.switch_mlp.down_proj.weight", weights)
        loose = [k for k in weights if re.search(r"\.experts\.\d+\.", k)]
        self.assertEqual(loose, [])

    def test_detect_falls_back_to_root_model_type(self):
        # Fork (C41 defect 2): HF Qwen3.5-MoE configs carry
        # text_config.model_type == "qwen3_5_moe_text" (unregistered) while the
        # root model_type is the registered base; detect must try both, first
        # REGISTERED wins -- not "first present".
        import tempfile
        from pathlib import Path

        from mlx_vlm.speculative.drafters.mtp_split import detect_mtp_splitter
        from mlx_vlm.speculative.drafters.qwen3_5_mtp.split import Qwen3_5MTPSplitter

        cfg = {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "mtp_num_hidden_layers": 1,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            src = self._write_source(
                tmp, cfg, {"mtp.norm.weight": mx.random.normal((8,))}
            )
            splitter = detect_mtp_splitter(Path(src))
        self.assertIsInstance(splitter, Qwen3_5MTPSplitter)

    def test_split_raises_on_loose_per_expert_keys(self):
        # Fork (C41 defect 3): if expert stacking cannot fire (here num_experts=3
        # but only experts 0..1 ship) the drafter would be written with loose
        # per-expert keys the runtime cannot load. Fail loudly, write nothing.
        import tempfile
        from pathlib import Path

        from mlx_vlm.split_mtp import split_mtp

        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            tensors = {
                "mtp.norm.weight": mx.random.normal((8,)),
                "mtp.layers.0.mlp.gate.weight": mx.random.normal((3, 8)),
            }
            for e in range(2):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    tensors[f"mtp.layers.0.mlp.experts.{e}.{proj}.weight"] = (
                        mx.random.normal((8, 8))
                    )
            src = self._write_source(
                tmp,
                {
                    "model_type": "qwen3_5_moe",
                    "text_config": {
                        "model_type": "qwen3_5_moe",
                        "num_experts": 3,
                        "mtp_num_hidden_layers": 1,
                    },
                },
                tensors,
            )
            with self.assertRaises(ValueError):
                split_mtp(src, out, model_type="qwen3_5_moe")
            self.assertFalse((Path(out) / "model.safetensors").exists())

    def test_qwen3_5_moe_mlx_source_stacks_separate_experts(self):
        # Fork (C45 N1): an already-MLX Qwen3.5-MoE source with SEPARATE
        # per-expert tensors takes the on_mlx_source branch of transform, which
        # returned BEFORE postprocess -- experts were never stacked and the C41
        # loose-key guard then rejected a legitimate source.
        import re
        import tempfile
        from pathlib import Path

        from mlx_vlm.split_mtp import split_mtp

        norm = mx.random.normal((8,))
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            tensors = {
                "mtp.norm.weight": norm,
                "mtp.layers.0.mlp.gate.weight": mx.random.normal((2, 8)),
            }
            for e in range(2):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    tensors[f"mtp.layers.0.mlp.experts.{e}.{proj}.weight"] = (
                        mx.random.normal((8, 8))
                    )
            src = self._write_mlx_source(
                tmp,
                {
                    "model_type": "qwen3_5_moe",
                    "text_config": {
                        "model_type": "qwen3_5_moe",
                        "num_experts": 2,
                        "mtp_num_hidden_layers": 1,
                    },
                },
                tensors,
            )
            split_mtp(src, out, model_type="qwen3_5_moe")  # must not raise
            weights = mx.load(str(Path(out) / "model.safetensors"))

        self.assertTrue(all(not k.startswith("mtp.") for k in weights))
        # the MLX branch was really taken: no +1.0 norm shift on an MLX source
        self.assertTrue(mx.array_equal(weights["norm.weight"], norm).item())
        for proj in ("gate_proj", "up_proj", "down_proj"):
            stacked = weights[f"layers.0.mlp.switch_mlp.{proj}.weight"]
            self.assertEqual(stacked.shape, (2, 8, 8))
            # expert order is preserved: stacked[e] is expert e
            for e in range(2):
                self.assertTrue(
                    mx.array_equal(
                        stacked[e],
                        tensors[f"mtp.layers.0.mlp.experts.{e}.{proj}.weight"],
                    ).item()
                )
        loose = [k for k in weights if re.search(r"\.experts\.\d+\.", k)]
        self.assertEqual(loose, [])

    def test_stack_separate_experts_infers_num_experts(self):
        # Fork (C45 N3a): with ``num_experts`` absent from the config the count
        # is inferred as max expert index + 1, and order is preserved.
        import re

        from mlx_vlm.speculative.drafters.qwen3_5_mtp.split import (
            _stack_separate_experts,
        )

        tensors = {"layers.0.mlp.gate.weight": mx.random.normal((3, 8))}
        for e in range(3):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                tensors[f"layers.0.mlp.experts.{e}.{proj}.weight"] = mx.random.normal(
                    (8, 8)
                )
        originals = dict(tensors)
        _stack_separate_experts(tensors, {})

        for proj in ("gate_proj", "up_proj", "down_proj"):
            stacked = tensors[f"layers.0.mlp.switch_mlp.{proj}.weight"]
            self.assertEqual(stacked.shape, (3, 8, 8))
            for e in range(3):
                self.assertTrue(
                    mx.array_equal(
                        stacked[e],
                        originals[f"layers.0.mlp.experts.{e}.{proj}.weight"],
                    ).item()
                )
        loose = [k for k in tensors if re.search(r"\.experts\.\d+\.", k)]
        self.assertEqual(loose, [])
        self.assertIn("layers.0.mlp.gate.weight", tensors)
