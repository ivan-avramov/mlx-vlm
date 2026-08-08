"""Guards for upstream hunks that were merged, then dropped in a resolution.

A fork-only file, deliberately. These guards cover behaviour that upstream ships
but never wrote a test for, so restoring an upstream test file is not an option —
and a merge that drops a feature hunk usually drops its tests in the same
resolution, which is why most of the real bugs found in this fork had no failing
test. `tests/test_utils.py` is excluded from the suite
(`--ignore=tests/test_utils.py`, 5 pre-existing failures), so a guard placed
there would never run; see `tests/test_model_registry.py` for the same reasoning.

Each guard names the upstream commit whose content it protects.
"""

import mlx.nn as nn

from mlx_vlm.generate import ar
from mlx_vlm.utils import skip_multimodal_module


class TestMinimaxM3Support:
    """`ecc457b2` — Minimax m3 support (#1374)."""

    def test_make_cache_honors_a_caches_own_to_batch(self):
        """A cache that ships `to_batch` must not fall through the isinstance chain.

        Without the guard at the top of `_make_cache.to_batch_cache`, MiniMax M3's
        sparse index-key side cache reaches the final `else` and raises
        "does not yet support batching", so the model cannot run on any
        batch or server path.
        """
        from mlx_vlm.models.minimax_m3_vl.language import (
            MiniMaxM3BatchKVCache,
            MiniMaxM3KVCache,
        )

        class _Model(nn.Module):
            def make_cache(self):
                return [MiniMaxM3KVCache()]

        caches = ar._make_cache(_Model(), [0, 1])

        assert len(caches) == 1
        assert isinstance(caches[0], MiniMaxM3BatchKVCache)

    def test_patch_merge_mlp_is_a_multimodal_module(self):
        """`patch_merge_mlp` must be skipped when skipping multimodal modules.

        MiniMax M3 VL names its patch-merge projector `patch_merge_mlp`; leaving it
        out of the suffix list quantizes/loads it as a text module.
        """
        assert skip_multimodal_module("x.patch_merge_mlp.fc1") is True


class TestQuantizedKVStartOnTheBatchPath:
    """`dab4cb45` — Honor quantized_kv_start on the batch TurboQuant path (#1582).

    Upstream's own tests (`test_generate.py::TestBatchTurboQuantizedKVStart`) cover
    `_make_cache` in isolation. This guard covers the *plumbing* instead, which is
    where the fork's bug actually was: `--quantized-kv-start` reached
    `BatchGenerator.self.quantized_kv_start` and was read by nothing, so
    TurboQuant quantized from token 0 no matter what the operator configured.
    """

    def test_prompt_processing_batch_threads_it_into_make_cache(self, monkeypatch):
        import mlx.core as mx

        seen = {}

        def fake_make_cache(model, left_padding, **kwargs):
            seen.update(kwargs)
            return []

        monkeypatch.setattr(ar, "_make_cache", fake_make_cache)

        ar.PromptProcessingBatch(
            model=nn.Module(),
            uids=[1, 2],
            input_ids=[[4, 5], [6, 7, 8]],
            max_tokens=[1, 1],
            inputs_embeds=mx.ones((2, 3, 4)),
            prompt_kwargs={},
            prefill_step_size=None,
            kv_bits=3.5,
            kv_quant_scheme="turboquant",
            quantized_kv_start=5000,
        )

        assert seen["quantized_kv_start"] == 5000
        # `prefill_length` must be the padded prompt length, so the deferral
        # decision is made against the real prefill size.
        assert seen["prefill_length"] == 3


class TestEmbeddingServingIsWired:
    """`40757df3` — Add native embedding serving infra.

    `server/embeddings.py` and `models/pooling.py` landed byte-identical to
    upstream and had **zero importers**; only the wiring was dropped, so
    `/v1/embeddings` 404'd while README documented it. A dropped import is
    invisible to both audits — parity only sees missing files and the symbol
    check only sees missing `def`/`class` names.
    """

    def test_embeddings_route_is_registered(self):
        from mlx_vlm.server.app import app

        assert "/v1/embeddings" in {route.path for route in app.routes}

    def test_embedding_models_get_their_own_cache_group(self):
        """Without this branch an embedding model lands in `text_generation`.

        That is the same registry slot as the served language model, so loading
        an embedding model would evict the LLM (and vice versa) instead of
        living alongside it.
        """
        from mlx_vlm.server.app import _cache_group_for_cache

        assert _cache_group_for_cache({"model_kind": "embedding"}) == "embedding"
