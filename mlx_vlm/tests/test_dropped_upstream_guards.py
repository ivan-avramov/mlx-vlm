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

    def test_embeddings_route_is_served(self):
        """Behavioural on purpose: reachability, not registration.

        An earlier version of this guard scanned `app.routes`, which broke the
        moment #1714 moved the protocol surfaces onto `inference_router` —
        FastAPI >= 0.141 does not flatten `include_router()` into `app.routes`.
        Asserting the request does not 404 is version-proof and is the thing
        that actually regressed (the route 404'd for real).
        """
        from fastapi.testclient import TestClient

        from mlx_vlm.server.app import app

        response = TestClient(app).post("/v1/embeddings", json={"input": "hello"})

        assert response.status_code != 404
        # No embedding model configured -> 400, which proves the handler ran.
        assert response.status_code == 400

    def test_embedding_models_get_their_own_cache_group(self):
        """Without this branch an embedding model lands in `text_generation`.

        That is the same registry slot as the served language model, so loading
        an embedding model would evict the LLM (and vice versa) instead of
        living alongside it.
        """
        from mlx_vlm.server.app import _cache_group_for_cache

        assert _cache_group_for_cache({"model_kind": "embedding"}) == "embedding"


class TestOneBitLoaderWiring:
    """`960b26f9` — Add 1-bit inference kernel (Python-hosted) (#1597).

    The kernel (`quantization/one_bit.py`) and its tests landed; only `utils.py`'s
    7 lines of loader wiring were dropped, so stock MLX saw `bits=1` and rejected
    it — a 494-line backend reachable by nothing. `test_one_bit.py` exercises the
    kernel directly and so passed throughout.
    """

    def test_loader_imports_the_one_bit_hooks(self):
        """`utils` must hold both hooks, and they must be the real ones.

        This is also an autoflake tripwire: `--remove-all-unused-imports` would
        delete this import outright if the two call sites were ever lost again.
        """
        import mlx_vlm.utils as utils
        from mlx_vlm.quantization import one_bit

        assert utils.replace_one_bit_modules is one_bit.replace_one_bit_modules
        assert utils._quantization_for_path is one_bit._quantization_for_path

    def test_per_path_quantization_reports_one_bit_layers(self):
        """The predicate keys off this, so a wrong shape silently re-quantizes."""
        from mlx_vlm.quantization.one_bit import _quantization_for_path

        quantization = {"bits": 4, "group_size": 64, "model.layers.0.mlp": {"bits": 1}}

        assert _quantization_for_path(quantization, "model.layers.0.mlp")["bits"] == 1
        assert _quantization_for_path(quantization, "model.layers.1.mlp")["bits"] == 4


class TestSamplerModesAreReachable:
    """`36331ea7` / #1653 / #1663 — top-nσ, p-less and locally-typical sampling.

    `sample_utils.make_sampler` accepted all three the whole time; nothing carried
    a request's value to it. `GenerationArguments` had no fields for them and the
    request schemas did not declare them, so the three modes were unreachable
    through the API — a dead-end at the *entry* rather than the exit.

    This guard walks the whole chain, because each link was independently broken
    and a unit test on any single one would have passed.
    """

    def test_request_schemas_declare_the_three_modes(self):
        from mlx_vlm.server.schemas import OpenAIRequest, VLMRequest

        for cls in (OpenAIRequest, VLMRequest):
            fields = cls.model_fields
            for name in ("top_n_sigma", "p_less", "typical_p"):
                assert name in fields, f"{cls.__name__} is missing {name}"

    def test_request_value_reaches_the_sampler(self):
        from types import SimpleNamespace

        from mlx_vlm.sample_utils import make_sampler
        from mlx_vlm.server.app import _build_gen_args

        request = SimpleNamespace(
            max_tokens=8,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
            min_p=0.0,
            top_n_sigma=1.5,
            p_less=True,
            typical_p=0.9,
            seed=None,
            logprobs=False,
            model_fields_set=set(),
        )

        args = _build_gen_args(request)
        assert (args.top_n_sigma, args.p_less, args.typical_p) == (1.5, True, 0.9)

        kwargs = args.to_generate_kwargs()
        assert kwargs["top_n_sigma"] == 1.5
        assert kwargs["p_less"] is True
        assert kwargs["typical_p"] == 0.9

        # The exit was never the problem; assert it still lines up anyway.
        assert callable(
            make_sampler(
                temp=0.7,
                top_p=0.9,
                top_n_sigma=kwargs["top_n_sigma"],
                p_less=kwargs["p_less"],
                typical_p=kwargs["typical_p"],
            )
        )

    def test_diffusion_kwargs_are_inert_by_default(self):
        """The restored diffusion fields must not leak kwargs into normal requests.

        `to_generate_kwargs` now merges `diffusion_kwargs()`, which emits only
        explicitly-supplied values. If that ever started emitting defaults, every
        text request would carry 13 unexpected kwargs into `generate()`.
        """
        from mlx_vlm.server.generation import GenerationArguments

        assert GenerationArguments().diffusion_kwargs() == {}


class TestLagunaTokenizationContract:
    """`53052569` — fix(laguna): preserve provider tokenization contract.

    Upstream *does* ship a test for this
    (`test_processors.py::TestLagunaProcessor::test_chat_template_owns_laguna_special_tokens`),
    but `TestLagunaProcessor` is defined **twice** in that file, so the earlier
    definition — the one holding this test — is shadowed by the later one and is
    never collected. That is why a missing `utils.should_add_special_tokens`
    failed nothing here despite a test for it sitting in the tree. Our
    `test_processors.py` is byte-identical to upstream, so the shadowing is an
    upstream bug too; the guard lives here instead of being duplicated there.
    """

    def test_laguna_chat_template_owns_its_special_tokens(self):
        from types import SimpleNamespace

        from mlx_vlm.utils import should_add_special_tokens

        processor = SimpleNamespace(chat_template="{{ messages }}")

        assert should_add_special_tokens("laguna", processor) is False
        assert should_add_special_tokens("llama", processor) is True
        # A model whose template owns the markers but that has no template must
        # fall back to adding them.
        assert should_add_special_tokens("laguna", SimpleNamespace()) is True

    def test_generate_paths_use_the_shared_helper(self):
        """Both generate paths must go through the helper, not an inline gemma list.

        The inline conditional they replaced listed only the gemma variants, so
        Laguna fell through to `True` and the provider's markers were duplicated.
        """
        from mlx_vlm.generate import ar, dispatch
        from mlx_vlm.utils import should_add_special_tokens

        assert ar.should_add_special_tokens is should_add_special_tokens
        assert dispatch.should_add_special_tokens is should_add_special_tokens

    def test_upstreams_own_laguna_test_is_still_shadowed(self):
        """Tripwire: if upstream ever de-duplicates the class, drop this guard.

        Asserts the shadowing that makes the guard above necessary. When this
        starts failing, upstream's test is collected and this class is redundant.
        """
        import ast
        from pathlib import Path

        source = Path(__file__).with_name("test_processors.py").read_text()
        names = [
            node.name
            for node in ast.parse(source).body
            if isinstance(node, ast.ClassDef) and node.name == "TestLagunaProcessor"
        ]
        assert len(names) == 2, (
            "TestLagunaProcessor is no longer duplicated in test_processors.py — "
            "upstream's own laguna test now collects, so TestLagunaTokenizationContract "
            "can be removed."
        )
