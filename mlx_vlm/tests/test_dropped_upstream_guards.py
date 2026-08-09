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


class TestLLGuidanceFastPath:
    """`6d5603b3` — Speed up llguidance structured decoding (#1628).

    `tests/test_structured.py` is byte-identical to upstream and passed with the
    *old* implementation, because #1628 is a behaviour-preserving performance
    rewrite: same masked logits, different machinery. So nothing in the suite
    distinguished the two, which is why 4 unreviewed `.symbol-exclusions` entries
    sat on this file. These guards assert the machinery, not the output.
    """

    def test_mask_kernel_masks_exactly_the_disallowed_tokens(self):
        """The Metal kernel replaced a per-row Python loop over the batch."""
        import mlx.core as mx

        from mlx_vlm.structured import _allocate_shared_bitmask, _apply_llguidance_mask

        mask_mx, view = _allocate_shared_bitmask(1, 64)
        view[:] = 0
        view[0, 0] = 0b1011  # allow tokens 0, 1 and 3
        logits = mx.arange(64, dtype=mx.float32)[None, :]

        out = _apply_llguidance_mask(logits, mask_mx)
        mx.eval(out)
        row = out[0].tolist()

        allowed = [i for i, v in enumerate(row) if v != float("-inf")]
        assert allowed == [0, 1, 3]
        assert [row[i] for i in allowed] == [0.0, 1.0, 3.0]

    def test_shared_bitmask_is_a_writable_contiguous_view(self):
        """The whole speedup rests on CPU-writable MLX memory.

        `_allocate_shared_bitmask` raises if MLX ever stops exposing that, so this
        pins the property rather than trusting it.
        """
        from mlx_vlm.structured import _allocate_shared_bitmask

        mask_mx, view = _allocate_shared_bitmask(2, 100)

        assert mask_mx.shape == (2, (100 + 31) // 32)
        assert view.flags["C_CONTIGUOUS"] and view.flags["WRITEABLE"]

    def test_processor_requests_an_immediate_decode_yield(self):
        """A constrained token must not wait behind unrelated prefill work.

        `BatchGenerator._next` reads this flag to return right after decoding; a
        latency fix that is invisible to any correctness test.
        """
        from mlx_vlm.structured import LLGuidanceLogitsProcessor

        assert LLGuidanceLogitsProcessor.requires_immediate_decode_yield is True

    def test_batch_generator_honours_the_yield_flag(self):
        import ast
        import inspect
        import textwrap

        from mlx_vlm.generate import ar

        source = inspect.getsource(ar.BatchGenerator._next)
        assert "requires_immediate_decode_yield" in source
        assert "yield_after_decode" in source
        # The flag is only useful if something returns on it. `textwrap.dedent`,
        # not `inspect.cleandoc` — the latter is for docstrings and leaves method
        # source indented, so `ast.parse` raises and the walk silently finds
        # nothing, which passes a badly-written version of this assertion.
        tree = ast.parse(textwrap.dedent(source))
        returns_on_flag = any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "yield_after_decode"
            and any(isinstance(s, ast.Return) for s in node.body)
            for node in ast.walk(tree)
        )
        assert returns_on_flag


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


class TestGenArgsUnion:
    """`221fe0b3` (#1644) — `server/request_normalization.py`, wired 2026-08-09.

    Upstream moved the gen-args helpers into that module; the resolution dropped
    the move and kept `app.py`'s copies, so the module shipped with zero
    importers. Wiring it was a **union, not a restore**: `app.py`'s copy of
    `_build_gen_args` had grown three fork behaviours the module never had, and the
    obvious reading ("delete app.py's duplicates") would have silently dropped all
    three. The suite could not have caught that — upstream has no test for the fork
    half, by definition, which is what these guards are for.

    The first two fork behaviours do have coverage in `test_server.py`
    (`test_repeat_penalty_alias_recognized`, `test_generation_defaults_*`); those
    tests exercise them through `server._build_gen_args` and so keep passing
    wherever the implementation lives. The guards here instead pin the *union*
    itself: that the delegation is real, and that the upstream half now works.
    """

    def _chat_request(self, **kwargs):
        from mlx_vlm import server

        return server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            **kwargs,
        )

    def test_app_delegates_to_the_request_normalization_module(self):
        """The module must be reached, not merely present.

        It sat in the tree with zero importers for long enough to be catalogued as
        an orphan; a regression would look exactly like that again.
        """
        import ast
        import importlib
        import inspect

        # NOT `from mlx_vlm.server import app`: server/__init__.py mirrors app.py's
        # names into the package, and app.py defines `app` as the FastAPI instance,
        # so that spelling hands back the ASGI object. Same shadowing trap as
        # `import mlx_vlm.convert` returning the re-exported function.
        app = importlib.import_module("mlx_vlm.server.app")

        source = inspect.getsource(app._build_gen_args)
        calls = [
            node
            for node in ast.walk(ast.parse(source.lstrip()))
            if isinstance(node, ast.Call)
        ]
        targets = {ast.unparse(c.func) for c in calls}
        assert "_request_normalization._build_gen_args" in targets, (
            "app._build_gen_args no longer delegates to request_normalization; the "
            "module is orphaned again"
        )

    def test_repeat_penalty_alias_survives_the_union(self):
        """Fork behaviour 1: Ollama's `repeat_penalty` spelling.

        OpenWebUI's Advanced Params slider sends the Ollama name on every endpoint.
        Upstream's copy reads `repetition_penalty` only, so taking it wholesale
        would drop the knob silently — no error, just an ignored slider.
        """
        from mlx_vlm import server

        args = server._build_gen_args(self._chat_request(repeat_penalty=1.15))
        assert args.repetition_penalty == 1.15

        # An explicit `repetition_penalty` still wins over the alias.
        both = server._build_gen_args(
            self._chat_request(repetition_penalty=1.20, repeat_penalty=1.05)
        )
        assert both.repetition_penalty == 1.20

    def test_generation_defaults_apply_only_to_omitted_fields(self, monkeypatch):
        """Fork behaviour 2: registry `generation_defaults`, and its precedence.

        Precedence is request > yaml > checkpoint > hardcoded. The subtle half is
        the alias rule: `max_tokens` and `max_output_tokens` are the same knob, so a
        request setting *either* counts as explicit and the yaml value must not
        overwrite it. Note this rule governs the **overlay only** — the base build
        reads `max_tokens` first and `ChatRequest.max_tokens` has a non-None
        default_factory, so the alias never feeds the base value on this request
        type. What the rule prevents is the registry clobbering a caller who
        expressed the limit under the other name.
        """
        import json

        from mlx_vlm import server

        monkeypatch.setenv(
            "MLX_VLM_GENERATION_DEFAULTS",
            json.dumps({"temperature": 0.3, "max_tokens": 111}),
        )

        omitted = server._build_gen_args(self._chat_request())
        assert omitted.temperature == 0.3
        assert omitted.max_tokens == 111

        explicit = server._build_gen_args(self._chat_request(max_tokens=7))
        assert explicit.max_tokens == 7, "explicit max_tokens must beat the default"

        aliased = server._build_gen_args(self._chat_request(max_output_tokens=9))
        assert aliased.max_tokens != 111, (
            "max_output_tokens is an alias for max_tokens, so setting it counts as "
            "explicit and the registry default must not be overlaid"
        )

    def test_resolved_sampling_is_logged(self, caplog):
        """Fork behaviour 3: the resolved-sampling line.

        It is the only way to observe which registry default or request override
        actually took effect, and it must stay on the `mlx_vlm.server` logger that
        operators configure — moving the code into `request_normalization` would
        otherwise have silently moved it to that module's own logger name.
        """
        import logging

        from mlx_vlm import server

        with caplog.at_level(logging.INFO, logger="mlx_vlm.server"):
            server._build_gen_args(self._chat_request(temperature=0.42))

        lines = [
            r.getMessage()
            for r in caplog.records
            if r.name == "mlx_vlm.server" and "resolved sampling" in r.getMessage()
        ]
        assert (
            lines
        ), "the resolved-sampling line is gone from the mlx_vlm.server logger"
        assert "temperature=0.42" in lines[-1]

    def test_reasoning_effort_none_disables_thinking(self):
        """Upstream half, previously inert: OpenAI-standard reasoning controls.

        `reasoning_effort` arrived on the request (the schemas are `extra='allow'`)
        and nothing read it, so a client asking for `"none"` got whatever the server
        default happened to be — the same shape of bug as #1714, an API accepting a
        parameter and doing nothing with it.
        """
        from mlx_vlm import server

        off = server._build_gen_args(self._chat_request(reasoning_effort="none"))
        assert off.enable_thinking is False

        on = server._build_gen_args(self._chat_request(reasoning_effort="high"))
        assert on.enable_thinking is True

    def test_explicit_enable_thinking_beats_reasoning_effort(self):
        """Precedence: the explicit fork/vendor field wins over the standard one.

        Without this, adding the standard controls would have changed behaviour for
        callers who already send `enable_thinking`.
        """
        from mlx_vlm import server

        args = server._build_gen_args(
            self._chat_request(enable_thinking=True, reasoning_effort="none")
        )
        assert args.enable_thinking is True

    def test_diffusion_passthroughs_reach_the_args(self):
        """Upstream half: the 13 diffusion knobs `29b6c00b` (#1508) added.

        `GenerationArguments` grew the fields in `7ebb5690`; nothing populated them
        until the module was wired, so every diffusion request silently ran at the
        engine defaults.
        """
        from mlx_vlm import server

        args = server._build_gen_args(
            self._chat_request(max_denoising_steps=5, block_length=16, threshold=0.7)
        )
        assert args.max_denoising_steps == 5
        assert args.block_length == 16
        assert args.threshold == 0.7
