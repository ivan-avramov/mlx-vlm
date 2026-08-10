"""Guards for upstream hunks that were merged, then dropped in a resolution.

A fork-only file, deliberately. These guards cover behaviour that upstream ships
but never wrote a test for, so restoring an upstream test file is not an option —
and a merge that drops a feature hunk usually drops its tests in the same
resolution, which is why most of the real bugs found in this fork had no failing
test. Keep guards here rather than scattering them: `tests/test_utils.py` was
excluded from the suite for a long time (`--ignore`, 5 pre-existing failures), so a
guard placed there would never have run. That file is collected again as of
2026-08-09, but the habit stands — see `tests/test_model_registry.py` for the same
reasoning about registries.

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


class TestSamplerModesReachTheSampler:
    """`36331ea7` (top-nσ), `67ca1f05` (#1653 p-less), `b739dfa4` (#1663 typical-p).

    All three landed their `sample_utils.make_sampler` half and their schema half,
    so `docs/upstream-gaps.md` recorded them as "reachable end to end". They were
    not. The dropped halves were the two places that *choose* a sampler, and the
    failure was silent in both:

    * `generate_step` had no `top_n_sigma` / `p_less` / `typical_p` parameters at
      all. It has `**kwargs`, so `to_generate_kwargs()` handed them over and they
      were swallowed — the dead-parameter tell, with no error anywhere.
    * `ResponseGenerator._make_sampler` returned a `_PositionedTargetSampler`
      whenever `temperature > 0`. That class cannot express any of the three, so
      every server request with one set sampled as if it were unset.

    These guards pin the *selection* logic, which is what was missing, rather than
    the sampler maths, which was always present.
    """

    def _args(self, **kwargs):
        from mlx_vlm.server.generation import GenerationArguments

        return GenerationArguments(max_tokens=8, temperature=0.7, **kwargs)

    def _sampler_for(self, **kwargs):
        from mlx_vlm.server.generation import ResponseGenerator

        # ResponseGenerator.__init__ starts a worker thread; _make_sampler is pure,
        # so bind it to a bare instance rather than standing a server up.
        generator = ResponseGenerator.__new__(ResponseGenerator)
        return ResponseGenerator._make_sampler(generator, self._args(**kwargs))

    def test_generate_step_declares_the_three_sampler_params(self):
        """A `**kwargs` sink is why this needs asserting explicitly.

        Without a declared parameter the value cannot influence sampler choice, and
        nothing raises to tell you.
        """
        import inspect

        from mlx_vlm.generate import ar

        params = inspect.signature(ar.generate_step).parameters
        for name in ("top_n_sigma", "p_less", "typical_p"):
            assert name in params, f"generate_step swallows {name} into **kwargs"

    def test_positioned_sampler_guard_rejects_each_mode(self):
        """The seeded fast path must not claim a request it cannot honour.

        `_PositionedTargetSampler` takes temperature/top_p/seed only, so if the
        guard does not test these three, setting one is silently ignored.
        """
        import inspect

        from mlx_vlm.generate import ar

        source = inspect.getsource(ar.generate_step)
        guard = source.split("if sampler is None:", 1)[1].split("else:", 1)[0]
        assert "top_n_sigma == DEFAULT_TOP_N_SIGMA" in guard
        assert "not p_less" in guard
        assert "typical_p == 1.0" in guard

    def test_server_sampler_honours_top_n_sigma(self):
        from mlx_vlm.generate.ar import _PositionedTargetSampler as ArPositioned
        from mlx_vlm.server.generation import _PositionedTargetSampler

        sampler = self._sampler_for(top_n_sigma=1.5)
        assert not isinstance(sampler, (_PositionedTargetSampler, ArPositioned)), (
            "top_n_sigma must route through make_sampler; the position-keyed "
            "sampler cannot express it"
        )

    def test_server_sampler_honours_p_less(self):
        from mlx_vlm.server.generation import _PositionedTargetSampler

        sampler = self._sampler_for(p_less=True)
        assert not isinstance(sampler, _PositionedTargetSampler)

    def test_server_sampler_honours_typical_p(self):
        from mlx_vlm.server.generation import _PositionedTargetSampler

        sampler = self._sampler_for(typical_p=0.9)
        assert not isinstance(sampler, _PositionedTargetSampler)

    def test_server_sampler_keeps_the_seeded_path_when_no_mode_is_set(self):
        """The fast path must still be chosen by default.

        Losing it would make every seeded request non-reproducible across batch
        groupings, which is the whole point of the position-keyed sampler.
        """
        from mlx_vlm.server.generation import _PositionedTargetSampler

        sampler = self._sampler_for(seed=7)
        assert isinstance(sampler, _PositionedTargetSampler)

    def test_greedy_still_returns_none(self):
        from mlx_vlm.server.generation import GenerationArguments, ResponseGenerator

        generator = ResponseGenerator.__new__(ResponseGenerator)
        args = GenerationArguments(max_tokens=8, temperature=0.0)
        assert ResponseGenerator._make_sampler(generator, args) is None


class TestDiffusionUnification:
    """`29b6c00b` — unify diffusion generation path (#1508).

    The commit landed in six of its eleven files, which is what made it invisible:
    `generate/diffusion.py` is byte-identical to upstream, so
    `stream_diffusion_generate_from_kwargs` and the whole unified engine were
    present and even called from `generate/dispatch.py`. Only the *server* call
    site was dropped, and `check_dead_helpers.py` cannot see that because the
    helper does have a caller.

    What made it a live bug rather than dead code: the fork kept
    `_run_diffusion(family)` routing to `_generate_diffusion` when
    `family == "block"`, but `diffusion_generation_family` — upstream's
    backward-compatible shim — only ever returns `"diffusion"` or `None`. So the
    unified path was unreachable and *every* diffusion request fell through to the
    stale `_generate_masked_diffusion`, which read no request knobs at all. The
    fork's own test called `_run_diffusion("block")` with the literal, so it
    exercised a branch production never took and the suite stayed green.
    """

    def test_diffusion_lane_does_not_branch_on_a_family_the_router_cannot_return(self):
        """The regression that hid this: routing on a literal the router can't produce.

        `_run_diffusion` must not gate its generator choice on a family string.
        The two halves are checked together, because each looks fine alone: the
        shim legitimately returns only "diffusion", and a `family == "block"`
        branch is only wrong *given* that shim.
        """
        import ast
        import inspect

        from mlx_vlm.generate.diffusion import diffusion_generation_family
        from mlx_vlm.server.generation import ResponseGenerator

        producible = {
            node.value
            for node in ast.walk(
                ast.parse(inspect.getsource(diffusion_generation_family).lstrip())
            )
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert "block" not in producible

        lane = inspect.getsource(ResponseGenerator._run_diffusion)
        compared = {
            constant.value
            for node in ast.walk(ast.parse(lane.lstrip()))
            if isinstance(node, ast.Compare)
            for constant in node.comparators
            if isinstance(constant, ast.Constant)
        }
        unreachable = compared - producible
        assert not unreachable, (
            f"_run_diffusion branches on {sorted(unreachable)}, which "
            "diffusion_generation_family never returns — that branch is dead"
        )
        assert "self._generate_diffusion(" in lane

    def test_served_diffusion_path_reads_the_request_knobs(self):
        """`_generate_diffusion` is the only diffusion path, and it honours the request.

        The stale path read only max_tokens/temperature/top_p/seed and took its
        sampler tuning from `config.default_diffusion_*` attributes, so all
        thirteen request-level diffusion options were silently ignored, along with
        `skip_special_tokens` and `top_k`.
        """
        import ast
        import inspect

        from mlx_vlm.server.generation import ResponseGenerator

        assert not hasattr(ResponseGenerator, "_generate_masked_diffusion")

        source = inspect.getsource(ResponseGenerator._generate_diffusion)
        read = {
            node.attr
            for node in ast.walk(ast.parse(source.lstrip()))
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
        }
        for field in ("diffusion_kwargs", "skip_special_tokens", "top_k"):
            assert field in read, f"served diffusion path ignores args.{field}"

    def test_structured_output_is_not_rejected_for_diffusion(self):
        """The stale path raised ValueError; the unified engine takes the processors."""
        import inspect

        from mlx_vlm.server.generation import ResponseGenerator

        source = inspect.getsource(ResponseGenerator._generate_diffusion)
        assert "Structured response_format is not supported" not in source
        assert '"logits_processors"' in source

    def test_diffusion_request_options_are_declared_and_validated(self):
        """The schemas.py half — hidden by `extra="allow"`.

        `FlexibleBaseModel` accepts unknown fields, so the knobs still *reached*
        `GenerationArguments` while undeclared and upstream's own test passes
        either way. What was lost is type/enum validation and OpenAPI visibility:
        `diffusion_sampler="nonsense"` and `block_length="abc"` were forwarded
        straight into the generation path instead of returning 422.
        """
        import pytest

        from mlx_vlm.server.schemas import VLMRequest

        properties = VLMRequest.model_json_schema()["properties"]
        for field in (
            "max_denoising_steps",
            "block_length",
            "num_to_transfer",
            "max_transfer_per_step",
            "editing_threshold",
            "max_post_steps",
            "stability_steps",
            "diffusion_full_canvas",
            "diffusion_min_canvas_length",
            "diffusion_max_canvas_length",
            "diffusion_sampler",
            "threshold",
            "min_threshold",
        ):
            assert field in properties, f"{field} is undocumented and unvalidated"

        messages = [{"role": "user", "content": "hi"}]
        with pytest.raises(ValueError):
            VLMRequest(model="m", messages=messages, diffusion_sampler="nonsense")
        with pytest.raises(ValueError):
            VLMRequest(model="m", messages=messages, block_length="abc")

    def test_block_emitter_reports_generation_tps(self):
        """The fork's inline block-chunk loop omitted `generation_tps` entirely.

        `StreamingToken` has the field, so nothing failed — diffusion streaming
        responses just always reported a null decode rate.
        """
        from types import SimpleNamespace

        from mlx_vlm.server.generation import _DiffusionBlockEmitter

        result = SimpleNamespace(
            is_draft=False,
            text="hello",
            token=5,
            diffusion_block_complete=True,
            finish_reason=None,
            generation_tokens=1,
            peak_memory=1.0,
            prompt_tps=10.0,
            generation_tps=4.0,
        )
        (chunk,) = list(_DiffusionBlockEmitter().feed(result))
        assert chunk.generation_tps == 4.0
        assert chunk.text == "hello"


class TestApcCallSitesFromTheAdapterRefactor:
    """`8422ece8` — APC as a pluggable, capability-driven cache adapter (#1638).

    The commit's `server/app.py` and `generate/dispatch.py` halves were restored on
    2026-08-09 and it was recorded as closed. It was not: the same helpers have a
    *second* call site each, and `check_dead_helpers.py` is per-symbol rather than
    per-call-site, so one library caller satisfies it and the second dropped site is
    invisible. That is the surviving sub-case of "helper landed, call site dropped".
    """

    @staticmethod
    def _batch_generator():
        from mlx_vlm.generate.ar import BatchGenerator

        gen = BatchGenerator.__new__(BatchGenerator)
        gen.apc_manager = object()
        gen.model = None
        gen.processor = None
        # __del__ -> close() reads this; without it the GC raises during teardown.
        gen._wire_stack = None
        return gen

    def test_batch_apc_key_covers_audio_video_embeddings_and_masks(self):
        """The continuous-batching APC key ignored every non-image medium.

        `_apc_extra_hash` returned `tenant_scoped_hash(tenant, image_hash)`, so two
        requests differing *only* in audio, video, inputs_embeds or attention_mask
        hashed identically and shared a prefix cache — reusing KV computed for
        different media. The single-sequence `dispatch.py` path was fixed in the same
        commit's restore; this one was missed.
        """
        import mlx.core as mx

        from mlx_vlm.generate.ar import BatchGenerator

        gen = self._batch_generator()
        base = {"pixel_values": None, "_apc_tenant": None}
        hashes = {
            "none": BatchGenerator._apc_extra_hash(gen, base),
            "audio_a": BatchGenerator._apc_extra_hash(
                gen, dict(base, input_features=mx.zeros((1, 8, 4)))
            ),
            "audio_b": BatchGenerator._apc_extra_hash(
                gen, dict(base, input_features=mx.ones((1, 8, 4)))
            ),
            "video": BatchGenerator._apc_extra_hash(
                gen, dict(base, pixel_values_videos=mx.zeros((1, 2, 3, 4, 4)))
            ),
            "embeds": BatchGenerator._apc_extra_hash(
                gen, dict(base, inputs_embeds=mx.zeros((1, 5, 8)))
            ),
            "masks": BatchGenerator._apc_extra_hash(
                gen, dict(base, attention_mask=mx.ones((1, 6), dtype=mx.int32))
            ),
        }
        assert len(set(hashes.values())) == len(
            hashes
        ), f"APC keys collide across media: {hashes}"
        # Contract of semantic_extra_hash: reduces to tenant_scoped_hash with no media.
        assert hashes["none"] == 0

    def test_batch_prefix_lookup_uses_the_shared_plan_helper(self):
        """`_apc_pick_for` kept an ~85-line inline copy of `apc_lookup_plan`.

        Not a behaviour bug — verified equivalent across 512 scenarios including
        `release()` side-effect order — but two copies of the disk>exact>block
        precedence ladder drift, and only one of them gets upstream's fixes.
        """
        import inspect

        from mlx_vlm.generate.ar import BatchGenerator

        source = inspect.getsource(BatchGenerator._apc_pick_for)
        assert "apc_lookup_plan" in source
        assert (
            "lookup_prefix_disk_cache" not in source
        ), "the precedence ladder is inlined again instead of delegating"

    def test_single_sequence_block_commit_handles_quantized_caches(self):
        """dispatch.py hand-rolled the harvest instead of calling commit_prefix_blocks.

        Its inline snapshot did `c.keys[..., :offset, :]`, which raises TypeError on a
        quantized cache because those store keys as a tuple — swallowed by the
        surrounding `except Exception` into an "APC store failed" warning. So block-mode
        APC harvesting silently stored nothing whenever `--kv-bits` was in use.
        `layer_kv_for_apc` is the helper that exists precisely for this, and its
        docstring says so.
        """
        import inspect

        import mlx.core as mx

        from mlx_vlm import apc as _apc
        from mlx_vlm.generate import dispatch

        assert "commit_prefix_blocks" in inspect.getsource(dispatch)

        class _QuantishCache:
            def __init__(self):
                self.offset = 3
                self._k = mx.zeros((1, 2, 4, 4))
                self._v = mx.ones((1, 2, 4, 4))
                self.keys = (self._k, self._k, self._k)
                self.values = (self._v, self._v, self._v)

            def dequantize_for_apc(self):
                return self._k[..., : self.offset, :], self._v[..., : self.offset, :]

        cache = _QuantishCache()
        # The old inline approach is what must not come back.
        try:
            cache.keys[..., : cache.offset, :]
            raise AssertionError("expected tuple slicing to fail")
        except TypeError:
            pass
        keys, values = _apc.layer_kv_for_apc(cache, batch_idx=None)
        assert keys is not None and values is not None
        assert keys.shape[-2] == 3


class TestStoppingCriteriaDoesNotAliasCallerLists:
    """`0ae7f5e0` + `63965655` — copy eos_token_ids to prevent caller-list aliasing."""

    @staticmethod
    def _criteria(eos_token_ids):
        from types import SimpleNamespace

        from mlx_vlm.utils import StoppingCriteria

        tokenizer = SimpleNamespace(encode=lambda *a, **k: [0], eos_token_ids=[1])
        return StoppingCriteria(eos_token_ids, tokenizer=tokenizer)

    def test_init_copies_the_list(self):
        caller = [1, 2, 3]
        criteria = self._criteria(caller)
        criteria.add_eos_token_ids(999)
        assert caller == [1, 2, 3], "StoppingCriteria mutated the caller's list"
        assert criteria.eos_token_ids == [1, 2, 3, 999]

    def test_reset_copies_the_list(self):
        caller = [7, 8]
        criteria = self._criteria([1])
        criteria.reset(caller)
        criteria.add_eos_token_ids(555)
        assert caller == [7, 8], "reset() aliased the caller's list"
        assert criteria.eos_token_ids == [7, 8, 555]


class TestModelLoadFailureIsABadRequest:
    """`bd6cb123` — return model loading failures as bad requests (#1717)."""

    def test_failed_model_load_raises_400_not_500(self):
        """A bad `--model` / request model path is a client error, not a server fault.

        A 500 tells a client to retry and pages an operator; a 400 says the request
        named a model that cannot be loaded.
        """
        import inspect

        from mlx_vlm.server import generation as server_generation

        source = inspect.getsource(server_generation)
        assert 'status_code=400, detail=f"Failed to load model' in source
        assert 'status_code=500, detail=f"Failed to load model' not in source


class TestReasoningProtocolFields:
    """`221fe0b3` — fix streaming reasoning protocol compatibility (#1644).

    The commit's generation.py and responses_state.py halves had landed; the
    schemas.py half had not. As with the diffusion knobs, `FlexibleBaseModel`'s
    `extra="allow"` meant the fields still *reached* GenerationArguments —
    `_request_field_is_set` reads `model_fields_set`, which includes extras — so
    upstream's own tests for this pass either way. Only the declaration was lost,
    and with it OpenAPI visibility and `reasoning_effort`'s str coercion.
    """

    def test_reasoning_fields_are_declared_on_both_request_models(self):
        from mlx_vlm.server.schemas import OpenAIRequest, VLMRequest

        for model in (OpenAIRequest, VLMRequest):
            properties = model.model_json_schema()["properties"]
            for field in ("reasoning", "reasoning_effort"):
                assert field in properties, f"{model.__name__}.{field} undeclared"


class TestStreamingThinkingMarkerUnion:
    """`221fe0b3`'s streaming half, and a fork gap it exposed.

    The fork rewrote the streaming chat-completions thinking machinery around the
    `THINKING_FORMATS` registry (`_is_prompt_inside_thinking` +
    `_step_thinking_state`), while `/v1/responses` and the non-streaming chat path
    go through upstream's `ThinkingStreamState`. Those two marker sets are not the
    same — `ThinkingStreamState` carries a Cohere pair the registry has no family
    for — so restoring #1644's own test surfaced three layers of the same gap on
    the streaming chat path only:

      1. the prompt-side opener was not recognised, so `in_thinking` was never
         seeded and `reasoning_content` came back empty;
      2. the closer was not in the union format, so once seeded the *entire* reply
         stayed classified as reasoning;
      3. the closer's partial prefixes were absent, so `<|END_THINKING|>` split
         across two tokens leaked verbatim and could never match.

    Fixed as a union (the fork's positional scan x upstream's complete marker set),
    not by picking a side: the fork's structural check finds openers anywhere in the
    prompt, which upstream's `endswith` cannot, and that is load-bearing for Gemma 4's
    global opener.
    """

    def test_prompt_side_detection_covers_non_registry_markers(self):
        from mlx_vlm.server.openai import _is_prompt_inside_thinking

        assert _is_prompt_inside_thinking("prompt<|START_THINKING|>") is True
        assert (
            _is_prompt_inside_thinking("p<|START_THINKING|>x<|END_THINKING|>") is False
        )
        # Explicitly configured markers must work too (--thinking-start-token).
        assert _is_prompt_inside_thinking("p<<GO>>", "<<GO>>", "<<STOP>>") is True
        # The fork's positional generality must survive: opener not at the tail.
        assert _is_prompt_inside_thinking("a<|think|>b") is True
        assert _is_prompt_inside_thinking("plain") is False

    def test_union_format_carries_markers_and_their_partials(self):
        from mlx_vlm.server.openai import _union_thinking_format

        fmt = _union_thinking_format()
        assert "<|START_THINKING|>" in fmt.openers
        assert "<|END_THINKING|>" in fmt.closers
        # Registry families must still be present — this is a union, not a swap.
        assert "<think>" in fmt.openers and "</think>" in fmt.closers
        assert "<|END_THINKING|" in fmt.partial_buffers

    def test_split_closer_and_content_markers_do_not_leak(self):
        """The end-to-end shape: a closer split across tokens, then Cohere's
        structural content wrapper, which the fork's path never stripped."""
        from mlx_vlm.server.openai import _resolve_streaming_thinking_format
        from mlx_vlm.server.responses_state import _step_thinking_state

        fmt = _resolve_streaming_thinking_format("prompt<|START_THINKING|>")
        in_thinking, accumulated = True, ""
        reasoning, content = "", ""
        for token in (
            "North reasoning.",
            "<|END_THINK",
            "ING|><|START_TEXT|>North answer.<|END_TEXT|>",
        ):
            in_thinking, accumulated, delta_r, delta_c = _step_thinking_state(
                token, in_thinking, accumulated, fmt
            )
            reasoning += delta_r or ""
            content += delta_c or ""

        assert reasoning == "North reasoning."
        assert content == "North answer."
        assert in_thinking is False


class TestServerGenerationLogging:
    """`cfcc36d9` — improve server generation logging (#1634).

    The largest single dropped commit in this fork (~375 lines / 9 files). Two of its
    pieces had already landed and were inert, which is the part worth remembering:
    `GenerationMetrics.record_chunk` and its `rate` property were byte-identical to
    upstream, computing an instantaneous decode rate that **nothing read** — the
    `schemas.py` half that surfaces it in streaming responses was the dropped part.
    A value computed and discarded is the same shape as a helper with no call site,
    and neither audit can see it.
    """

    def test_progress_interval_is_configurable_and_fails_safe(self):
        import os

        from mlx_vlm.server.generation import (
            DEFAULT_LOG_PROGRESS_INTERVAL,
            get_log_progress_interval,
        )

        previous = os.environ.get("MLX_VLM_LOG_PROGRESS_INTERVAL")
        try:
            os.environ["MLX_VLM_LOG_PROGRESS_INTERVAL"] = "25"
            assert get_log_progress_interval() == 25
            # 0 disables periodic decode progress and must survive the max(0, ...).
            os.environ["MLX_VLM_LOG_PROGRESS_INTERVAL"] = "0"
            assert get_log_progress_interval() == 0
            # A bad value must not take the server down on the first request.
            os.environ["MLX_VLM_LOG_PROGRESS_INTERVAL"] = "not-a-number"
            assert get_log_progress_interval() == DEFAULT_LOG_PROGRESS_INTERVAL
        finally:
            if previous is None:
                os.environ.pop("MLX_VLM_LOG_PROGRESS_INTERVAL", None)
            else:
                os.environ["MLX_VLM_LOG_PROGRESS_INTERVAL"] = previous

    def test_cli_flag_reaches_the_getter(self):
        """`--log-progress-interval` is only wired through an env var.

        The flag, the export and the getter live in three files, so any one of them
        going missing leaves a flag that parses and does nothing — the same shape as
        the sampler modes that were accepted and discarded.
        """
        import inspect

        from mlx_vlm.server import cli

        source = inspect.getsource(cli.main)
        assert "--log-progress-interval" in source
        assert 'os.environ["MLX_VLM_LOG_PROGRESS_INTERVAL"]' in source

    def test_instantaneous_rate_is_exposed_to_streaming_clients(self):
        """`GenerationMetrics.rate` existed and no schema read it.

        `StreamingTimings` plus `GenerationTimings.from_metrics` preferring
        `metrics.rate` are what make the computed value observable. Without them the
        rate is calculated per chunk and thrown away on every streaming request.
        """
        from mlx_vlm.server.schemas import (
            ChatStreamChunk,
            GenerationTimings,
            ResponseOutputTextDeltaEvent,
            ResponseOutputTextDoneEvent,
            StreamingTimings,
        )

        assert StreamingTimings(predicted_per_second=4.0).predicted_per_second == 4.0
        for model in (ResponseOutputTextDeltaEvent, ResponseOutputTextDoneEvent):
            assert "timings" in model.model_fields

        # ChatStreamChunk must accept BOTH shapes: the terminal chunk carries full
        # GenerationTimings, the per-token chunks carry StreamingTimings.
        chunk = ChatStreamChunk(timings=StreamingTimings(predicted_per_second=1.5))
        assert chunk.timings.predicted_per_second == 1.5

        class _Metrics:
            rate = 7.5
            generation_tps = 1.0
            token_times = []
            cached_tokens = 0
            peak_memory = 0.0
            prompt_tps = None

        timings = GenerationTimings.from_metrics(_Metrics(), 4, 2)
        assert (
            timings.predicted_per_second == 7.5
        ), "from_metrics must prefer the instantaneous rate over generation_tps"

    def test_every_logging_helper_is_called_as_often_as_upstream(self):
        """A *second* dropped call site is invisible to `check_dead_helpers.py`.

        That check is per-symbol, so one caller satisfies it forever. It is how
        `8422ece8` read as closed twice while `commit_prefix_blocks` and
        `apc_lookup_plan` still had a missing call site each. These five helpers are
        wired into the fork's own rewritten GPU worker loops (`_run`, `_step`,
        `_run_speculative`, `_run_diffusion`), so they are exactly the shape that
        goes missing in a resolution — count them against upstream directly.
        """
        import re
        import subprocess

        helpers = (
            "_log_prefill_started",
            "_log_prefill_progress",
            "_log_prefill_completed",
            "_log_decode_progress",
            "_request_log_id",
        )
        ours = open(_generation_path()).read()
        upstream = subprocess.run(
            ["git", "show", "upstream/main:mlx_vlm/server/generation.py"],
            capture_output=True,
            text=True,
            cwd=_repo_root(),
        ).stdout
        if not upstream:
            import pytest

            pytest.skip("upstream/main not fetched")

        for helper in helpers:
            pattern = re.compile(rf"self\.{helper}\(")
            mine = len(pattern.findall(ours))
            theirs = len(pattern.findall(upstream))
            assert mine == theirs, (
                f"{helper}: {mine} call site(s) here vs {theirs} upstream — "
                "a dropped call site the per-symbol check cannot see"
            )


def _repo_root() -> str:
    import pathlib

    return str(pathlib.Path(__file__).resolve().parents[2])


def _generation_path() -> str:
    import pathlib

    return str(pathlib.Path(__file__).resolve().parents[1] / "server" / "generation.py")
