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
import pytest

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
    present and even called from `generate/dispatch.py`. The *server* call site
    was dropped outright, and `check_dead_helpers.py` cannot see that because the
    helper does have a caller.

    **[2026-08-10] This docstring used to end "Only the *server* call site was
    dropped." That was wrong, and it is why the second half survived another
    session.** The commit has TWO call sites and the `generate/dispatch.py` one
    was also incomplete — present, reached, but invoked with FEWER ARGUMENTS than
    upstream passes. Count the call sites, do not trust that one of them is fine
    because the other was fixed; see
    `test_library_diffusion_dispatch_forwards_skip_special_tokens_and_verbose`.

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

    def test_library_diffusion_dispatch_forwards_skip_special_tokens_and_verbose(
        self, monkeypatch
    ):
        """The commit's second call site: `generate/dispatch.py`.

        `stream_diffusion_generate_from_kwargs` declares `skip_special_tokens` and
        `verbose` keyword-only with `= False` defaults. `stream_generate` popped
        both out of `kwargs` and then forwarded neither, so they silently stayed
        False however the caller set them -- and because they were *popped*,
        `**kwargs` did not rescue them either.

        Not cosmetic: `skip_special_tokens` reaches `decode_generated()` in the
        diffusion language models, which decodes EVERY streamed token batch, so
        `--skip-special-tokens` was a no-op on LLaDA2.X, diffusion_gemma and
        nemotron_labs_diffusion alike. `verbose` reaches
        `visualize=bool(verbose or diffusion_show_unmasking)`, so `--verbose`
        could never activate `DiffusionUnmaskingVisualizer` -- while `generate()`
        kept the `response.text_already_printed` handling that only that
        visualizer sets, making the fork internally inconsistent with its own drop.

        Runtime, not source-inspected: a dropped keyword argument is invisible to
        all seven gating audits (the file is present, the def is present, nothing
        was deleted, the helper IS called, no registry entry moved), so this
        assertion is the only thing standing between the fix and a silent
        regression.
        """
        import types

        import mlx.core as mx

        from mlx_vlm.generate import dispatch

        captured = {}

        def fake_helper(
            model,
            processor,
            tokenizer,
            input_ids,
            pixel_values,
            mask,
            skip_special_token_ids,
            kwargs,
            *,
            skip_special_tokens=False,
            verbose=False,
            on_result=None,
        ):
            captured["skip_special_tokens"] = skip_special_tokens
            captured["verbose"] = verbose
            captured["skip_special_token_ids"] = skip_special_token_ids
            return iter(())

        monkeypatch.setattr(dispatch, "is_diffusion_model", lambda model, kwargs: True)
        monkeypatch.setattr(
            dispatch, "stream_diffusion_generate_from_kwargs", fake_helper
        )

        class Tok:
            all_special_ids = [1, 2, 3]

            def decode(self, *a, **k):
                return ""

            def encode(self, s, **k):
                return [0]

        class Proc:
            tokenizer = Tok()

        config = types.SimpleNamespace(
            model_type="llada2_moe", image_token_index=None, eos_token_id=0
        )
        model = types.SimpleNamespace(config=config, language_model=None)

        list(
            dispatch.stream_generate(
                model,
                Proc(),
                "hello",
                input_ids=mx.array([[5, 6, 7]]),
                skip_special_tokens=True,
                verbose=True,
            )
        )

        assert captured["skip_special_tokens"] is True, (
            "stream_generate dropped skip_special_tokens on the diffusion path; "
            "special tokens leak into every diffusion model's output"
        )
        assert captured["verbose"] is True, (
            "stream_generate dropped verbose on the diffusion path; the unmasking "
            "visualizer can never activate"
        )
        # The positional companion was always passed -- kept here so a future
        # refactor cannot quietly trade one for the other.
        assert captured["skip_special_token_ids"] == {1, 2, 3}


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

    def test_every_log_message_literal_appears_as_often_as_upstream(self):
        """The sibling guard above counts HELPERS, and that is why it missed two.

        `cfcc36d9` needed a THIRD restore pass. `682043f5` declared it whole and was
        14 lines short; `59d0229a` finished what the wide pass then reported; and on
        2026-08-10 two more of its hunks were still absent — a
        `logger.info("Generation cancelled: ...")` in the batch loop's cancellation
        path, and one of three `"Prefill completed: ..."` sites (the speculative
        one). Both are DIRECT `logger.info` calls rather than `self._log_*` helper
        calls, so the helper-count guard above could never see them, and both sat in
        blocks otherwise byte-identical to upstream.

        The technique that found them generalises to any logging commit: **count the
        distinctive message literals in both trees.** A log line has no caller, no
        symbol and no test, so literal-counting is the only handle on it — which is
        the concrete form of trap 6's corollary about `print` -> `logger` commits
        leaving no failing test and no audit hit.
        """
        import subprocess

        literals = (
            "Generation cancelled: request=%s generated_tokens=%d",
            "Prefill completed: request=%s prompt_tokens=%d ",
            "Prefill started: request=%s",
            "Generation completed: request=%s",
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

        for literal in literals:
            theirs = upstream.count(literal)
            if not theirs:
                continue  # upstream reworded it; not this guard's business
            mine = ours.count(literal)
            assert mine >= theirs, (
                f"{literal!r}: {mine} occurrence(s) here vs {theirs} upstream — "
                "a dropped logger call site, invisible to every audit"
            )


def _repo_root() -> str:
    import pathlib

    return str(pathlib.Path(__file__).resolve().parents[2])


def _generation_path() -> str:
    import pathlib

    return str(pathlib.Path(__file__).resolve().parents[1] / "server" / "generation.py")


class TestPreloadedModelIsNotReloaded:
    """`7bf4f7ea` — fixes #1402: every first request after startup reloads the model.

    Two lines, and the more instructive half is the second: the fork's copy of
    `test_get_cached_model_omitted_adapter_inherits_loaded_adapter` had been edited to
    assert the *buggy* cache key (`"auto"` instead of `"text_generation"`), so the
    suite actively certified the bug. That is trap 8 — "a test can be edited to
    tolerate its own bug" — and it is why a test-file union has to diff bodies rather
    than just definition names.

    The mechanism: `lifespan()` preloads with `model_kind="text_generation"`, every
    `/chat/completions` request uses the default `model_kind="auto"`, and the cache key
    embeds the *unnormalized* kind while both resolve to the same `text_generation`
    cache group. So the first real request saw a key mismatch on the single slot,
    evicted the preloaded model and loaded it again — making `--preload-model` worse
    than useless (it paid the load cost twice and doubled peak memory transiently).
    """

    def test_preload_then_default_request_loads_once(self, monkeypatch):
        from types import SimpleNamespace

        from mlx_vlm import server

        app_module = server._app_module
        loads = []

        class FakeResponseGenerator:
            def __init__(self, model_path, adapter_path=None, **kwargs):
                loads.append((model_path, adapter_path))
                self.model_path = model_path
                self.adapter_path = adapter_path
                self.model = SimpleNamespace()
                self.processor = SimpleNamespace()
                self.config = SimpleNamespace(model_type="qwen2_vl")

            def wait_until_ready(self):
                return self.model, self.processor, self.config

            def stop_and_join(self):
                pass

        monkeypatch.setattr(app_module, "ResponseGenerator", FakeResponseGenerator)
        monkeypatch.setattr(app_module._apc, "from_env", lambda *a, **k: None)
        monkeypatch.setattr(server.runtime, "model_cache", {})
        monkeypatch.setattr(server.runtime, "response_generator", None)
        monkeypatch.setattr(server.runtime, "apc_manager", None)

        # Exactly what lifespan() does for --preload-model.
        server.get_cached_model("demo-model", None, model_kind="text_generation")
        assert len(loads) == 1

        # Exactly what the first /chat/completions request does (model_kind defaults
        # to "auto"). It must reuse the preloaded model.
        server.get_cached_model("demo-model", None)
        assert (
            len(loads) == 1
        ), "preloaded model was reloaded on the first request (mlx-vlm #1402)"

        # And an explicitly-kinded request must land on the same slot too.
        server.get_cached_model("demo-model", None, model_kind="text_generation")
        assert len(loads) == 1


class TestThinkingBudgetDoesNotSynchronizeDecode:
    """`6a1704e6` — avoid thinking budget decode synchronization.

    Half-landed in the shape that is hardest to spot: the `utils.py` half is
    byte-identical to upstream (`ThinkingBudgetCriteria.pop_forced_token_id`, via the
    `565ca595` rename), and the `ar.py` half was dropped. So `GenerationBatch.next`
    called the *new* helper while keeping the *old* structure around it —
    `mx.eval(self._next_tokens)` then `.tolist()` — a full device synchronization on
    every decode step of every request with a thinking budget. The symbol was present,
    reachable and unit-tested; only the reason it existed was gone.
    """

    def test_forced_token_substitution_does_not_materialize_next_tokens(self):
        import inspect

        from mlx_vlm.generate.ar import GenerationBatch

        source = inspect.getsource(GenerationBatch.next)
        assert (
            "mx.eval(self._next_tokens)" not in source
        ), "forced-token handling synchronizes the decode step again"
        assert ".tolist()" not in source
        # The mask-based substitution is what replaces it.
        assert "mx.where(" in source
        assert "mx.async_eval(self._next_tokens)" in source


class TestQwen35MetalOnlyFastPathsAreGuarded:
    """`16cf6140` (#1423) — Qwen3.5 on CUDA: fix Metal-only crashes.

    A two-file commit whose `gated_delta.py` half landed byte-identical while all
    three of its `language.py` hunks were dropped, so the file read as ordinary fork
    divergence. Both surviving hunks reach `mx.fast.metal_kernel` unconditionally:
    `_TARGET_VERIFY_GEMV` is built at import time, and
    `_qwen3_5_ragged_decode_attention` launches its SDPA kernels with no backend
    check. On a non-Metal backend that is an import-time or decode-time crash rather
    than the intended fall-through to portable `scaled_dot_product_attention`.

    Not reproducible on Apple Silicon, where `mx.metal.is_available()` is always
    True — which is exactly why the drop survived. These guards patch the predicate
    the upstream hunks consult, so they fail against the pre-restore file here.
    """

    def test_ragged_decode_attention_declines_without_metal(self, monkeypatch):
        import mlx.core as mx

        from mlx_vlm.models.qwen3_5.language import _qwen3_5_ragged_decode_attention

        queries = mx.zeros((2, 4, 1, 64), dtype=mx.float16)
        keys = mx.zeros((2, 2, 32, 64), dtype=mx.float16)
        values = mx.zeros((2, 2, 32, 64), dtype=mx.float16)

        # Shapes chosen to satisfy every other precondition, so the only thing
        # that can decline the fast path is the backend check.
        assert (
            _qwen3_5_ragged_decode_attention(queries, keys, values, [0, 0], 0.125)
            is not None
        )

        monkeypatch.setattr(mx.metal, "is_available", lambda: False)
        assert (
            _qwen3_5_ragged_decode_attention(queries, keys, values, [0, 0], 0.125)
            is None
        )

    def test_use_target_verify_dense_declines_when_kernel_is_none(self, monkeypatch):
        import mlx.core as mx

        from mlx_vlm.models.qwen3_5 import language as qwen35_language

        linear = nn.Linear(64, 64)
        x = mx.zeros((1, 4, 64), dtype=mx.float16)

        assert qwen35_language._use_target_verify_dense(linear, x, True) is True

        # What the module global becomes on a non-Metal backend once the
        # `if mx.metal.is_available() else None` guard is in place.
        monkeypatch.setattr(qwen35_language, "_TARGET_VERIFY_GEMV", None)
        assert qwen35_language._use_target_verify_dense(linear, x, True) is False


class TestQwen35QuantizedVerifyPredicateIsFactored:
    """`7fbc7bc9` (#1598) — split `_can_target_verify_quantized`.

    Upstream extracted the weight-only half as `_can_target_verify_quantized_head`
    so `fused_greedy_decode` could reuse it. The extraction landed and the *rewrite*
    of the original function was dropped, so this tree kept the pre-split body: two
    copies of the same predicate, one of which no longer had a reason to exist.

    The two forms are provably equivalent (`x.dtype in (bf16, f16)` +
    `scales.dtype == x.dtype` + `biases.dtype == x.dtype` is the same constraint as
    upstream's `scales.dtype in (bf16, f16)` + `biases.dtype == scales.dtype` +
    `x.dtype == scales.dtype`, and both derive the same `K`), so there is no
    behavioural repro to write — only the duplication to keep from coming back.
    """

    def test_predicate_delegates_rather_than_duplicating(self):
        import inspect

        from mlx_vlm.models.qwen3_5.language import _can_target_verify_quantized

        source = inspect.getsource(_can_target_verify_quantized)
        assert "_can_target_verify_quantized_head(linear)" in source
        assert (
            'linear.mode != "affine"' not in source
        ), "the pre-split body is back; the head predicate is duplicated again"


class TestCompressedTensorsMxfp4FormatIsHonored:
    """`1551c71f` (#1746, "Add Kimi K3") — its 4-line `utils.py` hunk.

    Ten of that commit's eleven files landed (the whole `models/kimi_k3/` package
    and the `prompt_utils.py` format entry), so nothing looked missing. The dropped
    hunk is the one that reads `quantization_config["format"]`: without it every
    `quant_method: compressed-tensors` checkpoint is loaded as
    `mode: "affine"`, including the `mxfp4-pack-quantized` ones the commit added
    support for. `nn.quantize` then dequantizes mxfp4-packed weights with the
    affine formula — wrong numbers, no error, no failing test.
    """

    def _load_with_format(self, fmt):
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch

        import mlx.nn as mlx_nn

        from mlx_vlm.utils import load_model

        class FakeConfig:
            @classmethod
            def from_dict(cls, config):
                return cls()

        class FakeModel(mlx_nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.language_model = mlx_nn.Linear(2, 2, bias=False)

            def load_weights(self, weights, strict=True):
                self.loaded_weights = weights

        quantization_config = {"quant_method": "compressed-tensors"}
        if fmt is not None:
            quantization_config["format"] = fmt

        with (
            patch(
                "mlx_vlm.utils.load_config",
                return_value={
                    "model_type": "kimi_k3",
                    "quantization_config": quantization_config,
                },
            ),
            patch(
                "mlx_vlm.utils.glob.glob",
                return_value=["/tmp/model/model.safetensors"],
            ),
            patch("mlx_vlm.utils._load_safetensors", return_value={}),
            patch(
                "mlx_vlm.utils.get_model_and_args",
                return_value=(
                    SimpleNamespace(ModelConfig=FakeConfig, Model=FakeModel),
                    "kimi_k3",
                ),
            ),
            patch("mlx_vlm.utils.nn.quantize") as quantize,
        ):
            load_model(Path("/tmp/model"), lazy=True)

        return quantize.call_args.kwargs

    def test_mxfp4_packed_format_selects_mxfp4_mode(self):
        kwargs = self._load_with_format("mxfp4-pack-quantized")

        assert kwargs["mode"] == "mxfp4"
        assert kwargs["bits"] == 4
        assert kwargs["group_size"] == 32

    def test_other_formats_still_select_affine(self):
        """The `else` arm — so the guard cannot pass by hard-coding mxfp4."""
        assert self._load_with_format("int4-pack-quantized")["mode"] == "affine"
        assert self._load_with_format(None)["mode"] == "affine"


class TestSubmoduleSanitizeGuardsOnMissingConfigs:
    """`b93707d8` (#1498) — the `hasattr(model_config, "<x>_config")` guards.

    #1498 replaced a `hasattr(model, "thinker")` special case in `load_model` with
    three guarded `if hasattr(model_config, ...)` blocks, having moved the six-way
    thinker sanitize chain into `qwen3_omni_moe.Model.sanitize`. That model-side
    half landed byte-identical; the `utils.py` half did not, so this tree kept the
    pre-#1498 shape — which reaches `model_config.text_config` unguarded.

    `minimax_m3` (a real top-level `model_type`: TEXT_ONLY in `prompt_utils.py`,
    with its own tool parser) exports `LanguageModel` but its `ModelConfig` has no
    `text_config`, so `load_model` raised
    `AttributeError: 'ModelConfig' object has no attribute 'text_config'` and the
    model could not be loaded at all. Upstream's guard skips the block instead.
    Nothing failed here because `test_minimax_m3.py` builds the config and model
    directly and never goes through `load_model`.
    """

    def test_text_only_minimax_m3_loads_without_a_text_config(self):
        from pathlib import Path
        from unittest.mock import patch

        import mlx_vlm.models.minimax_m3 as minimax_m3
        from mlx_vlm.utils import load_model

        assert hasattr(minimax_m3, "LanguageModel")
        config = minimax_m3.ModelConfig.from_dict({"model_type": "minimax_m3"})
        # The precondition the guard exists for; if this ever becomes True the
        # guard stops being load-bearing and this test stops proving anything.
        assert not hasattr(config, "text_config")

        with (
            patch(
                "mlx_vlm.utils.load_config",
                return_value={"model_type": "minimax_m3"},
            ),
            patch(
                "mlx_vlm.utils.glob.glob",
                return_value=["/tmp/model/model.safetensors"],
            ),
            patch("mlx_vlm.utils._load_safetensors", return_value={}),
            patch(
                "mlx_vlm.utils.get_model_and_args",
                return_value=(minimax_m3, "minimax_m3"),
            ),
        ):
            model = load_model(Path("/tmp/model"), lazy=True, strict=False)

        assert model is not None

    def test_omni_still_sanitizes_every_submodule(self):
        """The thinker chain must live on the model, not in `load_model`.

        Removing `load_model`'s `thinker` branch is only safe because #1498's other
        half put the same six passes in `Model.sanitize`, which the unconditional
        `sanitize_weights(model, weights)` above reaches.
        """
        import inspect

        import mlx_vlm.models.qwen3_omni_moe as qwen3_omni_moe

        source = inspect.getsource(qwen3_omni_moe.Model.sanitize)
        for submodule in (
            "self.thinker.sanitize",
            "self.thinker.vision_tower.sanitize",
            "self.thinker.audio_tower.sanitize",
            "self.thinker.language_model.sanitize",
            "self.code2wav.sanitize",
            "self.talker.sanitize",
        ):
            assert submodule in source, f"{submodule} is no longer reached"


class TestApcStoreEmitsItsTrace:
    """`3de5bada` (#1566) — `APC_TRACE` store event, dropped from `apc.py`.

    The purest instance of "helper landed, call site dropped" in the tree, and it
    beat all six audits at once. `apc_trace` is present (parity, symbols), nothing
    was deleted (deletions), `apc.py` is allowlisted (fork markers), and
    `check_dead_helpers.py` is per-symbol — `apc_trace` has two other callers
    (`reject`, `self_check`), so it reads as reachable while the `store` call site
    is gone. Every other file of #1566 landed: `tests/test_apc_observability.py`
    byte-identical, `server/app.py`'s self-check wiring, and the README row
    promising "greppable store/reject/self-check log lines".

    And the restored upstream test does not catch it, which is the point:
    `test_trace_emits_logger_info_when_enabled` calls `apc_trace("store", ...)`
    **directly**, so it exercises the helper rather than the site. Its sibling
    `test_reject_records_emit_trace` drives `store_exact_cache` end to end — for
    the reject path only. This guard is the missing success-path half.
    """

    def test_successful_exact_store_logs_the_store_event(self, monkeypatch, caplog):
        import logging

        import mlx.core as mx

        from mlx_vlm.apc import APCManager
        from mlx_vlm.models.cache import KVCache

        monkeypatch.setenv("APC_TRACE", "1")

        cache = KVCache()
        cache.keys = mx.ones((1, 1, 4, 2))
        cache.values = mx.ones((1, 1, 4, 2))
        cache.offset = 4
        manager = APCManager(num_blocks=4, block_size=4)

        with caplog.at_level(logging.INFO, logger="mlx_vlm.apc"):
            assert manager.store_exact_cache([1, 2, 3, 4], [cache]) is True

        messages = [r.message for r in caplog.records]
        assert any(
            "APC_TRACE store" in m for m in messages
        ), f"store succeeded but emitted no trace; got {messages}"
        assert any("mode=exact" in m for m in messages)
        assert any("token_len=4" in m for m in messages)
        assert any("layers=1" in m for m in messages)

    def test_disk_trace_flag_goes_through_the_shared_helper(self):
        """`_env_truthy("APC_DISK_TRACE")` — #1566's other dropped `apc.py` site.

        The helper landed and this call site kept the pre-refactor inline
        `os.environ.get(...).lower() in (...)`. Equivalent, so the guard is
        structural: it keeps the two flags reading the same predicate.
        """
        import inspect

        import mlx_vlm.apc as apc

        source = inspect.getsource(apc.DiskBlockStore.load_layer_major_prefix)
        assert '_env_truthy("APC_DISK_TRACE")' in source
        assert 'os.environ.get("APC_DISK_TRACE"' not in source


class TestTurboQuantPrefillUsesTheFusedKernel:
    """`e3906673` (#1433, "improved turbo quant's prefill") — its one-line gate.

    A single-file, 8-line commit with three changes; the two Metal-kernel changes
    landed and the third did not. #1433 WIDENED `_TurboQuantMSECodec.quantize`'s
    fast path from `vectors.shape[-2] == 1 and self.bits > 0 and self.use_rht` to
    just `self.bits > 0`, so this tree kept the pre-#1433 narrow gate: anything
    with more than one token — i.e. every prefill — fell past both fused-kernel
    blocks to the slow `_quantize_unit` path. That is precisely the prefill the
    commit title says it improved, and with the gate restored `quantize` is
    byte-identical to upstream's.

    This is the mirror of the other guards here: not missing content, but a
    narrowing that upstream had already removed. Nothing failed — all 241
    turboquant tests pass either way, because both paths produce a valid
    quantization and none of them asserts which one ran.
    """

    def _multi_token_vectors(self):
        import mlx.core as mx

        # dim=64 is a power of two, so use_rht is True and the narrow gate's
        # `shape[-2] == 1` is the only clause that can reject a prefill.
        return mx.ones((1, 2, 8, 64), dtype=mx.float32)

    def test_multi_token_quantize_does_not_take_the_slow_path(self, monkeypatch):
        from mlx_vlm.turboquant import _TurboQuantMSECodec

        codec = _TurboQuantMSECodec(64, 3, seed=0)
        assert codec.use_rht is True

        calls = []
        original = codec._quantize_unit
        monkeypatch.setattr(
            codec,
            "_quantize_unit",
            lambda *a, **k: (calls.append(1), original(*a, **k))[1],
        )

        state = codec.quantize(self._multi_token_vectors())

        assert not calls, (
            "multi-token quantize fell through to _quantize_unit; "
            "the #1433 fast-path gate is narrowed again"
        )
        # The fused path must still produce a usable state, shaped per-vector.
        assert state.norms.shape == (1, 2, 8)

    def test_single_token_quantize_also_uses_it(self):
        """Both shapes take the fused path, which is what `self.bits > 0` means."""
        import mlx.core as mx

        from mlx_vlm.turboquant import _TurboQuantMSECodec

        codec = _TurboQuantMSECodec(64, 3, seed=0)
        state = codec.quantize(mx.ones((1, 2, 1, 64), dtype=mx.float32))

        assert state.norms.shape == (1, 2, 1)


class TestResponsesInputTokensNormalizesInstructions:
    """`eda1ec4f` (#1716) — "Fix Responses instructions for Qwen templates".

    A two-file commit that replaced the same pre-existing
    `chat_messages.insert(0, {"role": "system", ...})` in BOTH `/v1/responses` and
    `/v1/responses/input_tokens` with a call to the new
    `_normalize_response_instruction_messages`, which coalesces `instructions` plus
    every system/developer message into one leading system message and drops the
    originals. The helper landed, `responses_endpoint`'s call site landed, and
    `responses_input_tokens_endpoint`'s did not — it kept the pre-#1716 insert.

    Why nothing caught it, which is the reusable part:

    * `check_dead_helpers.py` is per-symbol, so the one surviving caller made the
      helper read as reachable — the same blind spot that hid two `8422ece8` call
      sites and `apc_trace("store", ...)`.
    * #1716's own two tests landed and pass, because both exercise
      `/v1/responses`. **A restored upstream test covering the call site that
      SURVIVED is worse than no test: it makes the commit look closed.** When a
      commit changes N call sites, count them (`git grep -c`) rather than trusting
      its tests.

    The consequence is a plain contradiction between two endpoints whose entire
    contract is to agree: `/v1/responses/input_tokens` exists to report the token
    count of the prompt `/v1/responses` would build, and it was counting a
    differently-shaped message list — instructions duplicated as a second system
    message and `developer`-role messages left in place, which is exactly what
    #1716's title says breaks Qwen templates.
    """

    @pytest.fixture
    def client(self):
        """Local fixture — `test_server.py`'s is not visible from this file.

        Defined here rather than promoted to `conftest.py` on purpose: this file is
        fork-only and `conftest.py` is shared with restored upstream tests, which are
        kept byte-identical so future merges apply.
        """
        from fastapi.testclient import TestClient

        import mlx_vlm.server as server

        with TestClient(server.app) as test_client:
            yield test_client

    def _captured_messages(self, client, monkeypatch, path):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import mlx.core as mx

        import mlx_vlm.server as server

        seen = {}

        def fake_apply_chat_template(processor, config, messages, **kwargs):
            seen["messages"] = [dict(m) for m in messages]
            return "prompt"

        monkeypatch.setattr(
            server.runtime,
            "response_generator",
            SimpleNamespace(
                _cpu_preprocess=MagicMock(
                    return_value={"input_ids": mx.array([[1, 2, 3]], dtype=mx.int32)}
                )
            ),
        )
        monkeypatch.setattr(
            server,
            "get_cached_model",
            MagicMock(
                return_value=(
                    SimpleNamespace(),
                    SimpleNamespace(),
                    SimpleNamespace(model_type="qwen2_vl"),
                )
            ),
        )
        monkeypatch.setattr(server, "apply_chat_template", fake_apply_chat_template)

        response = client.post(
            path,
            json={
                "model": "demo",
                "instructions": "Be terse.",
                "input": [
                    {"role": "developer", "content": "Prefer bullet points."},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )
        assert response.status_code == 200, response.text
        return seen["messages"]

    def test_developer_message_and_instructions_are_coalesced(
        self, client, monkeypatch
    ):
        messages = self._captured_messages(
            client, monkeypatch, "/responses/input_tokens"
        )

        systems = [m for m in messages if m.get("role") == "system"]
        assert len(systems) == 1, f"expected one coalesced system message: {messages}"
        assert "Be terse." in systems[0]["content"]
        assert "Prefer bullet points." in systems[0]["content"]
        # The originals must be GONE, not merely preceded by a new one. A leftover
        # `developer` role is what breaks the Qwen template #1716 was fixing.
        assert not [m for m in messages if m.get("role") == "developer"]
        assert messages[0] is systems[0] or messages[0] == systems[0]

    def test_it_matches_what_the_shared_normalizer_produces(self, client, monkeypatch):
        """The endpoint must agree with the helper `/v1/responses` calls.

        Comparing against `_normalize_response_instruction_messages` rather than
        against a live `/v1/responses` request is deliberate: driving that endpoint
        needs a whole generation mocked, and the contract under test is the message
        SHAPE, which is exactly what the helper defines. Asserting equality with the
        helper also keeps this guard honest if upstream changes the normalization —
        it tracks the helper instead of a hard-coded expectation.
        """
        from mlx_vlm.server.openai import _normalize_response_instruction_messages

        counted = self._captured_messages(
            client, monkeypatch, "/responses/input_tokens"
        )

        expected = [
            {"role": "developer", "content": "Prefer bullet points."},
            {"role": "user", "content": "Hello"},
        ]
        _normalize_response_instruction_messages(expected, "Be terse.")

        assert counted == expected


class TestOmniAudioFeaturesComeFromTheProcessor:
    """`b93707d8` (#1498) — its `prepare_inputs` half, the mirror of the usual shape.

    Every other guard here protects content a resolution DROPPED. This one protects a
    deletion: #1498 removed `prepare_inputs`' qwen3-omni audio special case
    (`is_qwen3_omni_moe` / `audio_inputs` / `audio_feature_lengths` / `is_lossy_audio`
    / `normalize_audio_features`) in favour of letting the processor do the feature
    extraction, and this tree had reverted that — our copy was #1498's deleted text
    verbatim apart from one `print` -> `logger`, i.e. stale upstream code that read
    exactly like fork work. Its model-side half
    (`qwen3_omni_moe.Model.sanitize`) landed byte-identical, and its `load_model` half
    is guarded by `TestSubmoduleSanitizeGuardsOnMissingConfigs`.

    The reason it stayed open so long is worth recording: **no test in `tests/` passed
    `audio=` to `prepare_inputs`**, so the ~60-line deletion could not be validated
    and marking it `# Fork:` would have laundered stale code into a documented
    feature. These guards are that missing coverage. They assert the contract the
    simplified path depends on — the processor receives audio ARRAYS and whatever it
    returns is forwarded — rather than re-testing the removed branch.
    """

    def _fake_omni_processor(self, captured):
        class _Tokenizer:
            pad_token = "<pad>"
            pad_token_id = 0

        class _FeatureExtractor:
            sampling_rate = 16000

        class Qwen3OmniMoeProcessor:
            """Named so the deleted branch's class-name sniff WOULD have fired."""

            tokenizer = _Tokenizer()
            feature_extractor = _FeatureExtractor()

            def __call__(self, text=None, images=None, audio=None, **kwargs):
                captured["audio"] = audio
                return {
                    "input_ids": [[1, 2, 3]],
                    "attention_mask": [[1, 1, 1]],
                    "input_features": [[[0.5, 0.25]]],
                    "feature_attention_mask": [[1, 1]],
                }

        return Qwen3OmniMoeProcessor()

    def test_processor_output_is_forwarded_and_audio_arrives_as_arrays(
        self, monkeypatch
    ):
        import numpy as np

        import mlx_vlm.utils as utils

        captured = {}
        monkeypatch.setattr(
            utils, "load_audio", lambda path, sr=16000: np.zeros(8, dtype=np.float32)
        )

        model_inputs = utils.prepare_inputs(
            self._fake_omni_processor(captured),
            audio=["clip.wav"],
            prompts=["hi"],
        )

        # The whole point of #1498: the processor extracts the features.
        assert "input_features" in model_inputs
        assert "feature_attention_mask" in model_inputs
        # And it must receive decoded arrays, not the file path. The deleted branch
        # left `audio` as paths for omni while converting it for everything else.
        assert captured["audio"] is not None
        assert not isinstance(captured["audio"][0], str)

    def test_a_missing_feature_extractor_falls_back_to_16k(self, monkeypatch):
        """The removed branch raised ValueError here; upstream's path defaults.

        Keeps the `getattr(..., "sampling_rate", 16000)` fallback honest — it is the
        only thing standing between a processor without a feature_extractor and a
        TypeError.
        """
        import numpy as np

        import mlx_vlm.utils as utils

        captured = {}
        seen_sr = {}

        def fake_load_audio(path, sr=16000):
            seen_sr["sr"] = sr
            return np.zeros(8, dtype=np.float32)

        monkeypatch.setattr(utils, "load_audio", fake_load_audio)
        processor = self._fake_omni_processor(captured)
        del type(processor).feature_extractor

        utils.prepare_inputs(processor, audio=["clip.wav"], prompts=["hi"])

        assert seen_sr["sr"] == 16000

    def test_the_stale_branch_has_not_come_back(self):
        """Structural, because the branch's absence is the fix.

        A future merge that re-offers pre-#1498 content would have no conflict here —
        the deleting commit is an ancestor of `main`, so git treats our old copy as
        deliberately kept. This is the tripwire for that.
        """
        import inspect

        import mlx_vlm.utils as utils

        source = inspect.getsource(utils.prepare_inputs)
        for stale in ("is_qwen3_omni_moe", "is_lossy_audio", "audio_feature_lengths"):
            assert stale not in source, f"pre-#1498 audio branch is back: {stale}"
        assert not hasattr(utils, "normalize_audio_features")


class TestUniformKvQuantSkipsRotatingCaches:
    """Not a dropped hunk — the inverse. `a492e47d`'s uniform path, kept on purpose.

    `generate/common.py::maybe_quantize_kv_cache`'s final (uniform, non-TurboQuant)
    loop is the one branch this fork rewrote rather than took, and it had no test.
    Upstream gates on `hasattr(c, "to_quantized")`. `RotatingKVCache` **has** that
    method — it raises `NotImplementedError("RotatingKVCache Quantization NYI")` — so
    upstream raises for any sliding-window model run with `--kv-bits` on a scheme that
    is neither hybrid nor TurboQuant. Recorded as `docs/upstream-bugs.md` item 4.

    That makes this the shape AGENTS.md warns about in the other direction: a
    divergence with no coverage reads as unexplained fork drift, and the next reader
    to "converge" it reintroduces a crash. The guard is the explanation.
    """

    @staticmethod
    def _rotating_cache_with_content():
        import mlx.core as mx

        from mlx_vlm.models import cache

        entry = cache.RotatingKVCache(max_size=32, keep=4)
        entry.update_and_fetch(mx.zeros((1, 2, 8, 4)), mx.zeros((1, 2, 8, 4)))
        return entry

    def test_the_method_upstream_gates_on_still_raises(self):
        """Pins the premise, so this guard cannot silently become vacuous.

        If mlx ever implements rotating-cache quantization, `hasattr` stops being the
        wrong test, the fork's skip becomes dead weight, and this assertion is what
        says so instead of the guard passing for the wrong reason.
        """
        from mlx_vlm.models import cache

        entry = self._rotating_cache_with_content()
        assert hasattr(entry, "to_quantized")
        with pytest.raises(NotImplementedError):
            entry.to_quantized(group_size=64, bits=4)
        assert isinstance(entry, cache.RotatingKVCache)

    def test_a_rotating_cache_survives_uniform_quantization(self):
        from mlx_vlm.generate.common import maybe_quantize_kv_cache
        from mlx_vlm.models import cache

        prompt_cache = [self._rotating_cache_with_content()]

        maybe_quantize_kv_cache(
            prompt_cache, quantized_kv_start=0, kv_group_size=64, kv_bits=4
        )

        assert isinstance(prompt_cache[0], cache.RotatingKVCache)

    def test_a_plain_cache_alongside_it_is_still_quantized(self):
        """The skip must be surgical: guarding the rotating layer must not disable
        quantization for the layers that can be quantized, which a `return` instead of
        a `continue` would do."""
        import mlx.core as mx

        from mlx_vlm.generate.common import maybe_quantize_kv_cache
        from mlx_vlm.models import cache

        plain = cache.KVCache()
        plain.update_and_fetch(mx.zeros((1, 2, 8, 64)), mx.zeros((1, 2, 8, 64)))
        prompt_cache = [self._rotating_cache_with_content(), plain]

        maybe_quantize_kv_cache(
            prompt_cache, quantized_kv_start=0, kv_group_size=64, kv_bits=4
        )

        assert isinstance(prompt_cache[0], cache.RotatingKVCache)
        assert isinstance(prompt_cache[1], cache.QuantizedKVCache)
