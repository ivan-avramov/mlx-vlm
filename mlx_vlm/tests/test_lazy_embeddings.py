"""M57 lazy prompt embeddings (fork-only, CPU, tiny random-weight qwen3_5).

Text-only prompts embed each prefill chunk from its token ids instead of
materialising the whole prompt's embeddings. Outputs must be bit-identical.
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.generate import ar as ar_module
from mlx_vlm.generate import common as common_module
from mlx_vlm.models.qwen3_5.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.qwen3_5.qwen3_5 import LazyTokenEmbeddings, Model

PROMPT = list(range(1, 14))
STEP = 4


@pytest.fixture(autouse=True)
def _cpu_device():
    previous = mx.default_device()
    tls = common_module._thread_local_streams
    previous_stream = getattr(tls, "stream", None)
    mx.set_default_device(mx.cpu)
    tls.stream = None
    try:
        yield
    finally:
        tls.stream = previous_stream
        mx.set_default_device(previous)


def _tiny_model(model_type="qwen3_5", lazy=True, multimodal=False):
    mx.random.seed(0)
    cfg = TextConfig(
        model_type="qwen3_5_text",
        hidden_size=32,
        intermediate_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_hidden_layers=3,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=1,
        max_position_embeddings=128,
        full_attention_interval=2,
        head_dim=16,
    )
    vision = VisionConfig(
        model_type="qwen3_5",
        depth=1,
        hidden_size=8,
        intermediate_size=16,
        out_hidden_size=32,
        num_heads=1,
        in_channels=3,
        patch_size=4,
        temporal_patch_size=2,
        spatial_merge_size=1,
        num_position_embeddings=4,
    )
    ids = (
        dict(
            image_token_id=10,
            video_token_id=11,
            vision_start_token_id=12,
            vision_end_token_id=13,
        )
        if multimodal
        else {}
    )
    model = Model(
        ModelConfig(
            text_config=cfg, vision_config=vision, model_type=model_type, **ids
        )
    )
    mx.eval(model.parameters())
    model.lazy_prompt_embeddings = lazy  # F5: the worker flag, resolved at load
    return model


class EmbedSpy:
    """Records the token count of every embed_tokens call."""

    def __init__(self, model):
        self.lengths = []
        embed = model.language_model.model.embed_tokens
        self._cls = type(embed)
        spy = self
        real = self._cls.__call__

        class Spied(self._cls):
            def __call__(self, x):
                spy.lengths.append(int(x.shape[-1]))
                return real(self, x)

        embed.__class__ = Spied


def _run(model, lazy, step=STEP, **gen_kwargs):
    if not lazy:
        model = _without_lazy(model)
    cache = model.language_model.make_cache()
    gen = ar_module.generate_step(
        input_ids=mx.array([PROMPT], dtype=mx.int32),
        model=model,
        pixel_values=None,
        mask=None,
        max_tokens=1,
        temperature=0.0,
        prefill_step_size=step,
        prompt_cache=cache,
        **gen_kwargs,
    )
    y, logprobs = next(gen)
    gen.close()
    states = []
    for c in cache:
        state = c.state
        states.extend(
            a for a in (state if isinstance(state, (list, tuple)) else [state])
            if isinstance(a, mx.array)
        )
    mx.eval(states, logprobs)
    return y, logprobs, states


def _without_lazy(model):
    """The same model with the hook hidden: the code path of `main`."""

    class Plain:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "get_lazy_text_embeddings":
                raise AttributeError(name)
            return getattr(self._inner, name)

    return Plain(model)


class TestAC10LazyEmbeddings:
    def test_ac10_text_prompt_state_and_logits_bit_identical_to_main(self):
        model = _tiny_model()
        y0, lp0, st0 = _run(model, lazy=False)
        y1, lp1, st1 = _run(model, lazy=True)
        assert y0 == y1
        assert mx.array_equal(lp0, lp1).item()
        assert len(st0) == len(st1) > 0
        assert all(mx.array_equal(a, b).item() for a, b in zip(st0, st1))

    def test_ac10_unchunked_prefill_bit_identical_to_main(self):
        model = _tiny_model()
        y0, lp0, st0 = _run(model, lazy=False, step=None)
        y1, lp1, st1 = _run(model, lazy=True, step=None)
        assert y0 == y1 and mx.array_equal(lp0, lp1).item()
        assert all(mx.array_equal(a, b).item() for a, b in zip(st0, st1))

    def test_ac10_checkpoint_captures_bit_identical_to_main(self):
        model = _tiny_model()
        captured = {False: [], True: []}
        for lazy in (False, True):

            def checkpoint(n, cache, lazy=lazy):
                mx.eval([c.state for c in cache])
                captured[lazy].append(
                    (n, [mx.array(a) for c in cache for a in _arrays(c.state)])
                )

            _run(
                model,
                lazy=lazy,
                prompt_cache_checkpoint=checkpoint,
                prompt_cache_checkpoint_lengths=[5, 9],
            )
        assert [n for n, _ in captured[True]] == [5, 9]
        assert [n for n, _ in captured[False]] == [5, 9]
        for (_, a), (_, b) in zip(captured[False], captured[True]):
            assert all(mx.array_equal(x, y).item() for x, y in zip(a, b))

    def test_ac10_whole_prompt_embedding_is_never_materialised(self):
        model = _tiny_model()
        spy = EmbedSpy(model)
        _run(model, lazy=False)
        assert max(spy.lengths) == len(PROMPT)  # known positive: main embeds it all
        main_total = sum(spy.lengths)  # prompt + the one decode-step token
        spy.lengths.clear()
        _run(model, lazy=True)
        assert max(spy.lengths) <= STEP, spy.lengths
        assert sum(spy.lengths) == main_total  # every token embedded exactly once

    def test_ac10_multimodal_prompt_takes_the_unchanged_path(self):
        model = _tiny_model()
        dummy = mx.zeros((1,))
        assert model.get_lazy_text_embeddings(mx.array([PROMPT]), dummy) is None
        assert (
            model.get_lazy_text_embeddings(
                mx.array([PROMPT]), None, pixel_values_videos=dummy
            )
            is None
        )
        assert (
            model.get_lazy_text_embeddings(
                mx.array([PROMPT]), None, image_grid_thw=mx.zeros((1, 3))
            )
            is None
        )

    def test_ac10_multimodal_generate_step_calls_get_input_embeddings(self):
        calls = []
        lm = _tiny_model().language_model

        class Fake:
            language_model = lm
            config = SimpleNamespace(model_type="qwen3_5")
            get_lazy_text_embeddings = Model.get_lazy_text_embeddings

            def get_input_embeddings(self, input_ids, pixel_values, mask=None, **kw):
                calls.append(pixel_values)
                return SimpleNamespace(
                    inputs_embeds=lm.model.embed_tokens(input_ids),
                    to_dict=lambda: {},
                )

        gen = ar_module.generate_step(
            input_ids=mx.array([PROMPT], dtype=mx.int32),
            model=Fake(),
            pixel_values=mx.zeros((1,)),
            mask=None,
            max_tokens=1,
            temperature=0.0,
            prefill_step_size=STEP,
            prompt_cache=lm.make_cache(),
        )
        next(gen)
        gen.close()
        assert len(calls) == 1 and calls[0] is not None

    def test_ac10_duck_typed_models_never_opt_in(self):
        from unittest.mock import MagicMock

        assert getattr(type(MagicMock()), "get_lazy_text_embeddings", None) is None

    def test_ac10_other_model_types_are_not_lazy(self):
        model = _tiny_model(model_type="qwen3_5_moe")
        assert model.get_lazy_text_embeddings(mx.array([PROMPT]), None) is None


def _arrays(state):
    return [a for a in (state if isinstance(state, (list, tuple)) else [state])
            if isinstance(a, mx.array)]


class TestLazyTokenEmbeddings:
    def _embed(self):
        mx.random.seed(1)
        return nn.Embedding(16, 8)

    def test_shape_and_prefix_slice_equal_the_eager_array(self):
        embed = self._embed()
        ids = mx.array([[1, 2, 3, 4, 5, 6]])
        lazy = LazyTokenEmbeddings(embed, ids)
        eager = embed(ids)
        assert tuple(lazy.shape) == tuple(eager.shape)
        assert mx.array_equal(lazy[:, :4], eager[:, :4]).item()
        assert mx.array_equal(lazy.materialize(), eager).item()

    def test_suffix_slice_stays_lazy_and_aligned(self):
        embed = self._embed()
        ids = mx.array([[1, 2, 3, 4, 5, 6]])
        rest = LazyTokenEmbeddings(embed, ids)[:, 4:]
        assert isinstance(rest, LazyTokenEmbeddings)
        assert rest.shape[1] == 2
        assert mx.array_equal(rest.materialize(), embed(ids)[:, 4:]).item()

    def test_unsupported_indexing_fails_loudly(self):
        lazy = LazyTokenEmbeddings(self._embed(), mx.array([[1, 2, 3]]))
        with pytest.raises(NotImplementedError):
            lazy[:, 1:2]
        with pytest.raises(NotImplementedError):
            lazy[0]


# ----------------------------------------------------------------- F5 switch
class TestF5Switch:
    def test_f5_hook_is_off_unless_the_flag_was_resolved_onto_the_instance(self):
        off = _tiny_model(lazy=False)
        assert off.get_lazy_text_embeddings(mx.array([PROMPT]), None) is None
        on = _tiny_model(lazy=True)
        assert on.get_lazy_text_embeddings(mx.array([PROMPT]), None) is not None
        bare = _tiny_model(lazy=True)
        del bare.lazy_prompt_embeddings
        assert bare.get_lazy_text_embeddings(mx.array([PROMPT]), None) is None

    def test_f5_default_path_embeds_the_whole_prompt_like_main(self):
        model = _tiny_model(lazy=False)
        spy = EmbedSpy(model)
        _run(model, lazy=True)  # hook present on the type, switch off
        assert max(spy.lengths) == len(PROMPT)

    def test_f5_env_handoff_resolves_once_onto_the_model(self, monkeypatch):
        from mlx_vlm.server import generation as gen

        monkeypatch.delenv("MLX_VLM_LAZY_PROMPT_EMBEDDINGS", raising=False)
        model = _tiny_model(lazy=False)
        gen._apply_lazy_embeddings_from_env(model)
        assert model.lazy_prompt_embeddings is False
        monkeypatch.setenv("MLX_VLM_LAZY_PROMPT_EMBEDDINGS", "1")
        gen._apply_lazy_embeddings_from_env(model)
        assert model.lazy_prompt_embeddings is True
        monkeypatch.setenv("MLX_VLM_LAZY_PROMPT_EMBEDDINGS", "0")
        assert model.lazy_prompt_embeddings is True  # not re-read after load

    def test_f5_cli_flag_writes_the_env_handoff(self, monkeypatch):
        import os
        import sys

        from mlx_vlm.server import cli

        monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **k: None)
        monkeypatch.setattr(cli, "_apply_mlx_memory_limits", lambda *a, **k: None)
        monkeypatch.setattr(cli, "_configure_session_manager", lambda **k: None)
        monkeypatch.setattr(sys, "argv", ["s", "--lazy-prompt-embeddings"])
        cli.main()
        assert os.environ["MLX_VLM_LAZY_PROMPT_EMBEDDINGS"] == "1"
        monkeypatch.setattr(sys, "argv", ["s"])
        cli.main()
        assert os.environ["MLX_VLM_LAZY_PROMPT_EMBEDDINGS"] == "0"


# ----------------------------------------------------------------- F6 traces
from mlx_vlm.models.qwen3_5.language import LanguageModel  # noqa: E402


def _snap(cache):
    mx.eval([c.state for c in cache])
    out = []
    for c in cache:
        offset = getattr(c, "offset", None)
        out.append(
            (
                [mx.array(a) for a in _arrays(c.state)],
                None if offset is None else int(offset),
            )
        )
    return out


def _trace(model, lazy, prime=0, ids=None, pixel=None, **gen_kwargs):
    """Raw logits and cache (arrays + offsets) after every language-model call."""
    runner = model if lazy else _without_lazy(model)
    cache = model.language_model.make_cache()
    logits, snaps = [], []
    real = LanguageModel.__call__

    def traced(self, *args, **kwargs):
        out = real(self, *args, **kwargs)
        mx.eval(out.logits)
        logits.append(mx.array(out.logits))
        snaps.append(_snap(cache))
        return out

    prompt = list(ids if ids is not None else PROMPT)
    LanguageModel.__call__ = traced
    try:
        if prime:  # eager priming, identical in both arms -> warm offset
            for y, _ in _gen(_without_lazy(model), prompt[:prime], cache, max_tokens=0):
                pass
            del logits[:], snaps[:]
        for _ in _gen(runner, prompt[prime:], cache, max_tokens=1, pixel=pixel,
                      **gen_kwargs):
            break
    finally:
        LanguageModel.__call__ = real
    return logits, snaps


def _gen(model, ids, cache, max_tokens, pixel=None, **kw):
    extra = {}
    if pixel is not None:
        extra = {"image_grid_thw": mx.array([[1, 2, 2]])}
    gen = ar_module.generate_step(
        input_ids=mx.array([ids], dtype=mx.int32),
        model=model,
        pixel_values=pixel,
        mask=None,
        max_tokens=max_tokens,
        temperature=0.0,
        prefill_step_size=STEP,
        prompt_cache=cache,
        **extra,
        **kw,
    )
    try:
        yield from gen
    finally:
        gen.close()


def _same(a, b):
    la, sa = a
    lb, sb = b
    assert len(la) == len(lb) > 0
    assert all(mx.array_equal(x, y).item() for x, y in zip(la, lb))  # RAW logits
    assert len(sa) == len(sb)
    for call_a, call_b in zip(sa, sb):
        for (arrs_a, off_a), (arrs_b, off_b) in zip(call_a, call_b):
            assert off_a == off_b
            assert len(arrs_a) == len(arrs_b)
            assert all(mx.array_equal(x, y).item() for x, y in zip(arrs_a, arrs_b))


class TestF6AgainstEager:
    def test_f6_raw_logits_and_cache_after_every_call_cold(self):
        model = _tiny_model()
        _same(_trace(model, False), _trace(model, True))

    def test_f6_prompt_end_cache_is_the_state_before_the_lookahead_decode(self):
        model = _tiny_model()
        eager, lazy = _trace(model, False), _trace(model, True)
        # calls: 3 chunks of 4, the last prompt token, then the lookahead decode
        assert len(eager[0]) == 5
        _same(([eager[0][3]], [eager[1][3]]), ([lazy[0][3]], [lazy[1][3]]))
        offsets = {off for _, off in lazy[1][3] if off is not None}
        assert offsets == {len(PROMPT)}  # prompt end, before any decode token

    def test_f6_warm_session_cache_with_nonzero_initial_offset(self):
        model = _tiny_model()
        eager = _trace(model, False, prime=6)
        lazy = _trace(model, True, prime=6)
        assert {o for _, o in eager[1][0] if o is not None} == {10}  # 6 + chunk 4
        _same(eager, lazy)

    def test_f6_snapshot_landing_boundary(self):
        model = _tiny_model()
        got = []
        for flag in (False, True):
            offsets, arrays, rot = [], [], []
            out = _trace(
                model, flag, snapshot_at_offset=6, anchor_capture_offset=offsets,
                arrays_snapshot_capture=arrays, rotating_snapshot_capture=rot,
            )
            got.append((out, offsets, arrays))
        assert got[0][1] == got[1][1] and got[0][1], got[0][1]
        _same(got[0][0], got[1][0])
        for a, b in zip(got[0][2], got[1][2]):
            assert (a is None) == (b is None)

    @pytest.mark.parametrize("retain_at", [None, 9])
    def test_f6_prompt_end_retention_boundary(self, retain_at):
        model = _tiny_model()
        got = []
        for flag in (False, True):
            offsets, arrays, rot = [], [], []
            out = _trace(
                model, flag, retain_at_offset=retain_at, prompt_end_offset=offsets,
                prompt_end_arrays_capture=arrays, prompt_end_rotating_capture=rot,
            )
            got.append((out, offsets, arrays))
        assert got[0][1] == got[1][1] and got[0][1], got[0][1]
        _same(got[0][0], got[1][0])

    def test_f6_real_multimodal_merge_takes_the_eager_path(self):
        model = _tiny_model(multimodal=True)
        spy = EmbedSpy(model)
        ids = [1, 2, 3, 12, 10, 10, 10, 10, 13, 5, 6, 7, 8, 9]
        mx.random.seed(3)
        pixel = mx.random.normal((4, 96))
        eager = _trace(model, False, ids=ids, pixel=pixel)
        eager_lengths = list(spy.lengths)
        spy.lengths.clear()
        lazy = _trace(model, True, ids=ids, pixel=pixel)
        assert max(eager_lengths) == len(ids)
        assert max(spy.lengths) == len(ids)  # whole prompt embedded: eager merge
        _same(eager, lazy)
