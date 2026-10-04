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


def _tiny_model(model_type="qwen3_5"):
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
    model = Model(
        ModelConfig(text_config=cfg, vision_config=vision, model_type=model_type)
    )
    mx.eval(model.parameters())
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
