"""Env-gated prefill component profiler (M57 step 2, 2026-10-04).

CPU-pinned, no checkpoint: a tiny random-weight ``qwen3_5`` language model
(one GatedDeltaNet layer + one full-attention layer) is driven through the real
``generate_step`` chunked-prefill loop.
"""

import re
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm import prefill_profile
from mlx_vlm.generate import ar as ar_module
from mlx_vlm.models import cache as cache_module
from mlx_vlm.models.qwen3_5.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel

ENV = prefill_profile.ENV
PHASES = (
    "sdpa",
    "kv_update",
    "attn_prep",
    "attn_out",
    "gdn",
    "mlp",
    "cache_post",
    "clear_cache",
    "other",
)
LINE_RE = re.compile(
    r"^\[prefill_profile\] chunks=(?P<chunks>\d+) tokens=(?P<tokens>\d+) "
    r"keys=(?P<keys>\d+) wall=(?P<wall>[-\d.]+) sdpa=(?P<sdpa>[-\d.]+) "
    r"kv_update=(?P<kv_update>[-\d.]+) attn_prep=(?P<attn_prep>[-\d.]+) "
    r"attn_out=(?P<attn_out>[-\d.]+) gdn=(?P<gdn>[-\d.]+) mlp=(?P<mlp>[-\d.]+) "
    r"cache_post=(?P<cache_post>[-\d.]+) clear_cache=(?P<clear_cache>[-\d.]+) "
    r"other=(?P<other>[-\d.]+) final=(?P<final>[01])$"
)
# 13 prompt tokens, step 4: three chunks of 4 (12 tokens), last token via _step.
PROMPT = list(range(1, 14))
STEP = 4


@pytest.fixture(autouse=True)
def _cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _tiny_lm():
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
        num_hidden_layers=2,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=1,
        max_position_embeddings=128,
        full_attention_interval=2,
        head_dim=16,
    )
    # get_rope_index reads vision_config/image ids even for text-only prompts.
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
    lm = LanguageModel(
        cfg,
        config=ModelConfig(
            text_config=cfg, vision_config=vision, model_type="qwen3_5"
        ),
    )
    mx.eval(lm.parameters())
    return lm


def _fake_model(lm):
    def get_input_embeddings(input_ids, pixel_values, mask=None, **kwargs):
        return SimpleNamespace(
            inputs_embeds=lm.model.embed_tokens(input_ids), to_dict=lambda: {}
        )

    return SimpleNamespace(
        language_model=lm, get_input_embeddings=get_input_embeddings
    )


def _run(lm=None):
    """One generate_step prefill + first token. Returns (token, logprobs, states)."""
    lm = lm if lm is not None else _tiny_lm()
    prompt_cache = lm.make_cache()
    gen = ar_module.generate_step(
        input_ids=mx.array([PROMPT], dtype=mx.int32),
        model=_fake_model(lm),
        pixel_values=None,
        mask=None,
        max_tokens=1,
        temperature=0.0,
        prefill_step_size=STEP,
        prompt_cache=prompt_cache,
    )
    y, logprobs = next(gen)
    gen.close()
    states = [mx.array(a) for c in prompt_cache for a in _flat_state(c)]
    mx.eval(states)
    return y, logprobs, states


def _flat_state(entry):
    state = entry.state
    if isinstance(state, (list, tuple)):
        return [a for a in state if isinstance(a, mx.array)]
    return [state]


def _lines(capsys):
    err = capsys.readouterr().err
    return [l for l in err.splitlines() if l.startswith("[prefill_profile]")]


def _count_sync(monkeypatch):
    real = mx.synchronize
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(mx, "synchronize", counting)
    return calls


class TestSwitchUnset:
    def test_zero_synchronize_no_line_active_none(self, monkeypatch, capsys):
        monkeypatch.delenv(ENV, raising=False)
        calls = _count_sync(monkeypatch)
        _run()
        assert calls["n"] == 0
        assert _lines(capsys) == []
        assert prefill_profile.active() is None
        assert prefill_profile.from_env() is None


class TestSwitchSet:
    def test_three_chunk_run_emits_final_line_with_every_field(
        self, monkeypatch, capsys
    ):
        monkeypatch.setenv(ENV, "1")
        _run()
        lines = _lines(capsys)
        assert len(lines) == 1, lines
        m = LINE_RE.match(lines[0])
        assert m is not None, lines[0]
        f = m.groupdict()
        assert f["final"] == "1"
        assert f["chunks"] == "3"
        assert f["tokens"] == "12"
        assert f["keys"] == "12"
        for key in ("wall",) + PHASES:
            value = float(f[key])
            assert value == value and value != float("inf")
            assert value >= 0.0, (key, value)
        # other is the residual; allow a hair of timer slop below zero -> clamped.
        assert float(f["other"]) >= 0.0
        named = sum(float(f[k]) for k in PHASES if k != "other")
        assert float(f["wall"]) >= named - 1e-6

    def test_window_line_every_n_chunks_precedes_final(self, monkeypatch, capsys):
        monkeypatch.setenv(ENV, "1")
        monkeypatch.setattr(prefill_profile.PrefillProfiler, "every", 2)
        _run()
        lines = _lines(capsys)
        assert [l.endswith("final=0") for l in lines] == [True, False]
        w, fin = LINE_RE.match(lines[0]), LINE_RE.match(lines[1])
        assert (w["chunks"], w["tokens"], w["keys"]) == ("2", "8", "8")
        # final line covers the whole generation
        assert (fin["chunks"], fin["tokens"], fin["keys"]) == ("3", "12", "12")

    def test_every_hook_fires_once_per_chunk_and_layer(self, monkeypatch, capsys):
        monkeypatch.setenv(ENV, "1")
        seen = []
        real = prefill_profile.PrefillProfiler.mark

        def spy(self, phase, *outputs):
            seen.append(phase)
            return real(self, phase, *outputs)

        monkeypatch.setattr(prefill_profile.PrefillProfiler, "mark", spy)
        _run()
        for phase, n in {
            "gdn": 3,
            "mlp": 6,
            "attn_prep": 3,
            "kv_update": 3,
            "sdpa": 3,
            "attn_out": 3,
            "cache_post": 3,
            "clear_cache": 3,
        }.items():
            assert seen.count(phase) == n, (phase, seen)

    def test_other_model_family_reports_chunk_level_phases_only(
        self, monkeypatch, capsys
    ):
        monkeypatch.setenv(ENV, "1")
        lm = _tiny_lm()
        # Hide the handle from layer code == a model family with no layer hooks.
        seen = []
        real_mark = prefill_profile.PrefillProfiler.mark
        monkeypatch.setattr(
            prefill_profile.PrefillProfiler,
            "mark",
            lambda self, phase, *o: (seen.append(phase), real_mark(self, phase, *o))[1],
        )
        monkeypatch.setattr(prefill_profile, "active", lambda: None)
        _run(lm)
        assert set(seen) == {"other_fence", "cache_post", "clear_cache"}
        m = LINE_RE.match(_lines(capsys)[-1])
        assert m is not None and m["chunks"] == "3"
        for key in ("sdpa", "kv_update", "attn_prep", "attn_out", "gdn", "mlp"):
            assert float(m[key]) == 0.0


class TestOutputsUnchanged:
    def test_logits_and_cache_state_identical_with_switch_set(self, monkeypatch):
        monkeypatch.delenv(ENV, raising=False)
        y0, lp0, s0 = _run()
        monkeypatch.setenv(ENV, "1")
        y1, lp1, s1 = _run()
        assert mx.array_equal(y0, y1).item()
        assert mx.array_equal(lp0, lp1).item()
        assert len(s0) == len(s1) > 0
        for a, b in zip(s0, s1):
            assert a.shape == b.shape
            assert mx.array_equal(a, b).item()


class TestRaisingProfiler:
    def test_raising_mark_does_not_break_generation_and_clears_handle(
        self, monkeypatch
    ):
        monkeypatch.delenv(ENV, raising=False)
        y0, lp0, s0 = _run()
        monkeypatch.setenv(ENV, "1")

        def boom(self, phase, *outputs):
            raise RuntimeError("profiler bug")

        monkeypatch.setattr(prefill_profile.PrefillProfiler, "mark", boom)
        y1, lp1, s1 = _run()
        assert mx.array_equal(y0, y1).item()
        assert mx.array_equal(lp0, lp1).item()
        assert prefill_profile.active() is None

    def test_raising_report_does_not_break_generation(self, monkeypatch):
        monkeypatch.setenv(ENV, "1")

        def boom(self, *a, **k):
            raise RuntimeError("report bug")

        monkeypatch.setattr(prefill_profile.PrefillProfiler, "_format_line", boom)
        _run()
        assert prefill_profile.active() is None


class TestActiveHandle:
    def test_none_after_profiled_generation(self, monkeypatch):
        monkeypatch.setenv(ENV, "1")
        _run()
        assert prefill_profile.active() is None

    def test_active_inside_chunk_and_none_after_exception(self, monkeypatch):
        monkeypatch.setenv(ENV, "1")
        lm = _tiny_lm()
        inside = []
        real_call = lm.__class__.__call__
        n = {"calls": 0}

        def failing_call(self, *args, **kwargs):
            n["calls"] += 1
            inside.append(prefill_profile.active())
            if n["calls"] == 2:
                raise ValueError("chunk failure")
            return real_call(self, *args, **kwargs)

        monkeypatch.setattr(lm.__class__, "__call__", failing_call)
        with pytest.raises(ValueError, match="chunk failure"):
            _run(lm)
        assert inside[0] is not None and inside[1] is not None
        assert prefill_profile.active() is None
