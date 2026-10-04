"""Env-gated prefill component profiler (M57 step 2, 2026-10-04).

CPU-pinned, no checkpoint: a tiny random-weight ``qwen3_5`` language model
(one GatedDeltaNet layer + one full-attention layer) is driven through the real
``generate_step`` chunked-prefill loop.
"""

import re
import threading
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm import prefill_profile
from mlx_vlm.generate import ar as ar_module
from mlx_vlm.generate import common as common_module
from mlx_vlm.speculative import mtp_profile
from mlx_vlm.models.qwen3_5 import language as qwen_language
from mlx_vlm.models.qwen3_5.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.qwen3_5.language import (
    LanguageModel,
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
)

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
_FIELDS = (
    r"chunks=(?P<chunks>\d+) tokens=(?P<tokens>\d+) "
    r"keys=(?P<keys>\d+) wall=(?P<wall>[-\d.]+) sdpa=(?P<sdpa>[-\d.]+) "
    r"kv_update=(?P<kv_update>[-\d.]+) attn_prep=(?P<attn_prep>[-\d.]+) "
    r"attn_out=(?P<attn_out>[-\d.]+) gdn=(?P<gdn>[-\d.]+) mlp=(?P<mlp>[-\d.]+) "
    r"cache_post=(?P<cache_post>[-\d.]+) clear_cache=(?P<clear_cache>[-\d.]+) "
    r"other=(?P<other>[-\d.]+) final=(?P<final>[01])(?P<broken> broken=1)?$"
)
LINE_RE = re.compile(r"^\[prefill_profile\] " + _FIELDS)
TOTAL_RE = re.compile(r"^\[prefill_profile_total\] " + _FIELDS)
# 13 prompt tokens, step 4: three chunks of 4 (12 tokens), last token via _step.
PROMPT = list(range(1, 14))
STEP = 4
CPU_CALLS = []


@pytest.fixture(autouse=True)
def _cpu_device():
    """Bind everything generation uses to the CPU, and PROVE it inside the call.

    ``_get_generation_stream`` caches a per-thread stream on first use; a stream
    cached by an earlier test under the GPU default device would put
    ``mx.stream(...)`` back on the GPU. Reset the cache around each test, and
    wrap the model call to assert CPU placement of device + stream.
    """
    previous_device = mx.default_device()
    tls = common_module._thread_local_streams
    previous_stream = getattr(tls, "stream", None)
    mx.set_default_device(mx.cpu)
    tls.stream = None
    real_call = LanguageModel.__call__
    CPU_CALLS.clear()

    def checked_call(self, *args, **kwargs):
        stream = mx.default_stream(mx.default_device())
        assert mx.default_device().type == mx.cpu
        assert stream.device.type == mx.cpu
        assert ar_module._get_generation_stream().device.type == mx.cpu
        CPU_CALLS.append(1)
        return real_call(self, *args, **kwargs)

    LanguageModel.__call__ = checked_call
    try:
        yield
    finally:
        LanguageModel.__call__ = real_call
        tls.stream = previous_stream
        mx.set_default_device(previous_device)


def _tiny_lm(layers=3):
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
        num_hidden_layers=layers,
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


def _run(lm=None, layers=3):
    """One generate_step prefill + first token. Returns (token, logprobs, states)."""
    lm = lm if lm is not None else _tiny_lm(layers)
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


def _err_lines(capsys):
    return capsys.readouterr().err.splitlines()


def _lines(capsys):
    """Window/final lines (not the generation total line)."""
    return [l for l in _err_lines(capsys) if l.startswith("[prefill_profile]")]


def _count_sync(monkeypatch):
    real = mx.synchronize
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(mx, "synchronize", counting)
    return calls


class _Clock:
    """Fake monotonic clock shared by the profiler and its timer base."""

    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _drive(prof, clock, chunks):
    """Synthetic chunks: [(tokens, keys, {phase: seconds})]. Time is fake; the
    mx.eval / mx.synchronize fences are real (CPU)."""
    try:
        for tokens, keys, phases in chunks:
            prof.begin_chunk(tokens)
            for phase, seconds in phases.items():
                clock.t += seconds
                prof.safe_mark(phase)
            clock.t += 0.001  # un-fenced tail -> lands in `other`
            prof.end_chunk(keys)
    finally:
        prof.clear_active()


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(prefill_profile, "perf_counter", c)
    monkeypatch.setattr(mtp_profile, "perf_counter", c)
    return c


class TestSwitchUnset:
    def test_zero_synchronize_no_line_active_none(self, monkeypatch, capsys):
        monkeypatch.delenv(ENV, raising=False)
        calls = _count_sync(monkeypatch)
        _run()
        assert calls["n"] == 0
        assert _err_lines(capsys) == [] or not any(
            l.startswith("[prefill_profile") for l in _err_lines(capsys)
        )
        assert prefill_profile.active() is None
        assert prefill_profile.active_layer() is None
        assert prefill_profile.from_env() is None
        assert CPU_CALLS  # the model really ran under the CPU assertions


class TestSwitchSet:
    def test_three_chunk_run_emits_final_and_total_lines(self, monkeypatch, capsys):
        monkeypatch.setenv(ENV, "1")
        _run()
        assert CPU_CALLS
        err = _err_lines(capsys)
        lines = [l for l in err if l.startswith("[prefill_profile]")]
        totals = [l for l in err if l.startswith("[prefill_profile_total]")]
        assert len(lines) == 1 and len(totals) == 1, err
        for text, rx in ((lines[0], LINE_RE), (totals[0], TOTAL_RE)):
            m = rx.match(text)
            assert m is not None, text
            f = m.groupdict()
            assert f["final"] == "1" and f["broken"] is None
            assert (f["chunks"], f["tokens"], f["keys"]) == ("3", "12", "12")
            for key in ("wall",) + PHASES:
                value = float(f[key])
                assert value == value and value != float("inf")
                assert value >= 0.0, (key, value)
            named = sum(float(f[k]) for k in PHASES if k != "other")
            assert float(f["wall"]) >= named - 1e-6

    def test_window_lines_cover_exactly_their_chunks_then_remainder(
        self, monkeypatch, capsys
    ):
        monkeypatch.setenv(ENV, "1")
        monkeypatch.setattr(prefill_profile.PrefillProfiler, "every", 2)
        _run()
        err = _err_lines(capsys)
        lines = [l for l in err if l.startswith("[prefill_profile]")]
        assert [l.endswith("final=0") for l in lines] == [True, False]
        w, rest = LINE_RE.match(lines[0]), LINE_RE.match(lines[1])
        assert (w["chunks"], w["tokens"], w["keys"]) == ("2", "8", "8")
        assert (rest["chunks"], rest["tokens"], rest["keys"]) == ("1", "4", "12")
        total = TOTAL_RE.match([l for l in err if "_total]" in l][0])
        assert (total["chunks"], total["tokens"], total["keys"]) == ("3", "12", "12")

    def test_no_final_line_when_no_chunks_remain(self, monkeypatch, capsys):
        monkeypatch.setenv(ENV, "1")
        monkeypatch.setattr(prefill_profile.PrefillProfiler, "every", 3)
        _run()
        err = _err_lines(capsys)
        lines = [l for l in err if l.startswith("[prefill_profile]")]
        assert len(lines) == 1 and lines[0].endswith("final=0")
        assert any(l.startswith("[prefill_profile_total]") for l in err)

    @pytest.mark.parametrize(
        "layers,expected",
        [
            # layers: GDN, attention, GDN(terminal). Terminal mlp is dead -> fence.
            (3, {"gdn": 6, "mlp": 6, "attn_prep": 3, "kv_update": 3, "sdpa": 3,
                 "attn_out": 3}),
            # layers: GDN, attention(terminal): its sdpa/attn_out/mlp are dead.
            (2, {"gdn": 3, "mlp": 3, "attn_prep": 3, "kv_update": 3, "sdpa": 0,
                 "attn_out": 0}),
        ],
    )
    def test_hooks_fire_per_chunk_and_layer_skipping_dead_terminal_work(
        self, monkeypatch, capsys, layers, expected
    ):
        monkeypatch.setenv(ENV, "1")
        seen = []
        real = prefill_profile.PrefillProfiler.mark

        def spy(self, phase, *outputs):
            seen.append(phase)
            return real(self, phase, *outputs)

        monkeypatch.setattr(prefill_profile.PrefillProfiler, "mark", spy)
        _run(layers=layers)
        expected = {**expected, "cache_post": 3, "clear_cache": 3}
        for phase, n in expected.items():
            assert seen.count(phase) == n, (phase, seen)

    def test_other_model_family_reports_chunk_level_phases_only(
        self, monkeypatch, capsys
    ):
        monkeypatch.setenv(ENV, "1")
        seen = []
        real_mark = prefill_profile.PrefillProfiler.mark
        monkeypatch.setattr(
            prefill_profile.PrefillProfiler,
            "mark",
            lambda self, phase, *o: (seen.append(phase), real_mark(self, phase, *o))[1],
        )
        # Hide the handle from layer code == a model family with no layer hooks.
        monkeypatch.setattr(prefill_profile, "active", lambda: None)
        monkeypatch.setattr(prefill_profile, "active_layer", lambda: None)
        _run()
        assert set(seen) == {"other_fence", "cache_post", "clear_cache"}
        m = LINE_RE.match(_lines(capsys)[-1])
        assert m is not None and m["chunks"] == "3"
        for key in ("sdpa", "kv_update", "attn_prep", "attn_out", "gdn", "mlp"):
            assert float(m[key]) == 0.0


class TestLayerReuseOutsideDecoderLayer:
    def test_attention_reused_in_another_layer_class_reports_chunk_phases_only(
        self, monkeypatch, capsys
    ):
        """qwen3_5_moe / qwen4_exp reuse Qwen3_5Attention in their own layer
        class; their attention marks must not fire (no gdn/mlp would pair)."""
        monkeypatch.setenv(ENV, "1")
        lm = _tiny_lm(2)
        args = lm.args

        class ForeignLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.is_linear = False
                self.self_attn = Qwen3_5Attention(args)

            def __call__(self, x, mask=None, cache=None, position_ids=None,
                         position_embeddings=None):
                return x + self.self_attn(
                    x,
                    mask=mask,
                    cache=cache,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )

        lm.model.layers = [ForeignLayer(), ForeignLayer()]
        lm.model.ssm_idx = 0
        lm.model.fa_idx = 1
        # no recurrent layer in this fake: skip the SSM mask (KVCache has none)
        monkeypatch.setattr(
            qwen_language, "_create_qwen3_5_ssm_mask", lambda h, c: None
        )
        seen = []
        real_mark = prefill_profile.PrefillProfiler.mark
        monkeypatch.setattr(
            prefill_profile.PrefillProfiler,
            "mark",
            lambda self, phase, *o: (seen.append(phase), real_mark(self, phase, *o))[1],
        )
        _run(lm)
        assert set(seen) == {"other_fence", "cache_post", "clear_cache"}, seen
        m = LINE_RE.match(_lines(capsys)[-1])
        assert m is not None and m["chunks"] == "3"
        for key in ("sdpa", "kv_update", "attn_prep", "attn_out", "gdn", "mlp"):
            assert float(m[key]) == 0.0


class TestOnlyProductionWorkIsEvaluated:
    @pytest.mark.parametrize("layers", [2, 3])
    def test_chunk_output_and_terminal_layer_output_never_evaluated(
        self, monkeypatch, layers
    ):
        monkeypatch.setenv(ENV, "1")
        lm = _tiny_lm(layers)
        keep = []  # hold references so ids cannot be recycled
        evaluated = set()
        real_eval = mx.eval

        def recording_eval(*args, **kwargs):
            found = []
            for a in args:
                mtp_profile._collect(a, found)
            keep.extend(found)
            evaluated.update(id(x) for x in found)
            return real_eval(*args, **kwargs)

        monkeypatch.setattr(mx, "eval", recording_eval)

        layer_outputs = {}
        real_layer_call = Qwen3_5DecoderLayer.__call__

        def layer_call(self, *a, **k):
            out = real_layer_call(self, *a, **k)
            layer_outputs.setdefault(self.m57_terminal, []).append(out)
            return out

        monkeypatch.setattr(Qwen3_5DecoderLayer, "__call__", layer_call)
        chunk_outputs = []
        real_lm_call = LanguageModel.__call__

        def lm_call(self, *a, **k):
            out = real_lm_call(self, *a, **k)
            chunk_outputs.append(out)
            return out

        monkeypatch.setattr(LanguageModel, "__call__", lm_call)

        _run(lm)
        # calls 4+ are the unchunked _step decode calls (not profiled)
        chunk_outputs = chunk_outputs[:3]
        assert len(chunk_outputs) == 3 and len(layer_outputs[True]) >= 3
        # sentinel: dead in production, so never directly evaluated
        for out in chunk_outputs:
            assert id(out.logits) not in evaluated
        for out in layer_outputs[True][:3]:
            assert id(out) not in evaluated
        # positive control: a live (non-terminal) layer output IS evaluated
        assert id(layer_outputs[False][0]) in evaluated


class TestOutputsUnchanged:
    @pytest.mark.parametrize("layers", [2, 3])
    def test_logits_and_cache_state_identical_with_switch_set(
        self, monkeypatch, layers
    ):
        monkeypatch.delenv(ENV, raising=False)
        y0, lp0, s0 = _run(layers=layers)
        monkeypatch.setenv(ENV, "1")
        y1, lp1, s1 = _run(layers=layers)
        assert mx.array_equal(y0, y1).item()
        assert mx.array_equal(lp0, lp1).item()
        assert len(s0) == len(s1) > 0
        for a, b in zip(s0, s1):
            assert a.shape == b.shape
            assert mx.array_equal(a, b).item()


class TestRaisingProfiler:
    def test_raising_mark_does_not_break_generation_and_clears_handle(
        self, monkeypatch, capsys
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
        assert not any(l.startswith("[prefill_profile") for l in _err_lines(capsys))

    def test_mark_failing_after_a_recorded_chunk_prints_broken_never_final(
        self, monkeypatch, capsys
    ):
        monkeypatch.delenv(ENV, raising=False)
        y0, lp0, _ = _run()
        capsys.readouterr()
        monkeypatch.setenv(ENV, "1")
        real = prefill_profile.PrefillProfiler.mark

        def flaky(self, phase, *outputs):
            if self.units >= 1:
                raise RuntimeError("late profiler bug")
            return real(self, phase, *outputs)

        monkeypatch.setattr(prefill_profile.PrefillProfiler, "mark", flaky)
        y1, lp1, _ = _run()
        assert mx.array_equal(y0, y1).item() and mx.array_equal(lp0, lp1).item()
        assert prefill_profile.active() is None
        err = _err_lines(capsys)
        pp = [l for l in err if l.startswith("[prefill_profile")]
        assert pp, err
        assert not any("final=1" in l for l in pp), pp
        for l in pp:
            assert l.endswith("final=0 broken=1"), l
        first = LINE_RE.match([l for l in pp if "_total]" not in l][0])
        assert first["chunks"] == "1"  # only the complete chunk is reported

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
        assert prefill_profile.active_layer() is None

    def test_active_inside_chunk_and_none_after_exception_which_is_not_masked(
        self, monkeypatch, capsys
    ):
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
        # finalisation ran from the outer finally: partial set -> never final=1
        pp = [l for l in _err_lines(capsys) if l.startswith("[prefill_profile")]
        assert pp and not any("final=1" in l for l in pp), pp

    def test_handle_is_thread_local_and_owned(self):
        prof = prefill_profile.PrefillProfiler()
        seen = {}

        def other_thread():
            seen["active"] = prefill_profile.active()
            seen["layer"] = prefill_profile.active_layer()

        prof.begin_chunk(4)
        assert prefill_profile.active() is prof
        t = threading.Thread(target=other_thread)
        t.start()
        t.join()
        assert seen == {"active": None, "layer": None}
        # a different profiler cannot clear this one's handle
        prefill_profile.PrefillProfiler().clear_active()
        assert prefill_profile.active() is prof
        prof.clear_active()
        assert prefill_profile.active() is None

    def test_two_overlapping_profiled_generations_stay_separate(
        self, monkeypatch, capsys
    ):
        monkeypatch.setenv(ENV, "1")
        lms = [_tiny_lm(), _tiny_lm()]
        barrier = threading.Barrier(2, timeout=20)
        owners = {}
        real_call = LanguageModel.__call__

        def call(self, *a, **k):
            tid = threading.get_ident()
            owners.setdefault(tid, set()).add(id(prefill_profile.active()))
            if len(owners[tid]) == 1 and not getattr(self, "_waited", False):
                self._waited = True
                barrier.wait()  # both threads are inside a profiled chunk now
            return real_call(self, *a, **k)

        monkeypatch.setattr(LanguageModel, "__call__", call)
        errors = []

        def work(lm):
            try:
                _run(lm)
            except BaseException as e:  # pragma: no cover - surfaced below
                errors.append(e)

        threads = [threading.Thread(target=work, args=(lm,)) for lm in lms]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        assert not errors, errors
        assert len(owners) == 2
        # each thread saw exactly one (its own) profiler inside chunks, and None
        # on the final unchunked _step call
        mine = [ids - {id(None)} for ids in owners.values()]
        assert all(len(ids) == 1 for ids in mine), owners
        assert mine[0] != mine[1]
        err = _err_lines(capsys)
        finals = [l for l in err if l.endswith("final=1")]
        assert len(finals) == 4  # 2 generations x (remaining + total)
        for l in finals:
            assert ("chunks=3 tokens=12 keys=12 ") in l


class TestReportSemantics:
    def test_unequal_durations_across_windows(self, monkeypatch, capsys, clock):
        monkeypatch.setattr(prefill_profile.PrefillProfiler, "every", 2)
        prof = prefill_profile.PrefillProfiler()
        sdpa = [0.010, 0.020, 0.030, 0.040, 0.050]
        gdn = [0.100, 0.100, 0.200, 0.200, 0.400]
        _drive(
            prof,
            clock,
            [
                (4 + i, 10 * (i + 1), {"sdpa": sdpa[i], "gdn": gdn[i]})
                for i in range(5)
            ],
        )
        prof.finish()
        err = _err_lines(capsys)
        lines = [l for l in err if l.startswith("[prefill_profile]")]
        assert len(lines) == 3
        w1, w2, rest = (LINE_RE.match(l) for l in lines)
        total = TOTAL_RE.match([l for l in err if "_total]" in l][0])

        def close(m, key, want):
            assert abs(float(m[key]) - want) < 0.02, (key, m[key], want)

        assert (w1["chunks"], w1["tokens"], w1["keys"], w1["final"]) == (
            "2", "9", "20", "0")
        close(w1, "sdpa", 15.0)
        close(w1, "gdn", 100.0)
        assert (w2["chunks"], w2["tokens"], w2["keys"], w2["final"]) == (
            "2", "13", "40", "0")
        close(w2, "sdpa", 35.0)
        close(w2, "gdn", 200.0)
        assert (rest["chunks"], rest["tokens"], rest["keys"], rest["final"]) == (
            "1", "8", "50", "1")
        close(rest, "sdpa", 50.0)
        close(rest, "gdn", 400.0)
        assert (total["chunks"], total["tokens"], total["keys"]) == ("5", "30", "50")
        close(total, "sdpa", 30.0)
        close(total, "gdn", 200.0)
        close(total, "wall", 231.0)
        close(total, "other", 1.0)  # the un-fenced 1 ms tail of every chunk
