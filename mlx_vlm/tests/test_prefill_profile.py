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
from mlx_vlm.models import cache as qwen_cache
from mlx_vlm.models.qwen3_5 import language as qwen_language
from mlx_vlm.models.qwen3_5.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.qwen3_5.language import (
    LanguageModel,
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
)
from mlx_vlm.speculative import mtp_profile

ENV = prefill_profile.ENV
PHASES = (
    "entry",
    "sdpa",
    "kv_update",
    "observe",
    "attn_prep",
    "attn_out",
    "gdn",
    "mlp",
    "forward",
    "cache_post",
    "clear_cache",
    "other",
)
_FIELDS = (
    r"chunks=(?P<chunks>\d+) tokens=(?P<tokens>\d+) keys=(?P<keys>\d+) "
    r"layers=(?P<layers>[-\d.]+) wall=(?P<wall>[-\d.]+) "
    r"entry=(?P<entry>[-\d.]+) sdpa=(?P<sdpa>[-\d.]+) "
    r"kv_update=(?P<kv_update>[-\d.]+) observe=(?P<observe>[-\d.]+) "
    r"attn_prep=(?P<attn_prep>[-\d.]+) attn_out=(?P<attn_out>[-\d.]+) "
    r"gdn=(?P<gdn>[-\d.]+) mlp=(?P<mlp>[-\d.]+) "
    r"forward=(?P<forward>[-\d.]+) cache_post=(?P<cache_post>[-\d.]+) "
    r"clear_cache=(?P<clear_cache>[-\d.]+) other=(?P<other>[-\d.]+) "
    r"final=(?P<final>[01])(?P<broken> broken=1)?$"
)
LINE_RE = re.compile(r"^\[prefill_profile\] " + _FIELDS)
TOTAL_RE = re.compile(r"^\[prefill_profile_total\] " + _FIELDS)
BROKEN0_RE = re.compile(
    r"^\[prefill_profile\] chunks=0 broken=1 reason=(?P<reason>.+)$"
)
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
        config=ModelConfig(text_config=cfg, vision_config=vision, model_type="qwen3_5"),
    )
    mx.eval(lm.parameters())
    return lm


def _fake_model(lm):
    def get_input_embeddings(input_ids, pixel_values, mask=None, **kwargs):
        return SimpleNamespace(
            inputs_embeds=lm.model.embed_tokens(input_ids), to_dict=lambda: {}
        )

    return SimpleNamespace(language_model=lm, get_input_embeddings=get_input_embeddings)


def _run(lm=None, layers=3, prompt_cache=None, **gen_kwargs):
    """One generate_step prefill + first token. Returns (token, logprobs, states)."""
    lm = lm if lm is not None else _tiny_lm(layers)
    prompt_cache = prompt_cache if prompt_cache is not None else lm.make_cache()
    gen = ar_module.generate_step(
        input_ids=mx.array([PROMPT], dtype=mx.int32),
        model=_fake_model(lm),
        pixel_values=None,
        mask=None,
        max_tokens=1,
        temperature=0.0,
        prefill_step_size=STEP,
        prompt_cache=prompt_cache,
        **gen_kwargs,
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
        err = _err_lines(capsys)  # read ONCE: capsys.readouterr() drains
        assert not any(l.startswith("[prefill_profile") for l in err), err
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
            (
                3,
                {
                    "gdn": 6,
                    "mlp": 6,
                    "attn_prep": 3,
                    "kv_update": 3,
                    "sdpa": 3,
                    "attn_out": 3,
                },
            ),
            # layers: GDN, attention(terminal): its sdpa/attn_out/mlp are dead.
            (
                2,
                {
                    "gdn": 3,
                    "mlp": 3,
                    "attn_prep": 3,
                    "kv_update": 3,
                    "sdpa": 0,
                    "attn_out": 0,
                },
            ),
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
        expected = {
            **expected,
            "entry": 3,
            "forward": 0,  # layers declared themselves -> state is read at cache_post
            "cache_post": 3,
            "clear_cache": 3,
        }
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
        assert set(seen) == {"forward", "other_fence", "cache_post", "clear_cache"}
        m = LINE_RE.match(_lines(capsys)[-1])
        assert m is not None and m["chunks"] == "3"
        assert float(m["layers"]) == 0.0
        for key in ("entry", "sdpa", "kv_update", "attn_prep", "attn_out"):
            assert float(m[key]) == 0.0
        for key in ("gdn", "mlp"):
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

            def __call__(
                self,
                x,
                mask=None,
                cache=None,
                position_ids=None,
                position_embeddings=None,
            ):
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
        # Qwen3_5Model's own entry fence still fires; no decoder layer declares
        # itself, so the whole forward is evaluated and reported as `forward`.
        assert set(seen) == {
            "entry",
            "forward",
            "other_fence",
            "cache_post",
            "clear_cache",
        }, seen
        m = LINE_RE.match(_lines(capsys)[-1])
        assert m is not None and m["chunks"] == "3"
        assert float(m["layers"]) == 0.0
        for key in ("sdpa", "kv_update", "attn_prep", "attn_out", "gdn", "mlp"):
            assert float(m[key]) == 0.0
        assert float(m["forward"]) > 0.0


# --- execution-sensitive sentinels ------------------------------------------
# MLX is lazy and exposes no "was this evaluated" flag, so a sentinel wraps a
# module's output with `+ ones(BIG).sum() * 0`: evaluating it allocates ~134 MB,
# which shows up in mx.get_peak_memory() (verified on the CPU stream); leaving it
# unevaluated costs nothing. Sentinels are armed only inside a profiled chunk,
# so the unchunked _step decode calls never trip them.
BIG = 2**25
THRESH = 100e6


def _sentinel_add(out):
    return out + (mx.ones((BIG,)).sum() * 0).astype(out.dtype)


class _Sentinel(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __call__(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        if prefill_profile.active() is None:
            return out
        return _sentinel_add(out)


def _wrap_terminal(lm, which):
    term = lm.model.layers[-1]
    if which == "proj":
        if term.is_linear:
            term.linear_attn.out_proj = _Sentinel(term.linear_attn.out_proj)
        else:
            term.self_attn.o_proj = _Sentinel(term.self_attn.o_proj)
    elif which == "q":
        term.self_attn.q_proj = _Sentinel(term.self_attn.q_proj)
    elif which == "mlp":
        term.mlp = _Sentinel(term.mlp)
    elif which == "norm":
        lm.model.norm = _Sentinel(lm.model.norm)
    elif which == "head":
        lm.lm_head = _Sentinel(lm.lm_head)
    else:  # pragma: no cover
        raise AssertionError(which)


def _peak_of(fn):
    mx.eval(mx.zeros(1))
    mx.reset_peak_memory()
    fn()
    return mx.get_peak_memory()


class TestSentinelSelfCheck:
    def test_sentinel_fires_when_consumed_only_inside_a_profiled_chunk(self):
        sentinel = _Sentinel(nn.Identity())
        x = mx.ones((2, 3))
        assert _peak_of(lambda: mx.eval(sentinel(x))) < THRESH  # not armed
        prof = prefill_profile.PrefillProfiler()
        prof.begin_chunk(1)
        try:
            lazy = sentinel(x)
            assert _peak_of(lambda: None) < THRESH  # built but not consumed
            assert _peak_of(lambda: mx.eval(lazy)) >= THRESH  # consumed
        finally:
            prof.clear_active()


class TestDeadWorkNeverExecuted:
    @pytest.mark.parametrize(
        "layers,which",
        [
            (2, "proj"),
            (2, "mlp"),
            (2, "norm"),
            (2, "head"),
            (3, "proj"),
            (3, "mlp"),
            (3, "norm"),
            (3, "head"),
        ],
    )
    def test_terminal_layer_final_norm_and_head_stay_unevaluated(
        self, monkeypatch, layers, which
    ):
        monkeypatch.setenv(ENV, "1")
        lm = _tiny_lm(layers)
        _wrap_terminal(lm, which)
        assert _peak_of(lambda: _run(lm)) < THRESH

    @pytest.mark.parametrize(
        "layers,which", [(2, "proj"), (2, "mlp"), (3, "proj"), (3, "mlp")]
    )
    def test_a_profiler_that_evaluates_dead_work_trips_the_sentinel(
        self, monkeypatch, layers, which
    ):
        """The reviewer mutation: layer_mark ignores terminal-ness."""
        monkeypatch.setenv(ENV, "1")

        def mutated(self, phase, *outputs, live_if_nonterminal=()):
            self.safe_mark(phase, *outputs, *live_if_nonterminal)

        monkeypatch.setattr(prefill_profile.PrefillProfiler, "layer_mark", mutated)
        lm = _tiny_lm(layers)
        _wrap_terminal(lm, which)
        assert _peak_of(lambda: _run(lm)) >= THRESH

    @pytest.mark.parametrize(
        "live_kwargs",
        [
            {"return_hidden": True},
            {"capture_layer_ids": [0]},
            {"capture_layer_ids": []},
        ],
    )
    @pytest.mark.parametrize(
        "layers,which", [(2, "proj"), (2, "mlp"), (3, "proj"), (3, "mlp")]
    )
    def test_terminal_is_live_when_chunk_kwargs_capture_hidden_states(
        self, monkeypatch, live_kwargs, layers, which
    ):
        monkeypatch.setenv(ENV, "1")
        lm = _tiny_lm(layers)
        _wrap_terminal(lm, which)
        assert _peak_of(lambda: _run(lm, **live_kwargs)) >= THRESH


class TestTerminalProductionWorkIsEvaluated:
    def _spy(self, monkeypatch):
        seen = []
        real = prefill_profile.PrefillProfiler.mark

        def spy(self, phase, *outputs):
            before = mx.get_peak_memory()
            result = real(self, phase, *outputs)
            seen.append((phase, before, mx.get_peak_memory()))
            return result

        monkeypatch.setattr(prefill_profile.PrefillProfiler, "mark", spy)
        return seen

    def test_terminal_kv_update_is_evaluated_before_its_phase_closes(self, monkeypatch):
        monkeypatch.setenv(ENV, "1")
        seen = self._spy(monkeypatch)
        real = qwen_cache.KVCache.update_and_fetch

        def update(self, keys, values):
            k, v = real(self, keys, values)
            if prefill_profile.active() is not None:
                k = _sentinel_add(k)
            return k, v

        monkeypatch.setattr(qwen_cache.KVCache, "update_and_fetch", update)
        _peak_of(lambda: _run(layers=2))  # layer 1 (attention) is terminal
        first = [t for t in seen if t[0] == "kv_update"][0]
        assert first[1] < THRESH <= first[2], first

    def test_terminal_recurrent_state_is_in_the_gdn_mark(self, monkeypatch):
        """The GatedDeltaNet forward evaluates its own inputs while building (a
        peak-memory sentinel on its projections trips before any mark), so
        execution cannot separate the phases here; assert instead that the
        terminal layer's `gdn` mark receives exactly its stored cache state."""
        monkeypatch.setenv(ENV, "1")
        stored = []
        real_gdn = qwen_language.Qwen3_5GatedDeltaNet.__call__

        def gdn_call(self, inputs, mask=None, cache=None):
            out = real_gdn(self, inputs, mask, cache)
            stored.append({id(a) for a in _flat_state(cache)})
            return out

        monkeypatch.setattr(qwen_language.Qwen3_5GatedDeltaNet, "__call__", gdn_call)
        marked = []
        real = prefill_profile.PrefillProfiler.mark

        def spy(self, phase, *outputs):
            if phase == "gdn":
                found = []
                mtp_profile._collect(outputs, found)
                marked.append({id(a) for a in found})
            return real(self, phase, *outputs)

        monkeypatch.setattr(prefill_profile.PrefillProfiler, "mark", spy)
        keep = []
        lm = _tiny_lm(3)  # layer 2 (GatedDeltaNet) is terminal
        _run(lm)
        # calls alternate layer 0, terminal layer; chunk 1 is stored[0:2]
        assert stored[1] and stored[1] <= marked[1], (stored[1], marked[1])
        # and the terminal layer's output `r` is NOT in its mark (dead)
        assert len(marked[1]) == len(stored[1])
        # non-terminal layer 0 additionally closes on its output
        assert len(marked[0]) == len(stored[0]) + 1


class TestTerminalStateEvaluatedBeforeGdnCloses:
    """Execution-sensitive: after the real GatedDeltaNet call stores its state,
    the TERMINAL layer's stored arrays are replaced by value-identical arrays
    carrying a peak-memory sentinel (so projection-side evaluation inside the
    call cannot trip it). The sentinel must have fired by the time the `gdn`
    mark returns."""

    def _install(self, monkeypatch, layers=3):
        lm = _tiny_lm(layers)
        terminal = lm.model.layers[-1].linear_attn
        real = qwen_language.Qwen3_5GatedDeltaNet.__call__

        def call(self, inputs, mask=None, cache=None):
            out = real(self, inputs, mask, cache)
            if self is terminal and prefill_profile.active() is not None:
                cache.cache = [
                    None if a is None else _sentinel_add(a) for a in cache.cache
                ]
            return out

        monkeypatch.setattr(qwen_language.Qwen3_5GatedDeltaNet, "__call__", call)
        # (phase, peak while the mark ran, peak between the previous mark and it)
        peaks = []
        real_mark = prefill_profile.PrefillProfiler.mark

        def spy(self, phase, *outputs):
            between = mx.get_peak_memory()
            mx.reset_peak_memory()
            result = real_mark(self, phase, *outputs)
            peaks.append((phase, mx.get_peak_memory(), between))
            return result

        monkeypatch.setattr(prefill_profile.PrefillProfiler, "mark", spy)
        return lm, peaks

    def test_terminal_state_fires_by_the_time_gdn_returns(self, monkeypatch):
        monkeypatch.setenv(ENV, "1")
        lm, peaks = self._install(monkeypatch)
        _run(lm)
        gdn = [p for ph, p, _ in peaks if ph == "gdn"]
        assert len(gdn) == 6  # per chunk: layer 0 (live, unwrapped), terminal
        for layer0, terminal in zip(gdn[0::2], gdn[1::2]):
            assert layer0 < THRESH  # positive control: unwrapped layer is quiet
            assert terminal >= THRESH  # state evaluated under `gdn`

    def test_mutation_skipping_state_evaluation_fails_that_assertion(self, monkeypatch):
        monkeypatch.setenv(ENV, "1")
        lm, peaks = self._install(monkeypatch)
        real_layer_mark = prefill_profile.PrefillProfiler.layer_mark

        def mutated(self, phase, *outputs, live_if_nonterminal=()):
            if phase == "gdn":
                return self.safe_mark(phase)  # skip evaluation
            return real_layer_mark(
                self, phase, *outputs, live_if_nonterminal=live_if_nonterminal
            )

        monkeypatch.setattr(prefill_profile.PrefillProfiler, "layer_mark", mutated)
        _run(lm)
        gdn = [p for ph, p, _ in peaks if ph == "gdn"]
        assert all(p < THRESH for p in gdn)  # the assertion above would fail
        # the state is only evaluated later: between the last layer mark and
        # the cache_post mark (the loop's mx.eval of the cache state)
        assert any(b >= THRESH for ph, _, b in peaks if ph == "cache_post")


class TestTerminalIsLastExecutedLayer:
    def test_profiler_terminal_logic(self):
        prof = prefill_profile.PrefillProfiler()
        try:
            prof.begin_chunk(4)
            prof.begin_layers(2, live_terminal=False)
            prof.enter_layer()
            assert prof.terminal is False
            prof.enter_layer()
            assert prof.terminal is True
            prof.begin_chunk(4)
            prof.begin_layers(2, live_terminal=True)
            prof.enter_layer()
            prof.enter_layer()
            assert prof.terminal is False
            prof.begin_chunk(4)  # model never declared its layer count
            prof.enter_layer()
            prof.enter_layer()
            assert prof.terminal is False
        finally:
            prof.clear_active()

    def test_model_declares_executed_layer_count_not_a_stored_index(self, monkeypatch):
        monkeypatch.setenv(ENV, "1")
        calls = []
        real = prefill_profile.PrefillProfiler.begin_layers

        def spy(self, n_exec, live_terminal=False):
            calls.append((n_exec, live_terminal))
            return real(self, n_exec, live_terminal)

        monkeypatch.setattr(prefill_profile.PrefillProfiler, "begin_layers", spy)
        _run(layers=3)
        assert calls == [(3, False)] * 3
        calls.clear()
        _run(layers=3, capture_layer_ids=[0])
        assert calls == [(3, True)] * 3


class TestEntryFence:
    def test_precomputed_position_embeddings_are_in_the_entry_fence(self, monkeypatch):
        monkeypatch.setenv(ENV, "1")
        lm = _tiny_lm(2)
        for layer in lm.model.layers:
            if not layer.is_linear:
                layer.self_attn.rotary_emb.fused_apply = False
        passed = []
        real_layer = Qwen3_5DecoderLayer.__call__

        def layer_call(self, x, *a, **k):
            passed.append(k.get("position_embeddings"))
            return real_layer(self, x, *a, **k)

        monkeypatch.setattr(Qwen3_5DecoderLayer, "__call__", layer_call)
        entry_ids = []
        real_mark = prefill_profile.PrefillProfiler.mark

        def spy(self, phase, *outputs):
            if phase == "entry":
                found = []
                mtp_profile._collect(outputs, found)
                entry_ids.extend(id(x) for x in found)
            return real_mark(self, phase, *outputs)

        monkeypatch.setattr(prefill_profile.PrefillProfiler, "mark", spy)
        keep = []
        _run(lm)
        pe = passed[0]
        assert pe is not None, "non-fused rotary path did not build embeddings"
        found = []
        mtp_profile._collect(pe, found)
        keep.extend(found)
        assert found and all(id(x) in entry_ids for x in found)

    def test_entry_is_its_own_field_and_not_in_other(self, monkeypatch, capsys, clock):
        prof = prefill_profile.PrefillProfiler()
        _drive(
            prof,
            clock,
            [(4, 4, {"entry": 0.050, "sdpa": 0.010}), (4, 8, {"entry": 0.030})],
        )
        prof.finish()
        lines = [l for l in _err_lines(capsys) if l.startswith("[prefill_profile]")]
        m = LINE_RE.match(lines[-1])
        assert abs(float(m["entry"]) - 40.0) < 0.02
        assert abs(float(m["other"]) - 1.0) < 0.02  # only the un-fenced 1 ms tail


class TestForwardAndLayers:
    @pytest.mark.parametrize("layers", [2, 3])
    def test_layers_field_counts_declared_decoder_layers_per_chunk(
        self, monkeypatch, capsys, layers
    ):
        monkeypatch.setenv(ENV, "1")
        _run(layers=layers)
        m = LINE_RE.match(_lines(capsys)[-1])
        assert float(m["layers"]) == layers
        assert float(m["forward"]) == 0.0

    def test_forward_replaces_cache_post_when_no_layer_declared_itself(
        self, monkeypatch, capsys
    ):
        monkeypatch.setenv(ENV, "1")
        monkeypatch.setattr(
            Qwen3_5DecoderLayer, "__call__", _undeclared_layer_call, raising=True
        )
        seen = {}
        real = prefill_profile.PrefillProfiler.mark

        def spy(self, phase, *outputs):
            found = []
            mtp_profile._collect(outputs, found)
            seen.setdefault(phase, []).append(len(found))
            return real(self, phase, *outputs)

        monkeypatch.setattr(prefill_profile.PrefillProfiler, "mark", spy)
        _run(layers=3)
        # the cache state is evaluated at `forward` (right after the model call)
        assert len(seen["forward"]) == 3 and all(n > 0 for n in seen["forward"])
        m = LINE_RE.match(_lines(capsys)[-1])
        assert float(m["layers"]) == 0.0 and float(m["forward"]) > 0.0


def _undeclared_layer_call(
    self, x, mask=None, cache=None, position_ids=None, position_embeddings=None
):
    """Qwen3_5DecoderLayer.__call__ minus every profiler hook (a foreign layer)."""
    if self.is_linear:
        r = self.linear_attn(self.input_layernorm(x), mask, cache)
    else:
        r = self.self_attn(
            self.input_layernorm(x),
            mask=mask,
            cache=cache,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
    h = x + r
    return h + self.mlp(self.post_attention_layernorm(h))


class TestObserve:
    def _epi(self, monkeypatch):
        monkeypatch.setenv("MLX_EPICACHE_BUDGET", "4")
        monkeypatch.setenv("MLX_EPICACHE_BLOCK", "2")

    def test_observe_hook_is_closed_on_its_own_arrays_and_reported(
        self, monkeypatch, capsys
    ):
        monkeypatch.setenv(ENV, "1")
        self._epi(monkeypatch)
        seen = []
        real = prefill_profile.PrefillProfiler.mark

        def spy(self, phase, *outputs):
            if phase == "observe":
                found = []
                mtp_profile._collect(outputs, found)
                seen.append(found)
            return real(self, phase, *outputs)

        monkeypatch.setattr(prefill_profile.PrefillProfiler, "mark", spy)
        _run(layers=3)
        assert seen, "observe never marked"
        assert all(len(found) == 1 and found[0].ndim == 1 for found in seen)
        m = LINE_RE.match(_lines(capsys)[-1])
        assert m is not None and float(m["observe"]) >= 0.0

    def test_terminal_queries_are_live_when_observe_consumes_them(self, monkeypatch):
        """Terminal attn_prep marks (keys, values) only; with an observing cache
        the queries are live too. (q_proj cannot be a peak sentinel: the fused
        rotary op consumes queries and keys together, so keys already pull it.)"""

        def terminal_attn_prep_counts():
            counts = []
            real = prefill_profile.PrefillProfiler.mark

            def spy(self, phase, *outputs):
                if phase == "attn_prep":
                    found = []
                    mtp_profile._collect(outputs, found)
                    counts.append(len(found))
                return real(self, phase, *outputs)

            monkeypatch.setattr(prefill_profile.PrefillProfiler, "mark", spy)
            _run(layers=2)  # layer 1 (attention) is terminal
            monkeypatch.setattr(prefill_profile.PrefillProfiler, "mark", real)
            return counts

        monkeypatch.setenv(ENV, "1")
        assert terminal_attn_prep_counts() == [2, 2, 2]  # keys, values
        self._epi(monkeypatch)
        # chunk 1: offset 0 + 4 <= budget 4; chunks 2, 3 observe -> + queries
        assert terminal_attn_prep_counts() == [2, 3, 3]


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
        pp = [l for l in _err_lines(capsys) if l.startswith("[prefill_profile")]
        assert len(pp) == 1, pp
        m = BROKEN0_RE.match(pp[0])
        assert m is not None and "profiler bug" in m["reason"], pp

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

    def test_exception_in_first_chunk_prints_chunks_zero_broken_line(
        self, monkeypatch, capsys
    ):
        monkeypatch.setenv(ENV, "1")
        lm = _tiny_lm()
        real_call = lm.__class__.__call__

        def failing_call(self, *args, **kwargs):
            raise ValueError("first chunk failure")

        monkeypatch.setattr(lm.__class__, "__call__", failing_call)
        with pytest.raises(ValueError, match="first chunk failure"):
            _run(lm)
        pp = [l for l in _err_lines(capsys) if l.startswith("[prefill_profile")]
        assert len(pp) == 1 and BROKEN0_RE.match(pp[0]), pp

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
            [(4 + i, 10 * (i + 1), {"sdpa": sdpa[i], "gdn": gdn[i]}) for i in range(5)],
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
            "2",
            "9",
            "20",
            "0",
        )
        close(w1, "sdpa", 15.0)
        close(w1, "gdn", 100.0)
        assert (w2["chunks"], w2["tokens"], w2["keys"], w2["final"]) == (
            "2",
            "13",
            "40",
            "0",
        )
        close(w2, "sdpa", 35.0)
        close(w2, "gdn", 200.0)
        assert (rest["chunks"], rest["tokens"], rest["keys"], rest["final"]) == (
            "1",
            "8",
            "50",
            "1",
        )
        close(rest, "sdpa", 50.0)
        close(rest, "gdn", 400.0)
        assert (total["chunks"], total["tokens"], total["keys"]) == ("5", "30", "50")
        close(total, "sdpa", 30.0)
        close(total, "gdn", 200.0)
        close(total, "wall", 231.0)
        close(total, "other", 1.0)  # the un-fenced 1 ms tail of every chunk
