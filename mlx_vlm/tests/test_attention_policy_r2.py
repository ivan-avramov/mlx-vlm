"""M57 review round 2 (Amendment 2): G1-G4. CPU only."""

import ast
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient
from test_attention_policy import (  # noqa: F401  (fixtures + helpers)
    NativeCache,
    Recorder,
    _cpu_device,
    _q,
    _tiny_lm,
    gpu,
    passthrough,
    sdpa,
)

import mlx_vlm.server as server
import mlx_vlm.server.openai as openai_module
from mlx_vlm import attention_policy as ap
from mlx_vlm.server import cli as cli_module


# ---------------------------------------------------------------------- G1
class _Metrics:
    rate = 12.5
    sdpa_forced = 3
    sdpa_auto = 2


class TestG1EveryTerminalShape:
    def test_g1_one_helper_builds_the_timings(self):
        timings = openai_module._streaming_timings(12.5, _Metrics())
        dumped = json.loads(timings.model_dump_json())
        assert dumped == {
            "predicted_per_second": 12.5,
            "sdpa_forced": 3,
            "sdpa_auto": 2,
        }
        bare = openai_module._streaming_timings(
            1.0, SimpleNamespace(sdpa_forced=None, sdpa_auto=None)
        )
        assert json.loads(bare.model_dump_json()) == {"predicted_per_second": 1.0}

    def test_g1_every_terminal_chunk_shape_uses_the_helper_with_metrics(self):
        """Structural: covers the tool-call terminal AND its fallback construction
        (the fallback never sees counters through an endpoint test)."""
        tree = ast.parse(Path(openai_module.__file__).read_text())
        hand_built, tool_terminals = [], 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", "")
            if name == "StreamingTimings" and any(
                kw.arg == "predicted_per_second" and "metrics" in ast.dump(kw.value)
                for kw in node.keywords
            ):
                hand_built.append(node.lineno)
            if name == "ChatStreamChunk":
                for kw in node.keywords:
                    value = kw.value
                    if (
                        kw.arg == "timings"
                        and getattr(getattr(value, "func", None), "id", "")
                        == "_streaming_timings"
                        and len(value.args) == 2  # (rate, metrics)
                    ):
                        tool_terminals += 1
        assert not hand_built, hand_built
        # _final_chat_chunk + the streamed tool-call terminal + its fallback
        assert tool_terminals == 3

    def test_g1_completions_final_chunk_carries_counters_only_when_set(self):
        chunk = openai_module._completion_final_chunk("id", "m", 1, "stop", _Metrics())
        assert json.loads(chunk.to_sse_json())["timings"]["sdpa_forced"] == 3
        plain = openai_module._completion_final_chunk("id", "m", 1, "stop")
        assert "timings" not in json.loads(plain.to_sse_json())
        unset = openai_module._completion_final_chunk(
            "id", "m", 1, "stop", SimpleNamespace(sdpa_forced=None, sdpa_auto=None)
        )
        assert "timings" not in json.loads(unset.to_sse_json())


from test_server import _MuseResponseTemplateTokenizer  # noqa: E402


def _tool_stream(client, monkeypatch, forced, auto):
    from mlx_vlm.tools.parsers import atem as tool_module

    text = (
        "to=self<|message|>I need the weather tool.<|eom|>"
        "<|start|>assistant to=get_weather<|message|>"
        '<atem:function_calls><atem:invoke name="get_weather">'
        '<atem:parameter name="city">Warsaw</atem:parameter>'
        "</atem:invoke></atem:function_calls>"
    )

    class Gen:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, *a, **k):
            return None

        def generate(self, prompt, images=None, audio=None, args=None, **kw):
            return server.GenerationContext(uid=1, prompt_tokens=10), iter(
                [
                    server.StreamingToken(
                        text=text,
                        token=1,
                        logprobs=0.0,
                        finish_reason="stop",
                        prompt_tps=20.0,
                        cached_tokens=2,
                        sdpa_forced=forced,
                        sdpa_auto=auto,
                    )
                ]
            )

    monkeypatch.setattr(server.runtime, "response_generator", Gen())
    with (
        patch.object(
            server,
            "get_cached_model",
            return_value=(
                SimpleNamespace(),
                SimpleNamespace(tokenizer=_MuseResponseTemplateTokenizer()),
                SimpleNamespace(model_type="muse_glimmer"),
            ),
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "_infer_tool_parser_from_processor", return_value="demo"),
        patch.object(server, "load_tool_module", return_value=tool_module),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                "stream": True,  # NO stream_options: no usage chunk
            },
        )
    assert response.status_code == 200
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


class TestG1ToolCallStream:
    @pytest.fixture
    def client(self):
        with TestClient(server.app) as c:
            yield c

    def test_g1_streamed_tool_call_turn_without_usage_chunk_has_counters(
        self, client, monkeypatch
    ):
        chunks = _tool_stream(client, monkeypatch, 3, 2)
        assert not any(c.get("usage") for c in chunks)
        tool = [
            c
            for c in chunks
            if c["choices"] and c["choices"][0]["finish_reason"] == "tool_calls"
        ]
        assert len(tool) == 1
        t = tool[0]["timings"]
        assert (t["sdpa_forced"], t["sdpa_auto"]) == (3, 2)

    def test_g1_streamed_tool_call_turn_under_auto_has_no_counter_keys(
        self, client, monkeypatch
    ):
        chunks = _tool_stream(client, monkeypatch, None, None)
        for c in chunks:
            t = c.get("timings") or {}
            assert "sdpa_forced" not in t and "sdpa_auto" not in t


# ---------------------------------------------------------------------- G2
class TestG2ThreadLocalSuspension:
    def test_g2_exception_safe_and_nestable(self):
        policy = ap.resolve_policy("fused_v1")
        q = _q(512)
        with policy.suspended():
            with policy.suspended():
                assert not policy.should_force(q, 4096, None)
            assert not policy.should_force(q, 4096, None)  # outer still active
        assert policy.should_force(q, 4096, None)
        with pytest.raises(RuntimeError):
            with policy.suspended():
                raise RuntimeError("boom")
        assert policy.should_force(q, 4096, None)
        assert policy._suspended == 0

    def test_g2_second_thread_keeps_forcing_while_the_first_is_suspended(self):
        policy = ap.resolve_policy("fused_v1")
        q = _q(512)
        inside, release = threading.Event(), threading.Event()
        seen = {}

        def holder():
            with policy.suspended():
                seen["holder"] = policy.should_force(q, 4096, None)
                inside.set()
                release.wait(10)

        thread = threading.Thread(target=holder)
        thread.start()
        assert inside.wait(10)
        try:
            seen["other"] = policy.should_force(q, 4096, None)
        finally:
            release.set()
            thread.join(10)
        assert seen == {"holder": False, "other": True}
        assert policy.should_force(q, 4096, None)


# ---------------------------------------------------------------------- G3
class TestG3PoolLimit:
    def test_g3_fused_v1_derivation_equals_auto(self, monkeypatch):
        monkeypatch.setattr(cli_module, "_model_num_attention_heads", lambda p: 24)
        policy = ap.resolve_policy("fused_v1")
        for step in (512, 1024):
            auto = cli_module._derive_cache_limit_gb("m", 262144, step)
            assert (
                cli_module._derive_cache_limit_gb("m", 262144, step, policy=policy)
                == auto
            )
        assert cli_module._derive_cache_limit_gb("m", 262144, 512) == 9.0

    def test_g3_policy_no_longer_advertises_an_unfused_score_bound(self):
        assert not hasattr(ap.FusedV1Policy, "max_unfused_score_bytes")


# ---------------------------------------------------------------------- G4
def _mixed(lm, which):
    """A model whose WEIGHT dtype differs from the dtype queries are computed in."""
    attn = [m for _, m in lm.named_modules() if type(m).__name__ == "Qwen3_5Attention"]
    for m in attn:
        if which == "q_norm_fp32":  # everything bf16, q_norm weight fp32
            m.q_norm.weight = m.q_norm.weight.astype(mx.float32)
        elif which == "q_proj_fp32":  # q_proj weight fp32, q_norm bf16
            m.q_proj.weight = m.q_proj.weight.astype(mx.float32)
    mx.eval(lm.parameters())
    return lm


class TestG4Dtype:
    def test_g4_dtype_is_read_where_the_policy_decides(self, passthrough):
        lm = _tiny_lm()
        rec = Recorder()
        ap.self_test_model(
            lm,
            ap.resolve_policy("fused_v1"),
            max_kv=262144,
            sdpa=rec,
            force_calls=True,
        )
        assert rec.calls and all(c[0].dtype == mx.bfloat16 for c in rec.calls)
        # the probe left the model unstamped and unchanged
        assert all(
            getattr(m, "attention_policy", None) is None for _, m in lm.named_modules()
        )

    @pytest.mark.parametrize("which", ["q_norm_fp32", "q_proj_fp32"])
    def test_g4_activation_dtype_not_a_weight_dtype(self, passthrough, which):
        """bf16 q_proj/q_norm weights somewhere, fp32 queries in the attention:
        a weight-dtype stand-in would call bf16 and pass; the real dtype is fp32,
        which the policy never forces => zero forced calls => refused."""
        lm = _mixed(_tiny_lm(), which)
        with pytest.raises(ap.AttentionPolicyError, match="zero"):
            ap.self_test_model(
                lm,
                ap.resolve_policy("fused_v1"),
                max_kv=262144,
                sdpa=Recorder(),
                force_calls=True,
            )

    def test_g4_probe_error_is_the_one_line_attention_policy_failure(self, capsys):
        lm = _tiny_lm()

        class Broken:  # a variant without the expected structure
            def named_modules(self):
                return lm.named_modules()

            def make_cache(self):
                raise AttributeError("no make_cache")

        with pytest.raises(ap.AttentionPolicyError, match="dtype probe"):
            ap.self_test_model(
                Broken(),
                ap.resolve_policy("fused_v1"),
                max_kv=262144,
                sdpa=Recorder(),
                force_calls=True,
            )
        err = [l for l in capsys.readouterr().err.splitlines() if l.strip()]
        assert len(err) == 1 and err[0].startswith("attention-policy fused_v1:")
