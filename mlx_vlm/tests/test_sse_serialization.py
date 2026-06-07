"""ChatStreamChunk SSE serialization: the llama.cpp-style `timings` extension
key must be omitted when unset — OpenWebUI's stream parser crashes on an
explicit null (dict.update(None)) and silently drops every chunk."""

import json

from mlx_vlm.server.schemas import ChatMessage, ChatStreamChoice, ChatStreamChunk


def _chunk(**kwargs):
    return ChatStreamChunk(
        id="chatcmpl-test",
        created=1,
        model="test-model",
        choices=[ChatStreamChoice(delta=ChatMessage(role="assistant", content="hi"))],
        **kwargs,
    )


def test_to_sse_json_omits_unset_timings():
    data = json.loads(_chunk().to_sse_json())
    assert "timings" not in data
    # other nulls stay OpenAI-conformant
    assert "usage" in data and data["usage"] is None
    assert data["choices"][0]["delta"]["content"] == "hi"


def test_to_sse_json_keeps_populated_timings():
    from mlx_vlm.server.schemas import GenerationTimings

    chunk = _chunk(
        timings=GenerationTimings(
            prompt_n=1,
            cache_n=0,
            predicted_n=4,
            prompt_ms=50.0,
            prompt_per_token_ms=50.0,
            prompt_per_second=20.0,
            predicted_ms=400.0,
            predicted_per_token_ms=100.0,
            predicted_per_second=10.0,
        )
    )
    data = json.loads(chunk.to_sse_json())
    assert "timings" in data and data["timings"] is not None
