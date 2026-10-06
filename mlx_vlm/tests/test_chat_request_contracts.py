"""Fork-only: chat-completions request contracts pinned across the v0.7.6 upstream sync.

1. Prompt identity: the messages handed to ``apply_chat_template`` match the ones
   fork 664c2ead produced (golden fixture), including the fork's stripping of
   prior assistant thinking.
2. Compaction is opt-in: without a non-empty ``context_management`` list a chat
   request is never compacted, re-rendered or re-tokenized, and the fork's soft
   clamp (``_apply_generation_budget``) is the only overflow policy.
3. ``mlx_vlm.server`` imports without ``cryptography``; compaction capsules then
   fail with a clear error instead of an import-time crash.
"""

import importlib
import json
import subprocess
import sys
import textwrap
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import mlx_vlm.server as server
from mlx_vlm.generate import GenerationResult
from mlx_vlm.server import generation

try:
    compaction = importlib.import_module("mlx_vlm.server.compaction")
except ImportError:  # pre-sync tree: no compaction module at all
    compaction = None

FIXTURE = Path(__file__).parent / "fixtures" / "chat_prompt_identity_664c2ead.json"
CASES = json.loads(FIXTURE.read_text())["cases"]
LIMIT = 262144
MAX_TOKENS = 102400  # deployed max_tokens; limit - MAX_TOKENS = 159744


def _result():
    return GenerationResult(
        text="ok",
        prompt_tokens=8,
        generation_tokens=1,
        total_tokens=9,
        prompt_tps=1.0,
        generation_tps=1.0,
        peak_memory=0.1,
    )


@pytest.fixture
def client():
    with TestClient(server.app) as test_client:
        yield test_client


def _patched(stack, config=None):
    config = config or NS(model_type="qwen3_5", max_position_embeddings=LIMIT)
    stack.enter_context(
        patch.object(server, "get_cached_model", return_value=(NS(), NS(), config))
    )
    templ = stack.enter_context(
        patch.object(server, "apply_chat_template", return_value="prompt")
    )
    gen = stack.enter_context(patch.object(server, "generate", return_value=_result()))
    stack.enter_context(patch.object(server.runtime, "response_generator", None))
    return templ, gen


@pytest.mark.parametrize("name", sorted(CASES))
def test_chat_messages_match_pre_sync_golden(client, name):
    case = CASES[name]
    with ExitStack() as stack:
        templ, _ = _patched(stack)
        response = client.post(
            "/v1/chat/completions", json={"model": "demo", **case["request"]}
        )
    assert response.status_code == 200, response.text
    expected = case["expected"]
    assert len(templ.call_args_list) == expected["calls"]
    call = templ.call_args_list[0]
    assert call.args[2] == expected["messages"]
    assert call.kwargs.get("num_images") == expected["num_images"]
    assert call.kwargs.get("tools") == expected["tools"]


def test_chat_message_to_prompt_owns_the_assistant_thinking_strip():
    """One code path: the compaction renderer must see what the endpoint sends."""
    normalization = importlib.import_module("mlx_vlm.server.request_normalization")
    if not hasattr(normalization, "_chat_message_to_prompt"):
        pytest.skip("pre-sync tree: no _chat_message_to_prompt")
    to_prompt = normalization._chat_message_to_prompt
    assert to_prompt({"role": "assistant", "content": "<think>x</think>\n\nA"}) == {
        "role": "assistant",
        "content": "A",
    }
    assert to_prompt({"role": "assistant", "content": "x</think>B"})["content"] == "B"
    # user and tool content are never stripped
    assert to_prompt({"role": "user", "content": "<think>u</think>"})["content"] == (
        "<think>u</think>"
    )


def _big_conversation():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "third"},
    ]


@pytest.mark.parametrize(
    "extra",
    [{}, {"context_management": None}, {"context_management": []}],
    ids=["absent", "null", "empty"],
)
def test_chat_without_context_management_is_never_compacted(client, monkeypatch, extra):
    """A prompt past limit - max_tokens is served as before: one render, no extra
    tokenization, no compaction call; the soft clamp handles the overflow."""
    monkeypatch.setattr(server.runtime.config, "max_kv_size", LIMIT)
    prepare_calls = []

    def huge_prompt(*args, **kwargs):  # 200K tokens > 159744
        prepare_calls.append(1)
        return {"input_ids": list(range(200_000))}

    with ExitStack() as stack:
        templ, gen = _patched(stack)
        compact = None
        if compaction is not None:
            stack.enter_context(
                patch.object(compaction, "prepare_inputs", side_effect=huge_prompt)
            )
            compact = stack.enter_context(
                patch.object(
                    compaction,
                    "compact_response_context",
                    wraps=compaction.compact_response_context,
                )
            )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "demo",
                "messages": _big_conversation(),
                "max_tokens": MAX_TOKENS,
                **extra,
            },
        )
    assert response.status_code == 200, response.text
    assert len(templ.call_args_list) == 1
    assert prepare_calls == []
    if compact is not None:
        compact.assert_not_called()
    assert [m["content"] for m in templ.call_args_list[0].args[2]] == [
        m["content"] for m in _big_conversation()
    ]
    assert gen.call_args.kwargs.get("max_tokens") == MAX_TOKENS


def test_chat_with_context_management_reaches_compaction(client, monkeypatch):
    """Positive control: the opt-in path is still wired to upstream's compaction."""
    if compaction is None:
        pytest.skip("pre-sync tree: no compaction module")
    monkeypatch.setattr(server.runtime.config, "max_kv_size", LIMIT)
    with ExitStack() as stack:
        _patched(stack)
        compact = stack.enter_context(
            patch.object(
                compaction,
                "compact_response_context",
                return_value=compaction.CompactedContext(
                    [dict(m, type="message") for m in _big_conversation()], 10, 10
                ),
            )
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "demo",
                "messages": _big_conversation(),
                "context_management": [
                    {"type": "compaction", "compact_threshold": 1000}
                ],
            },
        )
    assert response.status_code == 200, response.text
    compact.assert_called_once()


@pytest.mark.parametrize(
    "prompt_tokens,max_tokens,thinking_budget",
    [
        (1_000, 102_400, 81_920),
        (159_744, 102_400, 81_920),
        (159_745, 102_399, 81_919),
        (200_000, 62_144, 49_715),
    ],
)
def test_soft_clamp_is_unchanged(monkeypatch, prompt_tokens, max_tokens, thinking_budget):
    """Values computed by fork 664c2ead's _apply_generation_budget (limit 262144,
    deployed max_tokens 102400 / thinking_budget 81920, ratio 0.8, floor 2048)."""
    monkeypatch.setattr(server.runtime.config, "max_kv_size", LIMIT)
    monkeypatch.delenv("MIN_OUTPUT_TOKENS", raising=False)
    assert generation.THINKING_BUDGET_CLAMP_RATIO == 0.8
    args = NS(max_tokens=102_400, thinking_budget=81_920)
    generation._apply_generation_budget(args, prompt_tokens)
    assert (args.max_tokens, args.thinking_budget) == (max_tokens, thinking_budget)


@pytest.mark.parametrize("prompt_tokens", [260_500, 262_144])
def test_soft_clamp_rejects_below_the_floor(monkeypatch, prompt_tokens):
    monkeypatch.setattr(server.runtime.config, "max_kv_size", LIMIT)
    monkeypatch.delenv("MIN_OUTPUT_TOKENS", raising=False)
    with pytest.raises(generation.PromptTooLongError):
        generation._apply_generation_budget(
            NS(max_tokens=102_400, thinking_budget=81_920), prompt_tokens
        )


_BLOCK_CRYPTOGRAPHY = textwrap.dedent(
    """
    import importlib.abc, sys

    class _Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "cryptography" or name.startswith("cryptography."):
                raise ImportError("blocked: " + name)
            return None

    sys.meta_path.insert(0, _Block())
    """
)


def test_server_imports_without_cryptography():
    if compaction is None:
        pytest.skip("pre-sync tree: no compaction module")
    code = _BLOCK_CRYPTOGRAPHY + textwrap.dedent(
        """
        import mlx_vlm.server
        from mlx_vlm.server import compaction, openai
        from mlx_vlm.server.cli import main
        assert "cryptography" not in sys.modules
        try:
            compaction.seal([], model="m", tenant=None)
        except RuntimeError as exc:
            assert "cryptography" in str(exc), exc
            print("SEAL_ERROR_OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "SEAL_ERROR_OK" in proc.stdout
