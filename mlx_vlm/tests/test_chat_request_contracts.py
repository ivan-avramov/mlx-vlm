"""Fork-only: chat-completions request contracts pinned across the v0.7.6 upstream sync.

1. Prompt identity: the messages handed to ``apply_chat_template`` match the ones
   fork 664c2ead produced (golden fixture), including the fork's stripping of
   prior assistant thinking.
2. Chat compaction is a server switch (MLX_VLM_CHAT_COMPACTION / --chat-compaction,
   default off). Off: ``context_management`` is ignored whatever its value. On: it is
   validated as upstream does and a non-empty list runs upstream compaction. Otherwise
   a chat request is never compacted, re-rendered or re-tokenized, and the fork's soft
   clamp (``_apply_generation_budget``) is the only overflow policy.
3. ``cryptography`` is optional: ``mlx_vlm.server`` imports without it, and a request
   needing a compaction capsule gets a clear 400 instead of a crash.
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
    streamer = stack.enter_context(
        patch.object(
            server,
            "stream_generate",
            side_effect=lambda *a, **kw: iter(
                [server.StreamingToken(text="ok", token=1, logprobs=0.0, finish_reason="stop")]
            ),
        )
    )
    stack.enter_context(patch.object(server.runtime, "response_generator", None))
    return templ, gen, streamer


def _post(client, body, stream):
    return client.post(
        "/v1/chat/completions", json={"model": "demo", **body, "stream": stream}
    )


def _generation_call(gen, streamer, stream):
    return (streamer if stream else gen).call_args


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_chat_messages_match_pre_sync_golden(client, name, stream):
    case = CASES[name]
    with ExitStack() as stack:
        templ, _, _ = _patched(stack)
        response = _post(client, case["request"], stream)
        body = response.text  # drain the stream inside the patches
    assert response.status_code == 200, body
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


_VALID = [{"type": "compaction", "compact_threshold": 1000}]
_GARBAGE = {
    "dict_auto": {"type": "auto"},
    "string_off": "off",
    "missing_threshold": [{"type": "compaction"}],
    "zero_threshold": [{"type": "compaction", "compact_threshold": 0}],
    "two_entries": [
        {"type": "compaction", "compact_threshold": 5},
        {"type": "compaction", "compact_threshold": 6},
    ],
}
_ABSENT = object()
_GATE_OFF_VALUES = {
    "absent": _ABSENT,
    "null": None,
    "empty": [],
    "valid": _VALID,
    **_GARBAGE,
}


def _body(value):
    body = {"messages": _big_conversation(), "max_tokens": MAX_TOKENS}
    if value is not _ABSENT:
        body["context_management"] = value
    return body


def _spy_compaction(stack, prepare_calls):
    """Make compaction see a 200K-token prompt (> limit - max_tokens = 159744)
    and record whether it was entered at all."""
    if compaction is None:
        return None

    def huge_prompt(*args, **kwargs):
        prepare_calls.append(1)
        return {"input_ids": list(range(200_000))}

    stack.enter_context(
        patch.object(compaction, "prepare_inputs", side_effect=huge_prompt)
    )
    return stack.enter_context(
        patch.object(
            compaction,
            "compact_response_context",
            wraps=compaction.compact_response_context,
        )
    )


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.parametrize("value", sorted(_GATE_OFF_VALUES))
def test_gate_off_context_management_is_ignored(client, monkeypatch, value, stream):
    """Default server (gate OFF): any context_management value -- valid or garbage --
    is accepted and ignored exactly as at 664c2ead: 200, one template render, no
    compaction, no extra tokenization, unchanged generation kwargs."""
    monkeypatch.delenv("MLX_VLM_CHAT_COMPACTION", raising=False)
    monkeypatch.setattr(server.runtime.config, "max_kv_size", LIMIT)
    prepare_calls = []
    with ExitStack() as stack:
        templ, gen, streamer = _patched(stack)
        compact = _spy_compaction(stack, prepare_calls)
        response = _post(client, _body(_GATE_OFF_VALUES[value]), stream)
        text = response.text
    assert response.status_code == 200, text
    assert len(templ.call_args_list) == 1
    assert prepare_calls == []
    if compact is not None:
        compact.assert_not_called()
    assert [m["content"] for m in templ.call_args_list[0].args[2]] == [
        m["content"] for m in _big_conversation()
    ]
    assert _generation_call(gen, streamer, stream).kwargs.get("max_tokens") == MAX_TOKENS


def _gate_on(monkeypatch):
    if compaction is None:
        pytest.skip("pre-sync tree: no compaction module")
    monkeypatch.setenv("MLX_VLM_CHAT_COMPACTION", "1")
    monkeypatch.setattr(server.runtime.config, "max_kv_size", LIMIT)


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_gate_on_valid_list_reaches_compaction(client, monkeypatch, stream):
    _gate_on(monkeypatch)
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
        response = _post(client, _body(_VALID), stream)
        text = response.text
    assert response.status_code == 200, text
    compact.assert_called_once()
    assert compact.call_args.args[0].context_management[0].compact_threshold == 1000


@pytest.mark.parametrize("value", sorted(_GARBAGE))
def test_gate_on_garbage_is_rejected(client, monkeypatch, value):
    _gate_on(monkeypatch)
    with ExitStack() as stack:
        templ, _, _ = _patched(stack)
        compact = stack.enter_context(
            patch.object(compaction, "compact_response_context")
        )
        response = _post(client, _body(_GARBAGE[value]), False)
    assert response.status_code == 422, response.text
    compact.assert_not_called()
    templ.assert_not_called()


@pytest.mark.parametrize("value", ["absent", "null", "empty"])
def test_gate_on_without_a_list_never_compacts(client, monkeypatch, value):
    _gate_on(monkeypatch)
    prepare_calls = []
    with ExitStack() as stack:
        templ, _, _ = _patched(stack)
        compact = _spy_compaction(stack, prepare_calls)
        response = _post(client, _body(_GATE_OFF_VALUES[value]), False)
    assert response.status_code == 200, response.text
    compact.assert_not_called()
    assert prepare_calls == [] and len(templ.call_args_list) == 1


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
    """cryptography is optional: the worker imports without it, and a request that
    needs a compaction capsule gets a clear 400 instead of a 500."""
    if compaction is None:
        pytest.skip("pre-sync tree: no compaction module")
    code = _BLOCK_CRYPTOGRAPHY + textwrap.dedent(
        """
        import mlx_vlm.server
        from mlx_vlm.server import compaction, openai
        from mlx_vlm.server.cli import main
        from fastapi import HTTPException
        from fastapi.testclient import TestClient
        assert "cryptography" not in sys.modules
        try:
            compaction.seal([], model="m", tenant=None)
        except HTTPException as exc:
            assert exc.status_code == 400 and "cryptography" in str(exc.detail), exc
            print("SEAL_400_OK")
        capsule = {
            "type": "compaction",
            "encrypted_content": compaction.CAPSULE_PREFIX + "x",
        }
        with TestClient(mlx_vlm.server.app) as client:
            r = client.post(
                "/v1/responses",
                json={"model": "m", "input": [capsule, {"role": "user", "content": "hi"}]},
            )
        assert r.status_code == 400, (r.status_code, r.text)
        assert "cryptography" in r.text, r.text
        print("CAPSULE_400_OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "SEAL_400_OK" in proc.stdout and "CAPSULE_400_OK" in proc.stdout


def test_cryptography_is_not_a_hard_requirement():
    requirements = (Path(__file__).parents[2] / "requirements.txt").read_text()
    assert "cryptography" not in requirements


def _run_cli(monkeypatch, *extra):
    import os

    from mlx_vlm.server import cli

    monkeypatch.setattr(sys, "argv", ["mlx_vlm.server", "--port", "8080", *extra])
    with patch.dict(os.environ), patch.object(cli.uvicorn, "run"):
        for key in ("MLX_VLM_CHAT_COMPACTION", "MLX_VLM_GENERATION_DEFAULTS"):
            os.environ.pop(key, None)
        cli.main()
        return dict(os.environ)


@pytest.mark.parametrize("flag,expected", [((), "off"), (("--chat-compaction", "on"), "on")])
def test_cli_chat_compaction_flag_is_exported(monkeypatch, flag, expected):
    env = _run_cli(monkeypatch, *flag)
    assert env["MLX_VLM_CHAT_COMPACTION"] == expected


@pytest.mark.parametrize(
    "defaults,expected",
    [(None, "typical_p=1.0 (default)"), ('{"typical_p": 0.9}', "typical_p=0.9 (--generation-defaults)")],
)
def test_cli_logs_effective_typical_p(monkeypatch, caplog, defaults, expected):
    """typical_p != 1 changes sampling and the registry never sets it; make the
    effective value visible in the worker log at startup."""
    extra = ("--generation-defaults", defaults) if defaults else ()
    with caplog.at_level("INFO"):
        _run_cli(monkeypatch, *extra)
    assert any(expected in r.getMessage() for r in caplog.records), [
        r.getMessage() for r in caplog.records
    ]
