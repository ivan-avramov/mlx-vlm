import asyncio
import base64

# Fork (2026-10-06 v0.7.6 sync): imports for the tests ported from upstream below
import copy
import json
import logging
import math
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from types import SimpleNamespace
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import mlx.core as mx
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from huggingface_hub import scan_cache_dir
from PIL import Image
from transformers.utils.chat_parsing import ResponseParser, parse_response

import mlx_vlm.reranker_loader as reranker_loader
import mlx_vlm.server as server
import mlx_vlm.server.anthropic as server_anthropic
import mlx_vlm.server.cli as cli
import mlx_vlm.server.generation as server_generation
import mlx_vlm.server.generation as generation
import mlx_vlm.server.openai as server_openai
import mlx_vlm.server.openai as openai
import mlx_vlm.server.reranking as server_reranking
import mlx_vlm.speculative.utils as speculative_utils
from mlx_vlm import apc as apc_module
from mlx_vlm.apc import hash_image_payload
from mlx_vlm.generate import GenerationResult
from mlx_vlm.generate.image import ImageGenerationResult
from mlx_vlm.prompt_utils import (  # noqa: F401 -- kept: upstream's copy imports it (registry audit)
    apply_chat_template,
)
from mlx_vlm.server import compaction
from mlx_vlm.server.model_discovery import discover_models, is_model_directory
from mlx_vlm.server.responses_state import (
    ToolCallStreamState,
    _response_items_to_chat,
    strip_protocol_markers,
)
from mlx_vlm.server.runtime_config import RuntimeConfig

# Fork (2026-09-27 sync): imports for the tests ported from upstream's test_server.py
from mlx_vlm.tests.test_processors import MINICPM_MULTICALL
from mlx_vlm.tokenizer_utils import SPMStreamingDetokenizer, _ServerTokenStreamer
from mlx_vlm.tools import _infer_tool_parser, load_tool_module, process_tool_calls
from mlx_vlm.tools.parsers import minicpm5


def test_response_generator_prefill_step_override_wins_over_environment(monkeypatch):
    class DormantThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setenv("PREFILL_STEP_SIZE", "2048")
    monkeypatch.setattr(server_generation, "Thread", DormantThread)

    default_generator = server.ResponseGenerator(model_path="default")
    overridden_generator = server.ResponseGenerator(
        model_path="overridden",
        prefill_step_size=3072,
    )

    assert default_generator.prefill_step_size == 2048
    assert overridden_generator.prefill_step_size == 3072


def test_response_generator_clears_worker_streams(monkeypatch):
    gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
    error = RuntimeError("worker failed")
    gen._run_impl = MagicMock(side_effect=error)
    clear_streams = MagicMock()
    monkeypatch.setattr(server_generation, "clear_mlx_streams", clear_streams)

    with pytest.raises(RuntimeError, match="worker failed"):
        gen._run()

    gen._run_impl.assert_called_once_with()
    clear_streams.assert_called_once_with()


_MUSE_RESPONSE_TEMPLATE = {
    "defaults": {"role": "assistant"},
    "fields": {
        "content": {
            "close": ["<|eot|>", "<|eom|>"],
            "content": "text",
            "open_pattern": r"to=user<\|message\|>",
        },
        "reasoning_content": {
            "close": "<|eom|>",
            "content": "text",
            "open_pattern": r"to=self<\|message\|>",
        },
        "tool_calls": {
            "close": "</atem:invoke>",
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": (
                    r'<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"'
                    r"[^>]*?>(?P<value>.*?)</atem:parameter>"
                ),
                "value_parser": {
                    "args": {"allow_non_json": True},
                    "name": "json",
                },
            },
            "open_pattern": r'<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">',
            "repeats": True,
            "transform": {
                "function": {"arguments": "{content}", "name": "{name}"},
                "type": "function",
            },
        },
    },
    "start_anchor": "<|start|>assistant",
}


class _MuseResponseTemplateTokenizer:
    response_template = _MUSE_RESPONSE_TEMPLATE

    def parse_response(self, response, prefix=None):
        return parse_response(response, self.response_template, prefix=prefix)

    def get_response_parser(self, prefix=None):
        return ResponseParser(self.response_template, prefix=prefix)


@pytest.fixture
def client():
    with TestClient(server.app) as test_client:
        yield test_client


def _gemma_thinking_channel_chunks():
    return [
        server.StreamingToken(text="", token=100, logprobs=0.0, finish_reason=None),
        server.StreamingToken(text="", token=45518, logprobs=0.0, finish_reason=None),
        server.StreamingToken(text="", token=107, logprobs=0.0, finish_reason=None),
        server.StreamingToken(text="", token=101, logprobs=0.0, finish_reason=None),
        server.StreamingToken(text="", token=236832, logprobs=0.0, finish_reason=None),
        server.StreamingToken(
            text="<|channel>thought\n<channel|>7",
            token=808,
            logprobs=0.0,
            finish_reason=None,
        ),
        server.StreamingToken(
            text=" *", token=236743, logprobs=0.0, finish_reason=None
        ),
        server.StreamingToken(text="", token=236828, logprobs=0.0, finish_reason=None),
        server.StreamingToken(text=" 8", token=578, logprobs=0.0, finish_reason=None),
        server.StreamingToken(
            text=" =", token=236743, logprobs=0.0, finish_reason=None
        ),
        server.StreamingToken(text="", token=236810, logprobs=0.0, finish_reason=None),
        server.StreamingToken(text="", token=236825, logprobs=0.0, finish_reason=None),
        server.StreamingToken(
            text=" 56", token=106, logprobs=0.0, finish_reason="stop"
        ),
    ]


@pytest.mark.parametrize("value", [224, "22", [1.0], [1.5], [True], [1, 2, 3]])
def test_chat_completions_endpoint_rejects_invalid_resize_shape(client, value):
    response = client.post(
        "/chat/completions",
        json={
            "model": "demo",
            "messages": [{"role": "user", "content": "Hello"}],
            "resize_shape": value,
        },
    )

    assert response.status_code == 422


def test_chat_completions_endpoint_requires_model(client):
    response = client.post(
        "/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 422
    detail = response.json().get("detail", [])
    assert any(err.get("loc") == ["body", "model"] for err in detail)


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": ""}],
        [{"role": "user", "content": " \n\t "}],
        [{"role": "user", "content": [{"type": "text", "text": " "}]}],
    ],
)
def test_chat_completions_endpoint_rejects_empty_effective_input(client, messages):
    with patch.object(server_openai, "get_cached_model") as mock_get_cached_model:
        response = client.post(
            "/chat/completions",
            json={"model": "demo", "messages": messages},
        )

    assert response.status_code == 400
    assert "non-empty message content" in response.json()["detail"]
    mock_get_cached_model.assert_not_called()


@pytest.mark.parametrize(
    "input_value",
    [
        "",
        " \n\t ",
        [],
        [{"role": "user", "content": ""}],
        [{"role": "user", "content": [{"type": "input_text", "text": " "}]}],
    ],
)
def test_responses_endpoint_rejects_empty_effective_input(client, input_value):
    with patch.object(server_openai, "get_cached_model") as mock_get_cached_model:
        response = client.post(
            "/v1/responses",
            json={"model": "demo", "input": input_value},
        )

    assert response.status_code == 400
    assert "non-empty message content" in response.json()["detail"]
    mock_get_cached_model.assert_not_called()


def test_chat_request_schema_requires_model():
    assert "model" in server.ChatRequest.model_json_schema()["required"]


def test_chat_request_schema_declares_tool_choice_fields():
    properties = server.ChatRequest.model_json_schema()["properties"]

    assert "tools" in properties
    assert "tool_choice" in properties


def test_chat_request_schema_allows_one_or_two_resize_shape_values():
    resize_shape = server.ChatRequest.model_json_schema()["properties"]["resize_shape"]
    lengths = {
        (item["minItems"], item["maxItems"])
        for item in resize_shape["anyOf"]
        if item.get("type") == "array"
    }

    assert lengths == {(1, 1), (2, 2)}


def test_chat_completions_tool_choice_none_disables_tools(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(chat_template="<tool_call>\n<function=")
    )
    config = SimpleNamespace(model_type="qwen3_5")
    result = GenerationResult(
        text="No tool call.", prompt_tokens=5, generation_tokens=3
    )
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Use the tool."}],
                "tools": tools,
                "tool_choice": "none",
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["tool_calls"] is None
    assert mock_template.call_args.kwargs["tools"] is None
    assert mock_template.call_args.kwargs["tool_choice"] == "none"


def test_chat_completions_tool_parser_override(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    # A template with no tool markers: inference alone selects no parser.
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(chat_template="a plain template, no tool markers")
    )
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text='<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>',
        prompt_tokens=5,
        generation_tokens=3,
    )
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]

    def post(extra):
        with (
            patch.object(
                server, "get_cached_model", return_value=(model, processor, config)
            ),
            patch.object(server, "apply_chat_template", return_value="prompt"),
            patch.object(server, "generate", return_value=result),
        ):
            return client.post(
                "/v1/chat/completions",
                json={
                    "model": "demo",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": tools,
                    **extra,
                },
            )

    # Without an override the markerless template routes to no parser: no calls.
    base = post({})
    assert base.status_code == 200
    assert base.json()["choices"][0]["message"]["tool_calls"] is None

    # The override forces json_tools, which parses the emitted call.
    overridden = post({"tool_parser": "json_tools"})
    assert overridden.status_code == 200
    calls = overridden.json()["choices"][0]["message"]["tool_calls"]
    assert calls and calls[0]["function"]["name"] == "get_weather"

    # An unknown parser name is rejected at request validation.
    assert post({"tool_parser": "bogus"}).status_code == 422


def test_chat_completions_required_tool_choice_adds_instruction(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(text="done", prompt_tokens=5, generation_tokens=2)
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": tools,
                "tool_choice": "required",
            },
        )

    assert response.status_code == 200
    messages = mock_template.call_args.args[2]
    assert messages[0]["role"] == "user"
    assert "must call one or more" in messages[0]["content"]
    assert mock_template.call_args.kwargs["tools"] == tools
    assert mock_template.call_args.kwargs["tool_choice"] == "required"


def test_chat_completions_forced_tool_choice_filters_tools(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(text="done", prompt_tokens=5, generation_tokens=2)
    tools = [
        {
            "type": "function",
            "function": {"name": "get_time", "parameters": {"type": "object"}},
        },
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        },
    ]
    tool_choice = {"type": "function", "function": {"name": "get_weather"}}

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "demo",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Say hello."},
                ],
                "tools": tools,
                "tool_choice": tool_choice,
            },
        )

    assert response.status_code == 200
    messages = mock_template.call_args.args[2]
    assert messages[0]["content"].startswith("Be concise.")
    assert "must call the 'get_weather' function" in messages[0]["content"]
    assert "must call the 'get_weather' function" in messages[-1]["content"]
    selected_tools = mock_template.call_args.kwargs["tools"]
    assert [tool["function"]["name"] for tool in selected_tools] == ["get_weather"]
    assert mock_template.call_args.kwargs["tool_choice"] == tool_choice


@pytest.mark.parametrize(
    ("tools", "tool_choice", "detail"),
    [
        ([], "required", "requires at least one tool"),
        (
            [
                {
                    "type": "function",
                    "function": {"name": "get_weather"},
                }
            ],
            {"type": "function", "function": {"name": "missing"}},
            "unknown function 'missing'",
        ),
        ([], "sometimes", "Invalid tool_choice"),
    ],
)
def test_chat_completions_rejects_invalid_tool_choice(
    client, tools, tool_choice, detail
):
    with patch.object(server, "get_cached_model") as mock_get_cached_model:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": tools,
                "tool_choice": tool_choice,
            },
        )

    assert response.status_code == 400
    assert detail in response.json()["detail"]
    mock_get_cached_model.assert_not_called()


def test_speculative_server_dispatches_mtp_batch_loop():
    assert (
        speculative_utils.get_speculative_rounds_batch("mtp")
        is speculative_utils._mtp_rounds_batch
    )


def test_positioned_target_sampler_is_batch_grouping_invariant():
    sampler = server_generation._PositionedTargetSampler(
        temperature=0.7, top_p=1.0, seed=42
    )
    logits = mx.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ],
        dtype=mx.float32,
    )
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    batched = sampler.sample_target(
        logprobs,
        row_ids=[0, 0],
        positions=[5, 5],
    )
    single_0 = sampler.sample_target(
        logprobs[0:1],
        row_ids=[0],
        positions=[5],
    )
    single_1 = sampler.sample_target(
        logprobs[1:2],
        row_ids=[0],
        positions=[5],
    )
    mx.eval(batched, single_0, single_1)

    assert batched.tolist() == [single_0.item(), single_1.item()]


@pytest.mark.parametrize("top_p", [1.0, 0.95])
def test_positioned_target_sampler_honors_top_k(top_p):
    sampler = server_generation._PositionedTargetSampler(
        temperature=1.0, top_p=top_p, top_k=2, seed=42
    )
    logits = mx.array([[0.0, 1.0, 2.0, 3.0]], dtype=mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    repeated = mx.repeat(logprobs, 32, axis=0)

    tokens = sampler.sample_target(
        repeated,
        row_ids=[0] * 32,
        positions=list(range(32)),
    )
    mx.eval(tokens)

    assert set(tokens.tolist()) <= {2, 3}


def test_server_passes_top_k_to_positioned_sampler():
    generator = server.ResponseGenerator.__new__(server.ResponseGenerator)
    args = server_generation.GenerationArguments(
        max_tokens=1,
        temperature=1.0,
        top_k=7,
    )

    sampler = generator._make_sampler(args)

    assert sampler.top_k == 7


# Fork: extended filter, seeded row, and insertion coverage.
def test_positioned_target_sampler_honors_top_k_fork():
    # Fork: _PositionedTargetSampler is fork-only (the seeded, position-keyed
    # sampler); upstream has no equivalent to test.
    # top_k=1 collapses each row to its argmax token, regardless of the
    # position-keyed RNG -> draws are deterministic and equal to argmax.
    sampler = server_generation._PositionedTargetSampler(
        temperature=0.7, top_p=1.0, seed=42, top_k=1
    )
    logits = mx.array(
        [
            [0.0, 1.0, 2.0, 3.0],  # argmax index 3
            [3.0, 2.0, 1.0, 0.0],  # argmax index 0
        ],
        dtype=mx.float32,
    )
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tokens = sampler.sample_target(logprobs, row_ids=[0, 1], positions=[5, 5])
    mx.eval(tokens)
    assert tokens.tolist() == [3, 0]


def test_positioned_target_sampler_min_p_filters_tail():
    # A high min_p prunes the low-probability tail; with these logits only the
    # top token survives in row 0, so the draw is deterministic.
    sampler = server_generation._PositionedTargetSampler(
        temperature=0.7, top_p=1.0, seed=7, min_p=0.9
    )
    logits = mx.array([[0.0, 0.1, 0.2, 6.0]], dtype=mx.float32)  # token 3 dominates
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tokens = sampler.sample_target(logprobs, row_ids=[0], positions=[3])
    mx.eval(tokens)
    assert tokens.tolist() == [3]


def test_positioned_target_sampler_defaults_unchanged():
    # top_k=0 / min_p=0 / top_p=1 must reproduce the plain keyed categorical
    # draw -> no behavior change for existing callers.
    logits = mx.array([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]], dtype=mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    base = server_generation._PositionedTargetSampler(
        temperature=0.7, top_p=1.0, seed=42
    ).sample_target(logprobs, row_ids=[0, 1], positions=[5, 5])
    explicit_defaults = server_generation._PositionedTargetSampler(
        temperature=0.7, top_p=1.0, seed=42, top_k=0, min_p=0.0
    ).sample_target(logprobs, row_ids=[0, 1], positions=[5, 5])
    mx.eval(base, explicit_defaults)
    assert base.tolist() == explicit_defaults.tolist()


def test_positioned_target_sampler_seeds_override_per_row():
    """O30: sample_target's optional seeds= lets each row draw under ITS OWN
    seed, independent of the sampler's single construction-time seed. This is
    what makes per-request seeds take effect on the batched decode path,
    where one sampler instance is shared across many simultaneous requests.
    """
    sampler = server_generation._PositionedTargetSampler(
        temperature=1.0, top_p=1.0, seed=999999  # must be ignored when seeds= given
    )
    logits = mx.zeros((2, 4096), dtype=mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    different = sampler.sample_target(
        logprobs, row_ids=[0, 0], positions=[3, 3], seeds=[111, 222]
    )
    mx.eval(different)
    assert different.tolist()[0] != different.tolist()[1]


def test_positioned_target_sampler_seeds_reproduce_across_calls():
    """Same declared seed + same position -> identical draw (O30 requirement:
    determinism per request must be preserved)."""
    sampler = server_generation._PositionedTargetSampler(
        temperature=0.7, top_p=1.0, seed=0
    )
    logits = mx.array([[0.0, 1.0, 2.0, 3.0]], dtype=mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    first = sampler.sample_target(logprobs, row_ids=[0], positions=[7], seeds=[42])
    second = sampler.sample_target(logprobs, row_ids=[0], positions=[7], seeds=[42])
    mx.eval(first, second)
    assert first.tolist() == second.tolist()


def test_run_threads_per_request_seed_into_batch_insert(monkeypatch):
    """O30 end-to-end: ResponseGenerator._run must pass EACH request's OWN
    declared seed to BatchGenerator.insert(), not only the first request's
    seed used to construct the BatchGenerator's sampler.
    """
    captured = {"seeds": []}

    class FakeDetokenizer:
        def __init__(self):
            self.last_segment = ""

        def reset(self):
            self.last_segment = ""

        def add_token(self, token):
            self.last_segment = str(token)

        def finalize(self):
            pass

    class FakeBatchGenerator:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self._next_uid = 1
            self._active = {}

        def insert(self, *args, **kwargs):
            del args
            captured["seeds"].append(kwargs.get("seeds"))
            uid = self._next_uid
            self._next_uid += 1
            self._active[uid] = 0
            return (uid,)

        def remove(self, uid):
            return self._active.pop(uid, None) is not None

        @property
        def unprocessed_prompts(self):
            return []

        @property
        def has_pending_prompts(self):
            return False

        def next(self, **kwargs):
            del kwargs
            responses = []
            finished = []
            for uid in sorted(self._active):
                responses.append(
                    SimpleNamespace(
                        uid=uid, token=uid, token_logprob=0.0, finish_reason="stop"
                    )
                )
                finished.append(uid)
            for uid in finished:
                del self._active[uid]
            return [], responses

    monkeypatch.setattr(server_generation, "BatchGenerator", FakeBatchGenerator)
    monkeypatch.setattr(
        server_generation, "make_streaming_detokenizer", lambda _: FakeDetokenizer()
    )

    gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
    gen.model_path = "demo"
    gen.adapter_path = None
    gen.model = None
    gen.processor = None
    gen.config = None
    gen.stop_tokens = set()
    gen.vision_cache = None
    gen.draft_model = None
    gen.draft_kind = None
    gen.kv_bits = None
    gen.kv_group_size = server.DEFAULT_KV_GROUP_SIZE
    gen.kv_quant_scheme = server.DEFAULT_KV_QUANT_SCHEME
    gen.quantized_kv_start = server.DEFAULT_QUANTIZED_KV_START
    gen.top_logprobs_k = 0
    gen.apc_manager = None
    gen.tokenizer = SimpleNamespace()
    gen.requests = Queue()
    gen._stop = False
    gen._ready = Event()
    gen._load_error = None
    gen._cancelled = set()
    gen._cancel_lock = Lock()

    def fake_initialize_model():
        gen.model = SimpleNamespace(language_model=object())
        gen.processor = SimpleNamespace()
        gen.config = SimpleNamespace()
        gen.stop_tokens = set()
        gen.draft_model = None
        gen.draft_kind = None
        gen.tokenizer = SimpleNamespace()

    gen._initialize_model = fake_initialize_model
    gen._gpu_embed = lambda raw_inputs, images=None, apc_semantic_hash=None: (
        mx.array([[raw_inputs["request_id"]]], dtype=mx.int32),
        {},
    )

    request_queues = []
    seeds_requested = [111, None]
    for request_id, seed in enumerate(seeds_requested):
        rqueue = Queue()
        request_queues.append(rqueue)
        gen.requests.put(
            server_generation.QueuedGenerationRequest(
                rqueue=rqueue,
                raw_inputs={"request_id": request_id},
                prompt_tokens=1,
                args=server.GenerationArguments(max_tokens=2, seed=seed),
            )
        )

    worker = Thread(target=gen._run, daemon=True)
    worker.start()

    try:
        for rqueue in request_queues:
            ctx = rqueue.get(timeout=1)
            assert isinstance(ctx, server.GenerationContext)
            while True:
                item = rqueue.get(timeout=1)
                if item is None:
                    break
    finally:
        gen._stop = True
        gen.requests.put(None)
        worker.join(timeout=2)

    # Request 0 declared seed=111; request 1 declared no seed at all, which
    # must resolve to DEFAULT_SEED (unchanged seedless semantics) — NOT to
    # request 0's seed, which is what the pre-fix code did (only the first
    # request's args.seed ever reached the sampler).
    assert captured["seeds"] == [[111], [server_generation.DEFAULT_SEED]]


def test_speculative_server_dispatches_eagle3_batch_loop():
    assert (
        speculative_utils.get_speculative_rounds_batch("eagle3")
        is speculative_utils._eagle3_rounds_batch
    )


def test_speculative_server_keeps_dflash_default_batch_loop():
    assert (
        speculative_utils.get_speculative_rounds_batch("dflash")
        is speculative_utils._dflash_rounds_batch
    )


def test_speculative_server_rejects_unknown_draft_kind():
    with pytest.raises(ValueError):
        speculative_utils.get_speculative_rounds_batch("nope")


def test_speculative_server_prefill_kwargs_are_drafter_specific():
    drafter = SimpleNamespace(config=SimpleNamespace(target_layer_ids=[1, 2, 3]))

    assert speculative_utils.speculative_prefill_kwargs("mtp", drafter) == {
        "return_hidden": True,
        "return_shared_kv": True,
    }
    assert speculative_utils.speculative_prefill_kwargs("dflash", drafter) == {
        "capture_layer_ids": [1, 2, 3],
    }


def test_speculative_server_hidden_state_picks_last_layer_for_mtp():
    h = [mx.zeros((1, 1, 4)), mx.ones((1, 1, 4))]
    out = SimpleNamespace(hidden_states=h)

    assert speculative_utils.speculative_hidden_state("mtp", out) is h[-1]


def test_speculative_server_hidden_state_concatenates_for_dflash():
    h = [mx.zeros((1, 1, 4)), mx.ones((1, 1, 4))]
    out = SimpleNamespace(hidden_states=h)

    result = speculative_utils.speculative_hidden_state("dflash", out)
    assert result.shape == (1, 1, 8)


@pytest.mark.parametrize(
    "draft_kind,batch_size,left_padding",
    [
        ("mtp", 1, [0]),
        ("mtp", 2, [0, 1]),
        ("dflash", 1, [0]),
        ("dflash", 2, [0, 1]),
        ("eagle3", 1, [0]),
        (None, 1, [0]),
    ],
)
def test_speculative_prompt_cache_always_uses_supplied_make_cache(
    draft_kind, batch_size, left_padding
):
    # `make_cache` is what applies --kv-bits. Single-row speculation used to
    # bypass it for `cache.make_prompt_cache`, which left the KV unquantized
    # however the server was configured.  That shortcut covered every drafter
    # routed through here, so each one is checked for the single-row case.
    lm = object()
    batched_cache = object()

    assert (
        speculative_utils.make_speculative_prompt_cache(
            lm,
            draft_kind=draft_kind,
            batch_size=batch_size,
            left_padding=left_padding,
            make_cache=lambda *args, **kwargs: batched_cache,
        )
        is batched_cache
    )


def test_speculative_server_reads_draft_block_size_env(monkeypatch):
    monkeypatch.delenv("MLX_VLM_DRAFT_BLOCK_SIZE", raising=False)
    assert server._get_draft_block_size_from_env() is None

    monkeypatch.setenv("MLX_VLM_DRAFT_BLOCK_SIZE", "3")
    assert server._get_draft_block_size_from_env() == 3


def test_speculative_server_reads_batch_coalesce_env(monkeypatch):
    monkeypatch.delenv("MLX_VLM_SPEC_BATCH_COALESCE_MS", raising=False)
    assert server.get_speculative_batch_coalesce_s() == pytest.approx(0.005)

    monkeypatch.setenv("MLX_VLM_SPEC_BATCH_COALESCE_MS", "2.5")
    assert server.get_speculative_batch_coalesce_s() == pytest.approx(0.0025)

    monkeypatch.setenv("MLX_VLM_SPEC_BATCH_COALESCE_MS", "bad")
    assert server.get_speculative_batch_coalesce_s() == pytest.approx(0.005)


def test_get_cached_model_omitted_adapter_inherits_loaded_adapter(monkeypatch):
    class FakeResponseGenerator:
        def __init__(self, model_path, adapter_path=None, **kwargs):
            self.model_path = model_path
            self.adapter_path = adapter_path
            self.model = SimpleNamespace()
            self.processor = SimpleNamespace()
            self.config = SimpleNamespace(model_type="qwen2_vl")

        def wait_until_ready(self):
            return self.model, self.processor, self.config

        def stop_and_join(self):
            pass

    monkeypatch.setattr(server._app_module, "ResponseGenerator", FakeResponseGenerator)
    monkeypatch.setattr(server._app_module._apc, "from_env", lambda *_, **__: None)
    monkeypatch.setattr(server.runtime, "model_cache", {})
    monkeypatch.setattr(server.runtime, "response_generator", None)
    monkeypatch.setattr(server.runtime, "apc_manager", None)

    server.get_cached_model("demo-model", "adapter-a")
    server.get_cached_model("demo-model")

    cache_key = server.runtime.model_cache["cache_key"]
    assert cache_key[:3] == ("demo-model", "adapter-a", "text_generation")
    assert cache_key[3] == server.runtime.config.fingerprint(kinds={"text_generation"})
    assert server.runtime.model_cache["adapter_path"] == "adapter-a"


def test_unload_model_cache_group_resets_apc_around_generator_shutdown(
    monkeypatch,
):
    events = []

    class FakeAPCManager:
        def __init__(self):
            self.contents = ["old-model-prefix"]

        def clear(self):
            events.append(("clear", list(self.contents)))
            self.contents.clear()

    manager = FakeAPCManager()

    class FakeResponseGenerator:
        def stop_and_join(self):
            events.append(("stop", list(manager.contents)))
            # Simulate a store that was already in flight when shutdown began.
            manager.contents.append("draining-worker-prefix")

    response_generator = FakeResponseGenerator()
    registry = server.ModelCacheRegistry()
    registry.set(
        "text_generation",
        {
            "model_path": "old-model",
            "adapter_path": None,
            "response_generator": response_generator,
            "apc_manager": manager,
        },
    )
    monkeypatch.setattr(server.runtime, "model_cache", registry)
    monkeypatch.setattr(server.runtime, "response_generator", response_generator)
    monkeypatch.setattr(server.runtime, "apc_manager", manager)
    monkeypatch.setattr(server._app_module.gc, "collect", lambda: None)
    monkeypatch.setattr(server._app_module.mx, "clear_cache", lambda: None)

    assert server._app_module._unload_model_cache_group("text_generation") is True

    assert events == [
        ("clear", ["old-model-prefix"]),
        ("stop", []),
        ("clear", ["draining-worker-prefix"]),
    ]
    assert manager.contents == []
    assert registry.for_kind("text_generation") == {}
    assert server.runtime.response_generator is None
    assert server.runtime.apc_manager is None


def test_unload_model_cache_group_keeps_model_when_initial_apc_reset_fails(
    monkeypatch,
):
    class FailingAPCManager:
        def clear(self):
            raise RuntimeError("APC cleanup failed")

    response_generator = MagicMock()
    cache = {
        "model_path": "old-model",
        "adapter_path": None,
        "response_generator": response_generator,
        "apc_manager": FailingAPCManager(),
    }
    registry = server.ModelCacheRegistry()
    registry.set("text_generation", cache)
    monkeypatch.setattr(server.runtime, "model_cache", registry)

    with pytest.raises(RuntimeError, match="APC cleanup failed"):
        server._app_module._unload_model_cache_group("text_generation")

    response_generator.stop_and_join.assert_not_called()
    assert registry.for_kind("text_generation") is cache


@pytest.mark.parametrize(
    "load_error",
    [
        ValueError("Model type bert not supported."),
        RuntimeError("Unable to initialize model."),
    ],
)
def test_load_model_resources_returns_load_failure_as_bad_request(
    monkeypatch, load_error
):
    def reject_model(*_args, **_kwargs):
        raise load_error

    monkeypatch.setattr(server_generation, "load", reject_model)

    with pytest.raises(server.HTTPException) as exc_info:
        server_generation.load_model_resources(
            "google-bert/bert-base-multilingual-cased",
            None,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == f"Failed to load model: {load_error}"


def test_unsupported_model_request_does_not_crash_server(client, monkeypatch):
    def reject_model(*_args, **_kwargs):
        raise ValueError("Model type bert not supported.")

    monkeypatch.setattr(server_generation, "load", reject_model)
    monkeypatch.setattr(server._app_module._apc, "from_env", lambda *_, **__: None)
    monkeypatch.setattr(server.runtime, "model_cache", {})
    monkeypatch.setattr(server.runtime, "response_generator", None)
    monkeypatch.setattr(server.runtime, "apc_manager", None)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "google-bert/bert-base-multilingual-cased",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Failed to load model: Model type bert not supported."
    )
    assert client.get("/health").status_code == 200


def _unstarted_response_generator():
    gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
    gen.model_path = "demo"
    gen.adapter_path = None
    gen.model = None
    gen.processor = None
    gen.config = None
    gen.stop_tokens = set()
    gen.vision_cache = None
    gen.draft_model = None
    gen.draft_kind = None
    gen.draft_model_path = None
    gen.draft_kind_override = None
    gen.kv_bits = None
    gen.kv_group_size = server.DEFAULT_KV_GROUP_SIZE
    gen.kv_quant_scheme = server.DEFAULT_KV_QUANT_SCHEME
    gen.quantized_kv_start = server.DEFAULT_QUANTIZED_KV_START
    gen.top_logprobs_k = 0
    gen.apc_manager = None
    gen.apc_mode = None
    gen.tokenizer = None
    gen.requests = Queue()
    gen._stop = False
    gen._ready = Event()
    gen._load_error = None
    gen._cancelled = set()
    gen._cancel_lock = Lock()
    return gen


# Fork: O40 fail-loud drafter tests (715b06b) — no upstream counterpart.
def test_server_refuses_incompatible_mtp_drafter_by_default(monkeypatch):
    """Contract change (O40, 2026-08-24): an incompatible drafter REFUSES the start
    instead of silently demoting to plain decode — the silent demote is the exact
    failure mode that voided a week of MTP measurements (a server the operator
    believed had MTP on measured plain autoregressive decode). The old behavior
    survives behind MLX_VLM_DRAFT_ALLOW_FALLBACK=1 (next test)."""
    target_config = SimpleNamespace(
        model_type="gemma4_text",
        hidden_size=5376,
        eos_token_id=[],
    )
    model = SimpleNamespace(language_model=SimpleNamespace(config=target_config))
    processor = SimpleNamespace(tokenizer=SimpleNamespace())
    drafter = SimpleNamespace(
        config=SimpleNamespace(
            model_type="gemma4_assistant",
            backbone_hidden_size=1536,
        )
    )
    gen = _unstarted_response_generator()

    monkeypatch.delenv("MLX_VLM_DRAFT_ALLOW_FALLBACK", raising=False)
    monkeypatch.setenv("MLX_VLM_DRAFT_MODEL", "assistant")
    monkeypatch.setenv("MLX_VLM_DRAFT_KIND", "mtp")
    monkeypatch.setattr(
        server_generation,
        "load_model_resources",
        lambda *_args, **_kwargs: (model, processor, target_config),
    )
    monkeypatch.setattr(
        "mlx_vlm.speculative.drafters.load_drafter",
        lambda *_args, **_kwargs: (drafter, "mtp"),
    )

    with pytest.raises(RuntimeError, match="MLX_VLM_DRAFT_ALLOW_FALLBACK"):
        gen._initialize_model()


def test_server_demotes_incompatible_mtp_drafter_to_ar(monkeypatch):
    # Fork: O40 — the pre-O40 silent demote survives only behind the env gate
    # (715b06b). No upstream counterpart.
    monkeypatch.setenv("MLX_VLM_DRAFT_ALLOW_FALLBACK", "1")
    target_config = SimpleNamespace(
        model_type="gemma4_text",
        hidden_size=5376,
        eos_token_id=[],
    )
    model = SimpleNamespace(language_model=SimpleNamespace(config=target_config))
    processor = SimpleNamespace(tokenizer=SimpleNamespace())
    drafter = SimpleNamespace(
        config=SimpleNamespace(
            model_type="gemma4_assistant",
            backbone_hidden_size=1536,
        )
    )
    gen = _unstarted_response_generator()

    monkeypatch.setenv("MLX_VLM_DRAFT_MODEL", "assistant")
    monkeypatch.setenv("MLX_VLM_DRAFT_KIND", "mtp")
    monkeypatch.setattr(
        server_generation,
        "load_model_resources",
        lambda *_args, **_kwargs: (model, processor, target_config),
    )
    monkeypatch.setattr(
        "mlx_vlm.speculative.drafters.load_drafter",
        lambda *_args, **_kwargs: (drafter, "mtp"),
    )

    gen._initialize_model()

    assert gen.model is model
    assert gen.processor is processor
    assert gen.draft_model is None
    assert gen.draft_kind is None


def test_server_includes_processor_specific_stop_tokens(monkeypatch):
    config = SimpleNamespace(eos_token_id=[2])
    model = SimpleNamespace(language_model=SimpleNamespace(config=config))
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(),
        additional_eos_token_ids=[3],
    )
    gen = _unstarted_response_generator()

    monkeypatch.delenv("MLX_VLM_DRAFT_MODEL", raising=False)
    monkeypatch.delenv("MLX_VLM_DRAFT_KIND", raising=False)
    monkeypatch.setattr(
        server_generation,
        "load_model_resources",
        lambda *_args, **_kwargs: (model, processor, config),
    )

    gen._initialize_model()

    assert gen.stop_tokens == {2, 3}


def test_server_includes_tokenizer_eos_in_stop_tokens(monkeypatch):
    config = SimpleNamespace(eos_token_id=248044)
    model = SimpleNamespace(language_model=SimpleNamespace(config=config))
    processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=248046))
    gen = _unstarted_response_generator()

    monkeypatch.delenv("MLX_VLM_DRAFT_MODEL", raising=False)
    monkeypatch.delenv("MLX_VLM_DRAFT_KIND", raising=False)
    monkeypatch.setattr(
        server_generation,
        "load_model_resources",
        lambda *_args, **_kwargs: (model, processor, config),
    )

    gen._initialize_model()

    assert gen.stop_tokens == {248044, 248046}


def test_server_caches_apc_mode_when_model_initializes(monkeypatch):
    config = SimpleNamespace(eos_token_id=[])
    language_model = SimpleNamespace()
    model = SimpleNamespace(language_model=language_model)
    processor = SimpleNamespace(tokenizer=SimpleNamespace())
    gen = _unstarted_response_generator()
    gen.apc_manager = object()

    monkeypatch.delenv("MLX_VLM_DRAFT_MODEL", raising=False)
    monkeypatch.delenv("MLX_VLM_DRAFT_KIND", raising=False)
    monkeypatch.setattr(
        server_generation,
        "load_model_resources",
        lambda *_args, **_kwargs: (model, processor, config),
    )
    apc_mode = MagicMock(return_value="exact")
    monkeypatch.setattr(apc_module, "model_apc_mode", apc_mode)

    gen._initialize_model()

    assert gen.apc_mode == "exact"
    apc_mode.assert_called_once_with(language_model)


def test_server_serves_ar_requests_after_drafter_mismatch(monkeypatch):
    # Fork: O40 — degraded-start serving path behind the env gate (715b06b); the
    # deliberate degraded start now requires the gate (see
    # test_server_refuses_incompatible_mtp_drafter_by_default). No upstream
    # counterpart.
    monkeypatch.setenv("MLX_VLM_DRAFT_ALLOW_FALLBACK", "1")

    class FakeDetokenizer:
        def __init__(self):
            self.last_segment = ""

        def add_token(self, token):
            self.last_segment = str(token)

        def finalize(self):
            pass

    class FakeBatchGenerator:
        def __init__(self, *args, **kwargs):
            self.unprocessed_prompts = []
            self.has_pending_prompts = False

        def insert(self, *args, **kwargs):
            return (1,)

        def next(self, **kwargs):
            return [], [
                SimpleNamespace(
                    uid=1,
                    token=7,
                    token_logprob=0.0,
                    finish_reason="length",
                )
            ]

    target_config = SimpleNamespace(
        model_type="gemma4_text",
        hidden_size=5376,
        eos_token_id=[],
    )
    model = SimpleNamespace(language_model=SimpleNamespace(config=target_config))
    processor = SimpleNamespace(tokenizer=SimpleNamespace())
    drafter = SimpleNamespace(
        config=SimpleNamespace(
            model_type="gemma4_assistant",
            backbone_hidden_size=1536,
        )
    )
    gen = _unstarted_response_generator()

    monkeypatch.setenv("MLX_VLM_DRAFT_MODEL", "assistant")
    monkeypatch.setenv("MLX_VLM_DRAFT_KIND", "mtp")
    monkeypatch.setattr(server_generation, "BatchGenerator", FakeBatchGenerator)
    monkeypatch.setattr(
        server_generation,
        "make_streaming_detokenizer",
        lambda _processor: FakeDetokenizer(),
    )
    monkeypatch.setattr(
        server_generation,
        "load_model_resources",
        lambda *_args, **_kwargs: (model, processor, target_config),
    )
    monkeypatch.setattr(
        "mlx_vlm.speculative.drafters.load_drafter",
        lambda *_args, **_kwargs: (drafter, "mtp"),
    )
    gen._gpu_embed = lambda raw_inputs, images=None, apc_semantic_hash=None: (
        mx.array([[raw_inputs["token"]]], dtype=mx.int32),
        {},
    )

    rqueue = Queue()
    gen.requests.put(
        server_generation.QueuedGenerationRequest(
            rqueue=rqueue,
            raw_inputs={"token": 1},
            prompt_tokens=1,
            args=server.GenerationArguments(max_tokens=1),
        )
    )
    worker = Thread(target=gen._run, daemon=True)
    worker.start()
    try:
        ctx = rqueue.get(timeout=1)
        token = rqueue.get(timeout=1)
        done = rqueue.get(timeout=1)
    finally:
        gen._stop = True
        gen.requests.put(None)
        worker.join(timeout=2)

    assert isinstance(ctx, server.GenerationContext)
    assert token.text == "7"
    assert token.finish_reason == "length"
    assert done is None
    assert gen.draft_model is None
    assert gen.draft_kind is None


def test_ar_thread_exception_reaches_pending_client_queue(monkeypatch):
    class FakeBatchGenerator:
        def __init__(self, *_args, **_kwargs):
            self.has_work = False

        def close(self):
            pass

    gen = _unstarted_response_generator()

    def initialize_model():
        gen.model = SimpleNamespace(language_model=object())
        gen.processor = SimpleNamespace()
        gen.config = SimpleNamespace()
        gen.tokenizer = SimpleNamespace()

    error = RuntimeError("vision embedding failed")
    gen._initialize_model = initialize_model
    gen._gpu_embed = MagicMock(side_effect=error)
    monkeypatch.setattr(server_generation, "BatchGenerator", FakeBatchGenerator)

    rqueue = Queue()
    gen.requests.put(
        server_generation.QueuedGenerationRequest(
            rqueue=rqueue,
            raw_inputs={"input_ids": mx.array([[1]], dtype=mx.int32)},
            prompt_tokens=1,
            args=server.GenerationArguments(max_tokens=2),
        )
    )

    worker = Thread(target=gen._run, daemon=True)
    worker.start()
    try:
        assert rqueue.get(timeout=1) is error
        assert rqueue.get(timeout=1) is None
        assert worker.is_alive()
    finally:
        gen._stop = True
        gen.requests.put(None)
        worker.join(timeout=2)


def test_response_generator_diffusion_forwards_generation_options(monkeypatch):
    gen = _unstarted_response_generator()
    gen.model = SimpleNamespace()
    gen.processor = SimpleNamespace()
    gen.config = SimpleNamespace(eos_token_id=3)
    gen.tokenizer = SimpleNamespace(all_special_ids=[0])
    gen.prefill_step_size = 3072
    apc_manager = SimpleNamespace()
    gen.apc_manager = apc_manager
    gen.apc_mode = "exact"
    captured = {}

    def fake_stream_diffusion_generate_from_kwargs(
        model,
        processor,
        tokenizer,
        input_ids,
        pixel_values,
        attention_mask,
        skip_special_token_ids,
        kwargs,
        *,
        skip_special_tokens=False,
        on_result=None,
    ):
        captured.update(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            skip_special_token_ids=skip_special_token_ids,
            kwargs=dict(kwargs),
            skip_special_tokens=skip_special_tokens,
        )
        on_result(
            GenerationResult(
                text="ok",
                token=7,
                prompt_tokens=2,
                generation_tokens=1,
                total_tokens=3,
                prompt_tps=10.0,
                generation_tps=5.0,
                cached_tokens=1,
                finish_reason="length",
            )
        )
        if False:
            yield None

    monkeypatch.setattr(
        server_generation,
        "stream_diffusion_generate_from_kwargs",
        fake_stream_diffusion_generate_from_kwargs,
    )
    args = server.GenerationArguments(
        max_tokens=4,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=123,
        max_denoising_steps=7,
        block_length=16,
        num_to_transfer=3,
        max_transfer_per_step=2,
        editing_threshold=0.8,
        max_post_steps=5,
        stability_steps=1,
        diffusion_full_canvas=True,
        diffusion_min_canvas_length=4,
        diffusion_max_canvas_length=8,
        diffusion_sampler="entropy-bound",
        threshold=0.7,
        min_threshold=0.4,
    )
    rqueue = Queue()

    gen._generate_diffusion(
        uid=1,
        rqueue=rqueue,
        raw_inputs={
            "input_ids": mx.array([[11, 12]], dtype=mx.int32),
            "pixel_values": "pixels",
            "attention_mask": "mask",
            "mm_token_type_ids": "types",
        },
        args=args,
        cancelled=set(),
        apc_semantic_hash=73,
    )

    chunk = rqueue.get(timeout=1)
    assert chunk.text == "ok"
    assert chunk.finish_reason == "length"
    assert chunk.generation_tps == 5.0
    assert chunk.cached_tokens == 1
    assert captured["input_ids"].tolist() == [[11, 12]]
    assert captured["pixel_values"] == "pixels"
    assert captured["attention_mask"] == "mask"
    assert captured["skip_special_token_ids"] == {0}
    assert captured["kwargs"]["_apc_manager"] is apc_manager
    assert captured["kwargs"]["_apc_semantic_hash"] == 73
    assert captured["skip_special_tokens"] is True
    assert captured["kwargs"] == {
        "max_tokens": 4,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "mm_token_type_ids": "types",
        "prefill_step_size": 3072,
        "seed": 123,
        "max_denoising_steps": 7,
        "block_length": 16,
        "num_to_transfer": 3,
        "max_transfer_per_step": 2,
        "editing_threshold": 0.8,
        "max_post_steps": 5,
        "stability_steps": 1,
        "diffusion_full_canvas": True,
        "diffusion_min_canvas_length": 4,
        "diffusion_max_canvas_length": 8,
        "diffusion_sampler": "entropy-bound",
        "threshold": 0.7,
        "min_threshold": 0.4,
        "_apc_manager": apc_manager,
        "_apc_semantic_hash": 73,
    }


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/health"),
        ("get", "/metrics"),
        ("get", "/v1/metrics"),
        ("get", "/cache/stats"),
        ("get", "/v1/cache/stats"),
        ("post", "/cache/reset"),
        ("post", "/v1/cache/reset"),
        ("post", "/unload"),
    ],
)
def test_management_endpoints_allow_requests_without_configured_api_key(
    client, monkeypatch, method, path
):
    monkeypatch.delenv("MLX_VLM_SERVER_API_KEY", raising=False)

    response = getattr(client, method)(path)

    assert response.status_code == 200


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/health"),
        ("get", "/metrics"),
        ("get", "/v1/metrics"),
        ("get", "/cache/stats"),
        ("get", "/v1/cache/stats"),
        ("post", "/cache/reset"),
        ("post", "/v1/cache/reset"),
        ("post", "/unload"),
    ],
)
def test_management_endpoints_require_configured_api_key(
    client, monkeypatch, method, path
):
    monkeypatch.setenv("MLX_VLM_SERVER_API_KEY", "secret-token")

    missing = getattr(client, method)(path)
    invalid = getattr(client, method)(
        path,
        headers={"Authorization": "Bearer wrong-token"},
    )
    valid = getattr(client, method)(
        path,
        headers={"Authorization": "Bearer secret-token"},
    )

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert valid.status_code == 200


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/messages"),
        ("post", "/messages/count_tokens"),
        ("post", "/responses/input_tokens"),
        ("get", "/responses/missing"),
        ("delete", "/responses/missing"),
        ("post", "/responses/missing/cancel"),
        ("get", "/responses/missing/input_items"),
        ("post", "/responses"),
        ("post", "/chat/completions"),
        ("post", "/images/generations"),
        ("post", "/images/edits"),
        ("post", "/audio/speech"),
        ("post", "/audio/transcriptions"),
        ("post", "/audio/translations"),
        ("get", "/models"),
        ("post", "/v1/messages"),
        ("post", "/v1/messages/count_tokens"),
        ("post", "/v1/responses/input_tokens"),
        ("get", "/v1/responses/missing"),
        ("delete", "/v1/responses/missing"),
        ("post", "/v1/responses/missing/cancel"),
        ("get", "/v1/responses/missing/input_items"),
        ("post", "/v1/responses"),
        ("post", "/v1/chat/completions"),
        ("post", "/v1/images/generations"),
        ("post", "/v1/images/edits"),
        ("post", "/v1/audio/speech"),
        ("post", "/v1/audio/transcriptions"),
        ("post", "/v1/audio/translations"),
        ("get", "/v1/models"),
    ],
)
def test_inference_endpoints_require_configured_api_key(
    client, monkeypatch, method, path
):
    monkeypatch.setenv("MLX_VLM_SERVER_API_KEY", "secret-token")

    missing = getattr(client, method)(path)
    invalid = getattr(client, method)(
        path,
        headers={"Authorization": "Bearer wrong-token"},
    )

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert missing.headers["WWW-Authenticate"] == "Bearer"
    assert invalid.headers["WWW-Authenticate"] == "Bearer"


def test_inference_endpoint_accepts_configured_api_key(client, monkeypatch):
    monkeypatch.setenv("MLX_VLM_SERVER_API_KEY", "secret-token")

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret-token"},
        json={},
    )

    assert response.status_code == 422


def _fake_image_result(*, seed: int, output_path=None) -> ImageGenerationResult:
    image = Image.new("RGB", (16, 16), (seed % 255, 8, 16))
    data = ImageGenerationResult(
        array=mx.array(np.array(image)),
        seed=seed,
        width=16,
        height=16,
        steps=1,
        model="bonsai",
        family="bonsai",
        variant="ternary",
        guidance=1.0,
        peak_memory=0.0,
        prompt_tokens=5,
    )
    if output_path is not None:
        data.save(output_path)
    return data


def test_images_generations_returns_b64_json(client, monkeypatch):
    calls = []
    cache_calls = []

    def fake_get_cached_model(model, **kwargs):
        cache_calls.append((model, kwargs))
        return SimpleNamespace(), None, SimpleNamespace(model_type="bonsai")

    monkeypatch.setattr(
        server,
        "get_cached_model",
        fake_get_cached_model,
    )

    def fake_generate_image(model, request, **kwargs):
        calls.append(request)
        return _fake_image_result(seed=request.seed)

    monkeypatch.setattr(server_openai, "generate_image", fake_generate_image)

    response = client.post(
        "/v1/images/generations",
        json={
            "model": "bonsai-ternary",
            "prompt": "bonsai",
            "n": 2,
            "seed": 10,
            "size": "256x256",
            "steps": 1,
            "response_format": "b64_json",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["size"] == "256x256"
    assert [item["seed"] for item in payload["data"]] == [10, 11]
    assert all(item["b64_json"] for item in payload["data"])
    assert [call.seed for call in calls] == [10, 11]
    assert cache_calls == [("bonsai-ternary", {"model_kind": "image_generation"})]


def test_image_generation_lock_uses_image_cache_kind(monkeypatch):
    text_lock = object()
    image_lock = object()
    registry = server.ModelCacheRegistry()
    registry.set("text_generation", {"generation_lock": text_lock})
    registry.set("image_generation", {"generation_lock": image_lock})
    monkeypatch.setattr(server.runtime, "model_cache", registry)

    assert server_openai._runtime_cache_get("generation_lock") is text_lock
    assert (
        server_openai._runtime_cache_get("generation_lock", kind="image_generation")
        is image_lock
    )


def test_images_generations_forwards_prompt_expansion_model(client, monkeypatch):
    calls = []

    monkeypatch.setattr(
        server,
        "get_cached_model",
        lambda model, **kwargs: (
            SimpleNamespace(),
            None,
            SimpleNamespace(model_type="ideogram4"),
        ),
    )

    def fake_generate_image(model, request, **kwargs):
        calls.append(request)
        result = _fake_image_result(seed=request.seed)
        result.metadata["revised_prompt"] = '{"compositional_deconstruction":{}}'
        return result

    monkeypatch.setattr(server_openai, "generate_image", fake_generate_image)

    response = client.post(
        "/v1/images/generations",
        json={
            "model": "ideogram-ai/ideogram-4-fp8",
            "prompt": "A red cube.",
            "seed": 10,
            "size": "256x256",
            "steps": 1,
            "auto_json_caption": True,
            "prompt_expansion_model": "tiny-text-model",
            "response_format": "b64_json",
        },
    )

    assert response.status_code == 200
    assert calls[0].extra == {
        "auto_json_caption": True,
        "prompt_expansion_model": "tiny-text-model",
    }
    assert (
        response.json()["data"][0]["revised_prompt"]
        == '{"compositional_deconstruction":{}}'
    )


def test_images_generations_writes_paths(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        server,
        "get_cached_model",
        lambda model, **kwargs: (
            SimpleNamespace(),
            None,
            SimpleNamespace(model_type="bonsai"),
        ),
    )

    def fake_generate_image(model, request, **kwargs):
        return _fake_image_result(seed=request.seed, output_path=kwargs["output_path"])

    monkeypatch.setattr(server_openai, "generate_image", fake_generate_image)

    response = client.post(
        "/v1/images/generations",
        json={
            "model": "bonsai-ternary",
            "prompt": "bonsai",
            "n": 2,
            "seed": 20,
            "size": "256x256",
            "steps": 1,
            "response_format": "path",
            "output_dir": str(tmp_path),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    paths = [Path(item["path"]) for item in payload["data"]]
    assert [path.name for path in paths] == ["image-20.png", "image-21.png"]
    assert all(path.exists() for path in paths)
    assert all(item["b64_json"] is None for item in payload["data"])


def test_images_edits_returns_b64_json(client, monkeypatch):
    calls = []
    cache_calls = []

    def fake_get_cached_model(model, **kwargs):
        cache_calls.append((model, kwargs))
        return SimpleNamespace(), None, SimpleNamespace(model_type="flux2")

    monkeypatch.setattr(server, "get_cached_model", fake_get_cached_model)

    def fake_edit_image(model, request, **kwargs):
        calls.append((request, kwargs))
        return _fake_image_result(seed=request.seed)

    monkeypatch.setattr(server_openai, "edit_image", fake_edit_image)

    response = client.post(
        "/v1/images/edits",
        json={
            "model": "black-forest-labs/FLUX.2-klein-9b-kv",
            "prompt": "add sunglasses",
            "image": ["reference.png"],
            "n": 2,
            "seed": 30,
            "size": "256x256",
            "steps": 1,
            "response_format": "b64_json",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["size"] == "16x16"
    assert [item["seed"] for item in payload["data"]] == [30, 31]
    assert all(item["b64_json"] for item in payload["data"])
    assert [call[0].seed for call in calls] == [30, 31]
    assert calls[0][0].image_paths == ("reference.png",)
    assert cache_calls == [
        ("black-forest-labs/FLUX.2-klein-9b-kv", {"model_kind": "image_edit"})
    ]


def test_images_edits_writes_paths(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        server,
        "get_cached_model",
        lambda model, **kwargs: (
            SimpleNamespace(),
            None,
            SimpleNamespace(model_type="flux2"),
        ),
    )

    def fake_edit_image(model, request, **kwargs):
        return _fake_image_result(seed=request.seed, output_path=kwargs["output_path"])

    monkeypatch.setattr(server_openai, "edit_image", fake_edit_image)

    response = client.post(
        "/v1/images/edits",
        json={
            "model": "black-forest-labs/FLUX.2-klein-9b-kv",
            "prompt": "add sunglasses",
            "image": "reference.png",
            "n": 2,
            "seed": 40,
            "size": "256x256",
            "steps": 1,
            "response_format": "path",
            "output_dir": str(tmp_path),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    paths = [Path(item["path"]) for item in payload["data"]]
    assert [path.name for path in paths] == ["edit-40.png", "edit-41.png"]
    assert all(path.exists() for path in paths)
    assert all(item["b64_json"] is None for item in payload["data"])


def test_responses_endpoint_forwards_new_sampling_args(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", return_value=result) as mock_generate,
    ):
        response = client.post(
            "/responses",
            json={
                "model": "demo",
                "input": "Hello",
                "max_output_tokens": 12,
                "top_k": 40,
                "min_p": 0.08,
                "repetition_penalty": 1.15,
                "logit_bias": {"12": -1.5},
                "enable_thinking": False,
                "thinking_budget": 24,
                "thinking_start_token": "<think>",
                "thinking_end_token": "</think>",
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.kwargs["enable_thinking"] is False
    assert mock_template.call_args.kwargs["thinking_budget"] == 24
    assert mock_template.call_args.kwargs["thinking_start_token"] == "<think>"
    assert mock_template.call_args.kwargs["thinking_end_token"] == "</think>"
    assert mock_generate.call_args.kwargs["max_tokens"] == 12
    assert mock_generate.call_args.kwargs["top_k"] == 40
    assert mock_generate.call_args.kwargs["min_p"] == 0.08
    assert mock_generate.call_args.kwargs["repetition_penalty"] == 1.15
    assert mock_generate.call_args.kwargs["logit_bias"] == {12: -1.5}
    assert mock_generate.call_args.kwargs["enable_thinking"] is False
    assert mock_generate.call_args.kwargs["thinking_budget"] == 24
    assert mock_generate.call_args.kwargs["thinking_start_token"] == "<think>"
    assert mock_generate.call_args.kwargs["thinking_end_token"] == "</think>"


def test_responses_endpoint_merges_developer_message_with_instructions(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/responses",
            json={
                "model": "demo",
                "instructions": "Top-level instructions.",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Developer instructions.",
                            }
                        ],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello"}],
                    },
                ],
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.args[2] == [
        {
            "role": "system",
            "content": "Top-level instructions.\n\nDeveloper instructions.",
        },
        {"role": "user", "content": "Hello"},
    ]


def test_responses_endpoint_places_function_output_image_after_tool_result(
    client, monkeypatch
):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    image_url = "data:image/png;base64,ZmFrZS1pbWFnZQ=="
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "generate", return_value=result) as mock_generate,
    ):
        response = client.post(
            "/responses",
            json={
                "model": "demo",
                "input": [
                    {
                        "type": "function_call",
                        "name": "view_image",
                        "arguments": "{}",
                        "call_id": "call_view_image",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_view_image",
                        "output": [
                            {
                                "type": "input_image",
                                "image_url": image_url,
                                "detail": "high",
                            }
                        ],
                    },
                ],
            },
        )

    assert response.status_code == 200
    prompt = mock_generate.call_args.kwargs["prompt"]
    assert prompt.index("Tool:") < prompt.index("<image>")
    assert image_url not in prompt
    assert mock_generate.call_args.kwargs["image"] == [image_url]


def test_responses_endpoint_rejects_image_file_id(client):
    response = client.post(
        "/v1/responses",
        json={
            "model": "demo",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_view_image",
                    "output": [{"type": "input_image", "file_id": "file-image"}],
                }
            ],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "input_image.file_id is not supported by this server. "
        "Provide image_url instead."
    )


@pytest.mark.parametrize(
    ("include_adapter", "adapter_path", "expected_adapter"),
    [
        (False, None, server._INHERIT_ADAPTER),
        (True, "adapter-a", "adapter-a"),
        (True, None, None),
    ],
)
def test_responses_endpoint_forwards_adapter_path_or_inherits(
    client, monkeypatch, include_adapter, adapter_path, expected_adapter
):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=1,
        generation_tokens=1,
        total_tokens=2,
    )
    get_cached_model = MagicMock(return_value=(model, processor, config))
    payload = {"model": "demo", "input": "Hello"}
    if include_adapter:
        payload["adapter_path"] = adapter_path

    monkeypatch.setattr(server.runtime, "response_generator", None)
    monkeypatch.setattr(server, "get_cached_model", get_cached_model)
    monkeypatch.setattr(server, "apply_chat_template", MagicMock(return_value="prompt"))
    monkeypatch.setattr(server, "generate", MagicMock(return_value=result))

    response = client.post("/responses", json=payload)

    assert response.status_code == 200
    assert get_cached_model.call_args.args == ("demo", expected_adapter)


def test_responses_input_tokens_endpoint_forwards_adapter_path(client, monkeypatch):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    get_cached_model = MagicMock(return_value=(model, processor, config))
    response_generator = SimpleNamespace(
        _cpu_preprocess=MagicMock(
            return_value={"input_ids": mx.array([[1, 2, 3]], dtype=mx.int32)}
        )
    )

    monkeypatch.setattr(server.runtime, "response_generator", response_generator)
    monkeypatch.setattr(server, "get_cached_model", get_cached_model)
    monkeypatch.setattr(server, "apply_chat_template", MagicMock(return_value="prompt"))

    response = client.post(
        "/responses/input_tokens",
        json={"model": "demo", "input": "Hello", "adapter_path": "adapter-a"},
    )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 3}
    assert get_cached_model.call_args.args == ("demo", "adapter-a")


def test_responses_previous_response_id_replays_stored_items(client):
    server.response_store.clear()
    server.response_store_order.clear()
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    first = GenerationResult(text="First answer", prompt_tokens=3, generation_tokens=2)
    second = GenerationResult(
        text="Second answer", prompt_tokens=7, generation_tokens=2
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", side_effect=[first, second]),
    ):
        first_response = client.post(
            "/v1/responses", json={"model": "demo", "input": "First"}
        )
        assert first_response.status_code == 200
        previous_response_id = first_response.json()["id"]

        second_response = client.post(
            "/v1/responses",
            json={
                "model": "demo",
                "previous_response_id": previous_response_id,
                "input": "Second",
            },
        )

    assert second_response.status_code == 200
    replayed_messages = mock_template.call_args_list[1].args[2]
    assert replayed_messages == [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second"},
    ]
    retrieved = client.get(f"/v1/responses/{previous_response_id}")
    assert retrieved.status_code == 200
    input_items = client.get(f"/v1/responses/{previous_response_id}/input_items")
    assert input_items.status_code == 200
    assert input_items.json()["data"][0]["content"][0]["text"] == "First"


def test_responses_endpoint_returns_function_call_items(client):
    server.response_store.clear()
    server.response_store_order.clear()
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text='<tool_call>{"name":"get_weather","arguments":{"location":"SF"}}</tool_call>',
        prompt_tokens=8,
        generation_tokens=4,
    )
    tool_module = SimpleNamespace(
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        parse_tool_call=lambda call, tools: json.loads(call),
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", return_value=result),
        patch.object(server, "_infer_tool_parser_from_processor", return_value="demo"),
        patch.object(server, "load_tool_module", return_value=tool_module),
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "demo",
                "input": "weather?",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["output"][0]["type"] == "function_call"
    assert payload["output"][0]["name"] == "get_weather"
    assert payload["output"][0]["arguments"] == '{"location": "SF"}'
    assert (
        mock_template.call_args.kwargs["tools"][0]["function"]["name"] == "get_weather"
    )


def test_responses_endpoint_returns_reasoning_items(client):
    server.response_store.clear()
    server.response_store_order.clear()
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="<think>Check briefly.</think>\n\nDone.",
        prompt_tokens=8,
        generation_tokens=4,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/v1/responses", json={"model": "demo", "input": "hello"}
        )

    assert response.status_code == 200
    payload = response.json()
    assert [item["type"] for item in payload["output"]] == ["reasoning", "message"]
    assert payload["output"][0]["summary"][0]["text"] == "Check briefly."
    assert payload["output"][1]["content"][0]["text"] == "Done."
    assert payload["output_text"] == "Done."


def test_responses_endpoint_returns_native_shell_call_items(client):
    server.response_store.clear()
    server.response_store_order.clear()
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text='<tool_call>{"name":"shell","arguments":{"command":"pwd"}}</tool_call>',
        prompt_tokens=8,
        generation_tokens=4,
    )
    tool_module = SimpleNamespace(
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        parse_tool_call=lambda call, tools: json.loads(call),
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result),
        patch.object(server, "_infer_tool_parser_from_processor", return_value="demo"),
        patch.object(server, "load_tool_module", return_value=tool_module),
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "demo",
                "input": "run pwd",
                "tools": [{"type": "shell"}],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["output"][0]["type"] == "shell_call"
    assert payload["output"][0]["action"] == {"type": "exec", "command": "pwd"}


def _sse_events(body):
    events = []
    for block in body.split("\n\n"):
        event_type = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event_type = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
        if event_type and data:
            events.append((event_type, data))
    return events


def test_responses_streaming_emits_native_tool_call_items(client):
    server.response_store.clear()
    server.response_store_order.clear()
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    chunks = [
        GenerationResult(
            text='<tool_call>{"name":"shell","arguments":{"command":"pwd"}}</tool_call>',
            prompt_tokens=8,
            generation_tokens=4,
            prompt_tps=0.0,
            generation_tps=0.0,
            peak_memory=0.0,
        )
    ]
    tool_module = SimpleNamespace(
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        parse_tool_call=lambda call, tools: json.loads(call),
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "stream_generate", return_value=iter(chunks)),
        patch.object(server, "_infer_tool_parser_from_processor", return_value="demo"),
        patch.object(server, "load_tool_module", return_value=tool_module),
        patch.object(server.runtime, "response_generator", None),
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "demo",
                "input": "run pwd",
                "stream": True,
                "tools": [{"type": "shell"}],
            },
        )

    assert response.status_code == 200
    body = response.text
    assert '"type": "shell_call"' in body
    assert '"command": "pwd"' in body
    assert "<tool_call>" not in body


def test_responses_streaming_emits_reasoning_events(client):
    server.response_store.clear()
    server.response_store_order.clear()
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    chunks = [
        GenerationResult(text="<think>Check", prompt_tokens=8, generation_tokens=1),
        GenerationResult(
            text=" briefly.</think>\n\nDone.",
            prompt_tokens=8,
            generation_tokens=4,
            finish_reason="stop",
        ),
    ]

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "stream_generate", return_value=iter(chunks)),
        patch.object(server.runtime, "response_generator", None),
    ):
        response = client.post(
            "/v1/responses",
            json={"model": "demo", "input": "hello", "stream": True},
        )

    assert response.status_code == 200
    events = _sse_events(response.text)
    reasoning_events = [
        data
        for event_type, data in events
        if event_type == "response.reasoning_text.delta"
    ]
    text_delta_events = [
        data
        for event_type, data in events
        if event_type == "response.output_text.delta"
    ]
    done_event = next(
        data for event_type, data in events if event_type == "response.output_text.done"
    )
    completed = next(
        data["response"]
        for event_type, data in events
        if event_type == "response.completed"
    )

    assert "".join(event["delta"] for event in reasoning_events) == "Check briefly."
    assert "".join(event["delta"] for event in text_delta_events) == "Done."
    assert reasoning_events[0]["timings"]["predicted_per_second"] is None
    assert reasoning_events[1]["timings"]["predicted_per_second"] > 0
    assert (
        text_delta_events[0]["timings"]["predicted_per_second"]
        == reasoning_events[1]["timings"]["predicted_per_second"]
    )
    assert done_event["timings"]["predicted_per_second"] > 0
    assert [item["type"] for item in completed["output"]] == ["reasoning", "message"]
    assert completed["output"][0]["summary"][0]["text"] == "Check briefly."
    assert completed["output_text"] == "Done."


def test_responses_streaming_uses_prompt_opened_thinking_without_flag(client):
    server.response_store.clear()
    server.response_store_order.clear()
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="cohere2_moe")
    chunks = [
        GenerationResult(text="North reasoning.", prompt_tokens=8, generation_tokens=1),
        GenerationResult(
            text="<|END_THINKING|><|START_TEXT|>North answer.<|END_TEXT|>",
            prompt_tokens=8,
            generation_tokens=4,
            finish_reason="stop",
        ),
    ]
    template_kwargs = {}

    def fake_apply_chat_template(*args, **kwargs):
        template_kwargs.update(kwargs)
        return "prompt<|START_THINKING|>"

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server,
            "apply_chat_template",
            side_effect=fake_apply_chat_template,
        ),
        patch.object(server, "stream_generate", return_value=iter(chunks)),
        patch.object(server.runtime, "response_generator", None),
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "CohereLabs/North-Mini-Code-1.0-w4a16",
                "input": "hello",
                "reasoning": {"effort": "high", "summary": "auto"},
                "stream": True,
            },
        )

    assert response.status_code == 200
    events = _sse_events(response.text)
    reasoning = "".join(
        data["delta"]
        for event_type, data in events
        if event_type == "response.reasoning_text.delta"
    )
    content = "".join(
        data["delta"]
        for event_type, data in events
        if event_type == "response.output_text.delta"
    )

    assert reasoning == "North reasoning."
    assert content == "North answer."
    assert template_kwargs["enable_thinking"] is True
    assert template_kwargs["reasoning"] is True
    assert template_kwargs["reasoning_effort"] == "high"
    assert "<|END_THINKING|>" not in response.text
    assert "<|START_TEXT|>" not in response.text
    assert "<|END_TEXT|>" not in response.text


def test_responses_streaming_emits_function_call_arguments_done(client):
    server.response_store.clear()
    server.response_store_order.clear()
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    chunks = [
        GenerationResult(
            text='<tool_call>{"name":"get_weather","arguments":{"location":"SF"}}</tool_call>',
            prompt_tokens=8,
            generation_tokens=4,
            finish_reason="stop",
        )
    ]
    tool_module = SimpleNamespace(
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        parse_tool_call=lambda call, tools: json.loads(call),
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "stream_generate", return_value=iter(chunks)),
        patch.object(server, "_infer_tool_parser_from_processor", return_value="demo"),
        patch.object(server, "load_tool_module", return_value=tool_module),
        patch.object(server.runtime, "response_generator", None),
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "demo",
                "input": "weather?",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    events = _sse_events(response.text)
    done = next(
        data
        for event_type, data in events
        if event_type == "response.function_call_arguments.done"
    )
    assert done["item_id"].startswith("fc_")
    assert done["name"] == "get_weather"
    assert done["arguments"] == '{"location": "SF"}'


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
                "stream": True,
            },
        ),
        (
            "/v1/responses",
            {
                "model": "demo",
                "input": "Hello",
                "max_output_tokens": 4,
                "stream": True,
            },
        ),
    ],
)
def test_stream_endpoints_do_not_clear_mlx_cache_on_close(
    client, monkeypatch, path, payload
):
    class FakeResponseGenerator:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=3), iter(
                [
                    server.StreamingToken(
                        text="ok",
                        token=1,
                        logprobs=0.0,
                        finish_reason="stop",
                    )
                ]
            )

    calls = {"clear_cache": 0, "collect": 0}
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")

    monkeypatch.setattr(server.runtime, "response_generator", FakeResponseGenerator())
    monkeypatch.setattr(
        server, "get_cached_model", MagicMock(return_value=(model, processor, config))
    )
    monkeypatch.setattr(server, "apply_chat_template", MagicMock(return_value="prompt"))
    monkeypatch.setattr(
        server_openai.mx,
        "clear_cache",
        lambda: calls.__setitem__("clear_cache", calls["clear_cache"] + 1),
    )
    monkeypatch.setattr(
        server_openai.gc,
        "collect",
        lambda: calls.__setitem__("collect", calls["collect"] + 1),
    )

    response = client.post(path, json=payload)

    assert response.status_code == 200
    assert calls == {"clear_cache": 0, "collect": 0}


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
                "stream": True,
            },
        ),
        (
            "/v1/responses",
            {
                "model": "demo",
                "input": "Hello",
                "max_output_tokens": 4,
                "stream": True,
            },
        ),
    ],
)
def test_v1_stream_endpoints_reject_over_context_before_sse(
    client, monkeypatch, path, payload
):
    class OverBudgetResponseGenerator:
        generate_called = False

        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            raise server.PromptTooLongError(
                "Request needs 9 context tokens "
                "(5 prompt + 4 max generation), but MAX_KV_SIZE is 8."
            )

        def generate(self, *args, **kwargs):
            self.generate_called = True
            raise AssertionError("streaming should not start")

    response_generator = OverBudgetResponseGenerator()
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")

    monkeypatch.setattr(server.runtime, "metrics", server.ServerMetricsStore())
    monkeypatch.setattr(server.runtime, "response_generator", response_generator)
    monkeypatch.setattr(
        server, "get_cached_model", MagicMock(return_value=(model, processor, config))
    )
    monkeypatch.setattr(server, "apply_chat_template", MagicMock(return_value="prompt"))

    response = client.post(path, json=payload)

    assert response.status_code == 400
    assert "MAX_KV_SIZE is 8" in response.json()["detail"]
    assert response_generator.generate_called is False


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
            },
        ),
        (
            "/v1/responses",
            {
                "model": "demo",
                "input": "Hello",
                "max_output_tokens": 4,
            },
        ),
    ],
)
def test_v1_non_stream_endpoints_reject_over_context(
    client, monkeypatch, path, payload
):
    class OverBudgetResponseGenerator:
        def generate(self, *args, **kwargs):
            raise server.PromptTooLongError(
                "Request needs 9 context tokens "
                "(5 prompt + 4 max generation), but MAX_KV_SIZE is 8."
            )

    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")

    monkeypatch.setattr(server.runtime, "metrics", server.ServerMetricsStore())
    monkeypatch.setattr(
        server.runtime, "response_generator", OverBudgetResponseGenerator()
    )
    monkeypatch.setattr(
        server, "get_cached_model", MagicMock(return_value=(model, processor, config))
    )
    monkeypatch.setattr(server, "apply_chat_template", MagicMock(return_value="prompt"))

    response = client.post(path, json=payload)

    assert response.status_code == 400
    assert "MAX_KV_SIZE is 8" in response.json()["detail"]


def test_chat_completions_endpoint_forwards_explicit_sampling_args(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
        prompt_tps=10.0,
        generation_tps=5.0,
        peak_memory=0.1,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result) as mock_generate,
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 12,
                "top_k": 40,
                "min_p": 0.08,
                "repetition_penalty": 1.15,
                "logit_bias": {"12": -1.5},
                "resize_shape": [512],
            },
        )

    assert response.status_code == 200
    assert mock_generate.call_args.kwargs["max_tokens"] == 12
    assert mock_generate.call_args.kwargs["top_k"] == 40
    assert mock_generate.call_args.kwargs["min_p"] == 0.08
    assert mock_generate.call_args.kwargs["repetition_penalty"] == 1.15
    assert mock_generate.call_args.kwargs["logit_bias"] == {12: -1.5}
    assert mock_generate.call_args.kwargs["resize_shape"] == (512, 512)


def test_chat_completions_streaming_forwards_explicit_sampling_args(
    client, monkeypatch
):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    captured = {}

    class FakeResponseGenerator:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            captured["prompt"] = prompt
            captured["images"] = images
            captured["audio"] = audio
            captured["args"] = args
            return server.GenerationContext(uid=1, prompt_tokens=8), iter(
                [
                    server.StreamingToken(
                        text="done", token=1, logprobs=0.0, finish_reason="stop"
                    )
                ]
            )

    monkeypatch.setattr(server.runtime, "response_generator", FakeResponseGenerator())

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "max_tokens": 12,
                "top_k": 40,
                "min_p": 0.08,
                "repetition_penalty": 1.15,
                "logit_bias": {"12": -1.5},
            },
        )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert captured["args"].max_tokens == 12
    assert captured["args"].top_k == 40
    assert captured["args"].min_p == 0.08
    assert captured["args"].repetition_penalty == 1.15
    assert captured["args"].logit_bias == {12: -1.5}


def test_chat_completions_streaming_splits_gemma_thinking_channel_content(
    client, monkeypatch
):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="gemma4")

    class FakeResponseGenerator:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=8), iter(
                _gemma_thinking_channel_chunks()
            )

    monkeypatch.setattr(server.runtime, "response_generator", FakeResponseGenerator())

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "What's 7*8?"}],
                "stream": True,
                "enable_thinking": True,
            },
        )

    assert response.status_code == 200
    chunks = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    deltas = [
        chunk["choices"][0]["delta"]
        for chunk in chunks
        if chunk.get("choices") and chunk["choices"][0].get("delta")
    ]

    assert "".join(delta.get("content") or "" for delta in deltas) == "7 * 8 = 56"
    assert "".join(delta.get("reasoning_content") or "" for delta in deltas) == ""
    assert "<|channel>" not in response.text
    assert "<channel|>" not in response.text


def test_chat_completions_streaming_uses_custom_thinking_markers(client, monkeypatch):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="custom")

    class FakeResponseGenerator:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=8), iter(
                [
                    server.StreamingToken(
                        text="<analysis>Custom reasoning.</analysis>Custom answer.",
                        token=1,
                        logprobs=0.0,
                        finish_reason="stop",
                    )
                ]
            )

    monkeypatch.setattr(server.runtime, "response_generator", FakeResponseGenerator())

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "enable_thinking": True,
                "thinking_start_token": "<analysis>",
                "thinking_end_token": "</analysis>",
            },
        )

    assert response.status_code == 200
    chunks = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    deltas = [
        chunk["choices"][0]["delta"]
        for chunk in chunks
        if chunk.get("choices") and chunk["choices"][0].get("delta")
    ]

    assert "".join(delta.get("reasoning_content") or "" for delta in deltas) == (
        "Custom reasoning."
    )
    assert "".join(delta.get("reasoning") or "" for delta in deltas) == (
        "Custom reasoning."
    )
    assert "".join(delta.get("content") or "" for delta in deltas) == ("Custom answer.")


def test_chat_completions_streaming_uses_prompt_opened_thinking_without_flag(
    client, monkeypatch
):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="cohere2_moe")

    class FakeResponseGenerator:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=8), iter(
                [
                    server.StreamingToken(
                        text="North reasoning.",
                        token=1,
                        logprobs=0.0,
                        finish_reason=None,
                    ),
                    server.StreamingToken(
                        text="<|END_THINK",
                        token=2,
                        logprobs=0.0,
                        finish_reason=None,
                    ),
                    server.StreamingToken(
                        text="ING|><|START_TEXT|>North answer.<|END_TEXT|>",
                        token=3,
                        logprobs=0.0,
                        finish_reason="stop",
                    ),
                ]
            )

    monkeypatch.setattr(server.runtime, "response_generator", FakeResponseGenerator())

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server,
            "apply_chat_template",
            return_value="prompt<|START_THINKING|>",
        ),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "CohereLabs/North-Mini-Code-1.0-w4a16",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    chunks = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    deltas = [
        chunk["choices"][0]["delta"]
        for chunk in chunks
        if chunk.get("choices") and chunk["choices"][0].get("delta")
    ]

    assert "".join(delta.get("reasoning_content") or "" for delta in deltas) == (
        "North reasoning."
    )
    assert "".join(delta.get("reasoning") or "" for delta in deltas) == (
        "North reasoning."
    )
    assert "".join(delta.get("content") or "" for delta in deltas) == "North answer."
    assert "<|END_THINKING|>" not in response.text
    assert "<|START_TEXT|>" not in response.text
    assert "<|END_TEXT|>" not in response.text


def test_chat_completions_streaming_keeps_plain_output_as_content_when_thinking_enabled(
    client, monkeypatch
):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="lfm2_vl")

    class FakeResponseGenerator:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=8), iter(
                [
                    server.StreamingToken(
                        text="Hello", token=1, logprobs=0.0, finish_reason=None
                    ),
                    server.StreamingToken(
                        text="!", token=2, logprobs=0.0, finish_reason="stop"
                    ),
                ]
            )

    monkeypatch.setattr(server.runtime, "response_generator", FakeResponseGenerator())

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "liquidai/LFM2.5-VL-1.6B",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "enable_thinking": True,
            },
        )

    assert response.status_code == 200
    chunks = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    deltas = [
        chunk["choices"][0]["delta"]
        for chunk in chunks
        if chunk.get("choices") and chunk["choices"][0].get("delta")
    ]

    assert "".join(delta.get("content") or "" for delta in deltas) == "Hello!"
    assert "".join(delta.get("reasoning_content") or "" for delta in deltas) == ""
    assert "".join(delta.get("reasoning") or "" for delta in deltas) == ""


def test_chat_completions_response_uses_reasoning_content(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="custom")
    result = GenerationResult(
        text="<analysis>Custom reasoning.</analysis>Custom answer.",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
        prompt_tps=10.0,
        generation_tps=5.0,
        peak_memory=0.1,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "enable_thinking": True,
                "thinking_start_token": "<analysis>",
                "thinking_end_token": "</analysis>",
            },
        )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["reasoning_content"] == "Custom reasoning."
    assert message["reasoning"] == "Custom reasoning."
    assert message["content"] == "Custom answer."


def test_chat_completions_uses_processor_config_and_response_template(
    client, monkeypatch
):
    model = SimpleNamespace()
    config = SimpleNamespace(
        model_type="muse_glimmer",
        thinking_start_token="to=self<|message|>",
        thinking_end_token="<|eom|>",
    )
    processor = SimpleNamespace(
        config=config,
        tokenizer=_MuseResponseTemplateTokenizer(),
    )
    monkeypatch.delenv("MLX_VLM_THINKING_START_TOKEN", raising=False)
    monkeypatch.delenv("MLX_VLM_THINKING_END_TOKEN", raising=False)
    result = GenerationResult(
        text=(
            "to=self<|message|>Say exactly: hello world\n\n"
            "We need to say exactly: hello world ...\n\n"
            "No extra.<|eom|><|start|>assistant to=user<|message|>hello world"
        ),
        prompt_tokens=8,
        generation_tokens=32,
        total_tokens=40,
        prompt_tps=10.0,
        generation_tps=5.0,
        peak_memory=0.1,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="<|start|>assistant"),
        patch.object(server, "generate", return_value=result) as mock_generate,
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "mlx-community/Muse-Glimmer-30B-4bit",
                "messages": [{"role": "user", "content": "Say exactly: hello world"}],
            },
        )

    assert response.status_code == 200
    generate_kwargs = mock_generate.call_args.kwargs
    assert generate_kwargs["thinking_start_token"] == "to=self<|message|>"
    assert generate_kwargs["thinking_end_token"] == "<|eom|>"
    message = response.json()["choices"][0]["message"]
    expected_reasoning = (
        "Say exactly: hello world\n\n"
        "We need to say exactly: hello world ...\n\n"
        "No extra."
    )
    assert message["reasoning_content"] == expected_reasoning
    assert message["reasoning"] == expected_reasoning
    assert message["content"] == "hello world"


@pytest.mark.parametrize(
    "audio_data_factory",
    [
        lambda raw: base64.b64encode(raw).decode("ascii"),
        lambda raw: f"data:audio/wav;base64,{base64.b64encode(raw).decode('ascii')}",
    ],
)
def test_chat_completions_decodes_input_audio_base64(client, audio_data_factory):
    raw_audio = b"RIFF$\x00\x00\x00WAVEfmt "
    captured = {}

    def fake_generate(prompt, images=None, audio=None, **kwargs):
        captured["audio"] = audio
        return GenerationResult(
            text="audio ok",
            prompt_tokens=8,
            generation_tokens=4,
            total_tokens=12,
            prompt_tps=10.0,
            generation_tps=5.0,
            peak_memory=0.1,
        )

    with (
        patch.object(
            server,
            "get_cached_model",
            return_value=(
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(model_type="qwen2_vl"),
            ),
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", side_effect=fake_generate),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the audio."},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_data_factory(raw_audio),
                                    "format": "wav",
                                },
                            },
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert captured["audio"][0].getvalue() == raw_audio


def test_chat_completions_preserves_input_audio_references(client):
    audio_path = "/tmp/audio.wav"
    captured = {}

    def fake_generate(prompt, images=None, audio=None, **kwargs):
        captured["audio"] = audio
        return GenerationResult(
            text="audio ok",
            prompt_tokens=8,
            generation_tokens=4,
            total_tokens=12,
            prompt_tps=10.0,
            generation_tps=5.0,
            peak_memory=0.1,
        )

    with (
        patch.object(
            server,
            "get_cached_model",
            return_value=(
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(model_type="qwen2_vl"),
            ),
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", side_effect=fake_generate),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the audio."},
                            {
                                "type": "input_audio",
                                "input_audio": {"data": audio_path, "format": "wav"},
                            },
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert captured["audio"] == [audio_path]


def test_generation_timings_from_metrics():
    metrics = SimpleNamespace(
        cached_tokens=2,
        prompt_tps=20.0,
        generation_tps=8.0,
        token_times=[],
        peak_memory=0.5,
    )
    timings = server.GenerationTimings.from_metrics(metrics, 10, 4)

    assert (timings.prompt_n, timings.cache_n, timings.predicted_n) == (8, 2, 4)
    assert timings.prompt_ms == pytest.approx(500.0)
    assert timings.prompt_per_token_ms == pytest.approx(62.5)
    assert timings.prompt_per_second == pytest.approx(16.0)
    assert timings.predicted_ms == pytest.approx(500.0)
    assert timings.predicted_per_token_ms == pytest.approx(125.0)
    assert timings.predicted_per_second == pytest.approx(8.0)
    assert timings.peak_memory == pytest.approx(0.5)

    metrics = SimpleNamespace(
        cached_tokens=9,
        prompt_tps=None,
        generation_tps=None,
        token_times=[],
        peak_memory=0.0,
    )
    timings = server.GenerationTimings.from_metrics(metrics, 4, 1)
    assert timings.prompt_n == 0
    assert timings.prompt_ms == 0.0
    assert timings.prompt_per_token_ms == 0.0
    assert timings.predicted_ms == 0.0
    assert timings.predicted_per_token_ms == 0.0


def test_generation_metrics_reports_chunk_and_aggregate_rates():
    metrics = server_generation.GenerationMetrics()

    first_rate = metrics.record_chunk(
        SimpleNamespace(generation_tokens=1, emitted_at=10.0)
    )
    second_rate = metrics.record_chunk(
        SimpleNamespace(generation_tokens=4, emitted_at=10.25)
    )

    assert first_rate is None
    assert second_rate == pytest.approx(12.0)
    assert metrics.rate == pytest.approx(12.0)


def test_generation_timings_include_speculative_stats():
    metrics = SimpleNamespace(
        cached_tokens=0,
        prompt_tps=20.0,
        generation_tps=8.0,
        token_times=[],
        peak_memory=0.0,
        draft_kind="mtp",
        draft_rounds=5,
        draft_n_accepted=12,
        draft_n=20,
    )
    timings = server.GenerationTimings.from_metrics(metrics, 10, 17)

    assert timings.draft_kind == "mtp"
    assert timings.draft_rounds == 5
    assert timings.draft_n_accepted == 12
    assert timings.draft_n == 20
    assert timings.draft_n_accepted / timings.draft_n == pytest.approx(0.6)


def test_generation_timings_speculative_stats_default_to_none():
    metrics = SimpleNamespace(
        cached_tokens=0,
        prompt_tps=20.0,
        generation_tps=8.0,
        token_times=[],
        peak_memory=0.0,
    )
    timings = server.GenerationTimings.from_metrics(metrics, 10, 4)

    assert timings.draft_kind is None
    assert timings.draft_rounds is None
    assert timings.draft_n_accepted is None
    assert timings.draft_n is None


def test_generation_metrics_record_speculative_stats():
    metrics = server_generation.GenerationMetrics()

    metrics.record_chunk(SimpleNamespace(generation_tokens=1, emitted_at=10.0))
    metrics.record_chunk(
        SimpleNamespace(
            generation_tokens=6,
            emitted_at=10.5,
            draft_kind="dflash",
            draft_rounds=3,
            draft_n_accepted=4,
            draft_n=9,
        )
    )

    assert metrics.draft_kind == "dflash"
    assert metrics.draft_rounds == 3
    assert metrics.draft_n_accepted == 4
    assert metrics.draft_n == 9


def test_speculative_lifetime_counters_survive_reset():
    from mlx_vlm.speculative.common import (
        _record_speculative_round,
        speculative_stats_since,
        speculative_stats_snapshot,
    )

    drafter = SimpleNamespace(accept_lens=[], draft_lens=[])

    assert speculative_stats_since(drafter, speculative_stats_snapshot(drafter)) == (
        None,
        None,
        None,
    )

    snapshot = speculative_stats_snapshot(drafter)
    _record_speculative_round(drafter, 3, 7)
    _record_speculative_round(drafter, 2.5, 7)
    drafter.accept_lens = []
    drafter.draft_lens = []
    _record_speculative_round(drafter, 1.5, 7)

    rounds, accepted, drafted = speculative_stats_since(drafter, snapshot)
    assert (rounds, accepted, drafted) == (3, 7, 21)

    later_snapshot = speculative_stats_snapshot(drafter)
    _record_speculative_round(drafter, 2, 7)
    rounds, accepted, drafted = speculative_stats_since(drafter, later_snapshot)
    assert (rounds, accepted, drafted) == (1, 2, 7)


def test_chat_completions_returns_timings(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=10,
        generation_tokens=4,
        prompt_tps=20.0,
        generation_tps=8.0,
        peak_memory=0.1,
        cached_tokens=2,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 12,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 2
    assert (body["timings"]["cache_n"], body["timings"]["prompt_n"]) == (2, 8)
    assert body["timings"]["predicted_per_second"] == 8.0


def test_chat_completions_streaming_emits_timings_on_finish(client, monkeypatch):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")

    class FakeResponseGenerator:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=10), iter(
                [
                    server.StreamingToken(
                        text="hi",
                        token=1,
                        logprobs=0.0,
                        finish_reason=None,
                        prompt_tps=20.0,
                        cached_tokens=2,
                    ),
                    server.StreamingToken(
                        text="!",
                        token=2,
                        logprobs=0.0,
                        finish_reason="stop",
                        prompt_tps=20.0,
                        cached_tokens=2,
                    ),
                ]
            )

    monkeypatch.setattr(server.runtime, "response_generator", FakeResponseGenerator())

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

    assert response.status_code == 200
    chunks = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    timed_chunk = next(chunk for chunk in chunks if chunk.get("usage") is not None)
    assert timed_chunk["choices"] == []
    assert timed_chunk["timings"]["cache_n"] == 2
    assert timed_chunk["usage"]["prompt_tokens_details"]["cached_tokens"] == 2
    token_chunks = [
        chunk
        for chunk in chunks
        if chunk["choices"] and chunk["choices"][0]["delta"].get("content") is not None
    ]
    assert token_chunks[0]["timings"]["predicted_per_second"] is None
    assert token_chunks[1]["timings"]["predicted_per_second"] > 0
    terminal_chunk = next(
        chunk
        for chunk in chunks
        if chunk["choices"] and chunk["choices"][0]["finish_reason"] == "stop"
    )
    assert terminal_chunk["timings"]["predicted_per_second"] > 0
    assert (
        timed_chunk["timings"]["predicted_per_second"]
        == terminal_chunk["timings"]["predicted_per_second"]
    )


def test_chat_completions_streaming_response_template_tool_calls(client, monkeypatch):
    model = SimpleNamespace()
    processor = SimpleNamespace(tokenizer=_MuseResponseTemplateTokenizer())
    config = SimpleNamespace(model_type="muse_glimmer")

    class FakeResponseGenerator:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=10), iter(
                [
                    server.StreamingToken(
                        text=(
                            "to=self<|message|>I need the weather tool.<|eom|>"
                            "<|start|>assistant to=get_weather<|message|>"
                            '<atem:function_calls><atem:invoke name="get_weather">'
                            '<atem:parameter name="city">Warsaw</atem:parameter>'
                            "</atem:invoke></atem:function_calls>"
                        ),
                        token=1,
                        logprobs=0.0,
                        finish_reason="stop",
                        prompt_tps=20.0,
                        cached_tokens=2,
                    )
                ]
            )

    from mlx_vlm.tools.parsers import atem as tool_module

    monkeypatch.setattr(server.runtime, "response_generator", FakeResponseGenerator())

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
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
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

    assert response.status_code == 200
    chunks = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    tool_chunk = next(
        chunk
        for chunk in chunks
        if chunk["choices"] and chunk["choices"][0]["finish_reason"] == "tool_calls"
    )
    usage_chunk = next(chunk for chunk in chunks if chunk.get("usage") is not None)
    reasoning = "".join(
        chunk["choices"][0]["delta"].get("reasoning_content") or ""
        for chunk in chunks
        if chunk["choices"]
    )
    content = "".join(
        chunk["choices"][0]["delta"].get("content") or ""
        for chunk in chunks
        if chunk["choices"]
    )
    tool_call = tool_chunk["choices"][0]["delta"]["tool_calls"][0]

    assert tool_chunk.get("usage") is None
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "Warsaw"}
    assert reasoning == "I need the weather tool."
    assert content == ""
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["prompt_tokens_details"]["cached_tokens"] == 2


def test_chat_completions_endpoint_flattens_text_content_parts(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
        prompt_tps=10.0,
        generation_tps=5.0,
        peak_memory=0.1,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "First text block."},
                            {"type": "text", "text": "Second text block."},
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.args[2] == [
        {
            "role": "user",
            "content": "First text block. Second text block.",
        }
    ]


def test_chat_completions_endpoint_forwards_native_video_content(client):
    model = SimpleNamespace()
    processor = SimpleNamespace(
        video_processor=SimpleNamespace(),
        process=lambda text=None, images=None, videos=None, **kwargs: None,
    )
    config = SimpleNamespace(model_type="gemma4")
    result = GenerationResult(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
        prompt_tps=10.0,
        generation_tps=5.0,
        peak_memory=0.1,
    )
    from mlx_vlm.generate import video as video_module

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", return_value=result) as mock_generate,
        patch.object(video_module, "sample_video_frames") as mock_sample,
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video_url", "video_url": {"url": "clip.mp4"}},
                            {"type": "text", "text": "Describe this video."},
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.kwargs["video"] == ["clip.mp4"]
    assert mock_template.call_args.args[2] == [
        {"role": "user", "content": "Describe this video."}
    ]
    assert mock_generate.call_args.kwargs["video"] == ["clip.mp4"]
    mock_sample.assert_not_called()


def test_chat_completions_endpoint_falls_back_from_video_to_images(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="mage_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
        prompt_tps=10.0,
        generation_tps=5.0,
        peak_memory=0.1,
    )
    frames = [object(), object()]
    from mlx_vlm.generate import video as video_module

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", return_value=result) as mock_generate,
        patch.object(
            video_module,
            "sample_video_frames",
            return_value=(frames, 2.0),
        ) as mock_sample,
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video_url", "video_url": {"url": "clip.mp4"}},
                            {"type": "text", "text": "Describe this video."},
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.kwargs["num_images"] == 2
    assert mock_template.call_args.kwargs["video"] is None
    assert mock_generate.call_args.kwargs["image"] == frames
    assert mock_generate.call_args.kwargs["video"] == []
    mock_sample.assert_called_once_with(["clip.mp4"], 2.0, None)


def test_chat_completions_endpoint_preserves_assistant_reasoning_content(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
        prompt_tps=10.0,
        generation_tps=5.0,
        peak_memory=0.1,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {
                        "role": "assistant",
                        "content": "Hello",
                        "reasoning_content": "Prior thought",
                    },
                    {"role": "user", "content": "Continue"},
                ],
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.args[2][1] == {
        "role": "assistant",
        "content": "Hello",
        "reasoning_content": "Prior thought",
        "reasoning": "Prior thought",
    }


def test_anthropic_messages_endpoint_accepts_system_role_in_messages(
    client, monkeypatch
):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(text="done", prompt_tokens=4, generation_tokens=2)

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "demo",
                "system": "Use short answers.",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "Be precise."}],
                    },
                    {"role": "user", "content": "Introduce the project."},
                ],
                "max_tokens": 12,
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.args[2] == [
        {"role": "system", "content": "Use short answers."},
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "Be precise."},
        {"role": "user", "content": "Introduce the project."},
    ]


def test_anthropic_messages_usage_reports_cached_tokens(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=10,
        generation_tokens=4,
        cached_tokens=6,
        prompt_tps=20.0,
        generation_tps=8.0,
        peak_memory=0.1,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert response.json()["usage"] == {
        "input_tokens": 4,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 6,
        "output_tokens": 4,
    }


def test_anthropic_nonstreaming_preserves_thinking_with_tool_use(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    config = SimpleNamespace(
        model_type="muse_glimmer",
        thinking_start_token="to=self<|message|>",
        thinking_end_token="<|eom|>",
    )
    processor = SimpleNamespace(
        config=config,
        tokenizer=_MuseResponseTemplateTokenizer(),
    )
    result = GenerationResult(
        text=(
            "to=self<|message|>I need the weather tool.<|eom|>"
            "<|start|>assistant to=get_weather<|message|>"
            '<atem:function_calls><atem:invoke name="get_weather">'
            '<atem:parameter name="city">Warsaw</atem:parameter>'
            "</atem:invoke></atem:function_calls>"
        ),
        prompt_tokens=7,
        generation_tokens=6,
        prompt_tps=0.0,
        generation_tps=0.0,
        peak_memory=0.0,
    )
    from mlx_vlm.tools.parsers import atem as tool_module

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result),
        patch.object(server, "_infer_tool_parser_from_processor", return_value="demo"),
        patch.object(server, "load_tool_module", return_value=tool_module),
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
                "thinking": {"type": "enabled", "budget_tokens": 4},
                "max_tokens": 8,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["stop_reason"] == "tool_use"
    assert payload["content"][0] == {
        "type": "thinking",
        "thinking": "I need the weather tool.",
        "signature": "",
    }
    assert payload["content"][1]["type"] == "tool_use"
    assert payload["content"][1]["name"] == "get_weather"
    assert payload["content"][1]["input"] == {"city": "Warsaw"}
    assert "to=self" not in response.text
    assert "<atem:" not in response.text


def test_anthropic_messages_streaming_uses_anthropic_events(client, monkeypatch):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")

    class FakeResponseGenerator:
        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=3), iter(
                [
                    server.StreamingToken(
                        text="Hel",
                        token=1,
                        logprobs=0.0,
                        finish_reason=None,
                        cached_tokens=2,
                    ),
                    server.StreamingToken(
                        text="lo",
                        token=2,
                        logprobs=0.0,
                        finish_reason="stop",
                        cached_tokens=2,
                    ),
                ]
            )

    monkeypatch.setattr(server.runtime, "response_generator", FakeResponseGenerator())

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
                "stream": True,
            },
        )

    assert response.status_code == 200
    body = response.text
    assert "event: message_start" in body
    assert "event: content_block_start" in body
    assert "event: content_block_delta" in body
    assert '"text": "Hel"' in body
    assert "event: message_delta" in body
    assert '"stop_reason": "end_turn"' in body
    assert '"cache_read_input_tokens": 2' in body
    assert '"input_tokens": 1' in body
    assert "event: message_stop" in body


def test_anthropic_messages_streaming_splits_gemma_thinking_channel_content(
    client, monkeypatch
):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="gemma4")

    class FakeResponseGenerator:
        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=3), iter(
                _gemma_thinking_channel_chunks()
            )

    monkeypatch.setattr(server.runtime, "response_generator", FakeResponseGenerator())

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "What's 7*8?"}],
                "max_tokens": 16,
                "stream": True,
                "enable_thinking": True,
            },
        )

    assert response.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    deltas = [
        event["delta"] for event in events if event.get("type") == "content_block_delta"
    ]

    assert "".join(delta.get("text") or "" for delta in deltas) == "7 * 8 = 56"
    assert "".join(delta.get("thinking") or "" for delta in deltas) == ""
    assert "<|channel>" not in response.text
    assert "<channel|>" not in response.text


def test_anthropic_messages_streaming_uses_custom_thinking_markers(client, monkeypatch):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="custom")

    class FakeResponseGenerator:
        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=3), iter(
                [
                    server.StreamingToken(
                        text="<analysis>Custom reasoning.</analysis>Custom answer.",
                        token=1,
                        logprobs=0.0,
                        finish_reason="stop",
                    )
                ]
            )

    monkeypatch.setattr(server.runtime, "response_generator", FakeResponseGenerator())

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 16,
                "stream": True,
                "enable_thinking": True,
                "thinking_start_token": "<analysis>",
                "thinking_end_token": "</analysis>",
            },
        )

    assert response.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    deltas = [
        event["delta"] for event in events if event.get("type") == "content_block_delta"
    ]

    assert "".join(delta.get("thinking") or "" for delta in deltas) == (
        "Custom reasoning."
    )
    assert "".join(delta.get("text") or "" for delta in deltas) == "Custom answer."


ANTHROPIC_TOOLS = [
    {
        "name": "get_time",
        "description": "Get the current time",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_weather",
        "description": "Get the current weather",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def _anthropic_tool_choice_request(client, tool_choice, tools=ANTHROPIC_TOOLS):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(text="done", prompt_tokens=5, generation_tokens=2)

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": tools,
                "tool_choice": tool_choice,
                "max_tokens": 32,
            },
        )
    return response, mock_template


def test_anthropic_messages_tool_choice_none_disables_tools(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)

    response, mock_template = _anthropic_tool_choice_request(client, {"type": "none"})

    assert response.status_code == 200
    assert mock_template.call_args.kwargs["tools"] is None
    assert mock_template.call_args.kwargs["tool_choice"] == "none"


def test_anthropic_messages_any_tool_choice_adds_instruction(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)

    response, mock_template = _anthropic_tool_choice_request(client, {"type": "any"})

    assert response.status_code == 200
    messages = mock_template.call_args.args[2]
    assert "must call one or more" in messages[-1]["content"]
    selected_tools = mock_template.call_args.kwargs["tools"]
    assert [tool["function"]["name"] for tool in selected_tools] == [
        "get_time",
        "get_weather",
    ]
    assert mock_template.call_args.kwargs["tool_choice"] == "required"


def test_anthropic_messages_forced_tool_choice_filters_tools(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)

    response, mock_template = _anthropic_tool_choice_request(
        client, {"type": "tool", "name": "get_time"}
    )

    assert response.status_code == 200
    messages = mock_template.call_args.args[2]
    assert "must call the 'get_time' function" in messages[-1]["content"]
    selected_tools = mock_template.call_args.kwargs["tools"]
    assert [tool["function"]["name"] for tool in selected_tools] == ["get_time"]
    assert mock_template.call_args.kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_time"},
    }


@pytest.mark.parametrize(
    ("tool_choice", "tools", "message"),
    [
        (
            {"type": "tool", "name": "missing"},
            ANTHROPIC_TOOLS,
            "unknown function 'missing'",
        ),
        ({"type": "any"}, [], "requires at least one tool"),
    ],
)
def test_anthropic_messages_rejects_unsatisfiable_tool_choice(
    client, monkeypatch, tool_choice, tools, message
):
    monkeypatch.setattr(server.runtime, "response_generator", None)

    response, _ = _anthropic_tool_choice_request(client, tool_choice, tools=tools)

    assert response.status_code == 400
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"
    assert message in payload["error"]["message"]


def test_anthropic_count_tokens_applies_tool_choice(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server_anthropic, "prepare_inputs", return_value={}),
        patch.object(server_anthropic, "_count_prompt_tokens", return_value=7),
    ):
        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": ANTHROPIC_TOOLS,
                "tool_choice": {"type": "tool", "name": "get_time"},
            },
        )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 7}
    selected_tools = mock_template.call_args.kwargs["tools"]
    assert [tool["function"]["name"] for tool in selected_tools] == ["get_time"]


def test_cache_endpoints_report_disabled_stats_and_reset(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "apc_manager", None)

    response = client.get("/v1/cache/stats")
    assert response.status_code == 200
    assert response.json() == {"enabled": False}

    response = client.post("/v1/cache/reset")
    assert response.status_code == 200
    assert response.json() == {"enabled": False}

    manager = SimpleNamespace(
        stats_snapshot=MagicMock(return_value={"hits": 2, "pool_used": 1}),
        clear=MagicMock(),
    )
    monkeypatch.setattr(server.runtime, "apc_manager", manager)

    response = client.get("/v1/cache/stats")
    assert response.status_code == 200
    assert response.json() == {"hits": 2, "pool_used": 1, "enabled": True}

    response = client.post("/v1/cache/reset")
    assert response.status_code == 200
    assert response.json() == {"enabled": True, "status": "cleared"}
    manager.clear.assert_called_once_with()


def test_metrics_endpoint_reports_empty_state(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "metrics", server.ServerMetricsStore())
    monkeypatch.setattr(server.runtime, "apc_manager", None)
    monkeypatch.setattr(server.runtime, "response_generator", None)
    monkeypatch.setattr(server.runtime, "model_cache", {})

    response = client.get("/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["latest"] is None
    assert payload["recent"] == []
    assert payload["summary"]["requests_started"] == 0
    assert payload["summary"]["requests_completed"] == 0
    assert payload["summary"]["requests_failed"] == 0
    assert payload["server"]["loaded_model"] is None
    assert payload["server"]["apc"] == {"enabled": False}


def test_metrics_store_logs_request_lifecycle(caplog):
    caplog.set_level(logging.INFO, logger="mlx_vlm.server")
    metrics = server.ServerMetricsStore()
    metrics.begin_request(endpoint="/chat/completions", model="demo", stream=True)
    metrics.record_success(
        {
            "endpoint": "/chat/completions",
            "model": "demo",
            "stream": True,
            "backend": "continuous_batching",
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "generated_tokens": 4,
            "request_elapsed_s": 0.5,
            "decode_elapsed_s": 0.1,
            "prefill_tok_s": 100.0,
            "decode_tok_s": 40.0,
            "finish_reason": "stop",
        }
    )

    assert "Request started: endpoint=/chat/completions model=demo" in caplog.text
    assert "Request completed: endpoint=/chat/completions model=demo" in caplog.text
    assert "prefill=100.0 tok/s decode=40.0 tok/s" in caplog.text


def test_metrics_endpoint_records_chat_completion_metrics(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "metrics", server.ServerMetricsStore())
    monkeypatch.setattr(server.runtime, "apc_manager", None)
    monkeypatch.setattr(server.runtime, "response_generator", None)

    config = SimpleNamespace(
        text_config=SimpleNamespace(max_position_embeddings=4096),
    )
    processor = SimpleNamespace()
    model = SimpleNamespace()
    monkeypatch.setattr(
        server.runtime,
        "model_cache",
        {
            "model_path": "demo-model",
            "adapter_path": None,
            "config": config,
            "processor": processor,
        },
    )
    monkeypatch.setattr(
        server,
        "get_cached_model",
        MagicMock(return_value=(model, processor, config)),
    )
    monkeypatch.setattr(server, "apply_chat_template", MagicMock(return_value="prompt"))
    monkeypatch.setattr(
        server,
        "generate",
        MagicMock(
            return_value=GenerationResult(
                text="Hello there",
                prompt_tokens=12,
                generation_tokens=5,
                prompt_tps=120.0,
                generation_tps=50.0,
                peak_memory=1.25,
            )
        ),
    )

    response = client.post(
        "/chat/completions",
        json={
            "model": "demo-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    payload = metrics.json()

    latest = payload["latest"]
    assert latest["endpoint"] == "/chat/completions"
    assert latest["model"] == "demo-model"
    assert latest["stream"] is False
    assert latest["backend"] == "generate"
    assert latest["prompt_tokens"] == 12
    assert latest["completion_tokens"] == 5
    assert latest["generated_tokens"] == 5
    assert latest["prefill_tok_s"] == 120.0
    assert latest["decode_tok_s"] == 50.0
    assert latest["peak_memory_gb"] == 1.25
    assert latest["image_count"] == 0
    assert latest["audio_count"] == 0
    assert latest["apc_enabled"] is False

    assert len(payload["recent"]) == 1
    assert payload["summary"]["requests_started"] == 1
    assert payload["summary"]["requests_completed"] == 1
    assert payload["summary"]["requests_failed"] == 0
    assert payload["summary"]["prompt_tokens_total"] == 12
    assert payload["summary"]["completion_tokens_total"] == 5
    assert payload["summary"]["generated_tokens_total"] == 5
    assert payload["server"]["loaded_model"] == "demo-model"
    assert payload["server"]["loaded_context_size"] == 4096


# ── Continuous batching / ResponseGenerator tests ─────────────────────


class TestResponseGenerator:
    """Tests for the ResponseGenerator continuous batching engine."""

    # Ported from upstream (2026-09-27 sync): --model-discovery was removed upstream (53616323).
    def test_server_cli_sets_thinking_defaults(self, monkeypatch):
        flags = [
            ("model", "PRELOAD_MODEL", "demo"),
            ("image-model", "PRELOAD_IMAGE_MODEL", "image-demo"),
            ("tts-model", "PRELOAD_TTS_MODEL", "tts-demo"),
            ("stt-model", "PRELOAD_STT_MODEL", "stt-demo"),
            ("decision-model", "PRELOAD_DECISION_MODEL", "decision-demo"),
            ("reranker-model", "PRELOAD_RERANKER_MODEL", "reranker-demo"),
            ("thinking-budget", "THINKING_BUDGET", "128"),
            ("thinking-start-token", "THINKING_START_TOKEN", "<|START_THINKING|>"),
            ("thinking-eos-token", "THINKING_END_TOKEN", "<|END_THINKING|>"),
            ("api-key", "SERVER_API_KEY", "admin-token"),
        ]
        expected = {"MLX_VLM_" + env: value for _, env, value in flags}
        expected["MLX_VLM_ENABLE_THINKING"] = "1"
        argv = [
            "mlx_vlm.server",
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
            "--enable-thinking",
        ]
        argv += [arg for flag, _, value in flags for arg in ("--" + flag, value)]
        monkeypatch.setattr(sys, "argv", argv)
        with patch.dict(os.environ), patch.object(cli.uvicorn, "run") as run:
            for key in [
                *expected,
                "MLX_VLM_PRELOAD_ADAPTER",
                "MLX_VLM_VISION_CACHE_SIZE",
                "MLX_VLM_MAX_TOKENS",
                "PREFILL_STEP_SIZE",
                "KV_GROUP_SIZE",
                "KV_QUANT_SCHEME",
                "QUANTIZED_KV_START",
            ]:
                os.environ.pop(key, None)
            cli.main()
            _assert_fields(os.environ, **expected)
            assert run.call_args.kwargs["host"] == "127.0.0.1"

    def _bare_generator(self):
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.draft_model = None
        gen.wait_until_ready = lambda: None
        gen._cpu_preprocess = lambda prompt, images, audio: {"input_ids": [1, 2, 3]}
        return gen

    def test_generate_rejects_requests_over_configured_context_limit(self, monkeypatch):
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.wait_until_ready = lambda: None
        gen.draft_model = None
        gen.apc_manager = object()
        gen.apc_mode = "block"
        gen._preprocess_request = lambda prompt, images, audio, videos: {
            "input_ids": mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32),
            "pixel_values": mx.zeros((1, 3, 2, 2), dtype=mx.float32),
        }
        gen.requests = Queue()
        image_hash = MagicMock(wraps=apc_module.hash_image_payload)
        monkeypatch.setattr(apc_module, "hash_image_payload", image_hash)

        monkeypatch.setenv("MAX_KV_SIZE", "8")

        # Fork: upstream matches "MAX_KV_SIZE is 8" because its `generate` calls
        # `_check_configured_context_budget`. This fork calls
        # `_apply_generation_budget`, which reports the shortfall against
        # MIN_OUTPUT_TOKENS instead. Different message, same rejection — do not
        # "converge" this string without converging the budget helper too.
        with pytest.raises(server.PromptTooLongError, match="MIN_OUTPUT_TOKENS"):
            gen.generate("prompt", args=server.GenerationArguments(max_tokens=4))

        assert gen.requests.empty()
        image_hash.assert_not_called()

    def test_generate_serializes_budget_criteria_with_tokenizer_preprocessing(self):
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.wait_until_ready = lambda: None
        gen.draft_model = None
        gen._tokenizer_lock = Lock()
        gen._cancel = lambda uid: None

        state_lock = Lock()
        active = 0
        max_active = 0
        queued = []
        next_uid = 0

        def tokenizer_work():
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.01)
            with state_lock:
                active -= 1

        def preprocess(prompt, images=None, audio=None, videos=None):
            del prompt, images, audio, videos
            tokenizer_work()
            return {"input_ids": mx.array([[99]], dtype=mx.int32)}

        def make_criteria(args, input_ids):
            del args, input_ids
            tokenizer_work()
            return object()

        class Requests:
            def put(self, request):
                nonlocal next_uid
                next_uid += 1
                queued.append(request)
                request.rqueue.put(
                    server.GenerationContext(uid=next_uid, prompt_tokens=1)
                )

        gen._preprocess_request = preprocess
        gen._make_thinking_budget_criteria = make_criteria
        gen.requests = Requests()

        def generate_one(_):
            _, token_iter = gen.generate(
                "prompt",
                args=server.GenerationArguments(
                    max_tokens=1,
                    thinking_budget=512,
                ),
            )
            token_iter.close()

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(generate_one, range(4)))

        assert max_active == 1
        assert len(queued) == 4
        assert all(request.thinking_budget_criteria is not None for request in queued)

    def test_generate_precomputes_semantic_hash_from_processed_image_content(self):
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.wait_until_ready = lambda: None
        gen.draft_model = None
        gen.apc_manager = object()
        gen.apc_mode = "block"
        gen.model = SimpleNamespace(language_model=SimpleNamespace())
        gen.processor = SimpleNamespace()
        gen._cancel = lambda uid: None

        pixel_values = iter(
            [
                mx.zeros((1, 3, 2, 2), dtype=mx.float32),
                mx.ones((1, 3, 2, 2), dtype=mx.float32),
            ]
        )
        queued = []

        def preprocess(prompt, images=None, audio=None, videos=None):
            del prompt, images, audio, videos
            return {
                "input_ids": mx.array([[1, 2]], dtype=mx.int32),
                "pixel_values": next(pixel_values),
            }

        class Requests:
            def put(self, request):
                queued.append(request)
                request.rqueue.put(
                    server.GenerationContext(uid=len(queued), prompt_tokens=2)
                )

        gen._preprocess_request = preprocess
        gen.requests = Requests()

        for _ in range(2):
            _, token_iter = gen.generate(
                "prompt",
                images=["mutable-image.png"],
                args=server.GenerationArguments(max_tokens=1),
            )
            token_iter.close()

        assert queued[0].images == queued[1].images
        assert queued[0].apc_semantic_hash != queued[1].apc_semantic_hash
        assert queued[0].apc_semantic_hash == apc_module.semantic_extra_hash(
            image_hash=hash_image_payload(
                pixel_values=mx.zeros((1, 3, 2, 2), dtype=mx.float32)
            ),
            model=gen.model.language_model,
            processor=gen.processor,
        )
        assert queued[1].apc_semantic_hash == apc_module.semantic_extra_hash(
            image_hash=hash_image_payload(
                pixel_values=mx.ones((1, 3, 2, 2), dtype=mx.float32)
            ),
            model=gen.model.language_model,
            processor=gen.processor,
        )

    def test_generate_skips_semantic_hash_for_unsupported_apc_model(self, monkeypatch):
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.wait_until_ready = lambda: None
        gen.draft_model = None
        gen.apc_manager = object()
        gen.apc_mode = None
        gen._cancel = lambda uid: None
        gen._preprocess_request = lambda prompt, images, audio, videos: {
            "input_ids": mx.array([[1, 2]], dtype=mx.int32),
            "pixel_values": mx.zeros((1, 3, 2, 2), dtype=mx.float32),
        }
        queued = []

        class Requests:
            def put(self, request):
                queued.append(request)
                request.rqueue.put(server.GenerationContext(uid=1, prompt_tokens=2))

        gen.requests = Requests()
        image_hash = MagicMock(wraps=apc_module.hash_image_payload)
        semantic_hash = MagicMock(wraps=apc_module.semantic_extra_hash)
        monkeypatch.setattr(apc_module, "hash_image_payload", image_hash)
        monkeypatch.setattr(apc_module, "semantic_extra_hash", semantic_hash)

        _, token_iter = gen.generate(
            "prompt",
            images=["image.png"],
            args=server.GenerationArguments(max_tokens=1),
        )
        token_iter.close()

        assert queued[0].apc_semantic_hash is None
        image_hash.assert_not_called()
        semantic_hash.assert_not_called()

    def test_server_runtime_snapshot_reports_effective_context_limit(self, monkeypatch):
        monkeypatch.setenv("MAX_KV_SIZE", "8")
        monkeypatch.setattr(
            server.runtime,
            "model_cache",
            {
                "config": SimpleNamespace(
                    text_config=SimpleNamespace(max_position_embeddings=16)
                )
            },
        )
        monkeypatch.setattr(server.runtime, "response_generator", None)
        monkeypatch.setattr(server.runtime, "apc_manager", None)

        runtime = server._server_runtime_snapshot()

        assert runtime["loaded_context_size"] == 16
        assert runtime["configured_context_limit"] == 8
        assert runtime["effective_context_limit"] == 8

    def test_generate_arguments_defaults(self):
        args = server.GenerationArguments()
        assert args.max_tokens == server.DEFAULT_MAX_TOKENS
        assert args.temperature == server.DEFAULT_TEMPERATURE
        assert args.enable_thinking is False
        assert args.logit_bias is None

    def test_token_queue_timeout_defaults_to_long_prefill_window(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", raising=False)
        monkeypatch.setattr(server.runtime, "config", RuntimeConfig.from_env())

        assert server.get_token_queue_timeout() == 600.0

    def test_token_queue_timeout_accepts_namespaced_env(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", "42.5")
        monkeypatch.setattr(server.runtime, "config", RuntimeConfig.from_env())

        assert server.get_token_queue_timeout() == 42.5

    def test_token_queue_timeout_invalid_values_fall_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", "bad")
        monkeypatch.setattr(server.runtime, "config", RuntimeConfig.from_env())

        assert server.get_token_queue_timeout() == 600.0

    def test_token_queue_timeout_can_disable_timeout(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", "0")
        monkeypatch.setattr(server.runtime, "config", RuntimeConfig.from_env())

        assert server.get_token_queue_timeout() is None

    def test_token_iterator_reports_timeout_and_cancels_request(self, monkeypatch):
        gen = self._bare_generator()
        cancelled = []

        class Requests:
            def put(self, item):
                rqueue = item.rqueue
                rqueue.put(SimpleNamespace(uid="req-1"))

        gen.requests = Requests()
        gen._cancel = cancelled.append
        monkeypatch.setattr(server.runtime.config, "token_queue_timeout", 0.01)

        _, token_iter = gen.generate("hello")

        with pytest.raises(RuntimeError, match="Timed out waiting for 0.01s"):
            next(token_iter)

        assert cancelled == ["req-1"]

    def test_token_iterator_close_cancels_while_next_blocks(self):
        cancelled = []
        result = []

        class BlockingQueue(Queue):
            def __init__(self):
                super().__init__()
                self.waiting = Event()

            def get(self, *args, **kwargs):
                self.waiting.set()
                return super().get(*args, **kwargs)

        rqueue = BlockingQueue()
        token_iter = server_generation._TokenIterator(
            rqueue,
            "req-1",
            cancelled.append,
            None,
        )

        def consume():
            try:
                result.append(next(token_iter))
            except Exception as exc:
                result.append(exc)

        thread = Thread(target=consume)
        thread.start()
        assert rqueue.waiting.wait(timeout=1.0)

        token_iter.close()

        assert cancelled == ["req-1"]

        rqueue.put(None)
        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert isinstance(result[0], StopIteration)

    def test_token_iterator_waits_past_timeout_for_delayed_token(self, monkeypatch):
        import threading

        gen = self._bare_generator()
        cancelled = []
        token = SimpleNamespace(text="hi")
        timeout_s = 0.05
        delay_s = timeout_s * 3

        class Requests:
            def put(self, item):
                rqueue: Queue = item.rqueue
                rqueue.put(SimpleNamespace(uid="req-1"))

                def deliver():
                    rqueue.put(token)
                    rqueue.put(None)

                threading.Timer(delay_s, deliver).start()

        gen.requests = Requests()
        gen._cancel = cancelled.append
        monkeypatch.setattr(
            server.runtime.config, "token_queue_timeout", timeout_s * 10
        )

        _, token_iter = gen.generate("hello")

        start = time.monotonic()
        assert next(token_iter) is token
        assert time.monotonic() - start >= delay_s * 0.5
        with pytest.raises(StopIteration):
            next(token_iter)
        assert cancelled == []

    def test_collect_pending_requests_coalesces_after_first_item(self, monkeypatch):
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.requests = Queue()
        gen._stop = False
        first = object()
        second = object()
        gen.requests.put(first)

        def fake_sleep(seconds):
            assert seconds == pytest.approx(0.005)
            gen.requests.put(second)

        monkeypatch.setattr(server.time, "sleep", fake_sleep)

        pending, should_stop = gen._collect_pending_requests(
            active=False, coalesce_s=0.005
        )

        assert pending == [first, second]
        assert should_stop is False

    def test_step_streams_spm_subword_tokens_immediately(self):
        class SentencePieceTokenizer:
            vocab = {
                "▁hello": 0,
                "world": 1,
                "!": 2,
            }

            def decode(self, tokens):
                parts = []
                for token in tokens:
                    parts.append(
                        {
                            0: " hello",
                            1: "world",
                            2: "!",
                        }[token]
                    )
                return "".join(parts).lstrip()

        class SingleResponseBatch:
            def __init__(self, response):
                self.response = response

            def next(self, **kwargs):
                return [], [self.response]

        tokenizer = SentencePieceTokenizer()
        processor = SimpleNamespace(detokenizer=SPMStreamingDetokenizer(tokenizer))
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        rqueue = Queue()
        active = {
            1: {
                "rqueue": rqueue,
                "streamer": _ServerTokenStreamer(
                    tokenizer,
                    server.make_streaming_detokenizer(processor),
                ),
            }
        }

        for token in [0, 1, 2]:
            gen._step(
                SingleResponseBatch(
                    SimpleNamespace(
                        uid=1,
                        token=token,
                        token_logprob=0.0,
                        finish_reason=None,
                    )
                ),
                active,
            )
        gen._step(
            SingleResponseBatch(
                SimpleNamespace(
                    uid=1,
                    token=99,
                    token_logprob=0.0,
                    finish_reason="stop",
                )
            ),
            active,
        )

        segments = []
        while not rqueue.empty():
            item = rqueue.get()
            if item is not None:
                segments.append(item.text)

        assert segments == ["hello", "world", "!", ""]

    def test_server_token_streamer_flushes_incomplete_utf8_on_finalize(self):
        class ByteFallbackTokenizer:
            vocab = {
                "<0xF0>": 0,
                "<0x9F>": 1,
            }

            def decode(self, tokens):
                byte_values = {0: 0xF0, 1: 0x9F}
                return bytes(byte_values[token] for token in tokens).decode(
                    "utf-8", errors="replace"
                )

        tokenizer = ByteFallbackTokenizer()
        processor = SimpleNamespace(
            detokenizer=SPMStreamingDetokenizer(tokenizer, trim_space=False)
        )
        streamer = _ServerTokenStreamer(
            tokenizer,
            server.make_streaming_detokenizer(processor),
        )

        assert streamer.advance(0, None) == ""
        assert streamer.advance(1, None) == ""
        assert streamer.finalize() == "\ufffd"

    def test_step_streams_multiple_utf8_emojis_with_text_between_them(self):
        class MixedEmojiTokenizer:
            vocab = {
                "hi": 0,
                "<0xF0>": 1,
                "<0x9F>": 2,
                "<0x98>": 3,
                "<0x80>": 4,
                "▁mid": 5,
                "<0x82>": 6,
                "▁wow": 7,
                "<0x8E>": 8,
                "▁done": 9,
            }

            def decode(self, tokens):
                text = ""
                byte_buffer = bytearray()
                byte_values = {
                    1: 0xF0,
                    2: 0x9F,
                    3: 0x98,
                    4: 0x80,
                    6: 0x82,
                    8: 0x8E,
                }
                regular = {0: "hi", 5: "▁mid", 7: "▁wow", 9: "▁done"}

                def flush_bytes():
                    nonlocal text, byte_buffer
                    if byte_buffer:
                        text += byte_buffer.decode("utf-8", errors="replace")
                        byte_buffer = bytearray()

                for token in tokens:
                    if token in byte_values:
                        byte_buffer.append(byte_values[token])
                    else:
                        flush_bytes()
                        text += regular[token].replace("▁", " ")
                flush_bytes()
                return text

        class SingleResponseBatch:
            def __init__(self, response):
                self.response = response

            def next(self, **kwargs):
                return [], [self.response]

        tokenizer = MixedEmojiTokenizer()
        processor = SimpleNamespace(
            detokenizer=SPMStreamingDetokenizer(tokenizer, trim_space=False)
        )
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        rqueue = Queue()
        active = {
            1: {
                "rqueue": rqueue,
                "streamer": _ServerTokenStreamer(
                    tokenizer,
                    server.make_streaming_detokenizer(processor),
                ),
            }
        }

        for token in [0, 1, 2, 3, 4, 5, 1, 2, 3, 6, 7, 1, 2, 3, 8, 9, 1, 2, 3, 4]:
            gen._step(
                SingleResponseBatch(
                    SimpleNamespace(
                        uid=1,
                        token=token,
                        token_logprob=0.0,
                        finish_reason=None,
                    )
                ),
                active,
            )
        gen._step(
            SingleResponseBatch(
                SimpleNamespace(
                    uid=1,
                    token=99,
                    token_logprob=0.0,
                    finish_reason="stop",
                )
            ),
            active,
        )

        segments = []
        while not rqueue.empty():
            item = rqueue.get()
            if item is not None:
                segments.append(item.text)

        streamed_text = "".join(segments)
        assert segments == [
            "hi",
            "",
            "",
            "",
            "😀",
            " mid",
            "",
            "",
            "",
            "😂",
            " wow",
            "",
            "",
            "",
            "😎",
            " done",
            "",
            "",
            "",
            "😀",
            "",
        ]
        assert streamed_text == "hi😀 mid😂 wow😎 done😀"
        assert "\ufffd" not in streamed_text

    def test_run_batches_eight_streaming_requests(self, monkeypatch):
        batch_state = {}

        class FakeDetokenizer:
            def __init__(self):
                self.last_segment = ""

            def reset(self):
                self.last_segment = ""

            def add_token(self, token):
                self.last_segment = str(token)

            def finalize(self):
                pass

        class FakeBatchGenerator:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self._next_uid = 1
                self._active = {}
                self.inserted_uids = []
                self.next_active_sizes = []
                batch_state["instance"] = self

            def insert(self, *args, **kwargs):
                del args, kwargs
                uid = self._next_uid
                self._next_uid += 1
                self._active[uid] = 0
                self.inserted_uids.append(uid)
                return (uid,)

            def remove(self, uid):
                return self._active.pop(uid, None) is not None

            @property
            def unprocessed_prompts(self):
                return []

            @property
            def has_pending_prompts(self):
                return False

            def next(self, **kwargs):
                del kwargs
                self.next_active_sizes.append(len(self._active))
                responses = []
                finished = []
                for uid in sorted(self._active):
                    step = self._active[uid]
                    token = uid * 10 + step
                    finish_reason = None if step == 0 else "length"
                    responses.append(
                        SimpleNamespace(
                            uid=uid,
                            token=token,
                            token_logprob=0.0,
                            finish_reason=finish_reason,
                        )
                    )
                    if finish_reason is None:
                        self._active[uid] = step + 1
                    else:
                        finished.append(uid)
                for uid in finished:
                    del self._active[uid]
                return [], responses

        monkeypatch.setattr(server_generation, "BatchGenerator", FakeBatchGenerator)
        monkeypatch.setattr(
            server_generation,
            "make_streaming_detokenizer",
            lambda _: FakeDetokenizer(),
        )

        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.model_path = "demo"
        gen.adapter_path = None
        gen.model = None
        gen.processor = None
        gen.config = None
        gen.stop_tokens = set()
        gen.vision_cache = None
        gen.draft_model = None
        gen.draft_kind = None
        gen.kv_bits = None
        gen.kv_group_size = server.DEFAULT_KV_GROUP_SIZE
        gen.kv_quant_scheme = server.DEFAULT_KV_QUANT_SCHEME
        gen.quantized_kv_start = server.DEFAULT_QUANTIZED_KV_START
        gen.top_logprobs_k = 0
        gen.apc_manager = None
        gen.tokenizer = SimpleNamespace()
        gen.requests = Queue()
        gen._stop = False
        gen._ready = Event()
        gen._load_error = None
        gen._cancelled = set()
        gen._cancel_lock = Lock()

        def fake_initialize_model():
            gen.model = SimpleNamespace(language_model=object())
            gen.processor = SimpleNamespace()
            gen.config = SimpleNamespace()
            gen.stop_tokens = set()
            gen.draft_model = None
            gen.draft_kind = None
            gen.tokenizer = SimpleNamespace()

        gen._initialize_model = fake_initialize_model
        gen._gpu_embed = lambda raw_inputs, images=None, apc_semantic_hash=None: (
            mx.array([[raw_inputs["request_id"]]], dtype=mx.int32),
            {},
        )

        request_queues = []
        for request_id in range(8):
            rqueue = Queue()
            request_queues.append(rqueue)
            gen.requests.put(
                server_generation.QueuedGenerationRequest(
                    rqueue=rqueue,
                    raw_inputs={"request_id": request_id},
                    prompt_tokens=1,
                    args=server.GenerationArguments(max_tokens=2),
                )
            )

        worker = Thread(target=gen._run, daemon=True)
        worker.start()

        streamed_by_uid = {}
        try:
            for rqueue in request_queues:
                ctx = rqueue.get(timeout=1)
                assert isinstance(ctx, server.GenerationContext)
                assert ctx.prompt_tokens == 1

                items = []
                while True:
                    item = rqueue.get(timeout=1)
                    if item is None:
                        break
                    items.append((item.text, item.finish_reason))
                streamed_by_uid[ctx.uid] = items
        finally:
            gen._stop = True
            gen.requests.put(None)
            worker.join(timeout=2)

        batch_gen = batch_state["instance"]
        assert batch_gen.inserted_uids == list(range(1, 9))
        assert batch_gen.next_active_sizes[:2] == [8, 8]
        assert len(streamed_by_uid) == 8
        for uid, items in streamed_by_uid.items():
            assert items == [
                (str(uid * 10), None),
                (str(uid * 10 + 1), "length"),
            ]

    @pytest.mark.parametrize("draft_kind", ["dflash", "eagle3", "mtp"])
    def test_run_coalesces_idle_speculative_batch_generator(
        self, monkeypatch, draft_kind
    ):
        monkeypatch.setenv("MLX_VLM_SPEC_BATCH_COALESCE_MS", "37")
        calls = []
        draft_model = object()

        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.draft_model = None
        gen.draft_kind = None
        gen._stop = False
        gen._ready = Event()
        gen._load_error = None

        def fake_initialize_model():
            gen.model = SimpleNamespace(language_model=object())
            gen.processor = SimpleNamespace()
            gen.config = SimpleNamespace()
            gen.stop_tokens = set()
            gen.draft_model = draft_model
            gen.draft_kind = draft_kind
            gen.tokenizer = SimpleNamespace()

        def fake_collect_pending_requests(
            *, active, idle_timeout=0.1, coalesce_s=0.0, capacity=None
        ):
            del idle_timeout, capacity
            calls.append((active, coalesce_s))
            return [], True

        gen._initialize_model = fake_initialize_model
        gen._collect_pending_requests = fake_collect_pending_requests

        gen._run_impl()

        assert calls == [(False, 0.037)]

    def test_idle_batch_generator_is_recreated_for_new_sampler(self, monkeypatch):
        created = []
        next_uid = [1]

        class FakeDetokenizer:
            def __init__(self):
                self.last_segment = ""

            def reset(self):
                self.last_segment = ""

            def add_token(self, token):
                self.last_segment = str(token)

            def finalize(self):
                pass

        class FakeBatchGenerator:
            def __init__(self, *args, **kwargs):
                del args
                self.sampler = kwargs.get("sampler")
                self.closed = False
                self._active = {}
                created.append(self)

            def insert(self, *args, **kwargs):
                del args, kwargs
                uid = next_uid[0]
                next_uid[0] += 1
                self._active[uid] = True
                return (uid,)

            @property
            def has_work(self):
                return bool(self._active)

            @property
            def unprocessed_prompts(self):
                return []

            @property
            def has_pending_prompts(self):
                return False

            def next(self, **kwargs):
                del kwargs
                responses = [
                    SimpleNamespace(
                        uid=uid,
                        token=uid,
                        token_logprob=0.0,
                        finish_reason="length",
                    )
                    for uid in list(self._active)
                ]
                self._active.clear()
                return [], responses

            def remove(self, uid):
                return self._active.pop(uid, None) is not None

            def close(self):
                self.closed = True

        monkeypatch.setattr(server_generation, "BatchGenerator", FakeBatchGenerator)
        monkeypatch.setattr(
            server_generation,
            "make_streaming_detokenizer",
            lambda _: FakeDetokenizer(),
        )

        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.model_path = "demo"
        gen.adapter_path = None
        gen.model = None
        gen.processor = None
        gen.config = None
        gen.stop_tokens = set()
        gen.vision_cache = None
        gen.draft_model = None
        gen.draft_kind = None
        gen.kv_bits = None
        gen.kv_group_size = server.DEFAULT_KV_GROUP_SIZE
        gen.kv_quant_scheme = server.DEFAULT_KV_QUANT_SCHEME
        gen.quantized_kv_start = server.DEFAULT_QUANTIZED_KV_START
        gen.top_logprobs_k = 0
        gen.apc_manager = None
        gen.tokenizer = SimpleNamespace()
        gen.requests = Queue()
        gen._stop = False
        gen._ready = Event()
        gen._load_error = None
        gen._cancelled = set()
        gen._cancel_lock = Lock()
        gen._make_sampler = lambda args: f"sampler-{args.temperature}"

        def fake_initialize_model():
            gen.model = SimpleNamespace(language_model=object())
            gen.processor = SimpleNamespace()
            gen.config = SimpleNamespace()
            gen.stop_tokens = set()
            gen.draft_model = None
            gen.draft_kind = None
            gen.tokenizer = SimpleNamespace()

        gen._initialize_model = fake_initialize_model
        gen._gpu_embed = lambda raw_inputs, images=None, apc_semantic_hash=None: (
            mx.array([[raw_inputs["request_id"]]], dtype=mx.int32),
            {},
        )

        worker = Thread(target=gen._run, daemon=True)
        worker.start()

        def run_request(request_id, temperature):
            rqueue = Queue()
            gen.requests.put(
                server_generation.QueuedGenerationRequest(
                    rqueue=rqueue,
                    raw_inputs={"request_id": request_id},
                    prompt_tokens=1,
                    args=server.GenerationArguments(
                        max_tokens=1, temperature=temperature
                    ),
                )
            )
            ctx = rqueue.get(timeout=1)
            assert isinstance(ctx, server.GenerationContext)
            item = rqueue.get(timeout=1)
            assert item.finish_reason == "length"
            assert rqueue.get(timeout=1) is None

        try:
            run_request(1, 0.0)
            run_request(2, 0.6)
        finally:
            gen._stop = True
            gen.requests.put(None)
            worker.join(timeout=2)

        assert [bg.sampler for bg in created] == ["sampler-0.0", "sampler-0.6"]
        assert created[0].closed is True

    def test_step_attaches_prompt_metrics_from_prompt_progress(self):
        class SimpleTokenizer:
            vocab = {"hi": 0}

            def decode(self, tokens):
                return "hi" if tokens else ""

        class PromptProgressBatch:
            def next(self, **kwargs):
                return (
                    [SimpleNamespace(uid=1, prompt_tps=184.431, cached_tokens=7)],
                    [
                        SimpleNamespace(
                            uid=1,
                            token=0,
                            token_logprob=0.0,
                            finish_reason="stop",
                        )
                    ],
                )

        tokenizer = SimpleTokenizer()
        processor = SimpleNamespace(
            detokenizer=SPMStreamingDetokenizer(tokenizer, trim_space=False)
        )
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        rqueue = Queue()
        active = {
            1: {
                "rqueue": rqueue,
                "streamer": _ServerTokenStreamer(
                    tokenizer,
                    server.make_streaming_detokenizer(processor),
                ),
                "prompt_tps": None,
                "cached_tokens": 0,
            }
        }

        gen._step(PromptProgressBatch(), active)

        item = rqueue.get()
        assert item.prompt_tps == pytest.approx(184.431)
        assert item.cached_tokens == 7
        assert rqueue.get() is None

    def test_generate_arguments_to_generate_kwargs(self):
        processor = lambda tokens, logits: logits
        args = server.GenerationArguments(
            max_tokens=50,
            temperature=0.7,
            top_k=40,
            min_p=0.05,
            repetition_penalty=1.15,
            repetition_context_size=512,
            presence_penalty=0.2,
            presence_context_size=256,
            frequency_penalty=0.3,
            frequency_context_size=128,
            logit_bias={3: -0.5},
            enable_thinking=False,
            thinking_budget=100,
            thinking_start_token="<think>",
            thinking_end_token="</think>",
            logits_processors=[processor],
            tenant_id="tenant-a",
        )
        kw = args.to_generate_kwargs()
        assert kw["max_tokens"] == 50
        assert kw["top_k"] == 40
        assert kw["min_p"] == 0.05
        assert kw["repetition_penalty"] == 1.15
        assert kw["repetition_context_size"] == 512
        assert kw["presence_penalty"] == 0.2
        assert kw["presence_context_size"] == 256
        assert kw["frequency_penalty"] == 0.3
        assert kw["frequency_context_size"] == 128
        assert kw["logit_bias"] == {3: -0.5}
        assert kw["enable_thinking"] is False
        assert kw["thinking_budget"] == 100
        assert kw["thinking_start_token"] == "<think>"
        assert kw["thinking_end_token"] == "</think>"
        assert kw["logits_processors"] == [processor]
        assert kw["apc_tenant"] == "tenant-a"

    def test_generate_arguments_to_template_kwargs(self):
        args = server.GenerationArguments(
            enable_thinking=False,
            reasoning=True,
            reasoning_effort="high",
            thinking_budget=50,
            thinking_end_token="</think>",
        )
        kw = args.to_template_kwargs()
        assert kw["enable_thinking"] is False
        assert kw["reasoning"] is True
        assert kw["reasoning_effort"] == "high"
        assert kw["thinking_budget"] == 50
        assert kw["thinking_end_token"] == "</think>"

    def test_generate_arguments_omits_none_optionals(self):
        args = server.GenerationArguments()
        kw = args.to_generate_kwargs()
        assert "repetition_penalty" not in kw
        assert (
            kw["repetition_context_size"]
            == server_generation.DEFAULT_REPETITION_CONTEXT_SIZE
        )
        assert "presence_penalty" not in kw
        assert (
            kw["presence_context_size"]
            == server_generation.DEFAULT_REPETITION_CONTEXT_SIZE
        )
        assert "frequency_penalty" not in kw
        assert (
            kw["frequency_context_size"]
            == server_generation.DEFAULT_REPETITION_CONTEXT_SIZE
        )
        assert "logit_bias" not in kw
        assert "thinking_budget" not in kw

    def test_server_generation_builds_repetition_logits_processors(self, monkeypatch):
        custom_processor = lambda tokens, logits: logits
        calls = []

        def fake_make_logits_processors(
            logit_bias,
            repetition_penalty,
            repetition_context_size,
            presence_penalty,
            presence_context_size,
            frequency_penalty,
            frequency_context_size,
        ):
            calls.append(
                (
                    logit_bias,
                    repetition_penalty,
                    repetition_context_size,
                    presence_penalty,
                    presence_context_size,
                    frequency_penalty,
                    frequency_context_size,
                )
            )
            return ["repetition-processor"]

        monkeypatch.setattr(
            server_generation, "make_logits_processors", fake_make_logits_processors
        )

        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        args = server.GenerationArguments(
            repetition_penalty=1.2,
            repetition_context_size=512,
            presence_penalty=0.2,
            presence_context_size=256,
            frequency_penalty=0.3,
            frequency_context_size=128,
            logit_bias={5: -0.5},
            logits_processors=[custom_processor],
        )

        processors = gen._make_logits_processors(args)

        assert calls == [({5: -0.5}, 1.2, 512, 0.2, 256, 0.3, 128)]
        assert processors == ["repetition-processor", custom_processor]

    def test_server_generation_delays_structured_processors_for_thinking_prompt(
        self, monkeypatch
    ):
        class SimpleTokenizer:
            def encode(self, text, add_special_tokens=False):
                return {"<think>": [10], "</think>": [20]}[text]

        repetition_processor = lambda tokens, logits: logits
        structured_processor = lambda tokens, logits: logits

        monkeypatch.setattr(
            server_generation,
            "make_logits_processors",
            lambda *_args: [repetition_processor],
        )

        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.tokenizer = SimpleTokenizer()
        args = server.GenerationArguments(
            enable_thinking=True,
            thinking_start_token="<think>",
            thinking_end_token="</think>",
            logits_processors=[structured_processor],
        )

        processors = gen._make_logits_processors(
            args,
            mx.array([[1, 10, 3]], dtype=mx.int32),
        )

        assert processors[0] is repetition_processor
        assert isinstance(processors[1], server_generation.ThinkingAwareLogitsProcessor)
        assert processors[1].processor is structured_processor

    def test_server_generation_keeps_structured_processors_active_without_open_thinking(
        self, monkeypatch
    ):
        class SimpleTokenizer:
            def encode(self, text, add_special_tokens=False):
                return {"<think>": [10], "</think>": [20]}[text]

        structured_processor = lambda tokens, logits: logits
        monkeypatch.setattr(
            server_generation,
            "make_logits_processors",
            lambda *_args: [],
        )

        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.tokenizer = SimpleTokenizer()
        args = server.GenerationArguments(
            enable_thinking=True,
            thinking_start_token="<think>",
            thinking_end_token="</think>",
            logits_processors=[structured_processor],
        )

        processors = gen._make_logits_processors(
            args,
            mx.array([[1, 10, 3, 20]], dtype=mx.int32),
        )

        assert processors == [structured_processor]

    def test_server_generation_delays_structured_processors_for_self_opening_model(
        self, monkeypatch
    ):
        """Regression test for issue #1911."""

        class SimpleTokenizer:
            def encode(self, text, add_special_tokens=False):
                return {"<think>": [10], "</think>": [20]}[text]

        repetition_processor = lambda tokens, logits: logits
        structured_processor = lambda tokens, logits: logits

        monkeypatch.setattr(
            server_generation,
            "make_logits_processors",
            lambda *_args: [repetition_processor],
        )

        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.tokenizer = SimpleTokenizer()
        args = server.GenerationArguments(
            enable_thinking=True,
            thinking_start_token="<think>",
            thinking_end_token="</think>",
            logits_processors=[structured_processor],
        )

        processors = gen._make_logits_processors(
            args,
            mx.array([[1, 2, 3]], dtype=mx.int32),
        )

        assert processors[0] is repetition_processor
        assert isinstance(processors[1], server_generation.ThinkingAwareLogitsProcessor)
        assert processors[1].processor is structured_processor

    def test_build_gen_args_from_openai_request(self):
        req = SimpleNamespace(
            max_output_tokens=128,
            temperature=0.5,
            top_p=0.9,
            top_k=32,
            min_p=0.1,
            repetition_penalty=1.2,
            repetition_context_size=512,
            presence_penalty=0.2,
            presence_context_size=256,
            frequency_penalty=0.3,
            frequency_context_size=128,
            logit_bias={"5": -1.0},
            enable_thinking=False,
            thinking_budget=None,
            thinking_start_token=None,
            thinking_end_token=None,
        )
        args = server._build_gen_args(req, tenant_id="tenant-a")
        assert args.max_tokens == 128
        assert args.top_k == 32
        assert args.repetition_context_size == 512
        assert args.presence_penalty == 0.2
        assert args.presence_context_size == 256
        assert args.frequency_penalty == 0.3
        assert args.frequency_context_size == 128
        assert args.logit_bias == {5: -1.0}  # string keys converted to int
        assert args.to_generate_kwargs()["apc_tenant"] == "tenant-a"

    def test_build_gen_args_from_chat_request(self):
        req = SimpleNamespace(
            max_tokens=256,
            max_output_tokens=None,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            repetition_penalty=None,
            repetition_context_size=None,
            presence_penalty=None,
            presence_context_size=None,
            frequency_penalty=None,
            frequency_context_size=None,
            logit_bias=None,
            enable_thinking=True,
            thinking_budget=None,
            thinking_start_token=None,
            thinking_end_token=None,
        )
        args = server._build_gen_args(req)
        assert args.max_tokens == 256
        assert args.enable_thinking is True

    def test_build_gen_args_uses_model_generation_config_when_omitted(
        self, monkeypatch
    ):
        monkeypatch.setitem(
            server.runtime.model_cache,
            "config",
            SimpleNamespace(temperature=1.0, top_p=0.95, top_k=64),
        )
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
        )

        args = server._build_gen_args(req)

        assert args.temperature == 1.0
        assert args.top_p == 0.95
        assert args.top_k == 64

    def test_build_gen_args_request_sampling_overrides_model_generation_config(
        self, monkeypatch
    ):
        monkeypatch.setitem(
            server.runtime.model_cache,
            "config",
            SimpleNamespace(temperature=1.0, top_p=0.95, top_k=64),
        )
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        )

        args = server._build_gen_args(req)

        assert args.temperature == 0.0
        assert args.top_p == 1.0
        assert args.top_k == 0

    def test_generation_defaults_applied_when_request_omits(self, monkeypatch):
        """Registry generation_defaults fill every sampling field the request omits — the
        VS Code/Zed fix (those clients send no sampling)."""
        monkeypatch.setenv(
            "MLX_VLM_GENERATION_DEFAULTS",
            json.dumps(
                {
                    "temperature": 0.3,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0.05,
                    "presence_penalty": 0.0,
                    "enable_thinking": True,
                }
            ),
        )
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
        )

        args = server._build_gen_args(req)

        assert args.temperature == 0.3  # base default 0.0 -> yaml
        assert args.top_p == 0.95  # base default 1.0 -> yaml
        assert args.top_k == 20  # base default 0 -> yaml
        assert args.min_p == 0.05  # had NO middle layer before -> yaml
        assert args.presence_penalty == 0.0  # base None -> yaml
        assert args.enable_thinking is True

    def test_request_sampling_overrides_generation_defaults(self, monkeypatch):
        """Explicit request sampling always beats the registry default (request wins)."""
        monkeypatch.setenv(
            "MLX_VLM_GENERATION_DEFAULTS", json.dumps({"temperature": 0.3})
        )
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            temperature=0.9,
        )

        args = server._build_gen_args(req)

        assert args.temperature == 0.9

    def test_generation_defaults_override_checkpoint_config(self, monkeypatch):
        """Registry default beats the checkpoint's baked generation_config (precedence A):
        request > yaml > checkpoint > hardcoded. Without this, the distill's baked temp 1.0
        would still win for a no-sampling client."""
        monkeypatch.setitem(
            server.runtime.model_cache,
            "config",
            SimpleNamespace(temperature=1.0, top_p=0.95, top_k=64),
        )
        monkeypatch.setenv(
            "MLX_VLM_GENERATION_DEFAULTS", json.dumps({"temperature": 0.3})
        )
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
        )

        args = server._build_gen_args(req)

        assert args.temperature == 0.3  # yaml wins over the checkpoint's 1.0

    def test_generation_defaults_max_tokens_alias_not_clobbered(self, monkeypatch):
        """A request that sets the max_output_tokens alias suppresses the max_tokens
        default (the two are aliases; the overlay must not override the aliased request value).
        """
        monkeypatch.setenv(
            "MLX_VLM_GENERATION_DEFAULTS", json.dumps({"max_tokens": 102400})
        )
        req = server.OpenAIRequest(model="demo", input="hi", max_output_tokens=555)

        args = server._build_gen_args(req)

        assert args.max_tokens == 555

    def test_get_server_generation_defaults_rejects_unknown_key(self, monkeypatch):
        """An unknown/typo'd key fails loud and fast, not a silent no-op."""
        monkeypatch.setenv(
            "MLX_VLM_GENERATION_DEFAULTS", json.dumps({"temperatur": 0.3})
        )
        with pytest.raises(ValueError, match="temperatur"):
            server_generation.get_server_generation_defaults()

    def test_get_server_generation_defaults_empty_when_unset(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_GENERATION_DEFAULTS", raising=False)
        assert server_generation.get_server_generation_defaults() == {}

    def test_build_gen_args_logs_resolved_sampling(self, caplog):
        """The resolved (post-overlay) sampling is logged at INFO so runtime pass-through is
        observable — the hook the deploy smoke greps to prove every param is applied."""
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            temperature=0.42,
            top_p=0.91,
        )
        with caplog.at_level(logging.INFO, logger="mlx_vlm.server"):
            server._build_gen_args(req)
        blob = " ".join(r.getMessage() for r in caplog.records)
        assert "temperature=0.42" in blob
        assert "top_p=0.91" in blob

    def test_build_gen_args_defaults_penalty_context_sizes_when_omitted(self):
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            repetition_penalty=1.1,
            presence_penalty=0.2,
            frequency_penalty=0.3,
        )

        args = server._build_gen_args(req)

        assert (
            args.repetition_context_size
            == server_generation.DEFAULT_REPETITION_CONTEXT_SIZE
        )
        assert (
            args.presence_context_size
            == server_generation.DEFAULT_REPETITION_CONTEXT_SIZE
        )
        assert (
            args.frequency_context_size
            == server_generation.DEFAULT_REPETITION_CONTEXT_SIZE
        )

    def test_build_gen_args_defaults_penalty_context_sizes_when_null(self):
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            repetition_penalty=1.1,
            repetition_context_size=None,
            presence_penalty=0.2,
            presence_context_size=None,
            frequency_penalty=0.3,
            frequency_context_size=None,
        )

        args = server._build_gen_args(req)

        assert (
            args.repetition_context_size
            == server_generation.DEFAULT_REPETITION_CONTEXT_SIZE
        )
        assert (
            args.presence_context_size
            == server_generation.DEFAULT_REPETITION_CONTEXT_SIZE
        )
        assert (
            args.frequency_context_size
            == server_generation.DEFAULT_REPETITION_CONTEXT_SIZE
        )

    def test_build_gen_args_preserves_explicit_penalty_context_sizes(self):
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            repetition_context_size=64,
            presence_context_size=32,
            frequency_context_size=16,
        )

        args = server._build_gen_args(req)

        assert args.repetition_context_size == 64
        assert args.presence_context_size == 32
        assert args.frequency_context_size == 16

    def test_build_gen_args_uses_server_thinking_default_when_omitted(
        self, monkeypatch
    ):
        monkeypatch.setenv("MLX_VLM_ENABLE_THINKING", "1")
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
        )

        assert "enable_thinking" not in req.model_fields_set
        assert server._build_gen_args(req).enable_thinking is True

        monkeypatch.setenv("MLX_VLM_ENABLE_THINKING", "0")
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
        )

        assert server._build_gen_args(req).enable_thinking is False

    def test_build_gen_args_uses_server_thinking_token_defaults_when_omitted(
        self, monkeypatch
    ):
        processor = SimpleNamespace(
            config=SimpleNamespace(
                thinking_start_token="<model-analysis>",
                thinking_end_token="</model-analysis>",
            )
        )
        monkeypatch.setenv("MLX_VLM_THINKING_BUDGET", "256")
        monkeypatch.setenv("MLX_VLM_THINKING_START_TOKEN", "<analysis>")
        monkeypatch.setenv("MLX_VLM_THINKING_END_TOKEN", "</analysis>")
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
        )

        assert "thinking_budget" not in req.model_fields_set
        assert "thinking_start_token" not in req.model_fields_set
        assert "thinking_end_token" not in req.model_fields_set
        args = server._build_gen_args(req, processor)

        assert args.thinking_budget == 256
        assert args.thinking_start_token == "<analysis>"
        assert args.thinking_end_token == "</analysis>"

    def test_build_gen_args_uses_processor_config_thinking_tokens_when_omitted(
        self, monkeypatch
    ):
        monkeypatch.delenv("MLX_VLM_THINKING_START_TOKEN", raising=False)
        monkeypatch.delenv("MLX_VLM_THINKING_END_TOKEN", raising=False)
        processor = SimpleNamespace(
            config=SimpleNamespace(
                thinking_start_token="to=self<|message|>",
                thinking_end_token="<|eom|>",
            )
        )
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
        )

        args = server._build_gen_args(req, processor)

        assert args.thinking_start_token == "to=self<|message|>"
        assert args.thinking_end_token == "<|eom|>"

    def test_build_gen_args_request_thinking_overrides_server_default(
        self, monkeypatch
    ):
        monkeypatch.setenv("MLX_VLM_ENABLE_THINKING", "1")
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            enable_thinking=False,
        )

        assert server._build_gen_args(req).enable_thinking is False

        monkeypatch.setenv("MLX_VLM_ENABLE_THINKING", "0")
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            enable_thinking=True,
        )

        assert server._build_gen_args(req).enable_thinking is True

    def test_build_gen_args_request_thinking_tokens_override_server_defaults(
        self, monkeypatch
    ):
        processor = SimpleNamespace(
            config=SimpleNamespace(
                thinking_start_token="<model-analysis>",
                thinking_end_token="</model-analysis>",
            )
        )
        monkeypatch.setenv("MLX_VLM_THINKING_BUDGET", "256")
        monkeypatch.setenv("MLX_VLM_THINKING_START_TOKEN", "<analysis>")
        monkeypatch.setenv("MLX_VLM_THINKING_END_TOKEN", "</analysis>")
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            thinking_budget=32,
            thinking_start_token="<think>",
            thinking_end_token="</think>",
        )

        args = server._build_gen_args(req, processor)

        assert args.thinking_budget == 32
        assert args.thinking_start_token == "<think>"
        assert args.thinking_end_token == "</think>"

    def test_lifespan_preloads_configured_model_kinds(self, monkeypatch):
        preload_env = {
            "MLX_VLM_PRELOAD_MODEL": "language-demo",
            "MLX_VLM_PRELOAD_ADAPTER": "adapter-demo",
            "MLX_VLM_PRELOAD_IMAGE_MODEL": "image-demo",
            "MLX_VLM_PRELOAD_TTS_MODEL": "tts-demo",
            "MLX_VLM_PRELOAD_STT_MODEL": "stt-demo",
            "MLX_VLM_PRELOAD_RERANKER_MODEL": "reranker-demo",
        }
        for key, value in preload_env.items():
            monkeypatch.setenv(key, value)
        calls = []

        def fake_get_cached_model(model_path, adapter_path=None, *, model_kind="auto"):
            calls.append((model_path, adapter_path, model_kind))
            return SimpleNamespace(), None, SimpleNamespace(model_type=model_kind)

        monkeypatch.setattr(
            server._app_module, "get_cached_model", fake_get_cached_model
        )
        monkeypatch.setattr(server.runtime, "audio_queue", None)

        async def run_lifespan():
            async with server._app_module.lifespan(server.app):
                pass

        asyncio.run(run_lifespan())

        assert calls == [
            ("language-demo", "adapter-demo", "text_generation"),
            ("image-demo", None, "image_generation"),
            ("tts-demo", None, "audio_tts"),
            ("stt-demo", None, "audio_stt"),
            ("reranker-demo", None, "reranker"),
        ]
        for key in preload_env:
            assert key not in os.environ

    def test_lifespan_continues_when_optional_preload_fails(self, monkeypatch):
        kinds = dict(
            MODEL="text_generation",
            TTS_MODEL="audio_tts",
            STT_MODEL="audio_stt",
            EMBEDDING_MODEL="embedding",
            RERANKER_MODEL="reranker",
            DECISION_MODEL="decision",
        )
        for key, kind in kinds.items():
            monkeypatch.setenv("MLX_VLM_PRELOAD_" + key, kind)
        calls = []

        def load(model_path, adapter_path=None, *, model_kind="auto"):
            calls.append(model_kind)
            if model_kind == "audio_stt":
                raise server.HTTPException(
                    status_code=500, detail="Failed to load audio model: boom"
                )
            return NS(), None, NS(model_type=model_kind)

        monkeypatch.setattr(server._app_module, "get_cached_model", load)
        monkeypatch.setattr(server.runtime, "audio_queue", None)
        monkeypatch.setattr(server.runtime, "preload_failures", {})

        async def run():
            async with server._app_module.lifespan(server.app):
                pass

        asyncio.run(run())
        assert calls == list(kinds.values())
        failure = server.runtime.preload_failures["audio_stt"]
        assert (
            failure["model"] == "audio_stt"
            and "Failed to load audio model" in failure["error"]
        )
        assert "audio_tts" not in server.runtime.preload_failures

    def test_lifespan_propagates_language_model_failure(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_PRELOAD_MODEL", "language-demo")

        def fake_get_cached_model(model_path, adapter_path=None, *, model_kind="auto"):
            raise server.HTTPException(
                status_code=500, detail="language model exploded"
            )

        monkeypatch.setattr(
            server._app_module, "get_cached_model", fake_get_cached_model
        )
        monkeypatch.setattr(server.runtime, "audio_queue", None)

        async def run_lifespan():
            async with server._app_module.lifespan(server.app):
                pass

        with pytest.raises(server.HTTPException):
            asyncio.run(run_lifespan())

    def test_gpu_embed_hashes_pixel_values_without_image_ref(self):
        class Embed:
            def to_dict(self):
                return {"inputs_embeds": mx.zeros((1, 2, 4))}

        class Model:
            def get_input_embeddings(
                self, input_ids, pixel_values, mask=None, **kwargs
            ):
                return Embed()

        response_generator = SimpleNamespace(model=Model(), vision_cache=None)
        pixel_values = mx.array([[[[1.0, 2.0]]]])
        semantic_hash = apc_module.semantic_extra_hash(
            image_hash=hash_image_payload(pixel_values=pixel_values)
        )

        _, gen_kwargs = server.ResponseGenerator._gpu_embed(
            response_generator,
            {
                "input_ids": mx.array([[1, 2]]),
                "pixel_values": pixel_values,
                "attention_mask": mx.array([[1, 1]]),
            },
            images=None,
            apc_semantic_hash=semantic_hash,
        )

        assert gen_kwargs["_apc_semantic_hash"] == semantic_hash

    def test_gpu_embed_drops_none_embedding_fields(self):
        class Embed:
            def to_dict(self):
                return {
                    "inputs_embeds": mx.zeros((1, 2, 4)),
                    "position_ids": None,
                    "rope_deltas": None,
                }

        class Model:
            def get_input_embeddings(
                self, input_ids, pixel_values, mask=None, **kwargs
            ):
                return Embed()

        response_generator = SimpleNamespace(model=Model(), vision_cache=None)

        _, gen_kwargs = server.ResponseGenerator._gpu_embed(
            response_generator,
            {
                "input_ids": mx.array([[1, 2]]),
                "attention_mask": mx.array([[1, 1]]),
            },
            images=None,
        )

        assert "position_ids" not in gen_kwargs
        assert "rope_deltas" not in gen_kwargs
        assert "_apc_semantic_hash" not in gen_kwargs

    def test_gpu_embed_uses_precomputed_semantic_hash(self):
        class Embed:
            def to_dict(self):
                return {"inputs_embeds": mx.zeros((1, 2, 4))}

        class Model:
            def get_input_embeddings(
                self, input_ids, pixel_values, mask=None, **kwargs
            ):
                return Embed()

        response_generator = SimpleNamespace(model=Model(), vision_cache=None)
        pixel_values = mx.array([[[[1.0, 2.0]]]])
        images = ["image-a.png"]
        semantic_hash = apc_module.semantic_extra_hash(
            image_hash=hash_image_payload(pixel_values=pixel_values)
        )

        _, gen_kwargs = server.ResponseGenerator._gpu_embed(
            response_generator,
            {
                "input_ids": mx.array([[1, 2]]),
                "pixel_values": pixel_values,
                "attention_mask": mx.array([[1, 1]]),
            },
            images=images,
            apc_semantic_hash=semantic_hash,
        )

        assert gen_kwargs["_apc_semantic_hash"] == semantic_hash

    def test_extract_chat_response_format_json_schema(self):
        req = SimpleNamespace(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "animal",
                    "schema": {
                        "type": "object",
                        "properties": {"animal": {"type": "string"}},
                        "required": ["animal"],
                    },
                },
            },
            text=None,
        )

        schema = server._extract_response_format_schema(req)

        assert schema["properties"]["animal"]["type"] == "string"

    def test_extract_responses_text_format_json_schema(self):
        req = SimpleNamespace(
            response_format=None,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "animal",
                    "schema": {
                        "type": "object",
                        "properties": {"animal": {"type": "string"}},
                        "required": ["animal"],
                    },
                }
            },
        )

        schema = server._extract_response_format_schema(req)

        assert schema["required"] == ["animal"]

    @pytest.mark.parametrize("format_type", ["json_object", "object"])
    def test_extract_chat_response_format_json_object_aliases(self, format_type):
        req = SimpleNamespace(
            response_format={"type": format_type},
            text=None,
        )

        assert server._extract_response_format_schema(req) == {"type": "object"}

    @pytest.mark.parametrize("format_type", ["json_object", "object"])
    def test_extract_responses_text_format_json_object_aliases(self, format_type):
        req = SimpleNamespace(
            response_format=None,
            text={"format": {"type": format_type}},
        )

        assert server._extract_response_format_schema(req) == {"type": "object"}

    def test_build_structured_logits_processors_uses_tokenizer(self):
        req = SimpleNamespace(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "animal",
                    "schema": {"type": "object"},
                },
            },
            text=None,
        )
        proc = SimpleNamespace(tokenizer=object())

        with patch.object(
            server, "build_json_schema_logits_processor", return_value="processor"
        ) as mock_build:
            processors = server._build_structured_logits_processors(req, proc)

        assert processors == ["processor"]
        assert mock_build.call_args.args[1] == {"type": "object"}

    def test_build_gen_args_preserves_diffusion_options(self):
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            max_denoising_steps=7,
            block_length=16,
            num_to_transfer=3,
            max_transfer_per_step=2,
            editing_threshold=0.8,
            max_post_steps=5,
            stability_steps=1,
            diffusion_full_canvas=True,
            diffusion_min_canvas_length=4,
            diffusion_max_canvas_length=8,
            diffusion_sampler="entropy-bound",
            threshold=0.7,
            min_threshold=0.4,
        )

        args = server._build_gen_args(req)

        expected = {
            "max_denoising_steps": 7,
            "block_length": 16,
            "num_to_transfer": 3,
            "max_transfer_per_step": 2,
            "editing_threshold": 0.8,
            "max_post_steps": 5,
            "stability_steps": 1,
            "diffusion_full_canvas": True,
            "diffusion_min_canvas_length": 4,
            "diffusion_max_canvas_length": 8,
            "diffusion_sampler": "entropy-bound",
            "threshold": 0.7,
            "min_threshold": 0.4,
        }
        assert args.diffusion_kwargs() == expected
        for key, value in expected.items():
            assert args.to_generate_kwargs()[key] == value

    @pytest.mark.parametrize("format_type", ["json_object", "object"])
    def test_build_structured_logits_processors_for_json_object_aliases(
        self, format_type
    ):
        req = SimpleNamespace(
            response_format={"type": format_type},
            text=None,
        )
        proc = SimpleNamespace(tokenizer=object())

        with patch.object(
            server, "build_json_schema_logits_processor", return_value="processor"
        ) as mock_build:
            processors = server._build_structured_logits_processors(req, proc)

        assert processors == ["processor"]
        assert mock_build.call_args.args[1] == {"type": "object"}

    def test_build_gen_args_maps_responses_reasoning_configuration(self):
        req = server.OpenAIRequest(
            model="demo",
            input="hi",
            reasoning={"effort": "high", "summary": "auto"},
        )

        args = server._build_gen_args(req)

        assert args.enable_thinking is True
        assert args.reasoning is True
        assert args.reasoning_effort == "high"
        assert args.to_template_kwargs()["reasoning"] is True
        assert args.to_template_kwargs()["reasoning_effort"] == "high"

    def test_build_gen_args_maps_chat_reasoning_effort(self):
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            reasoning_effort="low",
        )

        args = server._build_gen_args(req)

        assert args.enable_thinking is True
        assert args.reasoning is True
        assert args.reasoning_effort == "low"

    def test_build_gen_args_maps_none_reasoning_effort_to_disabled(self):
        req = server.OpenAIRequest(
            model="demo",
            input="hi",
            reasoning={"effort": "none"},
        )

        args = server._build_gen_args(req)

        assert args.enable_thinking is False
        assert args.reasoning is False
        assert args.reasoning_effort == "none"

    def test_build_gen_args_explicit_thinking_overrides_standard_reasoning(self):
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            enable_thinking=False,
            reasoning_effort="high",
        )

        args = server._build_gen_args(req)

        assert args.enable_thinking is False
        assert args.reasoning is False
        assert args.reasoning_effort == "high"

    def test_log_progress_interval_is_configurable(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_LOG_PROGRESS_INTERVAL", raising=False)
        assert server.get_log_progress_interval() == 10

        monkeypatch.setenv("MLX_VLM_LOG_PROGRESS_INTERVAL", "7")
        assert server.get_log_progress_interval() == 7

        monkeypatch.setenv("MLX_VLM_LOG_PROGRESS_INTERVAL", "-1")
        assert server.get_log_progress_interval() == 0

    def test_debug_decode_logging_adds_token_details(self, monkeypatch, caplog):
        monkeypatch.setenv("MLX_VLM_LOG_PROGRESS_INTERVAL", "2")
        caplog.set_level(logging.DEBUG, logger="mlx_vlm.server")
        info = {
            "request_id": "req-1",
            "queued_at": time.perf_counter() - 0.1,
            "generated_tokens": 0,
            "decode_started_at": None,
        }

        for token_number in range(1, 4):
            server.ResponseGenerator._log_decode_progress(
                1,
                info,
                token=token_number,
                text=str(token_number),
                finish_reason="stop" if token_number == 3 else None,
            )

        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "Decode progress: request=req-1 generated_tokens=1" in m
            and "token_number=1 token_id=1 text='1'" in m
            for m in messages
        )
        assert not any("Token streamed:" in m for m in messages)
        assert any("Decode started: request=req-1" in m for m in messages)
        assert any(
            "Decode completed: request=req-1 generated_tokens=3" in m for m in messages
        )

    def test_info_decode_logging_uses_interval_without_token_details(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv("MLX_VLM_LOG_PROGRESS_INTERVAL", "2")
        caplog.set_level(logging.INFO, logger="mlx_vlm.server")
        info = {
            "request_id": "req-1",
            "queued_at": time.perf_counter(),
            "generated_tokens": 0,
            "decode_started_at": None,
        }

        for token_number in range(1, 3):
            server.ResponseGenerator._log_decode_progress(
                1,
                info,
                token=token_number,
                text=str(token_number),
                finish_reason=None,
            )

        progress = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("Decode progress:")
        ]
        assert len(progress) == 1
        assert "generated_tokens=2" in progress[0]
        assert "token_number=" not in progress[0]
        assert "token_id=" not in progress[0]
        assert "text=" not in progress[0]

    def test_decode_logging_uses_one_rate_field(self, monkeypatch, caplog):
        times = iter([10.0, 10.25])
        monkeypatch.setattr(server_generation.time, "perf_counter", lambda: next(times))
        caplog.set_level(logging.DEBUG, logger="mlx_vlm.server")
        info = {
            "request_id": "req-1",
            "queued_at": 9.0,
            "generated_tokens": 0,
            "decode_started_at": None,
            "last_token_at": None,
        }

        for token_number in range(1, 3):
            server.ResponseGenerator._log_decode_progress(
                1,
                info,
                token=token_number,
                text=str(token_number),
                finish_reason="stop" if token_number == 2 else None,
            )

        progress = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("Decode progress:")
        ]
        assert "rate=n/a" in progress[0]
        assert "rate=4.0 tok/s" in progress[1]
        assert not any("token_rate=" in message for message in progress)
        completed = next(
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("Decode completed:")
        )
        assert "rate=4.0 tok/s" in completed
        assert "token_rate=" not in completed

    def test_chunked_prefill_logging_reports_partial_progress(self, caplog):
        caplog.set_level(logging.INFO, logger="mlx_vlm.server")
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        prompt_batch = SimpleNamespace(
            _processed_prompt_columns=2,
            _inputs_embeds=mx.zeros((1, 4, 8)),
            uids=[1],
            _suffix_lens=[6],
            _cached_tokens_per_row=[0],
            _left_padding_per_row=[0],
            _right_pad_per_row=None,
        )
        active = {1: {"request_id": "req-1", "prefill_processed": -1}}

        gen._log_prefill_progress(SimpleNamespace(_prompt_batch=prompt_batch), active)

        assert "Prefill progress: request=req-1 tokens=2/6 (33.3%)" in caplog.text

    def test_generate_forwards_videos_to_preprocess_and_queue(self):
        # Fork: fork-only test (upstream #1492's server half, restored here).
        """videos must reach prepare_inputs AND ride the queue to the GPU thread.

        Upstream #1492's server half: without the queue field the vision
        embeddings for a video request are silently built from images only.
        """
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.wait_until_ready = lambda: None
        gen.draft_model = None
        gen._cancel = lambda uid: None
        seen = {}
        queued = []

        def fake_cpu_preprocess(prompt, images=None, audio=None, videos=None):
            seen["videos"] = videos
            return {"input_ids": mx.array([[1, 2, 3]], dtype=mx.int32)}

        # generate() blocks on rqueue.get() for the GPU thread's context, so
        # the fake queue has to answer inline.
        class Requests:
            def put(self, item):
                queued.append(item)
                item.rqueue.put(SimpleNamespace(uid="req-1"))

        gen._cpu_preprocess = fake_cpu_preprocess
        gen.requests = Requests()

        gen.generate(
            "describe the clip",
            videos=["clip.mp4"],
            args=server.GenerationArguments(max_tokens=4),
        )

        assert seen["videos"] == ["clip.mp4"]
        assert isinstance(queued[0], server_generation.QueuedGenerationRequest)
        assert queued[0].videos == ["clip.mp4"]

    def test_generate_omits_videos_arg_when_none(self):
        # Fork: fork-only test for the videos call-shape compatibility.
        """A videos=None request must still call the 3-arg _cpu_preprocess.

        ``_preprocess_request`` exists purely to keep that call shape, so
        overrides/fakes that predate the videos parameter keep working.
        """
        gen = self._bare_generator()  # _cpu_preprocess takes exactly 3 args
        gen._cancel = lambda uid: None
        queued = []

        class Requests:
            def put(self, item):
                queued.append(item)
                item.rqueue.put(SimpleNamespace(uid="req-1"))

        gen.requests = Requests()

        gen.generate("hello", args=server.GenerationArguments(max_tokens=4))

        assert queued[0].videos is None

    def test_diffusion_daemon_consumes_full_request_object(self):
        # Fork: fork-only regression test (the fork queues extra request fields).
        """Regression: the diffusion loop unpacked a bare 5-tuple.

        ``generate()`` has always queued more fields than that (the fork adds
        prompt_cache_state + prompt), so every diffusion request raised
        ValueError before the queue became a QueuedGenerationRequest.
        """
        gen = _unstarted_response_generator()
        rqueue: Queue = Queue()
        request = server_generation.QueuedGenerationRequest(
            rqueue=rqueue,
            raw_inputs={"input_ids": mx.array([[1, 2]], dtype=mx.int32)},
            prompt_tokens=2,
            args=server.GenerationArguments(max_tokens=1),
            prompt_cache_state=SimpleNamespace(),
            prompt="hello",
            apc_semantic_hash=37,
        )

        collected = {"count": 0}

        def fake_collect_diffusion_requests(**_kwargs):
            collected["count"] += 1
            if collected["count"] == 1:
                return [request], False
            return [], True

        handled = []
        gen._collect_pending_requests = fake_collect_diffusion_requests
        gen._generate_diffusion = lambda uid, rq, raw, args, cancelled, log_state=None, *, apc_semantic_hash: handled.append(
            (uid, raw, args, apc_semantic_hash)
        )

        gen._run_diffusion()

        assert len(handled) == 1
        assert handled[0][2].max_tokens == 1
        assert isinstance(rqueue.get_nowait(), server.GenerationContext)
        assert rqueue.get_nowait() is None

        assert handled[0][3] == 37


class TestSplitThinking:
    """Tests for thinking tag parsing."""

    def test_channel_tags(self):
        text = "<|channel>thought\nReasoning here.<channel|>The answer."
        reasoning, content = server._split_thinking(text)
        assert reasoning == "Reasoning here."
        assert content == "The answer."

    def test_think_tags(self):
        text = "<think>Thinking.</think>Answer."
        reasoning, content = server._split_thinking(text)
        assert reasoning == "Thinking."
        assert content == "Answer."

    def test_partial_close_tag_only(self):
        text = "Thinking text\n</think>\nAnswer."
        reasoning, content = server._split_thinking(text)
        assert reasoning == "Thinking text"
        assert content == "Answer."

    @pytest.mark.parametrize("prefix", ["", "thought\n"])
    def test_channel_close_only(self, prefix):
        assert server._split_thinking(f"{prefix}got it<channel|>42") == ("got it", "42")

    def test_no_thinking(self):
        text = "Just plain text."
        reasoning, content = server._split_thinking(text)
        assert reasoning is None
        assert content == "Just plain text."

    def test_prompt_opened_thinking_is_detected(self):
        assert server.prompt_has_open_thinking("<|im_start|>assistant\n<think>")
        assert not server.prompt_has_open_thinking("<|im_start|>assistant\n")

    def test_unterminated_thinking_without_markers_is_reasoning(self):
        text = "The user is asking me to say OK. This is a simple request"
        reasoning, content = server._split_thinking(text, starts_in_thinking=True)
        assert reasoning == text
        assert content == ""

    def test_unterminated_thinking_stays_content_when_not_in_block(self):
        text = "The user is asking me to say OK. This is a simple request"
        reasoning, content = server._split_thinking(text, starts_in_thinking=False)
        assert reasoning is None
        assert content == text

    def test_starts_in_thinking_still_splits_on_close_marker(self):
        text = "Reasoning first.</think>The answer."
        reasoning, content = server._split_thinking(text, starts_in_thinking=True)
        assert reasoning == "Reasoning first."
        assert content == "The answer."

    def test_starts_in_thinking_respects_paired_markers(self):
        text = "<think>Thinking.</think>Answer."
        reasoning, content = server._split_thinking(text, starts_in_thinking=True)
        assert reasoning == "Thinking."
        assert content == "Answer."

    def test_empty_content_after_thinking(self):
        text = "<|channel>thought\nOnly thinking.<channel|>"
        reasoning, content = server._split_thinking(text)
        assert reasoning == "Only thinking."
        assert content == ""

    def test_custom_thinking_markers(self):
        text = "<analysis>Custom reasoning.</analysis>Custom answer."
        reasoning, content = server._split_thinking(text, "<analysis>", "</analysis>")
        assert reasoning == "Custom reasoning."
        assert content == "Custom answer."

    def test_response_template_parses_reasoning_and_content(self):
        text = (
            "to=self<|message|>Muse reasoning.<|eom|>"
            "<|start|>assistant to=user<|message|>Muse answer."
        )
        reasoning, content = server._split_thinking(
            text,
            processor=SimpleNamespace(tokenizer=_MuseResponseTemplateTokenizer()),
        )
        assert reasoning == "Muse reasoning."
        assert content == "Muse answer."

    def test_cohere_thinking_markers_strip_text_markers(self):
        text = (
            "<|START_THINKING|>Custom reasoning.<|END_THINKING|>"
            "<|START_TEXT|>Custom answer.<|END_TEXT|>"
        )
        reasoning, content = server._split_thinking(text)
        assert reasoning == "Custom reasoning."
        assert content == "Custom answer."


class TestThinkingStreamState:
    """Tests for streaming thinking tag parsing."""

    def test_last_chunk_releases_text_held_for_an_unfinished_marker(self):
        state = server.ThinkingStreamState()

        assert state.feed("hello <").content == "hello "
        assert state.feed("", last=True).content == "<"

    def test_last_chunk_releases_reasoning_held_for_an_unfinished_marker(self):
        state = server.ThinkingStreamState()

        state.feed("<think>")
        assert state.feed("cut off </thi", last=True).reasoning == "cut off </thi"

    def test_last_chunk_does_not_disturb_a_complete_stream(self):
        state = server.ThinkingStreamState()
        reasoning, content = "", ""

        for chunk in ("<think>", "why", "</think>", "answer"):
            delta = state.feed(chunk, last=chunk == "answer")
            reasoning += delta.reasoning or ""
            content += delta.content or ""

        assert (reasoning, content) == ("why", "answer")

    def test_last_chunk_on_an_empty_buffer_emits_nothing(self):
        state = server.ThinkingStreamState()

        delta = state.feed("", last=True)

        assert (delta.reasoning, delta.content) == (None, None)

    def test_buffer_is_only_released_once(self):
        state = server.ThinkingStreamState()

        state.feed("hello <")
        assert state.feed("", last=True).content == "<"
        assert state.feed("", last=True).content is None

    def test_last_chunk_releases_to_reasoning_when_thinking_opened_in_the_prompt(self):
        state = server.ThinkingStreamState(enable_thinking=True)

        delta = state.feed("why </thi", last=True)

        assert (delta.reasoning, delta.content) == ("why </thi", None)

    def test_last_chunk_releases_a_partial_custom_end_token(self):
        state = server.ThinkingStreamState(
            thinking_start_token="<BEGIN>", thinking_end_token="<END>"
        )

        state.feed("<BEGIN>")
        assert state.feed("work <EN", last=True).reasoning == "work <EN"

    def test_last_chunk_releases_content_held_after_thinking_closed(self):
        state = server.ThinkingStreamState()

        state.feed("<think>x</think>")
        assert state.feed("done <|START_TE", last=True).content == "done <|START_TE"

    def test_last_chunk_still_reports_thinking_closed(self):
        state = server.ThinkingStreamState()

        state.feed("<think>why")
        delta = state.feed("</think>tail <", last=True)

        assert (delta.content, delta.thinking_closed) == ("tail <", True)

    def test_last_chunk_strips_complete_content_markers(self):
        state = server.ThinkingStreamState()

        assert state.feed("hi <|END_TEXT|>", last=True).content == "hi "

    def test_prompt_must_end_with_open_thinking_marker_to_start_in_thinking(self):
        assert server.prompt_has_open_thinking("prompt", enable_thinking=True) is False
        assert (
            server.prompt_has_open_thinking("prompt<think>\n", enable_thinking=True)
            is True
        )
        assert (
            server.prompt_has_open_thinking(
                "User: Say <think> literally\nAssistant:", enable_thinking=True
            )
            is False
        )
        assert (
            server.prompt_has_open_thinking(
                "prompt<analysis>",
                enable_thinking=True,
                thinking_start_token="<analysis>",
                thinking_end_token="</analysis>",
            )
            is True
        )

    def test_prompt_open_marker_is_authoritative_when_flag_is_disabled(self):
        assert (
            server.prompt_has_open_thinking(
                "prompt<|START_THINKING|>", enable_thinking=False
            )
            is True
        )

    @pytest.mark.parametrize("enable_thinking", [False, True])
    def test_gemma_channel_markers_and_content_in_same_delta(self, enable_thinking):
        state = server.ThinkingStreamState(enable_thinking=enable_thinking)
        reasoning = []
        content = []

        for token in _gemma_thinking_channel_chunks():
            delta = state.feed(token.text)
            if delta.reasoning:
                reasoning.append(delta.reasoning)
            if delta.content:
                content.append(delta.content)

        assert "".join(reasoning) == ""
        assert "".join(content) == "7 * 8 = 56"

    def test_think_close_can_emit_reasoning_tail_and_content(self):
        state = server.ThinkingStreamState(enable_thinking=True)

        first = state.feed("thinking")
        second = state.feed(" tail</think>\n\nAnswer")

        assert first.reasoning == "thinking"
        assert first.content is None
        assert first.thinking_closed is False
        assert second.reasoning == " tail"
        assert second.content == "Answer"
        assert second.thinking_closed is True

    def test_custom_markers_split_same_delta_content(self):
        state = server.ThinkingStreamState(
            enable_thinking=False,
            thinking_start_token="<analysis>",
            thinking_end_token="</analysis>",
        )

        first = state.feed("<ana")
        second = state.feed("lysis>Custom reasoning.</analysis>Custom answer.")

        assert first.reasoning is None
        assert first.content is None
        assert second.reasoning == "Custom reasoning."
        assert second.content == "Custom answer."
        assert second.thinking_closed is True

    def test_response_template_markers_split_across_chunks(self):
        state = server.make_response_stream_state(
            SimpleNamespace(tokenizer=_MuseResponseTemplateTokenizer()),
            thinking_start_token="unused-start",
            thinking_end_token="unused-end",
        )
        reasoning = []
        content = []

        chunks = (
            "to=self<|mes",
            "sage|>Muse reasoning.<|eom|><|start|>assistant ",
            "to=user<|message|>Muse answer.",
        )
        thinking_closed = False
        for index, chunk in enumerate(chunks):
            delta = state.feed(chunk, last=index == len(chunks) - 1)
            if delta.reasoning:
                reasoning.append(delta.reasoning)
            if delta.content:
                content.append(delta.content)
            thinking_closed = thinking_closed or delta.thinking_closed

        assert "".join(reasoning) == "Muse reasoning."
        assert "".join(content) == "Muse answer."
        assert thinking_closed is True

    def test_cohere_text_markers_are_suppressed_across_chunks(self):
        state = server.ThinkingStreamState(enable_thinking=True)
        reasoning = []
        content = []

        for chunk in [
            "Custom reasoning.",
            "<|END_THINKING|><|START_",
            "TEXT|>Custom answer.<|END_",
            "TEXT|>",
        ]:
            delta = state.feed(chunk)
            if delta.reasoning:
                reasoning.append(delta.reasoning)
            if delta.content:
                content.append(delta.content)

        assert "".join(reasoning) == "Custom reasoning."
        assert "".join(content) == "Custom answer."


class TestChatMessageSchema:
    """Tests for ChatMessage accepting tool-calling roles and fields."""

    def test_accepts_tool_role(self):
        msg = server.ChatMessage(role="tool", content="result", tool_call_id="tc_1")
        assert msg.role == "tool"
        assert msg.tool_call_id == "tc_1"

    def test_accepts_assistant_with_tool_calls(self):
        msg = server.ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[{"id": "tc_1", "function": {"name": "f", "arguments": "{}"}}],
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_reasoning_field(self):
        msg = server.ChatMessage(
            role="assistant", content="answer", reasoning="thought"
        )
        assert msg.reasoning == "thought"
        assert msg.reasoning_content == "thought"

    def test_reasoning_content_field(self):
        msg = server.ChatMessage(
            role="assistant", content="answer", reasoning_content="thought"
        )
        assert msg.reasoning_content == "thought"
        assert msg.reasoning == "thought"


class TestToolCallStreamState:
    """Tests for tool-call markup suppression in streaming."""

    def test_no_tool_module(self):
        state = server.ToolCallStreamState(None, None)
        assert state.feed("world") == "world"

    def test_normal_text_before_tool_call(self):
        state = server.ToolCallStreamState("<tool_call>", "</tool_call>")
        assert state.feed("I will call") == "I will call"
        assert state.in_tool_call is False

    def test_suppresses_on_start_marker(self):
        state = server.ToolCallStreamState("<tool_call>", "</tool_call>")
        assert state.feed("text<tool_call>") == "text"
        assert state.in_tool_call is True

    def test_suppresses_partial_marker(self):
        state = server.ToolCallStreamState("<tool_call>", "</tool_call>")
        assert state.feed("text<tool") == "text"
        assert state.buffer == "<tool"
        assert state.in_tool_call is False

    def test_stays_suppressed_after_entering(self):
        state = server.ToolCallStreamState("<tool_call>", "</tool_call>")
        assert state.feed("text<tool_call>") == "text"
        assert state.feed("get_weather") is None
        assert state.in_tool_call is True

    def test_pipe_delimited_marker(self):
        state = server.ToolCallStreamState("<|tool_call>", "<|tool_call_end|>")
        assert state.feed("text<|tool_call>call:get_weather") == "text"
        assert state.in_tool_call is True

    def test_pipe_delimited_partial_marker(self):
        state = server.ToolCallStreamState("<|tool_call>", "<|tool_call_end|>")
        assert state.feed("text<|tool") == "text"
        assert state.buffer == "<|tool"
        assert state.in_tool_call is False


class TestProcessToolCalls:
    """Tests for tool call parsing from model output."""

    def test_no_tool_calls(self):
        # Minimal tool module mock
        module = SimpleNamespace(tool_call_start="<tc>", tool_call_end="</tc>")
        result = server.process_tool_calls("Just text.", module, [])
        assert result.calls == []
        assert result.remaining_text == "Just text."

    def test_parser_can_return_multiple_tool_calls(self):
        module = SimpleNamespace(
            tool_call_start="<tc>",
            tool_call_end="</tc>",
            parse_tool_call=lambda call, tools: [
                {"name": "grep", "arguments": {"pattern": "foo"}},
                {"name": "read", "arguments": {"path": "file.py"}},
            ],
        )

        result = server.process_tool_calls("Before <tc>[]</tc> after", module, [])

        assert result.remaining_text == "Before   after"
        assert [call["function"]["name"] for call in result.calls] == [
            "grep",
            "read",
        ]
        assert json.loads(result.calls[0]["function"]["arguments"]) == {
            "pattern": "foo"
        }
        assert json.loads(result.calls[1]["function"]["arguments"]) == {
            "path": "file.py"
        }

    minicpm5_call = (
        '<function name="write_file"><param name="content">'
        "<![CDATA[  <html>\nA & B\n</html>  ]]></param>"
        '<param name="version">123</param><param name="count">3</param>'
        '<param name="enabled">True</param></function>'
    )

    def test_detects_minicpm5_chat_template(self):
        template = """{{ '<function name="' ~ tool_call.name ~ '">' }}
    {{ '<param name="' ~ param_name ~ '">' }}"""
        assert server.load_tool_module(_infer_tool_parser(template)) is minicpm5

    def test_minicpm5_cdata_and_argument_types(self):
        tools = [
            {
                "function": {
                    "name": "write_file",
                    "parameters": {"properties": {"version": {"type": "string"}}},
                }
            }
        ]
        result = minicpm5.parse_tool_call(self.minicpm5_call, tools)

        assert result == {
            "name": "write_file",
            "arguments": {
                "content": "  <html>\nA & B\n</html>  ",
                "version": "123",
                "count": 3,
                "enabled": True,
            },
        }

    @pytest.mark.parametrize(
        "text",
        [
            '<function name="lookup"><param name="value">unfinished',
            '<function name=""></function>',
            '<function name="lookup"><param>3</param></function>',
        ],
    )
    def test_minicpm5_rejects_malformed_calls(self, text):
        with pytest.raises(ValueError):
            minicpm5.parse_tool_call(text)


class TestCountThinkingTagTokens:
    """Tests for thinking tag token counting."""

    def test_channel_tags(self):
        assert (
            server._count_thinking_tag_tokens("<|channel>thought\ntext<channel|>answer")
            == 4
        )

    def test_think_tags(self):
        assert server._count_thinking_tag_tokens("<think>text</think>answer") == 2

    def test_no_tags(self):
        assert server._count_thinking_tag_tokens("plain text") == 0


class TestQuantizedKVBits:
    def test_kv_bits_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv("KV_BITS", raising=False)
        assert server_generation.get_quantized_kv_bits() is None

    def test_kv_bits_applies(self, monkeypatch):
        monkeypatch.setenv("KV_BITS", "3.5")
        assert server_generation.get_quantized_kv_bits() == 3.5

    @pytest.mark.parametrize(
        "model_path",
        [
            "mlx-community/gemma-4-31B-it-qat-mxfp4",
            "mlx-community/gemma-4-31B-it-QAT-mxfp4",
            "/models/qat-experiments/llama-3",
            "some-org/qatar-news-llm",
        ],
    )
    def test_kv_bits_not_suppressed_by_model_path(self, monkeypatch, model_path):
        # KV cache quantization is independent of how the weights were trained,
        # so nothing in the model path may suppress it (#1333).
        monkeypatch.setenv("KV_BITS", "3.5")
        monkeypatch.setenv("MAX_KV_SIZE", "0")
        assert server_generation.get_quantized_kv_bits() == 3.5
        assert server_generation.get_max_kv_size(model_path) is None

    def test_split_bits_agree_with_uniform_bits(self, monkeypatch):
        # The split path never had a model-path guard; both must behave alike.
        monkeypatch.setenv("KV_BITS", "3.5")
        monkeypatch.setenv("KV_KEY_BITS", "3")
        monkeypatch.setenv("KV_VALUE_BITS", "4")
        assert server_generation.get_quantized_kv_bits() == 3.5
        assert server_generation.get_quantized_kv_split_bits() == (3.0, 4.0)


class TestRuntimeConfig:
    def test_from_env_seeds_defaults(self, monkeypatch):
        monkeypatch.setenv("KV_QUANT_SCHEME", "group")
        monkeypatch.setenv("KV_BITS", "6")
        monkeypatch.setenv("APC_ENABLED", "1")
        monkeypatch.setenv("MLX_VLM_VISION_CACHE_SIZE", "33")
        cfg = RuntimeConfig.from_env()
        assert cfg.kv_quant_scheme == "group"
        assert cfg.kv_bits == 6.0
        assert cfg.apc_enabled is True
        assert cfg.vision_cache_size == 33

    def test_fingerprint_stable_and_scoped(self):
        cfg = RuntimeConfig.from_env()
        fp = cfg.fingerprint()
        assert fp == cfg.fingerprint()  # stable across calls

        cfg2 = RuntimeConfig.from_env()
        assert cfg2.fingerprint() == fp

        # toggling a dead APC knob while APC is off must not invalidate
        cfg2.apc_block_size = 64
        assert cfg2.fingerprint() == fp

        # toggling an effective knob must invalidate
        cfg2.vision_cache_size = 40
        assert cfg2.fingerprint() != fp

    def test_apply_changes_validates_and_coerces(self):
        cfg = RuntimeConfig.from_env()
        applied, rejected = cfg.apply_changes(
            {
                "kv_bits": "8",  # str -> float coercion
                "apc_enabled": "true",
                "vision_cache_size": "50",
                "not_a_knob": 1,
            }
        )
        assert applied == {
            "kv_bits": 8.0,
            "apc_enabled": True,
            "vision_cache_size": 50,
        }
        assert rejected == [{"name": "not_a_knob", "reason": "unknown knob"}]
        assert cfg.kv_bits == 8.0
        assert cfg.apc_enabled is True
        assert cfg.vision_cache_size == 50

    def test_apply_changes_rejects_bad_values(self):
        cfg = RuntimeConfig.from_env()
        applied, rejected = cfg.apply_changes({"kv_bits": "not-a-number"})
        assert applied == {}
        assert len(rejected) == 1
        assert rejected[0]["name"] == "kv_bits"
        assert cfg.kv_bits is None  # unchanged

    def test_reload_kinds_scoped(self):
        cfg = RuntimeConfig.from_env()
        applied, _ = cfg.apply_changes({"kv_quant_scheme": "turboquant"})
        assert cfg.reload_kinds(applied) == {"text_generation"}
        applied, _ = cfg.apply_changes({"vision_cache_size": 10})
        assert cfg.reload_kinds(applied) == {"image_generation", "image_edit"}

    def test_reload_kinds_excludes_live_knobs(self):
        cfg = RuntimeConfig.from_env()
        before = cfg.fingerprint()
        applied, _ = cfg.apply_changes(
            {"max_kv_size": 4096, "token_queue_timeout": 30.0}
        )
        assert applied == {"max_kv_size": 4096, "token_queue_timeout": 30.0}
        assert cfg.reload_kinds(applied) == set()
        assert cfg.fingerprint() == before

    def test_reload_kinds_excludes_apc_knobs_while_disabled(self):
        cfg = RuntimeConfig.from_env()
        cfg.apply_changes({"apc_enabled": False})
        before = cfg.fingerprint()

        applied, _ = cfg.apply_changes({"apc_block_size": 32})
        assert applied == {"apc_block_size": 32}
        assert cfg.reload_kinds(applied) == set()
        assert cfg.fingerprint() == before

        applied, _ = cfg.apply_changes({"apc_enabled": True, "apc_block_size": 64})
        assert cfg.reload_kinds(applied) == {"text_generation"}
        assert cfg.fingerprint() != before

    def test_settings_endpoints_get_and_patch(self, client, monkeypatch):
        monkeypatch.setattr(server.runtime, "config", RuntimeConfig.from_env())
        cfg = server.runtime.config

        r = client.get("/v1/settings")
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"schema", "current", "fingerprint"}
        names = {k["name"] for k in body["schema"]}
        assert "kv_bits" in names and "apc_enabled" in names
        assert body["current"]["kv_quant_scheme"] == cfg.kv_quant_scheme

        before = cfg.fingerprint()
        r = client.patch("/v1/settings", json={"kv_quant_scheme": "turboquant"})
        assert r.status_code == 200
        body = r.json()
        assert body["applied"] == {"kv_quant_scheme": "turboquant"}
        assert body["rejected"] == []
        assert body["reload_kinds"] == ["text_generation"]
        assert body["current"]["kv_quant_scheme"] == "turboquant"
        assert body["fingerprint"] != before

        r = client.get("/v1/settings")
        assert r.json()["current"]["kv_quant_scheme"] == "turboquant"

        # unknown knobs are never applied
        r = client.patch("/v1/settings", json={"bogus": 1})
        assert r.status_code == 200
        assert r.json()["applied"] == {}
        assert r.json()["rejected"] == [{"name": "bogus", "reason": "unknown knob"}]

    def test_settings_patch_requires_json_object(self, client, monkeypatch):
        monkeypatch.setattr(server.runtime, "config", RuntimeConfig.from_env())
        r = client.patch("/v1/settings", json=[1, 2, 3])
        assert r.status_code == 400


class TestRuntimeConfigAdditions:
    def test_schema_includes_live_and_reloadable_knobs(self):
        cfg = RuntimeConfig.from_env()
        spec = {k["name"]: k for k in cfg.schema()}
        for name in (
            "max_kv_size",
            "token_queue_timeout",
            "spec_draft_model",
            "spec_draft_kind",
        ):
            assert name in spec
            assert spec[name]["reload_kinds"] == ["text_generation"]

    def test_token_queue_timeout_is_live(self, client, monkeypatch):
        monkeypatch.delenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", raising=False)
        monkeypatch.setattr(server.runtime, "config", RuntimeConfig.from_env())
        cfg = server.runtime.config
        fingerprint = cfg.fingerprint()

        response = client.patch("/v1/settings", json={"token_queue_timeout": 1200})

        assert response.status_code == 200
        assert response.json()["applied"] == {"token_queue_timeout": 1200.0}
        assert server.get_token_queue_timeout() == 1200.0
        assert cfg.fingerprint() == fingerprint

        response = client.patch("/v1/settings", json={"token_queue_timeout": 0})

        assert response.json()["applied"] == {"token_queue_timeout": None}
        assert server.get_token_queue_timeout() is None

    def test_max_kv_size_is_live_context_limit(self, monkeypatch):
        import mlx_vlm.server.generation as server_generation

        monkeypatch.setattr(server.runtime.config, "max_kv_size", 4096)
        assert server_generation.get_configured_context_limit() == 4096

        monkeypatch.setattr(server.runtime.config, "max_kv_size", None)
        monkeypatch.delenv("MAX_KV_SIZE", raising=False)
        assert server_generation.get_configured_context_limit() is None

    def test_spec_draft_knob_reaches_generator(self, monkeypatch):
        class FakeResponseGenerator:
            last_kwargs = {}

            def __init__(self, *args, **kwargs):
                FakeResponseGenerator.last_kwargs = kwargs
                self.model = SimpleNamespace()
                self.processor = SimpleNamespace()
                self.config = SimpleNamespace(model_type="qwen2_vl")

            def wait_until_ready(self):
                return self.model, self.processor, self.config

            def stop_and_join(self):
                pass

        monkeypatch.setattr(
            server._app_module, "ResponseGenerator", FakeResponseGenerator
        )
        monkeypatch.setattr(server._app_module._apc, "from_env", lambda *_, **__: None)
        monkeypatch.setattr(server.runtime, "model_cache", {})
        monkeypatch.setattr(server.runtime, "response_generator", None)
        monkeypatch.setattr(server.runtime, "apc_manager", None)
        monkeypatch.setattr(server.runtime.config, "spec_draft_model", "draft-x")
        monkeypatch.setattr(server.runtime.config, "spec_draft_kind", "auto")

        server.get_cached_model("demo-model")
        assert FakeResponseGenerator.last_kwargs["draft_model_path"] == "draft-x"
        assert FakeResponseGenerator.last_kwargs["draft_kind"] == "auto"

    def test_settings_patch_replace_semantics(self, client, monkeypatch):
        monkeypatch.setattr(server.runtime, "config", RuntimeConfig.from_env())
        cfg = server.runtime.config
        assert cfg.apc_enabled is False

        client.patch(
            "/v1/settings",
            json={"kv_quant_scheme": "turboquant", "apc_enabled": True},
        )
        assert cfg.kv_quant_scheme == "turboquant"
        assert cfg.apc_enabled is True

        r = client.patch(
            "/v1/settings",
            json={"op": "replace", "values": {"kv_quant_scheme": "uniform"}},
        )
        body = r.json()
        assert body["op"] == "replace"
        assert cfg.kv_quant_scheme == "uniform"
        assert cfg.apc_enabled is False

        r = client.patch("/v1/settings", json={"op": "bogus", "values": {}})
        assert r.status_code == 400

        r = client.patch("/v1/settings", json={"op": "replace", "values": "x"})
        assert r.status_code == 400


def test_runtime_config_fingerprint_is_kind_scoped():
    cfg = RuntimeConfig.from_env()
    text_fp = cfg.fingerprint(kinds={"text_generation"})
    vision_fp = cfg.fingerprint(kinds={"image_generation"})

    cfg.apply_changes({"kv_quant_scheme": "turboquant"})
    assert cfg.fingerprint(kinds={"text_generation"}) != text_fp
    assert cfg.fingerprint(kinds={"image_generation"}) == vision_fp

    cfg.apply_changes({"vision_cache_size": 64})
    assert cfg.fingerprint(kinds={"image_generation"}) != vision_fp
    assert cfg.fingerprint(kinds={"text_generation"}) != text_fp


def test_runtime_config_live_knob_not_in_fingerprint():
    cfg = RuntimeConfig.from_env()
    fp = cfg.fingerprint()
    cfg.apply_changes({"max_kv_size": 8192})
    assert cfg.fingerprint() == fp


def test_runtime_config_enum_knobs_reject_invalid():
    cfg = RuntimeConfig.from_env()
    applied, rejected = cfg.apply_changes({"kv_quant_scheme": "bogus"})
    assert applied == {}
    assert rejected[0]["name"] == "kv_quant_scheme"
    assert "bogus" in rejected[0]["reason"]
    assert cfg.kv_quant_scheme == "uniform"

    applied, rejected = cfg.apply_changes({"kv_quant_scheme": "turboquant"})
    assert applied == {"kv_quant_scheme": "turboquant"}
    assert rejected == []


class TestReranking:
    def test_requires_model(self, client, monkeypatch):
        monkeypatch.delenv("MLX_VLM_PRELOAD_RERANKER_MODEL", raising=False)

        response = client.post("/v1/rerank", json={"query": "q", "documents": ["d"]})

        assert response.status_code == 400
        assert "No reranker model specified" in response.json()["detail"]

    def test_sorts_limits_and_returns_documents(self, client, monkeypatch):
        cache_calls = []

        def fake_get_cached_model(model, *, model_kind):
            cache_calls.append((model, model_kind))
            return object(), object(), SimpleNamespace(model_type="qwen3")

        monkeypatch.setattr(server, "get_cached_model", fake_get_cached_model)
        monkeypatch.setattr(
            server_reranking,
            "score_documents",
            lambda *args: ([0.2, 0.9, 0.5], 12),
        )

        response = client.post(
            "/v1/rerank",
            json={
                "model": "reranker",
                "query": "query",
                "documents": ["first", "second", "third"],
                "top_n": 2,
                "return_documents": True,
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "model": "reranker",
            "results": [
                {"index": 1, "relevance_score": 0.9, "document": "second"},
                {"index": 2, "relevance_score": 0.5, "document": "third"},
            ],
            "usage": {"prompt_tokens": 12, "total_tokens": 12},
        }
        assert cache_calls == [("reranker", "reranker")]

    def test_preserves_input_order_for_equal_scores(self, client, monkeypatch):
        monkeypatch.setattr(
            server,
            "get_cached_model",
            lambda *args, **kwargs: (
                object(),
                object(),
                SimpleNamespace(model_type="qwen3"),
            ),
        )
        monkeypatch.setattr(
            server_reranking,
            "score_documents",
            lambda *args: ([0.5, 0.5, 0.5], 3),
        )

        response = client.post(
            "/v1/rerank",
            json={
                "model": "reranker",
                "query": "query",
                "documents": ["a", "b", "c"],
            },
        )

        assert [result["index"] for result in response.json()["results"]] == [0, 1, 2]

    def test_uses_preloaded_model(self, client, monkeypatch):
        monkeypatch.setenv("MLX_VLM_PRELOAD_RERANKER_MODEL", "preloaded")
        seen = []

        def fake_get_cached_model(model, *, model_kind):
            seen.append((model, model_kind))
            return object(), object(), SimpleNamespace(model_type="qwen3")

        monkeypatch.setattr(server, "get_cached_model", fake_get_cached_model)
        monkeypatch.setattr(
            server_reranking, "score_documents", lambda *args: ([0.7], 4)
        )

        response = client.post(
            "/v1/rerank", json={"query": "query", "documents": ["document"]}
        )

        assert response.status_code == 200
        assert response.json()["model"] == "preloaded"
        assert seen == [("preloaded", "reranker")]

    def test_uses_cached_preload_after_environment_is_consumed(
        self, client, monkeypatch
    ):
        monkeypatch.delenv("MLX_VLM_PRELOAD_RERANKER_MODEL", raising=False)
        registry = server.ModelCacheRegistry()
        registry.set("reranker", {"model_path": "preloaded"})
        monkeypatch.setattr(server.runtime, "model_cache", registry)
        monkeypatch.setattr(
            server,
            "get_cached_model",
            lambda *args, **kwargs: (
                object(),
                object(),
                SimpleNamespace(model_type="qwen3"),
            ),
        )
        monkeypatch.setattr(
            server_reranking, "score_documents", lambda *args: ([0.7], 4)
        )

        response = client.post(
            "/v1/rerank", json={"query": "query", "documents": ["document"]}
        )

        assert response.status_code == 200
        assert response.json()["model"] == "preloaded"

    def test_uses_server_authentication(self, client, monkeypatch):
        monkeypatch.setenv("MLX_VLM_SERVER_API_KEY", "secret")
        monkeypatch.setattr(
            server,
            "get_cached_model",
            lambda *args, **kwargs: (
                object(),
                object(),
                SimpleNamespace(model_type="qwen3"),
            ),
        )
        monkeypatch.setattr(
            server_reranking, "score_documents", lambda *args: ([0.5], 1)
        )
        payload = {"model": "reranker", "query": "q", "documents": ["d"]}

        assert client.post("/v1/rerank", json=payload).status_code == 401
        response = client.post(
            "/v1/rerank",
            json=payload,
            headers={"Authorization": "Bearer secret"},
        )

        assert response.status_code == 200

    @pytest.mark.parametrize(
        "value,label,expected",
        [
            ("  text  ", "query", server_reranking.RerankItem(text="text")),
            (
                {"text": " text "},
                "query",
                server_reranking.RerankItem(text="text"),
            ),
            (
                {"image_url": {"url": " image.png "}},
                "documents[0]",
                server_reranking.RerankItem(image="image.png"),
            ),
            (
                {"video": " video.mp4 "},
                "documents[0]",
                server_reranking.RerankItem(video="video.mp4"),
            ),
        ],
    )
    def test_normalizes_items(self, value, label, expected):
        assert server_reranking.normalize_item(value, label) == expected

    @pytest.mark.parametrize("value", ["", "   ", {}, {"text": " "}, {"image": {}}])
    def test_rejects_empty_items(self, value):
        with pytest.raises(ValueError):
            server_reranking.normalize_item(value, "query")

    def test_text_model_rejects_media(self):
        with pytest.raises(ValueError, match="do not support image or video"):
            server_reranking.score_documents(
                object(),
                object(),
                SimpleNamespace(model_type="qwen3"),
                server_reranking.RerankItem(image="image.png"),
                [server_reranking.RerankItem(text="document")],
                "instruction",
            )

    def test_vl_messages_preserve_content_order(self):
        messages = server_reranking._vl_messages(
            server_reranking.RerankItem(text="query", image="query.png"),
            server_reranking.RerankItem(text="document", video="document.mp4"),
            "rank candidates",
        )

        assert messages[1]["content"] == [
            {"type": "text", "text": "<Instruct>: rank candidates"},
            {"type": "text", "text": "<Query>:"},
            {"type": "image"},
            {"type": "text", "text": "query"},
            {"type": "text", "text": "\n<Document>:"},
            {"type": "video"},
            {"type": "text", "text": "document"},
        ]

    def test_batches_without_reordering(self, monkeypatch):
        batches = []

        def fake_score_batch(model, processor, query, documents, instruction):
            del model, processor, query, instruction
            batches.append([document.text for document in documents])
            return [float(document.text) for document in documents], len(documents)

        monkeypatch.setenv("MLX_VLM_RERANK_BATCH_SIZE", "2")
        monkeypatch.setattr(server_reranking, "_score_text_batch", fake_score_batch)
        documents = [server_reranking.RerankItem(text=str(index)) for index in range(5)]

        scores, tokens = server_reranking.score_documents(
            object(),
            object(),
            SimpleNamespace(model_type="qwen3"),
            server_reranking.RerankItem(text="query"),
            documents,
            "instruction",
        )

        assert scores == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert tokens == 5
        assert batches == [["0", "1"], ["2", "3"], ["4"]]

    def test_generative_reranker_uses_default_instruction(self, monkeypatch):
        instructions = []

        def fake_score_batch(model, processor, query, documents, instruction):
            del model, processor, query
            instructions.append(instruction)
            return [0.5] * len(documents), len(documents)

        monkeypatch.setattr(server_reranking, "_score_text_batch", fake_score_batch)

        server_reranking.score_documents(
            object(),
            object(),
            SimpleNamespace(model_type="qwen3"),
            server_reranking.RerankItem(text="query"),
            [server_reranking.RerankItem(text="document")],
            None,
        )

        assert instructions == [server_reranking.DEFAULT_INSTRUCTION]

    def test_sequence_classifier_scores_tokenized_pairs(self):
        calls = []

        class Tokenizer:
            model_max_length = 6

            def __call__(self, queries, documents, **kwargs):
                calls.append((queries, documents, kwargs))
                return {
                    "input_ids": np.array([[1, 2, 3, 0], [1, 4, 5, 6]]),
                    "attention_mask": np.array([[1, 1, 1, 0], [1, 1, 1, 1]]),
                    "token_type_ids": np.array([[0, 0, 1, 0], [0, 0, 1, 1]]),
                }

        class Model:
            def __call__(self, **inputs):
                assert set(inputs) == {
                    "input_ids",
                    "attention_mask",
                    "token_type_ids",
                }
                return SimpleNamespace(logits=mx.array([[-2.0], [2.0]]))

        scores, tokens = server_reranking.score_documents(
            Model(),
            Tokenizer(),
            SimpleNamespace(model_type="bert", max_position_embeddings=4),
            server_reranking.RerankItem(text="query"),
            [
                server_reranking.RerankItem(text="first"),
                server_reranking.RerankItem(text="second"),
            ],
            None,
        )

        assert scores == pytest.approx([1 / (1 + math.exp(2)), 1 / (1 + math.exp(-2))])
        assert tokens == 7
        assert calls == [
            (
                ["query", "query"],
                ["first", "second"],
                {
                    "padding": True,
                    "truncation": True,
                    "max_length": 4,
                    "return_tensors": "np",
                },
            )
        ]

    @pytest.mark.parametrize(
        "query,documents,instruction,error",
        [
            (
                server_reranking.RerankItem(image="query.png"),
                [server_reranking.RerankItem(text="document")],
                None,
                "do not support image or video",
            ),
            (
                server_reranking.RerankItem(text="query"),
                [server_reranking.RerankItem(text="document")],
                "rank legal documents",
                "do not support custom instructions",
            ),
        ],
    )
    def test_sequence_classifier_rejects_unsupported_inputs(
        self, query, documents, instruction, error
    ):
        with pytest.raises(ValueError, match=error):
            server_reranking.score_documents(
                object(),
                object(),
                SimpleNamespace(model_type="modernbert"),
                query,
                documents,
                instruction,
            )

    def test_attention_mask_combines_padding_and_causality(self):
        mask = server_reranking._attention_mask(mx.array([[0, 1, 1], [1, 1, 0]]))

        assert mask.shape == (2, 1, 3, 3)
        assert mask[0, 0].tolist() == [
            [False, False, False],
            [False, True, False],
            [False, True, True],
        ]
        assert mask[1, 0].tolist() == [
            [True, False, False],
            [True, True, False],
            [False, False, False],
        ]

    def test_attention_mask_uses_native_causal_path_without_padding(self):
        assert server_reranking._attention_mask(mx.ones((2, 3))) == "causal"

    def test_binary_scores_pool_last_non_padding_token(self):
        model = SimpleNamespace(
            language_model=SimpleNamespace(lm_head=lambda hidden_states: hidden_states)
        )
        tokenizer = SimpleNamespace(
            unk_token_id=None,
            convert_tokens_to_ids=lambda token: {"no": 0, "yes": 1}[token],
        )
        hidden_states = mx.array(
            [
                [[9.0, -9.0], [2.0, 4.0], [1.0, 5.0]],
                [[4.0, 1.0], [8.0, 2.0], [-9.0, 9.0]],
            ]
        )

        scores = server_reranking._binary_scores(
            model, hidden_states, mx.array([[0, 1, 1], [1, 1, 0]]), tokenizer
        )

        assert scores == pytest.approx([1 / (1 + math.exp(-4)), 1 / (1 + math.exp(6))])

    @pytest.mark.parametrize(
        "value",
        [
            [1, 2, 3],
            {"input_ids": [1, 2, 3]},
            SimpleNamespace(input_ids=[1, 2, 3]),
            SimpleNamespace(input_ids=[[1, 2, 3]]),
            mx.array([1, 2, 3]),
        ],
    )
    def test_input_ids_accept_tokenizer_return_types(self, value):
        assert server_reranking._input_ids(value) == [1, 2, 3]

    def test_ensure_chat_template_loads_packaged_template(self, tmp_path, monkeypatch):
        (tmp_path / "chat_template.jinja").write_text("template", encoding="utf-8")
        processor = SimpleNamespace(
            chat_template=None,
            tokenizer=SimpleNamespace(chat_template=None),
        )
        monkeypatch.setattr(server_reranking, "get_model_path", lambda path: tmp_path)

        server_reranking.ensure_chat_template(processor, "reranker")

        assert processor.chat_template == "template"
        assert processor.tokenizer.chat_template == "template"

    def test_model_uses_isolated_cache(self, monkeypatch):
        registry = server.ModelCacheRegistry()
        text_cache = {
            "cache_key": ("language", None, "text_generation"),
            "model_kind": "text_generation",
        }
        registry.set("text_generation", text_cache)
        monkeypatch.setattr(server.runtime, "model_cache", registry)
        model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3"))
        processor = object()
        monkeypatch.setattr(
            reranker_loader, "load_reranker", lambda path: (model, processor)
        )
        monkeypatch.setattr(
            server._app_module, "ensure_reranker_chat_template", lambda *args: None
        )

        loaded = server.get_cached_model("reranker", None, model_kind="reranker")

        assert loaded == (model, processor, model.config)
        assert registry.for_kind("text_generation") is text_cache
        assert registry.for_kind("reranker")["cache_key"] == (
            "reranker",
            None,
            "reranker",
            server.runtime.config.fingerprint(kinds={"reranker"}),
        )

    def test_loader_rejects_unsupported_family(self, monkeypatch):
        monkeypatch.setattr(server.runtime, "model_cache", server.ModelCacheRegistry())
        model = SimpleNamespace(config=SimpleNamespace(model_type="deberta_v2"))
        monkeypatch.setattr(
            reranker_loader, "load_reranker", lambda path: (model, object())
        )

        with pytest.raises(
            server.HTTPException, match="Unsupported reranker model type"
        ) as exc:
            server.get_cached_model("reranker", None, model_kind="reranker")

        assert exc.value.status_code == 400

    def test_loader_skips_chat_template_for_sequence_classifier(self, monkeypatch):
        monkeypatch.setattr(server.runtime, "model_cache", server.ModelCacheRegistry())
        model = SimpleNamespace(config=SimpleNamespace(model_type="bert"))
        processor = object()
        monkeypatch.setattr(
            reranker_loader, "load_reranker", lambda path: (model, processor)
        )
        monkeypatch.setattr(
            server._app_module,
            "ensure_reranker_chat_template",
            lambda *args: pytest.fail("sequence classifiers do not use chat templates"),
        )

        loaded = server.get_cached_model("reranker", None, model_kind="reranker")

        assert loaded == (model, processor, model.config)


# ============================================================================
# Fork: fork-ported test classes (server-port adaptation, 2026-06-06). Everything
# below this banner is fork content; upstream's file ends before it.
# ============================================================================
# Patch-target remaps: TOKEN_QUEUE_TIMEOUT_SECS / CACHED_PATH_HEARTBEAT_INTERVAL_SECS
# moved to mlx_vlm.server.generation (patched via the server_generation alias).
# Functions read via server.<name> resolve through the package re-exports.
#
# Intentionally NOT ported (features replaced upstream):
#   - TestComputeThinkingBudget (fork auto-budget -> upstream ThinkingBudgetCriteria)
#   - TestStepThinkingState::test_round_trip_multiple_thinking_blocks
#     (ported _step_thinking_state lstrips leading newline after opener)
#   - TestCountThinkingTagTokens::test_channel_tags fork variant
#     (upstream's version above already counts <|channel>thought as 4 tokens)
# ============================================================================


class _FakeTokenizer:
    """Minimal tokenizer stub whose apply_chat_template emulates the
    rendering behavior of a real chat template for prefill-opener detection.
    """

    def __init__(self, suffix_with_gen: str, suffix_no_gen: str = ""):
        self._suffix_with_gen = suffix_with_gen
        self._suffix_no_gen = suffix_no_gen

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, **kwargs
    ):
        body = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        return body + (
            self._suffix_with_gen if add_generation_prompt else self._suffix_no_gen
        )


class _FakeProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class TestHasPrefilledOpener:
    """Tests for _has_prefilled_opener detection across template families."""

    def setup_method(self):
        server._PREFILL_FLAG_CACHE.clear()

    def test_unsloth_qwen_thinking_on_is_prefilled(self):
        # unsloth Qwen 3.6 with enable_thinking=True ends with <think>\n
        proc = _FakeProcessor(
            _FakeTokenizer(
                suffix_with_gen="<|im_start|>assistant\n<think>\n",
                suffix_no_gen="",
            )
        )
        assert server._has_prefilled_opener(proc, {"enable_thinking": True}) is True

    def test_unsloth_qwen_thinking_off_is_not_prefilled(self):
        # enable_thinking=False renders the empty pair; suffix ends with </think>
        proc = _FakeProcessor(
            _FakeTokenizer(
                suffix_with_gen="<|im_start|>assistant\n<think>\n\n</think>\n\n",
                suffix_no_gen="",
            )
        )
        assert server._has_prefilled_opener(proc, {"enable_thinking": False}) is False

    def test_canonical_qwen_thinking_on_is_not_prefilled(self):
        # canonical Qwen 3 leaves the assistant header bare; model emits both tags
        proc = _FakeProcessor(
            _FakeTokenizer(suffix_with_gen="<|im_start|>assistant\n", suffix_no_gen="")
        )
        assert server._has_prefilled_opener(proc, {"enable_thinking": True}) is False

    def test_gemma_native_opener_is_prefilled(self):
        # If a future Gemma-style template prefilled <|channel>thought
        proc = _FakeProcessor(
            _FakeTokenizer(
                suffix_with_gen="<start_of_turn>model\n<|channel>thought",
                suffix_no_gen="",
            )
        )
        assert server._has_prefilled_opener(proc, {"enable_thinking": True}) is True

    def test_template_render_failure_returns_false(self):
        class _BrokenTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                raise RuntimeError("template error")

        proc = _FakeProcessor(_BrokenTokenizer())
        assert server._has_prefilled_opener(proc, {}) is False

    def test_caches_result_on_repeat_calls(self):
        calls = {"count": 0}

        class _CountingTokenizer(_FakeTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                calls["count"] += 1
                return super().apply_chat_template(*args, **kwargs)

        proc = _FakeProcessor(
            _CountingTokenizer(suffix_with_gen="<|im_start|>assistant\n<think>\n")
        )
        kwargs = {"enable_thinking": True}

        assert server._has_prefilled_opener(proc, kwargs) is True
        first_call_count = calls["count"]
        # Second call with identical kwargs hits cache (no new renders)
        assert server._has_prefilled_opener(proc, kwargs) is True
        assert calls["count"] == first_call_count

    def test_distinct_kwargs_get_distinct_cache_entries(self):
        proc = _FakeProcessor(
            _FakeTokenizer(suffix_with_gen="<|im_start|>assistant\n<think>\n")
        )
        # Both should be True for this stub, but they exercise the cache key
        assert server._has_prefilled_opener(proc, {"enable_thinking": True}) is True
        assert server._has_prefilled_opener(proc, {"enable_thinking": False}) is True
        # Two distinct keys cached
        assert len(server._PREFILL_FLAG_CACHE) >= 2

    def test_unhashable_kwargs_skip_cache_but_still_compute(self):
        proc = _FakeProcessor(
            _FakeTokenizer(suffix_with_gen="<|im_start|>assistant\n<think>\n")
        )
        # dict value is unhashable; helper should fall through gracefully
        result = server._has_prefilled_opener(proc, {"tools": [{"name": "x"}]})
        assert result is True
        # No cache entry written for unhashable keys
        for k in server._PREFILL_FLAG_CACHE:
            assert "tools" not in dict(k[1])


class TestIsPromptInsideThinking:
    """Regression for the Gemma 4 leak where the streaming state machine
    failed to seed in_thinking=True because `_has_prefilled_opener`
    only checked the prompt tail, missing Gemma's global `<|think|>`
    marker at the system block (memory.md #29).

    `_is_prompt_inside_thinking` does a structural scan of the whole
    rendered prompt and returns True iff there's an opener with no
    closer following it — handling both tail-prefilled (Qwen 3.6
    unsloth) and globally-opened (Gemma 4) cases.
    """

    def test_gemma_global_think_marker_at_start(self):
        # The exact pattern in user logs: <|think|> at the top of the
        # system block, no closer anywhere in the prompt → model is
        # in-thinking from gen-start.
        prompt = (
            "<|think|>\n"
            "system content\n"
            "<|tool>declaration:foo<tool|>\n"
            "<turn|>\n"
            "<|turn>user\nhi<turn|>\n"
            "<|turn>model\n"
        )
        assert server._is_prompt_inside_thinking(prompt) is True

    def test_qwen_tail_prefilled_opener(self):
        # The unsloth Qwen 3.6 case the original `_has_prefilled_opener`
        # was designed for. Same-direction signal here.
        prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        assert server._is_prompt_inside_thinking(prompt) is True

    def test_closed_thinking_block_returns_false(self):
        # Gemma 4 with enable_thinking=False renders an empty block:
        # `<|channel>thought\n<channel|>`. Opener is followed by closer
        # → not in thinking.
        prompt = "<|turn>user\nhi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        assert server._is_prompt_inside_thinking(prompt) is False

    def test_no_thinking_format_in_prompt(self):
        prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        assert server._is_prompt_inside_thinking(prompt) is False

    def test_opener_then_closer_then_opener_again(self):
        # Pathological case: a previously-closed thinking block, then a
        # fresh opener at the tail. Latest opener has no following
        # closer → in thinking.
        prompt = (
            "earlier <|channel>thought\nfoo<channel|> done\n<|turn>model\n<|think|>"
        )
        assert server._is_prompt_inside_thinking(prompt) is True


class TestPartialTagStartPos:
    """Tests for the ends-with-prefix detector that replaces the buggy
    `p in accumulated` substring match. The substring check could only
    fire after `accumulated` had grown to the partial's full length;
    by then the tag's leading bytes had already streamed through as
    `delta.content` piecewise, leaking literal `<|c`/`<|ch`/`<|chan`
    fragments. The ends-with-prefix check fires from the very first
    matching byte.
    """

    def test_returns_none_when_no_partial_at_end(self):
        partials = ("<|channel", "<|think")
        assert server._partial_tag_start_pos("plain text", partials) is None

    def test_matches_single_char_prefix(self):
        # The exact failure mode: a single `<` arrives as one token.
        # The substring check would not fire (`"<|channel"` is 9 chars,
        # accumulated is 1). Ends-with-prefix sees `<` matches the
        # 1-char prefix of every `<…` partial.
        partials = ("<|channel", "<|think")
        assert server._partial_tag_start_pos("hello <", partials) == 6

    def test_matches_growing_prefix_across_calls(self):
        # Drive the accumulated buffer character by character. Every
        # state along the way should still be detected as partial.
        partials = ("<|channel", "<|think")
        for accum in (
            "<",
            "<|",
            "<|c",
            "<|ch",
            "<|cha",
            "<|chan",
            "<|chann",
            "<|channe",
            "<|channel",
        ):
            pos = server._partial_tag_start_pos(accum, partials)
            assert pos == 0, f"failed at accum={accum!r} got pos={pos}"

    def test_returns_earliest_match_when_multiple_overlapping(self):
        # Keep the leftmost partial-start position when two literals
        # have overlapping prefixes (`</think` and `<channel` both
        # contain `<`).
        partials = ("</think", "<channel")
        assert server._partial_tag_start_pos("text <", partials) == 5

    def test_no_match_for_string_in_middle_of_accumulated(self):
        # A complete tag in the middle of accumulated isn't a partial
        # at the end. Caller's tag-find logic handles complete tags
        # via the find()-based branch; partial detection is only for
        # the trailing region.
        partials = ("<|channel",)
        assert server._partial_tag_start_pos("<|channel> stuff", partials) is None

    def test_empty_accumulated(self):
        partials = ("<|channel",)
        assert server._partial_tag_start_pos("", partials) is None

    def test_empty_partials_tuple(self):
        # No format → no partials to track → no match.
        assert server._partial_tag_start_pos("anything", ()) is None


class TestStepThinkingState:
    """Pure-function tests for the streaming-state-machine helper that
    replaced the inline branch chain in chat_completions_endpoint.

    Pins the three failure modes from production logs:
      1. Token-spanning tags eating content (`<channel|>2` dropping
         the leading `2 + ` of "2 + 2 = 4").
      2. Tag-prefix bytes leaking as content because the partial check
         used substring instead of ends-with-prefix.
      3. Multiple state transitions in a single token (closer +
         visible + opener + reasoning all fused) silently dropping the
         second transition.
    """

    @pytest.fixture
    def gemma_fmt(self):
        from mlx_vlm.prompt_utils import THINKING_FORMATS

        return next(f for f in THINKING_FORMATS if f.name == "gemma")

    def _drive(self, tokens, fmt, in_thinking_start=False):
        """Run a sequence of tokens through the helper, accumulating
        the emitted reasoning + content streams. Returns
        ``(end_in_thinking, end_accumulated, full_reasoning, full_content)``.
        """
        in_thinking = in_thinking_start
        accumulated = ""
        reasoning_parts = []
        content_parts = []
        for t in tokens:
            in_thinking, accumulated, dr, dc = server._step_thinking_state(
                t, in_thinking, accumulated, fmt
            )
            if dr is not None:
                reasoning_parts.append(dr)
            if dc is not None:
                content_parts.append(dc)
        return (
            in_thinking,
            accumulated,
            "".join(reasoning_parts),
            "".join(content_parts),
        )

    # --- Headline failure modes from the production logs -----------------

    def test_token_spanning_closer_emits_pre_and_post(self, gemma_fmt):
        # Gemma 4 production bug: "2 + 2 = 4" rendered as "2 = 4"
        # because the closer + leading visible char came in one token.
        # Pre-fix: branch 2 fired, transitioned, dropped the entire
        # token. Post-fix: split at closer, emit "thinking content"
        # as reasoning and "2" as content for the same iteration.
        in_thinking, accumulated, dr, dc = server._step_thinking_state(
            "thinking content<channel|>2",
            True,
            "",
            gemma_fmt,
        )
        assert in_thinking is False
        assert accumulated == ""
        assert dr == "thinking content"
        assert dc == "2"

    def test_partial_opener_buffered_then_completed(self, gemma_fmt):
        # Gemma 4 production bug: `<|channel>thought` literal showed
        # up in delta.content. Cause was the substring partial check
        # — `<|channel` (9 chars) couldn't match accumulated until
        # accumulated had 9+ chars, so individual tag-prefix tokens
        # streamed straight to delta.content.
        # Post-fix: ends-with-prefix matches from the very first byte.
        in_thinking, accumulated, reasoning, content = self._drive(
            [
                "hello ",
                "<",
                "|",
                "channel>thought",
                "\nreasoning",
                "<channel|>",
                "visible",
            ],
            gemma_fmt,
            in_thinking_start=False,
        )
        assert in_thinking is False
        assert accumulated == ""
        assert content == "hello visible"
        assert reasoning == "\nreasoning"
        # Crucial invariant: no fragment of the opener literal leaked
        # into content.
        assert "<|" not in content
        assert "channel" not in content

    def test_multi_transition_token(self, gemma_fmt):
        # closer + visible + opener + reasoning all fused into one
        # token. Pre-fix: only the first transition fired; the second
        # and third were dropped. Post-fix: helper loops over all
        # transitions in the accumulated buffer.
        in_thinking, accumulated, dr, dc = server._step_thinking_state(
            "r1<channel|>between<|channel>thoughtr2",
            True,
            "",
            gemma_fmt,
        )
        assert in_thinking is True
        assert accumulated == ""
        assert dr == "r1" + "r2"
        assert dc == "between"

    # --- Other invariants ------------------------------------------------

    def test_no_format_passthrough(self):
        # Non-thinking model: no format detected. Token text streams
        # through as content unchanged; in_thinking stays False;
        # accumulated unchanged.
        in_thinking, accumulated, dr, dc = server._step_thinking_state(
            "plain text", False, "", None
        )
        assert in_thinking is False
        assert accumulated == ""
        assert dr is None
        assert dc == "plain text"

    def test_seeded_in_thinking_routes_first_tokens_to_reasoning(self, gemma_fmt):
        # Gemma 4 + enable_thinking=True: streaming starts with
        # in_thinking=True (seeded by `_is_prompt_inside_thinking`).
        # First tokens are reasoning until a closer arrives.
        in_thinking, accumulated, reasoning, content = self._drive(
            ["thinking ", "content ", "more"],
            gemma_fmt,
            in_thinking_start=True,
        )
        assert in_thinking is True
        # No closer arrived → still buffering nothing, all emitted as
        # reasoning streamed.
        assert reasoning == "thinking content more"
        assert content == ""

    # NOTE (upstream port): test_round_trip_multiple_thinking_blocks is
    # intentionally NOT ported. The ported _step_thinking_state lstrips the
    # leading newline after a per-turn opener (intentional behavior change),
    # so the fork's exact reasoning concatenation assertion
    # ("thinking-1" + "\nthinking-2") no longer holds.

    def test_partial_at_end_is_carried_forward(self, gemma_fmt):
        # A partial tag at the end of one token's accumulated should
        # be preserved verbatim across the call boundary so the next
        # token can complete (or invalidate) it.
        in_thinking, accumulated, dr, dc = server._step_thinking_state(
            "before <", False, "", gemma_fmt
        )
        assert in_thinking is False
        assert accumulated == "<"  # buffered
        assert dr is None
        assert dc == "before "

    def test_partial_buffer_invalidated_by_non_tag_continuation(self, gemma_fmt):
        # Partial `<` followed by a non-matching char like `a` should
        # be released back into content (it wasn't a tag after all).
        # The helper handles this by re-checking on each call: at next
        # call, accumulated="<a"; no opener matches; no partial at end
        # (`<a` doesn't end with a prefix of any partial); flush all
        # as content.
        # First token: buffer the `<`.
        s = server._step_thinking_state("<", False, "", gemma_fmt)
        assert s == (False, "<", None, None)
        # Second token: completion that's NOT a tag. The buffered `<`
        # must be released, plus the new content.
        in_thinking, accumulated, dr, dc = (
            server._step_thinking_state("a", *s[:2][::-1][::-1][:2], gemma_fmt)
            if False
            else server._step_thinking_state("a", s[0], s[1], gemma_fmt)
        )
        assert in_thinking is False
        assert accumulated == ""
        assert dr is None
        assert dc == "<a"

    def test_empty_token_no_op(self, gemma_fmt):
        in_thinking, accumulated, dr, dc = server._step_thinking_state(
            "", False, "", gemma_fmt
        )
        assert in_thinking is False
        assert accumulated == ""
        assert dr is None
        assert dc is None

    def test_helper_appends_token_internally_no_double_count(self, gemma_fmt):
        # Regression for the "every word doubled" bug observed in the
        # production Gemma 4 stream after the helper rewrite. The caller
        # must NOT pre-append `token.text` to `accumulated` before calling
        # the helper — the helper does it internally. If both append,
        # every byte of the token streams twice (visible content +
        # reasoning), which the user saw as "The The user user is is...".
        #
        # Pin the contract: drive a sequence of plain-text tokens (no
        # thinking transitions) and assert the concatenated emitted
        # content exactly equals the input concatenation, byte-for-byte.
        in_thinking = False
        accumulated = ""
        emitted = []
        tokens = ["Hello", " ", "world", ", ", "this is ", "a ", "test."]
        for t in tokens:
            in_thinking, accumulated, dr, dc = server._step_thinking_state(
                t, in_thinking, accumulated, gemma_fmt
            )
            assert dr is None, f"unexpected reasoning emit on plain token: {dr!r}"
            if dc is not None:
                emitted.append(dc)
        assert "".join(emitted) == "".join(
            tokens
        ), f"emitted content {''.join(emitted)!r} != input {''.join(tokens)!r}"

    def test_helper_appends_token_internally_with_buffered_partial(self, gemma_fmt):
        # Same byte-for-byte invariant when partial buffering is in
        # play. Tokens carry a `<` that ends up not being a tag (the
        # next token resolves it as plain content). Output across both
        # tokens must exactly equal the input concatenation.
        in_thinking, accum1, dr1, dc1 = server._step_thinking_state(
            "before <", False, "", gemma_fmt
        )
        in_thinking, accum2, dr2, dc2 = server._step_thinking_state(
            " after", in_thinking, accum1, gemma_fmt
        )
        assert dr1 is None and dr2 is None
        emitted = (dc1 or "") + (dc2 or "")
        assert (
            emitted == "before < after"
        ), f"got {emitted!r}, expected 'before < after'"

    def test_seeded_in_thinking_elides_per_turn_opener(self, gemma_fmt):
        # Production bug (Gemma 4 26B 8-bit, OWUI first-turn):
        # `<|think|>` global system marker seeds in_thinking=True. The
        # model's first emission is the per-turn opener
        # `<|channel>thought\n` followed by reasoning. Pre-fix the
        # state machine only scanned closers while in_thinking, so the
        # opener literal leaked into delta.reasoning and the user
        # saw `<|channel>thought\nThe user is asking...` rendered
        # verbatim in the thinking block.
        # Post-fix: openers seen while already in_thinking are
        # structural markers — elide without state transition, so
        # only the actual reasoning prose streams to delta.reasoning.
        in_thinking, accumulated, reasoning, content = self._drive(
            [
                "<|channel>thought\n",
                "The user is asking who I am.",
                "<channel|>",
                "I am a large language model.",
            ],
            gemma_fmt,
            in_thinking_start=True,
        )
        assert in_thinking is False
        assert accumulated == ""
        assert reasoning == "\nThe user is asking who I am."
        assert content == "I am a large language model."
        # Crucial invariant: no fragment of the per-turn opener
        # literal leaked into the reasoning stream.
        assert "<|channel" not in reasoning
        assert "channel>thought" not in reasoning

    def test_seeded_in_thinking_elides_opener_split_across_tokens(self, gemma_fmt):
        # Same bug, byte-streamed variant: the opener arrives byte-by-
        # byte from the tokenizer. The partial-buffer path must fire
        # immediately (ends-with-prefix) so no prefix bytes leak as
        # reasoning content while accumulated is too short to contain
        # the full literal.
        in_thinking, accumulated, reasoning, content = self._drive(
            ["<", "|", "channel", ">thought", "\nreasoning"],
            gemma_fmt,
            in_thinking_start=True,
        )
        assert in_thinking is True
        assert accumulated == ""
        assert reasoning == "\nreasoning"
        assert content == ""
        assert "<" not in reasoning
        assert "channel" not in reasoning


class TestIsTemplateThinkingAsymmetric:
    """Tests for the asymmetric-rendering heuristic that gates the
    SWA-snapshot path (memory.md #30).

    The heuristic: a thinking format detected in the rendered prompt
    means a thinking-aware client (OWUI, OpenAI SDK with reasoning
    suppression, etc.) will likely strip reasoning before echoing the
    assistant turn back. Cache holds full thinking content; next
    request's render lacks it → asymmetric. Engaging the snapshot
    path is the safe default for any thinking model.
    """

    def test_gemma_native_thinking_is_asymmetric(self):
        # Gemma 4: `<|think|>` opener anywhere in the rendered prompt
        # signals the model is in a thinking-aware regime.
        prompt = "<|think|>\nsystem stuff\n<turn|>...<turn|>model\n"
        assert server._is_template_thinking_asymmetric(prompt) is True

    def test_qwen_thinking_is_asymmetric(self):
        prompt = "<|im_start|>assistant\n<think>\nreasoning\n</think>\n"
        assert server._is_template_thinking_asymmetric(prompt) is True

    def test_gpt_oss_channel_is_asymmetric(self):
        prompt = "user x <|channel>thought\nfoo<channel|> visible"
        assert server._is_template_thinking_asymmetric(prompt) is True

    def test_no_thinking_format_is_symmetric(self):
        # Plain non-thinking prompt — no detection, no snapshot needed.
        # Cache anchors at end-of-asst (current symmetric behavior).
        prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        assert server._is_template_thinking_asymmetric(prompt) is False

    def test_empty_prompt_is_symmetric(self):
        assert server._is_template_thinking_asymmetric("") is False


class TestDetectThinkingFormat:
    """Helper now returns a `ThinkingFormat` (or None) instead of a
    string identifier. Each format has its own opener/closer literals
    in the registry; consumers read tag tuples off the returned object.
    """

    def test_gemma_native_opener(self):
        # Gemma 4's actual thinking opener is the pipe-delimited tag.
        # Earlier versions of `_detect_thinking_format` conflated this
        # with gpt-oss's `<|channel>thought`; the registry separates them.
        fmt = server._detect_thinking_format("foo <|think|> bar")
        assert fmt is not None
        assert fmt.name == "gemma"

    def test_channel_thought_opener_matches_gemma(self):
        # `<|channel>thought` is now in Gemma 4's openers tuple too
        # (per-turn inline thinking, same syntax as gpt-oss). Gemma is
        # registry-listed first, so first-match wins. Behavior-wise
        # identical to gpt-oss for streaming purposes; only the brand
        # differs.
        fmt = server._detect_thinking_format("user msg ... <|channel>thought\n")
        assert fmt is not None
        assert fmt.name == "gemma"

    def test_qwen_think_tag(self):
        # Qwen / DeepSeek / generic `<think>...</think>` family.
        fmt = server._detect_thinking_format("hello <think>\n")
        assert fmt is not None
        assert fmt.name == "qwen"

    def test_no_thinking_tags(self):
        assert server._detect_thinking_format("just a plain prompt") is None

    def test_gemma_takes_precedence_over_qwen(self):
        # If a prompt contains both `<|think|>` and `<think>` literals
        # (rare, but possible in pathological echoed history), gemma
        # wins — it's listed first in THINKING_FORMATS, and first-match
        # ordering is the registry's specificity contract.
        prompt = "<|think|> reasoning ... <think>"
        fmt = server._detect_thinking_format(prompt)
        assert fmt is not None
        assert fmt.name == "gemma"


# NOTE (upstream port): TestComputeThinkingBudget (fork's 7 auto-budget tests)
# is intentionally NOT ported. The fork's _compute_thinking_budget /
# THINKING_BUDGET_RATIO auto-budget mechanism was replaced upstream by
# ThinkingBudgetCriteria; those symbols no longer exist on the server.


class TestMakeLogprobContent:
    class _FakeTokenizer:
        """Stub tokenizer that maps known token ids to text."""

        def __init__(self, mapping):
            self.mapping = mapping

        def decode(self, ids):
            return self.mapping.get(int(ids[0]), f"<unk:{ids[0]}>")

    def test_chosen_token_logprob_only(self):
        tk = self._FakeTokenizer({42: "hello"})
        out = server._make_logprob_content(tk, token_id=42, logprob=-0.5)
        assert out.token == "hello"
        assert out.logprob == pytest.approx(-0.5)
        assert out.top_logprobs == []
        # bytes() of "hello" UTF-8.
        assert out.bytes == list(b"hello")

    def test_top_k_zero_skips_top_logprobs_even_when_provided(self):
        # The contract: top_k=0 means "don't include alternatives" — the
        # caller didn't ask for them. Honoring this matters because
        # building TopLogprob entries requires an extra decode per id.
        tk = self._FakeTokenizer({1: "a", 2: "b"})
        out = server._make_logprob_content(
            tk, token_id=1, logprob=-1.0, top_logprobs=[(2, -2.0)], top_k=0
        )
        assert out.top_logprobs == []

    def test_top_k_truncates_top_logprobs(self):
        tk = self._FakeTokenizer({1: "a", 2: "b", 3: "c", 4: "d"})
        out = server._make_logprob_content(
            tk,
            token_id=1,
            logprob=-1.0,
            top_logprobs=[(2, -2.0), (3, -3.0), (4, -4.0)],
            top_k=2,
        )
        assert [t.token for t in out.top_logprobs] == ["b", "c"]
        assert [t.logprob for t in out.top_logprobs] == [-2.0, -3.0]

    def test_decode_failure_yields_empty_string_not_crash(self):
        # _decode_token swallows exceptions — an unknown id with a
        # tokenizer that raises shouldn't take down the streaming loop.
        class _FailTokenizer:
            def decode(self, ids):
                raise RuntimeError("boom")

        out = server._make_logprob_content(_FailTokenizer(), token_id=999, logprob=-1.0)
        assert out.token == ""
        # Empty string -> empty bytes list (not None).
        assert out.bytes == []

    def test_logprob_coerced_to_float(self):
        # The streaming path passes scalar mx.array values via .item();
        # if a numpy float64 ever leaks through, the Pydantic model
        # should still accept it.
        import numpy as np

        tk = self._FakeTokenizer({1: "x"})
        out = server._make_logprob_content(tk, token_id=1, logprob=np.float64(-0.25))
        assert isinstance(out.logprob, float)
        assert out.logprob == pytest.approx(-0.25)


class TestBuildGenArgsPenaltyAndSeedPlumbing:
    """memory.md #24 — verify the four advanced-params knobs flow from
    request body / Ollama aliases all the way through to GenerationArguments.
    Regressions here are silent: the slider in OpenWebUI moves but the
    server ignores it.
    """

    def _base_request(self, **overrides):
        # Minimal request stub matching the attributes _build_gen_args
        # reads. Every field defaulted to None so individual tests only
        # set what they care about.
        defaults = dict(
            max_tokens=None,
            max_output_tokens=None,
            temperature=None,
            top_p=None,
            top_k=None,
            min_p=None,
            seed=None,
            repetition_penalty=None,
            repeat_penalty=None,
            presence_penalty=None,
            frequency_penalty=None,
            logit_bias=None,
            enable_thinking=False,
            thinking_budget=None,
            thinking_start_token=None,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_seed_plumbed_through(self):
        req = self._base_request(seed=42)
        args = server._build_gen_args(req)
        assert args.seed == 42

    def test_seed_default_is_none_not_zero(self):
        # Critical: a non-None default of 0 would silently re-seed every
        # request to the same value, eliminating sampling variance. The
        # contract is "omitted seed = don't reseed".
        req = self._base_request()
        args = server._build_gen_args(req)
        assert args.seed is None

    def test_repeat_penalty_alias_recognized(self):
        # Ollama / OpenWebUI native UI slider name. Must alias to
        # repetition_penalty when the OpenAI-style name isn't present.
        req = self._base_request(repeat_penalty=1.15)
        args = server._build_gen_args(req)
        assert args.repetition_penalty == 1.15

    def test_repetition_penalty_wins_when_both_set(self):
        # If a client somehow sends both, the OpenAI-style name takes
        # precedence (the alias is a fallback, not an override).
        req = self._base_request(repetition_penalty=1.20, repeat_penalty=1.05)
        args = server._build_gen_args(req)
        assert args.repetition_penalty == 1.20

    def test_repeat_penalty_falsy_falls_through_to_repetition(self):
        # Defensive: 0 / None / False on the alias must not poison a
        # real repetition_penalty value. Implementation uses `or`, so
        # any falsy alias falls through.
        req = self._base_request(repetition_penalty=1.10, repeat_penalty=None)
        args = server._build_gen_args(req)
        assert args.repetition_penalty == 1.10

    def test_presence_penalty_plumbed_through(self):
        # Qwen 3.x family REQUIRES presence_penalty (rep_penalty is
        # forbidden by the model creator). Losing this drops the only
        # sane loop-mitigation knob for those models.
        req = self._base_request(presence_penalty=1.5)
        args = server._build_gen_args(req)
        assert args.presence_penalty == 1.5

    def test_frequency_penalty_plumbed_through(self):
        # Llama 3.x family uses frequency_penalty. Same silent-drop risk.
        req = self._base_request(frequency_penalty=0.5)
        args = server._build_gen_args(req)
        assert args.frequency_penalty == 0.5

    def test_all_four_penalties_independent(self):
        req = self._base_request(
            seed=7,
            repetition_penalty=1.10,
            presence_penalty=1.20,
            frequency_penalty=0.30,
        )
        args = server._build_gen_args(req)
        assert args.seed == 7
        assert args.repetition_penalty == 1.10
        assert args.presence_penalty == 1.20
        assert args.frequency_penalty == 0.30

    def test_unset_penalties_are_none_not_zero(self):
        # Distinguishing None from 0 matters: mlx_lm's
        # make_logits_processors checks `is not None` to decide whether
        # to install each processor. A 0.0 default would install a no-op
        # processor and burn cycles per token.
        args = server._build_gen_args(self._base_request())
        assert args.repetition_penalty is None
        assert args.presence_penalty is None
        assert args.frequency_penalty is None
        assert args.seed is None


class TestTokenIteratorHeartbeat:
    """The streaming iterator filters KeepAlive heartbeats so slow
    prefill (which produces no real tokens for many seconds) doesn't
    trip the queue-timeout. The contract:

      - KeepAlive items don't yield, but reset the timeout (each
        rqueue.get returns one queue interaction).
      - None terminates the stream cleanly.
      - Exception items raise to the caller.
      - StreamingToken with finish_reason ends the stream after yield.
      - Real silence longer than TOKEN_QUEUE_TIMEOUT_SECS raises
        queue.Empty (not caught here — surfaces to the caller).
    """

    @staticmethod
    def _make_response_generator():
        """Bypass ResponseGenerator.__init__ — we only exercise
        _token_iterator and _cancel, neither of which touches model state.
        """
        rg = server.ResponseGenerator.__new__(server.ResponseGenerator)
        rg._cancelled = set()
        rg._cancel_lock = __import__("threading").Lock()
        return rg

    def test_keepalive_filtered_not_yielded(self):
        from queue import Queue

        rg = self._make_response_generator()
        q: Queue = Queue()
        # Many heartbeats then one real token then sentinel.
        q.put(server.KeepAlive())
        q.put(server.KeepAlive())
        q.put(server.KeepAlive())
        token = server.StreamingToken(
            text="hi", token=42, logprobs=-0.1, finish_reason=None
        )
        q.put(token)
        q.put(None)

        items = list(rg._token_iterator(q, uid=1))
        # All heartbeats consumed silently — only the real token reaches
        # the caller.
        assert len(items) == 1
        assert items[0].token == 42

    def test_finish_reason_token_ends_stream(self):
        from queue import Queue

        rg = self._make_response_generator()
        q: Queue = Queue()
        final = server.StreamingToken(
            text="bye", token=2, logprobs=0.0, finish_reason="stop"
        )
        q.put(final)
        # Sentinel never gets a chance to be read — the iterator should
        # end on finish_reason without blocking on the queue.

        items = list(rg._token_iterator(q, uid=2))
        assert len(items) == 1
        assert items[0].finish_reason == "stop"

    def test_none_sentinel_terminates_cleanly(self):
        from queue import Queue

        rg = self._make_response_generator()
        q: Queue = Queue()
        q.put(None)

        items = list(rg._token_iterator(q, uid=3))
        assert items == []
        # Ended cleanly — no cancellation queued.
        assert 3 not in rg._cancelled

    def test_exception_item_raises_to_caller(self):
        from queue import Queue

        rg = self._make_response_generator()
        q: Queue = Queue()
        q.put(RuntimeError("backend exploded"))

        gen = rg._token_iterator(q, uid=4)
        with pytest.raises(RuntimeError, match="backend exploded"):
            list(gen)

    def test_keepalive_bursts_dont_yield_anything(self):
        # Pure heartbeat stream followed by termination — verify the
        # iterator collapses cleanly without spurious yields.
        from queue import Queue

        rg = self._make_response_generator()
        q: Queue = Queue()
        for _ in range(50):
            q.put(server.KeepAlive())
        q.put(None)

        items = list(rg._token_iterator(q, uid=5))
        assert items == []

    def test_unfinished_iterator_cancels_uid(self, monkeypatch):
        # If the consumer breaks out early (or the iterator exits
        # without the daemon's None sentinel), the finally block must
        # call _cancel(uid) so the daemon stops generating tokens for a
        # client that's no longer listening.
        from queue import Queue

        rg = self._make_response_generator()
        q: Queue = Queue()
        token = server.StreamingToken(
            text="x", token=1, logprobs=0.0, finish_reason=None
        )
        q.put(token)
        # No None sentinel, no finish_reason — consumer breaks early.

        cancelled = []
        monkeypatch.setattr(rg, "_cancel", lambda uid: cancelled.append(uid))

        gen = rg._token_iterator(q, uid=99)
        # Consume the first token, then close without exhausting.
        first = next(gen)
        assert first.token == 1
        gen.close()

        assert cancelled == [99]

    def test_finished_iterator_does_not_cancel(self, monkeypatch):
        from queue import Queue

        rg = self._make_response_generator()
        q: Queue = Queue()
        q.put(None)  # immediate clean termination

        cancelled = []
        monkeypatch.setattr(rg, "_cancel", lambda uid: cancelled.append(uid))

        list(rg._token_iterator(q, uid=100))
        assert cancelled == []

    def test_heartbeat_resets_timeout_window(self, monkeypatch):
        # Critical regression guard: a steady drip of heartbeats keeps
        # the iterator alive even past TOKEN_QUEUE_TIMEOUT_SECS of real
        # wall time. Implementation detail: queue.get's timeout is a
        # per-call deadline, NOT a cumulative one — every successful
        # get (heartbeat included) resets the window.
        #
        # We don't sleep TOKEN_QUEUE_TIMEOUT_SECS in tests; instead we
        # patch the constant to a tiny value and prove that an
        # interleaved heartbeat-then-token stream completes successfully
        # despite each gap being shorter than the timeout but their sum
        # exceeding it — by simply emitting them faster than one step.
        from queue import Queue

        monkeypatch.setattr(server_generation, "TOKEN_QUEUE_TIMEOUT_SECS", 0.5)

        rg = self._make_response_generator()
        q: Queue = Queue()
        # 20 heartbeats interleaved with 5 tokens — total stream is
        # well under 0.5s wall clock since everything is pre-queued.
        for _ in range(20):
            q.put(server.KeepAlive())
        for i in range(5):
            q.put(
                server.StreamingToken(
                    text=str(i), token=i, logprobs=0.0, finish_reason=None
                )
            )
        q.put(None)

        items = list(rg._token_iterator(q, uid=11))
        assert [t.token for t in items] == [0, 1, 2, 3, 4]


class TestStepEmitsHeartbeatDuringPrefill:
    """Direct test for the daemon-side hook: when batch_gen.next() yields
    no responses (we're inside a prefill chunk), _step pushes a
    KeepAlive to every active rqueue. Without this, prefill chunks
    would silently consume time and the iterator's queue-get timer
    would interpret the gap as a daemon hang.
    """

    @staticmethod
    def _make_response_generator():
        rg = server.ResponseGenerator.__new__(server.ResponseGenerator)
        rg._cancelled = set()
        return rg

    @staticmethod
    def _fake_batch_gen(responses):
        # Upstream _step shape: batch_gen.next() returns
        # (prompt_responses, responses) and iterates prompt_responses, so
        # the first slot must be an iterable (empty list here), NOT None.
        return SimpleNamespace(next=lambda **kw: ([], responses))

    def test_empty_responses_emits_keepalive_per_active_uid(self):
        from queue import Queue

        rg = self._make_response_generator()
        q1, q2 = Queue(), Queue()
        active = {
            10: {"rqueue": q1, "tokens": [], "prev_text": ""},
            20: {"rqueue": q2, "tokens": [], "prev_text": ""},
        }
        rg._step(self._fake_batch_gen([]), active)

        ka1 = q1.get_nowait()
        ka2 = q2.get_nowait()
        assert isinstance(ka1, server.KeepAlive)
        assert isinstance(ka2, server.KeepAlive)
        # No further items — heartbeat is one-per-step, not a flood.
        assert q1.empty()
        assert q2.empty()

    def test_responses_present_skips_heartbeat(self):
        # When prefill is done and the step produced real responses,
        # we don't ALSO push a heartbeat — the response itself counts
        # as activity, and stacking heartbeats behind tokens just
        # wastes queue churn.
        from queue import Queue

        rg = self._make_response_generator()
        q = Queue()
        # Upstream _step turns a response token into text via
        # info["streamer"].advance(token, finish_reason); the active-dict
        # entry must carry a streamer stub rather than relying on a
        # tokenizer.decode shim.
        active = {
            7: {
                "rqueue": q,
                "tokens": [],
                "prev_text": "",
                "streamer": SimpleNamespace(
                    advance=lambda tok, fr: "x", finalize=lambda: ""
                ),
            }
        }

        # Construct a minimal real-shaped response.
        resp = SimpleNamespace(uid=7, token=42, finish_reason=None, token_logprob=-0.5)
        rg._step(self._fake_batch_gen([resp]), active)

        # First item is the StreamingToken; no KeepAlive emitted.
        item = q.get_nowait()
        assert isinstance(item, server.StreamingToken)
        assert item.token == 42
        assert q.empty()

    def test_no_active_uids_emits_nothing(self):
        # Defensive: no active uids means no rqueues to ping. Must not
        # crash on the empty-dict iteration.
        rg = self._make_response_generator()
        rg._step(self._fake_batch_gen([]), active={})
        # Nothing to assert — just no exception.


class TestCachedPathHeartbeatWatchdog:
    """The cached path's stream_generate yields nothing during its
    internal prefill loop. _process_cached_request runs a per-request
    timer thread that pumps KeepAlive sentinels into rqueue while the
    `for chunk in stream_generate(...)` loop is active, so the
    iterator's queue timer doesn't interpret legitimate prefill silence
    as a hang.

    These tests stub stream_generate so the watchdog logic can be
    exercised without spinning up a real model.
    """

    @staticmethod
    def _make_response_generator(stream_generate_stub):
        """Build a ResponseGenerator with the bare attributes
        _process_cached_request reads. The stream_generate symbol is
        imported locally inside the method, so we patch via a fake
        ``mlx_vlm.generate.stream_generate``.
        """
        rg = server.ResponseGenerator.__new__(server.ResponseGenerator)
        rg.model = SimpleNamespace()
        rg.processor = SimpleNamespace()
        rg.vision_cache = None
        rg.kv_bits = None
        rg.kv_group_size = None
        rg.kv_quant_scheme = None
        rg.quantized_kv_start = None
        rg._cancelled = set()
        rg._cancel_lock = __import__("threading").Lock()
        return rg

    @staticmethod
    def _stub_args():
        # GenerationArguments needs only to_generate_kwargs() for our
        # purposes. Use the real class with defaults so the kwargs dict
        # is realistic.
        return server.GenerationArguments()

    def test_slow_prefill_produces_heartbeats(self, monkeypatch):
        # Simulate a slow prefill: stream_generate sleeps before yielding
        # its first chunk. With the watchdog, the rqueue receives at
        # least one KeepAlive before the real chunk arrives, plus the
        # final StreamingToken and None sentinel.
        import time
        from queue import Queue

        # Tighten the heartbeat interval so the test runs in ~0.1s.
        monkeypatch.setattr(
            server_generation, "CACHED_PATH_HEARTBEAT_INTERVAL_SECS", 0.02
        )

        # Fake stream_generate: sleep 0.10s (≈ 5 heartbeat intervals),
        # then yield one terminal chunk. Mimics prefill silence followed
        # by a single end-of-generation token.
        def fake_stream_generate(**kwargs):
            time.sleep(0.10)
            yield SimpleNamespace(
                token=42,
                text="hi",
                logprobs=None,
                finish_reason="stop",
                peak_memory=0.0,
            )

        # mlx_vlm.generate the function shadows the submodule on the
        # package, so the dotted-path setattr resolves to the function.
        # Patch via sys.modules to reach the actual submodule that the
        # cached-path's local `from .generate import stream_generate`
        # resolves against.
        import sys

        monkeypatch.setattr(
            sys.modules["mlx_vlm.generate"], "stream_generate", fake_stream_generate
        )

        rg = self._make_response_generator(fake_stream_generate)
        rqueue: Queue = Queue()
        prompt_cache_state = SimpleNamespace()  # opaque; only forwarded

        rg._process_cached_request(
            rqueue=rqueue,
            prompt="hello",
            images=None,
            args=self._stub_args(),
            prompt_tokens=5,
            prompt_cache_state=prompt_cache_state,
        )

        # Drain the queue. Expected order:
        #   1. GenerationContext (always pushed first)
        #   2. >=1 KeepAlive (from the watchdog during prefill silence)
        #   3. StreamingToken (the real chunk)
        #   4. None (terminator)
        items = []
        while not rqueue.empty():
            items.append(rqueue.get_nowait())

        assert isinstance(items[0], server.GenerationContext)
        assert items[-1] is None
        keepalive_count = sum(1 for it in items if isinstance(it, server.KeepAlive))
        token_count = sum(1 for it in items if isinstance(it, server.StreamingToken))
        assert keepalive_count >= 1, (
            f"watchdog should have emitted at least one KeepAlive "
            f"during the 0.10s silence; got items={[type(i).__name__ for i in items]}"
        )
        assert token_count == 1

    def test_watchdog_stops_before_terminator(self, monkeypatch):
        # The finally block sets heartbeat_done BEFORE pushing the None
        # sentinel, so by the time None is on the queue the watchdog
        # is no longer pumping. After the iterator hits None it stops
        # reading; any straggler KeepAlive that landed earlier is
        # filtered by isinstance check.
        import time
        from queue import Queue

        monkeypatch.setattr(
            server_generation, "CACHED_PATH_HEARTBEAT_INTERVAL_SECS", 0.01
        )

        def fake_stream_generate(**kwargs):
            yield SimpleNamespace(
                token=1,
                text="x",
                logprobs=None,
                finish_reason="stop",
                peak_memory=0.0,
            )

        # mlx_vlm.generate the function shadows the submodule on the
        # package, so the dotted-path setattr resolves to the function.
        # Patch via sys.modules to reach the actual submodule that the
        # cached-path's local `from .generate import stream_generate`
        # resolves against.
        import sys

        monkeypatch.setattr(
            sys.modules["mlx_vlm.generate"], "stream_generate", fake_stream_generate
        )

        rg = self._make_response_generator(fake_stream_generate)
        rqueue: Queue = Queue()

        rg._process_cached_request(
            rqueue=rqueue,
            prompt="x",
            images=None,
            args=self._stub_args(),
            prompt_tokens=1,
            prompt_cache_state=SimpleNamespace(),
        )

        # Give the daemon thread a moment to fully exit; the watchdog's
        # join(timeout=1.0) should have stopped it deterministically.
        time.sleep(0.05)
        items = []
        while not rqueue.empty():
            items.append(rqueue.get_nowait())

        # Sentinel must be the LAST item. No further heartbeats arrive
        # after the None — that would be a leak past the finally.
        assert items[-1] is None
        none_index = items.index(None)
        assert (
            none_index == len(items) - 1
        ), f"None terminator must be last; got {[type(i).__name__ for i in items]}"

    def test_exception_in_stream_generate_still_stops_watchdog(self, monkeypatch):
        # If stream_generate raises, the except block puts the exception
        # on the queue, the finally block must still stop the watchdog
        # AND push the None sentinel.
        import time
        from queue import Queue

        monkeypatch.setattr(
            server_generation, "CACHED_PATH_HEARTBEAT_INTERVAL_SECS", 0.01
        )

        def fake_stream_generate(**kwargs):
            time.sleep(0.05)
            raise RuntimeError("backend exploded")
            yield  # unreachable; needed to make this a generator

        # mlx_vlm.generate the function shadows the submodule on the
        # package, so the dotted-path setattr resolves to the function.
        # Patch via sys.modules to reach the actual submodule that the
        # cached-path's local `from .generate import stream_generate`
        # resolves against.
        import sys

        monkeypatch.setattr(
            sys.modules["mlx_vlm.generate"], "stream_generate", fake_stream_generate
        )

        rg = self._make_response_generator(fake_stream_generate)
        rqueue: Queue = Queue()

        rg._process_cached_request(
            rqueue=rqueue,
            prompt="x",
            images=None,
            args=self._stub_args(),
            prompt_tokens=1,
            prompt_cache_state=SimpleNamespace(),
        )

        items = []
        while not rqueue.empty():
            items.append(rqueue.get_nowait())

        # Order: GenerationContext, [KeepAlives], Exception, None.
        assert isinstance(items[0], server.GenerationContext)
        assert items[-1] is None
        # The error gets logged AND surfaced as an Exception item.
        assert any(isinstance(it, RuntimeError) for it in items)


class TestCachedPathDraftCounters:
    """O40 follow-up (mlx_local_stack C24): the cached/inline path must attach
    the same per-request draft counters the batched path puts on its final
    StreamingToken (draft_kind/draft_rounds/draft_n/draft_n_accepted), computed
    by snapshot-and-diff of the drafter's lifetime counters. Without this the
    response timings carry nulls while the inline mtp loop is engaged (measured
    live 2026-08-24: 1.93x decode with the drafter loaded, all counters null),
    so an engagement tripwire reading the response cannot certify engagement.
    """

    @staticmethod
    def _rg(drafter=None):
        rg = server.ResponseGenerator.__new__(server.ResponseGenerator)
        rg.model = SimpleNamespace()
        rg.processor = SimpleNamespace()
        rg.vision_cache = None
        rg.kv_bits = None
        rg.kv_group_size = None
        rg.kv_quant_scheme = None
        rg.quantized_kv_start = None
        rg._cancelled = set()
        rg._cancel_lock = __import__("threading").Lock()
        if drafter is not None:
            rg.draft_kind = "mtp"
            rg.draft_model = drafter
        return rg

    @staticmethod
    def _final_token(rqueue):
        items = []
        while not rqueue.empty():
            items.append(rqueue.get_nowait())
        tokens = [it for it in items if isinstance(it, server.StreamingToken)]
        assert tokens, f"no StreamingToken in {[type(i).__name__ for i in items]}"
        return tokens[-1]

    def _run(self, rg, fake_stream_generate, monkeypatch):
        import sys
        from queue import Queue

        monkeypatch.setattr(
            sys.modules["mlx_vlm.generate"], "stream_generate", fake_stream_generate
        )
        rqueue: Queue = Queue()
        rg._process_cached_request(
            rqueue=rqueue,
            prompt="x",
            images=None,
            args=server.GenerationArguments(),
            prompt_tokens=1,
            prompt_cache_state=SimpleNamespace(),
        )
        return self._final_token(rqueue)

    def test_engaged_request_reports_counters(self, monkeypatch):
        drafter = SimpleNamespace(
            speculative_total_rounds=10,
            speculative_total_accepted=20.0,
            speculative_total_drafted=30,
        )

        def fake_stream_generate(**kwargs):
            # The inline round loop bumps the drafter's lifetime counters
            # (_record_speculative_round); simulate 3 rounds / 7 drafted / 5
            # accepted happening during this request.
            drafter.speculative_total_rounds += 3
            drafter.speculative_total_accepted += 5.0
            drafter.speculative_total_drafted += 7
            yield SimpleNamespace(
                token=1, text="x", logprobs=None, finish_reason="stop", peak_memory=0.0
            )

        tok = self._run(self._rg(drafter), fake_stream_generate, monkeypatch)
        assert tok.draft_kind == "mtp"
        assert tok.draft_rounds == 3
        assert tok.draft_n == 7
        assert tok.draft_n_accepted == 5

    def test_wired_but_zero_rounds_reports_none(self, monkeypatch):
        # Same null-not-zero semantics as the batched path: a request that
        # wired a drafter but ran no rounds (e.g. the ar.py processor fallback
        # dropped to plain decode) reports None everywhere, so a tripwire
        # reads it as NOT engaged.
        drafter = SimpleNamespace(
            speculative_total_rounds=10,
            speculative_total_accepted=20.0,
            speculative_total_drafted=30,
        )

        def fake_stream_generate(**kwargs):
            yield SimpleNamespace(
                token=1, text="x", logprobs=None, finish_reason="stop", peak_memory=0.0
            )

        tok = self._run(self._rg(drafter), fake_stream_generate, monkeypatch)
        assert tok.draft_kind is None
        assert tok.draft_rounds is None
        assert tok.draft_n is None
        assert tok.draft_n_accepted is None

    def test_no_drafter_reports_none(self, monkeypatch):
        def fake_stream_generate(**kwargs):
            yield SimpleNamespace(
                token=1, text="x", logprobs=None, finish_reason="stop", peak_memory=0.0
            )

        tok = self._run(self._rg(), fake_stream_generate, monkeypatch)
        assert tok.draft_kind is None
        assert tok.draft_rounds is None
        assert tok.draft_n is None
        assert tok.draft_n_accepted is None


class TestStreamingChatThinkingStateSelection:
    """The union in `chat_completions`' ResponseGenerator streaming branch.

    Upstream builds one `thinking_state` and drives it with `.feed()`. The fork
    keeps that for models whose tokenizer ships a `parse_response` response
    template -- upstream's new capability, authoritative when present -- and
    otherwise routes tokens through `_step_thinking_state`, whose scan handles
    several thinking transitions inside a single streamed token. `.feed()` does
    not: it takes the first transition and leaks the rest into content.

    Both halves of that branch need a test, because a suite that only covers one
    makes the discriminator look load-bearing when it is not. Verified by pinning
    `use_response_template` to each constant in turn: True breaks this test, False
    breaks `test_chat_completions_streaming_response_template_tool_calls`.
    """

    MULTI_TRANSITION = "<think>a</think>b<think>c</think>d"

    @staticmethod
    def _fake_response_generator(text):
        class FakeResponseGenerator:
            tokenizer = SimpleNamespace(decode=lambda tokens: "")

            def validate_context_budget(
                self, prompt, images=None, audio=None, args=None
            ):
                return None

            # **kwargs, not the exact upstream signature: whether the endpoint
            # passes prompt_cache_state depends on suite-wide state an earlier
            # test may have left set, and the fake has no business asserting it.
            def generate(self, prompt, images=None, audio=None, args=None, **kwargs):
                return server.GenerationContext(uid=1, prompt_tokens=10), iter(
                    [
                        server.StreamingToken(
                            text=text,
                            token=1,
                            logprobs=0.0,
                            finish_reason="stop",
                        )
                    ]
                )

        return FakeResponseGenerator()

    def _stream(self, client, monkeypatch, processor, config):
        monkeypatch.setattr(
            server.runtime,
            "response_generator",
            self._fake_response_generator(self.MULTI_TRANSITION),
        )
        with (
            patch.object(
                server,
                "get_cached_model",
                return_value=(SimpleNamespace(), processor, config),
            ),
            patch.object(server, "apply_chat_template", return_value="prompt"),
        ):
            response = client.post(
                "/chat/completions",
                json={
                    "model": "demo",
                    "messages": [{"role": "user", "content": "Hi?"}],
                    "stream": True,
                },
            )
        assert response.status_code == 200
        chunks = [
            json.loads(line[len("data: ") :])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        reasoning = "".join(
            chunk["choices"][0]["delta"].get("reasoning_content") or ""
            for chunk in chunks
            if chunk["choices"]
        )
        content = "".join(
            chunk["choices"][0]["delta"].get("content") or ""
            for chunk in chunks
            if chunk["choices"]
        )
        return reasoning, content

    def test_plain_processor_uses_the_forks_multi_transition_scan(
        self, client, monkeypatch
    ):
        reasoning, content = self._stream(
            client,
            monkeypatch,
            SimpleNamespace(),
            SimpleNamespace(model_type="qwen2_vl"),
        )

        # Both thinking blocks are reasoning and both gaps are content.
        # Upstream's `.feed()` yields ("a", "b<think>c</think>d") here.
        assert (reasoning, content) == ("ac", "bd")

    def test_response_template_processor_defers_to_the_template_parser(
        self, client, monkeypatch
    ):
        # A Muse-template tokenizer routes through ResponseTemplateStreamState,
        # so the fork's registry scan must NOT claim these tokens. The template
        # has no <think> region, so nothing is reasoning.
        reasoning, _content = self._stream(
            client,
            monkeypatch,
            SimpleNamespace(tokenizer=_MuseResponseTemplateTokenizer()),
            SimpleNamespace(model_type="muse_glimmer"),
        )

        assert reasoning == ""


# ---------------------------------------------------------------------------
# Fork: legacy OpenAI text-completions endpoint (/v1/completions). Upstream has
# no CompletionRequest/CompletionResponse at all, so every test below this banner
# is fork-only.
# ---------------------------------------------------------------------------


def _completion_fake_generator(tokens, prompt_tokens=8, captured=None):
    """Build a FakeResponseGenerator emitting the given StreamingToken list."""

    class FakeResponseGenerator:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            if captured is not None:
                captured["prompt"] = prompt
                captured["images"] = images
                captured["audio"] = audio
                captured["args"] = args
            return server.GenerationContext(uid=1, prompt_tokens=prompt_tokens), iter(
                list(tokens)
            )

    return FakeResponseGenerator()


def test_completions_basic_non_streaming(client, monkeypatch):
    # Fork: fork-only — the legacy /v1/completions endpoint is fork work (b75c18b7).
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    captured = {}
    tokens = [
        server.StreamingToken(
            text="Hello", token=1, logprobs=0.0, finish_reason=None, prompt_tps=20.0
        ),
        server.StreamingToken(
            text=" world", token=2, logprobs=0.0, finish_reason="stop", prompt_tps=20.0
        ),
    ]
    monkeypatch.setattr(
        server.runtime,
        "response_generator",
        _completion_fake_generator(tokens, prompt_tokens=6, captured=captured),
    )

    with patch.object(
        server, "get_cached_model", return_value=(model, processor, config)
    ):
        response = client.post(
            "/v1/completions",
            json={"model": "demo", "prompt": "Continue: "},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["id"].startswith("cmpl-")
    assert body["model"] == "demo"
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["text"] == "Hello world"
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["logprobs"] is None
    # usage accounting
    assert body["usage"]["prompt_tokens"] == 6
    assert body["usage"]["completion_tokens"] == 2
    assert body["usage"]["total_tokens"] == 8


def test_completions_prompt_is_not_chat_templated(client, monkeypatch):
    # Fork: fork-only — the legacy /v1/completions endpoint is fork work (b75c18b7).
    """The raw prompt must reach the model verbatim — no apply_chat_template."""
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    captured = {}
    tokens = [
        server.StreamingToken(
            text="ok", token=1, logprobs=0.0, finish_reason="stop", prompt_tps=20.0
        )
    ]
    monkeypatch.setattr(
        server.runtime,
        "response_generator",
        _completion_fake_generator(tokens, captured=captured),
    )

    raw = "<|im_start|>user\nNOT A TEMPLATE\n[INST] verbatim [/INST]"
    template_mock = MagicMock(return_value="TEMPLATED-PROMPT")
    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", template_mock),
    ):
        response = client.post(
            "/v1/completions",
            json={"model": "demo", "prompt": raw, "enable_thinking": True},
        )

    assert response.status_code == 200
    # apply_chat_template is never called on the completions path.
    template_mock.assert_not_called()
    # The model receives the prompt byte-for-byte.
    assert captured["prompt"] == raw
    # Thinking is forced off regardless of the request asking for it.
    assert captured["args"].enable_thinking is False
    assert captured["args"].thinking_budget is None


def test_completions_stop_sequence_truncates_non_streaming(client, monkeypatch):
    # Fork: fork-only — the legacy /v1/completions endpoint is fork work (b75c18b7).
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    tokens = [
        server.StreamingToken(
            text="keep this", token=1, logprobs=0.0, finish_reason=None
        ),
        server.StreamingToken(
            text="<STOP>drop this", token=2, logprobs=0.0, finish_reason="stop"
        ),
    ]
    monkeypatch.setattr(
        server.runtime, "response_generator", _completion_fake_generator(tokens)
    )

    with patch.object(
        server, "get_cached_model", return_value=(model, processor, config)
    ):
        response = client.post(
            "/v1/completions",
            json={"model": "demo", "prompt": "p", "stop": "<STOP>"},
        )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == "keep this"
    assert choice["finish_reason"] == "stop"


def test_completions_echo_prepends_prompt_non_streaming(client, monkeypatch):
    # Fork: fork-only — the legacy /v1/completions endpoint is fork work (b75c18b7).
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    tokens = [
        server.StreamingToken(
            text=" answer", token=1, logprobs=0.0, finish_reason="stop"
        )
    ]
    monkeypatch.setattr(
        server.runtime, "response_generator", _completion_fake_generator(tokens)
    )

    with patch.object(
        server, "get_cached_model", return_value=(model, processor, config)
    ):
        response = client.post(
            "/v1/completions",
            json={"model": "demo", "prompt": "Question:", "echo": True},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "Question: answer"


def test_completions_streaming_emits_deltas_and_done(client, monkeypatch):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    tokens = [
        server.StreamingToken(
            text="foo", token=1, logprobs=0.0, finish_reason=None, prompt_tps=20.0
        ),
        server.StreamingToken(
            text="bar", token=2, logprobs=0.0, finish_reason="stop", prompt_tps=20.0
        ),
    ]
    monkeypatch.setattr(
        server.runtime,
        "response_generator",
        _completion_fake_generator(tokens, prompt_tokens=5),
    )

    with patch.object(
        server, "get_cached_model", return_value=(model, processor, config)
    ):
        response = client.post(
            "/v1/completions",
            json={"model": "demo", "prompt": "p", "stream": True},
        )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    chunks = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert all(chunk["object"] == "text_completion" for chunk in chunks)
    # Reconstruct streamed text from text-bearing choices.
    text = "".join(
        chunk["choices"][0]["text"]
        for chunk in chunks
        if chunk.get("choices") and chunk["choices"][0].get("text")
    )
    assert text == "foobar"
    # A terminal choice carries finish_reason="stop".
    finish_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("choices") and chunk["choices"][0].get("finish_reason") == "stop"
    ]
    assert finish_chunks
    # A trailing usage chunk reports accounting.
    usage_chunk = next(chunk for chunk in chunks if chunk.get("usage") is not None)
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["prompt_tokens"] == 5
    assert usage_chunk["usage"]["completion_tokens"] == 2


def test_completions_streaming_echo_first_chunk(client, monkeypatch):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    tokens = [
        server.StreamingToken(text="gen", token=1, logprobs=0.0, finish_reason="stop")
    ]
    monkeypatch.setattr(
        server.runtime, "response_generator", _completion_fake_generator(tokens)
    )

    with patch.object(
        server, "get_cached_model", return_value=(model, processor, config)
    ):
        response = client.post(
            "/v1/completions",
            json={"model": "demo", "prompt": "PROMPT", "echo": True, "stream": True},
        )

    assert response.status_code == 200
    chunks = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    text = "".join(
        chunk["choices"][0]["text"]
        for chunk in chunks
        if chunk.get("choices") and chunk["choices"][0].get("text")
    )
    assert text == "PROMPTgen"


def test_completions_streaming_stop_sequence_truncates(client, monkeypatch):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    tokens = [
        server.StreamingToken(text="abc", token=1, logprobs=0.0, finish_reason=None),
        server.StreamingToken(
            text="DEFstop", token=2, logprobs=0.0, finish_reason=None
        ),
        server.StreamingToken(text="zzz", token=3, logprobs=0.0, finish_reason="stop"),
    ]
    monkeypatch.setattr(
        server.runtime, "response_generator", _completion_fake_generator(tokens)
    )

    with patch.object(
        server, "get_cached_model", return_value=(model, processor, config)
    ):
        response = client.post(
            "/v1/completions",
            json={"model": "demo", "prompt": "p", "stop": ["stop"], "stream": True},
        )

    assert response.status_code == 200
    chunks = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    text = "".join(
        chunk["choices"][0]["text"]
        for chunk in chunks
        if chunk.get("choices") and chunk["choices"][0].get("text")
    )
    # "abc" + "DEF" (text before the "stop" sequence); "zzz" never streamed.
    assert text == "abcDEF"
    finish_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("choices") and chunk["choices"][0].get("finish_reason") == "stop"
    ]
    assert finish_chunks


def test_completions_rejects_n_greater_than_one(client, monkeypatch):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    monkeypatch.setattr(
        server.runtime, "response_generator", _completion_fake_generator([])
    )

    with patch.object(
        server, "get_cached_model", return_value=(model, processor, config)
    ):
        response = client.post(
            "/v1/completions",
            json={"model": "demo", "prompt": "p", "n": 2},
        )

    assert response.status_code == 400
    assert "n=2" in response.json()["detail"]


def test_completions_requires_model(client):
    response = client.post("/v1/completions", json={"prompt": "hi"})
    assert response.status_code == 400


def test_completions_generate_fallback_path(client, monkeypatch):
    """With no ResponseGenerator the endpoint uses generate() with the raw prompt."""
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    captured = {}

    def fake_generate(prompt, image=None, audio=None, **kwargs):
        captured["prompt"] = prompt
        return GenerationResult(
            text="raw continuation",
            prompt_tokens=4,
            generation_tokens=3,
            total_tokens=7,
            prompt_tps=10.0,
            generation_tps=5.0,
            peak_memory=0.1,
            finish_reason="length",
        )

    template_mock = MagicMock(return_value="TEMPLATED")
    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", template_mock),
        patch.object(server, "generate", side_effect=fake_generate),
    ):
        response = client.post(
            "/completions",
            json={"model": "demo", "prompt": "verbatim prompt"},
        )

    assert response.status_code == 200
    template_mock.assert_not_called()
    assert captured["prompt"] == "verbatim prompt"
    body = response.json()
    assert body["choices"][0]["text"] == "raw continuation"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["prompt_tokens"] == 4
    assert body["usage"]["completion_tokens"] == 3


def _registered_paths():
    """Every route path the app serves, including routers it includes.

    FastAPI >= 0.141 no longer flattens `include_router()` into `app.routes` — it
    inserts one lazy `_IncludedRouter` node whose `path` is `None` and resolves
    matches at request time. So scanning `app.routes` alone silently misses every
    inference route once they live on `inference_router` (#1714). Union in the
    router we own rather than reaching into the private node.
    """
    paths = {r.path for r in server.app.routes if getattr(r, "path", None)}
    paths |= {
        r.path
        for r in server._app_module.inference_router.routes
        if getattr(r, "path", None)
    }
    return paths


def test_completions_both_routes_registered():
    paths = _registered_paths()
    assert "/completions" in paths
    assert "/v1/completions" in paths


def test_inference_routes_are_served_not_just_registered(client):
    """Companion to the above: registration on a router is not reachability.

    Guards the failure mode where `include_router()` is forgotten — the paths
    would still be on `inference_router` and the test above would still pass,
    while every request 404'd.
    """
    for path in ("/completions", "/v1/completions", "/v1/chat/completions"):
        assert client.post(path, json={}).status_code != 404
    assert client.get("/v1/models").status_code != 404


@dataclass
class _FakeAlignedToken:
    id: int
    text: str
    start: float
    duration: float
    end: float = 0.0

    def __post_init__(self):
        self.end = self.start + self.duration


@dataclass
class _FakeAlignedSentence:
    text: str
    tokens: list
    start: float = 0.0
    end: float = 0.0

    def __post_init__(self):
        self.start = self.tokens[0].start
        self.end = self.tokens[-1].end


@dataclass
class _FakeAlignedResult:
    text: str
    sentences: list


@dataclass
class _FakeSTTOutput:
    text: str
    segments: list = None
    language: str = None


@dataclass
class _FakeStreamingResult:
    text: str
    tokens: list
    is_final: bool
    start_time: float
    end_time: float


def _fake_parakeet_result():
    first = _FakeAlignedSentence(
        "Hello world.",
        [
            _FakeAlignedToken(1, "Hello", 0.0, 0.4),
            _FakeAlignedToken(2, " world.", 0.4, 0.5),
        ],
    )
    second = _FakeAlignedSentence("Bye.", [_FakeAlignedToken(3, "Bye.", 1.0, 0.3)])
    return _FakeAlignedResult("Hello world. Bye.", [first, second])


class TestSTTSegmentSerialization:
    """Serialization of STT results into OpenAI-style transcription payloads.

    Regression coverage for NeMo-alignment models (Parakeet/Canary) whose
    ``AlignedResult`` exposes ``sentences`` rather than ``segments`` (issue 2183).
    """

    def test_derives_segments_from_nemo_sentences(self):
        from mlx_vlm.server.audio import _stt_item_to_dict

        data = _stt_item_to_dict(_fake_parakeet_result())

        assert "segments" in data
        segments = data["segments"]
        assert [s["text"] for s in segments] == ["Hello world.", "Bye."]
        assert segments[0]["start"] == 0.0
        assert abs(segments[0]["end"] - 0.9) < 1e-6
        assert segments[1]["start"] == 1.0
        assert abs(segments[1]["end"] - 1.3) < 1e-6

    def test_pipeline_preserves_nemo_segments(self):
        from mlx_vlm.server.audio import (
            _iter_stt_items,
            _sanitize_for_json,
            _stt_item_to_dict,
            _transcription_result_from_chunks,
        )

        chunks = [
            json.dumps(_sanitize_for_json(_stt_item_to_dict(item))) + "\n"
            for item in _iter_stt_items(_fake_parakeet_result())
        ]
        result = _transcription_result_from_chunks(chunks)

        assert result["text"].startswith("Hello world.")
        assert len(result.get("segments") or []) == 2

    def test_whisper_segments_unchanged(self):
        from mlx_vlm.server.audio import _stt_item_to_dict

        whisper = _FakeSTTOutput(
            "hi", segments=[{"start": 0.0, "end": 1.0, "text": "hi"}], language="en"
        )
        data = _stt_item_to_dict(whisper)

        assert data["segments"] == [{"start": 0.0, "end": 1.0, "text": "hi"}]
        assert data["language"] == "en"
        assert "sentences" not in data

    def test_plain_text_item_unchanged(self):
        from mlx_vlm.server.audio import _stt_item_to_dict

        assert _stt_item_to_dict("just text") == {"text": "just text"}

    def test_streaming_result_gets_no_segments(self):
        from mlx_vlm.server.audio import _stt_item_to_dict

        data = _stt_item_to_dict(
            _FakeStreamingResult("partial", [1, 2], False, 0.0, 0.5)
        )

        assert "segments" not in data
        assert data["is_final"] is False

    def test_real_aligned_result_if_available(self):
        pytest.importorskip("mlx_audio.stt.models.nemo.alignment")
        from mlx_audio.stt.models.nemo.alignment import (
            AlignedSentence,
            AlignedToken,
            sentences_to_result,
        )

        from mlx_vlm.server.audio import _stt_item_to_dict

        result = sentences_to_result(
            [
                AlignedSentence(
                    "hello",
                    [
                        AlignedToken(id=1, text="hel", start=0.0, duration=0.2),
                        AlignedToken(id=2, text="lo", start=0.2, duration=0.3),
                    ],
                )
            ]
        )
        data = _stt_item_to_dict(result)

        assert data["segments"] == [
            {"id": 0, "start": 0.0, "end": 0.5, "text": "hello"}
        ]


# Ported from upstream c36708d3 (2026-09-27 sync): the newlines that follow the
# thinking close marker are dropped even when they arrive in LATER stream chunks,
# so streamed content matches the non-streamed response.
def _feed_thinking(state, chunks, last=False):
    return [
        state.feed(text, last=last and i == len(chunks) - 1)
        for i, text in enumerate(chunks)
    ]


def _joined(deltas, key):
    return "".join(item.get(key) or "" for item in deltas)


def _thoughts(deltas):
    return tuple(
        _joined([vars(delta) for delta in deltas], key)
        for key in ("reasoning", "content")
    )


@pytest.mark.parametrize(
    "chunks,enabled,expected",
    [
        (["<think>plan</think>\n\nAnswer."], False, ("plan", "Answer.")),
        (
            ["<think>", "plan", "</think>", "\n\n", "Answer."],
            False,
            ("plan", "Answer."),
        ),
        (["<think>plan</think>", "\n", "\n", "Answer."], False, ("plan", "Answer.")),
        (["<think>plan</thi", "nk>\n", "\nAnswer."], False, ("plan", "Answer.")),
        (["plan", "</think>", "\n\n", "Answer."], True, ("plan", "Answer.")),
        (["<think>plan</think>", "\n\n"], False, ("plan", "")),
        (
            ["<think>plan</think>", "Answer.", "\n\nMore."],
            False,
            ("plan", "Answer.\n\nMore."),
        ),
    ],
    ids=[
        "same-chunk",
        "separate-chunk",
        "one-per-chunk",
        "split-marker",
        "preopened",
        "only-newlines",
        "keep-later-newlines",
    ],
)
def test_thinking_stream_strips_newlines_after_close(chunks, enabled, expected):
    state = server.ThinkingStreamState(enable_thinking=enabled)
    assert _thoughts(_feed_thinking(state, chunks, last=True)) == expected


_JSON_TOOLS = NS(
    tool_call_start="<tool_call>",
    tool_call_end="</tool_call>",
    parse_tool_call=lambda call, tools: json.loads(call),
)


def _assert_fields(actual, **expected):
    assert {key: actual[key] for key in expected} == expected


def _sse_events_all(body):
    """Upstream's SSE parser (2026-09-27 sync): skips the ``[DONE]`` sentinel and yields
    event-less chat chunks too. The fork's ``_sse_events`` keeps its stricter list contract
    for the fork's own tests."""
    for block in body.split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in block.splitlines() if ": " in line
        )
        if "data" in fields and fields["data"] != "[DONE]":
            yield fields.get("event"), json.loads(fields["data"])


def _data(response):
    assert response.status_code == 200, response.text
    return [data for _, data in _sse_events_all(response.text)]


def _deltas(response, api="chat"):
    data = _data(response)
    if api == "chat":
        return [item["choices"][0]["delta"] for item in data if item.get("choices")]
    if api == "messages":
        return [
            item["delta"] for item in data if item.get("type") == "content_block_delta"
        ]
    return data


@contextmanager
def _endpoint(
    *,
    model_type="qwen2_vl",
    processor=None,
    config=None,
    result=None,
    chunks=(),
    generator=None,
    template="prompt",
    parser=None,
):
    model, processor = NS(), processor or NS()
    config = config or NS(model_type=model_type)
    with ExitStack() as stack:

        def mock(name, **kwargs):
            return stack.enter_context(patch.object(server, name, **kwargs))

        cached = mock("get_cached_model", return_value=(model, processor, config))
        templating = mock("apply_chat_template", return_value=template)
        generation = mock("generate", return_value=result or _result())
        streaming = mock("stream_generate", side_effect=lambda *a, **kw: iter(chunks))
        stack.enter_context(
            patch.object(server.runtime, "response_generator", generator)
        )
        if parser:
            mock("_infer_tool_parser_from_processor", return_value="demo")
            mock("load_tool_module", return_value=parser)
        yield NS(
            cache=cached,
            template=templating,
            generate=generation,
            stream=streaming,
            config=config,
        )


def _post(client, api="chat", **payload):
    paths = dict(
        chat="/v1/chat/completions", responses="/v1/responses", messages="/v1/messages"
    )
    path = paths.get(api, api)
    body = {"model": "demo"}
    body["input" if "responses" in path else "messages"] = (
        "Hello" if "responses" in path else [_msg()]
    )
    if path == "/v1/messages":
        body["max_tokens"] = 4
    return client.post(path, json={**body, **payload})


def _reset_runtime(monkeypatch, **overrides):
    state = dict(
        model_cache=server.ModelCacheRegistry(),
        response_generator=None,
        apc_manager=None,
    )
    for name, value in (state | overrides).items():
        monkeypatch.setattr(server.runtime, name, value)


def _result(text="done", **kwargs):
    return GenerationResult(
        **(
            dict(
                text=text,
                prompt_tokens=8,
                generation_tokens=4,
                total_tokens=12,
                prompt_tps=10.0,
                generation_tps=5.0,
                peak_memory=0.1,
            )
            | kwargs
        )
    )


def _stream_response(
    client,
    tokens,
    api="/chat/completions",
    *,
    prompt_tokens=3,
    endpoint=None,
    **payload,
):
    with _endpoint(generator=_streaming(tokens, prompt_tokens), **(endpoint or {})):
        return _post(client, api, stream=True, **payload)


def _token(text="", token=1, finish_reason=None, **kwargs):
    return server.StreamingToken(
        text=text, token=token, logprobs=0.0, finish_reason=finish_reason, **kwargs
    )


def _tool(name="get_weather", api="chat"):
    if api == "messages":
        return dict(
            name=name, description="Get weather", input_schema={"type": "object"}
        )
    return dict(
        type="function", function=dict(name=name, parameters={"type": "object"})
    )


def _msg(content="Hello", role="user", **extra):
    return dict(role=role, content=content, **extra)


def _streaming(chunks, prompt_tokens=3):
    return NS(
        tokenizer=NS(decode=lambda tokens: ""),
        validate_context_budget=MagicMock(),
        generate=MagicMock(
            return_value=(
                server.GenerationContext(uid=1, prompt_tokens=prompt_tokens),
                iter(chunks),
            )
        ),
    )


# ---------------------------------------------------------------------------
# Ported from upstream at the 2026-09-27 sync (behaviour changed upstream; the fork's
# stale copies were dropped): model discovery (53616323), Anthropic tool-use streaming
# (67599f2e/8ff71517), tool-call stream parity (f16c98f2..3c001d01).
# ---------------------------------------------------------------------------


class TestModelDiscovery:
    @staticmethod
    def _model_directory(path):
        path.mkdir(parents=True)
        (path / "config.json").write_text('{"model_type": "qwen2_vl"}')
        (path / "model.safetensors").write_bytes(b"weights")
        return path

    @pytest.mark.parametrize(
        "config,valid",
        [
            ('{"model_type": "qwen2_vl"}', True),
            ('{"model_type": "custom", "model_file": "model.py"}', True),
            ("not json", False),
            ("{}", False),
        ],
        ids=["no-tokenizer", "custom-code", "malformed", "empty"],
    )
    def test_metadata(self, tmp_path, config, valid):
        model = self._model_directory(tmp_path / "model")
        (model / "config.json").write_text(config)
        (model / "model.py").write_text("raise RuntimeError('must not execute')")
        assert is_model_directory(model) is valid

    @pytest.mark.parametrize(
        "shard,valid",
        [(None, False), (b"", False), (b"weights", True)],
        ids=["missing", "empty", "complete"],
    )
    def test_shards(self, tmp_path, shard, valid):
        model = self._model_directory(tmp_path / "model")
        (model / "model.safetensors.index.json").write_text(
            '{"weight_map": {"a": "model.safetensors", "b": "second.safetensors"}}'
        )
        if shard is not None:
            (model / "second.safetensors").write_bytes(shard)
        assert is_model_directory(model) is valid

    def test_rejects_adapters_and_broken_links(self, tmp_path):
        model = self._model_directory(tmp_path / "adapter")
        (model / "model.safetensors").rename(model / "adapter_model.safetensors")
        assert not is_model_directory(model)
        (model / "model.safetensors").symlink_to(model / "missing.safetensors")
        assert not is_model_directory(model)

    def test_pipeline_components(self, tmp_path):
        pipeline = tmp_path / "pipeline"
        component = self._model_directory(pipeline / "transformer")
        (pipeline / "model_index.json").write_text('{"_class_name": "FluxPipeline"}')
        (pipeline / "tokenizer").mkdir()
        assert is_model_directory(pipeline)
        (component / "model.safetensors").unlink()
        assert not is_model_directory(pipeline)
        self._model_directory(pipeline / "text_encoder")
        (component / "model.safetensors.index.json").write_text(
            '{"weight_map": {"a": "missing.safetensors"}}'
        )
        assert not is_model_directory(pipeline)

    @pytest.mark.parametrize("main", ["absent", "complete", "incomplete"])
    def test_revisions_and_local_alias(self, tmp_path, main):
        repo = tmp_path / "models--local--vision"
        snapshots = [
            self._model_directory(repo / "snapshots" / (revision * 40))
            for revision in "ab"
        ]
        for modified, snapshot in zip((100, 200), snapshots):
            for path in (snapshot, *snapshot.iterdir()):
                os.utime(path, (modified, modified))
        if main != "absent":
            (repo / "refs").mkdir()
            (repo / "refs" / "main").write_text("a" * 40)
        if main == "incomplete":
            (snapshots[0] / "model.safetensors").unlink()
        selected = snapshots[0] if main == "complete" else snapshots[1]
        cache = scan_cache_dir(tmp_path)
        found = discover_models(cache)
        assert found == [
            dict(
                id="local/vision" if main == "complete" else str(selected),
                path=selected,
                created=100 if main == "complete" else 200,
            )
        ]
        assert discover_models(cache, [str(selected)]) == found

    @pytest.mark.parametrize("source", ["parent", "home", "model", "alias", "combined"])
    def test_custom_roots_and_aliases(self, tmp_path, source):
        root = tmp_path / "models"
        model = self._model_directory(root / "custom")
        alias = root / "alias"
        alias.symlink_to(model, target_is_directory=True)
        (root / "unrelated").mkdir()
        sources = dict(
            parent=str(root),
            home="~/" + os.path.relpath(root, Path.home()),
            model=str(model),
            alias=str(alias),
            missing=str(root / "missing"),
        )
        paths = list(sources.values()) if source == "combined" else [sources[source]]
        found = discover_models(NS(repos=[]), paths)
        assert (
            len(found) == 1
            and found[0]["id"] == str(model)
            and found[0]["path"] == model
        )

    @pytest.fixture
    def model_listing(self, client, monkeypatch, tmp_path):
        _reset_runtime(monkeypatch)
        monkeypatch.delenv("MLX_VLM_MODEL_PATHS", raising=False)
        cache_root = tmp_path / "cache"
        model = self._model_directory(
            cache_root / "models--local--vision" / "snapshots" / ("a" * 40)
        )
        refs = model.parent.parent / "refs"
        refs.mkdir()
        (refs / "main").write_text("a" * 40)
        scan = Mock(side_effect=lambda: scan_cache_dir(cache_root))
        monkeypatch.setattr(server, "scan_cache_dir", scan)

        def get(endpoint="/v1/models", **kwargs):
            response = client.get(endpoint, **kwargs)
            assert response.status_code == 200
            entries = response.json()["data"]
            ids = [m["id"] for m in entries]
            assert ids == sorted(set(ids), key=str.lower)
            return {m["id"]: m["loaded"] for m in entries}

        return NS(get=get, scan=scan, path=model, registry=server.runtime.model_cache)

    def test_endpoint_cache_and_loaded_status(self, model_listing):
        listing = model_listing
        for kind, model in (
            ("text_generation", "local/vision"),
            ("embedding", "/loaded/embedding"),
            ("tts", "/loaded/tts"),
        ):
            listing.registry.set(kind, {"model_path": model})
        expected = {
            "local/vision": True,
            "/loaded/embedding": True,
            "/loaded/tts": True,
        }
        assert listing.get("/models") == listing.get() == expected
        listing.registry.clear()
        assert listing.get() == {"local/vision": False}
        (listing.path / "model.safetensors").unlink()
        assert listing.get() == {}

    @pytest.mark.parametrize("cached", [False, True], ids=["missing-cache", "cached"])
    @pytest.mark.parametrize("source", ["environment", "query"])
    def test_endpoint_custom_paths(self, model_listing, monkeypatch, cached, source):
        listing = model_listing
        path = str(listing.path)
        if not cached:
            listing.scan.side_effect = server.CacheNotFound("missing cache", "/missing")
        params = {"model_dir": path} if source == "query" else {}
        if source == "environment":
            monkeypatch.setenv("MLX_VLM_MODEL_PATHS", path)
        listing.registry.set("text_generation", {"model_path": path})
        listing.registry.set("embedding", {"model_path": "/loaded/embedding"})
        assert listing.get(params=params) == {path: True, "/loaded/embedding": True}
        listing.registry.clear()
        assert listing.get(params=params) == {"local/vision" if cached else path: False}

    def test_endpoint_query_paths_are_additive_and_temporary(
        self, model_listing, monkeypatch, tmp_path
    ):
        configured = self._model_directory(tmp_path / "configured")
        requested = self._model_directory(
            tmp_path / "requested" / "model with spaces & symbols"
        )
        another = self._model_directory(tmp_path / "another")
        monkeypatch.setenv("MLX_VLM_MODEL_PATHS", str(configured))
        baseline = {"local/vision": False, str(configured): False}
        params = [("model_dir", str(path)) for path in (requested.parent, another, "")]
        assert model_listing.get(params=params) == {
            **baseline,
            str(requested): False,
            str(another): False,
        }
        assert os.environ["MLX_VLM_MODEL_PATHS"] == str(configured)
        assert model_listing.get() == baseline

    @pytest.mark.parametrize("use_cli_paths", [False, True])
    def test_cli_custom_model_paths(self, monkeypatch, tmp_path, use_cli_paths):
        monkeypatch.setattr(os, "environ", dict(os.environ))
        monkeypatch.setenv("MLX_VLM_MODEL_PATHS", "/existing/models")
        paths = [str(tmp_path / "model with spaces"), str(tmp_path / "other")]
        flags = (
            [arg for path in paths for arg in ("--model-dir", path)]
            if use_cli_paths
            else []
        )
        monkeypatch.setattr(sys, "argv", ["mlx_vlm.server", *flags])
        with patch.object(cli.uvicorn, "run") as run:
            cli.main()
        assert os.environ["MLX_VLM_MODEL_PATHS"] == (
            os.pathsep.join(paths) if use_cli_paths else "/existing/models"
        )
        run.assert_called_once()


def test_anthropic_messages_streaming_emits_tool_use_events(client):
    token = _token(
        '<tool_call>{"name":"get_weather","arguments":{"location":"SF"}}</tool_call> After the call.',
        finish_reason="stop",
    )
    response = _stream_response(
        client,
        [token],
        "messages",
        endpoint=dict(parser=_JSON_TOOLS),
        tools=[_tool(api="messages")],
    )
    assert response.status_code == 200
    for fragment in (
        '"type": "tool_use"',
        '"name": "get_weather"',
        '"type": "input_json_delta"',
        '"partial_json": "{\\"location\\": \\"SF\\"}"',
        '"text": "After the call."',
        '"stop_reason": "tool_use"',
    ):
        assert fragment in response.text


@pytest.mark.parametrize(
    "chunks,start_marker,end_marker,expected,inside",
    [
        (["text<tool_call>"], "<tool_call>", "</tool_call>", "text<tool_call>", True),
        (
            ["Before ", "<tool_call>", '{"name": "a"}', " trailing"],
            "<tool_call>",
            "",
            "Before",
            True,
        ),
        (["A literal <tool"], "<tool_call>", "</tool_call>", "A literal <tool", False),
        (
            [*MINICPM_MULTICALL, ""],
            "<function",
            "</function>",
            "Before Between After",
            False,
        ),
        (
            ["<tool_call>a</tool_call>", "\n", "<tool_call>b</tool_call>", "\n"],
            "<tool_call>",
            "</tool_call>",
            "",
            False,
        ),
        (
            list("<tool_call>a</tool_call>\n<tool_call>b</tool_call>\n"),
            "<tool_call>",
            "</tool_call>",
            "",
            False,
        ),
        (
            ["<tool_call>a</tool_call>", "\n", "Done", "."],
            "<tool_call>",
            "</tool_call>",
            "Done.",
            False,
        ),
        (
            list("A<tool_call>x</tool_call> \n<tool_call>y</tool_call>B"),
            "<tool_call>",
            "</tool_call>",
            "A  \n B",
            False,
        ),
        (
            list("A<tool_call>x</tool"),
            "<tool_call>",
            "</tool_call>",
            "A<tool_call>x</tool",
            True,
        ),
        (
            list("  <tool_call>x</tool_call>B"),
            "<tool_call>",
            "</tool_call>",
            "B",
            False,
        ),
        (
            ["<tool_call>x</tool_call> B", "  "],
            "<tool_call>",
            "</tool_call>",
            "B",
            False,
        ),
        (
            ["Before ", "[TOOL_CALLS]foo[ARGS]{}", "\nAfter"],
            "[TOOL_CALLS]",
            "",
            "Before  After",
            False,
        ),
    ],
    ids=[
        "start-marker",
        "missing-end-marker",
        "unfinished-start-marker",
        "minicpm-character-chunks",
        "whitespace-between-calls",
        "whitespace-between-calls-character-chunks",
        "text-after-call",
        "whitespace-between-text",
        "unfinished-call",
        "leading-whitespace",
        "trailing-whitespace",
        "no-end-marker-ends-at-newline",
    ],
)
def test_tool_stream_finalization(chunks, start_marker, end_marker, expected, inside):
    state = ToolCallStreamState(start_marker, end_marker)
    visible = [
        state.feed(chunk, last=i == len(chunks) - 1) for i, chunk in enumerate(chunks)
    ]
    assert "".join(delta for delta in visible if delta) == expected
    assert state.in_tool_call is inside


_CALL = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'


@pytest.mark.parametrize(
    "parser,text",
    [
        ("json_tools", f"{_CALL}\n{_CALL}\n"),
        ("json_tools", f"Hi {_CALL}\n{_CALL}\n bye"),
        ("json_tools", f"{_CALL} \nDone."),
        ("json_tools", f"A{_CALL}B"),
        ("json_tools", f"A{_CALL}\nB<tool_call>unfinished"),
        ("json_tools", "A <tool_call>unfinished"),
        ("json_tools", "No calls\n\n"),
        ("minicpm5", '<function name="get_time"></function>Use <function as a prefix.'),
        ("mistral", 'Before [TOOL_CALLS]foo[ARGS]{"a": 1}\nAfter'),
        ("mistral", "[TOOL_CALLS]foo[ARGS]{}\n[TOOL_CALLS]bar[ARGS]{}"),
    ],
)
def test_tool_stream_matches_non_streamed_content(parser, text):
    # Streamed content equals the non-streamed content, whatever the chunking:
    # beside a parsed call, the text process_tool_calls leaves with protocol
    # markers removed; otherwise the whole output. Both are stripped.
    module = load_tool_module(parser)
    parsed = process_tool_calls(text, module, None)
    expected = (
        strip_protocol_markers(parsed.remaining_text, module)
        if parsed.calls
        else text.strip()
    )
    for chunks in ([text], list(text)):
        state = ToolCallStreamState(module.tool_call_start, module.tool_call_end)
        streamed = "".join(
            state.feed(chunk, last=i == len(chunks) - 1) or ""
            for i, chunk in enumerate(chunks)
        )
        assert streamed == expected


_WEATHER_CALL = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'


def test_chat_fallback_stream_parses_tool_calls(client):
    # Without a response generator the stream_generate fallback streamed the
    # raw tool-call markup as content and never emitted tool_calls.
    result = _result(f"Checking.{_WEATHER_CALL}", finish_reason="stop")
    with _endpoint(chunks=[result], parser=_JSON_TOOLS):
        response = _post(client, stream=True, tools=[_tool()])
    deltas = _deltas(response)
    assert _joined(deltas, "content") == "Checking."
    calls = [call for delta in deltas for call in delta.get("tool_calls") or []]
    assert [call["function"]["name"] for call in calls] == ["get_weather"]
    reasons = [
        choice["finish_reason"]
        for chunk in _data(response)
        for choice in chunk.get("choices") or []
        if choice.get("finish_reason")
    ]
    assert reasons == ["tool_calls"]


@pytest.mark.parametrize("api", ["chat", "responses", "messages"])
def test_stream_without_finish_token_flushes_held_text(client, api):
    # The iterator stops without a finish reason: the unfinished call is not a
    # call, so its text is content, as in the non-streamed response.
    tool = _tool(api="messages") if api == "messages" else _tool()
    if api == "responses":
        tool = dict(type="function", name="get_weather", parameters={"type": "object"})
    response = _stream_response(
        client,
        [_token("A <tool_call>unfinished")],
        api,
        endpoint=dict(parser=_JSON_TOOLS),
        tools=[tool],
    )
    deltas = _deltas(response, api)
    if api == "chat":
        text = _joined(deltas, "content")
    elif api == "messages":
        text = _joined(deltas, "text")
    else:
        text = _joined(
            [d for d in deltas if d.get("type") == "response.output_text.delta"],
            "delta",
        )
    assert text == "A <tool_call>unfinished"


@pytest.mark.parametrize("api", ["chat", "responses", "messages"])
def test_tool_call_content_keeps_angle_bracket_text(client, api):
    result = _result(
        f"<think>r</think>Use <b>bold</b>.<|im_end|></think> {_WEATHER_CALL}"
        " </tool_call>"
    )
    tool = (
        dict(type="function", name="get_weather", parameters={"type": "object"})
        if api == "responses"
        else _tool(api=api)
    )
    with _endpoint(result=result, parser=_JSON_TOOLS):
        response = _post(client, api, tools=[tool])
    assert response.status_code == 200, response.text
    body = response.json()
    if api == "chat":
        message = body["choices"][0]["message"]
        assert message["content"] == "Use <b>bold</b>."
        assert message["tool_calls"][0]["function"]["name"] == "get_weather"
    elif api == "messages":
        assert [b["text"] for b in body["content"] if b["type"] == "text"] == [
            "Use <b>bold</b>."
        ]
        assert [b["name"] for b in body["content"] if b["type"] == "tool_use"] == [
            "get_weather"
        ]
    else:
        texts = [
            part["text"]
            for item in body["output"]
            if item.get("type") == "message"
            for part in item["content"]
        ]
        assert (
            "Use <b>bold</b>." in texts or body.get("output_text") == "Use <b>bold</b>."
        )


# Fork (2026-10-06 v0.7.6 sync): tests upstream added between 967bf90b and v0.7.6
# (#2408/#2418 compaction, Responses replay/stream fixes, #2402 decisions), ported
# byte-identically into this fork-owned copy (C104(c)). Fork deviations are marked inline:
# upstream's `_sse_events` generator is `_sse_events_all` here (2026-09-27 convention), and
# chat compaction is OPT-IN in the fork, so chat cases that relied on automatic
# compaction now send an explicit context_management list or are dropped.


# Upstream helpers the ported tests use (absent from the fork copy until now).
def _input_message(text, role="user"):
    return _msg([dict(type="input_text", text=text)], role, type="message")


def _input_image(url, **options):
    return dict(type="input_image", image_url=url, **options)


def _function_result(output, name=None, call_id="call_view_image"):
    items = (
        [dict(type="function_call", name=name, arguments="{}", call_id=call_id)]
        if name
        else []
    )
    return [*items, dict(type="function_call_output", call_id=call_id, output=output)]


def test_responses_tool_arguments_are_normalized_without_mutating_input():
    items = [
        {
            "type": "function_call",
            "name": "exec_command",
            "call_id": "c1",
            "arguments": '{"cmd":"cat log.txt"}',
        }
    ]
    original = copy.deepcopy(items)
    messages, _ = _response_items_to_chat(items)
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {
        "cmd": "cat log.txt"
    }
    assert messages[0]["content"] == ""
    assert items == original


@pytest.mark.parametrize(
    "content", [None, "", [], [{"type": "output_text", "text": ""}]]
)
@pytest.mark.parametrize("sealed", [False, True])
def test_responses_replay_ignores_empty_assistant_turns(
    client, content, sealed, tmp_path, monkeypatch
):
    monkeypatch.setenv("MLX_VLM_COMPACTION_KEY_FILE", str(tmp_path / "key"))
    history = [
        _input_message("Read the file."),
        _msg(content, "assistant", type="message"),
        *_function_result("File contents", name="read_file"),
    ]
    if sealed:
        history = [_seal_context(history)]
    with _endpoint() as fake:
        response = _post(
            client, "responses", input=[*history, _input_message("Continue.")]
        )
    assert response.status_code == 200, response.text
    messages = fake.template.call_args.args[2]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert messages[2]["content"] == "File contents"


@pytest.mark.parametrize(
    "extra",
    [
        {"reasoning_content": "Check the file first."},
        {"reasoning": "Check the file first."},
        {
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ]
        },
        {"content": [_input_image("https://example.com/image.png")]},
    ],
)
def test_responses_replay_preserves_assistant_payloads_without_text(extra):
    item = {"type": "message", "role": "assistant", "content": [], **extra}
    original = copy.deepcopy(item)
    messages, images = _response_items_to_chat([item])
    assert messages
    if "content" in extra:
        assert images == ["https://example.com/image.png"]
    elif "tool_calls" in extra:
        assert messages[0]["tool_calls"][0]["function"]["name"] == "read_file"
    else:
        assert messages[0]["reasoning_content"] == "Check the file first."
    assert item == original


def _completed_response(response):
    events = _data(response)
    final = next(e["response"] for e in events if e["type"] == "response.completed")
    output = final["output"]
    for kind in ("added", "done"):
        items = [e for e in events if e["type"] == f"response.output_item.{kind}"]
        assert [e["output_index"] for e in items] == list(range(len(output)))
        assert [e["item"]["id"] for e in items] == [item["id"] for item in output]
        if kind == "done":
            assert [e["item"] for e in items] == output
    added = set()
    for event in events:
        if event["type"] == "response.output_item.added":
            added.add(event["item"]["id"])
        elif event["type"] == "response.output_item.done":
            assert event["item"]["id"] in added
        if "item_id" in event:
            assert event["item_id"] in added
            assert output[event["output_index"]]["id"] == event["item_id"]
    reasoning = "".join(
        part["text"]
        for item in output
        if item["type"] == "reasoning"
        for part in item["summary"]
    )
    for kind, expected in (
        ("output_text", final["output_text"]),
        ("reasoning_text", reasoning),
    ):
        for suffix, field in (("delta", "delta"), ("done", "text")):
            chunks = [e for e in events if e["type"] == f"response.{kind}.{suffix}"]
            assert _joined(chunks, field) == expected
    return final


def _response(client, api="responses", *, status=200, **payload):
    response = _post(client, api, **payload)
    assert response.status_code == status, response.text
    return (
        _completed_response(response)
        if status == 200 and payload.get("stream")
        else response.json()
    )


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize(
    "parts,types",
    [
        (
            ["<tool_call>", '{"name":"get_weather","arguments":{}}', "</tool_call>"],
            ["function_call"],
        ),
        (["Checking. ", _WEATHER_CALL], ["message", "function_call"]),
        (
            ["<think>", "Check first.", "</think>", _WEATHER_CALL],
            ["reasoning", "function_call"],
        ),
        (["<think>Check first.</think>", "Sunny."], ["reasoning", "message"]),
        (["Hello", " world."], ["message"]),
        ([""], []),
        (["<think>Check first.</think>"], ["reasoning"]),
        ([" ", "\n"], []),
        ([" \n", "<think>", "Check first.", "</think>"], ["reasoning"]),
        (["<think>", " \n", "</think>"], []),
        ([" \n", "Hello ", "\n  ", "world. ", "\n"], ["message"]),
        (["<think> \n", "Check ", "\n  ", "first. \n</think>", " \n"], ["reasoning"]),
    ],
    ids=[
        "tool",
        "text-tool",
        "reasoning-tool",
        "reasoning-text",
        "text",
        "empty",
        "reasoning",
        "whitespace",
        "space-before-reasoning",
        "empty-reasoning",
        "text-spacing",
        "reasoning-spacing",
    ],
)
def test_responses_stream_items_match_completed_output(
    client, continuous, parts, types
):
    chunks = [
        (_token if continuous else _result)(
            text, finish_reason="stop" if i == len(parts) - 1 else None
        )
        for i, text in enumerate(parts)
    ]
    with _endpoint(
        chunks=chunks,
        generator=_streaming(chunks) if continuous else None,
        parser=_JSON_TOOLS,
    ):
        final = _response(
            client, stream=True, tools=[_tool()] if "function_call" in types else []
        )
    assert [item["type"] for item in final["output"]] == types


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_responses_truncated_preopened_reasoning(client, continuous, stream):
    text = "Check first."
    with _endpoint(
        template="prompt<think>",
        result=_result(text, finish_reason="length"),
        chunks=[_result(text, finish_reason="length")],
        generator=(
            _streaming([_token(text, finish_reason="length")]) if continuous else None
        ),
    ):
        response = _response(client, stream=stream)
    assert response["output_text"] == ""
    assert len(response["output"]) == 1
    item = response["output"][0]
    assert item["type"] == "reasoning"
    assert item["summary"] == [{"type": "summary_text", "text": text}]


def _compaction_message(text, role="user"):
    return {"type": "message", "role": role, "content": text}


def _seal_context(items, **scope):
    return compaction.seal(items, **{"model": "demo", "tenant": None, **scope})


def _resolve_context(items, **scope):
    return compaction.resolve(items, **{"model": "demo", "tenant": None, **scope})


def _compaction_history():
    return [
        _compaction_message("Follow the user's constraints.", "system"),
        _compaction_message(
            "Project ORCHID. The chosen port is 7319. Never edit secrets.env."
        ),
        *_function_result("old log entry\n" * 1500, name="read_file", call_id="c1"),
        _compaction_message("Read the log. Next: verify configuration.", "assistant"),
        _compaction_message("What is the port?"),
    ]


def _real_post(client, api="responses", **payload):
    return _response(
        client.http,
        api,
        model=client.model,
        temperature=0,
        enable_thinking=False,
        **payload,
    )


def _real_answer(client, items):
    return _real_post(client, input=items, max_output_tokens=96, store=False)


def _real_compact(client, items):
    return client.sdk.responses.compact(
        model=client.model,
        input=items,
        extra_body={
            "keep_tokens": 256,
            "max_output_tokens": 768,
            "temperature": 0,
            "enable_thinking": False,
        },
    )


def _assert_recalled(text, port="7319"):
    assert all(fact in text for fact in ("ORCHID", port, "secrets.env")), text


class TestCompaction:
    """Responses compaction contracts and opt-in real-model integration."""

    @pytest.fixture(autouse=True)
    def isolated_state(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MLX_VLM_COMPACTION_KEY_FILE", str(tmp_path / "key"))
        # Fork: chat compaction is a server switch (default off); upstream's
        # contracts are exercised with it on.
        monkeypatch.setenv("MLX_VLM_CHAT_COMPACTION", "1")
        monkeypatch.setattr(server.runtime.config, "max_kv_size", None)
        server.response_store.clear()
        server.response_store_order.clear()

    @pytest.fixture
    def mocked(self, client):
        with (
            _endpoint(
                config=NS(model_type="qwen2_vl", max_position_embeddings=32768),
                result=_result(
                    "Goal: ORCHID. Port: 7319. Constraint: never edit secrets.env.",
                    cached_tokens=32,
                ),
            ) as fake,
            patch.object(openai, "prepare_inputs") as prepare,
            patch.object(compaction, "prepare_inputs", prepare),
        ):
            fake.template.side_effect = (
                lambda processor, config, messages, **kw: json.dumps(messages)
            )
            prepare.side_effect = lambda processor, prompts, **kw: {
                "input_ids": np.zeros((1, max(1, len(prompts) // 4)), dtype=np.int32)
            }
            fake.request = partial(_response, client)
            fake.compact = partial(fake.request, "/responses/compact")
            fake.count = partial(fake.request, "/responses/input_tokens")
            yield fake

    @pytest.fixture
    def requirements(self):
        return [
            _compaction_message(text)
            for text in (
                "Project ORCHID. Never edit secrets.env.",
                "Use port 8421 instead of 7319.",
                "What are the current requirements?",
            )
        ]

    @pytest.mark.parametrize("path", ["/responses/compact", "/v1/responses/compact"])
    def test_compact_round_trip_and_token_count(self, mocked, path):
        original = _compaction_history()
        data = mocked.request(
            path, input=original, keep_tokens=0, max_output_tokens=256
        )
        assert data["object"] == "response.compaction"
        assert data["usage"]["input_tokens_details"]["cached_tokens"] == 32
        item = data["output"][0]
        assert item["type"] == "compaction"
        assert "ORCHID" not in item["encrypted_content"]
        restored = _resolve_context([item])
        assert restored[0] == original[0] and restored[-1] == original[-1]
        assert restored[1] == original[1]
        assert restored[2]["role"] == "assistant"
        assert "7319" in restored[2]["content"][0]["text"]
        assert "old log entry" not in json.dumps(restored)
        for inputs in ([item], original + [item], restored):
            response = mocked.count(input=inputs)
            if inputs == [item]:
                expected = response
            assert response == expected
        mocked.request(
            input=[item, _compaction_message("Continue")],
            instructions="New current instructions",
        )
        rendered = json.loads(mocked.generate.call_args.kwargs["prompt"])
        assert "New current instructions" in rendered[0]["content"]
        assert rendered[-1]["content"] == "Continue"

    @pytest.mark.parametrize("change", ["tenant", "model", "tamper", "foreign"])
    def test_capsules_reject_invalid_scope_and_payload(self, change):
        item = _seal_context([_compaction_message("private")], tenant="one")
        model, tenant = "demo", "one"
        if change == "tenant":
            tenant = "two"
        elif change == "model":
            model = "other"
        elif change == "tamper":
            value = item["encrypted_content"]
            at = len(compaction.CAPSULE_PREFIX) + 50
            item["encrypted_content"] = (
                value[:at] + ("A" if value[at] != "A" else "B") + value[at + 1 :]
            )
        else:
            item["encrypted_content"] = "openai-issued-opaque-state"
        with pytest.raises(HTTPException) as caught:
            _resolve_context([item], model=model, tenant=tenant)
        assert caught.value.status_code == 400

    def test_latest_capsule_replaces_previous_context(self):
        first = _seal_context([_compaction_message("old")])
        second = _seal_context([_compaction_message("new")])
        assert _resolve_context(
            [first, _compaction_message("between"), second, _compaction_message("last")]
        ) == [_compaction_message("new"), _compaction_message("last")]

    def test_resent_codex_instructions_do_not_accumulate_in_capsules(self):
        prefix = _compaction_message("Current permission constraints.", "developer")
        other = _compaction_message("A separate requirement.", "developer")
        original = [prefix, other, _compaction_message("handoff", "assistant")]
        for _ in range(5):
            capsule = _seal_context(original)
            original = _resolve_context(
                [
                    capsule,
                    {**prefix, "id": "new-client-id"},
                    _compaction_message("Continue"),
                ]
            )
            assert sum(x.get("content") == prefix["content"] for x in original) == 1
            assert other in original

    def test_codex_retained_messages_survive_capsule_replay(self, requirements):
        instruction = _compaction_message("Keep user constraints.", "developer")
        requirement, correction, question = requirements
        repeated = _compaction_message("Continue")
        latest = _compaction_message("Read the log")
        call = {"type": "function_call", "call_id": "c1", "name": "read_file"}
        output = {"type": "function_call_output", "call_id": "c1", "output": "done"}
        context = [
            instruction,
            _compaction_message("Routine logs inspected.", "assistant"),
            latest,
            call,
            output,
        ]
        retained = [requirement, repeated, repeated, correction]
        expected = context[:1] + retained + context[1:]
        for cycle in range(3):
            capsule = _seal_context(context)
            # Clients may regenerate message IDs when rebuilding the window.
            inputs = [
                {**item, "id": f"{cycle}-{index}"}
                for index, item in enumerate([instruction, *retained, latest])
            ] + [capsule, question]
            original = copy.deepcopy(inputs)
            context = _resolve_context(inputs)
            assert inputs == original
            assert [
                {key: value for key, value in item.items() if key != "id"}
                for item in context
            ] == expected + [inputs[-1]]
            context = context[:-1]

    @pytest.mark.parametrize("stream", [False, True])
    def test_codex_retained_messages_reach_generation_and_token_count(
        self, mocked, stream, requirements
    ):
        requirement, correction, question = requirements
        context = [_compaction_message("Routine logs inspected.", "assistant")]
        capsule = _seal_context(context)
        inputs = [requirement, correction, capsule, question]
        expected = [requirement, correction, *context, question]
        counted = mocked.count(input=inputs)
        plain = mocked.count(input=expected)
        assert counted == plain
        mocked.request(input=inputs, stream=stream)
        rendered = mocked.template.call_args.args[2]
        assert [item["content"] for item in rendered] == [
            item["content"] for item in expected
        ]

    def test_codex_retained_messages_count_toward_context_budget(
        self, mocked, monkeypatch
    ):
        capsule = _seal_context([_compaction_message("Short handoff", "assistant")])
        monkeypatch.setattr(server.runtime.config, "max_kv_size", 1024)
        response = mocked.request(
            input=[_compaction_message("Retained user text. " * 1000), capsule],
            max_output_tokens=64,
            context_management=[{"type": "compaction", "compact_threshold": 10000}],
            status=400,
        )
        assert (
            response["detail"]
            == "Protected conversation exceeds the available context budget."
        )
        mocked.generate.assert_not_called()

    def test_pending_parallel_tools_cannot_be_split(self):
        items = [
            _compaction_message("start"),
            {"type": "function_call", "call_id": "a"},
            {"type": "function_call", "call_id": "b"},
            {"type": "function_call_output", "call_id": "a"},
            _compaction_message("steering during tools"),
            {"type": "function_call_output", "call_id": "b"},
            _compaction_message("next"),
        ]
        assert compaction.safe_boundaries(items) == [0, 6]

    @pytest.mark.parametrize("api", ["standalone", "trigger", "automatic"])
    @pytest.mark.parametrize("limit", [4096, 16384])
    def test_bounded_retention_survives_replay_and_recompaction(
        self, mocked, monkeypatch, api, limit, requirements
    ):
        monkeypatch.setattr(server.runtime.config, "max_kv_size", limit)
        mocked.generate.return_value = _result("Routine work completed.")
        requirement, correction, question = requirements
        retained = [requirement]
        window = [requirement]
        for cycle in range(2):
            log = _compaction_message(
                f"Log {cycle}: " + "routine evidence " * (limit // 20)
            )
            log["id"] = f"log-{cycle}"
            retained.extend([log, correction, question])
            window += [log, correction, question]
            options = dict(input=window, max_output_tokens=128, store=False)
            if api == "standalone":
                options["keep_tokens"] = 0
                response = mocked.compact(**options)
            else:
                if api == "trigger":
                    options["input"] = window + [{"type": "compaction_trigger"}]
                else:
                    options["context_management"] = [
                        {"type": "compaction", "compact_threshold": 1}
                    ]
                response = mocked.request(**options)
            capsule = response["output"][0]
            assert capsule["type"] == "compaction"
            context = _resolve_context([capsule])
            assert requirement in context and correction in context
            assert "routine evidence" not in json.dumps(context)
            resent = [
                {
                    **x,
                    "id": f"replayed-{cycle}-{i}",
                    "content": [{"type": "input_text", "text": x["content"]}],
                }
                for i, x in enumerate(retained)
            ]
            before = mocked.count(input=window)
            after = mocked.count(input=[capsule])
            replay = mocked.count(input=resent + [capsule])
            assert replay == after
            assert after["input_tokens"] < before["input_tokens"] * 0.6
            assert _resolve_context(resent + [capsule]) == context
            shortened = {**log, "content": "routine evidence [truncated]"}
            assert _resolve_context([shortened, capsule]) == context
            fresh = _compaction_message("A new requirement.")
            assert _resolve_context(resent + [fresh, capsule, log]) == [
                fresh,
                *context,
                log,
            ]
            mocked.generate.reset_mock()
            unchanged = mocked.compact(
                input=[capsule],
                keep_tokens=limit,
                max_output_tokens=128,
            )
            mocked.generate.assert_not_called()
            window = resent + unchanged["output"]
            assert _resolve_context(window) == context

    def test_retention_budget_preserves_whole_messages_in_order(self):
        messages = [_compaction_message(f"Requirement {i}. " * 10) for i in range(8)]
        latest = _compaction_message("Continue")

        async def count(items):
            return len(json.dumps(items))

        async def summarize(items):
            return "Earlier requirements summarized.", None

        result = asyncio.run(
            compaction.compact(
                messages + [latest],
                count=count,
                summarize=summarize,
                keep_tokens=0,
                target_tokens=1600,
                retain_tokens=500,
            )
        )
        originals = [x for x in result.items[:-1] if x.get("role") == "user"]
        assert originals == messages[-2:]
        base = [x for x in result.items if x not in originals]
        assert result.after_tokens - asyncio.run(count(base)) <= 500
        capsule = _seal_context(result.items, covered=result.covered)
        assert _resolve_context(messages + [capsule, latest]) == result.items + [latest]

    @pytest.mark.parametrize(
        "text, tokens, status, detail",
        [
            pytest.param("", 4, 502, "empty summary", id="empty"),
            pytest.param("partial summary", 256, 502, "output limit", id="length"),
            pytest.param(
                "huge " * 10000,
                4,
                400,
                "could not reach the context budget",
                id="too-large",
            ),
        ],
    )
    def test_summary_failures_preserve_original_context(
        self, mocked, text, tokens, status, detail
    ):
        original = _compaction_history()
        mocked.generate.return_value = _result(text, generation_tokens=tokens)
        with patch.object(
            compaction,
            "compact_response_context",
            wraps=compaction.compact_response_context,
        ) as compact:
            response = mocked.compact(
                input=original,
                max_output_tokens=256,
                keep_tokens=0,
                status=status,
            )
        assert detail in response["detail"]
        compact.assert_awaited_once()
        assert compact.call_args.args[1] == original and not server.response_store

    def test_short_history_is_a_noop(self, mocked):
        items = [_compaction_message("hello")]
        response = mocked.compact(input=items)
        assert response["output"] == items
        mocked.generate.assert_not_called()

    def test_summary_budget_overrides_legacy_max_tokens(self, mocked):
        mocked.compact(
            input=_compaction_history(),
            keep_tokens=0,
            max_output_tokens=256,
            max_tokens=4096,
        )
        assert mocked.generate.call_args.kwargs["max_tokens"] == 256

    def test_auto_below_threshold_only_generates_the_answer(self, mocked):
        response = mocked.request(
            input="hello",
            max_output_tokens=64,
            context_management=[{"type": "compaction", "compact_threshold": 1000}],
        )
        assert all(item["type"] != "compaction" for item in response["output"])
        assert mocked.generate.call_count == 1

    @pytest.mark.parametrize("stream", [False, True])
    # Fork: "chat-default" (automatic chat compaction) dropped; the fork's opt-in rule
    # is pinned by tests/test_chat_request_contracts.py.
    @pytest.mark.parametrize("api", ["responses", "chat"])
    def test_output_budget_overrides_compaction_threshold(
        self, mocked, client, monkeypatch, api, stream
    ):
        items = [_compaction_message("old evidence " * 150) for _ in range(6)]
        items.append(_compaction_message("Continue"))
        before = mocked.count(input=items)["input_tokens"]
        monkeypatch.setattr(server.runtime.config, "max_kv_size", before + 32)
        options = dict(
            stream=stream,
            context_management=[{"type": "compaction", "compact_threshold": 100000}],
        )
        if api.startswith("chat"):
            options.update(messages=items, max_tokens=256)
            if api == "chat-default":
                options.pop("context_management")
            api = "chat"
        else:
            options.update(input=items, max_output_tokens=256)
        response = _post(client, api, **options)
        assert response.status_code == 200, response.text
        assert mocked.generate.call_count == (1 if stream else 2)
        answer_call = mocked.stream.call_args if stream else mocked.generate.call_args
        assert len(answer_call.kwargs["prompt"]) // 4 + 256 <= before + 32
        assert "Conversation handoff" in answer_call.kwargs["prompt"]

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("mode", ["automatic", "trigger", "chat"])
    def test_oversized_history_uses_bounded_summary_calls(
        self, mocked, client, monkeypatch, mode, stream
    ):
        monkeypatch.setattr(server.runtime.config, "max_kv_size", 2048)
        items = [_compaction_message("Keep the constraints.", "system")]
        for index in range(6):
            items += [
                _compaction_message(f"batch-{index} " + "evidence " * 200),
                *_function_result("done", name="read_file", call_id=f"c{index}"),
            ]
        items.append(_compaction_message("Continue"))
        original = copy.deepcopy(items)
        options = dict(stream=stream)
        api = "responses"
        if mode == "trigger":
            options.update(input=items + [{"type": "compaction_trigger"}])
        elif mode == "chat":
            api = "chat"
            options.update(
                messages=[_compaction_message("old evidence " * 180) for _ in range(10)]
                + [items[-1]],
                max_tokens=64,
                # Fork: chat compaction is opt-in
                context_management=[
                    {"type": "compaction", "compact_threshold": 100000}
                ],
            )
        else:
            options.update(
                input=items,
                max_output_tokens=64,
                context_management=[
                    {"type": "compaction", "compact_threshold": 100000}
                ],
            )
        response = _post(client, api, **options)
        assert response.status_code == 200, response.text
        calls = mocked.generate.call_args_list
        summary_calls = [
            call
            for call in calls
            if json.loads(call.kwargs["prompt"])[-1]["content"]
            == compaction.SUMMARY_INSTRUCTION
        ]
        assert 1 < len(summary_calls) <= compaction.MAX_SUMMARY_PASSES
        for call in summary_calls:
            assert len(call.kwargs["prompt"]) // 4 + call.kwargs["max_tokens"] <= 2048
            messages = json.loads(call.kwargs["prompt"])
            ids = {tc["id"] for m in messages for tc in m.get("tool_calls", [])}
            assert ids == {
                m["tool_call_id"] for m in messages if m.get("role") == "tool"
            }
        assert items == original
        if mode != "chat":
            final = _completed_response(response) if stream else response.json()
            assert final["output"][0]["type"] == "compaction"
            assert mocked.count(input=[final["output"][0]])["input_tokens"] <= 2048 - 64
            if mode == "trigger":
                assert final["usage"]["input_tokens"] == 8 * len(summary_calls)
                assert final["usage"]["output_tokens"] == 4 * len(summary_calls)
            if stream:
                kinds = [event["type"] for event in _data(response)]
                assert (
                    kinds.count("mlx.compaction.started")
                    == kinds.count("mlx.compaction.completed")
                    == 1
                )

    def test_standalone_compaction_still_requires_input_to_fit(
        self, mocked, monkeypatch
    ):
        monkeypatch.setattr(server.runtime.config, "max_kv_size", 2048)
        result = mocked.compact(input=_compaction_history(), status=400)
        assert "input must fit" in result["detail"]
        mocked.generate.assert_not_called()

    def test_later_summary_failure_never_publishes_partial_compaction(
        self, mocked, client, monkeypatch
    ):
        monkeypatch.setattr(server.runtime.config, "max_kv_size", 2048)
        mocked.generate.side_effect = [
            _result("Earlier requirements preserved."),
            _result(""),
        ]
        items = [_compaction_message("old evidence " * 180) for _ in range(10)] + [
            _compaction_message("Continue")
        ]
        original = copy.deepcopy(items)
        events = _data(
            _post(
                client,
                "responses",
                input=items,
                stream=True,
                max_output_tokens=64,
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )
        assert mocked.generate.call_count == 2
        assert events[-1]["type"] == "response.failed"
        assert events[-1]["response"]["error"]["code"] == "server_error"
        assert [event["type"] for event in events].count("mlx.compaction.started") == 1
        assert not any(
            event["type"]
            in (
                "response.output_item.added",
                "response.completed",
                "mlx.compaction.completed",
            )
            for event in events
        )
        assert not server.response_store and items == original

    @pytest.mark.parametrize(
        "failure", ["oversized-exchange", "nonreducing-summary", "pass-limit"]
    )
    def test_bounded_summary_recovery_stops_without_mutating_input(
        self, monkeypatch, failure
    ):
        items = [_compaction_message("old evidence " * 200) for _ in range(4)]
        original = copy.deepcopy(items)
        monkeypatch.setattr(compaction, "MAX_SUMMARY_PASSES", 2)

        async def count(items):
            return len(json.dumps(items))

        async def fits(items):
            return (
                failure != "oversized-exchange"
                and sum(x.get("role") == "user" for x in items) <= 1
            )

        summarize = AsyncMock(
            return_value=(
                "huge " * 1000 if failure == "nonreducing-summary" else "handoff",
                server.OpenAIUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )
        )
        with pytest.raises(compaction.ContextBudgetError):
            asyncio.run(
                compaction._summarize_bounded(
                    items, fits=fits, summarize=summarize, count=count
                )
            )
        assert (
            summarize.await_count
            == {"oversized-exchange": 0, "nonreducing-summary": 1, "pass-limit": 2}[
                failure
            ]
        )
        assert items == original

    @pytest.mark.parametrize("options", [{}, {"context_management": []}])
    def test_chat_within_budget_does_not_compact(
        self, mocked, client, monkeypatch, options
    ):
        messages = [_compaction_message("earlier notes " * 120) for _ in range(4)]
        messages.append(_compaction_message("Continue"))
        before = mocked.count(input=messages)["input_tokens"]
        monkeypatch.setattr(server.runtime.config, "max_kv_size", before + 64)
        response = _post(client, messages=messages, max_tokens=64, **options)
        assert response.status_code == 200
        assert mocked.generate.call_count == 1
        assert "Conversation handoff" not in mocked.generate.call_args.kwargs["prompt"]

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        "outcome", ["recover", "disabled", "summary-fails", "protected-too-large"]
    )
    def test_chat_overflow_preserves_tools_and_only_generates_after_recovery(
        self, mocked, client, monkeypatch, stream, outcome
    ):
        monkeypatch.setattr(server.runtime.config, "max_kv_size", 2048)
        history = [_msg("Preserve requirements.", "system")]
        history += [_msg("older evidence " * 150) for _ in range(5)]
        recent = [
            _msg("Inspect this file."),
            _msg(
                "",
                "assistant",
                tool_calls=[
                    {
                        "id": "latest",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            ),
            _msg(
                "current evidence " * (600 if outcome == "protected-too-large" else 30),
                "tool",
                tool_call_id="latest",
            ),
        ]
        original = copy.deepcopy(history + recent)
        if outcome == "summary-fails":
            mocked.generate.return_value = _result("")
        # Fork: chat compaction is opt-in, so the recovering outcomes ask for it.
        options = {
            "context_management": (
                []
                if outcome == "disabled"
                else [{"type": "compaction", "compact_threshold": 100000}]
            )
        }
        response = _post(
            client,
            messages=original,
            tools=[_tool()],
            max_tokens=64,
            stream=stream,
            **options,
        )
        assert original == history + recent
        if outcome in ("summary-fails", "protected-too-large"):
            assert response.status_code == (502 if outcome == "summary-fails" else 400)
            assert response.headers["content-type"] == "application/json"
            assert mocked.generate.call_count == (
                1 if outcome == "summary-fails" else 0
            )
            mocked.stream.assert_not_called()
            assert not server.response_store
            return
        assert response.status_code == 200, response.text
        answer = mocked.stream.call_args if stream else mocked.generate.call_args
        messages = json.loads(answer.kwargs["prompt"])
        assert messages[0] == original[0]
        for expected, actual in zip(recent, messages[-3:]):
            for key in ("role", "content", "tool_call_id"):
                assert actual.get(key) == expected.get(key)
        call = messages[-2]["tool_calls"][0]
        assert call["id"] == messages[-1]["tool_call_id"] == "latest"
        assert call["function"] == {"name": "get_weather", "arguments": {}}
        if outcome == "disabled":
            assert mocked.generate.call_count == (0 if stream else 1)
            assert len(answer.kwargs["prompt"]) // 4 + 64 > 2048
        else:
            summaries = mocked.generate.call_args_list
            if not stream:
                summaries = summaries[:-1]
            assert 1 <= len(summaries) <= compaction.MAX_SUMMARY_PASSES
            for call in summaries:
                assert (
                    json.loads(call.kwargs["prompt"])[-1]["content"]
                    == compaction.SUMMARY_INSTRUCTION
                )
                assert (
                    len(call.kwargs["prompt"]) // 4 + call.kwargs["max_tokens"] <= 2048
                )
            assert len(answer.kwargs["prompt"]) // 4 + 64 <= 2048
            assert "Conversation handoff" in answer.kwargs["prompt"]

    @pytest.mark.parametrize(
        "choice",
        [
            None,
            "none",
            "required",
            {"type": "function", "function": {"name": "get_weather"}},
        ],
    )
    @pytest.mark.parametrize("explicit", [False, True])
    def test_chat_compaction_counts_the_generation_prompt(
        self, mocked, client, choice, explicit
    ):
        messages = [
            _msg(
                [
                    {"type": "text", "text": "First."},
                    {"type": "text", "text": "Second."},
                ],
                "developer",
            ),
            _msg("", "assistant", reasoning_content="Earlier reasoning."),
            _msg(
                [
                    {"type": "text", "text": "Inspect"},
                    _input_image("data:image/png;base64,example"),
                ]
            ),
        ]
        response = _post(
            client,
            messages=messages,
            tools=[_tool()],
            tool_choice=choice,
            **(
                {
                    "context_management": [
                        {"type": "compaction", "compact_threshold": 100000}
                    ]
                }
                if explicit
                else {}
            ),
        )
        assert response.status_code == 200, response.text
        counted, generated = (
            mocked.template.call_args_list[0],
            mocked.template.call_args_list[-1],
        )
        assert counted.args == generated.args
        for field in ("tools", "tool_choice", "num_images"):
            assert counted.kwargs.get(field) == generated.kwargs.get(field)

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("short", [False, True])
    def test_codex_compaction_trigger_returns_only_one_capsule(
        self, mocked, client, stream, short
    ):
        items = [_compaction_message("hello")] if short else _compaction_history()
        response = _post(
            client,
            "responses",
            input=items + [{"type": "compaction_trigger"}],
            stream=stream,
        )
        assert response.status_code == 200, response.text
        if stream:
            events = _data(response)
            data = _completed_response(response)
            assert not any(x["type"] == "response.output_text.delta" for x in events)
            if short:
                assert not any(x["type"].startswith("mlx.compaction.") for x in events)
        else:
            data = response.json()
        assert [x["type"] for x in data["output"]] == ["compaction"]
        assert data["output_text"] == ""
        assert mocked.generate.call_count == (0 if short else 1)
        mocked.request(
            input=data["output"] + [_compaction_message("Continue")],
        )
        assert "compaction_trigger" not in mocked.generate.call_args.kwargs["prompt"]

    def test_fixed_instructions_are_excluded_from_reduction_target(self, mocked):
        inputs = _compaction_history()
        options = dict(input=inputs, instructions="Static instruction. " * 3000)
        before = mocked.count(**options)["input_tokens"]
        response = mocked.compact(keep_tokens=0, **options)
        after = mocked.count(
            input=response["output"],
            instructions=options["instructions"],
        )["input_tokens"]
        assert before * 0.6 < after < before

    @pytest.mark.parametrize(
        "threshold, output_tokens, status",
        [
            pytest.param(1, 32768, 400, id="budget-above-threshold"),
            pytest.param(100000, 32768, 400, id="budget-below-threshold"),
            pytest.param(0, 64, 422, id="zero-threshold"),
            pytest.param(-1, 64, 422, id="negative-threshold"),
            pytest.param("bad", 64, 422, id="nonnumeric-threshold"),
        ],
    )
    def test_invalid_compaction_budget_rejected_before_generation(
        self, mocked, threshold, output_tokens, status
    ):
        mocked.request(
            input=_compaction_history(),
            max_output_tokens=output_tokens,
            context_management=[{"type": "compaction", "compact_threshold": threshold}],
            status=status,
        )
        mocked.generate.assert_not_called()

    @pytest.mark.parametrize(
        "api, item",
        [
            pytest.param(
                "responses", {"type": "compaction_trigger"}, id="nonterminal-trigger"
            ),
            pytest.param(
                "/responses/compact",
                {"type": "item_reference", "id": "missing"},
                id="item-reference",
            ),
            pytest.param(
                "/responses/compact",
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_audio"}],
                },
                id="input-audio",
            ),
        ],
    )
    def test_invalid_compaction_input_rejected_before_generation(
        self, mocked, api, item
    ):
        mocked.request(
            api, input=[copy.deepcopy(item)] + _compaction_history(), status=400
        )
        mocked.generate.assert_not_called()

    def test_tail_with_image_and_pending_tool_call_is_retained(self):
        instructions = _compaction_message(
            "Keep the original constraints.", "developer"
        )
        tail = [
            _msg(
                [
                    {"type": "input_text", "text": "Inspect this"},
                    _input_image("data:image/png;base64,example"),
                ],
                type="message",
            ),
            _function_result("", name="read_file", call_id="pending")[0],
            _compaction_message("A correction while the tool is running"),
        ]
        original = [
            instructions,
            _compaction_message("old " * 1000),
            _compaction_message("done", "assistant"),
            *tail,
        ]

        async def count(items):
            return len(json.dumps(items))

        async def summarize(items):
            assert items == original[:3]
            return "Prior work completed.", None

        result = asyncio.run(
            compaction.compact(
                original,
                count=count,
                summarize=summarize,
                keep_tokens=0,
                target_tokens=2000,
            )
        )
        assert result.changed and result.items[0] == instructions
        assert result.items[-len(tail) :] == tail
        assert _resolve_context([_seal_context(result.items)]) == result.items

    @pytest.mark.parametrize(
        "config",
        [
            NS(text_config=NS(max_position_embeddings=8192)),
            NS(text_config={"max_position_embeddings": 8192}),
            {"text_config": {"max_position_embeddings": 8192}},
        ],
    )
    def test_context_limit_respects_model_and_server(self, config, monkeypatch):
        assert compaction._context_limit(config) == 8192
        monkeypatch.setattr(server.runtime.config, "max_kv_size", 4096)
        assert compaction._context_limit(config) == 4096

    @pytest.mark.parametrize("stream", [False, True])
    def test_auto_compaction_replay_and_stream_indices(self, mocked, stream):
        final = mocked.request(
            input=_compaction_history(),
            max_output_tokens=64,
            context_management=[{"type": "compaction", "compact_threshold": 1000}],
            stream=stream,
        )
        assert final["output"][0]["type"] == "compaction"
        resolved = _resolve_context(_compaction_history() + final["output"])
        assert "old log entry" not in json.dumps(resolved)
        mocked.request(input="next", previous_response_id=final["id"])
        assert "old log entry" not in mocked.generate.call_args.kwargs["prompt"]

    def test_auto_compaction_tool_stream_preserves_item_indices(self, mocked):
        text = "<think>Check first.</think>" + _WEATHER_CALL
        mocked.stream.side_effect = lambda *a, **kw: iter(
            [_result(text, finish_reason="stop")]
        )
        with (
            patch.object(
                server, "_infer_tool_parser_from_processor", return_value="demo"
            ),
            patch.object(server, "load_tool_module", return_value=_JSON_TOOLS),
        ):
            final = mocked.request(
                input=_compaction_history(),
                stream=True,
                tools=[_tool()],
                max_output_tokens=64,
                context_management=[{"type": "compaction", "compact_threshold": 1000}],
            )
        assert [item["type"] for item in final["output"]] == [
            "compaction",
            "reasoning",
            "function_call",
        ]

    @pytest.mark.parametrize("trigger", [False, True])
    def test_compaction_progress_arrives_before_summary_finishes(self, mocked, trigger):
        release, running = Event(), Event()
        summary = mocked.generate.return_value

        def generate(**kwargs):
            running.set()
            assert release.wait(
                5
            ), "Summary released only after start event is received"
            return summary

        mocked.generate.side_effect = generate
        payload = dict(
            model="demo", input=_compaction_history(), stream=True, max_output_tokens=64
        )
        if trigger:
            payload["input"].append({"type": "compaction_trigger"})
        else:
            payload["context_management"] = [
                {"type": "compaction", "compact_threshold": 1000}
            ]
        before = mocked.count(input=_compaction_history())["input_tokens"]

        async def consume():
            response = await openai.responses_endpoint(
                NS(json=AsyncMock(return_value=payload), headers={})
            )
            events = []
            try:
                async for chunk in response.body_iterator:
                    events.extend(data for _, data in _sse_events_all(chunk))
                    if events[-1]["type"] == "mlx.compaction.started":
                        assert await asyncio.to_thread(running.wait, 2)
                        assert events[-1]["input_tokens"] == before
                        assert not release.is_set()
                        release.set()
            finally:
                release.set()
                await response.body_iterator.aclose()
            return events

        events = asyncio.run(consume())
        kinds = [event["type"] for event in events]
        assert kinds[:3] == [
            "response.created",
            "response.in_progress",
            "mlx.compaction.started",
        ]
        assert kinds.index("mlx.compaction.completed") < kinds.index(
            "response.output_item.added"
        )
        final = events[-1]["response"]
        progress = [
            event for event in events if event["type"].startswith("mlx.compaction.")
        ]
        assert len(progress) == 2 and all(
            event["response_id"] == final["id"] for event in progress
        )
        after = mocked.count(input=[final["output"][0]])["input_tokens"]
        assert progress[1]["input_tokens_before"] == before
        assert progress[1]["input_tokens_after"] == after < before

    @pytest.mark.parametrize("threshold", [1, 1000])
    def test_no_summary_emits_no_progress(self, mocked, client, threshold):
        response = _post(
            client,
            "responses",
            input="hello",
            stream=True,
            context_management=[{"type": "compaction", "compact_threshold": threshold}],
        )
        assert not any(e["type"].startswith("mlx.compaction.") for e in _data(response))
        _completed_response(response)
        mocked.generate.assert_not_called()

    @pytest.mark.parametrize("trigger", [False, True])
    @pytest.mark.parametrize(
        "text, tokens, code",
        [("", 4, 502), ("partial", 1024, 502), ("huge " * 10000, 4, 400)],
    )
    def test_failed_streamed_summary_never_completes(
        self, mocked, client, trigger, text, tokens, code
    ):
        mocked.generate.return_value = _result(text, generation_tokens=tokens)
        items = _compaction_history()
        options = {}
        if trigger:
            items.append({"type": "compaction_trigger"})
        else:
            options["context_management"] = [
                {"type": "compaction", "compact_threshold": 1000}
            ]
        events = _data(_post(client, "responses", input=items, stream=True, **options))
        assert [event["type"] for event in events] == [
            "response.created",
            "response.in_progress",
            "mlx.compaction.started",
            "response.failed",
        ]
        failed = events[-1]["response"]
        assert failed["id"] == events[0]["response"]["id"]
        assert failed["status"] == "failed" and not failed["output"]
        assert failed["error"]["code"] == (
            "context_length_exceeded" if code == 400 else "server_error"
        )
        assert "Compaction" in failed["error"]["message"]
        assert not server.response_store
        mocked.stream.assert_not_called()

    @pytest.mark.parametrize("trigger", [False, True])
    def test_closing_compaction_stream_cancels_summary_worker(
        self, mocked, monkeypatch, trigger
    ):
        queue, running, cancelled = Queue(), Event(), Event()

        def cancel(uid):
            cancelled.set()
            queue.put(None)

        iterator = generation._TokenIterator(queue, 1, cancel, 5)
        worker = _streaming([])
        worker._cpu_preprocess = lambda prompt, *args: {
            "input_ids": np.zeros((1, len(prompt) // 4), dtype=np.int32)
        }

        def generate(**kwargs):
            running.set()
            return NS(prompt_tokens=100), iterator

        worker.generate.side_effect = generate
        monkeypatch.setattr(server.runtime, "response_generator", worker)
        payload = dict(model="demo", input=_compaction_history(), stream=True)
        if trigger:
            payload["input"].append({"type": "compaction_trigger"})
        else:
            payload["context_management"] = [
                {"type": "compaction", "compact_threshold": 1000}
            ]

        async def disconnect():
            response = await openai.responses_endpoint(
                NS(json=AsyncMock(return_value=payload), headers={})
            )
            try:
                async for chunk in response.body_iterator:
                    event = next(_sse_events_all(chunk))[1]
                    if event["type"] == "mlx.compaction.started":
                        assert await asyncio.to_thread(running.wait, 2)
                        break
            finally:
                await asyncio.wait_for(response.body_iterator.aclose(), 2)
                assert cancelled.is_set()

        asyncio.run(disconnect())
        assert not server.response_store

    def test_compaction_progress_is_isolated_between_requests(self, mocked):
        async def consume(size):
            items = _compaction_history()
            items[-1]["content"] += " extra" * size
            payload = dict(
                model="demo",
                input=items,
                stream=True,
                context_management=[{"type": "compaction", "compact_threshold": 1000}],
            )
            response = await openai.responses_endpoint(
                NS(json=AsyncMock(return_value=payload), headers={})
            )
            return [
                data
                async for chunk in response.body_iterator
                for _, data in _sse_events_all(chunk)
            ]

        async def concurrent():
            return await asyncio.gather(consume(1), consume(100))

        first, second = asyncio.run(concurrent())
        assert first[-1]["response"]["id"] != second[-1]["response"]["id"]
        assert first[2]["input_tokens"] < second[2]["input_tokens"]
        for events in (first, second):
            response_id = events[-1]["response"]["id"]
            assert all(
                e["response_id"] == response_id
                for e in events
                if e["type"].startswith("mlx.compaction.")
            )

    def test_streamed_compaction_budget_error_precedes_start(self, mocked, client):
        events = _data(
            _post(
                client,
                "responses",
                input=_compaction_history(),
                stream=True,
                max_output_tokens=32768,
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )
        assert [event["type"] for event in events] == [
            "response.created",
            "response.in_progress",
            "response.failed",
        ]
        assert events[-1]["response"]["error"]["code"] == "invalid_prompt"
        mocked.generate.assert_not_called()
        mocked.stream.assert_not_called()

    def test_stored_compaction_survives_parent_eviction(self, mocked):
        first = mocked.request(input="original")
        capsule = _seal_context([_compaction_message("compacted")])
        second = mocked.request(
            "responses", input=[capsule], previous_response_id=first["id"]
        )
        server.response_store.pop(first["id"])
        mocked.request("responses", input="next", previous_response_id=second["id"])

    @pytest.mark.parametrize(
        "finish_reason, status",
        [
            pytest.param("stop", 200, id="success"),
            pytest.param(None, 400, id="iterator-error"),
        ],
    )
    def test_summary_uses_generation_worker_and_closes_iterator(
        self, mocked, monkeypatch, finish_reason, status
    ):

        def chunks():
            yield _token("Goal: ORCHID. Port: 7319.", finish_reason=finish_reason)
            raise server.PromptTooLongError("summary worker failed")

        iterator = MagicMock()
        iterator.__iter__.return_value = chunks()
        generator = _streaming([])
        context, _ = generator.generate.return_value
        generator.generate.return_value = (context, iterator)
        generator._cpu_preprocess = lambda prompt, images, audio: {
            "input_ids": np.zeros((1, max(1, len(prompt) // 4)), dtype=np.int32)
        }
        monkeypatch.setattr(server.runtime, "response_generator", generator)
        response = mocked.compact(
            input=_compaction_history(),
            keep_tokens=0,
            status=status,
        )
        if status == 400:
            assert response["detail"] == "summary worker failed"
        generator.generate.assert_called_once()
        mocked.generate.assert_not_called()
        iterator.close.assert_called_once_with()

    @pytest.fixture
    def real_server(self, tmp_path):
        model = os.environ.get("MLX_VLM_COMPACTION_TEST_MODEL")
        if not model:
            pytest.skip("Set MLX_VLM_COMPACTION_TEST_MODEL for real inference")
        from openai import OpenAI

        @contextmanager
        def start():
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = dict(
                os.environ,
                APC_ENABLED="1",
                APC_DISK_ENABLED="0",
                APC_NUM_BLOCKS="2048",
                MLX_VLM_COMPACTION_KEY_FILE=str(tmp_path / "compaction.key"),
            )
            env.pop("MLX_VLM_SERVER_API_KEY", None)
            log_path = tmp_path / "server.log"
            with log_path.open("a") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "mlx_vlm.server",
                        "--model",
                        model,
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    url = f"http://127.0.0.1:{port}"
                    headers = {"X-APC-Tenant": "compaction-test"}
                    with (
                        httpx.Client(
                            base_url=url, timeout=180, headers=headers
                        ) as client,
                        OpenAI(
                            base_url=url + "/v1",
                            api_key="test",
                            default_headers=headers,
                        ) as sdk,
                    ):
                        for _ in range(240):
                            assert process.poll() is None, log_path.read_text()
                            try:
                                if client.get("/health", timeout=1).is_success:
                                    break
                            except httpx.TransportError:
                                pass
                            time.sleep(0.25)
                        else:
                            pytest.fail(
                                f"Model server did not become ready:\n{log_path.read_text()}"
                            )
                        yield NS(http=client, sdk=sdk, model=model)
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

        return start

    @pytest.fixture
    def real_client(self, real_server):
        with real_server() as client:
            yield client

    @pytest.fixture
    def real_history(self):
        items = [
            _msg(
                "Follow the user's requirements. Answer factual questions concisely.",
                "system",
            ),
            _msg(
                "We are working on project ORCHID. The deployment port is 7319. Never edit secrets.env. Remember all three facts."
            ),
        ]
        for index in range(8):
            items.extend(
                [
                    _msg(
                        f"Inspection {index} finished. "
                        + "The routine build log contains no new decisions. " * 70,
                        "assistant",
                    ),
                    _msg(
                        f"Continue inspection {index + 1}, keeping the original requirements."
                    ),
                ]
            )
        return items + [
            _msg(
                "What is the project name, deployment port, and file you must never edit? Give all three."
            )
        ]

    def test_real_compaction_replay_and_apc(self, real_client, real_history):
        client = real_client
        before = _real_post(client, "/v1/responses/input_tokens", input=real_history)
        baseline = _real_answer(client, real_history)
        compacted = _real_compact(client, real_history)
        output = compacted.model_dump()["output"]
        assert output[0]["type"] == "compaction"
        after = _real_post(client, "/v1/responses/input_tokens", input=output)
        assert after["input_tokens"] < before["input_tokens"] * 0.6
        cold = _real_answer(client, output)
        warm = _real_answer(client, output)
        assert (
            warm["usage"]["input_tokens_details"]["cached_tokens"]
            > cold["usage"]["input_tokens_details"]["cached_tokens"]
        )
        for response in (baseline, cold, warm):
            _assert_recalled(response["output_text"])
        replay = client.sdk.responses.create(
            model=client.model,
            input=compacted.output,
            max_output_tokens=96,
            temperature=0,
            store=False,
            extra_body={"enable_thinking": False},
        )
        _assert_recalled(replay.output_text)
        # Full-history and capsule-only replay must render identical tokenized inputs.
        assert (
            _real_post(
                client, "/v1/responses/input_tokens", input=real_history + output
            )
            == after
        )
        client.http.post("/v1/cache/reset").raise_for_status()
        reset = _real_answer(client, output)
        assert reset["usage"]["input_tokens_details"]["cached_tokens"] == 0
        _assert_recalled(reset["output_text"])

    def test_real_automatic_compaction_stream(self, real_client, real_history):
        payload = dict(
            model=real_client.model,
            input=real_history,
            temperature=0,
            enable_thinking=False,
            max_output_tokens=96,
            store=False,
            stream=True,
            context_management=[{"type": "compaction", "compact_threshold": 2000}],
        )
        started = time.perf_counter()
        lines, progress = [], []
        with real_client.http.stream("POST", "/v1/responses", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                lines.append(line)
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    if event.get("type", "").startswith("mlx.compaction."):
                        progress.append((time.perf_counter() - started, event))
        completed = _completed_response(
            httpx.Response(200, text="\n".join(lines) + "\n")
        )
        assert completed["output"][0]["type"] == "compaction"
        _assert_recalled(completed["output_text"])
        assert [event["type"] for _, event in progress] == [
            "mlx.compaction.started",
            "mlx.compaction.completed",
        ]
        assert progress[0][0] < progress[1][0]
        sdk_payload = {
            key: value for key, value in payload.items() if key != "enable_thinking"
        }
        with real_client.sdk.responses.create(
            **sdk_payload, extra_body={"enable_thinking": False}
        ) as stream:
            sdk_events = list(stream)
        assert sdk_events[-1].type == "response.completed"
        assert any(event.type == "mlx.compaction.started" for event in sdk_events)
        counts = progress[1][1]
        assert counts["input_tokens_before"] == progress[0][1]["input_tokens"]
        assert counts["input_tokens_after"] == completed["usage"]["input_tokens"]
        print(
            f"Compaction started at {progress[0][0]:.3f}s, completed at {progress[1][0]:.3f}s; "
            f"{counts['input_tokens_before']} -> {counts['input_tokens_after']} input tokens"
        )

    def test_real_repeated_compaction_keeps_corrections(
        self, real_client, real_history
    ):
        output = _real_compact(real_client, real_history).model_dump()["output"]
        corrected = output + [
            _msg("Confirmed.", "assistant"),
            _msg(
                "Correction: deployment port is now 8421. Preserve the other requirements."
            ),
            _msg(
                "Port updated to 8421. " + "Routine verification passed. " * 800,
                "assistant",
            ),
            real_history[-1],
        ]
        again = _real_post(
            real_client,
            "/v1/responses/compact",
            input=corrected,
            keep_tokens=0,
            max_output_tokens=768,
        )
        _assert_recalled(
            _real_answer(real_client, again["output"])["output_text"], port="8421"
        )

    @pytest.mark.parametrize("api", ["chat", "messages"])
    def test_real_client_summary_cache(self, real_client, real_history, api):
        items = [
            real_history[0],
            _msg(
                "Prior conversation summary: Project ORCHID; deployment port 7319; never edit secrets.env."
            ),
            _msg("I will preserve those requirements.", "assistant"),
            real_history[-1],
        ]
        body = {"messages": items, "max_tokens": 96}
        if api == "messages":
            body.update(system=items[0]["content"], messages=items[1:])
        first, warm = [_real_post(real_client, api, **body) for _ in range(2)]
        if api == "messages":
            text = "".join(part.get("text", "") for part in warm["content"])
            counts = [
                result["usage"].get("cache_read_input_tokens", 0)
                for result in (first, warm)
            ]
        else:
            text = warm["choices"][0]["message"]["content"]
            counts = [
                result["usage"]["prompt_tokens_details"]["cached_tokens"]
                for result in (first, warm)
            ]
        _assert_recalled(text)
        assert counts[1] > counts[0]

    def test_real_compaction_survives_restart(self, real_server, real_history):
        with real_server() as client:
            output = _real_compact(client, real_history).model_dump()["output"]
        # Restart with the same key, without a response registry or populated APC pool.
        with real_server() as client:
            restored = _real_answer(client, output)
            assert restored["usage"]["input_tokens_details"]["cached_tokens"] == 0
            _assert_recalled(restored["output_text"])


@pytest.mark.parametrize("kind", ["choice", "score", "bool", "multi_label"])
def test_decisions_endpoint_uses_shared_prediction(client, kind):
    criteria = None if kind == "bool" else ["low", "high"]
    questions = {"result": {"type": kind, "criteria": criteria}}
    result = {
        "answers": {"result": {"type": kind, "value": "low"}},
        "usage": {"input_tokens": 7},
    }
    model = NS(decision_types=(kind,), predict=MagicMock(return_value=result))
    processor = object()
    with patch.object(
        server, "get_cached_model", return_value=(model, processor, {})
    ) as load:
        response = client.post(
            "/v1/decisions",
            json={"model": "decision", "state": "text", "questions": questions},
        )
    assert response.status_code == 200
    assert response.json() == {**result, "model": "decision"}
    load.assert_called_once_with("decision", model_kind="decision")
    model.predict.assert_called_once_with(processor, "text", questions)


def test_decisions_rejects_unsupported_type(client):
    model = NS(decision_types=("choice",), predict=MagicMock())
    with patch.object(server, "get_cached_model", return_value=(model, None, {})):
        response = client.post(
            "/v1/decisions",
            json={
                "model": "decision",
                "state": "text",
                "questions": {"x": {"type": "bool"}},
            },
        )
    assert response.status_code == 400
    model.predict.assert_not_called()


def test_decisions_auth_and_schema_before_loading(client, monkeypatch):
    monkeypatch.setenv("MLX_VLM_SERVER_API_KEY", "secret-token")
    with patch.object(server, "get_cached_model") as load:
        assert client.post("/v1/decisions", json={}).status_code == 401
        assert (
            client.post(
                "/v1/decisions",
                json={},
                headers={"Authorization": "Bearer secret-token"},
            ).status_code
            == 422
        )
    load.assert_not_called()


def test_decision_cache_reuses_standard_loader_and_preserves_text(monkeypatch):
    _reset_runtime(monkeypatch)
    registry = server.runtime.model_cache
    text_cache = {"model_path": "text", "model_kind": "text_generation"}
    registry.set("text_generation", text_cache)
    model, processor = NS(config={}, decision_types=("choice",)), object()
    with (
        patch("mlx_vlm.utils.load", return_value=(model, processor)) as load,
        patch.object(server._app_module, "ResponseGenerator") as generator,
    ):
        for _ in range(2):
            assert server.get_cached_model("decision", model_kind="decision") == (
                model,
                processor,
                {},
            )
    load.assert_called_once_with("decision")
    generator.assert_not_called()
    assert registry.for_kind("text_generation") is text_cache
    assert registry.for_kind("decision")["model"] is model


def test_decision_cache_rejects_non_decision_model(monkeypatch):
    _reset_runtime(monkeypatch)
    with patch("mlx_vlm.utils.load", return_value=(NS(config={}), None)):
        with pytest.raises(server.HTTPException) as error:
            server.get_cached_model("text", model_kind="decision")
    assert error.value.status_code == 400
    assert not server.runtime.model_cache.for_kind("decision")


@pytest.mark.parametrize("status", [404, 500])
def test_decisions_preserves_loader_errors(client, status):
    with patch.object(
        server,
        "get_cached_model",
        side_effect=server.HTTPException(status_code=status, detail="load failed"),
    ):
        response = client.post(
            "/v1/decisions",
            json={
                "model": "missing",
                "state": "text",
                "questions": {"x": {"type": "bool"}},
            },
        )
    assert response.status_code == status
    assert response.json()["detail"] == "load failed"


@pytest.mark.parametrize("preloaded", [False, True])
def test_decisions_default_model(client, monkeypatch, preloaded):
    _reset_runtime(monkeypatch)
    monkeypatch.delenv("MLX_VLM_PRELOAD_DECISION_MODEL", raising=False)
    if preloaded:
        server.runtime.model_cache.set("decision", {"model_path": "preloaded"})
    model = NS(
        decision_types=("bool",), predict=MagicMock(return_value={"answers": {}})
    )
    with patch.object(
        server, "get_cached_model", return_value=(model, None, {})
    ) as load:
        response = client.post(
            "/v1/decisions",
            json={"state": "text", "questions": {"x": {"type": "bool"}}},
        )
    if preloaded:
        assert response.status_code == 200
        assert response.json()["model"] == "preloaded"
        load.assert_called_once_with("preloaded", model_kind="decision")
    else:
        assert response.status_code == 400
        load.assert_not_called()


def test_decisions_default_model_from_env(client, monkeypatch):
    _reset_runtime(monkeypatch)
    monkeypatch.setenv("MLX_VLM_PRELOAD_DECISION_MODEL", "env-decision")
    model = NS(
        decision_types=("bool",), predict=MagicMock(return_value={"answers": {}})
    )
    with patch.object(
        server, "get_cached_model", return_value=(model, None, {})
    ) as load:
        response = client.post(
            "/v1/decisions",
            json={"state": "text", "questions": {"x": {"type": "bool"}}},
        )
    assert response.status_code == 200
    assert response.json()["model"] == "env-decision"
    load.assert_called_once_with("env-decision", model_kind="decision")


@pytest.mark.parametrize("threshold", ["bad", None, [], {}, True, -0.1, 1.1])
def test_decisions_rejects_invalid_threshold_before_prediction(client, threshold):
    model = NS(decision_types=("multi_label",), predict=MagicMock())
    with patch.object(server, "get_cached_model", return_value=(model, None, {})):
        response = client.post(
            "/v1/decisions",
            json={
                "model": "decision",
                "state": "text",
                "questions": {
                    "tags": {
                        "type": "multi_label",
                        "criteria": ["refund"],
                        "threshold": threshold,
                    }
                },
            },
        )
    assert response.status_code == 400
    assert (
        response.json()["detail"] == "threshold must be a number between zero and one"
    )
    model.predict.assert_not_called()


@pytest.mark.parametrize("threshold", [0, 0.5, 1])
def test_decisions_preserves_valid_threshold(client, threshold):
    model = NS(
        decision_types=("multi_label",), predict=MagicMock(return_value={"answers": {}})
    )
    questions = {
        "tags": {"type": "multi_label", "criteria": ["refund"], "threshold": threshold}
    }
    with patch.object(server, "get_cached_model", return_value=(model, None, {})):
        response = client.post(
            "/v1/decisions",
            json={"model": "decision", "state": "text", "questions": questions},
        )
    assert response.status_code == 200
    model.predict.assert_called_once_with(None, "text", questions)


def test_decisions_generic_prediction_failure_returns_500(client):
    model = NS(
        decision_types=("bool",),
        predict=MagicMock(side_effect=RuntimeError("boom")),
    )
    with patch.object(server, "get_cached_model", return_value=(model, None, {})):
        response = client.post(
            "/v1/decisions",
            json={
                "model": "decision",
                "state": "text",
                "questions": {"x": {"type": "bool"}},
            },
        )
    assert response.status_code == 500
    assert response.json()["detail"] == "Decision prediction failed"
    model.predict.assert_called_once()
