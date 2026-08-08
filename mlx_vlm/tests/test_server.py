import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import mlx_vlm.server as server
import mlx_vlm.server.cli as server_cli
import mlx_vlm.server.generation as server_generation
import mlx_vlm.server.openai as server_openai
import mlx_vlm.speculative.utils as speculative_utils
from mlx_vlm.apc import hash_image_payload
from mlx_vlm.generate import GenerationResult
from mlx_vlm.generate.image import ImageGenerationResult
from mlx_vlm.tokenizer_utils import SPMStreamingDetokenizer, _ServerTokenStreamer


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


def test_chat_request_schema_requires_model():
    assert "model" in server.ChatRequest.model_json_schema()["required"]


def test_chat_request_schema_allows_one_or_two_resize_shape_values():
    resize_shape = server.ChatRequest.model_json_schema()["properties"]["resize_shape"]
    lengths = {
        (item["minItems"], item["maxItems"])
        for item in resize_shape["anyOf"]
        if item.get("type") == "array"
    }

    assert lengths == {(1, 1), (2, 2)}


def test_speculative_server_dispatches_mtp_batch_loop():
    assert (
        speculative_utils.get_speculative_rounds_batch("mtp")
        is speculative_utils._mtp_rounds_batch
    )


def test_speculative_server_samples_first_bonus_like_decode_step():
    seen = {}
    logits = mx.array(
        [
            [[1.0, 2.0, 3.0]],
            [[4.0, 1.0, 0.0]],
        ],
        dtype=mx.float32,
    )

    def sampler(logprobs):
        seen["shape"] = logprobs.shape
        seen["values"] = logprobs
        return mx.argmax(logprobs, axis=-1)

    tokens = server_generation._sample_last_token(logits, sampler)
    expected_logprobs = logits[:, -1, :] - mx.logsumexp(
        logits[:, -1, :], axis=-1, keepdims=True
    )
    mx.eval(tokens, seen["values"], expected_logprobs)

    assert seen["shape"] == (2, 3)
    assert tokens.tolist() == [2, 0]
    assert bool(mx.allclose(seen["values"], expected_logprobs).item())


def test_speculative_server_samples_first_bonus_with_positioned_sampler():
    seen = {}
    logits = mx.array(
        [
            [[1.0, 2.0, 3.0]],
            [[4.0, 1.0, 0.0]],
        ],
        dtype=mx.float32,
    )

    class Sampler:
        def __call__(self, logprobs):
            raise AssertionError("positioned sampler was not used")

        def sample_target(self, logprobs, *, row_ids, positions):
            seen["shape"] = logprobs.shape
            seen["row_ids"] = list(row_ids)
            seen["positions"] = list(positions)
            return mx.argmax(logprobs, axis=-1)

    tokens = server_generation._sample_last_token(
        logits,
        Sampler(),
        row_ids=[0, 0],
        positions=[0, 0],
    )
    mx.eval(tokens)

    assert seen == {
        "shape": (2, 3),
        "row_ids": [0, 0],
        "positions": [0, 0],
    }
    assert tokens.tolist() == [2, 0]


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


def test_positioned_target_sampler_honors_top_k():
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


def test_speculative_prompt_cache_uses_unbatched_cache_for_single_mtp(monkeypatch):
    lm = object()
    unbatched_cache = object()
    batched_cache = object()

    monkeypatch.setattr(
        speculative_utils.cache, "make_prompt_cache", lambda target: unbatched_cache
    )

    result = speculative_utils.make_speculative_prompt_cache(
        lm,
        draft_kind="mtp",
        batch_size=1,
        left_padding=[0],
        make_cache=lambda *args, **kwargs: batched_cache,
    )

    assert result is unbatched_cache


def test_speculative_prompt_cache_uses_batched_cache_for_batch_or_dflash(monkeypatch):
    lm = object()
    batched_cache = object()

    monkeypatch.setattr(
        speculative_utils.cache, "make_prompt_cache", lambda target: pytest.fail()
    )

    assert (
        speculative_utils.make_speculative_prompt_cache(
            lm,
            draft_kind="mtp",
            batch_size=2,
            left_padding=[0, 1],
            make_cache=lambda *args, **kwargs: batched_cache,
        )
        is batched_cache
    )
    assert (
        speculative_utils.make_speculative_prompt_cache(
            lm,
            draft_kind="dflash",
            batch_size=1,
            left_padding=[0],
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

    assert server.runtime.model_cache["cache_key"] == (
        "demo-model",
        "adapter-a",
        "auto",
    )
    assert server.runtime.model_cache["adapter_path"] == "adapter-a"


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
    gen.kv_bits = None
    gen.kv_group_size = server.DEFAULT_KV_GROUP_SIZE
    gen.kv_quant_scheme = server.DEFAULT_KV_QUANT_SCHEME
    gen.quantized_kv_start = server.DEFAULT_QUANTIZED_KV_START
    gen.top_logprobs_k = 0
    gen.apc_manager = None
    gen.tokenizer = None
    gen.requests = Queue()
    gen._stop = False
    gen._ready = Event()
    gen._load_error = None
    gen._cancelled = set()
    gen._cancel_lock = Lock()
    return gen


def test_server_demotes_incompatible_mtp_drafter_to_ar(monkeypatch):
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


def test_server_serves_ar_requests_after_drafter_mismatch(monkeypatch):
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
    gen._gpu_embed = lambda raw_inputs, images=None: (
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


def test_speculative_thread_exception_reaches_client_queue(monkeypatch):
    gen = _unstarted_response_generator()
    gen.model = SimpleNamespace(language_model=SimpleNamespace())
    gen.processor = SimpleNamespace()
    gen.draft_model = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[1, 2]), accept_lens=[]
    )
    gen.draft_kind = "dflash"
    gen.stop_tokens = set()

    rqueue = Queue()
    pending = [
        server_generation.QueuedGenerationRequest(
            rqueue=rqueue,
            raw_inputs={"input_ids": mx.array([[1]], dtype=mx.int32)},
            prompt_tokens=1,
            args=server.GenerationArguments(max_tokens=2),
        )
    ]
    calls = {"count": 0}

    def collect_pending_requests(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return pending, False
        return [], True

    error = RuntimeError("speculative prefill failed")
    gen._collect_pending_requests = collect_pending_requests
    gen._gpu_embed = MagicMock(side_effect=error)
    monkeypatch.setattr(
        "mlx_vlm.speculative.utils.speculative_prefill_kwargs",
        lambda *_args, **_kwargs: {},
    )

    gen._run_speculative()

    assert rqueue.get(timeout=1) is error
    assert rqueue.get(timeout=1) is None


def test_speculative_thread_exception_skips_broken_queues(monkeypatch):
    gen = _unstarted_response_generator()
    gen.model = SimpleNamespace(language_model=SimpleNamespace())
    gen.processor = SimpleNamespace()
    gen.draft_model = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[1, 2]), accept_lens=[]
    )
    gen.draft_kind = "dflash"
    gen.stop_tokens = set()

    class BrokenQueue:
        def put(self, item):
            raise RuntimeError("client went away")

    good_queue = Queue()
    pending = [
        server_generation.QueuedGenerationRequest(
            rqueue=BrokenQueue(),
            raw_inputs={"input_ids": mx.array([[1]], dtype=mx.int32)},
            prompt_tokens=1,
            args=server.GenerationArguments(max_tokens=2),
        ),
        server_generation.QueuedGenerationRequest(
            rqueue=good_queue,
            raw_inputs={"input_ids": mx.array([[1]], dtype=mx.int32)},
            prompt_tokens=1,
            args=server.GenerationArguments(max_tokens=2),
        ),
    ]
    calls = {"count": 0}

    def collect_pending_requests(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return pending, False
        return [], True

    error = RuntimeError("speculative prefill failed")
    gen._collect_pending_requests = collect_pending_requests
    gen._gpu_embed = MagicMock(side_effect=error)

    gen._run_speculative()

    assert good_queue.get(timeout=1) is error
    assert good_queue.get(timeout=1) is None


def test_speculative_thread_exception_clears_runtime_cache(monkeypatch):
    gen = _unstarted_response_generator()
    gen.model = SimpleNamespace(language_model=SimpleNamespace())
    gen.processor = SimpleNamespace()
    gen.draft_model = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[1, 2]), accept_lens=[]
    )
    gen.draft_kind = "dflash"
    gen.stop_tokens = set()
    rqueue = Queue()

    calls = {"clear_cache": 0, "collect": 0}
    collect_calls = {"count": 0}

    def collect_pending_requests(**_kwargs):
        collect_calls["count"] += 1
        if collect_calls["count"] > 1:
            return [], True
        return [
            server_generation.QueuedGenerationRequest(
                rqueue=rqueue,
                raw_inputs={"input_ids": mx.array([[1]], dtype=mx.int32)},
                prompt_tokens=1,
                args=server.GenerationArguments(max_tokens=2),
            )
        ], False

    gen._collect_pending_requests = collect_pending_requests
    gen._gpu_embed = MagicMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(
        server_generation.mx,
        "clear_cache",
        lambda: calls.__setitem__("clear_cache", calls["clear_cache"] + 1),
    )
    monkeypatch.setattr(
        server_generation.gc,
        "collect",
        lambda: calls.__setitem__("collect", calls["collect"] + 1),
    )

    gen._run_speculative()

    assert calls == {"clear_cache": 1, "collect": 1}


def test_models_endpoint_lists_single_file_safetensors_models(client, monkeypatch):
    def repo(repo_id, file_names):
        return SimpleNamespace(
            repo_id=repo_id,
            repo_type="model",
            last_modified=123.0,
            refs={
                "main": SimpleNamespace(
                    files=[
                        SimpleNamespace(file_path=SimpleNamespace(name=file_name))
                        for file_name in file_names
                    ]
                )
            },
        )

    monkeypatch.setattr(
        server,
        "scan_cache_dir",
        lambda: SimpleNamespace(
            repos=[
                repo(
                    "local/single-file-model",
                    ["config.json", "model.safetensors", "tokenizer_config.json"],
                ),
                repo(
                    "local/sharded-model",
                    [
                        "config.json",
                        "model.safetensors.index.json",
                        "tokenizer_config.json",
                    ],
                ),
                repo("missing/weights", ["config.json", "tokenizer_config.json"]),
            ]
        ),
    )

    response = client.get("/v1/models")

    assert response.status_code == 200
    ids = {model["id"] for model in response.json()["data"]}
    assert "local/single-file-model" in ids
    assert "local/sharded-model" in ids
    assert "missing/weights" not in ids


def test_models_endpoint_includes_loaded_local_model_without_hf_cache(
    client, monkeypatch
):
    monkeypatch.setattr(
        server,
        "scan_cache_dir",
        MagicMock(side_effect=server.CacheNotFound("missing cache", "/missing")),
    )
    monkeypatch.setitem(server.runtime.model_cache, "model_path", "/models/local-qwen")

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"] == [
        {
            "id": "/models/local-qwen",
            "object": "model",
            "created": response.json()["data"][0]["created"],
        }
    ]


def test_models_endpoint_deduplicates_loaded_model_from_hf_cache(client, monkeypatch):
    def repo(repo_id, file_names):
        return SimpleNamespace(
            repo_id=repo_id,
            repo_type="model",
            last_modified=123.0,
            refs={
                "main": SimpleNamespace(
                    files=[
                        SimpleNamespace(file_path=SimpleNamespace(name=file_name))
                        for file_name in file_names
                    ]
                )
            },
        )

    monkeypatch.setattr(
        server,
        "scan_cache_dir",
        lambda: SimpleNamespace(
            repos=[
                repo(
                    "local/sharded-model",
                    [
                        "config.json",
                        "model.safetensors.index.json",
                        "tokenizer_config.json",
                    ],
                ),
            ]
        ),
    )
    monkeypatch.setitem(server.runtime.model_cache, "model_path", "local/sharded-model")

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert [model["id"] for model in response.json()["data"]].count(
        "local/sharded-model"
    ) == 1


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


class _RecordingSpeculativeLM:
    def __init__(self, draft_kind):
        self.calls = []
        self.draft_kind = draft_kind
        self._position_ids = "stale"
        self._rope_deltas = "stale"

    def __call__(self, inputs, cache=None, **kwargs):
        self.calls.append({"inputs": inputs, "cache": cache, **kwargs})
        batch_size, seq_len = inputs.shape
        logits = mx.broadcast_to(
            mx.array([[[0.0, 1.0, 0.0, 0.0, 0.0]]], dtype=mx.float32),
            (batch_size, seq_len, 5),
        )
        hidden = mx.ones((batch_size, seq_len, 2), dtype=mx.float32)
        if self.draft_kind == "mtp":
            return SimpleNamespace(
                logits=logits,
                hidden_states=[hidden],
                shared_kv_states={"full_attention": ("k", "v")},
            )
        return SimpleNamespace(
            logits=logits,
            hidden_states=[hidden, hidden],
            shared_kv_states=None,
        )


def _run_speculative_prefill_once(monkeypatch, *, draft_kind, request_specs):
    lm = _RecordingSpeculativeLM(draft_kind)
    gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
    gen.model = SimpleNamespace(language_model=lm)
    gen.processor = SimpleNamespace()
    gen.draft_model = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[1, 2]), accept_lens=[]
    )
    gen.draft_kind = draft_kind
    gen.stop_tokens = {99}
    gen.requests = Queue()
    gen._stop = False
    gen._make_sampler = lambda args: None
    gen.tokenizer = SimpleNamespace(
        decode=lambda tokens: "".join(str(tok) for tok in tokens)
    )

    specs_iter = iter(request_specs)

    def fake_gpu_embed(raw_inputs, images=None):
        del raw_inputs, images
        spec = next(specs_iter)
        return spec["input_ids"], spec["gen_kwargs"]

    gen._gpu_embed = fake_gpu_embed

    monkeypatch.setattr(server_generation, "_make_cache", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        server_generation, "_get_draft_block_size_from_env", lambda: None
    )
    monkeypatch.setattr(
        server_generation, "get_speculative_batch_coalesce_s", lambda: 0.0
    )

    class _FakeDetokenizer:
        def __init__(self):
            self.last_segment = ""

        def reset(self):
            self.last_segment = ""

        def add_token(self, token):
            self.last_segment = str(token)

        def finalize(self):
            pass

    monkeypatch.setattr(
        server_generation,
        "make_streaming_detokenizer",
        lambda processor: _FakeDetokenizer(),
    )

    def fake_rounds(*args, **kwargs):
        del args
        gen.round_kwargs = kwargs
        gen._stop = True
        yield ([4] * int(kwargs["first_bonus"].shape[0]), None)

    monkeypatch.setattr(server_generation, "run_speculative_server_rounds", fake_rounds)

    args = server.GenerationArguments(max_tokens=2, temperature=0)
    for spec in request_specs:
        gen.requests.put(
            server_generation.QueuedGenerationRequest(
                rqueue=Queue(),
                raw_inputs={"input_ids": spec["input_ids"]},
                prompt_tokens=int(spec["input_ids"].shape[1]),
                args=args,
            )
        )

    gen._run_speculative()
    call = lm.calls[0]
    call["round_kwargs"] = gen.round_kwargs
    return call


def test_speculative_server_threads_greedy_flag_to_mtp_loop(monkeypatch):
    call = _run_speculative_prefill_once(
        monkeypatch,
        draft_kind="mtp",
        request_specs=[
            {
                "input_ids": mx.array([[11, 12, 13]], dtype=mx.int32),
                "gen_kwargs": {"inputs_embeds": mx.ones((1, 3, 4), dtype=mx.float32)},
            },
            {
                "input_ids": mx.array([[21, 22, 23]], dtype=mx.int32),
                "gen_kwargs": {"inputs_embeds": mx.ones((1, 3, 4), dtype=mx.float32)},
            },
        ],
    )

    assert call["round_kwargs"]["greedy_sampling"] is True


def test_speculative_server_prefill_threads_gemma4_per_layer_inputs(monkeypatch):
    call = _run_speculative_prefill_once(
        monkeypatch,
        draft_kind="mtp",
        request_specs=[
            {
                "input_ids": mx.array([[11, 12, 13]], dtype=mx.int32),
                "gen_kwargs": {
                    "inputs_embeds": mx.ones((1, 3, 4), dtype=mx.float32),
                    "per_layer_inputs": mx.array(
                        [[[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]], dtype=mx.float32
                    ),
                },
            },
            {
                "input_ids": mx.array([[21, 22]], dtype=mx.int32),
                "gen_kwargs": {
                    "inputs_embeds": mx.full((1, 2, 4), 7.0, dtype=mx.float32),
                    "per_layer_inputs": mx.array(
                        [[[4.0, 4.5], [5.0, 5.5]]], dtype=mx.float32
                    ),
                },
            },
        ],
    )

    assert call["return_hidden"] is True
    assert call["return_shared_kv"] is True
    assert call["per_layer_inputs"].shape == (2, 3, 2)
    assert call["per_layer_inputs"].tolist()[1][0] == [0.0, 0.0]
    assert call["inputs_embeds"].shape == (2, 3, 4)


def test_speculative_server_prefill_threads_qwen_dflash_prompt_kwargs(monkeypatch):
    call = _run_speculative_prefill_once(
        monkeypatch,
        draft_kind="dflash",
        request_specs=[
            {
                "input_ids": mx.array([[31, 32, 33]], dtype=mx.int32),
                "gen_kwargs": {
                    "inputs_embeds": mx.ones((1, 3, 4), dtype=mx.float32),
                    "image_grid_thw": mx.array([[1, 2, 3]], dtype=mx.int32),
                    "_apc_image_hash": 123,
                    "_apc_tenant": "tenant-a",
                },
            },
            {
                "input_ids": mx.array([[41, 42]], dtype=mx.int32),
                "gen_kwargs": {
                    "inputs_embeds": mx.full((1, 2, 4), 9.0, dtype=mx.float32),
                    "image_grid_thw": mx.array([[4, 5, 6]], dtype=mx.int32),
                },
            },
        ],
    )

    assert call["capture_layer_ids"] == [1, 2]
    assert call["image_grid_thw"].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert call["inputs_embeds"].shape == (2, 3, 4)
    assert call["inputs_embeds"].tolist()[1][0] == [0.0, 0.0, 0.0, 0.0]
    assert "_apc_image_hash" not in call
    assert "_apc_tenant" not in call


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
    assert "".join(delta.get("reasoning") or "" for delta in deltas) == ""
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

    assert "".join(delta.get("reasoning") or "" for delta in deltas) == (
        "Custom reasoning."
    )
    assert "".join(delta.get("content") or "" for delta in deltas) == ("Custom answer.")


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
    timed_chunks = [chunk for chunk in chunks if chunk.get("timings") is not None]
    assert len(timed_chunks) == 1
    timed_chunk = timed_chunks[0]
    assert timed_chunk["choices"] == []
    assert timed_chunk["timings"]["cache_n"] == 2
    assert timed_chunk["usage"]["prompt_tokens_details"]["cached_tokens"] == 2


def test_chat_completions_streaming_tool_calls_emit_usage_chunk(client, monkeypatch):
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
                        text=(
                            '<tool_call>{"name":"get_weather",'
                            '"arguments":{"location":"SF"}}</tool_call>'
                        ),
                        token=1,
                        logprobs=0.0,
                        finish_reason="stop",
                        prompt_tps=20.0,
                        cached_tokens=2,
                    )
                ]
            )

    tool_module = SimpleNamespace(
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        parse_tool_call=lambda call, tools: json.loads(call),
    )
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

    assert tool_chunk.get("usage") is None
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


def test_chat_completions_endpoint_forwards_video_content(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
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


# ---------------------------------------------------------------------------
# Legacy OpenAI text-completions endpoint (/v1/completions)
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


def test_completions_both_routes_registered():
    paths = {r.path for r in server.app.routes if hasattr(r, "path")}
    assert "/completions" in paths
    assert "/v1/completions" in paths


def test_chat_completions_endpoint_preserves_assistant_reasoning(client):
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
                        "reasoning": "Prior thought",
                    },
                    {"role": "user", "content": "Continue"},
                ],
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.args[2][1] == {
        "role": "assistant",
        "content": "Hello",
        "reasoning": "Prior thought",
    }


def test_anthropic_messages_endpoint_maps_text_and_images(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
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
        patch.object(server, "generate", return_value=result) as mock_generate,
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "demo",
                "system": "You are concise.",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe it."},
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": "https://example.com/image.png",
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 12,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "message"
    assert payload["role"] == "assistant"
    assert payload["content"] == [{"type": "text", "text": "done"}]
    assert payload["stop_reason"] == "end_turn"
    assert payload["usage"] == {
        "input_tokens": 8,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 4,
    }
    assert mock_template.call_args.args[2] == [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Describe it."},
    ]
    assert mock_generate.call_args.kwargs["image"] == ["https://example.com/image.png"]
    assert mock_generate.call_args.kwargs["max_tokens"] == 12


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
        {"role": "system", "content": "Use short answers.\nBe precise."},
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "Introduce the project."},
    ]


def test_anthropic_messages_endpoint_converts_tool_result_inputs(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=5,
        generation_tokens=2,
        prompt_tps=0.0,
        generation_tps=0.0,
        peak_memory=0.0,
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
            "/v1/messages",
            json={
                "model": "demo",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "get_weather",
                                "input": {"location": "SF"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "72F",
                            }
                        ],
                    },
                ],
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.args[2] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "SF"}, ensure_ascii=False),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": "72F", "name": None},
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


def test_anthropic_messages_endpoint_preserves_tool_result_images(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=5,
        generation_tokens=2,
        prompt_tps=0.0,
        generation_tps=0.0,
        peak_memory=0.0,
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
            "/v1/messages",
            json={
                "model": "demo",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "render_chart",
                                "input": {"kind": "bar"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": [
                                    {"type": "text", "text": "Rendered chart."},
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": "aW1n",
                                        },
                                    },
                                ],
                            }
                        ],
                    },
                ],
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.args[2] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "render_chart",
                        "arguments": json.dumps({"kind": "bar"}, ensure_ascii=False),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_1",
            "content": [
                {"type": "text", "text": "Rendered chart."},
                {"type": "image"},
            ],
            "name": None,
        },
    ]
    assert mock_generate.call_args.kwargs["image"] == ["data:image/png;base64,aW1n"]


def test_anthropic_messages_endpoint_returns_tool_use_blocks(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text='<tool_call>{"name":"get_weather","arguments":{"location":"SF"}}</tool_call>',
        prompt_tokens=7,
        generation_tokens=6,
        prompt_tps=0.0,
        generation_tps=0.0,
        peak_memory=0.0,
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
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    }
                ],
                "max_tokens": 8,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["stop_reason"] == "tool_use"
    assert payload["content"][0]["type"] == "tool_use"
    assert payload["content"][0]["name"] == "get_weather"
    assert payload["content"][0]["input"] == {"location": "SF"}


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


def test_anthropic_messages_streaming_emits_tool_use_events(client, monkeypatch):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    tool_module = SimpleNamespace(
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        parse_tool_call=lambda call, tools: json.loads(call),
    )

    class FakeResponseGenerator:
        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=3), iter(
                [
                    server.StreamingToken(
                        text='<tool_call>{"name":"get_weather","arguments":{"location":"SF"}}</tool_call>',
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
                        "input_schema": {"type": "object"},
                    }
                ],
                "max_tokens": 4,
                "stream": True,
            },
        )

    assert response.status_code == 200
    body = response.text
    assert '"type": "tool_use"' in body
    assert '"name": "get_weather"' in body
    assert '"type": "input_json_delta"' in body
    assert '"partial_json": "{\\"location\\": \\"SF\\"}"' in body
    assert '"stop_reason": "tool_use"' in body


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

    def _bare_generator(self):
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.draft_model = None
        gen.wait_until_ready = lambda: None
        gen._cpu_preprocess = lambda prompt, images, audio: {"input_ids": [1, 2, 3]}
        return gen

    def test_generate_forwards_videos_to_preprocess_and_queue(self):
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
        )

        collected = {"count": 0}

        def collect_pending_requests(**_kwargs):
            collected["count"] += 1
            if collected["count"] == 1:
                return [request], False
            return [], True

        handled = []
        gen._collect_pending_requests = collect_pending_requests
        gen._generate_diffusion = lambda uid, rq, raw, args, cancelled: handled.append(
            (uid, raw, args)
        )

        gen._run_diffusion("block")

        assert len(handled) == 1
        assert handled[0][2].max_tokens == 1
        assert isinstance(rqueue.get_nowait(), server.GenerationContext)
        assert rqueue.get_nowait() is None

    def test_generate_rejects_requests_over_configured_context_limit(self, monkeypatch):
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.wait_until_ready = lambda: None
        gen.draft_model = None
        gen._cpu_preprocess = lambda prompt, images, audio: {
            "input_ids": mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        }
        gen.requests = Queue()

        monkeypatch.setenv("MAX_KV_SIZE", "8")

        with pytest.raises(server.PromptTooLongError, match="MIN_OUTPUT_TOKENS"):
            gen.generate("prompt", args=server.GenerationArguments(max_tokens=4))

        assert gen.requests.empty()

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

        assert server.get_token_queue_timeout() == 600.0

    def test_token_queue_timeout_accepts_namespaced_env(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", "42.5")

        assert server.get_token_queue_timeout() == 42.5

    def test_token_queue_timeout_invalid_values_fall_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", "bad")

        assert server.get_token_queue_timeout() == 600.0

    def test_token_queue_timeout_can_disable_timeout(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", "0")

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
        monkeypatch.setenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", "0.01")

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
        monkeypatch.setenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", str(timeout_s * 10))

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
        gen._gpu_embed = lambda raw_inputs, images=None: (
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

    def test_run_routes_mtp_through_batch_generator(self, monkeypatch):
        batch_state = {}
        draft_model = object()

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
                batch_state["kwargs"] = kwargs
                self._next_uid = 1
                self._active = {}
                self.next_active_sizes = []
                batch_state["instance"] = self

            def insert(self, *args, **kwargs):
                del args, kwargs
                uid = self._next_uid
                self._next_uid += 1
                self._active[uid] = True
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
                responses = [
                    SimpleNamespace(
                        uid=uid,
                        token=uid + 100,
                        token_logprob=0.0,
                        finish_reason="length",
                    )
                    for uid in sorted(self._active)
                ]
                self._active.clear()
                return [], responses

        monkeypatch.setattr(server_generation, "BatchGenerator", FakeBatchGenerator)
        monkeypatch.setattr(
            server_generation,
            "_get_draft_block_size_from_env",
            lambda: 6,
        )
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
            gen.draft_model = draft_model
            gen.draft_kind = "mtp"
            gen.tokenizer = SimpleNamespace()

        gen._initialize_model = fake_initialize_model
        gen._run_speculative = lambda: pytest.fail("MTP should use BatchGenerator")
        gen._gpu_embed = lambda raw_inputs, images=None: (
            mx.array([[raw_inputs["request_id"]]], dtype=mx.int32),
            {},
        )

        request_queues = []
        for request_id in range(2):
            rqueue = Queue()
            request_queues.append(rqueue)
            gen.requests.put(
                server_generation.QueuedGenerationRequest(
                    rqueue=rqueue,
                    raw_inputs={"request_id": request_id},
                    prompt_tokens=1,
                    args=server.GenerationArguments(max_tokens=1, temperature=0),
                )
            )

        worker = Thread(target=gen._run, daemon=True)
        worker.start()

        try:
            for rqueue in request_queues:
                ctx = rqueue.get(timeout=1)
                assert isinstance(ctx, server.GenerationContext)
                item = rqueue.get(timeout=1)
                assert item.finish_reason == "length"
                assert rqueue.get(timeout=1) is None
        finally:
            gen._stop = True
            gen.requests.put(None)
            worker.join(timeout=2)

        kwargs = batch_state["kwargs"]
        assert kwargs["draft_model"] is draft_model
        assert kwargs["draft_kind"] == "mtp"
        assert kwargs["draft_block_size"] == 6
        assert kwargs["greedy_sampling"] is True
        assert kwargs["compute_logprobs"] is False
        assert batch_state["instance"].next_active_sizes == [2]

    def test_run_coalesces_idle_mtp_batch_generator(self, monkeypatch):
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
            gen.draft_kind = "mtp"
            gen.tokenizer = SimpleNamespace()

        def fake_collect_pending_requests(
            *, active, idle_timeout=0.1, coalesce_s=0.0, capacity=None
        ):
            del idle_timeout, capacity
            calls.append((active, coalesce_s))
            return [], True

        gen._initialize_model = fake_initialize_model
        gen._run_speculative = lambda: pytest.fail("MTP should use BatchGenerator")
        gen._collect_pending_requests = fake_collect_pending_requests

        gen._run()

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
        gen._gpu_embed = lambda raw_inputs, images=None: (
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
            thinking_budget=50,
            thinking_end_token="</think>",
        )
        kw = args.to_template_kwargs()
        assert kw["enable_thinking"] is False
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
        args = server._build_gen_args(req)

        assert args.thinking_budget == 256
        assert args.thinking_start_token == "<analysis>"
        assert args.thinking_end_token == "</analysis>"

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

        args = server._build_gen_args(req)

        assert args.thinking_budget == 32
        assert args.thinking_start_token == "<think>"
        assert args.thinking_end_token == "</think>"

    def test_server_cli_sets_thinking_defaults(self, monkeypatch):
        for env_var in (
            "MLX_VLM_ENABLE_THINKING",
            "MLX_VLM_PRELOAD_MODEL",
            "MLX_VLM_PRELOAD_ADAPTER",
            "MLX_VLM_VISION_CACHE_SIZE",
            "MLX_VLM_MAX_TOKENS",
            "MLX_VLM_THINKING_BUDGET",
            "MLX_VLM_THINKING_START_TOKEN",
            "MLX_VLM_THINKING_END_TOKEN",
            "MLX_VLM_SERVER_API_KEY",
            "PREFILL_STEP_SIZE",
            "KV_GROUP_SIZE",
            "KV_QUANT_SCHEME",
            "QUANTIZED_KV_START",
        ):
            monkeypatch.delenv(env_var, raising=False)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "mlx_vlm.server",
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
                "--model",
                "demo",
                "--enable-thinking",
                "--thinking-budget",
                "128",
                "--thinking-start-token",
                "<|START_THINKING|>",
                "--thinking-eos-token",
                "<|END_THINKING|>",
                "--api-key",
                "admin-token",
            ],
        )
        run_calls = []
        monkeypatch.setattr(
            server_cli.uvicorn,
            "run",
            lambda *args, **kwargs: run_calls.append((args, kwargs)),
        )

        try:
            server_cli.main()

            assert os.environ["MLX_VLM_ENABLE_THINKING"] == "1"
            assert os.environ["MLX_VLM_THINKING_BUDGET"] == "128"
            assert os.environ["MLX_VLM_THINKING_START_TOKEN"] == "<|START_THINKING|>"
            assert os.environ["MLX_VLM_THINKING_END_TOKEN"] == "<|END_THINKING|>"
            assert os.environ["MLX_VLM_SERVER_API_KEY"] == "admin-token"
            assert run_calls[0][1]["host"] == "127.0.0.1"
        finally:
            for env_var in (
                "MLX_VLM_ENABLE_THINKING",
                "MLX_VLM_PRELOAD_MODEL",
                "MLX_VLM_PRELOAD_ADAPTER",
                "MLX_VLM_VISION_CACHE_SIZE",
                "MLX_VLM_MAX_TOKENS",
                "MLX_VLM_THINKING_BUDGET",
                "MLX_VLM_THINKING_START_TOKEN",
                "MLX_VLM_THINKING_END_TOKEN",
                "MLX_VLM_SERVER_API_KEY",
            ):
                os.environ.pop(env_var, None)

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

        _, gen_kwargs = server.ResponseGenerator._gpu_embed(
            response_generator,
            {
                "input_ids": mx.array([[1, 2]]),
                "pixel_values": pixel_values,
                "attention_mask": mx.array([[1, 1]]),
            },
            images=None,
        )

        assert gen_kwargs["_apc_image_hash"] == hash_image_payload(
            pixel_values=pixel_values
        )

    def test_gpu_embed_prefers_image_ref_for_apc_hash(self):
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

        _, gen_kwargs = server.ResponseGenerator._gpu_embed(
            response_generator,
            {
                "input_ids": mx.array([[1, 2]]),
                "pixel_values": pixel_values,
                "attention_mask": mx.array([[1, 1]]),
            },
            images=images,
        )

        assert gen_kwargs["_apc_image_hash"] == hash_image_payload(image_ref=images)
        assert gen_kwargs["_apc_image_hash"] != hash_image_payload(
            pixel_values=pixel_values
        )

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


class TestSuppressToolCallContent:
    """Tests for tool-call markup suppression in streaming."""

    def test_no_tool_module(self):
        in_tc, content = server.suppress_tool_call_content(
            "Hello world", False, None, "world"
        )
        assert in_tc is False
        assert content == "world"

    def test_normal_text_before_tool_call(self):
        in_tc, content = server.suppress_tool_call_content(
            "I will call", False, "<tool_call>", "call"
        )
        assert in_tc is False
        assert content == "call"

    def test_suppresses_on_start_marker(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<tool_call>", False, "<tool_call>", ">"
        )
        assert in_tc is True
        assert content is None

    def test_suppresses_partial_marker(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<tool", False, "<tool_call>", "<tool"
        )
        assert in_tc is False
        assert content is None

    def test_stays_suppressed_after_entering(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<tool_call>get_weather", True, "<tool_call>", "weather"
        )
        assert in_tc is True
        assert content is None

    def test_pipe_delimited_marker(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<|tool_call>call:get_weather", False, "<|tool_call>", "weather"
        )
        assert in_tc is True
        assert content is None

    def test_pipe_delimited_partial_marker(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<|tool", False, "<|tool_call>", "<|tool"
        )
        assert in_tc is False
        assert content is None

    def test_literal_less_than_is_not_suppressed(self):
        in_tc, content = server.suppress_tool_call_content(
            "if n <", False, "<tool_call>", "<"
        )
        assert in_tc is False
        assert content == "<"


class TestProcessToolCalls:
    """Tests for tool call parsing from model output."""

    def test_no_tool_calls(self):
        # Minimal tool module mock
        module = SimpleNamespace(tool_call_start="<tc>", tool_call_end="</tc>")
        result = server.process_tool_calls("Just text.", module, [])
        assert result["calls"] == []
        assert result["remaining_text"] == "Just text."

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

        assert result["remaining_text"] == "Before   after"
        assert [call["function"]["name"] for call in result["calls"]] == [
            "grep",
            "read",
        ]
        assert json.loads(result["calls"][0]["function"]["arguments"]) == {
            "pattern": "foo"
        }
        assert json.loads(result["calls"][1]["function"]["arguments"]) == {
            "path": "file.py"
        }


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


# ============================================================================
# Fork-ported test classes (server-port adaptation, 2026-06-06)
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
