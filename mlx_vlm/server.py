import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, List, Literal, Optional, Union

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from huggingface_hub import scan_cache_dir
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Required, TypeAlias, TypedDict

from .generate import (
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_END_TOKEN,
    DEFAULT_THINKING_START_TOKEN,
    DEFAULT_TOP_P,
    generate,
    normalize_resize_shape,
    stream_generate,
    PromptCacheState,
)
import functools
from .prompt_utils import apply_chat_template
from .tool_parsers import _infer_tool_parser, load_tool_module
from .utils import load
from .version import __version__
from .vision_cache import VisionFeatureCache
import logging

from collections import OrderedDict
from .turboquant import register_allocation_hook, unregister_allocation_hook
from .utils import sanitize_strict_json

DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 8080

class MultiCacheManager:
    def __init__(self, threshold_percent: float):
        self.caches = OrderedDict()
        self.threshold_percent = threshold_percent
        self.active_session = None

    def get_free_percent(self):
        active = mx.metal.get_active_memory()
        limit = mx.device_info().get("max_recommended_working_set_size", active * 1.5)
        return max(0.0, ((limit - active) / limit) * 100.0)

    def check_and_evict(self):
        evicted = False
        # Keep evicting while memory is below threshold, as long as dormant caches exist
        while len(self.caches) > 0 and self.get_free_percent() < self.threshold_percent:
            # Find the oldest session that is NOT the currently active session
            oldest_session = None
            for session_id in self.caches:
                if session_id != self.active_session:
                    oldest_session = session_id
                    break

            if oldest_session is None:
                # Only the active session remains; we cannot evict it
                break

            oldest_state = self.caches.pop(oldest_session)
            logging.info(
                f"Multi-Cache | Critical Memory ({self.get_free_percent():.1f}% free < {self.threshold_percent}%). "
                f"Evicting dormant session: {oldest_session}"
            )

            # Destroy references to force memory release
            oldest_state.cache = None
            oldest_state.token_ids = None
            del oldest_state
            evicted = True

        if evicted:
            mx.metal.clear_cache()
            gc.collect()

    def get_or_create(self, session_id):
        self.active_session = session_id
        if session_id in self.caches:
            state = self.caches.pop(session_id)
            self.caches[session_id] = state  # Move to MRU position
            logging.info(f"Multi-Cache | Resuming session: {session_id}")
        else:
            state = PromptCacheState()
            self.caches[session_id] = state
            logging.info(f"Multi-Cache | New session allocated: {session_id}")

        # Check memory immediately upon assignment
        self.check_and_evict()
        return state

def _resolve_tool_calls(text: str, tool_parser_type=None, tool_module=None, tools=None):
    if tool_parser_type is None or tool_module is None:
        return {"calls": [], "remaining_text": text}

    # 1. Let the native MLX parser do the heavy lifting
    tool_calls = process_tool_calls(
        model_output=text,
        tool_module=tool_module,
        tools=tools,
    )

    # 2. Enforce OpenAI API / OpenWebUI Schema Strictness
    if tool_calls and "calls" in tool_calls:
        for call in tool_calls["calls"]:
            # Add missing mandatory fields
            if "id" not in call:
                call["id"] = f"call_{uuid.uuid4().hex[:8]}"
            if "type" not in call:
                call["type"] = "function"

            # Stringify the arguments dict (OpenWebUI expects a string, not a dict)
            func_data = call.get("function", {})
            if "arguments" in func_data and isinstance(func_data["arguments"], dict):
                func_data["arguments"] = json.dumps(func_data["arguments"])

    return tool_calls


class StreamingTranslator:
    """Consumptive buffer for reliable translation of stream tags using stateful boundary detection."""

    def __init__(self, thinking_start_token: str = DEFAULT_THINKING_START_TOKEN, thinking_end_token: str = DEFAULT_THINKING_END_TOKEN):
        self.buffer = ""
        self.in_thought = False
        self.is_beginning = True
        self.thinking_start_token = thinking_start_token
        self.thinking_end_token = thinking_end_token

    def translate(self, text: str):
        while True:
            open_match = re.search(r"<\|channel>\s*thought[\s\n]*", text, flags=re.IGNORECASE)
            close_match = re.search(r"<channel\|>", text, flags=re.IGNORECASE)

            hallucinated_open = None
            if not self.in_thought:
                hallucinated_open = re.search(r"(?:^|\n)\s*(?:\**thought\**\s*[:\n]|\**thinking\**\s*[:\n])", text, flags=re.IGNORECASE)

            open_idx = open_match.start() if open_match else float("inf")
            close_idx = close_match.start() if close_match else float("inf")
            hallucinated_idx = hallucinated_open.start() if hallucinated_open else float("inf")

            if open_idx == float("inf") and close_idx == float("inf") and hallucinated_idx == float("inf"):
                break

            min_idx = min(open_idx, close_idx, hallucinated_idx)

            if min_idx == open_idx:
                text = f"{text[:open_idx]}{self.thinking_start_token}\n{text[open_match.end() :]}"

                self.in_thought = True
            elif min_idx == hallucinated_idx:
                idx = hallucinated_open.start()
                match_str = text[idx:hallucinated_open.end()]
                prefix = text[:idx]
                if match_str.startswith('\n'):
                    text = f"{prefix}\n{self.thinking_start_token}\n{text[hallucinated_open.end() :]}"
                else:
                    text = f"{prefix}{self.thinking_start_token}\n{text[hallucinated_open.end() :]}"
                self.in_thought = True
            else:
                if self.in_thought:
                    text = f"{text[:close_idx]}\n{self.thinking_end_token}\n{text[close_match.end() :]}"
                    self.in_thought = False
                else:
                    text = text[:close_idx] + text[close_match.end() :]

        return text

    def feed(self, text: str):
        self.buffer += text

        # Guard at the very beginning of the stream
        if self.is_beginning:
            self.buffer = re.sub(r"^\s*(?:response|answer)\s*[:\.]\s*[\r\n]*", "", self.buffer, flags=re.IGNORECASE)
            self.buffer = re.sub(r"^\s*(?:[a-zA-Z0-9/_'-]+\s*){1,2}[\.\:]\s*[\r\n]+", "", self.buffer)

            # Un-fuse missing spaces on the very first word of the generation
            self.buffer = re.sub(
                r"\b(This|The|Here|There)(is|provided|search|user|context|results|model)\b",
                r"\1 \2",
                self.buffer,
                flags=re.IGNORECASE,
            )

        self.buffer = self.translate(self.buffer)

        # 15-character delay prevents splitting words before the regex can catch them
        split_idx = max(0, len(self.buffer) - 15)

        last_open = self.buffer.rfind("<")
        last_close = self.buffer.rfind(">")
        if last_open != -1 and last_open > last_close:
            split_idx = min(split_idx, last_open)

        channel_idx = self.buffer.rfind("<|channel>")
        if channel_idx != -1:
            split_idx = min(split_idx, channel_idx)

        chunk = self.buffer[:split_idx]
        self.buffer = self.buffer[split_idx:]

        if chunk.strip() and not chunk.endswith(f"{self.thinking_start_token}\n"):
            self.is_beginning = False

        return chunk

    def pre_tool_flush(self):
        self.buffer = self.translate(self.buffer)
        split_idx = self.buffer.find("<|tool_call>")
        if split_idx != -1:
            chunk = self.buffer[:split_idx]

            has_close_tag = f"{self.thinking_end_token}" in chunk

            if has_close_tag:
                tag_idx = chunk.find(f"{self.thinking_end_token}") + len(f"{self.thinking_end_token}")
                pre_tag = chunk[:tag_idx]
                post_tag = chunk[tag_idx:]

                if len(post_tag.strip()) < 60:
                    chunk = pre_tag + "\n"
                else:
                    chunk = pre_tag + re.sub(r"\s*[\w\./\\]+[\.\:]?\s*$", "\n", post_tag)
            else:
                if len(chunk.strip()) < 60:
                    chunk = ""
                else:
                    chunk = re.sub(r"\s*[\w\./\\]+[\.\:]?\s*$", "\n", chunk)

            if self.in_thought:
                chunk += f"\n{self.thinking_end_token}\n"
                self.in_thought = False

            self.buffer = self.buffer[split_idx:]
            return chunk
        return ""

    def flush(self):
        chunk = self.translate(self.buffer)
        if self.in_thought:
            chunk += f"\n{self.thinking_end_token}\n"
            self.in_thought = False
        self.buffer = ""
        return chunk

def get_prefill_step_size():
    return int(os.environ.get("PREFILL_STEP_SIZE", DEFAULT_PREFILL_STEP_SIZE))


def get_quantized_kv_bits(model: str):
    kv_bits = float(os.environ.get("KV_BITS", 0))
    if kv_bits == 0:
        return None
    if "qat" in model:
        logging.warning(
            f"Model {model} is quantization aware, (Rotating)KVCache cache will not be quantized to {kv_bits} bits, use --max-kv-size [tokens] instead."
        )
        return None
    return kv_bits


def get_kv_group_size():
    return int(os.environ.get("KV_GROUP_SIZE", DEFAULT_KV_GROUP_SIZE))


def get_kv_quant_scheme():
    return os.environ.get("KV_QUANT_SCHEME", DEFAULT_KV_QUANT_SCHEME)


def get_max_kv_size(model: str):
    max_kv_tokens = int(os.environ.get("MAX_KV_SIZE", 0))
    if max_kv_tokens == 0:
        return None
    if get_quantized_kv_bits(model) is not None:
        logging.warning(
            f"Model {model} uses QuantizedKVCache cache. MAX_KV_SIZE={max_kv_tokens} will be used for static pre-allocation."
        )
    return max_kv_tokens


def get_quantized_kv_start():
    return int(os.environ.get("QUANTIZED_KV_START", DEFAULT_QUANTIZED_KV_START))


@asynccontextmanager
async def lifespan(app):
    # Startup
    model_path = os.environ.get("PRELOAD_MODEL")
    adapter_path = os.environ.get("PRELOAD_ADAPTER") or None
    if model_path:
        try:
            logging.info(f"Preloading model: {model_path}")
            get_cached_model(model_path, adapter_path)
        except Exception as e:
            logging.error(f"Failed to preload model: {e}")
            logging.error("Server will continue without a preloaded model.")
    yield
    unload_model_sync()


app = FastAPI(
    title="MLX-VLM Inference API",
    description="API for using Vision Language Models (VLMs) and Omni Models (Vision, Audio and Video support) with MLX.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_IMAGES = 10  # Maximum number of images to process at once

# Loading/unloading utilities

model_cache = {}


class FlexibleBaseModel(BaseModel):
    """Base model that ignores/accepts any unknown OpenAI SDK fields."""

    model_config = ConfigDict(extra="allow")

    def dump_kwargs(self, *fields: str) -> dict[str, Any]:
        return self.model_dump(include=set(fields), exclude_none=True)


def load_model_resources(model_path: str, adapter_path: Optional[str]):
    """
    Loads model, processor, and config based on paths.
    Handles potential loading errors.
    """
    try:
        logging.info(f"Loading model from: {model_path}")
        if adapter_path:
            logging.info(f"Loading adapter from: {adapter_path}")
        # Use the load function from utils.py which handles path resolution and loading
        trust_remote_code = (
            os.environ.get("MLX_TRUST_REMOTE_CODE", "false").lower() == "true"
        )
        model, processor = load(
            model_path, adapter_path, trust_remote_code=trust_remote_code
        )
        config = model.config
        logging.info("Model and processor loaded successfully.")
        return model, processor, config
    except Exception as e:
        logging.error(f"Error loading model {model_path}: {e}")
        traceback.print_exc()  # Print detailed traceback for debugging
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


def get_cached_model(model_path: str, adapter_path: Optional[str] = None):
    """
    Factory function to get or load the appropriate model resources from cache or by loading.
    """
    global model_cache

    cache_key = (model_path, adapter_path)

    # Return from cache if already loaded and matches the requested paths
    if model_cache.get("cache_key") == cache_key:
        logging.info(f"Using cached model: {model_path}, Adapter: {adapter_path}")
        return model_cache["model"], model_cache["processor"], model_cache["config"]

    # If cache exists but doesn't match, clear it
    if model_cache:
        logging.info("New model request, clearing existing cache...")
        unload_model_sync()  # Use a synchronous version for internal call

    # Load the model resources
    model, processor, config = load_model_resources(model_path, adapter_path)

    model_cache = {
        "cache_key": cache_key,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "model": model,
        "processor": processor,
        "config": config,
        "vision_cache": VisionFeatureCache(),
        "multi_cache_manager": None, # lazy initialization to perform registration of allocation hook only when a model is loaded
    }

    return model, processor, config


# Synchronous unload function for internal use
def unload_model_sync():
    global model_cache
    if not model_cache:
        return False

    logging.info(
        f"Unloading model: {model_cache.get('model_path')}, Adapter: {model_cache.get('adapter_path')}"
    )
    # Clear vision cache before dropping references
    if "vision_cache" in model_cache:
        model_cache["vision_cache"].clear()
    # Unregister allocation hook and purge multi-cache sessions before dropping references
    mcm = model_cache.get("multi_cache_manager")
    if mcm is not None:
        unregister_allocation_hook(mcm.check_and_evict)
        for state in mcm.caches.values():
            state.cache = None
            state.token_ids = None
        mcm.caches.clear()
    model_cache = {}
    # Force garbage collection
    gc.collect()
    mx.clear_cache()
    logging.info("Model unloaded and cache cleared.")
    return True


# OpenAI API Models

# Models for /responses endpoint


class ResponseInputTextParam(TypedDict, total=False):
    text: Required[str]
    type: Required[
        Literal["input_text", "text"]
    ]  # The type of the input item. Always `input_text`.


class ResponseInputImageParam(TypedDict, total=False):
    detail: Literal["high", "low", "auto"] = Field(
        "auto", description="The detail level of the image to be sent to the model."
    )
    """The detail level of the image to be sent to the model.

    One of `high`, `low`, or `auto`.
    """
    type: Required[
        Literal["input_image"]
    ]  # The type of the input item. Always `input_image`.
    image_url: Required[str]
    file_id: Optional[str]
    """The ID of the file to be sent to the model.
     NOTE : wouldn't this help the model if we passed the file_id as well to the vlm models
    """


class InputAudio(TypedDict, total=False):
    data: Required[str]
    format: Required[str]


class ResponseInputAudioParam(TypedDict, total=False):
    type: Required[
        Literal["input_audio"]
    ]  # The type of the input item. Always `input_audio`.
    input_audio: Required[InputAudio]


class ImageUrl(TypedDict, total=False):
    url: Required[str]


class ResponseImageUrlParam(TypedDict, total=False):
    type: Required[
        Literal["image_url"]
    ]  # The type of the input item. Always`image_url`.
    image_url: Required[ImageUrl]


ResponseInputContentParam: TypeAlias = Union[
    ResponseInputTextParam,
    ResponseInputImageParam,
    ResponseImageUrlParam,
    ResponseInputAudioParam,
]

ResizeShapeInput: TypeAlias = tuple[int] | tuple[int, int]

ResponseInputMessageContentListParam: TypeAlias = List[ResponseInputContentParam]


class ResponseOutputText(TypedDict, total=False):
    text: Required[str]
    type: Required[
        Literal["output_text"]
    ]  # The type of the output item. Always `output_text`


ResponseOutputMessageContentList: TypeAlias = List[ResponseOutputText]


class ChatMessage(FlexibleBaseModel):
    role: Literal["user", "assistant", "system", "developer", "tool"] = Field(
        ...,
        description="Role of the message sender (e.g., 'system', 'user', 'assistant').",
    )
    content: Optional[
        Union[
            str,
            ResponseInputMessageContentListParam,
            ResponseOutputMessageContentList,
        ]
    ] = Field(None, description="Content of the message.")
    tool_calls: Optional[List] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class GenerationParams(FlexibleBaseModel):
    temperature: float = Field(
        DEFAULT_TEMPERATURE,
        description="Temperature for sampling.",
    )
    top_p: float = Field(
        DEFAULT_TOP_P,
        description="Top-p sampling.",
    )
    top_k: Optional[int] = Field(
        None,
        description="Top-k sampling cutoff.",
    )
    min_p: Optional[float] = Field(
        None,
        description="Min-p sampling threshold.",
    )
    repetition_penalty: Optional[float] = Field(
        None, description="Penalty applied to repeated tokens."
    )
    logit_bias: Optional[dict[int, float]] = Field(
        None, description="Additive logit bias keyed by token id."
    )

    def shared_generation_kwargs(self) -> dict[str, Any]:
        return self.dump_kwargs(
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "logit_bias",
        )


class TemplateParams(FlexibleBaseModel):
    enable_thinking: Optional[bool] = Field(
        None,
        description="Enable thinking mode in the chat template.",
    )
    thinking_budget: Optional[int] = Field(
        None,
        description="Maximum number of thinking tokens before forcing the end token.",
    )
    thinking_start_token: Optional[str] = Field(
        DEFAULT_THINKING_START_TOKEN,
        description="Token that marks the start of a thinking block.",
    )
    thinking_end_token: Optional[str] = Field(
        DEFAULT_THINKING_END_TOKEN,
        description="Token that marks the end of a thinking block.",
    )

    def template_kwargs(self) -> dict[str, Any]:
        kwargs = self.dump_kwargs(
            "enable_thinking",
            "thinking_budget",
            "thinking_start_token",
            "thinking_end_token",
        )
        kwargs.setdefault("enable_thinking", False)
        return kwargs


class OpenAIRequest(GenerationParams, TemplateParams):
    """
    OpenAI-compatible request structure.
    Using this structure : https://github.com/openai/openai-python/blob/main/src/openai/resources/responses/responses.py
    """

    input: Union[str, List[ChatMessage]] = Field(
        ..., description="Input text or list of chat messages."
    )
    model: str = Field(..., description="The model to use for generation.")
    max_output_tokens: int = Field(
        DEFAULT_MAX_TOKENS,
        description="Maximum number of tokens to generate.",
    )
    stream: bool = Field(
        False, description="Whether to stream the response chunk by chunk."
    )

    def generation_kwargs(self) -> dict[str, Any]:
        kwargs = self.dump_kwargs("max_output_tokens")
        kwargs["max_tokens"] = kwargs.pop("max_output_tokens")
        return {**kwargs, **self.shared_generation_kwargs()}


class OpenAIUsage(BaseModel):
    """Token usage details including input tokens, output tokens, breakdown, and total tokens used."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class OpenAIErrorObject(BaseModel):
    """Error object returned when the model fails to generate a Response."""

    code: Optional[str] = None
    message: Optional[str] = None
    param: Optional[str] = None
    type: Optional[str] = None


class OpenAIResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for this Response")
    object: Literal["response"] = Field(
        ..., description="The object type of this resource - always set to response"
    )
    created_at: int = Field(
        ..., description="Unix timestamp (in seconds) of when this Response was created"
    )
    status: Literal["completed", "failed", "in_progress", "incomplete"] = Field(
        ..., description="The status of the response generation"
    )
    error: Optional[OpenAIErrorObject] = Field(
        None,
        description="An error object returned when the model fails to generate a Response",
    )
    instructions: Optional[str] = Field(
        None,
        description="Inserts a system (or developer) message as the first item in the model's context",
    )
    max_output_tokens: Optional[int] = Field(
        None,
        description="An upper bound for the number of tokens that can be generated for a response",
    )
    model: str = Field(..., description="Model ID used to generate the response")
    output: List[Union[ChatMessage, Any]] = Field(
        ..., description="An array of content items generated by the model"
    )
    output_text: Optional[str] = Field(
        None,
        description="SDK-only convenience property containing aggregated text output",
    )
    temperature: Optional[float] = Field(
        None, ge=0, le=2, description="Sampling temperature between 0 and 2"
    )
    top_p: Optional[float] = Field(
        None, ge=0, le=1, description="Nucleus sampling probability mass"
    )
    truncation: Union[Literal["auto", "disabled"], str] = Field(
        "disabled", description="The truncation strategy to use"
    )
    usage: OpenAIUsage = Field(
        ..., description="Token usage details"
    )  # we need the model to return stats
    user: Optional[str] = Field(
        None, description="A unique identifier representing your end-user"
    )


class BaseStreamEvent(BaseModel):
    type: str


class ContentPartOutputText(BaseModel):
    type: Literal["output_text"]
    text: str
    annotations: List[str] = []


class MessageItem(BaseModel):
    id: str
    type: Literal["message"]
    status: Literal["in_progress", "completed"]
    role: str
    content: List[ContentPartOutputText] = []


class ResponseCreatedEvent(BaseStreamEvent):
    type: Literal["response.created"]
    response: OpenAIResponse


class ResponseInProgressEvent(BaseStreamEvent):
    type: Literal["response.in_progress"]
    response: OpenAIResponse


class ResponseOutputItemAddedEvent(BaseStreamEvent):
    type: Literal["response.output_item.added"]
    output_index: int
    item: MessageItem


class ResponseContentPartAddedEvent(BaseStreamEvent):
    type: Literal["response.content_part.added"]
    item_id: str
    output_index: int
    content_index: int
    part: ContentPartOutputText


class ResponseOutputTextDeltaEvent(BaseStreamEvent):
    type: Literal["response.output_text.delta"]
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseOutputTextDoneEvent(BaseStreamEvent):
    type: Literal["response.output_text.done"]
    item_id: str
    output_index: int
    content_index: int
    text: str


class ResponseContentPartDoneEvent(BaseStreamEvent):
    type: Literal["response.content_part.done"]
    item_id: str
    output_index: int
    content_index: int
    part: ContentPartOutputText


class ResponseOutputItemDoneEvent(BaseStreamEvent):
    type: Literal["response.output_item.done"]
    output_index: int
    item: MessageItem


class ResponseCompletedEvent(BaseStreamEvent):
    type: Literal["response.completed"]
    response: OpenAIResponse


StreamEvent = Union[
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseContentPartAddedEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ResponseContentPartDoneEvent,
    ResponseOutputItemDoneEvent,
    ResponseCompletedEvent,
]

# Models for /chat/completion endpoint


class VLMRequest(GenerationParams, TemplateParams):
    model: str = Field(
        DEFAULT_MODEL_PATH,
        description="The path to the local model directory or Hugging Face repo.",
    )
    adapter_path: Optional[str] = Field(
        None, description="The path to the adapter weights."
    )
    max_tokens: int = Field(
        DEFAULT_MAX_TOKENS,
        description="Maximum number of tokens to generate.",
    )
    seed: int = Field(DEFAULT_SEED, description="Seed for random generation.")
    resize_shape: Optional[ResizeShapeInput] = Field(
        None,
        description="Resize shape for the image. Provide one integer for a square resize or two integers for (height, width).",
    )
    user: Optional[str] = Field(None, description="OpenAI standard user/session identifier.")
    session_id: Optional[str] = Field(None, description="Custom session identifier.")

    @field_validator("resize_shape", mode="before")
    @classmethod
    def normalize_resize_shape_field(cls, value):
        return normalize_resize_shape(value)

    def generation_kwargs(self) -> dict[str, Any]:
        return {
            **self.dump_kwargs("max_tokens", "resize_shape"),
            **self.shared_generation_kwargs(),
        }


class GenerationRequest(VLMRequest):
    """
    Inherits from VLMRequest and adds additional fields for the generation request.
    """

    stream: bool = Field(
        False, description="Whether to stream the response chunk by chunk."
    )


class UsageStats(OpenAIUsage):
    """
    Inherits from OpenAIUsage and adds additional fields for usage statistics.
    """

    prompt_tps: float = Field(..., description="Tokens per second for the prompt.")
    generation_tps: float = Field(
        ..., description="Tokens per second for the generation."
    )
    peak_memory: float = Field(
        ..., description="Peak memory usage during the generation."
    )


class ChatRequest(GenerationRequest):
    messages: List[ChatMessage]
    tools: Optional[List[Any]] = None


class ChatChoice(BaseModel):
    finish_reason: str
    message: ChatMessage


class ChatResponse(BaseModel):
    model: str
    choices: List[ChatChoice]
    usage: Optional[UsageStats]


class ChatStreamChoice(BaseModel):
    index: int = 0
    finish_reason: Optional[str] = None
    delta: ChatMessage


class ChatStreamChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatStreamChoice]
    usage: Optional[UsageStats]


def get_serialize_kv_quantization():
    return os.environ.get("SERIALIZE_KV_QUANTIZATION", "false").lower() == "true"


def build_generation_kwargs(
    request: Any,
    template_kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "prefill_step_size": get_prefill_step_size(),
        "kv_bits": get_quantized_kv_bits(request.model),
        "kv_group_size": get_kv_group_size(),
        "kv_quant_scheme": get_kv_quant_scheme(),
        "max_kv_size": get_max_kv_size(request.model),
        "quantized_kv_start": get_quantized_kv_start(),
        "serialize_kv_quantization": get_serialize_kv_quantization(),
        **request.generation_kwargs(),
        **template_kwargs,
    }


def process_tool_calls(model_output: str, tool_module, tools):
    called_tools = []
    remaining = model_output

    if tool_module.tool_call_start in model_output:
        if tool_module.tool_call_end == "":
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?(?:\n|$)", re.DOTALL
            )

        else:
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?{re.escape(tool_module.tool_call_end)}",
                re.DOTALL,
            )

        matches = re.findall(pattern, model_output)
        if matches:
            remaining = re.sub(pattern, " ", model_output).strip()
            tool_call_index = 0
            for match in matches:
                call = (
                    match.strip()
                    .removeprefix(tool_module.tool_call_start)
                    .removesuffix(tool_module.tool_call_end)
                )
                try:
                    tool_call = tool_module.parse_tool_call(call, tools)
                    called_tool = {}
                    called_tool["type"] = "function"
                    called_tool["index"] = tool_call_index
                    called_tool["id"] = str(uuid.uuid4())
                    called_tool["function"] = {}
                    called_tool["function"]["name"] = tool_call["name"].strip()
                    called_tool["function"]["arguments"] = json.dumps(
                        tool_call["arguments"], ensure_ascii=False
                    )
                    called_tools.append(called_tool)
                    tool_call_index += 1
                except Exception:
                    logging.error(f"Invalid tool call: {call}")
    return dict(calls=called_tools, remaining_text=remaining)


# Models for /models endpoint


class ModelInfo(BaseModel):
    id: str
    object: str
    created: int


class ModelsResponse(BaseModel):
    object: Literal["list"]
    data: List[ModelInfo]


# OpenAI compatile endpoints


@app.post("/responses")
@app.post("/v1/responses", include_in_schema=False)
async def responses_endpoint(openai_request: OpenAIRequest):
    """
    OpenAI-compatible endpoint for generating text based on a prompt and optional images.

    using client.responses.create method.

    example:

    from openai import OpenAI

    API_URL = "http://0.0.0.0:8000"
    API_KEY = 'any'

    def run_openai(prompt, img_url,system, stream=False, max_output_tokens=512, model="mlx-community/Qwen2.5-VL-3B-Instruct-8bit"):
        ''' Calls the OpenAI API
        '''

        client = OpenAI(base_url=f"{API_URL}", api_key=API_KEY)

        try :
            response = client.responses.create(
                model=model,
                input=[
                    {"role":"system",
                    "content": f"{system}"
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": f"{img_url}"},
                        ],
                    }
                ],
                max_output_tokens=max_output_tokens,
                stream=stream
            )
            if not stream:
                logging.info(response.output[0].content[0].text)
                logging.info(response.usage)
            else:
                for event in response:
                    # Process different event types if needed
                    if hasattr(event, 'delta') and event.delta:
                        logging.info(event.delta)
                    elif event.type == 'response.completed':
                        logging.info("\n--- Usage ---")
                        logging.info(event.response.usage)

        except Exception as e:
            # building a response object to match the one returned when request is successful so that it can be processed in the same way
            return {"model - error":str(e),"content":{}, "model":model}

    """

    try:
        # Get model, processor, config - loading if necessary
        model, processor, config = get_cached_model(openai_request.model)

        chat_messages = []
        images = []
        instructions = None
        if openai_request.input:
            if isinstance(openai_request.input, str):
                # If input is a string, treat it as a single text message
                chat_messages.append({"role": "user", "content": openai_request.input})
            elif isinstance(openai_request.input, list):
                # If input is a list, treat it as a series of chat messages
                for message in openai_request.input:
                    if isinstance(message, ChatMessage):
                        if message.content is None:
                            chat_messages.append({"role": message.role, "content": ""})
                        elif isinstance(message.content, str):
                            chat_messages.append(
                                {"role": message.role, "content": message.content}
                            )
                            if message.role == "system":
                                instructions = message.content
                        elif isinstance(message.content, list):
                            # Handle list of content items
                            for item in message.content:
                                if isinstance(item, dict):
                                    if item["type"] == "input_text":
                                        chat_messages.append(
                                            {
                                                "role": message.role,
                                                "content": item["text"],
                                            }
                                        )
                                        if message.role == "system":
                                            instructions = item["text"]
                                    # examples for multiple images (https://platform.openai.com/docs/guides/images?api-mode=responses)
                                    elif item["type"] == "input_image":
                                        images.append(item["image_url"])
                                    else:
                                        logging.error(
                                            f"invalid input item type: {item['type']}"
                                        )
                                        raise HTTPException(
                                            status_code=400,
                                            detail="Invalid input item type.",
                                        )
                                else:
                                    logging.error(
                                        f"Invalid message content item format: {item}"
                                    )
                                    raise HTTPException(
                                        status_code=400,
                                        detail="Missing type in input item.",
                                    )
                        else:
                            logging.error("Invalid message content format.")
                            raise HTTPException(
                                status_code=400, detail="Invalid input format."
                            )
                    else:
                        logging.error("not a ChatMessage")
                        raise HTTPException(
                            status_code=400, detail="Invalid input format."
                        )
            else:
                logging.error("neither string not list")
                raise HTTPException(status_code=400, detail="Invalid input format.")

        else:
            logging.error("no input")
            raise HTTPException(status_code=400, detail="Missing input.")

        template_kwargs = openai_request.template_kwargs()
        formatted_prompt = apply_chat_template(
            processor,
            config,
            chat_messages,
            num_images=len(images),
            **template_kwargs,
        )
        generation_kwargs = build_generation_kwargs(openai_request, template_kwargs)
        thinking_start_token = generation_kwargs.get("thinking_start_token", DEFAULT_THINKING_START_TOKEN)
        thinking_end_token = generation_kwargs.get("thinking_end_token", DEFAULT_THINKING_END_TOKEN)

        generated_at = datetime.now().timestamp()
        response_id = f"resp_{uuid.uuid4().hex}"
        message_id = f"msg_{uuid.uuid4().hex}"

        if openai_request.stream:
            # Streaming response
            async def stream_generator():
                token_iterator = None
                try:
                    # Create base response object (to match the openai pipeline)
                    base_response = OpenAIResponse(
                        id=response_id,
                        object="response",
                        created_at=int(generated_at),
                        status="in_progress",
                        instructions=instructions,
                        max_output_tokens=openai_request.max_output_tokens,
                        model=openai_request.model,
                        output=[],
                        output_text="",
                        temperature=openai_request.temperature,
                        top_p=openai_request.top_p,
                        usage={
                            "input_tokens": 0,  # get prompt tokens
                            "output_tokens": 0,
                            "total_tokens": 0,
                        },
                    )

                    # Send response.created event  (to match the openai pipeline)
                    yield f"event: response.created\ndata: {ResponseCreatedEvent(type='response.created', response=base_response).model_dump_json()}\n\n"

                    # Send response.in_progress event  (to match the openai pipeline)
                    yield f"event: response.in_progress\ndata: {ResponseInProgressEvent(type='response.in_progress', response=base_response).model_dump_json()}\n\n"

                    # Send response.output_item.added event  (to match the openai pipeline)
                    message_item = MessageItem(
                        id=message_id,
                        type="message",
                        status="in_progress",
                        role="assistant",
                        content=[],
                    )
                    yield f"event: response.output_item.added\ndata: {ResponseOutputItemAddedEvent(type='response.output_item.added', output_index=0, item=message_item).model_dump_json()}\n\n"

                    # Send response.content_part.added event
                    content_part = ContentPartOutputText(
                        type="output_text", text="", annotations=[]
                    )
                    yield f"event: response.content_part.added\ndata: {ResponseContentPartAddedEvent(type='response.content_part.added', item_id=message_id, output_index=0, content_index=0, part=content_part).model_dump_json()}\n\n"

                    # Stream text deltas
                    token_iterator = stream_generate(
                        model=model,
                        processor=processor,
                        prompt=formatted_prompt,
                        image=images,
                        vision_cache=model_cache.get("vision_cache"),
                        **generation_kwargs,
                    )

                    full_text = ""
                    translator = StreamingTranslator(thinking_start_token=thinking_start_token, thinking_end_token=thinking_end_token)
                    usage_stats = {"input_tokens": 0, "output_tokens": 0}

                    for chunk in token_iterator:
                        if chunk is None or not hasattr(chunk, "text"):
                            continue

                        delta = translator.feed(chunk.text)
                        usage_stats = {
                            "input_tokens": chunk.prompt_tokens,
                            "output_tokens": chunk.generation_tokens,
                        }

                        if delta:
                            full_text += delta
                            yield f"event: response.output_text.delta\ndata: {ResponseOutputTextDeltaEvent(type='response.output_text.delta', item_id=message_id, output_index=0, content_index=0, delta=delta).model_dump_json()}\n\n"

                    final_delta = translator.flush()
                    if final_delta:
                        full_text += final_delta
                        yield f"event: response.output_text.delta\ndata: {ResponseOutputTextDeltaEvent(type='response.output_text.delta', item_id=message_id, output_index=0, content_index=0, delta=final_delta).model_dump_json()}\n\n"

                    yield f"event: response.output_text.done\ndata: {ResponseOutputTextDoneEvent(type='response.output_text.done', item_id=message_id, output_index=0, content_index=0, text=full_text).model_dump_json()}\n\n"

                    # Send response.content_part.done event (to match the openai pipeline)
                    final_content_part = ContentPartOutputText(
                        type="output_text", text=full_text, annotations=[]
                    )
                    yield f"event: response.content_part.done\ndata: {ResponseContentPartDoneEvent(type='response.content_part.done', item_id=message_id, output_index=0, content_index=0, part=final_content_part).model_dump_json()}\n\n"

                    # Send response.output_item.done event (to match the openai pipeline)
                    final_message_item = MessageItem(
                        id=message_id,
                        type="message",
                        status="completed",
                        role="assistant",
                        content=[final_content_part],
                    )
                    yield f"event: response.output_item.done\ndata: {ResponseOutputItemDoneEvent(type='response.output_item.done', output_index=0, item=final_message_item).model_dump_json()}\n\n"

                    # Send response.completed event (to match the openai pipeline)
                    completed_response = base_response.model_copy(
                        update={
                            "status": "completed",
                            "output": [final_message_item],
                            "usage": {
                                "input_tokens": usage_stats["input_tokens"],
                                "output_tokens": usage_stats["output_tokens"],
                                "total_tokens": usage_stats["input_tokens"]
                                + usage_stats["output_tokens"],
                            },
                        }
                    )
                    logging.info(f"Responses Stream Usage: {json.dumps(completed_response.usage)}")
                    yield f"event: response.completed\ndata: {ResponseCompletedEvent(type='response.completed', response=completed_response).model_dump_json()}\n\n"

                except Exception as e:
                    logging.error(f"Error during stream generation: {e}")
                    traceback.print_exc()
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"

                finally:
                    mx.clear_cache()
                    gc.collect()
                    logging.info("Stream finished, cleared cache.")

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            # Non-streaming response
            try:
                # Use generate from generate.py
                result = generate(
                    model=model,
                    processor=processor,
                    prompt=formatted_prompt,
                    image=images,
                    verbose=False,  # stats are passed in the response
                    **generation_kwargs,
                )
                # Clean up resources
                mx.clear_cache()
                gc.collect()
                logging.info("Generation finished, cleared cache.")

                translator = StreamingTranslator(thinking_start_token=thinking_start_token, thinking_end_token=thinking_end_token)
                translated_text = translator.feed(result.text) + translator.flush()
                translated_text = sanitize_strict_json(translated_text) if translated_text else translated_text

                response = OpenAIResponse(
                    id=response_id,
                    object="response",
                    created_at=int(generated_at),
                    status="completed",
                    instructions=instructions,
                    max_output_tokens=openai_request.max_output_tokens,
                    model=openai_request.model,
                    output=[
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": translated_text,
                                }
                            ],
                        }
                    ],
                    output_text=translated_text,
                    temperature=openai_request.temperature,
                    top_p=openai_request.top_p,
                    usage={
                        "input_tokens": result.prompt_tokens,
                        "output_tokens": result.generation_tokens,
                        "total_tokens": result.total_tokens,
                    },
                )
                return response

            except Exception as e:
                logging.error(f"Error during generation: {e}")
                traceback.print_exc()
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions (like model loading failure)
        raise http_exc
    except Exception as e:
        # Catch unexpected errors
        logging.error(f"Unexpected error in /responses endpoint: {e}")
        traceback.print_exc()
        mx.clear_cache()
        gc.collect()
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@app.post(
    "/chat/completions", response_model=None
)  # Response model handled dynamically based on stream flag
@app.post("/v1/chat/completions", response_model=None, include_in_schema=False)
async def chat_completions_endpoint(request: ChatRequest):
    """
    Generate text based on a prompt and optional images.
    Prompt must be a list of chat messages, including system, user, and assistant messages.
    System message will be ignored if not already in the prompt.
    Can operate in streaming or non-streaming mode.
    """

    try:
        # Get model, processor, config - loading if necessary
        model, processor, config = get_cached_model(request.model, request.adapter_path)

        images = []
        audio = []
        processed_messages = []
        template_kwargs = request.template_kwargs()
        generation_kwargs = build_generation_kwargs(request, template_kwargs)
        thinking_start_token = generation_kwargs.get("thinking_start_token", DEFAULT_THINKING_START_TOKEN)
        thinking_end_token = generation_kwargs.get("thinking_end_token", DEFAULT_THINKING_END_TOKEN)

        for message in request.messages:
            msg_dict = {"role": message.role}

            if hasattr(message, "tool_calls") and message.tool_calls:
                parsed_tool_calls = []
                for tc in message.tool_calls:
                    tc_dict = tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
                    if "function" in tc_dict and isinstance(
                        tc_dict["function"].get("arguments"), str
                    ):
                        try:
                            tc_dict["function"]["arguments"] = json.loads(
                                tc_dict["function"]["arguments"]
                            )
                        except Exception as e:
                            logging.error(
                                f"Failed to parse tool arguments to dict: {e}"
                            )
                    parsed_tool_calls.append(tc_dict)
                msg_dict["tool_calls"] = parsed_tool_calls

            if hasattr(message, "tool_call_id") and message.tool_call_id:
                msg_dict["tool_call_id"] = message.tool_call_id
            if hasattr(message, "name") and message.name:
                msg_dict["name"] = message.name
            logging.debug(f"msg:({message.content})")
            if message.content is None:
                msg_dict["content"] = ""
            elif isinstance(message.content, str):
                content = message.content
                if message.role == "assistant":
                    # Strip OpenWebUI thought blocks entirely to prevent token fragmentation
                    content = re.sub(rf"{re.escape(thinking_start_token)}.*?{re.escape(thinking_end_token)}\n*", "", content, flags=re.DOTALL)
                msg_dict["content"] = content.strip()

            elif isinstance(message.content, list):
                new_content = []
                for item in message.content:
                    if isinstance(item, dict):
                        if item["type"] == "input_image":
                            images.append(item["image_url"])
                            # CRITICAL: Rename to standard HF schema so the Jinja template renders the token
                            new_content.append({"type": "image"})
                        elif item["type"] == "image_url":
                            if isinstance(item["image_url"], dict):
                                images.append(item["image_url"]["url"])
                            else:
                                images.append(item["image_url"])
                            # CRITICAL: Rename to standard HF schema so the Jinja template renders the token
                            new_content.append({"type": "image"})
                        elif item["type"] == "input_audio":
                            audio.append(item["input_audio"]["data"])
                            new_content.append({"type": "audio"})
                        elif item["type"] in ("text", "input_text"):
                            text_val = item.get("text", "")
                            if message.role == "assistant":
                                # Strip thought blocks from multimodal history
                                text_val = re.sub(rf"{re.escape(thinking_start_token)}.*?{re.escape(thinking_end_token)}\n*", "", text_val, flags=re.DOTALL)
                            if text_val.strip():
                                new_content.append(
                                    {"type": "text", "text": text_val.strip()}
                                )

                if len(new_content) == 1 and new_content[0].get("type") == "text":
                    msg_dict["content"] = new_content[0]["text"]
                elif len(new_content) == 0:
                    msg_dict["content"] = ""
                else:
                    msg_dict["content"] = new_content

            processed_messages.append(msg_dict)

        # --- GHOST PROMPT MATH GUARD ---
        math_guard = "Enclose inline math in \\( and \\). Enclose display math in \\[ and \\] on their own lines."
        has_system_prompt = False

        for msg in processed_messages:
            if msg.get("role") == "system":
                # Ensure content is a string before appending (handles multimodal lists)
                if isinstance(msg["content"], str):
                    msg["content"] += f"\n{math_guard}"
                elif isinstance(msg["content"], list):
                    msg["content"].append({"type": "text", "text": f"\n{math_guard}"})
                has_system_prompt = True
                break

        if not has_system_prompt:
            processed_messages.insert(0, {"role": "system", "content": math_guard})

        tools = None
        if hasattr(request, "tools") and request.tools:
            tools = request.tools

        tool_parser_type = None
        tool_module = None
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )

        chat_template = getattr(tokenizer, "chat_template", None)

        if chat_template is None and hasattr(tokenizer, "default_chat_template"):
            chat_template = tokenizer.default_chat_template

        if chat_template is not None:
            tokenizer.chat_template = chat_template

            tool_parser_type = _infer_tool_parser(chat_template)
            if tool_parser_type is not None:
                tool_module = load_tool_module(tool_parser_type)

        try:
            formatted_prompt = tokenizer.apply_chat_template(
                processed_messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        except Exception as e:
            logging.error(f"Native template failed, falling back: {e}")
            formatted_prompt = apply_chat_template(
                processor,
                config,
                processed_messages,
                num_images=len(images),
                num_audios=len(audio),
                tools=tools,
                **template_kwargs,
            )

        # --- LRU MULTI-CACHE MANAGER ---
        # 1. Extract or generate the session fingerprint
        current_session_id = request.session_id or request.user

        # Fallback: Hash the root message or generate a transient ID
        if not current_session_id:
            if len(request.messages) > 0:
                root_content = request.messages[0].content
                if isinstance(root_content, list):
                    # Handle multimodal root messages
                    root_content = str([item.get("text", "") for item in root_content if isinstance(item, dict)])
                current_session_id = str(hash(str(root_content)))
            else:
                current_session_id = f"transient_{uuid.uuid4().hex[:8]}"

        # Lazily initialize the manager and register the hook
        if model_cache.get("multi_cache_manager") is None:
            min_free = float(os.environ.get("LRU_MIN_FREE_RAM_PERCENT", "10.0"))
            manager = MultiCacheManager(min_free)
            register_allocation_hook(manager.check_and_evict)
            model_cache["multi_cache_manager"] = manager

        # Universal cache application (for both stream and non-stream)
        manager = model_cache["multi_cache_manager"]
        generation_kwargs["prompt_cache_state"] = manager.get_or_create(current_session_id)

        if request.stream:
            # Main chat interactions use streaming. Link the global state-aware cache.

            # Streaming response
            async def stream_generator():
                token_iterator = None
                try:
                    # Use stream_generate from utils
                    token_iterator = stream_generate(
                        model=model,
                        processor=processor,
                        prompt=formatted_prompt,
                        image=images,
                        audio=audio,
                        vision_cache=model_cache.get("vision_cache"),
                        **generation_kwargs,
                    )

                    full_output_text = ""
                    request_id = f"chatcmpl-{uuid.uuid4()}"
                    is_tool_call_sequence = False
                    translator = StreamingTranslator(thinking_start_token=thinking_start_token, thinking_end_token=thinking_end_token)

                    usage_stats = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "prompt_tps": 0.0,
                        "generation_tps": 0.0,
                        "peak_memory": 0.0,
                    }

                    for chunk in token_iterator:
                        if chunk is None or not hasattr(chunk, "text"):
                            logging.warning(f"Warning: Received unexpected chunk format: '{chunk}'")
                            continue

                        full_output_text += chunk.text

                        # Yield chunks in Server-Sent Events (SSE) format
                        usage_stats = {
                            "input_tokens": chunk.prompt_tokens,
                            "output_tokens": chunk.generation_tokens,
                            "total_tokens": chunk.prompt_tokens
                            + chunk.generation_tokens,
                            "prompt_tps": chunk.prompt_tps,
                            "generation_tps": chunk.generation_tps,
                            "peak_memory": chunk.peak_memory,
                        }

                        if not is_tool_call_sequence:
                            if "<|tool_call>" in translator.buffer + chunk.text:
                                translator.buffer += chunk.text
                                delta = translator.pre_tool_flush()
                                is_tool_call_sequence = True
                            else:
                                delta = translator.feed(chunk.text)

                            if delta:
                                choices = [
                                    ChatStreamChoice(
                                        delta=ChatMessage(
                                            role="assistant", content=delta
                                        )
                                    )
                                ]
                                chunk_data = ChatStreamChunk(
                                    id=request_id,
                                    created=int(time.time()),
                                    model=request.model,
                                    usage=usage_stats,
                                    choices=choices,
                                )
                                logging.debug(f"DEBUG STREAMING DELTA: {chunk_data}")
                                yield f"data: {chunk_data.model_dump_json()}\n\n"
                        else:
                            translator.buffer += chunk.text

                    if not is_tool_call_sequence:
                        final_delta = translator.flush()
                        if final_delta:
                            choices = [
                                ChatStreamChoice(
                                    delta=ChatMessage(
                                        role="assistant", content=final_delta
                                    )
                                )
                            ]
                            chunk_data = ChatStreamChunk(
                                id=request_id,
                                created=int(time.time()),
                                model=request.model,
                                usage=usage_stats,
                                choices=choices,
                            )
                            logging.debug(f"DEBUG STREAMING FINAL DELTA: {chunk_data}")
                            yield f"data: {chunk_data.model_dump_json()}\n\n"

                    tool_calls = _resolve_tool_calls(
                        full_output_text, tool_parser_type, tool_module, tools
                    )
                    has_tools = len(tool_calls.get("calls", [])) > 0

                    if has_tools:
                        final_content = None
                    else:
                        # Rescue the buffered text if the model started a tool sequence but failed to write valid JSON
                        if is_tool_call_sequence:
                            rescued = StreamingTranslator(thinking_start_token=thinking_start_token, thinking_end_token=thinking_end_token).translate(tool_calls["remaining_text"])
                            final_content = sanitize_strict_json(rescued) if rescued else ""
                        else:
                            final_content = ""

                    choices = [
                        ChatStreamChoice(
                            finish_reason="tool_calls" if has_tools else "stop",
                            delta=ChatMessage(
                                role="assistant",
                                content=final_content,
                                tool_calls=(
                                    tool_calls.get("calls") if has_tools else None
                                ),
                            ),
                        )
                    ]

                    chunk_data = ChatStreamChunk(
                        id=request_id,
                        created=int(time.time()),
                        model=request.model,
                        usage=usage_stats,
                        choices=choices,
                    )
                    logging.debug(f"DEBUG STREAMING CONTENT post token iterator: {chunk_data}")

                    yield f"data: {chunk_data.model_dump_json()}\n\n"
                    logging.info(f"Chat Stream Usage: {json.dumps(usage_stats)}")
                    yield "data: [DONE]\n\n"

                except Exception as e:
                    logging.error(f"Error during stream generation: {e}")
                    traceback.print_exc()
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"

                finally:
                    mx.clear_cache()
                    gc.collect()
                    logging.debug("Stream finished, cleared cache.")

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            # Non-streaming response
            try:
                # Use generate from generate.py
                gen_result = generate(
                    model=model,
                    processor=processor,
                    prompt=formatted_prompt,
                    image=images,
                    audio=audio,
                    verbose=False,  # Keep API output clean
                    vision_cache=model_cache.get("vision_cache"),
                    **generation_kwargs,
                )
                # Clean up resources
                mx.clear_cache()
                gc.collect()
                logging.debug("Generation finished, cleared cache.")

                usage_stats = UsageStats(
                    input_tokens=gen_result.prompt_tokens,
                    output_tokens=gen_result.generation_tokens,
                    total_tokens=gen_result.total_tokens,
                    prompt_tps=gen_result.prompt_tps,
                    generation_tps=gen_result.generation_tps,
                    peak_memory=gen_result.peak_memory,
                )
                logging.info(f"Chat Completion Usage: {usage_stats.model_dump_json()}")
                tool_calls = _resolve_tool_calls(
                    gen_result.text, tool_parser_type, tool_module, tools
                )
                has_tools = len(tool_calls.get("calls", [])) > 0

                if not has_tools:
                    translator = StreamingTranslator(thinking_start_token=thinking_start_token, thinking_end_token=thinking_end_token)
                    processed_chunk = translator.feed(tool_calls["remaining_text"])
                    final_content = processed_chunk + translator.flush()
                else:
                    final_content = None

                final_content = sanitize_strict_json(final_content) if final_content else None

                choices = [
                    ChatChoice(
                        finish_reason="tool_calls" if has_tools else "stop",
                        message=ChatMessage(
                            role="assistant",
                            content=final_content,
                            tool_calls=tool_calls.get("calls") if has_tools else None,
                        ),
                    )
                ]

                result = ChatResponse(
                    model=request.model, usage=usage_stats, choices=choices
                )
                logging.debug(f"DEBUG FINAL CONTENT: {result}")
                return result

            except Exception as e:
                logging.error(f"Error during generation: {e}")
                traceback.print_exc()
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions (like model loading failure)
        logging.error(f"HTTPException error in /generate endpoint: {http_exc}")
        raise http_exc
    except Exception as e:
        # Catch unexpected errors
        logging.error(f"Unexpected error in /generate endpoint: {e}")
        traceback.print_exc()
        mx.clear_cache()
        gc.collect()
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@app.get("/models", response_model=ModelsResponse)
@app.get("/v1/models", response_model=ModelsResponse, include_in_schema=False)
def models_endpoint():
    """
    Return list of locally downloaded MLX models.
    """

    files = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]

    def probably_mlx_lm(repo):
        if repo.repo_type != "model":
            return False
        if "main" not in repo.refs:
            return False
        file_names = {f.file_path.name for f in repo.refs["main"].files}
        return all(f in file_names for f in files)

    # Scan the cache directory for downloaded mlx models
    hf_cache_info = scan_cache_dir()
    downloaded_models = [repo for repo in hf_cache_info.repos if probably_mlx_lm(repo)]

    # Create a list of available models
    models = [
        {"id": repo.repo_id, "object": "model", "created": int(repo.last_modified)}
        for repo in downloaded_models
    ]

    response = {"object": "list", "data": models}

    return response


# MLX_VLM API endpoints


@app.get("/health")
async def health_check():
    """
    Check if the server is healthy and what model is loaded.
    """
    return {
        "status": "healthy",
        "loaded_model": model_cache.get("model_path", None),
        "loaded_adapter": model_cache.get("adapter_path", None),
    }


@app.post("/unload")
async def unload_model_endpoint():
    """
    Unload the currently loaded model from memory.
    """
    unloaded_info = {
        "model_name": model_cache.get("model_path", None),
        "adapter_name": model_cache.get("adapter_path", None),
    }

    if not unload_model_sync():  # Use the synchronous unload function
        return {"status": "no_model_loaded", "message": "No model is currently loaded"}

    return {
        "status": "success",
        "message": "Model unloaded successfully",
        "unloaded": unloaded_info,
    }


def main():
    parser = argparse.ArgumentParser(description="MLX VLM Http Server.")
    parser.add_argument(
        "--model",
        type=str,
        help="Optional path to the MLX model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_SERVER_HOST,
        help="Host for the HTTP server (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help="Port for the HTTP server (default: %(default)s)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading models from Hugging Face Hub.",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=DEFAULT_PREFILL_STEP_SIZE,
        help="Number of tokens to process per prefill step. "
        "Lower values reduce peak memory usage but may be slower. "
        "Try 512 or 256 if you hit GPU memory errors during prefill.",
    )
    parser.add_argument(
        "--kv-bits",
        type=float,
        default=0,
        help="Number of bits for KV cache quantization.",
    )
    parser.add_argument(
        "--kv-quant-scheme",
        type=str,
        choices=("uniform", "turboquant"),
        default=DEFAULT_KV_QUANT_SCHEME,
        help="KV cache quantization backend. Fractional --kv-bits values use "
        "TurboQuant automatically.",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=DEFAULT_KV_GROUP_SIZE,
        help="Group size for uniform KV cache quantization.",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=0,
        help="Maximum KV size for the prompt cache (tokens).",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
        help="Start index (of token) for the quantized KV cache.",
    )
    parser.add_argument(
        "--serialize-kv-quantization",
        action="store_true",
        default=False,
        help="Evaluate each layer's KV cache immediately after quantization "
        "to prevent graph explosion OOM spikes at the quantization threshold. "
        "Recommended for large models (e.g. 31B+) with long contexts.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload on file changes (development only). "
        "WARNING: watches the entire working directory — can cause excessive memory "
        "usage with large models in repos with frequent file changes.",
    )
    parser.add_argument(
        "--lru-min-free-ram-percent",
        type=float,
        default=10.0,
        help="Minimum free system memory percentage before LRU eviction of multi-caches (default: %(default)s).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="log_file",
        help="File to write logs to.",
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='DEBUG',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level (default: INFO)',
    )

    args = parser.parse_args()
    if args.trust_remote_code:
        os.environ["MLX_TRUST_REMOTE_CODE"] = "true"
    if args.model:
        os.environ["PRELOAD_MODEL"] = args.model
    if args.adapter_path:
        os.environ["PRELOAD_ADAPTER"] = args.adapter_path
    os.environ["PREFILL_STEP_SIZE"] = str(args.prefill_step_size)
    os.environ["KV_BITS"] = str(args.kv_bits)
    os.environ["KV_GROUP_SIZE"] = str(args.kv_group_size)
    os.environ["KV_QUANT_SCHEME"] = args.kv_quant_scheme
    os.environ["MAX_KV_SIZE"] = str(args.max_kv_size)
    os.environ["QUANTIZED_KV_START"] = str(args.quantized_kv_start)
    os.environ["SERIALIZE_KV_QUANTIZATION"] = str(args.serialize_kv_quantization).lower()
    os.environ["LRU_MIN_FREE_RAM_PERCENT"] = str(args.lru_min_free_ram_percent)
    if args.log_file != "<stdout>":
        logging.basicConfig(
            level=args.log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            filename=args.log_file,
            filemode='a'
        )
    else:
        logging.basicConfig(
            level=args.log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout
        )

    logging.debug(f"Command-line arguments: {args}")
    logging.debug(f"Environment variables: {os.environ}")
    logging.info("Starting MLX VLM Server with the following configuration:")
    logging.info(f"Host: {args.host}")
    logging.info(f"Port: {args.port}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Adapter Path: {args.adapter_path}")
    logging.info(f"Trust Remote Code: {args.trust_remote_code}")
    logging.info(f"Prefill Step Size: {args.prefill_step_size}")
    logging.info(f"KV Bits: {args.kv_bits}")
    logging.info(f"KV Group Size: {args.kv_group_size}")
    logging.info(f"KV Quant Scheme: {args.kv_quant_scheme}")
    logging.info(f"Max KV Size: {args.max_kv_size}")
    logging.info(f"Quantized KV Start: {args.quantized_kv_start}")
    logging.info(f"LRU Min Free RAM Percent: {args.lru_min_free_ram_percent}")
    logging.info(f"Log File: {args.log_file}")
    logging.info(f"Log Level: {args.log_level}")

    uvicorn.run(
        "mlx_vlm.server:app",
        host=args.host,
        port=args.port,
        workers=1,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
