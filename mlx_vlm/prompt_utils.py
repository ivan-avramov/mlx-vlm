from enum import Enum
from functools import partial
from typing import Any, Dict, List, Union

# ---------------------------------------------------------------------------
# Chat-template kwargs for prefix-cache-friendly multi-turn rendering.
#
# Some chat templates render the LATEST assistant turn differently from
# PRIOR assistant turns in the conversation history. For example, the
# unsloth-modified Qwen 3.6 template injects an empty `<think>\n\n</think>\n\n`
# block in the latest-assistant header but renders prior assistant turns as
# bare `<|im_start|>assistant\n{content}` — i.e. without the think wrapper.
#
# That asymmetry breaks prefix-cache reuse: turn N's CACHED tokens (which
# contain the think wrapper because it was in the prefill prompt) won't
# match turn N+1's RENDERED tokens (which omit the wrapper for the now-
# "prior" assistant turn). Hybrid models (DeltaNet, Mamba, etc.) — whose
# non-trimmable recurrent state can't be safely rewound across the divergence
# point without a snapshot — fall back to full re-prefill on every multi-turn
# request, costing ~10x in prefill time.
#
# Templates that are aware of this issue expose escape-hatch kwargs that
# force symmetric rendering across turns. This registry is the canonical
# list of such kwargs; the server merges them into apply_chat_template
# calls when a per-chat PromptCacheState is active.
#
# Invariant: every kwarg here has been verified to be a TRUE NO-OP on
# templates that don't reference it. Jinja's `{% if X is defined %}` guard
# is what makes that work — passing an undefined kwarg silently does
# nothing. Verification was done empirically by rendering identical
# multi-turn conversations with and without each kwarg on canonical
# templates (Qwen3-Next, Qwen3-32B): tokens were byte-identical.
#
# Adding a new entry:
#   1. Discover a chat template that breaks cache reuse via a new
#      asymmetric pattern.
#   2. Confirm the template exposes a kwarg that toggles symmetric
#      rendering (search the template source for `is defined` patterns).
#   3. Verify the new kwarg is a no-op on the official templates we care
#      about by tokenizing a multi-turn conversation with and without it
#      and checking the token sequences are identical.
#   4. Add the entry below with a comment naming the template family it
#      addresses.
# ---------------------------------------------------------------------------
CACHE_ALIGNMENT_KWARGS: Dict[str, Any] = {
    # Unsloth Qwen 3.x ports (e.g. `unsloth/Qwen3.6-27B-UD-MLX-6bit`):
    # latest-assistant header injects `<think>\n\n</think>\n\n` while prior
    # assistants render bare. `preserve_thinking=True` forces the same
    # wrapper on prior assistants. No-op on official Qwen templates
    # (Qwen3-Next, Qwen3-32B, Qwen3-30B-A3B-2507, Qwen3-235B-A22B-2507) —
    # they don't reference the kwarg.
    "preserve_thinking": True,
}


def get_cache_alignment_kwargs() -> Dict[str, Any]:
    """Return chat-template kwargs known to enable cache-friendly rendering.

    Callers pass these to `apply_chat_template` whenever a per-chat
    PromptCacheState is active. See CACHE_ALIGNMENT_KWARGS for the
    invariant and entry-addition criteria.
    """
    return dict(CACHE_ALIGNMENT_KWARGS)


# ---------------------------------------------------------------------------
# Thinking-format registry.
#
# Different model families wrap their reasoning blocks in different tag
# pairs. The streaming SSE state machine (`server.py:chat_completions_endpoint`)
# uses these tags to split outgoing tokens between `delta.reasoning`
# (thinking content) and `delta.content` (visible output). The same tags
# also drive `_compute_thinking_budget`, `_count_thinking_tag_tokens`,
# `_split_thinking`, and `_has_prefilled_opener`.
#
# Centralizing the tags here avoids the drift bug we hit before (multiple
# call sites hardcoding their own subsets — e.g. `_detect_thinking_format`
# recognized Gemma's `<|think|>` opener but `_split_thinking` and the
# streaming state machine looked only for `<think>`/`</think>` and
# `<|channel>thought`/`<channel|>`, so Gemma's reasoning block leaked
# into `delta.content` and the leading content token was eaten at the
# `</think>` boundary).
#
# Adding a new format:
#   1. Inspect the model's chat template + a sample completion to identify
#      the opener and closer literals.
#   2. List any prefixes the streaming state machine should buffer
#      mid-token to avoid leaking partial-tag bytes into delta.content
#      (typically the `<` or `<|` prefix of each tag).
#   3. Estimate `tag_token_count` — the per-turn token cost of the
#      opener+closer pair, used to debit `completion_tokens`. For most
#      formats this is 2 (one token each); gpt-oss splits the channel
#      marker, hence its higher count.
#   4. Add an entry to THINKING_FORMATS, ordered most-specific first.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThinkingFormat:
    """Tag literals for one model family's reasoning-block wrapper.

    All consumers (detection, streaming state machine, splitter, budget
    enforcer, prefilled-opener guard) read the same tag tuples here so
    they can't drift out of sync.
    """

    name: str
    openers: Tuple[str, ...]
    closers: Tuple[str, ...]
    # Substrings to suppress mid-token in the streaming state machine.
    # Without this, a single token like `</think>` (or its byte-prefix
    # split across two tokens) leaks into delta.content while the state
    # machine is still deciding whether it's a tag. Each entry should
    # be the longest unambiguous prefix of an opener or closer.
    partial_buffers: Tuple[str, ...]
    # Estimated token count consumed by one opener+closer pair on this
    # family's tokenizer. Used to subtract from `completion_tokens` so
    # the count reflects user-visible content only.
    tag_token_count: int


THINKING_FORMATS: Tuple[ThinkingFormat, ...] = (
    # Gemma 4: the chat template injects `<|think|>` at the system
    # block when enable_thinking=True (a global "thinking on" marker,
    # not a per-turn delimiter). The model's actual per-turn thinking
    # content is bracketed by `<|channel>thought ... <channel|>` —
    # same syntax as gpt-oss. Empirically the model also emits a bare
    # `</think>` as a closer in some samples, so both closer literals
    # are listed. The streaming state machine matches ANY listed opener
    # to enter thinking and ANY listed closer to exit.
    ThinkingFormat(
        name="gemma",
        openers=("<|think|>", "<|channel>thought"),
        closers=("</think>", "<channel|>"),
        partial_buffers=("<|think", "<|channel", "</think", "<channel"),
        tag_token_count=2,
    ),
    # Qwen 3.x family + most generic `<think>...</think>` thinkers.
    # Distinct opener from Gemma; the registry's first-match-wins
    # ordering puts Gemma above Qwen so prompts containing both
    # literals resolve to the more specific one.
    ThinkingFormat(
        name="qwen",
        openers=("<think>",),
        closers=("</think>",),
        partial_buffers=("<think", "</think"),
        tag_token_count=2,
    ),
    # gpt-oss / OpenAI-style channeled thought. Distinct from Gemma
    # in detection only (it doesn't use the `<|think|>` global marker).
    # Tag literals overlap with Gemma's per-turn syntax — both are
    # listed so the streaming state machine works either way once the
    # family is identified.
    ThinkingFormat(
        name="gpt-oss",
        openers=("<|channel>thought",),
        closers=("<channel|>",),
        partial_buffers=("<|channel>", "<channel"),
        tag_token_count=4,
    ),
)


def detect_thinking_format(text: str) -> Optional[ThinkingFormat]:
    """Match a chat-template prompt or model output against the registry.

    Returns the first format whose opener literal appears in `text`, or
    None if no known thinking format is detected. First-match-wins
    ordering is determined by the THINKING_FORMATS tuple — most-specific
    formats appear first.

    Used both at request time (against the rendered prompt) and at
    output time (against accumulated generation, e.g. by `_split_thinking`
    when the format wasn't already known).
    """
    for fmt in THINKING_FORMATS:
        for opener in fmt.openers:
            if opener in text:
                return fmt
    return None


class MessageFormat(Enum):
    """Enum for different message format types."""

    LIST_WITH_IMAGE = "list_with_image"
    LIST_WITH_IMAGE_FIRST = "list_with_image_first"
    LIST_WITH_IMAGE_URL_FIRST = "list_with_image_url_first"
    LIST_WITH_IMAGE_TYPE = "list_with_image_type"
    LIST_WITH_IMAGE_TYPE_TEXT = "list_with_image_type_text"
    LIST_WITH_IMAGE_TYPE_TEXT_IMAGE_LAST = "list_with_image_type_text_image_last"
    IMAGE_TOKEN = "image_token"
    IMAGE_TOKEN_PIPE = "image_token_pipe"
    START_IMAGE_TOKEN = "start_image_token"
    IMAGE_TOKEN_NEWLINE = "image_token_newline"
    NUMBERED_IMAGE_TOKENS = "numbered_image_tokens"
    PROMPT_ONLY = "prompt_only"
    PROMPT_WITH_IMAGE_TOKEN = "prompt_with_image_token"
    PROMPT_WITH_START_IMAGE_TOKEN = "prompt_with_start_image_token"
    VIDEO_WITH_TEXT = "video_with_text"


# Model configuration mapping
MODEL_CONFIG = {
    # List with image format models
    "jina_vlm": MessageFormat.IMAGE_TOKEN_PIPE,
    "jvlm": MessageFormat.IMAGE_TOKEN_PIPE,
    "idefics2": MessageFormat.LIST_WITH_IMAGE,
    "idefics3": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "lfm2-vl": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "lfm2_vl": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "aya_vision": MessageFormat.LIST_WITH_IMAGE,
    "cohere2_vision": MessageFormat.LIST_WITH_IMAGE,
    "paddleocr_vl": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "qwen2_vl": MessageFormat.LIST_WITH_IMAGE,
    "qwen2_5_vl": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "qwen3_vl": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "qwen3_vl_moe": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "qwen3_5": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "qwen3_5_moe": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "qwen3_omni_moe": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "minicpmo": MessageFormat.IMAGE_TOKEN,
    "mistral3": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "glm4v": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "glm4v_moe": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "glm_ocr": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "dots_ocr": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "ernie4_5_moe_vl": MessageFormat.LIST_WITH_IMAGE_URL_FIRST,
    "internvl_chat": MessageFormat.LIST_WITH_IMAGE_TYPE,
    "nemotron_h_nano_omni": MessageFormat.LIST_WITH_IMAGE_TYPE,
    "nemotronh_nano_omni_reasoning_v3": MessageFormat.LIST_WITH_IMAGE_TYPE,
    "kimi_vl": MessageFormat.LIST_WITH_IMAGE,
    "kimi_k25": MessageFormat.LIST_WITH_IMAGE,
    "gemma3": MessageFormat.START_IMAGE_TOKEN,
    "gemma3n": MessageFormat.LIST_WITH_IMAGE_TYPE_TEXT,
    "gemma4": MessageFormat.LIST_WITH_IMAGE_TYPE_TEXT,
    "llama4": MessageFormat.LIST_WITH_IMAGE,
    "smolvlm": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "llava": MessageFormat.LIST_WITH_IMAGE,
    "llava_next": MessageFormat.LIST_WITH_IMAGE,
    "granite_vision": MessageFormat.LIST_WITH_IMAGE,
    "granite4_vision": MessageFormat.LIST_WITH_IMAGE,
    "mllama": MessageFormat.LIST_WITH_IMAGE,
    "pixtral": MessageFormat.LIST_WITH_IMAGE_TYPE_TEXT,
    "molmo2": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "molmo_point": MessageFormat.LIST_WITH_IMAGE_FIRST,
    # Token-based models
    "llava-qwen2": MessageFormat.IMAGE_TOKEN_NEWLINE,
    "llava_qwen2": MessageFormat.IMAGE_TOKEN_NEWLINE,  # fastvlm
    "bunny-llama": MessageFormat.IMAGE_TOKEN_NEWLINE,
    "phi3_v": MessageFormat.NUMBERED_IMAGE_TOKENS,
    "phi4mm": MessageFormat.NUMBERED_IMAGE_TOKENS,
    "multi_modality": MessageFormat.IMAGE_TOKEN,
    "deepseek_vl_v2": MessageFormat.IMAGE_TOKEN_NEWLINE,
    "deepseekocr_2": MessageFormat.IMAGE_TOKEN_NEWLINE,
    "deepseekocr": MessageFormat.IMAGE_TOKEN_NEWLINE,
    "phi4-siglip": MessageFormat.IMAGE_TOKEN_NEWLINE,
    "hunyuan_vl": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "youtu_vl": MessageFormat.LIST_WITH_IMAGE_FIRST,
    # Prompt-only models
    "florence2": MessageFormat.PROMPT_ONLY,
    "molmo": MessageFormat.PROMPT_ONLY,
    "moondream3": MessageFormat.PROMPT_ONLY,
    "falcon_ocr": MessageFormat.PROMPT_ONLY,
    "paligemma": MessageFormat.PROMPT_WITH_IMAGE_TOKEN,
}

# Models that don't support multi-image
SINGLE_IMAGE_ONLY_MODELS = {
    "llava_next",
    "llava-qwen2",
    "bunny-llama",
    "paligemma",
    "multi_modality",
    "mllama",
    "falcon_ocr",
}


def extract_text_from_content(content: Any) -> str:
    """
    Extract text from multimodal content.

    When using OpenAI-compatible multimodal API, content can be a list like:
    [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]

    This function extracts only the text parts, preventing base64 image data
    from being tokenized as text (which would cause token explosion).

    Args:
        content: Either a string or a list of content items

    Returns:
        A string containing only the text content
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                # Extract text from text-type items
                if item_type in ("text", "input_text"):
                    text = item.get("text", "") or item.get("content", "")
                    if text:
                        text_parts.append(text)
                # Skip image_url, input_image, input_audio - these are handled separately
        return " ".join(text_parts).strip() if text_parts else ""

    # Fallback: convert to string (shouldn't happen in normal usage)
    return str(content) if content else ""


def _get_role_content(item: Any) -> Union[tuple[str, Any], None]:
    """Return (role, content) for a message-like item (dict or object with .role/.content), else None."""
    if isinstance(item, dict):
        return item.get("role", "user"), item.get("content")
    if hasattr(item, "role") and hasattr(item, "content"):
        return getattr(item, "role", "user"), getattr(item, "content", "")
    return None


class MessageBuilder:
    """Builder for creating messages in various formats."""

    @staticmethod
    def text_message(text: str) -> Dict[str, str]:
        """Create a simple text message."""
        return {"type": "text", "text": text, "content": text}

    @staticmethod
    def content_message(content: str) -> Dict[str, str]:
        """Create a content-type text message."""
        return {"type": "text", "text": content, "content": content}

    @staticmethod
    def image_message() -> Dict[str, str]:
        """Create an image message."""
        return {"type": "image"}

    @staticmethod
    def image_url_message() -> Dict[str, str]:
        """Create an image_url message (for models like ERNIE that expect this format)."""
        return {"type": "image_url"}

    @staticmethod
    def audio_message() -> Dict[str, str]:
        """Create an audio message."""
        return {"type": "audio"}

    @staticmethod
    def video_message(
        video_path: str, max_pixels: int = 224 * 224, fps: int = 1
    ) -> Dict[str, Any]:
        """Create a video message."""
        return {
            "type": "video",
            "video": video_path,
            "max_pixels": max_pixels,
            "fps": fps,
        }


class MessageFormatter:
    """Handles formatting messages for different model types."""

    def __init__(self, model_name: str):
        self.model_name = model_name.lower()
        self.format_type = MODEL_CONFIG.get(self.model_name)
        # Unknown models (e.g. text-only models loaded via mlx_lm fallback)
        # get plain text formatting — no image/audio tokens to insert.
        if not self.format_type:
            self.format_type = None

    def format_message(
        self,
        prompt: str,
        role: str = "user",
        skip_image_token: bool = False,
        skip_audio_token: bool = False,
        num_images: int = 1,
        num_audios: int = 1,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """Format a message based on the model type."""

        # Check multi-image support
        if num_images > 1 and self.model_name in SINGLE_IMAGE_ONLY_MODELS:
            raise ValueError(
                f"Model {self.model_name} does not support multi-image chat. "
                f"Please only use 1 image."
            )

        # Handle video format for specific models
        if self.model_name in [
            "qwen2_vl",
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
            "qwen3_omni_moe",
            "gemma4",
        ] and kwargs.get("video"):
            return self._format_video_message(prompt, role, **kwargs)

        # Route to appropriate formatter
        formatter_map = {
            MessageFormat.LIST_WITH_IMAGE: self._format_list_with_image,
            MessageFormat.LIST_WITH_IMAGE_FIRST: partial(
                self._format_list_with_image, image_first=True
            ),
            MessageFormat.LIST_WITH_IMAGE_URL_FIRST: partial(
                self._format_list_with_image, image_first=True, use_image_url=True
            ),
            MessageFormat.LIST_WITH_IMAGE_TYPE: self._format_list_with_image_type,
            MessageFormat.LIST_WITH_IMAGE_TYPE_TEXT: partial(
                self._format_list_with_image_type, message_type="text"
            ),
            MessageFormat.LIST_WITH_IMAGE_TYPE_TEXT_IMAGE_LAST: partial(
                self._format_list_with_image_type,
                message_type="text",
                image_first=False,
            ),
            MessageFormat.IMAGE_TOKEN: partial(
                self._format_with_token, token="<image>"
            ),
            MessageFormat.IMAGE_TOKEN_PIPE: partial(
                self._format_with_token, token="<|image|>"
            ),
            MessageFormat.START_IMAGE_TOKEN: partial(
                self._format_with_token, token="<start_of_image>", image_first=False
            ),
            MessageFormat.IMAGE_TOKEN_NEWLINE: partial(
                self._format_with_token, token="<image>\n"
            ),
            MessageFormat.NUMBERED_IMAGE_TOKENS: self._format_numbered_tokens,
            MessageFormat.PROMPT_ONLY: lambda *args, **kw: prompt,
            MessageFormat.PROMPT_WITH_IMAGE_TOKEN: lambda *args, **kw: "<image>"
            * num_images
            + prompt,
            MessageFormat.PROMPT_WITH_START_IMAGE_TOKEN: lambda *args, **kw: prompt
            + "<start_of_image>" * num_images,
            MessageFormat.VIDEO_WITH_TEXT: self._format_video_message,
        }

        formatter = formatter_map.get(self.format_type)
        if formatter is None:
            # Unknown model type — return plain text message
            return {"role": role, "content": prompt}
        return formatter(
            prompt,
            role,
            skip_image_token,
            skip_audio_token,
            num_images,
            num_audios,
            **kwargs,
        )

    def _format_list_with_image(
        self,
        prompt: str,
        role: str,
        skip_image_token: bool,
        skip_audio_token: bool,
        num_images: int,
        num_audios: int,
        image_first: bool = False,
        use_image_url: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Format as a list with image tokens."""
        content = [MessageBuilder.text_message(prompt)]

        if role == "user" and not skip_image_token and num_images > 0:
            image_builder = (
                MessageBuilder.image_url_message
                if use_image_url
                else MessageBuilder.image_message
            )
            image_tokens = [image_builder()] * num_images
            content = image_tokens + content if image_first else content + image_tokens

        return {"role": role, "content": content}

    def _format_list_with_image_type(
        self,
        prompt: str,
        role: str,
        skip_image_token: bool,
        skip_audio_token: bool,
        num_images: int,
        num_audios: int,
        message_type: str = "content",
        image_first: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Format as a list with typed messages."""
        msg_func = (
            MessageBuilder.content_message
            if message_type == "content"
            else MessageBuilder.text_message
        )
        message = {"role": role, "content": [msg_func(prompt)]}

        if role == "user":
            if not skip_image_token and num_images > 0:
                message["content"] = (
                    [MessageBuilder.image_message()] * num_images + message["content"]
                    if image_first
                    else message["content"]
                    + [MessageBuilder.image_message()] * num_images
                )
            if not skip_audio_token and num_audios > 0:
                message["content"] = (
                    message["content"] + [MessageBuilder.audio_message()] * num_audios
                )

        if role == "assistant":
            message["content"] = message["content"][0].get(
                "content", message["content"][0].get("text")
            )

        return message

    def _format_with_token(
        self,
        prompt: str,
        role: str,
        skip_image_token: bool,
        skip_audio_token: bool,
        num_images: int,
        num_audios: int,
        token: str,
        image_first: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Format with image tokens in the text."""
        content = prompt

        if role == "user" and not skip_image_token and num_images > 0:
            prefix = token * num_images
            content = f"{prefix}{content}" if image_first else f"{content}{prefix}"

        if role == "user" and not skip_audio_token and num_audios > 0:
            audio_prefix = "".join([f"<|audio_{i+1}|>" for i in range(num_audios)])
            content = f"{audio_prefix}{content}"

        return {"role": role, "content": content}

    def _format_numbered_tokens(
        self,
        prompt: str,
        role: str,
        skip_image_token: bool,
        skip_audio_token: bool,
        num_images: int,
        num_audios: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """Format with numbered image/audio tokens.

        Order follows Phi-4 convention: <|image_N|> before <|audio_N|>.
        """
        content = prompt

        if role == "user":
            # Build prefix: images first, then audio (matches HF model format)
            prefix_parts = []
            if not skip_image_token and num_images > 0:
                prefix_parts.append(
                    "".join([f"<|image_{i+1}|>" for i in range(num_images)])
                )
            if not skip_audio_token and num_audios > 0:
                prefix_parts.append(
                    "".join([f"<|audio_{i+1}|>" for i in range(num_audios)])
                )
            if prefix_parts:
                content = f"{''.join(prefix_parts)}{content}"

        return {"role": role, "content": content}

    def _format_video_message(
        self,
        prompt: str,
        role: str = "user",
        skip_image_token: bool = False,
        skip_audio_token: bool = False,
        num_images: int = 0,
        num_audios: int = 0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Format a video message with text.

        Accepts either a single path in ``video=`` or a list of paths; emits
        one ``{"type": "video", ...}`` content item per video, followed by the
        text. ``fps`` may be a scalar (applied to all) or a list of the same
        length as the videos.
        """
        videos = kwargs["video"]
        if not isinstance(videos, list):
            videos = [videos]

        max_pixels = kwargs.get("max_pixels", 224 * 224)
        fps = kwargs.get("fps", 1)
        fps_list = fps if isinstance(fps, list) else [fps] * len(videos)
        if len(fps_list) != len(videos):
            raise ValueError(
                f"Got {len(fps_list)} fps values for {len(videos)} videos."
            )

        content = [
            MessageBuilder.video_message(v, max_pixels, f)
            for v, f in zip(videos, fps_list)
        ]
        content.append(MessageBuilder.text_message(prompt))
        return {"role": role, "content": content}


def get_message_json(
    model_name: str,
    prompt: str,
    role: str = "user",
    skip_image_token: bool = False,
    skip_audio_token: bool = False,
    num_images: int = 0,
    num_audios: int = 0,
    **kwargs,
) -> Union[str, Dict[str, Any]]:
    """
    Get the appropriate JSON message based on the specified model.

    Args:
        model_name: The model for which to generate the message
        prompt: The text prompt to be included in the message
        role: The role of the message (default: "user")
        skip_image_token: Whether to skip adding image tokens
        skip_audio_token: Whether to skip adding audio tokens
        num_images: Number of image tokens to add
        num_audios: Number of audio tokens to add
        **kwargs: Additional arguments (e.g., video path, max_pixels, fps)

    Returns:
        A dictionary or string representing the message for the specified model
    """
    formatter = MessageFormatter(model_name)

    return formatter.format_message(
        prompt,
        role,
        skip_image_token,
        skip_audio_token,
        num_images,
        num_audios,
        **kwargs,
    )


def get_chat_template(
    processor,
    messages: List[Dict[str, Any]],
    add_generation_prompt: bool,
    tokenize: bool = False,
    **kwargs,
) -> Any:
    """Apply chat template using processor's tokenizer."""

    def _get_image_token() -> str:
        if processor is None:
            return "<image>"

        image_token = getattr(processor, "image_token", None)
        if isinstance(image_token, str) and image_token:
            return image_token

        tokenizer = getattr(processor, "tokenizer", None)
        image_token = getattr(tokenizer, "image_token", None)
        if isinstance(image_token, str) and image_token:
            return image_token

        return "<image>"

    def _flatten_content(content: Any, image_token: str) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            audio_marker = kwargs.get("audio_token", "<audio>")
            multimodal_markers = {image_token, audio_marker, "<audio>", "<video>"}
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type in ("text", "input_text"):
                        text = item.get("text", "") or item.get("content", "")
                        if text:
                            parts.append(str(text))
                    elif item_type in ("image", "image_url", "input_image"):
                        parts.append(image_token)
                    elif item_type in ("audio", "input_audio"):
                        parts.append("<audio>")
                    elif item_type == "video":
                        parts.append("<video>")
                    else:
                        text = item.get("text", "") or item.get("content", "")
                        if text:
                            parts.append(str(text))
                elif item is not None:
                    parts.append(str(item))

            stitched_parts = []
            prev_is_marker = False
            for part in parts:
                if not part:
                    continue
                current_is_marker = part in multimodal_markers
                if prev_is_marker and not current_is_marker and not part[0].isspace():
                    stitched_parts.append(" ")
                stitched_parts.append(part)
                prev_is_marker = current_is_marker

            return "".join(stitched_parts).strip()

        if isinstance(content, dict):
            text = content.get("text", "") or content.get("content", "")
            return str(text) if text else ""

        return str(content) if content is not None else ""

    def _messages_to_plain_prompt() -> str:
        image_token = _get_image_token()
        normalized = []

        for message in messages:
            if isinstance(message, str):
                normalized.append({"role": "user", "content": message})
                continue

            if isinstance(message, dict):
                normalized.append(
                    {
                        "role": message.get("role", "user"),
                        "content": _flatten_content(
                            message.get("content", ""), image_token
                        ),
                    }
                )
                continue

            normalized.append({"role": "user", "content": str(message)})

        if not normalized:
            return ""

        if len(normalized) == 1 and normalized[0]["role"] == "user":
            return normalized[0]["content"]

        lines = []
        for message in normalized:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role in ("system", "user", "assistant", "tool"):
                prefix = role.capitalize()
                lines.append(f"{prefix}: {content}" if content else f"{prefix}:")
            else:
                lines.append(content if content else "")

        if add_generation_prompt:
            lines.append("Assistant:")

        return "\n".join(lines).strip()

    def _missing_template_error(error: Exception) -> bool:
        message = str(error)
        return (
            "chat_template is not set" in message
            or "no template argument was passed" in message
        )

    chat_template_override = kwargs.get("chat_template", None)

    try:
        template_processor = None
        if (
            processor is not None
            and hasattr(processor, "apply_chat_template")
            and (
                chat_template_override is not None
                or getattr(processor, "chat_template", None) is not None
            )
        ):
            template_processor = processor
        elif (
            processor is not None
            and hasattr(processor, "tokenizer")
            and hasattr(processor.tokenizer, "apply_chat_template")
            and (
                chat_template_override is not None
                or getattr(processor.tokenizer, "chat_template", None) is not None
            )
        ):
            template_processor = processor.tokenizer
        elif processor is not None and hasattr(processor, "apply_chat_template"):
            # Handles tokenizers passed directly.
            if (
                chat_template_override is not None
                or getattr(processor, "chat_template", None) is not None
            ):
                template_processor = processor

        if template_processor is None:
            return _messages_to_plain_prompt()

        try:
            return template_processor.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
        except ValueError as e:
            if chat_template_override is None and _missing_template_error(e):
                return _messages_to_plain_prompt()
            raise
    except AttributeError:
        return _messages_to_plain_prompt()


def apply_chat_template(
    processor,
    config: Union[Dict[str, Any], Any],
    prompt: Union[str, Dict[str, Any], List[Any]],
    add_generation_prompt: bool = True,
    return_messages: bool = False,
    num_images: int = 0,
    num_audios: int = 0,
    **kwargs,
) -> Union[List[Dict[str, Any]], str, Any]:
    """
    Apply chat template to prompts.

    Args:
        processor: The processor with chat template functionality
        config: Model configuration
        prompt: Single prompt string, dict, or list of prompts
        add_generation_prompt: Whether to add generation prompt
        return_messages: Whether to return messages list instead of template
        num_images: Number of images in the input
        num_audios: Number of audio files in the input
        **kwargs: Additional arguments for message formatting

    Returns:
        Formatted messages or chat template
    """
    config = config if isinstance(config, dict) else config.__dict__
    model_type = config["model_type"]

    # Build messages from prompts
    messages = []

    if isinstance(prompt, str):
        # Single string prompt
        messages.append(
            get_message_json(
                model_type,
                prompt,
                num_images=num_images,
                num_audios=num_audios,
                **kwargs,
            )
        )
    elif isinstance(prompt, dict):
        # Single dict prompt
        content = extract_text_from_content(prompt["content"])
        messages.append(
            get_message_json(
                model_type,
                content,
                prompt["role"],
                num_images=num_images,
                num_audios=num_audios,
                **kwargs,
            )
        )
    elif isinstance(prompt, list):
        # List of prompts — find the last user message to place image/audio tokens
        last_user_idx = -1
        for i, p in enumerate(prompt):
            if isinstance(p, str):
                last_user_idx = i
            elif (rc := _get_role_content(p)) is not None:
                if rc[0] not in ("system", "assistant", "tool"):
                    last_user_idx = i

        for i, p in enumerate(prompt):
            if isinstance(p, str):
                is_target = i == last_user_idx
                messages.append(
                    get_message_json(
                        model_type,
                        p,
                        skip_image_token=not is_target,
                        skip_audio_token=not is_target,
                        num_images=num_images,
                        num_audios=num_audios,
                        **kwargs,
                    )
                )
            elif (role_content := _get_role_content(p)) is not None:
                role, content = role_content
                # Tool-calling messages: pass through as-is to preserve
                # tool_calls, tool_call_id, name for the Jinja template.
                has_tool_metadata = isinstance(p, dict) and (
                    "tool_calls" in p or "tool_call_id" in p or role == "tool"
                )
                if has_tool_metadata:
                    messages.append(p)
                else:
                    # Handle multimodal content: extract only text, skip image/audio URLs
                    content = extract_text_from_content(content)
                    is_target = i == last_user_idx
                    messages.append(
                        get_message_json(
                            model_type,
                            content,
                            role,
                            skip_image_token=not is_target
                            or role in ["system", "assistant"],
                            skip_audio_token=not is_target
                            or role in ["system", "assistant"],
                            num_images=num_images,
                            num_audios=num_audios,
                            **kwargs,
                        )
                    )

    if return_messages:
        return messages

    # Some models only need the last message
    if model_type in ["paligemma", "molmo", "florence2", "falcon_ocr"]:
        return messages[-1]

    return get_chat_template(processor, messages, add_generation_prompt, **kwargs)
