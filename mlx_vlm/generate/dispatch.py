import argparse
import codecs
import json
import logging
import os  # Fork: MLX_VLM_LOG_NAME override on the logger below (5e9b9503)
import time
from collections.abc import Sequence
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer

from .. import apc as _apc
from ..kv_quant import from_legacy as kv_quant_from_legacy  # Fork: kv-quant config
from ..models import cache
from ..prompt_utils import (  # Fork: all but apply_chat_template are fork-only (1c3f1e50, 7e3477b0) — the THINKING_FORMATS registry
    apply_chat_template,
    cached_special_token_encode,
    detect_thinking_format,
    prompt_is_inside_thinking,
)
from ..speculative.utils import format_speculative_stats
from ..tokenizer_utils import make_streaming_detokenizer
from ..utils import (
    StoppingCriteria,
    ThinkingBudgetCriteria,
    load,
    prepare_inputs,
    should_add_special_tokens,
)
from .common import (  # Fork: the snapshot-ring helpers (_capture/_restore_*, _rotating_rewind_safe, _has_non_trimmable) and _get_generation_stream are fork-only; they replace upstream's _prefix_cache_trim_amount/_cache_fully_retained
    DEFAULT_DIFFUSION_MAX_DENOISING_STEPS,
    DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH,
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_P,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_REPETITION_CONTEXT_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    GenerationResult,
    _capture_rotating_layers_for_snapshot,
    _compute_anchor_before_latest_user_offset,
    _get_generation_stream,
    _has_non_trimmable,
    _is_rotating_kv_layer,
    _restore_arrays_layers_from_snapshots,
    _restore_deltanet_state,
    _restore_rotating_layers_from_snapshots,
    _rotating_rewind_safe,
    _trim_cache,
    wired_limit,
)
from .image import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IMAGE_STEPS,
    DEFAULT_IMAGE_TASK,
    run_image_generation_cli,
)
from .video_generation import DEFAULT_VIDEO_STEPS, run_video_generation_cli

# Fork: MLX_VLM_LOG_NAME (5e9b9503) so an embedding host can re-root the logger
# tree; upstream hardcodes "mlx_vlm.generate". Same name when the var is unset.
logger = logging.getLogger(f"{os.environ.get('MLX_VLM_LOG_NAME', 'mlx_vlm')}.generate")

DEFAULT_MODEL_PATH = "mlx-community/nanoLLaVA-1.5-8bit"
DEFAULT_IMAGE = None
DEFAULT_AUDIO = None
DEFAULT_VIDEO = None
DEFAULT_PROMPT = "What are these?"
DEFAULT_SEED = 0
DEFAULT_THINKING_START_TOKEN = "<think>"
DEFAULT_THINKING_END_TOKEN = "</think>"


def parse_arguments():
    # Fork: PURE ADDITIONS to upstream's parser — 23 added lines, 0 removed. The
    # "suffix" choice on --draft-kind plus --suffix-min-match and --draft-cooldown
    # (863441c9, drafter-free n-gram speculation, which upstream does not have), and
    # the --draft-block-size help text extended to describe its meaning for that
    # kind. Every other flag here is upstream's, byte-identical.
    parser = argparse.ArgumentParser(
        description="Generate text, an image, or a video with a supported model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--output-modality",
        type=str,
        choices=("text", "image", "video"),
        default="text",
        help=(
            "Generate text with a VLM, an image with a supported image model, "
            "or a video with a supported video model."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for image or video generation.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=("generate", "edit"),
        default=DEFAULT_IMAGE_TASK,
        help="Image task to run when --output-modality image is selected.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default=None,
        help=(
            "Output size as WIDTHxHEIGHT. Image generation defaults to "
            f"{DEFAULT_IMAGE_SIZE}; image editing defaults to the first reference "
            "image size, and video uses the model default when omitted."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Number of inference steps. Defaults to "
            f"{DEFAULT_IMAGE_STEPS} for images and {DEFAULT_VIDEO_STEPS} for videos."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "PRNG seed for reproducible sampling and diffusion canvas init. "
            "Image and video generation default to a random 32-bit seed."
        ),
    )
    parser.add_argument(
        "--workflow",
        choices=("t2va", "fl2va", "ref2va"),
        default=None,
        help=(
            "Video-generation workflow. Inferred from --image/--last-image or "
            "--reference when omitted."
        ),
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Requested number of generated video frames.",
    )
    parser.add_argument(
        "--last-image",
        type=str,
        default=None,
        help="Last-frame conditioning image for FL2VA video generation.",
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=None,
        metavar="KIND=PATH",
        help=(
            "Ordered Ref2VA reference; KIND is image, video, or audio. Repeat "
            "the argument to preserve semantic reference order."
        ),
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="Classifier-free guidance for image generation/editing.",
    )
    parser.add_argument(
        "--prompt-expansion-model",
        type=str,
        default=None,
        help=(
            "Text model path or Hugging Face repo used to expand plain image "
            "prompts into Ideogram 4 JSON captions."
        ),
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="The path to the adapter weights.",
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        default=DEFAULT_IMAGE,
        help="URL or path of the image to process.",
    )
    parser.add_argument(
        "--audio",
        type=str,
        nargs="+",
        default=DEFAULT_AUDIO,
        help="URL or path of the audio to process.",
    )
    parser.add_argument(
        "--video",
        type=str,
        nargs="+",
        default=DEFAULT_VIDEO,
        help="URL or path of the video to process.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames-per-second to sample from --video.",
    )
    parser.add_argument(
        "--video-max-frames",
        type=int,
        default=16,
        help="Cap on frames sent when video falls back to ordered images "
        "(long clips are re-sampled evenly to this count).",
    )
    parser.add_argument(
        "--resize-shape",
        type=int,
        nargs="+",
        default=None,
        help="Resize shape for the image.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default=DEFAULT_PROMPT,
        help="Message to be processed by the model.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System message for the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--max-denoising-steps",
        type=int,
        default=None,
        help=(
            "Maximum denoising steps for diffusion generation. "
            "Default: the checkpoint's generation config (typically "
            f"{DEFAULT_DIFFUSION_MAX_DENOISING_STEPS}). Adaptive stopping "
            "usually converges canvases earlier; set lower to hard-cap "
            "throughput."
        ),
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=None,
        help="Block length for diffusion text generation.",
    )
    parser.add_argument(
        "--num-to-transfer",
        type=int,
        default=None,
        help="Target number of masked tokens to transfer per diffusion denoising step.",
    )
    parser.add_argument(
        "--max-transfer-per-step",
        type=int,
        default=None,
        help="Maximum confident masked tokens to transfer per denoising step.",
    )
    parser.add_argument(
        "--editing-threshold",
        type=float,
        default=None,
        help="Confidence threshold for diffusion post-fill token edits.",
    )
    parser.add_argument(
        "--max-post-steps",
        type=int,
        default=None,
        help="Maximum diffusion post-fill editing steps per block.",
    )
    parser.add_argument(
        "--stability-steps",
        type=int,
        default=None,
        help="Stop post-fill refinement after this many stable no-edit steps.",
    )
    parser.add_argument(
        "--diffusion-full-canvas",
        action="store_true",
        help=(
            "Use the checkpoint canvas length for diffusion generation even when "
            "--max-tokens requests a partial block."
        ),
    )
    parser.add_argument(
        "--diffusion-min-canvas-length",
        type=int,
        default=None,
        help=(
            "Minimum active canvas length for diffusion partial blocks. "
            f"Default: {DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH}."
        ),
    )
    parser.add_argument(
        "--diffusion-max-canvas-length",
        type=int,
        default=None,
        help=(
            "Maximum active canvas length for diffusion generation. Default: the "
            "checkpoint canvas length; set lower to trade quality for "
            "throughput."
        ),
    )
    parser.add_argument(
        "--diffusion-sampler",
        choices=["entropy-bound", "confidence-threshold"],
        default="confidence-threshold",
        help=(
            "Canvas update sampler for diffusion generation. Use entropy-bound "
            "for reference-style denoising; confidence-threshold is faster for "
            "quantized block-diffusion checkpoints."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Token probability threshold for diffusion confidence transfer. "
            f"Default: {DEFAULT_DIFFUSION_CONFIDENCE_THRESHOLD:g} for "
            "confidence-threshold sampling; "
            "masked-diffusion models use their checkpoint reference defaults."
        ),
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=None,
        help="Lowest token probability threshold for masked diffusion transfer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Nucleus sampling: keep the smallest set of tokens whose "
        "probabilities sum to this. 1.0 disables it.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Keep only the k most probable tokens. 0 disables it.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=DEFAULT_MIN_P,
        help="Drop tokens whose probability is below this fraction of the "
        "most probable token's. 0 disables it.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Penalty factor for previously generated tokens.",
    )
    parser.add_argument(
        "--repetition-context-size",
        type=int,
        default=DEFAULT_REPETITION_CONTEXT_SIZE,
        help="Number of recent generated tokens used for repetition penalty.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Additive penalty for tokens that already appeared.",
    )
    parser.add_argument(
        "--presence-context-size",
        type=int,
        default=DEFAULT_REPETITION_CONTEXT_SIZE,
        help="Number of recent generated tokens used for presence penalty.",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=None,
        help="Additive penalty scaled by token frequency.",
    )
    parser.add_argument(
        "--frequency-context-size",
        type=int,
        default=DEFAULT_REPETITION_CONTEXT_SIZE,
        help="Number of recent generated tokens used for frequency penalty.",
    )
    parser.add_argument("--chat", action="store_true", help="Chat in multi-turn style.")
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable detailed output and progress bars. By default only the final "
            "result is printed."
        ),
    )
    parser.add_argument(
        "--eos-tokens",
        type=str,
        nargs="+",
        default=None,
        help="EOS tokens to add to the tokenizer.",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Maximum KV size for the prompt cache.",
    )
    parser.add_argument(
        "--kv-bits",
        type=float,
        default=None,
        help="Number of bits to quantize the KV cache to.",
    )
    parser.add_argument(
        "--kv-key-bits",
        type=float,
        default=None,
        help="Override the TurboQuant key bit-width (defaults to floor(--kv-bits)).",
    )
    parser.add_argument(
        "--kv-value-bits",
        type=float,
        default=None,
        help="Override the TurboQuant value bit-width (defaults to ceil(--kv-bits)).",
    )
    parser.add_argument(
        "--kv-key-scheme",
        type=str,
        choices=("uniform", "turboquant"),
        default=None,
        help="Override the KV quantization backend for keys only.",
    )
    parser.add_argument(
        "--kv-value-scheme",
        type=str,
        choices=("uniform", "turboquant"),
        default=None,
        help="Override the KV quantization backend for values only.",
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
        "--quantized-kv-start",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
        help="Start index for the quantized KV cache.",
    )
    parser.add_argument(
        "--skip-special-tokens",
        action="store_true",
        help="Skip special tokens in the detokenizer.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force download the model from Hugging Face.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The specific model version to use (branch, tag, commit).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the model.",
    )
    parser.add_argument(
        "--quantize-activations",
        "-qa",
        action="store_true",
        help="Enable activation quantization for QQLinear layers. "
        "Only supported for models quantized with 'nvfp4' or 'mxfp8' modes.",
    )
    parser.add_argument(
        "--processor-kwargs",
        type=json.loads,
        default={},
        help="Extra processor kwargs as JSON. "
        'Example: --processor-kwargs \'{"cropping": false, "max_patches": 3}\'',
    )
    parser.add_argument(
        "--gen-kwargs",
        type=json.loads,
        default={},
        help="Extra generation kwargs as JSON. "
        "Example: --gen-kwargs '{\"custom_arg\": true}'",
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
        "--draft-model",
        type=str,
        default=None,
        help="Speculative drafter path or HF id (e.g. z-lab/Qwen3.5-4B-DFlash).",
    )
    parser.add_argument(
        "--draft-kind",
        type=str,
        default=None,
        choices=["dflash", "eagle3", "mtp", "suffix"],
        help="Drafter family. Supported: 'dflash' (Qwen3.5 DFlash), "
        "'eagle3' (Speculators/SGLang EAGLE-3), "
        "'mtp' (Gemma 4 Multi-Token Prediction / Assistant model), "
        "'suffix' (drafter-free n-gram / prompt-lookup; no --draft-model). "
        "Default: auto-detected from the drafter's HF model_type.",
    )
    parser.add_argument(
        "--draft-block-size",
        type=int,
        default=None,
        help="Override the drafter's configured block size. For "
        "--draft-kind suffix, this is the maximum draft (proposal) length.",
    )
    parser.add_argument(
        "--suffix-min-match",
        type=int,
        default=2,
        help="Minimum n-gram match length for drafter-free suffix decoding "
        "(--draft-kind suffix). Default: 2.",
    )
    parser.add_argument(
        "--draft-cooldown",
        type=int,
        default=None,
        help="For --draft-kind suffix: after N consecutive 0-accept verify "
        "rounds, pause proposing for a growing window (then probe). Avoids "
        "wasted verifies on novel text. Default: off (always propose).",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Enable thinking in the chat template. Templates that use "
            "thinking_mode receive thinking_mode='enabled'."
        ),
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("enabled", "disabled", "adaptive"),
        default=None,
        help=(
            "Set the chat-template thinking mode when supported. "
            "Choices: enabled, disabled, adaptive."
        ),
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Maximum number of thinking tokens before forcing the end-of-thinking token.",
    )
    parser.add_argument(
        "--thinking-start-token",
        type=str,
        default=DEFAULT_THINKING_START_TOKEN,
        help="Token that marks the start of a thinking block (default: %(default)s).",
    )
    parser.add_argument(
        "--thinking-end-token",
        type=str,
        default=DEFAULT_THINKING_END_TOKEN,
        help="Token that marks the end of a thinking block (default: %(default)s).",
    )

    return parser.parse_args()


def normalize_resize_shape(
    values: Optional[Sequence[int]],
) -> Optional[Tuple[int, int]]:
    if values is None:
        return None
    if not (
        isinstance(values, Sequence)
        and not isinstance(values, (str, bytes))
        and len(values) in (1, 2)
        and all(type(value) is int for value in values)
    ):
        raise ValueError("resize_shape must contain 1 or 2 integers")
    return (values[0], values[0]) if len(values) == 1 else tuple(values)


from .diffusion import (
    DEFAULT_DIFFUSION_CONFIDENCE_THRESHOLD,
    DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH,
    DiffusionOutputHandler,
    diffusion_kwargs_from_args,
    is_diffusion_model,
    stream_diffusion_generate_from_kwargs,
)
from .types import GenerateKwargs, ProcessorLike, Unpack


def _prime_cached_prefix_rope_state(
    model: nn.Module,
    full_input_ids: mx.array,
    mask: Optional[mx.array],
    kwargs: Dict[str, Any],
) -> bool:
    """Prime Qwen-style mRoPE metadata before a cached-prefix trim.

    Qwen VL language models keep ``_rope_deltas`` on the model object and use
    it when continuing from a non-empty KV cache. If APC trims the prompt to
    only the uncached suffix, the suffix alone is not enough to recompute the
    original prompt's RoPE delta, so derive it from the full prompt first.
    """
    lm = getattr(model, "language_model", None)
    get_rope_index = getattr(lm, "get_rope_index", None)
    if not callable(get_rope_index):
        return True
    if not (hasattr(lm, "_rope_deltas") or hasattr(lm, "_position_ids")):
        return True
    try:
        position_ids, rope_deltas = get_rope_index(
            full_input_ids,
            kwargs.get("image_grid_thw", None),
            kwargs.get("video_grid_thw", None),
            mask,
        )
    except Exception as e:
        logger.warning(
            "Could not prime cached-prefix RoPE state; falling back to cold prefill: %s",
            e,
        )
        return False
    if hasattr(lm, "_position_ids"):
        lm._position_ids = position_ids
    if hasattr(lm, "_rope_deltas"):
        lm._rope_deltas = rope_deltas
    kwargs["rope_deltas"] = rope_deltas
    return True


from .ar import generate_step


def stream_generate(
    model: nn.Module,
    processor: ProcessorLike | PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str], None] = None,
    audio: Union[str, List[str], None] = None,
    video: Union[str, List[str], None] = None,
    **kwargs: Unpack[GenerateKwargs],
) -> Generator[GenerationResult, None, None]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        model (nn.Module): The model to use for generation.
        processor (PreTrainedTokenizer): The tokenizer/processor.
        prompt (str): The input prompt text.
        image (Union[str, List[str]], optional): Image path(s) or URL(s).
        audio (Union[str, List[str]], optional): Audio file path(s).
        prefill_step_size (int, optional): Number of tokens to process per prefill
          step. When set, enables chunked prefill which processes long prompts in
          smaller chunks to reduce peak memory usage.
        kwargs: Additional options passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Yields:
        Generator[GenerationResult]: A generator producing GenerationResult objects
          containing the generated text, tokens, and statistics.
    """
    # Fork: a rewrite of upstream's body (405 -> 647 lines). 80 upstream lines are
    # absent and ALL of them are accounted for, per site, in two groups:
    #
    #  1. Upstream's prefix-reuse gate — `_prefix_cache_trim_amount(kv_cache,
    #     prefix_len)` plus its `n_drop` trim loop. Both symbols are REVIEWED
    #     .symbol-exclusions entries: this fork gates rewinds with
    #     `_rotating_rewind_safe` + the snapshot ring instead (c503fa7b, mlx-vlm
    #     #1715), and running both would give two different notions of when a
    #     rewind is safe. This is also why the audit reports a permanent
    #     whitespace probe near `from .ar import generate_step`: upstream's two
    #     absent functions sit right above it, so the differ pairs the blank lines
    #     differently. Both trees have identical blanks there; there is nothing to
    #     converge, which is why this file keeps its allowlist entry.
    #  2. Upstream's `enable_thinking` gate —
    #     `thinking_start_token_id in input_ids`. DELIBERATELY replaced by
    #     `prompt_is_inside_thinking(decoded_prompt)` + `detect_thinking_format`
    #     (1c3f1e50): for families like Gemma 4 the hardcoded <think>/</think>
    #     defaults are not real tokens, so upstream's id-in-prompt check forced
    #     enable_thinking False and the forced closer would have been a bare ">".
    #     The replacement is a strict superset — see the comment at the
    #     `thinking_budget` block below and prompt_utils.THINKING_FORMATS.
    #
    # Individually verified as still done, by reading the sites: the
    # vision-feature cache, the APC media-safe-prefix helpers
    # (`multimodal_token_ids_from_config`, `media_safe_prefix_min`,
    # `prefix_leaves_text_only_suffix`, `prefix_contains_media_tokens`) and
    # `prompt_cache_state.update`.
    #
    # This marker used to close with "everything else upstream does here is still
    # done ... all appear here at >= upstream's count", and that claim was FALSE.
    # An occurrence count over the symbols a marker happens to name says nothing
    # about the ones it does not: the diffusion dispatch below had silently
    # dropped upstream's `skip_special_tokens=` and `verbose=` kwargs, so
    # --skip-special-tokens was a no-op on every diffusion model while this
    # comment asserted completeness. Both gates, all seven audits and the whole
    # suite stayed green, because a helper that is present, imported, called and
    # unit-tested but called with FEWER ARGUMENTS is invisible to all of them.
    # Do not restore a blanket "everything else" claim here; per-site or nothing.
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    verbose = kwargs.pop("verbose", False)
    # Preserve only explicitly supplied sequence tensors as semantic APC
    # inputs. Tensors produced by prepare_inputs span the complete prompt and
    # therefore change whenever text is appended, even when the old token
    # prefix is identical.
    custom_inputs_embeds = kwargs.get("inputs_embeds")
    custom_mask = kwargs.get("mask")

    # Ensure stopping criteria reflects the model's EOS token IDs.
    # generate() does this before calling us, but direct callers (e.g. the
    # server) skip generate() and would otherwise use stale criteria. Guarded
    # on the criteria object's API so minimal/callable-only criteria keep
    # working unchanged.
    _stopping_criteria = getattr(tokenizer, "stopping_criteria", None)
    eos_tokens = kwargs.pop("eos_tokens", None)
    # Respect a caller-provided custom stopping_criteria (set by generate())
    # — don't clobber it with a reset.
    _custom_stopping_criteria = kwargs.get("stopping_criteria", None) is not None
    if _stopping_criteria is not None and not _custom_stopping_criteria:
        if eos_tokens is not None and hasattr(_stopping_criteria, "add_eos_token_ids"):
            _stopping_criteria.add_eos_token_ids(eos_tokens)
        elif eos_tokens is None and hasattr(_stopping_criteria, "reset"):
            _stopping_criteria.reset(model.config.eos_token_id)

            # Some model configs only list <eos> but omit chat-template stop
            # tokens like <end_of_turn>.  Resolve them from the tokenizer's
            # vocab and merge so generation stops at the right place.
            _chat_stop_tokens = ["<end_of_turn>", "<|endoftext|>", "<|im_end|>"]
            if hasattr(tokenizer, "convert_tokens_to_ids") and hasattr(
                _stopping_criteria, "eos_token_ids"
            ):
                for tok in _chat_stop_tokens:
                    tid = tokenizer.convert_tokens_to_ids(tok)
                    # convert_tokens_to_ids returns unk_token_id for unknowns
                    unk = getattr(tokenizer, "unk_token_id", None)
                    if (
                        tid is not None
                        and tid != unk
                        and tid not in _stopping_criteria.eos_token_ids
                    ):
                        _stopping_criteria.eos_token_ids.append(tid)

    # Set up thinking budget criteria if requested
    thinking_budget = kwargs.pop("thinking_budget", None)
    thinking_end_token = kwargs.pop("thinking_end_token", DEFAULT_THINKING_END_TOKEN)
    thinking_start_token = kwargs.pop(
        "thinking_start_token", DEFAULT_THINKING_START_TOKEN
    )
    enable_thinking = kwargs.pop("enable_thinking", False)

    # Skip special tokens
    skip_special_tokens = kwargs.pop("skip_special_tokens", False)
    skip_special_token_ids = (
        set(tokenizer.all_special_ids)
        if skip_special_tokens and hasattr(tokenizer, "all_special_ids")
        else []
    )

    add_special_tokens = should_add_special_tokens(model.config.model_type, processor)

    resize_shape = normalize_resize_shape(kwargs.pop("resize_shape", None))
    image_token_index = getattr(model.config, "image_token_index", None)
    vision_cache = kwargs.pop("vision_cache", None)
    prompt_cache_state = kwargs.pop("prompt_cache_state", None)
    apc_manager: Optional[_apc.APCManager] = kwargs.pop("apc_manager", None)
    apc_tenant: Optional[str] = kwargs.pop("apc_tenant", None)
    image = image or None
    audio = audio or None
    video = video or None

    # Asymmetric-template detection result (from the server's
    # _is_template_thinking_asymmetric on the (processor, template_kwargs)
    # pair). When True the chat template strips thinking content from prior
    # assistant messages → cache must anchor BEFORE the latest user message.
    # Read from PromptCacheState (server sets it once per session) with an
    # explicit kwarg override for tests / callers without a PromptCacheState.
    is_asymmetric_rendering = bool(kwargs.pop("is_asymmetric_rendering", False))
    if prompt_cache_state is not None and not is_asymmetric_rendering:
        is_asymmetric_rendering = bool(
            getattr(prompt_cache_state, "is_asymmetric_rendering", False)
        )

    if kwargs.get("input_ids", None) is not None:
        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values", None)
        mask = kwargs.pop("mask", None)
    else:
        inputs = prepare_inputs(
            processor,
            images=image,
            audio=audio,
            videos=video,
            prompts=prompt,
            image_token_index=image_token_index,
            resize_shape=resize_shape,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )
        input_ids = inputs.get("input_ids", None)
        pixel_values = inputs.get("pixel_values", None)
        mask = inputs.get("attention_mask", None)
        data_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        kwargs.update(data_kwargs)

    # Vision feature caching: reuse cached image features across turns
    if vision_cache is not None and image is not None and pixel_values is not None:
        cached = vision_cache.get(image)
        if cached is not None:
            kwargs["cached_image_features"] = cached
        elif hasattr(model, "encode_image"):
            features = model.encode_image(pixel_values)
            mx.eval(features)
            vision_cache.put(image, features)
            kwargs["cached_image_features"] = features

    # Prompt cache reuse: skip common prefix from previous turn
    reused_prefix_len = 0
    original_prompt_length = input_ids.size
    full_input_ids_list = input_ids.flatten().tolist()
    apc_blocks_in_use: List[_apc.APCBlock] = []
    apc_extra_hash = 0
    apc_mode: Optional[str] = None

    multimodal_token_ids = _apc.multimodal_token_ids_from_config(model.config)
    apc_safe_prefix_min = _apc.media_safe_prefix_min(
        full_input_ids_list,
        multimodal_token_ids,
    )
    apc_safe_prefix_lookup_min = max(0, apc_safe_prefix_min - 1)

    def _apc_suffix_is_text_only(prefix_len: int) -> bool:
        return _apc.prefix_leaves_text_only_suffix(
            full_input_ids_list,
            prefix_len,
            multimodal_token_ids,
        )

    def _apc_prefix_has_media_tokens(prefix_len: int) -> bool:
        return _apc.prefix_contains_media_tokens(
            full_input_ids_list,
            prefix_len,
            multimodal_token_ids,
        )

    if is_diffusion_model(model, kwargs):
        yield from stream_diffusion_generate_from_kwargs(
            model,
            processor,
            tokenizer,
            input_ids,
            pixel_values,
            mask,
            skip_special_token_ids,
            kwargs,
            skip_special_tokens=skip_special_tokens,
            verbose=verbose,
        )
        return

    if apc_manager is not None:
        apc_mode = _apc.model_apc_mode(model.language_model)
        if apc_mode is None:
            apc_manager = None

    if apc_manager is not None:
        image_hash = _apc.hash_image_payload(pixel_values=pixel_values, image_ref=image)
        audio_features = kwargs.get("input_features")
        video_features = kwargs.get("pixel_values_videos")
        apc_extra_hash = _apc.semantic_extra_hash(
            tenant=apc_tenant,
            image_hash=image_hash,
            media={
                "audio": audio_features if audio_features is not None else audio,
                "video": video_features if video_features is not None else video,
                "embeddings": custom_inputs_embeds,
                "masks": custom_mask,
            },
            model=model,
            processor=processor,
        )

    if prompt_cache_state is not None and prompt_cache_state.cache is not None:
        prefix_len = prompt_cache_state.find_prefix_length(full_input_ids_list)

        # SWA Ring Buffer Corruption Guard: a RotatingKVCache that wrapped
        # during a prior turn can't be rewound into its overwritten region.
        # Falling back to full re-prefill is the only safe option.
        if prefix_len > 0 and prefix_len < len(prompt_cache_state.token_ids):
            if not _rotating_rewind_safe(prompt_cache_state.cache, prefix_len):
                logger.debug(
                    "SWA Ring Buffer Corruption Guard: rewind to token %d "
                    "is in the overwritten region. Forcing full re-prefill.",
                    prefix_len,
                )
                prefix_len = 0

        # Hybrid-Cache Rewind Guard: models with non-trimmable recurrent
        # layers (GatedDeltaNet/Mamba via ArraysCache) can only rewind via a
        # captured snapshot. Restore from the nearest snapshot and replay from
        # there, or fall back to full re-prefill when none is available.
        if prefix_len > 0 and prefix_len < len(prompt_cache_state.token_ids):
            if _has_non_trimmable(prompt_cache_state.cache):
                ring = getattr(prompt_cache_state, "snapshot_ring", None)
                rewind_enabled = getattr(prompt_cache_state, "rewind_enabled", True)
                snap = (
                    ring.find_nearest(prefix_len)
                    if (ring is not None and ring.enabled and rewind_enabled)
                    else None
                )
                if snap is None:
                    logger.warning(
                        "Hybrid-Cache Rewind Guard: no snapshot available "
                        "for rewind to %d (ring=%s, rewind_enabled=%s). "
                        "Forcing full re-prefill.",
                        prefix_len,
                        len(ring) if ring else "none",
                        rewind_enabled,
                    )
                    prefix_len = 0
                else:
                    _restore_deltanet_state(prompt_cache_state.cache, snap.states)
                    logger.warning(
                        "DeltaNet snapshot rewind: restored from offset %d, "
                        "replaying %d tokens to reach prefix %d.",
                        snap.offset,
                        prefix_len - snap.offset,
                        prefix_len,
                    )
                    # Treat the snapshot offset as the new prefix point; the
                    # trim+prefill below advances both KV and DeltaNet state
                    # from the restored snapshot.
                    prefix_len = snap.offset

        if prefix_len > 0 and prefix_len < input_ids.shape[1]:
            if _apc_suffix_is_text_only(prefix_len) and _prime_cached_prefix_rope_state(
                model, input_ids, mask, kwargs
            ):
                reused_prefix_len = prefix_len
                # Trim to only new tokens
                input_ids = input_ids[:, prefix_len:]
                # Only skip vision if the new (trimmed) tokens carry no image
                # tokens — otherwise the trimmed prefill still needs them.
                image_token_id = getattr(
                    model.config, "image_token_id", None
                ) or getattr(model.config, "image_token_index", None)
                new_ids = input_ids.flatten().tolist()
                has_image_in_new = (
                    image_token_id is not None and image_token_id in new_ids
                )
                if not has_image_in_new:
                    pixel_values = None
                    kwargs.pop("cached_image_features", None)
                # Reuse the saved KV cache (recursively trimmed to prefix_len;
                # handles hybrid/rotating/quantized layouts correctly, unlike a
                # blind physical slice).
                kv_cache = prompt_cache_state.cache
                for c in kv_cache:
                    _trim_cache(c, prefix_len)
                kwargs["prompt_cache"] = kv_cache

    if prompt_cache_state is not None:
        logger.info(
            "Prefix Cache Telemetry | Total Prompt: %d | Skipped Context: %d "
            "| Prompt Delta: %d",
            original_prompt_length,
            reused_prefix_len,
            input_ids.size,
        )

    # APC: cross-request, hash-based prefix lookup. Only consulted if a per-turn
    # PromptCacheState didn't already produce a hit.
    # APC: cross-request, hash-based prefix lookup. Only consulted if a per-turn
    # PromptCacheState didn't already produce a hit.
    if apc_manager is not None and reused_prefix_len == 0:
        plan = _apc.apc_lookup_plan(
            apc_manager,
            full_input_ids_list,
            extra_hash=apc_extra_hash,
            apc_mode=apc_mode,
            safe_lookup_min=apc_safe_prefix_lookup_min,
            suffix_is_text_only=_apc_suffix_is_text_only,
            prefix_has_media=_apc_prefix_has_media_tokens,
        )
        if plan is not None:
            plen = plan["prefix_len"]
            warm_cache = plan.get("warm_cache")
            matched_blocks = plan.get("matched_blocks") or []
            primed = _prime_cached_prefix_rope_state(model, input_ids, mask, kwargs)
            if primed:
                reused_prefix_len = plen
                input_ids = input_ids[:, plen:]
                pixel_values = None
                kwargs.pop("cached_image_features", None)
                if warm_cache is not None:
                    kwargs["prompt_cache"] = warm_cache
                else:
                    apc_blocks_in_use = matched_blocks
                    # Warm-restored layers must come back the same *type* live
                    # generation would have built, or continuous-batching
                    # `extend` can try to join differently-typed peers. Without
                    # this the warm cache was always float KVCache even with
                    # kv-bits on.
                    _quant_policy = kv_quant_from_legacy(
                        kwargs.get("kv_bits"),
                        kwargs.get("kv_quant_scheme"),
                        kwargs.get("kv_group_size", 64),
                        kwargs.get("kv_key_bits"),
                        kwargs.get("kv_value_bits"),
                        kwargs.get("kv_key_scheme"),
                        kwargs.get("kv_value_scheme"),
                    )
                    _quant_cfg = (
                        _quant_policy.to_config() if _quant_policy is not None else None
                    )
                    kwargs["prompt_cache"] = _apc.make_warm_kv_cache(
                        matched_blocks,
                        min_capacity_tokens=plen + input_ids.shape[1] + 1,
                        kv_quant_config=_quant_cfg,
                    )
            elif warm_cache is None and matched_blocks:
                apc_manager.release(matched_blocks)

    if thinking_budget is not None:
        # Fork: detect an open thinking block across ALL registered formats, and
        # feed the criteria the model's REAL delimiters — not the hardcoded
        # <think>/</think>. For families like Gemma 4 (opener <|think|>,
        # per-turn closer <channel|>) the defaults are not real tokens — they
        # tokenize to subword pieces both ending in ">", so upstream's
        # `<think>`-id-in-prompt check reads False and the forced closer would
        # have been a bare ">". See prompt_utils.THINKING_FORMATS.
        decoded_prompt = tokenizer.decode(input_ids.flatten().tolist())
        prompt_preopens_thinking = prompt_is_inside_thinking(decoded_prompt)
        eff_start_token, eff_end_token = thinking_start_token, thinking_end_token
        fmt = detect_thinking_format(decoded_prompt)
        if fmt is not None:
            eff_start_token = fmt.openers[0]
            # Prefer a single-token closer so the forced close sequence is
            # exact (the criteria keys the forced/stop token off the LAST
            # token id of the closer string).
            eff_end_token = fmt.closers[0]
            for _closer in fmt.closers:
                if len(cached_special_token_encode(tokenizer, _closer)) == 1:
                    eff_end_token = _closer
                    break
        tokenizer.thinking_budget_criteria = ThinkingBudgetCriteria(
            tokenizer=tokenizer,
            thinking_budget=thinking_budget,
            thinking_end_token=eff_end_token,
            thinking_start_token=eff_start_token,
            enable_thinking=enable_thinking,
            prompt_preopens_thinking=prompt_preopens_thinking,
        )
        kwargs["thinking_budget_criteria"] = tokenizer.thinking_budget_criteria
    else:
        tokenizer.thinking_budget_criteria = None

    # Ensure we have a prompt_cache we can track for reuse.
    if "prompt_cache" not in kwargs:
        kwargs["prompt_cache"] = cache.make_prompt_cache(
            model.language_model,
            max_kv_size=kwargs.get("max_kv_size", None),
        )
    tracked_cache = kwargs["prompt_cache"]

    total_prompt_tokens = reused_prefix_len + input_ids.size

    # Asymmetric-rendering anchor: compute the token offset of the LAST
    # user-turn-open marker in the rendered prompt and ask generate_step to
    # capture cache state at that boundary during chunked prefill. Persisting
    # the cache anchored at this offset (instead of end-of-user) means the next
    # request's re-rendering of the latest user message (e.g. OpenWebUI RAG
    # `<context>` wrapping once a search tool returns) won't trigger a backward
    # trim. Three parallel side-channels:
    #   * rotating — RotatingKVCache (Gemma 4 SWA layers)
    #   * arrays   — ArraysCache (Qwen 3.5/3.6 DeltaNet layers, Mamba)
    #   * offset   — single-entry list holding the actual offset reached
    # The offset marker is what the post-gen branch keys on — it fires the
    # anchor path even on pure plain-attention models (just trim KV layers).
    snapshot_at_offset: Optional[int] = None
    mid_prefill_rotating_capture: List[Any] = []
    mid_prefill_arrays_capture: List[Optional[List[mx.array]]] = []
    mid_prefill_anchor_offset: List[int] = []
    if is_asymmetric_rendering and prompt_cache_state is not None:
        snapshot_at_offset = _compute_anchor_before_latest_user_offset(
            prompt, tokenizer
        )
        if snapshot_at_offset is not None:
            kwargs["snapshot_at_offset"] = snapshot_at_offset
            kwargs["rotating_snapshot_capture"] = mid_prefill_rotating_capture
            kwargs["arrays_snapshot_capture"] = mid_prefill_arrays_capture
            kwargs["anchor_capture_offset"] = mid_prefill_anchor_offset

    with wired_limit(model, [_get_generation_stream()]):
        detokenizer = make_streaming_detokenizer(processor)
        thinking_criteria = getattr(tokenizer, "thinking_budget_criteria", None)
        exact_checkpoint_len = None
        exact_checkpoint = None
        if apc_manager is not None and apc_mode == "exact" and reused_prefix_len == 0:
            exact_checkpoint_len = _apc.adjust_prefix_to_text_suffix_boundary(
                full_input_ids_list,
                len(full_input_ids_list) - apc_manager.exact_cache_guard_tokens,
                multimodal_token_ids,
                max_prefix_tokens=len(full_input_ids_list) - 1,
            )
            if exact_checkpoint_len <= 0:
                exact_checkpoint_len = None

            def exact_checkpoint(prefix_len: int, prompt_cache: List[Any]) -> None:
                apc_manager.store_exact_cache(
                    full_input_ids_list[:prefix_len],
                    prompt_cache,
                    extra_hash=apc_extra_hash,
                )

        gen = generate_step(
            input_ids,
            model,
            pixel_values,
            mask,
            prompt_cache_checkpoint=exact_checkpoint,
            prompt_cache_checkpoint_len=exact_checkpoint_len,
            verbose=verbose,
            **kwargs,
        )
        tic = time.perf_counter()

        generated_tokens = []
        finish_reason: Optional[str] = None
        # Rotating-layer snapshots captured at end-of-prefill, used at
        # post-generation to restore the SWA layers' state to the end-of-user
        # boundary (fallback when the mid-prefill anchor wasn't reached).
        rotating_snapshots: List[Any] = []
        for n, (token, logprobs) in enumerate(gen):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = total_prompt_tokens / prompt_time
                tic = time.perf_counter()
                if (
                    apc_manager is not None
                    and apc_mode == "exact"
                    and reused_prefix_len == 0
                ):
                    try:
                        apc_manager.store_exact_cache(
                            full_input_ids_list,
                            tracked_cache,
                            extra_hash=apc_extra_hash,
                        )
                    except Exception as e:
                        logger.warning("APC exact-cache store failed: %s", e)

                # End-of-prefill cache state has just been produced (the first
                # yielded token is sampled from post-prefill logits but not yet
                # written into the cache). Capture rotating-layer state now if
                # the chat template renders prior asst turns asymmetrically; we
                # restore after generation so the cache anchors at end-of-user.
                if is_asymmetric_rendering and prompt_cache_state is not None:
                    from ..snapshot import capture_rotating

                    rotating_snapshots = _capture_rotating_layers_for_snapshot(
                        tracked_cache, capture_rotating
                    )
                    if rotating_snapshots:
                        logger.debug(
                            "Captured %d rotating-layer snapshot(s) at "
                            "end-of-prefill for asymmetric-rendering session.",
                            len(rotating_snapshots),
                        )

            generated_tokens.append(token)

            # Check thinking budget and force token if needed
            if thinking_criteria is not None:
                thinking_criteria(token)

            # Stop generation if the token is in the eos_token_ids
            if tokenizer.stopping_criteria(token):
                finish_reason = "stop"
                break

            detokenizer.add_token(token, skip_special_token_ids=skip_special_token_ids)

            # Yield the last segment if streaming
            yield GenerationResult(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                prompt_tokens=total_prompt_tokens,
                generation_tokens=n + 1,
                total_tokens=total_prompt_tokens + n + 1,
                prompt_tps=prompt_tps,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
                cached_tokens=reused_prefix_len,
            )
        else:
            # generate_step exhausted its budget without stopping_criteria firing.
            finish_reason = "length"

        if not generated_tokens:
            prompt_time = time.perf_counter() - tic
            prompt_tps = total_prompt_tokens / prompt_time if prompt_time > 0 else 0.0
            yield GenerationResult(
                text="",
                token=None,
                logprobs=None,
                prompt_tokens=total_prompt_tokens,
                generation_tokens=0,
                total_tokens=total_prompt_tokens,
                prompt_tps=prompt_tps,
                generation_tps=0.0,
                peak_memory=mx.get_peak_memory() / 1e9,
                cached_tokens=reused_prefix_len,
                finish_reason="length",
            )
            return

        detokenizer.finalize()
        yield GenerationResult(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            prompt_tokens=total_prompt_tokens,
            generation_tokens=n + 1,
            total_tokens=total_prompt_tokens + n + 1,
            prompt_tps=prompt_tps,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
            cached_tokens=reused_prefix_len,
            finish_reason=finish_reason,
        )

        all_ids: Optional[List[int]] = full_input_ids_list + [
            t.item() if hasattr(t, "item") else t for t in generated_tokens
        ]

        # APC: harvest new blocks from the post-generation KV state. Runs
        # BEFORE the PromptCacheState asymmetric anchoring below, because that
        # path restores/trims ``tracked_cache`` in-place and would desync the
        # layer offsets from ``all_ids``.
        if apc_manager is not None and apc_mode == "block":
            try:
                if all_ids is None:
                    all_ids = full_input_ids_list + [
                        t.item() if hasattr(t, "item") else t for t in generated_tokens
                    ]
                _apc.commit_prefix_blocks(
                    apc_manager,
                    tracked_cache,
                    all_ids,
                    extra_hash=apc_extra_hash,
                    skip_first_n_tokens=reused_prefix_len,
                    blocks_in_use=apc_blocks_in_use,
                )
            except Exception as e:
                logger.warning("APC store failed: %s", e)
                apc_manager.release(apc_blocks_in_use)

        # Save cache state for potential reuse on next turn. Two paths,
        # selected by ``is_asymmetric_rendering`` (computed in the server via
        # ``_is_template_thinking_asymmetric`` and passed in via kwargs /
        # PromptCacheState).
        #
        # ASYMMETRIC (e.g. Gemma 4 — chat template strips thinking content
        # from prior assistant messages on every render): anchor the cache
        # BEFORE the latest user message. Prefer the mid-prefill snapshot at
        # ``snapshot_at_offset``; fall back to end-of-user (= end-of-prefill)
        # when that boundary wasn't located or chunked prefill didn't run.
        #
        # SYMMETRIC (e.g. Qwen 3.x official, Gemma 3, Llama): persist the FULL
        # end-of-asst state; the next request's prefix-match extends naturally.
        if prompt_cache_state is not None:
            prefill_len = len(full_input_ids_list)
            if is_asymmetric_rendering:
                if mid_prefill_anchor_offset and snapshot_at_offset is not None:
                    # Anchor at the captured offset (authoritative; may differ
                    # from ``snapshot_at_offset`` by a few tokens when chunked
                    # prefill couldn't land exactly). Three independent restore
                    # steps, each a no-op when its snapshot list is empty.
                    anchor_offset = mid_prefill_anchor_offset[0]
                    _restore_rotating_layers_from_snapshots(
                        tracked_cache, mid_prefill_rotating_capture
                    )
                    _restore_arrays_layers_from_snapshots(
                        tracked_cache, mid_prefill_arrays_capture
                    )
                    for c in tracked_cache:
                        if not _is_rotating_kv_layer(c):
                            _trim_cache(c, anchor_offset)
                    anchor_token_ids = full_input_ids_list[:anchor_offset]
                    prompt_cache_state.update(anchor_token_ids, tracked_cache)
                    logger.debug(
                        "Asymmetric path: persisted cache BEFORE latest user "
                        "message (anchor=%d [%s of target %d] / prefill_len %d).",
                        anchor_offset,
                        "exact" if anchor_offset == snapshot_at_offset else "fallback",
                        snapshot_at_offset,
                        prefill_len,
                    )
                else:
                    # Fallback: anchor at end-of-user (= end-of-prefill).
                    if rotating_snapshots:
                        _restore_rotating_layers_from_snapshots(
                            tracked_cache, rotating_snapshots
                        )
                    for c in tracked_cache:
                        if not _is_rotating_kv_layer(c):
                            _trim_cache(c, prefill_len)
                    prompt_cache_state.update(full_input_ids_list, tracked_cache)
                    logger.debug(
                        "Asymmetric path (fallback): persisted cache at "
                        "end-of-user (offset %d); restored %d rotating layer(s).",
                        prefill_len,
                        len(rotating_snapshots),
                    )
            else:
                # Symmetric: persist full end-of-asst state.
                prompt_cache_state.update(all_ids, tracked_cache)

        # Cleanup after generation
        mx.clear_cache()


def generate(
    model: nn.Module,
    processor: ProcessorLike | PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str], None] = None,
    audio: Union[str, List[str], None] = None,
    video: Union[str, List[str], None] = None,
    verbose: bool = False,
    **kwargs: Unpack[GenerateKwargs],
) -> GenerationResult:
    """
    Generate text from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       temperature (float): The temperature for sampling (default 0).
       max_tokens (int): The maximum number of tokens (default 100).
       verbose (bool): If ``True``, print tokens and timing information
           (default ``False``).
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       repetition_penalty (float, optional): The penalty factor for repeating tokens.
       repetition_context_size (int, optional): The number of tokens to consider for repetition penalty.
    """

    if verbose:
        print("=" * 10)
        files = []
        if image is not None:
            files.extend(image)
        if audio is not None:
            files.extend(audio)
        if video is not None:
            files.extend(video if isinstance(video, list) else [video])

        print(f"Files: {files}", "\n")

        print("Prompt:", prompt)

    text = ""
    last_response = None

    eos_tokens = kwargs.get("eos_tokens", None)
    stopping_criteria = kwargs.get("stopping_criteria", None)

    # Get the tokenizer
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    diffusion_output = DiffusionOutputHandler(model, kwargs, verbose)

    # Add custom EOS tokens to the stopping criteria
    if eos_tokens is not None:
        tokenizer.stopping_criteria.add_eos_token_ids(eos_tokens)

    # Use custom stopping criteria
    elif stopping_criteria is not None:
        if isinstance(stopping_criteria, StoppingCriteria) or callable(
            stopping_criteria
        ):
            tokenizer.stopping_criteria = stopping_criteria
        else:
            raise ValueError(
                "stopping_criteria must be an instance of StoppingCriteria or a callable"
            )
    else:
        tokenizer.stopping_criteria.reset(model.config.eos_token_id)

    for response in stream_generate(
        model, processor, prompt, image, audio, video, verbose=verbose, **kwargs
    ):
        if response.is_draft:
            diffusion_output.handle_draft(response)
            last_response = response
            continue

        if (
            verbose
            and not response.text_already_printed
            and not diffusion_output.handle_text(response.text)
        ):
            print(response.text, end="", flush=True)
        text += response.text
        last_response = response

    if last_response is None:
        return GenerationResult(text=text, peak_memory=mx.get_peak_memory() / 1e9)

    if verbose:
        diffusion_output.finish(text)
        print("\n" + "=" * 10)
        if len(text) == 0:
            print("No text generated for this prompt")
        print(
            f"Prompt: {last_response.prompt_tokens} tokens, "
            f"{last_response.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"Generation: {last_response.generation_tokens} tokens, "
            f"{last_response.generation_tps:.3f} tokens-per-sec"
        )
        print(f"Peak memory: {last_response.peak_memory:.3f} GB")

    return GenerationResult(
        text=text,
        token=last_response.token,
        logprobs=last_response.logprobs,
        prompt_tokens=last_response.prompt_tokens,
        generation_tokens=last_response.generation_tokens,
        total_tokens=last_response.total_tokens,
        prompt_tps=last_response.prompt_tps,
        generation_tps=last_response.generation_tps,
        peak_memory=last_response.peak_memory,
        cached_tokens=last_response.cached_tokens,
        finish_reason=last_response.finish_reason,
        diffusion_canvas_tokens=last_response.diffusion_canvas_tokens,
        diffusion_denoising_steps=last_response.diffusion_denoising_steps,
        diffusion_work_tokens=last_response.diffusion_work_tokens,
        diffusion_canvas_tps=last_response.diffusion_canvas_tps,
        diffusion_work_tps=last_response.diffusion_work_tps,
    )


def main():
    # Fork: PURE ADDITIONS — 17 added lines, 0 removed: the `elif args.draft_kind
    # == "suffix"` arm that builds a SuffixDecodingProposer and passes it as
    # `draft_model`, so upstream's existing dispatch fires with no weights and no
    # extra memory (863441c9). Everything else in this function is upstream's.
    args = parse_arguments()

    if getattr(args, "output_modality", "text") == "image":
        run_image_generation_cli(args)
        return
    if getattr(args, "output_modality", "text") == "video":
        run_video_generation_cli(args)
        return

    if getattr(args, "seed", None) is not None:
        mx.random.seed(args.seed)

    diffusion_arg_defaults = {
        "max_denoising_steps": None,
        "diffusion_full_canvas": False,
        "diffusion_min_canvas_length": None,
        "diffusion_max_canvas_length": None,
        "diffusion_sampler": "confidence-threshold",
        "threshold": None,
        "min_threshold": None,
        "block_length": None,
        "num_to_transfer": None,
        "max_transfer_per_step": None,
        "editing_threshold": None,
        "max_post_steps": None,
        "stability_steps": None,
        "gen_kwargs": {},
    }
    for name, default in diffusion_arg_defaults.items():
        if not hasattr(args, name):
            setattr(args, name, default)

    if isinstance(args.image, str):
        args.image = [args.image]
    if isinstance(args.audio, str):
        args.audio = [args.audio]
    if isinstance(args.video, str):
        args.video = [args.video]

    model, processor = load(
        args.model,
        args.adapter_path,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        quantize_activations=args.quantize_activations,
    )
    config = model.config

    draft_model = None
    if args.draft_model is not None:
        from ..speculative.drafters import load_drafter, validate_drafter_compatibility

        print(f"Loading drafter ({args.draft_kind or 'auto'}): {args.draft_model}")
        draft_model, resolved_kind = load_drafter(
            args.draft_model, kind=args.draft_kind
        )
        if args.draft_kind is None:
            print(f"  → auto-detected --draft-kind={resolved_kind!r}.")
        elif resolved_kind != args.draft_kind:
            print(
                f"  → drafter requires --draft-kind={resolved_kind!r}; "
                f"using {resolved_kind!r} instead of {args.draft_kind!r}."
            )
        args.draft_kind = resolved_kind
        try:
            validate_drafter_compatibility(model, draft_model, args.draft_kind)
        except ValueError as e:
            print(
                "Speculative drafter is incompatible with the target model; "
                f"falling back to autoregressive generation. {e}"
            )
            draft_model = None
            args.draft_kind = None
    elif args.draft_kind == "suffix":
        # Drafter-free speculative decoding: construct the n-gram proposer
        # internally and pass it as ``draft_model`` so the existing dispatch
        # fires. No weights, no extra memory.
        from ..speculative.suffix_decoding import SuffixDecodingProposer

        draft_model = SuffixDecodingProposer(
            min_match=args.suffix_min_match,
            max_draft=args.draft_block_size,
            cooldown=args.draft_cooldown,
        )
        print(
            "Using drafter-free suffix decoding "
            f"(min_match={args.suffix_min_match}, "
            f"max_draft={args.draft_block_size or 'default'}, "
            f"cooldown={args.draft_cooldown or 'off'})."
        )

    prompt = args.prompt

    if args.system:
        prompt = [{"role": "system", "content": args.system}] + (
            prompt if isinstance(prompt, list) else [prompt]
        )

    # Processors without native video support used to drop --video silently:
    # the frames were loaded, the processor ignored the kwarg, and the model
    # hallucinated an answer with no visual input at all. Fall back to sending
    # sampled frames as ordered images (see generate/video.py).
    gen_kwargs_extra = {}
    video_prompt = None
    if args.video:
        from .video import (
            pair_adjacent_frames,
            processor_handles_video,
            resolve_video_inputs,
            sample_video_frames,
            timestamped_frame_messages,
        )

        if not processor_handles_video(processor):
            max_frames = max(2, getattr(args, "video_max_frames", 16) or 16)
            pair_hook = getattr(model, "prepare_video_frame_pairs", None)
            if pair_hook is not None:
                frames, frame_fps = sample_video_frames(args.video, args.fps or 2.0)
                anchors, first_frames, second_frames = pair_adjacent_frames(
                    frames, max_frames
                )
                gen_kwargs_extra.update(pair_hook(processor, second_frames))
                still_count = len(args.image or [])
                args.image = (args.image or []) + first_frames
                user_text = (
                    " ".join(args.prompt)
                    if isinstance(args.prompt, list)
                    else str(args.prompt)
                )
                msgs = timestamped_frame_messages(
                    user_text,
                    args.system,
                    still_count,
                    [a / max(frame_fps, 1e-6) for a in anchors],
                )
                _tok = (
                    processor.tokenizer
                    if hasattr(processor, "tokenizer")
                    else processor
                )
                video_prompt = _tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False
                )
                args.video = None
            else:
                resolution = resolve_video_inputs(
                    processor,
                    args.video,
                    images=args.image,
                    fps=args.fps or 2.0,
                    max_frames=max_frames,
                )
                print(
                    f"{processor.__class__.__name__} has no native video "
                    f"support; sending {resolution.selected_count} of "
                    f"{resolution.sampled_count} sampled "
                    f"frames as ordered images."
                )
                args.image = resolution.images
                args.video = resolution.videos or None

    num_images = len(args.image) if args.image is not None else 0
    num_audios = len(args.audio) if args.audio is not None else 0

    chat_template_kwargs = {"enable_thinking": args.enable_thinking}
    if args.thinking_mode is not None:
        chat_template_kwargs["thinking_mode"] = args.thinking_mode
    if args.video:
        chat_template_kwargs["video"] = args.video
        chat_template_kwargs["fps"] = args.fps

    if video_prompt is not None:
        prompt = video_prompt
    else:
        prompt = apply_chat_template(
            processor,
            config,
            prompt,
            num_images=num_images,
            num_audios=num_audios,
            **chat_template_kwargs,
        )

    kwargs = {}

    if args.eos_tokens is not None:
        eos_tokens = []
        for token in args.eos_tokens:
            try:
                decoded_token = codecs.decode(token, "unicode_escape")
                eos_tokens.append(decoded_token)
            except (UnicodeDecodeError, UnicodeError):
                eos_tokens.append(token)
        kwargs["eos_tokens"] = eos_tokens

    if args.skip_special_tokens:
        kwargs["skip_special_tokens"] = args.skip_special_tokens

    # Add processor kwargs from JSON
    if args.processor_kwargs:
        kwargs.update(args.processor_kwargs)

    # Add generation kwargs from JSON
    if args.gen_kwargs:
        kwargs.update(args.gen_kwargs)

    # Add thinking kwargs
    kwargs["enable_thinking"] = args.enable_thinking
    if args.thinking_budget is not None:
        kwargs["thinking_budget"] = args.thinking_budget
        kwargs["thinking_end_token"] = args.thinking_end_token
        if args.thinking_start_token is not None:
            kwargs["thinking_start_token"] = args.thinking_start_token

    if args.chat:
        from ..vision_cache import VisionFeatureCache

        vision_cache = VisionFeatureCache()
        chat = []
        if args.system:
            chat.append({"role": "system", "content": args.system})
        while user := input("User:"):
            chat.append({"role": "user", "content": user})
            prompt = apply_chat_template(
                processor,
                config,
                chat,
                num_images=num_images,
                num_audios=num_audios,
                **chat_template_kwargs,
            )
            response = ""
            print("Assistant:", end="")
            stream_kwargs = {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "min_p": args.min_p,
                "repetition_penalty": args.repetition_penalty,
                "repetition_context_size": args.repetition_context_size,
                "presence_penalty": args.presence_penalty,
                "presence_context_size": args.presence_context_size,
                "frequency_penalty": args.frequency_penalty,
                "frequency_context_size": args.frequency_context_size,
                "vision_cache": vision_cache,
                **kwargs,
            }
            if args.resize_shape is not None:
                stream_kwargs["resize_shape"] = args.resize_shape
            if args.prefill_step_size is not None:
                stream_kwargs["prefill_step_size"] = args.prefill_step_size
            stream_kwargs.update(diffusion_kwargs_from_args(args, config))

            diffusion_output = DiffusionOutputHandler(model, stream_kwargs, True)
            for chunk in stream_generate(
                model,
                processor,
                prompt,
                args.image,
                args.audio,
                args.video,
                **stream_kwargs,
            ):
                if chunk.is_draft:
                    diffusion_output.handle_draft(chunk)
                    continue
                response += chunk.text
                if not diffusion_output.handle_text(chunk.text):
                    print(chunk.text, end="")

            chat.append({"role": "assistant", "content": response})
            diffusion_output.finish(response)
            print()

    else:
        gen_kwargs = {
            **gen_kwargs_extra,
            "image": args.image,
            "audio": args.audio,
            "video": args.video,
            "fps": args.fps,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "max_tokens": args.max_tokens,
            "repetition_penalty": args.repetition_penalty,
            "repetition_context_size": args.repetition_context_size,
            "presence_penalty": args.presence_penalty,
            "presence_context_size": args.presence_context_size,
            "frequency_penalty": args.frequency_penalty,
            "frequency_context_size": args.frequency_context_size,
            "verbose": args.verbose,
            "max_kv_size": args.max_kv_size,
            "kv_bits": args.kv_bits,
            "kv_key_bits": getattr(args, "kv_key_bits", None),
            "kv_value_bits": getattr(args, "kv_value_bits", None),
            "kv_key_scheme": getattr(args, "kv_key_scheme", None),
            "kv_value_scheme": getattr(args, "kv_value_scheme", None),
            "kv_group_size": args.kv_group_size,
            "kv_quant_scheme": getattr(
                args, "kv_quant_scheme", DEFAULT_KV_QUANT_SCHEME
            ),
            "quantized_kv_start": args.quantized_kv_start,
            **kwargs,
        }
        if args.resize_shape is not None:
            gen_kwargs["resize_shape"] = args.resize_shape
        if args.prefill_step_size is not None:
            gen_kwargs["prefill_step_size"] = args.prefill_step_size
        gen_kwargs.update(diffusion_kwargs_from_args(args, config))
        if draft_model is not None:
            gen_kwargs["draft_model"] = draft_model
            gen_kwargs["draft_kind"] = args.draft_kind
            if args.draft_block_size is not None:
                gen_kwargs["draft_block_size"] = args.draft_block_size

        result = generate(
            model,
            processor,
            prompt,
            **gen_kwargs,
        )
        if not args.verbose:
            print(result.text)

        if draft_model is not None:
            stats = format_speculative_stats(draft_model)
            if stats is not None:
                print(stats)


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_vlm.generate ...` directly is deprecated."
        " Use `mlx_vlm generate` or `python -m mlx_vlm generate` instead."
    )
    main()
