import argparse
import logging
import os

import uvicorn

from ..generate import (
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
)
from ..snapshot import DEFAULT_RING_SIZE
from .generation import DEFAULT_ENABLE_THINKING, get_server_max_tokens
from .session_manager import _env_choice, _env_int
from .session_manager import configure as _configure_session_manager

DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 8080

logger = logging.getLogger("mlx_vlm.server")


def main():
    parser = argparse.ArgumentParser(description="MLX VLM Http Server.")
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_SERVER_HOST,
        help="Host for the HTTP server (default:0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help="Port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading models from Hugging Face Hub.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Pre-load a model at startup (e.g. mlx-community/Qwen2.5-VL-3B-Instruct-4bit).",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Adapter weights to load with the model.",
    )
    parser.add_argument(
        "--vision-cache-size",
        type=int,
        default=20,
        help="Max number of cached vision features (default: 20).",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=DEFAULT_PREFILL_STEP_SIZE,
        help="Tokens per prefill step (default: %(default)s).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=get_server_max_tokens(),
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=DEFAULT_ENABLE_THINKING,
        help=(
            "Enable thinking mode by default for requests that do not set "
            "enable_thinking explicitly."
        ),
    )
    parser.add_argument(
        "--kv-bits",
        type=float,
        default=None,
        help="Number of bits for KV cache quantization (e.g. 3.5 for TurboQuant).",
    )
    parser.add_argument(
        "--kv-quant-scheme",
        type=str,
        choices=("uniform", "turboquant"),
        default=DEFAULT_KV_QUANT_SCHEME,
        help="KV cache quantization backend.",
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
        default=None,
        help="Maximum KV cache size in tokens.",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
        help="Start index for quantized KV cache.",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help=(
            "Speculative drafter path or HF id "
            "(e.g. z-lab/Qwen3.5-4B-DFlash, google/gemma-4-31B-it-assistant)."
        ),
    )
    parser.add_argument(
        "--draft-kind",
        type=str,
        default=None,
        choices=["dflash", "eagle3", "mtp"],
        help="Drafter family -- 'dflash', 'eagle3', or 'mtp' (Gemma 4). "
        "Default: auto-detected from the drafter's HF model_type.",
    )
    parser.add_argument(
        "--draft-block-size",
        type=int,
        default=None,
        help="Override the drafter's configured block size.",
    )
    parser.add_argument(
        "--top-logprobs-k",
        type=int,
        default=None,
        help=(
            "Server-side cap for per-token top_logprobs (0-20, default 0 = "
            "disabled). Maps to the TOP_LOGPROBS_K env var."
        ),
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload for development.",
    )
    parser.add_argument(
        "--cache-session-max",
        type=int,
        default=_env_int("MLX_VLM_CACHE_SESSION_MAX", 8),
        help=(
            "Maximum number of per-chat PromptCacheState sessions retained "
            "concurrently. Used for prefix-cache reuse across turns and "
            "DeltaNet snapshot rewind on hybrid models. LRU eviction at this "
            "cap. Set to 0 to disable per-chat caching entirely. Env fallback: "
            "MLX_VLM_CACHE_SESSION_MAX. Default: 8."
        ),
    )
    parser.add_argument(
        "--cache-chat-id-header",
        type=str,
        default=os.environ.get("MLX_VLM_CACHE_CHAT_ID_HEADER", "X-MLX-VLM-Chat-Id"),
        help=(
            "HTTP header name carrying the chat_id used to key per-chat "
            "PromptCacheState. Falls back to body fields chat_id / "
            "metadata.chat_id if the header is absent. Env fallback: "
            "MLX_VLM_CACHE_CHAT_ID_HEADER. Default: X-MLX-VLM-Chat-Id."
        ),
    )
    parser.add_argument(
        "--cache-anon-sessions",
        type=str,
        default=_env_choice("MLX_VLM_CACHE_ANON_SESSIONS", "on", ["on", "off"]),
        choices=["on", "off"],
        help=(
            "Route requests that arrive without an explicit chat_id to the "
            "per-chat session whose stored turn-hash chain shares the longest "
            "prefix with this request. Default: on. Turn off in multi-user "
            "deployments where cache-hit timing could be a side-channel. Env "
            "fallback: MLX_VLM_CACHE_ANON_SESSIONS."
        ),
    )
    parser.add_argument(
        "--deltanet-rewind",
        type=str,
        default=_env_choice("MLX_VLM_DELTANET_REWIND", "auto", ["on", "off", "auto"]),
        choices=["on", "off", "auto"],
        help=(
            "Hybrid-cache rewind master switch for models with non-trimmable "
            "layers (Qwen 3.5/3.6 GatedDeltaNet, Mamba-style, etc.). 'auto' and "
            "'on' enable the snapshot-restore path; 'off' forces full re-prefill "
            "on every hybrid-model rewind. Env fallback: MLX_VLM_DELTANET_REWIND. "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--deltanet-ring-size",
        type=int,
        default=_env_int("MLX_VLM_DELTANET_RING_SIZE", DEFAULT_RING_SIZE),
        help=(
            "Number of DeltaNet state snapshots retained per session (FIFO). "
            "Set to 0 to disable snapshots (forces full re-prefill on hybrid "
            "rewind). Env fallback: MLX_VLM_DELTANET_RING_SIZE. Default: 3."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )
    args = parser.parse_args()
    if args.trust_remote_code:
        os.environ["MLX_TRUST_REMOTE_CODE"] = "true"
    if args.model:
        os.environ["MLX_VLM_PRELOAD_MODEL"] = args.model
        if args.adapter_path:
            os.environ["MLX_VLM_PRELOAD_ADAPTER"] = args.adapter_path
    os.environ["MLX_VLM_VISION_CACHE_SIZE"] = str(args.vision_cache_size)
    if args.draft_model:
        os.environ["MLX_VLM_DRAFT_MODEL"] = args.draft_model
        if args.draft_kind is not None:
            os.environ["MLX_VLM_DRAFT_KIND"] = args.draft_kind
        if args.draft_block_size is not None:
            os.environ["MLX_VLM_DRAFT_BLOCK_SIZE"] = str(args.draft_block_size)
    if args.prefill_step_size:
        os.environ["PREFILL_STEP_SIZE"] = str(args.prefill_step_size)
    os.environ["MLX_VLM_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["MLX_VLM_ENABLE_THINKING"] = "1" if args.enable_thinking else "0"
    if args.kv_bits is not None:
        os.environ["KV_BITS"] = str(args.kv_bits)
    os.environ["KV_GROUP_SIZE"] = str(args.kv_group_size)
    os.environ["KV_QUANT_SCHEME"] = args.kv_quant_scheme
    if args.max_kv_size is not None:
        os.environ["MAX_KV_SIZE"] = str(args.max_kv_size)
    os.environ["QUANTIZED_KV_START"] = str(args.quantized_kv_start)
    if args.top_logprobs_k is not None:
        os.environ["TOP_LOGPROBS_K"] = str(args.top_logprobs_k)

    # Publish per-chat / DeltaNet config to the session manager. argparse has
    # already resolved the CLI-arg > env-var > default precedence chain.
    _configure_session_manager(
        deltanet_ring_size=max(0, int(args.deltanet_ring_size)),
        deltanet_rewind_enabled=args.deltanet_rewind.lower() != "off",
        session_cache_max=max(0, int(args.cache_session_max)),
        chat_id_header=args.cache_chat_id_header,
        cache_anon_sessions=args.cache_anon_sessions.lower() != "off",
    )
    logger.info(
        "Per-chat cache: explicit-chat-id %s, anonymous hash-chain matching %s",
        "enabled" if args.cache_session_max > 0 else "disabled",
        (
            "enabled"
            if args.cache_anon_sessions.lower() != "off" and args.cache_session_max > 0
            else "disabled"
        ),
    )

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(log_level)

    uvicorn.run(
        "mlx_vlm.server:app",
        host=args.host,
        port=args.port,
        workers=1,
        reload=args.reload,
        server_header=False,
    )
