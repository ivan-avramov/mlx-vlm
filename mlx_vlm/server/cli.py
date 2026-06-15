import argparse
import logging
import os
import sys

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

_LOG_NAME = os.environ.get("MLX_VLM_LOG_NAME", "mlx_vlm")
logger = logging.getLogger(f"{_LOG_NAME}.server")


def _model_num_attention_heads(model_path):
    """Read the language model's query-head count from config.json (cheap: only the
    config file is fetched, not weights). Returns None if it can't be determined."""
    import json

    try:
        if model_path and os.path.isdir(model_path):
            cfg_file = os.path.join(model_path, "config.json")
        else:
            from huggingface_hub import hf_hub_download

            cfg_file = hf_hub_download(model_path, "config.json")
        with open(cfg_file, encoding="utf-8") as f:
            cfg = json.load(f)
        tc = cfg.get("text_config", cfg)
        return tc.get("num_attention_heads") or cfg.get("num_attention_heads")
    except Exception as e:  # noqa: BLE001 - best-effort; fall back to a safe default
        logger.warning("cache-limit auto-derive: could not read num_attention_heads (%s)", e)
        return None


def _derive_cache_limit_gb(model_path, max_kv_size, prefill_step):
    """Auto-size the buffer-pool cap to one full-attention layer's QK^T score tensor at
    the model's MAX context, so it never undershoots at runtime (real ctx <= max_kv_size).

    cap = ceil( n_heads x prefill_step x max_kv_size x 2 bytes / 1e9 ) + 2 GB margin

    The +2 GB and round-up cover the co-resident transients (Q/MLP/dequant/hidden,
    measured ~1 GB) and absorb estimation slack. Heads come from the model config;
    fallback is 32 (the largest in this stack) so the fallback over- rather than
    under-shoots (undershoot only costs prefill speed, never correctness).
    """
    import math

    if not (max_kv_size and prefill_step):
        return None
    heads = _model_num_attention_heads(model_path) or 32
    scores_gb = heads * prefill_step * max_kv_size * 2 / 1e9
    return math.ceil(scores_gb) + 2.0


def _apply_mlx_memory_limits(
    cache_limit_gb, memory_limit_frac, model_path=None, max_kv_size=None, prefill_step=None
):
    """Bound MLX's Metal allocator at server startup.

    Long-context prefill's apparent "memory leak" is reclaimable buffer-pool
    retention (``get_cache_memory``), not KV/weights: the pool grows toward
    physical RAM and pages before MLX's default limit (~device size) evicts it.
    ``cache_limit_gb`` caps the pool (RSS ≈ active_peak + cache_limit_gb); when not
    given explicitly it is auto-derived from the model's heads x prefill_step x
    max_kv_size. ``memory_limit_frac`` sets a total-memory backstop so MLX evicts the
    pool before the OS swaps, and adapts to each machine's physical RAM.
    """
    import mlx.core as mx

    GB = 1024**3
    if not (cache_limit_gb and cache_limit_gb > 0):
        derived = _derive_cache_limit_gb(model_path, max_kv_size, prefill_step)
        if derived:
            logger.info(
                "MLX buffer-pool cache limit auto-derived: %.0f GB "
                "(heads x %s step x %s ctx x 2B + 2GB margin)",
                derived,
                prefill_step,
                max_kv_size,
            )
            cache_limit_gb = derived
    if cache_limit_gb and cache_limit_gb > 0:
        mx.set_cache_limit(int(cache_limit_gb * GB))
        logger.info("MLX buffer-pool cache limit: %.1f GB", cache_limit_gb)
    if memory_limit_frac and memory_limit_frac > 0:
        try:
            phys = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, AttributeError, OSError):
            phys = 0
        try:
            rec = int(mx.device_info().get("max_recommended_working_set_size", 0))
        except Exception:
            rec = 0
        candidates = [v for v in (int(memory_limit_frac * phys), rec) if v > 0]
        if candidates:
            limit = min(candidates)
            mx.set_memory_limit(limit)
            logger.info(
                "MLX memory limit: %.1f GB (%.2f×%.0f GB phys; rec WSS %.1f GB)",
                limit / GB,
                memory_limit_frac,
                (phys / GB) if phys else 0,
                (rec / GB) if rec else 0,
            )


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
        "--cache-limit-gb",
        type=float,
        default=None,
        help=(
            "Cap MLX's Metal buffer-reuse pool (get_cache_memory) at this many GB. "
            "Bounds RSS to ~= active_peak + cache_limit_gb instead of letting the pool "
            "grow toward physical RAM during long-context prefill. Should be >= one "
            "attention layer's score tensor at your max context (~6 GB for <=128K, "
            "~10 GB for 256K). Default: unset (MLX default ~= device size)."
        ),
    )
    parser.add_argument(
        "--memory-limit-frac",
        type=float,
        default=None,
        help=(
            "Set MLX's total-memory limit to this fraction of physical RAM (capped at "
            "the Metal recommended working-set size). MLX evicts the buffer pool before "
            "crossing it, preventing OS swap. e.g. 0.85. Default: unset."
        ),
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
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level. Env: MLX_VLM_LOG_LEVEL (default: INFO).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log output destination. Use '<stdout>' for stdout, or a file path. "
        "Env: MLX_VLM_LOG_FILE (default: <stdout>).",
    )
    args = parser.parse_args()

    # Configure logging — CLI args override env vars, env vars override defaults
    log_level_str = args.log_level or os.environ.get("MLX_VLM_LOG_LEVEL", "INFO")
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    log_file = args.log_file or os.environ.get("MLX_VLM_LOG_FILE", "<stdout>")

    log_kwargs = {
        "level": log_level,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }
    if log_file == "<stdout>":
        log_kwargs["stream"] = sys.stdout
    else:
        log_kwargs["filename"] = log_file

    logging.basicConfig(**log_kwargs)
    # Set level on the base logger so all mlx_vlm.* loggers inherit it
    logging.getLogger(_LOG_NAME).setLevel(log_level)
    logger.setLevel(log_level)

    _apply_mlx_memory_limits(
        args.cache_limit_gb,
        args.memory_limit_frac,
        model_path=args.model,
        max_kv_size=args.max_kv_size,
        prefill_step=args.prefill_step_size or DEFAULT_PREFILL_STEP_SIZE,
    )

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

    logger.debug("Command-line arguments: %s", args)
    logger.info("Starting MLX VLM Server")
    logger.info("Host: %s, Port: %s", args.host, args.port)
    logger.info("Model: %s, Adapter: %s", args.model, args.adapter_path)

    uvicorn.run(
        "mlx_vlm.server:app",
        host=args.host,
        port=args.port,
        workers=1,
        reload=args.reload,
        server_header=False,
    )
