from __future__ import annotations

import contextlib
import logging
import os
import threading as _threading
from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce

from ..models import cache
from ..models.cache import PreallocKVCache, PreallocQuantizedKVCache
from ..turboquant import TurboQuantKVCache, turboquant_enabled

_LOG_NAME = os.environ.get("MLX_VLM_LOG_NAME", "mlx_vlm")
logger = logging.getLogger(f"{_LOG_NAME}.generate")

DEFAULT_KV_GROUP_SIZE = 64
DEFAULT_KV_QUANT_SCHEME = "uniform"
DEFAULT_QUANTIZED_KV_START = 5000


# ---------------------------------------------------------------------------
# Lazy per-thread generation stream.
#
# Upstream used a single module-level ``mx.new_thread_local_stream`` singleton.
# That breaks when generation runs on worker threads (asyncio executors,
# ThreadPoolExecutor, the server's BatchGenerator GPU thread): a stream object
# created on one thread is not the one ``mx.async_eval`` resolves on another,
# producing "no Stream(gpu, N) in current thread" errors.
#
# The architecturally-sound fix is to lazily resolve each thread's own default
# stream the first time that thread enters generation code. Call sites use
# ``_get_generation_stream()`` directly. The module-level ``generation_stream``
# attribute is preserved via ``__getattr__`` for backward compatibility (older
# callers / re-exports), materializing the calling thread's stream on access.
# ---------------------------------------------------------------------------
_thread_local_streams = _threading.local()


def _get_generation_stream():
    """Return (and lazily resolve) the calling thread's generation stream.

    Uses ``mx.default_stream(device)`` rather than
    ``mx.new_thread_local_stream``: the former returns each thread's
    auto-registered default stream (matching the BatchGenerator GPU thread's
    pattern in the server), while the latter creates a new stream object that
    ``mx.async_eval`` cannot find on subsequent calls — the "no Stream(gpu, N)
    in current thread" error.
    """
    stream = getattr(_thread_local_streams, "stream", None)
    if stream is None:
        stream = mx.default_stream(mx.default_device())
        _thread_local_streams.stream = stream
    return stream


# ---------------------------------------------------------------------------
# Cache trimming / inspection utilities.
#
# These power the prompt-cache reuse path (trim cached state to the common
# prefix before resuming forward) and the post-generation anchor path (trim
# the just-generated assistant turn back off so the persisted cache always
# ends at an end-of-user-turn boundary).
# ---------------------------------------------------------------------------
def _kv_seq_axis(shape) -> int:
    """Return the sequence-length axis for a KV cache tensor.

    MLX convention is [B, H, L, D] (axis 2).  Some models use [B, L, H, D]
    (axis 1).  We disambiguate by checking ndim and falling back to 2 when the
    two middle dims are equal (which is the common MLX layout).
    """
    if len(shape) < 3:
        return 1
    # Unambiguous cases
    if shape[1] > shape[2]:
        return 1
    if shape[2] > shape[1]:
        return 2
    # Equal dims — default to the standard MLX layout [B, H, L, D]
    return 2


def _cache_kind_names(c) -> frozenset:
    """Every class name in ``type(c)``'s MRO.

    Cache-type dispatch here is by class *name* rather than ``isinstance`` so
    the fork does not have to import cache classes that only exist in some
    mlx_lm versions. Matching against the whole MRO instead of
    ``type(c).__name__`` is what makes that safe for subclasses: our own
    ``BufferedRotatingKVCache(RotatingKVCache)`` is a rotating ring buffer and
    must be treated as one everywhere, but an exact-name check silently
    classifies it as a plain flat cache.
    """
    return frozenset(t.__name__ for t in type(c).__mro__)


def _trim_cache(c, target_len):
    """Recursively trim a KV cache (or nested container of caches) so its
    sequence dimension is exactly ``target_len``.

    Handles list / tuple containers, hybrid wrappers exposing ``.caches``,
    and the common per-layer cache shapes. Static caches (TurboQuant,
    Rotating SWA) manage their own internal structure via ``.trim()`` /
    ``.truncate()``; standard caches additionally need a physical slice
    on the K/V tensors so ghost tokens / step-padding don't desync RoPE
    on the next forward pass.

    Used in two places: (1) start-of-generation prefix-cache reuse —
    trim cached state to the common prefix before resuming forward;
    (2) end-of-generation discard of the assistant turn — trim back to
    prefill length so the persisted cache state always ends at an
    end-of-user-turn boundary, sidestepping the asymmetric-rendering
    problem (thinking content present in cache but absent in next
    request's echoed assistant message).
    """
    if isinstance(c, (list, tuple)):
        for sub_c in c:
            _trim_cache(sub_c, target_len)
        return
    if hasattr(c, "caches"):
        for sub_c in c.caches:
            _trim_cache(sub_c, target_len)
        return
    if not hasattr(c, "offset"):
        return
    current_offset = int(c.offset.item() if hasattr(c.offset, "item") else c.offset)
    if current_offset <= target_len:
        return
    # Update MLX native state pointers
    if hasattr(c, "trim"):
        c.trim(current_offset - target_len)
    elif hasattr(c, "truncate"):
        c.truncate(target_len)
    else:
        c.offset = target_len

    # Caches that own their internal storage layout — leave the K/V
    # structure alone after the trim/truncate above moved the offset.
    #   - TurboQuantKVCache, RotatingKVCache, BatchRotatingKVCache:
    #     ring/static-sized buffers; physical slicing would corrupt
    #     them.
    #   - QuantizedKVCache, BatchQuantizedKVCache: `.keys` and
    #     `.values` are 3-element lists `[quantized_uint32, scales,
    #     biases]` (a single layer of state, not a chunked sequence).
    #     The chunked-list branch below would treat the 3 components
    #     as 3 sequence chunks and silently strip scales+biases —
    #     producing a 1-element list that breaks the next call to
    #     `mx.quantized_matmul(queries, *q_keys, ...)` with a "missing
    #     scales argument" TypeError.
    #   Matched against the full MRO: BufferedRotatingKVCache subclasses
    #   RotatingKVCache, and slicing its ring buffer physically while
    #   ``.trim()`` above already moved the offset logically leaves the ring
    #   index desynced from ``offset`` -- silent wrong output, or a broadcast
    #   crash on the next update.
    if _cache_kind_names(c) & {
        "TurboQuantKVCache",
        "RotatingKVCache",
        "BatchRotatingKVCache",
        "QuantizedKVCache",
        "BatchQuantizedKVCache",
    }:
        return

    k_attr = "keys" if hasattr(c, "keys") else "k" if hasattr(c, "k") else None
    v_attr = "values" if hasattr(c, "values") else "v" if hasattr(c, "v") else None
    if not (k_attr and v_attr):
        return
    k_arr = getattr(c, k_attr)
    v_arr = getattr(c, v_attr)
    if k_arr is None or v_arr is None:
        return

    if isinstance(k_arr, mx.array):
        seq_axis = _kv_seq_axis(k_arr.shape)
        if seq_axis == 1:
            setattr(c, k_attr, k_arr[:, :target_len, ...])
            setattr(c, v_attr, v_arr[:, :target_len, ...])
        else:
            setattr(c, k_attr, k_arr[:, :, :target_len, ...])
            setattr(c, v_attr, v_arr[:, :, :target_len, ...])
    elif isinstance(k_arr, list):
        # ChunkedKVCache list slicing
        new_k, new_v = [], []
        current_len = 0
        for k, v in zip(k_arr, v_arr):
            seq_axis = _kv_seq_axis(k.shape)
            chunk_len = k.shape[seq_axis]
            if current_len + chunk_len <= target_len:
                new_k.append(k)
                new_v.append(v)
                current_len += chunk_len
            elif current_len < target_len:
                sl = target_len - current_len
                if seq_axis == 1:
                    new_k.append(k[:, :sl, ...])
                    new_v.append(v[:, :sl, ...])
                else:
                    new_k.append(k[:, :, :sl, ...])
                    new_v.append(v[:, :, :sl, ...])
                break
            else:
                break
        setattr(c, k_attr, new_k)
        setattr(c, v_attr, new_v)

    # Force execution to clean graph
    mx.eval(getattr(c, k_attr), getattr(c, v_attr))


def _first_kv_offset(entries) -> int:
    """Return the offset of the first cache layer that exposes an
    ``offset`` attribute, recursing into nested ``.caches`` containers.

    Hybrid topologies (Qwen 3.5/3.6 GatedDeltaNet, Mamba-style) mix
    ``ArraysCache`` (recurrent state, no ``offset``) with attention
    KV layers (``KVCache``/``RotatingKVCache``/``TurboQuantKVCache``,
    which do have ``offset``). All KV layers advance in lockstep
    during prefill, so reading any one is correct — but the first
    layer is not guaranteed to be one of them. Returns 0 for empty
    input or all-recurrent topologies.
    """
    for c in entries:
        if hasattr(c, "offset"):
            return int(c.offset.item() if hasattr(c.offset, "item") else c.offset)
        if hasattr(c, "caches"):
            inner = _first_kv_offset(c.caches)
            if inner:
                return inner
        elif isinstance(c, (list, tuple)):
            inner = _first_kv_offset(c)
            if inner:
                return inner
    return 0


def _policy_enabled(policy) -> bool:
    return bool(getattr(policy, "enabled", policy))


def _chunked_prefill_enabled(
    model,
    *,
    input_ids=None,
    inputs_embeds=None,
    prompt_cache=None,
    draft_model=None,
    draft_kind=None,
    prefill_kwargs=None,
) -> bool:
    prefill_kwargs = prefill_kwargs or {}
    candidates = [model]
    language_model = getattr(model, "language_model", None)
    if language_model is not None and language_model is not model:
        candidates.append(language_model)

    for candidate in candidates:
        policy = getattr(candidate, "chunked_prefill_policy", None)
        if callable(policy):
            return _policy_enabled(
                policy(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    prompt_cache=prompt_cache,
                    draft_model=draft_model,
                    draft_kind=draft_kind,
                    prefill_kwargs=prefill_kwargs,
                )
            )

    if any(getattr(candidate, "no_chunked_prefill", False) for candidate in candidates):
        return False

    # Hidden-state speculative prefill is model-contract dependent. Keep unknown
    # target models conservative unless they expose a chunked_prefill_policy.
    return draft_model is None


def maybe_quantize_kv_cache(
    prompt_cache,
    quantized_kv_start,
    kv_group_size,
    kv_bits,
    kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
    max_kv_size: Optional[int] = None,
    serialize_kv_quantization: bool = False,
    kv_prealloc_tokens: Optional[int] = None,
):
    if kv_bits is None:
        return

    if turboquant_enabled(kv_bits, kv_quant_scheme):

        def quantize_entry(entry):
            if isinstance(entry, TurboQuantKVCache):
                return entry
            if isinstance(entry, cache.RotatingKVCache):
                return entry
            # Support SimpleKVCache and ChunkedKVCache in addition to KVCache
            if isinstance(
                entry, (cache.KVCache, cache.SimpleKVCache, cache.ChunkedKVCache)
            ):
                current_offset = getattr(entry, "offset", 0)
                if current_offset == 0:
                    # Empty: replace so update_and_fetch quantizes on the fly
                    return TurboQuantKVCache(
                        bits=kv_bits,
                        max_kv_size=max_kv_size,
                        prealloc_tokens=kv_prealloc_tokens,
                    )
                if current_offset < quantized_kv_start:
                    return entry
                return TurboQuantKVCache.from_cache(
                    entry,
                    bits=kv_bits,
                    max_kv_size=max_kv_size,
                    prealloc_tokens=kv_prealloc_tokens,
                )
            if isinstance(entry, cache.CacheList):
                entry.caches = [quantize_entry(sub_entry) for sub_entry in entry.caches]
                return entry
            if isinstance(entry, list):
                for i, sub_entry in enumerate(entry):
                    entry[i] = quantize_entry(sub_entry)
                return entry
            if isinstance(entry, tuple):
                return tuple(quantize_entry(sub_entry) for sub_entry in entry)
            return entry

        # Skip the last layer (before final norm/LM head) — it's highly
        # sensitive to quantization in deep models (e.g. gemma-4-31b).
        last_idx = len(prompt_cache) - 1 if len(prompt_cache) > 2 else -1
        for index, layer_cache in enumerate(prompt_cache):
            if index == last_idx:
                continue

            # Check if this layer is currently unquantized
            was_unquantized = isinstance(
                layer_cache,
                (cache.KVCache, cache.SimpleKVCache, cache.ChunkedKVCache),
            )

            prompt_cache[index] = quantize_entry(layer_cache)

            # Serialize Graph Compilation: evaluate each layer immediately after
            # conversion to prevent MLX from building one giant FP32 intermediate
            # graph for all layers at once (which causes multi-GB OOM spikes).
            if (
                serialize_kv_quantization
                and was_unquantized
                and isinstance(prompt_cache[index], TurboQuantKVCache)
            ):
                mx.eval(prompt_cache[index].keys, prompt_cache[index].values)
        return

    # Uniform quantization path — replaces upstream mlx_maybe_quantize_kv_cache
    # to safely skip RotatingKVCache (which raises NotImplementedError) and
    # support serialized per-layer evaluation.
    kv_bits_int = int(kv_bits)
    for index, c in enumerate(prompt_cache):
        if isinstance(c, cache.RotatingKVCache):
            continue
        if not hasattr(c, "to_quantized") or c.offset < quantized_kv_start:
            continue
        prompt_cache[index] = c.to_quantized(group_size=kv_group_size, bits=kv_bits_int)
        if serialize_kv_quantization:
            keys, values = prompt_cache[index].state
            if keys is not None:
                mx.eval(keys, values)


def maybe_preallocate_kv_cache(prompt_cache, kv_prealloc_tokens):
    """Convert leftover plain fp16 / uniform-quantized caches to their pre-allocating
    variants (including non-empty ones — copies content). Runs AFTER
    maybe_quantize_kv_cache so it never fp16-pre-allocs a to-be-quantized layer.
    Idempotent: a cache already a Prealloc*/TQ variant is left as-is (only its floor
    is refreshed)."""
    if not kv_prealloc_tokens:
        return
    floor = int(kv_prealloc_tokens)
    for i, entry in enumerate(prompt_cache):
        # Already a pre-alloc variant → just refresh the floor (idempotent).
        if isinstance(entry, (PreallocKVCache, PreallocQuantizedKVCache)):
            entry.prealloc_tokens = floor
        # QuantizedKVCache first (PreallocQuantizedKVCache subclasses it, handled above).
        elif isinstance(entry, cache.QuantizedKVCache):
            prompt_cache[i] = PreallocQuantizedKVCache.from_quantized(entry, floor)
        # Plain KVCache last (PreallocKVCache subclasses it, handled above).
        elif isinstance(entry, cache.KVCache):
            prompt_cache[i] = PreallocKVCache.from_kvcache(entry, floor)
        # TurboQuantKVCache, RotatingKVCache, ChunkedKVCache, linear-attn, CacheList: untouched.


@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """Temporarily set the wired memory limit for generation.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    if not mx.metal.is_available():
        yield
        return

    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    max_rec_size = mx.device_info()["max_recommended_working_set_size"]
    if model_bytes > 0.9 * max_rec_size:
        model_mb = model_bytes // 2**20
        max_rec_mb = max_rec_size // 2**20
        logger.warning(
            "Generating with a model that requires %d MB "
            "which is close to the maximum recommended size of %d "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models",
            model_mb,
            max_rec_mb,
        )
    old_limit = mx.set_wired_limit(max_rec_size)
    try:
        yield
    finally:
        if streams is not None:
            for stream in streams:
                mx.synchronize(stream)
        else:
            mx.synchronize()
        mx.set_wired_limit(old_limit)


@dataclass
class GenerationResult:
    text: str = ""
    token: Optional[int] = None
    logprobs: Optional[List[float]] = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    total_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory: float = 0.0
    cached_tokens: int = 0
    # Populated only on the terminal chunk yielded by ``stream_generate``:
    #   "stop"   - tokenizer.stopping_criteria matched (EOS / chat-template stop)
    #   "length" - max_tokens reached without a stop token
    #   None     - mid-stream chunk; no terminal signal yet
    finish_reason: Optional[str] = None
    diffusion_canvas_tokens: int = 0
    diffusion_denoising_steps: int = 0
    diffusion_work_tokens: int = 0
    diffusion_canvas_tps: float = 0.0
    diffusion_work_tps: float = 0.0
    is_draft: bool = False
    draft_text: str = ""
    text_already_printed: bool = False
    diffusion_step: int = 0
    diffusion_total_steps: int = 0
    diffusion_canvas_index: int = 0
    diffusion_block_complete: bool = False


class PromptCacheState:
    """Holds KV cache and token history across conversation turns.

    Pass this to stream_generate via the ``prompt_cache_state`` kwarg to
    reuse the KV cache from previous turns.  Only the new tokens (after
    the common prefix) are processed, avoiding redundant prefill.

    For hybrid models with non-trimmable layers (Qwen 3.5/3.6 GatedDeltaNet,
    Mamba-style, etc.), a ``snapshot_ring`` (auto-created by default) enables
    rewind support. Without one, mid-conversation rewinds (regenerate, edit)
    fall back to full re-prefill since the standard ``_trim_cache`` path can't
    safely rewind recurrent state.

    ``rewind_enabled`` is the master toggle for the snapshot-restore code
    path. ``False`` means snapshots are still captured (cheap; future
    sessions might toggle on) but never used to restore — the rewind guard
    falls through to full re-prefill. Useful for A/B testing or when an
    operator wants to disable the restore path without rebuilding sessions.

    ``is_asymmetric_rendering`` is set by the server after asymmetry detection
    on the (processor, template_kwargs) pair. When True, post-generation cache
    anchors at end-of-user via rotating-layer snapshot+restore. When False
    (default), cache anchors at end-of-asst (forward extension on next
    request).
    """

    def __init__(self, snapshot_ring=None, rewind_enabled: bool = True):
        self.cache: Optional[List[Any]] = None
        self.token_ids: Optional[List[int]] = None
        self.rewind_enabled = rewind_enabled
        self.is_asymmetric_rendering: bool = False
        # Local import so this module stays importable without snapshot.py
        # being touched by callers that don't need it.
        if snapshot_ring is None:
            from ..snapshot import DeltaNetSnapshotRing

            snapshot_ring = DeltaNetSnapshotRing()
        self.snapshot_ring = snapshot_ring

    def find_prefix_length(self, new_ids: list) -> int:
        """Return the number of leading tokens that match the cached ids."""
        if self.token_ids is None:
            return 0
        max_len = min(len(self.token_ids), len(new_ids))
        for i in range(max_len):
            if self.token_ids[i] != new_ids[i]:
                return i
        return max_len

    def update(self, token_ids: list, kv_cache: list):
        """Store the full token sequence and corresponding KV cache.

        Also captures a DeltaNet state snapshot at this offset (the current
        turn boundary) when a snapshot ring is attached and the cache
        contains non-trimmable layers. The capture is refcount-cheap.

        Snapshot invariant: each ``DeltaNetSnapshot`` represents the
        recurrent state after processing the cached token sequence's
        first ``offset`` tokens. When the new ``token_ids`` diverges from
        the previously-cached sequence at position ``d``, every snapshot
        with ``offset > d`` references state conditioned on tokens that
        are no longer in the live cache (e.g. the prior turn's assistant
        message was edited or replaced). Restoring such a snapshot would
        produce a DeltaNet state inconsistent with the trimmed KV layers
        and silently drift generation. Drop them eagerly here.
        """
        new_ids = list(token_ids)

        if (
            self.token_ids is not None
            and self.snapshot_ring is not None
            and self.snapshot_ring.enabled
        ):
            divergence = 0
            limit = min(len(self.token_ids), len(new_ids))
            while (
                divergence < limit and self.token_ids[divergence] == new_ids[divergence]
            ):
                divergence += 1
            # If the new sequence is shorter than the cached one OR diverges
            # mid-prefix, snapshots past `divergence` are stale.
            if divergence < len(self.token_ids):
                dropped = self.snapshot_ring.drop_after(divergence)
                if dropped:
                    logger.debug(
                        "Snapshot ring: dropped %d stale snapshot(s) past "
                        "token-sequence divergence at offset %d (cached len=%d, "
                        "new len=%d).",
                        dropped,
                        divergence,
                        len(self.token_ids),
                        len(new_ids),
                    )

        self.token_ids = new_ids
        self.cache = kv_cache
        if self.snapshot_ring is not None and self.snapshot_ring.enabled:
            self.snapshot_ring.capture(offset=len(self.token_ids), cache=kv_cache)


# ---------------------------------------------------------------------------
# SWA / rotating / hybrid-cache rewind guards + asymmetric-template anchoring.
#
# These guard silent-correctness invariants on multi-turn cache reuse:
#   * RotatingKVCache (SWA) ring buffers lose history once they wrap; rewinding
#     into the overwritten region produces ghost tokens.
#   * Hybrid models (GatedDeltaNet / Mamba via ArraysCache) advance recurrent
#     state monotonically with no rewind primitive — restore from a snapshot
#     or fall back to full re-prefill.
#   * Asymmetric chat templates (Gemma 4 thinking strip, OpenWebUI RAG-wrap)
#     re-render the latest user message differently per turn; anchoring the
#     cache BEFORE the latest user message keeps reuse safe.
# ---------------------------------------------------------------------------
def _rotating_rewind_safe(entries, target_len) -> bool:
    """Return True if every RotatingKVCache in ``entries`` (recursing
    into nested ``.caches`` and list/tuple containers) can safely rewind
    to ``target_len``.

    Ring buffers lose historical context once they wrap (offset > max_size).
    Rewinding into the overwritten region creates a "Memory Hole" and ghost
    tokens that cause Softmax anomalies (hallucination loops). However,
    if the buffer hasn't wrapped, all positions are still valid and a
    rewind is safe — just trim the offset.
    """
    for c in entries:
        if hasattr(c, "caches"):
            if not _rotating_rewind_safe(c.caches, target_len):
                return False
            continue
        if isinstance(c, (list, tuple)):
            if not _rotating_rewind_safe(c, target_len):
                return False
            continue

        # Buffered / chunked caches evict by advancing start_position, so
        # anything before it is gone no matter what the ring arithmetic below
        # says. BufferedRotatingKVCache (what speculative decoding installs)
        # reports itself trimmable even after evicting, so this has to be
        # checked explicitly rather than inferred from .is_trimmable().
        start_position = getattr(c, "start_position", None)
        if start_position is not None and target_len < int(start_position):
            return False

        # Subclass-aware on purpose: a BufferedRotatingKVCache built directly
        # (rather than via .from_cache) keeps start_position at 0 while its ring
        # wraps, so the check above cannot be the only one that covers it.
        if _cache_kind_names(c) & {"RotatingKVCache", "BatchRotatingKVCache"}:
            offset = int(c.offset.item() if hasattr(c.offset, "item") else c.offset)
            max_size = getattr(c, "max_size", None)
            if max_size is not None and offset > max_size:
                # Buffer has wrapped — data before (offset - max_size) is gone
                if target_len < (offset - max_size):
                    return False
    return True


ROTATING_CACHE_KINDS = frozenset({"RotatingKVCache", "BatchRotatingKVCache"})


def _is_rotating_kv_layer(c) -> bool:
    """True if `c` is a rotating (sliding-window) KV cache, subclasses included.

    This is a *routing* predicate, not a label: the post-generation path uses it
    to decide whether a layer is rewound by snapshot restore (rotating) or by
    ``_trim_cache`` (flat). Matching on ``type(c).__name__`` sent
    ``BufferedRotatingKVCache`` down the flat path, where it was neither
    captured by ``_capture_rotating_layers_for_snapshot`` nor safe to trim --
    a wrapped ring rewound with no snapshot to restore from is exactly the
    "memory hole" this machinery exists to prevent.
    """
    return bool(_cache_kind_names(c) & ROTATING_CACHE_KINDS)


def _anchor_within_loop_range(
    initial_cache_offset: int,
    prompt_len: int,
    snapshot_at_offset: Optional[int],
) -> bool:
    """Return True iff ``snapshot_at_offset`` falls strictly inside the
    chunked-prefill loop's reachable range.

    The loop processes tokens in [initial_cache_offset,
    initial_cache_offset + prompt_len - 1) — the final token is
    reserved for the post-loop ``_step`` forward pass that produces
    the first generation logits, so it can't be a snapshot landing
    point. Anchors equal to ``initial_cache_offset`` are also
    excluded: the cache is already at that offset, no capture is
    needed (and the loop wouldn't iterate at that boundary anyway).

    Used to gate loop entry: when the prompt is shorter than
    ``prefill_step_size`` (typical after good cache reuse) the loop
    is normally skipped, but if an anchor target falls inside the
    range we MUST enter the loop or lose the asymmetric-rendering
    anchor for that turn — leading to a full re-prefill on the next
    turn when the client re-renders the user message differently.
    """
    if snapshot_at_offset is None:
        return False
    return (
        initial_cache_offset < snapshot_at_offset
        and snapshot_at_offset < initial_cache_offset + prompt_len - 1
    )


def _should_capture_anchor_pre_prefill(
    snapshot_at_offset: Optional[int],
    initial_cache_offset: int,
    prompt_cache_present: bool,
) -> bool:
    """Return True iff the anchor capture should fire BEFORE the
    chunked-prefill loop runs (capturing from the cache state at
    start of ``generate_step``).

    Fires when the cache state at start of this turn IS the anchor
    state we want to persist — i.e. the latest-user-turn marker is
    at or before the current cache offset. Two situations match:

      * **Equality** (``initial == snapshot``): typical OWUI tool-
        continuation steady state. The user-turn marker hasn't
        moved across multiple chat-completion calls, and the prior
        call already persisted at exactly that anchor.

      * **Strict-greater** (``initial > snapshot``): degenerate
        transient where a prior call's fallback advanced the cache
        past the anchor we'd ideally want. We can't go back to the
        original anchor (SWA ring may have wrapped past it), so we
        capture at ``initial`` as best-available.

    The in-loop capture won't fire in either case because
    ``_classify_snapshot_action`` returns ``"skip"`` once
    ``cumulative_offset > snapshot_at_offset``, which happens after
    the first chunk. Cold-start (``prompt_cache_present=False``) is
    excluded — there's no live cache state to capture.
    """
    if snapshot_at_offset is None:
        return False
    if not prompt_cache_present:
        return False
    return initial_cache_offset >= snapshot_at_offset


def _adjust_chunk_for_snapshot_landing(
    cumulative_offset: int,
    n_to_process: int,
    snapshot_at_offset: Optional[int],
    snapshot_done: bool,
) -> int:
    """Shrink ``n_to_process`` so the chunked-prefill loop lands EXACTLY
    on ``snapshot_at_offset`` when the about-to-process chunk would
    otherwise cross it. No-op when there's no target, the target was
    already captured, or the chunk doesn't cross the target.

    Pure function — the boundary-alignment math is unit-tested without
    driving the model. The ``> snapshot_at_offset`` (strict inequality)
    is deliberate: if the chunk would land EXACTLY on the target with no
    shrink, no adjustment is needed.
    """
    if snapshot_at_offset is None or snapshot_done:
        return n_to_process
    if (
        cumulative_offset < snapshot_at_offset
        and cumulative_offset + n_to_process > snapshot_at_offset
    ):
        return snapshot_at_offset - cumulative_offset
    return n_to_process


def _classify_snapshot_action(
    cumulative_offset: int,
    snapshot_at_offset: Optional[int],
    snapshot_done: bool,
) -> str:
    """At a chunked-prefill chunk boundary, classify what to do with
    the rotating-layer snapshot. Returns one of:

      * ``"skip"`` — no target set, target already captured, or we
        overshot the target without a usable boundary on this chunk.
      * ``"capture_and_finalize"`` — chunk landed EXACTLY on the
        target offset. Capture and stop scanning further boundaries.
      * ``"capture_as_fallback"`` — chunk ended at an offset before
        the target. Capture as the best-available approximation,
        replacing any prior fallback so the latest (closest-to-target)
        wins. Don't finalize — a later iteration may land closer or
        on-target.

    The fallback path matters when ``snapshot_at_offset`` can't be hit
    exactly during chunked prefill (BPE context-sensitivity makes the
    helper's prefix tokenization disagree by a few tokens with the
    prefill's cumulative tally).
    """
    if snapshot_at_offset is None or snapshot_done:
        return "skip"
    if cumulative_offset == snapshot_at_offset:
        return "capture_and_finalize"
    if cumulative_offset < snapshot_at_offset:
        return "capture_as_fallback"
    return "skip"


def _compute_anchor_before_latest_user_offset(
    formatted_prompt: str,
    tokenizer,
) -> Optional[int]:
    """Find the token offset just before the LATEST user-turn-open marker
    in ``formatted_prompt``. Returns None if no known user-turn marker
    is present.

    The asymmetric-rendering cache path uses this offset as the anchor:
    the cache is persisted holding tokens [0, offset), so the next
    request's prompt can re-render the latest user message in any
    shape (e.g. OpenWebUI's RAG ``<context>`` wrapping) without
    triggering a backward trim. The last user message itself is
    re-prefilled forward on every turn.

    Per-template markers come from ``prompt_utils.USER_TURN_OPEN_MARKERS``;
    we scan all of them and pick the LAST occurrence of ANY marker. The
    character position of that marker becomes the anchor's substring
    boundary; we re-tokenize just the prefix and return its token count.
    """
    from ..prompt_utils import USER_TURN_OPEN_MARKERS

    last_pos = -1
    for marker in USER_TURN_OPEN_MARKERS:
        idx = formatted_prompt.rfind(marker)
        if idx > last_pos:
            last_pos = idx
    if last_pos < 0:
        return None

    prefix = formatted_prompt[:last_pos]
    try:
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    except TypeError:
        # Some tokenizers (e.g. SentencePiece-only) don't accept
        # ``add_special_tokens``; fall back to plain encode.
        prefix_ids = tokenizer.encode(prefix)
    return len(prefix_ids)


def _capture_rotating_layers_for_snapshot(cache_list, capture_fn):
    """Walk a flat cache list and snapshot every rotating layer.

    Returns a list of ``RotatingKVSnapshot`` objects (one per rotating
    layer found). Each snapshot records its ``layer_index`` so the
    restore path can match it back to the same position.
    """
    snapshots = []
    for idx, c in enumerate(cache_list):
        if _is_rotating_kv_layer(c):
            snapshots.append(capture_fn(c, idx))
    return snapshots


def _restore_rotating_layers_from_snapshots(cache_list, snapshots) -> None:
    """Restore each snapshot back into its corresponding live layer."""
    from ..snapshot import restore_rotating

    by_idx = {s.layer_index: s for s in snapshots}
    for idx, c in enumerate(cache_list):
        snap = by_idx.get(idx)
        if snap is not None and _is_rotating_kv_layer(c):
            restore_rotating(c, snap)


def _capture_arrays_layers_for_snapshot(cache_list):
    """Walk a flat cache list and snapshot every ``ArraysCache`` layer.

    Returns a list aligned positionally with ``cache_list``: ``None`` for
    non-ArraysCache layers, ``list(c.state)`` (refcount-cheap copy of the
    layer's state list) for ArraysCache layers. Returns an empty list if
    no ArraysCache layers are present (pure-attention model — caller
    skips the restore step entirely).

    Used by the asymmetric-rendering anchor path on hybrid topologies
    (Qwen 3.5/3.6 GatedDeltaNet, Mamba) where recurrent (``ArraysCache``)
    state needs to be preserved at the before-latest-user boundary.
    """
    from mlx_vlm.models.cache import ArraysCache

    snapshots = []
    any_arrays = False
    for c in cache_list:
        if isinstance(c, ArraysCache):
            snapshots.append(list(c.state))
            any_arrays = True
        else:
            snapshots.append(None)
    return snapshots if any_arrays else []


def _restore_arrays_layers_from_snapshots(cache_list, snapshots) -> None:
    """Restore captured ``ArraysCache`` state back onto the live layers.

    Snapshots align positionally with ``cache_list`` — None entries
    correspond to non-ArraysCache layers (skipped). No-op when
    ``snapshots`` is empty (pure-attention model).
    """
    if not snapshots:
        return
    from mlx_vlm.models.cache import ArraysCache

    for c, s in zip(cache_list, snapshots):
        if s is not None and isinstance(c, ArraysCache):
            c.state = s


def _capture_anchor_state(
    prompt_cache,
    offset: int,
    rotating_capture: Optional[List[Any]],
    arrays_capture: Optional[List[Optional[List[mx.array]]]],
    anchor_offset_list: Optional[List[int]],
) -> None:
    """Capture all three side-channels for an asymmetric-anchor landing:
    rotating-layer snapshots, ArraysCache snapshots, and the offset marker.

    Used at two sites: pre-prefill (cache state at start of generate_step)
    and in-loop (chunk boundary at or before ``snapshot_at_offset``).

    Each list is cleared before populating so latest-wins semantics hold
    across multiple in-loop calls (an exact landing replaces a prior
    fallback). ``None`` for any list means "this side-channel is not active
    for this request".
    """
    from ..snapshot import capture_rotating

    if rotating_capture is not None:
        captured_rot = _capture_rotating_layers_for_snapshot(
            prompt_cache, capture_rotating
        )
        if captured_rot:
            rotating_capture.clear()
            rotating_capture.extend(captured_rot)

    if arrays_capture is not None:
        captured_arr = _capture_arrays_layers_for_snapshot(prompt_cache)
        if captured_arr:
            arrays_capture.clear()
            arrays_capture.extend(captured_arr)

    if anchor_offset_list is not None:
        anchor_offset_list.clear()
        anchor_offset_list.append(offset)


def _rotating_post_gen_trim_safe(entries, target_len) -> bool:
    """Strict variant of ``_rotating_rewind_safe`` for the *post-generation*
    cache trim path.

    ``_rotating_rewind_safe`` only flags as unsafe the case where the trim
    target sits below the oldest-still-valid logical position
    (``offset - max_size``). That's correct for forward extension after a
    small backward trim — newly-prefilled tokens overwrite the slots near
    the trim point and the wraparound resolves itself.

    The post-gen trim is different: we trim AT the moment of persisting,
    before any further writes. If the ring buffer wrapped during generation
    (``offset > max_size``), the ring is no longer a coherent representation
    of the prefix [0, target_len), and the next request's attention will
    read garbage K/V from the misaligned slots. Empirically: Gemma 4 26B
    8-bit produced repetition loops on turn 2 when the lenient check passed
    but the strict invariant was violated.

    The strictly-safe condition is: the ring never wrapped at all
    (``offset <= max_size``).

    SUPERSEDED -- do not wire this into the post-generation path.
    ------------------------------------------------------------
    This helper has no production caller by design, and adding one would be a
    pessimization, not a fix. It predates the mid-prefill snapshot primitive,
    which solves the same problem strictly better:

      * The old plan was "if the ring wrapped, refuse to trim, persist the full
        end-of-asst state, and let the next request's rewind guard force a full
        re-prefill." Correct, but it gives up all prefix reuse.
      * The snapshot path instead captures rotating-layer state EXACTLY at the
        anchor offset *during* chunked prefill, so post-gen never has to trim a
        rotating layer at all -- it restores one. ``dispatch.py``'s asymmetric
        branch does that and trims only non-rotating layers; the symmetric
        branch does not trim.

    Wiring this in would reject the asymmetric anchoring path almost always: at
    the anchor offset the ring is typically already wrapped (Gemma 4's
    ``max_size`` is 1024, so any non-trivial conversation wraps well before the
    anchor), which is exactly the case the snapshot machinery was built to make
    reusable. The tests below are kept because they document the real SWA
    invariant -- the "repetition loops on turn 2" incident above -- not because
    the function is on a live path.
    """
    for c in entries:
        if hasattr(c, "caches"):
            if not _rotating_post_gen_trim_safe(c.caches, target_len):
                return False
            continue
        if isinstance(c, (list, tuple)):
            if not _rotating_post_gen_trim_safe(c, target_len):
                return False
            continue

        # Buffered/chunked rings evict by advancing start_position; a non-zero
        # watermark means the ring has already lost early tokens, which is the
        # same "not strictly safe" condition as a wrapped offset.
        start_position = getattr(c, "start_position", None)
        if start_position is not None and int(start_position) > 0:
            return False

        if _is_rotating_kv_layer(c):
            offset = int(c.offset.item() if hasattr(c.offset, "item") else c.offset)
            max_size = getattr(c, "max_size", None)
            if max_size is not None and offset > max_size:
                return False
    return True


def _has_non_trimmable(entries) -> bool:
    """Return True if any cache in ``entries`` (recursing into nested
    ``.caches`` and list/tuple containers) reports ``is_trimmable() ==
    False``.

    Hybrid models (Qwen 3.5/3.6 GatedDeltaNet via ``ArraysCache``,
    Mamba-style hybrids) advance recurrent state monotonically with no
    rewind primitive. ``_trim_cache`` only handles caches with an
    ``offset`` attribute, so non-trimmable layers would silently retain
    state past the rewind point. Callers use this signal to gate
    snapshot-restore-or-full-re-prefill.
    """
    for c in entries:
        if hasattr(c, "is_trimmable") and not c.is_trimmable():
            return True
        if hasattr(c, "caches"):
            if _has_non_trimmable(c.caches):
                return True
        elif isinstance(c, (list, tuple)):
            if _has_non_trimmable(c):
                return True
    return False


def _restore_deltanet_state(entries, snapshot_states) -> None:
    """Restore non-trimmable cache state from a snapshot list. The
    snapshot list aligns positionally with ``entries`` — None entries
    correspond to trimmable (KV) layers and are skipped here.
    """
    from mlx_vlm.models.cache import ArraysCache

    for c, s in zip(entries, snapshot_states):
        if s is not None and isinstance(c, ArraysCache):
            c.state = s


def __getattr__(name):
    """Backward-compat shim: ``generation_stream`` resolves lazily to the
    calling thread's stream rather than a frozen module-level singleton.
    """
    if name == "generation_stream":
        return _get_generation_stream()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
