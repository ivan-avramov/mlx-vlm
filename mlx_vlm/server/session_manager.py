"""Per-chat session cache + anonymous hash-chain routing.

This module owns the server's per-conversation ``PromptCacheState``
sessions: an LRU-capped ``OrderedDict`` keyed by chat_id, plus the
chained-sha256 turn-hash machinery that routes anonymous requests (no
explicit chat_id) to the session whose conversation prefix matches.

The state lives at module scope (not on ``runtime``) because the unit
tests treat these as importable module globals — they reset
``_session_caches`` / ``_session_cache_max`` / ``_chat_id_header``
between cases and read them directly. ``server/__init__.py`` re-exports
the public names so ``mlx_vlm.server.<name>`` resolves to them.

Phase 2 (``generation.ResponseGenerator._process_cached_request``)
routes chat_id'd requests through the daemon thread that owns the model,
applying prefix-cache reuse and DeltaNet snapshot rewind on hybrid
models.
"""

import hashlib
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from threading import Lock
from typing import List, Optional, Tuple

from ..generate import PromptCacheState
from ..snapshot import DEFAULT_RING_SIZE, DeltaNetSnapshotRing

logger = logging.getLogger(f"{os.environ.get('MLX_VLM_LOG_NAME', 'mlx_vlm')}.server")


# --- Per-chat-id PromptCacheState session manager ---
# Maps chat_id -> {state: PromptCacheState, last_used: float, turn_hashes}.
# OrderedDict gives O(1) LRU behavior via move_to_end / popitem(last=False).
_session_caches: "OrderedDict[str, dict]" = OrderedDict()
_session_caches_lock = Lock()

# --- Per-chat / DeltaNet config (set by cli.main() from argparse) ---
# argparse defaults are populated from env vars at parser-construction time
# (see cli.main()), so by the time these holders are written they already
# reflect the full precedence chain CLI arg > env var > module default.
# Module-level defaults below are what direct importers (tests) see if
# cli.main() doesn't run.
_deltanet_ring_size: int = DEFAULT_RING_SIZE
_deltanet_rewind_enabled: bool = True
# Default OFF at import time so direct importers (and the server test
# fixtures that don't run cli.main) see the upstream no-session-cache
# baseline. cli.main() publishes the user-facing default of 8 via
# configure(), and the session-cache unit tests set this per-case.
_session_cache_max: int = 0
_chat_id_header: str = "X-MLX-VLM-Chat-Id"
# Anonymous-session hash-chain matching: when a request arrives without
# any explicit chat_id, route it to whichever session shares the longest
# turn-hash prefix (>= _MIN_HASH_PREFIX_MATCH turns). Default on; flip
# off in multi-user deployments where two distinct conversations sharing
# a prefix would constitute a covert side-channel.
_cache_anon_sessions: bool = True
# Minimum number of leading turn-hashes that must match before we'll
# reuse an existing session for an anonymous request. 2 = "system + at
# least one user message" — a system-only match (length 1) saves only
# the few system tokens of prefill and is not worth the routing risk.
_MIN_HASH_PREFIX_MATCH: int = 2


def _env_int(name: str, default: int) -> int:
    """Read an env var as int with graceful fallback on missing/invalid.

    Used at argparse construction time to populate ``default=`` for int
    options that accept env-var fallback. Failing soft (rather than
    raising) keeps a typo'd env var from breaking server startup; the
    user gets the documented default and the typo is visible in --help.
    """
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_choice(name: str, default: str, choices: list) -> str:
    """Read an env var with choice validation. Falls back to default if
    the env value isn't one of the allowed choices (case-insensitive)."""
    val = os.environ.get(name, default).lower()
    return val if val in choices else default


def _resolve_chat_id(raw_request, parsed_request) -> Optional[str]:
    """Pull an explicit chat_id off the request (header / body / metadata).

    Order of precedence:
      1. HTTP header (configurable name; default X-MLX-VLM-Chat-Id).
      2. Top-level ``chat_id`` field on the request body.
      3. ``metadata.chat_id`` on the body (some clients including OpenWebUI
         send conversation metadata here).
      4. None — caller (``_resolve_session``) falls through to anonymous
         hash-prefix matching when enabled.
    """
    if raw_request is not None:
        header_val = raw_request.headers.get(_chat_id_header)
        if header_val:
            return str(header_val).strip()
    direct = getattr(parsed_request, "chat_id", None)
    if direct:
        return str(direct).strip()
    metadata = getattr(parsed_request, "metadata", None)
    if isinstance(metadata, dict):
        meta_chat_id = metadata.get("chat_id")
        if meta_chat_id:
            return str(meta_chat_id).strip()
    return None


# Fields whose presence affects per-turn cache identity. ``reasoning`` is
# excluded because clients (notably OpenWebUI) drop it when echoing
# assistant messages back; including it would deterministically miss on
# every multi-turn conversation. Everything else round-trips reliably.
_HASHABLE_FIELDS: Tuple[str, ...] = (
    "role",
    "content",
    "tool_calls",
    "tool_call_id",
    "name",
)


def _coerce_hashable(value):
    """Recursively coerce pydantic models within a value to plain dicts.

    Needed because ``json.dumps(..., default=str)`` would otherwise
    stringify nested pydantic objects via their repr, which embeds memory
    addresses and breaks hash stability across runs.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, list):
        return [_coerce_hashable(v) for v in value]
    if isinstance(value, dict):
        return {k: _coerce_hashable(v) for k, v in value.items()}
    return value


def _canonical_message(msg) -> str:
    """Render a message into a stable JSON string for hashing.

    Accepts pydantic ChatMessage models or plain dicts. Keeps only the
    ``_HASHABLE_FIELDS`` allow-list, drops None values, and sorts keys so
    the byte representation is deterministic regardless of input shape.
    """
    if isinstance(msg, dict):
        d = {k: msg[k] for k in _HASHABLE_FIELDS if msg.get(k) is not None}
    else:
        d = {}
        for k in _HASHABLE_FIELDS:
            v = getattr(msg, k, None)
            if v is None:
                continue
            d[k] = _coerce_hashable(v)
    d = _coerce_hashable(d)
    return json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)


def _chain_hash(prev_digest: bytes, canonical_str: str) -> bytes:
    h = hashlib.sha256()
    h.update(prev_digest)
    h.update(b"\x00")
    h.update(canonical_str.encode("utf-8"))
    return h.digest()


def _compute_turn_hashes(messages: list) -> List[str]:
    """Build a chained sha256 hash list — one entry per message.

    h[0] = sha256(canonical(messages[0]))
    h[k] = sha256(h[k-1] || 0x00 || canonical(messages[k]))

    Each h[k] uniquely identifies the conversation prefix through
    message k. Two requests share a hash prefix of length m iff their
    first m messages are pairwise byte-identical under canonicalization.
    """
    if not isinstance(messages, list):
        return []
    hashes: List[str] = []
    prev = b""
    for msg in messages:
        digest = _chain_hash(prev, _canonical_message(msg))
        hashes.append(digest.hex())
        prev = digest
    return hashes


def _hash_assistant_turn(
    prev_hash_hex: str,
    content: Optional[str],
    tool_calls: Optional[list],
) -> str:
    """Compute h[k] for the just-generated assistant message.

    Mirrors ``_canonical_message`` for an assistant turn so the resulting
    hash equals what we'd get if the same content+tool_calls dict were
    echoed back in a future request and re-hashed via the request path.
    """
    msg_dict: dict = {"role": "assistant"}
    if content is not None:
        msg_dict["content"] = content
    if tool_calls:
        msg_dict["tool_calls"] = _coerce_hashable(list(tool_calls))
    canon = json.dumps(msg_dict, sort_keys=True, ensure_ascii=False, default=str)
    prev = bytes.fromhex(prev_hash_hex) if prev_hash_hex else b""
    return _chain_hash(prev, canon).hex()


def get_or_create_prompt_cache_state(chat_id: str) -> PromptCacheState:
    """Look up or create a PromptCacheState for the given chat_id, applying
    LRU eviction at the configured session cap.

    New PromptCacheState instances are constructed with the snapshot ring
    size and rewind-enabled toggle resolved at CLI startup. snapshot.py
    itself doesn't read env vars; configuration flows down explicitly
    from this layer.
    """
    max_sessions = _session_cache_max
    with _session_caches_lock:
        if chat_id in _session_caches:
            entry = _session_caches[chat_id]
            entry["last_used"] = time.time()
            _session_caches.move_to_end(chat_id)
            return entry["state"]

        state = PromptCacheState(
            snapshot_ring=DeltaNetSnapshotRing(max_size=_deltanet_ring_size),
            rewind_enabled=_deltanet_rewind_enabled,
        )
        _session_caches[chat_id] = {
            "state": state,
            "last_used": time.time(),
            # Chained per-message sha256s. Populated by ``_sync_session_hashes``
            # on each request and ``_append_session_assistant_hash`` after each
            # generation. Used by ``_find_session_by_hash_prefix`` to route
            # anonymous requests to the right session.
            "turn_hashes": [],
        }
        while max_sessions > 0 and len(_session_caches) > max_sessions:
            evicted_id, _evicted = _session_caches.popitem(last=False)
            logger.debug(
                "Evicting per-chat PromptCacheState for chat_id=%s (LRU, max=%d)",
                evicted_id,
                max_sessions,
            )
        return state


def clear_session_caches() -> None:
    """Drop all per-chat caches. Called on model unload."""
    with _session_caches_lock:
        _session_caches.clear()


def _find_session_by_hash_prefix(
    request_hashes: List[str],
) -> Tuple[Optional[str], int]:
    """Pick the session whose stored turn_hashes shares the longest prefix
    with the request's hashes.

    Returns ``(session_id, match_len)``. ``session_id`` is None when no
    session matches at least ``_MIN_HASH_PREFIX_MATCH`` turns — system-only
    matches don't justify cache reuse and would create a side-channel
    where every fresh chat with the same system prompt latches onto a
    prior unrelated session.

    O(sessions x turns) under the LRU cap (~8 x ~50). Negligible.
    """
    if not request_hashes:
        return None, 0
    best_id: Optional[str] = None
    best_len = 0
    with _session_caches_lock:
        for sid, entry in _session_caches.items():
            stored = entry.get("turn_hashes") or []
            common = 0
            for a, b in zip(stored, request_hashes):
                if a != b:
                    break
                common += 1
            if common > best_len:
                best_len = common
                best_id = sid
    if best_len < _MIN_HASH_PREFIX_MATCH:
        return None, 0
    return best_id, best_len


def _sync_session_hashes(session_id: str, request_hashes: List[str]) -> None:
    """Replace the session's stored chain with the request's chain.

    Each request is the source of truth for the conversation's history
    up through its last message; previously-stored hashes for those
    positions may be stale (client edited a past turn, regenerated, etc.).
    Token-level divergence inside ``PromptCacheState.update`` independently
    handles the cache-trim half of the story; this helper only keeps the
    routing metadata in sync so the next request can find this session.
    """
    with _session_caches_lock:
        entry = _session_caches.get(session_id)
        if entry is not None:
            entry["turn_hashes"] = list(request_hashes)
            entry["last_used"] = time.time()


def _append_session_assistant_hash(session_id: str, assistant_hash: str) -> None:
    """Append the just-generated assistant turn's hash to the session.

    Called after generation completes so the next request — which echoes
    that assistant message back as part of the conversation history —
    can match through it via ``_find_session_by_hash_prefix``.
    """
    with _session_caches_lock:
        entry = _session_caches.get(session_id)
        if entry is not None:
            entry["turn_hashes"].append(assistant_hash)
            entry["last_used"] = time.time()


def _resolve_session(
    raw_request, parsed_request
) -> Tuple[Optional[str], Optional[PromptCacheState], List[str]]:
    """Pick the right per-chat session for this request.

    Returns ``(session_id, prompt_cache_state, request_turn_hashes)``.
    The first two are None when caching is disabled or no usable
    identifier can be derived.

    Resolution order:
      1. Explicit chat_id (header / body / metadata) — wins; treated as
         a stable user-supplied key. Hash chain is still synced onto the
         entry so anonymous requests can fall through to it later if the
         client drops the explicit id.
      2. Anonymous hash-prefix match against existing sessions, when
         ``_cache_anon_sessions`` is on. Routes to the session whose
         stored turn_hashes shares the longest prefix with this request,
         requiring at least ``_MIN_HASH_PREFIX_MATCH`` matching turns.
      3. New anonymous session keyed by an internal uuid, when anon
         caching is on and the request has at least one hashable
         message.
      4. None — caching disabled or no messages to hash; caller falls
         through to the non-cached path.
    """
    if _session_cache_max <= 0:
        return None, None, []

    messages = getattr(parsed_request, "messages", None)
    if messages is None:
        messages = getattr(parsed_request, "input", None)
    request_hashes = (
        _compute_turn_hashes(messages) if isinstance(messages, list) else []
    )

    explicit_id = _resolve_chat_id(raw_request, parsed_request)
    if explicit_id:
        state = get_or_create_prompt_cache_state(explicit_id)
        if request_hashes:
            _sync_session_hashes(explicit_id, request_hashes)
        return explicit_id, state, request_hashes

    if not _cache_anon_sessions or not request_hashes:
        return None, None, request_hashes

    matched_id, match_len = _find_session_by_hash_prefix(request_hashes)
    if matched_id is not None:
        state = get_or_create_prompt_cache_state(matched_id)
        _sync_session_hashes(matched_id, request_hashes)
        logger.debug(
            "Anonymous session hash-prefix match: id=%s prefix=%d/%d turns",
            matched_id,
            match_len,
            len(request_hashes),
        )
        return matched_id, state, request_hashes

    new_id = f"anon:{uuid.uuid4().hex[:16]}"
    state = get_or_create_prompt_cache_state(new_id)
    _sync_session_hashes(new_id, request_hashes)
    logger.debug(
        "Anonymous session created: id=%s turns=%d", new_id, len(request_hashes)
    )
    return new_id, state, request_hashes


def configure(
    *,
    deltanet_ring_size: Optional[int] = None,
    deltanet_rewind_enabled: Optional[bool] = None,
    session_cache_max: Optional[int] = None,
    chat_id_header: Optional[str] = None,
    cache_anon_sessions: Optional[bool] = None,
) -> None:
    """Publish CLI-resolved config to the module-level holders.

    Called from ``cli.main()`` after argparse has resolved the
    CLI-arg/env-var/default precedence chain. Only non-None values are
    applied so callers can update a subset.
    """
    global _deltanet_ring_size, _deltanet_rewind_enabled
    global _session_cache_max, _chat_id_header, _cache_anon_sessions
    if deltanet_ring_size is not None:
        _deltanet_ring_size = max(0, int(deltanet_ring_size))
    if deltanet_rewind_enabled is not None:
        _deltanet_rewind_enabled = bool(deltanet_rewind_enabled)
    if session_cache_max is not None:
        _session_cache_max = max(0, int(session_cache_max))
    if chat_id_header is not None:
        _chat_id_header = chat_id_header
    if cache_anon_sessions is not None:
        _cache_anon_sessions = bool(cache_anon_sessions)
