"""Tests for the per-chat session cache + chat_id resolution in server.py.

Covers memory.md #11 (LRU eviction at session cap) and #25 (chat_id resolution
order, env-var helpers, get_or_create_prompt_cache_state, clear_session_caches).

The code under test is module-level state (_session_caches OrderedDict +
_session_cache_max + _chat_id_header), so each test resets it via
clear_session_caches() and the autouse fixture below.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mlx_vlm import server as srv
from mlx_vlm.generate import PromptCacheState


@pytest.fixture(autouse=True)
def _reset_session_state():
    """Reset module-globals before each test. Module imports under pytest
    share state across test files, so prior runs of server tests can leave
    `_session_caches` populated or change `_session_cache_max`.
    """
    srv.clear_session_caches()
    saved = {
        "max": srv._session_cache_max,
        "header": srv._chat_id_header,
        "ring": srv._deltanet_ring_size,
        "rewind": srv._deltanet_rewind_enabled,
        "anon_sessions": srv._cache_anon_sessions,
    }
    yield
    srv.clear_session_caches()
    srv._session_cache_max = saved["max"]
    srv._chat_id_header = saved["header"]
    srv._deltanet_ring_size = saved["ring"]
    srv._deltanet_rewind_enabled = saved["rewind"]
    srv._cache_anon_sessions = saved["anon_sessions"]


def _make_request(headers=None, chat_id=None, metadata=None):
    """Build a (raw_request, parsed_request) pair for _resolve_chat_id.

    raw_request mirrors FastAPI's Request — only `.headers.get(name)` is
    accessed. parsed_request is a Pydantic-like namespace exposing
    `chat_id` and `metadata` attributes.
    """
    raw = MagicMock()
    raw.headers.get = (headers or {}).get
    parsed = SimpleNamespace(chat_id=chat_id, metadata=metadata)
    return raw, parsed


class TestGetOrCreatePromptCacheState:
    def test_creates_new_state_on_miss(self):
        state = srv.get_or_create_prompt_cache_state("chat-A")
        assert isinstance(state, PromptCacheState)
        assert "chat-A" in srv._session_caches

    def test_returns_existing_state_on_hit(self):
        first = srv.get_or_create_prompt_cache_state("chat-A")
        second = srv.get_or_create_prompt_cache_state("chat-A")
        assert first is second

    def test_uses_configured_ring_size(self):
        srv._deltanet_ring_size = 7
        state = srv.get_or_create_prompt_cache_state("chat-A")
        assert state.snapshot_ring.max_size == 7

    def test_propagates_rewind_enabled_flag(self):
        srv._deltanet_rewind_enabled = False
        state = srv.get_or_create_prompt_cache_state("chat-A")
        assert state.rewind_enabled is False

    def test_distinct_chat_ids_get_distinct_states(self):
        a = srv.get_or_create_prompt_cache_state("chat-A")
        b = srv.get_or_create_prompt_cache_state("chat-B")
        assert a is not b
        assert a.snapshot_ring is not b.snapshot_ring


class TestLRUEviction:
    def test_evicts_least_recently_used_at_cap(self):
        srv._session_cache_max = 3
        a = srv.get_or_create_prompt_cache_state("A")
        b = srv.get_or_create_prompt_cache_state("B")
        c = srv.get_or_create_prompt_cache_state("C")
        # Insert D — A is the oldest and gets evicted.
        srv.get_or_create_prompt_cache_state("D")
        assert "A" not in srv._session_caches
        assert set(srv._session_caches.keys()) == {"B", "C", "D"}
        # Re-requesting A creates a fresh state, not the evicted one.
        new_a = srv.get_or_create_prompt_cache_state("A")
        assert new_a is not a

    def test_cache_hit_moves_to_most_recent(self):
        srv._session_cache_max = 3
        srv.get_or_create_prompt_cache_state("A")
        srv.get_or_create_prompt_cache_state("B")
        srv.get_or_create_prompt_cache_state("C")
        # Touch A — now B becomes the LRU.
        srv.get_or_create_prompt_cache_state("A")
        srv.get_or_create_prompt_cache_state("D")
        assert "B" not in srv._session_caches
        assert set(srv._session_caches.keys()) == {"A", "C", "D"}

    def test_zero_or_negative_cap_disables_eviction(self):
        # max_sessions <= 0 means "no cap" (the while-loop guards on
        # max_sessions > 0). Useful for CI / single-user setups that
        # want unlimited sessions.
        srv._session_cache_max = 0
        for i in range(20):
            srv.get_or_create_prompt_cache_state(f"chat-{i}")
        assert len(srv._session_caches) == 20

    def test_last_used_timestamp_refreshed_on_hit(self):
        srv._session_cache_max = 5
        srv.get_or_create_prompt_cache_state("A")
        first_ts = srv._session_caches["A"]["last_used"]
        # Hit should update timestamp (move_to_end + last_used assignment).
        # Real wall-clock may not advance enough; assert the field is
        # set to a non-decreasing value rather than strictly greater.
        srv.get_or_create_prompt_cache_state("A")
        second_ts = srv._session_caches["A"]["last_used"]
        assert second_ts >= first_ts


class TestClearSessionCaches:
    def test_empties_the_dict(self):
        srv.get_or_create_prompt_cache_state("A")
        srv.get_or_create_prompt_cache_state("B")
        assert len(srv._session_caches) == 2
        srv.clear_session_caches()
        assert len(srv._session_caches) == 0

    def test_safe_when_already_empty(self):
        # Called from unload_model_sync even when nothing was cached.
        srv.clear_session_caches()
        srv.clear_session_caches()  # no-op


class TestResolveChatId:
    def test_header_takes_precedence(self):
        srv._chat_id_header = "X-Chat"
        raw, parsed = _make_request(
            headers={"X-Chat": "header-id"},
            chat_id="body-id",
            metadata={"chat_id": "meta-id"},
        )
        assert srv._resolve_chat_id(raw, parsed) == "header-id"

    def test_falls_back_to_body_chat_id(self):
        srv._chat_id_header = "X-Chat"
        raw, parsed = _make_request(headers={}, chat_id="body-id")
        assert srv._resolve_chat_id(raw, parsed) == "body-id"

    def test_falls_back_to_metadata_chat_id(self):
        srv._chat_id_header = "X-Chat"
        raw, parsed = _make_request(headers={}, metadata={"chat_id": "meta-id"})
        assert srv._resolve_chat_id(raw, parsed) == "meta-id"

    def test_returns_none_when_nothing_configured(self):
        # _resolve_chat_id only checks header / body / metadata. The
        # anon-hash-prefix matching is the responsibility of the
        # surrounding `_resolve_session`, not this helper.
        raw, parsed = _make_request(headers={})
        assert srv._resolve_chat_id(raw, parsed) is None

    def test_strips_whitespace(self):
        srv._chat_id_header = "X-Chat"
        raw, parsed = _make_request(headers={"X-Chat": "  spaced  "})
        assert srv._resolve_chat_id(raw, parsed) == "spaced"

    def test_uses_configured_header_name(self):
        srv._chat_id_header = "X-Custom-Id"
        raw, parsed = _make_request(headers={"X-Custom-Id": "via-custom"})
        assert srv._resolve_chat_id(raw, parsed) == "via-custom"

    def test_empty_header_value_falls_through_to_body(self):
        srv._chat_id_header = "X-Chat"
        raw, parsed = _make_request(headers={"X-Chat": ""}, chat_id="body-id")
        assert srv._resolve_chat_id(raw, parsed) == "body-id"

    def test_non_dict_metadata_is_ignored(self):
        srv._chat_id_header = "X-Chat"
        # Some clients send metadata as a string or list; must not crash.
        raw, parsed = _make_request(headers={}, metadata="not-a-dict")
        assert srv._resolve_chat_id(raw, parsed) is None

    def test_handles_missing_raw_request(self):
        # Some endpoints don't pass the FastAPI Request through — body-only.
        parsed = SimpleNamespace(chat_id="body-only", metadata=None)
        assert srv._resolve_chat_id(None, parsed) == "body-only"


class TestEnvHelpers:
    def test_env_int_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("TEST_INT_X", raising=False)
        assert srv._env_int("TEST_INT_X", 42) == 42

    def test_env_int_parses_valid_value(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_X", "99")
        assert srv._env_int("TEST_INT_X", 42) == 99

    def test_env_int_falls_back_on_invalid_value(self, monkeypatch):
        # Failing soft is intentional: a typo'd env var must not crash
        # server startup. The default is documented in --help.
        monkeypatch.setenv("TEST_INT_X", "not-a-number")
        assert srv._env_int("TEST_INT_X", 42) == 42

    def test_env_int_handles_negative_and_zero(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_X", "-5")
        assert srv._env_int("TEST_INT_X", 42) == -5
        monkeypatch.setenv("TEST_INT_X", "0")
        assert srv._env_int("TEST_INT_X", 42) == 0

    def test_env_choice_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("TEST_CHOICE", raising=False)
        assert (
            srv._env_choice("TEST_CHOICE", "auto", ["auto", "on", "off"]) == "auto"
        )

    def test_env_choice_accepts_valid_choice(self, monkeypatch):
        monkeypatch.setenv("TEST_CHOICE", "off")
        assert srv._env_choice("TEST_CHOICE", "auto", ["auto", "on", "off"]) == "off"

    def test_env_choice_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("TEST_CHOICE", "ON")
        assert srv._env_choice("TEST_CHOICE", "auto", ["auto", "on", "off"]) == "on"

    def test_env_choice_falls_back_on_unknown(self, monkeypatch):
        monkeypatch.setenv("TEST_CHOICE", "garbage")
        assert (
            srv._env_choice("TEST_CHOICE", "auto", ["auto", "on", "off"]) == "auto"
        )
