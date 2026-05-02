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


def _make_msg(role, content, **extras):
    """Build a pydantic-like ChatMessage stand-in.

    `tool_calls`, `tool_call_id`, `name`, and `reasoning` are passed
    through as attributes when supplied so canonical_message can pick
    them up via getattr just like it would on a real ChatMessage.
    Default-None for the unset ones so the allow-list filter behaves.
    """
    fields = dict(
        role=role,
        content=content,
        tool_calls=None,
        tool_call_id=None,
        name=None,
        reasoning=None,
    )
    fields.update(extras)
    return SimpleNamespace(**fields)


class TestCanonicalMessage:
    def test_dict_and_pydantic_produce_same_output(self):
        as_obj = _make_msg("user", "hello")
        as_dict = {"role": "user", "content": "hello"}
        assert srv._canonical_message(as_obj) == srv._canonical_message(as_dict)

    def test_drops_none_fields(self):
        # Reasoning is excluded entirely; tool_calls=None must not appear.
        msg = _make_msg("assistant", "ok")
        canon = srv._canonical_message(msg)
        assert "tool_calls" not in canon
        assert "reasoning" not in canon

    def test_excludes_reasoning_even_when_present(self):
        # Reasoning is omitted from `_HASHABLE_FIELDS` because clients
        # (e.g. OWUI) drop it when echoing the assistant message back.
        with_reasoning = _make_msg("assistant", "ok", reasoning="thinking...")
        without = _make_msg("assistant", "ok")
        assert srv._canonical_message(with_reasoning) == srv._canonical_message(
            without
        )

    def test_includes_tool_calls(self):
        a = _make_msg(
            "assistant", None, tool_calls=[{"id": "1", "function": {"name": "f"}}]
        )
        b = _make_msg(
            "assistant", None, tool_calls=[{"id": "2", "function": {"name": "f"}}]
        )
        assert srv._canonical_message(a) != srv._canonical_message(b)

    def test_keys_sorted_so_output_stable(self):
        # Build the same logical message in two orders. JSON output must match.
        a = {"role": "user", "content": "hi", "name": "alice"}
        b = {"name": "alice", "content": "hi", "role": "user"}
        assert srv._canonical_message(a) == srv._canonical_message(b)


class TestComputeTurnHashes:
    def test_empty_messages_returns_empty(self):
        assert srv._compute_turn_hashes([]) == []

    def test_non_list_returns_empty(self):
        assert srv._compute_turn_hashes(None) == []
        assert srv._compute_turn_hashes("not-a-list") == []

    def test_one_hash_per_message(self):
        msgs = [_make_msg("system", "s"), _make_msg("user", "u")]
        hashes = srv._compute_turn_hashes(msgs)
        assert len(hashes) == 2
        assert all(isinstance(h, str) and len(h) == 64 for h in hashes)

    def test_chain_property_h1_depends_on_h0(self):
        # If we change message[0], h[1] must also change (chain).
        a = srv._compute_turn_hashes(
            [_make_msg("system", "A"), _make_msg("user", "u")]
        )
        b = srv._compute_turn_hashes(
            [_make_msg("system", "B"), _make_msg("user", "u")]
        )
        assert a[0] != b[0]
        assert a[1] != b[1]

    def test_role_change_diverges_hash(self):
        a = srv._compute_turn_hashes([_make_msg("user", "x")])
        b = srv._compute_turn_hashes([_make_msg("assistant", "x")])
        assert a != b

    def test_dict_messages_match_pydantic(self):
        as_objs = [_make_msg("system", "s"), _make_msg("user", "u")]
        as_dicts = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ]
        assert srv._compute_turn_hashes(as_objs) == srv._compute_turn_hashes(
            as_dicts
        )

    def test_extends_growing_conversation(self):
        # Turn 2's chain extends turn 1 — first hash matches.
        turn1 = srv._compute_turn_hashes([_make_msg("user", "hi")])
        turn2 = srv._compute_turn_hashes(
            [_make_msg("user", "hi"), _make_msg("assistant", "hello")]
        )
        assert turn2[0] == turn1[0]
        assert len(turn2) == 2

    def test_is_deterministic_across_runs(self):
        msgs = [_make_msg("system", "s"), _make_msg("user", "u")]
        h1 = srv._compute_turn_hashes(msgs)
        h2 = srv._compute_turn_hashes(msgs)
        assert h1 == h2

    def test_hash_includes_content_separator_safety(self):
        # Edge case: "AB" + "" must not equal "A" + "B" via concatenation.
        # Chain hash uses 0x00 separator + canonical JSON which embeds
        # role boundaries, so this is a sanity check that the expected
        # property holds.
        a = srv._compute_turn_hashes(
            [_make_msg("system", "AB"), _make_msg("user", "")]
        )
        b = srv._compute_turn_hashes(
            [_make_msg("system", "A"), _make_msg("user", "B")]
        )
        assert a != b


class TestHashAssistantTurn:
    def test_extends_chain(self):
        prior = srv._compute_turn_hashes([_make_msg("user", "hi")])
        asst_hash = srv._hash_assistant_turn(prior[-1], "hello", None)
        assert asst_hash != prior[-1]
        assert len(asst_hash) == 64

    def test_matches_request_recomputed_chain(self):
        # Server pre-computes h[k] for an assistant message it just
        # generated; if the next request echoes that assistant verbatim,
        # the request-side _compute_turn_hashes must reproduce h[k]
        # identically. This is the "no extra prefill on multi-turn"
        # invariant that makes the whole design pay off.
        msgs = [_make_msg("user", "hi")]
        prior = srv._compute_turn_hashes(msgs)
        asst_hash = srv._hash_assistant_turn(prior[-1], "hello", None)

        echoed = msgs + [_make_msg("assistant", "hello")]
        recomputed = srv._compute_turn_hashes(echoed)
        assert recomputed[0] == prior[0]
        assert recomputed[1] == asst_hash

    def test_tool_calls_included(self):
        prior = srv._compute_turn_hashes([_make_msg("user", "do x")])
        a = srv._hash_assistant_turn(prior[-1], None, [{"id": "1", "f": "x"}])
        b = srv._hash_assistant_turn(prior[-1], None, [{"id": "2", "f": "x"}])
        assert a != b

    def test_empty_prev_hash_handled(self):
        # If somehow called with no prior chain (defensive), shouldn't crash.
        h = srv._hash_assistant_turn("", "hi", None)
        assert len(h) == 64


class TestFindSessionByHashPrefix:
    def test_empty_pool_returns_none(self):
        assert srv._find_session_by_hash_prefix(["a", "b"]) == (None, 0)

    def test_empty_request_hashes_returns_none(self):
        srv.get_or_create_prompt_cache_state("S1")
        srv._sync_session_hashes("S1", ["a", "b"])
        assert srv._find_session_by_hash_prefix([]) == (None, 0)

    def test_single_full_match(self):
        srv.get_or_create_prompt_cache_state("S1")
        srv._sync_session_hashes("S1", ["a", "b", "c"])
        sid, n = srv._find_session_by_hash_prefix(["a", "b", "c", "d"])
        assert sid == "S1"
        assert n == 3

    def test_below_min_match_threshold_returns_none(self):
        # MIN=2: a single-turn match (system only) must NOT be returned.
        srv.get_or_create_prompt_cache_state("S1")
        srv._sync_session_hashes("S1", ["a", "x"])
        sid, n = srv._find_session_by_hash_prefix(["a", "y"])
        assert sid is None
        assert n == 0

    def test_longest_prefix_wins(self):
        srv.get_or_create_prompt_cache_state("S1")
        srv._sync_session_hashes("S1", ["a", "b"])
        srv.get_or_create_prompt_cache_state("S2")
        srv._sync_session_hashes("S2", ["a", "b", "c"])
        sid, n = srv._find_session_by_hash_prefix(["a", "b", "c", "d"])
        assert sid == "S2"
        assert n == 3

    def test_no_match_when_first_hash_differs(self):
        srv.get_or_create_prompt_cache_state("S1")
        srv._sync_session_hashes("S1", ["a", "b", "c"])
        sid, n = srv._find_session_by_hash_prefix(["x", "b", "c"])
        assert sid is None
        assert n == 0

    def test_session_with_no_hashes_skipped(self):
        srv.get_or_create_prompt_cache_state("S1")
        # Don't sync any hashes — turn_hashes stays []
        sid, n = srv._find_session_by_hash_prefix(["a", "b"])
        assert sid is None and n == 0


class TestSyncAndAppendHashHelpers:
    def test_sync_replaces_chain(self):
        srv.get_or_create_prompt_cache_state("S1")
        srv._sync_session_hashes("S1", ["a", "b"])
        srv._sync_session_hashes("S1", ["x", "y", "z"])
        assert srv._session_caches["S1"]["turn_hashes"] == ["x", "y", "z"]

    def test_sync_no_op_for_unknown_session(self):
        # Should not crash if session was evicted between resolve and sync.
        srv._sync_session_hashes("never-existed", ["a", "b"])

    def test_append_extends_chain(self):
        srv.get_or_create_prompt_cache_state("S1")
        srv._sync_session_hashes("S1", ["a", "b"])
        srv._append_session_assistant_hash("S1", "c")
        assert srv._session_caches["S1"]["turn_hashes"] == ["a", "b", "c"]

    def test_append_no_op_for_unknown_session(self):
        srv._append_session_assistant_hash("never-existed", "c")

    def test_sync_updates_last_used(self):
        srv.get_or_create_prompt_cache_state("S1")
        before = srv._session_caches["S1"]["last_used"]
        srv._sync_session_hashes("S1", ["a"])
        after = srv._session_caches["S1"]["last_used"]
        assert after >= before


class TestResolveSession:
    """End-to-end tests for the unified resolver. Covers the four
    branches: explicit chat_id, anon hash-prefix match, anon new
    session, anon disabled."""

    def _req(self, messages, headers=None, chat_id=None, metadata=None):
        raw, parsed = _make_request(
            headers=headers, chat_id=chat_id, metadata=metadata
        )
        parsed.messages = messages
        return raw, parsed

    def test_explicit_chat_id_wins_over_hash_match(self):
        srv._cache_anon_sessions = True
        # Pre-populate an existing anon session that *would* match.
        msgs = [_make_msg("system", "s"), _make_msg("user", "u")]
        srv.get_or_create_prompt_cache_state("anon:existing")
        srv._sync_session_hashes(
            "anon:existing", srv._compute_turn_hashes(msgs)
        )
        # Request with explicit header — should use that, not hash-match.
        raw, parsed = self._req(msgs, headers={srv._chat_id_header: "explicit"})
        sid, state, hashes = srv._resolve_session(raw, parsed)
        assert sid == "explicit"
        assert state is not None
        assert hashes == srv._compute_turn_hashes(msgs)

    def test_anon_match_reuses_existing_session(self):
        srv._cache_anon_sessions = True
        msgs = [_make_msg("system", "s"), _make_msg("user", "u")]
        # Seed an existing session with the same prefix.
        srv.get_or_create_prompt_cache_state("anon:original")
        srv._sync_session_hashes(
            "anon:original", srv._compute_turn_hashes(msgs)
        )
        raw, parsed = self._req(msgs)
        sid, state, _hashes = srv._resolve_session(raw, parsed)
        assert sid == "anon:original"
        assert state is srv._session_caches["anon:original"]["state"]

    def test_anon_no_match_creates_new_session(self):
        srv._cache_anon_sessions = True
        msgs = [_make_msg("system", "s"), _make_msg("user", "u")]
        raw, parsed = self._req(msgs)
        sid, state, _hashes = srv._resolve_session(raw, parsed)
        assert sid is not None
        assert sid.startswith("anon:")
        assert sid in srv._session_caches

    def test_anon_disabled_returns_none(self):
        srv._cache_anon_sessions = False
        msgs = [_make_msg("system", "s"), _make_msg("user", "u")]
        raw, parsed = self._req(msgs)
        sid, state, _hashes = srv._resolve_session(raw, parsed)
        assert sid is None
        assert state is None

    def test_caching_disabled_globally(self):
        srv._session_cache_max = 0
        msgs = [_make_msg("user", "hi")]
        raw, parsed = self._req(msgs)
        sid, state, hashes = srv._resolve_session(raw, parsed)
        assert sid is None
        assert state is None
        assert hashes == []

    def test_no_messages_anonymous_falls_through(self):
        srv._cache_anon_sessions = True
        raw, parsed = _make_request(headers={})
        # No messages attribute at all.
        sid, state, hashes = srv._resolve_session(raw, parsed)
        assert sid is None
        assert state is None
        assert hashes == []

    def test_explicit_id_seeds_chain_on_first_request(self):
        # Even with explicit id, chain is recorded so future anon
        # requests can match it.
        msgs = [_make_msg("system", "s"), _make_msg("user", "u")]
        raw, parsed = self._req(msgs, headers={srv._chat_id_header: "abc"})
        sid, _state, hashes = srv._resolve_session(raw, parsed)
        assert sid == "abc"
        assert srv._session_caches["abc"]["turn_hashes"] == hashes


class TestSameOpeningDifferentConversations:
    """Two chat sessions whose first user message is identical but whose
    assistant responses diverge.

    The chain extends only on inbound requests — the server never
    pre-computes an assistant hash. So convo A's *first* turn stores
    chain length 2 ([sys, user_1]); B's first turn collides on that
    prefix and reattaches to the same anon session (correctly — both
    requests are byte-identical at that point). Forking happens on
    each conversation's *second* request, when the request carries
    the divergent assistant message back as part of the echoed history.
    """

    def test_after_full_assistant_divergence_resolver_picks_correct_branch(self):
        """Pre-existing two sessions whose chains differ at the assistant
        message — resolver routes each next-turn request to its own
        branch, no cross-contamination. Setup mimics what would be in
        the pool after each conversation has had its second request
        sync'd in (chain length 3, ending at the divergent asst)."""
        srv._cache_anon_sessions = True
        msgs = [_make_msg("system", "s"), _make_msg("user", "joke?")]
        # Build each session's stored chain by computing the hash of
        # the request that *would have arrived* on its second turn — i.e.
        # the messages list with the assistant message echoed back.
        msgs_a = msgs + [_make_msg("assistant", "knock knock")]
        msgs_b = msgs + [_make_msg("assistant", "why did the chicken")]
        chain_a = srv._compute_turn_hashes(msgs_a)
        chain_b = srv._compute_turn_hashes(msgs_b)
        srv.get_or_create_prompt_cache_state("anon:A")
        srv._session_caches["anon:A"]["turn_hashes"] = list(chain_a)
        srv.get_or_create_prompt_cache_state("anon:B")
        srv._session_caches["anon:B"]["turn_hashes"] = list(chain_b)

        # A's next request: includes "knock knock" assistant + new user.
        msgs_a_next = msgs + [
            _make_msg("assistant", "knock knock"),
            _make_msg("user", "another"),
        ]
        raw_a, parsed_a = _make_request(headers={})
        parsed_a.messages = msgs_a_next
        sid_a, _state, _hashes = srv._resolve_session(raw_a, parsed_a)
        assert sid_a == "anon:A"

        # B's next request: includes its own assistant + new user.
        msgs_b_next = msgs + [
            _make_msg("assistant", "why did the chicken"),
            _make_msg("user", "another"),
        ]
        raw_b, parsed_b = _make_request(headers={})
        parsed_b.messages = msgs_b_next
        sid_b, _state, _hashes = srv._resolve_session(raw_b, parsed_b)
        assert sid_b == "anon:B"
        assert sid_a != sid_b

    def test_first_turn_collision_then_third_turn_fork(self):
        """End-to-end: two convos that share only the opening collide
        on turn 1, fork on turn 2 (when the assistant message arrives
        as part of the echoed history)."""
        srv._cache_anon_sessions = True
        msgs1 = [_make_msg("system", "be helpful"), _make_msg("user", "joke?")]

        # ---- Convo A, turn 1 ----
        raw, parsed = _make_request(headers={})
        parsed.messages = msgs1
        sid_a1, _s, _h = srv._resolve_session(raw, parsed)
        # Server generates "knock knock" — but the chain is NOT pre-extended.
        assert len(srv._session_caches[sid_a1]["turn_hashes"]) == 2

        # ---- Convo B, turn 1 (same opening, will get a different assistant) ----
        raw, parsed = _make_request(headers={})
        parsed.messages = msgs1
        sid_b1, _s, _h = srv._resolve_session(raw, parsed)
        # Routes to the same session — both first turns are byte-identical.
        assert sid_b1 == sid_a1

        # ---- Convo A, turn 2 ----
        msgs_a2 = msgs1 + [
            _make_msg("assistant", "knock knock"),
            _make_msg("user", "another"),
        ]
        raw, parsed = _make_request(headers={})
        parsed.messages = msgs_a2
        sid_a2, _s, _h = srv._resolve_session(raw, parsed)
        # Chain after sync now includes A's assistant.
        assert srv._session_caches[sid_a2]["turn_hashes"] == (
            srv._compute_turn_hashes(msgs_a2)
        )

        # ---- Convo B, turn 2 (different assistant in echoed history) ----
        msgs_b2 = msgs1 + [
            _make_msg("assistant", "why did the chicken"),
            _make_msg("user", "another"),
        ]
        raw, parsed = _make_request(headers={})
        parsed.messages = msgs_b2
        sid_b2, _s, _h = srv._resolve_session(raw, parsed)
        # B's chain diverges from the stored A-chain at index 2 (asst);
        # match length is 2, still meets MIN_MATCH=2, so the resolver
        # routes B back to the same anon session. The chain is then
        # rewritten to B's. Documented limitation: when MIN_MATCH=2,
        # convos sharing an opening serialize onto the same session
        # and the daemon's PromptCacheState.update() handles the
        # token-level cache trim. Multi-user deployments should set
        # `--cache-anon-sessions=off` to avoid this entirely.
        assert srv._session_caches[sid_b2]["turn_hashes"] == (
            srv._compute_turn_hashes(msgs_b2)
        )


class TestRewindAndEdit:
    """When the user rewinds (regenerates last assistant) or edits a
    past turn, the new request's hash chain shares a prefix with the
    stored chain but diverges at the edit point. The resolver must
    pick that session (longest-prefix wins) and re-sync the chain so
    the daemon's token-level trim handles the cache."""

    def test_regenerate_last_assistant_keeps_session(self):
        # Original turn-2 request brought chain length 3 [sys, user1,
        # asst1]; that's what's stored after the second resolve call
        # in the lifetime of the session. User regenerates → next
        # request drops the assistant and is just [sys, user1]. Match
        # length = 2; routes to the same session.
        srv._cache_anon_sessions = True
        prev_request = [
            _make_msg("system", "s"),
            _make_msg("user", "hi"),
            _make_msg("assistant", "first response"),
        ]
        srv.get_or_create_prompt_cache_state("anon:orig")
        srv._session_caches["anon:orig"]["turn_hashes"] = (
            srv._compute_turn_hashes(prev_request)
        )

        # Regenerate request — same messages, no assistant.
        regen = [_make_msg("system", "s"), _make_msg("user", "hi")]
        raw, parsed = _make_request(headers={})
        parsed.messages = regen
        sid, _state, hashes = srv._resolve_session(raw, parsed)
        assert sid == "anon:orig"
        # Chain re-synced to the request shape (no asst entry).
        assert srv._session_caches["anon:orig"]["turn_hashes"] == hashes
        assert len(srv._session_caches["anon:orig"]["turn_hashes"]) == 2

    def test_edit_mid_turn_routes_to_session_and_truncates_chain(self):
        # 4-turn conversation. User edits turn 3's user message.
        # Request still shares turns 0 and 1 (sys, user1, asst1) but
        # turn 3 differs.
        srv._cache_anon_sessions = True
        sys_m = _make_msg("system", "s")
        u1 = _make_msg("user", "first")
        a1 = _make_msg("assistant", "alpha")
        u2_orig = _make_msg("user", "second-original")
        u2_edit = _make_msg("user", "second-edited")

        original = [sys_m, u1, a1, u2_orig]
        edited = [sys_m, u1, a1, u2_edit]

        # Seed session with the original chain.
        original_chain = srv._compute_turn_hashes(original)
        srv.get_or_create_prompt_cache_state("anon:edit")
        srv._session_caches["anon:edit"]["turn_hashes"] = list(original_chain)

        # Edited request comes in.
        raw, parsed = _make_request(headers={})
        parsed.messages = edited
        sid, _state, hashes = srv._resolve_session(raw, parsed)
        # Same session (matched 3/4 turns).
        assert sid == "anon:edit"
        # Stored chain now matches the edited request, not the original.
        assert srv._session_caches["anon:edit"]["turn_hashes"] == hashes
        # Sanity: original turns 0..2 hash equally; turn 3 differs.
        edited_chain = srv._compute_turn_hashes(edited)
        assert edited_chain[:3] == original_chain[:3]
        assert edited_chain[3] != original_chain[3]

    def test_regenerate_does_not_match_below_min_threshold(self):
        # If the only common prefix is the system message (length 1),
        # MIN_MATCH=2 prevents reuse and a new session is created.
        srv._cache_anon_sessions = True
        srv.get_or_create_prompt_cache_state("anon:other")
        srv._session_caches["anon:other"]["turn_hashes"] = (
            srv._compute_turn_hashes(
                [_make_msg("system", "s"), _make_msg("user", "totally-different")]
            )
        )
        # New request shares system, but user message is different.
        raw, parsed = _make_request(headers={})
        parsed.messages = [
            _make_msg("system", "s"),
            _make_msg("user", "fresh-conversation"),
        ]
        sid, _state, _hashes = srv._resolve_session(raw, parsed)
        assert sid != "anon:other"
        # New anon session created.
        assert sid is not None and sid.startswith("anon:")

