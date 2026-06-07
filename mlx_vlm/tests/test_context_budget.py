"""Tests for soft-max context budget resolution (clamp-with-floor)."""

import pytest

from mlx_vlm.server.generation import (
    GenerationArguments,
    PromptTooLongError,
    _apply_generation_budget,
    _resolve_generation_budget,
)


def test_resolve_returns_requested_when_it_fits(monkeypatch):
    monkeypatch.setenv("MAX_KV_SIZE", "1000")
    assert _resolve_generation_budget(100, 200) == 200


def test_resolve_clamps_to_remaining_context(monkeypatch):
    monkeypatch.setenv("MAX_KV_SIZE", "1000")
    monkeypatch.setenv("MIN_OUTPUT_TOKENS", "100")
    assert _resolve_generation_budget(300, 1000) == 700


def test_resolve_rejects_below_floor(monkeypatch):
    monkeypatch.setenv("MAX_KV_SIZE", "1000")
    monkeypatch.setenv("MIN_OUTPUT_TOKENS", "100")
    with pytest.raises(PromptTooLongError):
        _resolve_generation_budget(950, 1000)


def test_resolve_no_configured_limit_passes_through(monkeypatch):
    monkeypatch.delenv("MAX_KV_SIZE", raising=False)
    assert _resolve_generation_budget(10_000_000, 500) == 500


def test_resolve_request_that_fits_is_never_floor_checked(monkeypatch):
    # An explicit small max_tokens that fits must be honored even when the
    # remaining budget is below the floor (old behavior accepted it too).
    monkeypatch.setenv("MAX_KV_SIZE", "1000")
    monkeypatch.setenv("MIN_OUTPUT_TOKENS", "500")
    assert _resolve_generation_budget(700, 200) == 200


def test_apply_clamps_max_tokens_and_scales_thinking_budget(monkeypatch):
    monkeypatch.setenv("MAX_KV_SIZE", "1000")
    monkeypatch.setenv("MIN_OUTPUT_TOKENS", "100")
    args = GenerationArguments(max_tokens=1000, thinking_budget=900)
    _apply_generation_budget(args, 500)
    assert args.max_tokens == 500
    assert args.thinking_budget == 400  # 80% of the clamped budget


def test_apply_without_clamp_leaves_args_untouched(monkeypatch):
    monkeypatch.setenv("MAX_KV_SIZE", "1000")
    args = GenerationArguments(max_tokens=200, thinking_budget=150)
    _apply_generation_budget(args, 100)
    assert args.max_tokens == 200
    assert args.thinking_budget == 150


def test_apply_clamp_without_thinking_budget(monkeypatch):
    monkeypatch.setenv("MAX_KV_SIZE", "1000")
    monkeypatch.setenv("MIN_OUTPUT_TOKENS", "100")
    args = GenerationArguments(max_tokens=1000)
    _apply_generation_budget(args, 500)
    assert args.max_tokens == 500
    assert args.thinking_budget is None


def test_resolve_rejects_when_prompt_exceeds_limit(monkeypatch):
    monkeypatch.setenv("MAX_KV_SIZE", "500")
    with pytest.raises(PromptTooLongError, match="context window is only 500"):
        _resolve_generation_budget(600, 200)  # remaining = -100


def test_apply_clamp_keeps_smaller_thinking_budget(monkeypatch):
    monkeypatch.setenv("MAX_KV_SIZE", "1000")
    monkeypatch.setenv("MIN_OUTPUT_TOKENS", "100")
    args = GenerationArguments(max_tokens=1000, thinking_budget=300)
    _apply_generation_budget(args, 500)
    assert args.max_tokens == 500
    assert args.thinking_budget == 300  # already below 80% of 500, untouched
