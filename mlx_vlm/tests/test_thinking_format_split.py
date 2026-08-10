"""Tests for the fork's registry-driven thinking splitter.

Fork-only, deliberately: `tests/test_responses_state.py` is byte-identical to upstream
(5 of 5 shared bodies) and restored upstream tests are kept that way so future merges
apply cleanly. `_split_thinking_by_format` and `_strip_thinking_quirks` are fork-only
functions inside a shared file, so their tests belong in a fork-only file — the same
pattern as `test_sanitize_strict_json.py` and `test_context_budget.py`.

**Why now: both had zero test references.** `dev/find_untested_fork_code.py` listed
them, and reading them for a contract turned up a real bug in the first one
(`_strip_thinking_quirks` mangled any gpt-oss reasoning whose first word merely began
with "thought"). That is the seventh direction working as intended: every gating audit
compares against `upstream/main`, so fork-only code is invisible to all of them.
"""

import pytest

from mlx_vlm.prompt_utils import THINKING_FORMATS
from mlx_vlm.server.responses_state import (
    _split_thinking_by_format,
    _strip_thinking_quirks,
)


def _fmt(name):
    for candidate in THINKING_FORMATS:
        if candidate.name == name:
            return candidate
    raise AssertionError(f"THINKING_FORMATS has no {name!r}")


@pytest.fixture(scope="module")
def gpt_oss():
    return _fmt("gpt-oss")


@pytest.fixture(scope="module")
def gemma():
    return _fmt("gemma")


class TestStripThinkingQuirks:
    """gpt-oss leaves a bare "thought" token after its opener, because the
    tokenization splits the channel name in `<|channel>thought`. Stripping it is
    correct; stripping a *prefix* of a longer word is not."""

    def test_the_leftover_channel_token_is_removed(self, gpt_oss):
        assert (
            _strip_thinking_quirks(gpt_oss, "thought the answer is 7")
            == "the answer is 7"
        )

    def test_a_bare_leftover_becomes_empty(self, gpt_oss):
        assert _strip_thinking_quirks(gpt_oss, "thought") == ""

    def test_a_newline_after_the_leftover_is_consumed(self, gpt_oss):
        assert _strip_thinking_quirks(gpt_oss, "thought\nnext line") == "next line"

    @pytest.mark.parametrize(
        "reasoning",
        [
            "thoughts about the problem",
            "thoughtful analysis here",
            "thoughtfully, the answer is 7",
            "thought_x = 1",
        ],
    )
    def test_a_word_merely_beginning_with_thought_is_left_alone(
        self, gpt_oss, reasoning
    ):
        """The regression. `startswith("thought")` had no word boundary, so
        "thoughts about the problem" was served as "s about the problem" and
        "thoughtful analysis" as "ful analysis" — corrupted reasoning text, visible to
        the user, with no test and no audit that could see it (fork-only code in a
        shared file).

        `thought_x` is included because `_` is a word character, so `\\b` must not
        match there either.
        """
        assert _strip_thinking_quirks(gpt_oss, reasoning) == reasoning

    def test_the_quirk_is_scoped_to_gpt_oss(self, gemma):
        """Other families are documented as clean. Applying the strip to all of them
        would corrupt any reasoning starting with the word "thought"."""
        assert (
            _strip_thinking_quirks(gemma, "thought the answer is 7")
            == "thought the answer is 7"
        )

    def test_surrounding_whitespace_is_stripped_for_every_format(self, gemma):
        assert _strip_thinking_quirks(gemma, "  padded  ") == "padded"

    def test_an_empty_reasoning_stays_empty(self, gpt_oss, gemma):
        assert _strip_thinking_quirks(gpt_oss, "") == ""
        assert _strip_thinking_quirks(gemma, "   ") == ""


class TestSplitThinkingByFormat:
    def test_a_complete_pair_splits(self, gemma):
        reasoning, content = _split_thinking_by_format(
            f"{gemma.openers[0]}weighing it{gemma.closers[0]}the answer", gemma
        )

        assert reasoning == "weighing it"
        assert content == "the answer"

    def test_a_prefilled_opener_is_handled(self, gemma):
        """A chat template can seed the model mid-thinking, so the closer arrives with
        no opener. Everything before it is reasoning."""
        reasoning, content = _split_thinking_by_format(
            f"already thinking{gemma.closers[0]}the answer", gemma
        )

        assert reasoning == "already thinking"
        assert content == "the answer"

    def test_no_closer_means_reasoning_still_in_progress(self, gemma):
        """Content must be empty rather than echoing the partial reasoning — the
        caller streams `content` to the user."""
        reasoning, content = _split_thinking_by_format(
            f"{gemma.openers[0]}still going", gemma
        )

        assert reasoning == "still going"
        assert content == ""

    def test_the_earliest_closer_wins(self, gemma):
        """Not the first listed. A format with several closers must cut at whichever
        appears first, or trailing content leaks into reasoning."""
        if len(gemma.closers) < 2:
            pytest.skip("gemma declares a single closer")
        text = f"r{gemma.closers[1]}mid{gemma.closers[0]}tail"
        reasoning, _content = _split_thinking_by_format(text, gemma)
        assert reasoning == "r"

    def test_openers_are_removed_from_the_reasoning_span(self, gemma):
        reasoning, _content = _split_thinking_by_format(
            f"{gemma.openers[0]}a{gemma.openers[0]}b{gemma.closers[0]}c", gemma
        )

        assert gemma.openers[0] not in reasoning
        assert reasoning == "ab"

    def test_empty_reasoning_returns_none_not_empty_string(self, gemma):
        """`None` is what the caller tests to decide whether to emit a reasoning
        block at all; `""` would emit an empty one."""
        reasoning, content = _split_thinking_by_format(
            f"{gemma.openers[0]}{gemma.closers[0]}just content", gemma
        )

        assert reasoning is None
        assert content == "just content"

    def test_gpt_oss_end_to_end_keeps_a_thought_prefixed_first_word(self, gpt_oss):
        """The bug as a user would hit it: the whole split, not just the helper."""
        text = f"{gpt_oss.openers[0]}thoughts on this{gpt_oss.closers[0]}42"

        reasoning, content = _split_thinking_by_format(text, gpt_oss)

        assert reasoning == "thoughts on this"
        assert content == "42"

    def test_gpt_oss_end_to_end_still_strips_the_real_leftover(self, gpt_oss):
        """And the fix must not disable the strip it exists for. The opener literal is
        removed, then a split-off bare "thought" token remains."""
        text = f"{gpt_oss.openers[0]}thought weighing it{gpt_oss.closers[0]}42"

        reasoning, content = _split_thinking_by_format(text, gpt_oss)

        assert reasoning == "weighing it"
        assert content == "42"
