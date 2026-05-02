"""Tests for utils.sanitize_strict_json — strict-JSON pre-escape pipeline.

Covers memory.md #21: pre-escape LaTeX math blocks before json_repair sees
them, otherwise valid JSON escapes (\\f, \\b, \\n) destroy math content.

Two failure modes guarded here:
  1. Plain prose silently corrupted because the function thinks it's JSON.
  2. JSON-intended math like ``$\\frac{1}{2}$`` mangled because json_repair
     interprets ``\\f`` as a control char.

The contract: pass through untouched if not JSON-intended; otherwise
return JSON whose ``json.loads`` reproduces the math content verbatim.
"""

import json

import pytest

from mlx_vlm.utils import sanitize_strict_json


class TestPlainProsePassthrough:
    def test_prose_returned_unchanged(self):
        text = "The result is approximately 0.5 and that is final."
        assert sanitize_strict_json(text) == text

    def test_leading_whitespace_does_not_trigger_sanitize(self):
        # Stripping happens internally only for intent detection — the
        # original (unstripped) text must be returned for non-JSON.
        text = "  hello world  "
        assert sanitize_strict_json(text) == text

    def test_text_with_dollar_signs_but_not_json(self):
        # "$5" is a price, not a math block, and the text isn't JSON-shaped.
        text = "It costs $5 to make."
        assert sanitize_strict_json(text) == text

    def test_empty_string(self):
        assert sanitize_strict_json("") == ""


class TestRawJSONObjectIntent:
    def test_simple_object_passes_through(self):
        out = sanitize_strict_json('{"a": 1}')
        assert json.loads(out) == {"a": 1}

    def test_array_intent_recognized(self):
        out = sanitize_strict_json("[1, 2, 3]")
        assert json.loads(out) == [1, 2, 3]

    def test_leading_whitespace_before_brace_still_recognized(self):
        out = sanitize_strict_json('  \n  {"a": 1}')
        assert json.loads(out) == {"a": 1}


class TestMarkdownJSONFenceIntent:
    def test_fenced_json_block_unwrapped(self):
        text = '```json\n{"x": 1}\n```'
        out = sanitize_strict_json(text)
        assert json.loads(out) == {"x": 1}


class TestMathBlockEscaping:
    def test_inline_dollar_math_survives_roundtrip(self):
        # Without the pre-escape, json_repair turns \\f into form-feed
        # and \\frac{1}{2} loses its leading \\f. This is the exact
        # silent-corruption bug memory.md #21 calls out.
        original = r'{"eq": "$\frac{1}{2}$"}'
        out = sanitize_strict_json(original)
        parsed = json.loads(out)
        assert parsed["eq"] == r"$\frac{1}{2}$"

    def test_double_dollar_block_math_survives(self):
        original = r'{"display": "$$\sum_{i=1}^n x_i$$"}'
        out = sanitize_strict_json(original)
        parsed = json.loads(out)
        assert parsed["display"] == r"$$\sum_{i=1}^n x_i$$"

    def test_bracket_display_math_survives(self):
        original = r'{"latex": "\[\int_0^1 x dx\]"}'
        out = sanitize_strict_json(original)
        parsed = json.loads(out)
        assert parsed["latex"] == r"\[\int_0^1 x dx\]"

    def test_paren_inline_math_survives(self):
        original = r'{"latex": "\(x^2 + y^2\)"}'
        out = sanitize_strict_json(original)
        parsed = json.loads(out)
        assert parsed["latex"] == r"\(x^2 + y^2\)"

    def test_already_double_escaped_backslashes_preserved(self):
        # If the model already double-escaped, the negative lookbehind in
        # _escape_math_block must not over-escape into quadruple slashes.
        original = r'{"eq": "$\\alpha$"}'
        out = sanitize_strict_json(original)
        parsed = json.loads(out)
        # Either the original double-escape OR a normalized single-backslash
        # representation is acceptable, as long as the rendered string is
        # \\alpha or \alpha — NOT 4-slash garbage.
        assert parsed["eq"] in (r"$\\alpha$", r"$\alpha$")

    def test_quote_escapes_inside_math_not_doubled(self):
        # Backslash-quote inside a math span is a JSON-level escape and
        # must remain a single backslash. The lookbehind in _escape_math_block
        # specifically excludes \\" to prevent breaking string escapes.
        original = r'{"eq": "$\"x\"$"}'
        out = sanitize_strict_json(original)
        parsed = json.loads(out)
        assert parsed["eq"] == r'$"x"$'


class TestMixedContent:
    def test_object_without_math_unchanged_semantically(self):
        original = '{"name": "alice", "age": 30}'
        out = sanitize_strict_json(original)
        assert json.loads(out) == {"name": "alice", "age": 30}

    def test_object_with_text_and_math_field(self):
        original = (
            r'{"explanation": "the answer is", "formula": "$E = mc^2$"}'
        )
        out = sanitize_strict_json(original)
        parsed = json.loads(out)
        assert parsed["explanation"] == "the answer is"
        assert parsed["formula"] == "$E = mc^2$"


class TestMalformedInput:
    def test_repaired_json_still_valid(self):
        # json_repair rescues common LLM failure modes (trailing commas,
        # unclosed strings). The function should still return parseable JSON.
        original = '{"a": 1, "b": 2,}'  # trailing comma
        out = sanitize_strict_json(original)
        # If json_repair ate it, json.loads must succeed.
        json.loads(out)
