import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import mlx_vlm.server as server


@pytest.fixture
def client():
    with TestClient(server.app) as test_client:
        yield test_client


@pytest.mark.parametrize("value", [224, "22", [1.0], [1.5], [True], [1, 2, 3]])
def test_chat_completions_endpoint_rejects_invalid_resize_shape(client, value):
    response = client.post(
        "/chat/completions",
        json={
            "model": "demo",
            "messages": [{"role": "user", "content": "Hello"}],
            "resize_shape": value,
        },
    )

    assert response.status_code == 422


def test_chat_request_schema_allows_one_or_two_resize_shape_values():
    resize_shape = server.ChatRequest.model_json_schema()["properties"]["resize_shape"]
    lengths = {
        (item["minItems"], item["maxItems"])
        for item in resize_shape["anyOf"]
        if item.get("type") == "array"
    }

    assert lengths == {(1, 1), (2, 2)}


def test_responses_endpoint_forwards_new_sampling_args(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = SimpleNamespace(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", return_value=result) as mock_generate,
    ):
        response = client.post(
            "/responses",
            json={
                "model": "demo",
                "input": "Hello",
                "max_output_tokens": 12,
                "top_k": 40,
                "min_p": 0.08,
                "repetition_penalty": 1.15,
                "logit_bias": {"12": -1.5},
                "enable_thinking": False,
                "thinking_budget": 24,
                "thinking_start_token": "<think>",
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.kwargs["enable_thinking"] is False
    assert mock_template.call_args.kwargs["thinking_budget"] == 24
    assert mock_template.call_args.kwargs["thinking_start_token"] == "<think>"
    assert mock_generate.call_args.kwargs["max_tokens"] == 12
    assert mock_generate.call_args.kwargs["top_k"] == 40
    assert mock_generate.call_args.kwargs["min_p"] == 0.08
    assert mock_generate.call_args.kwargs["repetition_penalty"] == 1.15
    assert mock_generate.call_args.kwargs["logit_bias"] == {12: -1.5}
    assert mock_generate.call_args.kwargs["enable_thinking"] is False
    assert mock_generate.call_args.kwargs["thinking_budget"] == 24
    assert mock_generate.call_args.kwargs["thinking_start_token"] == "<think>"


def test_chat_completions_endpoint_forwards_explicit_sampling_args(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = SimpleNamespace(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
        prompt_tps=10.0,
        generation_tps=5.0,
        peak_memory=0.1,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result) as mock_generate,
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 12,
                "top_k": 40,
                "min_p": 0.08,
                "repetition_penalty": 1.15,
                "logit_bias": {"12": -1.5},
                "resize_shape": [512],
            },
        )

    assert response.status_code == 200
    assert mock_generate.call_args.kwargs["max_tokens"] == 12
    assert mock_generate.call_args.kwargs["top_k"] == 40
    assert mock_generate.call_args.kwargs["min_p"] == 0.08
    assert mock_generate.call_args.kwargs["repetition_penalty"] == 1.15
    assert mock_generate.call_args.kwargs["logit_bias"] == {12: -1.5}
    assert mock_generate.call_args.kwargs["resize_shape"] == (512, 512)


# ── Continuous batching / ResponseGenerator tests ─────────────────────


class TestResponseGenerator:
    """Tests for the ResponseGenerator continuous batching engine."""

    def test_generate_arguments_defaults(self):
        args = server.GenerationArguments()
        assert args.max_tokens == server.DEFAULT_MAX_TOKENS
        assert args.temperature == server.DEFAULT_TEMPERATURE
        assert args.enable_thinking is False
        assert args.logit_bias is None

    def test_generate_arguments_to_generate_kwargs(self):
        processor = lambda tokens, logits: logits
        args = server.GenerationArguments(
            max_tokens=50,
            temperature=0.7,
            top_k=40,
            min_p=0.05,
            repetition_penalty=1.15,
            logit_bias={3: -0.5},
            enable_thinking=False,
            thinking_budget=100,
            logits_processors=[processor],
        )
        kw = args.to_generate_kwargs()
        assert kw["max_tokens"] == 50
        assert kw["top_k"] == 40
        assert kw["min_p"] == 0.05
        assert kw["repetition_penalty"] == 1.15
        assert kw["logit_bias"] == {3: -0.5}
        assert kw["enable_thinking"] is False
        assert kw["thinking_budget"] == 100
        assert kw["logits_processors"] == [processor]

    def test_generate_arguments_to_template_kwargs(self):
        args = server.GenerationArguments(enable_thinking=False, thinking_budget=50)
        kw = args.to_template_kwargs()
        assert kw["enable_thinking"] is False
        assert kw["thinking_budget"] == 50

    def test_generate_arguments_omits_none_optionals(self):
        args = server.GenerationArguments()
        kw = args.to_generate_kwargs()
        assert "repetition_penalty" not in kw
        assert "logit_bias" not in kw
        assert "thinking_budget" not in kw

    def test_build_gen_args_from_openai_request(self):
        req = SimpleNamespace(
            max_output_tokens=128,
            temperature=0.5,
            top_p=0.9,
            top_k=32,
            min_p=0.1,
            repetition_penalty=1.2,
            logit_bias={"5": -1.0},
            enable_thinking=False,
            thinking_budget=None,
            thinking_start_token=None,
        )
        args = server._build_gen_args(req)
        assert args.max_tokens == 128
        assert args.top_k == 32
        assert args.logit_bias == {5: -1.0}  # string keys converted to int

    def test_build_gen_args_from_chat_request(self):
        req = SimpleNamespace(
            max_tokens=256,
            max_output_tokens=None,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            repetition_penalty=None,
            logit_bias=None,
            enable_thinking=True,
            thinking_budget=None,
            thinking_start_token=None,
        )
        args = server._build_gen_args(req)
        assert args.max_tokens == 256
        assert args.enable_thinking is True  # explicitly passed True

    def test_extract_chat_response_format_json_schema(self):
        req = SimpleNamespace(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "animal",
                    "schema": {
                        "type": "object",
                        "properties": {"animal": {"type": "string"}},
                        "required": ["animal"],
                    },
                },
            },
            text=None,
        )

        schema = server._extract_response_format_schema(req)

        assert schema["properties"]["animal"]["type"] == "string"

    def test_extract_responses_text_format_json_schema(self):
        req = SimpleNamespace(
            response_format=None,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "animal",
                    "schema": {
                        "type": "object",
                        "properties": {"animal": {"type": "string"}},
                        "required": ["animal"],
                    },
                }
            },
        )

        schema = server._extract_response_format_schema(req)

        assert schema["required"] == ["animal"]

    def test_build_structured_logits_processors_uses_tokenizer(self):
        req = SimpleNamespace(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "animal",
                    "schema": {"type": "object"},
                },
            },
            text=None,
        )
        proc = SimpleNamespace(tokenizer=object())

        with patch.object(
            server, "build_json_schema_logits_processor", return_value="processor"
        ) as mock_build:
            processors = server._build_structured_logits_processors(req, proc)

        assert processors == ["processor"]
        assert mock_build.call_args.args[1] == {"type": "object"}


class TestSplitThinking:
    """Tests for thinking tag parsing."""

    def test_channel_tags(self):
        text = "<|channel>thought\nReasoning here.<channel|>The answer."
        reasoning, content = server._split_thinking(text)
        assert reasoning == "Reasoning here."
        assert content == "The answer."

    def test_think_tags(self):
        text = "<think>Thinking.</think>Answer."
        reasoning, content = server._split_thinking(text)
        assert reasoning == "Thinking."
        assert content == "Answer."

    def test_partial_close_tag_only(self):
        text = "Thinking text\n</think>\nAnswer."
        reasoning, content = server._split_thinking(text)
        assert reasoning == "Thinking text"
        assert content == "Answer."

    def test_no_thinking(self):
        text = "Just plain text."
        reasoning, content = server._split_thinking(text)
        assert reasoning is None
        assert content == "Just plain text."

    def test_empty_content_after_thinking(self):
        text = "<|channel>thought\nOnly thinking.<channel|>"
        reasoning, content = server._split_thinking(text)
        assert reasoning == "Only thinking."
        assert content == ""


class TestChatMessageSchema:
    """Tests for ChatMessage accepting tool-calling roles and fields."""

    def test_accepts_tool_role(self):
        msg = server.ChatMessage(role="tool", content="result", tool_call_id="tc_1")
        assert msg.role == "tool"
        assert msg.tool_call_id == "tc_1"

    def test_accepts_assistant_with_tool_calls(self):
        msg = server.ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[{"id": "tc_1", "function": {"name": "f", "arguments": "{}"}}],
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_reasoning_field(self):
        msg = server.ChatMessage(
            role="assistant", content="answer", reasoning="thought"
        )
        assert msg.reasoning == "thought"


class TestSuppressToolCallContent:
    """Tests for tool-call markup suppression in streaming."""

    def test_no_tool_module(self):
        in_tc, content = server.suppress_tool_call_content(
            "Hello world", False, None, "world"
        )
        assert in_tc is False
        assert content == "world"

    def test_normal_text_before_tool_call(self):
        in_tc, content = server.suppress_tool_call_content(
            "I will call", False, "<tool_call>", "call"
        )
        assert in_tc is False
        assert content == "call"

    def test_suppresses_on_start_marker(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<tool_call>", False, "<tool_call>", ">"
        )
        assert in_tc is True
        assert content is None

    def test_suppresses_partial_marker(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<tool", False, "<tool_call>", "<tool"
        )
        assert in_tc is False
        assert content is None

    def test_stays_suppressed_after_entering(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<tool_call>get_weather", True, "<tool_call>", "weather"
        )
        assert in_tc is True
        assert content is None

    def test_pipe_delimited_marker(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<|tool_call>call:get_weather", False, "<|tool_call>", "weather"
        )
        assert in_tc is True
        assert content is None

    def test_pipe_delimited_partial_marker(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<|tool", False, "<|tool_call>", "<|tool"
        )
        assert in_tc is False
        assert content is None


class TestProcessToolCalls:
    """Tests for tool call parsing from model output (memory.md #4).

    The wrapper layers the OpenAI-spec adapters (id, type=function,
    json.dumps(arguments)) on top of the model's native tool-call parser.
    Regressions here surface as Pydantic validation errors in OpenWebUI
    when it sees a tool_calls list missing the required fields.
    """

    @staticmethod
    def _module(parse_returns, tool_call_start="<tc>", tool_call_end="</tc>"):
        """Build a fake tool_module whose parse_tool_call returns canned values.

        ``parse_returns`` is either a single dict (return-as-is for every call)
        or a list of (dict|Exception) — replayed in order so multi-call
        scenarios can mix valid + invalid parses.
        """
        if not isinstance(parse_returns, list):
            parse_returns = [parse_returns]
        cursor = {"i": 0}

        def parse_tool_call(text, tools):
            i = cursor["i"]
            cursor["i"] += 1
            value = parse_returns[i] if i < len(parse_returns) else parse_returns[-1]
            if isinstance(value, Exception):
                raise value
            return value

        return SimpleNamespace(
            tool_call_start=tool_call_start,
            tool_call_end=tool_call_end,
            parse_tool_call=parse_tool_call,
        )

    def test_no_tool_calls(self):
        module = self._module({})
        result = server.process_tool_calls("Just text.", module, [])
        assert result["calls"] == []
        assert result["remaining_text"] == "Just text."

    def test_single_call_emits_openai_shape(self):
        module = self._module({"name": "get_weather", "arguments": {"city": "SF"}})
        out = server.process_tool_calls(
            'Reply: <tc>{"name": "get_weather", "arguments": {"city": "SF"}}</tc>',
            module,
            [],
        )

        assert len(out["calls"]) == 1
        call = out["calls"][0]
        # OpenAI-mandated fields — OpenWebUI's Pydantic validation
        # requires these exact keys.
        assert call["type"] == "function"
        assert call["index"] == 0
        assert "id" in call and isinstance(call["id"], str) and call["id"]
        assert call["function"]["name"] == "get_weather"
        # Arguments must be a JSON STRING (not dict) per OpenAI spec.
        assert call["function"]["arguments"] == '{"city": "SF"}'

    def test_dict_arguments_serialized_to_json_string(self):
        # Even nested dicts and unicode must round-trip cleanly.
        module = self._module({"name": "fn", "arguments": {"q": "héllo", "n": 3}})
        out = server.process_tool_calls(
            "<tc>...</tc>", module, []
        )
        args = out["calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        # ensure_ascii=False keeps non-ASCII verbatim — OpenWebUI displays
        # this in a chat bubble; escaped sequences would look wrong to
        # the user.
        parsed = json.loads(args)
        assert parsed == {"q": "héllo", "n": 3}

    def test_string_arguments_passed_through_unchanged(self):
        # Some parsers return arguments already-JSON-stringified — the
        # wrapper must NOT re-encode (which would double-escape quotes).
        module = self._module({"name": "fn", "arguments": '{"x": 1}'})
        out = server.process_tool_calls("<tc>...</tc>", module, [])
        assert out["calls"][0]["function"]["arguments"] == '{"x": 1}'

    def test_multiple_calls_get_distinct_indices(self):
        module = self._module(
            [
                {"name": "first", "arguments": {}},
                {"name": "second", "arguments": {}},
                {"name": "third", "arguments": {}},
            ]
        )
        out = server.process_tool_calls(
            "<tc>a</tc> mid <tc>b</tc> mid <tc>c</tc>",
            module,
            [],
        )
        assert [c["function"]["name"] for c in out["calls"]] == [
            "first",
            "second",
            "third",
        ]
        assert [c["index"] for c in out["calls"]] == [0, 1, 2]
        # Each call gets a distinct id (UUID).
        ids = {c["id"] for c in out["calls"]}
        assert len(ids) == 3

    def test_invalid_call_is_skipped_valid_calls_survive(self):
        # Rescue path: a malformed call must not poison neighboring
        # valid calls. memory.md #15 — the streaming pipeline expects
        # at most 'valid_count' entries back.
        module = self._module(
            [
                {"name": "ok", "arguments": {}},
                ValueError("malformed"),
                {"name": "also_ok", "arguments": {}},
            ]
        )
        out = server.process_tool_calls(
            "<tc>good</tc> <tc>bad</tc> <tc>good2</tc>",
            module,
            [],
        )
        names = [c["function"]["name"] for c in out["calls"]]
        assert "ok" in names
        assert "also_ok" in names
        assert len(out["calls"]) == 2

    def test_tool_call_markup_stripped_from_remaining_text(self):
        module = self._module({"name": "fn", "arguments": {}})
        out = server.process_tool_calls(
            "Before <tc>{...}</tc> after.", module, []
        )
        # The matched markup is replaced with a single space and
        # surrounding whitespace stripped.
        assert "<tc>" not in out["remaining_text"]
        assert "</tc>" not in out["remaining_text"]
        assert "Before" in out["remaining_text"]
        assert "after" in out["remaining_text"]

    def test_empty_tool_call_end_uses_newline_terminator(self):
        # Some parsers (Ollama-flavored) signal a tool call by start
        # marker only, terminating at newline. The pattern must switch
        # to newline-anchored when tool_call_end == "".
        module = self._module(
            {"name": "fn", "arguments": {}},
            tool_call_start="<<call>>",
            tool_call_end="",
        )
        out = server.process_tool_calls(
            "lead <<call>>get_weather city=SF\nMore text.",
            module,
            [],
        )
        assert len(out["calls"]) == 1
        assert "<<call>>" not in out["remaining_text"]
        # Trailing prose preserved on the next line.
        assert "More text" in out["remaining_text"]

    def test_name_whitespace_stripped(self):
        # Defensive: parsers occasionally leak whitespace into the name
        # field. The wrapper strips it — OpenAI clients reject names
        # with leading/trailing whitespace.
        module = self._module({"name": "  spaced_fn  ", "arguments": {}})
        out = server.process_tool_calls("<tc>x</tc>", module, [])
        assert out["calls"][0]["function"]["name"] == "spaced_fn"


class TestCountThinkingTagTokens:
    """Tests for thinking tag token counting."""

    def test_channel_tags(self):
        # `<|channel>thought ... <channel|>` content resolves to
        # gemma (its openers tuple includes `<|channel>thought`),
        # whose tag_token_count is 2. A pure-gpt-oss-style output
        # would historically count as 4; the merge into gemma costs
        # ~2 tokens of accounting precision on completion_tokens —
        # cosmetic only.
        assert (
            server._count_thinking_tag_tokens("<|channel>thought\ntext<channel|>answer")
            == 2
        )

    def test_think_tags(self):
        assert server._count_thinking_tag_tokens("<think>text</think>answer") == 2

    def test_no_tags(self):
        assert server._count_thinking_tag_tokens("plain text") == 0


class _FakeTokenizer:
    """Minimal tokenizer stub whose apply_chat_template emulates the
    rendering behavior of a real chat template for prefill-opener detection.
    """

    def __init__(self, suffix_with_gen: str, suffix_no_gen: str = ""):
        self._suffix_with_gen = suffix_with_gen
        self._suffix_no_gen = suffix_no_gen

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, **kwargs
    ):
        body = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        return body + (
            self._suffix_with_gen if add_generation_prompt else self._suffix_no_gen
        )


class _FakeProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class TestHasPrefilledOpener:
    """Tests for _has_prefilled_opener detection across template families."""

    def setup_method(self):
        server._PREFILL_FLAG_CACHE.clear()

    def test_unsloth_qwen_thinking_on_is_prefilled(self):
        # unsloth Qwen 3.6 with enable_thinking=True ends with <think>\n
        proc = _FakeProcessor(
            _FakeTokenizer(
                suffix_with_gen="<|im_start|>assistant\n<think>\n",
                suffix_no_gen="",
            )
        )
        assert server._has_prefilled_opener(proc, {"enable_thinking": True}) is True

    def test_unsloth_qwen_thinking_off_is_not_prefilled(self):
        # enable_thinking=False renders the empty pair; suffix ends with </think>
        proc = _FakeProcessor(
            _FakeTokenizer(
                suffix_with_gen="<|im_start|>assistant\n<think>\n\n</think>\n\n",
                suffix_no_gen="",
            )
        )
        assert server._has_prefilled_opener(proc, {"enable_thinking": False}) is False

    def test_canonical_qwen_thinking_on_is_not_prefilled(self):
        # canonical Qwen 3 leaves the assistant header bare; model emits both tags
        proc = _FakeProcessor(
            _FakeTokenizer(suffix_with_gen="<|im_start|>assistant\n", suffix_no_gen="")
        )
        assert server._has_prefilled_opener(proc, {"enable_thinking": True}) is False

    def test_gemma_native_opener_is_prefilled(self):
        # If a future Gemma-style template prefilled <|channel>thought
        proc = _FakeProcessor(
            _FakeTokenizer(
                suffix_with_gen="<start_of_turn>model\n<|channel>thought",
                suffix_no_gen="",
            )
        )
        assert server._has_prefilled_opener(proc, {"enable_thinking": True}) is True

    def test_template_render_failure_returns_false(self):
        class _BrokenTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                raise RuntimeError("template error")

        proc = _FakeProcessor(_BrokenTokenizer())
        assert server._has_prefilled_opener(proc, {}) is False

    def test_caches_result_on_repeat_calls(self):
        calls = {"count": 0}

        class _CountingTokenizer(_FakeTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                calls["count"] += 1
                return super().apply_chat_template(*args, **kwargs)

        proc = _FakeProcessor(
            _CountingTokenizer(suffix_with_gen="<|im_start|>assistant\n<think>\n")
        )
        kwargs = {"enable_thinking": True}

        assert server._has_prefilled_opener(proc, kwargs) is True
        first_call_count = calls["count"]
        # Second call with identical kwargs hits cache (no new renders)
        assert server._has_prefilled_opener(proc, kwargs) is True
        assert calls["count"] == first_call_count

    def test_distinct_kwargs_get_distinct_cache_entries(self):
        proc = _FakeProcessor(
            _FakeTokenizer(suffix_with_gen="<|im_start|>assistant\n<think>\n")
        )
        # Both should be True for this stub, but they exercise the cache key
        assert server._has_prefilled_opener(proc, {"enable_thinking": True}) is True
        assert server._has_prefilled_opener(proc, {"enable_thinking": False}) is True
        # Two distinct keys cached
        assert len(server._PREFILL_FLAG_CACHE) >= 2

    def test_unhashable_kwargs_skip_cache_but_still_compute(self):
        proc = _FakeProcessor(
            _FakeTokenizer(suffix_with_gen="<|im_start|>assistant\n<think>\n")
        )
        # dict value is unhashable; helper should fall through gracefully
        result = server._has_prefilled_opener(proc, {"tools": [{"name": "x"}]})
        assert result is True
        # No cache entry written for unhashable keys
        for k in server._PREFILL_FLAG_CACHE:
            assert "tools" not in dict(k[1])


class TestIsPromptInsideThinking:
    """Regression for the Gemma 4 leak where the streaming state machine
    failed to seed in_thinking=True because `_has_prefilled_opener`
    only checked the prompt tail, missing Gemma's global `<|think|>`
    marker at the system block (memory.md #29).

    `_is_prompt_inside_thinking` does a structural scan of the whole
    rendered prompt and returns True iff there's an opener with no
    closer following it — handling both tail-prefilled (Qwen 3.6
    unsloth) and globally-opened (Gemma 4) cases.
    """

    def test_gemma_global_think_marker_at_start(self):
        # The exact pattern in user logs: <|think|> at the top of the
        # system block, no closer anywhere in the prompt → model is
        # in-thinking from gen-start.
        prompt = (
            "<|think|>\n"
            "system content\n"
            "<|tool>declaration:foo<tool|>\n"
            "<turn|>\n"
            "<|turn>user\nhi<turn|>\n"
            "<|turn>model\n"
        )
        assert server._is_prompt_inside_thinking(prompt) is True

    def test_qwen_tail_prefilled_opener(self):
        # The unsloth Qwen 3.6 case the original `_has_prefilled_opener`
        # was designed for. Same-direction signal here.
        prompt = (
            "<|im_start|>user\nhi<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )
        assert server._is_prompt_inside_thinking(prompt) is True

    def test_closed_thinking_block_returns_false(self):
        # Gemma 4 with enable_thinking=False renders an empty block:
        # `<|channel>thought\n<channel|>`. Opener is followed by closer
        # → not in thinking.
        prompt = (
            "<|turn>user\nhi<turn|>\n<|turn>model\n"
            "<|channel>thought\n<channel|>"
        )
        assert server._is_prompt_inside_thinking(prompt) is False

    def test_no_thinking_format_in_prompt(self):
        prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        assert server._is_prompt_inside_thinking(prompt) is False

    def test_opener_then_closer_then_opener_again(self):
        # Pathological case: a previously-closed thinking block, then a
        # fresh opener at the tail. Latest opener has no following
        # closer → in thinking.
        prompt = (
            "earlier <|channel>thought\nfoo<channel|> done"
            "\n<|turn>model\n<|think|>"
        )
        assert server._is_prompt_inside_thinking(prompt) is True


class TestPartialTagStartPos:
    """Tests for the ends-with-prefix detector that replaces the buggy
    `p in accumulated` substring match. The substring check could only
    fire after `accumulated` had grown to the partial's full length;
    by then the tag's leading bytes had already streamed through as
    `delta.content` piecewise, leaking literal `<|c`/`<|ch`/`<|chan`
    fragments. The ends-with-prefix check fires from the very first
    matching byte.
    """

    def test_returns_none_when_no_partial_at_end(self):
        partials = ("<|channel", "<|think")
        assert server._partial_tag_start_pos("plain text", partials) is None

    def test_matches_single_char_prefix(self):
        # The exact failure mode: a single `<` arrives as one token.
        # The substring check would not fire (`"<|channel"` is 9 chars,
        # accumulated is 1). Ends-with-prefix sees `<` matches the
        # 1-char prefix of every `<…` partial.
        partials = ("<|channel", "<|think")
        assert server._partial_tag_start_pos("hello <", partials) == 6

    def test_matches_growing_prefix_across_calls(self):
        # Drive the accumulated buffer character by character. Every
        # state along the way should still be detected as partial.
        partials = ("<|channel", "<|think")
        for accum in ("<", "<|", "<|c", "<|ch", "<|cha", "<|chan", "<|chann", "<|channe", "<|channel"):
            pos = server._partial_tag_start_pos(accum, partials)
            assert pos == 0, f"failed at accum={accum!r} got pos={pos}"

    def test_returns_earliest_match_when_multiple_overlapping(self):
        # Keep the leftmost partial-start position when two literals
        # have overlapping prefixes (`</think` and `<channel` both
        # contain `<`).
        partials = ("</think", "<channel")
        assert server._partial_tag_start_pos("text <", partials) == 5

    def test_no_match_for_string_in_middle_of_accumulated(self):
        # A complete tag in the middle of accumulated isn't a partial
        # at the end. Caller's tag-find logic handles complete tags
        # via the find()-based branch; partial detection is only for
        # the trailing region.
        partials = ("<|channel",)
        assert server._partial_tag_start_pos("<|channel> stuff", partials) is None

    def test_empty_accumulated(self):
        partials = ("<|channel",)
        assert server._partial_tag_start_pos("", partials) is None

    def test_empty_partials_tuple(self):
        # No format → no partials to track → no match.
        assert server._partial_tag_start_pos("anything", ()) is None


class TestStepThinkingState:
    """Pure-function tests for the streaming-state-machine helper that
    replaced the inline branch chain in chat_completions_endpoint.

    Pins the three failure modes from production logs:
      1. Token-spanning tags eating content (`<channel|>2` dropping
         the leading `2 + ` of "2 + 2 = 4").
      2. Tag-prefix bytes leaking as content because the partial check
         used substring instead of ends-with-prefix.
      3. Multiple state transitions in a single token (closer +
         visible + opener + reasoning all fused) silently dropping the
         second transition.
    """

    @pytest.fixture
    def gemma_fmt(self):
        from mlx_vlm.prompt_utils import THINKING_FORMATS

        return next(f for f in THINKING_FORMATS if f.name == "gemma")

    def _drive(self, tokens, fmt, in_thinking_start=False):
        """Run a sequence of tokens through the helper, accumulating
        the emitted reasoning + content streams. Returns
        ``(end_in_thinking, end_accumulated, full_reasoning, full_content)``.
        """
        in_thinking = in_thinking_start
        accumulated = ""
        reasoning_parts = []
        content_parts = []
        for t in tokens:
            in_thinking, accumulated, dr, dc = server._step_thinking_state(
                t, in_thinking, accumulated, fmt
            )
            if dr is not None:
                reasoning_parts.append(dr)
            if dc is not None:
                content_parts.append(dc)
        return (
            in_thinking,
            accumulated,
            "".join(reasoning_parts),
            "".join(content_parts),
        )

    # --- Headline failure modes from the production logs -----------------

    def test_token_spanning_closer_emits_pre_and_post(self, gemma_fmt):
        # Gemma 4 production bug: "2 + 2 = 4" rendered as "2 = 4"
        # because the closer + leading visible char came in one token.
        # Pre-fix: branch 2 fired, transitioned, dropped the entire
        # token. Post-fix: split at closer, emit "thinking content"
        # as reasoning and "2" as content for the same iteration.
        in_thinking, accumulated, dr, dc = server._step_thinking_state(
            "thinking content<channel|>2",
            True,
            "",
            gemma_fmt,
        )
        assert in_thinking is False
        assert accumulated == ""
        assert dr == "thinking content"
        assert dc == "2"

    def test_partial_opener_buffered_then_completed(self, gemma_fmt):
        # Gemma 4 production bug: `<|channel>thought` literal showed
        # up in delta.content. Cause was the substring partial check
        # — `<|channel` (9 chars) couldn't match accumulated until
        # accumulated had 9+ chars, so individual tag-prefix tokens
        # streamed straight to delta.content.
        # Post-fix: ends-with-prefix matches from the very first byte.
        in_thinking, accumulated, reasoning, content = self._drive(
            ["hello ", "<", "|", "channel>thought", "\nreasoning", "<channel|>", "visible"],
            gemma_fmt,
            in_thinking_start=False,
        )
        assert in_thinking is False
        assert accumulated == ""
        assert content == "hello visible"
        assert reasoning == "\nreasoning"
        # Crucial invariant: no fragment of the opener literal leaked
        # into content.
        assert "<|" not in content
        assert "channel" not in content

    def test_multi_transition_token(self, gemma_fmt):
        # closer + visible + opener + reasoning all fused into one
        # token. Pre-fix: only the first transition fired; the second
        # and third were dropped. Post-fix: helper loops over all
        # transitions in the accumulated buffer.
        in_thinking, accumulated, dr, dc = server._step_thinking_state(
            "r1<channel|>between<|channel>thoughtr2",
            True,
            "",
            gemma_fmt,
        )
        assert in_thinking is True
        assert accumulated == ""
        assert dr == "r1" + "r2"
        assert dc == "between"

    # --- Other invariants ------------------------------------------------

    def test_no_format_passthrough(self):
        # Non-thinking model: no format detected. Token text streams
        # through as content unchanged; in_thinking stays False;
        # accumulated unchanged.
        in_thinking, accumulated, dr, dc = server._step_thinking_state(
            "plain text", False, "", None
        )
        assert in_thinking is False
        assert accumulated == ""
        assert dr is None
        assert dc == "plain text"

    def test_seeded_in_thinking_routes_first_tokens_to_reasoning(self, gemma_fmt):
        # Gemma 4 + enable_thinking=True: streaming starts with
        # in_thinking=True (seeded by `_is_prompt_inside_thinking`).
        # First tokens are reasoning until a closer arrives.
        in_thinking, accumulated, reasoning, content = self._drive(
            ["thinking ", "content ", "more"],
            gemma_fmt,
            in_thinking_start=True,
        )
        assert in_thinking is True
        # No closer arrived → still buffering nothing, all emitted as
        # reasoning streamed.
        assert reasoning == "thinking content more"
        assert content == ""

    def test_round_trip_multiple_thinking_blocks(self, gemma_fmt):
        # Real Gemma 4 pattern: thinks, closes, emits content, opens
        # again, thinks again, closes, emits content. State machine
        # must transition cleanly through every boundary.
        in_thinking, accumulated, reasoning, content = self._drive(
            [
                "thinking-1",
                "<channel|>",
                "visible-1 ",
                "<|channel>thought\n",
                "thinking-2",
                "<channel|>",
                "visible-2",
            ],
            gemma_fmt,
            in_thinking_start=True,
        )
        assert in_thinking is False
        assert accumulated == ""
        assert reasoning == "thinking-1" + "\nthinking-2"
        assert content == "visible-1 visible-2"

    def test_partial_at_end_is_carried_forward(self, gemma_fmt):
        # A partial tag at the end of one token's accumulated should
        # be preserved verbatim across the call boundary so the next
        # token can complete (or invalidate) it.
        in_thinking, accumulated, dr, dc = server._step_thinking_state(
            "before <", False, "", gemma_fmt
        )
        assert in_thinking is False
        assert accumulated == "<"  # buffered
        assert dr is None
        assert dc == "before "

    def test_partial_buffer_invalidated_by_non_tag_continuation(self, gemma_fmt):
        # Partial `<` followed by a non-matching char like `a` should
        # be released back into content (it wasn't a tag after all).
        # The helper handles this by re-checking on each call: at next
        # call, accumulated="<a"; no opener matches; no partial at end
        # (`<a` doesn't end with a prefix of any partial); flush all
        # as content.
        # First token: buffer the `<`.
        s = server._step_thinking_state("<", False, "", gemma_fmt)
        assert s == (False, "<", None, None)
        # Second token: completion that's NOT a tag. The buffered `<`
        # must be released, plus the new content.
        in_thinking, accumulated, dr, dc = server._step_thinking_state(
            "a", *s[:2][::-1][::-1][:2], gemma_fmt
        ) if False else server._step_thinking_state(
            "a", s[0], s[1], gemma_fmt
        )
        assert in_thinking is False
        assert accumulated == ""
        assert dr is None
        assert dc == "<a"

    def test_empty_token_no_op(self, gemma_fmt):
        in_thinking, accumulated, dr, dc = server._step_thinking_state(
            "", False, "", gemma_fmt
        )
        assert in_thinking is False
        assert accumulated == ""
        assert dr is None
        assert dc is None

    def test_helper_appends_token_internally_no_double_count(self, gemma_fmt):
        # Regression for the "every word doubled" bug observed in the
        # production Gemma 4 stream after the helper rewrite. The caller
        # must NOT pre-append `token.text` to `accumulated` before calling
        # the helper — the helper does it internally. If both append,
        # every byte of the token streams twice (visible content +
        # reasoning), which the user saw as "The The user user is is...".
        #
        # Pin the contract: drive a sequence of plain-text tokens (no
        # thinking transitions) and assert the concatenated emitted
        # content exactly equals the input concatenation, byte-for-byte.
        in_thinking = False
        accumulated = ""
        emitted = []
        tokens = ["Hello", " ", "world", ", ", "this is ", "a ", "test."]
        for t in tokens:
            in_thinking, accumulated, dr, dc = server._step_thinking_state(
                t, in_thinking, accumulated, gemma_fmt
            )
            assert dr is None, f"unexpected reasoning emit on plain token: {dr!r}"
            if dc is not None:
                emitted.append(dc)
        assert "".join(emitted) == "".join(tokens), (
            f"emitted content {''.join(emitted)!r} != input {''.join(tokens)!r}"
        )

    def test_helper_appends_token_internally_with_buffered_partial(self, gemma_fmt):
        # Same byte-for-byte invariant when partial buffering is in
        # play. Tokens carry a `<` that ends up not being a tag (the
        # next token resolves it as plain content). Output across both
        # tokens must exactly equal the input concatenation.
        in_thinking, accum1, dr1, dc1 = server._step_thinking_state(
            "before <", False, "", gemma_fmt
        )
        in_thinking, accum2, dr2, dc2 = server._step_thinking_state(
            " after", in_thinking, accum1, gemma_fmt
        )
        assert dr1 is None and dr2 is None
        emitted = (dc1 or "") + (dc2 or "")
        assert emitted == "before < after", (
            f"got {emitted!r}, expected 'before < after'"
        )

    def test_seeded_in_thinking_elides_per_turn_opener(self, gemma_fmt):
        # Production bug (Gemma 4 26B 8-bit, OWUI first-turn):
        # `<|think|>` global system marker seeds in_thinking=True. The
        # model's first emission is the per-turn opener
        # `<|channel>thought\n` followed by reasoning. Pre-fix the
        # state machine only scanned closers while in_thinking, so the
        # opener literal leaked into delta.reasoning and the user
        # saw `<|channel>thought\nThe user is asking...` rendered
        # verbatim in the thinking block.
        # Post-fix: openers seen while already in_thinking are
        # structural markers — elide without state transition, so
        # only the actual reasoning prose streams to delta.reasoning.
        in_thinking, accumulated, reasoning, content = self._drive(
            [
                "<|channel>thought\n",
                "The user is asking who I am.",
                "<channel|>",
                "I am a large language model.",
            ],
            gemma_fmt,
            in_thinking_start=True,
        )
        assert in_thinking is False
        assert accumulated == ""
        assert reasoning == "\nThe user is asking who I am."
        assert content == "I am a large language model."
        # Crucial invariant: no fragment of the per-turn opener
        # literal leaked into the reasoning stream.
        assert "<|channel" not in reasoning
        assert "channel>thought" not in reasoning

    def test_seeded_in_thinking_elides_opener_split_across_tokens(self, gemma_fmt):
        # Same bug, byte-streamed variant: the opener arrives byte-by-
        # byte from the tokenizer. The partial-buffer path must fire
        # immediately (ends-with-prefix) so no prefix bytes leak as
        # reasoning content while accumulated is too short to contain
        # the full literal.
        in_thinking, accumulated, reasoning, content = self._drive(
            ["<", "|", "channel", ">thought", "\nreasoning"],
            gemma_fmt,
            in_thinking_start=True,
        )
        assert in_thinking is True
        assert accumulated == ""
        assert reasoning == "\nreasoning"
        assert content == ""
        assert "<" not in reasoning
        assert "channel" not in reasoning


class TestIsTemplateThinkingAsymmetric:
    """Tests for the asymmetric-rendering heuristic that gates the
    SWA-snapshot path (memory.md #30).

    The heuristic: a thinking format detected in the rendered prompt
    means a thinking-aware client (OWUI, OpenAI SDK with reasoning
    suppression, etc.) will likely strip reasoning before echoing the
    assistant turn back. Cache holds full thinking content; next
    request's render lacks it → asymmetric. Engaging the snapshot
    path is the safe default for any thinking model.
    """

    def test_gemma_native_thinking_is_asymmetric(self):
        # Gemma 4: `<|think|>` opener anywhere in the rendered prompt
        # signals the model is in a thinking-aware regime.
        prompt = "<|think|>\nsystem stuff\n<turn|>...<turn|>model\n"
        assert server._is_template_thinking_asymmetric(prompt) is True

    def test_qwen_thinking_is_asymmetric(self):
        prompt = "<|im_start|>assistant\n<think>\nreasoning\n</think>\n"
        assert server._is_template_thinking_asymmetric(prompt) is True

    def test_gpt_oss_channel_is_asymmetric(self):
        prompt = "user x <|channel>thought\nfoo<channel|> visible"
        assert server._is_template_thinking_asymmetric(prompt) is True

    def test_no_thinking_format_is_symmetric(self):
        # Plain non-thinking prompt — no detection, no snapshot needed.
        # Cache anchors at end-of-asst (current symmetric behavior).
        prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        assert server._is_template_thinking_asymmetric(prompt) is False

    def test_empty_prompt_is_symmetric(self):
        assert server._is_template_thinking_asymmetric("") is False


class TestDetectThinkingFormat:
    def test_gemma_channel_opener(self):
        # The native Gemma 4 thinking marker.
        prompt = "user msg ... <|channel>thought\n"
        assert server._detect_thinking_format(prompt) == "gemma"

    def test_gemma_alternate_think_marker(self):
        # The "<|think|>" alternate marker also resolves to gemma.
        assert server._detect_thinking_format("foo <|think|> bar") == "gemma"

    def test_channel_thought_opener_matches_gemma(self):
        # `<|channel>thought` is now in Gemma 4's openers tuple too
        # (per-turn inline thinking, same syntax as gpt-oss). Gemma is
        # registry-listed first, so first-match wins. Behavior-wise
        # identical to gpt-oss for streaming purposes; only the brand
        # differs.
        fmt = server._detect_thinking_format("user msg ... <|channel>thought\n")
        assert fmt is not None
        assert fmt.name == "gemma"

    def test_qwen_think_tag(self):
        # Qwen / DeepSeek / generic `<think>...</think>` family.
        fmt = server._detect_thinking_format("hello <think>\n")
        assert fmt is not None
        assert fmt.name == "qwen"

    def test_no_thinking_tags(self):
        assert server._detect_thinking_format("just a plain prompt") is None

    def test_gemma_takes_precedence_over_generic(self):
        # If both appear, gemma wins (it's checked first). This matters
        # because some templates emit both markers in nested chains.
        prompt = "<|channel>thought ... <think>"
        assert server._detect_thinking_format(prompt) == "gemma"


class TestComputeThinkingBudget:
    def test_returns_none_when_no_format(self):
        # Non-thinking models get no budget enforcement at all.
        assert server._compute_thinking_budget(1000, None, None) is None

    def test_returns_none_even_with_client_budget_and_no_format(self):
        # Client-supplied budget is ignored when the model doesn't emit
        # thinking tags — there's nothing to count.
        assert server._compute_thinking_budget(1000, 500, None) is None

    def test_auto_budget_from_max_tokens(self):
        # Default formula: 80% of max_tokens. Floor via int().
        out = server._compute_thinking_budget(1000, None, "gemma")
        assert out == 800

    def test_auto_budget_uses_ratio_constant(self):
        # If THINKING_BUDGET_RATIO ever changes, this assertion catches
        # accidental drift between server.py and the documented ratio.
        out = server._compute_thinking_budget(2000, None, "generic")
        assert out == int(2000 * server.THINKING_BUDGET_RATIO)

    def test_client_budget_overrides_auto(self):
        # Explicit thinking_budget on the request wins, even when smaller
        # than the auto formula.
        assert server._compute_thinking_budget(10000, 500, "gemma") == 500

    def test_client_budget_can_exceed_auto(self):
        # And even when larger — clients can opt into deeper thinking.
        assert server._compute_thinking_budget(1000, 5000, "generic") == 5000

    def test_zero_client_budget_disables_thinking_immediately(self):
        # 0 is a meaningful explicit value — "no thinking" — distinct
        # from None. Server-side enforcement should fire at first thinking
        # token.
        assert server._compute_thinking_budget(1000, 0, "gemma") == 0


class TestMakeLogprobContent:
    class _FakeTokenizer:
        """Stub tokenizer that maps known token ids to text."""

        def __init__(self, mapping):
            self.mapping = mapping

        def decode(self, ids):
            return self.mapping.get(int(ids[0]), f"<unk:{ids[0]}>")

    def test_chosen_token_logprob_only(self):
        tk = self._FakeTokenizer({42: "hello"})
        out = server._make_logprob_content(tk, token_id=42, logprob=-0.5)
        assert out.token == "hello"
        assert out.logprob == pytest.approx(-0.5)
        assert out.top_logprobs == []
        # bytes() of "hello" UTF-8.
        assert out.bytes == list(b"hello")

    def test_top_k_zero_skips_top_logprobs_even_when_provided(self):
        # The contract: top_k=0 means "don't include alternatives" — the
        # caller didn't ask for them. Honoring this matters because
        # building TopLogprob entries requires an extra decode per id.
        tk = self._FakeTokenizer({1: "a", 2: "b"})
        out = server._make_logprob_content(
            tk, token_id=1, logprob=-1.0, top_logprobs=[(2, -2.0)], top_k=0
        )
        assert out.top_logprobs == []

    def test_top_k_truncates_top_logprobs(self):
        tk = self._FakeTokenizer({1: "a", 2: "b", 3: "c", 4: "d"})
        out = server._make_logprob_content(
            tk,
            token_id=1,
            logprob=-1.0,
            top_logprobs=[(2, -2.0), (3, -3.0), (4, -4.0)],
            top_k=2,
        )
        assert [t.token for t in out.top_logprobs] == ["b", "c"]
        assert [t.logprob for t in out.top_logprobs] == [-2.0, -3.0]

    def test_decode_failure_yields_empty_string_not_crash(self):
        # _decode_token swallows exceptions — an unknown id with a
        # tokenizer that raises shouldn't take down the streaming loop.
        class _FailTokenizer:
            def decode(self, ids):
                raise RuntimeError("boom")

        out = server._make_logprob_content(_FailTokenizer(), token_id=999, logprob=-1.0)
        assert out.token == ""
        # Empty string -> empty bytes list (not None).
        assert out.bytes == []

    def test_logprob_coerced_to_float(self):
        # The streaming path passes scalar mx.array values via .item();
        # if a numpy float64 ever leaks through, the Pydantic model
        # should still accept it.
        import numpy as np

        tk = self._FakeTokenizer({1: "x"})
        out = server._make_logprob_content(tk, token_id=1, logprob=np.float64(-0.25))
        assert isinstance(out.logprob, float)
        assert out.logprob == pytest.approx(-0.25)


class TestBuildGenArgsPenaltyAndSeedPlumbing:
    """memory.md #24 — verify the four advanced-params knobs flow from
    request body / Ollama aliases all the way through to GenerationArguments.
    Regressions here are silent: the slider in OpenWebUI moves but the
    server ignores it.
    """

    def _base_request(self, **overrides):
        # Minimal request stub matching the attributes _build_gen_args
        # reads. Every field defaulted to None so individual tests only
        # set what they care about.
        defaults = dict(
            max_tokens=None,
            max_output_tokens=None,
            temperature=None,
            top_p=None,
            top_k=None,
            min_p=None,
            seed=None,
            repetition_penalty=None,
            repeat_penalty=None,
            presence_penalty=None,
            frequency_penalty=None,
            logit_bias=None,
            enable_thinking=False,
            thinking_budget=None,
            thinking_start_token=None,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_seed_plumbed_through(self):
        req = self._base_request(seed=42)
        args = server._build_gen_args(req)
        assert args.seed == 42

    def test_seed_default_is_none_not_zero(self):
        # Critical: a non-None default of 0 would silently re-seed every
        # request to the same value, eliminating sampling variance. The
        # contract is "omitted seed = don't reseed".
        req = self._base_request()
        args = server._build_gen_args(req)
        assert args.seed is None

    def test_repeat_penalty_alias_recognized(self):
        # Ollama / OpenWebUI native UI slider name. Must alias to
        # repetition_penalty when the OpenAI-style name isn't present.
        req = self._base_request(repeat_penalty=1.15)
        args = server._build_gen_args(req)
        assert args.repetition_penalty == 1.15

    def test_repetition_penalty_wins_when_both_set(self):
        # If a client somehow sends both, the OpenAI-style name takes
        # precedence (the alias is a fallback, not an override).
        req = self._base_request(repetition_penalty=1.20, repeat_penalty=1.05)
        args = server._build_gen_args(req)
        assert args.repetition_penalty == 1.20

    def test_repeat_penalty_falsy_falls_through_to_repetition(self):
        # Defensive: 0 / None / False on the alias must not poison a
        # real repetition_penalty value. Implementation uses `or`, so
        # any falsy alias falls through.
        req = self._base_request(repetition_penalty=1.10, repeat_penalty=None)
        args = server._build_gen_args(req)
        assert args.repetition_penalty == 1.10

    def test_presence_penalty_plumbed_through(self):
        # Qwen 3.x family REQUIRES presence_penalty (rep_penalty is
        # forbidden by the model creator). Losing this drops the only
        # sane loop-mitigation knob for those models.
        req = self._base_request(presence_penalty=1.5)
        args = server._build_gen_args(req)
        assert args.presence_penalty == 1.5

    def test_frequency_penalty_plumbed_through(self):
        # Llama 3.x family uses frequency_penalty. Same silent-drop risk.
        req = self._base_request(frequency_penalty=0.5)
        args = server._build_gen_args(req)
        assert args.frequency_penalty == 0.5

    def test_all_four_penalties_independent(self):
        req = self._base_request(
            seed=7,
            repetition_penalty=1.10,
            presence_penalty=1.20,
            frequency_penalty=0.30,
        )
        args = server._build_gen_args(req)
        assert args.seed == 7
        assert args.repetition_penalty == 1.10
        assert args.presence_penalty == 1.20
        assert args.frequency_penalty == 0.30

    def test_unset_penalties_are_none_not_zero(self):
        # Distinguishing None from 0 matters: mlx_lm's
        # make_logits_processors checks `is not None` to decide whether
        # to install each processor. A 0.0 default would install a no-op
        # processor and burn cycles per token.
        args = server._build_gen_args(self._base_request())
        assert args.repetition_penalty is None
        assert args.presence_penalty is None
        assert args.frequency_penalty is None
        assert args.seed is None


class TestTokenIteratorHeartbeat:
    """The streaming iterator filters KeepAlive heartbeats so slow
    prefill (which produces no real tokens for many seconds) doesn't
    trip the queue-timeout. The contract:

      - KeepAlive items don't yield, but reset the timeout (each
        rqueue.get returns one queue interaction).
      - None terminates the stream cleanly.
      - Exception items raise to the caller.
      - StreamingToken with finish_reason ends the stream after yield.
      - Real silence longer than TOKEN_QUEUE_TIMEOUT_SECS raises
        queue.Empty (not caught here — surfaces to the caller).
    """

    @staticmethod
    def _make_response_generator():
        """Bypass ResponseGenerator.__init__ — we only exercise
        _token_iterator and _cancel, neither of which touches model state.
        """
        rg = server.ResponseGenerator.__new__(server.ResponseGenerator)
        rg._cancelled = set()
        rg._cancel_lock = __import__("threading").Lock()
        return rg

    def test_keepalive_filtered_not_yielded(self):
        from queue import Queue

        rg = self._make_response_generator()
        q: Queue = Queue()
        # Many heartbeats then one real token then sentinel.
        q.put(server.KeepAlive())
        q.put(server.KeepAlive())
        q.put(server.KeepAlive())
        token = server.StreamingToken(
            text="hi", token=42, logprobs=-0.1, finish_reason=None
        )
        q.put(token)
        q.put(None)

        items = list(rg._token_iterator(q, uid=1))
        # All heartbeats consumed silently — only the real token reaches
        # the caller.
        assert len(items) == 1
        assert items[0].token == 42

    def test_finish_reason_token_ends_stream(self):
        from queue import Queue

        rg = self._make_response_generator()
        q: Queue = Queue()
        final = server.StreamingToken(
            text="bye", token=2, logprobs=0.0, finish_reason="stop"
        )
        q.put(final)
        # Sentinel never gets a chance to be read — the iterator should
        # end on finish_reason without blocking on the queue.

        items = list(rg._token_iterator(q, uid=2))
        assert len(items) == 1
        assert items[0].finish_reason == "stop"

    def test_none_sentinel_terminates_cleanly(self):
        from queue import Queue

        rg = self._make_response_generator()
        q: Queue = Queue()
        q.put(None)

        items = list(rg._token_iterator(q, uid=3))
        assert items == []
        # Ended cleanly — no cancellation queued.
        assert 3 not in rg._cancelled

    def test_exception_item_raises_to_caller(self):
        from queue import Queue

        rg = self._make_response_generator()
        q: Queue = Queue()
        q.put(RuntimeError("backend exploded"))

        gen = rg._token_iterator(q, uid=4)
        with pytest.raises(RuntimeError, match="backend exploded"):
            list(gen)

    def test_keepalive_bursts_dont_yield_anything(self):
        # Pure heartbeat stream followed by termination — verify the
        # iterator collapses cleanly without spurious yields.
        from queue import Queue

        rg = self._make_response_generator()
        q: Queue = Queue()
        for _ in range(50):
            q.put(server.KeepAlive())
        q.put(None)

        items = list(rg._token_iterator(q, uid=5))
        assert items == []

    def test_unfinished_iterator_cancels_uid(self, monkeypatch):
        # If the consumer breaks out early (or the iterator exits
        # without the daemon's None sentinel), the finally block must
        # call _cancel(uid) so the daemon stops generating tokens for a
        # client that's no longer listening.
        from queue import Queue

        rg = self._make_response_generator()
        q: Queue = Queue()
        token = server.StreamingToken(
            text="x", token=1, logprobs=0.0, finish_reason=None
        )
        q.put(token)
        # No None sentinel, no finish_reason — consumer breaks early.

        cancelled = []
        monkeypatch.setattr(rg, "_cancel", lambda uid: cancelled.append(uid))

        gen = rg._token_iterator(q, uid=99)
        # Consume the first token, then close without exhausting.
        first = next(gen)
        assert first.token == 1
        gen.close()

        assert cancelled == [99]

    def test_finished_iterator_does_not_cancel(self, monkeypatch):
        from queue import Queue

        rg = self._make_response_generator()
        q: Queue = Queue()
        q.put(None)  # immediate clean termination

        cancelled = []
        monkeypatch.setattr(rg, "_cancel", lambda uid: cancelled.append(uid))

        list(rg._token_iterator(q, uid=100))
        assert cancelled == []

    def test_heartbeat_resets_timeout_window(self, monkeypatch):
        # Critical regression guard: a steady drip of heartbeats keeps
        # the iterator alive even past TOKEN_QUEUE_TIMEOUT_SECS of real
        # wall time. Implementation detail: queue.get's timeout is a
        # per-call deadline, NOT a cumulative one — every successful
        # get (heartbeat included) resets the window.
        #
        # We don't sleep TOKEN_QUEUE_TIMEOUT_SECS in tests; instead we
        # patch the constant to a tiny value and prove that an
        # interleaved heartbeat-then-token stream completes successfully
        # despite each gap being shorter than the timeout but their sum
        # exceeding it — by simply emitting them faster than one step.
        from queue import Queue

        monkeypatch.setattr(server, "TOKEN_QUEUE_TIMEOUT_SECS", 0.5)

        rg = self._make_response_generator()
        q: Queue = Queue()
        # 20 heartbeats interleaved with 5 tokens — total stream is
        # well under 0.5s wall clock since everything is pre-queued.
        for _ in range(20):
            q.put(server.KeepAlive())
        for i in range(5):
            q.put(
                server.StreamingToken(
                    text=str(i), token=i, logprobs=0.0, finish_reason=None
                )
            )
        q.put(None)

        items = list(rg._token_iterator(q, uid=11))
        assert [t.token for t in items] == [0, 1, 2, 3, 4]


class TestStepEmitsHeartbeatDuringPrefill:
    """Direct test for the daemon-side hook: when batch_gen.next() yields
    no responses (we're inside a prefill chunk), _step pushes a
    KeepAlive to every active rqueue. Without this, prefill chunks
    would silently consume time and the iterator's queue-get timer
    would interpret the gap as a daemon hang.
    """

    @staticmethod
    def _make_response_generator():
        rg = server.ResponseGenerator.__new__(server.ResponseGenerator)
        rg._cancelled = set()
        return rg

    @staticmethod
    def _fake_batch_gen(responses):
        # batch_gen.next() returns (stats, responses). _step ignores
        # stats so the first slot can be anything.
        return SimpleNamespace(next=lambda **kw: (None, responses))

    def test_empty_responses_emits_keepalive_per_active_uid(self):
        from queue import Queue

        rg = self._make_response_generator()
        q1, q2 = Queue(), Queue()
        active = {
            10: {"rqueue": q1, "tokens": [], "prev_text": ""},
            20: {"rqueue": q2, "tokens": [], "prev_text": ""},
        }
        rg._step(self._fake_batch_gen([]), active)

        ka1 = q1.get_nowait()
        ka2 = q2.get_nowait()
        assert isinstance(ka1, server.KeepAlive)
        assert isinstance(ka2, server.KeepAlive)
        # No further items — heartbeat is one-per-step, not a flood.
        assert q1.empty()
        assert q2.empty()

    def test_responses_present_skips_heartbeat(self):
        # When prefill is done and the step produced real responses,
        # we don't ALSO push a heartbeat — the response itself counts
        # as activity, and stacking heartbeats behind tokens just
        # wastes queue churn.
        from queue import Queue

        rg = self._make_response_generator()
        rg.tokenizer = SimpleNamespace(decode=lambda toks: "x" * len(toks))
        q = Queue()
        active = {7: {"rqueue": q, "tokens": [], "prev_text": ""}}

        # Construct a minimal real-shaped response.
        resp = SimpleNamespace(
            uid=7, token=42, finish_reason=None, token_logprob=-0.5
        )
        rg._step(self._fake_batch_gen([resp]), active)

        # First item is the StreamingToken; no KeepAlive emitted.
        item = q.get_nowait()
        assert isinstance(item, server.StreamingToken)
        assert item.token == 42
        assert q.empty()

    def test_no_active_uids_emits_nothing(self):
        # Defensive: no active uids means no rqueues to ping. Must not
        # crash on the empty-dict iteration.
        rg = self._make_response_generator()
        rg._step(self._fake_batch_gen([]), active={})
        # Nothing to assert — just no exception.


class TestCachedPathHeartbeatWatchdog:
    """The cached path's stream_generate yields nothing during its
    internal prefill loop. _process_cached_request runs a per-request
    timer thread that pumps KeepAlive sentinels into rqueue while the
    `for chunk in stream_generate(...)` loop is active, so the
    iterator's queue timer doesn't interpret legitimate prefill silence
    as a hang.

    These tests stub stream_generate so the watchdog logic can be
    exercised without spinning up a real model.
    """

    @staticmethod
    def _make_response_generator(stream_generate_stub):
        """Build a ResponseGenerator with the bare attributes
        _process_cached_request reads. The stream_generate symbol is
        imported locally inside the method, so we patch via a fake
        ``mlx_vlm.generate.stream_generate``.
        """
        rg = server.ResponseGenerator.__new__(server.ResponseGenerator)
        rg.model = SimpleNamespace()
        rg.processor = SimpleNamespace()
        rg.vision_cache = None
        rg.kv_bits = None
        rg.kv_group_size = None
        rg.kv_quant_scheme = None
        rg.quantized_kv_start = None
        rg._cancelled = set()
        rg._cancel_lock = __import__("threading").Lock()
        return rg

    @staticmethod
    def _stub_args():
        # GenerationArguments needs only to_generate_kwargs() for our
        # purposes. Use the real class with defaults so the kwargs dict
        # is realistic.
        return server.GenerationArguments()

    def test_slow_prefill_produces_heartbeats(self, monkeypatch):
        # Simulate a slow prefill: stream_generate sleeps before yielding
        # its first chunk. With the watchdog, the rqueue receives at
        # least one KeepAlive before the real chunk arrives, plus the
        # final StreamingToken and None sentinel.
        from queue import Queue
        import time

        # Tighten the heartbeat interval so the test runs in ~0.1s.
        monkeypatch.setattr(server, "CACHED_PATH_HEARTBEAT_INTERVAL_SECS", 0.02)

        # Fake stream_generate: sleep 0.10s (≈ 5 heartbeat intervals),
        # then yield one terminal chunk. Mimics prefill silence followed
        # by a single end-of-generation token.
        def fake_stream_generate(**kwargs):
            time.sleep(0.10)
            yield SimpleNamespace(
                token=42,
                text="hi",
                logprobs=None,
                finish_reason="stop",
                peak_memory=0.0,
            )

        # mlx_vlm.generate the function shadows the submodule on the
        # package, so the dotted-path setattr resolves to the function.
        # Patch via sys.modules to reach the actual submodule that the
        # cached-path's local `from .generate import stream_generate`
        # resolves against.
        import sys

        monkeypatch.setattr(
            sys.modules["mlx_vlm.generate"], "stream_generate", fake_stream_generate
        )

        rg = self._make_response_generator(fake_stream_generate)
        rqueue: Queue = Queue()
        prompt_cache_state = SimpleNamespace()  # opaque; only forwarded

        rg._process_cached_request(
            rqueue=rqueue,
            prompt="hello",
            images=None,
            args=self._stub_args(),
            prompt_tokens=5,
            prompt_cache_state=prompt_cache_state,
        )

        # Drain the queue. Expected order:
        #   1. GenerationContext (always pushed first)
        #   2. >=1 KeepAlive (from the watchdog during prefill silence)
        #   3. StreamingToken (the real chunk)
        #   4. None (terminator)
        items = []
        while not rqueue.empty():
            items.append(rqueue.get_nowait())

        assert isinstance(items[0], server.GenerationContext)
        assert items[-1] is None
        keepalive_count = sum(1 for it in items if isinstance(it, server.KeepAlive))
        token_count = sum(1 for it in items if isinstance(it, server.StreamingToken))
        assert keepalive_count >= 1, (
            f"watchdog should have emitted at least one KeepAlive "
            f"during the 0.10s silence; got items={[type(i).__name__ for i in items]}"
        )
        assert token_count == 1

    def test_watchdog_stops_before_terminator(self, monkeypatch):
        # The finally block sets heartbeat_done BEFORE pushing the None
        # sentinel, so by the time None is on the queue the watchdog
        # is no longer pumping. After the iterator hits None it stops
        # reading; any straggler KeepAlive that landed earlier is
        # filtered by isinstance check.
        from queue import Queue
        import time

        monkeypatch.setattr(server, "CACHED_PATH_HEARTBEAT_INTERVAL_SECS", 0.01)

        def fake_stream_generate(**kwargs):
            yield SimpleNamespace(
                token=1,
                text="x",
                logprobs=None,
                finish_reason="stop",
                peak_memory=0.0,
            )

        # mlx_vlm.generate the function shadows the submodule on the
        # package, so the dotted-path setattr resolves to the function.
        # Patch via sys.modules to reach the actual submodule that the
        # cached-path's local `from .generate import stream_generate`
        # resolves against.
        import sys

        monkeypatch.setattr(
            sys.modules["mlx_vlm.generate"], "stream_generate", fake_stream_generate
        )

        rg = self._make_response_generator(fake_stream_generate)
        rqueue: Queue = Queue()

        rg._process_cached_request(
            rqueue=rqueue,
            prompt="x",
            images=None,
            args=self._stub_args(),
            prompt_tokens=1,
            prompt_cache_state=SimpleNamespace(),
        )

        # Give the daemon thread a moment to fully exit; the watchdog's
        # join(timeout=1.0) should have stopped it deterministically.
        time.sleep(0.05)
        items = []
        while not rqueue.empty():
            items.append(rqueue.get_nowait())

        # Sentinel must be the LAST item. No further heartbeats arrive
        # after the None — that would be a leak past the finally.
        assert items[-1] is None
        none_index = items.index(None)
        assert none_index == len(items) - 1, (
            f"None terminator must be last; got "
            f"{[type(i).__name__ for i in items]}"
        )

    def test_exception_in_stream_generate_still_stops_watchdog(self, monkeypatch):
        # If stream_generate raises, the except block puts the exception
        # on the queue, the finally block must still stop the watchdog
        # AND push the None sentinel.
        from queue import Queue
        import time

        monkeypatch.setattr(server, "CACHED_PATH_HEARTBEAT_INTERVAL_SECS", 0.01)

        def fake_stream_generate(**kwargs):
            time.sleep(0.05)
            raise RuntimeError("backend exploded")
            yield  # unreachable; needed to make this a generator

        # mlx_vlm.generate the function shadows the submodule on the
        # package, so the dotted-path setattr resolves to the function.
        # Patch via sys.modules to reach the actual submodule that the
        # cached-path's local `from .generate import stream_generate`
        # resolves against.
        import sys

        monkeypatch.setattr(
            sys.modules["mlx_vlm.generate"], "stream_generate", fake_stream_generate
        )

        rg = self._make_response_generator(fake_stream_generate)
        rqueue: Queue = Queue()

        rg._process_cached_request(
            rqueue=rqueue,
            prompt="x",
            images=None,
            args=self._stub_args(),
            prompt_tokens=1,
            prompt_cache_state=SimpleNamespace(),
        )

        items = []
        while not rqueue.empty():
            items.append(rqueue.get_nowait())

        # Order: GenerationContext, [KeepAlives], Exception, None.
        assert isinstance(items[0], server.GenerationContext)
        assert items[-1] is None
        # The error gets logged AND surfaced as an Exception item.
        assert any(isinstance(it, RuntimeError) for it in items)
