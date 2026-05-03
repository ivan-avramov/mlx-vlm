"""Tests for prompt_utils module, specifically multimodal content handling."""

import pytest

from mlx_vlm.prompt_utils import (
    CACHE_ALIGNMENT_KWARGS,
    MessageBuilder,
    MessageFormat,
    MessageFormatter,
    extract_text_from_content,
    get_cache_alignment_kwargs,
    get_chat_template,
)


class TestExtractTextFromContent:
    """Tests for the extract_text_from_content function."""

    def test_string_content_passthrough(self):
        """String content should be returned as-is."""
        content = "Hello, describe this image."
        result = extract_text_from_content(content)
        assert result == "Hello, describe this image."

    def test_empty_string(self):
        """Empty string should return empty string."""
        result = extract_text_from_content("")
        assert result == ""

    def test_multimodal_content_with_text_and_image(self):
        """Should extract only text from multimodal content, skipping image_url."""
        content = [
            {"type": "text", "text": "Describe this image in detail."},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANS..."},
            },
        ]
        result = extract_text_from_content(content)
        assert result == "Describe this image in detail."

    def test_multimodal_content_with_input_image(self):
        """Should handle input_image type (alternative format)."""
        content = [
            {"type": "text", "text": "What do you see?"},
            {"type": "input_image", "image_url": "data:image/jpeg;base64,/9j/4AAQ..."},
        ]
        result = extract_text_from_content(content)
        assert result == "What do you see?"

    def test_multimodal_content_with_multiple_text_parts(self):
        """Should concatenate multiple text parts."""
        content = [
            {"type": "text", "text": "First part."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            {"type": "text", "text": "Second part."},
        ]
        result = extract_text_from_content(content)
        assert result == "First part. Second part."

    def test_multimodal_content_with_input_text(self):
        """Should handle input_text type."""
        content = [
            {"type": "input_text", "text": "Alternative text format."},
            {"type": "image_url", "image_url": {"url": "..."}},
        ]
        result = extract_text_from_content(content)
        assert result == "Alternative text format."

    def test_multimodal_content_with_content_field(self):
        """Should handle text items with 'content' instead of 'text' field."""
        content = [
            {"type": "text", "content": "Using content field."},
            {"type": "image_url", "image_url": {"url": "..."}},
        ]
        result = extract_text_from_content(content)
        assert result == "Using content field."

    def test_multimodal_content_with_audio(self):
        """Should skip audio content."""
        content = [
            {"type": "text", "text": "Transcribe this audio."},
            {"type": "input_audio", "input_audio": {"data": "base64audiodata..."}},
        ]
        result = extract_text_from_content(content)
        assert result == "Transcribe this audio."

    def test_empty_list(self):
        """Empty list should return empty string."""
        result = extract_text_from_content([])
        assert result == ""

    def test_list_with_only_images(self):
        """List with only images should return empty string."""
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        ]
        result = extract_text_from_content(content)
        assert result == ""

    def test_none_content(self):
        """None should return empty string."""
        result = extract_text_from_content(None)
        assert result == ""

    def test_large_base64_not_included(self):
        """Ensure large base64 strings are NOT included in output.

        This is the critical test case for the bug fix.
        A 428x1000 pixel image encoded as base64 is ~570KB.
        If this were tokenized as text, it would produce ~422k tokens.
        """
        # Simulate a realistic multimodal message with large base64
        large_base64 = "iVBOR" + "A" * 570000  # ~570KB of base64 data
        content = [
            {"type": "text", "text": "Extract product information from this image."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{large_base64}"},
            },
        ]
        result = extract_text_from_content(content)

        # Result should be the text only, not the base64
        assert result == "Extract product information from this image."
        # Result should be short, not hundreds of KB
        assert len(result) < 1000

    def test_real_world_openai_format(self):
        """Test with exact format sent by OpenAI-compatible clients."""
        content = [
            {
                "type": "text",
                "text": "이미지에서 상품명, 가격, 설명을 추출해주세요.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                },
            },
        ]
        result = extract_text_from_content(content)
        assert result == "이미지에서 상품명, 가격, 설명을 추출해주세요."


class TestApplyChatTemplateIntegration:
    """Integration tests for apply_chat_template with multimodal content.

    These tests verify the actual bug fix works end-to-end, not just the helper.
    Uses return_messages=True to inspect intermediate messages without mocking.
    """

    def test_nemotron_omni_formats_image_and_audio_messages(self):
        """Nemotron Omni should use typed multimodal content for HF templates."""
        from mlx_vlm.prompt_utils import apply_chat_template

        for model_type in (
            "nemotron_h_nano_omni",
            "nemotronh_nano_omni_reasoning_v3",
        ):
            result = apply_chat_template(
                None,
                {"model_type": model_type},
                "Describe the inputs.",
                return_messages=True,
                num_images=1,
                num_audios=1,
            )

            assert result == [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": "Describe the inputs.",
                            "content": "Describe the inputs.",
                        },
                        {"type": "audio"},
                    ],
                }
            ]

    def test_multimodal_message_does_not_include_base64_in_prompt(self):
        """Critical regression test: base64 should NOT appear in formatted messages.

        This test reproduces the exact bug scenario:
        - OpenAI-compatible multimodal message with base64 image
        - apply_chat_template should extract only text for tokenization
        """
        from mlx_vlm.prompt_utils import apply_chat_template

        config = {"model_type": "qwen2_vl"}

        # Multimodal message with base64 (the bug trigger)
        large_base64 = "iVBOR" + "A" * 10000  # Smaller for test speed
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{large_base64}"},
                    },
                ],
            }
        ]

        # Use return_messages=True to get intermediate messages without needing processor
        result = apply_chat_template(
            None,  # processor not needed with return_messages=True
            config,
            messages,
            return_messages=True,
            num_images=1,
        )

        # Verify: the messages should NOT contain base64
        assert isinstance(result, list)
        for msg in result:
            content = msg.get("content", "")
            if isinstance(content, str):
                assert "iVBOR" not in content, "Base64 data leaked into text content!"
                assert (
                    len(content) < 1000
                ), f"Content too long ({len(content)}), likely contains base64"
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text", "") or item.get("content", "")
                        assert "iVBOR" not in str(
                            text
                        ), "Base64 data leaked into text content!"

    def test_pydantic_basemodel_content_extraction(self):
        """Test that BaseModel message objects are handled correctly."""
        from pydantic import BaseModel

        from mlx_vlm.prompt_utils import apply_chat_template

        class ChatMessage(BaseModel):
            role: str
            content: list

        config = {"model_type": "qwen2_vl"}

        # BaseModel with multimodal content
        message = ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,ABC123XYZ"},
                },
            ],
        )

        result = apply_chat_template(
            None,
            config,
            [message],
            return_messages=True,
            num_images=1,
        )

        # Should extract text, not include base64
        assert isinstance(result, list)
        for msg in result:
            content = msg.get("content", "")
            if isinstance(content, str):
                assert "ABC123" not in content, "Base64 leaked from BaseModel content!"
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text", "") or item.get("content", "")
                        assert "ABC123" not in str(
                            text
                        ), "Base64 leaked from BaseModel content!"

    def test_single_dict_prompt_multimodal(self):
        """Single dict prompt with multimodal content should not include base64.

        This tests the isinstance(prompt, dict) code path, which is different
        from isinstance(prompt, list) where we pass a list of message dicts.
        """
        from mlx_vlm.prompt_utils import apply_chat_template

        config = {"model_type": "qwen2_vl"}

        # Single dict prompt (NOT a list of dicts)
        single_prompt = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this single prompt image."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,SINGLEBASE64DATA"},
                },
            ],
        }

        result = apply_chat_template(
            None,
            config,
            single_prompt,  # Note: dict, not [dict]
            return_messages=True,
            num_images=1,
        )

        assert isinstance(result, list)
        for msg in result:
            content = msg.get("content", "")
            if isinstance(content, str):
                assert (
                    "SINGLEBASE64" not in content
                ), "Base64 leaked from single dict prompt!"
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text", "") or item.get("content", "")
                        assert "SINGLEBASE64" not in str(text), "Base64 leaked!"

    def test_text_content_preserved_correctly(self):
        """Verify that text content is preserved correctly after extraction."""
        from mlx_vlm.prompt_utils import apply_chat_template

        config = {"model_type": "qwen2_vl"}

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please analyze this product image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ]

        result = apply_chat_template(
            None,
            config,
            messages,
            return_messages=True,
            num_images=1,
        )

        # Find the text content in the result
        found_text = False
        for msg in result:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "") or item.get("content", "")
                        if "analyze this product" in text:
                            found_text = True
            elif isinstance(content, str) and "analyze this product" in content:
                found_text = True

        assert found_text, "Original text content was not preserved!"


class TestExtractTextFromContentEdgeCases:
    """Edge case tests for extract_text_from_content."""

    def test_malformed_content_item_no_type(self):
        """Items without 'type' should be skipped."""
        content = [
            {"text": "No type field"},
            {"type": "text", "text": "Has type field"},
        ]
        result = extract_text_from_content(content)
        assert result == "Has type field"

    def test_content_with_non_dict_items(self):
        """Non-dict items in list should be skipped."""
        content = [
            "just a string",
            {"type": "text", "text": "Valid item"},
            123,
            None,
        ]
        result = extract_text_from_content(content)
        assert result == "Valid item"

    def test_text_item_with_empty_text(self):
        """Text items with empty text should not add extra spaces."""
        content = [
            {"type": "text", "text": ""},
            {"type": "text", "text": "Actual content"},
            {"type": "text", "text": ""},
        ]
        result = extract_text_from_content(content)
        assert result == "Actual content"


class TestCacheAlignmentKwargs:
    """memory.md #27 — cache-friendly chat-template kwargs registry.

    Invariant: every kwarg here is a true no-op on templates that don't
    reference it (Jinja's `is defined` guard makes that safe). The
    registry is consumed by the server when a per-chat PromptCacheState
    is active to keep multi-turn renderings byte-aligned.
    """

    def test_registry_returns_a_copy(self):
        # Mutating the result must not poison subsequent calls — server
        # threads merge this into apply_chat_template kwargs, and one
        # request mutating the dict would affect every other.
        first = get_cache_alignment_kwargs()
        first["intruder"] = True
        second = get_cache_alignment_kwargs()
        assert "intruder" not in second
        assert "intruder" not in CACHE_ALIGNMENT_KWARGS

    def test_registry_includes_preserve_thinking_for_unsloth_qwen(self):
        # Specific entry guard — losing this regresses cache reuse on
        # `unsloth/Qwen3.6-*` ports without producing any visible error
        # (just ~10x slower multi-turn prefill on hybrid models).
        kwargs = get_cache_alignment_kwargs()
        assert kwargs.get("preserve_thinking") is True

    def test_all_values_are_truthy_or_explicit_false(self):
        # Boolean toggles only; any None value would make the no-op
        # invariant unverifiable (`{% if x is defined %}` evaluates
        # `None` as defined).
        for key, value in get_cache_alignment_kwargs().items():
            assert value is not None, f"{key} must not be None in registry"


class TestThinkingFormatRegistry:
    """memory.md #29 — centralized opener/closer literals per family.

    Drift between sites (detection, splitter, streaming state machine,
    budget enforcer, prefilled-opener guard) was the root cause of the
    Gemma 4 leading-content-token-eating artifact. Tests pin the
    registry's contract: every consumer must agree on the tags.
    """

    def test_registry_has_known_families(self):
        names = [f.name for f in THINKING_FORMATS]
        assert names == ["gemma", "qwen", "gpt-oss"], (
            "ordering matters — first-match-wins resolves ambiguous "
            "prompts; specificity is encoded in the tuple order."
        )

    def test_each_format_is_frozen(self):
        # Frozen dataclass = registry entries can't be mutated at runtime
        # without raising, which guards against accidental drift if a
        # caller patches openers in-place.
        for fmt in THINKING_FORMATS:
            with pytest.raises(Exception):
                fmt.openers = ("hacked",)  # type: ignore[misc]

    def test_each_format_has_at_least_one_opener_and_closer(self):
        for fmt in THINKING_FORMATS:
            assert fmt.openers, f"{fmt.name} has no openers"
            assert fmt.closers, f"{fmt.name} has no closers"
            assert fmt.tag_token_count >= 1, f"{fmt.name} tag_token_count <= 0"

    def test_partial_buffers_are_prefixes_of_real_tags(self):
        # The streaming state-machine uses `partial_buffers` to suppress
        # mid-token leakage. Each entry must be a prefix of either an
        # opener or a closer; otherwise it'd never match a partial-tag
        # arrival and we'd leak bytes.
        for fmt in THINKING_FORMATS:
            all_tags = fmt.openers + fmt.closers
            for partial in fmt.partial_buffers:
                assert any(tag.startswith(partial) for tag in all_tags), (
                    f"{fmt.name}: partial_buffer {partial!r} is not a prefix "
                    f"of any opener/closer"
                )

    def test_gemma_carries_both_global_and_per_turn_openers(self):
        # Gemma 4 is trained on both shapes: `<|think|>` is the global
        # marker injected at the system block when enable_thinking=True,
        # and `<|channel>thought` is the per-turn opener the model emits
        # for in-line thinking blocks. Both must be in the registry's
        # openers tuple so the streaming state machine recognizes
        # either one.
        gemma = next(f for f in THINKING_FORMATS if f.name == "gemma")
        assert "<|think|>" in gemma.openers
        assert "<|channel>thought" in gemma.openers
        # Likewise, the model emits both `</think>` and `<channel|>`
        # as closers (cross-format trained); both must match.
        assert "</think>" in gemma.closers
        assert "<channel|>" in gemma.closers

    def test_qwen_uses_plain_think_opener(self):
        qwen = next(f for f in THINKING_FORMATS if f.name == "qwen")
        assert qwen.openers == ("<think>",)
        assert qwen.closers == ("</think>",)


class TestDetectThinkingFormat:
    """`detect_thinking_format` is the single entry point all server-side
    helpers use to learn what tags to expect. Tests pin both positive
    matches per family and the first-match-wins ambiguity resolution."""

    def test_detects_gemma_native_opener(self):
        assert detect_thinking_format("foo <|think|> bar").name == "gemma"

    def test_detects_qwen_open_tag(self):
        assert detect_thinking_format("hello <think>\n").name == "qwen"

    def test_channel_thought_opener_resolves_to_gemma(self):
        # Gemma 4 lists `<|channel>thought` among its openers (per-turn
        # inline thinking syntax) and is registry-listed first, so an
        # input containing only that opener resolves to gemma rather
        # than gpt-oss. Streaming-wise both formats use the same closer
        # so the behavior is identical; only the format-name brand
        # changes.
        assert detect_thinking_format("ms <|channel>thought x").name == "gemma"

    def test_returns_none_for_plain_text(self):
        assert detect_thinking_format("just a plain prompt") is None

    def test_returns_none_when_only_closer_present(self):
        # Closer-only outputs are handled at split-time (the prefilled-
        # opener case), but the detection-from-prompt path requires an
        # opener to identify the family unambiguously.
        assert detect_thinking_format("trailing </think> only") is None

    def test_first_match_wins_for_ambiguous_prompts(self):
        # `<|think|>` (gemma) listed before `<think>` (qwen). A prompt
        # that contains both literals resolves to gemma.
        prompt = "<|think|> first ... <think> second"
        assert detect_thinking_format(prompt).name == "gemma"

    def test_qwen_match_does_not_steal_from_channel_format(self):
        # `<think>` vs `<|channel>thought` are unrelated literals; Qwen
        # detection must not match a channel-formatted prompt. With
        # the merged Gemma registry, channel-formatted prompts now
        # resolve to gemma (Gemma 4 includes the channel openers).
        prompt = "<|channel>thought reasoning <channel|>"
        assert detect_thinking_format(prompt).name == "gemma"


class TestMessageFormatterTextOnlyFallback:
    """memory.md #20 — pure text models (Qwen 2.5, gemma3_text) load via
    mlx_lm and aren't in MODEL_CONFIG. MessageFormatter must return a
    plain text dict instead of raising.
    """

    def test_unknown_model_returns_plain_dict(self):
        formatter = MessageFormatter("definitely-not-a-real-model")
        msg = formatter.format_message(prompt="hello", role="user")
        assert msg == {"role": "user", "content": "hello"}

    def test_unknown_model_preserves_role(self):
        formatter = MessageFormatter("ghost_model")
        msg = formatter.format_message(prompt="sys text", role="system")
        assert msg["role"] == "system"
        assert msg["content"] == "sys text"

    def test_unknown_model_format_type_is_none(self):
        # Internal invariant: the fallback path is gated on
        # `format_type is None` (the formatter_map .get returns None).
        formatter = MessageFormatter("nonexistent_vlm")
        assert formatter.format_type is None

    def test_unknown_model_ignores_image_request(self):
        # No image tokens to insert for text-only models — the
        # num_images > 0 path must NOT crash.
        formatter = MessageFormatter("unknown_model")
        msg = formatter.format_message(prompt="hi", role="user", num_images=1)
        assert msg == {"role": "user", "content": "hi"}


class TestMessageFormatterImageSchema:
    """memory.md #3 — Gemma's chat template (and several others) require
    multimodal content as a list with `{type: image}` items, not OpenAI's
    `{type: image_url}`. The vision schema rewrite happens inside
    MessageFormatter._format_list_with_image, not in server.py.
    """

    def test_gemma_produces_type_image_dict(self):
        # gemma4 uses LIST_WITH_IMAGE per MODEL_CONFIG; vision content
        # must be `{type: image}` (NOT `{type: image_url}`).
        formatter = MessageFormatter("gemma4")
        msg = formatter.format_message(prompt="describe", role="user", num_images=1)
        assert msg["role"] == "user"
        types = [item.get("type") for item in msg["content"] if isinstance(item, dict)]
        assert "image" in types
        # Must be the bare `image` form, not `image_url`.
        assert "image_url" not in types

    def test_image_url_preferring_model_uses_image_url_schema(self):
        # Some models (e.g. ERNIE) actually want `{type: image_url}`.
        # The MessageFormatter switches on use_image_url; this guards
        # against accidentally rewriting their schema as well.
        msg = MessageBuilder.image_url_message()
        assert msg == {"type": "image_url"}

    def test_user_role_skip_image_token_omits_image(self):
        # Multi-message conversations skip image tokens on prior user
        # turns (only the latest user message gets the image).
        formatter = MessageFormatter("gemma4")
        msg = formatter.format_message(
            prompt="prior turn", role="user", num_images=1, skip_image_token=True
        )
        types = [item.get("type") for item in msg["content"] if isinstance(item, dict)]
        assert "image" not in types

    def test_assistant_role_never_gets_image(self):
        formatter = MessageFormatter("gemma4")
        msg = formatter.format_message(prompt="response", role="assistant", num_images=1)
        types = [item.get("type") for item in msg["content"] if isinstance(item, dict)]
        assert "image" not in types


class TestGetChatTemplateProcessorFallback:
    """memory.md #5 — AutoProcessor sometimes loads without exposing
    chat_template; get_chat_template must fall back to processor.tokenizer.
    """

    class _StubTokenizer:
        """Tokenizer with a real chat_template; records what it received."""

        def __init__(self, chat_template="dummy template"):
            self.chat_template = chat_template
            self.last_call = None

        def apply_chat_template(self, messages, **kwargs):
            self.last_call = {"messages": messages, "kwargs": kwargs}
            # Return something deterministic so callers can assert the
            # fallback ran.
            return f"RENDERED({len(messages)} msgs)"

    def test_processor_with_template_used_directly(self):
        # Happy path: processor itself has chat_template + apply_chat_template.
        proc = self._StubTokenizer(chat_template="proc_template")
        result = get_chat_template(
            proc, [{"role": "user", "content": "hi"}], add_generation_prompt=True
        )
        assert result == "RENDERED(1 msgs)"

    def test_falls_back_to_tokenizer_when_processor_lacks_template(self):
        # Processor exposes apply_chat_template but no chat_template;
        # the helper should walk through to processor.tokenizer.
        class _ProcessorWithoutTemplate:
            chat_template = None

            def __init__(self, tk):
                self.tokenizer = tk

            def apply_chat_template(self, messages, **kwargs):
                raise AssertionError("processor path should have been skipped")

        tk = self._StubTokenizer(chat_template="tokenizer_template")
        proc = _ProcessorWithoutTemplate(tk)
        result = get_chat_template(
            proc, [{"role": "user", "content": "hi"}], add_generation_prompt=True
        )
        assert result == "RENDERED(1 msgs)"
        # Confirms it actually went through the tokenizer.
        assert tk.last_call is not None

    def test_falls_back_to_plain_prompt_when_no_template_anywhere(self):
        # Both processor and tokenizer lack chat_template — the helper
        # builds a Role: content style flat prompt rather than crashing.
        class _BlankProcessor:
            chat_template = None

            class _BlankTokenizer:
                chat_template = None

                def apply_chat_template(self, *args, **kwargs):
                    raise AssertionError("plain path should bypass tokenizer")

            tokenizer = _BlankTokenizer()

            def apply_chat_template(self, *args, **kwargs):
                raise AssertionError("plain path should bypass processor")

        result = get_chat_template(
            _BlankProcessor(),
            [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "hi"},
            ],
            add_generation_prompt=True,
        )
        assert isinstance(result, str)
        assert "hi" in result
        # Multi-message conversations render with Role: content prefixes
        # and terminate with `Assistant:` when add_generation_prompt is
        # set. Single-user-message conversations short-circuit to bare
        # content (covered by the next test).
        assert result.rstrip().endswith("Assistant:")
        assert "you are helpful" in result

    def test_plain_prompt_single_user_message_returns_bare_content(self):
        # Special-case in _messages_to_plain_prompt: a single user
        # message renders as just its content, no Role: prefix or
        # trailing Assistant: marker. Models that lack a chat template
        # entirely behave most predictably this way.
        class _BlankProcessor:
            chat_template = None

            class _BlankTokenizer:
                chat_template = None

            tokenizer = _BlankTokenizer()

        result = get_chat_template(
            _BlankProcessor(),
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
        )
        assert result == "hi"

    def test_chat_template_override_kwarg_respected(self):
        # `chat_template` kwarg lets callers override even when the
        # processor exposes its own template.
        proc = self._StubTokenizer(chat_template="builtin")
        get_chat_template(
            proc,
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
            chat_template="custom override",
        )
        # The override is forwarded to apply_chat_template via **kwargs.
        assert proc.last_call["kwargs"].get("chat_template") == "custom override"
