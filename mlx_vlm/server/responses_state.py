import json
import os
import re
import uuid
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from ..prompt_utils import THINKING_FORMATS, ThinkingFormat, detect_thinking_format

RESPONSE_STORE_LIMIT = int(os.environ.get("MLX_VLM_RESPONSE_STORE_LIMIT", "1024"))
_CONTENT_MARKERS = ("<|START_TEXT|>", "<|END_TEXT|>")


def _strip_content_markers(text: str) -> str:
    for marker in _CONTENT_MARKERS:
        text = text.replace(marker, "")
    return text


@dataclass
class StoredResponse:
    response: Dict[str, Any]
    input_items: List[Dict[str, Any]]
    output_items: List[Dict[str, Any]]
    previous_response_id: Optional[str] = None


@dataclass
class ThinkingStreamDelta:
    reasoning: Optional[str] = None
    content: Optional[str] = None
    thinking_closed: bool = False


class ThinkingStreamState:
    """Split streamed thinking delimiters from user-visible content."""

    _DEFAULT_OPEN_CLOSE_MARKERS = (
        ("<|channel>thought", "<channel|>"),
        ("<think>", "</think>"),
        ("<|START_THINKING|>", "<|END_THINKING|>"),
    )

    @staticmethod
    def _registry_open_close_markers() -> Tuple[Tuple[str, str], ...]:
        """Open/close marker pairs sourced from prompt_utils.THINKING_FORMATS.

        Every consumer of thinking tags (this streaming state machine,
        ``_split_thinking``, the budget enforcer) reads the same registry
        so the family tag literals (Gemma's pipe-delimited ``<|think|>``,
        Qwen's ``<think>``, gpt-oss's ``<|channel>thought``) can't drift
        out of sync. Each format's openers x closers are expanded into
        pairs so any listed opener flips into thinking and any listed
        closer flips back out.
        """
        pairs: List[Tuple[str, str]] = []
        for fmt in THINKING_FORMATS:
            for opener in fmt.openers:
                for closer in fmt.closers:
                    pair = (opener, closer)
                    if pair not in pairs:
                        pairs.append(pair)
        return tuple(pairs)

    def __init__(
        self,
        enable_thinking: bool = False,
        thinking_start_token: Optional[str] = None,
        thinking_end_token: Optional[str] = None,
    ):
        self.open_close_markers = self._build_open_close_markers(
            thinking_start_token, thinking_end_token
        )
        self.open_markers = tuple(marker for marker, _ in self.open_close_markers)
        self.close_markers = tuple(marker for _, marker in self.open_close_markers)
        self.in_thinking = bool(enable_thinking)
        self.thinking_done = False
        self.buffer = ""

    def feed(self, text: str) -> ThinkingStreamDelta:
        self.buffer += text or ""
        reasoning = []
        content = []
        thinking_closed = False

        while self.buffer:
            if self.in_thinking:
                idx, marker = self._find_first(self.buffer, self.close_markers)
                if idx < 0:
                    emit, self.buffer = self._split_partial(
                        self.buffer, self.close_markers
                    )
                    emit = self._strip_open_marker(emit)
                    if emit:
                        reasoning.append(emit)
                    break

                before = self._strip_open_marker(self.buffer[:idx])
                if before:
                    reasoning.append(before)

                self.buffer = self.buffer[idx + len(marker) :].lstrip("\n")
                self.in_thinking = False
                self.thinking_done = True
                thinking_closed = True
                continue

            if self.thinking_done:
                emit, self.buffer = self._split_partial(self.buffer, _CONTENT_MARKERS)
                emit = _strip_content_markers(emit)
                if emit:
                    content.append(emit)
                break

            idx, marker = self._find_first(self.buffer, self.open_markers)
            if idx < 0:
                emit, self.buffer = self._split_partial(self.buffer, self.open_markers)
                emit = _strip_content_markers(emit)
                if emit:
                    content.append(emit)
                break

            if idx:
                emit = _strip_content_markers(self.buffer[:idx])
                if emit:
                    content.append(emit)

            self.buffer = self.buffer[idx + len(marker) :].lstrip("\n")
            self.in_thinking = True

        return ThinkingStreamDelta(
            reasoning="".join(reasoning) or None,
            content="".join(content) or None,
            thinking_closed=thinking_closed,
        )

    @classmethod
    def _build_open_close_markers(
        cls,
        thinking_start_token: Optional[str],
        thinking_end_token: Optional[str],
    ) -> Tuple[Tuple[str, str], ...]:
        markers = []
        if thinking_start_token and thinking_end_token:
            markers.append((thinking_start_token, thinking_end_token))
        # Registry-sourced family literals first (most-specific ordering
        # from THINKING_FORMATS). This puts Gemma's pipe-delimited
        # ``<|think|>`` ahead of the generic ``<think>`` default so a
        # ``<|think|>...</think>`` block isn't mis-split by the looser
        # ``<think>`` pair (which never matches the opener but does match
        # the shared ``</think>`` closer, stranding the real opener).
        for marker_pair in cls._registry_open_close_markers():
            if marker_pair not in markers:
                markers.append(marker_pair)
        for marker_pair in cls._DEFAULT_OPEN_CLOSE_MARKERS:
            if marker_pair not in markers:
                markers.append(marker_pair)
        return tuple(markers)

    @staticmethod
    def _find_first(text: str, markers: Tuple[str, ...]) -> Tuple[int, str]:
        found_idx = -1
        found_marker = ""
        for marker in markers:
            idx = text.find(marker)
            if idx >= 0 and (found_idx < 0 or idx < found_idx):
                found_idx = idx
                found_marker = marker
        return found_idx, found_marker

    @staticmethod
    def _split_partial(text: str, markers: Tuple[str, ...]) -> Tuple[str, str]:
        hold = 0
        for marker in markers:
            max_len = min(len(marker) - 1, len(text))
            for length in range(max_len, 0, -1):
                if text.endswith(marker[:length]):
                    hold = max(hold, length)
                    break
        if hold:
            return text[:-hold], text[-hold:]
        return text, ""

    def _strip_open_marker(self, text: str) -> str:
        for marker in self.open_markers:
            if marker in text:
                before, after = text.split(marker, 1)
                return before + after.lstrip("\n")
        return text


response_store: Dict[str, StoredResponse] = {}
response_store_order: deque = deque()
response_store_lock = Lock()


def suppress_tool_call_content(
    full_output: str,
    in_tool_call: bool,
    tc_start: Optional[str],
    delta_content: Optional[str],
) -> Tuple[bool, Optional[str]]:
    """Suppress tool-call markup from streamed delta.content."""
    if not tc_start:
        return in_tool_call, delta_content
    if not in_tool_call:
        if tc_start in full_output:
            return True, None

        if any(full_output.endswith(tc_start[:j]) for j in range(2, len(tc_start))):
            return False, None
    else:
        return True, None
    return in_tool_call, delta_content


def process_tool_calls(model_output: str, tool_module, tools):
    """Parse tool calls from model output using the appropriate tool parser."""
    called_tools = []
    remaining = model_output

    if tool_module.tool_call_start in model_output:
        if tool_module.tool_call_end == "":
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?(?:\n|$)", re.DOTALL
            )
        else:
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?{re.escape(tool_module.tool_call_end)}",
                re.DOTALL,
            )

        matches = re.findall(pattern, model_output)
        if matches:
            remaining = re.sub(pattern, " ", model_output).strip()
            for match in matches:
                call = (
                    match.strip()
                    .removeprefix(tool_module.tool_call_start)
                    .removesuffix(tool_module.tool_call_end)
                )
                try:
                    parsed = tool_module.parse_tool_call(call, tools)
                    parsed_calls = parsed if isinstance(parsed, list) else [parsed]
                    for tool_call in parsed_calls:
                        args = tool_call["arguments"]
                        called_tools.append(
                            {
                                "type": "function",
                                "index": len(called_tools),
                                "id": str(uuid.uuid4()),
                                "function": {
                                    "name": tool_call["name"].strip(),
                                    "arguments": (
                                        args
                                        if isinstance(args, str)
                                        else json.dumps(args, ensure_ascii=False)
                                    ),
                                },
                            },
                        )
                except Exception:
                    print(f"Invalid tool call: {call}")
    return dict(calls=called_tools, remaining_text=remaining)


def _as_plain_dict(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


def _sse_event(event_type: str, payload: Dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, default=_jsonable)}\n\n"


def _clean_reasoning(reasoning: str, start_marker: str) -> str:
    reasoning = reasoning.replace(start_marker, "")
    if start_marker == "<|channel>thought":
        reasoning = reasoning.lstrip("thought")
    return reasoning.strip()


def _strip_thinking_quirks(fmt: ThinkingFormat, reasoning: str) -> str:
    """Apply per-format reasoning-text fixups after opener removal.

    gpt-oss leaves a literal "thought" word right after the opener even
    when the opener literal ``<|channel>thought`` is stripped (the
    tokenization splits the channel name). Other formats are clean.
    """
    cleaned = reasoning.strip()
    if fmt.name == "gpt-oss":
        if cleaned.startswith("thought"):
            cleaned = cleaned[len("thought") :].lstrip()
    return cleaned


def _split_thinking_by_format(
    text: str, fmt: ThinkingFormat
) -> Tuple[Optional[str], str]:
    """Split using a single resolved ThinkingFormat's tag literals.

    Mirrors the fork's registry-driven splitter: strips all of the
    family's openers from the reasoning span and splits at the earliest
    closer occurrence. Handles the prefilled-opener case (closer present
    with no opener) by treating everything up to the closer as reasoning.
    """
    closer_idx = -1
    closer_used = ""
    for cl in fmt.closers:
        idx = text.find(cl)
        if idx >= 0 and (closer_idx < 0 or idx < closer_idx):
            closer_idx = idx
            closer_used = cl

    if closer_idx < 0:
        # Opener present but no closer — entire text is in-progress reasoning.
        reasoning = text
        for op in fmt.openers:
            reasoning = reasoning.replace(op, "")
        return _strip_thinking_quirks(fmt, reasoning) or None, ""

    reasoning = text[:closer_idx]
    content = text[closer_idx + len(closer_used) :]
    for op in fmt.openers:
        reasoning = reasoning.replace(op, "")
    return (
        _strip_thinking_quirks(fmt, reasoning) or None,
        _strip_content_markers(content).strip(),
    )


def _split_thinking(
    text: str,
    thinking_start_token: Optional[str] = None,
    thinking_end_token: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """Split thinking tags from content. Returns (reasoning, content).

    When no explicit start/end tokens are supplied, prefer the
    THINKING_FORMATS registry (most-specific-first) so family literals
    like Gemma's pipe-delimited ``<|think|>`` resolve correctly and don't
    collide with the generic ``<think>`` default. Falls back to the
    hard-coded marker-pair loop for explicit tokens or the
    ``<|START_THINKING|>`` style defaults that aren't in the registry.
    """
    if not text:
        return None, text

    # Registry path: only when the caller didn't pin explicit tokens.
    if not (thinking_start_token and thinking_end_token):
        fmt = detect_thinking_format(text)
        if fmt is None:
            # Prefilled-opener case: a closer is present without any
            # opener (chat template seeded the model mid-thinking).
            for candidate in THINKING_FORMATS:
                if any(cl in text for cl in candidate.closers):
                    fmt = candidate
                    break
        if fmt is not None:
            return _split_thinking_by_format(text, fmt)

    for start_marker, end_marker in ThinkingStreamState._build_open_close_markers(
        thinking_start_token, thinking_end_token
    ):
        start = text.find(start_marker)
        end = text.find(end_marker, start if start >= 0 else 0)
        if start >= 0 and start < end:
            reasoning = text[start + len(start_marker) : end].strip()
            content = _strip_content_markers(
                text[:start] + text[end + len(end_marker) :]
            ).strip()
            return reasoning or None, content

        if end_marker in text:
            reasoning, content = text.split(end_marker, 1)
            reasoning = _clean_reasoning(reasoning, start_marker)
            return reasoning or None, _strip_content_markers(content).strip()

        if start_marker in text:
            reasoning = _clean_reasoning(text, start_marker)
            return reasoning or None, ""

    return None, _strip_content_markers(text).strip()


def _partial_tag_start_pos(
    accumulated: str, partial_buffers: Tuple[str, ...]
) -> Optional[int]:
    """Return the earliest position in ``accumulated`` where a partial
    tag prefix starts, OR None if accumulated does not end with a
    partial-tag prefix.

    A "partial tag" is any non-empty proper prefix of a known partial
    buffer literal that ``accumulated`` ends with. Crucially uses
    ENDSWITH a prefix rather than CONTAINS substring — the substring
    check would never fire until accumulated had grown large enough to
    contain the full partial, by which point the partial-tag bytes had
    already leaked into delta.content as multiple short tokens.
    """
    earliest: Optional[int] = None
    for p in partial_buffers:
        # Try every prefix length from longest to shortest so we get
        # the earliest start position when multiple lengths match.
        for k in range(len(p), 0, -1):
            tail_start = len(accumulated) - k
            if tail_start < 0:
                continue
            if accumulated[tail_start:] == p[:k]:
                if earliest is None or tail_start < earliest:
                    earliest = tail_start
                break
    return earliest


def _step_thinking_state(
    token_text: str,
    in_thinking: bool,
    accumulated: str,
    fmt: Optional[ThinkingFormat],
) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """Process one streamed token through the thinking-state machine.

    Returns ``(new_in_thinking, new_accumulated, delta_reasoning,
    delta_content)``. Each returned delta is either ``None`` (no emission
    of that kind for this token) or a non-empty string ready to send to
    the SSE consumer.

    Handles three classes of bug that the inline branch chain accumulated:

      1. Token-spanning tags eating content — splits at the tag boundary,
         emits pre-tag content under the OLD state, transitions, then
         emits post-tag content under the NEW state.
      2. Partial-buffer matching via ends-with-prefix (``_partial_tag_start_pos``)
         instead of substring, so tag prefixes never leak as content.
      3. Multiple transitions in one token — the internal ``while`` loop
         re-enters at the new state with the residual accumulated.

    A ``None`` ``fmt`` means no thinking format is detected; pass the
    token through as content unchanged (non-thinking models).
    """
    if fmt is None:
        return in_thinking, accumulated, None, (token_text or None)

    accumulated = accumulated + token_text
    reasoning_parts: List[str] = []
    content_parts: List[str] = []

    while True:
        # Scan ALL tags, distinguishing state-changing vs structural
        # markers. Closers are state-changing while in_thinking; openers
        # are state-changing while not in_thinking. The other direction
        # (e.g. an opener while already in_thinking) is a redundant
        # structural marker elided without a transition — needed for
        # Gemma 4's global ``<|think|>`` seeding, where the model emits a
        # per-turn opener ``<|channel>thought`` while already in_thinking.
        if in_thinking:
            state_tags = fmt.closers
            marker_tags = fmt.openers
        else:
            state_tags = fmt.openers
            marker_tags = fmt.closers

        best_idx = -1
        best_tag = ""
        best_changes_state = False
        for tag in state_tags:
            idx = accumulated.find(tag)
            if idx >= 0 and (best_idx < 0 or idx < best_idx):
                best_idx = idx
                best_tag = tag
                best_changes_state = True
        for tag in marker_tags:
            idx = accumulated.find(tag)
            if idx >= 0 and (best_idx < 0 or idx < best_idx):
                best_idx = idx
                best_tag = tag
                best_changes_state = False

        if best_idx >= 0:
            pre = accumulated[:best_idx]
            if pre:
                if in_thinking:
                    reasoning_parts.append(pre)
                else:
                    content_parts.append(pre)
            entering_thinking = best_changes_state and not in_thinking
            if best_changes_state:
                in_thinking = not in_thinking
            accumulated = accumulated[best_idx + len(best_tag) :]
            # Strip a leading newline immediately after an opener that enters
            # thinking — chat templates emit `<opener>\n` and the newline is
            # template noise, not reasoning content (matches the leading-\n
            # trim ThinkingStreamState applies).
            if entering_thinking:
                accumulated = accumulated.lstrip("\n")
            continue

        # No complete tag in accumulated. Check whether accumulated ENDS
        # with a partial-tag prefix; if so, emit everything before that
        # partial under the current state and keep the partial buffered
        # for the next token.
        partial_start = _partial_tag_start_pos(accumulated, fmt.partial_buffers)
        if partial_start is not None:
            pre = accumulated[:partial_start]
            if pre:
                if in_thinking:
                    reasoning_parts.append(pre)
                else:
                    content_parts.append(pre)
            accumulated = accumulated[partial_start:]
        else:
            if accumulated:
                if in_thinking:
                    reasoning_parts.append(accumulated)
                else:
                    content_parts.append(accumulated)
            accumulated = ""
        break

    delta_reasoning = "".join(reasoning_parts) if reasoning_parts else None
    delta_content = "".join(content_parts) if content_parts else None
    return in_thinking, accumulated, delta_reasoning, delta_content


def _response_output_items_from_text(
    full_text: str,
    message_id: str,
    tool_module: Any,
    chat_tools: List[Any],
    tool_registry: Dict[str, str],
    thinking_start_token: Optional[str] = None,
    thinking_end_token: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, Optional[str], str]:
    reasoning, content = _split_thinking(
        full_text, thinking_start_token, thinking_end_token
    )
    if tool_module is not None and chat_tools:
        tc = process_tool_calls(full_text, tool_module, chat_tools)
        if tc["calls"]:
            items = [
                _tool_call_to_response_item(call, tool_registry) for call in tc["calls"]
            ]
            _, remaining = _split_thinking(
                tc.get("remaining_text") or "",
                thinking_start_token,
                thinking_end_token,
            )
            remaining = re.sub(r"<\|[^>]+\|>|<[^>]+>", "", remaining).strip()
            return items, remaining, reasoning, "tool_calls"
    item = {
        "id": message_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": content, "annotations": []}],
    }
    if reasoning:
        item["reasoning"] = reasoning
    return [item], content, reasoning, "stop"


def _normalize_response_input(input_value: Any) -> List[Dict[str, Any]]:
    if isinstance(input_value, str):
        return [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": input_value}],
            }
        ]
    if not isinstance(input_value, list):
        raise HTTPException(status_code=400, detail="Invalid input format.")

    items = []
    for item in input_value:
        item = _as_plain_dict(item)
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="Invalid input format.")
        item_type = item.get("type")
        if item_type is None and item.get("role") is not None:
            item = {**item, "type": "message"}
        items.append(item)
    return items


def _response_call_to_chat_tool_call(item: Dict[str, Any]) -> Dict[str, Any]:
    call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"
    name = item.get("name")
    arguments = item.get("arguments")
    if item.get("type") == "shell_call":
        name = name or "shell"
        action = item.get("action") or {}
        arguments = arguments or json.dumps(action, ensure_ascii=False)
    elif item.get("type") == "apply_patch_call":
        name = name or "apply_patch"
        arguments = arguments or item.get("patch") or item.get("input") or "{}"
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, ensure_ascii=False)
    return {
        "type": "function",
        "id": call_id,
        "function": {"name": name or "tool", "arguments": arguments},
    }


def _response_image_source(part: Dict[str, Any]) -> Optional[Any]:
    part_type = part.get("type")
    if part_type == "image_url":
        image_url = part.get("image_url")
        return image_url.get("url") if isinstance(image_url, dict) else image_url
    if part_type != "input_image":
        return None

    if part.get("file_id") is not None:
        raise HTTPException(
            status_code=400,
            detail=(
                "input_image.file_id is not supported by this server. "
                "Provide image_url instead."
            ),
        )
    image_url = part.get("image_url")
    return image_url or None


def _response_tool_output_to_text_and_images(
    output: Any,
) -> Tuple[str, List[Any]]:
    if isinstance(output, str):
        return output, []
    if not isinstance(output, list):
        return json.dumps(output, ensure_ascii=False), []

    text_parts = []
    remaining_parts = []
    output_images = []
    for part in output:
        part = _as_plain_dict(part)
        if not isinstance(part, dict):
            remaining_parts.append(part)
            continue
        part_type = part.get("type")
        if part_type in ("input_text", "output_text", "text"):
            text_parts.append(str(part.get("text", "")))
        elif part_type in ("input_image", "image_url"):
            image = _response_image_source(part)
            if image:
                output_images.append(image)
            else:
                remaining_parts.append(part)
        else:
            remaining_parts.append(part)

    if remaining_parts:
        text_parts.append(json.dumps(remaining_parts, ensure_ascii=False))
    if output_images:
        text_parts.append("[Image output attached in the next message]")
    return "\n".join(part for part in text_parts if part), output_images


def _response_image_message(image_count: int) -> Dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "image"} for _ in range(image_count)],
    }


def _append_response_item_to_prompt(
    item: Dict[str, Any],
    chat_messages: List[Dict[str, Any]],
    images: List[Any],
):
    item_type = item.get("type")
    if item_type == "message":
        role = item.get("role") or "user"
        content = item.get("content")
        if isinstance(content, list):
            content_parts = []
            item_images = []
            for part in content:
                part = _as_plain_dict(part)
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in ("input_text", "output_text", "text"):
                    text = str(part.get("text", ""))
                    if text:
                        content_parts.append({"type": "text", "text": text})
                elif part_type in ("input_image", "image_url"):
                    image = _response_image_source(part)
                    if image:
                        item_images.append(image)
                        content_parts.append({"type": "image"})
            images.extend(item_images)
            if item_images and role not in ("user",):
                text = "\n".join(
                    part["text"] for part in content_parts if part.get("type") == "text"
                )
                chat_messages.append({"role": role, "content": text})
                chat_messages.append(_response_image_message(len(item_images)))
                return
            if item_images:
                content = content_parts
            else:
                content = "\n".join(
                    part["text"] for part in content_parts if part.get("type") == "text"
                )
        chat_messages.append({"role": role, "content": content or ""})
        return

    if item_type in ("function_call", "shell_call", "apply_patch_call"):
        chat_messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_response_call_to_chat_tool_call(item)],
            }
        )
        return

    if item_type in (
        "function_call_output",
        "shell_call_output",
        "apply_patch_call_output",
        "tool_result",
    ):
        output = item.get("output", item.get("content", ""))
        output, output_images = _response_tool_output_to_text_and_images(output)
        chat_messages.append(
            {
                "role": "tool",
                "tool_call_id": item.get("call_id") or item.get("tool_call_id"),
                "content": output,
            }
        )
        if output_images:
            images.extend(output_images)
            chat_messages.append(_response_image_message(len(output_images)))


def _response_chain_items(previous_response_id: Optional[str]) -> List[Dict[str, Any]]:
    if not previous_response_id:
        return []
    chain: List[StoredResponse] = []
    seen = set()
    current_id = previous_response_id
    with response_store_lock:
        while current_id:
            if current_id in seen:
                break
            seen.add(current_id)
            stored = response_store.get(current_id)
            if stored is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Previous response not found: {current_id}",
                )
            chain.append(stored)
            current_id = stored.previous_response_id

    items: List[Dict[str, Any]] = []
    for stored in reversed(chain):
        items.extend(stored.input_items)
        items.extend(stored.output_items)
    return items


def _response_items_to_chat(
    items: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    chat_messages: List[Dict[str, Any]] = []
    images: List[Any] = []
    for item in items:
        _append_response_item_to_prompt(item, chat_messages, images)
    return chat_messages, images


def _store_response(
    response: Any,
    input_items: List[Dict[str, Any]],
    output_items: List[Dict[str, Any]],
    previous_response_id: Optional[str],
):
    if getattr(response, "store", True) is False:
        return
    payload = response.model_dump(exclude_none=True)
    with response_store_lock:
        response_store[response.id] = StoredResponse(
            response=payload,
            input_items=input_items,
            output_items=output_items,
            previous_response_id=previous_response_id,
        )
        response_store_order.append(response.id)
        while len(response_store_order) > RESPONSE_STORE_LIMIT:
            old_id = response_store_order.popleft()
            response_store.pop(old_id, None)


def _response_tool_to_chat_tool(tool: Any) -> Optional[Dict[str, Any]]:
    tool = _as_plain_dict(tool)
    if not isinstance(tool, dict):
        return None
    tool_type = tool.get("type")
    if tool_type == "function" and isinstance(tool.get("function"), dict):
        return tool
    if tool_type == "function":
        return {
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("parameters") or {},
            },
        }
    if tool_type == "shell":
        return {
            "type": "function",
            "function": {
                "name": tool.get("name") or "shell",
                "description": tool.get("description") or "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    if tool_type == "apply_patch":
        return {
            "type": "function",
            "function": {
                "name": tool.get("name") or "apply_patch",
                "description": tool.get("description") or "Apply a patch to files.",
                "parameters": {
                    "type": "object",
                    "properties": {"patch": {"type": "string"}},
                    "required": ["patch"],
                },
            },
        }
    return None


def _response_tool_registry(
    tools: Optional[List[Any]],
) -> Tuple[List[Any], Dict[str, str]]:
    chat_tools = []
    registry: Dict[str, str] = {}
    for tool in tools or []:
        plain = _as_plain_dict(tool)
        chat_tool = _response_tool_to_chat_tool(plain)
        if chat_tool is None:
            continue
        chat_tools.append(chat_tool)
        function = chat_tool.get("function", {})
        name = function.get("name")
        if name:
            registry[name] = (plain or {}).get("type", "function")
    return chat_tools, registry


def _tool_call_to_response_item(
    call: Dict[str, Any],
    registry: Dict[str, str],
) -> Dict[str, Any]:
    function = call.get("function", {})
    name = function.get("name") or "tool"
    arguments = function.get("arguments") or "{}"
    call_id = call.get("id") or f"call_{uuid.uuid4().hex}"
    tool_type = registry.get(name, "function")
    if tool_type == "shell":
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except Exception:
            parsed = {"command": arguments}
        command = parsed.get("command", parsed) if isinstance(parsed, dict) else parsed
        return {
            "id": f"sh_{uuid.uuid4().hex}",
            "type": "shell_call",
            "call_id": call_id,
            "status": "completed",
            "action": {"type": "exec", "command": command},
        }
    if tool_type == "apply_patch":
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except Exception:
            parsed = {"patch": arguments}
        patch = parsed.get("patch", parsed) if isinstance(parsed, dict) else parsed
        return {
            "id": f"apc_{uuid.uuid4().hex}",
            "type": "apply_patch_call",
            "call_id": call_id,
            "status": "completed",
            "patch": patch,
        }
    return {
        "id": f"fc_{uuid.uuid4().hex}",
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }
