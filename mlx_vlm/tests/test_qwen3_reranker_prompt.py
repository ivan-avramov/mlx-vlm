"""Qwen3 text rerankers are prompted with the published format, not the checkpoint template.

Fork-only test. Qwen/Qwen3-Reranker-0.6B (revision e61197e, 2026) ships a chat_template.jinja
written for sentence-transformers that reads messages with roles "query" and "document";
rendered with the generic system/user messages it yields an empty query and document, and
every pair scores identically (0.0049 on five unrelated documents). The server now renders the
format from the model card directly; see docs/upstream-bugs.md item 7.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_vlm.server import reranking as rr


class _Tokenizer:
    """Encodes by splitting on spaces; a template that blanks the content, like the checkpoint's."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [hash(w) % 1000 + 2 for w in text.split()]

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(
            "<|im_start|>system judge <|im_start|>user <Instruct>: <Query>: <Document>:"
        )


def test_text_prompt_carries_query_document_and_instruction():
    prompt = rr._text_prompt(
        rr.RerankItem(text="notice period"), rr.RerankItem(text="120 days"), "Find it."
    )
    assert prompt.startswith(
        "<|im_start|>system\n" + rr.SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\n"
    )
    assert "<Instruct>: Find it." in prompt
    assert "<Query>: notice period" in prompt
    assert "<Document>: 120 days" in prompt
    assert not prompt.endswith(
        rr.TEXT_SUFFIX
    )  # the suffix is appended as tokens after truncation


def test_score_text_batch_tokenizes_the_rendered_prompt_not_the_template(monkeypatch):
    tokenizer = _Tokenizer()
    seen = {}

    def fake_padded(tok, sequences):
        seen["sequences"] = sequences
        length = max(map(len, sequences))
        ids = mx.array([[0] * (length - len(s)) + s for s in sequences])
        return ids, (ids != 0).astype(mx.int32)

    monkeypatch.setattr(rr, "_padded_tokens", fake_padded)
    monkeypatch.setattr(
        rr, "_binary_scores", lambda model, h, mask, tok: [0.5] * h.shape[0]
    )

    class _Inner:
        def embed_tokens(self, ids):
            return mx.zeros((*ids.shape, 4))

        layers = []

        def norm(self, h):
            return h

    model = type("M", (), {"language_model": type("L", (), {"model": _Inner()})()})()
    rr._score_text_batch(
        model,
        tokenizer,
        rr.RerankItem(text="alpha beta"),
        [rr.RerankItem(text="gamma"), rr.RerankItem(text="delta epsilon")],
        "inst",
    )
    expected = [
        tokenizer.encode(
            rr._text_prompt(
                rr.RerankItem(text="alpha beta"), rr.RerankItem(text=d), "inst"
            )
        )
        + tokenizer.encode(rr.TEXT_SUFFIX)
        for d in ("gamma", "delta epsilon")
    ]
    assert seen["sequences"] == expected
    assert seen["sequences"][0] != seen["sequences"][1]
