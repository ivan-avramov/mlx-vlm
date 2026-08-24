"""MTP dispatch on the cached/inline path + the drafter-conflict gate (O40, 2026-08-24).

Fork-only file. The third gap that made native MTP undeployable: the session-cached
inline path (`_process_cached_request` -> stream_generate -> generate_step) wired ONLY
`draft_kind == "suffix"`, so a server started with an MTP drafter served plain decode
to every session request; and `generate()` hard-rejected `thinking_budget` for every
loaded non-suffix drafter regardless of path, though the inline mtp round loop now
honors the budget (test_mtp_thinking_budget.py).

Contract, factored into two pure helpers so it is testable without a model:
  - `_inline_draft_kwargs(draft_kind, draft_model, args)`: the gen_kwargs wiring for
    the inline path. suffix AND mtp wire through; structured output (logits_processors)
    falls back to plain decode for both (grammar-blind drafts + no grammar mask on the
    speculative path — the suffix precedent); other kinds never wire inline.
  - `_drafter_conflict(draft_kind, draft_model_loaded, args, cached)`: the generate()
    gate. thinking_budget is OK with suffix (any path) and with mtp ON THE CACHED PATH
    (where the inline round loop enforces it); everything else that used to error
    still errors — the batched paths have no criteria support yet.
"""

from types import SimpleNamespace

from mlx_vlm.server.generation import _drafter_conflict, _inline_draft_kwargs


def _args(logits_processors=None, thinking_budget=None):
    return SimpleNamespace(
        logits_processors=logits_processors, thinking_budget=thinking_budget
    )


DRAFTER = object()


class TestInlineDraftKwargs:
    def test_suffix_wires_through(self):
        kw = _inline_draft_kwargs("suffix", DRAFTER, _args())
        assert kw == {"draft_model": DRAFTER, "draft_kind": "suffix"}

    def test_mtp_wires_through(self):
        kw = _inline_draft_kwargs("mtp", DRAFTER, _args())
        assert kw == {"draft_model": DRAFTER, "draft_kind": "mtp"}

    def test_structured_output_falls_back_to_plain_decode(self):
        assert _inline_draft_kwargs("mtp", DRAFTER, _args(logits_processors=[1])) == {}
        assert (
            _inline_draft_kwargs("suffix", DRAFTER, _args(logits_processors=[1])) == {}
        )

    def test_other_kinds_never_wire_inline(self):
        assert _inline_draft_kwargs("eagle3", DRAFTER, _args()) == {}
        assert _inline_draft_kwargs("dflash", DRAFTER, _args()) == {}

    def test_no_drafter_no_wiring(self):
        assert _inline_draft_kwargs("mtp", None, _args()) == {}
        assert _inline_draft_kwargs(None, None, _args()) == {}


class TestDrafterConflictGate:
    def test_budget_with_mtp_on_cached_path_is_allowed(self):
        assert (
            _drafter_conflict("mtp", True, _args(thinking_budget=81920), cached=True)
            is None
        )

    def test_budget_with_mtp_on_batched_path_still_errors(self):
        msg = _drafter_conflict("mtp", True, _args(thinking_budget=81920), cached=False)
        assert msg is not None and "thinking_budget" in msg

    def test_budget_with_suffix_is_allowed_any_path(self):
        for cached in (True, False):
            assert (
                _drafter_conflict(
                    "suffix", True, _args(thinking_budget=81920), cached=cached
                )
                is None
            )

    def test_budget_with_eagle3_still_errors(self):
        msg = _drafter_conflict(
            "eagle3", True, _args(thinking_budget=81920), cached=True
        )
        assert msg is not None and "thinking_budget" in msg

    def test_structured_with_mtp_on_cached_path_is_allowed_plain_fallback(self):
        assert (
            _drafter_conflict("mtp", True, _args(logits_processors=[1]), cached=True)
            is None
        )

    def test_structured_with_mtp_on_batched_path_still_errors(self):
        msg = _drafter_conflict("mtp", True, _args(logits_processors=[1]), cached=False)
        assert msg is not None and "response_format" in msg

    def test_no_drafter_never_conflicts(self):
        assert (
            _drafter_conflict(
                None, False, _args(thinking_budget=81920, logits_processors=[1]),
                cached=False,
            )
            is None
        )
