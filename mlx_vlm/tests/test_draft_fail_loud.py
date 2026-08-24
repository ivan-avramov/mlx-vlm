"""Fail-loud guards for speculative-drafter configuration (F2; campaign O40, 2026-08-24).

Fork-only file. The defect these pin: the server's load path is `if draft_model_path:` —
so `--draft-kind mtp` WITHOUT `--draft-model` skips drafter loading entirely and serves
plain autoregressive decode while the operator believes MTP is on. Every MTP measurement
taken through that path between 2026-08-16 and 2026-08-23 (including a milestone-closing
one) measured plain decode at ~1.0x. Same class: an incompatible drafter downgraded to a
WARNING + silent plain-decode fallback.

Contract:
  (a) a weights-backed draft kind (KNOWN_DRAFTER_KINDS) with no drafter path REFUSES at
      load time — never serves plain decode silently;
  (b) `suffix` (drafter-free) and the no-drafter config stay valid without a path;
  (c) an incompatible drafter REFUSES by default; MLX_VLM_DRAFT_ALLOW_FALLBACK=1 restores
      the old warn-and-fall-back behavior for deliberate degraded starts.

The server worker surfaces load-time raises via `_load_error` -> `wait_until_ready()`,
so a refusal here fails the server start loudly rather than hanging a request.
"""

import pytest

from mlx_vlm.speculative.drafters import (
    KNOWN_DRAFTER_KINDS,
    drafter_incompat_policy,
    require_draft_config,
)


class TestRequireDraftConfig:
    @pytest.mark.parametrize("kind", sorted(KNOWN_DRAFTER_KINDS))
    def test_weights_kind_without_path_refuses(self, kind):
        with pytest.raises(ValueError, match="requires a drafter path"):
            require_draft_config(kind, None)

    @pytest.mark.parametrize("kind", sorted(KNOWN_DRAFTER_KINDS))
    def test_weights_kind_with_path_passes(self, kind):
        require_draft_config(kind, "/models/some-drafter")

    def test_suffix_without_path_passes(self):
        """Suffix is drafter-free by construction — no weights, no path."""
        require_draft_config("suffix", None)

    def test_no_drafter_config_passes(self):
        """Plain decode explicitly asked for is not an error."""
        require_draft_config(None, None)

    def test_path_without_kind_passes(self):
        """Auto-detection from the drafter's model_type is the documented default."""
        require_draft_config(None, "/models/some-drafter")


class TestIncompatPolicy:
    def test_incompatible_drafter_refuses_by_default(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_DRAFT_ALLOW_FALLBACK", raising=False)
        with pytest.raises(RuntimeError, match="MLX_VLM_DRAFT_ALLOW_FALLBACK"):
            drafter_incompat_policy(ValueError("hidden size mismatch"))

    def test_env_gate_restores_the_fallback(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_DRAFT_ALLOW_FALLBACK", "1")
        # Returns (None, None) = serve plain decode, deliberately.
        assert drafter_incompat_policy(ValueError("hidden size mismatch")) == (None, None)

    def test_env_gate_zero_still_refuses(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_DRAFT_ALLOW_FALLBACK", "0")
        with pytest.raises(RuntimeError):
            drafter_incompat_policy(ValueError("hidden size mismatch"))
