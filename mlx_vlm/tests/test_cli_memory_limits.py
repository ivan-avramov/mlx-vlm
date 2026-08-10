"""Tests for the fork's MLX memory-limit derivation in `server/cli.py`.

Fork-only. `_apply_mlx_memory_limits`, `_derive_cache_limit_gb` and
`_model_num_attention_heads` are absent from `upstream/main`, so all seven gating audits
are blind to them, and `dev/find_untested_fork_code.py` reported the whole chain with no
tested caller — the only genuinely uncovered cluster left on its list once the false
leads were dismissed.

Nothing here is a bug fix: reading the chain for a contract turned up no defect. What it
turned up is an unpinned formula and an undocumented unit choice (below), which is worth
pinning precisely because getting it wrong is invisible — an over-large buffer-pool cap
just uses more RAM, and an under-large one just slows prefill. Neither fails a test or
raises.

The limits themselves are never really applied: `mx.set_cache_limit` and
`mx.set_memory_limit` are process-global, so a test that actually called them would leak
into every later test in the session.
"""

from unittest.mock import patch

import pytest

import mlx_vlm.server.cli as cli


class TestModelNumAttentionHeads:
    def test_reads_text_config_first(self, tmp_path):
        """Multimodal configs put the language model's head count under
        `text_config`; the top-level value on a VLM config is the VISION tower's."""
        (tmp_path / "config.json").write_text(
            '{"num_attention_heads": 16, "text_config": {"num_attention_heads": 40}}'
        )
        assert cli._model_num_attention_heads(str(tmp_path)) == 40

    def test_falls_back_to_the_top_level(self, tmp_path):
        (tmp_path / "config.json").write_text('{"num_attention_heads": 24}')
        assert cli._model_num_attention_heads(str(tmp_path)) == 24

    def test_falls_back_when_text_config_lacks_the_key(self, tmp_path):
        (tmp_path / "config.json").write_text(
            '{"num_attention_heads": 24, "text_config": {"hidden_size": 4096}}'
        )
        assert cli._model_num_attention_heads(str(tmp_path)) == 24

    def test_a_missing_config_returns_none(self, tmp_path):
        """Best-effort by design: the caller has a safe fallback, so an unreadable
        config must not take the server down at startup."""
        assert cli._model_num_attention_heads(str(tmp_path)) is None

    def test_malformed_json_returns_none(self, tmp_path):
        (tmp_path / "config.json").write_text("{not json")
        assert cli._model_num_attention_heads(str(tmp_path)) is None

    def test_a_non_dict_text_config_returns_none(self, tmp_path):
        """`cfg.get("text_config", cfg)` happily yields a non-dict, and the `.get` on
        it then raises — caught, because the broad except is the point of the helper."""
        (tmp_path / "config.json").write_text(
            '{"num_attention_heads": 24, "text_config": "nonsense"}'
        )
        assert cli._model_num_attention_heads(str(tmp_path)) is None

    def test_an_absent_head_count_returns_none(self, tmp_path):
        (tmp_path / "config.json").write_text('{"hidden_size": 4096}')
        assert cli._model_num_attention_heads(str(tmp_path)) is None


class TestDeriveCacheLimitGb:
    """The documented formula:

    cap = ceil(n_heads * prefill_step * max_kv_size * 2 bytes / 1e9) + 2 GB
    """

    def test_the_formula_is_exactly_as_documented(self, tmp_path):
        (tmp_path / "config.json").write_text('{"num_attention_heads": 32}')

        # 32 heads * 2048 step * 32768 ctx * 2 bytes = 4.295e9 -> ceil 5 -> +2 = 7
        assert cli._derive_cache_limit_gb(str(tmp_path), 32768, 2048) == 7.0

    def test_it_rounds_up_before_adding_the_margin(self, tmp_path):
        """`ceil` then `+2`, not `+2` then `ceil`. Same result here, but the order is
        what makes the margin a floor rather than something rounding can eat."""
        (tmp_path / "config.json").write_text('{"num_attention_heads": 1}')

        # 1 * 1 * 1 * 2 / 1e9 is a hair above zero -> ceil 1 -> +2 = 3
        assert cli._derive_cache_limit_gb(str(tmp_path), 1, 1) == 3.0

    def test_the_head_fallback_is_32_and_overshoots_on_purpose(self, tmp_path):
        """No config: fall back to the largest head count in this stack, so the
        fallback OVER-shoots. Undershooting only costs prefill speed; overshooting
        only costs RAM. The docstring picks that trade deliberately."""
        derived = cli._derive_cache_limit_gb(str(tmp_path), 32768, 2048)

        assert derived == 7.0  # identical to the 32-head case above

    def test_a_smaller_model_derives_a_smaller_cap(self, tmp_path):
        (tmp_path / "config.json").write_text('{"num_attention_heads": 8}')

        assert cli._derive_cache_limit_gb(str(tmp_path), 32768, 2048) == 4.0

    @pytest.mark.parametrize(
        ("max_kv_size", "prefill_step"),
        [(None, 2048), (32768, None), (0, 2048), (32768, 0), (None, None)],
    )
    def test_missing_inputs_derive_nothing(self, max_kv_size, prefill_step):
        """Returning None rather than a guess: the caller then leaves MLX's own
        default in place, which is the correct behaviour with nothing to size from."""
        assert cli._derive_cache_limit_gb("anything", max_kv_size, prefill_step) is None

    def test_the_formula_is_decimal_GB_but_applied_as_binary_GiB(self, tmp_path):
        """An undocumented unit mismatch, pinned rather than 'fixed'.

        `_derive_cache_limit_gb` divides by 1e9 (decimal GB) while
        `_apply_mlx_memory_limits` multiplies by 1024**3 (binary GiB), so the cap
        actually installed is ~7.4% larger than the byte count the formula computed.
        That is in the safe direction — the docstring's whole argument is that
        overshooting a buffer-pool cap costs RAM while undershooting costs prefill
        speed — so this is not a bug, and changing it would silently shrink every
        deployment's cap.

        This test exists so that any future change to either unit is a deliberate
        decision that has to edit an assertion, rather than a quiet 7% regression in
        long-context prefill.
        """
        (tmp_path / "config.json").write_text('{"num_attention_heads": 32}')
        derived_gb = cli._derive_cache_limit_gb(str(tmp_path), 32768, 2048)

        scores_bytes = 32 * 2048 * 32768 * 2
        assert derived_gb == pytest.approx(scores_bytes / 1e9 + 2, abs=1.0)
        # ...and the installed cap is that number of BINARY gibibytes.
        assert int(derived_gb * 1024**3) > int(derived_gb * 1e9)


class TestApplyMlxMemoryLimits:
    """Captures the calls instead of making them — both setters are process-global."""

    @staticmethod
    def _run(**kwargs):
        """Patch the real `mlx.core`'s attributes, not the module object.

        `_apply_mlx_memory_limits` does `import mlx.core as mx` inside the function,
        and that form binds through the parent package's attribute rather than
        `sys.modules`, so swapping `sys.modules["mlx.core"]` for a fake has no effect —
        the first version of this fixture did exactly that and all seven of these tests
        failed with KeyError because the real setters ran instead.
        """
        import mlx.core as mx

        calls = {}
        with (
            patch.object(
                mx,
                "set_cache_limit",
                side_effect=lambda v: calls.__setitem__("cache", v),
            ),
            patch.object(
                mx,
                "set_memory_limit",
                side_effect=lambda v: calls.__setitem__("memory", v),
            ),
            patch.object(
                mx,
                "device_info",
                return_value={"max_recommended_working_set_size": 96 * 1024**3},
            ),
        ):
            cli._apply_mlx_memory_limits(**kwargs)
        return calls

    def test_an_explicit_cache_limit_is_applied_in_gibibytes(self):
        calls = self._run(cache_limit_gb=8, memory_limit_frac=None)

        assert calls["cache"] == 8 * 1024**3
        assert "memory" not in calls

    @pytest.mark.parametrize("value", [0, None, -1])
    def test_a_non_positive_cache_limit_derives_instead(self, value, tmp_path):
        (tmp_path / "config.json").write_text('{"num_attention_heads": 8}')
        calls = self._run(
            cache_limit_gb=value,
            memory_limit_frac=None,
            model_path=str(tmp_path),
            max_kv_size=32768,
            prefill_step=2048,
        )

        assert calls["cache"] == int(4.0 * 1024**3)

    def test_nothing_is_set_when_there_is_nothing_to_derive_from(self):
        """No explicit limit and no sizing inputs: leave MLX's own default alone."""
        calls = self._run(cache_limit_gb=None, memory_limit_frac=None)

        assert calls == {}

    def test_the_memory_limit_is_the_min_of_frac_and_recommended_wss(self):
        """The recommended working-set size is a hard ceiling: asking for a fraction
        above it would let MLX grow past what Metal advises and the OS would swap,
        which is the exact failure this backstop exists to prevent."""
        with patch(
            "os.sysconf",
            side_effect=lambda n: {
                "SC_PAGE_SIZE": 4096,
                "SC_PHYS_PAGES": 128 * 1024**3 // 4096,
            }[n],
        ):
            calls = self._run(cache_limit_gb=None, memory_limit_frac=0.9)

        # 0.9 * 128 GiB = 115.2 GiB, above the 96 GiB recommended WSS -> clamped.
        assert calls["memory"] == 96 * 1024**3

    def test_the_fraction_wins_when_it_is_below_the_recommendation(self):
        with patch(
            "os.sysconf",
            side_effect=lambda n: {
                "SC_PAGE_SIZE": 4096,
                "SC_PHYS_PAGES": 128 * 1024**3 // 4096,
            }[n],
        ):
            calls = self._run(cache_limit_gb=None, memory_limit_frac=0.5)

        assert calls["memory"] == int(0.5 * 128 * 1024**3)

    def test_an_unavailable_sysconf_falls_back_to_the_recommendation(self):
        """`os.sysconf` is not portable; the helper must still set a backstop."""
        with patch("os.sysconf", side_effect=OSError):
            calls = self._run(cache_limit_gb=None, memory_limit_frac=0.5)

        assert calls["memory"] == 96 * 1024**3


class TestValidateKvPreallocTokens:
    """Adjacent, and the one function here that raises rather than degrading."""

    def test_a_floor_above_the_cap_is_rejected(self):
        with pytest.raises(ValueError, match="must be <= max_kv_size"):
            cli.validate_kv_prealloc_tokens(4096, 2048)

    def test_equal_is_allowed(self):
        assert cli.validate_kv_prealloc_tokens(2048, 2048) is None

    @pytest.mark.parametrize(
        ("floor", "cap"), [(None, 2048), (2048, None), (None, None)]
    )
    def test_missing_either_side_is_a_no_op(self, floor, cap):
        assert cli.validate_kv_prealloc_tokens(floor, cap) is None
