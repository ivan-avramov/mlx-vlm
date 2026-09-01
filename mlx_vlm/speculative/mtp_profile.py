"""Env-gated MTP round/head-loop profiler (M29 H1, 2026-08-31).

MLX is lazy: a phase boundary must ``mx.eval`` that phase's outputs THEN
``mx.synchronize()``, or all compute collapses into whichever call evals
first downstream (typically verify's ``mx.eval(target_tokens)``). The eval
fences that make per-phase timing possible also kill async overlap between
phases -- an accepted observer effect, which is why this only runs behind
``MLX_VLM_MTP_PROFILE`` / ``MLX_VLM_MTP_PROFILE_HEAD`` and is a strict no-op
otherwise (see ``round_profiler_from_env`` / ``head_profiler_from_env``: with
the env unset, callers see ``None`` and skip every profiler call -- no
``mx.synchronize``, no timers, one ``os.environ`` lookup per generation).
"""

import dataclasses
import os
import sys
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

import mlx.core as mx

ENV_ROUNDS = "MLX_VLM_MTP_PROFILE"
ENV_HEAD = "MLX_VLM_MTP_PROFILE_HEAD"


def _collect(value: Any, out: List[mx.array]) -> None:
    if isinstance(value, mx.array):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect(item, out)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _collect(getattr(value, field.name), out)


def cache_state_arrays(prompt_cache: Iterable[Any]) -> List[mx.array]:
    """Collect state arrays out of a prompt cache for the rollback phase mark."""
    out: List[mx.array] = []
    for entry in prompt_cache:
        state = getattr(entry, "state", None)
        if state is None:
            continue
        _collect(state, out)
    return out


class _PhaseTimer:
    """Shared accumulate-and-report machinery for the round/head profilers."""

    every = 200

    def __init__(self, tag: str, phases: List[str], unit_label: str):
        self.tag = tag
        self.phases = phases
        self.unit_label = unit_label
        self.totals: Dict[str, float] = {phase: 0.0 for phase in phases}
        self.counters: Dict[str, float] = {}
        self.units = 0
        self._last = 0.0

    def begin(self) -> None:
        mx.synchronize()
        self._last = perf_counter()

    def mark(self, phase: str, *outputs: Any) -> None:
        arrays: List[mx.array] = []
        for value in outputs:
            _collect(value, arrays)
        if arrays:
            mx.eval(*arrays)
        mx.synchronize()
        now = perf_counter()
        self.totals[phase] += now - self._last
        self._last = now

    def end_unit(self, **counters: float) -> None:
        self.units += 1
        for key, value in counters.items():
            self.counters[key] = self.counters.get(key, 0.0) + value
        if self.units % self.every == 0:
            self.report(final=False)

    def _per_unit_ms(self, phase: str) -> float:
        if self.units == 0:
            return 0.0
        return 1000.0 * self.totals[phase] / self.units

    def report(self, final: bool) -> None:
        """Print ONE stderr line. Never raises -- a profiler bug must not
        take down generation."""
        try:
            line = self._format_line(final)
        except Exception:
            return
        try:
            print(line, file=sys.stderr, flush=True)
        except Exception:
            pass
        try:
            self._after_report(final)
        except Exception:
            pass

    def _format_line(self, final: bool) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def _after_report(self, final: bool) -> None:
        pass


class MTPRoundProfiler(_PhaseTimer):
    """Per-round phase timing for ``_mtp_rounds`` (server single-request path)."""

    def __init__(self) -> None:
        super().__init__(
            "mtp_profile",
            ["draft", "verify", "walk", "rollback", "other"],
            "round",
        )

    def _format_line(self, final: bool) -> str:
        draft = self._per_unit_ms("draft")
        verify = self._per_unit_ms("verify")
        walk = self._per_unit_ms("walk")
        rollback = self._per_unit_ms("rollback")
        other = self._per_unit_ms("other")
        round_ms = draft + verify + walk + rollback + other
        units = self.units
        emitted_rd = self.counters.get("emitted", 0.0) / units if units else 0.0
        accepted_rd = self.counters.get("accepted", 0.0) / units if units else 0.0
        n_draft_rd = self.counters.get("n_draft", 0.0) / units if units else 0.0
        return (
            f"[mtp_profile] rounds={units} draft={draft:.2f} verify={verify:.2f} "
            f"walk={walk:.2f} rollback={rollback:.2f} other={other:.2f} "
            f"emitted/rd={emitted_rd:.2f} accepted/rd={accepted_rd:.2f} "
            f"n_draft/rd={n_draft_rd:.2f} round_ms={round_ms:.2f} "
            f"final={1 if final else 0}"
        )

    def _after_report(self, final: bool) -> None:
        if final and _head_profiler_instance is not None:
            _head_profiler_instance.report(final=True)


class MTPHeadProfiler(_PhaseTimer):
    """Per-draft-token phase timing for ``NemotronHMTPDraftModel.draft_block``."""

    def __init__(self) -> None:
        super().__init__(
            "mtp_profile_head",
            ["proj_layers", "lm_head", "sampler", "eval"],
            "draft_token",
        )

    def _format_line(self, final: bool) -> str:
        proj_layers = self._per_unit_ms("proj_layers")
        lm_head = self._per_unit_ms("lm_head")
        sampler = self._per_unit_ms("sampler")
        eval_ms = self._per_unit_ms("eval")
        return (
            f"[mtp_profile_head] draft_tokens={self.units} "
            f"proj_layers={proj_layers:.2f} lm_head={lm_head:.2f} "
            f"sampler={sampler:.2f} eval={eval_ms:.2f} final={1 if final else 0}"
        )


_head_profiler_instance: Optional[MTPHeadProfiler] = None


def round_profiler_from_env() -> Optional[MTPRoundProfiler]:
    """New profiler per generation, or ``None`` if unset (the hot-path guard)."""
    if os.environ.get(ENV_ROUNDS, "") != "1":
        return None
    return MTPRoundProfiler()


def head_profiler_from_env() -> Optional[MTPHeadProfiler]:
    """Module singleton (``draft_block`` has no end-of-generation signal to
    report from), or ``None`` if unset."""
    global _head_profiler_instance
    if os.environ.get(ENV_HEAD, "") != "1":
        return None
    if _head_profiler_instance is None:
        _head_profiler_instance = MTPHeadProfiler()
    return _head_profiler_instance


def reset_head_profiler() -> None:
    """Test-only: clear the head singleton between test cases."""
    global _head_profiler_instance
    _head_profiler_instance = None
