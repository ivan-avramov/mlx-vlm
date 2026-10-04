"""Env-gated chunked-prefill component profiler (M57 step 2, 2026-10-04).

Attributes the per-chunk wall time of the single-sequence chunked prefill loop
(``generate/ar.py``) to components. Diagnostic only: with
``MLX_VLM_PREFILL_PROFILE`` unset, ``from_env()`` returns ``None``, every hook
is a single ``is not None`` check, and nothing evals, synchronizes or times.

Same conventions as ``speculative/mtp_profile.py`` (whose ``_PhaseTimer`` is
reused): a phase mark is ``mx.eval(outputs)`` then ``mx.synchronize()`` then a
timestamp, and reporting never raises. The eval fences kill async overlap
between phases -- an accepted observer effect; compare shares, not absolutes,
when profiled wall is far above unprofiled wall.

Phase boundaries (a mark closes the time since the previous mark):

* ``attn_prep``: input layernorm (when the layer is full attention), q/k/v
  projections, norms, rotary -- everything before the cache update.
* ``kv_update``: ``cache.update_and_fetch`` (and any preallocation inside it).
* ``sdpa``: the ``scaled_dot_product_attention`` call (plus the EpiCache
  ``observe`` hook when that cache is in use).
* ``attn_out``: transpose/reshape, gate, ``o_proj``.
* ``gdn``: input layernorm + the whole ``linear_attn`` call.
* ``mlp``: residual add, post-attention norm, ``mlp`` (both layer kinds).
* ``cache_post`` / ``clear_cache``: chunk-loop work after the model call.
* ``other``: chunk wall minus the sum of the above (embedding slice, final norm,
  lm_head, Python, snapshot handling).

Active-profiler handle: ``active()`` is single-request DIAGNOSTIC state, set by
the chunk loop for the duration of one model call and cleared in ``finally``.
It is ``None`` outside a profiled chunk. It is not thread-safe and must not be
used to key behaviour; layer code only reads it to place marks.
"""

import os
from time import perf_counter
from typing import Any, Dict, List, Optional

from .speculative.mtp_profile import _PhaseTimer

ENV = "MLX_VLM_PREFILL_PROFILE"

# Reported phases, in line order. ``other`` is the chunk-wall residual.
_NAMED = (
    "sdpa",
    "kv_update",
    "attn_prep",
    "attn_out",
    "gdn",
    "mlp",
    "cache_post",
    "clear_cache",
)
# ``other_fence``: a mark that only closes a time span (its time is part of
# ``other``, which is derived as a residual, so it is never reported itself).
_FENCE = "other_fence"

_active: Optional["PrefillProfiler"] = None


def active() -> Optional["PrefillProfiler"]:
    """The profiler for the chunk currently inside the model call, else None."""
    return _active


class PrefillProfiler(_PhaseTimer):
    every = 32

    def __init__(self) -> None:
        super().__init__("prefill_profile", list(_NAMED) + [_FENCE], "chunk")
        # One record per chunk: (wall_s, tokens, keys, {phase: seconds}).
        self._records: List[Any] = []
        self._window_start = 0
        self._chunk_t0 = 0.0
        self._chunk_tokens = 0
        self._chunk_base: Dict[str, float] = {}
        self.broken = False

    # -- chunk lifecycle (called by the chunk loop; never raise) -----------

    def begin_chunk(self, tokens: int) -> None:
        """Start timing a chunk and publish the active handle."""
        global _active
        try:
            self._chunk_tokens = int(tokens)
            self._chunk_base = dict(self.totals)
            self.begin()
            self._chunk_t0 = self._last
        except Exception:
            self.broken = True
        _active = None if self.broken else self

    def clear_active(self) -> None:
        global _active
        if _active is self:
            _active = None

    def safe_mark(self, phase: str, *outputs: Any) -> None:
        """``mark`` that cannot take generation down: on any error the profiler
        disables itself for the rest of the generation."""
        if self.broken:
            return
        try:
            self.mark(phase, *outputs)
        except Exception:
            self.broken = True
            self.clear_active()

    def end_chunk(self, keys: int) -> None:
        if self.broken:
            return
        try:
            wall = perf_counter() - self._chunk_t0
            phases = {p: self.totals[p] - self._chunk_base[p] for p in _NAMED}
            self._records.append((wall, self._chunk_tokens, int(keys), phases))
            self.units += 1
            if self.units % self.every == 0:
                self.report(final=False)
                self._window_start = len(self._records)
        except Exception:
            self.broken = True

    def finish(self) -> None:
        """Final line, covering every chunk of the generation."""
        if self.broken and not self._records:
            return
        self.clear_active()
        self.report(final=True)

    # -- reporting ---------------------------------------------------------

    def _format_line(self, final: bool) -> str:
        records = self._records if final else self._records[self._window_start :]
        n = len(records)
        if n == 0:
            raise ValueError("no chunks")
        wall = sum(r[0] for r in records)
        named = {p: sum(r[3][p] for r in records) for p in _NAMED}
        other = max(0.0, wall - sum(named.values()))

        def ms(seconds: float) -> str:
            return f"{1000.0 * seconds / n:.2f}"

        return (
            f"[prefill_profile] chunks={n} tokens={sum(r[1] for r in records)} "
            f"keys={records[-1][2]} wall={ms(wall)} sdpa={ms(named['sdpa'])} "
            f"kv_update={ms(named['kv_update'])} "
            f"attn_prep={ms(named['attn_prep'])} "
            f"attn_out={ms(named['attn_out'])} gdn={ms(named['gdn'])} "
            f"mlp={ms(named['mlp'])} cache_post={ms(named['cache_post'])} "
            f"clear_cache={ms(named['clear_cache'])} other={ms(other)} "
            f"final={1 if final else 0}"
        )


def from_env() -> Optional[PrefillProfiler]:
    """New profiler per generation, or ``None`` if unset (the hot-path guard)."""
    if os.environ.get(ENV, "") != "1":
        return None
    return PrefillProfiler()
