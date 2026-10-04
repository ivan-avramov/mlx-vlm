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

* ``other_fence`` entry fence: this chunk's input embeddings, position/mask
  setup (charged to ``other``, not to layer 0).
* ``attn_prep``: input layernorm (when the layer is full attention), q/k/v
  projections, norms, rotary, mask -- everything before the cache update.
* ``kv_update``: ``cache.update_and_fetch`` (and any preallocation inside it).
* ``sdpa``: the ``scaled_dot_product_attention`` call (plus the EpiCache
  ``observe`` hook when that cache is in use).
* ``attn_out``: transpose/reshape, gate, ``o_proj``.
* ``gdn``: input layernorm + the whole ``linear_attn`` call, closed on the layer
  output AND the recurrent/conv state it stores in its cache entry.
* ``mlp``: residual add, post-attention norm, ``mlp`` (both layer kinds).
* ``cache_post`` / ``clear_cache``: chunk-loop work after the model call;
  ``cache_post`` closes on the cache state AFTER the eviction hook.
* ``other``: chunk wall minus the sum of the above (embedding slice, final norm,
  lm_head, Python, snapshot handling).

Only work production prefill does is evaluated: the chunk output (logits), the
terminal decoder layer's attention output / o_proj / MLP, the final norm and the
head are dead in production (the loop evaluates cache state only) and are never
passed to a mark.

Active-profiler handle: ``active()`` is request-local DIAGNOSTIC state, kept
per thread, set by the chunk loop for the duration of one model call and
cleared in ``finally``; it is owned by the profiler that published it. It is
``None`` outside a profiled chunk and in every other thread. Layer hooks are
valid only inside ``Qwen3_5DecoderLayer`` (``active_layer()``).
"""

import contextlib
import os
import sys
import threading
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
# Work that production prefill never evaluates when it belongs to the TERMINAL
# decoder layer (production evaluates cache state only, so the terminal layer's
# attention output, o_proj and MLP are dead). Marks for them evaluate nothing.
_DEAD_WHEN_TERMINAL = ("sdpa", "attn_out", "mlp")

_tls = threading.local()


def active() -> Optional["PrefillProfiler"]:
    """The profiler for the chunk currently inside the model call on THIS
    thread, else None (thread-local; another thread, or this thread outside a
    profiled chunk, sees None)."""
    return getattr(_tls, "active", None)


def active_layer() -> Optional["PrefillProfiler"]:
    """Like ``active()`` but only while a ``Qwen3_5DecoderLayer`` has declared
    itself via ``enter_layer``; attention marks use this so a model that reuses
    ``Qwen3_5Attention`` inside another layer class reports chunk phases only."""
    prof = getattr(_tls, "active", None)
    if prof is not None and prof.layer_open:
        return prof
    return None


class PrefillProfiler(_PhaseTimer):
    every = 32

    def __init__(self) -> None:
        super().__init__("prefill_profile", list(_NAMED) + [_FENCE], "chunk")
        # One record per COMPLETE chunk: (wall_s, tokens, keys, {phase: seconds}).
        self._records: List[Any] = []
        self._window_start = 0
        self._chunk_t0 = 0.0
        self._chunk_tokens = 0
        self._chunk_base: Dict[str, float] = {}
        self.broken = False
        self.layer_open = False
        self.terminal = False

    # -- chunk lifecycle (called by the chunk loop; never raise) -----------

    def begin_chunk(self, tokens: int) -> None:
        """Start timing a chunk and publish the thread-local active handle."""
        try:
            self._chunk_tokens = int(tokens)
            self._chunk_base = dict(self.totals)
            self.layer_open = False
            self.begin()
            self._chunk_t0 = self._last
        except Exception:
            self.broken = True
        _tls.active = None if self.broken else self

    def clear_active(self) -> None:
        """Retract the handle (only if this profiler owns it)."""
        self.layer_open = False
        if getattr(_tls, "active", None) is self:
            _tls.active = None

    def enter_layer(self, terminal: bool) -> None:
        self.layer_open = True
        self.terminal = bool(terminal)

    def exit_layer(self) -> None:
        self.layer_open = False

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

    def layer_mark(self, phase: str, *outputs: Any, live_if_nonterminal=()) -> None:
        """Phase mark from layer code. In the terminal decoder layer the dead
        phases evaluate nothing (time goes to ``other``) and
        ``live_if_nonterminal`` arrays are left unevaluated."""
        if self.terminal:
            if phase in _DEAD_WHEN_TERMINAL:
                self.safe_mark(_FENCE)
                return
        else:
            outputs = outputs + tuple(live_if_nonterminal)
        self.safe_mark(phase, *outputs)

    def end_chunk(self, keys: int) -> None:
        if self.broken:
            return
        try:
            wall = perf_counter() - self._chunk_t0
            phases = {p: self.totals[p] - self._chunk_base[p] for p in _NAMED}
            self._records.append((wall, self._chunk_tokens, int(keys), phases))
            self.units += 1
            if self.units % self.every == 0:
                self._print(self._records[self._window_start :], "", final=False)
                self._window_start = len(self._records)
        except Exception:
            self.broken = True

    def finish(self, aborted: bool = False) -> None:
        """Last line (chunks since the previous window line; omitted if none)
        plus the generation total line. A disabled or aborted profiler marks
        both ``broken=1`` and never ``final=1``."""
        self.clear_active()
        if not self._records:
            return
        partial = self.broken or aborted
        remaining = self._records[self._window_start :]
        if remaining:
            self._print(remaining, "", final=not partial, broken=partial)
        self._print(self._records, "_total", final=not partial, broken=partial)

    # -- reporting ---------------------------------------------------------

    def _print(self, records, suffix: str, final: bool, broken: bool = False) -> None:
        """Print ONE stderr line. Never raises."""
        try:
            line = self._format_line(records, suffix, final, broken)
        except Exception:
            return
        try:
            print(line, file=sys.stderr, flush=True)
        except Exception:
            pass

    def _format_line(self, records, suffix: str, final: bool, broken: bool) -> str:
        n = len(records)
        if n == 0:
            raise ValueError("no chunks")
        wall = sum(r[0] for r in records)
        named = {p: sum(r[3][p] for r in records) for p in _NAMED}
        other = max(0.0, wall - sum(named.values()))

        def ms(seconds: float) -> str:
            return f"{1000.0 * seconds / n:.2f}"

        return (
            f"[prefill_profile{suffix}] chunks={n} "
            f"tokens={sum(r[1] for r in records)} "
            f"keys={records[-1][2]} wall={ms(wall)} sdpa={ms(named['sdpa'])} "
            f"kv_update={ms(named['kv_update'])} "
            f"attn_prep={ms(named['attn_prep'])} "
            f"attn_out={ms(named['attn_out'])} gdn={ms(named['gdn'])} "
            f"mlp={ms(named['mlp'])} cache_post={ms(named['cache_post'])} "
            f"clear_cache={ms(named['clear_cache'])} other={ms(other)} "
            f"final={1 if final else 0}" + (" broken=1" if broken else "")
        )


def from_env() -> Optional[PrefillProfiler]:
    """New profiler per generation, or ``None`` if unset (the hot-path guard)."""
    if os.environ.get(ENV, "") != "1":
        return None
    return PrefillProfiler()


@contextlib.contextmanager
def _finalizing(prof: PrefillProfiler):
    aborted = True
    try:
        yield prof
        aborted = False
    finally:
        # Never mask the original exception: finish() swallows its own errors.
        try:
            prof.finish(aborted=aborted)
        except Exception:
            pass


def finalizing(prof: Optional[PrefillProfiler]):
    """Context manager for the chunk loop: flushes the report on any exit."""
    if prof is None:
        return contextlib.nullcontext()
    return _finalizing(prof)
