"""Layer-scoped expert-budget expansion (M34): inference-only, training-free MoE
routing change shared by `qwen3_5_moe` and `nemotron_h`.

Source: Agrillo, "Layer-Scoped Expert-Budget Expansion Discovers Succinct
Convergence in Sparse MoE Reasoning" (preprint, 2026-09). Full definition:
`docs/specs/m34-moe-expert-expansion.md` in the mlx_local_stack repo.

Per token at a MoE layer with router probabilities/scores `p[e]` over `E`
experts and a native top-`K`:

1. Rank experts by `p` descending; ties broken by lowest expert id.
2. Ranks are 1-based `j = 1..E`, but only the top `N` ranks are ever
   candidates. `R = max(floor(N/2), 1)`; `pref = p` at rank `R`.
3. Ranks `1..K` are ALWAYS kept (the native top-K is never pruned -- this is
   what makes `N == K` byte-identical to the native path, a hard requirement
   of the spec, item 7). `T == 0`: additionally keep every rank up to `N`.
   `T > 0`: for ranks beyond `max(K, floor(N/4))`, keep rank `j` iff
   `p_j >= T * pref`. This is the closest faithful reading of the spec's
   literal "keep rank j iff j <= floor(N/4) or p_j >= T*pref" that is also
   consistent with its own byte-identical-at-N==K requirement: read literally
   (with no floor above K), a peaked distribution with T close to 1 could
   prune a rank inside the native top-K, which native never does. Folding the
   native top-K into the "always kept" floor (`max(K, floor(N/4))` instead of
   bare `floor(N/4)`) removes that contradiction while leaving the rule
   identical to the literal spec whenever `floor(N/4) >= K` (the common case).
4. Decay for extra ranks `j = K+1..N`:
   `factor(j) = 0.99 - (0.99 - D) * (j - (K+1)) / (N - (K+1))`;
   when `N == K+1`, `factor = (0.99 + D) / 2`. Ranks `<= K`: factor 1.
5. Weights `w_j = p_j * factor(j) * kept_j`, renormalized over kept experts.

Tie-break determinism: `_composite_sort_key` builds an INTEGER composite key
from each expert's IEEE-754 bit pattern (sign-aware monotonic embedding of
float order into integers -- the standard "float radix sort" trick) combined
with the expert id in the low digits. This is a composite KEY, built with
exact integer arithmetic, not a floating-point epsilon perturbation of `p` --
it cannot reorder two distinct `p` values no matter how close they are, and it
does not rely on `mx.argsort`'s tie-handling being stable.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx

_U32_SIGN_BIT = 0x80000000
_U32_ALL_BITS = 0xFFFFFFFF


def _composite_sort_key(p: mx.array) -> mx.array:
    """Integer key: ascending key order == (p descending, expert id ascending).

    Built purely from bit patterns and integer arithmetic (no float addition),
    so equal keys can only arise from bit-identical `p` values at the same id,
    which cannot happen for distinct ids. See module docstring.
    """
    e = p.shape[-1]
    bits = p.astype(mx.float32).view(mx.uint32).astype(mx.int64)
    sign = (bits >> 31) & 1
    # IEEE-754 monotonic embedding: negative floats need every bit flipped;
    # non-negative floats need only the sign bit set. This maps the full
    # float ordering onto integer ordering, for both signs.
    monotonic = mx.where(
        sign.astype(mx.bool_),
        bits ^ mx.array(_U32_ALL_BITS, dtype=mx.int64),
        bits | mx.array(_U32_SIGN_BIT, dtype=mx.int64),
    )
    ids = mx.arange(e, dtype=mx.int64)
    # Descending p -> ascending key: negate. `ids` in the low digits breaks
    # ties ascending by id; E is tiny relative to the int64 range so this can
    # never cross into a neighbouring p-bucket.
    return -(monotonic) * mx.array(e + 1, dtype=mx.int64) + ids


def decay_factor(j: int, k: int, n: int, d: float) -> float:
    """Decay factor for 1-based rank `j`, given native top-`k` and budget `n`.

    Ranks `<= k` are unweighted (factor 1). `Appendix A` check (N=20, K=8,
    D=0.5): ranks 9/12/15/18/20 -> 0.990/0.856/0.723/0.589/0.500. `N == K+1`
    fallback (D=0.5): 0.745.
    """
    if j <= k:
        return 1.0
    if n == k + 1:
        return (0.99 + d) / 2
    return 0.99 - (0.99 - d) * (j - (k + 1)) / (n - (k + 1))


def _rank_and_factor(
    p: mx.array, k: int, n: int, t: float, d: float
) -> Tuple[mx.array, mx.array, mx.array]:
    """Shared rank/kept/factor machinery for `expand_route` and
    `expand_route_with_weight_base`. Returns `(inds, factor, kept)`, each
    shape `(..., n)`, aligned rank-for-rank (rank 1 first)."""
    key = _composite_sort_key(p)
    order = mx.argsort(key, axis=-1)
    inds = order[..., :n]
    ranked_p = mx.take_along_axis(p, inds, axis=-1)

    r = max(n // 2, 1)
    pref = ranked_p[..., r - 1 : r]

    ranks = mx.arange(1, n + 1)
    if t == 0.0:
        kept = mx.ones(ranked_p.shape, dtype=mx.bool_)
    else:
        floor_guard = max(n // 4, k)
        kept = (ranks <= floor_guard) | (ranked_p >= t * pref)

    factors = [1.0] * min(k, n) + [
        decay_factor(j, k, n, d) for j in range(k + 1, n + 1)
    ]
    factor = mx.array(factors[:n], dtype=mx.float32)

    return inds, factor, kept


def _normalized_weights(
    weight_base: mx.array, factor: mx.array, kept: mx.array, normalize: bool
) -> mx.array:
    raw = weight_base * factor * kept.astype(weight_base.dtype)
    if not normalize:
        return raw
    denom = raw.sum(axis=-1, keepdims=True)
    return raw / denom


def expand_route(
    p: mx.array, k: int, n: int, t: float, d: float
) -> Tuple[mx.array, mx.array]:
    """Expanded routing over a single probability/score array `p`.

    `p` is used both to rank/select experts AND as the weight numerator
    (matches qwen3_5_moe: softmax gates serve both roles, so this is the
    native top-K rule extended). Returns `(inds, weights)`, both shape
    `(..., n)`; pruned ranks carry weight 0 (kept shape, per the static-kernel
    requirement). When `n == k`, `weights` reduces exactly to the native
    top-K renormalization (`p / p.sum()` over the kept set).
    """
    inds, factor, kept = _rank_and_factor(p, k, n, t, d)
    ranked_p = mx.take_along_axis(p, inds, axis=-1)
    weights = _normalized_weights(ranked_p, factor, kept, normalize=True)
    return inds, weights


def expand_route_with_weight_base(
    rank_p: mx.array,
    weight_p: mx.array,
    k: int,
    n: int,
    t: float,
    d: float,
    normalize: bool = True,
) -> Tuple[mx.array, mx.array]:
    """Like `expand_route`, but ranking and weighting use DIFFERENT arrays.

    nemotron_h selects on the bias-corrected, group-masked sigmoid score
    (`rank_p`) but weighs by the plain sigmoid score (`weight_p`) at the same
    indices -- native's `orig_scores` vs `scores`. `normalize` mirrors
    native's `norm_topk_prob` gate (renormalization is skipped, not just a
    no-op division, when the caller says so).
    """
    inds, factor, kept = _rank_and_factor(rank_p, k, n, t, d)
    weight_base = mx.take_along_axis(weight_p, inds, axis=-1)
    weights = _normalized_weights(weight_base, factor, kept, normalize=normalize)
    return inds, weights


def active_expert_count(weights: mx.array) -> mx.array:
    """Experts-per-token instrumentation: count of nonzero weights per token."""
    return (weights != 0).sum(axis=-1)


@dataclass(frozen=True)
class MoeExpansion:
    """Parsed `--moe-expand` config: `layers` is an inclusive, ABSOLUTE,
    0-based `(Ls, Le)` decoder-layer range; `n` the expanded expert budget;
    `t` the keep-threshold in `[0, 1]`; `d` the decay floor in `(0, 1]`."""

    layers: Tuple[int, int]
    n: int
    t: float
    d: float

    def in_range(self, layer_idx: int) -> bool:
        return self.layers[0] <= layer_idx <= self.layers[1]


def parse_moe_expand(spec: str) -> MoeExpansion:
    """Parse the CLI string `LS-LE:N:T:D` (e.g. `27-39:20:0.8:0.5`)."""
    try:
        layers_part, n_part, t_part, d_part = spec.split(":")
        ls_part, le_part = layers_part.split("-")
        ls, le = int(ls_part), int(le_part)
        n = int(n_part)
        t = float(t_part)
        d = float(d_part)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"malformed --moe-expand value {spec!r}: want 'LS-LE:N:T:D'"
        ) from exc

    if ls < 0 or le < ls:
        raise ValueError(
            f"malformed --moe-expand layer range in {spec!r}: LS<=LE, LS>=0"
        )
    if n < 1:
        raise ValueError(f"malformed --moe-expand N in {spec!r}: N must be >= 1")
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"malformed --moe-expand T in {spec!r}: T must be in [0, 1]")
    if not (0.0 < d <= 1.0):
        raise ValueError(f"malformed --moe-expand D in {spec!r}: D must be in (0, 1]")

    return MoeExpansion(layers=(ls, le), n=n, t=t, d=d)


def format_moe_expand(exp: MoeExpansion) -> str:
    """Inverse of `parse_moe_expand` (round-trips canonical strings)."""
    ls, le = exp.layers
    return f"{ls}-{le}:{exp.n}:{exp.t}:{exp.d}"


def apply_moe_expansion(model, spec: str) -> int:
    """Parse `spec` and apply it to `model`'s language model ONLY.

    A speculative MTP drafter is a SEPARATE `nn.Module` bound to the target
    via `drafter.bind(target_model)` (see e.g.
    `speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py`) -- it is never an
    attribute of the target model, so this can never reach it. Returns the
    number of MoE layers whose absolute layer index falls inside `spec`'s
    range (0 if the model has no MoE blocks in range, or none at all).
    Raises `ValueError` if the model has no `set_moe_expansion` (not an
    MoE-capable language model).
    """
    exp = parse_moe_expand(spec)
    language_model = getattr(model, "language_model", model)
    set_fn = getattr(language_model, "set_moe_expansion", None)
    if set_fn is None:
        raise ValueError(
            "--moe-expand is not supported by model type "
            f"{type(language_model).__name__!r} (no MoE blocks)"
        )
    return set_fn(exp)
