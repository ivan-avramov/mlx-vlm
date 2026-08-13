"""The GQA tile-reuse decode kernel must be A/B-able at runtime, or its claim is unfalsifiable.

WHY THIS EXISTS. `_decode_2pass_use_legacy` selects the R-redundant legacy pass-1 kernel instead of
the GQA tile-reuse one, "for the apples-to-apples micro-bench baseline" per its own comment. But it
occurred in exactly ONE place in the whole fork — the `getattr` that READS it. Nothing ever set it:
no env var, no constructor argument, no test. It served a one-off micro-bench and then became dead
code.

The consequence, found while re-verifying Phase-2 on 2026-08-13: the recorded win for this kernel
("lossless, ~1.3x over legacy TQ, +2-7% end-to-end") **cannot be re-measured at runtime**. That
matters because the same re-verification pass found APC — another shipped, documented Phase-2 win —
to be completely inert in production, and the only reason that was provable is that APC exposes
`/metrics` counters. This kernel had no such escape hatch.

So the toggle gets the same treatment every other TQ A/B knob already has: an env var following the
`TQ_FUSED_PREFILL` / `TQ_PREFILL_IMPL` idiom, plus a constructor argument, with the tile-reuse path
staying the DEFAULT so shipped behaviour is unchanged.
"""
import mlx.core as mx

from mlx_vlm.turboquant import TurboQuantKVCache


def test_tile_reuse_is_the_default():
    """Shipped behaviour must not change: the fused GQA tile-reuse kernel stays default."""
    cache = TurboQuantKVCache(bits=3, seed=0)
    assert cache._decode_2pass_use_legacy is False


def test_legacy_selectable_via_constructor():
    cache = TurboQuantKVCache(bits=3, seed=0, decode_2pass_use_legacy=True)
    assert cache._decode_2pass_use_legacy is True


def test_legacy_selectable_via_env(monkeypatch):
    """The env var is what makes a same-process A/B possible from a benchmark harness."""
    monkeypatch.setenv("TQ_DECODE_2PASS_LEGACY", "1")
    assert TurboQuantKVCache(bits=3, seed=0)._decode_2pass_use_legacy is True


def test_env_accepts_the_same_truthy_spellings_as_the_other_TQ_flags(monkeypatch):
    for val in ("1", "true", "TRUE", "yes"):
        monkeypatch.setenv("TQ_DECODE_2PASS_LEGACY", val)
        assert TurboQuantKVCache(bits=3, seed=0)._decode_2pass_use_legacy is True, val
    for val in ("0", "false", "no", ""):
        monkeypatch.setenv("TQ_DECODE_2PASS_LEGACY", val)
        assert TurboQuantKVCache(bits=3, seed=0)._decode_2pass_use_legacy is False, val


def test_constructor_WINS_over_env(monkeypatch):
    """An explicit argument must beat ambient environment, or a harness cannot pin the arm it wants
    while an operator has the variable exported — the same precedence bug that made
    MLX_BENCH_RESULTS able to defeat a test's monkeypatch in the bench harness."""
    monkeypatch.setenv("TQ_DECODE_2PASS_LEGACY", "1")
    assert TurboQuantKVCache(bits=3, seed=0,
                             decode_2pass_use_legacy=False)._decode_2pass_use_legacy is False
    monkeypatch.delenv("TQ_DECODE_2PASS_LEGACY", raising=False)
    assert TurboQuantKVCache(bits=3, seed=0,
                             decode_2pass_use_legacy=True)._decode_2pass_use_legacy is True


def test_both_arms_produce_the_same_decode_output():
    """The kernel is claimed LOSSLESS (fp32-exact), so the A/B must differ in SPEED only. If this
    ever fails, the '+2-7% end-to-end for free' claim is not free and the arms are not comparable.

    Note the kernel's benefit is conditional: heads_per_group = 2 only when n_repeats is EVEN, and at
    G=1 tile-reuse degenerates to the legacy read. n_repeats = 4 here (4 q-heads / 1 kv-head).
    """
    def run(legacy):
        cache = TurboQuantKVCache(bits=3, seed=0, decode_2pass_use_legacy=legacy)
        k = mx.random.normal((1, 1, 128, 64), key=mx.random.key(0))
        v = mx.random.normal((1, 1, 128, 64), key=mx.random.key(1))
        ks, vs = cache.update_and_fetch(k, v)
        q = mx.random.normal((1, 4, 1, 64), key=mx.random.key(2))
        return cache.decode_attention(q, cache._unwrap(ks), cache._unwrap(vs))

    a, b = run(False), run(True)
    if a is None or b is None:
        return          # this dim/bit combination has no fused decode path; nothing to compare
    assert mx.abs(a - b).max().item() < 1e-4, "tile-reuse and legacy decode disagree"
