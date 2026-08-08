"""EpiCache integration into qwen3_5 LanguageModel.make_cache (Phase B port).

qwen3_5 is a GatedDeltaNet/full-attention hybrid: layers with
``is_linear=(idx+1)%full_attention_interval != 0`` use a bounded ArraysCache (SSM
state); the periodic full-attention layers use a growing KVCache. EpiCache must wrap
ONLY the full-attention KVCache layers (the linear layers are already state-bounded),
exactly mirroring the gemma4 make_cache integration.

Run (parent fork venv, full deps):
  PYTHONPATH=. \
    ./.venv/bin/python \
    mlx_vlm/tests/test_epicache_qwen3_5.py
"""

import os
import types

from mlx_lm.models.cache import KVCache

from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.epicache import EpiCacheKVCache
from mlx_vlm.models.qwen3_5.language import LanguageModel

_INTERVAL = 4  # full_attention_interval; full-attn at idx 3,7,11,...


def _fake_self(n_layers=12, interval=_INTERVAL):
    # mirror Qwen3_5DecoderLayer.is_linear = (idx+1) % full_attention_interval != 0
    layers = [
        types.SimpleNamespace(is_linear=((i + 1) % interval != 0))
        for i in range(n_layers)
    ]
    return types.SimpleNamespace(layers=layers)


def _is_full_attn(i, interval=_INTERVAL):
    return (i + 1) % interval == 0


def test_make_cache_off_is_behaviour_preserving():
    """Budget unset -> full-attn layers are PLAIN KVCache (no EpiCache wrap)."""
    os.environ.pop("MLX_EPICACHE_BUDGET", None)
    caches = LanguageModel.make_cache(_fake_self())
    for i, c in enumerate(caches):
        if _is_full_attn(i):
            assert isinstance(c, KVCache) and not isinstance(c, EpiCacheKVCache), (
                i,
                type(c),
            )
        else:
            assert isinstance(c, ArraysCache), (i, type(c))


def test_make_cache_wraps_only_full_attn_when_budget_set():
    """Budget set -> full-attn layers wrapped in EpiCacheKVCache; linear layers untouched."""
    os.environ["MLX_EPICACHE_BUDGET"] = "4096"
    os.environ["MLX_EPICACHE_BLOCK"] = "1024"
    try:
        caches = LanguageModel.make_cache(_fake_self())
        n_epi = 0
        for i, c in enumerate(caches):
            if _is_full_attn(i):
                assert isinstance(c, EpiCacheKVCache), (i, type(c))
                assert c.budget == 4096 and c.block_size == 1024
                n_epi += 1
            else:
                # linear layers must NEVER be EpiCache-wrapped (they're SSM state, not KV)
                assert isinstance(c, ArraysCache), (i, type(c))
                assert not isinstance(c, EpiCacheKVCache)
        assert n_epi == 3, n_epi  # 12 layers, interval 4 -> 3 full-attn
    finally:
        os.environ.pop("MLX_EPICACHE_BUDGET", None)
        os.environ.pop("MLX_EPICACHE_BLOCK", None)


def test_make_cache_budget_zero_is_off():
    """Explicit budget=0 behaves like unset (no wrap) — guards the >0 gate."""
    os.environ["MLX_EPICACHE_BUDGET"] = "0"
    try:
        caches = LanguageModel.make_cache(_fake_self())
        for i, c in enumerate(caches):
            if _is_full_attn(i):
                assert not isinstance(c, EpiCacheKVCache), (i, type(c))
    finally:
        os.environ.pop("MLX_EPICACHE_BUDGET", None)


def test_scalar_positions_plain_cache_uses_offset():
    """Non-EpiCache cache: RoPE positions and kv-length both derive from the physical offset
    (behaviour unchanged)."""
    from mlx_vlm.models.qwen3_5.language import _qwen35_scalar_positions

    cache = types.SimpleNamespace(offset=10)  # plain cache, no rope_offset attribute
    pos, kv_delta = _qwen35_scalar_positions(cache, 10, 4)
    assert tuple(pos.shape) == (3, 1, 4), pos.shape  # tiled for MRoPE's 3 sections
    assert pos[0, 0].tolist() == [10, 11, 12, 13], pos[0, 0].tolist()
    assert kv_delta == 11, kv_delta  # cache_offset + 1


def test_scalar_positions_epicache_uses_rope_offset_for_rope_physical_for_mask():
    """EpiCache post-eviction: physical offset shrank to 20 but the true sequence position is
    50. RoPE must use 50 (so kept keys + query share a frame); kv-length/mask must use the
    PHYSICAL 20 (the actual cached KV count)."""
    from mlx_vlm.models.qwen3_5.language import _qwen35_scalar_positions

    cache = types.SimpleNamespace(offset=20, rope_offset=50)
    pos, kv_delta = _qwen35_scalar_positions(cache, 20, 3)
    assert pos[0, 0].tolist() == [50, 51, 52], pos[
        0, 0
    ].tolist()  # RoPE at TRUE position
    assert kv_delta == 21, kv_delta  # mask uses PHYSICAL offset


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nALL {len(tests)} qwen3_5 EpiCache make_cache tests PASS")
