"""P405 F3: the KV codec a worker must serve, derived from its argv, and the exact per-layer leaf manifest (fork-only, CPU).

P405 served affine 4-bit KV where the argv asked TurboQuant, and the harness saw it only as a leaf-class count. The worker now
asserts every resolved ``ResponseGenerator`` KV field against the argv, and the smoke gate requires the live cache to match,
layer path by layer path, the manifest the argv implies under the served path's policy (``maybe_quantize_kv_cache``: the
TurboQuant branch keeps the last top-level cache entry native when there are more than two; the uniform branch converts
every KV layer).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import kvmanifest  # noqa: E402

from mlx_vlm.models.cache import (  # noqa: E402
    ArraysCache,
    CacheList,
    KVCache,
    PreallocKVCache,
    PreallocQuantizedKVCache,
)
from mlx_vlm.turboquant import TurboQuantKVCache  # noqa: E402

TQ_ARGV = ["--model", "m", "--kv-bits", "4", "--kv-quant-scheme", "turboquant"]
TQ_ARGV += ["--quantized-kv-start", "0"]


def test_kv_from_argv_applies_the_cli_defaults():
    assert kvmanifest.kv_from_argv(TQ_ARGV) == dict(
        kv_bits=4.0,
        kv_key_bits=None,
        kv_value_bits=None,
        kv_key_scheme=None,
        kv_value_scheme=None,
        kv_quant_scheme="turboquant",
        kv_group_size=64,
        quantized_kv_start=0,
        tq_kv_quant_mode="mse",
    )
    native = kvmanifest.kv_from_argv(["--model", "m", "--kv-bits", "0"])
    assert native["kv_bits"] is None and native["kv_quant_scheme"] == "uniform"
    assert native["quantized_kv_start"] == 5000  # the CLI default
    prod = kvmanifest.kv_from_argv(TQ_ARGV + ["--kv-quant-mode", "prod"])
    assert prod["tq_kv_quant_mode"] == "prod"


def _rg(**changes):
    fields = dict(
        kv_bits=4.0,
        kv_key_bits=None,
        kv_value_bits=None,
        kv_key_scheme=None,
        kv_value_scheme=None,
        kv_quant_scheme="turboquant",
        kv_group_size=64,
        quantized_kv_start=0,
    )
    fields.update(changes)
    return SimpleNamespace(**fields)


def test_served_kv_reads_every_resolved_generator_field_and_the_mode():
    got = kvmanifest.served_kv(_rg(), {"TQ_KV_QUANT_MODE": "mse"})
    assert got == kvmanifest.kv_from_argv(TQ_ARGV)


@pytest.mark.parametrize(
    "changes",
    [
        dict(kv_quant_scheme="uniform"),  # the P405 shape
        dict(kv_bits=None),
        dict(kv_key_bits=3.0),
        dict(kv_value_bits=3.0),
        dict(kv_key_scheme="uniform"),
        dict(kv_value_scheme="turboquant"),
        dict(kv_group_size=32),
        dict(quantized_kv_start=5000),
    ],
)
def test_kv_problems_names_each_field_the_server_got_wrong(changes):
    served = kvmanifest.served_kv(_rg(**changes), {"TQ_KV_QUANT_MODE": "mse"})
    problems = kvmanifest.kv_problems(served, kvmanifest.kv_from_argv(TQ_ARGV))
    assert [p.split(" ")[0] for p in problems] == list(changes)


def test_kv_problems_names_a_wrong_mode():
    served = kvmanifest.served_kv(_rg(), {"TQ_KV_QUANT_MODE": "prod"})
    problems = kvmanifest.kv_problems(served, kvmanifest.kv_from_argv(TQ_ARGV))
    assert [p.split(" ")[0] for p in problems] == ["tq_kv_quant_mode"]


# ------------------------------------------------------------------ the per-layer manifest
def _template(n_blocks=4):
    """A hybrid stack like qwen3_5: three linear layers then one full-attention layer, per block."""
    layers = []
    for _ in range(n_blocks):
        layers += [ArraysCache(size=2), ArraysCache(size=2), ArraysCache(size=2)]
        layers.append(KVCache())
    return layers


def _served_tq(template, mode="mse", key_bits=None, last_native=True):
    out = []
    last = len(template) - 1
    for i, c in enumerate(template):
        if isinstance(c, KVCache) and not (last_native and i == last):
            out.append(TurboQuantKVCache(bits=4, kv_quant_mode=mode, key_bits=key_bits))
        elif isinstance(c, KVCache):
            out.append(PreallocKVCache())
        else:
            out.append(c)
    return out


def _served_affine(template):
    return [
        PreallocQuantizedKVCache(group_size=64, bits=4) if isinstance(c, KVCache) else c
        for c in template
    ]


def _problems(served, template, argv=TQ_ARGV):
    return kvmanifest.manifest_problems(
        kvmanifest.manifest(served),
        kvmanifest.manifest(template),
        kvmanifest.kv_from_argv(argv),
    )


def test_the_manifest_is_ordered_by_layer_path_with_codec_details():
    nested = [ArraysCache(size=2), KVCache(), CacheList(KVCache(), ArraysCache(size=1))]
    got = kvmanifest.manifest(nested)
    assert [(m["path"], m["kind"]) for m in got] == [
        ("0", "ArraysCache"),
        ("1", "native"),
        ("2.0", "native"),
        ("2.1", "ArraysCache"),
    ]
    tq = kvmanifest.manifest([TurboQuantKVCache(bits=4)])[0]
    assert tq == dict(
        path="0",
        kind="tq",
        cls="TurboQuantKVCache",
        bits=4.0,
        key_bits=4.0,
        value_bits=4.0,
        mode="mse",
    )
    affine = kvmanifest.manifest([PreallocQuantizedKVCache(group_size=64, bits=4)])[0]
    assert affine == dict(
        path="0", kind="affine", cls="PreallocQuantizedKVCache", bits=4, group_size=64
    )


def test_a_correct_turboquant_cache_has_no_problems():
    template = _template()
    assert _problems(_served_tq(template), template) == []


def test_a_correct_affine_cache_has_no_problems_under_a_uniform_argv():
    template = _template()
    uniform = ["--model", "m", "--kv-bits", "4", "--quantized-kv-start", "0"]
    assert _problems(_served_affine(template), template, uniform) == []


@pytest.mark.parametrize(
    "case",
    [
        "affine_under_tq",
        "mixed",
        "missing",
        "extra",
        "key_width",
        "mode",
        "exempt_quantized",
        "exempt_moved",
    ],
)
def test_the_manifest_check_refuses_by_layer_path(case):
    template = _template()
    served = _served_tq(template)
    if case == "affine_under_tq":  # the P405 shape
        served = _served_affine(template)
    elif case == "mixed":
        served[3] = PreallocQuantizedKVCache(group_size=64, bits=4)
    elif case == "missing":
        served = served[:-1]
    elif case == "extra":
        served = served + [KVCache()]
    elif case == "key_width":
        served = _served_tq(template, key_bits=3)
    elif case == "mode":
        served = _served_tq(template, mode="prod")
    elif case == "exempt_quantized":
        served = _served_tq(template, last_native=False)
    elif case == "exempt_moved":
        served[3], served[15] = PreallocKVCache(), TurboQuantKVCache(bits=4)
    problems = _problems(served, template)
    assert problems, case
    assert all(
        p.startswith("layer ") or p.startswith("layers ") for p in problems
    ), problems


def test_a_non_zero_quantized_kv_start_is_refused_not_guessed():
    template = _template()
    argv = [
        "--model",
        "m",
        "--kv-bits",
        "4",
        "--kv-quant-scheme",
        "turboquant",
    ]  # start: CLI default 5000
    with pytest.raises(ValueError, match="quantized_kv_start"):
        _problems(_served_tq(template), template, argv)


def test_a_native_argv_wants_every_kv_layer_native():
    template = _template()
    native = ["--model", "m", "--kv-bits", "0"]
    served = [PreallocKVCache() if isinstance(c, KVCache) else c for c in template]
    assert _problems(served, template, native) == []
    assert _problems(_served_tq(template), template, native)
