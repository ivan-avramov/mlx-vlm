"""Cheap constructor guards; live lifecycle coverage is in test_turboquant_storage_contract."""

import ast
from pathlib import Path
from types import SimpleNamespace

_SOURCE = Path(__file__).parents[1] / "turboquant.py"


def _constructors():
    tree = ast.parse(_SOURCE.read_text())
    wanted = {"TurboQuantKVCache", "BatchTurboQuantKVCache"}
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in wanted:
            constructor = next(
                member
                for member in node.body
                if isinstance(member, ast.FunctionDef) and member.name == "__init__"
            )
            classes.append(
                ast.ClassDef(
                    name=node.name,
                    bases=[],
                    keywords=[],
                    body=[constructor],
                    decorator_list=[],
                )
            )
    module = ast.Module(body=classes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Optional": __import__("typing").Optional,
        "DEFAULT_TURBOQUANT_SEED": 0,
        "resolve_kv_bits": lambda bits, key, value: (
            bits,
            bits if key is None else key,
            bits if value is None else value,
        ),
        "mx": SimpleNamespace(array=lambda values: list(values)),
    }
    exec(compile(module, str(_SOURCE), "exec"), namespace)
    return namespace["TurboQuantKVCache"], namespace["BatchTurboQuantKVCache"]


def test_concrete_cache_retains_full_cap_constructor_and_attention_options(monkeypatch):
    monkeypatch.delenv("TQ_FUSED_PREFILL", raising=False)
    monkeypatch.delenv("TQ_KV_QUANT_MODE", raising=False)
    single, _ = _constructors()
    cache = single(bits=4, max_kv_size=262144, prealloc_tokens=262144)
    assert cache.max_kv_size == cache.prealloc_tokens == 262144
    assert cache._needs_refloor is False
    assert cache._fused_prefill_enabled is False
    assert cache._kv_quant_mode == "mse"


def test_batch_initializes_shared_attention_flags_without_scalarizing_offsets(
    monkeypatch,
):
    monkeypatch.setenv("TQ_FUSED_PREFILL", "1")
    monkeypatch.setenv("TQ_PREFILL_IMPL", "fused")
    _, batch = _constructors()
    cache = batch([0, 3], bits=4, prealloc_tokens=262144)
    assert cache.prealloc_tokens == 262144
    assert cache.offset == [0, -3]
    assert cache.left_padding == [0, 3]
    assert cache._fused_prefill_enabled is True
    assert cache._prefill_impl == "fused"
    assert cache._fused_attention_eligible is False


def test_cache_classes_do_not_shadow_method_definitions():
    tree = ast.parse(_SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in {
            "_TurboQuantAttentionMixin",
            "TurboQuantKVCache",
            "BatchTurboQuantKVCache",
        }:
            methods = [
                member.name
                for member in node.body
                if isinstance(member, ast.FunctionDef)
                and not any(
                    isinstance(d, ast.Attribute) and d.attr == "setter"
                    for d in member.decorator_list
                )
            ]
            assert len(methods) == len(set(methods)), node.name
