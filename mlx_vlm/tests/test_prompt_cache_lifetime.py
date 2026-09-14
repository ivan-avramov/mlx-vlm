"""CPU-only ownership checks for the full production stream_generate body.

Compile the unchanged definitions with inert boundary dependencies so this suite
can run alongside a capacity probe without importing MLX or touching the GPU.
Weak references check payload lifetime at replacement allocation, not merely
whether a state field was cleared.
"""

import ast
import logging
import time
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


class _Payload:
    pass


class _Layer:
    def __init__(self):
        self.payload = _Payload()
        self.offset = 4


class _Ids:
    def __init__(self, values):
        self.values = list(values)
        self.size = len(values)
        self.shape = (1, self.size)

    def flatten(self):
        return self

    def tolist(self):
        return list(self.values)

    def __getitem__(self, key):
        return _Ids(self.values[key[1]])


class _Ring:
    enabled = True

    def __init__(self):
        self.snapshots = []
        self.captured = []

    def __len__(self):
        return len(self.snapshots)

    def clear(self):
        self.snapshots.clear()

    def find_nearest(self, offset):
        return next((s for s in reversed(self.snapshots) if s.offset <= offset), None)

    def drop_after(self, offset):
        before = len(self.snapshots)
        self.snapshots[:] = [s for s in self.snapshots if s.offset <= offset]
        return before - len(self.snapshots)

    def capture(self, *, offset, cache):
        self.captured.append(offset)


def _definition(module, name, namespace):
    path = Path(__file__).parents[1] / "generate" / module
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    body = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(body)
    exec(compile(body, str(path), "exec"), namespace)
    return namespace[name]


def _harness(
    *, ids=(1, 2, 9), hybrid=False, rotating_safe=True, media_safe=True, rope_safe=True
):
    namespace = {
        "logger": logging.getLogger(__name__),
        "time": time,
        "DEFAULT_THINKING_END_TOKEN": "</think>",
        "DEFAULT_THINKING_START_TOKEN": "<think>",
        "_SESSION_SHRINK_ON_RETIRE": False,
        "_prepare_generation_inputs": lambda *a: (_Ids(ids), None, None, None),
        "_rotating_rewind_safe": lambda *a: rotating_safe,
        "_has_non_trimmable": lambda *a: hybrid,
        "_restore_deltanet_state": lambda *a: None,
        "_prime_cached_prefix_rope_state": lambda *a: rope_safe,
        "_trim_cache": lambda c, offset: setattr(c, "offset", offset),
        "_get_generation_stream": lambda: None,
        "wired_limit": lambda *a: nullcontext(),
        "is_diffusion_model": lambda *a: False,
        "GenerationResult": SimpleNamespace,
        "mx": SimpleNamespace(clear_cache=lambda: None, get_peak_memory=lambda: 0),
        "make_streaming_detokenizer": lambda *a: SimpleNamespace(
            last_segment="x", add_token=lambda *a, **k: None, finalize=lambda: None
        ),
        "_apc": SimpleNamespace(
            multimodal_token_ids_from_config=lambda *a: [],
            media_safe_prefix_min=lambda *a: 0,
            prefix_leaves_text_only_suffix=lambda *a: media_safe,
            prefix_contains_media_tokens=lambda *a: False,
        ),
    }
    state_class = _definition("common.py", "PromptCacheState", namespace)
    state = state_class(snapshot_ring=_Ring())
    state.cache = [_Layer()]
    state.token_ids = [1, 2, 3, 4]
    old_payload = weakref.ref(state.cache[0].payload)
    calls = []

    def allocate(*args, **kwargs):
        assert (
            old_payload() is None
        ), "retired full-cap payload survived replacement allocation"
        assert state.cache is None and state.token_ids is None
        assert not state.snapshot_ring.snapshots
        calls.append("allocate")
        return [_Layer()]

    def generate_step(*args, **kwargs):
        assert kwargs["kv_prealloc_tokens"] == 262144
        assert kwargs["max_kv_size"] == 262144
        calls.append(("generate", args[0].tolist(), id(kwargs["prompt_cache"])))
        yield 7, None

    namespace["cache"] = SimpleNamespace(make_prompt_cache=allocate)
    namespace["generate_step"] = generate_step
    stream = _definition("dispatch.py", "stream_generate", namespace)
    model = SimpleNamespace(language_model=object(), config=SimpleNamespace())
    processor = SimpleNamespace(stopping_criteria=lambda token: False)

    def run(**kwargs):
        return stream(
            model,
            processor,
            "prompt",
            prompt_cache_state=state,
            kv_prealloc_tokens=262144,
            max_kv_size=262144,
            **kwargs,
        )

    return SimpleNamespace(
        state=state, old_payload=old_payload, namespace=namespace, calls=calls, run=run
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"ids": (9, 8, 7)},
        {"ids": (1, 2, 3, 4)},
        {"ids": (1, 2)},
        {"hybrid": True},
        {"rotating_safe": False},
        {"media_safe": False},
        {"rope_safe": False},
    ],
    ids=[
        "no-prefix",
        "identical-prompt",
        "shorter-prompt",
        "hybrid-no-snapshot",
        "rotating-unsafe",
        "media-unsafe",
        "rope-unsafe",
    ],
)
def test_declined_reuse_retires_payload_before_replacement(settings):
    h = _harness(**settings)
    list(h.run())
    assert h.calls[0] == "allocate"
    assert h.state.cache is not None
    assert h.state.token_ids == list(settings.get("ids", (1, 2, 9))) + [7]


def test_selected_snapshot_is_not_retained_by_generator_after_reuse_declined():
    h = _harness(hybrid=True, rope_safe=False)
    h.state.snapshot_ring.snapshots.append(SimpleNamespace(offset=1, states=_Payload()))
    snapshot_payload = weakref.ref(h.state.snapshot_ring.snapshots[0].states)
    allocate = h.namespace["cache"].make_prompt_cache

    def check_snapshot(*args, **kwargs):
        assert snapshot_payload() is None, "local snapshot outlived cache retirement"
        return allocate(*args, **kwargs)

    h.namespace["cache"].make_prompt_cache = check_snapshot
    list(h.run())


@pytest.mark.parametrize("hybrid", [False, True])
def test_valid_reuse_keeps_cache_identity_and_snapshot_rewind(hybrid):
    h = _harness(hybrid=hybrid)
    original_cache_id = id(h.state.cache)
    if hybrid:
        h.state.snapshot_ring.snapshots.append(
            SimpleNamespace(offset=1, states=_Payload())
        )
    list(h.run())
    assert h.old_payload() is not None
    assert id(h.state.cache) == original_cache_id
    assert h.calls == [("generate", [2, 9] if hybrid else [9], original_cache_id)]
    assert h.state.cache[0].offset == (1 if hybrid else 2)


def test_replacement_failure_leaves_session_empty_and_retryable():
    h = _harness(ids=(9, 8, 7))
    allocate = h.namespace["cache"].make_prompt_cache

    def fail(*args, **kwargs):
        allocate(*args, **kwargs)
        raise RuntimeError("replacement failed")

    h.namespace["cache"].make_prompt_cache = fail
    with pytest.raises(RuntimeError, match="replacement failed"):
        list(h.run())
    assert h.old_payload() is None
    assert h.state.cache is None and h.state.token_ids is None
    h.namespace["cache"].make_prompt_cache = allocate
    list(h.run())
    assert h.state.token_ids == [9, 8, 7, 7]


def test_clear_preserves_session_options_and_ring_configuration():
    h = _harness()
    ring = h.state.snapshot_ring
    h.state.rewind_enabled = False
    h.state.is_asymmetric_rendering = True
    ring.snapshots.append(SimpleNamespace(offset=1, states=_Payload()))
    snapshot_payload = weakref.ref(ring.snapshots[0].states)
    h.state.clear()
    assert h.old_payload() is None and snapshot_payload() is None
    assert h.state.token_ids is None and h.state.cache is None
    assert h.state.snapshot_ring is ring and ring.enabled
    assert h.state.rewind_enabled is False
    assert h.state.is_asymmetric_rendering is True


@pytest.mark.parametrize("after_token", [False, True])
def test_failed_reprefill_or_decode_cannot_reuse_old_token_history(after_token):
    h = _harness(ids=(9, 8, 7))

    def fail(*args, **kwargs):
        if after_token:
            yield 7, None
        raise RuntimeError("generation failed")

    h.namespace["generate_step"] = fail
    with pytest.raises(RuntimeError, match="generation failed"):
        list(h.run())
    assert h.old_payload() is None
    assert h.state.cache is None and h.state.token_ids is None
    assert not h.state.snapshot_ring.snapshots


def test_apc_materialization_also_releases_discarded_session_first():
    h = _harness(ids=(9, 8, 7))

    def materialize(*args, **kwargs):
        assert h.old_payload() is None
        assert h.state.cache is None and h.state.token_ids is None
        h.calls.append("materialize")
        return [_Layer()]

    coordinator = SimpleNamespace(
        enabled=True,
        is_checkpoint=False,
        prepare_prefill=lambda *a: None,
        lookup=lambda *a, **kw: {"prefix_len": 1},
        materialize_single=materialize,
        commit=lambda *a, **kw: None,
    )
    h.namespace["_apc"].APCCoordinator = lambda *a: coordinator
    h.namespace["_apc"].hash_image_payload = lambda **kw: 0
    h.namespace["_apc"].semantic_extra_hash = lambda **kw: 0
    h.namespace["kv_quant_from_legacy"] = lambda *a: None
    list(h.run(apc_manager=object()))
    assert h.calls[0] == "materialize"
    assert h.state.token_ids == [9, 8, 7, 7]


def test_forward_extension_preserves_cache_without_snapshot_restore():
    h = _harness(ids=(1, 2, 3, 4, 5), hybrid=True)
    original_cache_id = id(h.state.cache)
    list(h.run())
    assert h.old_payload() is not None
    assert h.calls == [("generate", [5], original_cache_id)]
    assert h.state.token_ids == [1, 2, 3, 4, 5, 7]


def test_cancellation_before_publication_leaves_discarded_session_empty():
    h = _harness(ids=(9, 8, 7))
    gen = h.run()
    next(gen)
    gen.close()
    assert h.old_payload() is None
    assert h.state.cache is None and h.state.token_ids is None


def test_explicit_caller_cache_remains_owned_by_caller():
    h = _harness(ids=(9, 8, 7))
    caller_cache = [_Layer()]
    list(h.run(prompt_cache=caller_cache))
    assert h.old_payload() is None
    assert h.state.cache is caller_cache
    assert h.calls == [("generate", [9, 8, 7], id(caller_cache))]
