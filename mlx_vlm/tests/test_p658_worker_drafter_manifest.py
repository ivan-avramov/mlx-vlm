"""P658 (a) Interface 4 / acceptance 4: the worker's qualification record carries ``drafter_manifest`` only when the JOB asks
(fork-only, dev; CPU only).

Written from ``.p658-material/spec.md`` before the implementation.  The first line of each test body is a ``# class:`` comment.
The test drives ``worker.run_job`` for a ``qualify`` job with the model / server boundary faked (the booted server, the HTTP
post, prompt sizing, the memory-limit recorder, the capture) and a fake ``rg`` whose ``draft_model`` has ``make_cache()`` and
``_cache``.  The qualification request itself (``qualify_request``) and ``kvmanifest.manifest`` are the real code.  The fake
``post`` populates the drafter's ``_cache`` (as serving does) and the fake ``release_session`` clears it (as releasing the
session does), so a record that matches the populated cache proves the read happened after the response and before the release.
"""

import sys
import types
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(DEV))

from p369_phase2 import kvmanifest, telemetry, worker  # noqa: E402


# leaf classes named as the real ones are: kvmanifest classifies by class name through the MRO
class KVCache:
    pass


class PreallocKVCache(KVCache):
    pass


class QuantizedKVCache:
    bits = 8
    group_size = 64


class PreallocQuantizedKVCache(QuantizedKVCache):
    pass


class ArraysCache:
    pass


class _Capture:
    backend = "cached_session"
    completed = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def reset(self):
        pass


class _Client:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _LM:
    args = types.SimpleNamespace(num_attention_heads=24)

    def make_cache(self):
        return [ArraysCache(), KVCache(), ArraysCache(), KVCache()]

    def __call__(self, *a, **k):
        return None


class _Drafter:
    def __init__(self, served):
        self._cache = None
        self._served = served  # what serving leaves in ``_cache`` after the request

    def make_cache(self):
        return [KVCache(), KVCache()]


def _manifest(leaves):
    return kvmanifest.manifest(leaves)


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """``run(job_extra, served=[...]) -> (record, events, drafter)``."""
    import mlx_vlm.server as server
    from mlx_vlm.generate import ar

    def run(*, served=None, **job_extra):
        served = [PreallocKVCache(), PreallocKVCache()] if served is None else served
        draft = _Drafter(served)
        rg = types.SimpleNamespace(
            model=types.SimpleNamespace(language_model=_LM()),
            draft_model=draft,
            draft_kind="mtp",
        )
        events = []

        def post(client, payload, chat_id, capture, timeout_s):
            events.append("post")
            list(ar.run_speculative_rounds())  # the recorded prefill / decode boundary
            draft._cache = draft._served  # serving populates the drafter's own cache
            return dict(
                chat_id=chat_id,
                final=dict(
                    cached_tokens=0,
                    finish_reason="stop",
                    generation_tokens=3,
                    prompt_tps=1.0,
                    generation_tps=2.0,
                ),
                prompt=dict(tokens=3, sha256="x"),
                response=dict(sdpa_forced=0),
                wall_s=0.1,
            )

        def release_session(chat_id):
            events.append("release")
            draft._cache = None  # releasing the session clears the caches

        monkeypatch.setattr(server.runtime, "response_generator", rg, raising=False)
        monkeypatch.setattr(ar, "run_speculative_rounds", lambda *a, **k: iter(()))
        for name, value in dict(
            record_memory_limits=lambda: (dict(phys=1, rec_wss=1), lambda: None),
            boot=lambda argv: _Client(),
            served_state=lambda: {},
            assert_served=lambda state, expect: None,
            check_sources=lambda root, dev: None,
            check_snapshots=lambda argv, revisions: {},
            Capture=_Capture,
            prompt_tokens=lambda client, messages, base: 3,
            prompt_ids=lambda client, messages, base: [1, 2, 3],
            size_prompt=lambda count, target, prefix, suffix: ("text", target),
            check_server_main=lambda: True,
            post=post,
            release_session=release_session,
        ).items():
            monkeypatch.setattr(worker, name, value)
        monkeypatch.setattr(telemetry, "manifest_from_model", lambda lm: {})
        job = dict(
            name="m/qualify",
            mode="qualify",
            argv=["--model", "some/model"],
            expect={},
            source_root=str(tmp_path),
            dev_dir=str(DEV),
            out_dir=str(tmp_path / "out"),
            requests=[
                dict(
                    name="max-context",
                    model="m",
                    target=260096,
                    prefix="p",
                    suffix="s",
                    seed=7,
                    generation_defaults=dict(temperature=0.7),
                    timeout_s=10,
                )
            ],
            timeout_s=60,
            **job_extra,
        )
        (tmp_path / "out").mkdir(exist_ok=True)
        result = dict(requests=[])
        worker.run_job(job, tmp_path / "out", result)
        (request,) = result["requests"]
        return request["qualify"], events, draft

    return run


# today's record keys for this harness (the request is exercised as the real code does it)
TODAY = {
    "prompt",
    "chunks",
    "prompt_step_tokens",
    "boundary_seen",
    "peaks",
    "lazy_calls",
    "shrink_calls",
    "memory",
    "template",
    "delegates_to_cli_main",
    "manifest",
    "capacities_by_path",
    "cached_tokens",
    "backend",
    "finish_reason",
    "generation_tokens",
    "rates",
    "sdpa_forced",
    "server_defaults",
    "wall_s",
    "drafter_live_checks",
}


def test_the_record_gains_the_drafters_observed_and_template_manifests(harness):
    # class: exercises new behaviour
    rec, events, draft = harness(drafter_manifest=True)
    assert events == ["post", "release"]
    assert set(rec["drafter_manifest"]) == {"observed", "template"}
    assert rec["drafter_manifest"]["template"] == _manifest([KVCache(), KVCache()])
    assert rec["drafter_manifest"]["observed"] == _manifest(
        [PreallocKVCache(), PreallocKVCache()]
    )
    assert rec["drafter_manifest"]["observed"] == [
        dict(path="0", kind="native", cls="PreallocKVCache", batch=False),
        dict(path="1", kind="native", cls="PreallocKVCache", batch=False),
    ]
    assert set(rec) == TODAY | {"drafter_manifest"}  # nothing else changes


def test_the_drafter_cache_is_read_after_the_response_and_before_the_session_is_released(
    harness,
):
    # class: exercises new behaviour
    rec, events, draft = harness(drafter_manifest=True)
    assert events == ["post", "release"]
    assert (
        draft._cache is None
    )  # the release did clear it: a late read would have seen nothing
    assert rec["drafter_manifest"][
        "observed"
    ], "the drafter cache was read before the response or after the release"


def test_a_quantized_drafter_leaf_is_recorded_as_it_is(harness):
    # class: exercises new behaviour
    served = [PreallocQuantizedKVCache(), PreallocKVCache()]
    rec, _, _ = harness(drafter_manifest=True, served=served)
    observed = rec["drafter_manifest"]["observed"]
    assert [o["kind"] for o in observed] == [
        "affine",
        "native",
    ]  # recorded, not judged, in the worker
    assert rec["drafter_manifest"]["template"] == _manifest([KVCache(), KVCache()])


def test_a_drafter_with_no_cache_after_the_request_records_an_empty_observed_list(
    harness,
):
    # class: exercises new behaviour
    rec, _, _ = harness(drafter_manifest=True, served=None)
    assert rec["drafter_manifest"]["observed"] != []  # control: the default populates
    rec, _, _ = harness(drafter_manifest=True, served=[])
    assert rec["drafter_manifest"]["observed"] == []
    assert rec["drafter_manifest"]["template"] == _manifest([KVCache(), KVCache()])


def test_without_the_job_field_the_record_has_exactly_todays_keys(harness):
    # class: preserves an invariant
    rec, events, _ = harness()
    assert set(rec) == TODAY
    assert "drafter_manifest" not in rec
    assert events == ["post", "release"]


@pytest.mark.parametrize("value", (False, None, 0))
def test_a_falsy_job_field_records_nothing(harness, value):
    # class: preserves an invariant
    rec, _, _ = harness(drafter_manifest=value)
    assert "drafter_manifest" not in rec
    assert set(rec) == TODAY


def test_the_target_part_of_the_record_is_the_same_with_and_without_the_field(harness):
    # class: exercises new behaviour
    plain, _, _ = harness()
    asked, _, _ = harness(drafter_manifest=True)
    assert "drafter_manifest" in asked
    asked = dict(asked)
    asked.pop("drafter_manifest")
    assert asked == plain
