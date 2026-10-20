"""P405: a CLI-launched server serves the CLI's settings (fork-only, CPU).

The server package builds ``runtime.config`` (``RuntimeConfig.from_env``) when it is IMPORTED, before ``cli.main`` writes the
CLI's flags into the environment; ``app.py`` then resolved ``cfg.kv_quant_scheme or get_kv_quant_scheme()`` from the stale
truthy default, so ``--kv-quant-scheme turboquant`` served affine 4-bit KV while the log printed ``scheme=turboquant``. Every
case starts from the import-time config with every CLI-written env var cleared, so no ambient value can mask the defect.
"""

import importlib
import logging
import os
import sys

import pytest

from mlx_vlm.server import cli
from mlx_vlm.server.runtime import ModelCacheRegistry, runtime
from mlx_vlm.server.runtime_config import RuntimeConfig

server_app = importlib.import_module(
    "mlx_vlm.server.app"
)  # the package re-exports `app`, the FastAPI object

CLI_ENV = (
    "KV_BITS",
    "KV_KEY_BITS",
    "KV_VALUE_BITS",
    "KV_KEY_SCHEME",
    "KV_VALUE_SCHEME",
    "KV_GROUP_SIZE",
    "KV_QUANT_SCHEME",
    "TQ_KV_QUANT_MODE",
    "QUANTIZED_KV_START",
    "MAX_KV_SIZE",
    "KV_PREALLOC_TOKENS",
    "MLX_VLM_DRAFT_MODEL",
    "MLX_VLM_DRAFT_KIND",
    "MLX_VLM_VISION_CACHE_SIZE",
    "MLX_VLM_PRELOAD_MODEL",
    "MLX_VLM_SERVER_API_KEY",
)

# every RuntimeConfig knob the CLI writes: (flags, field, the value the served config must hold)
KNOBS = [
    (["--kv-bits", "4"], "kv_bits", 4.0),
    (["--kv-key-bits", "4"], "kv_key_bits", 4.0),
    (["--kv-value-bits", "3"], "kv_value_bits", 3.0),
    (["--kv-key-scheme", "turboquant"], "kv_key_scheme", "turboquant"),
    (["--kv-value-scheme", "turboquant"], "kv_value_scheme", "turboquant"),
    (["--kv-group-size", "32"], "kv_group_size", 32),
    (["--kv-quant-scheme", "turboquant"], "kv_quant_scheme", "turboquant"),
    (["--quantized-kv-start", "2"], "quantized_kv_start", 2),
    (["--max-kv-size", "4096"], "max_kv_size", 4096),
    (["--draft-model", "/drafts/mtp"], "spec_draft_model", "/drafts/mtp"),
    (["--draft-kind", "mtp"], "spec_draft_kind", "mtp"),
    (["--vision-cache-size", "7"], "vision_cache_size", 7),
]


@pytest.fixture
def launch(monkeypatch):
    """``launch(*flags)`` runs ``cli.main`` against the config the package import built; returns ``runtime.config``."""
    saved = dict(os.environ)
    for name in CLI_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runtime, "config", RuntimeConfig.from_env())  # import time
    monkeypatch.setattr(runtime, "model_cache", ModelCacheRegistry())
    monkeypatch.setattr(runtime, "response_generator", None)
    monkeypatch.setattr(runtime, "apc_manager", None)
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_apply_mlx_memory_limits", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_configure_session_manager", lambda **k: None)

    def run(*flags):
        monkeypatch.setattr(sys, "argv", ["mlx_vlm.server", *flags])
        cli.main()
        return runtime.config

    yield run
    os.environ.clear()  # cli.main writes os.environ directly
    os.environ.update(saved)


def test_t1_the_cli_scheme_reaches_the_runtime_config(launch):
    assert launch(
        "--kv-bits", "4", "--kv-quant-scheme", "turboquant"
    ).kv_quant_scheme == ("turboquant")


@pytest.mark.parametrize(("flags", "field", "value"), KNOBS, ids=[k[1] for k in KNOBS])
def test_t2_every_cli_written_knob_reaches_the_runtime_config(
    launch, flags, field, value
):
    assert getattr(launch(*flags), field) == value


class _Generator:
    """Captures what ``get_cached_model`` hands the generator; loads nothing."""

    seen = []

    def __init__(self, **kwargs):
        _Generator.seen.append(kwargs)

    def wait_until_ready(self):
        return object(), object(), {}

    def stop_and_join(self):
        pass


@pytest.fixture
def loaded(launch, monkeypatch):
    _Generator.seen = []
    monkeypatch.setattr(server_app, "ResponseGenerator", _Generator)
    monkeypatch.setattr(server_app._apc, "from_env", lambda **k: None)

    def run(*flags):
        launch(*flags)
        server_app.get_cached_model("/models/tiny", None, model_kind="text_generation")
        (kwargs,) = _Generator.seen
        return kwargs

    return run


def test_t3_the_load_path_builds_the_generator_with_the_cli_scheme(loaded):
    kwargs = loaded("--kv-bits", "4", "--kv-quant-scheme", "turboquant")
    assert (kwargs["kv_bits"], kwargs["kv_quant_scheme"]) == (4.0, "turboquant")


def test_t4_settings_replace_restores_the_cli_values(launch):
    from fastapi.testclient import TestClient

    launch("--kv-bits", "4", "--kv-quant-scheme", "turboquant")
    client = TestClient(server_app.app)  # no lifespan: nothing preloads
    r = client.patch("/v1/settings", json={"kv_quant_scheme": "uniform"})
    assert r.status_code == 200 and r.json()["current"]["kv_quant_scheme"] == "uniform"
    r = client.patch("/v1/settings", json={"op": "replace", "values": {}})
    assert r.status_code == 200, r.text
    assert r.json()["current"]["kv_quant_scheme"] == "turboquant"
    assert r.json()["current"]["kv_bits"] == 4.0


def test_t5_the_load_log_states_the_resolved_codec(loaded, caplog):
    with caplog.at_level(logging.INFO, logger="mlx_vlm"):
        kwargs = loaded("--kv-bits", "4", "--kv-quant-scheme", "turboquant")
    lines = [r.getMessage() for r in caplog.records]
    resolved = [m for m in lines if m.startswith("KV cache resolved:")]
    assert len(resolved) == 1, lines
    for name in (
        "kv_bits",
        "kv_key_bits",
        "kv_value_bits",
        "kv_key_scheme",
        "kv_value_scheme",
        "kv_quant_scheme",
        "kv_group_size",
        "quantized_kv_start",
    ):
        assert f"{name}={kwargs[name]!r}" in resolved[0], (name, resolved[0])
    assert (
        "TQ_KV_QUANT_MODE='mse'" in resolved[0]
    )  # K6: the RESOLVED default, as TurboQuantKVCache applies it


def test_t5_the_env_derived_preload_log_line_is_gone():
    import inspect

    assert 'os.environ.get("KV_QUANT_SCHEME"' not in inspect.getsource(
        server_app.lifespan
    )


def test_k6_the_logged_mode_is_the_resolved_default_without_the_env(
    launch, monkeypatch, caplog
):
    """P405 code round 1 K6: a non-CLI load (no TQ_KV_QUANT_MODE) logs the mode TurboQuantKVCache applies, 'mse'."""
    _Generator.seen = []
    monkeypatch.setattr(server_app, "ResponseGenerator", _Generator)
    monkeypatch.setattr(server_app._apc, "from_env", lambda **k: None)
    launch("--kv-bits", "4", "--kv-quant-scheme", "turboquant")
    monkeypatch.delenv("TQ_KV_QUANT_MODE", raising=False)
    with caplog.at_level(logging.INFO, logger="mlx_vlm"):
        server_app.get_cached_model("/models/tiny", None, model_kind="text_generation")
    (line,) = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("KV cache resolved:")
    ]
    assert "TQ_KV_QUANT_MODE='mse'" in line, line
