"""CPU-only test modules must not change other tests' device selection."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

MODULES = ("test_ssm_with_states", "test_mtp_profile", "test_nemotron_h_rollback")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_probe(source):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("module_name", MODULES)
def test_import_preserves_default_device_without_switching(module_name):
    # Real MLX and real imports, in a fresh process. Selecting a device alone
    # launches no kernels; record even temporary switches during collection.
    _run_probe(f"""
        import importlib
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        previous = mx.default_device()
        changes = []
        original_set = mx.set_default_device
        def record_set(device):
            changes.append(device)
            original_set(device)
        mx.set_default_device = record_set

        importlib.import_module("mlx_vlm.tests.{module_name}")

        assert mx.default_device() == previous, (
            "{module_name} changed the default device during import",
            previous, mx.default_device(),
        )
        assert changes == [], ("collection switched devices", changes)
        """)


@pytest.mark.parametrize("module_name", MODULES)
@pytest.mark.parametrize("initial_device", ("cpu", "gpu"))
def test_cpu_fixture_restores_device_after_success_and_failure(
    module_name, initial_device
):
    _run_probe(f"""
        import mlx.core as mx
        import pytest

        mx.set_default_device(mx.{initial_device})
        previous = mx.default_device()

        def success():
            assert mx.default_device() == mx.cpu
        class ProbeFailure(Exception):
            pass
        def failure():
            assert mx.default_device() == mx.cpu
            raise ProbeFailure

        class ProbePlugin:
            def pytest_collection_modifyitems(self, items):
                # Keep real module fixture discovery, replacing model/kernel
                # tests with probes of the autouse fixture's public behavior.
                parent = items[0].getparent(pytest.Module)
                good = pytest.Function.from_parent(
                    parent, name="test_success_probe", callobj=success
                )
                bad = pytest.Function.from_parent(
                    parent, name="test_failure_probe", callobj=failure
                )
                bad.add_marker(pytest.mark.xfail(raises=ProbeFailure, strict=True))
                items[:] = [good, bad]

            @pytest.hookimpl(hookwrapper=True, tryfirst=True)
            def pytest_runtest_teardown(self, item, nextitem):
                yield
                assert mx.default_device() == previous

        result = pytest.main(
            ["-q", "mlx_vlm/tests/{module_name}.py"], plugins=[ProbePlugin()]
        )
        assert result == 0
        assert mx.default_device() == previous
        """)


def test_explicit_gpu_test_overrides_cpu_fixture_and_restores_it():
    # Stop at input construction: exercise nested device scopes, no GPU work.
    _run_probe("""
        from contextlib import contextmanager
        import mlx.core as mx
        from mlx_vlm.tests import test_ssm_with_states as module

        class InputsReached(Exception):
            pass
        def stop_before_arrays(*args, **kwargs):
            assert mx.default_device() == mx.gpu
            raise InputsReached
        module._random_inputs = stop_before_arrays

        mx.set_default_device(mx.gpu)
        with contextmanager(module._cpu_device.__wrapped__)():
            assert mx.default_device() == mx.cpu
            test = module.TestMetalKernelMatchesSingleStepKernel()
            try:
                test.test_kernel_matches_ssm_update_kernel_applied_t_times()
            except InputsReached:
                pass
            else:
                raise AssertionError("GPU test did not construct its inputs")
            assert mx.default_device() == mx.cpu
        assert mx.default_device() == mx.gpu
        """)
