"""`ssm_update_with_states` (mamba2 with-states kernel + ops twin).

Fork-only: gives mamba2 (`nemotron_h`) the same per-position-state capture
`qwen3_5`'s GatedDeltaNet already has via
`qwen3_5/gated_delta.py:gated_delta_update_with_states`, so an MTP verify no
longer has to replay the block one position at a time through the whole
backbone to observe intermediate `(conv_state, ssm_state)` snapshots (see
`mlx_vlm/models/recurrent_rollback.py`'s docstring and
`mlx_vlm/models/nemotron_h/language.py`'s `NemotronHModel.__call__`).

CPU-only and tiny synthetic tensors throughout: a real benchmark run may be
using this machine's GPU concurrently, so this module must never touch it.
The one GPU-path test is skipped unless `MLX_VLM_GPU_TESTS=1` is set, which
this suite never sets itself.
"""

import os

import mlx.core as mx

# Fork: this benchmark host may have a live GPU benchmark running; every test
# in this module (other than the explicitly gated GPU one) must stay off it.
mx.set_default_device(mx.cpu)

import pytest

from mlx_vlm.models.ssm import (
    _ssm_with_states_ops,
    compute_dt,
    ssm_update,
    ssm_update_kernel,
    ssm_update_with_states,
)

TIME_STEP_LIMIT = (0.001, 100.0)


def _random_inputs(b, t, h, dh, g, ds, seed):
    mx.random.seed(seed)
    x = mx.random.normal((b, t, h, dh))
    A_log = mx.random.normal((h,)) * 0.1
    B = mx.random.normal((b, t, g, ds))
    C = mx.random.normal((b, t, g, ds))
    D = mx.random.normal((h,))
    dt = mx.random.normal((b, t, h))
    dt_bias = mx.random.normal((h,))
    state0 = mx.random.normal((b, h, dh, ds)).astype(mx.float32)
    return x, A_log, B, C, D, dt, dt_bias, state0


def _looped_single_step_reference(x, A_log, B, C, D, dt, dt_bias, state0):
    """Ground truth: call the CPU dispatcher (`ssm_update`, which on CPU
    always takes the `ssm_attn` chunked-scan path) one position at a time,
    threading the evolving state through -- exactly what ordinary
    one-token-at-a-time decode does. Returns per-position outputs and the
    state snapshot captured AFTER each position, plus the final state."""
    b, t_len, h, dh = x.shape
    state = state0
    ys = []
    states = []
    for t in range(t_len):
        y_t, state = ssm_update(
            x[:, t : t + 1],
            A_log,
            B[:, t : t + 1],
            C[:, t : t + 1],
            D,
            dt[:, t : t + 1],
            dt_bias,
            state,
            TIME_STEP_LIMIT,
        )
        ys.append(y_t)
        states.append(state)
    return mx.concatenate(ys, axis=1), state, mx.stack(states, axis=1)


B_T_CASES = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4)]


class TestOpsTwinMatchesLoopedSingleStep:
    @pytest.mark.parametrize("b,t", B_T_CASES)
    def test_outputs_and_states_match(self, b, t):
        h, dh, g, ds = 4, 8, 2, 16
        x, A_log, B, C, D, dt, dt_bias, state0 = _random_inputs(
            b, t, h, dh, g, ds, seed=b * 100 + t
        )

        dt_clipped = compute_dt(dt, dt_bias, TIME_STEP_LIMIT)
        y_ops, final_state_ops, states_ops = _ssm_with_states_ops(
            x, A_log, B, C, D, dt_clipped, state0
        )

        y_ref, final_state_ref, states_ref = _looped_single_step_reference(
            x, A_log, B, C, D, dt, dt_bias, state0
        )

        mx.eval(y_ops, final_state_ops, states_ops, y_ref, final_state_ref, states_ref)

        assert y_ops.shape == y_ref.shape
        assert states_ops.shape == states_ref.shape == (b, t, h, dh, ds)
        assert mx.allclose(y_ops, y_ref, rtol=1e-4, atol=1e-4).item()
        assert mx.allclose(states_ops, states_ref, rtol=1e-4, atol=1e-4).item()
        assert mx.allclose(
            final_state_ops, final_state_ref, rtol=1e-4, atol=1e-4
        ).item()

    @pytest.mark.parametrize("b,t", B_T_CASES)
    def test_final_state_equals_last_position_state(self, b, t):
        h, dh, g, ds = 4, 8, 2, 16
        x, A_log, B, C, D, dt, dt_bias, state0 = _random_inputs(
            b, t, h, dh, g, ds, seed=1000 + b * 100 + t
        )
        dt_clipped = compute_dt(dt, dt_bias, TIME_STEP_LIMIT)
        _, final_state, states = _ssm_with_states_ops(
            x, A_log, B, C, D, dt_clipped, state0
        )
        mx.eval(final_state, states)
        assert mx.allclose(final_state, states[:, -1], rtol=1e-6, atol=1e-6).item()

    def test_t_equals_one_matches_single_step_exactly(self):
        b, t, h, dh, g, ds = 2, 1, 4, 8, 2, 16
        x, A_log, B, C, D, dt, dt_bias, state0 = _random_inputs(
            b, t, h, dh, g, ds, seed=42
        )
        dt_clipped = compute_dt(dt, dt_bias, TIME_STEP_LIMIT)
        y_ops, final_state_ops, states_ops = _ssm_with_states_ops(
            x, A_log, B, C, D, dt_clipped, state0
        )
        y_ref, final_state_ref, states_ref = _looped_single_step_reference(
            x, A_log, B, C, D, dt, dt_bias, state0
        )
        mx.eval(y_ops, final_state_ops, states_ops, y_ref, final_state_ref, states_ref)

        assert mx.allclose(y_ops, y_ref, rtol=1e-4, atol=1e-4).item()
        assert mx.allclose(states_ops, states_ref, rtol=1e-4, atol=1e-4).item()
        assert mx.allclose(
            final_state_ops, final_state_ref, rtol=1e-4, atol=1e-4
        ).item()


class TestOpsTwinMask:
    def test_false_position_freezes_state_and_zeroes_y(self):
        """A masked-out (`mask == False`) position must carry the state
        forward UNCHANGED and write `y == 0` there -- mirrors
        `_gated_delta_with_states_ops`'s masked branch. `ssm_attn` (the
        CPU dispatcher's chunked-scan path) has no equivalent masked-`y`
        behavior, so this is checked against a hand-computed frozen state,
        not against `ssm_update`/`ssm_attn`."""
        b, t, h, dh, g, ds = 2, 4, 4, 8, 2, 16
        x, A_log, B, C, D, dt, dt_bias, state0 = _random_inputs(
            b, t, h, dh, g, ds, seed=55
        )
        dt_clipped = compute_dt(dt, dt_bias, TIME_STEP_LIMIT)

        # Row 0: position 2 invalid. Row 1: fully valid (control).
        mask = mx.array([[True, True, False, True], [True, True, True, True]])

        y, final_state, states = _ssm_with_states_ops(
            x, A_log, B, C, D, dt_clipped, state0, mask
        )
        mx.eval(y, final_state, states)

        # Masked position: state frozen at the prior position's value,
        # y zeroed.
        assert mx.array_equal(states[0, 2], states[0, 1]).item()
        assert mx.allclose(y[0, 2], mx.zeros_like(y[0, 2])).item()
        # Un-masked positions still evolve normally (not frozen/zeroed).
        assert not mx.array_equal(states[0, 1], states[0, 0]).item()
        assert not mx.allclose(y[0, 1], mx.zeros_like(y[0, 1])).item()
        # Control row (no masked positions) never freezes.
        for pos in range(1, t):
            assert not mx.array_equal(states[1, pos], states[1, pos - 1]).item()

        # states[:, -1] still equals the final state even with a masked
        # position earlier in the block.
        assert mx.allclose(final_state, states[:, -1], rtol=1e-6, atol=1e-6).item()


class TestDispatcherOnCPU:
    def test_ssm_update_with_states_matches_ops_twin_on_cpu(self):
        """`ssm_update_with_states` must route to the ops twin off-GPU (this
        whole module is CPU-pinned) -- this is the regression guard that the
        dispatcher's device gate is wired correctly."""
        b, t, h, dh, g, ds = 1, 3, 4, 8, 2, 16
        x, A_log, B, C, D, dt, dt_bias, state0 = _random_inputs(
            b, t, h, dh, g, ds, seed=7
        )
        y_disp, final_state_disp, states_disp = ssm_update_with_states(
            x, A_log, B, C, D, dt, dt_bias, state0, TIME_STEP_LIMIT
        )
        dt_clipped = compute_dt(dt, dt_bias, TIME_STEP_LIMIT)
        y_ops, final_state_ops, states_ops = _ssm_with_states_ops(
            x, A_log, B, C, D, dt_clipped, state0
        )
        mx.eval(
            y_disp, final_state_disp, states_disp, y_ops, final_state_ops, states_ops
        )
        assert mx.array_equal(y_disp, y_ops).item()
        assert mx.array_equal(states_disp, states_ops).item()
        assert mx.array_equal(final_state_disp, final_state_ops).item()

    def test_lengths_raises_not_implemented(self):
        """`lengths` has no with-states counterpart (see
        `ssm_update_with_states`'s docstring) -- must fail loud, not
        silently ignore it."""
        b, t, h, dh, g, ds = 1, 2, 4, 8, 2, 16
        x, A_log, B, C, D, dt, dt_bias, state0 = _random_inputs(
            b, t, h, dh, g, ds, seed=11
        )
        lengths = mx.array([t])
        with pytest.raises(NotImplementedError, match="lengths"):
            ssm_update_with_states(
                x,
                A_log,
                B,
                C,
                D,
                dt,
                dt_bias,
                state0,
                TIME_STEP_LIMIT,
                lengths=lengths,
            )

    def test_state_none_starts_from_zeros(self):
        b, t, h, dh, g, ds = 1, 2, 4, 8, 2, 16
        x, A_log, B, C, D, dt, dt_bias, _ = _random_inputs(b, t, h, dh, g, ds, seed=9)
        y_none, final_none, states_none = ssm_update_with_states(
            x, A_log, B, C, D, dt, dt_bias, None, TIME_STEP_LIMIT
        )
        zeros = mx.zeros((b, h, dh, ds), dtype=mx.float32)
        y_zeros, final_zeros, states_zeros = ssm_update_with_states(
            x, A_log, B, C, D, dt, dt_bias, zeros, TIME_STEP_LIMIT
        )
        mx.eval(y_none, final_none, states_none, y_zeros, final_zeros, states_zeros)
        assert mx.array_equal(y_none, y_zeros).item()
        assert mx.array_equal(states_none, states_zeros).item()
        assert mx.array_equal(final_none, final_zeros).item()


@pytest.mark.skipif(
    os.environ.get("MLX_VLM_GPU_TESTS") != "1",
    reason="GPU-only: exercises the Metal ssm_with_states_kernel; set "
    "MLX_VLM_GPU_TESTS=1 explicitly to run (never set in this repo's CI or "
    "by any other test in this suite -- this machine may have a live "
    "benchmark on the GPU).",
)
class TestMetalKernelMatchesSingleStepKernel:
    def test_kernel_matches_ssm_update_kernel_applied_t_times(self):
        previous_device = mx.default_device()
        mx.set_default_device(mx.gpu)
        try:
            b, t, h, dh, g, ds = 2, 4, 4, 32, 2, 32
            x, A_log, B, C, D, dt, dt_bias, state0 = _random_inputs(
                b, t, h, dh, g, ds, seed=123
            )

            y_kernel, final_state_kernel, states_kernel = ssm_update_with_states(
                x, A_log, B, C, D, dt, dt_bias, state0, TIME_STEP_LIMIT
            )

            state = state0
            ys = []
            states = []
            for tt in range(t):
                y_t, state = ssm_update_kernel(
                    x[:, tt : tt + 1],
                    A_log,
                    B[:, tt : tt + 1],
                    C[:, tt : tt + 1],
                    D,
                    dt[:, tt : tt + 1],
                    dt_bias,
                    state,
                    TIME_STEP_LIMIT,
                )
                ys.append(y_t)
                states.append(state)
            y_ref = mx.concatenate(ys, axis=1)
            states_ref = mx.stack(states, axis=1)
            final_state_ref = state

            mx.eval(
                y_kernel,
                final_state_kernel,
                states_kernel,
                y_ref,
                final_state_ref,
                states_ref,
            )

            assert mx.allclose(y_kernel, y_ref, rtol=1e-6, atol=1e-6).item()
            assert mx.allclose(states_kernel, states_ref, rtol=1e-6, atol=1e-6).item()
            assert mx.allclose(
                final_state_kernel, final_state_ref, rtol=1e-6, atol=1e-6
            ).item()
        finally:
            mx.set_default_device(previous_device)
