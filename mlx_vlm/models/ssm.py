from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@mx.compile
def compute_dt(dt, dt_bias, time_step_limit):
    dt = dt.astype(mx.float32)
    dt = nn.softplus(dt + dt_bias)
    return mx.clip(dt, time_step_limit[0], time_step_limit[1])


def make_ssm_kernel():
    if not mx.metal.is_available():
        return None
    source = """
        auto n = thread_position_in_grid.z;
        auto h_idx = n % H;
        auto g_idx = n / G;
        constexpr int n_per_t = Ds / 32;

        auto x = X + n * Dh;
        out += n * Dh;
        auto i_state = state_in + n * Dh * Ds;
        auto o_state = state_out + n * Dh * Ds;

        // C and B have shape [batch, group, state_dim]
        // C and B need to be offset by group size
        auto C_ = C + g_idx * Ds;
        auto B_ = B + g_idx * Ds;

        auto ds_idx = thread_position_in_threadgroup.x;
        auto d_idx = thread_position_in_grid.y;

        auto dt_ = static_cast<float>(dt[n]);
        auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
        auto dA = fast::exp(A * dt_);

        float acc = 0.0;
        auto x_ = static_cast<float>(x[d_idx]);

        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * ds_idx + i;
            auto idx = d_idx * Ds + s_idx;
            auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
            auto state = dA * i_state[idx] + dB_by_x;
            o_state[idx] = static_cast<U>(state);
            acc += state * C_[s_idx];
        }
        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
        }
    """
    return mx.fast.metal_kernel(
        name="ssm_kernel",
        input_names=["X", "A_log", "B", "C", "D", "dt", "state_in"],
        output_names=["out", "state_out"],
        source=source,
    )


_ssm_kernel = make_ssm_kernel()


def ssm_update_kernel(
    hidden_states: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: mx.array,
    time_step_limit: Tuple[float, float],
):
    n, _, h, d = hidden_states.shape
    input_type = hidden_states.dtype
    state_type = state.dtype
    hb, ds = B.shape[-2:]
    dt = compute_dt(dt, dt_bias, time_step_limit)
    return _ssm_kernel(
        inputs=[hidden_states, A_log, B, C, D, dt, state],
        template=[
            ("T", input_type),
            ("U", state_type),
            ("Dh", d),
            ("Ds", ds),
            ("H", h),
            ("G", h // hb),
        ],
        grid=(32, d, h * n),
        threadgroup=(32, 8, 1),
        output_shapes=[(n, 1, h, d), state.shape],
        output_dtypes=[input_type, state_type],
    )


def make_ssm_with_states_kernel():
    """Same recurrence as `make_ssm_kernel`'s single-step body, looped over
    `T` positions inside one thread (state carried in a register across the
    loop) instead of one launch per position -- mirrors
    `qwen3_5/gated_delta.py:_make_gated_delta_with_states_kernel`. Emits the
    per-position state after every t (`states`) in addition to `out` and the
    final `state_out`, so a verify block no longer needs the caller to
    replay the model one token at a time to observe intermediate state."""
    if not mx.metal.is_available():
        return None
    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / H;
        auto h_idx = n % H;
        constexpr int heads_per_group = H / G;
        auto g_idx = h_idx / heads_per_group;
        constexpr int n_per_t = Ds / 32;

        auto d_idx = thread_position_in_grid.y;
        auto ds_idx = thread_position_in_threadgroup.x;

        auto x_ = X + (b_idx * T * H + h_idx) * Dh;
        auto out_ = out + (b_idx * T * H + h_idx) * Dh;
        auto dt_ = dt + b_idx * T * H + h_idx;
        auto B_ = B + b_idx * T * G * Ds + g_idx * Ds;
        auto C_ = C + b_idx * T * G * Ds + g_idx * Ds;
        auto states_ = states + (b_idx * T * H + h_idx) * Dh * Ds + d_idx * Ds;

        auto i_state = state_in + (n * Dh + d_idx) * Ds;
        auto o_state = state_out + (n * Dh + d_idx) * Ds;

        auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
        auto D_h = static_cast<float>(D[h_idx]);

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * ds_idx + i;
            state[i] = static_cast<float>(i_state[s_idx]);
        }

        for (int t = 0; t < T; ++t) {
            auto dt_t = static_cast<float>(dt_[0]);
            auto dA = fast::exp(A * dt_t);
            auto x_t = static_cast<float>(x_[d_idx]);

            float acc = 0.0;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * ds_idx + i;
                auto dB_by_x = x_t * dt_t * static_cast<float>(B_[s_idx]);
                state[i] = dA * state[i] + dB_by_x;
                acc += state[i] * C_[s_idx];
            }
            acc = simd_sum(acc);
            if (thread_index_in_simdgroup == 0) {
                out_[d_idx] = static_cast<InT>(acc + x_t * D_h);
            }

            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * ds_idx + i;
                states_[s_idx] = static_cast<StT>(state[i]);
            }

            x_ += H * Dh;
            out_ += H * Dh;
            dt_ += H;
            B_ += G * Ds;
            C_ += G * Ds;
            states_ += H * Dh * Ds;
        }

        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * ds_idx + i;
            o_state[s_idx] = static_cast<StT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="ssm_with_states_kernel",
        input_names=["X", "A_log", "B", "C", "D", "dt", "state_in"],
        output_names=["out", "state_out", "states"],
        source=source,
    )


_ssm_with_states_kernel = make_ssm_with_states_kernel()


def _ssm_with_states_ops(
    x: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Pure-mx per-position twin of `ssm_with_states_kernel` (CPU fallback,
    and the reference the kernel is checked against): mirrors
    `qwen3_5/gated_delta.py:_gated_delta_with_states_ops` -- a plain Python
    loop over `t` applying exactly the single-step recurrence
    (`make_ssm_kernel`'s body), so results match that kernel applied T
    times. `dt` must already be time_step_limit-clipped (`compute_dt`),
    same convention `ssm_update_kernel` uses. `mask` (`[B, T]`, from
    `ArraysCache.make_mask`) freezes the state and zeroes `y` at invalid
    (padded) positions, mirroring `_gated_delta_with_states_ops`'s masked
    branch -- there is no `lengths` handling here (see
    `ssm_update_with_states`'s docstring for why that's out of scope).
    """
    b, l, h, dh = x.shape
    _, _, g, ds = B.shape
    repeats = h // g
    A = -mx.exp(A_log.astype(mx.float32))
    D = D.astype(mx.float32)

    ys = []
    states = []
    for t in range(l):
        old_state = state
        dt_t = dt[:, t]  # [b, h]
        dA = mx.exp(A.reshape(1, h) * dt_t)  # [b, h]
        x_t = x[:, t].astype(mx.float32)  # [b, h, dh]
        B_t = B[:, t]  # [b, g, ds]
        C_t = C[:, t]
        if repeats > 1:
            B_t = mx.repeat(B_t, repeats, axis=1)
            C_t = mx.repeat(C_t, repeats, axis=1)
        dB_by_x = x_t[..., None] * dt_t[..., None, None] * B_t[:, :, None, :]
        state = dA[..., None, None] * state + dB_by_x
        y_t = (state * C_t[:, :, None, :]).sum(axis=-1) + x_t * D.reshape(1, h, 1)

        if mask is not None:
            valid = mask[:, t]
            state = mx.where(valid[:, None, None, None], state, old_state)
            y_t = mx.where(valid[:, None, None], y_t, 0)

        ys.append(y_t.astype(x.dtype))
        states.append(state)

    y = mx.stack(ys, axis=1)
    stacked_states = mx.stack(states, axis=1)
    return y, state, stacked_states


def ssm_update_with_states(
    hidden_states: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    time_step_limit: Tuple[float, float] = (0.001, 100.0),
    mask: Optional[mx.array] = None,
    lengths: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Dispatcher for the with-states path: returns `(y, final_state,
    states)` where `states[:, t]` is the state AFTER position t. Kernel on
    GPU+Metal with no mask/lengths (the single in-thread-per-position loop
    has no notion of a padding mask); the ops twin otherwise (CPU, or a
    masked batch on GPU). `lengths` (`ssm_attn`'s cross-chunk
    valid-remaining-steps bookkeeping, driven by `ArraysCache.lengths`) has
    no with-states counterpart -- this path only exists for the nemotron_h
    MTP-verify capture, which already requires `cache.lengths is None`
    (see `NemotronHMamba2Mixer._conv`'s capture branch), so this fails
    loud rather than silently mishandling it."""
    if lengths is not None:
        raise NotImplementedError(
            "ssm_update_with_states: `lengths` (cross-chunk continuation "
            "bookkeeping) has no with-states implementation; only `mask` "
            "(per-position validity) is supported."
        )
    n, t_len, h, d = hidden_states.shape
    hb, ds = B.shape[-2:]
    if state is None:
        state = mx.zeros((n, h, d, ds), dtype=mx.float32)
    dt = compute_dt(dt, dt_bias, time_step_limit)

    if (
        mask is not None
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or _ssm_with_states_kernel is None
    ):
        return _ssm_with_states_ops(hidden_states, A_log, B, C, D, dt, state, mask)

    input_type = hidden_states.dtype
    state_type = state.dtype
    return _ssm_with_states_kernel(
        inputs=[hidden_states, A_log, B, C, D, dt, state],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dh", d),
            ("Ds", ds),
            ("H", h),
            ("G", hb),
            ("T", t_len),
        ],
        grid=(32, d, h * n),
        threadgroup=(32, 8, 1),
        output_shapes=[(n, t_len, h, d), state.shape, (n, t_len, h, d, ds)],
        output_dtypes=[input_type, state_type, state_type],
    )


def segsum(x, mask=None):
    l = x.shape[-1]
    if mask is not None:
        mask = mx.expand_dims(mask, 1)
        x = x * mask
    x = mx.repeat(x[..., None], l, axis=-1)
    x = mx.tril(x, -1)
    x_segsum = mx.cumsum(x, axis=-2)
    if mask is not None:
        x_segsum = mx.where(
            mask[..., None, :] * mask[..., None], x_segsum, -float("inf")
        )
    return x_segsum


def ssm_attn(
    x: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    time_step_limit: Tuple[float, float] = (0.001, 100.0),
    mask: Optional[mx.array] = None,
    lengths: Optional[mx.array] = None,
    step: int = 256,
) -> Tuple[mx.array, mx.array]:
    b, l, h, dh = x.shape
    _, _, g, d = B.shape

    dt = compute_dt(dt, dt_bias, time_step_limit)
    repeats = h // g
    A = -mx.exp(A_log).astype(dt.dtype)
    dtA = dt * A.reshape(1, 1, -1)
    dtx = dt.reshape(b, l, h, 1) * x

    def _step(dtx, dtA, B, C, state, mask):
        s = dtx.shape[1]
        B = mx.transpose(B, (0, 2, 3, 1))

        CB = mx.swapaxes(C, 1, 2) @ B
        CB = mx.repeat(CB, repeats, axis=1)

        decay = mx.exp(segsum(dtA.swapaxes(1, 2), mask=mask))

        surrogate_attention_matrix = mx.tril(CB * decay, 0)

        y = surrogate_attention_matrix @ dtx.swapaxes(1, 2)
        y = mx.swapaxes(y, 1, 2)

        if lengths is not None:
            pos = mx.maximum(mx.minimum(lengths, step) - 1, 0)
            pos = mx.expand_dims(pos, (1, 2, 3))
            decay = mx.take_along_axis(decay, pos, axis=2)
        else:
            decay = decay[:, :, -1:, :]

        decay = decay.transpose(0, 3, 1, 2)
        B = mx.repeat(B, h // g, axis=1).swapaxes(2, 3)
        dtxdecay = dtx * decay
        dtxdecay = dtxdecay.swapaxes(1, 2).swapaxes(2, 3)

        next_state = dtxdecay @ B

        if state is not None:
            exp_dtA_cumsum = mx.exp(mx.cumsum(dtA, axis=-2))
            next_state += exp_dtA_cumsum[:, -1, :, None, None] * state
            C = C.reshape(b, s, g, 1, d, 1)
            y_prev = (
                (state.reshape((b, 1, g, repeats, dh, d)) @ C).squeeze(-1).flatten(2, 3)
            )
            y += exp_dtA_cumsum[..., None] * y_prev
        if lengths is not None and state is not None:
            next_state = mx.where(
                mx.expand_dims(lengths < 0, (1, 2, 3)), state, next_state
            )

        return y.astype(x.dtype), next_state

    ys = []
    for i in range(0, l, step):
        y, state = _step(
            dtx[:, i : i + step],
            dtA[:, i : i + step],
            B[:, i : i + step],
            C[:, i : i + step],
            state,
            None if mask is None else mask[..., i : i + step],
        )
        if lengths is not None:
            lengths = lengths - step
        ys.append(y)
    y = mx.concatenate(ys, axis=1) + x * D.reshape(1, 1, h, 1)
    return y, state


def ssm_update(
    hidden_states: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    time_step_limit: Tuple[float, float] = (0.001, 100.0),
    mask: Optional[mx.array] = None,
    lengths: Optional[mx.array] = None,
):
    seq_len = hidden_states.shape[1]
    if (
        seq_len > 1
        or state is None
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
    ):
        return ssm_attn(
            hidden_states,
            A_log,
            B,
            C,
            D,
            dt,
            dt_bias,
            state,
            time_step_limit,
            mask=mask,
            lengths=lengths,
        )
    else:
        return ssm_update_kernel(
            hidden_states,
            A_log,
            B,
            C,
            D,
            dt,
            dt_bias,
            state,
            time_step_limit,
        )
