"""``singleton_quantized_linear`` equals the decode call bit for bit, narrow K included.

Fork-only file. MLX routes a plain ``quantized_matmul`` with K == 128 (or K == 64 and bits >= 2) to
``qmv_quad``, which ``gather_qmm`` never uses, so the single-``gather_qmm`` block differs from the decode
call by one ulp on some inputs; on mlx 0.32.4 so does an output width N that is not a multiple of 8
(``qmv_fast`` with partial rows). The fork computes K <= 128 or N % 8 != 0 rows with the decode call itself;
other shapes keep the ``gather_qmm`` path (see ``docs/upstream-bugs.md`` item 14).
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.models.quantized_verifier import singleton_quantized_linear

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="qmv kernels are Metal")

SEEDS = 40


def _linear(bits, k, bias, seed, n=64, dtype=mx.bfloat16):
    mx.random.seed(seed)
    layer = nn.Linear(k, n, bias=bias)
    layer.weight = (mx.random.normal(layer.weight.shape) * 0.5).astype(dtype)
    if bias:
        layer.bias = mx.random.normal(layer.bias.shape).astype(dtype)
    quantized = nn.QuantizedLinear.from_linear(layer, group_size=64, bits=bits)
    mx.eval(quantized.parameters())
    return quantized


def _rows_equal(linear, x):
    out = singleton_quantized_linear(linear, x)
    assert out is not None and out.shape == (*x.shape[:2], linear.weight.shape[0])
    for b in range(x.shape[0]):
        for t in range(x.shape[1]):
            want = linear(x[b : b + 1, t : t + 1])
            if not bool(mx.array_equal(out[b : b + 1, t : t + 1], want).item()):
                return False
    return True


def _baseline(linear, x):
    """The pre-fix single ``mx.gather_qmm`` computation."""
    indices = mx.zeros((*x.shape[:2], 1), dtype=mx.int32)
    out = (
        mx.gather_qmm(
            mx.expand_dims(mx.contiguous(x), (-2, -3)),
            linear.weight[None],
            linear.scales[None],
            None if linear.biases is None else linear.biases[None],
            rhs_indices=indices,
            transpose=True,
            group_size=linear.group_size,
            bits=linear.bits,
            mode=linear.mode,
            sorted_indices=False,
        )
        .squeeze(-2)
        .squeeze(-2)
    )
    if "bias" in linear:
        out = out + linear["bias"]
    return out


@metal
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("shape", [(1, 2), (2, 3)])
@pytest.mark.parametrize("k", [64, 128])
@pytest.mark.parametrize("bits", [2, 4, 8])
def test_narrow_rows_equal_decode_bitwise(bits, k, shape, bias):
    for seed in range(SEEDS):
        linear = _linear(bits, k, bias, seed)
        x = mx.random.normal((*shape, k)).astype(mx.bfloat16)
        assert _rows_equal(linear, x), (bits, k, shape, bias, seed)


@metal
def test_the_baseline_gather_path_mismatches_for_8bit_k128():
    # the regression is real on the installed MLX, so the test above is not vacuous
    mismatched = 0
    for seed in range(SEEDS):
        linear = _linear(8, 128, False, seed)
        x = mx.random.normal((1, 2, 128)).astype(mx.bfloat16)
        out = _baseline(linear, x)
        for t in range(2):
            want = linear(x[:, t : t + 1])
            mismatched += not bool(mx.array_equal(out[:, t : t + 1], want).item())
    assert mismatched > 0


@metal
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("n", [7, 12, 100])
@pytest.mark.parametrize("bits", [4, 8])
def test_odd_output_width_rows_equal_decode_bitwise(bits, n, dtype, bias):
    for seed in range(SEEDS):
        linear = _linear(bits, 512, bias, seed, n=n, dtype=dtype)
        x = mx.random.normal((1, 3, 512)).astype(dtype)
        assert _rows_equal(linear, x), (bits, n, dtype, bias, seed)


@metal
@pytest.mark.parametrize("bias", [False, True])
def test_wide_k_keeps_the_gather_path(bias):
    linear = _linear(4, 512, bias, 0)
    x = mx.random.normal((2, 3, 512)).astype(mx.bfloat16)
    out = singleton_quantized_linear(linear, x)
    assert out is not None
    assert bool(mx.array_equal(out, _baseline(linear, x)).item())
    for b in range(2):
        for t in range(3):
            want = linear(x[b : b + 1, t : t + 1])
            assert bool(mx.array_equal(out[b : b + 1, t : t + 1], want).item())


@metal
def test_unsupported_inputs_return_none():
    linear = _linear(4, 128, False, 0)
    assert singleton_quantized_linear(linear, mx.zeros((2, 128))) is None
    assert singleton_quantized_linear(linear, mx.zeros((1, 2, 64))) is None
    assert singleton_quantized_linear(nn.Linear(128, 64), mx.zeros((1, 2, 128))) is None
