"""T1 independent behavioural regressions; all execution requires Metal."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="T1 quantized arithmetic requires Metal"
)


def _linear(width, bits, bias):
    dense = nn.Linear(width, 128, bias=bias)
    dense.weight = dense.weight.astype(mx.bfloat16)
    if bias:
        # Nonzero bias must participate in the decode reference too.
        dense.bias = mx.random.normal((128,)).astype(mx.bfloat16)
    return nn.QuantizedLinear.from_linear(
        dense, group_size=64, bits=bits, mode="affine"
    )


def _check_rows(linear, x, seed, output_width=128, dtype=mx.bfloat16):
    from mlx_vlm.models.quantized_verifier import singleton_quantized_linear

    actual = singleton_quantized_linear(linear, x)
    assert actual is not None
    assert actual.shape == (*x.shape[:2], output_width)
    assert actual.dtype == dtype
    for b in range(x.shape[0]):
        for t in range(x.shape[1]):
            expected = linear(x[b : b + 1, t : t + 1])
            assert expected.dtype == dtype
            got = actual[b : b + 1, t : t + 1]
            mx.eval(got, expected)
            np.testing.assert_array_equal(
                np.array(got.view(mx.uint16)),
                np.array(expected.view(mx.uint16)),
                err_msg=f"seed={seed}, batch={b}, time={t}, mlx={mx.__version__}",
            )


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("batch,length", [(1, 2), (2, 3)])
@pytest.mark.parametrize("bias", [False, True], ids=["no-bias", "bias"])
def test_narrow_rows_are_bitwise_singleton_decode(bits, width, batch, length, bias):
    """Reproduces a failure: narrow gather reductions differ from native decode."""
    for seed in range(40):
        mx.random.seed(seed)
        linear = _linear(width, bits, bias)
        x = mx.random.normal((batch, length, width)).astype(mx.bfloat16)
        _check_rows(linear, x, seed)


@pytest.mark.parametrize("output_width", [7, 12, 100])
def test_partial_output_width_fp16_rows_are_bitwise_singleton_decode(output_width):
    """Reproduces B: wide-K partial output rows choose different qmv reductions."""
    for seed in range(40):
        mx.random.seed(seed)
        dense = nn.Linear(512, output_width, bias=False)
        dense.weight = mx.random.normal((output_width, 512)).astype(mx.float16)
        linear = nn.QuantizedLinear.from_linear(
            dense, group_size=64, bits=4, mode="affine"
        )
        x = mx.random.normal((1, 3, 512)).astype(mx.float16)
        _check_rows(linear, x, seed, output_width=output_width, dtype=mx.float16)


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("bias", [False, True], ids=["no-bias", "bias"])
def test_wide_rows_keep_bitwise_singleton_decode(bits, bias):
    """Preserves an invariant: the existing wide affine projection stays exact."""
    for seed in range(4):
        mx.random.seed(seed)
        linear = _linear(512, bits, bias)
        x = mx.random.normal((2, 3, 512)).astype(mx.bfloat16)
        _check_rows(linear, x, seed)


@pytest.mark.parametrize("width", [128, 512])
@pytest.mark.parametrize("kind", ["module", "rank2", "rank4", "width"])
def test_unsupported_inputs_return_none(width, kind):
    """Preserves an invariant: unsupported modules, ranks and K return None."""
    from mlx_vlm.models.quantized_verifier import singleton_quantized_linear

    linear = _linear(width, 8, False)
    shape = (2, 3, width)
    if kind == "module":
        linear = nn.Linear(width, 128)
    elif kind == "rank2":
        shape = (3, width)
    elif kind == "rank4":
        shape = (1, 2, 3, width)
    else:
        shape = (2, 3, width + 64)
    x = mx.zeros(shape, dtype=mx.bfloat16)
    assert singleton_quantized_linear(linear, x) is None
