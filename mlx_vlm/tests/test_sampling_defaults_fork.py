"""Fork: upstream #2545's seeded sampling-defaults cases against the fork's sampler.

`test_sampling_defaults.py::test_python_sampling_defaults` is kept verbatim and
its five seeded, sampled parametrizations are strict-xfailed in conftest.py
(FORK_DIVERGENT_UPSTREAM_TESTS): the fork's `_PositionedTargetSampler` also
receives `min_p` (C26), so upstream's exact call assertion cannot match. This
file runs the same seeded cases with the fork's call shape, so checkpoint-default
resolution on that path, the emitted tokens and config immutability stay pinned.
"""

import contextlib
from unittest.mock import patch

import mlx.core as mx
import pytest

from mlx_vlm.generate import ar, dispatch
from mlx_vlm.tests.test_diffusion_models import FakeProcessor
from mlx_vlm.tests.test_sampling_defaults import (  # noqa: F401  (fixture)
    _sampling_model,
    sampling_case,
)


@pytest.mark.parametrize("entrypoint", ["generate_step", "stream_generate", "generate"])
def test_seeded_sampling_defaults_reach_the_fork_sampler(sampling_case, entrypoint):
    config, overrides, expected = sampling_case
    seed = 42
    original_config = vars(config).copy()
    model = _sampling_model(config)
    options = dict(max_tokens=32, seed=seed, prompt_cache=[])
    with (
        patch.object(ar, "make_sampler", wraps=ar.make_sampler) as sampler,
        patch.object(
            ar, "_PositionedTargetSampler", wraps=ar._PositionedTargetSampler
        ) as positioned,
        patch.object(dispatch, "wired_limit", return_value=contextlib.nullcontext()),
    ):
        mx.random.seed(42)
        if entrypoint == "generate_step":
            tokens = [
                token
                for token, _ in ar.generate_step(
                    mx.array([[1]]), model, None, None, **options, **overrides
                )
            ]
        else:
            result = getattr(dispatch, entrypoint)(
                model,
                FakeProcessor(),
                "",
                input_ids=mx.array([[1]]),
                **options,
                **overrides,
            )
            if entrypoint == "generate":
                tokens = result.token_ids
            else:
                tokens = [
                    chunk.token for chunk in result if chunk.finish_reason is None
                ]
    if expected["temperature"] > 0:
        positioned.assert_called_once_with(**expected, seed=seed, min_p=0.0)
        sampler.assert_not_called()
    else:
        actual = sampler.call_args.kwargs
        assert (actual["temp"], actual["top_p"], actual["top_k"]) == tuple(
            expected.values()
        )
        positioned.assert_not_called()
    assert len(tokens) == 32
    assert set(tokens) == ({1, 2} if expected["temperature"] > 0 else {2})
    assert vars(config) == original_config
