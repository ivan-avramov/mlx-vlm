"""MTP speculative-verify rollback contract for `nemotron_h`.

Before this, serving a `nemotron_h` target under `--draft-kind mtp` crashed in
`mlx_vlm/speculative/mtp.py`'s `_mtp_rounds_batch` with ``RuntimeError:
LanguageModel does not implement rollback_speculative_cache`` -- nemotron_h's
mamba2 layers keep a fixed-size recurrent state per layer (an ArraysCache
``(conv_state, ssm_state)``) that a plain KV-cache ``.trim()`` cannot restore
after a speculative round partially rejects a verify block.

``mlx_vlm/models/recurrent_rollback.py``'s ``RecurrentStateRollbackMixin``
(mixed into ``mlx_vlm.models.nemotron_h.language.LanguageModel``) implements
the five-method contract mtp.py needs
(``speculative_verify_logits``/``_hidden``, ``speculative_logits_from_hidden``,
``speculative_argmax_from_hidden``, ``rollback_speculative_cache``). This file
proves it is EXACT: rolling back a verify block to `accepted` must leave every
layer's cache identical -- allclose arrays, equal KV offsets -- to plain
sequential decoding of exactly the first `accepted + 1` block tokens, for
`accepted` anywhere in `[0, block_size - 1]`, matching what the model would
have produced had the drafter proposed nothing beyond the accepted prefix.

CPU-only and tiny synthetic weights throughout: a real benchmark run may be
using this machine's GPU concurrently, so this module must never touch it and
never load a real checkpoint.
"""

import mlx.core as mx

# Fork: this benchmark host may have a live GPU benchmark running; every test
# in this module must stay off it.
mx.set_default_device(mx.cpu)

import pytest

from mlx_vlm.models.cache import ArraysCache, KVCache
from mlx_vlm.models.nemotron_h.config import ModelConfig
from mlx_vlm.models.nemotron_h.language import LanguageModel

HIDDEN_SIZE = 16
VOCAB_SIZE = 32


def _config(**overrides) -> ModelConfig:
    # "M*-": one mamba2 layer, one attention layer, one MLP layer -- the
    # smallest pattern that exercises every cache kind rollback has to know
    # about (recurrent ArraysCache, trimmable KVCache, and the `None` cache
    # of a stateless layer).
    kwargs = dict(
        model_type="nemotron_h",
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=32,
        num_hidden_layers=3,
        max_position_embeddings=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        attention_bias=False,
        mamba_num_heads=2,
        mamba_head_dim=8,
        mamba_proj_bias=False,
        ssm_state_size=8,
        conv_kernel=4,
        n_groups=1,
        mlp_bias=False,
        layer_norm_epsilon=1e-5,
        use_bias=False,
        use_conv_bias=True,
        hybrid_override_pattern="M*-",
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def _model() -> LanguageModel:
    mx.random.seed(0)
    return LanguageModel(_config())


def _greedy_sampler(logits):
    return mx.argmax(logits, axis=-1)


def _snapshot(caches, row=None):
    """Materialize a comparable snapshot of every cache's live state.

    KVCache's backing buffer is a mutable, over-allocated ring that later
    writes can alias into (`step=256` growth, in-place slice assignment), so
    a snapshot used for a later comparison must force a real copy rather than
    hold a reference into the live buffer.

    ``row``, when given, slices every array down to that one batch row (kept
    as a length-1 leading dim) so a batched cache can be compared against a
    single-row reference cache.
    """

    def sl(x):
        return x if row is None else x[row : row + 1]

    out = []
    for c in caches:
        if c is None:
            out.append(None)
        elif isinstance(c, ArraysCache):
            out.append(("ssm", mx.array(sl(c[0])), mx.array(sl(c[1]))))
        elif isinstance(c, KVCache):
            keys, values = c.state
            out.append(("kv", c.offset, mx.array(sl(keys)), mx.array(sl(values))))
        else:
            raise AssertionError(f"unexpected cache type in test: {type(c)}")
    return out


def _assert_snapshots_match(a, b):
    assert len(a) == len(b)
    for entry_a, entry_b in zip(a, b):
        if entry_a is None:
            assert entry_b is None
            continue
        assert entry_a[0] == entry_b[0]
        if entry_a[0] == "ssm":
            _, conv_a, ssm_a = entry_a
            _, conv_b, ssm_b = entry_b
            assert mx.allclose(conv_a, conv_b, rtol=1e-5, atol=1e-6).item()
            assert mx.allclose(ssm_a, ssm_b, rtol=1e-5, atol=1e-6).item()
        else:
            _, offset_a, keys_a, values_a = entry_a
            _, offset_b, keys_b, values_b = entry_b
            assert offset_a == offset_b
            assert mx.allclose(keys_a, keys_b, rtol=1e-5, atol=1e-6).item()
            assert mx.allclose(values_a, values_b, rtol=1e-5, atol=1e-6).item()


def _decode_sequential(model, prefix, tokens):
    """Plain one-token-at-a-time decode of `tokens` after `prefix`."""
    cache = model.make_cache()
    model(prefix, cache=cache)
    last_out = None
    for tok in tokens:
        last_out = model(mx.array([[tok]], dtype=mx.int32), cache=cache)
    return cache, last_out


PREFIX = mx.array([[3, 5, 7]], dtype=mx.int32)
BLOCK = [11, 12, 13, 14]  # 4-token verify block
BLOCK_SIZE = len(BLOCK)


class TestKnownPositiveRollback:
    def test_partial_accept_matches_sequential_decode_and_continuation(self):
        model = _model()

        # Path A: sequential decode through the first 2 block tokens
        # (accepted=1 -> n=2 tokens kept). This is the ground truth.
        cache_seq, _ = _decode_sequential(model, PREFIX, BLOCK[:2])
        seq_snapshot = _snapshot(cache_seq)

        # Path B: verify the whole 4-token block in one call (capturing
        # per-position recurrent state), then roll back to accepted=1.
        cache_verify = model.make_cache()
        model(PREFIX, cache=cache_verify)
        block_input = mx.array([BLOCK], dtype=mx.int32)
        hidden, shared_kv, states = model.speculative_verify_hidden(
            block_input, cache_verify
        )
        mx.eval(hidden)
        assert shared_kv == {}

        max_a = model.rollback_speculative_cache(
            cache_verify, states, accepted=1, block_size=BLOCK_SIZE
        )
        assert max_a == 1

        verify_snapshot = _snapshot(cache_verify)
        _assert_snapshots_match(seq_snapshot, verify_snapshot)

        # Continuing both paths with the same next token (t3, the block's
        # 3rd token) must now produce identical logits.
        next_tok = mx.array([[BLOCK[2]]], dtype=mx.int32)
        out_seq = model(next_tok, cache=cache_seq)
        out_verify = model(next_tok, cache=cache_verify)
        mx.eval(out_seq.logits, out_verify.logits)
        assert mx.allclose(
            out_seq.logits, out_verify.logits, rtol=1e-5, atol=1e-6
        ).item()

    def test_accepted_zero(self):
        model = _model()

        cache_seq, _ = _decode_sequential(model, PREFIX, BLOCK[:1])
        seq_snapshot = _snapshot(cache_seq)

        cache_verify = model.make_cache()
        model(PREFIX, cache=cache_verify)
        block_input = mx.array([BLOCK], dtype=mx.int32)
        hidden, _, states = model.speculative_verify_hidden(block_input, cache_verify)
        mx.eval(hidden)

        max_a = model.rollback_speculative_cache(
            cache_verify, states, accepted=0, block_size=BLOCK_SIZE
        )
        assert max_a == 0

        _assert_snapshots_match(seq_snapshot, _snapshot(cache_verify))

    def test_accepted_all(self):
        """accepted == block_size - 1: nothing is rejected."""
        model = _model()

        cache_seq, _ = _decode_sequential(model, PREFIX, BLOCK)
        seq_snapshot = _snapshot(cache_seq)

        cache_verify = model.make_cache()
        model(PREFIX, cache=cache_verify)
        block_input = mx.array([BLOCK], dtype=mx.int32)
        hidden, _, states = model.speculative_verify_hidden(block_input, cache_verify)
        mx.eval(hidden)

        max_a = model.rollback_speculative_cache(
            cache_verify, states, accepted=BLOCK_SIZE - 1, block_size=BLOCK_SIZE
        )
        assert max_a == BLOCK_SIZE - 1

        _assert_snapshots_match(seq_snapshot, _snapshot(cache_verify))


PREFIX_ROW0 = [3, 5, 7]
PREFIX_ROW1 = [4, 6, 9]
BLOCK_ROW0 = [11, 12, 13, 14]
BLOCK_ROW1 = [21, 22, 23, 24]


class TestBatchedUniformAcceptRollback:
    """B=2, same `accepted` value for every row (the only batched case this
    mixin supports exactly -- see rollback_speculative_cache's docstring).
    Ground truth is each row decoded independently and sequentially, exactly
    as TestKnownPositiveRollback does for B=1; the two rows use distinct
    prefixes/block tokens so the comparison isn't trivially symmetric.
    """

    @pytest.mark.parametrize("accepted", [0, 1, 3])
    def test_batch_two_uniform_acceptance_matches_per_row_sequential_decode(
        self, accepted
    ):
        model = _model()

        cache_row0, _ = _decode_sequential(
            model,
            mx.array([PREFIX_ROW0], dtype=mx.int32),
            BLOCK_ROW0[: accepted + 1],
        )
        cache_row1, _ = _decode_sequential(
            model,
            mx.array([PREFIX_ROW1], dtype=mx.int32),
            BLOCK_ROW1[: accepted + 1],
        )

        cache_verify = model.make_cache()
        prefix_batch = mx.array([PREFIX_ROW0, PREFIX_ROW1], dtype=mx.int32)
        model(prefix_batch, cache=cache_verify)
        block_batch = mx.array([BLOCK_ROW0, BLOCK_ROW1], dtype=mx.int32)
        hidden, shared_kv, states = model.speculative_verify_hidden(
            block_batch, cache_verify
        )
        mx.eval(hidden)
        assert shared_kv == {}

        max_a = model.rollback_speculative_cache(
            cache_verify, states, accepted=[accepted, accepted], block_size=BLOCK_SIZE
        )
        assert max_a == accepted

        _assert_snapshots_match(_snapshot(cache_row0), _snapshot(cache_verify, row=0))
        _assert_snapshots_match(_snapshot(cache_row1), _snapshot(cache_verify, row=1))


class TestVerifyReturnContract:
    def test_speculative_verify_hidden_returns_three_tuple(self):
        model = _model()
        cache = model.make_cache()
        model(PREFIX, cache=cache)

        result = model.speculative_verify_hidden(
            mx.array([BLOCK], dtype=mx.int32), cache
        )
        assert len(result) == 3
        hidden, shared_kv, states = result
        mx.eval(hidden)
        assert hidden.shape == (1, BLOCK_SIZE, HIDDEN_SIZE)
        assert shared_kv == {}
        # states is aligned with `cache`: None at the attention/MLP-derived
        # slots, a list of per-position (conv_state, ssm_state) pairs at the
        # mamba slot.
        assert len(states) == len(cache)
        mamba_slots = [i for i, c in enumerate(cache) if isinstance(c, ArraysCache)]
        assert len(mamba_slots) == 1
        (mamba_idx,) = mamba_slots
        for i, c in enumerate(cache):
            if i == mamba_idx:
                assert isinstance(states[i], list)
                assert len(states[i]) == BLOCK_SIZE
                for conv_state, ssm_state in states[i]:
                    assert conv_state.shape[0] == 1
                    assert ssm_state.shape[0] == 1
            else:
                assert states[i] is None

    def test_speculative_verify_logits_returns_four_tuple(self):
        model = _model()
        cache = model.make_cache()
        model(PREFIX, cache=cache)

        result = model.speculative_verify_logits(
            mx.array([BLOCK], dtype=mx.int32), cache, _greedy_sampler
        )
        assert len(result) == 4
        hidden, shared_kv, states, target_tokens = result
        mx.eval(hidden, target_tokens)
        assert hidden.shape == (1, BLOCK_SIZE, HIDDEN_SIZE)
        assert shared_kv == {}
        assert len(states) == len(cache)
        assert target_tokens.shape == (1, BLOCK_SIZE)

    def test_speculative_logits_from_hidden_matches_model_own_logits(self):
        """`hidden_sink` holds PRE-`norm_f` hidden (the drafter needs it), but
        the model's real logits are `lm_head(norm_f(hidden))`
        (Model.__call__, language.py). `speculative_logits_from_hidden` must
        re-apply `norm_f` itself via `speculative_final_norm` -- comparing
        against a manual bare `lm_head(hidden)` (as an earlier version of
        this test did) is vacuous, since that's exactly the unfixed bug's own
        computation. Comparing against the model's OWN logits from the same
        forward is the real check.

        `norm_f.weight` is forced non-uniform: RMSNorm under a uniform
        (all-ones, the default init) weight is a pure positive-scalar rescale
        of the hidden vector, so `lm_head(norm_f(h))` and `lm_head(h)` would
        still differ in VALUE but happen to preserve the ratio structure less
        obviously than under a non-uniform weight -- pin non-uniform so this
        test exercises what `norm_f` actually does per-dimension.
        """
        model = _model()
        model.backbone.norm_f.weight = mx.random.uniform(
            low=0.5, high=2.0, shape=(HIDDEN_SIZE,)
        )
        cache = model.make_cache()
        out = model(PREFIX, cache=cache, return_hidden=True)
        mx.eval(out.logits)

        actual = model.speculative_logits_from_hidden(out.hidden_states[-1])
        mx.eval(actual)
        assert mx.allclose(out.logits, actual, rtol=1e-5, atol=1e-6).item()

    def test_speculative_argmax_from_hidden_matches_model_own_logits_argmax(self):
        """Same bug as above, but for argmax specifically: a UNIFORM
        `norm_f.weight` rescales every logit by the same positive scalar, so
        `argmax(lm_head(h))` and `argmax(lm_head(norm_f(h)))` coincide even
        when the missing-norm bug is present -- that coincidence is exactly
        what hid F1 behind the original (uniform-weight) version of this
        test. A non-uniform weight changes the per-dimension weighting before
        the lm_head matmul, so a missing `norm_f` application can actually
        flip the argmax and this test can actually catch it.
        """
        model = _model()
        model.backbone.norm_f.weight = mx.random.uniform(
            low=0.5, high=2.0, shape=(HIDDEN_SIZE,)
        )
        cache = model.make_cache()
        out = model(PREFIX, cache=cache, return_hidden=True)
        mx.eval(out.logits)
        expected = mx.argmax(out.logits, axis=-1)

        actual = model.speculative_argmax_from_hidden(out.hidden_states[-1])
        mx.eval(expected, actual)
        assert mx.array_equal(expected, actual).item()


class TestRollbackErrorHandling:
    def test_missing_capture_raises_loud(self):
        """Rolling back without having run a captured verify must not
        silently no-op -- it must fail loud."""
        model = _model()
        cache = model.make_cache()
        model(PREFIX, cache=cache)
        states = [None] * len(cache)

        with pytest.raises(RuntimeError, match="missing recurrent-state"):
            model.rollback_speculative_cache(
                cache, states, accepted=0, block_size=BLOCK_SIZE
            )

    def test_states_none_fails_early_with_explicit_cause(self):
        """`states=None` (not a list of Nones -- genuinely absent) happens
        when `hasattr(lm, "rollback_speculative_cache")` lets a non-mtp
        draft_kind (dflash/eagle3/suffix) route through mtp.py's rollback
        call without ever having called this mixin's own
        speculative_verify_hidden/_logits (the only place that populates
        `gdn_states`/`capture_recurrent_states`). That must fail immediately
        with a message naming the actual cause, not the generic per-cache
        "missing recurrent-state snapshot" message from a later loop
        iteration."""
        model = _model()
        cache = model.make_cache()
        model(PREFIX, cache=cache)

        # Matched on wording unique to the early, dedicated check -- the
        # generic per-cache "missing recurrent-state snapshot" message (hit
        # if this check were removed and states=None fell through to the
        # loop) also happens to mention capture_recurrent_states, so a
        # looser match wouldn't actually distinguish "failed early and
        # clearly" from "failed late with a less specific message."
        with pytest.raises(
            RuntimeError, match="verify ran without capture_recurrent_states"
        ):
            model.rollback_speculative_cache(
                cache, None, accepted=0, block_size=BLOCK_SIZE
            )

    def test_non_uniform_batch_accept_fails_loud(self):
        model = _model()
        cache = model.make_cache()

        with pytest.raises(NotImplementedError, match="uniform acceptance"):
            model.rollback_speculative_cache(
                cache, [None] * len(cache), accepted=[0, 2], block_size=BLOCK_SIZE
            )

    def test_accepted_exceeding_block_size_raises(self):
        model = _model()
        cache = model.make_cache()

        with pytest.raises(ValueError, match="exceeding block_size"):
            model.rollback_speculative_cache(
                cache, [None] * len(cache), accepted=BLOCK_SIZE, block_size=BLOCK_SIZE
            )
