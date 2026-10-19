"""E1c in-process observers: full-state hash, neutral probe, ON / OFF recorders.

- ``state_hash(caches)``: ONE SHA-256 over the complete logical valid state of a cache list. Per leaf (in
  ``iter_leaf_caches`` order): its type, ``offset``, ``meta_state``, its type-specific logical metadata
  (``ArraysCache``: the effective ``left_padding`` / ``lengths`` mask inputs; ``TurboQuantKVCache``: the codec
  mode and each codec's kind, bits and ``_snapshot_kv_codec`` coordinates — dim and split ``low_idx`` /
  ``high_idx``), then every array of its VALID state (``.state``, which slices KV caches to ``offset``;
  TurboQuant sliced with ``_slice_state`` directly so its ``_cached_state`` is never touched) with dtype, shape
  and exact bytes. Preallocated tails are never hashed; an unknown state element or an open speculative
  transaction raises (fail closed). ``ArraysCache.metadata_revision`` is NOT hashed: it is a derived-metadata
  invalidation counter that every speculative commit / abort bumps (every ON block, every probe), not state.
- ``probe(lm, caches, token)``: a transactional width-one verify forward of ``token`` (the production
  ``_mtp_verify_target``), its full logit row, then ``abort()``.
- ``Recorder(mode)``: patches the production sampler (``_PositionedTargetSampler.sample_target``) to record
  every row it sees by emitted position — ON: the prefill row and every verifier row; OFF: the plain rows,
  with ``force_tokens`` replacing the sampled tokens so the replay walks ON's history. It captures the live
  cache from ``dispatch.generate_step`` and snapshots the state (hash, probe, hash again = neutrality): ON at
  the prefill row and after EVERY target commit (block commits and width-one pending-token commits); OFF at
  the sampler call whose cache offset is in ``hash_offsets``. Everything is restored on exit.
"""

import hashlib
import sys

from . import compare

_UINT = {1: "uint8", 2: "uint16", 4: "uint32", 8: "uint64"}


def _mx():
    import mlx.core as mx

    return mx


def _flatten(obj, out):
    mx = _mx()
    if obj is None:
        out.append(("none",))
    elif isinstance(obj, mx.array):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        out.append(("seq", type(obj).__qualname__, len(obj)))
        for item in obj:
            _flatten(item, out)
    elif getattr(obj, "__dataclass_fields__", None):
        out.append(("dataclass", type(obj).__qualname__))
        for name in obj.__dataclass_fields__:
            _flatten(getattr(obj, name), out)
    elif type(obj) in (bool, int, float, str):
        out.append(("scalar", type(obj).__name__, repr(obj)))
    else:
        raise TypeError(f"unknown cache state element {type(obj).__qualname__}")


def _logical_metadata(leaf):
    """Type-specific logical metadata outside ``state`` / ``meta_state`` (see the module docstring)."""
    from mlx_vlm.models.cache import ArraysCache

    tq = sys.modules.get("mlx_vlm.turboquant")
    if tq is not None and isinstance(leaf, tq.TurboQuantKVCache):

        def codec(c):
            if c is None:
                return None
            return (type(c).__qualname__, c.bits, tq._snapshot_kv_codec(c))

        return (
            "turboquant",
            leaf._kv_quant_mode,
            codec(leaf.key_codec),
            codec(leaf.value_codec),
        )
    if isinstance(leaf, ArraysCache):
        if leaf.is_speculating:
            raise RuntimeError("state_hash: active speculative transaction on a leaf")
        return ("arrays", leaf.left_padding, leaf.lengths)
    return None


def _valid_state(leaf):
    tq = sys.modules.get("mlx_vlm.turboquant")
    if tq is not None and isinstance(leaf, tq.TurboQuantKVCache):
        if leaf.keys is None:
            return (None, None)
        return (
            tq._slice_state(leaf.keys, leaf.offset),
            tq._slice_state(leaf.values, leaf.offset),
        )
    return leaf.state


def _hash_array(h, array):
    mx = _mx()
    import numpy as np

    mx.eval(array)
    dtype = str(array.dtype).split(".")[-1]
    h.update(f"A|{dtype}|{tuple(array.shape)}|".encode())
    if array.size == 0:
        return
    if array.dtype == mx.bool_:
        raw = np.array(array.astype(mx.uint8))
    else:
        raw = np.array(array.view(getattr(mx, _UINT[array.itemsize])))
    h.update(np.ascontiguousarray(raw).tobytes())


def _cache_offset(caches):
    from mlx_vlm.speculative import mtp

    return mtp._mtp_cache_offset_max(caches)


def state_hash(caches):
    """Hex SHA-256 of the complete logical valid state (see the module docstring)."""
    from mlx_vlm.speculative.cache_state import iter_leaf_caches

    h = hashlib.sha256()
    for index, leaf in enumerate(iter_leaf_caches(caches)):
        kind = f"{type(leaf).__module__}.{type(leaf).__qualname__}"
        offset = getattr(leaf, "offset", None)
        offset = offset.tolist() if hasattr(offset, "tolist") else offset
        meta = getattr(leaf, "meta_state", None)
        h.update(f"L|{index}|{kind}|{offset!r}|{meta!r}|".encode())
        parts = []
        _flatten(_logical_metadata(leaf), parts)
        _flatten(_valid_state(leaf), parts)
        for part in parts:
            if isinstance(part, tuple):
                h.update(repr(part).encode())
            else:
                _hash_array(h, part)
    return h.hexdigest()


def probe(lm, caches, token):
    """The full logit row of a transactional width-one verify forward of ``token``; then ``abort()``."""
    mx = _mx()
    from mlx_vlm.speculative import mtp

    with mx.stream(mtp.generation_stream):
        verify = mtp._mtp_verify_target(
            lm,
            mx.array([[int(token)]], dtype=mx.int32),
            caches,
            None,
            sample_target_tokens=False,
        )
        logits = mtp._mtp_logits_from_hidden(lm, verify.hidden)
    try:
        mx.eval(logits)
        return compare.as_record(logits.reshape(-1))
    finally:
        verify.abort()


def compare_states(on, off):
    """Match ON's prompt-end + commit snapshots against OFF's snapshots at the same offsets."""
    want = {}
    if on.prompt_end is not None:
        want[on.prompt_end["offset"]] = on.prompt_end
    for snap in on.commits:
        want[snap["offset"]] = snap
    report = dict(
        compared=0,
        missing_off=[],
        mismatched_offsets=[],
        probe_mismatches=[],
        non_neutral=[],
    )
    for offset in sorted(want):
        mine, theirs = want[offset], off.states.get(offset)
        if theirs is None:
            report["missing_off"].append(offset)
            continue
        report["compared"] += 1
        if mine["sha"] != theirs["sha"]:
            report["mismatched_offsets"].append(offset)
        if (mine["probe"] is None) != (theirs["probe"] is None) or (
            mine["probe"] is not None
            and compare._bits(mine["probe"])[0] != compare._bits(theirs["probe"])[0]
            or mine["probe"] is not None
            and not (
                compare._bits(mine["probe"])[1] == compare._bits(theirs["probe"])[1]
            ).all()
        ):
            report["probe_mismatches"].append(offset)
        if not (mine["neutral"] and theirs["neutral"]):
            report["non_neutral"].append(offset)
    report["pass"] = bool(
        report["compared"] > 0
        and not report["missing_off"]
        and not report["mismatched_offsets"]
        and not report["probe_mismatches"]
        and not report["non_neutral"]
    )
    return report


class Recorder:
    """See the module docstring. ``mode`` is ``"on"`` or ``"off"``."""

    def __init__(
        self,
        mode,
        *,
        force_tokens=None,
        hash_offsets=None,
        lm=None,
        probe_token=None,
        before_hash=None,
    ):
        if mode not in ("on", "off"):
            raise ValueError(f"mode must be 'on' or 'off', not {mode!r}")
        self.mode = mode
        self.force_tokens = list(force_tokens or [])
        self.hash_offsets = set(hash_offsets or [])
        self.lm = lm
        self.probe_token = probe_token
        self.before_hash = before_hash
        self.rows = []
        self.commits = []
        self.states = {}
        self.prompt_end = None
        self.draft_rounds = 0
        self._cache = None
        self._in_pending = False
        self._patched = []

    # -- snapshots
    def _snapshot(self, caches, kind):
        offset = _cache_offset(caches)
        if self.before_hash is not None:
            self.before_hash(caches, offset)
        sha = state_hash(caches)
        probe_rec, neutral = None, True
        if self.lm is not None and self.probe_token is not None:
            probe_rec = probe(self.lm, caches, self.probe_token)
            neutral = state_hash(caches) == sha
        return dict(offset=offset, sha=sha, probe=probe_rec, neutral=neutral, kind=kind)

    # -- patches
    def _patch(self, owner, name, value):
        self._patched.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def __enter__(self):
        mx = _mx()
        from mlx_vlm.generate import ar, dispatch
        from mlx_vlm.speculative import mtp

        recorder = self
        sample_target = ar._PositionedTargetSampler.sample_target

        def recording_sample_target(
            sampler, logprobs, *, row_ids, positions, seeds=None
        ):
            out = sample_target(
                sampler, logprobs, row_ids=row_ids, positions=positions, seeds=seeds
            )
            mx.eval(logprobs, out)
            rows = logprobs.reshape(len(positions), -1)
            for i, position in enumerate(positions):
                recorder.rows.append((int(position), compare.as_record(rows[i])))
            if recorder.mode == "off" and recorder.force_tokens:
                values = out.tolist()
                for i, position in enumerate(positions):
                    if int(position) < len(recorder.force_tokens):
                        values[i] = int(recorder.force_tokens[int(position)])
                out = mx.array(values, dtype=out.dtype)
            caches = recorder._cache
            if caches is not None:
                if recorder.mode == "on":
                    if recorder.prompt_end is None and list(positions) == [0]:
                        recorder.prompt_end = recorder._snapshot(caches, "prompt_end")
                else:
                    offset = _cache_offset(caches)
                    if (
                        offset in recorder.hash_offsets
                        and offset not in recorder.states
                    ):
                        recorder.states[offset] = recorder._snapshot(caches, "plain")
            return out

        generate_step = dispatch.generate_step

        def capturing_generate_step(*args, **kwargs):
            recorder._cache = kwargs.get("prompt_cache")
            yield from generate_step(*args, **kwargs)

        commit = mtp._MTPVerifyResult.commit
        pending = mtp._mtp_commit_pending_token

        def recording_commit(verify, model, caches, accepted, block_size):
            result = commit(verify, model, caches, accepted, block_size)
            if not recorder._in_pending:
                recorder.draft_rounds += 1
                recorder.commits.append(recorder._snapshot(caches, "block"))
            return result

        def recording_pending(lm, prompt_cache, token, sampler, token_dtype):
            recorder._in_pending = True
            try:
                result = pending(lm, prompt_cache, token, sampler, token_dtype)
            finally:
                recorder._in_pending = False
            recorder.commits.append(recorder._snapshot(prompt_cache, "pending"))
            return result

        self._patch(
            ar._PositionedTargetSampler, "sample_target", recording_sample_target
        )
        self._patch(dispatch, "generate_step", capturing_generate_step)
        if self.mode == "on":
            self._patch(mtp._MTPVerifyResult, "commit", recording_commit)
            self._patch(mtp, "_mtp_commit_pending_token", recording_pending)
        return self

    def __exit__(self, *exc):
        while self._patched:
            owner, name, value = self._patched.pop()
            setattr(owner, name, value)
        return False


class KeyLengths:
    """Every verifier attention block's key-length span (Q19): per call of
    ``Qwen3_5BatchInvariantForward._attention`` with width > 1, the first query sees ``offset + 1`` keys and
    the last ``offset + width`` (``offset`` = the layer cache's offset before the block is appended).
    """

    def __init__(self):
        self.blocks = []
        self._patched = None

    def __enter__(self):
        from mlx_vlm.models.qwen3_5 import speculative_verifier as sv

        cls = sv.Qwen3_5BatchInvariantForward
        real = cls._attention
        blocks = self.blocks

        def recording_attention(verifier, attention, x, mask, cache, *args, **kwargs):
            width = int(x.shape[1])
            if width > 1 and cache is not None:
                offset = getattr(cache, "offset", None)
                offset = (
                    int(offset.max().item()) if hasattr(offset, "max") else int(offset)
                )
                blocks.append(
                    dict(
                        width=width,
                        first_key=offset + 1,
                        last_key=offset + width,
                        cache=type(cache).__name__,
                    )
                )
            return real(verifier, attention, x, mask, cache, *args, **kwargs)

        self._patched = (cls, real)
        cls._attention = recording_attention
        return self

    def __exit__(self, *exc):
        cls, real = self._patched
        cls._attention = real
        return False


def straddles(blocks, threshold):
    """True when some verifier block has queries on both sides of the key-length ``threshold``."""
    return any(b["first_key"] <= threshold < b["last_key"] for b in blocks)
