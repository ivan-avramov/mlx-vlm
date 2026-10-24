"""Per-instance serialization of Hugging Face fast tokenizers.

Fork-only module (docs/proposals/tokenizer-concurrency.md, docs/upstream-bugs.md
item 8). ``transformers`` mutates the tokenizer's Rust object (padding and
truncation) at the top of every encode whenever the call's target differs from the
object's current state; a mutable borrow while another thread is inside
``encode_batch`` raises ``RuntimeError: Already borrowed``. The setter and the
encode must be one critical section, so the lock is taken around ``_encode_plus``
(which calls ``set_truncation_and_padding``) and around the setter itself.

The lock lives in a module-level ``WeakKeyDictionary``, not in the instance, so
``copy.deepcopy``, ``pickle`` and ``save_pretrained`` keep working. The locked
methods are instances of a small module-level callable class, one per (tokenizer,
name), stored on the tokenizer instance under the method name. Its ``__reduce__``
rebuilds the callable from the restored tokenizer and its name; the pickled state
already carries both callables and the mark, so ``copy.deepcopy`` and a pickle round
trip return a serialized tokenizer without running any code on a half-built object.
A pickle of a serialized tokenizer therefore references ``mlx_vlm.tokenizer_lock``:
unpickling needs the fork importable. The lock is a leaf: nothing here calls a server
lock, a cache lock or a model.

``copy.copy`` of a serialized tokenizer is unsupported: the clone's instance-bound
methods act on the original and the two share one Rust backend. Use
``copy.deepcopy``.
"""

import logging
import threading
import weakref

from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

_REQUIRED = ("_encode_plus", "set_truncation_and_padding")
_MARK = "_mlx_vlm_serialized"

_LOCKS = weakref.WeakKeyDictionary()
_LOCKS_GUARD = threading.Lock()


def resolve_fast_tokenizer(obj):
    """The fast tokenizer behind ``obj``, or None.

    Stops at an ``isinstance(x, PreTrainedTokenizerFast)``; otherwise follows only
    ``vars(x).get("tokenizer")`` then ``vars(x).get("_tokenizer")`` (never getattr,
    so forwarding wrappers are not mistaken for tokenizers), at most four steps.
    """
    current = obj
    for _ in range(5):
        if isinstance(current, PreTrainedTokenizerFast):
            return current
        try:
            attrs = vars(current)
        except TypeError:
            return None
        current = attrs.get("tokenizer")
        if current is None:
            current = attrs.get("_tokenizer")
        if current is None:
            return None
    return None


def tokenizer_lock(tok) -> threading.RLock:
    """The lock of one fast tokenizer instance, created on first use.

    ``tok`` must be the fast tokenizer itself (pass ``resolve_fast_tokenizer(obj)``).
    """
    lock = _LOCKS.get(tok)
    if lock is None:
        with _LOCKS_GUARD:
            lock = _LOCKS.get(tok)
            if lock is None:
                lock = threading.RLock()
                _LOCKS[tok] = lock
    return lock


class _Locked:
    """``tok.<name>`` under the instance lock, delegating to the class's own method."""

    __slots__ = ("tok", "name")

    def __init__(self, tok, name):
        self.tok = tok
        self.name = name

    def __call__(self, *args, **kwargs):
        with tokenizer_lock(self.tok):
            return getattr(type(self.tok), self.name)(self.tok, *args, **kwargs)

    def __reduce__(self):
        return (_Locked, (self.tok, self.name))


def is_serialized(obj) -> bool:
    """True when the fast tokenizer behind ``obj`` carries the mark and both
    required names are bound on the instance."""
    tok = resolve_fast_tokenizer(obj)
    if tok is None:
        return False
    attrs = vars(tok)
    if not attrs.get(_MARK):
        return False
    for name in _REQUIRED:
        bound = attrs.get(name)
        # Identity of ``bound.tok`` is deliberately not checked: a shallow copy
        # carries the original's callables and is unsupported, not repaired.
        if not (isinstance(bound, _Locked) and bound.name == name):
            return False
    return True


def serialize_fast_tokenizer(obj, *, logger=logger):
    """Make the fast tokenizer behind ``obj`` safe to call from several threads.

    Returns ``obj`` itself in every case. Nothing resolves (a slow tokenizer, a
    processor without a tokenizer): unchanged. Idempotent. If the class lacks a
    required name, or binding raises, one ERROR is logged and ``obj`` is left
    unprotected and unchanged: serialization is never a silent no-op.
    """
    tok = resolve_fast_tokenizer(obj)
    if tok is None or is_serialized(tok):
        return obj
    for name in _REQUIRED:
        if not callable(getattr(type(tok), name, None)):
            logger.error(
                "Cannot serialize %s: it has no %s; concurrent requests may fail "
                "with 'Already borrowed' (docs/proposals/tokenizer-concurrency.md)",
                type(tok).__name__,
                name,
            )
            return obj
    attrs = vars(tok)
    absent = object()
    prior = {key: attrs.get(key, absent) for key in (*_REQUIRED, _MARK)}
    attempting = None
    try:
        for name in _REQUIRED:
            attempting = name
            setattr(tok, name, _Locked(tok, name))
        attempting = _MARK
        setattr(tok, _MARK, True)
        if not is_serialized(tok):
            # A class whose __setattr__ stores attributes elsewhere accepts the
            # assignments without binding them on the instance.
            raise RuntimeError("binding did not take effect")
    except Exception as exc:  # noqa: BLE001 - reported, never silent
        # Restore through the instance dict: a class that refuses assignment must
        # not be able to refuse the rollback. The exception's text is not logged,
        # it can carry arbitrary content.
        for key, value in prior.items():
            if value is absent:
                attrs.pop(key, None)
            else:
                attrs[key] = value
        logger.error(
            "Cannot serialize %s: binding %s failed (%s); the tokenizer is left "
            "unchanged and concurrent requests may fail with 'Already borrowed' "
            "(docs/proposals/tokenizer-concurrency.md)",
            type(tok).__name__,
            attempting,
            type(exc).__name__,
        )
    return obj
