# Tokenizer concurrency in the server: serialize each fast tokenizer where it is loaded

Specification draft 5, 2026-10-24, for a four-eyes run (light, two design-review rounds at the operator's request); draft 5 amends draft 4 after the fix round, on the operator's word. Fork-only change. Draft 1 proposed a lock at three call sites; round 1 (codex gpt-6-astra, opus) found the site inventory incomplete, the lock held over media and GPU work and a lock-order hazard, and draft 2 moved the lock into the tokenizer instance; draft 3 added the retry warning the operator settled; round 2 (same members, fresh sessions) found the lock storage breaking deep-copy and pickling of processors, an unverified upstream-provenance claim, an under-specified resolution rule and loader fixture, and the audit criterion contradicting the settled constraint. Draft 4 resolved round 2 and the side note on version drift. Draft 5 (after Go, operator-approved) changes two things the fix round surfaced: the serialized methods are callable objects that restore themselves through pickle and deep-copy, replacing plain bound functions, and copy safety is promised for the tokenizer and for holders without the fork's streaming detokenizer, which cannot be deep-copied on the baseline (a pre-existing defect, recorded).

## Goal

The server fails one of two concurrent requests to the same served model with `RuntimeError: Already borrowed` and returns a 500, because the Hugging Face fast tokenizer's Rust object is used from more than one thread with differing padding or truncation settings: `transformers` mutates the Rust object (`enable_padding`, `no_padding`, truncation) whenever a call's target differs from the object's current state, and a mutable borrow while another thread holds a borrow panics. In Premisa's task server this happens at every chat start, when Open WebUI sends title generation and query generation together (five failures in the first minutes of each study launch on 2026-10-23); the failing request is a missing title or a query rewrite that silently falls back. Solving it means that no request to one served model fails or degrades for this reason under concurrent use, that tokenizer and detokenizer outputs are byte-identical to today's, and that no GPU work is serialized by the fix.

## Problem, as found

- Only GIL-releasing encode calls collide. Measured on 2026-10-23 with an in-memory `PreTrainedTokenizerFast` (`tokenizers` 0.22.2; `transformers` 5.14.1 in the fork's `.venv` and 5.15.0 in Premisa's bundled interpreter, same results on both): one thread alternating `padding=True` and `padding=False` (a mutation on every call) against a second thread running `encode` fails with `Already borrowed` within the first iterations; the same mutator against `decode` of 3200 ids, against `batch_decode` of 64 sequences of 400 ids, and against `convert_tokens_to_ids` of 600 tokens never fails over 3000 iterations. `encode_batch` releases the GIL while holding a shared borrow; decoding and id lookups hold the GIL, so the mutator, which also holds the GIL, cannot observe them mid-borrow. The victim set is the encode entry point and the setter.
- The setter and the encode must be one critical section: if thread A sets padding and thread B then sets a different padding before A encodes, A encodes with B's settings and returns a silently wrong result. So the lock must span `set_truncation_and_padding` together with `encode_batch`, which is what `_encode_plus` does; locking the Rust calls individually would be unsafe.
- Provenance. The path both tracebacks of 2026-10-23 end in, the GPU worker's own tokenization inside `_process_cached_request` through `_stream_generate` and `prepare_inputs`, is fork-only (ported in cb585a75; upstream's `generation.py` has no `_process_cached_request` and its comment at the `_tokenizer_lock` block says its design keeps tokenization on the caller side so the worker never races request threads). The fork's retention and anchor encodes in `server/openai.py` and `generate/common.py` are fork-only too. Upstream is exposed only where two request threads use one tokenizer with differing settings outside `_tokenizer_lock`, which its embeddings and reranking routes allow (text against image requests); that exposure is reasoned, not reproduced upstream.
- The tree already carries three partial mitigations for this class: upstream's `_tokenizer_lock` (ff295e36) serializes the caller-side preprocessing against other request threads; the fork's 7e3477b0 added `cached_special_token_encode`, `prewarm_special_token_encodes` and `_encode_retrying_on_borrow_error` in `mlx_vlm/prompt_utils.py`, with fifteen tests in `mlx_vlm/tests/test_tokenizer_concurrency.py`. None reaches the worker's encodes, the request-side encodes outside the preprocessing block, or the direct tokenizer calls in `server/embeddings.py::_embed` and `server/reranking.py`.
- A call-site inventory is the wrong unit: round 1 found sites draft 1 missed in `openai.py`, `dispatch.py`, `common.py`, `embeddings.py`, `reranking.py`, `app.py` and the server token streamer. The lock belongs to the object.

## Scope

Files in scope: new `mlx_vlm/tokenizer_lock.py`; `mlx_vlm/tokenizer_utils.py` (one hook in `load_tokenizer`, one marked import); `mlx_vlm/utils.py` (one hook in `load_processor`, one marked import); `mlx_vlm/prompt_utils.py` (a marked `import logging`, a module logger, and the warning in `_encode_retrying_on_borrow_error`); `mlx_vlm/tests/test_tokenizer_concurrency.py` (tests added, none removed or renamed); `docs/upstream-bugs.md` (new item 8); `.registry-exclusions` (one line, below); `mlx_vlm/tests/test_registry_check.py` (the pinned count of reviewed registry exclusions, five to six, keeping the test's id) and the one `.registry-exclusions` baseline cell in AGENTS.md's exclusion table (50 to 51), both directed by the change review; and only if the symbol audit demands it for the new module, `.symbol-exclusions` with a one-line reason naming this specification. Nothing else. Decided in advance: `token_classification.py` and `privacy_filter.py` construct tokenizers directly but are not server paths and are out of scope; processors that build their own tokenizer are covered through `.tokenizer`.

## Settled constraints

- (Operator, 2026-10-22, Premisa forest P600) A fix the fork needs is made in the fork as marked divergence: every changed definition in a file upstream also has carries a `# Fork:` comment with the reason, and so does every added import; a test; an entry in `docs/upstream-bugs.md`.
- (Fork, AGENTS.md) Failing test first, minimal fix, full suite; the eight audits in `dev/` exit 0 against the fork's last sync point `952d4f6b`, with reasoned exclusion entries; no upstream merge in this change.
- (Operator, 2026-10-22) No in-place package changes in Premisa's built environments; the fix reaches the product through the fork and the stack pin (operator, 2026-10-23: after any fork fix the stack bumps its pin and passes its gate, then Premisa bumps and passes its gate).
- (Operator, 2026-10-23) The fork's borrow-error retry logs a warning each time it fires; no user-facing surface.
- (Designer, from the goal) Outputs unchanged: the change adds synchronization only; no tokenizer call changes its arguments, strategy or result.
- (Designer, from the goal) No lock is held across a model forward pass, a vision encoder call, media preprocessing or any other GPU work.

## Design

Each fast tokenizer instance the fork loads is serialized once, at load, by rebinding on the instance the method that owns the critical section, `_encode_plus`, and the setter it calls, `set_truncation_and_padding`, so that both run under one re-entrant lock per instance. Every caller, present or future, in any thread, then goes through the lock without knowing it. The lock lives outside the instance, so copying, pickling and `save_pretrained` keep working.

### Interface

New module `mlx_vlm/tokenizer_lock.py`, importing only the standard library and `transformers.PreTrainedTokenizerFast` (so `tokenizer_utils.py` and `utils.py` import it without a cycle):

```python
_REQUIRED = ("_encode_plus", "set_truncation_and_padding")

def resolve_fast_tokenizer(obj):
    """The fast tokenizer behind `obj`, or None. Stops at an `isinstance(x,
    PreTrainedTokenizerFast)`; otherwise follows only `vars(x).get("tokenizer")` then
    `vars(x).get("_tokenizer")` (never getattr, so forwarding wrappers are not mistaken
    for tokenizers), at most four steps."""

def tokenizer_lock(tok) -> threading.RLock:
    """The lock of one fast tokenizer instance, from a module-level WeakKeyDictionary
    keyed by the instance, created under a module lock on first use. `tok` must be the
    fast tokenizer itself (callers pass `resolve_fast_tokenizer(obj)`)."""

def serialize_fast_tokenizer(obj, *, logger=module logger):
    """Make the fast tokenizer behind `obj` safe to call from several threads. Resolves
    with `resolve_fast_tokenizer`; returns `obj` unchanged when nothing resolves (a slow
    tokenizer, a processor without a tokenizer). Idempotent: an instance already
    serialized is returned unchanged. For each name in _REQUIRED, stores on the instance
    a callable object `_Locked(instance, name)` whose call takes `tokenizer_lock(instance)`
    and delegates to `type(instance).<name>`, and whose `__reduce__` returns `(_Locked, (instance, name))`, so that
    unpickling and deep-copying a serialized tokenizer rebuild each callable bound to the
    restored object directly, without touching a half-built instance (second fix round, G2);
    then sets `instance._mlx_vlm_serialized = True`. The two bindings and the mark are one
    transaction: if any assignment fails, the prior attributes are restored exactly, one
    ERROR names the class, the attribute and the exception's class (never its text), and
    `obj` is returned unchanged. If the class lacks a required name, one ERROR names the
    class and the name and `obj` is returned unchanged: serialization is never a silent
    no-op."""

def is_serialized(obj) -> bool:
    """True when the fast tokenizer behind `obj` carries the mark and both required
    names are bound on the instance (usable as a one-line check from any interpreter)."""
```

Behaviour: `tokenizer_lock(resolve_fast_tokenizer(p)) is tokenizer_lock(resolve_fast_tokenizer(p.tokenizer)) is tokenizer_lock(resolve_fast_tokenizer(TokenizerWrapper(p.tokenizer)))`; two instances have two locks; `serialize_fast_tokenizer` applied twice leaves the stored callables unchanged; `copy.deepcopy`, a `pickle` round trip and `save_pretrained` of a serialized tokenizer succeed and the restored object `is_serialized` without any further call; a holder object without a detokenizer that keeps the serialized tokenizer deep-copies and saves. A shallow copy (`copy.copy`) of a serialized tokenizer is unsupported: its stored callables still act on the original and both share one Rust backend; this is documented in the module and in item 8 and pinned by a test, not made safe (no caller in the tree shallow-copies a tokenizer). `is_serialized` reports the presence of the callables under both names plus the mark; it does not check the receiver, so it is true for such a clone. `_add_tokens`, `_decode`, `batch_decode`, id lookups and the detokenizers are not rebound: `_add_tokens` runs at load, before serving; the others are measured not to collide, and leaving them free keeps the GPU worker's per-token decode off the lock.

### Where it is called

1. `mlx_vlm/tokenizer_utils.py::load_tokenizer`: the `AutoTokenizer.from_pretrained(...)` result passes through `serialize_fast_tokenizer` before it is wrapped in `TokenizerWrapper`. The import and the call carry `# Fork:` with the reason: the fork's cached-session path tokenizes on the GPU worker, outside upstream's caller-side lock.
2. `mlx_vlm/utils.py::load_processor`: the `AutoProcessor.from_pretrained(...)` result passes through `serialize_fast_tokenizer` (it resolves `processor.tokenizer`, or the processor itself when `AutoProcessor` returned a bare tokenizer). Covers the chat path, `load_embedding_processor` and the reranker loader, which both call `load_processor`, and the embedding loader's tokenizer-only fallback through `load_tokenizer`. Marked the same way.
3. `mlx_vlm/prompt_utils.py::_encode_retrying_on_borrow_error`: one `logger.warning` per caught borrow error, naming the attempt number and the tokenizer's class and nothing from the prompt (prompt text is user data). The file has no logger today: `import logging` and `logger = logging.getLogger(__name__)` are added with inline `# Fork:` markers.
4. Nothing else changes. Upstream's `self._tokenizer_lock` block in `generation.py` stays (it also guards thinking-budget state). The retry, pre-warm and encode cache stay; their retirement belongs to the upstream-merge session.

### Lock order

The instance lock is a leaf: the locked functions call only `type(self)`'s implementation and the Rust object, never a server lock, a cache lock or a model. Every existing lock (`_tokenizer_lock`, `_TOKEN_ENCODE_CACHE_LOCK`, the generation locks) is therefore taken before it and never inside it, in every thread, so no ordering can invert; round 2 traced this and found no inversion. Re-entry (`_encode_plus` calling `set_truncation_and_padding`) re-enters the `RLock`.

### What the GPU worker can wait for

The lock spans the whole of `_encode_plus`, including the conversion of encodings to Python lists, because the setter and the encode must be one critical section. The worker's own encodes therefore wait for whatever encode is in flight on a request thread: microseconds for a chat turn, tens to a few hundred milliseconds for a 100k-token retention encode. This is accepted: it is bounded by one encode, never by GPU or media work, and the alternative, a tokenizer per thread, is a non-goal. Criterion 1(h) records the bound.

### Success path

Open WebUI sends title generation and query generation together for the task model. Request A's thread encodes under upstream's `_tokenizer_lock`; inside, `_encode_plus` takes the instance lock for the setter plus `encode_batch`. The worker, preparing request B's prompt, calls `prepare_inputs`, whose `tokenizer(...)` call blocks on the instance lock until A's encode returns, then proceeds. No borrow error; both requests produce the tokens an uncontended run produces.

### Failure path

A slow tokenizer, or a processor with no fast tokenizer behind it: nothing resolves, nothing is rebound, behaviour unchanged. A fast tokenizer whose class lacks a required name (a future `transformers`): one ERROR log at load naming class and name; serving continues unprotected and the fork's test suite fails on that version (criterion 1(g)), so the dependency session sees it before any pin advances. A deep copy or an unpickled copy of a serialized tokenizer: its callables are rebuilt bound to the copy by `__reduce__`, its state carries the mark, and the copy gets its own lock on first use. A shallow copy: unsupported, see Behaviour. An exception inside the Rust call: the `with` block releases the lock; the exception propagates unchanged.

## Acceptance criteria

1. Tests are added to the existing `mlx_vlm/tests/test_tokenizer_concurrency.py`; no existing test id is removed or renamed (compare `pytest --collect-only -q` between baseline and candidate). The new tests use an in-memory `PreTrainedTokenizerFast` built with `tokenizers` (WordLevel over a few words, a pad token), nothing from the Hugging Face cache; stress prompts are thousands of words long so `encode_batch` holds its borrow long enough to collide. Required:
   (a) Reproduction and fix in one test: threads started behind a `Barrier`, one alternating `padding=True` and `padding=False` on a two-prompt batch, one calling `encode`, exceptions collected from every thread into a list and asserted empty, a stop event so the first error ends the run, running until 5000 iterations per thread or 10 seconds, whichever comes first. The test imports `serialize_fast_tokenizer` with a fallback to the identity function when the import fails, so on the baseline it fails by collision (the recorded baseline run shows `Already borrowed` in 5 of 5 runs), not by `ImportError`; on the candidate it passes.
   (b) Critical section: an event-controlled proxy backend installed as the instance's `_tokenizer`, delegating to the real backend and asserting `tokenizer_lock(tok)._is_owned()` inside `encode_batch`, `enable_padding`, `no_padding`, `enable_truncation` and `no_truncation`; one thread holds an encode open on an event while another attempts a settings-changing call and is shown not to reach the setter until release. Driven through `tok(...)`, `tok.encode(...)` and `tok.apply_chat_template(..., tokenize=True)`.
   (c) Identity: processor-like object, `TokenizerWrapper` and the tokenizer resolve to one lock, calling `tokenizer_lock` on the wrapper's resolution first on a fresh tokenizer; two tokenizers do not share; a second `serialize_fast_tokenizer` leaves the bound function objects unchanged; a forwarding wrapper is never itself rebound.
   (d) Outputs: `input_ids` and `attention_mask` for padded, unpadded and truncated calls on the same prompts are equal before and after serialization, and equal to a single-threaded run during a concurrent stress.
   (e) Loaders: a temporary directory holding the in-memory tokenizer's `tokenizer.json`, a minimal `tokenizer_config.json` and a `config.json` whose `model_type` makes `AutoProcessor` fall back to the tokenizer (the implementer picks one that resolves on the installed version and says which); `load_tokenizer` returns a wrapper whose inner tokenizer `is_serialized`; `load_processor(add_detokenizer=False)` returns an object that `is_serialized`.
   (f) Retry tripwire: with the existing `_FlakyTokenizer`, a retry that succeeds emits exactly one warning record per retried attempt through `caplog`, the record does not contain the prompt text, and an uncontended encode emits none.
   (g) Version guard: the installed fast tokenizer class exposes both `_REQUIRED` names; and `serialize_fast_tokenizer` on a fast-tokenizer-like object whose class lacks one of them logs one ERROR and returns it unchanged.
   (h) Copy safety: `copy.deepcopy` and a `pickle.dumps`/`loads` round trip of a serialized tokenizer succeed, the result `is_serialized` without a further call and passes the stress of (a), and `save_pretrained(tmp_path)` of the tokenizer and of a holder object without a detokenizer succeeds. The fork's `NaiveStreamingDetokenizer` cannot be deep-copied or unpickled on the baseline (a pre-existing defect, recorded for the fork's backlog), so a processor holding the tokenizer through it is outside this criterion.
   (k) Saved artifacts unchanged: `save_pretrained` of a serialized tokenizer and of the tokenizer-fallback processor produces directories byte-for-byte identical to those of an unserialized instance (operator, 2026-10-24: the byte-identical constraint covers what the convert path writes).
   (j) Shallow copy is pinned as unsupported: a `copy.copy` clone's stored callables act on the original, and `serialize_fast_tokenizer(clone)` does not change that.
   (i) Bound: during the stress of (a) with the long prompts, the longest wait any thread spent acquiring the lock is recorded and asserted under one second.
   The new tests together run under 40 seconds on the fork's `.venv`.
2. The full suite passes: `(cd mlx_vlm && ../.venv/bin/python -m pytest -q ./tests --ignore=tests/test_smoke.py)` with 6031 passed, 13 skipped, 2 xfailed plus the new tests, and no baseline test id missing.
3. All eight audits exit 0 against `--upstream-ref 952d4f6b`. The one pre-existing finding (`check_upstream_registries`: `mlx_vlm/server/app.py` no longer re-exports `load_processor`, since 07228726 replaced it with `load_embedding_processor` for mistral3 text-only loads) is resolved by one `.registry-exclusions` line with that reason; fork-marker findings are satisfied by `# Fork:` comments, never by exclusions.
4. `docs/upstream-bugs.md` item 8, in the style of items 6 and 7, titled as a fork-introduced race with partial upstream exposure: the mechanism and measurement; the fork-only paths (cached-session worker tokenization, retention and anchor encodes) and the upstream-exposed one (embeddings and reranking request threads with differing settings, reasoned, not reproduced); the three partial mitigations and why they stay; the divergence (two hooks, the new module, the logger, the markers); the tests; the fix upstream could take (the same serialization in `load_tokenizer` and `load_processor`).
5. The implementer's report names the rebound methods and confirms that no `with tokenizer_lock(...)` or `tokenizer_lock(` call appears outside `tokenizer_lock.py` and the tests.
6. Bundled-interpreter check, run by the designer before the stack's pin bump, not by the helpers: with Premisa's `python-inference` interpreter and the candidate on `PYTHONPATH`, a one-liner loads the in-memory tokenizer, calls `serialize_fast_tokenizer` and asserts `is_serialized`, then runs the (a) stress inline; recorded in the run report.

## Non-goals

Merging upstream; locking `decode`, `batch_decode` or the detokenizers; per-site locks; retiring the retry, pre-warm or encode cache; changing `_tokenizer_lock`; tokenizer copies per thread; performance work beyond criterion 1(i); the per-call `padding_side` writes in `cohere_compass` and `internvl_chat` processors (a different cause, recorded for the operator); the embedding concurrency bound Premisa sets in Open WebUI; any change to Premisa itself.

## Assumptions (designer's)

- Rebinding on the instance is honoured by `__call__`, `encode` and `apply_chat_template(tokenize=True)`: verified on 2026-10-23 on transformers 5.14.1 and 5.15.0 (class `TokenizersBackend`, which is `PreTrainedTokenizerFast`; `_batch_encode_plus` does not exist in 5.x). The test (g) keeps this true for later versions.
- `ProcessorMixin.to_dict` deep-copies the instance dictionary minus the tokenizer attributes, which still reaches a tokenizer retained by a detokenizer (round 2, reproduced by codex); the detokenizer itself cannot be deep-copied on the baseline, so criterion 1(h) covers the tokenizer and holders without a detokenizer, and the detokenizer defect is recorded separately.
- Uncontended `RLock` acquisition per encode is negligible.

## Open questions

None. Draft 5's two amendments (callable restoration, copy-safety scope) were put to the operator after Go at P616 and approved on 2026-10-24; criterion 1(k) was added at P617 on the operator's reading of the byte-identical constraint. The second fix round (P617, approved) restores the registry test's id, simplifies the restore path, verifies the bind, and corrects markers and documentation.

## Design review dispositions

Round 1 (draft 1): see the ledger in the run directory; every blocker, major and minor adopted by moving the lock into the instance; one scope request declined (golden outputs from the baseline commit).

Round 2 (draft 3). Adopted: lock in `__dict__` breaks deep-copy and pickling (both) → WeakKeyDictionary plus module-level functions bound with `types.MethodType`, criterion 1(h); upstream provenance unestablished (opus) → verified fork-only, item 8 retitled; resolution by `hasattr` misled by forwarding wrappers (opus) → `isinstance` stop and `vars()` lookups; "microseconds" overstated (opus) → the wait section and criterion 1(i); loader fixture lacks `config.json` (both) → criterion 1(e); fake backend under-specified (opus) and no proof of one critical section (codex) → criterion 1(b); iteration wording and probabilistic baseline (both) → long prompts, 5 of 5, criterion 1(a); logger and imports need markers (opus) → Scope and item 3; grep finds non-server constructors (opus) → decided in Scope; `batch_decode` unmeasured (opus) → measured, no collision; economy of fallback tiers and `_add_tokens` (opus) and attribute-refusing instances (codex) → single WeakKeyDictionary, `_add_tokens` dropped, supported target is a mutable fast tokenizer, failure is a logged error (also the side note's "never a silent no-op"); audit criterion contradicts the settled constraint (both) → all eight exit 0, the registry exclusion with its reason; prompt text must not reach the log (opus) → criterion 1(f). Declined: the `padding_side` save-and-restore in two processors (opus, scope request): a different cause, recorded for the operator in the Premisa handoff. Side note (version drift): adopted as the ERROR log, criterion 1(g) and criterion 6.
