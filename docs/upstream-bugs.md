# Upstream bugs found here — our own record of what comes from where

Bugs that live in `Blaizzy/mlx-vlm` itself, not in this fork. Each was found while
resolving a merge or draining an audit, and each is reproducible against
`upstream/main` with no fork code involved.

**This file is bookkeeping, and that is the whole of its purpose.** Its job is to
stop each of these being re-investigated as a fork problem every time it surfaces in
an audit, to record which `.symbol-exclusions` entries exist *because* upstream has a
defect, and to explain the one divergence we carry *because* upstream is wrong
(item 4). Two consequences, both deliberate:

* **These are not to be filed with upstream.** Engaging upstream's issue tracker is
  out of scope for this fork (decided 2026-08-10). An earlier handoff invented that
  task; it was never asked for and nothing else in `AGENTS.md`, `upstream-gaps.md` or
  `memory.md` ever proposed it. Don't re-propose it.
* **Patched here only when something downstream needs it** (rule changed 2026-10-22 by
  the operator, Premisa P600). A local fix to upstream code is a new permanent conflict
  site, so items 1-3, 5, 9, 10, 11 and 13, which block nothing, stay byte-identical to upstream. Items
  4, 6, 7, 8 and 12 are deliberate divergences: each carries a `# Fork:` marker, a test, and an
  entry below saying what upstream does wrong. A fix that a Premisa candidate or the
  stack needs goes in the same way, as ordinary fork work; it is not avoided as a
  one-off and never patched in a downstream repository.

So the correct action on an entry below is: read it, understand why our tree looks
the way it does, and move on; add a fix only with a marker, a test and an entry.

Line numbers are `upstream/main` at base `ffd7aeff` unless stated. Verified
2026-08-10; none had been filed by anyone else as of that date
(`gh search issues --repo Blaizzy/mlx-vlm`), which is recorded only because it
confirms they are genuinely unreported rather than known-and-wontfix.

---

## 1. `BatchTurboQuantKVCache.zero_row_tail` is defined twice; the first copy is dead

`mlx_vlm/turboquant.py`, `class BatchTurboQuantKVCache`:

* lines **6230-6244** — a `_map_state`-based version with a nested `_z(arr, ndim)`
  helper that indexes 3-D and 4-D state differently.
* lines **6271-6273** — a `_zero_state_row_tail`-based version.

Python binds the later definition, so the first is unreachable, and with it the
nested `_z`. Nothing else in the file references either.

Not to be confused with the `state`/`meta_state` pairs in `TurboQuantKVCache`,
`BatchTurboQuantKVCache` and `HybridQuantKVCache`, which are also two defs each and
are legitimate `@property` / `@x.setter` pairs. `zero_row_tail` carries no decorators
on either copy — this is real shadowing.

```python
import ast, subprocess
from collections import Counter
src = subprocess.run(["git","show","upstream/main:mlx_vlm/turboquant.py"],
                     capture_output=True, text=True).stdout
for n in ast.walk(ast.parse(src)):
    if isinstance(n, ast.ClassDef):
        for m in n.body:
            if getattr(m, "name", "") == "zero_row_tail":
                print(n.name, m.lineno, m.end_lineno,
                      [ast.unparse(d) for d in m.decorator_list])
# BatchTurboQuantKVCache 6230 6244 []
# BatchTurboQuantKVCache 6271 6273 []
```

This fork carries only the live copy, byte-identical to upstream's second one, which
is why `.symbol-exclusions` has entries for the dead sibling and its `_z`. Deleting
the dead copy upstream would let both entries be pruned here.

## 2. `TestLagunaProcessor` is defined twice, so one of its tests never runs

`mlx_vlm/tests/test_processors.py`, lines **2323-2330** and **2770-2851**. Our copy
of this file is byte-identical to upstream's, so this is upstream's bug.

The first class holds one test, `test_chat_template_owns_laguna_special_tokens`. The
second shadows it, so that test is never collected — confirmed by
`pytest --collect-only`, which reports only the second class's two tests:

```
tests/test_processors.py::TestLagunaProcessor::test_auto_processor_patch_intercepts_laguna
tests/test_processors.py::TestLagunaProcessor::test_from_pretrained_loads_fast_tokenizer_directly
```

Renaming the first class (say to `TestLagunaChatTemplate`) collects all three.

A green suite is what makes this durable: nothing fails, and the count looks right.
It is also why a symbol the shadowed test exercises can go missing without any test
noticing. (`utils.should_add_special_tokens` was the example when this was first
recorded; it is present in both trees today, so the shadowing is currently costing
coverage rather than hiding a missing symbol.)

## 3. `skip_special_tokens` is inert on the normal text path

Three call sites set it for tool-calling requests:

* `mlx_vlm/server/openai.py:994`
* `mlx_vlm/server/openai.py:1704`
* `mlx_vlm/server/anthropic.py:489`

all of the form `gen_args.skip_special_tokens = False`, so that tool-call and channel
markers survive detokenisation and the tool parser can see them.

But `args.skip_special_tokens` is read in exactly two places in
`mlx_vlm/server/generation.py` — lines **1906** and **1971** — and both are inside
`ResponseGenerator._generate_diffusion`. The autoregressive text path never reads it,
so all three assignments are no-ops for every non-diffusion model. The field defaults
to `True` at `generation.py:734`.

```python
# enclosing scope of every `args.skip_special_tokens` read, upstream/main
# generation.py:1906 -> ResponseGenerator._generate_diffusion
# generation.py:1971 -> ResponseGenerator._generate_diffusion
```

Confirmed identical in this fork (our `anthropic.py:489` is byte-identical to
upstream's), so `gen_args.skip_special_tokens = False` does nothing in **both**
trees. Recorded as `docs/upstream-gaps.md` item 9(b) and deliberately not patched
here.

## 4. Uniform KV quantization raises on any sliding-window model

`mlx_vlm/generate/common.py::maybe_quantize_kv_cache`'s final loop — the uniform,
non-hybrid, non-TurboQuant path — is:

```python
for index, layer_cache in enumerate(prompt_cache):
    if (
        hasattr(layer_cache, "to_quantized")
        and layer_cache.offset >= quantized_kv_start
    ):
        prompt_cache[index] = layer_cache.to_quantized(...)
```

`hasattr(..., "to_quantized")` is the wrong gate. `RotatingKVCache` **has** that
method, and it is a stub:

```python
def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
    raise NotImplementedError("RotatingKVCache Quantization NYI")
```

So the call is made and raises. Every other branch of the same function guards
`RotatingKVCache` explicitly (`if isinstance(entry, cache.RotatingKVCache): return
entry` in both `hybridize` and `quantize_entry`) — only the uniform path does not,
which is what makes this look like an oversight rather than a design choice.

Reachable for any sliding-window model (Gemma-family SWA layers produce a
`RotatingKVCache`) run with `--kv-bits` set on the default `uniform` scheme.
Reproduced directly against upstream's function:

```python
import ast, subprocess, textwrap
import mlx.core as mx
from mlx_vlm.generate import common as ourmod
from mlx_vlm.models import cache

src = subprocess.run(["git", "show", "upstream/main:mlx_vlm/generate/common.py"],
                     capture_output=True, text=True).stdout
tree, lines = ast.parse(src), src.splitlines()
fn = next(textwrap.dedent("\n".join(lines[n.lineno - 1:n.end_lineno]))
          for n in tree.body if getattr(n, "name", "") == "maybe_quantize_kv_cache")
ns = dict(vars(ourmod)); exec(fn, ns)

entry = cache.RotatingKVCache(max_size=32, keep=4)
entry.update_and_fetch(mx.zeros((1, 2, 8, 4)), mx.zeros((1, 2, 8, 4)))
ns["maybe_quantize_kv_cache"]([entry], quantized_kv_start=0, kv_group_size=64, kv_bits=4)
# NotImplementedError: RotatingKVCache Quantization NYI
```

**This fork already does not have the bug**, and that is the one entry here where our
divergence is load-bearing rather than incidental: our uniform loop skips
`RotatingKVCache` with a `continue`. It had no test until 2026-08-10 — see
`TestUniformKvQuantSkipsRotatingCaches` in `tests/test_dropped_upstream_guards.py`,
which also pins the premise (that `to_quantized` still raises), so it cannot become
vacuous if mlx ever implements it.

The fix upstream is one line: the same `isinstance` guard its other two branches
already use.

## 5. Anthropic streaming leaks the stop sequence; non-streaming does not

`mlx_vlm/server/anthropic.py`'s streaming path emits every text delta verbatim and
applies `_apply_stop_sequences` only at the very end, **discarding the truncated
text**:

```python
stop_sequence = None
if not parsed_tool_calls:
    _, stop_sequence = _apply_stop_sequences(      # line 782 — note the `_,`
        text_output, request.stop_sequences
    )
```

Only the matched sequence name survives, for `message_delta.stop_reason`. The
non-streaming path at line 953 keeps both halves and does truncate:

```python
response_text, stop_sequence = _apply_stop_sequences(
    response_text, request.stop_sequences
)
```

So the same request returns different text depending on `stream`. Demonstrated against
this tree with `stop_sequences: ["END"]` and a generation of `"keep" + "END" + "drop"`:

| | text | stop_reason | stop_sequence |
|---|---|---|---|
| non-streaming | `'keep'` | `stop_sequence` | `END` |
| streaming | `'keepENDdrop'` | `stop_sequence` | `END` |

The streaming client is told a stop sequence fired *and* handed the sequence plus
everything after it. `stop_sequences` exists precisely to prevent that.

**This fork carries `anthropic.py` byte-identical to upstream** — `check_body_divergence.py
--file` reports 21 of 21 shared bodies identical and 28 of 28 module statements
identical — and `git log -S` puts the block in upstream's #1203. So it is not patched
here, per this file's header: a local fix would turn a perfectly aligned file into a
permanent conflict site.

Reproduce (drives the real endpoint through `TestClient`):

```python
import json
from types import SimpleNamespace
from unittest.mock import patch
from fastapi.testclient import TestClient
import mlx_vlm.server as server
import mlx_vlm.server.anthropic as anth
from mlx_vlm.generate import GenerationResult

BODY = {"model": "demo", "max_tokens": 32, "stop_sequences": ["END"],
        "messages": [{"role": "user", "content": "hi"}]}
fake = lambda: (SimpleNamespace(), SimpleNamespace(), SimpleNamespace(model_type="qwen2_vl"))
toks = [GenerationResult(text=t, prompt_tokens=5, generation_tokens=1,
                         finish_reason=("stop" if t == "drop" else None))
        for t in ["keep", "END", "drop"]]

with TestClient(server.app) as c:
    server.runtime.response_generator = None
    with (patch.object(anth, "get_cached_model", return_value=fake()),
          patch.object(anth, "apply_chat_template", return_value="p"),
          patch.object(anth, "stream_generate", side_effect=lambda **k: iter(toks))):
        r = c.post("/v1/messages", json={**BODY, "stream": True})
    print("".join(json.loads(l[6:])["delta"]["text"]
                  for l in r.text.splitlines()
                  if l.startswith("data: ")
                  and json.loads(l[6:]).get("type") == "content_block_delta"))
    # keepENDdrop  — expected: keep
```

The fix upstream is to truncate the accumulated text before emitting each delta and
hold back a trailing *partial* sequence, which is what
`responses_state._partial_tag_start_pos` already does for thinking markers. **The
fork's own `/v1/completions` endpoint had the partial half of this same bug** — see
the `# Fork:` comment on its streaming stop hold-back and
`tests/test_completions_endpoint.py`.

---

## 6. ModernBERT masks overflow to `-inf` in float16 and NaN short texts in a padded batch

`mlx_vlm/models/modernbert/modernbert.py::Model._masks` fills the additive mask with
`-1e9` and then casts it to the compute dtype. float16 tops out at 65504, so the cast
yields `-inf`. A padding query position whose whole sliding window (`local_attention
// 2` on each side, 64 for the published configs) is also padding gets a row of
`-inf` scores, and `mx.fast.scaled_dot_product_attention` softmaxes that row to NaN.
The NaN sits in a padding position, but mean pooling multiplies the hidden state by
the zero mask and `NaN * 0` is NaN, so the whole embedding of the short text is NaN.
Single-text embedding and float32 compute are unaffected; bfloat16 represents `-1e9`
and is unaffected.

Reproduced with `Alibaba-NLP/gte-modernbert-base` (float16 weights) on sixteen
retrieval chunks batched together, token lengths 66 to 448: every row padded by more
than 64 tokens (six of sixteen) came back NaN. `tests/test_modernbert_fp16_mask.py`
reproduces it on a tiny random model and fails against upstream's `_masks`.

**This fork diverges deliberately here**, the second load-bearing divergence after
item 4: `_masks` uses `mx.finfo(dtype).min` instead of `-1e9`, as `bert.py` and
`xlm_roberta.py` already do, marked `# Fork:`. The fused kernel keeps a fully masked
row finite with that constant even at scores of several thousand (checked directly).
Decided 2026-10-21 in the Premisa forest (P570): the embedder cannot move onto the
MLX task server with NaN rows in batched embedding, and the server batches.

The fix upstream is the same one-line constant change.

---

## 7. Qwen3 text rerankers score every pair the same with the checkpoint's own chat template

`mlx_vlm/server/reranking.py::_score_text_batch` builds the prompt by passing generic
`system` and `user` messages through `tokenizer.apply_chat_template`. The
`Qwen/Qwen3-Reranker-0.6B` repository (revision `e61197e`) now ships a
`chat_template.jinja` written for sentence-transformers' CrossEncoder: it reads the
instruction from the `system` message and the query and document from messages with
roles `query` and `document`. Rendered with the generic messages it emits
`<Query>: ` and `<Document>: ` empty and puts the system text where the instruction
goes, so every (query, document) pair tokenizes to the same prompt and scores
identically.

Reproduced on the real checkpoint on 2026-10-22: five documents, one clearly relevant,
all scored 0.0049 through `score_documents`; the rendered prompt showed the empty
fields.

**This fork diverges deliberately here** (`_text_prompt`, marked `# Fork:`): the prompt
is rendered from the format on the model card (system block, `<Instruct>`, `<Query>`,
`<Document>`, then the fixed assistant suffix the code already appended as tokens),
with no dependency on whatever template the checkpoint carries. After the change the
same five documents score 1.0, 0.0, 1.0, 0.0, 0.0004. Test:
`tests/test_qwen3_reranker_prompt.py`. Decided in the Premisa forest (P568 screening,
2026-10-22): the candidate cannot be screened, nor served, with constant scores.

The fix upstream is the same: render the published format, or pass `query` and
`document` roles when the checkpoint's template declares them.

---

## 8. Fast tokenizers race on their Rust object: a fork-introduced race with partial upstream exposure

The Hugging Face fast tokenizer mutates its Rust object (`enable_padding`,
`no_padding`, truncation) at the top of every encode whenever the call's target
differs from the object's current state. Those are `&mut self` pyo3 methods, and
`encode_batch` releases the GIL while holding a shared borrow, so a second thread's
settings change fails with `RuntimeError: Already borrowed`. The server turns that into
a 500 for one of two concurrent requests to the same served model.

Measured 2026-10-23 on an in-memory `PreTrainedTokenizerFast` (`tokenizers` 0.22.2;
`transformers` 5.14.1 here, 5.15.0 in Premisa's interpreter): one thread alternating
`padding=True` and `padding=False` against a second thread running `encode` fails
within the first iterations, 5 of 5 runs. The same mutator against `decode` of 3200
ids, `batch_decode` of 64 sequences of 400 ids and `convert_tokens_to_ids` of 600
tokens never fails over 3000 iterations: those hold the GIL, so the mutator cannot
observe them mid-borrow. The victim set is the encode entry point and the setter, and
the setter and the encode must be one critical section (`_encode_plus` calls the
setter); locking the Rust calls individually would let a second thread's settings
change land between them and return a silently wrong result.

**Where it comes from.** The path both production tracebacks end in, the GPU worker's
own tokenization inside `_process_cached_request` through `_stream_generate` and
`prepare_inputs`, is fork-only (ported in `cb585a75`; upstream's `generation.py` has no
`_process_cached_request`, and its comment at the `_tokenizer_lock` block says its
design keeps tokenization on the caller side so the worker never races request
threads). The fork's retention and anchor encodes in `server/openai.py` and
`generate/common.py` are fork-only too. **Upstream is exposed only where two request
threads use one tokenizer with differing settings outside `_tokenizer_lock`**, which
its embeddings and reranking routes allow (text against image requests). That
exposure is reasoned, not reproduced upstream. In Premisa's task server it shows at
every chat start, when Open WebUI sends title generation and query generation
together.

**Three partial mitigations stay.** Upstream's `_tokenizer_lock` (`ff295e36`)
serializes the caller-side preprocessing, and also guards thinking-budget state. The
fork's `7e3477b0` added `cached_special_token_encode`, `prewarm_special_token_encodes`
and `_encode_retrying_on_borrow_error` in `prompt_utils.py`. None reaches the worker's
encodes, the request-side encodes outside the preprocessing block, or the direct
tokenizer calls in `server/embeddings.py::_embed` and `server/reranking.py`; a
call-site inventory is the wrong unit (round 1 of the design review kept finding sites).
They stay because retiring them belongs to the upstream-merge session.

**This fork diverges deliberately here**, marked `# Fork:` throughout: the new module
`mlx_vlm/tokenizer_lock.py` rebinds, on each fast tokenizer instance, `_encode_plus` and
`set_truncation_and_padding` to callables that take one re-entrant lock per
instance (a `WeakKeyDictionary` outside the instance, so `copy.deepcopy`,
`pickle.dumps` and `save_pretrained` keep working). Two hooks apply it once at load:
`tokenizer_utils.py::load_tokenizer` and `utils.py::load_processor`, which together
cover the chat path, the embedding and reranker loaders and the tokenizer-only
fallback. `prompt_utils.py` gains a module logger and a warning per retried borrow
error (operator, 2026-10-23), naming the attempt and the tokenizer class and nothing
from the prompt. Decode, `batch_decode`, id lookups and the detokenizers are not
locked, which keeps the worker's per-token decode off the lock; the lock is a leaf and
is never held across a model call, a vision encoder call or media work. The worker's
encodes can wait for one in-flight encode on a request thread, bounded by that encode.
The bound methods are instances of a small callable class whose `__reduce__`
rebuilds it from the restored tokenizer, so `copy.deepcopy` and a pickle round trip
return a serialized tokenizer. A pickled serialized tokenizer references
`mlx_vlm.tokenizer_lock`, so unpickling needs the fork importable. `copy.copy` of a serialized tokenizer is unsupported: the clone's
instance-bound methods act on the original and the two share one Rust backend; use
`copy.deepcopy`. A class lacking either method, or refusing the assignment, logs one
ERROR at load (the class, the attribute and the exception's class name) and is left
unchanged, instead of silently running unprotected.

Tests: `tests/test_tokenizer_concurrency.py` (the instance-level block): the stress
reproduction, lock ownership inside every padding, truncation and encode call driven
through `__call__`, `encode` and `apply_chat_template`, a settings change waiting for an
encode in flight, lock identity, output equality, the loaders, the retry warning, the
version guard and copy safety.

The fix upstream could take is the same serialization applied in `load_tokenizer` and
`load_processor`, or a lock around `_encode_plus` and `set_truncation_and_padding` in
whatever creates the tokenizer.

## 9. The Qwen3.5 exact speculative verifier ignores `norm_topk_prob`

`mlx_vlm/models/qwen3_5/speculative_verifier.py`,
`Qwen3_5ExactSpeculativeVerifier._feed_forward` (upstream `4623cf9d`):

```python
scores = mx.take_along_axis(gates, indices, axis=-1)
scores = scores / scores.sum(axis=-1, keepdims=True)
```

#2457 (`c32e07b8`) gave `Qwen3_5MoeSparseMoeBlock` a `norm_topk_prob` flag
(`TextConfig.norm_topk_prob`, default `True`) and made the block renormalize only
when it is set. The exact verifier re-implements the block's routing and still
renormalizes unconditionally, so for a checkpoint with `norm_topk_prob: false` the
verified logits differ from the target's own forward pass and exact speculative
decoding stops being exact.

Reproduce: `test_qwen_exact_verifier_moe_matches_singleton_path` with
`norm_topk_prob=False` added to its `SimpleNamespace` config fails (max abs diff
well above the bf16 tolerance); with `True` it passes.

**Not patched here** (P600): no Qwen3.5/3.6/3.8 MoE checkpoint the stack serves sets
`norm_topk_prob: false` (the HF configs omit it, so the default `True` applies), so
nothing downstream needs the fix. Found during the 2026-10-25 staged sync (stage A
review). The fix upstream could take is
`if getattr(feed_forward, "norm_topk_prob", True):` around the division.

## 10. Nemotron MTP sampled bf16 video generation drifts from plain decode

`mlx_vlm/tests/test_speculative.py::test_nemotron_mtp_generation_matches_autoregressive`
(upstream `b8bf04c6`, #2528), parametrization `[0.7-dtype1-video]` (temperature 0.7,
`bfloat16`, video input): the MTP-speculative token stream differs from plain
`generate_step` at token 1. The other eleven parametrizations, including bf16 text
and image at 0.7 and every greedy case, pass.

Reproduced on upstream's own tree (`git worktree add <dir> b8bf04c6`, run that
test from `<dir>/mlx_vlm`, which imports the worktree's package) with mlx 0.32.4 on
2026-10-25, so it is not fork code. Likely a bf16 numerics difference between the
batched verify block and the singleton decode step under sampling; not investigated.

**Not patched here** (P600): the stack serves Nemotron draft-OFF. The ported test is
kept verbatim and marked strict-xfail through `tests/conftest.py`'s
`KNOWN_UPSTREAM_FAILURES`, so it turns red when upstream fixes it.

## 11. `nemotron_h_nano_omni`'s `output_hidden_states` disagrees with its layers' inputs

`mlx_vlm/tests/test_models.py::test_model_contract[nemotron_bolt]` and
`[nemotron_super_vl]` (both `nemotron_h_nano_omni`, #2528): the
`language_hidden_states` check compares `output.hidden_states[:-1]` with the inputs
each layer actually received, and every element differs by an almost constant
per-position factor (about 0.64). That looks like a normalization or residual
scaling applied to the returned states but not to the layer inputs.

Reproduced on upstream's own trees (`b8bf04c6` and `e9af3766`, worktree, mlx 0.32.3
and 0.32.4) on 2026-10-25, so it is not fork code. Not investigated further.

**Not patched here** (P600): nothing downstream reads `output_hidden_states` from
this model. Both parametrizations are strict-xfailed through
`tests/conftest.py`'s `KNOWN_UPSTREAM_FAILURES`.

## 12. The Qwen3.5 exact verifier's q/k normalization no longer matches decode

`mlx_vlm/models/qwen3_5/language.py::Qwen3_5GatedDeltaNet._normalize_qk` changed its
RMSNorm epsilon from `1e-6` to `1e-6 / head_width` (upstream `b8bf04c6`, #2541). That
makes it the L2 normalization upstream intended. The verifier's copy,
`speculative_verifier.py::Qwen3_5ExactSpeculativeVerifier._normalize_gated_delta_qk`,
kept `1e-6`. Verify and plain decode therefore build slightly different gated-delta
states. Greedy emission still matches, but the recurrent and KV state after an MTP
round no longer equals the plain path's.

Found by the fork's exact-MTP tests (`test_mtp_stop_state.py`,
`test_mtp_session_cancel.py`, `test_mtp_thinking_budget.py`,
`test_mtp_zero_budget.py`). Seven of them failed after the stage C merge and pass
with the fix.

**This fork diverges deliberately here**: the stack's MTP serving relies on
verify == decode state (P369's session exactness). `_normalize_gated_delta_qk`
delegates to the layer's own `_normalize_qk` when it has one (marked `# Fork:`),
falling back to upstream's body otherwise, so any later change to the layer flows
through. The fix upstream could take is the same delegation, or the same epsilon
in both copies.

## 13. LoRA/PEFT cannot target NVFP4 projections that carry a global scale

#2437 (`035d6abb`) loads NVFP4 linears that have a per-tensor global scale
(`weight_scale_2` / `weight_global_scale`) as `quantization/nvfp4.py::ScaledQuantizedLinear`,
an `nn.Module` wrapper. Before that change they loaded as ordinary
`nn.QuantizedLinear`. `trainer/peft/utils.py` (target discovery) and
`trainer/peft/lora.py` (`_to_lora`, restore and fuse) accept only `nn.Linear` /
`nn.QuantizedLinear`. On such a checkpoint, adapter discovery silently skips those
projections, and explicitly restoring an adapter that targets one raises
`Can't convert layer of type ScaledQuantizedLinear to LoRA`.

Found by the stage A change review (codex) of the 2026-10-25 sync. Confirmed by
reading `trainer/peft/`, which is byte-identical to upstream `e9af3766`.

**Not patched here** (P600): the stack does not fine-tune NVFP4 checkpoints.

## 14. `singleton_quantized_linear` is not bitwise equal to decode for K <= 128

`models/quantized_verifier.py::singleton_quantized_linear` (#2127) computes a
`[B, T, K]` block with one `mx.gather_qmm` so each row gets "singleton qmv
arithmetic". MLX's `dispatch_qmv` (`mlx/backend/metal/quantized.cpp`, identical in
v0.32.3 and v0.32.4) routes a plain `quantized_matmul` with `K == 128 || (K == 64 &&
bits >= 2)`, power-of-two bits and no global scale to `qmv_quad` (quad-group
reduction); `gather_qmv` never does (simd reduction). The reduction trees differ, so
for K <= 128 verify differs from decode by one output ulp on some inputs.

Measured over 40 seeds per configuration, `[1, 2, K]` bf16: on mlx 0.32.3 only 8-bit
K in {64, 128} mismatch; on 0.32.4 the 2- and 4-bit cases do too. Every other K and
bit width matched. In the fork's E1a mixed fixture the 8-bit `linear_attn.out_proj`
(K = 128) reaches it through `speculative/ops/linear.py::_target_verify_quantized_linear`,
which is upstream-identical; the fork's exact affine kernel does not apply because
`supports_optimized_affine_head` rejects K % 512 != 0.

A second geometry, **mlx 0.32.4 only**: when the output width `N` is not a multiple
of 8 the plain call takes `qmv_fast` with partial rows, while `gather_qmv` is still
gated on `N % 8 == 0` and never does. Confirmed for bf16 and fp16, 4- and 8-bit,
K in {512, 1024}, N in {7, 12, 100}; none on 0.32.3.

Upstream already knows the shape: `models/linear.py::native_batch_linear` guards its
own call with `K % 512 == 0`; the speculative path has no such guard.

**This fork diverges deliberately here**: for `x.shape[-1] <= 128` or `N % 8 != 0` the function runs
each `[b, t]` row as the decode call itself and concatenates (marked `# Fork:`), exact
by construction; the `gather_qmm` path is kept only when K > 128 and N % 8 == 0. Pinned by
`tests/test_singleton_quantized_narrow.py`. The fix upstream could take is the same
per-row path, or a guard at the speculative caller requiring both `K % 512 == 0` and `N % 8 == 0` (the K guard alone does not cover the odd-N geometry).
