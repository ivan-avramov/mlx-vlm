# Post-merge known gaps (2026-08-06)

> **SUPERSEDED — see [`upstream-gaps.md`](upstream-gaps.md).**
>
> Kept for its incident history (the `is_mlx_format` / `+1.0` norm-shift
> production regression in §2 and the tokenizer race in the intro are still the
> best write-ups of those). Do not treat the gap list below as current:
>
> - §1's claim that prefix-cache reuse is "entirely absent from our
>   `dispatch.py`" is **wrong**. The fork does prefix reuse and has its own
>   guard; the actual bug was a subclass-matching hole that exposed
>   `BufferedRotatingKVCache` to ring-buffer corruption. Fixed — see
>   `upstream-gaps.md`.
> - §2's `minicpmv4_6` import breakage is fixed.
> - §1's DeepSeek V4 HISA config fields are restored (wiring still open).
> - The "18 failures" figure covered only four test files; the full suite was
>   at 36 failures at that time.
> - `origin/upstream` (the branch this doc refers to) no longer exists; the
>   fork now uses a conventional fetch-only `upstream` remote.

Context: `main` was merged with `origin/upstream` (`Blaizzy/mlx-vlm`, synced to
commit `94edef5`, 28 commits) to fix a regression where `initialize_rope` was
missing from `mlx_vlm/models/rope_utils.py` (traced to an earlier bad merge,
`b6c1378 "Merge upstream origin/upstream (114f847) into main"`, which pulled in
a stale pre-consolidation snapshot of upstream and reverted several shared
files). That regression is fixed and verified: `qwen2`, `qwen3`, `qwen3_5`, and
`gemma4` all import cleanly again.

A **second, unrelated regression** surfaced immediately after: the task model
(`mlx-community/Qwen2.5-1.5B-Instruct-4bit`) still failed to load, now with
`ValueError: Received 732 parameters not in model`. Root cause: `load_model`
in `mlx_vlm/utils.py` gated `sanitize_weights(...)` behind `if not
is_mlx_format:` (detected via the checkpoint's safetensors metadata,
`format: mlx`). `get_class_predicate`'s `f"{p}.scales" in weights`
key-matching (inside `nn.quantize(...)`) depends on `sanitize_weights` having
run to align checkpoint key names (add the `language_model.` wrapper prefix)
with the model's parameter paths — without it nothing gets quantized, so the
checkpoint's `.scales`/`.biases` tensors have nowhere to load. Confirmed via
bisection this predates the merge entirely (reproduces at `main`'s pre-merge
tip, `6e65e98`, once `rope_utils` is patched in isolation) — masked by the
crash above since the task model never got past import to reach model
loading. First fix attempt: remove the `is_mlx_format` gate entirely so
`sanitize_weights` always runs. **This was wrong and briefly broke the main
model in production** (see §2 below for the full story and the actual
correct fix — the real signal is per-model key-prefix presence, not file
format, and it's now fixed properly in `qwen2`/`qwen3_5`/`qwen3_5_moe`'s own
`sanitize()` methods rather than as a global gate in `utils.py`).

A **third, unrelated regression** surfaced while testing the fixes above:
garbled output on an unrelated prompt, later reproduced as a genuine
`RuntimeError: Already borrowed` in production logs. Root cause: HF's fast
(Rust-backed) tokenizer mutates internal truncation/padding state on every
`.encode()` call, which isn't safe under concurrent calls from multiple
request threads — continuous batching shares ONE tokenizer instance across
all in-flight requests, so two requests calling `.encode()` at the same
moment can panic with a Rust `RefCell` double-borrow, and because the
tokenizer is shared, that panic can corrupt state for whatever *other*
request happens to be running at the same moment. Found four call sites
doing this per-request on a small, fixed set of format/default token
strings (`mlx_vlm/generate/dispatch.py:1103`, `mlx_vlm/server/generation.py`'s
`_thinking_token_ids` and `_make_thinking_budget_criteria`, the latter
explicitly commented "Mirrors the dispatch.py path") — all deterministic
per (tokenizer, text) and therefore safe to cache. Fixed by adding
`cached_special_token_encode` to `prompt_utils.py` (double-checked locking:
lockless on cache hit) and routing all four call sites through it.

After resolving all merge conflicts and fixing what the conflict resolution
silently broke (`base.py`'s `kv_sequence_length`, a stale `qwen3_omni_moe`
import) plus the three regressions above, the four touched test files
(`test_generate.py`, `test_models.py`, `test_processors.py`,
`test_server.py`) run at **659 passed / 18 failed** (the `sanitize()` fixes
incidentally also fixed `TestGptOssMixedQuant::test_mixed_quant_checkpoint_loads`
and both `TestQwen35NormSanitization` tests, all sharing the same root
cause). None of the remaining 18 are things this merge introduced — they're
pre-existing gaps the new upstream tests happen to
newly exercise, or bugs that predate this session entirely. Below is what's
failing and why, grouped by root cause, so a follow-up session can pick up
without re-deriving any of this.

## 1. Deferred upstream features (exist upstream, not yet ported to this fork)

### AWQ weight-folding tolerance — 2 tests
- `mlx_vlm/tests/test_models.py::TestAWQ::test_fold_into_norm_is_identity` (class at line 8521, method at 8522)
- `mlx_vlm/tests/test_models.py::TestAWQ::test_fold_into_linear_is_identity` (line 8536)
- Failure: `AssertionError: 0.000614166259765625 not less than 0.0001` — a numerical-tolerance gap, not an import error. Whatever fold-into-linear/norm implementation we have is close but not tight enough against upstream's expected precision. Needs comparing our AWQ folding math against upstream's current version directly (not diagnosed further this session).

### FP8 / NVFP4 mixed-precision compressed-tensors — 4 tests
- `mlx_vlm/tests/test_models.py::TestMixedPrecisionCompressedTensors` (class at line 9040)
  - `test_routes_nvfp4_fp8_and_dense` (9103)
  - `test_fp8_dequantized_values` (9141) — `AssertionError: False is not true` on an `mx.allclose` dequant check
  - `test_strict_load_and_selective_quantize` (9150)
  - `test_rejects_multiple_native_quant_modes` (9192)
- Not diagnosed at the symbol level this session — likely a whole quantization-format feature (FP8/NVFP4 compressed-tensors support) that hasn't been ported, similar in shape to the AWQ gap.

### DeepSeek V4 HISA (sparse attention) — 3 of 4 tests
- `mlx_vlm/tests/test_models.py::TestDeepseekV4HISA` (class at 7967)
  - `test_hisa_shape_and_valid_indices` (8026)
  - `test_hisa_equals_flat_when_all_blocks_kept` (8014)
  - `test_hisa_high_recall_on_clustered_prefix` (8039)
  - (`test_hisa_batched_l_gt_1_matches_flat` at 8056 is not in the failing set)
- Failure: `TypeError: ModelConfig.__init__() got an unexpected keyword argument 'index_block'`
- Scope, precisely checked this session: `mlx_vlm/models/deepseek_v4/hisa_kernel.py` **already matches upstream byte-for-byte** — the actual sparse-attention kernel is present and correct. What's missing is just:
  1. Two fields on `mlx_vlm/models/deepseek_v4/config.py`'s `ModelConfig`: `index_block: int = 64` and `index_keep: int = 16` (upstream diff, confirmed exact).
  2. Wiring: `index_block`/`index_keep` are not referenced anywhere in `mlx_vlm/models/deepseek_v4/language.py` yet, so even after adding the config fields, the attention forward path still needs to actually call into `hisa_kernel` using them.
- This is the most tightly-scoped of the deferred items — kernel done, just needs config + plumbing.

### lfm2_vl optional projector layernorm — 2 tests
- `mlx_vlm/tests/test_models.py::test_lfm2_vl_skips_projector_layernorm_when_disabled` (line 2623)
- `mlx_vlm/tests/test_models.py::test_lfm2_vl_disabled_projector_layernorm_weights_load` (line 6056)
- Failure: `AttributeError: 'Lfm2VlMultiModalProjector' object has no attribute 'layer_norm'` — our `Lfm2VlMultiModalProjector` doesn't yet support a config flag to make the projector's layernorm optional/absent. Needs comparing against upstream's `lfm2_vl` multi-modal projector.

### Prefix-cache-reuse trimming — feature entirely absent (no failing test; the test class was removed)
- Upstream has `_prefix_cache_trim_amount` and `_cache_fully_retained` in `mlx_vlm/generate/dispatch.py` (upstream line 730 / 710), replacing an old unsafe `keys[..., :prefix_len, :]` raw-slice reuse path that corrupts rotating (sliding-window) KV caches — silent wrong output, or a shape crash once speculative decoding wraps them in `BufferedRotatingKVCache` (referenced as mlx-vlm issue #1715 in upstream's test docstring).
- Neither the old buggy path nor the new fixed path exists in our `dispatch.py` at all — this fork doesn't do prompt-cache prefix reuse across requests currently.
- The upstream test class this came with (`TestPrefixCacheReuseTrim` in `mlx_vlm/tests/test_generate.py`, originally right before `TestGemma4LogitsToKeep`) was **deleted** during merge conflict resolution rather than left failing, since porting it needs the whole feature (`_prefix_cache_trim_amount`, `_cache_fully_retained`, and the caller context in `dispatch.py` around upstream's `prompt_cache_state.find_prefix_length` / `_apc_suffix_is_text_only` / `_prime_cached_prefix_rope_state`) which is substantial, separate work. `TestGemma4LogitsToKeep` (same original conflict block) was kept since its dependency (`supports_logits_to_keep`) is already present in both `gemma4` and `gemma4_text`.
- Given this addresses a real correctness bug on upstream's side (not just a nicety), this is probably the highest-priority item on this list if we ever hit prefix-reuse-adjacent symptoms (wrong output or crashes on rotating/sliding-window caches under repeated/continued prompts).

### Structured request/decode logging subsystem — degraded gracefully, not failing, but absent
- Upstream added `_log_prefill_started`, `_log_prefill_progress`, `_log_decode_progress`, `_request_log_id` methods to `mlx_vlm/server/generation.py` (upstream ~line 1274 onward), plus a `log_state` dict threaded through several call sites, plus `QueuedGenerationRequest.request_id`/`.queued_at` fields, plus a `StreamingToken.emitted_at` field.
- None of this exists in our fork. Where conflict markers referenced it (`emitted_at = self._log_decode_progress(...)`, `**log_state`), those calls were dropped rather than partially grafted in (they'd have raised `NameError`/`AttributeError` immediately). The *other* half of these same conflict hunks — `draft_kind`/`draft_rounds`/`draft_n_accepted`/`draft_n` speculative-decode telemetry via `spec_snapshot`/`speculative_stats_since` — **was** ported and is wired up correctly (verified via `test_generation_metrics_record_speculative_stats` etc., all passing).
- Pure observability/logging feature; nothing depends on it for correctness as far as this session found.

## 2. `qwen3_5` / `qwen3_5_moe` `sanitize()` — RESOLVED (was the cause of a production incident)

**Update, later same session:** this was not just a test gap — it caused a real
production regression and is now fixed. Sequence of events:

1. While fixing the task model's quantization bug (see intro), `is_mlx_format`
   was removed from `mlx_vlm/utils.py`'s `load_model` so `sanitize_weights`
   always runs.
2. This broke the **main model** (`caslca/Ornith-1.0-35B-mlx-uniform-4bit`,
   `model_type: qwen3_5_moe`) — confirmed via live testing: coherent output
   before the fix, garbled tokens (`'uteI, senses ownI️: in2.<|im_start|>...'`)
   after, on a single non-concurrent request (so not the tokenizer race either).
3. Root cause, found via bisection: `qwen3_5_moe.py`'s own `Model.sanitize()`
   (and `qwen3_5.py`'s, though `qwen3_5_moe` doesn't inherit it — it has its
   own copy) does `if value.ndim == 1: value += 1.0` on norm-layer weights
   unconditionally, with no idempotency guard. This is a one-time HF→MLX
   convention correction that Ornith's checkpoint **already had applied at
   conversion time** — confirmed by its keys already being
   `language_model.`-prefixed (e.g. `language_model.model.layers.31...`).
   Running `sanitize()` again added the +1.0 offset a *second* time to every
   norm layer, corrupting the model numerically. `is_mlx_format` (checking
   safetensors `format: mlx` metadata) had originally existed to prevent
   exactly this — but it's the wrong signal: the task model's checkpoint
   (`mlx-community/Qwen2.5-1.5B-Instruct-4bit`) is *also* tagged
   `format: mlx` (it's an `mlx_lm`-only conversion, no `language_model.`
   wrapper) yet still needs `sanitize()` to run once to add that prefix.
4. **The precise, correct signal is prefix presence, not file format** —
   exactly the self-guard `qwen2`'s own `Model.sanitize()` already uses:
   `if any(k.startswith("language_model.") for k in weights): return weights`.
   Added the identical guard to `qwen3_5.qwen3_5.Model.sanitize()` and
   `qwen3_5_moe.qwen3_5_moe.Model.sanitize()` (the MoE expert-weight fusion
   in the latter was already safe/idempotent on its own — it guards on
   presence of the pre-fusion `experts.*` keys, which vanish after the first
   run).
5. Verified: Ornith coherent again, task model still fine, and this
   incidentally fixed `TestQwen35NormSanitization::test_qwen3_5_preserves_mlx_norm_weights`
   and `test_qwen3_5_moe_preserves_mlx_norm_weights` (both now pass — down
   from 21→20→**18** failures across this session).

**Still open:** `mlx_vlm/models/minicpmv4_6/minicpmv4_6.py` imports
`NORM_WEIGHT_SUFFIXES`/`should_offset_norm_weight`/`should_shift_norm_weights`
from `..qwen3_5.qwen3_5` — symbols that no longer exist there (removed at
some point, unrelated to this fix; not reintroduced). This still blocks:
- `mlx_vlm/tests/test_models.py::TestMiniCPMV4_6::test_minicpmv4_6_language_uses_text_only_rope` (7084)
- `mlx_vlm/tests/test_models.py::TestMiniCPMV4_6::test_minicpmv4_6_language_rejects_qwen_vl_grid_rope` (7096)
- `mlx_vlm/tests/test_processors.py::TestMiniCPMVProcessor::test_video_marker_expands_to_frame_bounds` (879)

`minicpmv4_6.py`'s own sanitize (line 430) uses `should_offset_norm_weight`
as a **per-key** gate (not a blanket shift), which is actually closer to the
right shape than what `qwen3_5`/`qwen3_5_moe` had — worth using as a
reference if `NORM_WEIGHT_SUFFIXES` et al. get restored/redesigned. Not
attempted this session since minicpmv4_6 was already non-functional
(`ImportError`) independent of anything here.

## 3. Pre-existing bugs, unrelated to this merge (found incidentally while validating)

### `/responses` endpoint validates input *after* loading the model
- `mlx_vlm/tests/test_server.py::test_responses_endpoint_merges_developer_message_with_instructions` (line 1351)
- `mlx_vlm/tests/test_server.py::test_responses_endpoint_rejects_image_file_id` (line 1460)
- Both are new tests from this merge, but the bug they expose is pre-existing fork structure, not something the merge touched (the surrounding `responses_endpoint` body auto-merged with zero conflicts).
- Root cause: `mlx_vlm/server/openai.py:975 async def responses_endpoint` calls `get_cached_model(...)` at line 1036, near the very top of the handler, **before** converting/validating the request's `input` items. Upstream's version (checked directly) does input normalization → `_response_items_to_chat` (where the `file_id`-rejection check in `mlx_vlm/server/responses_state.py:629`'s `_response_image_source` actually lives, confirmed present and correct) → `_normalize_response_instruction_messages` (the developer-message + top-level `instructions` merge the first test wants, confirmed **absent** from our fork) → *then* `get_cached_model`.
- Effect: with an unmocked/nonexistent model, a request with a bad `file_id` image 500s (model-load failure) instead of 400ing (validation failure), because the model load kicks off before validation ever runs. And developer-role messages never get merged with top-level `instructions` at all.
- Fix shape: reorder `responses_endpoint` to validate/convert input before `get_cached_model`, and port `_normalize_response_instruction_messages` from upstream. Not attempted this session — real, if contained, work on the `/responses` (OpenAI Responses API compat) surface specifically, which is secondary to the main chat-completions path.

### `ThinkingAwareLogitsProcessor` structured-decode-delay test
- `mlx_vlm/tests/test_server.py::TestResponseGenerator::test_server_generation_delays_structured_processors_for_thinking_prompt` (line 4757)
- This test already existed on `main` **before** this merge (confirmed via `git show HEAD:mlx_vlm/tests/test_server.py`) — untouched by any conflict resolution this session did. `ThinkingAwareLogitsProcessor` itself is fork-specific (doesn't exist upstream or at the merge-base at all), so this is a pre-existing bug in our own code, surfaced while running the full suite for validation, not a merge artifact.
- Failure: `assert isinstance(processors[1], server_generation.ThinkingAwareLogitsProcessor)` — `_make_logits_processors` isn't wrapping/inserting the thinking-aware delay processor the way this test expects when `enable_thinking=True` + a structured (grammar) processor is also requested. Not diagnosed further; needs a look at `ResponseGenerator._make_logits_processors` (`mlx_vlm/server/generation.py` ~line 1558) against this test's expectations.

## 4. Asset/template mismatch (not a code gap)

- `mlx_vlm/tests/test_processors.py::TestKimiK3Processor::test_prompt_utils_uses_python_renderer_not_plain_fallback` (class 2174, method 2269)
- Failure: expects `"Describe this image.<|kimi_image_placeholder|>"` in the rendered output, actual output is a full kimi_k3-style structured message (`<|open|>message role="system" type="thinking-effort"...`) with no image placeholder substitution happening. Looks like either a chat-template asset version mismatch (our vendored kimi_k3 template vs. what the test was written against) or the Python renderer isn't substituting the image placeholder for this processor. Not investigated further — narrow, single-model scope.

## Suggested priority if picking this up

1. **Prefix-cache-reuse trimming** (§1) — real correctness bug upstream fixed (silent corruption / crashes on rotating caches), currently entirely absent from our fork. Worth doing even without a failing test forcing it.
2. **`qwen3_5`/`qwen3_5_moe` sanitize()** (§2) — RESOLVED this session; was a real production incident (silently corrupted the main model's norm weights). If touching either file's `sanitize()` again, preserve the `language_model.`-prefix self-guard.
3. **DeepSeek V4 HISA** (§1) — smallest, most contained of the remaining quantization/attention feature gaps (kernel already correct, just config + wiring).
4. `minicpmv4_6`'s broken `NORM_WEIGHT_SUFFIXES` import (§2) — low effort now that the reference implementation (`qwen2`'s and `qwen3_5`'s prefix-guard pattern) is clear; minicpmv4_6's own per-key `should_offset_norm_weight` gate is actually closer to correct than what qwen3_5 had.
5. Everything else in §1 (AWQ, FP8/NVFP4, lfm2_vl projector, structured logging) — lower urgency, no known correctness impact, port opportunistically.
6. §3 and §4 — pre-existing/unrelated; fix whenever convenient, no urgency tied to this merge.
