# Post-merge known gaps (2026-08-06)

Context: `main` was merged with `origin/upstream` (`Blaizzy/mlx-vlm`, synced to
commit `94edef5`, 28 commits) to fix a regression where `initialize_rope` was
missing from `mlx_vlm/models/rope_utils.py` (traced to an earlier bad merge,
`b6c1378 "Merge upstream origin/upstream (114f847) into main"`, which pulled in
a stale pre-consolidation snapshot of upstream and reverted several shared
files). That regression is fixed and verified: `qwen2`, `qwen3`, `qwen3_5`, and
`gemma4` all import cleanly again.

After resolving all merge conflicts and fixing what the conflict resolution
silently broke (`base.py`'s `kv_sequence_length`, a stale `qwen3_omni_moe`
import), the four touched test files (`test_generate.py`, `test_models.py`,
`test_processors.py`, `test_server.py`) run at **656 passed / 21 failed**.
None of the 21 are things this merge introduced — they're pre-existing gaps
the new upstream tests happen to newly exercise, or bugs that predate this
session entirely. Below is what's failing and why, grouped by root cause, so a
follow-up session can pick up without re-deriving any of this.

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

### gpt_oss mixed-quant checkpoint loading — 1 test
- `mlx_vlm/tests/test_models.py::TestGptOssMixedQuant::test_mixed_quant_checkpoint_loads` (class at 9226, method at 9282)
- Failure: `ValueError: Received 71 parameters not in model: ...` — a per-layer key-remapping mismatch when loading a mixed-quant gpt_oss checkpoint (expert layers get keys the current `sanitize`/quantize-predicate path doesn't expect). Related in spirit to the `qwen3_5` sanitize gap below but a different model family.

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

## 2. `qwen3_5` `sanitize()` — needs a product decision, not a quick fix

- `mlx_vlm/tests/test_models.py::TestQwen35NormSanitization::test_qwen3_5_preserves_mlx_norm_weights` (line 8149)
- `mlx_vlm/tests/test_models.py::TestQwen35NormSanitization::test_qwen3_5_moe_preserves_mlx_norm_weights` (line 8161)
- Also blocks (via `ImportError: cannot import name 'NORM_WEIGHT_SUFFIXES'`):
  - `mlx_vlm/tests/test_models.py::TestMiniCPMV4_6::test_minicpmv4_6_language_uses_text_only_rope` (7084)
  - `mlx_vlm/tests/test_models.py::TestMiniCPMV4_6::test_minicpmv4_6_language_rejects_qwen_vl_grid_rope` (7096)
  - `mlx_vlm/tests/test_processors.py::TestMiniCPMVProcessor::test_video_marker_expands_to_frame_bounds` (line 879; imports `NORM_WEIGHT_SUFFIXES`/`should_offset_norm_weight`/`should_shift_norm_weights` from `..qwen3_5.qwen3_5` via `mlx_vlm/models/minicpmv4_6/minicpmv4_6.py:8`)
- What happened: at the merge-base, `mlx_vlm/models/qwen3_5/qwen3_5.py` had `NORM_WEIGHT_SUFFIXES`, `should_shift_norm_weights(weights)`, and `should_offset_norm_weight(original_key, shift_norm_weights)`, used inside `Model.sanitize()` to decide whether MTP-shard/unsanitized-conv1d checkpoints need their norm weights shifted.
- Our fork's `main` **replaced** that logic with a simpler `weights = {k: v for k, v in weights.items() if "mtp." not in k}` (just dropping MTP weights, no norm-weight shift/offset decision at all). This was alongside a real, deliberate, unrelated change in the same commit region: `get_input_embeddings` now defers `get_rope_index` to only when there's an actual image/video grid, storing `_position_ids`/`_rope_deltas` directly on the language model (comment: "Pre-calculate position_ids for chunked prefill") instead of returning them via `InputEmbeddingsFeatures`.
- Because both changes landed together and the norm-weight simplification could plausibly be intentional (e.g. if it turned out the shift/offset logic was unnecessary for checkpoints we actually load, or was itself a source of bugs), **do not just restore the three upstream functions verbatim** without checking: (a) whether any checkpoint we actually load needs the shift/offset behavior, and (b) whether the simplification was deliberate or fell out of the same bad-merge pattern that broke `rope_utils.py`. Worth a `git log -p` / blame pass on this specific hunk in isolation, and possibly asking whoever wrote the chunked-prefill position-id change if the norm-weight simplification was intentional.
- `minicpmv4_6` and the minicpm video-marker processor test are collateral: they import `NORM_WEIGHT_SUFFIXES` etc. from `qwen3_5.qwen3_5` for their own (correct, presumably still-needed) purposes. Fixing the `qwen3_5` question fixes these transitively.

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
2. **`qwen3_5` sanitize()** (§2) — blocks 5 tests across 3 files and touches active-campaign model families; needs the git-history/intent check described above before touching.
3. **DeepSeek V4 HISA** (§1) — smallest, most contained of the quantization/attention feature gaps (kernel already correct, just config + wiring).
4. Everything else in §1 (AWQ, FP8/NVFP4, gpt_oss mixed-quant, lfm2_vl projector, structured logging) — lower urgency, no known correctness impact, port opportunistically.
5. §3 and §4 — pre-existing/unrelated; fix whenever convenient, no urgency tied to this merge.
