Here is the fully updated, production-ready context memory, incorporating all architectural discoveries, bug fixes, and the TurboQuant investigation findings.

***

**System Overview: Local Gemma-4 Vision-MoE + OpenWebUI Pipeline**

**Goal:** We successfully built a flagship-tier local AI pipeline using quantized Gemma 4 (31B 6-bit) running on Apple Silicon via `mlx-vlm`, interfaced with OpenWebUI.

**Capabilities Achieved:** Native multimodal vision, agentic web searching, autonomous multi-tool execution (e.g., sequentially writing and executing Python scripts in Pyodide to self-verify math), document RAG, multi-turn accordion thought blocks (`<think>`), perfect LaTeX math rendering, and a highly-optimized 128k/256k context window (for the gemma-4-31B-6bit/gemma-4-31B-4bit models respectively)

**Current Workload:** Autonomous Agentic Coding. We specifically run the 6-bit quantization to prioritize lossless reasoning, strict JSON adherence, and rigid syntactic logic over the 4-bit model's larger context window.

**Why mlx-vlm instead of mlx-lm?**
While `mlx-lm` works fine out of the box for text-only Gemma 4, we explicitly built on top of `mlx-vlm` because we must have the ability to process visual tensors (screenshots, image uploads, and document PDFs) alongside the text and tool-calling agentic loops.

**The Core Architecture Fixes (Minimalist Production Build)**

**1. Tokenizer Race Condition (utils.py)**
* **The Problem:** Words leaked out of the thought block after the closing `<channel|>` tag due to a race condition in the custom `_ChannelTokenDetokenizer`.
* **The Fix:** Deleted the custom wrapper class. We now rely entirely on the native tokenizer and `StoppingCriteria`, which perfectly streams tokens in chronological order.

**2. The StreamingTranslator Engine & Token Un-Fuser (server.py)**
* **The Problem:** OpenWebUI requires `<think>` instead of Gemma's native `<|channel>thought`. Furthermore, quantization degraded the model's attention mechanism on the very first token of a zero-shot prompt, causing token space-fusion (e.g., generating "Thisis").
* **The Fix:** Built a robust, stateful `StreamingTranslator` with a 15-character delay buffer to translate tags mid-stream without tearing. Added a surgical regex bound to an `is_beginning` flag to dynamically un-fuse missing spaces on targeted vocabulary words.

**3. Vision Schema Alignment (server.py)**
* **The Problem:** OpenWebUI sends images as `{"type": "image_url"}`, but the Hugging Face `gemma_4_template.jinja` requires `{"type": "image"}` to render the `<image>` visual token. The model was going blind.
* **The Fix:** Updated the `message.content` parser to intercept `image_url` and rewrite it to `type: image`, perfectly aligning the prompt with the visual tensors.

**4. Tool Parsing & OpenAI API Strictness (server.py)**
* **The Problem:** OpenWebUI's Pydantic validation crashes if tools don't perfectly match the OpenAI spec.
* **The Fix:** Implemented a lightweight `_resolve_tool_calls` wrapper. It lets the native MLX `process_tool_calls` do the heavy parsing, but safely injects the mandatory `id`, `type="function"`, and `json.dumps()` stringified arguments that OpenWebUI demands.

**5. Processor Chat Template Fallback (prompt_utils.py)**
* **The Problem:** AutoProcessor models frequently fail to expose their `chat_template`, causing mlx-vlm to fall back to a raw, broken prompt.
* **The Fix:** Added a fallback block inside `apply_chat_template` that checks if `processor.chat_template` is missing, and natively inherits it from `processor.tokenizer.chat_template`.

**6. The Quantization Interceptor Bug (generate.py)**
* **The Problem:** The pipeline was blowing past 64GB of RAM on long contexts because `maybe_quantize_kv_cache` was only checking for `cache.KVCache`, failing to intercept and quantize Gemma's newer `cache.SimpleKVCache` and `cache.ChunkedKVCache`.
* **The Fix:** Added the newer cache types to the `isinstance` tuple, allowing the pipeline to successfully migrate FP16 histories into quantized caches at the `QUANTIZED_KV_START` (5000) threshold.

**7. The Threshold Graph Explosion (generate.py)**
* **The Problem:** When breaching the 5000-token threshold, MLX lazily built FP32 intermediate buffers for all layers simultaneously, causing a massive 9GB+ out-of-memory spike during compilation.
* **The Fix:** Serialized the graph compilation by injecting `mx.eval()` directly inside the layer loop. This compiles and frees the intermediate FP32 buffers layer-by-layer, dropping the threshold spike to ~150MB. This is now configurable via the `--serialize-kv-quantization` CLI flag (defaults to off). It works for both the TurboQuant and uniform quantization paths. Recommended for large models (31B+) with long contexts.

**8. The Double-Buffer Reallocation Spike (server.py & turboquant.py)**
* **The Problem:** Late-stage cache geometric growth (e.g., 76K -> 95K tokens) required the GPU to hold both the old and new cache in VRAM simultaneously, creating a massive 11GB+ double-buffer memory spike.
* **The Fix:** Threaded the `MAX_KV_SIZE` environment variable all the way down into the `TurboQuantKVCache` constructor. By setting `MAX_KV_SIZE=131072`, the backend statically pre-allocates the entire 130K block upfront, bypassing O(N^2) reallocation thrashing and late-stage OOM crashes. `get_max_kv_size()` now returns the value even when `kv_bits` is set (previously it returned `None` in that case, blocking it from reaching TurboQuant).

**9. The Prefill Step Size (server.py)**
* **The Problem:** `PREFILL_STEP_SIZE=2048` (the default) created a massive 9.3GB N^2 activation memory spike during prompt ingestion. `128` crashed the Metal driver due to unrolled shader alignment assumptions.
* **The Fix:** The prefill step size is configurable via the `--prefill-step-size` CLI argument and `PREFILL_STEP_SIZE` environment variable (default: 2048). The recommended value for Apple Silicon UMA with Gemma 4 31B is **512**, which shrinks the activation footprint by nearly 5GB without sacrificing `prompt_tps` throughput, avoiding macOS GPU watchdog kills.

**10. Telemetry Pipeline (server.py)**
* **The Problem:** Server usage metrics (tokens, TPS, peak memory) were vanishing into the API ether.
* **The Fix:** Injected `logging.info` blocks across all four OpenAI-compatible endpoints (`/chat/completions` and `/responses`, both streaming and non-streaming) to pipe live generation metrics directly into `log_file`.

**11. The Multi-Cache LRU Manager (server.py)**
* **The Problem:** OpenWebUI background tasks (e.g., chat titles, tags) used distinct prompt structures, triggering cache divergence. Since the server originally used a single-slot cache, these background tasks constantly destroyed the 45GB main chat context.
* **The Fix:** Implemented a `MultiCacheManager` using `OrderedDict`. It tracks multiple sessions independently and dynamically evicts dormant Least Recently Used (LRU) caches when macOS active unified memory breaches a configurable limit (e.g., 10% free RAM via `--lru-min-free-ram-percent`). The manager and its allocation hooks are properly cleaned up on model unload to prevent stale hook accumulation.

**12. Mid-Prefill Allocation Hooks (turboquant.py & server.py)**
* **The Problem:** Caches could scale geometrically mid-prefill. Request-boundary eviction was unsafe, as massive context growth could hit the VRAM ceiling and OOM crash before the manager regained control.
* **The Fix:** Injected `_trigger_allocation_hooks()` directly into `turboquant._reserve_state_capacity`. The Metal allocator now calls back to the Python memory manager to halt the thread and purge dormant LRU sessions on demand to satisfy contiguous memory requests. An `unregister_allocation_hook()` function was added to allow clean teardown on model unload.

**13a. Hybrid-Cache Rewind: Guard then Snapshot Restore (generate.py + snapshot.py — uncommitted as of 2026-04-28)**
* **The Problem:** Models with non-trimmable cache layers (Qwen 3.5 / 3.6 GatedDeltaNet uses `ArraysCache` for recurrent state, which advances monotonically with no rewind primitive) silently corrupt on chat-rewind / regenerate. The fork's `_trim_cache` (generate.py:1208) only handles caches with an `offset` attribute — `ArraysCache.is_trimmable()` returns `False` and has no `offset`, so it gets silently skipped. Consequence: KV layers correctly trim to the new prefix length; DeltaNet state retains advancement past the rewind point; generation drifts (no crash, just wrong outputs). Upstream `mlx_lm.can_trim_prompt_cache` correctly refuses this case via `all(c.is_trimmable() for c in cache)`, but the fork's custom trimmer bypasses that check.
* **The Fix (initial guard, conservative):** Added a `_has_non_trimmable()` recursive walker that runs before partial-prefix reuse. When ANY cache layer reports `is_trimmable() == False`, force `prefix_len = 0` (full re-prefill). Generic across cache types via the canonical `is_trimmable()` API, not `isinstance(c, ArraysCache)`.
* **The Fix (snapshot-based, proper):** New `mlx_vlm/snapshot.py` module with `DeltaNetSnapshot` dataclass and `DeltaNetSnapshotRing` (FIFO, default size 3). `PromptCacheState` owns a ring; `update()` captures a snapshot at every turn boundary (free — `mx.array` is immutable, `list(c.state)` is refcount-only). On rewind, the guard tries `ring.find_nearest(prefix_len)`; if found, restores DeltaNet state via `c.state = snap.states[i]` and sets `prefix_len = snap.offset` so the existing prefill machinery replays `[snap.offset, full_len)` through the model — both KV and DeltaNet end up coherent. If the ring is empty / disabled / no usable snapshot, falls through to the conservative full-re-prefill path.
* **Why no WAL / per-token logging:** considered and rejected. Per-token WAL (`k_t, v_t, β_t, G_t`) for Qwen 3.6 27B costs ~1.15 MB/token across 48 DeltaNet layers — a 16K-token window = 18 GB. Snapshot replay processes the same forward pass as a fresh prefill but starting from a much later offset, so it's bounded and cheaper than full re-prefill. Snapshots-only is dramatically simpler with ~80× less memory; replay cost is acceptable for chat (re-running a turn's worth of tokens, typically <2K, takes seconds).
* **Why no token-interval snapshots:** OpenWebUI doesn't expose mid-generation rewind. Real rewinds (regenerate, edit prior message) all land on turn boundaries. Token-interval captures would consume memory for a use case that doesn't exist.
* **Why no replay-distance cap:** replay distance is bounded by the ring contents — if the user rewinds beyond the oldest snapshot, `find_nearest` returns None and we fall through to fresh re-prefill. Adding an explicit max-replay knob would route some legitimate replays to fresh re-prefill, which is *strictly slower* (it processes more tokens). The implicit ring cap is the only legitimate fallback trigger.
* **Configuration (CLI primary, env fallback):**
  - `--deltanet-ring-size N` / `MLX_VLM_DELTANET_RING_SIZE` (default 3): snapshots per session. 0 disables. ~75 MB per snapshot on Qwen 3.6 27B (48 DeltaNet layers × 1.57 MB), so 3 = 225 MB.
  - `--deltanet-rewind {on,off,auto}` / `MLX_VLM_DELTANET_REWIND` (default auto): master switch. `auto`/`on` enable the restore path; `off` forces full re-prefill on hybrid rewinds. Internally collapses to a `rewind_enabled: bool` on PromptCacheState.
  - **Resolution boundary:** values are resolved once at CLI parse time in `server.py:_resolve_deltanet_config` (CLI arg > env var > module default), stashed on `_deltanet_ring_size` / `_deltanet_rewind_enabled`, and passed through `PromptCacheState(snapshot_ring=DeltaNetSnapshotRing(max_size=...), rewind_enabled=...)`. **`snapshot.py` itself reads no env vars and exports no defaults beyond a single `DEFAULT_RING_SIZE` constant** — the data module is pure, configuration flows down explicitly. Direct `PromptCacheState()` calls in chat.py / chat_ui.py / tests get sensible defaults via the constructor signatures.
* **Architecture notes:**
  - The snapshot ring lives on `PromptCacheState`. Capture happens in `PromptCacheState.update()`, which is called at the end of each generation in `stream_generate()`. So the rewind benefit applies to the chat.py / chat_ui.py / non-streaming server endpoints — the streaming/batched server path doesn't currently use prefix-cache reuse (each request fresh-prefills the full conversation), so no rewind issue exists there.
  - Vision-mode interaction is naturally handled: turn-boundary snapshots are post-vision-encoding by construction (a turn doesn't end mid-image).
  - Speculative-decoding interaction is naturally handled: snapshots are taken AFTER per-token commit in the standard generation path; the existing `rollback_speculative_cache` (qwen3_5/language.py:450-558) operates intra-block and finishes before the token commit point.
  - Restore is `c.state = snap.states[i]` — the `state` setter exists on `ArraysCache` (just assigns to `self.cache`).
* **Validation:** Smoke tests cover: capture/restore mx.array refs survive cache mutation; FIFO eviction at max_size; duplicate-offset rejection; `find_nearest` correctness; pure-attention models (no ArraysCache) return None; disabled ring no-ops; PromptCacheState auto-attaches a ring with default size 3 from env.
* **Stale-snapshot drop on token divergence (generate.py + snapshot.py — uncommitted 2026-04-29):** The original ring captured at turn boundaries but never invalidated entries when the live cached token sequence changed underneath them. Concretely: turn N captures `snap@27` reflecting DeltaNet state from the cached tokens [0..27). Turn N+1 forces full re-prefill (e.g. divergence at offset 15) — `prompt_cache_state.cache` is REPLACED with a brand-new cache, but `snap@27` still holds refs to the OLD cache's state arrays. Turn N+2 then rewinds to offset 38 (in the new cache); `find_nearest(38)` returns `snap@27`; restore installs OLD-cache state into the NEW cache's DeltaNet layers. KV layers carry new-cache content for [0..27); DeltaNet state was conditioned on old-cache content for [0..27). Inconsistent → wrong logits → silently wrong outputs (deterministically, at temp=0.0). **Fix:** new `DeltaNetSnapshotRing.drop_after(offset)` plus a divergence walker in `PromptCacheState.update()`. Before overwriting `self.token_ids`, compute the position where new vs. old tokens first differ; if it's strictly inside the old sequence, call `ring.drop_after(divergence)`. All snapshots whose offset references tokens that are no longer in the live cache get evicted. The mx.array refs are released as a memory bonus. Logged at DEBUG: `Snapshot ring: dropped N stale snapshot(s) past token-sequence divergence at offset M (cached len=X, new len=Y).` Verified end-to-end: pre-fix Turn 3a/b/c on the test scenario deterministically produced "3" instead of the ground-truth "6"; post-fix produces "6" reliably.

**13. RoPE Desync, Step-Padding, and the SWA Ring Buffer Rewind (generate.py)**
* **The Problem:** Rewinding context during a chat branch caused severe hallucination loops. Padded standard caches (inflated by `PREFILL_STEP_SIZE`) bypassed shape-based slice checks, leaving ghost tokens. Worse, Gemma 3/4 uses Sliding Window Attention (`RotatingKVCache`). Rewinding into the overwritten region of a wrapped ring buffer creates a "Memory Hole", causing uniform probability anomalies during Softmax.
* **The Fix:** Implemented a type-aware "Smart Rewind". For SWA ring buffers, a `_rotating_rewind_safe()` check determines whether the rewind target is still within the buffer's valid data range (`offset <= max_size` means the buffer hasn't wrapped, so all positions are valid). Only if the target is in the overwritten region (`target < offset - max_size`) is the cache dropped and a full re-prefill forced. For typical multi-turn conversations under the window size (1024 for Gemma 3, 512 for Gemma 4), the rewind is always safe and prefix cache reuse works across turns. For standard dynamic caches, the arrays are unconditionally physically sliced using `_kv_seq_axis()` to strip both ghost tokens and step-padding, ensuring RoPE indices remain perfectly aligned.

**14. Single-Worker Queue Bottlenecks (OpenWebUI)**
* **The Problem:** Follow-up questions appeared to hang the server. In reality, OpenWebUI was firing sequential background tasks to the 31B model, monopolizing Uvicorn's single worker queue while the UI waited.
* **The Fix:** Identified as an architectural constraint in OpenWebUI (not an mlx-vlm code change). Required configuring OpenWebUI to isolate "Task Models" (titles, tags, autocomplete) to a secondary, lightweight local API (e.g., 2B quantization) to keep the primary reasoning worker unblocked.

**15. The Tool Call Swallower (server.py)**
* **The Problem:** If the model stuttered and hallucinated a broken `<|tool_call>` sequence, the `StreamingTranslator` buffered the output indefinitely waiting for valid JSON, effectively silencing the stream and hanging the UI.
* **The Fix:** Patched the stream generator to safely rescue and yield the buffered text as standard conversational output if a tool call fails to resolve, preventing silent drops. Rescued text is also passed through `sanitize_strict_json` for consistency.

**16. Ghost Prompt Math Guard — REMOVED (server.py)**
* **The Problem:** A hidden system prompt ("Enclose inline math in \( and \)...") was injected into every chat request to enforce KaTeX-compatible delimiters for OpenWebUI.
* **Why Removed:** The ghost prompt caused small models (1B-3B) to hallucinate LaTeX math instead of following the user's actual request — the math instruction overwhelmed their limited attention budget. The server should not silently inject system prompts. Users who need math formatting should include it in their own system prompt.

**17. EOS Token Fix for stream_generate() (generate.py)**
* **The Problem:** `stream_generate()` never reset the stopping criteria — only `generate()` did (at line 1009). Since the server calls `stream_generate()` directly, the stopping criteria was stale. Worse, some model configs (e.g., Gemma 3 1B) only list `<eos>` (ID 1) as the EOS token but omit `<end_of_turn>` (ID 106/107), so the model generated past the real response and filled remaining tokens with garbage.
* **The Fix:** `stream_generate()` now resets the stopping criteria with `model.config.eos_token_id` on each call. It also merges chat-template stop tokens (`<end_of_turn>`, `<|im_end|>`, `<|endoftext|>`) by resolving them from the tokenizer's vocab via `convert_tokens_to_ids`, filtering out unknown tokens. This covers Gemma, Qwen/ChatML, and GPT-style models.

**18. enable_thinking Default (server.py)**
* **The Problem:** `TemplateParams.template_kwargs()` defaulted `enable_thinking` to `True`, passing it to `tokenizer.apply_chat_template()` for every request. Models not trained with thinking tokens (e.g., Gemma 3 1B) received `<think>` in their prompt and hallucinated. Every other code path in the project (CLI, chat.py, generate.py) defaulted to `False`.
* **The Fix:** Changed `kwargs.setdefault("enable_thinking", True)` to `False`. Users can still send `"enable_thinking": true` explicitly for models that support it.

**19. Chat Stop Tokens in the Batch Path (server.py — commit `e039083`)**
* **The Problem:** The `ResponseGenerator`/`BatchGenerator` path bypasses `stream_generate()` and uses `stop_tokens` derived from `config.eos_token_id` directly. The fix in #17 only patched `stream_generate()`, so models like gemma-3-1b that list only `<eos>` (ID 1) but omit `<end_of_turn>` (ID 107) ran past the response boundary on the batch path.
* **The Fix:** Resolve chat-template stop tokens (`<end_of_turn>`, `<|im_end|>`, `<|endoftext|>`) from the tokenizer vocab inside `get_cached_model()` and merge them into the batch-path `stop_tokens`, mirroring the streaming-path behavior.

**20. Text-Only Model Support (utils.py & prompt_utils.py — commits `e3f2157`, `a75f1db`, `9457990`; follow-up fixes uncommitted as of 2026-04-28)**
* **The Problem:** Pure text models like `gemma3_text` and Qwen 2.5 Instruct aren't VLMs — they have no vision tower and aren't in `MODEL_CONFIG`. Loading them through mlx-vlm's normal path raised `ValueError "not supported"`, and `MessageFormatter` raised `ValueError` for unknown model types.
* **The Fix:** Added `_TextOnlyLanguageModel`, `_TextOnlyModelWrapper`, and `_SimpleNamespace` wrappers in `utils.py` that adapt mlx_lm text models to the VLM interface. `load()` catches the unsupported-model `ValueError` and falls back to `mlx_lm.utils.load` with tokenizer compatibility patches. `_TextOnlyLanguageModel` inspects the model's `__call__` signature and only forwards accepted kwargs, stripping VLM-specific extras like `attention_mask_4d`. `MessageFormatter` (prompt_utils.py) now falls back to plain text formatting for unknown model types instead of raising — there are no image/audio tokens to insert anyway.
* **Follow-up fix #1 — `make_cache` contract violation (utils.py, uncommitted 2026-04-28):** The original `_TextOnlyLanguageModel.make_cache()` returned `None` when the underlying mlx_lm model didn't implement `make_cache` (e.g., Qwen 2.5, which uses external `make_prompt_cache` instead). But `generate._make_cache()` (generate.py:1630) does `hasattr(model, "make_cache")` to decide between calling it and falling back to building default `KVCache` objects from `model.layers`. The wrapper always answered `True` to `hasattr`, the fallback never ran, and `len(None)` blew up at generate.py:1632 with `TypeError: object of type 'NoneType' has no len()`. **Fix:** in `_TextOnlyLanguageModel.__init__`, conditionally bind `self.make_cache = lm_model.make_cache` *only* when the underlying model has it. `nn.Module.__setattr__` accepts bound-method assignment cleanly, so `hasattr` correctly reports whether there's a real implementation to forward to. mlx_lm models without `make_cache` now take the proper fallback path. Verified with `mlx-community/Qwen2.5-1.5B-Instruct-4bit` — loads in 1 GB peak, responds in <0.4s.
* **Follow-up fix #2 — Misleading log severity (utils.py:227, uncommitted 2026-04-28):** `get_model_and_args()` logged `ERROR` and raised `ValueError` when the model wasn't in `mlx_vlm.models` or `mlx_vlm.speculative.drafters`, but every known caller funnels through `load()` which catches the ValueError and falls back to mlx_lm. The ERROR was therefore always premature noise during the normal text-model loading path. **Fix:** demoted to `logger.warning(...)` with a clearer message that names both packages tried and notes the caller may fall back. The "not supported" phrase in the message body is load-bearing — `load()` greps for it to decide whether to invoke the fallback — so it stayed.

**21. Strict JSON Output Sanitization — Non-Streaming Endpoints Only (utils.py & server.py — commits `e3f2157`, `cbe1ac8`)**
* **The Problem:** When a model is asked to emit raw JSON (especially with embedded LaTeX math), `json_repair`'s default behavior treats valid JSON escapes like `\f`, `\b`, `\n` as control characters and silently destroys the math content. Plain-prose responses must not be touched.
* **The Fix:** `sanitize_strict_json(text)` in `utils.py` first detects intent — only acts if the text starts with `{`, `[`, or `` ```json ``. For non-JSON output it returns the input untouched. For JSON-intended output, it pre-escapes math blocks (`$$…$$`, `$…$`, `\[…\]`, `\(…\)`) by double-escaping any backslash not already followed by a backslash or quote, then runs `json_repair`. Wired into the non-streaming `/v1/chat/completions` (server.py:2573) and `/v1/responses` (server.py:1944) endpoints. **The streaming path is intentionally NOT sanitized** — token-by-token deltas can't be repaired without a buffering rewrite, and streaming users shouldn't be paying that latency cost. This means streamed code blocks pass through verbatim (relevant context for the code-fence findings below).

**22. Server-Side Thinking Budget Enforcement (server.py — commit `ffad5df`)**
* **The Problem:** Reasoning models can enter self-reverification loops on multi-constraint prompts (see "Model Behavior Findings" below). With no hard cap, a single request can burn 80K+ thinking tokens before producing any visible output.
* **The Fix:** `_compute_thinking_budget()` (server.py:935) returns `0.80 × max_tokens` when a thinking format is detected (`<|channel>thought` for Gemma, `<think>` for generic), or honors an explicit `thinking_budget` field on the request. The streaming loop at server.py:2218 increments `thinking_tokens` while `in_thinking` is true; once it exceeds the budget, the server emits `THINKING_TRUNCATION_MSG` with `finish_reason="length"` and breaks the stream. `THINKING_BUDGET_RATIO` is the constant at server.py:58. Format detection lives in `_detect_thinking_format()` (server.py:910).

**23. Structured Logging & `--log-file` Flag (server.py, generate.py, utils.py — commits `119d0b9`, `170c30f`)**
* **The Problem:** Diagnostic output was a mix of `print()` calls and ad-hoc `logging.info` blocks. No way to redirect to a file, no consistent module loggers, no preview-truncation for long prompts (which spammed terminals at 6K+ tokens).
* **The Fix:** Replaced all `print()` with `logger.info/warning/error/debug` across server, generate, and utils. Module loggers derive from `MLX_VLM_LOG_NAME` (default `mlx_vlm`); `MLX_VLM_LOG_LEVEL` env var and `--log-name` CLI arg control verbosity. Added `DEBUG_PREVIEW_CHARS` constant and `_truncate()` helper for head+tail previews of long prompts/responses. `traceback` import dropped in favor of `exc_info=True` on `logger.error`. The `--log-file PATH` flag (default: `<stdout>`) lets ops redirect to file or journald cleanly. Supersedes the original ad-hoc telemetry pipeline mentioned in #10.

**26. Per-Thread MLX Streams in generate.py (uncommitted as of 2026-04-28)**
* **The Problem:** `generate.py:295` originally registered `generation_stream = mx.new_thread_local_stream(mx.default_device())` at module import time. The stream is thread-local — only registered on the import thread (typically FastAPI's main thread). Inside `generate_step`, ops are wrapped in `with mx.stream(generation_stream)`, binding result arrays to that specific stream's id. Trailing ops outside the `with` (e.g. `mx.async_eval(y)` at generate.py:995) require that exact stream to be available on the calling thread. Worker threads (asyncio's ThreadPoolExecutor, server's per-chat session executor) hit `RuntimeError: There is no Stream(gpu, 1) in current thread` because the stream id is only valid on the import thread.
* **The Fix (architectural):** Replaced the module-level singleton with a lazy per-thread stream:
  ```python
  _thread_local_streams = threading.local()
  def _get_generation_stream():
      stream = getattr(_thread_local_streams, "stream", None)
      if stream is None:
          stream = mx.new_thread_local_stream(mx.default_device())
          _thread_local_streams.stream = stream
      return stream
  ```
  All 6 `with mx.stream(generation_stream):` sites + `wired_limit(model, [generation_stream])` + `BatchGenerator.__init__`'s `self._stream = stream or generation_stream` now call `_get_generation_stream()`. Each thread that enters generate.py for the first time gets its own thread-local stream lazily; arrays computed there are bound to the calling thread's stream and trailing async_eval ops find it correctly.
* **Backward compatibility:** Existing imports of `generation_stream` as a module attribute still resolve via a module-level `__getattr__` shim that materializes the calling thread's stream on attribute access. No caller needs to change.
* **Why this beats the workaround approaches considered:**
  - Monkey-patching `generate.generation_stream` from a dedicated worker thread — fragile, mutates shared state, breaks any other thread that uses the module.
  - Single-thread executor + import-time stream — limits server throughput to one concurrent legacy call, and still requires the executor's worker thread to "own" the import-time stream (impossible without monkey-patching).
  - Wrapping calls in `with mx.stream(generation_stream):` from worker threads — doesn't fix trailing ops outside the `with`, since arrays are bound by stream id, not by active context.
* **Server-side cleanup:** With this fix in place, the workaround machinery in `server.py` (the `_ensure_worker_thread_stream` helper, outer `with mx.stream(...)` wrappers around legacy paths) is no longer needed and was removed. The legacy chat-completion paths just call `generate()` / `stream_generate()` directly; per-thread stream creation happens transparently.
* **Validation:** Smoke tests confirm: distinct streams across 4 concurrent threads, stable cache within the same thread, backward-compat module attribute access, end-to-end generation through `asyncio.to_thread` with a real model (Qwen 2.5 1.5B) producing correct output.

**25. Server-Side Per-Chat PromptCacheState (server.py — uncommitted; Phase 1 reverted, Phase 2 shipped 2026-04-29)**
* **The Goal:** Per-chat prefix-cache reuse and DeltaNet snapshot rewind for the streaming server, keyed by `chat_id`. Without this, every request fresh-prefills the entire conversation and the snapshot rewind (#13a) is never exercised through the server (it lives inside `stream_generate`'s prefix-cache reuse logic).
* **Phase 1 attempt (reverted 2026-04-29):** routed chat_id'd requests through the legacy `stream_generate`/`generate` path via `asyncio.to_thread` worker. Fundamentally broken because **the model is loaded on `ResponseGenerator`'s daemon thread** (`server.py:_run` calls `_initialize_model()` before the BatchGenerator loop). Model weights become mx arrays bound to streams of the loading thread; any forward pass on those weights from a different thread fails at `mx.async_eval(y)` with `RuntimeError: There is no Stream(gpu, 1) in current thread`. MLX's threading model: arrays are bound to streams, streams are accessible only on the thread that owns them. Per-thread stream registration (`mx.new_thread_local_stream` / `mx.default_stream`) on a worker doesn't help — the WEIGHTS' stream is what matters.
* **Phase 2 (shipped 2026-04-29):** route the cached path through the same daemon thread that owns the model. `ResponseGenerator._run` (server.py:644) dispatches by inspecting incoming requests for an attached `prompt_cache_state`; non-cached requests flow into the existing continuous-batching path; cached requests go to a new `_process_cached_request()` method (server.py:444) that runs the prefix-match, snapshot-rewind, prefill, and generation loop inline on the daemon thread. Shares the model and stream context with BatchGenerator so cross-thread issues vanish. **Speculative decoding is bypassed for cached requests** (TODO 1 below); cached path uses non-speculative `stream_generate`. **Implementation breakdown:**
  - **Phase 2.A** — request submission plumbing. Request queue tuple expanded from 5 to 7 elements: `(rqueue, raw_inputs, prompt_tokens, args, images, prompt_cache_state, formatted_prompt)`. `ResponseGenerator.generate()` accepts `prompt_cache_state` kwarg. Both `_run` and `_run_speculative` updated to unpack the new tuple.
  - **Phase 2.B** — `_process_cached_request(rqueue, prompt, images, args, prompt_tokens, prompt_cache_state)` runs synchronously on the daemon thread. Builds `gen_kwargs` from `args.to_generate_kwargs()` plus `prompt_cache_state` and KV-quant config; calls `stream_generate(...)`; converts each `GenerationResult` into a `StreamingToken` and pushes to rqueue. Cooperative cancellation via `self._cancelled` / `self._cancel_lock`. Final `None` sentinel pushed in `finally`.
* **Audit-pass fixes (during Phase 2.B implementation, uncommitted 2026-04-29):**
  - **`uid` allocation bug:** original handler did `uid = self.uid_count; self.uid_count += 1` but `ResponseGenerator` has no `uid_count` attribute. Would AttributeError on first cached request. Fixed by switching to `uid = id(rqueue)` to match the BatchGenerator convention (server.py:778). Same `_cancel(uid)` machinery works uniformly.
  - **`logprobs` shape mismatch:** `GenerationResult.logprobs` is `Optional[List[float]]` (full per-vocab logprob vector), but `StreamingToken.logprobs: float` expects the chosen-token scalar. Original handler did `float(chunk.logprobs)` → TypeError on list. Fixed by extracting the chosen token's logprob: `chunk.logprobs[token_id].item()` with `0.0` fallback when None or indexing fails. Matches BatchGenerator's `r.token_logprob` scalar contract that `_make_logprob_content` (server.py:2624) expects.
  - **`finish_reason` not surfaced:** `GenerationResult` had no `finish_reason` field, so the cached path's terminal chunk couldn't tell the OpenAI SSE consumer whether to emit `"stop"` or `"length"`. Architecturally-correct fix instead of a synthetic terminator: added `finish_reason: Optional[str] = None` to `GenerationResult` (generate.py:472); `stream_generate` tracks `stopped_by_criteria` flag inside the generation loop and sets `finish_reason="stop"` (criteria match) or `"length"` (generator exhaustion) on the post-loop final yield only (generate.py:1547). Per-token chunks always carry `None`; only the terminal chunk carries the reason. Also fixes the legacy fallback path (server.py:2740) which had the same gap.
* **What survived the Phase 1 revert (still in working tree, consumed by Phase 2):**
  - `_session_caches` `OrderedDict` + LRU + `get_or_create_prompt_cache_state` + `clear_session_caches` (called from `unload_model_sync`)
  - `_resolve_chat_id` (header → body → metadata; no fallback hash)
  - Module-level config holders `_session_cache_max`, `_chat_id_header`, `_deltanet_ring_size`, `_deltanet_rewind_enabled`. `main()` populates them after argparse, which itself bakes env-var fallbacks into `default=` (see `_env_int` / `_env_choice` helpers in server.py). Earlier `get_session_cache_max()` / `get_chat_id_header()` getter functions were eliminated — defaults belong in argparse, not in on-demand env reads.
  - CLI args: `--cache-session-max`, `--cache-chat-id-header`
  - Endpoint integration: `chat_id` is resolved and `prompt_cache_state` is looked up on every chat completion. Phase 2 plumbs it into `response_generator.generate(..., prompt_cache_state=...)`.
* **What was removed in the revert:**
  - `_legacy_generate_blocking` worker (non-streaming branch)
  - `_legacy_stream_worker` + `Queue` bridge (streaming branch)
  - `use_legacy_path` routing (`if response_generator is not None and not use_legacy_path:`)
  - The cross-thread workaround helpers (`_ensure_worker_thread_stream`, `mx.stream(...)` wrappers)
* **Configuration (still active, will be consumed by Phase 2):**
  - `--cache-session-max N` / `MLX_VLM_CACHE_SESSION_MAX=N` (default 8)
  - `--cache-chat-id-header NAME` / `MLX_VLM_CACHE_CHAT_ID_HEADER=NAME` (default `X-MLX-VLM-Chat-Id`)
* **Architectural lesson:** in MLX 0.31.2+ the model is the anchor — all forward passes on a model must run on the thread that loaded it. Async offload patterns (`asyncio.to_thread`, separate executors) don't work for inference on the same model. Either the model is loaded on the worker thread (impossible to change after server start) OR all work is routed to the loading thread (Phase 2 approach). Phase 2 lesson recorded in feedback memory.
* **End-to-end validation (2026-04-29):** `dev/test_session_cache_curl.sh` covers (1) main multi-turn diagnostic (Turn 3a/b/c at temp=0.0 must produce identical output → confirms divergence-drop fix), (2) streaming SSE final-chunk `finish_reason="stop"`, (3) restore-path scenario (R1→R2 verbatim extension → R3 regenerate). Verified on `unsloth/Qwen3.6-27B-UD-MLX-6bit`: R3 fires `WARNING - DeltaNet snapshot rewind: restored from offset 27, replaying 21 tokens to reach prefix 48`, output matches R2 byte-for-byte → restore-and-replay code is correct end-to-end.
* **Phase 2 followup TODOs (deferred; revisit after stabilization):**
  - **TODO 1 — Speculative decoding + cached path integration.** Phase 2 cached requests run on the daemon thread via a non-speculative `stream_generate`-style path. With `--draft-model` configured, non-cached requests get speculative speedup but cached requests don't. Wiring `prompt_cache_state` through `_run_speculative` requires: skipping fresh `_make_cache(...)` construction, trimming the cache to `prefix_len` BEFORE speculative prefill runs, DeltaNet snapshot restore, suffix-only speculative prefill, and coordinating snapshot capture with speculative-rollback boundaries (intra-block rollback in `qwen3_5/language.py:450-558`). Estimated ~50-80 additional lines plus interaction testing between snapshot rewind (turn-boundary) and speculative rollback (intra-step) — they shouldn't conflict but worth verifying. Workload impact when integrated: cached requests gain ~1.5-3x per-token speedup. For long multi-turn conversations dominated by prefill, this is incremental (cache reuse already saves ~10-100x on prefill). Worth doing if profiling shows per-token rate matters.
  - **TODO 2 — Cache-aware batching for cached requests (Phase 3 territory).** Phase 2 processes cached requests serially. Multiple concurrent `chat_id`s = sequential daemon dispatch. Batching them requires per-uid cache slicing inside `BatchGenerator` (different prefix lengths, different snapshot offsets, different DeltaNet states). Significantly more complex than single-request cached mode. Relevant for multi-user OpenWebUI; not for single-user local use.
  - **TODO 3 — Streaming `BatchGenerator` cache-state integration.** Natural extension of TODO 2 — cached requests bypass continuous batching entirely; non-cached requests benefit from it. Same cache-slicing complexity blocks it. Implement only if multi-user batching becomes a real workload requirement.

**24. OpenWebUI Advanced Params End-to-End Plumbing (server.py)**
* **The Problem:** OpenWebUI v0.9.2 forwards Advanced Params and Custom Parameters verbatim to OpenAI-compatible endpoints (no allowlist filter on the OpenAI router), but several knobs the user expected to work were silently dropped server-side:
  - **`repeat_penalty`** — OpenWebUI's native UI slider uses the Ollama-style name. The server only read `repetition_penalty`. The slider was theater.
  - **`seed`** — declared on `VLMRequest` with default `DEFAULT_SEED=0` and never plumbed into `_build_gen_args` or applied to the sampler. Worse, the non-None default meant every request was implicitly "seeded with 0", which would have eliminated variance if the seed had been wired up.
  - Other knobs (`thinking_budget`, `thinking_start_token`, `repetition_penalty` proper) worked end-to-end but were not surfaced in the OpenWebUI UI — required manual Custom Parameters JSON.
* **The Fix (server.py):**
  - `_build_gen_args` (server.py:863): accept `repeat_penalty` as an alias for `repetition_penalty` so OpenWebUI's native slider works without Custom Params. One-liner: `repetition_penalty = getattr(request, "repetition_penalty", None) or getattr(request, "repeat_penalty", None)`.
  - `_build_gen_args` (server.py:869): plumb `seed=getattr(request, "seed", None)` through into `GenerationArguments`.
  - `_make_sampler` (server.py:377): when `args.seed is not None`, call `mx.random.seed(args.seed)` once at sampler construction time. Comment documents the caveat — MLX PRNG is process-global, so under continuous batching this is best-effort determinism (interleaved batches share state).
  - `VLMRequest.seed` (server.py:1448): changed from `int = Field(DEFAULT_SEED)` to `Optional[int] = Field(None)` so omitted seed means "don't reseed" instead of "reseed to 0 every time". Removed the now-unused `DEFAULT_SEED` import.
* **The Fix (`openwebui/` filter family):** Seven Filter Functions, each a self-contained file. Toggle them on/off per-chat in the chat tools menu. Mutual-exclusion enforced via shared body markers — second filter to detect a conflicting marker raises `ValueError` with a clear message.

  | File | Title in OpenWebUI | Sets marker | Errors on |
  |---|---|---|---|
  | `thinking.py` | Thinking | `_mlx_thinking_active` | `_mlx_advanced_active` |
  | `advanced.py` | Advanced Params | `_mlx_advanced_active` | any other marker |
  | `profile_strict.py` | Profile · Strict | `_mlx_active_profile` | another profile, or advanced |
  | `profile_explore.py` | Profile · Explore | same | same |
  | `profile_math.py` | Profile · Math | same | same |
  | `profile_casual.py` | Profile · Casual | same | same |
  | `profile_creative.py` | Profile · Creative | same | same |

  **Compatibility matrix:** Thinking + Profile = OK. Profile + Profile = error. Advanced + anything else = error (Advanced is "manual mode" and refuses to coexist).

* **Family-aware profile params (added 2026-04-28).** Each profile detects the model family from the request's `model` id and applies family-correct params. Detection is most-specific-first substring match; community/quantizer prefix (`mlx-community/`, `unsloth/`, `lmstudio-community/`, etc.) is stripped before matching, since UD/dynamic vs uniform quantization does not change recommended sampling params per Unsloth and upstream docs.

  Supported families per profile (researched against official model cards and `generation_config.json` files):
  - **Gemma 4 / 3** — uses `repetition_penalty` for loop mitigation; cap at 1.10 (formatting fidelity degrades above)
  - **Qwen 3.x / 3.5 / 3.6** — uses `presence_penalty` (Qwen team explicitly forbids `repetition_penalty != 1.0`; causes language mixing); separate `qwen3_moe` entry for the 35B-A3B MoE variant
  - **Qwen 2.5** (incl. Coder, VL) — distinct from 3.x: ALLOWS `repetition_penalty=1.05`. Detection table matches `qwen2.5` strictly before falling through to a generic `qwen` rule.
  - **Llama 3.x** (3.1/3.2/3.3 share params) — uses `frequency_penalty`; do NOT set `top_k`
  - **DeepSeek R1** — temperature bounded to 0.5–0.7 (R1 produces endless repetition outside that range); deliberately excluded from Creative profile
  - **DeepSeek V3** — supports `<think>` opt-in; tools available
  - **Mistral Small 3.x** — temperature ceiling at 0.7 (officially tuned for 0.15); profiles cap accordingly

  Unrecognized families error with: *"model X is from an unrecognized family. Use the Advanced Params filter, or add this family to the profile's `_FAMILY_DETECTION` and `_PARAMS_BY_FAMILY` tables."* The user's "Qwen attempting to set pinned values" concern is now structurally impossible — profiles never write `repetition_penalty` on Qwen 3.x families because their `_PARAMS_BY_FAMILY[qwen3]` entry uses `presence_penalty` instead.

* **Profile semantics — meaningfully tuned per workflow:**
  - **Strict** — implementation under constraints; loop-mitigation profile. Low temp, family-correct repetition control, capped thinking budget.
  - **Explore** — design brainstorming, architectural research, tech-doc writing. Higher temp, mild repetition control, large thinking budget.
  - **Math** — calculus, proofs, formal logic. Low temp, NO repetition penalty (math benefits from re-stating equations), highest thinking budget.
  - **Casual** — quick everyday Q&A. Optimized for snappy responses, low max_tokens, no thinking_budget set (defers to Thinking filter).
  - **Creative** — essays, fiction, non-technical writing. High temp, widened top_p, model-family-specific ceiling enforcement (Mistral capped at 0.7, R1 excluded entirely).

* **Advanced filter footgun warnings (in `advanced.py` docstring):** Qwen 3.x rep_penalty pin = 1.0; Qwen 3.x temperature must be > 0; Gemma rep_pen ceiling = 1.10; DeepSeek R1 temperature range 0.5–0.7; Mistral Small temperature ceiling 0.7; Llama do not set `top_k`. These document the known model-creator-mandated constraints — the Advanced filter doesn't enforce them (it's manual mode and lets the user experiment), but the docstring lists them so users know what to avoid.

* **UserValves only on Thinking and Advanced filters.** Profile filters intentionally have no UserValves — their params are baked, deterministic per family. If a user wants to tweak a single knob within a profile, they switch to Advanced (which is incompatible with profiles by design — that mutual-exclusion is the feature, not a bug).

* **Bootstrap script (per-model OpenWebUI Advanced Params) should bake:**
  - `top_k` per family (Gemma 4: 64, Qwen: 20)
  - `repetition_penalty=1.0` for Qwen 3.x (defense in depth even though profiles auto-route)
  - Reasoning Tags (`<|channel>thought` / `<channel|>` for Gemma 4, `<think>` / `</think>` for Qwen 3.x and DeepSeek)
  - Native tools enable/disable per family
  - Context window cap (max_kv_size)
  Leave `temperature`, `top_p`, `max_tokens`, `thinking_budget`, repetition controls UNSET in per-model defaults — those are governed by the per-chat profile/thinking/advanced toggles.

* **Penalty plumbing — full streaming-path fix (uncommitted as of 2026-04-28):** Until this fix, `repetition_penalty` was plumbed only through the legacy non-streaming `generate()` path; the streaming/batched path (which OpenWebUI almost always uses) silently dropped it. `BatchGenerator` only accepted a `sampler` callable — `GenerationBatch._step` called `self.sampler(logprobs)` with no logits-processor chain. So the user's profile filters and OpenWebUI's `repeat_penalty` slider have been theater on streaming requests up to this point. Adding `presence_penalty` and `frequency_penalty` via the same plumbing would have been more theater. The fix:
  - `GenerationArguments` (server.py:127): added `presence_penalty`, `frequency_penalty` fields; `to_generate_kwargs` emits them when set.
  - `_build_gen_args` (server.py:864): extracts them from the request.
  - `VLMRequest` (server.py:1467): added Pydantic Field declarations with descriptions noting Qwen 3.x's required `presence_penalty` and Llama's `frequency_penalty`.
  - `generate_step` (generate.py:766): added `presence_penalty`, `presence_context_size`, `frequency_penalty`, `frequency_context_size` params; forwards them to `make_logits_processors` (mlx_lm already supports all 7 args, the fork was just calling with 3 of them).
  - `GenerationBatch` (generate.py:1751): added `logits_processors` param (per-uid: `List[Optional[List[Callable]]]` after the 2026-05-01 upstream merge, which switched from a flat shared list to per-sequence lists alongside a new `token_context` parameter). Tracks per-sequence rolling token history in `token_context`, capped at `_PENALTY_HISTORY_SIZE=64`; applies processors per-row in `_step` on raw logits by slicing `logits[i:i+1]` and passing `mx.array(self.token_context[i])`. Necessary because mlx_lm's processors are written for single-sequence inputs (`logits[:, tokens]` indexes the same tokens across all batch rows). `extend()` / `filter()` / `empty()` maintain `token_context` correctly when sequences are inserted/removed mid-batch. (Originally tracked in a separate `_histories` field; collapsed into `token_context` on 2026-05-01 after the upstream merge made `token_context` the canonical buffer.)
  - `BatchGenerator` (generate.py:2215): added `logits_processors` param; forwards to `GenerationBatch.empty` and to `PromptProcessingBatch.generate` at both transition sites.
  - `PromptProcessingBatch.generate` (generate.py:2147): forwards `logits_processors` to the new `GenerationBatch` it creates.
  - Server `_run` (server.py:507): builds processors via new `_make_logits_processors(args)` helper that calls `mlx_lm.sample_utils.make_logits_processors` with all four penalty/bias args. Returns `[]` when nothing is configured so the per-row processor loop is skipped entirely.
  - Server `_make_logits_processors` (server.py): mirrors `_make_sampler` — same continuous-batching caveat applies (processors are constructed from the FIRST request's args and shared across the active batch). Acceptable for single-user OpenWebUI workloads.
* **Verified end-to-end:** Smoke tests confirm `make_logits_processors` returns the right number of processors for each penalty config (0 if none, 1 for Qwen presence-only, 3 for full Gemma/Llama setup) and applies penalties correctly (e.g., presence_penalty=1.5 subtracts 1.5 from logits of tokens already in history). All schema fields, kwargs, and call sites verified via Python introspection.

**27. Cache-friendly chat-template rendering — `CACHE_ALIGNMENT_KWARGS` (prompt_utils.py + server.py — uncommitted as of 2026-04-29)**
* **The Problem:** Some chat templates render the LATEST assistant header differently from PRIOR assistant turns. The asymmetry breaks prefix-cache reuse — turn N's CACHED tokens won't match turn N+1's RENDERED tokens at the prior-assistant boundary, forcing full re-prefill. On hybrid models (DeltaNet / `ArraysCache`) this is catastrophic because the rewind guard requires a snapshot at offset ≤ divergence_point, which doesn't exist when divergence is mid-turn-prefix.
* **Concrete observation (Qwen 3.6 unsloth port):** `unsloth/Qwen3.6-27B-UD-MLX-6bit`'s template injects `<think>\n\n</think>\n\n` in the latest-assistant header (when `enable_thinking=False`) but renders prior assistants as bare `<|im_start|>assistant\n{content}`. T1 cached (27 tokens) vs T2 rendered (44 tokens default) → common prefix only 15. With `preserve_thinking=True` template kwarg + bare assistant content from OpenWebUI: T2 renders to 48 tokens, common prefix 27/27 — full alignment.
* **Critical finding from cross-template verification:** This asymmetry is ENTIRELY in unsloth's modified template, NOT in canonical Qwen. Empirically verified across `Qwen/Qwen3-Next-80B-A3B-Instruct` (the official DeltaNet flagship), `Qwen/Qwen3-32B`, `Qwen/Qwen3-30B-A3B-Instruct-2507`, `Qwen/Qwen3-235B-A22B-Instruct-2507`, `Qwen/Qwen3-VL-30B-A3B-Instruct`: all render assistant turns symmetrically, T1 is a strict prefix of T2 by default, cache reuse works natively. `preserve_thinking` is exclusively an unsloth-port escape hatch.
* **The Fix (registry-style):** New `CACHE_ALIGNMENT_KWARGS` flat dict in `prompt_utils.py` lists chat-template kwargs known to enable symmetric rendering. Initial entry: `{"preserve_thinking": True}`. Server's `chat_completions_endpoint` resolves `chat_id` BEFORE `apply_chat_template`; when `prompt_cache_state is not None`, merges the registry kwargs into the call. Ordering change at server.py:2456 → chat_id lookup moved above the template render.
* **Why no model-family / org-prefix gating:** the kwargs are gated by Jinja's `is defined` guard inside the template — passing a kwarg that the template doesn't reference is a true no-op (verified: T2 tokens byte-identical with vs without `preserve_thinking` on canonical Qwen templates). So always-on when caching is active is safe AND robust to model swaps. No `model_type`-keyed dispatch, no `unsloth/` prefix sniffing — just a flat list of "known cache-alignment kwargs" applied unconditionally.
* **What this fix covers:**
  - Templates that DEFINE the kwarg → fixes alignment (unsloth-style ports)
  - Templates that DON'T DEFINE the kwarg → silent no-op (canonical Qwen, Gemma, others)
  - Asymmetric templates with NOVEL kwarg names → still broken, requires registry addition (graceful failure: full re-prefill, correct outputs, observable via the now-WARNING-level `Snapshot ring: dropped ...` and `Hybrid-Cache Rewind Guard: no snapshot available ...` logs)
* **Adding a new entry:** discover a chat template that breaks cache reuse, find the `is defined` escape-hatch kwarg in its source, verify the kwarg is a no-op on canonical templates by tokenizing identical multi-turn conversations and comparing token sequences. Then add the entry with a comment naming the template family.
* **Log-level bumps (generate.py):** `DeltaNet snapshot rewind: restored from offset N` and `Hybrid-Cache Rewind Guard: no snapshot available` both bumped from DEBUG to WARNING. Rewind indicates unusual flow (regenerate / message edit) worth surfacing without DEBUG enabled. The "no snapshot available" warning specifically signals "consider whether the chat template needs a CACHE_ALIGNMENT_KWARGS entry."
* **Validation:** end-to-end test `dev/test_session_cache_curl.sh` covers a separate restore-path scenario (R1 fresh prefill → R2 multi-turn extension → R3 regenerate). With `preserve_thinking=True` plumbed: R2's prompt becomes a strict extension of R1's cache, snapshot at offset 27 stays valid through R2's update, R3 fires the snapshot-restore-and-replay path. Without the fix: R2 diverges at offset 15, snapshot is invalidated by `drop_after`, R3 falls through to full re-prefill (still correct, but doesn't exercise the restore code).
* **Limitation:** the `responses_endpoint` (server.py:1944 area) does not currently resolve `chat_id` and so doesn't apply the alignment kwargs. Only `chat_completions_endpoint` benefits. Worth aligning if `responses_endpoint` ever gains chat_id support.

---

**Pipeline Configuration**

**Primary model:** Gemma 4 31B 6-bit quantization. Chosen over 4-bit because the workload is autonomous agentic coding — lossless reasoning, strict JSON adherence, and rigid syntactic logic outweigh the 4-bit model's larger context window. 6-bit gives 128k context; 4-bit gives 256k but degrades reasoning quality. Lightweight task routing also tested with Gemma 3 1B 4-bit.

**Key runtime parameters:**
- `PREFILL_STEP_SIZE=512` — sweet spot. 2048 OOMs (9.3 GB activation spike), 128 crashes Metal driver.
- `MAX_KV_SIZE=131072` — static pre-allocation avoids late-stage reallocation spikes (see #8).
- `QUANTIZED_KV_START=5000` — threshold for FP16 → 4-bit migration (see #6, #7).
- `enable_thinking` defaults to `False` in server (see #18) — only enable explicitly for models trained with thinking tokens.
- `--kv-quant-scheme=uniform --kv-bits=8` — preferred over `turboquant` for multi-step symbolic reasoning (see TurboQuant Investigation below).

**OpenWebUI integration constraints:**
- Task models (titles, tags, autocomplete) must be routed to a separate lightweight model (e.g., 2B) to avoid blocking the main reasoning worker on Uvicorn's single worker queue (see #14).
- Images come as `image_url` from OpenWebUI but Gemma's jinja expects `image` (see #3).
- Think blocks use `<think>` (OpenWebUI) not `<|channel>thought` (Gemma native); StreamingTranslator handles the translation (see #2).
- Math rendering: users who need KaTeX-compatible delimiters should include formatting instructions in their own system prompt — the server no longer injects this automatically (see #16).

---

**Model Behavior Findings (Gemma 4 + multi-constraint prompts, 2026-04-27)**

These are NOT mlx-vlm bugs — they are model behaviors that users of this fork need to know about, because the symptoms can look like server bugs. The streaming pipeline does not strip code fences (server.py emits `token.text` directly as `delta_content` at server.py:2257); whatever the UI sees is what the model produced.

**A. Code-fence dropping is variant-dependent.**
On the 12-constraint prime-filter test, the model would generate properly indented Python in its reasoning trace using ```python fences, then emit the *final* answer as bare unfenced code. In OpenWebUI this renders as collapsed paragraph text (markdown strips leading whitespace), looking truncated. Cause: the model commits to *"I will provide just the code"* inside its reasoning; by the time the final answer streams, the system-prompt instruction is thousands of tokens behind. High temperature compounds the variance.

| variant | mitigation result |
|---|---|
| `mlx-community/gemma-4-26b-a4b-it-8bit` | temp ≤ 0.7 + strict directive (*"Every code snippet, including single lines, MUST be enclosed in triple-backtick fences with a language tag. Bare unfenced code is invalid output."*) — **reliable** |
| Gemma 4 31B 4-bit UD (unsloth) | same fix is **intermittent** — first run produced incomplete code block, second run on the same prompt worked. The 4-bit UD quantization lowers format reliability so failures still surface stochastically even at temp 0.7. If reliability matters, lower temp further (0.3–0.5) or prefer the 26B-8bit variant. |

**B. Self-reverification thinking loop on multi-constraint prompts.**
On prompts with 10+ independent constraints, Gemma 4 reasoning can enter a runaway loop: *"One last thing: did I cover constraint N?"*, *"Wait, let me re-verify…"*. Each occurrence of the phrase increases the next occurrence's probability (induction-head dynamics), so the loop self-reinforces. Observed exceeding 80K thinking tokens with no convergence on both 26B-8bit and 31B-4bit-UD.

**Mitigation recipe — send all four together:**

| knob | value | role |
|---|---|---|
| `temperature` | 0.3–0.5 | suppresses the recheck branch entirely |
| `repetition_penalty` | 1.08–1.15 | down-weights "one last thing" / "wait" before they snowball |
| `thinking_budget` | 8000–12000 | hard cap, overrides 80%-of-max formula at server.py:935 (see #22) |
| `max_tokens` | 16384 | cap on total work; auto-budget then ~13K |

`repetition_penalty` is the highest-leverage knob — it directly attacks the loop pattern. The server already exposes both `repetition_penalty` and `logit_bias` (server.py:137-138, server.py:863-864); they're unset by default. OpenWebUI surfaces `repetition_penalty` in Advanced Params per model. If you need a server-side default rather than per-request, patch around server.py:863 to default `repetition_penalty` to ~1.10 for Gemma family models.

---

**TurboQuant Investigation: KV Cache Quantization and Gemma 4 Reasoning Degradation**

**The Observation:** With `kv_bits=8, kv_quant_scheme=turboquant`, the model solves `integral of sec^3` correctly but enters hallucination loops on the follow-up `integral of sec^5`.

**The Debunked Theory (Attention Logit Soft-Capping):** An initial theory blamed TurboQuant for bypassing Gemma's "attention logit soft-capping" (tanh-based score clamping). This is **incorrect**. Code inspection confirmed that Gemma 4 **does not use attention-level soft-capping**. The `logit_softcap()` function exists in `gemma4/language.py:42` but is only applied to **final output logits** (line 565-566, before the LM head), never to attention scores. The config only has `final_logit_softcapping: float = 30.0` — no attention-level soft-cap parameter exists.

**The Real Cause:** TurboQuant completely replaces the entire attention computation (Q*K^T scoring, softmax, V-weighted sum) with codec-based fused Metal kernels. The dispatch happens in `base.py:194` — when a `TurboQuantKVCache` is detected, control bypasses MLX's native `mx.fast.scaled_dot_product_attention` entirely. Even at 8-bit, the accumulated approximation error across the 11 full-attention layers (out of 35 total, with `sliding_window_pattern=5`) compounds enough to degrade multi-step symbolic reasoning. Prompt 1 (sec^3) works because most reasoning happens in FP16 before the 5000-token threshold; Prompt 2 (sec^5) fails because ALL generation occurs under TurboQuant attention when the context exceeds the threshold.

**The Solution: Uniform KV Quantization.** Switching to `--kv-quant-scheme=uniform --kv-bits=8` uses MLX's native `quantized_scaled_dot_product_attention` (in `mlx_lm/models/base.py`), which computes attention via `mx.quantized_matmul` directly on quantized data. This preserves the standard attention flow (scores -> mask -> softmax -> weighted sum) while still saving memory. The only loss is the quantize/dequantize round-trip, which is far less impactful than TurboQuant's full attention replacement.

**Memory Budget (Gemma 4 31B, 6-bit, device limit ~55.7 GB):**
- Architecture: 35 layers, 7 full-attention (KVCache, 4 KV heads, head_dim=512) + 28 sliding (RotatingKVCache, window=512)
- 15 concrete caches (layers 0-14), 20 shared layers. SWA memory is negligible (~25 MB).
- FP16 KV at 256k: ~23 GB cache + ~23 GB weights = ~46 GB (83% of limit — tight)
- 8-bit uniform KV at 256k: ~12 GB cache + ~23 GB weights = ~35 GB (63% — comfortable)
- 4-bit uniform KV at 256k: ~6.5 GB cache + ~23 GB weights = ~29.5 GB (53% — lots of headroom)

**Required Code Fix for Uniform Path:** The upstream `mlx_maybe_quantize_kv_cache` from `mlx_lm` calls `c.to_quantized()` on all caches that cross the threshold, including `RotatingKVCache` which raises `NotImplementedError`. Since Gemma 4's SWA layers have a continuously-incrementing offset (doesn't wrap), they cross 5000 and crash. Fixed by replacing the upstream call with an inline loop that explicitly skips `RotatingKVCache`.

**TurboQuant is still valid** for models that don't require high-precision multi-step reasoning (summarization, simple Q&A), where fused Metal kernel speed matters more than mathematical accuracy.

---

**Dependencies**

- **`mlx>=0.31.2`** (commit `96bf370`): `uv.lock` was bumped because upstream now requires `mx.new_thread_local_stream()`, which lands in 0.31.2. The lockfile previously pinned 0.31.1.

---

**Change Log Over Upstream**

Refresh authoritative commit list with `git log --pretty=format:"%h|%ad|%s" --date=short upstream..HEAD`.

**Upstream base:** `c2058a5` *Fix Kimi VL concurrent Metal crash and mixed-batch text degradation (#1039)*

**Local commits over upstream** (oldest → newest, as of 2026-05-01):

| commit | date | purpose | section |
|---|---|---|---|
| `f34d678` | 2026-04-24 | port: generation pipeline fixes (EOS reset, SWA corruption guard, smart KV-cache rewind) | #6, #13, #17 |
| `44524b5` | 2026-04-24 | port: TurboQuant enhancements (allocation hooks, geometric growth, from_cache) | #8, #12 |
| `e3f2157` | 2026-04-24 | port: text-only model support and JSON sanitization (initial) | #20, #21 |
| `ac19dae` | 2026-04-24 | port: server fixes (`enable_thinking` default → False, thinking-tag strip) | #18 |
| `9457990` | 2026-04-24 | fix: handle unknown model types in MessageFormatter for text-only models | #20 |
| `a75f1db` | 2026-04-24 | fix: text-only model compat (kwarg filtering, unknown model types) | #20 |
| `e039083` | 2026-04-24 | fix: merge chat stop tokens into server's batch-path `stop_tokens` | #19 |
| `96bf370` | 2026-04-24 | chore: update `uv.lock` for `mlx>=0.31.2` | Dependencies |
| `119d0b9` | 2026-04-24 | refactor: convert `print` statements to structured logging across server, generate, utils | #23 |
| `170c30f` | 2026-04-24 | logs configured via `--log-file` with `<stdout>` default | #23 |
| `cbe1ac8` | 2026-04-24 | add back strict JSON output sanitization (`sanitize_strict_json`, non-streaming endpoints only) | #21 |
| `ffad5df` | 2026-04-25 | add thinking budget enforcement (auto = 80% × max_tokens, override via `thinking_budget`) | #22 |
| `59c7843` | 2026-04-27 | bump debug preview size | #23 |
| *(uncommitted)* | 2026-04-27 | OpenWebUI advanced params: server seed plumbing + `repeat_penalty` alias + `openwebui/` filter file | #24 |
| *(uncommitted)* | 2026-04-28 | Text-only model fixes: `_TextOnlyLanguageModel.make_cache` only present when underlying has it; `get_model_and_args` log demoted from ERROR to WARNING (recoverable via mlx_lm fallback) | #20 |
| *(uncommitted)* | 2026-04-28 | OpenWebUI filter family redesign: split single `mlx_vlm_advanced_params_filter.py` into `thinking.py` + `advanced.py` + 5 family-aware profile filters (Strict/Explore/Math/Casual/Creative) with mutual-exclusion via body markers and per-family param routing for Gemma 4/3, Qwen 3.x/2.5, Llama 3.x, DeepSeek R1/V3, Mistral Small 3.x | #24 |
| *(uncommitted)* | 2026-04-28 | Hybrid-cache rewind guard: force full re-prefill when any cache layer reports `is_trimmable()==False` (covers Qwen 3.5/3.6 GatedDeltaNet's ArraysCache; prevents silent state drift on chat-rewind) | #13a |
| *(uncommitted)* | 2026-04-28 | Hybrid-cache rewind: snapshot-based restore via new `mlx_vlm/snapshot.py` (DeltaNetSnapshotRing, FIFO size 3 default). Captures recurrent state at turn boundaries, restores on rewind, replays through prefill machinery. CLI: `--deltanet-rewind`, `--deltanet-ring-size`. Env: `MLX_VLM_DELTANET_REWIND`, `MLX_VLM_DELTANET_RING_SIZE` | #13a |
| *(uncommitted)* | 2026-04-28 → 2026-04-29 | Server-side per-chat PromptCacheState — Phase 1 attempt (legacy-path routing) reverted on 2026-04-29 due to MLX model-thread affinity (model loaded on daemon thread can't be used from worker threads). Infrastructure retained (session manager, chat_id resolution, CLI args) for Phase 2. CLI: `--cache-session-max`, `--cache-chat-id-header`. Env: `MLX_VLM_CACHE_SESSION_MAX`, `MLX_VLM_CACHE_CHAT_ID_HEADER` | #25 |
| *(uncommitted)* | 2026-04-28 | generate.py per-thread MLX streams: replaced module-level `generation_stream = mx.new_thread_local_stream(...)` with lazy `_get_generation_stream()` backed by `threading.local`. Each thread that enters legacy generate / stream_generate gets its own stream — fixes "There is no Stream(gpu, N) in current thread" when called from FastAPI worker threads or any non-import thread. Backward-compat module attribute via `__getattr__` so existing imports of `generation_stream` still resolve. | #26 |
| *(uncommitted)* | 2026-04-28 | Penalty plumbing — full streaming-path fix: added `presence_penalty` + `frequency_penalty` to GenerationArguments / VLMRequest / generate_step; wired logits_processors through BatchGenerator and GenerationBatch with per-sequence token history. Fixes long-standing bug where `repetition_penalty` was silently dropped on the streaming/batched path | #24 |
| *(uncommitted)* | 2026-04-29 | Phase 2 server-side per-chat PromptCacheState **shipped**: `_process_cached_request` on the daemon thread plus 7-element request-tuple plumbing through `ResponseGenerator.generate` / `_run` / `_run_speculative`. Audit-pass fixes during 2.B implementation: `uid = id(rqueue)` (no `uid_count` on class), chosen-token scalar logprob extraction, `GenerationResult.finish_reason` field set in `stream_generate`'s post-loop yield. | #25 |
| *(uncommitted)* | 2026-04-29 | Snapshot ring stale-entry invalidation: `DeltaNetSnapshotRing.drop_after(offset)` plus divergence walker in `PromptCacheState.update()`. When new vs. cached `token_ids` diverge inside the cached length, snapshots referencing the now-discarded suffix are dropped before the new one is captured. Fixes deterministic-wrong-output bug (Turn 3 returned "3" instead of "6" pre-fix) where a stale snapshot from a discarded cache was restored into a fresh cache with different content. | #13a |
| *(uncommitted)* | 2026-04-29 | `CACHE_ALIGNMENT_KWARGS` registry in prompt_utils.py; flat dict of chat-template kwargs known to enable cache-friendly multi-turn rendering (initial entry: `preserve_thinking=True`). Server merges these into `apply_chat_template` when `prompt_cache_state` is active. Verified empirically: no-op on canonical Qwen 3.x templates (`Qwen3-Next`, `Qwen3-32B`, etc.); fixes alignment on unsloth-modified templates. Cross-template verification confirmed the asymmetry is unsloth-port-specific, not a general Qwen design. | #27 |
| *(uncommitted)* | 2026-04-29 | Log level bumps in generate.py: `DeltaNet snapshot rewind: restored ...` and `Hybrid-Cache Rewind Guard: no snapshot available ...` both promoted from DEBUG to WARNING. Rewind = unusual flow (regenerate / message edit) worth surfacing without DEBUG. The "no snapshot" warning explicitly hints at `CACHE_ALIGNMENT_KWARGS` as the likely root cause if it fires every multi-turn request. | #13a, #27 |
| *(uncommitted)* | 2026-04-29 | Strip env-var coupling from snapshot.py and consolidate config-resolution at the argparse layer: removed `get_ring_size` / `get_enable_mode` / env constants from snapshot.py, plus `get_session_cache_max` / `get_chat_id_header` getter functions and the intermediate `_resolve_deltanet_config` helper from server.py. CLI args now bake env-var fallbacks directly into `default=` via `_env_int` / `_env_choice` helpers; `main()` copies args to module-level holders (`_deltanet_ring_size`, `_deltanet_rewind_enabled`, `_session_cache_max`, `_chat_id_header`). PromptCacheState gains `rewind_enabled: bool` field; stream_generate reads it from the cache state instead of calling a global helper. snapshot.py is pure data with one `DEFAULT_RING_SIZE` constant. | #13a |
| `aa3fa67` | 2026-05-01 | merge upstream (`0f903f9` → `c2058a5`): 6 commits — server `json_schema` response_format (new `mlx_vlm/structured.py` + `_build_structured_logits_processors` in server.py), Kimi VL concurrent Metal crash fix, mistral3 sanitize/quantization fixes, Nemotron H Nano Omni model + processor, server `Server` response header. Conflicts resolved in `server.py` (kept both `snapshot` + `structured` imports; kept Ollama `repeat_penalty` alias on top of upstream's two-step `args = GenerationArguments(...)` pattern that sets `args.logits_processors`) and `generate.py` (took upstream's `GenerationBatch.__init__` signature with `token_context` + per-uid `logits_processors: List[Optional[List[Callable]]]`; kept local `_histories` field initialization alongside upstream's `token_context = []` in the `empty()` classmethod). | Change Log |
| `62b47d4` | 2026-05-01 | fix merge artefacts: (1) auto-merge produced duplicate `logits_processors` parameter in `BatchGenerator.__init__` (SyntaxError), removed one. (2) `GenerationBatch._step` had two logits-processor application blocks — upstream's per-uid block on raw logits (correct) and the legacy local block on logprobs iterating the now-per-uid list as if flat (TypeError: list not callable). Removed the broken local block; upstream's path supersedes. | Change Log |
| *(uncommitted)* | 2026-05-01 | collapse `_histories` into capped `token_context` (GenerationBatch). After the merge, per-sequence token history was tracked twice — upstream's `token_context` (read by logits processors) and the local `_histories` (capped at `_PENALTY_HISTORY_SIZE = 64` but no longer read after the broken duplicate apply block was removed in `62b47d4`). Consolidated to a single buffer: dropped `_histories` from `__init__`, `_step` append/cap loop, `extend()` merging, `filter()` trim, and `empty()` classmethod. Ported the cap onto `token_context` append in `_step` (mlx_lm penalty processors slice `[-context_size:]` internally so behavior is unchanged — net 20 lines removed, 3 added). | #24 |
| `4035278` | 2026-05-01 | **Crash fix — empty GenerationBatch was seeded with flat broadcast logits_processors.** `BatchGenerator.__init__` passed `self.logits_processors` (flat `List[Callable]`) into `GenerationBatch.empty(...)`; once `extend()` appended the first per-uid entry from `PromptProcessingBatch` the structure became mixed (`[<rep_penalty>, [None]]`) and `_step` raised `TypeError: 'function' object is not iterable` at the first generation step. Fix: pass `logits_processors=None` to the empty batch (per-uid entries flow in via `extend()`); also corrected the `GenerationBatch.empty` type annotation from flat `List[Callable]` to per-uid `List[Optional[List[Callable]]]`. **Plus silent-drop fix:** `server.py` always passed `logits_processors=[args.logits_processors]` per-insert; when `args.logits_processors is None` (no JSON-schema), this was `[None]`, which suppressed the broadcast in `insert()` (line 2502 only broadcasts when the kwarg is None). Result: penalty processors silently never applied — OpenWebUI's `repetition_penalty` slider has been theater since the upstream merge. New behavior combines `_make_logits_processors(args)` (penalty) + `args.logits_processors` (structured) per request and passes the merged list per-uid. Now penalties reflect the CURRENT request's args (strictly better than the prior "first request locks the broadcast" design). Regression test in `test_generate.py::TestBatchGenerator::test_batch_generator_empty_generation_batch_does_not_seed_flat_processors`. | #24 |
| `3bca51c` | 2026-05-01 | Lift rewind helpers to module level: `_rotating_rewind_safe`, `_has_non_trimmable`, `_restore_deltanet_state` were nested closures inside `stream_generate` (capturing nothing). Promoted to module-level so they're directly importable and unit-testable. Behavior identical. Bundled with the initial unit-test backfill (#13, #13a, #21, #25). | #13, #13a |

Sections #1–#16 (the pre-port architecture work — StreamingTranslator, MultiCacheManager, prefill-step tuning, RoPE desync fix, ghost-prompt removal, etc.) predate the current `upstream` branch tip and are not individually itemized in this commit table; they're reflected in the cumulative diff `git diff upstream..HEAD`.

---

**Test Coverage Sweep (2026-05-01)**

Audit + buildout to close the high/medium-severity unit-test gaps for local mods. Net: **~160 new tests across 5 new files + 4 extended files; full suite 619 passing** (up from baseline 511); 5 pre-existing failures unchanged (audio I/O, two known logging/signature drifts in `test_generate.py`, two upstream-merged Nemotron model tests). Initial backfill committed in `3bca51c`; round-3/4 test extensions (`test_prompt_utils.py`, `test_server.py` ProcessToolCalls, `test_turboquant.py`) staged but uncommitted as of this entry.

**New test files:**
- `tests/test_snapshot.py` (20 tests) — `DeltaNetSnapshotRing` capture/FIFO/`find_nearest`/`drop_after`/`clear`, dataclass `frozen=True`, refcount-cheap snapshot semantics. Covers #13a.
- `tests/test_rewind_guard.py` (32 tests) — `_rotating_rewind_safe` (wrapped/unwrapped buffers, mx.array offsets, recursion through `.caches` and lists), `_has_non_trimmable`, `_restore_deltanet_state`, `PromptCacheState.update` divergence walker (the silent-correctness bug), `find_prefix_length`, default ring attachment. Covers #13 and #13a.
- `tests/test_session_cache.py` (28 tests) — `_session_caches` LRU eviction at cap, move-to-end on hit, zero-cap disables eviction, `clear_session_caches`, `_resolve_chat_id` precedence (header → body → metadata → None), header strip + custom name, non-dict metadata ignored, `_env_int` / `_env_choice` graceful fallbacks. Covers #11 + #25.
- `tests/test_sanitize_strict_json.py` (17 tests) — plain prose passthrough, raw `{`/`[` JSON intent, ` ```json ` markdown fence intent, math-block escape survival round-trip (`\frac`, `$$\sum$$`, `\[\int\]`, `\(x^2\)`), already-double-escaped backslashes preserved, `\"` not over-escaped. Covers #21.
- (no separate file for thinking-budget / detect-format / logprob — extended into `test_server.py`).

**Extended test files:**
- `tests/test_server.py` — added 4 test classes (43 new tests total):
  - `TestDetectThinkingFormat` (5) — gemma vs generic vs None; gemma precedence over generic.
  - `TestComputeThinkingBudget` (7) — None when no format; 80% × max_tokens auto formula; client_budget override; zero client budget.
  - `TestMakeLogprobContent` (5) — chosen-token scalar; top_k=0 skips alternatives; truncation; decode-failure tolerance; numpy float64 acceptance.
  - `TestBuildGenArgsPenaltyAndSeedPlumbing` (9) — `seed` plumbed through (None default); `repeat_penalty` Ollama alias; `repetition_penalty` wins when both set; `presence_penalty` / `frequency_penalty` propagated; unset penalties stay None (not 0).
  - `TestProcessToolCalls` extended from 1 → 9 tests — OpenAI shape (id/type/function); dict args → JSON; pre-stringified args passthrough; multi-call distinct indices; **rescue path** (invalid call skipped, neighbors survive); markup stripped from remaining_text; Ollama `tool_call_end=""` newline-anchored; whitespace-stripped name. Covers #4 / #15.
- `tests/test_prompt_utils.py` — added 4 test classes (16 new tests):
  - `TestCacheAlignmentKwargs` (3) — `get_cache_alignment_kwargs()` returns defensive copy; `preserve_thinking` entry guarded; values must not be `None`. Covers #27.
  - `TestMessageFormatterTextOnlyFallback` (4) — unknown `model_type` returns plain `{role, content}`; preserves role; `format_type is None` invariant; ignores image requests. Covers #20.
  - `TestMessageFormatterImageSchema` (4) — Gemma's `LIST_WITH_IMAGE` produces `{type: image}` (not `image_url`); ERNIE's image_url variant; `skip_image_token` and assistant role omit image. Covers #3.
  - `TestGetChatTemplateProcessorFallback` (5) — happy-path direct use; tokenizer fallback when processor lacks `chat_template`; plain renderer when both lack; single-user-message bare-content shortcut; `chat_template` kwarg override forwarding. Covers #5.
- `tests/test_turboquant.py` — added 2 test classes (12 new tests):
  - `TestAllocationHookLifecycle` (7) — register/unregister round-trip; idempotent re-registration; unregister of unknown is a no-op; ordered firing; trigger-with-no-hooks no-op. Autouse fixture saves/restores `_ALLOCATION_HOOKS` to keep tests hermetic. Covers #12.
  - `TestMaxKVSizePlumbing` (5) — default None; constructor stores; `from_cache` propagates; `from_cache` default None; explicit 0 documented. Covers #8.
- `tests/test_generate.py` — added 1 regression test (`TestBatchGenerator::test_batch_generator_empty_generation_batch_does_not_seed_flat_processors`) for the per-uid shape crash fix.

**Audit findings (memory.md drift discovered during the sweep):**
- **#11 (MultiCacheManager) is partially overstated.** The current implementation is just an `OrderedDict` LRU on `_session_caches` with count-based eviction (`_session_cache_max`); there is no RAM-aware free-memory threshold class, no `psutil` integration, and the "MultiCacheManager" name doesn't appear in the code. The original design described in #11 may have been planned but never shipped, or was simplified. Tests cover what's actually there (count-based LRU + chat_id resolution).
- **#2 (StreamingTranslator engine) is largely already covered** by existing `_split_thinking`, `suppress_tool_call_content`, `_count_thinking_tag_tokens`, and `_has_prefilled_opener` tests. The "translator engine" name corresponds to inline streaming-SSE state-machine code in `server.py:2700-2820`, which is integration-test territory, not unit. No new file added for it.

**Remaining gaps (lower leverage, deferred):**
- **Medium:** #7 layer-by-layer `mx.eval` at KV-quant threshold (needs Metal + KV mocking — meaningful integration setup); #13 SWA step-padding strip in `_trim_cache` (buried in inner closure, would need another helper-lift refactor).
- **Low:** #10 telemetry (logging side effects), #16 ghost-prompt removal regression guard, #19 batch-path stop-token merge, #22 streaming-loop budget counter, #26 per-thread MLX streams (thread-local state hard to assert cleanly).
