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

**20. Text-Only Model Support (utils.py & prompt_utils.py — commits `e3f2157`, `a75f1db`, `9457990`)**
* **The Problem:** Pure text models like `gemma3_text` aren't VLMs — they have no vision tower and aren't in `MODEL_CONFIG`. Loading them through mlx-vlm's normal path raised `ValueError "not supported"`, and `MessageFormatter` raised `ValueError` for unknown model types.
* **The Fix:** Added `_TextOnlyLanguageModel`, `_TextOnlyModelWrapper`, and `_SimpleNamespace` wrappers in `utils.py` that adapt mlx_lm text models to the VLM interface. `load()` catches the unsupported-model `ValueError` and falls back to `mlx_lm.utils.load` with tokenizer compatibility patches. `_TextOnlyLanguageModel` inspects the model's `__call__` signature and only forwards accepted kwargs, stripping VLM-specific extras like `attention_mask_4d`. `MessageFormatter` (prompt_utils.py) now falls back to plain text formatting for unknown model types instead of raising — there are no image/audio tokens to insert anyway.

**21. Strict JSON Output Sanitization — Non-Streaming Endpoints Only (utils.py & server.py — commits `e3f2157`, `cbe1ac8`)**
* **The Problem:** When a model is asked to emit raw JSON (especially with embedded LaTeX math), `json_repair`'s default behavior treats valid JSON escapes like `\f`, `\b`, `\n` as control characters and silently destroys the math content. Plain-prose responses must not be touched.
* **The Fix:** `sanitize_strict_json(text)` in `utils.py` first detects intent — only acts if the text starts with `{`, `[`, or `` ```json ``. For non-JSON output it returns the input untouched. For JSON-intended output, it pre-escapes math blocks (`$$…$$`, `$…$`, `\[…\]`, `\(…\)`) by double-escaping any backslash not already followed by a backslash or quote, then runs `json_repair`. Wired into the non-streaming `/v1/chat/completions` (server.py:2573) and `/v1/responses` (server.py:1944) endpoints. **The streaming path is intentionally NOT sanitized** — token-by-token deltas can't be repaired without a buffering rewrite, and streaming users shouldn't be paying that latency cost. This means streamed code blocks pass through verbatim (relevant context for the code-fence findings below).

**22. Server-Side Thinking Budget Enforcement (server.py — commit `ffad5df`)**
* **The Problem:** Reasoning models can enter self-reverification loops on multi-constraint prompts (see "Model Behavior Findings" below). With no hard cap, a single request can burn 80K+ thinking tokens before producing any visible output.
* **The Fix:** `_compute_thinking_budget()` (server.py:935) returns `0.80 × max_tokens` when a thinking format is detected (`<|channel>thought` for Gemma, `<think>` for generic), or honors an explicit `thinking_budget` field on the request. The streaming loop at server.py:2218 increments `thinking_tokens` while `in_thinking` is true; once it exceeds the budget, the server emits `THINKING_TRUNCATION_MSG` with `finish_reason="length"` and breaks the stream. `THINKING_BUDGET_RATIO` is the constant at server.py:58. Format detection lives in `_detect_thinking_format()` (server.py:910).

**23. Structured Logging & `--log-file` Flag (server.py, generate.py, utils.py — commits `119d0b9`, `170c30f`)**
* **The Problem:** Diagnostic output was a mix of `print()` calls and ad-hoc `logging.info` blocks. No way to redirect to a file, no consistent module loggers, no preview-truncation for long prompts (which spammed terminals at 6K+ tokens).
* **The Fix:** Replaced all `print()` with `logger.info/warning/error/debug` across server, generate, and utils. Module loggers derive from `MLX_VLM_LOG_NAME` (default `mlx_vlm`); `MLX_VLM_LOG_LEVEL` env var and `--log-name` CLI arg control verbosity. Added `DEBUG_PREVIEW_CHARS` constant and `_truncate()` helper for head+tail previews of long prompts/responses. `traceback` import dropped in favor of `exc_info=True` on `logger.error`. The `--log-file PATH` flag (default: `<stdout>`) lets ops redirect to file or journald cleanly. Supersedes the original ad-hoc telemetry pipeline mentioned in #10.

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

**Upstream base:** `0f903f9` *Fix Gemma 4 LoRA training: vision backward NaN + audio_tower freeze leak (#1052)*

**Local commits over upstream** (oldest → newest, as of 2026-04-27):

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

Sections #1–#16 (the pre-port architecture work — StreamingTranslator, MultiCacheManager, prefill-step tuning, RoPE desync fix, ghost-prompt removal, etc.) predate the current `upstream` branch tip and are not individually itemized in this commit table; they're reflected in the cumulative diff `git diff upstream..HEAD`.
