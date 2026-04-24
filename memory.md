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
