# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLX-VLM is a Python package for inference and fine-tuning of Vision Language Models (VLMs) and Omni Models (audio/video) on Apple Silicon using the MLX framework. It supports 50+ model architectures with a plugin-based model system.

## Fork & Branches

This repo is a fork of the upstream project (Blaizzy/mlx-vlm). `main` is the
only branch on the fork and carries local custom changes on top of upstream.

Two remotes: `origin` (the fork, `ivan-avramov/mlx-vlm`) and `upstream`
(`Blaizzy/mlx-vlm`, fetch-only — its push URL is `no_push`).

Merge, never rebase. `main` is published, and it carries ~818 local commits and
91 merge commits — a rebase would replay all of them and silently flatten the
merge topology, to reach the same tree a merge reaches in a handful of hunks.

```bash
git fetch upstream
git merge upstream/main
python dev/check_upstream_parity.py     # no upstream file silently dropped
python dev/check_upstream_symbols.py    # no upstream def/class silently dropped
cd mlx_vlm/ && pytest ./tests --ignore=tests/test_smoke.py --ignore=tests/test_utils.py
git push origin main
```

**Run both checks after every merge — this is not optional.** A merge resolution
that drops an upstream file or hunk is permanent and self-healing-proof: the
upstream commit still becomes an ancestor of `main`, so git treats the content
as deliberately deleted and no later merge re-offers it. An earlier merge lost 16
upstream test files exactly this way, invisibly. `docs/upstream-gaps.md` explains
the mechanism and is the current gap list; `docs/post-merge-gaps-2026-08-06.md`
is superseded.

When resolving conflicts, prefer a **union** over picking a side. The fork and
upstream both add to shared lists (`__all__`, lazy import tuples, skip-lists);
taking one side drops the other's contribution and often only fails at runtime,
not at import.

Use `upstream/main` to refer to upstream, and `upstream/main...HEAD` (three
dots, i.e. from the merge base) for the cumulative diff of local changes.
See `memory.md` for the change log over upstream.

## Common Commands

### Install (editable)
```bash
pip install -e .
```

A local venv with test deps (used by the commands below):
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e . pytest pytest-asyncio
```

### Run tests
```bash
cd mlx_vlm/ && pytest -s ./tests --ignore=tests/test_smoke.py --ignore=tests/test_utils.py
```

The suite is **not green**: 16 failed / 2175 passed. Compare against that number
rather than expecting zero.

All 16 remaining failures are in `test_diffusion_gemma.py` (11) and
`test_diffusion_models.py` (5) — no other test file has a failing test. They need
a design decision on reconciling `generate/diffusion.py`, which has diverged from
upstream in both directions; see `docs/upstream-gaps.md`.

**Compare failing test IDs, not counts.** A change that fixes one test and breaks
another shows the same total. Diff the sorted `FAILED ...` lines:

```bash
# before and after your change
pytest -q ./tests --ignore=tests/test_smoke.py --ignore=tests/test_utils.py \
  | grep '^FAILED' | sed 's/ - .*//' | sort > /tmp/after.txt
comm -13 /tmp/before.txt /tmp/after.txt   # regressions
comm -23 /tmp/before.txt /tmp/after.txt   # fixed
```

`mlx_vlm/tests/conftest.py` skips 2 restored upstream test files, both a
deliberate permanent divergence (quantized-SDPA masks; this fork dequantizes, so
the hazard they cover cannot occur). Restored upstream tests are kept
byte-identical to upstream so future merges apply cleanly to them; put skip
decisions in `conftest.py`, not in the test files.

### Run a single test file
```bash
cd mlx_vlm/ && pytest -s ./tests/test_generate.py
```

### Run a single test
```bash
cd mlx_vlm/ && pytest -s ./tests/test_generate.py::TestClassName::test_method_name
```

### Formatting & linting (pre-commit)
```bash
pre-commit run --all-files
```
Uses **black** (formatting), **isort** (import sorting, profile=black), and **autoflake** (unused imports). Files `mlx_vlm/models/base.py` and `mlx_vlm/models/cache.py` are excluded from autoflake.

### CLI entry points
```bash
python -m mlx_vlm.generate --model <model> --image <path> --prompt "..."
python -m mlx_vlm.chat --model <model>
python -m mlx_vlm.server --model <model> --port 8080
python -m mlx_vlm.convert --hf-path <model> -q 4
python -m mlx_vlm.chat_ui --model <model>  # requires gradio
```

## Architecture

### Model Plugin System

Models are discovered dynamically via `config.json`'s `model_type` field. The loading path:

1. `utils.py:load()` downloads from HF Hub (if needed) and reads `config.json`
2. `utils.py:get_model_and_args()` does `importlib.import_module(f"mlx_vlm.models.{model_type}")` 
3. `MODEL_REMAPPING` dict in `utils.py` maps alternate names (e.g., `"llava_qwen2"` -> `"fastvlm"`)

Each model directory (`mlx_vlm/models/{model_type}/`) contains:
- `__init__.py` — imports and optionally patches the processor
- `config.py` — `ModelConfig`, `TextConfig`, `VisionConfig` dataclasses (inherit from `BaseModelConfig`)
- `{model_type}.py` — Main `Model` class with `get_input_embeddings()` method
- `language.py` — `LanguageModel` (the LLM backbone)
- `vision.py` — `VisionModel` (the vision encoder)
- `processing_{model}.py` — Processor patches for transformers compatibility

### Generation Pipeline

`generate.py` is the core engine (~1760 lines):
- `stream_generate()` — main entry point, yields `GenerationResult` per token
- `generate_step()` — single forward pass + sampling with logits processors
- `prepare_inputs()` (in `utils.py`) — processes images/audio/text into model inputs
- Vision feature caching (`vision_cache.py`) — LRU cache skips redundant vision encoder calls
- `PromptCacheState` — KV cache reuse across multi-turn conversations

### Multimodal Input Flow

1. Images/audio loaded and preprocessed (`utils.py:prepare_inputs()`)
2. Vision encoder produces embeddings
3. `Model.get_input_embeddings()` merges vision/audio embeddings with text token embeddings
4. Language model generates tokens autoregressively
5. `StreamingDetokenizer` (`tokenizer_utils.py`) decodes tokens in real time

### Public API (`mlx_vlm/__init__.py`)

Core exports: `load`, `generate`, `stream_generate`, `batch_generate`, `apply_chat_template`, `prepare_inputs`, `process_image`, `convert`, `VisionFeatureCache`, `GenerationResult`, `PromptCacheState`

### Server (`server.py`)

FastAPI server with OpenAI-compatible endpoints (`/v1/chat/completions`, `/chat/completions`). Uses `MultiCacheManager` for model lifecycle and `StreamingTranslator` for async streaming.

### Fine-Tuning (`trainer/`)

LoRA/QLoRA fine-tuning via `SFTTrainer` and `ORPOTrainer`. Entry point is `lora.py`.

### Prompt Formatting (`prompt_utils.py`)

Maps `model_type` to a `MessageFormat` enum that controls how image/audio tokens are inserted into chat templates. This is where multi-image and multi-modal formatting logic lives.

### KV Cache Variants (`models/cache.py`)

Standard `KVCache`, `RotatingKVCache` (sliding window), `StaticKVCache`. Also `TurboQuant` (`turboquant.py`) for compressed KV caches with custom Metal kernels.

## Adding a New Model

1. Create `mlx_vlm/models/{model_type}/` matching the `model_type` from HF `config.json`
2. Implement `config.py`, model class, `language.py`, `vision.py`, and processor patch
3. Add tests to `mlx_vlm/tests/test_models.py`
4. Run `pre-commit run --all-files` before submitting

## CI

PR tests run on macOS-14, Python 3.10. CI checks pre-commit formatting and runs pytest (excluding `test_smoke.py` and `test_utils.py`).

## Key Dependencies

- `mlx` >= 0.30.0, `mlx-lm` >= 0.31.0 — core ML framework
- `transformers` >= 5.1.0 — model configs, tokenizers, processors
- `huggingface-hub` — model downloads
- `Pillow`, `opencv-python`, `miniaudio` — image/video/audio processing
- `fastapi`, `uvicorn` — server
