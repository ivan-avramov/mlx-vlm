# AGENTS.md

Guidance for coding agents working in this repository.

MLX-VLM is a Python package for inference and fine-tuning of Vision Language
Models and Omni (audio/video) models on Apple Silicon using MLX. 50+ model
architectures via a plugin-based model system.

## Fork & branches

A fork of Blaizzy/mlx-vlm. `main` is the only branch and carries local work on
top of upstream. Two remotes: `origin` (`ivan-avramov/mlx-vlm`) and `upstream`
(`Blaizzy/mlx-vlm`, push URL is `no_push`).

**Merge, never rebase.** `main` is published and carries ~800 local commits plus
~90 merge commits; a rebase would replay all of them and flatten the merge
topology to reach the same tree a merge reaches in a handful of hunks.

```bash
git fetch upstream
git merge upstream/main
python dev/check_upstream_parity.py     # no upstream file silently dropped
python dev/check_upstream_symbols.py    # no upstream def/class silently dropped
python dev/find_dropped_hunks.py        # no upstream HUNK silently dropped
python dev/check_upstream_deletions.py  # no upstream DELETION silently reverted
cd mlx_vlm/ && pytest ./tests --ignore=tests/test_smoke.py --ignore=tests/test_utils.py
```

**Run all four checks after every merge — this is not optional.** When a merge
resolution drops content from a commit that upstream already merged, that commit
is still an ancestor of `main`, so git records the content as *deliberately
deleted* and no later merge re-offers it. There is no conflict and no warning.

### Divergence has two directions — always check both

The first three checks all ask the same question: *what does upstream have that we
lack?* For a long time nothing asked the mirror: **what did upstream delete that we
kept?** Same mechanism, opposite sign.

    upstream deletes a file or symbol -> its deleting commit becomes an ancestor
    of `main` through a merge -> our resolution keeps our side -> the content
    lives on as stale upstream code that reads exactly like fork work

`dev/check_upstream_deletions.py` is that check. It reports two things, and like
`find_dropped_hunks.py` it is a **lead generator, not an oracle** — confirm every
hit with `git log -S` before acting:

- **Files** we carry that `upstream/main` lacks, where a commit that deleted the
  file is an ancestor of `upstream/main`. The conclusive test is
  `git diff <deleting-commit>^:<path> <path>`: **empty output means our copy is
  upstream's last pre-deletion copy**, so we never touched it — STALE, not FORK.
- **Symbols** we define in a shared file that upstream defined there once and no
  longer does.

Reviewed hits go in `.deletion-exclusions` with a real reason. It reads the git
**index**, like the other two audits, so it can gate a commit. Takes ~70s.

**Most symbol hits are the second half of a dropped commit.** When upstream moves
or renames a symbol A -> B and the resolution drops the B half, we keep A.
`check_upstream_symbols.py` then reports B as *missing* while this script reports A
as *kept*, and nothing connects the two halves of one event. Three examples, all
real:

- `server/app.py`'s `_as_plain_dict`, `_request_field_or_default` and
  `_model_config_field_or_default` are defined **twice** in this tree — here, and
  in `server/request_normalization.py`, which is where upstream moved them
  (#1644). Upstream has only the latter. That is *why* that module is orphaned.
- `utils.py::apply_forced_token` was renamed to `pop_forced_token_id` with a
  changed contract (`(next_y) -> mx.array` became `() -> Optional[int]`). We had
  the old name; seven `.symbol-exclusions` entries excused the "missing" new one.
- `mlx_vlm/video_generate.py` was 645 lines of code upstream removed in #1454.
  7 of that commit's 8 files had applied, so every doc reference was stripped
  while the module and its `__main__.py` registration survived — a working,
  undocumented command no check could see.

**Corollary for `AGENTS.md` itself: do not describe a symbol as "fork-only"
without running `git log -S` on it first.** This section's own guidance got this
wrong — see the KV-cache note below.

When resolving conflicts, prefer a **union** over picking a side. Both sides add
to shared lists (`__all__`, lazy-import tuples, skip-lists, registries); taking
one side drops the other's contribution and usually only fails at runtime.

`upstream/main...HEAD` (three dots, from the merge base) is the cumulative local
diff. `memory.md` has the change log; `docs/upstream-gaps.md` is the current gap
list and the record of how this fork diverges and why.

## The one rule that matters most

**Never conclude anything about a divergence from a `git diff --numstat` line
count.** That measure cannot distinguish three situations that need opposite
responses: content we dropped (restore), fork work (keep), and code upstream
itself later replaced (delete). Six wrong calls have been made this way; they are
listed at the end of `docs/upstream-gaps.md`.

Establish provenance first, every time:

```bash
git log --oneline -S'<distinctive symbol or literal>' --all -- <path>
git merge-base --is-ancestor <commit> HEAD           # merged-then-dropped?
git merge-base --is-ancestor <commit> upstream/main  # still current upstream?
git show --stat <commit>                             # its WHOLE file set
```

Then classify: **DROPPED** (restore), **FORK** (keep), **STALE** (delete ours).

Two corollaries learned the hard way:

- **A dropped commit usually spans several files, and they are internally
  consistent while stale.** Restoring one file of a multi-file dropped commit is
  worse than restoring none.
- **A "fork-only" symbol is often stale *upstream* code** that upstream later
  removed. `git log -S` is what tells them apart.

`dev/find_dropped_hunks.py` automates the search: for each diverged file it finds
which upstream lines are absent, then attributes them to the commits that
introduced them (by line content, not blame — blame follows our side of history,
which is the side that lost the content). It is a **lead generator, not an
oracle**: heavy fork rewrites and upstream-superseded code both surface as false
positives. Every hit still needs `git log -S` and a read.

## Tests

```bash
cd mlx_vlm/ && pytest -s ./tests --ignore=tests/test_smoke.py --ignore=tests/test_utils.py
```

The suite is **green: 2380 passed, 5 skipped, 0 failed.** Keep it that way.

**Compare failing test IDs, not counts.** A change that fixes one test and breaks
another shows the same total:

```bash
pytest -q ./tests --ignore=tests/test_smoke.py --ignore=tests/test_utils.py \
  | grep '^FAILED' | sed 's/ - .*//' | sort > /tmp/after.txt
comm -13 /tmp/before.txt /tmp/after.txt   # regressions
comm -23 /tmp/before.txt /tmp/after.txt   # fixed
```

**A green suite only proves that *collected* tests pass.** Check
`pytest --collect-only` counts too. Two live examples:

- `tests/test_utils.py` is **excluded from the suite** (`--ignore`, 5 pre-existing
  failures). A regression test placed there never runs. Put fork tests in a
  collected file; `tests/test_model_registry.py` is the fork-only example.
- `TestLagunaProcessor` is defined **twice** in `tests/test_processors.py`, so the
  earlier definition is shadowed and its test is never collected — which is why a
  missing `utils.should_add_special_tokens` fails nothing. Our copy is
  byte-identical to upstream, so this is an upstream bug too.

**Green does not mean converged.** Of the real bugs found in this fork, most had
no failing test — because the merge that dropped a feature hunk usually dropped
its tests in the same resolution. 56% of the content still missing from upstream
is test code.

Restored upstream tests are kept byte-identical to upstream so future merges
apply cleanly; put skip decisions in `mlx_vlm/tests/conftest.py`, not in the test
files. `UNPORTED_UPSTREAM_TESTS` there is currently **empty** — keep the
mechanism, since removing an entry is the definition of done for porting a
feature, and it announces what it skips at collection time.

Run a single test file or test:

```bash
cd mlx_vlm/ && pytest -s ./tests/test_generate.py
cd mlx_vlm/ && pytest -s ./tests/test_generate.py::TestClassName::test_method_name
```

## Install

```bash
pip install -e .
# local venv with test deps (used by the commands above):
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e . pytest pytest-asyncio
```

## Formatting & linting

```bash
pre-commit run --all-files
```

Pinned in `.pre-commit-config.yaml`: **black 26.3.1**, **isort 5.13.2**
(profile=black), **autoflake 2.2.1**. The pins are identical to upstream's, and
upstream's tree is clean under them — so an unformatted file here has *drifted
from* upstream and formatting it **converges**. That premise was wrong in an
earlier version of these notes; see `docs/upstream-gaps.md`.

**autoflake is the hook with teeth** — black and isort only move whitespace,
autoflake deletes imports. Check each removal by hand. `mlx_vlm/models/base.py`
and `mlx_vlm/models/cache.py` are excluded from it because they are re-export
surfaces; keep that exclusion.

## CLI entry points

```bash
python -m mlx_vlm.generate --model <model> --image <path> --prompt "..."
python -m mlx_vlm.chat --model <model>
python -m mlx_vlm.server --model <model> --port 8080
python -m mlx_vlm.convert --hf-path <model> -q 4
python -m mlx_vlm.chat_ui --model <model>  # requires gradio
```

## Architecture

### Model plugin system

Models are discovered via `config.json`'s `model_type`:

1. `utils.py:load()` downloads from HF Hub if needed and reads `config.json`
2. `utils.py:get_model_and_args()` does
   `importlib.import_module(f"mlx_vlm.models.{model_type}")`
3. `MODEL_REMAPPING` in `utils.py` maps alternate names
   (`"llava_qwen2"` -> `"fastvlm"`)

Each `mlx_vlm/models/{model_type}/` contains `__init__.py` (imports, optional
processor patch), `config.py` (`ModelConfig`/`TextConfig`/`VisionConfig`
dataclasses off `BaseModelConfig`), `{model_type}.py` (the `Model` class with
`get_input_embeddings()`), `language.py`, `vision.py`, and
`processing_{model}.py`.

**Registries are a known blind spot.** `MODEL_REMAPPING`, prompt-format maps,
drafter registries, tool parsers and `__init__.py` re-exports are all invisible to
both audits — parity only sees missing files, the symbol check only sees missing
`def`/`class` names. A dropped dict entry or re-export passes both. Four
`MODEL_REMAPPING` entries and a `Gemma4VideoProcessor` re-export were lost exactly
this way. `tests/test_model_registry.py` exercises the entries; extend it when you
add a registry.

### Generation (`mlx_vlm/generate/`, a package)

`dispatch.py:stream_generate()` is the main entry point, yielding
`GenerationResult` per token. `ar.py` holds autoregressive generation and the
continuous-batching `BatchGenerator`. Also `common.py` (cache trim/rewind
helpers, snapshot ring), `diffusion.py`, `image.py`, `edit_image.py`, `video.py`,
`video_generation.py`, `types.py`, `cli.py`.

`utils.py:prepare_inputs()` turns images/audio/video/text into model inputs.
`vision_cache.py` is an LRU that skips redundant vision-encoder calls.
`PromptCacheState` carries KV reuse across multi-turn conversations.

### Multimodal input flow

1. Media loaded and preprocessed (`utils.py:prepare_inputs()`)
2. Vision/audio encoders produce embeddings
3. `Model.get_input_embeddings()` merges them with text token embeddings
4. Language model generates autoregressively
5. `StreamingDetokenizer` (`tokenizer_utils.py`) decodes in real time

### Server (`mlx_vlm/server/`, a package)

FastAPI, OpenAI-compatible. `app.py` builds the app and owns model caching
(`get_cached_model`); `runtime.py:ModelCacheRegistry` handles model lifecycle.
Protocol surfaces are `openai.py`, `anthropic.py`, `audio.py`, `embeddings.py`,
with `schemas.py`, `responses_state.py`, `session_manager.py`, and `generation.py`
(the `ResponseGenerator` + GPU worker thread).

### Public API (`mlx_vlm/__init__.py`)

`load`, `generate`, `stream_generate`, `batch_generate`, `apply_chat_template`,
`prepare_inputs`, `process_image`, `convert`, `VisionFeatureCache`,
`GenerationResult`, `PromptCacheState`.

### Prompt formatting (`prompt_utils.py`)

Maps `model_type` to a `MessageFormat` controlling how image/audio/video tokens
are inserted into chat templates. Also holds the fork's `THINKING_FORMATS`
registry and `CACHE_ALIGNMENT_KWARGS`.

### KV caches (`models/cache.py`)

**This file has a boundary comment.** Everything above it is vendored from
upstream and should stay byte-identical apart from two hunks marked `# Fork:`
(`prealloc_tokens`). Add fork work **below** the boundary. Fork-only classes:
`PreallocKVCache`, `PreallocQuantizedKVCache`. `turboquant.py` holds compressed
KV caches with custom Metal kernels.

**[correction]** This list used to include `SlidingWindowCache` and
`StaticKVCache` as fork-only. They were not. Both came from upstream's #391
(Gemma3n) and upstream **removed** them in `a492e47d` "Consolidate caches + RoPE
handling (#1494)", which is an ancestor of both `HEAD` and `upstream/main`. They
had **zero references anywhere** outside `cache.py` — 148 lines of dead stale
upstream code, deleted now. Being *below* the boundary comment is not evidence of
fork authorship; a resolution can leave stale code anywhere. This is exactly the
trap the "divergence has two directions" section warns about, and this file fell
into it.

### Fine-tuning (`trainer/`)

LoRA/QLoRA via `SFTTrainer` and `ORPOTrainer`; entry point `lora.py`. Adapter
layers are native: `trainer/lora_layers.py`, `trainer/dora_layers.py`,
`trainer/adapter_utils.py`.

## mlx-lm: vendored, not depended on

Upstream vendored its infrastructure and stopped importing `mlx_lm` in library
code; this fork now matches. **Exactly two library files reference it**, the same
two as upstream:

- `models/text_only.py` — a lazy, optional `from mlx_lm.utils import _get_classes`
  inside a function, wrapped in `try/except ImportError` with a real message. This
  is the intended escape hatch for text-only architectures with no native
  implementation. Keep it.
- `models/qwen3_5/gated_delta.py` — a provenance *comment*, not an import.

So use the native modules: `models/cache.py`, `models/base.py`,
`models/activations.py`, `models/rope_utils.py`, `models/switch_layers.py`,
`models/gated_delta.py`, `models/mla.py`, `models/pipeline.py`, `sample_utils.py`,
`quant_utils.py`. **Do not add new `mlx_lm` imports to library code.** If
something seems missing, it is almost certainly in one of those.

`mlx-lm` stays in `requirements.txt` — it is still needed for the text-only
fallback and by two test files.

## Deliberate divergences — do not "fix" these

- **`gemma4_assistant/masks.py` must keep its absolute import**
  (`from mlx_vlm.models.cache import dynamic_roll`). It looks like gratuitous
  divergence from upstream's relative form, but that module is imported standalone
  by its test, where a relative import has no parent package. Taking upstream's
  broke two tests.
- **`_rotating_post_gen_trim_safe` is intentionally unwired.** The mid-prefill
  snapshot supersedes it; wiring it in would force full re-prefill on wrapped SWA
  rings. See `memory.md:274` and `docs/upstream-gaps.md`.
- **`models/base.py`'s `cache.quantized_attention` call** — chunked over Q and K
  with online softmax and `mx.eval` between K-tiles, so peak memory is ~O(chunk)
  rather than O(context). Upstream dequantizes the whole state. This is the only
  reason `base.py` is not byte-identical to upstream.
- **`qwen3_5{,_moe}` MoE `sanitize()`'s expert-fusion branch** handles an unfused
  expert layout. Note the *norm-shift gate* in those files is **no longer a
  divergence** — it is byte-identical to upstream, which landed the same fix in
  #1556. Earlier guidance said not to touch these files because that gate was
  protected fork work; that was wrong, and it deterred inspection of the exact
  file where a live `+2.0` double-shift bug was hiding.

## Adding a new model

1. Create `mlx_vlm/models/{model_type}/` matching HF `config.json`'s `model_type`
2. Implement `config.py`, the model class, `language.py`, `vision.py`, processor
3. Add tests to `mlx_vlm/tests/test_models.py`
4. Add a `MODEL_REMAPPING` entry if the HF name differs, and make sure
   `tests/test_model_registry.py` still passes
5. `pre-commit run --all-files`

## CI

- `tests.yml` — runs on **pull_request only**, macOS-14 / Python 3.10. Runs
  `pre-commit run --all` and fails if `git diff` is non-empty, then pytest
  (excluding `test_smoke.py` and `test_utils.py`). Because it is PR-only, pushes
  straight to `main` are never style-checked — which is how style drift went
  unnoticed.
- `upstream-parity.yml` — runs on pushes to `main` as well as PRs, since this fork
  is usually committed to directly. Runs both audit scripts.

## Key dependencies

- `mlx` >= 0.31.2 — core framework
- `mlx-lm` >= 0.31.3 — text-only fallback only; not imported by library code
- `mlx-audio` >= 0.4.3
- `transformers` >= 5.5.0 — configs, tokenizers, processors
- `huggingface-hub` — model downloads
- `Pillow`, `opencv-python`, `miniaudio` — image/video/audio
- `fastapi`, `uvicorn` — server

## Working agreements

- **Never push without being asked.** Committing is fine when the work calls for
  it; stop at the commit and say the branch is ready.
- Full suite before every commit; report fixed/regressed test **IDs**.
- Prove each fix: repro before, repro after. Add a regression test in a file the
  suite actually collects.
- Prune `.symbol-exclusions` after each restore — entries go stale silently. Its
  "baseline: pre-existing divergence, unreviewed" reason means *nobody looked*,
  not *this is fine*.
- Format touched files with the pinned hooks before committing.
- `memory.md`'s change-log table takes an entry per merge; its refresh command is
  in that section (not `upstream/main..HEAD`, which returns ~837 commits). A row
  cannot contain its own hash — expect a one-line follow-up commit after an
  `--amend`.
