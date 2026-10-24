# Text-only embedding load for `mistral3` retrieval checkpoints (Nemotron 3.5 Embed 1B) — spec draft 4

Status: design-reviewed once (four-eyes light, codex and opus, 2026-10-22; ledger in Premisa `docs/reviews/nemotron35-embed-loader/`); Go given by the operator on 2026-10-22 (P589). Draft 3 corrected the fallback's return type after the implementer and the test-writer both found that `load_tokenizer` returns a non-callable wrapper. Draft 4 folds in the change review (codex and opus, 2026-10-22): detection from the raw config.json, `config=None` treated as omitted, a WARNING-level fallback log with the message, `model.lm_head.` dropped, and the residuals named below. Both drafts are the designer's, made after Go, and are shown to the operator at Accept.

## Goal

`nvidia/Nemotron-3.5-Embed-1B-BF16` is a `mistral3` checkpoint: a Pixtral vision tower, a multimodal projector and a `ministral3` text model (initialised from Nemotron 3 Embed 1B) that sentence-transformers mean-pools and L2-normalises into 2048-dimensional embeddings, with `query: ` and `passage: ` prefixes supplied by the caller. The fork's embedding loader maps `ministral3` to its text embedding model but has no path for `mistral3`, so the checkpoint cannot be loaded as an embedder at all. The problem is to load it as a text-only embedder: the loader builds the fork's `ministral3_embedding` model from the checkpoint's `text_config` and its `language_model.*` tensors, drops the vision tower and projector so they are never loaded into the model, and produces, for text inputs, the same embeddings the reference produces when it is given text only (the reference runs just the language model when there are no pixel inputs). Solving it means the task server's `/v1/embeddings` path and the Premisa spike can embed text with this checkpoint, at BF16 and from an MLX affine-quantized copy, with no other change.

## Scope

- `mlx_vlm/embedding_loader.py`: recognise a `mistral3` config whose `text_config.model_type` is `ministral3` and load it through `ministral3_embedding`; add `load_embedding_processor(model_dir, **kwargs)`.
- `mlx_vlm/models/ministral3_embedding/ministral3_embedding.py`: `sanitize` accepts the multimodal key layout and rejects alias conflicts.
- `mlx_vlm/server/app.py`: the embedding load uses `load_embedding_processor` instead of `load_processor` (one call site).
- `mlx_vlm/tests/test_mistral3_text_embedding.py`: new, fork-only.
- All three source files are shared with upstream; every changed definition carries a `# Fork:` marker (`dev/check_fork_markers.py` must stay clean for them).

## Settled constraints (settled by the operator in the Premisa forest, 2026-10-22)

- The candidate is screened text-only; image retrieval is not in scope.
- Fork work stays on the fork's terms: no Premisa product code changes; the change is a fork feature, not a patch of an upstream bug.
- Downloads are user-started; the checkpoint is already in the local Hugging Face cache.

## Verified facts (2026-10-22, replacing the earlier assumptions A1 and A2)

- F1. The reference `Mistral3Model`, given `input_ids` and `attention_mask` only, runs its `language_model` (`Ministral3Model`) and returns its last hidden state; sentence-transformers mean-pools and normalises it. Built offline in the torch environment with transformers 5.5.4 (`AutoModel`, CPU float32): six texts, relevant query-passage cosine 0.689, irrelevant 0.062; vectors saved for the parity criterion. sentence-transformers itself cannot load the checkpoint there (image extra missing), so the parity reference is the transformers forward plus mean pooling, which is what the card's transformers example does.
- F2. `text_config.is_causal` is false, as in Nemotron 3 Embed 1B; the fork's `ministral3_embedding` forward uses a padding-only (bidirectional) mask. The two configs differ only in bookkeeping keys and in `rope_theta`, which the fork reads from `rope_parameters`; `rope_parameters` also carries `apply_yarn_scaling: false`, which transformers reports as unrecognised, so the parity check is the arbiter of the rope path.
- F3. The checkpoint's tokenizer adds no BOS or EOS to a plain text (`query: ...` starts at the word token); `model_max_length` and the sentence-transformers `max_seq_length` are 32,768. The server's `_embed` caps at 8,192 tokens: embeddings are identical below that length, which every study chunk is, and this is a stated difference above it.
- F4. `load_processor` on the checkpoint fails offline in the MLX environment (`PixtralProcessor` requires PyTorch), so the server's embedding route needs a tokenizer fallback; the model path needs only the tokenizer.

## Assumptions (the designer's)

- A3. Pooling comes from `1_Pooling/config.json` (mean) through the existing `pooling_config` mechanism; the loader carries no top-level `pooling` key.
- A4. The prefixes are the caller's responsibility (sentence-transformers' `prompts`), as they were for the Nemotron 3 Embed 1B study arm; the server does not read `config_sentence_transformers.json`.

## Interfaces

### `load_embedding_model(model_path, lazy=False, **kwargs)`

Current behaviour: reads `config.json`, applies `EMBEDDING_MODEL_REMAPPING` to `model_type`, loads every safetensors tensor in the directory, builds the model and calls `sanitize`.

New behaviour, added before the existing path and only when the caller passed no `config` (a `config=None` argument counts as none; an explicit dict keeps its meaning and skips detection): the loader reads the raw `config.json` with `json.load` (no transformers normalisation, no caller kwargs; a missing file falls through to the existing path, which reports it), and if `config["model_type"] == "mistral3"`, `config.get("text_config", {}).get("model_type") == "ministral3"` and the text config's `is_causal` is not `true` (a causal text model is an instruct checkpoint, not an embedder; added at the second change review, with a negative test), derives a text config:

```
text = dict(config["text_config"])
text["model_type"] = "ministral3"            # remapped to ministral3_embedding by the existing table
if "quantization" in config:                 # the key the fork's quantized copies carry at top level
    text.setdefault("quantization", config["quantization"])
```

and calls `load_encoder_model(model_path, model_remapping=..., config=text, ...)` with that config. Everything else (weight reading, `sanitize`, quantization predicate, `pooling_config` set by the server) is unchanged. Any other `mistral3` checkpoint (no `ministral3` text config) falls through to the existing path and fails as today.

### `ministral3_embedding.Model.sanitize(weights)`

Current: drops `lm_head.*`; prefixes bare keys with `model.`. New: `model.lm_head.*` is dropped as well.

New: first drops every key starting with `vision_tower.`, `multi_modal_projector.` or `model.vision_tower.` / `model.multi_modal_projector.`; strips a leading `language_model.` (or `model.language_model.`); then the current logic. Result for the reference layout: `language_model.layers.3.self_attn.q_proj.weight` → `model.layers.3.self_attn.q_proj.weight`; `language_model.embed_tokens.weight` → `model.embed_tokens.weight`; `language_model.norm.weight` → `model.norm.weight`. A quantized copy written by the Premisa spike already carries `model.*` keys and passes through unchanged. If two source keys map to the same destination (for example `language_model.norm.weight` beside `model.norm.weight`), `sanitize` raises `ValueError` naming both; it never keeps whichever came last.

### `load_embedding_processor(model_dir, **kwargs)` (new, `embedding_loader.py`)

Returns `load_processor(model_dir, add_detokenizer=False, **kwargs)`; if that raises (an image processor whose dependencies are absent, as with the Pixtral processor offline), logs one WARNING line naming the exception class and its message and returns a callable tokenizer: the result of `load_tokenizer(model_dir)` when it is callable, otherwise the Hugging Face tokenizer it wraps (its `_tokenizer` attribute), since the server's `_embed` calls the tokenizer directly and `TokenizerWrapper` is not callable (draft 3 correction). The server's embedding load (`server/app.py`, the `load_as_embedding` branch) calls it in place of `load_processor`; the text embedding path uses only the tokenizer. Residual, recorded at the change review and not built against: the fallback applies to every embedder the server loads, so an image-capable embedder whose processor fails to build now loads text-only and reports "does not support image inputs" on an image request instead of failing at load; the WARNING line is the trace. Scoping the fallback to text-only checkpoints is a possible follow-up fork item; it would invalidate three frozen tests written to the unscoped rule. A second residual: if the Pixtral processor does build (PyTorch present), the text-only model ignores `pixel_values`, as every text embedder in the fork does; the Premisa inference environment has no PyTorch.

### Success path

`load_embedding_model(<Nemotron 3.5 dir>)` returns a `ministral3_embedding.Model` with 16 layers and no vision parameters; `model.pooling_config = read_pooling_config(dir)` gives mean pooling; `_embed(model, processor, ["query: x", "passage: y"])` returns two finite unit vectors of length 2048.

### Failure path

A `mistral3` checkpoint without a `ministral3` text config, or with vision-only weights, raises the loader's existing error (no model class or missing weights); nothing is silently loaded as text-only. A `language_model.*` key set that does not match the text config raises `load_weights`' strict mismatch, as today.

### Processor

See `load_embedding_processor`. The Premisa spike (test tooling in the Premisa repository, not product code) already carries the same fallback for its own `_load`.

## Acceptance criteria

1. Synthetic mapping test (the `language_model.model.*` layout of transformers-4 `Mistral3ForConditionalGeneration` saves is pinned as a fifth parametrisation): a temporary directory holding a `mistral3` config with a tiny `ministral3` `text_config` and a tiny `vision_config`, random text, `vision_tower.*` and `multi_modal_projector.*` tensors in one safetensors file, loads through `load_embedding_model`; the model has no parameter whose path contains `vision_tower` or `multi_modal_projector`; its embeddings for a fixed token batch equal, to 1e-5, those of a plain `ministral3` directory holding the same text weights under bare keys. Parametrised over the text key layouts `language_model.*`, `model.language_model.*`, `language_model.*` plus a `language_model.lm_head.weight`, and bare `model.*`. This proves the mapping, not the decoder (criterion 6 does that).
2. Quantized round trip: the synthetic text weights quantized in memory with `nn.quantize` 8-bit (group 64), embedded, saved in the spike's layout (`model.*` keys, the `mistral3` config plus `"quantization": {"group_size": 64, "bits": 8, "mode": "affine"}`), reloaded through `load_embedding_model`, embedded again: identical to 1e-5, and cosine at or above 0.99 against the BF16 embeddings of the same synthetic weights.
3. Negatives: a `mistral3` config whose `text_config.model_type` is not `ministral3` falls through and raises as before; a directory whose text keys appear under two aliases raises `ValueError` from `sanitize` naming both keys; a missing or shape-mismatched text tensor raises from strict loading.
4. Processor fallback: `load_embedding_processor` returns the processor when `load_processor` succeeds and the tokenizer when it raises (unit test with a patched `load_processor`); on the real checkpoint offline it returns a tokenizer whose ids and mask for `query: x` equal the reference tokenizer's.
5. `dev/check_fork_markers.py` reports no unmarked site in the three files; the fork suite passes in full in a checkout with `.git` (observed: 6,075 passed); in a plain copy without `.git` only the allowlist-pinning test of `test_audit_scoped_exclusions.py` fails, because it runs git.
6. Reference parity (designer-run, reported, not a unit test): on the six saved reference texts (short and a 2,235-token passage, both prefixes), the fork at BF16 produces the same token ids and masks and per-row cosine at or above 0.999 against the transformers reference of F1; the minimum is reported. Then the 1,875-chunk set embeds with zero nonfinite rows, with peak process memory reported.
   Measured 2026-10-22 on the candidate: token ids and masks identical on all six texts; per-row cosine 0.99997 to 0.99999 (minimum 0.99997); batched against single-text cosine minimum 0.99998; the model loads in 0.7 s with 1,140M parameters and no vision tensors; peak process RSS 2.78 GB at BF16 for the six texts.
   Corpus runs 2026-10-22 on the candidate (batch 16, prefixes `query: ` and `passage: `): 1,875 chunks and 230 questions on the first corpus and 1,868 and 230 on the second, zero nonfinite rows, 38 to 41 s per corpus at BF16; the batched run pads six texts of 6 to 2,235 tokens through the fallback tokenizer (right padding, pad token present), so the pad-token concern raised at the change review does not arise. Real-checkpoint 8-bit round trip in the spike's layout (every module, group 64, 1.1 GB, 8.50 bits per weight stored): reloaded through the loader, chunk cosine 0.99988 minimum against BF16, top-8 neighbour overlap 0.991. Task-server peak with the 8-bit copy beside the 8-bit reranker and the task model: 3,871,297,904 bytes.

## Non-goals

Image or mixed inputs for this checkpoint; applying `config_sentence_transformers.json` prompts in the server; the NVFP4 variant; any change to `mistral3` generation; quantized layouts other than the spike's (an mlx-vlm `convert` of the full multimodal model, with per-path overrides keyed under `language_model.`, is not supported by this change).

## Open questions

- Q1. Should the server's embedding route apply the checkpoint's `prompts` (`query: `, `passage: `) automatically? Recommendation: no (A4); Open WebUI carries prefix settings and the study already used them.
