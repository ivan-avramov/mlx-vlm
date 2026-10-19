# Sequence-classification reranking for bidirectional Llama (llama-nemotron-rerank-1b-v2)

Status: specification, draft 4 (2026-10-17), after three review-panel rounds (two members each; the
round-three findings are applied here and not re-reviewed). Not approved for build. Repository: this fork of mlx-vlm. Requested by the operator for Premisa's retrieval
study (revision 6 reranker arm) and the stack's reranker socket.

## Goal

`/v1/rerank` serves `nvidia/llama-nemotron-rerank-1b-v2` (model_type `llama_bidirec`, architecture
`LlamaBidirectionalForSequenceClassification`, one label, `pooling: "avg"`, `temperature: 1.0`,
`pad_token_id: 128001`, bf16 weights, `score.weight` of shape `[1, 2048]`) as a sequence-classification
reranker with measured parity against the reference implementation, with no change to existing reranker
kinds, to the embedding behaviour of `models/llama_bidirec`, or to the embedding checkpoint that shares
its `model_type`.

## What exists (verified in the tree and the checkpoint on 2026-10-17; the fixture generator re-asserts the checkpoint facts mechanically)

- `mlx_vlm/models/llama_bidirec/`: `Model` runs the Llama tower with a full (non-causal) 4-D mask built
  from the 2-D attention mask, mean-pools and normalises. `__init__.py` exports only `Model` and
  `ModelConfig`. `sanitize` drops `lm_head`, `rotary_emb.inv_freq`, `position_ids`, strips a
  `language_model.` prefix and prefixes every key that does not start with `model.`.
- `mlx_vlm/reranker.py`: `_RERANKER_KINDS` keyed on `model_type`; `reranker_loader.load_reranker`
  loads `SEQUENCE_CLASSIFIER` kinds through `load_encoder_model(model_class_name=
  "SequenceClassificationModel")`, resolving the class on the model package; requires an architecture
  ending in `ForSequenceClassification` and exactly one label. The server calls `reranker_kind(model.
  config)` again on the typed config after loading (`server/app.py`), so fields it needs must survive
  `BaseModelConfig.from_dict`, which drops undeclared fields.
- `mlx_vlm/server/reranking.py`: `normalize_item` strips query and document text;
  `_score_sequence_batch` tokenises `(query, document)` pairs with the tokenizer's pair encoding,
  truncates at `_sequence_max_length(config, tokenizer)` (the minimum of the server cap, the config's
  positions and the tokenizer's `model_max_length`, which is 4096 here), expects one logit per pair and
  applies a sigmoid before the route sorts.
- The server's model cache key is `(model_path, adapter, effective_model_kind, fingerprint)` with a
  separate cache group per kind.
- The checkpoint: tokenizer `model_max_length` 4096 and `truncation_side` as configured in its
  `tokenizer_config.json` (recorded by the fixture generator); EOS and pad are both `<|end_of_text|>`
  (128001); the post-processor adds one BOS (128000) and no EOS to a single sequence; no chat template;
  the reference formats a pair as one string `f"question:{q} \n \n passage:{p}"`, left padding,
  example `max_length` 512, 8192 documented as the supported total length; reference pooling masks
  padding to zero then averages; `pooled_logits = score(pooled) / temperature`, returned raw.

## Design

1. `models/llama_bidirec/llama_bidirec.py`: `SequenceClassificationModel(Model)` with
   `self.score = nn.Linear(hidden_size, num_labels, bias=False)`. `__call__(input_ids,
   attention_mask=None)` runs the tower with a **key-only** mask broadcast over queries using a finite
   large negative value (never `-inf`, so padded query rows cannot produce NaN), pools by masked mean
   computed in float32 with `mx.where(mask, h, 0)` (never multiplication by the mask), divides by
   `config.temperature`, returns `SequenceClassifierOutput(logits)` of shape `(batch, 1)` in float32.
   `pooling` other than `avg` raises `ValueError` naming the value. `sanitize` applies the existing
   `language_model.` stripping first, keeps keys starting with `model.` or `score.` as they are, drops
   `lm_head`, `rotary_emb.inv_freq`, `position_ids`, and prefixes everything else with `model.`; the
   embedding `Model.sanitize` output is pinned by a test, and a deterministic baseline test pins the
   embedding `Model`'s hidden states and normalised embeddings on fixed inputs, padded and unpadded.
   Position ids stay an arange (as the reference); batch invariance is asserted at 1e-4 for that reason.
2. `models/llama_bidirec/__init__.py` exports `SequenceClassificationModel`.
3. `models/llama_bidirec/config.py`: `ModelConfig` gains `architectures: Optional[list[str]] = None`
   (so the typed config still dispatches), `pooling: str = "avg"`, `temperature: float = 1.0`,
   `num_labels: int = 1` (falls back to `len(id2label)`), `pad_token_id: Optional[int] = None`,
   `reranker_pair_template: Optional[str] = None` (validated at load time by rendering with sentinel
   values; only `{query}` and `{document}` named fields), `reranker_max_length: Optional[int] = None`
   (validated as a positive integer).
4. `reranker.py`: `reranker_kind` returns `SEQUENCE_CLASSIFIER` for `llama_bidirec` only when an
   architecture ends with `ForSequenceClassification`; otherwise the config is not a reranker. Every
   other site that dispatches on `model_type` is enumerated in the implementation report (embedding
   kind detection, preload routing, model listing); embedding kind detection rejects configs whose
   architecture ends with `ForSequenceClassification`, so the reranker checkpoint cannot be loaded as
   an embedder and the embedding checkpoint keeps its behaviour. `load_encoder_model` is reused; the
   report states whether it needed a change (expected: none) and lists it among touched files if so.
5. `server/reranking.py`:
   - `_pair_inputs(config, tokenizer, query, documents, max_length)` returns ids and masks. Encoder
     classifiers: the existing pair encoding, unchanged. The bidirectional Llama classifier: one string
     per pair from `config.reranker_pair_template or "question:{query} \n \n passage:{document}"`
     rendered with `str.format(query=..., document=...)`, tokenised **without padding** with
     `add_special_tokens` at the tokenizer default (one BOS, no appended EOS), no chat template, the
     tokenizer's configured `truncation_side` honoured on the complete formatted string with
     `max_length` passed explicitly; then left-padded manually with the pad id (`config.pad_token_id`,
     else the tokenizer's, else 128001) and masks built from sequence lengths. The cached tokenizer is
     never mutated. Raw query and document text (no stripping) for this kind; whitespace-only inputs
     still rejected.
   - Scoring length for this kind: `min(server cap, request max_length if given, config.
     reranker_max_length or 512, 8192)`; exceeding the tokenizer's `model_max_length` of 4096 is
     intended and documented. A request field `max_length` (clamped) lets a study select 512 or 8192
     without editing the pinned checkpoint config. The length used is returned in the response metadata.
   - Output: the sort key is the raw logit for **all** sequence-classifier kinds (fixing float32
     saturation for the existing kinds too); `relevance_score` stays the sigmoid for every kind so the
     API contract does not change per model; the raw logit is returned in an additional field
     (`raw_score`) for the study. Documented in the route.
6. No CLI change.

## Acceptance criteria

Unit (synthetic two-layer config, `num_labels` 1, random weights, key names copied from the real
checkpoint's safetensors header into the fixture):
- Loads through the public `load_reranker` path and through a cold `/v1/rerank` request via the real
  cache loader, followed by a cache hit; `reranker_kind(model.config)` on the typed config still
  returns `SEQUENCE_CLASSIFIER`.
- Bidirectionality oracle: with an attention mask supplied, perturbing token *k* changes the final
  hidden state at positions before *k*; a tiny numpy full-attention reference agrees on a two-token
  example.
- Logits equal `score(masked mean of the final hidden state) / temperature` computed independently;
  left- and right-padded inputs with the same real tokens give equal logits; an item scored alone equals
  the same item inside a batch with a five-times-longer peer within 1e-4 in float32, through
  `/v1/rerank`; varying the pad id does not change results; no NaN at any position for a heavily
  left-padded batch in float32 and bf16.
- `pooling` other than `avg` raises. `sanitize` on real key names yields exactly the model's parameter
  names; embedding `sanitize` output and the embedding `Model` baseline (hidden states, normalised
  embeddings, padded batches) are unchanged; embedding and reranker cache groups loaded in both orders
  leave embedding outputs unchanged.
- `reranker_kind` on the embedding checkpoint's config is not a reranker; embedding kind detection
  rejects the reranker checkpoint's config.
- Config round trip: `architectures`, `reranker_pair_template`, `reranker_max_length` and
  `pad_token_id` survive `config.json` → public loader → cached typed config → tokenizer input and
  response metadata. An invalid template fails at load time.
- `_pair_inputs` for the Llama kind reproduces committed golden ids for two known pairs: the first
  unmasked token is 128000, no 128000 elsewhere, no EOS appended to the unpadded sequence (padding is
  identified by the mask, not by id), no chat-template tokens; left padding; truncation by the
  tokenizer's configured side on the complete string, including a case where the query alone exceeds
  the budget; with a real tokenizer whose pad token is unset, padding still succeeds. Significant
  leading and trailing whitespace in both request forms reaches the template intact. `bert` pair
  encoding unchanged.
- Route: sort on raw logits for large positive values (20 and 21 rank the second first) for the Llama
  kind and for an encoder kind; reversed input order; `top_n`; `relevance_score` in [0, 1] for every
  kind; `raw_score` present for every sequence-classifier kind; request `max_length` clamped and
  echoed in metadata.

Parity (real checkpoint; revision hash and tokenizer and config hashes pinned; fixtures committed;
tests skip without the weights, but one recorded run on the pinned revision is attached to the dispatch
report before acceptance):
- Fixture generation pins the `transformers` version, records `truncation_side`, asserts the reference
  is bidirectional (changing the last token changes position 0's hidden state) and that the reference
  ran in float32 at batch size 1; `pip freeze` committed beside the fixtures.
- Parity A (float32 against float32): the card's three pairs plus a committed set of 13 pairs, of which
  four exceed 512 tokens, two exceed 4096 and one approaches 8192, scored at 512 and at 8192 as two
  fixture sets: absolute difference at most 1e-3 on logits; mean-pooled vector cosine at least 0.9999;
  identical within-query ranking where the reference gap exceeds 2e-3.
- Parity B (serving precision, bf16 weights with float32 pooling and head, against the float32
  reference): a tolerance of 0.1 absolute fixed in advance on a calibration subset of four pairs, then
  evaluated on the remaining twelve; identical within-query ranking where the reference gap is at least
  0.1; Kendall τ reported per query, not pooled.
- Through the route at serving precision: the same real pairs scored alone, beside long peers and
  across the server's batch boundary, at 512 and at 8192, with ids, masks and discrepancies recorded.

Procedure:
- Touched files listed as upstream or fork-only with expected hunks (`llama_bidirec/{llama_bidirec,
  config,__init__}.py`, `reranker.py`, `server/reranking.py`, tests and fixtures; any other site found
  in the dispatch enumeration); fork markers on every changed hunk; provenance checks before marking
  shared hunks; the full suite, every `dev/check_*.py` and the `dev/find_*.py` lead generators as the
  fork's AGENTS.md prescribes; pinned formatting hooks; a changelog or fork-change register entry and
  documentation of the new response fields.
- The implementation base is pinned: either the current head, with a trial merge
  (`git merge --no-commit upstream/main`, then abort) showing no conflict in every touched file
  including `config.py` and `__init__.py`, or the operator's merge, with final validation on the
  resulting tree.

## Round-three amendments (applied, not re-reviewed)

- **Length resolution.** `requested = request.max_length if supplied else (config.reranker_max_length or
  512)`; `effective = min(server cap, requested, 8192)`; the config field is a default, never a ceiling. A
  route test proves tokens beyond 4096 reach the model: evidence placed after token 4096 changes the
  score at an 8192 request and not at 512.
- **Serving-precision route gates.** Every bf16 route arrangement (alone, beside long peers, across the
  batch boundary, reversed order, at 512 and at 8192) has explicit gates: absolute logit error at most
  the fixed 0.1 against the float32 reference, and within-query ranking preserved for gaps above it;
  batch invariance at bf16 is gated at 0.05 absolute with the reference's own batch-versus-single
  difference measured separately as the attributable baseline. The 1e-4 invariance gate applies to the
  synthetic float32 model only.
- **Tokenizer sharing.** Encoding uses a request-local tokenizer clone, or encoding is serialised,
  since fast-tokenizer calls mutate backend truncation and padding state; if the server tokenises
  concurrently, a mixed-length concurrent test is required. "Never mutated" means the cached instance's
  configuration is unchanged by any request.
- **Reference arithmetic.** Parity A's reference disables autocast and TF32, records the attention
  implementation and hardware, asserts float32 operands at the pooling and head, and is kept distinct
  from the card's bf16 example.
- **Fixture composition.** The committed pairs form query groups with several competing documents in
  each length regime (short, over 512, over 4096, near 8192); the pre-registered gap thresholds yield a
  nonzero comparison count per regime, and an undefined τ (singleton group) is reported as such, never as
  agreement.
- **Truncation side.** The checkpoint's `tokenizer_config.json` carries neither `truncation_side` nor
  `padding_side`; the fixture generator records both the absence and the loaded tokenizer's effective
  side (right by library default), and the golden truncation tests use the effective value; left
  padding is the card's explicit choice, not a checkpoint field.

## Non-goals

Qwen3-Reranker and mxbai templates; the 8B Nemotron reranker; `cls` or `last` pooling; CLI changes;
Open WebUI; Premisa's host switch to the external reranking engine.

## Open questions

- The operator's decision on the upstream merge fixes the implementation base.
- Whether the study's arm scores at 512 (card parity) or 8192 (card maximum); both are selectable per
  request and recorded in the response metadata.
