# Upstream gaps and the silent-merge-loss failure mode

Supersedes `post-merge-gaps-2026-08-06.md`, which is kept for its incident
history but should not be treated as current. Corrections to it are marked
**[correction]** below.

## The failure mode, precisely

This is the thing worth internalising, because it explains why gaps here do not
fix themselves and why "just merge again" never works.

When a merge resolution drops a file or hunk that upstream added, the upstream
commit that added it **still becomes an ancestor of `main`** through the merge's
second parent. Git therefore records that content as *merged, then deliberately
deleted*. Every later three-way merge sees

    merge base: has it   /   ours: deleted it   /   theirs: unchanged

and resolves that to **keep the deletion, silently**. No conflict, no warning.

Consequences that make this hard to notice:

- `git log --diff-filter=D -- <path>` finds **nothing** — the deletion never
  happened in any commit; it happened *between* commits, inside a merge.
- The file/symbol will never be re-offered by any future merge, forever.
- The upstream commit shows as merged in `git log`, so history looks complete.

Verified instance: 16 upstream test files were absent from `main` while
`94edef5` (their source) was an ancestor of `main` — reachable only via a second
parent, never on the first-parent trunk. A trial merge confirmed **16/16 stayed
missing** while all 21 genuinely-new upstream files arrived normally.

## Tooling that now guards this

Two checks, wired into `.github/workflows/upstream-parity.yml` (runs on pushes
to `main` too, not just PRs, since this fork is usually committed to directly):

| Check | Catches | Baseline file |
|---|---|---|
| `dev/check_upstream_parity.py` | whole files in `upstream/main` missing here | `.merge-exclusions` |
| `dev/check_upstream_symbols.py` | `def`/`class` names missing from our copy of a shared file | `.symbol-exclusions` |

Both require a `# reason` on every exclusion and fail if one is missing. Both
read the git **index**, so they can gate a commit rather than only report on one.

Run locally before pushing a merge:

```bash
git fetch upstream
python dev/check_upstream_parity.py
python dev/check_upstream_symbols.py
```

`.symbol-exclusions` currently holds a **401-entry baseline**, down from 446 as gaps
were closed. It was captured against
`upstream/main`. That number is a snapshot of existing divergence, not a defect
count: it mixes never-ported upstream features, modules this fork deliberately
rewrote (`sample_utils`, `apc`, `cache`, `server/generation`), and genuine
dropped hunks. It is entry-per-symbol rather than per-file on purpose, so the
*next* symbol lost from an already-diverged file still trips the check.
Shrinking it is good; adding to it needs a specific reason.

## Restored, with a caveat

All 16 dropped test files are back, restored **byte-identical to upstream**.
That is deliberate: unmodified files let future `git merge upstream/main` apply
upstream's own edits to them cleanly, and let the parity check prove they have
not gone missing again.

Four of the six that originally could not even be collected have since been
fixed and are running. **Two remain skipped**, in one reviewable place in
`mlx_vlm/tests/conftest.py`, and they are a deliberate permanent divergence
rather than a backlog item:

| Skipped file | Needs | Why it stays skipped |
|---|---|---|
| `test_quant_sdpa_mask.py` | `base.quantized_scaled_dot_product_attention` | the symbol genuinely does not exist here: plain quantized caches delegate to `mlx_lm`'s SDPA, and TurboQuant caches use their own `quantized_attention`. The hazard those tests cover is handled elsewhere — see below. |
| `test_quant_sdpa_mask_adversarial.py` | as above | as above |

Deleting an entry from `UNPORTED_UPSTREAM_TESTS` is the definition of done for
porting that feature.

### **[correction]** why those two stay skipped — the old reason was wrong

This entry previously read: "this fork dequantizes and runs dense attention
(`models/base.py:251`), so 5D scores never exist and the hazard is structurally
impossible." **Both halves are false**, and the error was load-bearing, so it is
worth spelling out.

`models/base.py` no longer dequantizes — it calls `cache.quantized_attention(...)`
(chunked over Q and K with online softmax, `mx.eval` between K-tiles, so peak
memory is ~O(chunk) rather than O(context)). And `TurboQuantKVCache.quantized_attention`
reshapes queries to `(B, n_kv_heads, n_repeats, L, D)`, so **its scores are 5D —
exactly the shape family that makes upstream #1567 possible.**

What actually prevents the hazard is `TurboQuantKVCache._apply_attention_mask`
(`turboquant.py:5419`):

```python
mask_chunk = mask[..., q_start:q_end, k_start:k_end]
if mask_chunk.ndim == scores.ndim - 1:
    mask_chunk = mx.expand_dims(mask_chunk, axis=2)
```

That inserts the singleton axis *before* the trailing `(L, K)` pair, which is
upstream's `align_attention_mask_to_scores` by another name. So the conclusion
("the tests would be dead code here") still holds — but for a completely
different reason than recorded, and via **three lines that were untested**.
Nothing in the suite referenced `_apply_attention_mask`, so a refactor could have
deleted that `expand_dims` and reintroduced #1567 while this page asserted the
hazard was impossible. `test_apply_attention_mask_aligns_batch_not_heads_on_5d_scores`
and `..._causal_string_broadcasts_over_5d_scores` in `tests/test_turboquant.py`
now pin it; the first was verified to fail when the `expand_dims` is removed
(row 0 silently receives row 1's mask — no exception, just wrong numbers).

**One narrower-than-upstream point, left as-is deliberately.** The guard is a
single `if ... == scores.ndim - 1`, where upstream loops
`while mask.ndim < scores.ndim`. A mask two or more ranks short — e.g. a 3D
`(B, L, K)` — would therefore still right-align and alias `B` with `n_repeats`.
Masks reaching here are 4D or the `"causal"` string, so this is latent rather
than live; noted so that whoever makes 3D masks reachable knows to widen the
guard to a loop first.

## Fixed in this pass

- **`prompt_has_open_thinking` (merge completion).** Upstream's #1811 call sites
  auto-merged into `server/openai.py` but neither the definition nor the import
  came with them, so *every* chat-completions request raised `NameError` and
  500'd. Ported the definition into `responses_state.py` and wired the import
  and re-export. This accounted for 9 of the 10 failures the merge introduced.

- **Rotating-cache guard bypass (correctness, issue #1715).** See below.

- **`minicpmv4_6` import.** `NORM_WEIGHT_SUFFIXES`,
  `should_offset_norm_weight`, `should_shift_norm_weights` were restored to
  `models/qwen3_5/qwen3_5.py`. Only those three helpers were restored — **not**
  upstream's `sanitize()`, which would have regressed the production `+1.0`
  double-shift fix. The `language_model.`-prefix self-guard is intact at
  `qwen3_5.py:149` and `qwen3_5_moe.py:27`.

- **DeepSeek V4 HISA — now fully wired.** The config fields were restored first,
  then `Indexer` was replaced with upstream's (verified purely additive plus a
  reorder), adding `_hisa_select`, the L>1 batched path via `hisa_select`, and the
  kernel import. `hisa_kernel.py` had matched upstream byte-for-byte all along, so
  only the wiring was ever missing. All 4 `TestDeepseekV4HISA` tests pass.

- **APC under `--kv-bits`.** `generate/ar.py` disabled APC entirely whenever
  `kv_bits` was set, so quantized runs got zero prefix caching. That guard was
  justified when warm restore always rebuilt float caches; threading
  `kv_quant_config` through the warm builders removed the reason, so it was
  removed. Likely the most user-visible change here, since the served models are
  quantized.

### The rotating-cache bug, in detail

**[correction]** The old doc said prefix-cache reuse was "entirely absent from
our `dispatch.py`" and therefore had no impact. Both halves are wrong. The fork
*does* do prefix reuse (`generate/dispatch.py` ~898-981) and *does* have its own
guard (`_rotating_rewind_safe` + a snapshot ring + `_trim_cache`) — a different
design from upstream's `_prefix_cache_trim_amount`/`_cache_fully_retained`, not
a missing one. The real problem was narrower and more serious than "not ported":

Both guards dispatched on `type(c).__name__`, which does not match subclasses.
`BufferedRotatingKVCache(RotatingKVCache)` — installed by speculative decoding
(`speculative/mtp.py:509`, `drafters/qwen3_dflash/dflash.py:161`, reachable
whenever a draft model is configured, which mlx-serve does via `--draft-model`)
— was therefore treated as a plain flat cache by both:

1. `_rotating_rewind_safe` skipped the ring-wrap check and returned `True` for a
   wrapped, partly-evicted cache, declaring an unsafe rewind safe.
2. `_trim_cache`'s protective skip-list missed it, so after `.trim()` had already
   moved the offset logically, the ring buffer was *also* physically sliced
   (16 → 5 positions with `offset` still 36), desyncing the ring index.

Plain `RotatingKVCache` was protected; only the fork's own subclass was exposed.
Fixed via `_cache_kind_names()` (matches the whole MRO, keeping the existing
name-based style so optional `mlx_lm` classes need not be imported) plus honouring
`start_position`, the eviction watermark for buffered/chunked caches.
`TestPrefixCacheReuseTrim` (8 tests, the fork counterpart of upstream's dropped
class) covers it; 5 of the 8 fail without the fix.

Follow-up, now done (commit `087c91a3`): `_is_rotating_kv_layer` had the same
exact-name hole, and it is the *routing* predicate — the post-generation path
uses it to choose snapshot-restore vs. `_trim_cache`. A buffered layer got
neither (the same predicate gates `_capture_rotating_layers_for_snapshot`, so no
snapshot existed) and then fell into the trim branch. Fixed, together with
`start_position` plumbed through `RotatingKVSnapshot` / `capture_rotating` /
`restore_rotating`, since restoring K/V and offset while leaving the eviction
watermark stale would leave the layer claiming tokens the buffer no longer holds.

**`_rotating_post_gen_trim_safe` is intentionally dead — do not wire it in.**
Commit `087c91a3`'s message called it "a guard not wired into the path it was
written to protect"; that was wrong. `memory.md:274` records it being
*deliberately* unhooked when the mid-prefill snapshot landed, and `memory.md:315`
explains why the snapshot is strictly better: rather than refusing to trim a
wrapped ring (correct, but forfeits all prefix reuse), it captures rotating state
exactly at the anchor offset during chunked prefill so post-gen restores instead
of trimming. Wiring the strict guard would reject the asymmetric anchoring path
almost always — at the anchor the ring is normally already wrapped (Gemma 4's
`max_size` is 1024) — turning the cache-reuse win back into full re-prefill. Its
subclass hole was still fixed, and its docstring now says all of this so the
trap is not re-sprung; its tests are kept as documentation of the SWA invariant
behind the "repetition loops on turn 2" incident.

## Test-suite state

**The suite is green: 2197 passed, 5 skipped, 0 failed.**

| Stage | Result |
|---|---|
| Before the merge | 36 failed, 1805 passed |
| After the merge | 46 failed, 1833 passed (10 regressions, the `prompt_has_open_thinking` break) |
| After restoring 16 dropped test files | 76 failed, 1924 passed (44 pre-existing gaps became *visible*) |
| **Now** | **0 failed, 2197 passed** |

The rise to 76 was not a regression — restoring the dropped test files exposed
gaps that had been invisible because the tests for them were missing.

Skipped files: 2, the deliberate quant-SDPA divergence (down from 6).

**Compare failing test IDs, not counts**, when validating a change:
`comm -13`/`comm -23` over the sorted `FAILED ...` lines. A fix-one-break-one swap
shows an identical total.

## Green does not mean converged

This is the most important thing on this page.

Of the last five real bugs found in this fork, **four had no failing test**:

| Bug | Impact | Test that caught it |
|---|---|---|
| Gemma 4 tool-call normalization dropped (`tool_parsers/gemma4.py`, +0/−8) | `":name{...}"` and `"name{...}"` call shapes silently fail to parse | none |
| Anthropic endpoint seeded thinking state from `enable_thinking` instead of `prompt_has_open_thinking(...)` | wrong thinking state; also `skip_special_tokens` left on, stripping tool markup | none |
| `glm4_moe_lite_mtp` / `inkling_mtp` missing from the drafter registry | those drafters cannot be auto-detected | none |
| 5 models missing from the prompt-format registry | prompts formatted by the fallback path | none |
| `gemma4`/`gemma4_unified` processors stuck pre-#1492 | `AutoProcessor` rejects the DiffusionGemma processor | yes |

All four silent ones surfaced only by **diffing files that a dropped upstream
commit had touched**. Neither audit can see them: the files are present and every
symbol is present.

### The two history commands that actually settle things

Learned the hard way, twice, on the diffusion subsystem:

```bash
git log -S'<symbol>' --all -- <path>    # who introduced it, and was that upstream?
git show --stat <commit>                # EVERY file that commit touched
```

1. **A "fork-only" symbol is not necessarily fork work.** `diffusion.py`'s seven
   fork-only symbols looked like a two-way divergence needing an architecture
   decision. `git log -S` showed all seven came from upstream PRs #1347/#1348 and
   were deliberately removed upstream by #1359 and #1508. The fork was carrying
   upstream's June implementation. There was no design decision — just staleness.
2. **A dropped commit usually spans several files, and they are internally
   consistent while stale.** #1492 changed `gemma4`'s and `gemma4_unified`'s
   processors together. Restoring only one *broke* two passing tests. Restoring
   one file of a multi-file dropped commit is worse than restoring none.

### Classify divergence by direction

Cheap triage that finds losses without waiting for a test:

```bash
git diff --numstat upstream/main -- 'mlx_vlm/**/*.py' |   awk '$1==0 && $2>0 {print "PURE LOSS -"$2"\t"$3}'
```

`+0/−N` means upstream content absent with nothing of ours added — safe to take
verbatim. As of this writing there are **0** such files (22 are fork-only, ~63
diverge both ways and need case-by-case review).

## The recurring failure mode, in full

| Shape | Example | Caught by tooling? |
|---|---|---|
| Whole file dropped | 16 upstream test files | yes (parity) |
| Symbol dropped | `dynamic_roll`, `layer_kv_for_apc` | yes (symbols) |
| Definition dropped, call sites merged | `prompt_has_open_thinking` → every chat request 500'd | yes (symbols) |
| Call site dropped, definition merged | `_image_model_type_from_component_indexes`; `anthropic.py`'s thinking seed | **no** |
| Module constant dropped | `_COMPRESSED_TENSORS_DROP_SUFFIXES` | **no** |
| Single guard/line dropped | `alias_model_class.supports_model(model)` — broke every quantized flux2 repo | **no** |
| Registry entry dropped | `"kimi_k3"`, `"mage"`, `inkling_mtp`, 5 prompt formats | **no** |
| Hunk dropped from a present file | gemma4 processors stuck pre-#1492 | **no** |
| Duplicate authority kept instead of delegating | 5 cache-classification ladders, 117 lines | **no** |
| Stale *upstream* code retained after upstream replaced it | the whole diffusion subsystem | **no** |
| Stale *fork* test kept over upstream's rewrite | `lfm2_vl` tests asserted the inverse of our own code | **no** |
| Positional payload widened, one consumer left at the old arity | `self.requests` grew to a 7-tuple; `_run_diffusion` still unpacked 5 → `ValueError` on every diffusion request | **no** |
| Parameters kept, the hunk that uses them dropped | `_format_video_message` still declares `skip_audio_token` / `num_audios`; the body and the call site that pass them are gone, so they are dead | **no** |

Nine of thirteen are invisible to the audits. They catch structural loss only.

The last two share a tell worth internalising: **a signature that nothing
exercises**. Both a positional payload read at two different arities and a
parameter no caller supplies are cheap to grep for and neither needs upstream as
a reference — they are internally inconsistent on their own terms. Upstream's
move to a dataclass (`QueuedGenerationRequest`) removes the first shape by
construction, which is a better reason to take it than byte-convergence.

## Closed: upstream #1492's video-input support

Ported. The `−66` lines `prompt_utils.py` carried against upstream turned out to
be **three** dropped commits, not one — worth recording, because reading the
line count as a single feature is what made this look smaller than it was:

| Missing piece | Owning upstream commit | Status |
|---|---|---|
| `_get_video_token()`, video markers + `("video","input_video","video_url")` in `_flatten_content`, `_messages_to_plain_prompt` wiring | `4d468e85` (#1492) | **restored** |
| `_format_video_message` audio preservation + its positional call site | `ff2a6daa` | still open, see below |
| `_template_references_kw` + the `thinking_mode="enabled"` block | `ecc457b2` (#1374) | still open, see below |

`server/schemas.py`'s four TypedDicts (`VideoUrl`, `ResponseInputVideoParam`,
`ResponseVideoUrlParam`, `ResponseVideoParam`) plus the
`ResponseInputContentParam` union are restored byte-identical to upstream.
`server/openai.py` has `_extract_video_reference` and the full request path
(collection → `apply_chat_template(video=…)` → preflight → all four generate
call sites). `server/app.py`'s preflight takes `videos`.

**The queue rewrite was the substantive part.** `server/generation.py` carried a
7-tuple on `self.requests` — `(rqueue, raw_inputs, prompt_tokens, args, images,
prompt_cache_state, prompt)` — read three different ways: `_item[:5]` in the
batch loop, `*_extra` in the speculative loop, and **a bare 5-tuple in
`_run_diffusion`**. That last one meant every diffusion request submitted through
`generate()` raised `ValueError: too many values to unpack`. Adopting upstream's
`QueuedGenerationRequest` dataclass (upstream's 6 fields, plus the fork's
`prompt_cache_state` / `prompt`) fixes it structurally and converges the shape.
No test covered `_run_diffusion` — another entry for the "green does not mean
converged" table above. Three tests now do.

`_preprocess_request` is worth keeping even though the fork's `prepare_inputs`
already accepts `videos`: it preserves the 3-argument `_cpu_preprocess(prompt,
images, audio)` call shape when `videos is None`, which is what lets existing
test fakes (and any subclass override) keep working.

**One more dropped hunk, found by ratio-triage rather than by the port.**
`models/gemma4/__init__.py` was `+1/−5`: it did not re-export
`Gemma4VideoProcessor`, from `bc3461b1` (#1523). Every other file that commit
touched had landed — `processing_gemma4.py` and `processing_gemma4_unified.py`
byte-identical, `gemma4/rope_utils.py` correctly deleted — so only the export was
missing, while the class itself was present. That matters because
`diffusion_gemma/processing_diffusion_gemma.py:71` declares
`video_processor_class = "Gemma4VideoProcessor"` and `AutoProcessor` resolves it
by name off the model package.

**Neither audit could see it, by construction:** the parity check works on whole
files (present), and the symbol check only compares `def`/`class` names — an
`__init__.py` defines none, and the class *is* defined in the file it lives in.
A re-export is invisible to both. Worth remembering that package `__init__.py`
files are a structural blind spot for the current tooling; `git diff --numstat
upstream/main` on them is the only thing that finds this shape.

## The both-ways sweep: one divergence explains most of it

Triaging the ~63 both-ways-diverged files by loss ratio
(`−N/+M`, largest first) surfaces this at the top:

| ratio | file | verdict |
|---|---|---|
| 396:1 | `tests/test_processors.py` | mostly cosmetic + grouping |
| 258:1 | `tests/test_sample_utils.py` | expected — the fork rewrote `sample_utils` |
| 250:1 | `tests/test_trainer.py` | expected |
| 185:1 | `models/qwen3_5/gated_delta.py` | **mlx-lm removal** (below) |
| 86:1 | `trainer/datasets.py` | cosmetic (`numpy` import) |
| 9.4:1 | `models/base.py` | **deliberate** — the quant-SDPA divergence + mlx-lm |
| 4:1 | `speculative/drafters/qwen3_dflash/dflash.py` | **mlx-lm removal** |
| 4:1 | `models/text_only.py` | **mlx-lm removal** |
| 3.2:1 | `models/gemma4/gemma4.py` | real gap, see below |

**The ratio heuristic works, but nearly every top hit is the same one thing.**
Upstream has completed an mlx-lm *removal* series (#1593, #1594, #1616, plus
`893f659a` and `4dac60c6`): it vendored the text architectures, activations,
RoPE helpers and cache classes into `mlx_vlm/` and stopped importing `mlx_lm`
in library code.

    # upstream/main
    $ grep -rl 'from mlx_lm\|import mlx_lm' --include='*.py' mlx_vlm | wc -l
    0
    # here
    31        (19 outside mlx_vlm/tests)

`models/cache.py` alone: **0** mlx_lm references upstream, **25** here.

Both remotes still *declare* `mlx-lm>=0.31.3` in `requirements.txt`, so this is
about imports in library code, not the dependency itself.

**This is an architecture decision, not a dropped hunk, and it should not be
resolved file-by-file.** Two coherent positions:

- **Follow upstream.** Removes the largest single source of future merge
  conflict. But the fork's APC work is built *on* mlx-lm internals — e.g.
  `models/cache.py` grafts `dequantize_for_apc` onto mlx_lm's
  `QuantizedKVCache`, and `BatchRotatingKVCache` subclasses mlx_lm's to carry a
  right-pad prefill fix mlx_lm lacks. Vendoring means owning all of that.
- **Stay on mlx-lm and declare it permanent.** Then these ~10 files should be
  recorded here as intentional divergence (like the quant-SDPA entry) so future
  sweeps stop re-triaging them, and `.symbol-exclusions` reasons should say
  "mlx-lm delegation" rather than "pre-existing divergence, unreviewed".

Either way the *next* action is a decision, not a patch. Until it is made,
treat every "we import `mlx_lm`, upstream defines it locally" hunk as explained
rather than as a gap.

**One genuine gap the sweep did find**, independent of the above:
`models/gemma4/gemma4.py` (`+12/−38`) is missing two hunks from `bc3461b1`
(#1523) — the same commit whose `__init__.py` export this branch just restored:

1. `image_position_ids` / `video_position_ids` plumbing through
   `get_input_embeddings` and `encode_image`, with `_encode_image` /
   `_encode_video` split out of `_encode_vision`.
2. Conv-weight layout *detection* in `sanitize()`. Upstream transposes only when
   the incoming shape does not already match the expected MLX layout
   (`expected_in` derived from `audio_config.subsampling_conv_channels`); we
   transpose unconditionally. Upstream's commit message for this is "Handle
   Gemma4 audio weights in MLX layout", i.e. **already-converted checkpoints get
   double-transposed here.**

Item 2 is a correctness bug with no failing test, and it is independent — it can
be fixed on its own.

**Item 1 is a textbook lesson-2 trap, and worth checking before you touch it.**
`gemma4.py` and `vision.py` are consistently stale *together*:

```
ours:      def __call__(self, pixel_values) -> mx.array:                          # vision.py
upstream:  def __call__(self, pixel_values, pixel_position_ids=None) -> mx.array:
```

So restoring `gemma4.py`'s `_encode_image` / `_encode_video` split alone makes it
call `self.vision_tower(pixels, position_ids)` against a one-argument
`VisionModel.__call__` — an immediate `TypeError` on every Gemma 4 image request.
The pair must land together, and `vision.py` carries fork work (`+50/−67`), so
that half is an interleaved port, not a checkout.

## Still open

Both of the dropped-commit items previously listed here are **done** —
`ff2a6daa` (audio preservation in Gemma 4 video prompts, all four files) and
`ecc457b2`/#1374 (`_template_references_kw` + the `thinking_mode` block).
`prompt_utils.py` is now `+280/-2` against upstream, and those two remaining
`-` lines are the fork's own text-only fallback (`MessageFormatter` returning
plain messages for unknown `model_type` instead of raising). **No upstream
content is missing from `prompt_utils.py`.**

What is left is one decision and one verified lead:

1. **The mlx-lm removal divergence** — an architecture decision, written up
   above under "The both-ways sweep". Nothing should be patched file-by-file
   until it is made.
2. **`models/gemma4/gemma4.py`'s `image_position_ids` / `video_position_ids`
   plumbing** (from `bc3461b1`/#1523) — a verified lesson-2 trap: it must land
   together with `vision.py`, whose `VisionModel.__call__` still takes only
   `(pixel_values)` where upstream takes `(pixel_values, pixel_position_ids=None)`.
   Restoring the caller alone `TypeError`s on every Gemma 4 image request.
   `vision.py` carries fork work (`+50/-67`), so that half is an interleaved port.

Also open, low priority: `_rotating_post_gen_trim_safe` stays intentionally
unwired (see above) — correct, not a gap.

## Formatting is convergent here, not divergent — settled

This was deferred for a while on the reasoning that "several of the unformatted
files are files upstream also has, so reformatting them diverges from upstream's
bytes and creates future merge conflicts." **That is false, and it is worth
recording why so it is not deferred on the same reasoning again.**

    $ git diff upstream/main -- .pre-commit-config.yaml     # -> empty; same pins
    $ git archive upstream/main | tar -x -C /tmp/up && cd /tmp/up
    $ black --check $(find . -name '*.py')
    All done! 1148 files would be left unchanged.

Upstream's entire tree is already clean under the *same* pinned
black 26.3.1 / isort 5.13.2 / autoflake 2.2.1 this repo declares. So an
unformatted file here is not "ours, differently styled" — it has **drifted away
from upstream's formatting**, and reformatting it moves it *toward* upstream.

Measured over the 12 shared files that were unformatted:

| | lines differing from `upstream/main` |
|---|---|
| before | 5690 |
| after | 5685 (**−5**) |

and `speculative/mtp.py`, `tests/test_cohere_tool_parser.py` and
`tests/test_smoke.py` became **byte-identical to upstream** (they had differed
only by a magic trailing comma that newer black adds). The files that went
slightly *up* (`server/app.py` +6, `tests/test_server.py` +16) did so only by
reformatting fork-added code, which upstream does not have either way.

Two notes for whoever runs this next:

- **autoflake is the hook with teeth.** black and isort only move whitespace;
  autoflake deletes imports. It removed `top_p_sampling` from
  `server/generation.py` and `DEFAULT_DIFFUSION_CONFIDENCE_THRESHOLD` from
  `generate/dispatch.py` — both verified unreferenced in the importing module and
  not re-exports. `models/base.py` and `models/cache.py` are excluded from
  autoflake in `.pre-commit-config.yaml` precisely because they *are* re-export
  surfaces; keep that exclusion, and check any new removal by hand.
- `black` warns "Python 3.12 cannot parse code formatted for Python 3.14" and
  skips its AST safety check. It still exits 0, so this does not fail
  `pre-commit`, and formatting is derived from file content rather than the
  running interpreter — but CI runs Python 3.10, so that path is verified only
  by CI actually going green, not from here.

## Not a source of known issues

`docs/report_issues.md` is four lines of upstream boilerplate pointing at
Blaizzy's issue tracker. It contains no fork-specific content and is
intentionally left untouched to avoid merge friction.
