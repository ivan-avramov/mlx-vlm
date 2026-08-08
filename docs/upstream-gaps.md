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

`.symbol-exclusions` currently holds a **300-entry baseline**, down from 446 as gaps
were closed (the mlx-lm vendoring alone retired 88). It was captured against
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

All six that originally could not even be collected now run. **Nothing is
skipped**, and `UNPORTED_UPSTREAM_TESTS` in `mlx_vlm/tests/conftest.py` is empty.

Keep the mechanism even while empty: removing an entry is the definition of done
for porting an upstream feature, and it announces what it skips at collection
time (`pytest_report_collectionfinish`) rather than hiding it.

### **[correction]** the two quant-SDPA files were never a permanent divergence

They were skipped for two releases on this reasoning, recorded here and in
`conftest.py`:

> upstream's quantized attention reshapes scores to 5D for `mx.quantized_matmul`,
> which is what makes right-aligning a 4D mask alias `B` with `n_kv_heads`
> (upstream #1567). This fork dequantizes and runs dense attention
> (`models/base.py:251`), so 5D scores never exist and the hazard is
> structurally impossible. Porting the helper would be dead code.

**Every clause of that is false**, and it is a good case study in how a
divergence rationale rots:

1. `models/base.py` had stopped dequantizing — it calls
   `cache.quantized_attention(...)`, a chunked online-softmax path.
2. `TurboQuantKVCache.quantized_attention` *does* build 5D
   `(B, n_kv_heads, n_repeats, L, D)` scores, so the hazard's precondition was
   present all along.
3. What actually prevented #1567 was three untested lines in
   `TurboQuantKVCache._apply_attention_mask` that insert the singleton axis
   before the trailing `(L, K)` pair — upstream's
   `align_attention_mask_to_scores` under another name. Those are now pinned by
   `test_apply_attention_mask_aligns_batch_not_heads_on_5d_scores`, verified to
   fail when the `expand_dims` is removed.
4. "Porting the helper would be dead code" was backwards. Adopting upstream's
   `models/base.py` during the mlx-lm vendoring brought both
   `quantized_scaled_dot_product_attention` and
   `align_attention_mask_to_scores` in as a matter of course, and **130
   previously-skipped tests now pass**.

The lesson generalises: a "permanent divergence" is a claim about code that keeps
changing underneath it. This one survived because nothing re-checked it — the
skip made the tests invisible, and the audits only see missing *symbols*, which
was exactly what the entry excused.

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

Skipped files: **0** (down from 6). The last two ran once the mlx-lm vendoring
brought upstream's quant-SDPA helpers with it — see below.

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

Triaging the ~63 both-ways-diverged files by loss ratio (`-N/+M`, largest first)
puts the same root cause at the top over and over:

| ratio | file | verdict |
|---|---|---|
| 396:1 | `tests/test_processors.py` | mostly cosmetic + test grouping |
| 258:1 | `tests/test_sample_utils.py` | expected — the fork rewrote `sample_utils` |
| 250:1 | `tests/test_trainer.py` | expected |
| 185:1 | `models/qwen3_5/gated_delta.py` | **mlx-lm delegation** (below) |
| 86:1 | `trainer/datasets.py` | cosmetic (`numpy` import) |
| 9.4:1 | `models/base.py` | **mlx-lm delegation** + the quant-SDPA divergence |
| 4:1 | `speculative/drafters/qwen3_dflash/dflash.py` | **mlx-lm delegation** |
| 4:1 | `models/text_only.py` | **mlx-lm delegation** |

**Almost every top hit is one thing: upstream vendored its infrastructure and we
still import it from `mlx_lm`.** Upstream did this across #1593, #1594, #1616,
`893f659a` and `4dac60c6`, and its motivation is stated in the code — the header
of upstream's `models/qwen3_5/gated_delta.py` reads: *"Vendored from
mlx_lm.models.gated_delta (mlx-lm 0.31.3) to drop the module-level mlx_lm import
so qwen3_5 loads without mlx-lm (and under transformers>=5.13)."*

### **[correction]** upstream is not mlx-lm-free, and the numbers here were wrong

An earlier version of this section claimed upstream imports `mlx_lm` in **0**
library files against 31 here. That came from a buggy shell loop. Measured
properly, against a clean `git archive upstream/main`:

| | files importing `mlx_lm` (library code, excl. tests) |
|---|---|
| upstream/main | **2** — and only **1** is a real import |
| here | **19** |

Upstream's two are worth knowing precisely, because together they *are* the target
state:

- `models/qwen3_5/gated_delta.py` — the `mlx_lm` string is only the provenance
  comment quoted above. Genuinely vendored, no import.
- `models/text_only.py` — a **lazy, optional** `from mlx_lm.utils import
  _get_classes` inside a function, wrapped in `try/except ImportError` that
  re-raises with "Loading text-only model_type '…' relies on mlx-lm's model
  registry (no native mlx-vlm implementation). Install it with `pip install
  mlx-lm`." That is a deliberate, contained escape hatch for text-only
  architectures mlx-vlm has no native implementation of — not a dependency.

So "eliminate mlx-lm" is achievable, and upstream shows exactly what the residue
looks like: one lazy optional fallback with a good error message.

### Done — the vendoring, and what it turned up

**Library files importing `mlx_lm`: 19 -> 2.** That is now exactly upstream's
count, in exactly upstream's two files: `models/text_only.py`'s lazy optional
`_get_classes` fallback, and a provenance comment (not an import) in
`models/qwen3_5/gated_delta.py`.

| file | before | after |
|---|---|---|
| `sample_utils.py` | fork rewrite, 1 symbol | **byte-identical to upstream** (18 symbols) |
| `models/text_only.py` | `+4/-16` | **byte-identical** |
| `models/qwen3_5/gated_delta.py` | `+2/-371` | **byte-identical** |
| `speculative/drafters/qwen3_dflash/dflash.py` | `+4/-16` | **byte-identical** |
| `models/base.py` | `+11/-103` | `+6/-4` |
| `models/cache.py` | `+506/-1499` | `+297/-4` |

`.symbol-exclusions` shed **88** entries (401 -> 300).

**The grafts really were pure workaround.** Upstream's vendored classes match
every one of them, down to a detail that looked like fork cleverness:
`QuantizedKVCache.dequantize_for_apc`, `BatchRotatingKVCache.batch_size` *and*
`is_single_row`, and `ArraysCache.batch_size` **without** `is_single_row` — the
asymmetry `models/cache.py` explained at length as necessary to stop
`apc_adapters.clone_cache_entry` recursing forever. That is upstream's own
design. Deleted: the module-level function, three `hasattr` guards, and the
`BatchRotatingKVCache` subclass whose docstring conceded *"only instances built
through this module get the fix."* That hole is now closed by construction.

**Two bugs fell out that the analysis had not predicted:**

1. **The quant-SDPA "permanent divergence" dissolved** — 130 skipped tests now
   run. See the `[correction]` above.
2. **`BatchQuantizedKVCache` was missing `prepare()` / `finalize()`**, upstream's
   right-pad lifecycle. The `qwen3_5_mtp` call site is `hasattr`-guarded, so
   quantized batch caches **silently skipped the right-padding rollup** instead
   of raising. Vendoring supplies the methods.

### What stayed fork work, and how it is marked

`models/cache.py` keeps a boundary comment; everything above it is vendored and
should stay byte-identical apart from two hunks marked `# Fork:`.

- **`prealloc_tokens`** on `BatchKVCache` and `BatchQuantizedKVCache` — a
  first-fill allocation floor so later appends write in place. Merged *into* the
  vendored classes rather than subclassed, deliberately: subclassing is what
  produced the "only our instances get it" hazard, and the whole point of owning
  the code is that we no longer have to.
- **`PreallocKVCache`, `PreallocQuantizedKVCache`, `SlidingWindowCache`,
  `StaticKVCache`** — fork-only classes, below the boundary comment.
- **`models/base.py`'s `cache.quantized_attention` call** — chunked over Q and K
  with online softmax and `mx.eval` between K-tiles, so peak memory is ~O(chunk)
  rather than O(context). Upstream dequantizes the whole state and runs dense
  SDPA. This is a genuine fork optimisation and the only reason `base.py` is not
  byte-identical.

### One behavioural change worth knowing

Quantized **non-TurboQuant** caches previously reached `mlx_lm`'s
`scaled_dot_product_attention` (via `cache=cache`); they now take upstream's
`quantized_scaled_dot_product_attention`. Both are `mx.quantized_matmul`-based,
and upstream's carries the #1567 mask-alignment fix — but it is a real numerics
change on a path the served quantized models use. It is now covered by the 130
tests above, which is more coverage than the old path ever had here.

### The gemma4 gap the sweep found — resolved, with two of my own errors

The sweep flagged `models/gemma4/gemma4.py` (`+12/-38`). Chasing it turned up a
**second dropped commit and a live crash**, and both of the characterisations
recorded here earlier were wrong. Kept as a worked example, because the mistakes
are the instructive part.

**What it actually was.** `#1503` ("Support Gemma 4 variable soft-token budgets")
landed *half*: `processing_gemma4.py` got the `max_soft_tokens` / `patch_size` /
`pooling_kernel_size` forwarding, but `vision.py` did not, so the tower still
padded to a fixed `self.max_patches` and pooled to `self.default_output_length`.
Half-landed commits are worse than fully-dropped ones — the two halves disagree at
runtime:

```
budget=16  processor says 16 soft tokens
ValueError: [broadcast_shapes] Shapes (1,64,32) and (1,16,32) cannot be broadcast
```

Stock defaults happen to agree (processor `max_soft_tokens=280` ==
`VisionConfig.default_output_length=280`), so exposure is a caller passing
`max_soft_tokens`, or a repo whose `preprocessor_config.json` disagrees with its
`config.json`.

**Error 1: "`vision.py` carries fork work (`+50/-67`), so that half is an
interleaved port, not a checkout."** False. `git log` shows no fork-only commit
ever touched `vision.py`; the `+50` was stale pre-`#1503` upstream code, including
a dead `all_same_size = True` / `else:` branch. `#1052`'s NaN-gradient fix was
shared context, not ours. Both `vision.py` and `gemma4.py` were taken from
upstream byte-for-byte.

**Error 2: "the position_ids plumbing is inert on its own."** Also false, in the
useful direction: restoring `mm_token_type_ids` / `token_type_ids` to
`Model.__call__`'s LM kwargs filter matters beyond video. Without it `_make_masks`
always saw `None`, so the bidirectional vision overlay never engaged on the
`Model.__call__` path. The server path was unaffected because
`server/generation.py` passes `mm_token_type_ids` directly — a **CLI-only silent
difference**, invisible to every test.

**Both errors have the same cause:** reasoning from the `+N/-M` line count instead
of running `git log -S` on the symbols. That is exactly what the two history
commands above exist to prevent, and the ratio table is a triage heuristic, not
evidence. Use it to pick what to look at, never to conclude.

The coupling was real, though — just the other way round. With `gemma4.py`
restored and `vision.py` stale, the tests fail `TypeError: VisionModel.__call__()
takes 2 positional arguments but 3 were given`. Landing the pair together is
still the rule.

## Still open — the systematic audit's backlog

A three-way audit plus `dev/find_dropped_hunks.py` replaced the previous
file-ratio triage. Result: **72 upstream commits, ~2749 lines, all
merged-then-dropped** — `git log HEAD..upstream/main` is empty and always was.
**56% of the missing content is tests**, which is why so much of this stayed
invisible: the merge that dropped a feature hunk usually dropped its tests in the
same resolution.

Re-run the scanner after every merge:

```bash
python dev/find_dropped_hunks.py            # ranked, with owning commits
```

It is a lead generator, not an oracle — a commit whose content upstream later
replaced also surfaces, and heavy fork rewrites (`server/generation.py`) show up
as false positives. Every hit still needs `git log -S` and a read.

### Fixed already

`#1492`, `ff2a6daa`, `ecc457b2` (partial), `bc3461b1`, `#1503`, `46ee12dd`,
`#1554` (MTP `+2.0` norm shift), 4 `MODEL_REMAPPING` entries, DeepSeek V4's
`swiglu_limit` clamp + `zero_row_tail`, and the mlx-lm vendoring.

### Ranked backlog — each item has a demonstrated failure

| # | Item | Owning commit | Demonstrated effect |
|---|---|---|---|
| 1 | **Inference routes unauthenticated** | `4993eac1` (#1714) | with `MLX_VLM_SERVER_API_KEY` set, `/v1/models` returns 200 and `/v1/chat/completions` reaches body validation without auth, while `/v1/cache/stats` correctly 401s. Upstream wraps them in `APIRouter(dependencies=[Depends(_require_management_api_key)])`; we register on `app` directly |
| 2 | **MiniMax M3 VL cannot run on any batch/server path** | `ecc457b2` (#1374) | `ar._make_cache(model, [0,1])` -> `ValueError: MiniMaxM3KVCache does not yet support batching`. 3-line `to_batch` guard at the top of `to_batch_cache` |
| 3 | **Streaming `/v1/responses` streams raw chain-of-thought as visible output** | `7c233155`, `cfcc36d9` | `openai.py` streaming path is `delta = chunk.text`; zero `response.reasoning*` events. Non-streaming drops the item too: `_response_output_items_from_text` yields `['message']` where upstream yields `['reasoning','message']` |
| 4 | **`/v1/embeddings` 404s although the implementation ships** | `40757df3` | `server/embeddings.py` + `models/pooling.py` byte-identical to upstream with **zero importers**; the `app.py`/`cli.py` wiring was dropped |
| 5 | **`--quantized-kv-start` silently ignored on the server** | `dab4cb45` (#1582) | plumbed all the way to `ar.py`'s `self.quantized_kv_start = ...` and **read by nothing**. TurboQuant quantizes from token 0 regardless. A dead-parameter tell |
| 6 | **mRoPE cluster** | `a8642018`, `b8671991`, `#1527`, `#1741` | must land WITH `generate/ar.py`: our batcher lacks the MRoPE helpers, so `(3,B,L)` `position_ids` is truncated to `(1,B,L)`. Restoring `qwen3_5` alone *creates* a bug. **Live today for qwen2_vl / qwen2_5_vl / qwen3_vl / qwen3_vl_moe** — qwen3_5's staleness is what masks it |
| 7 | **TurboQuant trim cluster** | `#1447`, `#1453`, `#1432` | `BatchTurboQuantKVCache.is_trimmable()` -> `False`, `trim(2)` -> `0` (upstream: `True`, `2`). Under `--kv-bits` the cache falls into the SSM branch and is indexed `c[0]`/`c[1]` -> `TypeError`; on the non-crashing path every later `gdn_states[j]` index shifts, restoring GatedDeltaNet state from the wrong layer. `zero_row_tail` exists with **no caller anywhere** |
| 8 | **`tool_choice` ignored on `/v1/chat/completions`** | `2394fcf1` (#1611) | `"tools" in ChatRequest.model_fields` -> False; also lost the `if not tools: tool_module = None` guard, so tool markup is parsed for callers who sent no tools |
| 9 | **`skip_special_tokens` is an undeclared attribute** | `cfcc36d9`/`29b6c00b` | `anthropic.py:489` writes `gen_args.skip_special_tokens = False`; the field does not exist on `GenerationArguments` and `generation.py` hardcodes `True`. The recorded "Anthropic tool-markup" fix is **inert** |
| 10 | **1-bit repos cannot load** | `960b26f9` (#1597) | kernel, `__init__` export and 123-line test all landed; only `utils.py`'s 7 lines of `replace_one_bit_modules` wiring dropped |
| 11 | **AWQ quantization absent** | `3c0232ed` (#1666) | 104 lines of `convert.py`, 91% of the hunk |
| 12 | **Empty-input rejection absent** | `a344713a` (#1491) | empty requests reach model load instead of a 400 |
| 13 | `should_add_special_tokens` missing | `53052569` | see the shadowed-test note below; gemma3/3n/4/4_unified/laguna get markers added outside the template |
| 14 | `load_image(PIL.Image)` raises | `84025353` | `ValueError: Unsupported image source type: Image` |
| 15 | `request_normalization.py` orphaned | `221fe0b3` (#1644) | 242 lines, byte-identical, **zero importers**, and it *raises on first call* (passes `top_n_sigma` to a `GenerationArguments` that lacks it) — a landmine for anyone who "fixes" the import |
| 16 | Three sampler modes unreachable | `36331ea7`, `#1653`, `#1663` | `sample_utils.make_sampler` accepts `top_n_sigma`/`p_less`/`typical_p`; no request field reaches them |
| 17 | **`test_models.py` wholesale take** | 24 commits | proven strict subset (AST: 52 methods only upstream, **0** only ours); +46 tests, retires all 57 `test_models.py` exclusions. 3 fail until item 6 lands |

Plus: generation-logging subsystem (`cfcc36d9`), concurrent thinking-budget race
(`ff295e36`), `reasoning_content` alias (`6a8cdff6`), multi-kind preload flags
(`c6084344`), `/v1/responses/input_tokens` counting a different prompt than
`/v1/responses` builds (`eda1ec4f`, half-landed).

### A test that cannot fail — worth internalising

`should_add_special_tokens` is absent from `utils.py`, and
`tests/test_processors.py` imports it. That should be a hard `ImportError`, yet
the suite is green. Reason: **`TestLagunaProcessor` is defined twice** in that
file, so Python's later definition shadows the earlier one and
`test_chat_template_owns_laguna_special_tokens` is **never collected**. Our copy
is byte-identical to upstream, so this is an upstream test bug as well, and worth
reporting there.

Two habits follow. A green suite proves only that *collected* tests pass — check
collection counts, not just results (`pytest --collect-only`). And
`tests/test_utils.py` is excluded from the suite entirely
(`--ignore=tests/test_utils.py`, 5 pre-existing failures), so any guard placed
there never runs; put fork regression tests in a file that is actually collected.

### The methodological rule this page has now paid for six times

Never conclude anything about a divergence from a `+N/-M` line count. Run
`git log -S'<symbol>' --all -- <path>`, check `git merge-base --is-ancestor`
against **both** HEAD and `upstream/main`, and `git show --stat` the owning commit
to get its whole file set. Line counts are for choosing what to look at. Wrong
calls made from them in this repo so far: `vision.py` "needs an interleaved port"
(it was a clean checkout), position-ids plumbing "inert" (it silently disabled a
mask overlay), `trainer/datasets.py` "cosmetic numpy import" (a dropped feature),
`swiglu_limit` "upstream is broken" (wrong constructor), `gemma4_assistant/masks.py`
"import-style only, zero risk" (its absolute import is load-bearing — broke two
tests), and the qwen3_5 `sanitize()` guardrail in CLAUDE.md (protecting a
divergence that no longer exists, which deterred inspection of the exact file
where the `+2.0` MTP bug was hiding).

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
