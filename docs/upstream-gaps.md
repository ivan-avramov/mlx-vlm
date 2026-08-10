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

**Six** checks. Five are wired into `.github/workflows/upstream-parity.yml`
(which runs on pushes to `main` too, not just PRs, since this fork is usually
committed to directly); `find_dropped_hunks.py` stays manual because it is a
ranked report rather than a pass/fail gate.

| Check | Catches | Baseline file |
|---|---|---|
| `dev/check_upstream_parity.py` | whole files in `upstream/main` missing here | `.merge-exclusions` |
| `dev/check_upstream_symbols.py` | `def`/`class` names missing from our copy of a shared file | `.symbol-exclusions` |
| `dev/check_upstream_deletions.py` | the **reverse**: files/symbols upstream deleted that we kept | `.deletion-exclusions` |
| `dev/check_fork_markers.py` | fork changes to a shared file that carry no `# Fork:` marker | `.fork-marker-allowlist` |
| `dev/check_dead_helpers.py` | helpers upstream calls from library code that we reach only from tests | `.dead-helper-exclusions` |
| `dev/find_dropped_hunks.py` | dropped *hunks*, ranked by owning commit (Python **and** docs/config) | — |

All five gating checks require a `# reason` on every exclusion and fail if one is
missing, and all read the git **index**, so they can gate a commit rather than only
report on one. `check_upstream_symbols.py` and `check_fork_markers.py` additionally
warn when an entry no longer excuses anything — a stale exclusion is a hole in the
next audit, and it has caught retired entries three times since it was added.

Run locally before pushing a merge:

```bash
git fetch upstream
python dev/check_upstream_parity.py
python dev/check_upstream_symbols.py
python dev/check_upstream_deletions.py    # ~70s
python dev/check_fork_markers.py          # ~35s; --summary ranks the rollout
python dev/check_dead_helpers.py          # ~10s; dropped call sites
python dev/find_dropped_hunks.py          # slow; ranked report
```

`.symbol-exclusions` currently holds **57 entries**, down from 446 (the mlx-lm
vendoring retired 88, the `test_models.py` take another 57). That number is a
snapshot of existing divergence, not a defect count: it mixes never-ported upstream
features, modules this fork deliberately rewrote (`sample_utils`, `apc`, `cache`,
`server/generation`), and genuine dropped hunks. It is entry-per-symbol rather than
per-file on purpose, so the *next* symbol lost from an already-diverged file still
trips the check. Shrinking it is good; adding to it needs a specific reason.

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

**`c503fa7b` is therefore CLOSED as a reviewed deliberate divergence, not a gap.**
`find_dropped_hunks.py` ranks it third (87 lines: 51 in `tests/test_generate.py`,
36 in `generate/dispatch.py`) and will keep doing so, because the fork solved the
same problem a different way. Both of its symbols are excused with real reasons in
`.symbol-exclusions`, and its 51 test lines are the tests *for those symbols*, so
they are superseded by `TestPrefixCacheReuseTrim` rather than missing. Do not
"restore" it: adopting `_prefix_cache_trim_amount`/`_cache_fully_retained` alongside
the snapshot ring would give two guards with different notions of when a rewind is
safe. This is the clearest standing example of a permanent, legitimate entry in that
report — the report is a lead generator, and this lead has been followed.

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

## The dropped-hunk report has bottomed out — 19 commits, all closed by review

**Settled 2026-08-10, and this is the section to read before re-investigating any
`find_dropped_hunks.py` hit.** The wide pass
(`--min-lines 1 --min-share 0.05 --max-commits 400`) went 72 -> 43 -> 24 -> **19**
commits. Those 19 will keep appearing forever, and that is correct: **every one of
them now has an established disposition, and none is missing content.** The report
attributes by line CONTENT, so a reviewed supersession, a rewrite, or even a local
variable rename reads identically to a dropped hunk.

`.fork-marker-allowlist` is empty and `.deletion-exclusions` is empty, so the report
count is no longer a progress signal of any kind. **The exclusion baselines are**
(`.symbol-exclusions` 14, everything else 0).

Closed because the fork deliberately supersedes them — see each file's `# Fork:`
markers, which name the commit:

* **`c503fa7b`** (88) — the fork guards rotating-cache prefix reuse with
  `_rotating_rewind_safe` + a snapshot ring rather than upstream's
  `_prefix_cache_trim_amount` / `_cache_fully_retained`. `generate/common.py`'s
  `start_position` branch names `BufferedRotatingKVCache` and explains that it
  "reports itself trimmable even after evicting" — exactly what `c503fa7b` fixed.
  Running both would give two different notions of when a rewind is safe. Its 51
  `test_generate.py` lines are the tests for those symbols, and all 9 are REVIEWED
  `.symbol-exclusions` entries superseded by the fork's own `TestPrefixCacheReuseTrim`.
* **`eb7537b9`** (#1210, 20) — reduces **entirely** to `_sample_top_p_one`. Every
  identifier the report flags (`argsort`, `sorted_probs`, `cumulative_probs`,
  `take_along_axis`, `top_probs`, `zeros_like`, `softmax`, `top_p_sampling`) lives in
  that one method's body, which the fork replaced with a `_filter()` applying
  top_p / min_p / top_k. A REVIEWED `.symbol-exclusions` entry.
* **`a492e47d`** (6) — `generate/common.py`'s inline uniform-quant path, replaced
  deliberately. Its `apc.py` line is a content-attribution artifact: all 12 of the
  lines it added there are present.
* **`b590c747`** (8) — fork work (`f0d50c90`, ours, four days before upstream's own
  fix), marked at `qwen3_5_moe.py:59`. **Reclassify rather than close:** the only
  difference is probing `gate_proj` vs `up_proj`, equivalent for any real SwiGLU
  checkpoint, so it is a convergence candidate carrying a permanent conflict site.

Closed because upstream's own copy is the defect:

* **`1171888e`** (9) — our `zero_row_tail` is **byte-identical to upstream's live
  copy**. Upstream defines it **twice** inside `BatchTurboQuantKVCache`, so its first
  copy is dead code and the report flags that shadowed one as "missing". Confirmed
  mechanically in 2026-08-10, not by spot check. Same shape as the doubly-defined
  `TestLagunaProcessor` in `tests/test_processors.py`; both are worth reporting
  upstream.

Closed because the content is present and the report is measuring alignment:

* **`9edb3c6b`** (#1299, 7) and **`ec0f2354`** (#1228, 3) — both reduce to
  `thinking_start_token_id` / `thinking_end_token_id` at 2 occurrences vs upstream's 4.
  Ours calls the same `_thinking_token_ids` helper and unpacks it into `start_id` /
  `end_id`. **The reported lines are a local variable rename**, with byte-identical
  logic either side. This is the purest illustration of why the count is not a signal.
* **`4d468e8575`** (#1492, 4) and **`b3d2380d1d`** (#1266, 3) — both verified FULLY
  LANDED in `server/openai.py` (video plumbing; per-token thinking split). Said so in
  that file's markers so nobody re-derives them from its divergence.
* **`7267aff2`** (#1313, 2) — both halves landed; the reported lines sit inside the
  fork's EpiCache refactor (`_qwen35_scalar_positions`) of the region it introduced.
* **`182eef66`** (#1229, 14), **`5034c609`** (#1203, 10), **`ffd7aeff`** (3),
  **`477d3aeb`** (1), **`dab4cb45`** (1), **`5788472570`** (#628, 1),
  **`473692ea73`** (1), **`d85ca4d0`** (#1181, 1), **`3947dd03`** (#1029, 1) — each
  lands in a file that was reviewed site-by-site during the `.fork-marker-allowlist`
  rollout (`generate/common.py`, `generate/__init__.py`, `server/cli.py`,
  `generate/dispatch.py`, `generate/ar.py`, `prompt_utils.py`,
  `tests/test_prompt_utils.py`, `utils.py`) and is accounted for by that file's
  markers — predominantly the fork's `print`/`logging` -> module-logger conversions
  and its generation-engine port.

**How to use this list.** If a hit is here, do not re-investigate it; extend the entry
if you learn something. If a hit is NOT here, it is new — treat it as a lead and run
`git log -S`. That distinction is the only thing that makes a permanently-nonzero
report usable.

## Still open — the systematic audit's backlog

A three-way audit plus `dev/find_dropped_hunks.py` replaced the previous
file-ratio triage. Result: **72 upstream commits, ~2749 lines, all
merged-then-dropped** — `git log HEAD..upstream/main` is empty and always was.
**56% of the missing content is tests**, which is why so much of this stayed
invisible: the merge that dropped a feature hunk usually dropped its tests in the
same resolution.

**Progress as of 2026-08-09 (second pass).** 72 -> **33** commits with missing
content, 39 -> **34** diverged files, 239 -> **107** `.symbol-exclusions`
entries (zero of them stale). Suite 2380 -> **2514 passed, 5 skipped, 0 failed**.
**Everything that needed an operator decision landed** — #1 (auth), #6 (mRoPE)
and #17 (`test_models.py`) after confirmation, plus #1454's `video_generate`
removal. Also closed: #1611 `tool_choice`, #1491 empty-input rejection,
`load_image(PIL)`, the 1-bit wiring, laguna tokenization, the TurboQuant trim
cluster, three dropped dependency commits, and the whole `convert.py` cluster
(#1439, #1666 AWQ, 2066c930) — `convert.py` is now byte-identical to upstream.

Both audit blind spots found this pass are now closed by tooling:
`dev/check_upstream_deletions.py` covers the reverse direction (deletions we
reverted), and `find_dropped_hunks.py` now scans docs/config with a threshold that
fits them. Four checks, not three; see AGENTS.md.

**Progress as of 2026-08-09 (third pass).** Merged four upstream commits — #1807
(per-tensor KV quantization), #1814, Bonsai detection — the first merge under the
weekly rule; seven files conflicted and the resolutions are itemised in that merge
commit. Suite 2514 -> **2564 passed, 5 skipped, 0 failed**. Two dropped hunks
restored out of merge conflicts rather than from the backlog: `8422ece8`'s (#1638)
`server/app.py` half, which had left the on-disk APC namespace not fingerprinting
the KV-quant config, and both halves of `a492e47d`'s (#1494) `masks.py`/test pair.
`.symbol-exclusions` lost 8 entries (107 -> 99) to convergence when #1807 deleted
upstream's own `dispatch.py` duplicates; `.deletion-exclusions` 11 -> 8. **Fifth
check added:** `dev/check_fork_markers.py`, the fork-content oracle — see below.

**Progress as of 2026-08-09 (fourth pass, same day).** `find_dropped_hunks.py`
**34 -> 12** commits, four of which are now closed-by-review rather than open work;
diverged files 31 -> 28. Suite **2623 passed / 5 skipped / 0 failed**, and `tests/test_utils.py` is collected again (its 5 "pre-existing failures"
were stale test code — the only remaining `--ignore` is `test_smoke.py`). Exclusions:
`.symbol-exclusions` 107 -> **57**, `.deletion-exclusions` 11 -> **4**,
`.fork-marker-allowlist` 32 -> **19 files**, and the new `.dead-helper-exclusions` is
**empty**. **Sixth check added:** `dev/check_dead_helpers.py`.

Four test files were union-merged with upstream by the same method — `test_utils.py`,
`test_trainer_utils.py`, `test_batch_quantized_cache.py`, `test_speculative.py` — which
is what retired most of those exclusions. The method, and the order of its steps, is
written up under "Per-file convergence".

Restored this pass, all with `git log -S` provenance first: `8e2638b7` (#1558),
`c1821e93` (#1329), `26220e71`, `9afc59ce`, `57dc1fb5` (#1450), `e3906673` (#1433,
kernel half only), `6a8cdff6` (#1411), `13d1ff4e`, `e7a0f0f0` + `21aeb8a5` +
`3de5bada`, `7fbc7bc9` (#1598) with `6d5603b3` (#1628) and `ea2edd68` (#1583) riding
on it, `36331ea7` + `67ca1f05` + `b739dfa4`, `e029e2b3`, `a30180a6`, `8422ece8`
(#1638), `ff295e36`, `f044f36a` (#1359). Converged to byte-identical:
`trainer/utils.py`, `tests/test_trainer_utils.py`, `models/gemma4/README.md`,
`generate/image.py`, `gemma4_assistant/masks.py` + its test.

**The through-line of this pass: five of the defects were "the symbol exists, and
nothing calls it."** Not missing code — missing *wiring*, with the helper present,
importable and usually unit-tested. That is why the suite stayed green through all of
them, and why `dev/check_dead_helpers.py` now gates it. Three had user-visible
consequences: embedding models silently ignored their `1_Pooling/config.json`, the
APC cache key ignored audio/video/embeddings/masks so requests differing only in
media **shared a prefix cache**, and three sampler modes were accepted by the API and
discarded.

> **[correction]** Commit `70c99d01`'s message, and an earlier version of this
> paragraph, claim `find_dropped_hunks.py` went "33 -> 18 commits". **That is
> wrong**, and the mistake is instructive: the number was read off a
> `find_dropped_hunks.py | tail -60` and only the visible tail was counted. The
> real figure is **34**, against 33 before the merge — essentially flat, and up by
> one because merging four upstream commits widens the scan's scope. The commit
> message cannot be fixed without rewriting a merge commit, so the correction lives
> here. **Count the report's own header line** (`N upstream commit(s) with content
> missing here`); never a truncated pipe.
>
> The related claim that the report "no longer reports `8422ece8`" is also wrong.
> It still does, at 49 lines — but now for `generate/dispatch.py` and
> `generate/ar.py`, not the `server/app.py` half that was restored. This is how the
> report behaves in general: it is *per commit*, aggregated over files, so
> restoring one file of a multi-file commit removes that file's line and leaves the
> commit listed. `a492e47d` and `221fe0b3` both still appear for the same reason
> after having a half fixed each. A commit leaving the report entirely is the
> exception, not the measure of progress.

Re-run the scanner after every merge:

```bash
python dev/find_dropped_hunks.py            # ranked, with owning commits
```

It is a lead generator, not an oracle — a commit whose content upstream later
replaced also surfaces, and heavy fork rewrites (`server/generation.py`) show up
as false positives. Every hit still needs `git log -S` and a read.

**[correction, 2026-08-09] The default `--min-lines 3` Python floor is itself a blind
spot, and it hid a live regression.** `f044f36a` (#1359) touches three files. The
report listed two — `generate/dispatch.py` (9 lines) and `server/openai.py` (3) — and
omitted its `server/generation.py` half entirely, because that hunk is 18 lines made
of 1-2 line changes and every individual change falls under the floor. The commit was
then "restored" from the report's own file list, leaving two hardcoded
`skip_special_tokens=True` where upstream reads `args.skip_special_tokens`.

Two habits follow, and they are cheap:

- **Re-run at `--min-lines 1` before calling the report clean.** The floor exists to
  suppress noise and succeeds at suppressing signal too. This is the Python-side twin
  of the `--min-lines-config 1` fix that recovered three dropped dependency commits.
- **Never take the report's per-commit file list as that commit's file set.** Use
  `git show --stat <commit>` and diff every file it touches. The report tells you
  *which commits to look at*, never *how much of one is missing*.

### Fixed already

`#1492`, `ff2a6daa`, `bc3461b1`, `#1503`, `46ee12dd`, `#1554` (MTP `+2.0` norm
shift), 4 `MODEL_REMAPPING` entries, DeepSeek V4's `swiglu_limit` clamp +
`zero_row_tail`, and the mlx-lm vendoring.

Closed in the 2026-08-08 second pass — items **2, 4, 5, 7, 10, 13** below plus the
multi-kind preload flags:

| Commit here | Restores | Was |
|---|---|---|
| `c8609a0c` | `ecc457b2` (#1374), now complete | MiniMax M3 VL unusable on every batch/server path; `--thinking-mode` unreachable |
| `b7ae124b` | `c6084344` | `--image-model` / `--tts-model` / `--stt-model` documented in README but rejected by argparse |
| `f5b74a9a` | `dab4cb45` (#1582) | `--quantized-kv-start` a dead parameter |
| `609bdf95` | `40757df3` | `/v1/embeddings` 404 with 246 lines of implementation in the tree |
| `92620167` | `#1447`, `#1453`, `#1432` | TurboQuant batch cache untrimmable — a live `IndexError`/`TypeError` |
| `510f16e9` | `960b26f9` (#1597), `53052569` | 1-bit backend unreachable; laguna markers duplicated |
| `a91f9b4d` | — (tooling) | stale `.symbol-exclusions` entries now warned about instead of rotting silently |
| `8839c2ff` | `4993eac1` (#1714) | inference routes unauthenticated while management was gated — and the README already claimed otherwise |
| `540a3189` | `b8671991`, `a8642018`, `#1527`, `#1741` | mRoPE `(3,B,L)` mangled on every batched path; live for four byte-identical-to-upstream models |
| `d684708e` | upstream's `test_models.py` | proven strict subset; +46 tests, 57 exclusions retired |
| `499fbc2a` | `e48ed11b` (#1454) | the opposite sign — a file upstream *deleted* that we kept |

Three things worth carrying forward from that pass:

- **Docs landing without their code is the most reliable tell in this repo.**
  Four of the six restores were findable that way, not from line counts: README
  documented `--image-model`/`--tts-model`/`--stt-model`, `--embedding-model`, a
  whole `/v1/embeddings` section, and (see #1) "Bearer token required for
  inference, model discovery, and management endpoints" — all while the code did
  none of it. Grep the README against argparse when triaging.
- **The stale-exclusion warning paid for itself immediately**, surfacing 5 dead
  entries after the TurboQuant restore and 1 after the laguna restore. It is a
  warning, not an error, so `upstream-parity.yml` does not start failing on the
  pre-existing backlog.
- **One deliberate non-restore.** Upstream's `BatchTurboQuantKVCache` has *two*
  `zero_row_tail` definitions (upstream:6193 from #1447, upstream:6234 from
  #1432) in the same class body, so the first is shadowed and unreachable. Ours
  has the surviving one. Do not "restore" the dead duplicate.

Three more from the second half of the pass, all new:

- **The audits have a blind side, not just blind spots.** Every check here —
  both audit scripts and `find_dropped_hunks.py` — asks "what does upstream have
  that we lack?". None asks the reverse. `mlx_vlm/video_generate.py` was 645 lines
  of *stale upstream code* we kept after upstream deleted it in `e48ed11b` (#1454);
  7 of that commit's 8 files had applied, so README/docs/examples had every
  reference stripped while the module and its `__main__.py` registration survived —
  a working, undocumented command. The check that settles it is
  `git diff <deleting-commit>^:<path> <path>`: empty output means our copy is
  upstream's last pre-deletion copy, i.e. STALE rather than FORK. Worth automating:
  for each file we have that `upstream/main` lacks, run
  `git log --oneline --all --diff-filter=D -- <path>` instead of assuming fork-only.

- **A test edited to tolerate its own bug.** Taking upstream's `test_models.py`
  showed our copy had *weakened* an assertion rather than deleted it:
  `assertEqual(shape, (3,1,1))` had become
  `assertIn(shape, {(1,1), (3,1,1)})` with a branch for each. That is the mRoPE bug
  being accepted as valid by the test written to catch it. AST-diffing bodies of
  shared functions — not just comparing names — is what surfaced it; the symbol
  audit sees `def` names only and was perfectly happy.

- **`app.routes` stopped meaning what it used to.** FastAPI >= 0.141 does not
  flatten `include_router()` into `app.routes`; it inserts one lazy
  `_IncludedRouter` node with `path=None`. `requirements.txt` pins
  `fastapi>=0.95.1`, unpinned, so we get 0.141.x. Routing is unaffected —
  introspection is. Any test asserting a path is in `app.routes` silently measures
  the wrong thing once those routes move onto a router, which is exactly what
  #1714 does. Prefer asserting a request does not 404.

### Ranked backlog — each item has a demonstrated failure

| # | Item | Owning commit | Demonstrated effect |
|---|---|---|---|
| 1 | ~~Inference routes unauthenticated~~ **FIXED `8839c2ff`** | `4993eac1` (#1714) | with `MLX_VLM_SERVER_API_KEY` set, `/v1/models` returns 200 and `/v1/chat/completions` reaches body validation without auth, while `/v1/cache/stats` correctly 401s. Upstream wraps them in `APIRouter(dependencies=[Depends(_require_management_api_key)])`; we register on `app` directly |
| 2 | ~~MiniMax M3 VL cannot run on any batch/server path~~ **FIXED `c8609a0c`** | `ecc457b2` (#1374) | `ar._make_cache(model, [0,1])` -> `ValueError: MiniMaxM3KVCache does not yet support batching`. 3-line `to_batch` guard at the top of `to_batch_cache` |
| 3 | ~~Streaming `/v1/responses` streams raw chain-of-thought as visible output~~ **FIXED `b6ed5d8d`** (`7c233155` half; `cfcc36d9` logging still open) | `7c233155`, `cfcc36d9` | `openai.py` streaming path is `delta = chunk.text`; zero `response.reasoning*` events. Non-streaming drops the item too: `_response_output_items_from_text` yields `['message']` where upstream yields `['reasoning','message']` |
| 4 | ~~`/v1/embeddings` 404s although the implementation ships~~ **FIXED `609bdf95`** | `40757df3` | `server/embeddings.py` + `models/pooling.py` byte-identical to upstream with **zero importers**; the `app.py`/`cli.py` wiring was dropped |
| 5 | ~~`--quantized-kv-start` silently ignored on the server~~ **FIXED `f5b74a9a`** | `dab4cb45` (#1582) | plumbed all the way to `ar.py`'s `self.quantized_kv_start = ...` and **read by nothing**. TurboQuant quantizes from token 0 regardless. A dead-parameter tell |
| 6 | ~~mRoPE cluster~~ **FIXED `540a3189`** | `a8642018`, `b8671991`, `#1527`, `#1741` | must land WITH `generate/ar.py`: our batcher lacks the MRoPE helpers, so `(3,B,L)` `position_ids` is truncated to `(1,B,L)`. Restoring `qwen3_5` alone *creates* a bug. **Live today for qwen2_vl / qwen2_5_vl / qwen3_vl / qwen3_vl_moe** — qwen3_5's staleness is what masks it |
| 7 | ~~TurboQuant trim cluster~~ **FIXED `92620167`** | `#1447`, `#1453`, `#1432` | `BatchTurboQuantKVCache.is_trimmable()` -> `False`, `trim(2)` -> `0` (upstream: `True`, `2`). Under `--kv-bits` the cache falls into the SSM branch and is indexed `c[0]`/`c[1]` -> `TypeError`; on the non-crashing path every later `gdn_states[j]` index shifts, restoring GatedDeltaNet state from the wrong layer. `zero_row_tail` exists with **no caller anywhere** |
| 8 | ~~`tool_choice` ignored on `/v1/chat/completions`~~ **FIXED `0bb0b07f`** | `2394fcf1` (#1611) | `"tools" in ChatRequest.model_fields` -> False; also lost the `if not tools: tool_module = None` guard, so tool markup is parsed for callers who sent no tools |
| 9 | `skip_special_tokens` — **RECLASSIFIED TWICE; the diffusion half is a REGRESSION, not a gap** | `f044f36a` (#1359) / upstream for the rest | Two separate things wear this name. **(a) The diffusion half is ours and is a regression.** `f044f36a` introduced `skip_special_tokens=args.skip_special_tokens` in `server/generation.py`; our copy hardcodes `True` in two spots. That commit was restored on 2026-08-09 — but only its `dispatch.py` and `openai.py` halves, because `find_dropped_hunks.py`'s default `--min-lines 3` Python floor hid an 18-line hunk made of 1-2 line changes. Fixable directly; **not** blocked behind `29b6c00b` as an earlier note claimed. **(b) The normal text path is an upstream bug**: upstream honours the flag only inside `_generate_diffusion` and its `anthropic.py:489` is byte-identical to ours, so `gen_args.skip_special_tokens = False` is inert in both trees. Report upstream; do not invent it here |
| 10 | ~~1-bit repos cannot load~~ **FIXED `510f16e9`** | `960b26f9` (#1597) | kernel, `__init__` export and 123-line test all landed; only `utils.py`'s 7 lines of `replace_one_bit_modules` wiring dropped |
| 11 | ~~AWQ quantization absent~~ **FIXED `1d6e8c9b`** | `3c0232ed` (#1666) | 104 lines of `convert.py`, 91% of the hunk |
| 12 | ~~Empty-input rejection absent~~ **FIXED `4d91b912`** | `a344713a` (#1491) | empty requests reach model load instead of a 400 |
| 13 | ~~`should_add_special_tokens` missing~~ **FIXED `510f16e9`** | `53052569` | see the shadowed-test note below; gemma3/3n/4/4_unified/laguna get markers added outside the template |
| 14 | ~~`load_image(PIL.Image)` raises~~ **FIXED `4d91b912`** | `84025353` | `ValueError: Unsupported image source type: Image` |
| 15 | ~~`request_normalization.py` orphaned~~ **FIXED as a union** | `221fe0b3` (#1644) | module had zero importers; `app.py` now delegates to it, exactly as upstream's `app.py` does. Landed as a **union**: the three fork behaviours in `app.py:_build_gen_args` (Ollama `repeat_penalty` alias, registry `generation_defaults` overlay, resolved-sampling log) moved into the module rather than being dropped, and the four byte-identical helper duplicates were deleted (retiring 3 `.deletion-exclusions` entries). Newly live: `reasoning`/`reasoning_effort` now drive thinking mode, and the 13 diffusion pass-throughs reach `GenerationArguments`. Guarded in `tests/test_dropped_upstream_guards.py::TestGenArgsUnion`, each fork hunk verified to fail its own guard when removed |
| 16 | ~~Three sampler modes unreachable~~ **FIXED `7ebb5690`** | `36331ea7`, `#1653`, `#1663` | reachable end to end now: schema -> `_build_gen_args` -> `to_generate_kwargs` -> `make_sampler` |
| 17 | ~~`test_models.py` wholesale take~~ **FIXED `d684708e`** | 24 commits | proven strict subset (AST: 52 methods only upstream, **0** only ours); +46 tests, retires all 57 `test_models.py` exclusions. 3 fail until item 6 lands |

Plus: generation-logging subsystem (`cfcc36d9`), ~~concurrent thinking-budget
race (`ff295e36`)~~ **FIXED**, ~~`reasoning_content` alias (`6a8cdff6`)~~
**FIXED**, multi-kind preload flags (`c6084344`),
`/v1/responses/input_tokens` counting a different prompt than `/v1/responses`
builds (`eda1ec4f`, half-landed).

`ff295e36` was a live data race, not a missing feature.
`_make_thinking_budget_criteria` tokenizes, and it was being called on the **GPU
worker thread** while request threads tokenized concurrently in
`_preprocess_request` — two threads through the same mutable fast-tokenizer
backend. The fix moves it to the caller side under a new `_tokenizer_lock` and
carries the result on `QueuedGenerationRequest`. Upstream's own test makes the race
visible (four threads, `assert max_active == 1`); neutering the lock turns it into
`assert 4 == 1`.

### `8422ece8` (#1638) — the APC adapter refactor, and two silent bugs in it

Closed 2026-08-09. This one is worth its own entry because it is the best example of
why "the symbol exists" proves nothing, and because both bugs it was hiding were
invisible to the whole suite.

The commit replaced `dispatch.py`'s inline APC lookup with a pluggable
`_apc.apc_lookup_plan(...)`. Our tree had `apc.py`'s half — `apc_lookup_plan`,
`semantic_extra_hash`, `snapshot_prompt_cache_row`, all **byte-identical to
upstream** and all unit-tested — and none of the three call sites. `0f359e5` had even
recorded "`apc.py` fully converged with upstream", which was true of the file and
false of the feature. The three helpers are what motivated
`dev/check_dead_helpers.py`; it found the third one, which no gap list had.

Restoring the call sites (113 inline lines in `dispatch.py` became an 8-line call)
fixed two defects, neither with a failing test:

- **The APC cache key ignored non-image media.** Ours computed
  `tenant_scoped_hash(tenant, image_hash)`; `semantic_extra_hash` also folds in
  audio, video, `inputs_embeds` and masks. So two requests differing *only* in audio
  hashed identically and **shared a prefix cache**, reusing KV computed for other
  media. `0f359e5`'s note that "`semantic_extra_hash` reduces exactly to
  `tenant_scoped_hash` with no extras" is correct — and was read as "so the call site
  does not matter", when the extras are the entire point.
- **APC stored a live, growing cache.** `_apc_prompt_cache_for_store` returned
  `self.prompt_cache` **by reference** for single-row batches. The APC store then
  held the cache generation was still mutating, so a checkpoint keyed at N tokens
  kept growing past N. `snapshot_prompt_cache_row` clones, and also dequantizes
  quantized layers into float `KVCache` entries, which our path never did.

Its last piece, `apc.py::_merge_arrays_cache_entries`, was **deleted** rather than
kept: zero references anywhere, absent from upstream entirely, and byte-identical to
upstream's last pre-deletion copy — STALE by the conclusive test, its only caller
having gone with the refactor.

### Per-file convergence: which files can be retired, and which must not be

The cheapest permanent win here is not restoring hunks, it is **shrinking the
shared-file surface**. Every fork hunk in a file upstream also edits is a future
conflict and a future silent-drop site. When a file's delta is no longer worth
carrying, `git checkout upstream/main -- <path>` makes it byte-identical and
retires it as a drop site forever. `convert.py` (`1d6e8c9b`) and `structured.py`
(`b616f889`) both reached that state, and three more did on 2026-08-09:
`generate/image.py` (its entire delta was one blank line plus a 20-line function
relocation), and the `gemma4_assistant/masks.py` + `test_gemma4_assistant_masks_static.py`
pair described under "The deliberate divergence that wasn't".

The safe test is **not** the line count. It is: *does our copy define anything
upstream does not?* Run this before considering any convergence:

```python
# for each diverged .py file, AST-compare def/class names both ways
only_ours = ours_names - upstream_names      # fork-only definitions
only_upstream = upstream_names - ours_names  # dropped or relocated
```

Current standing (2026-08-09): **12 diverged files have zero fork-only
definitions**, and 22 carry some.

**But zero fork-only definitions does NOT mean convergeable**, and this is the
important half. Ten of those twelve also have *zero* missing upstream definitions
— they diverge only in **bodies**, and a body can be load-bearing fork work. Two
of the ten are divergences this very document tells you never to touch:

- `models/base.py` — its `cache.quantized_attention` call is chunked with online
  softmax so peak memory is ~O(chunk); upstream dequantizes the whole state.
- ~~`speculative/drafters/gemma4_assistant/masks.py`~~ — **RETRACTED.** The absolute
  import was not load-bearing; it was half of dropped `a492e47d`. Both that file and
  its test are byte-identical to upstream now. See "The deliberate divergence that
  wasn't" below — this bullet contradicted it for a while.

So the symbol check narrows the candidate list; it does not authorise the
checkout. Body diffs still need reading.

The two candidates with genuinely missing upstream symbols were both classified:

- `generate/dispatch.py` (10) — **not convergeable, and nothing is lost.** Seven
  symbols were *relocated* by the fork's own `5e9b9503` generation-engine port and
  now live in `generate/common.py` (verified present, and `GenerationResult` /
  `wired_limit` are still re-exported from `dispatch`). The other two,
  `_cache_fully_retained` and `_prefix_cache_trim_amount`, are upstream's
  rotating-cache guard, deliberately superseded by the fork's
  `_rotating_rewind_safe` + snapshot ring. All ten now carry real reasons in
  `.symbol-exclusions` instead of "unreviewed".
- `tests/test_utils.py` (4) — **closed 2026-08-09.** Zero fork-only definitions, but
  *not* a clean checkout: our copy had fork content in **bodies** (mocking a
  `safetensors.safe_open` metadata check that exists in neither tree any more).
  Union-merged instead — upstream's file plus one genuine fork adaptation. The 5
  "pre-existing failures" that got the file excluded from the suite were stale test
  code, not product bugs: it is now **34 passed / 0 failed and collected**.

**Where the remaining 57 exclusions sit.** Down from 107. The two test files that
held 23 of them are **converged**: `tests/test_batch_quantized_cache.py` (12) and
`tests/test_speculative.py` (11) were union-merged with upstream on 2026-08-09 and
both left `.fork-marker-allowlist` as well. What is left is concentrated in
`tests/test_server.py` (31, gated on `cfcc36d9` and `29b6c00b`),
`tests/test_generate.py` (22) and `server/generation.py` (11). All five carry substantial fork-only definitions (124, 11, 17, 23 and 14
respectively), so none is a checkout candidate; they need per-symbol review. (Was
107; the 2026-08-09 merge retired the 8 `dispatch.py` entries when #1807 deleted
upstream's own duplicates of the symbols the fork had relocated.)

**The fork-content oracle — built 2026-08-09, `dev/check_fork_markers.py`.**
Nothing used to tell you which hunks in a shared file are fork work; `cache.py`'s
`# Fork:` convention was the right answer and existed in exactly one file. The
check now enforces it, and "which side is this?" is a grep.

Three design decisions, each of which was forced by something that went wrong:

- **The unit is the enclosing top-level definition, not the hunk.** Calibrating
  against `cache.py` — the one file already following the convention — showed a
  per-hunk rule *fails the file that defines the convention*: it diverges in 7
  hunks but carries 3 markers, because the `prealloc_tokens` feature touches a
  signature, a constructor body and a growth calculation and one marker on the body
  explains all three. A rule that fails its own exemplar is the wrong rule.
- **Raw `-U0` hunk counts are meaningless for a rewritten file.** `apc.py` reports
  **361** of them inside just **4** default-context regions, because diff finds
  many small common substrings in a rewrite. The report counts *sites* instead.
  Tree-wide that is **312 sites**, not the ~658 hunks previously estimated.
- **A site is any top-level statement, not just a `def`/`class`.** Adding one name
  to a parenthesized `from x import (...)` block forced this: isort hoists any
  standalone comment in such a block up to the `import (` header, so a marker
  *cannot* be placed adjacent to the added name. Multi-line statements therefore
  count as enclosing spans; single-line ones still need the marker on the line.
- **The allowlist is consulted after coverage is computed, not before.** Checking
  it first looks equivalent and is not: an entry naming an already-marked file
  would be silently accepted as if it were doing work, so the stale-entry warning
  could never fire for it. Caught by adding an entry for `models/base.py` (already
  marked) and watching it pass unremarked.

**The trap this check sets, and it must be stated wherever the check is
mentioned.** A site that differs from upstream is *not* automatically fork work —
it can equally be content a merge resolution dropped. Marking such a site
`# Fork:` launders a bug into a documented feature that no later audit will ever
question. So a marker is a *provenance conclusion*, and requires the same
`git log -S` / `git merge-base --is-ancestor` work AGENTS.md demands. The
allowlist tags files with known `find_dropped_hunks.py` leads accordingly.

That trap is not hypothetical; building the check found an instance of it already
recorded as fact. See "the deliberate divergence that wasn't" below.

Standing at introduction: **7 of 32 files** under the convention (3 converged to
byte-identical, 4 marked), **25 files / 334 sites** allowlisted with reasons.

### The union cluster — item 15 landed 2026-08-09; 9 and `cfcc36d9` remain

**Status.** The `_build_gen_args` union is **done** (item 15). `app.py` now
delegates to `request_normalization`, in the same shape upstream's `app.py` uses,
with all three fork behaviours carried into the module and the four byte-identical
helper duplicates deleted. That made `reasoning` / `reasoning_effort` and the 13
diffusion pass-throughs live for the first time. Still open in this cluster:

- **item 9** — `skip_special_tokens`, now **reclassified**. Investigated after the
  union and it is mostly *not* a fork gap: upstream honours the flag only inside
  `_generate_diffusion`, and upstream's `anthropic.py:489` is byte-identical to
  ours, so the write is ignored on the normal text path in both trees. The
  diffusion half is ours and is blocked behind `29b6c00b`, because our two
  hardcoded sites sit in the stale `_generate_masked_diffusion`. See the table row.
- **`cfcc36d9`** (#1634, generation logging) — a 9-file, 716-line dropped commit,
  surfaced independently by the 2026-08-09 merge when `server/__init__.py`
  conflicted over importing `get_log_progress_interval`, which exists nowhere in
  this tree. Seven `.symbol-exclusions` entries marked "unreviewed" are its
  content. Restore whole or not at all.

The rest of this section is the pre-landing analysis, kept because the reasoning is
the reusable part.

Items **9**, **15**, and the generation-logging (`cfcc36d9`) and
diffusion-unification (`29b6c00b` / #1508) commits all converge on the same
gen-args path in `server/app.py` and `server/generation.py` — the file AGENTS.md
names as the false-positive-prone fork rewrite.

`GenerationArguments` itself is done: `7ebb5690` took upstream's class after
proving it a strict superset (39 fields vs 20, **0 fields only ours**, shared
methods differing only by our removals). So the *fields* for
`reasoning`, `reasoning_effort`, `skip_special_tokens` and the 13 diffusion
knobs now exist. What is not done is populating them, and that is where the shape
changes.

`app.py:_build_gen_args` and `request_normalization._build_gen_args` have
**diverged in both directions**:

| only in `app.py` (fork) | only in `request_normalization.py` (upstream) |
|---|---|
| `repeat_penalty` alias fallback for `repetition_penalty` | `_standard_reasoning_control` → three-way `enable_thinking` resolution |
| `get_server_generation_defaults()` applied to fields absent from `model_fields_set` | the three sampler pass-throughs (now also in ours) |
| the `resolved sampling: ...` info log | all 13 diffusion pass-throughs |

So the obvious reading of item 15 — "wire the module, delete `app.py`'s
duplicates" — **would silently drop three pieces of fork behaviour**. It needed a
hand-merged union of the two functions, then the delegation, which is what landed.

The other four helpers (`_as_plain_dict`, `_request_field_or_default`,
`_model_config_field_or_default`, `_extract_response_format_schema`) were confirmed
**byte-identical** between the two files by AST comparison — and, usefully, the same
comparison showed `_build_structured_logits_processors` was **not** identical, so it
became a delegating wrapper rather than a deletion. The four are gone now; their
three `.deletion-exclusions` entries retired with them.

Two things worth keeping from doing it:

- **Upstream had already solved the structure.** Its `app.py` carries exactly the
  thin delegating wrappers this needed, *including* the `_server_package_attr`
  indirection for the logits factory that looked like fork work. Reading upstream's
  target shape before hand-merging turned a rewrite into a convergence. Check for
  that before designing a resolution.
- **Removing the moved code made ten of `app.py`'s imports unused**, and autoflake
  wanted them gone. Each had to be checked for reachability through
  `mlx_vlm.server.<name>` (the package mirrors `app.py`'s namespace) and for
  monkeypatching in tests before removing. They were safe, and dropping them moved
  `app.py`'s import list toward upstream's — but this is exactly the hook AGENTS.md
  warns has teeth.

Also note item 15's original description was stale by the time it was fixed: the
module no longer "raises on first call", because the `top_n_sigma` it passes exists
as of `7ebb5690`. That was the entire reason it was marked a landmine.

### The deliberate divergence that wasn't — `gemma4_assistant/masks.py`

AGENTS.md carried this as a rule for a while:

> `gemma4_assistant/masks.py` must keep its absolute import. It looks like
> gratuitous divergence from upstream's relative form, but that module is imported
> standalone by its test, where a relative import has no parent package. Taking
> upstream's broke two tests.

Every clause of that is an accurate observation. The conclusion is still wrong, and
the shape of the error is worth more than the fix.

`a492e47d` (#1494) is a merged-then-dropped commit that changed **two** files as one
atomic edit:

- `masks.py`: `from mlx_lm.models.cache import dynamic_roll` -> `from ....models.cache import dynamic_roll`
- `tests/test_gemma4_assistant_masks_static.py`: fake the `mlx_vlm.*` package chain
  and load the module under its real dotted name, so the *relative* import resolves

Only the first half was dropped. So our tree had a test still faking
`mlx_lm.models.cache` — inert, because nothing imports that any more and the real
`mlx_vlm.models.cache` loads regardless — and an absolute import in `masks.py` that
was invented as a workaround for the missing half. Someone then tried upstream's
`masks.py` *alone*, correctly observed two tests break, and codified the workaround
as a deliberate divergence. Taking **both** halves makes both files byte-identical
to upstream with the two tests passing.

Three things generalise:

- This is AGENTS.md's own "a dropped commit usually spans several files, and
  restoring one file of a multi-file dropped commit is worse than restoring none"
  rule, encountered from the other end: *testing* one file of a multi-file dropped
  commit and concluding from the failure that our side is load-bearing.
- "Taking upstream's version broke a test" is not evidence that our version is
  correct. It is evidence that the two sides are internally consistent and you are
  holding half of each.
- The note that got this wrong is the second AGENTS.md "do not touch this"
  guardrail found to be protecting a non-divergence, after the `qwen3_5` norm-shift
  gate. Both deterred inspection of exactly the file that needed it. A guardrail
  needs a `git log -S` behind it, and the ones written without that are liabilities
  precisely because they read as settled.

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
`tests/test_utils.py` **was** excluded from the suite entirely
(`--ignore=tests/test_utils.py`, 5 pre-existing failures), so any guard placed there
never ran. Resolved 2026-08-09: the failures were stale test code, the file is now
green and collected. The habit still stands — check what is collected, not just what
passes — but this particular hole is closed.

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

  **[correction, 2026-08-09] "Unreferenced" is not the same as "dead", and one of
  those two removals proves it.** `DEFAULT_DIFFUSION_CONFIDENCE_THRESHOLD` was
  unreferenced in `generate/dispatch.py` *because* `f044f36a` (#1359) had been
  dropped — that commit is the only thing that uses it, in the `--threshold` help
  string. Deleting the import tidied away the last trace of a missing hunk, and
  `find_dropped_hunks.py` lost a line of evidence with it. Restoring #1359 put the
  import straight back.

  So before accepting an autoflake removal in a file that diverges from upstream,
  check whether **upstream** references the name. If it does, the import is not
  dead — it is the shadow of a dropped call site, and the right response is to
  restore the hunk, not delete the import. This is the same reasoning
  `dev/check_dead_helpers.py` automates one level up, for definitions rather than
  imports.
- `black` warns "Python 3.12 cannot parse code formatted for Python 3.14" and
  skips its AST safety check. It still exits 0, so this does not fail
  `pre-commit`, and formatting is derived from file content rather than the
  running interpreter — but CI runs Python 3.10, so that path is verified only
  by CI actually going green, not from here.

## Not a source of known issues

`docs/report_issues.md` is four lines of upstream boilerplate pointing at
Blaizzy's issue tracker. It contains no fork-specific content and is
intentionally left untouched to avoid merge friction.
