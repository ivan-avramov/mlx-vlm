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
| `test_quant_sdpa_mask.py` | `base.quantized_scaled_dot_product_attention` | upstream's quantized attention reshapes scores to 5D for `mx.quantized_matmul`, which is what makes right-aligning a 4D mask alias `B` with `n_kv_heads` (upstream #1567). This fork dequantizes and runs dense attention (`models/base.py:251`), so 5D scores never exist and the hazard is structurally impossible. Porting the helper would be dead code. |
| `test_quant_sdpa_mask_adversarial.py` | as above | as above |

Deleting an entry from `UNPORTED_UPSTREAM_TESTS` is the definition of done for
porting that feature.

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

Eight of eleven are invisible to the audits. They catch structural loss only.

## Still open

**Upstream #1492's video-input support.** `prompt_utils.py` is −66 lines against
upstream, and those 66 are one coherent feature: `_get_video_token()`, video
handling in `_flatten_content`, `_template_references_kw`, video-message routing.
It pairs with `server/schemas.py`'s missing `VideoUrl`,
`ResponseInputVideoParam`, `ResponseVideoUrlParam`, `ResponseVideoParam`.

The full feature spans `prompt_utils.py`, `schemas.py`, `server/openai.py`,
`server/generation.py`, `server/app.py`. Three carry heavy fork work, so this is
an interleaved port, not a `git checkout`. Start from
`git show --stat 4d468e85` and reconcile each file — per lesson 2 above, do not
restore them one at a time.

Also open, low priority: `_rotating_post_gen_trim_safe` stays intentionally
unwired (see above) — correct, not a gap.

## Not a source of known issues

`docs/report_issues.md` is four lines of upstream boilerplate pointing at
Blaizzy's issue tracker. It contains no fork-specific content and is
intentionally left untouched to avoid merge friction.
