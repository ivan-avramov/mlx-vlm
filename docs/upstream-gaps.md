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

Full suite, `cd mlx_vlm/ && pytest ./tests --ignore=tests/test_smoke.py
--ignore=tests/test_utils.py`:

| Stage | Result |
|---|---|
| Before the merge | 36 failed, 1805 passed, 3 skipped |
| After the merge | 46 failed, 1833 passed (10 regressions, all from the `prompt_has_open_thinking` break) |
| After the restore | 76 failed, 1924 passed (44 pre-existing gaps became *visible*) |
| **Now** | **16 failed, 2175 passed, 5 skipped** |

Read the middle rows carefully: the rise to 76 was not a regression. Restoring
the 16 dropped test files exposed 44 gaps that had been invisible because the
tests for them were missing. Everything from there down was closing real gaps.

**Every remaining failure is in `test_diffusion_gemma.py` (11) or
`test_diffusion_models.py` (5). No other test file has a failing test.**

Skipped files: 2, both the deliberate quant-SDPA divergence described above
(down from 6).

Compare failing *test IDs*, not counts, when validating a change here — a
same-count swap is otherwise invisible. `comm -13`/`comm -23` over the sorted
`FAILED ...` lines is enough.

## The recurring failure mode, in full

Every gap closed in this fork traced to one of these shapes. Worth knowing the
list, because two of them are invisible to `dev/check_upstream_parity.py` and
`dev/check_upstream_symbols.py`:

| Shape | Example | Caught by tooling? |
|---|---|---|
| Whole file dropped | 16 upstream test files | yes (parity) |
| Symbol dropped | `dynamic_roll`, `layer_kv_for_apc` | yes (symbols) |
| Definition dropped, call sites merged | `prompt_has_open_thinking` → every chat request 500'd | yes (symbols) |
| Call site dropped, definition merged | `_image_model_type_from_component_indexes`, `layer_kv_for_apc` | **no** |
| Module constant dropped | `_COMPRESSED_TENSORS_DROP_SUFFIXES`, `IMAGE_COMPONENT_INDEX_DOWNLOAD_PATTERNS` | **no** (not a def/class) |
| Single guard/line dropped | `alias_model_class.supports_model(model)` — broke every quantized flux2 repo | **no** |
| Registry entry dropped | `"kimi_k3"`, `"mage"`, `minimax_m3_vl` | **no** |
| Duplicate authority kept instead of delegating | 5 cache-classification ladders in `apc.py`, 117 lines | **no** |
| Stale *fork* version kept over upstream's rewrite | `lfm2_vl` projector tests asserted the inverse of our own code | **no** |

The lesson for future merges: the audits catch structural loss (files, symbols).
They cannot catch a dropped line, a dropped dict entry, or a kept-but-stale test.
Only running the suite and comparing test IDs finds those.

## Still open

**The diffusion subsystem — needs a design decision, not a port.**

16 failures, all in the two diffusion test files, both of which are
**byte-identical to upstream**, so they test upstream's design.

`mlx_vlm/generate/diffusion.py` has diverged in *both* directions: 188 insertions
and 332 deletions against upstream. Upstream has 9 symbols this fork lacks (the
model-owned generator path: `_stream_model_diffusion_generate`,
`_uses_model_diffusion_generator`, `_normalize_decoder_input_ids`,
`_diffusion_initial_canvas`, …). This fork has 7 upstream lacks
(`_diffusion_soft_embeddings`, `_diffusion_entropy_probs_chain`,
`_diffusion_entropy_and_soft_embeddings`, `_diffusion_soft_embedding_weight`,
`_diffusion_prefill_cache`, `_is_diffusion_config`, `is_masked_diffusion_model`),
and `is_masked_diffusion_model` is wired into `generate/dispatch.py`.

Taking upstream's file wholesale would delete working fork functionality. The
open question is whether this fork's soft-embedding/entropy path should sit on
top of upstream's model-owned generator or is superseded by it. That is an
architecture call about intent, not a mechanical merge, so it was deliberately
left alone rather than guessed at.

Also open, low priority: `_rotating_post_gen_trim_safe` remains intentionally
unwired (see above) — that is correct, not a gap.

## Not a source of known issues

`docs/report_issues.md` is four lines of upstream boilerplate pointing at
Blaizzy's issue tracker. It contains no fork-specific content and is
intentionally left untouched to avoid merge friction.
