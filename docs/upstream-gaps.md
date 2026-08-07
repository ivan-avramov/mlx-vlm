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

`.symbol-exclusions` currently holds a **446-entry baseline** captured against
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

Six of them import symbols this fork never ported, so collection is skipped for
those six in `mlx_vlm/tests/conftest.py` — in one reviewable place, with a
reason each, rather than by editing the upstream files. Porting the symbol alone
would have made the tests pass without testing anything real:

| Skipped file | Needs | Why not just port it |
|---|---|---|
| `test_apc_semantic_key.py` | `apc._hash_payload` | fork hashes via `_hash_tokens`/`_hash_use_sha256`; key derivation differs by design |
| `test_apc_observability.py` | `apc.APCSelfCheckResult` | upstream's APC self-check subsystem, not ported |
| `test_apc_quantized.py` | `cache.should_quantize_kv_layer` | upstream needs `_make_cache`, stream quantize **and** APC warm restore to share this policy; the fork decides inline, so a lone helper would test nothing |
| `test_quant_sdpa_mask.py` | `cache.dynamic_roll` | upstream vendors its cache classes; the fork delegates base classes to `mlx_lm` |
| `test_quant_sdpa_mask_adversarial.py` | `cache.dynamic_roll` | as above |
| `test_minimax_m3.py` | `base.align_attention_mask_to_scores` | mask/score alignment helper, not ported |

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

- **DeepSeek V4 HISA config.** `index_block: int = 64` / `index_keep: int = 16`
  restored to `models/deepseek_v4/config.py`. Wiring is still absent — see below.

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

**Known remaining gap:** `_is_rotating_kv_layer` — used by snapshot
capture/restore, *not* by the reuse guard any more — still matches exact names,
so buffered layers are skipped by `_capture_rotating_layers_for_snapshot`.
Fixing that also needs `start_position` plumbed through `RotatingKVSnapshot` and
`restore_rotating`, which is a separate change and was deliberately not rushed
in alongside a correctness fix.

## Test-suite state

Full suite, `cd mlx_vlm/ && pytest ./tests --ignore=tests/test_smoke.py
--ignore=tests/test_utils.py`:

| Stage | Result |
|---|---|
| Before the merge | 36 failed, 1805 passed, 3 skipped |
| After the merge | 46 failed, 1833 passed (10 regressions, all from the `prompt_has_open_thinking` break) |
| After the fixes + restore | **76 failed, 1924 passed, 4 skipped** |

The rise from 46 to 76 is **not** a regression. 13 failures were fixed (the 10
merge regressions plus 3 `minicpmv4_6` tests unblocked by the import fix), and
44 pre-existing gaps became *visible* for the first time because the test files
that exercise them had been missing. That was the point of restoring them.

Those 44 group into a handful of causes, all unported upstream APC-adapter work:

| Cause | Failures |
|---|---|
| `mlx_vlm.apc.snapshot_prompt_cache_row` absent | ~10 |
| `mlx_vlm.apc.layer_kv_for_apc` absent | ~3 |
| `.extract()` missing on `BatchQuantizedKVCache` / `BatchTurboQuantKVCache` | ~6 |
| remaining APC adapter/storage/registry divergence, `mage_flow`, `qwen3_5_mtp_sanitize` | rest |

Restored files are otherwise fully passing: 10 of the 16 collect and run, and
`test_inkling.py`, `test_one_bit.py`, `test_minimax_m3.py`,
`test_paddleocr_vl_vision.py` and `test_mage_vl_positions.py` contributed no
failures at all.

## Still open

Priority order. Everything here was re-verified against the current tree.

1. **DeepSeek V4 HISA wiring** — config fields now exist and
   `hisa_kernel.py` already matches upstream byte-for-byte, but
   `models/deepseek_v4/language.py` still never references `index_block` /
   `index_keep`, so the attention path does not call into the kernel. The three
   `TestDeepseekV4HISA` failures moved on from
   `TypeError: unexpected keyword argument 'index_block'` to
   `AttributeError: 'Indexer' object has no attribute '_hisa_select'`, which
   names the exact missing piece. Smallest remaining item.
2. **APC adapter surface** — the 44 newly-visible failures above. Largest
   remaining block; needs `snapshot_prompt_cache_row`, `layer_kv_for_apc` and
   the batch-cache `.extract()` methods, or a decision that this fork's APC
   design diverges permanently (in which case the tests should move to
   `conftest.py`'s skip manifest with that reason).
3. **`_is_rotating_kv_layer` / snapshot `start_position`** — see above.
4. **`/responses` validates input after loading the model.** `server/openai.py`'s
   `responses_endpoint` calls `get_cached_model(...)` near the top, before
   converting/validating `input`. Upstream normalises input →
   `_response_items_to_chat` → `_normalize_response_instruction_messages` →
   *then* loads. Effect: a bad `file_id` 500s instead of 400ing, and
   developer-role messages are never merged with top-level `instructions`
   (`_normalize_response_instruction_messages` is still absent here).
5. **`ThinkingAwareLogitsProcessor` structured-decode delay** — fork-specific
   bug, predates all of this. `_make_logits_processors` does not insert the
   thinking-aware processor when `enable_thinking=True` and a grammar processor
   is also requested.
6. **AWQ fold tolerance / FP8+NVFP4 compressed-tensors / lfm2_vl optional
   projector layernorm** — unported upstream features, no known correctness
   impact. Port opportunistically.
7. **kimi_k3 chat-template asset mismatch** — narrow, single-model.

## Not a source of known issues

`docs/report_issues.md` is four lines of upstream boilerplate pointing at
Blaizzy's issue tracker. It contains no fork-specific content and is
intentionally left untouched to avoid merge friction.
