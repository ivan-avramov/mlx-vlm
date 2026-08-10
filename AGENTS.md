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

**Merge weekly. Never let upstream accumulate.** This is a rule, not a
preference. Every gap catalogued in `docs/upstream-gaps.md` — 72 commits' worth of
silently dropped content — came out of resolving large accumulations, because:

- A 90-commit merge produces a resolution nobody can review hunk by hunk, and an
  unreviewable resolution is where content gets dropped without a conflict.
- The six audits below only work **if they run**. They are cheap after a
  five-commit merge and psychologically skippable after a ninety-commit one.
- A dropped hunk is unrecoverable by any later merge (see the failure mode
  below), so the cost of a bad resolution is permanent, while the cost of merging
  often is a few minutes a week.

If upstream has moved and it has been a week, merge — even if the diff looks
boring, and even mid-task. `git log HEAD..upstream/main` being empty is the
steady state to maintain, not a coincidence to notice later.

```bash
git fetch upstream
git merge upstream/main
python dev/check_upstream_parity.py     # no upstream file silently dropped
python dev/check_upstream_symbols.py    # no upstream def/class silently dropped
python dev/find_dropped_hunks.py        # no upstream HUNK silently dropped
python dev/check_upstream_deletions.py  # no upstream DELETION silently reverted
python dev/check_fork_markers.py        # every fork hunk in a shared file is marked
python dev/check_dead_helpers.py        # no upstream-called helper left unreachable
cd mlx_vlm/ && pytest ./tests --ignore=tests/test_smoke.py
```

**Run all six checks after every merge — this is not optional.** When a merge
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

### The third direction: which side wrote this hunk?

Both questions above are about *presence*. Neither can tell you, for a file both
trees have, **which side authored a given change** — the question every conflict
resolution actually turns on. `dev/check_fork_markers.py` is that check: for each
file `upstream/main` also has, every diverging top-level `def`/`class` must carry a
`# Fork:` comment saying what deviates and why, or sit below a fork-boundary banner
(`models/cache.py` is the model). Files not yet annotated live in
`.fork-marker-allowlist`, which is a **rollout schedule, not an exemption list** —
each entry means "not yet reviewed", and removing one is the definition of done.
`--summary` ranks the remaining work by file.

**A marker is a provenance conclusion, not a label.** A site that differs from
upstream is *not* automatically fork work — it can equally be content a merge
resolution dropped, and marking that `# Fork:` launders a bug into a documented
feature that no later audit will ever question. Establish provenance the usual way
first. Allowlist entries tagged `LEADS` have open `find_dropped_hunks.py`
candidates; resolve those before marking anything in the file.

The reverse of the AGENTS.md corollary above therefore also holds: **do not mark a
hunk `# Fork:` without running `git log -S` on it either.** Building this check
found `gemma4_assistant/masks.py` — recorded in this very file as a deliberate
divergence — to be half of a dropped commit; see the `[correction]` under
"Deliberate divergences".

**Before working an allowlist entry, run the AST body diff in "the fifth direction"
below.** A file's site and hunk counts measure alignment, so a *reordered* file looks
like the biggest job on the list and is the smallest. Marking is the fallback; the
better exit is usually CONVERGE.

**A pure-deletion hunk cannot carry a marker, so `.symbol-exclusions` covers it
(case 4).** When upstream defines a symbol we legitimately do not have, the diff
attributes that deletion to whatever line of ours sits at the seam — the blank line
after the preceding definition, which no span reaches — so there is no enclosing
definition to annotate and no whitespace to converge. `generate/dispatch.py` and
`server/generation.py` were both stuck this way with every real site marked. The
check now treats such a hunk as covered when **every** `def`/`class` name it removes
is already excused in `.symbol-exclusions` for that path, and prints the names it
covered on each run so the coverage stays auditable. Two things to know:

- **Such a hunk looks exactly like a whitespace probe and is not one.** Run
  `git diff -U0` on it before concluding either way — a real probe is fixable by
  converging the ordering, this is not.
- **The rule matches methods too, and that is the point.** "Every name excused" is
  the safety property; the indent is not. Anchoring at column 0 was an over-narrow
  first attempt that left `server/generation.py` stuck on a deleted *method*
  (`_sample_top_p_one`) whose class is present and marked.

`mlx_vlm/tests/test_fork_marker_check.py` pins that narrowness, and is the only test
of any `dev/` audit. Worth extending rather than replacing: a bug that makes one of
these checks **more permissive** fails nothing, still prints OK, and silently stops
reporting dropped content.

### The fourth direction: is the thing that exists actually reachable?

The three questions above are all about *code*. This one is about *wiring*, and it
turned out to be the dominant residual failure mode. Call it **helper landed, call
site dropped**:

    upstream adds a helper AND its call site in one commit -> our merge applies the
    helper's file but drops the call site's hunk -> the symbol exists, imports fine,
    is often unit-tested, and is reachable from nothing

Every other check is blind to it *by construction*: the file is present (parity),
the `def` is present (symbols), nothing was deleted (deletions), and the missing
hunk lives in a different file that is usually already allowlisted (fork markers).
The tests keep passing, because the helper's own unit tests still exercise it
directly. That is what makes this shape so durable.

`dev/check_dead_helpers.py` is that check, and the instances it exists for were all
real defects with no failing test:

- `apc.apc_disk_namespace` — the on-disk APC namespace stopped fingerprinting the
  KV-quant config, so two runs with different `--kv-bits` shared a prefix cache.
- `embedding_loader.load_embedding_model` + `models.pooling.read_pooling_config` —
  every embedding model silently used default pooling and ignored its
  `1_Pooling/config.json`. Wrong embeddings, no error.
- `apc.self_check_model_apc` — the APC layout dry-run never ran.
- `apc.apc_lookup_plan`, `semantic_extra_hash`, `snapshot_prompt_cache_row` — still
  open; `8422ece8`'s `dispatch.py` refactor. The last of these was found *by the
  check*, not by hand, and was in no prior gap list.

It compares against upstream rather than flagging every unused helper: a fork-only
helper nothing calls yet is our business, one upstream *does* call is a dropped
hunk. `.dead-helper-exclusions` distinguishes **REVIEWED** (intentionally
unreachable here, say what you established) from **CONFIRMED-DROPPED** (a real
dropped call site, restore tracked and pending — a debt marker, not approval).

**"A dead parameter or an unreachable helper is a tell."** That phrase already
appears in `docs/upstream-gaps.md` next to several fixed items; this check turns it
from a habit into a gate. When something looks unused, ask whether upstream calls
it before concluding it is fork scaffolding.

**A log line is the same shape with no handle at all.** It has no caller, no symbol
and no test, so nothing above can see it — and a commit whose content is mostly
logging is therefore the hardest kind to restore completely. `cfcc36d9` (#1634) took
**three** passes: declared whole once and 14 lines short, finished against the wide
pass, and still missing two hunks a month later. What found them was neither audit
nor report but counting the commit's own distinctive message literals in both trees:

```bash
git show upstream/main:<path> | grep -c 'Generation cancelled: request=%s'
grep -c 'Generation cancelled: request=%s' <path>
```

Do that for every logging commit. Counting the *helpers* is not enough — a guard in
`test_dropped_upstream_guards.py` was already comparing `self._log_*(` call counts
against upstream and passed throughout, because both survivors were direct
`logger.info(...)` calls in otherwise byte-identical blocks.

**Corollary for `.symbol-exclusions`: "baseline: pre-existing divergence,
unreviewed" is a to-do, and reviewing one is often what settles a whole file.** The
fifth direction below surfaces those entries as its `up-only` column, so the two
techniques compose: AST-diff the bodies, then resolve whatever `up-only` reports.
Both of `server/generation.py`'s unreviewed entries turned out to be genuine,
tested, deliberate divergences (`_check_configured_context_budget` replaced by the
fork's clamp-instead-of-reject budget; `_sample_top_p_one` superseded by a `_filter`
that also applies min_p and top_k) — but neither said so, so neither could be
trusted, and the file could not be signed off while they read "unreviewed".

### The fifth direction: is this divergence CONTENT, or only ALIGNMENT?

The four questions above are all about *what exists where*. None of them, and none of
the five gating scripts, can tell you the one thing you need before sizing a diverged
file: **how much of it actually differs.** A file whose definitions have merely been
*reordered* reports as maximally diverged by every line-based measure, and there is no
check that says otherwise.

**Do this before reading a diff, before sizing a file, before scheduling the work.** It
is ~25 lines and takes a second:

```python
# .venv/bin/python - <<'PY'    (set P)
import subprocess, ast, difflib
R = "/Users/ia87221/ws/mlx-vlm"; P = "mlx_vlm/server/openai.py"
up = subprocess.run(["git","-C",R,"show",f"upstream/main:{P}"],
                    capture_output=True, text=True).stdout
ours = open(f"{R}/{P}").read()
def parse(src):
    t = ast.parse(src); L = src.splitlines(); d = {}; mods = []
    for n in t.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            s = min([n.lineno] + [x.lineno for x in n.decorator_list])
            d[n.name] = "\n".join(L[s-1:n.end_lineno])
        else:
            mods.append("\n".join(L[n.lineno-1:n.end_lineno]))
    return d, mods
U, UM = parse(up); O, OM = parse(ours)
sh = sorted(set(U) & set(O))
print(f"shared={len(sh)} identical={sum(U[n]==O[n] for n in sh)} "
      f"ours-only={sorted(set(O)-set(U))} up-only={sorted(set(U)-set(O))}")
print(f"module stmts: up={len(UM)} ours={len(OM)} identical={UM==OM}")
for n in sh:
    if U[n] == O[n]: continue
    ch = [x for x in difflib.unified_diff(U[n].splitlines(), O[n].splitlines(),
          lineterm="", n=0) if x[:1] in "+-" and not x.startswith(("---","+++"))]
    print(f"  DIFF {n:48s} up={len(U[n].splitlines()):4d} "
          f"ours={len(O[n].splitlines()):4d} changed={len(ch)}")
PY
```

Three results so far, and the first one is why this section exists:

- **`apc.py`** — the largest `.fork-marker-allowlist` entry for weeks (36 sites, 75
  hunks, 554 missing upstream lines), described in this file's own notes and in the
  handoff as "a rewrite rather than a set of hunks". **76 shared definitions, 74 with
  byte-identical bodies, 23/23 module statements identical.** The entire delta was four
  block moves left over from the fork's own cluster-by-cluster port. Real fork content:
  two fork-only helpers and two call sites.
- **`turboquant.py`** — 96 shared, **91 identical**. Reduced "118 missing lines, mostly
  fork rewrite" to five functions, one of which held a dropped hunk that three prior
  passes had missed because all three read `find_dropped_hunks.py` instead of diffing.
- **`tests/test_generate.py`** — 49 shared, **47 identical**, and the residual was
  method ordering. Converged 5 sites to 0.

**The exit this unlocks is the one `.fork-marker-allowlist` calls CONVERGE**, and it is
`docs/handoff-2026-08-10.md` §2f's test-file method applied to library files: take
upstream's file as the base, re-apply the fork content, mark it. `apc.py` went from 554
missing lines to 2. Verify the rebuild lost nothing by AST-diffing **before against
after** — same definition count, zero lost, zero new, module statements identical, only
the intended texts changed. For a *test* file also compare `pytest --collect-only`
counts on both sides; that is the only thing that catches a shadowed definition.

Two caveats, both paid for:

- **Comments are not AST nodes.** A node-based reorder will land a relocated definition
  on the wrong side of a `# ===` banner. Check banner positions by eye afterwards.
- **`ours >= upstream` occurrence counts mean "rewrite"; `ours < upstream` is the only
  signal worth chasing.** That is the cheaper identifier-count technique in the handoff,
  and it is strictly weaker than this one: it answers "is this symbol present in both?",
  not "is this body upstream's?". Use it on whatever the AST diff flags, not instead.

**This is also the general form of the rule below.** "Never conclude from a
`--numstat` line count" was being obeyed to the letter while the same mistake was made
with "missing lines", "sites" and "hunks" — every wrong claim in the handoff's history
has been a number or a framing derived from a line count. Line counts measure
alignment. AST-diffing bodies measures content.

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

**`--numstat` is only the most obvious form.** "Missing lines", "sites" and "hunks"
are the same measure wearing different hats, and reading them as a proxy for *how
much differs* is how `apc.py` was mis-sized by two orders of magnitude — see "the
fifth direction" above, and run that AST comparison before you size anything. A
fourth situation this rule cannot distinguish, added because it cost a cycle: **a
file whose definitions were merely reordered.**

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
cd mlx_vlm/ && pytest -s ./tests --ignore=tests/test_smoke.py
```

The suite is **green: 2685 passed, 5 skipped, 0 failed.** Keep it that way. (This
line goes stale on every restore that adds a guard — trust the run, not the number.)

**Compare failing test IDs, not counts.** A change that fixes one test and breaks
another shows the same total:

```bash
pytest -q ./tests --ignore=tests/test_smoke.py \
  | grep '^FAILED' | sed 's/ - .*//' | sort > /tmp/after.txt
comm -13 /tmp/before.txt /tmp/after.txt   # regressions
comm -23 /tmp/before.txt /tmp/after.txt   # fixed
```

**A green suite only proves that *collected* tests pass.** Check
`pytest --collect-only` counts too. Two live examples:

- **[resolved 2026-08-09]** `tests/test_utils.py` used to be **excluded from the
  suite** (`--ignore`, 5 pre-existing failures), so a regression test placed there
  never ran. It is now collected and green. The 5 failures were **stale test code,
  not product bugs**: our copy had been adapted around a `safetensors.safe_open`
  metadata check that no longer exists in either tree, and upstream's copy passes
  against our library. Taking upstream's file plus one genuine fork adaptation (our
  `load_processor` reads the config first, so `load_config` must be patched too)
  gives 34 passed / 0 failed. Note CI's `tests.yml` never excluded this file — only
  `test_smoke.py` — so CI would have been red on any PR.
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

- **[correction]** This list used to open with "`gemma4_assistant/masks.py` must
  keep its absolute import ... taking upstream's broke two tests." **It was not a
  divergence at all.** `a492e47d` (#1494) changed `masks.py` to a relative import
  *and* rewrote its test's module fakes to match, as one atomic edit; only the
  first half survived our merge. So the test still faked `mlx_lm.models.cache`
  (inert) and the absolute import was a workaround for the missing half. Taking
  **both** halves makes both files byte-identical to upstream with the tests
  passing, which is now the state of the tree. "Taking upstream's version broke a
  test" means the two sides are internally consistent and you are holding half of
  each — it is not evidence that our side is load-bearing. See
  `docs/upstream-gaps.md`, "The deliberate divergence that wasn't". This is the
  second guardrail in this file found to protect a non-divergence, after the
  `qwen3_5` norm-shift gate below; **do not add one without `git log -S` behind
  it.**
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
  (excluding `test_smoke.py` only — this file previously claimed it also excluded
  `test_utils.py`; it never did, so CI would have been red on any PR while that
  file had failures). Because it is PR-only, pushes straight to `main` are never
  style-checked — which is how style drift went unnoticed.
- `upstream-parity.yml` — runs on pushes to `main` as well as PRs, since this fork
  is usually committed to directly. Runs the five gating audit scripts.

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
