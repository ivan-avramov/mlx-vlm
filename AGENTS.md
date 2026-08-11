# AGENTS.md

Guidance for coding agents working in this repository.

MLX-VLM is a Python package for inference and fine-tuning of Vision Language
Models and Omni (audio/video) models on Apple Silicon using MLX. 50+ model
architectures via a plugin-based model system.

## Starting a session: verify the tree before believing anything

Takes ~7 minutes (the suite is ~4 of them) and it has repeatedly been worth it. Every
number in this file goes stale eventually; the tree does not lie.

```bash
cd /Users/ia87221/ws/mlx-vlm
git status --short                                           # expect clean
git rev-parse origin/main                                    # expect == HEAD
git fetch upstream && git log --oneline HEAD..upstream/main   # MUST be empty; merge if not

(cd mlx_vlm && ../.venv/bin/python -m pytest -q ./tests --ignore=tests/test_smoke.py)

.venv/bin/python dev/check_upstream_parity.py
.venv/bin/python dev/check_upstream_symbols.py
.venv/bin/python dev/check_upstream_deletions.py     # ~70s
.venv/bin/python dev/check_dead_helpers.py
.venv/bin/python dev/check_fork_markers.py
.venv/bin/python dev/check_body_divergence.py        # ~3s
.venv/bin/python dev/check_upstream_registries.py    # ~13s
.venv/bin/python dev/check_call_arguments.py         # ~30s

grep -cE '^mlx_vlm' .symbol-exclusions .deletion-exclusions .fork-marker-allowlist \
    .dead-helper-exclusions .body-divergence-exclusions .registry-exclusions \
    .call-argument-exclusions
# expect 14, 0, 0, 0, 0, 5, 2 — see "Exclusion baselines" below
grep -cE '^mlx_vlm.*# baseline: pre-existing divergence, unreviewed' .symbol-exclusions
# expect 0 — every remaining exclusion carries a real reason
```

The pytest `cd` **must** be in a subshell as written. After any bare `cd mlx_vlm`,
`.venv/bin/python` is *not found* — come back to the repo root.

### Exclusion baselines

Eight gating audits, each with its own reviewed baseline. **These are the progress
signal, not any report's count.**

| file | baseline | meaning |
|---|---|---|
| `.symbol-exclusions` | **14** | upstream symbols we deliberately lack; zero unreviewed |
| `.deletion-exclusions` | **0** | upstream deletions we deliberately kept |
| `.fork-marker-allowlist` | **0** | files not yet under the `# Fork:` convention |
| `.dead-helper-exclusions` | **0** | upstream-called helpers unreachable here |
| `.body-divergence-exclusions` | **0** | misalignments claimed deliberate |
| `.registry-exclusions` | **5** | re-exports the fork replaced, all reviewed |
| `.call-argument-exclusions` | **2** | calls passing fewer args, reviewed (one `[**]`, one `[arity]`) |

An empty file is **not** a reason to delete it — each keeps its header and RESOLVED
notes, and the next merge that adds a fork hunk to a shared file will need an entry.

Two non-gating lead generators, which are read rather than gated on:
`find_dropped_hunks.py` (**18** commits, all closed by review — see `upstream-gaps.md`,
"the dropped-hunk report has bottomed out"; that count will never reach zero) and
`find_untested_fork_code.py` (**10**, all confirmed executed by a coverage run).

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
- The eight audits below only work **if they run**. They are cheap after a
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
python dev/find_dropped_hunks.py --min-lines 1 --min-share 0.05 \
        --max-commits 400              # no upstream HUNK silently dropped
python dev/check_upstream_deletions.py  # no upstream DELETION silently reverted
python dev/check_fork_markers.py        # every fork hunk in a shared file is marked
python dev/check_dead_helpers.py        # no upstream-called helper left unreachable
python dev/check_body_divergence.py     # no divergence that is only ALIGNMENT
python dev/check_upstream_registries.py # no registry entry / re-export / field dropped
python dev/check_call_arguments.py      # no call passing fewer args than upstream
python dev/check_body_divergence.py --summary   # then size what conflicted
python dev/check_body_divergence.py --sweep     # then re-review the markers
python dev/find_untested_fork_code.py   # fork-only code no test mentions (non-gating)
cd mlx_vlm/ && pytest ./tests --ignore=tests/test_smoke.py
```

**`find_dropped_hunks.py` has THREE floors and the bare invocation is decoration.**
`--min-lines 3`, `--min-share 0.5` and `--max-commits 80` are all defaults, and each
hides real content: `f044f36a` was hidden by the *share* floor (its `generation.py`
share is 45%), which nobody expected. The default pass reports ~12 commits; the wide
pass above reported 43. **Run the wide one, always**, and treat a hit as a lead:
`docs/upstream-gaps.md` has a section listing every commit that permanently appears in
it together with why each is closed — check that before investigating anything.

Two further limits, both hit for real. The report **cannot see deletions**, so a
commit that mostly removes code (`29b6c00b`) shows only its insertions and
`check_upstream_deletions.py` is what sees the rest. And a commit can be **fully
closed while still listed**, because attribution is by line content — the exclusion
baselines, not the commit count, are the progress signal.

**Run all ten checks after every merge — this is not optional.** Eight of them gate
(they exit non-zero and CI runs them); `find_dropped_hunks.py` and
`find_untested_fork_code.py` are lead generators and do not. When a merge
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

- **[resolved 2026-08-09 by `08723a3f`; kept because the SHAPE is the lesson]**
  `server/app.py` used to define `_as_plain_dict`, `_request_field_or_default` and
  `_model_config_field_or_default` a second time, duplicating
  `server/request_normalization.py`, which is where upstream moved them (#1644).
  Upstream has only the latter, and carrying both is *why* that module sat
  orphaned. `app.py` now imports it (`app.py:29`) and delegates
  (`app.py:60,173,227,239`), guarded by
  `test_dropped_upstream_guards.py::test_app_delegates_to_the_request_normalization_module`.
  Do not "converge" the `_as_plain_dict` pair that remains in
  `request_normalization.py` + `responses_state.py`: **upstream defines it in both
  of those files too**, so that duplication is parity, not drift.
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

**Before working an allowlist entry, run `dev/check_body_divergence.py --file <path>`
("the fifth direction" below).** A file's site and hunk counts measure alignment, so a *reordered* file looks
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

`mlx_vlm/tests/test_fork_marker_check.py` pins that narrowness. It and
`test_body_divergence_check.py` are the only tests of any `dev/` audit; every new
gating script needs one, because a bug that makes one of these checks **more
permissive** fails nothing, still prints OK, and silently stops reporting dropped
content. Extend them rather than replacing them.

**A marker is unverified prose, and five of them have been false.** This is the
convention's structural weakness and worth stating plainly: `check_fork_markers.py`
proves a `# Fork:` comment is **present**; nothing proves it is **true**. A false
marker is worse than a missing one, because a missing one reports and a false one
makes the site invisible to every later audit. The five:

- three `# Fork: placement only ... Nothing to converge` claims on definitions that
  were byte-identical to upstream and merely *out of order* (`3105b598`);
- `maybe_quantize_kv_cache`'s claim that the fork "additionally skips the last layer"
  and "honours the TurboQuant split key/value bit widths" — upstream's own hybrid and
  TurboQuant branches do both, with upstream's own comment (`0670f556`). That one
  overstated fork ownership of shared code, which is the shape that *deters*
  inspection; the same mistake around the `qwen3_5` norm-shift gate hid a live `+2.0`
  double-shift bug.
- `dispatch.py::stream_generate`'s claim that *"everything else upstream does here is
  still done ... all appear here at >= upstream's count"*, while the diffusion dispatch
  right below it had dropped upstream's `skip_special_tokens=` and `verbose=`
  (`eb044b7d`). See "the eighth direction".

**A marker's CLAIM SHAPE is itself a risk signal, and this is the cheap check.** The
three shapes rank by how much damage a false one does:

    per-site   "this line differs because X"        -> falsifiable against `gone`
    scoped     "these N sites are the only ones"    -> falsifiable by counting them
    blanket    "everything else is still done"      -> NOT falsifiable; assume false

A blanket claim is unverifiable by construction — it asserts something about the lines
it does *not* name — so it launders every unexamined site in the definition into
"reviewed". `stream_generate`'s was the only one in the tree, found by grepping for the
phrasing rather than by reading 61 markers:

```bash
git grep -n "everything else\|>= upstream\|upstream's count\|all present\|still done" \
    -- 'mlx_vlm/**.py' | grep -v '^mlx_vlm/tests/'
```

Distinguish a **completeness** claim ("nothing else is missing") from a **provenance**
claim ("everything else in this class is upstream's"). The latter is fine and two
markers legitimately make it. **Never write a blanket completeness claim: per-site or
nothing.** And note what made this one especially bad — it invoked the
identifier-occurrence count, which this file already calls "strictly weaker" than
diffing bodies, and that count *would* have caught the bug (4 occurrences here against
upstream's 5) had it been applied to the dropped name instead of a hand-picked subset.
A weak technique aimed at the wrong symbol reads exactly like a strong one.

The check for it is `dev/check_body_divergence.py --file <path>`'s **`gone`**
column, which lists the upstream lines missing from our whole copy of the file. A
marker saying the fork ADDED something that appears in that list is false — we did not
add what upstream already has. Use `gone`, not a plain diff: a rewritten body reports
every reordered line as changed, so a real omission hides in the noise. Then
`git log -S` the distinctive lines, because `gone > 0` means *either* a fork
replacement *or* content a resolution dropped, and nothing distinguishes those
mechanically.

Note what `gone = 0` does and does not prove. It means every upstream line is
*somewhere* in our file, so a marker claiming upstream LACKS something cannot be
supported by absence — but a marker can still be false the other way, by claiming the
fork authored something both trees have. That is exactly `0670f556`, whose
`# Skip the last layer` comment was present in *upstream's* body and absent from
neither. Only reading the two bodies settles that.

**So: review a marker whenever you touch its file, and never write one from the
diff alone.**

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

**A restored upstream test that covers the call site which SURVIVED is worse than no
test — it makes the commit look closed.** `eda1ec4f` (#1716) changed the same line in
two endpoints; one landed, one did not, and both of the commit's own tests landed and
passed because both exercise the endpoint that got the fix. The result was
`/v1/responses/input_tokens` reporting the token count of a differently-shaped prompt
than `/v1/responses` actually builds — the one thing that endpoint exists not to do.
So when a commit touches N call sites, **count them** rather than trusting its tests:

```bash
git grep -c '<helper>(' upstream/main -- '*.py'   # then the same here
```

**[correction 2026-08-10] That grep counts CALLS, and a call site can be dropped
without changing the count.** It would NOT have caught `29b6c00b`'s second half: both
trees call `stream_diffusion_generate_from_kwargs` exactly twice, and what our copy
dropped was two *keyword arguments* at one of them, making `--skip-special-tokens` a
no-op on every diffusion model. Run the grep to find the sites, then
`dev/check_call_arguments.py` for what each site passes — see "the eighth direction".
The two questions are *is it called* and *is it called correctly*, and this grep only
answers the first.

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

**[update 2026-08-10] The fifth direction now covers this, for the common case.**
`check_body_divergence.py --file`'s `gone` list is line-content based, so a dropped log
line *inside a shared definition* appears in it verbatim. Verified by deleting exactly
`cfcc36d9`'s cancellation log line from `server/generation.py`: `ResponseGenerator`
went gone=16 -> 18 and the report named
`"Generation cancelled: request=%s generated_tokens=%d",` outright, while parity,
symbols, deletions, fork-markers and registries all stayed green. So "nothing above can
see it" was true when written and is now false — run `--sweep` after a logging commit
and the literals show up on their own.

Two limits keep the hand-count worth knowing. `gone` only sees content inside a
definition **both trees define**, so a log line at module scope or in a fork-only
function is still invisible to it; and it is per-file, so a line the fork moved to
another module reads as present. Counting the *helpers* is not enough either — a guard in
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

The four questions above are all about *what exists where*. None of them, and no
line-based measure, can tell you the one thing you need before sizing a diverged file:
**how much of it actually differs.** A file whose definitions have merely been
*reordered* reports as maximally diverged by `--numstat`, by "missing lines", by
"sites" and by "hunks" alike — those are all the same measure wearing different hats.

`dev/check_body_divergence.py` is that check. It was prose plus a copy-paste snippet
in this file until 2026-08-10, which meant it only ran when someone remembered.

**Run `--summary` before reading a diff, before sizing a file, before scheduling the
work.** It takes ~3s over the whole tree:

```bash
python dev/check_body_divergence.py --summary        # rank FILES by CONTENT
python dev/check_body_divergence.py --sweep          # rank DEFINITIONS by `gone`
python dev/check_body_divergence.py --file <path>    # per-definition report
python dev/check_body_divergence.py                  # the gate
```

`--summary` prints each file's `content` score next to the same file's `--numstat`
delta, on purpose: `tests/test_server.py` reads +2185/-262 and scores 33;
`turboquant.py` reads +539/-102 and scores 9. The line column is there to be
distrusted. A file with `content=0` is a pure reordering — converge it rather than
reading its diff.

`--file` also prints, per differing definition, two counts: **`absent`** (upstream
lines missing from our *body*) and **`gone`** (missing from the whole *file*), and
lists the `gone` ones. **Read `gone`.** `absent` alone cannot tell *moved* from
*lost* — `turboquant.py`'s rewritten kernel reports absent=66 / gone=2, because the
fork kept upstream's body as an ours-only `_legacy` sibling, so reading the first
number as content loss overstates it 33x. That is AGENTS.md's central rule one level
down: a measure that conflates two situations needing opposite responses.
`gone` is the column a `# Fork:` marker's claim gets checked against — see "a marker
is unverified prose" above; it is what caught `0670f556`.

**`--sweep` ranks every content-differing definition in the tree by `gone`.** That is
the marker-review worklist: 61 definitions differ, 22 are strict supersets (pure fork
addition, nothing upstream lost) and 39 have upstream lines we do not have. It is
regenerated on demand rather than stored, deliberately — a per-definition ledger of
reviewed entries would be ~50 rows nobody had looked at, which is the
"baseline: pre-existing divergence, unreviewed" anti-pattern `.symbol-exclusions` took
months to drain. The conclusion belongs in the marker or a `docs/` note, not in a
list.

It **gates only on alignment masquerading as content**, which is a much narrower
claim than the report makes: a whole file whose divergence is zero content, a
definition whose bodies differ only in blank lines or trailing whitespace, and a
definition byte-identical to upstream's sitting outside upstream's order. All three
have mechanical fixes, so `.body-divergence-exclusions` should stay empty. It
deliberately does **not** demand a per-definition review ledger — `check_fork_markers.py`
already covers each diverging site, and a second list would just add ~50 unreviewed
entries.

Three results from the snippet era, and the first one is why this section exists:

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
the union method below, applied to a library file: take
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

The first caveat was observed live by the script itself. `tests/test_server.py` had
`# ── Continuous batching / ResponseGenerator tests ──` sitting above a metrics test
instead of above `class TestResponseGenerator` where upstream has it — a previous
node-based move had landed on the wrong side of it. After acting on a RELOCATED
finding, check banners by eye.

**Two shapes to expect when reading `gone`, both from real passes:**

- **A textual `gone` cannot see a BEHAVIOURAL superset.**
  `server/app.py::_count_thinking_tag_tokens` reads gone=7 and its "strict superset"
  marker holds — upstream's accumulator (`count = 4` / `elif … count = 2` /
  `return count`) versus our early returns is the same function. Read the bodies before
  concluding a marker is false.
- **A big `gone` is not a big problem.** `tests/test_generate.py::TestPrefixCacheReuseTrim`
  reads gone=60, the largest single number in the sweep, and every line is the body of
  one of 9 already-reviewed `.symbol-exclusions` methods.

**A third caveat, and the reason this check gates rather than merely reports:
`# Fork: placement only` is a lie waiting to be written.** All three markers in this
tree that said it were false — two claimed "byte-identical body ... Nothing to
converge" about definitions that were simply in the wrong order, and one blamed a
"line number shift" for an ordering swap. `git log -S` showed only upstream's own
commits behind every side of both pairs. A marker is what makes a site invisible to
every later audit, so *placement* is exactly the wrong thing to excuse with one; the
gate now refuses it. See `3105b598`.

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

### The sixth direction: is it even a DEFINITION?

Every check above keys on a `def`, a `class`, a file or a hunk. Nothing keyed on the
other half of the language. A dropped dict entry, a dropped `__init__.py` re-export and
a dropped dataclass field are `ast.Assign` / `ast.AnnAssign` / `ast.Import*`, and they
pass all five: the file is present (parity), no `def`/`class` name vanished (symbols),
nothing was deleted (deletions), the hunk sits in a file whose fork sites are marked
(fork markers), and the entry is not a helper anyone calls (dead helpers).

`dev/check_upstream_registries.py` is that check, and **the gap it closes was
measured, not assumed.** Deleting `models/gemma4/__init__.py`'s
`Gemma4VideoProcessor` re-export — a real historical loss in this fork — left **the
entire suite passing (2817 tests at the time) and all five other audits green.**
Only this one reports it.

Four shapes, each of which has cost a real loss here:

- **registry entry** — a key or element missing from a module-level container both
  trees assign to the same name. Four `MODEL_REMAPPING` entries went this way, and
  "unlimited-ocr" / "inkling_mm_model" failed with *"Model type X not supported"*
  while their implementations sat in the tree byte-identical to upstream.
- **registry** — a whole module-level container upstream assigns and we do not. A
  fork *rename* reports as this, which is correct: a rename is exactly the event that
  loses entries in a merge.
- **re-export** — a name an upstream import binds that ours does not. A package
  `__init__.py` is nothing but this.
- **class attribute** — a class-level assignment we lack. Dataclass fields are the
  case that matters: every `models/*/config.py` `ModelConfig` is one, and a dropped
  field silently changes model behaviour rather than raising.

**It compares PRESENCE only, never values.** `{"a": 1}` vs `{"a": 2}` passes, and that
is deliberate — values diverge legitimately across this fork (every tuned constant)
while keys almost never do. A wrong value is `check_body_divergence.py --file`'s job;
that is the shape of #1402's test-certified cache-key bug.

Reviewed hits go in `.registry-exclusions`, currently **5**, all re-exports the fork
replaced (`generation_stream` ×3 for the lazy per-thread stream,
`_check_configured_context_budget` for the clamp-instead-of-reject budget,
`top_p_sampling` for the `_filter` that also applies min_p and top_k). Runs in ~13s.

**Corollary, and it is why this section exists at all: a docstring claiming coverage
is not coverage.** `check_upstream_symbols.py` listed
`deepseek_v4/config.py -- ModelConfig.index_block / .index_keep` among its "verified
instances" for months. It cannot see either — they are `AnnAssign` nodes and that
script collects only `def`/`class` names. The fields are present today, so they were
restored, but not by the check that claimed them, and the claim made a whole category
look covered while nothing covered it.

### The seventh direction: who protects the FORK's own code?

Every direction above compares against `upstream/main`. That means **fork-only code is
invisible to all eight gating checks by construction** — no upstream copy to diff, no
upstream symbol to miss, no upstream hunk to drop, no marker to demand. The whole
apparatus protects the upstream content this fork carries. Nothing protected the fork's
own ~800 commits.

`dev/find_untested_fork_code.py` is the lead generator for that, and it earned its place
on the first run: the fork's legacy `/v1/completions` endpoint — `completions_endpoint`
(409 lines), five schemas, four helpers, with both `/completions` and `/v1/completions`
as live registered routes — had **zero test references.** Writing the tests it lacked
found a real user-visible bug immediately (a stop sequence split across token boundaries
had its prefix streamed to the client; `b75c18b7`).

```bash
python dev/find_untested_fork_code.py          # the untested ones, ~19s
python dev/find_untested_fork_code.py --all    # every fork-only definition + counts
```

**It does not gate, and should not.** Two reasons, the first being the same argument
that kept `.body-divergence-exclusions` from becoming a ledger: an exclusions file would
need ~18 written reasons for things that are mostly fine (a three-line predicate
exercised through its caller does not need an entry), which produces ~18 unreviewed
ones. And unlike every other check here, a hit is not a correctness claim — a dropped
upstream hunk is *wrong*, untested fork code is *risk*, and risk gets ranked and worked
down rather than gated.

**The method that worked, and it is the point.** Read the code for its contract, write
the tests that pin it, and let the failures find the bug. Of five items worked this way,
two were bugs (`/v1/completions`' partial stop sequence, gpt-oss's mangled first word)
and three were not — and *saying so* is the deliverable, not a reason to manufacture a
fix. But: **do not write tests that assert current behaviour without deciding it is
correct first.** That is how the suite came to certify #1402's cache-key bug.

**Read the columns asymmetrically.** It counts textual references, not coverage. A zero
in `tests` is a real signal worth acting on; a non-zero is **not** evidence of coverage
(it may be an unrelated mention), and a zero may still be exercised through a caller.

**`dev/fork_coverage_report.py` is the confirmation step**, and it is what settles those
dismissals rather than arguing them:

```bash
uv pip install --python .venv/bin/python coverage
cd mlx_vlm && ../.venv/bin/python -m coverage run \
    --source=/Users/ia87221/ws/mlx-vlm/mlx_vlm --data-file=/tmp/cov.data \
    -m pytest -q ./tests --ignore=tests/test_smoke.py
.venv/bin/python dev/fork_coverage_report.py --data-file=/tmp/cov.data [--all]
```

Per-definition, not per-file: `server/openai.py` is mostly upstream code and the fork's
dozen definitions vanish in its total. First run: **95 definitions, 1467 statements,
92.2% covered, ZERO entirely uncovered** — every one of the 12 "no test mentions"
dismissals confirmed. The two partial ones worth closing were `_trim_cache` (30/58, the
whole `[B, L, H, D]` layout path) and `/v1/completions`' ResponseGenerator backend.

**Measure the file alone, not just the suite.** `completions_endpoint`'s second backend
read as *covered* in the whole-suite run and *missed* when only its own test file ran —
another module had left `runtime.response_generator` set, so the branch was executed by
accident and asserted by nobody. A whole-suite number flatters an individual file; when a
definition matters, re-run coverage over its own test file.

### The eighth direction: is the call passing everything upstream passes?

Every direction above asks whether something **exists** — a file, a `def`, a deletion,
a hunk, a registry entry, a body, a caller. None can see a call that exists, is
reached, and passes **fewer keyword arguments** than upstream's:

    upstream adds a parameter AND passes it at N call sites -> our merge applies the
    callee's file byte-identically and applies M < N of the call sites -> the
    parameter exists, is keyword-only with a default, is documented, is unit-tested
    through the callee, and is silently never supplied

`dev/check_call_arguments.py` is that check. **`check_dead_helpers.py` is the near
miss, and understanding why matters:** it asks whether an upstream-called helper is
*reachable* here. The founding instance was reachable — it had a caller. *Called* and
*called correctly* are different questions, and only this script asks the second.

The instance it was written for (2026-08-10, found during a cold audit):
`generate/dispatch.py::stream_generate` called
`stream_diffusion_generate_from_kwargs(...)` without upstream's
`skip_special_tokens=` and `verbose=`. Both keyword-only with `= False` defaults, and
`skip_special_tokens` is *popped* out of `kwargs` first, so `**kwargs` did not rescue
it either. `--skip-special-tokens` was therefore a **no-op on every diffusion model** —
it reaches the `decode_generated()` that decodes every streamed token batch. All seven
other audits green, 3001 tests green, `.fork-marker-allowlist` empty, and the site
carried a `# Fork:` marker asserting *"everything else upstream does here is still
done"*. It was the second half of `29b6c00b` (#1508), whose *other* call site had been
restored a session earlier — the `eda1ec4f` shape: **when a commit touches N call
sites, count them.**

```bash
python dev/check_call_arguments.py                # the gate, ~30s
python dev/check_call_arguments.py --file <path>  # one file, verbose
python dev/check_call_arguments.py --summary      # rank files by hits
```

Three design rules, each paid for by a false positive in an earlier draft:

- **Names, never values** — the same rule as `check_upstream_registries.py`.
  `make_cache=_make_cache` vs `make_cache=functools.partial(_make_cache, ...)` is a
  fork enhancement. A wrong *value* is `check_body_divergence.py --file`'s job.
- **Read it asymmetrically**, like `find_untested_fork_code.py`'s `tests` column:
  `ours < upstream` is the only direction worth chasing. `ours > upstream` is fork
  work and stays silent, or `_make_cache`'s prealloc kwargs would report forever.
- **Two measures, and the second is weaker on purpose.** A keyword name is
  unambiguous, but a purely positional call has no name to be missing — `f(a, b, c)`
  losing `c` is invisible to it. So each pair also compares **total supplied argument
  count** (positional excluding `*x`, plus distinct keyword names), tagged `[arity]`.
  *Total*, not positional: moving an argument between positional and keyword form
  changes nothing and the fork does it freely. Measured both ways before choosing.
  Two rules keep it honest — the **deferred-call idiom is skipped** (when our call
  passes a `lambda` the arguments live inside its body, which is what
  `asyncio.to_thread(lambda: gen(a, b, c))` does in `server/openai.py`), and arity is
  reported **only when the keyword check found nothing** for that pair, so one defect
  is not counted twice. An `[arity]` hit is a **pointer to a diff, not a conclusion**:
  being a count it cannot say which argument went.

**`[**]` means our call forwards `**something`, so the name may still arrive through
the dict.** Those are tagged and still gate rather than being filtered — the founding
bug is *adjacent* to that shape (its callee takes a `kwargs` dict positionally), and a
check that goes quiet where the answer is hard is the failure mode `dev/` exists to
prevent. The single baseline entry is one of these: `server/cli.py::main` really does
supply `level=` and `format=` via `basicConfig(**log_kwargs)`. Confirm which dict, and
say so in the reason.

It covers module-scope calls too, filed under `<module>` — a registration or
middleware call (`app.add_middleware(...)`) is exactly the shape that loses a keyword
in a merge, and `check_body_divergence.py`'s `gone` cannot see module scope either, so
nothing else would cover it. Added after measuring that it reports zero hits tree-wide,
so the coverage was free. Scoped to files that differ from upstream — a byte-identical
file cannot have this defect — which is what keeps it at ~30s instead of >5min.
`mlx_vlm/tests/test_call_argument_check.py` pins it, per the rule that **every new
gating script needs one**: a bug making one of these checks *more permissive* fails
nothing and still prints OK.

### Adding a ninth direction — what the eighth cost, as a checklist

There will be a ninth: each of the eight was found by hand first, and the pattern is
always "a category of thing no existing check keys on". What the eighth needed, in the
order that mattered:

1. **A real instance first.** Do not build a check for a hypothesis. Every one of these
   eight exists because a concrete bug was found by hand and the question "what would
   have caught this?" had no answer.
2. **Name which existing check is the NEAR MISS, and why it misses.** For the eighth it
   was `check_dead_helpers.py`, and the distinction ("called" vs "called correctly") is
   what defines the new check's scope. If no check is close, the direction is probably
   too broad to gate.
3. **Compare names/presence, not values.** Every check here that stayed useful compares
   presence; values diverge legitimately across this fork. `check_upstream_registries.py`
   states this outright and the eighth copied it.
4. **Decide the asymmetry.** `ours < upstream` is a defect, `ours > upstream` is fork
   work. A check that reports both is a check that gets switched off.
5. **Scope to diverged files.** A byte-identical file cannot carry the defect, and this
   is usually a 10x speed difference — the eighth was >5min tree-wide and ~30s scoped.
   A gate slow enough to skip is not a gate.
6. **Validate it FIRES before trusting it green.** Revert the fix, confirm it reports
   the exact site and exits 1, restore. Write that procedure into the exclusions-file
   header so the next session can re-run it. An audit that has never been seen to fail
   is indistinguishable from one that cannot.
7. **Write its test in `mlx_vlm/tests/`**, covering both directions plus the exclusions
   contract. Not optional: a permissive bug here fails nothing and still prints OK.
8. **Wire it into `.github/workflows/upstream-parity.yml`** and add its baseline to the
   table above, the cold-start block, and the post-merge block. Three places, all of
   which have gone stale before.

The exclusions file wants a **written reason per entry and a hard error when one is
missing** — every one of these scripts enforces that, and it is the only thing that
stops a baseline becoming the "pre-existing divergence, unreviewed" swamp
`.symbol-exclusions` took months to drain.

## The one rule that matters most

**Never conclude anything about a divergence from a `git diff --numstat` line
count.** That measure cannot distinguish three situations that need opposite
responses: content we dropped (restore), fork work (keep), and code upstream
itself later replaced (delete). Six wrong calls have been made this way; they are
listed at the end of `docs/upstream-gaps.md`.

**`--numstat` is only the most obvious form.** "Missing lines", "sites" and "hunks"
are the same measure wearing different hats, and reading them as a proxy for *how
much differs* is how `apc.py` was mis-sized by two orders of magnitude — see "the
fifth direction" above, and run `dev/check_body_divergence.py --summary` before you
size anything. A
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

The suite is **green: 3079 passed, 5 skipped, 0 failed.** Keep it that way. (This
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
# only for dev/fork_coverage_report.py; not a package dependency:
uv pip install --python .venv/bin/python coverage
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

**Registries were the longest-standing blind spot; `dev/check_upstream_registries.py`
is now the check.** See "the sixth direction" below. `tests/test_model_registry.py`
still exercises the specific entries and re-exports it knows about — extend it when you
add a registry — but the audit is what covers the ones nobody thought to test.

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
- **`models/qwen3_5_moe/qwen3_5_moe.py` is byte-identical to upstream ON PURPOSE**
  (`6be3f881`). The fork had its own unfused-expert probe keyed on `gate_proj` where
  upstream keys on `up_proj`; the two were shown identical on every layout that can
  exist and exact mirror images on the two that cannot (an expert with no gate
  projection is not SwiGLU; with no up projection it is not a gated MLP). Do not
  reintroduce the fork probe. `tests/test_qwen3_5_moe_sanitize.py` is fork-only and is
  now the sole carrier of that knowledge, with
  `test_a_partial_expert_layout_raises` there to stop it coming back.
- **`utils.py`'s remaining 12 missing upstream lines** are the fork's `print`/`logging`
  → module-logger conversions (`2f5e01dd`) plus one comment inside the same converted
  block. Not a gap.
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
  is usually committed to directly. Runs the eight gating audit scripts.

## Key dependencies

- `mlx` >= 0.31.2 — core framework
- `mlx-lm` >= 0.31.3 — text-only fallback only; not imported by library code
- `mlx-audio` >= 0.4.3
- `transformers` >= 5.5.0 — configs, tokenizers, processors
- `huggingface-hub` — model downloads
- `Pillow`, `opencv-python`, `miniaudio` — image/video/audio
- `fastapi`, `uvicorn` — server

## Union-merging a test file, or any file upstream also has

Four for four on `test_utils.py`, `test_trainer_utils.py`,
`test_batch_quantized_cache.py`, `test_speculative.py`, then on `test_server.py`,
`test_generate.py` and — as a library file — `apc.py`. **The step order is the point:**

1. AST-compare definition names both ways. **`only_ours` empty does NOT mean
   checkout-safe** — `test_utils.py` had zero fork-only definitions and real fork
   content in *bodies*.
2. Diff the **bodies** of shared definitions: `dev/check_body_divergence.py --file
   <path>` (see "the fifth direction"). Ours-only
   content is either a fork adaptation (keep, mark `# Fork:`) or a weakened assertion
   (take upstream's). **These look identical in a diff**; `memory.md`'s change log is
   what distinguishes them.
3. Take upstream's file as the base so future merges apply, then re-append fork-only
   definitions below a `# Fork additions below this line` banner.
4. Union the import block, and mark fork-only imports **inline** (`# Fork:` on the
   line) — a standalone comment gets hoisted by isort.
5. Re-run `check_upstream_symbols.py`, `check_fork_markers.py` and
   `check_body_divergence.py`, and prune what stops
   being excused, **in the same commit**.

Step 2 is the step that pays. In `test_server.py`, 10 of 232 shared bodies differed and
split four ways: take upstream (5, all asserting behaviour restored the same session),
keep ours (1, a real fork adaptation, now commented so nobody "converges" it), cosmetic
(1), and one **test-certified bug** — a fork test edited to assert #1402's buggy cache
key, so the suite was actively defending it.

## Formatting when `pre-commit` is not installed

`pre-commit` is not in `.venv`. The pinned hooks run standalone:

```bash
uvx black@26.3.1 <files>
uvx isort@5.13.2 --profile=black <files>
uvx autoflake@2.2.1 --check --remove-all-unused-imports --ignore-init-module-imports <files>
```

`black` warns that Python 3.13 cannot verify code targeting 3.14; that warning is
expected and the reformat is still correct.

## Traps — every one of these cost a real cycle

Collected from the retired handoff files (`handoff-2026-08-10.md`, then
`handoff-2026-08-11.md`), each deleted per its own "delete this when §2 is empty"
instruction once its task list was drained. Ordered roughly by how often they bite.

**There is no handoff file now, deliberately.** Both were task lists, and a task list
holding no tasks is the document most likely to go stale next — the last one needed four
separate correction commits for numbers that restated themselves, including three
consecutive revisions of a single unpushed-commit count. Standing work lives in this
file as rules ("merge weekly", "re-run the sweep after each merge"); state lives in the
tree and is verified by the cold-start section at the top; history lives in `memory.md`
and `docs/upstream-gaps.md`. If a future session needs to hand off *tasks*, write a new
one and delete it on the same trigger.

**Environment and tooling**

1. **Always `git -C /Users/ia87221/ws/mlx-vlm`.** `cd` persists between tool calls, and
   a wrong relative path returns **empty output, not an error** — three times that made
   a file look fork-only or look byte-identical to upstream. Same class: a zsh
   `--include=*.py` glob silently swallows a `grep`, and `| head` reports `exit=0` over
   a hard failure.
2. **`cp` is aliased to `cp -i`.** A restore-from-backup silently did nothing and left
   a neutering edit in the tree. Use `command cp -f`, then check `git status`.
   Better still, **restore from git** (`git checkout HEAD -- <path>`) rather than from
   `/tmp`: a stale `/tmp` backup from an earlier session will restore the wrong content
   and the `-i` prompt is what tells you, if you read it.
3. **Never read a count off a truncated pipe.** Every audit prints its own totals.
   Its sibling: **an EMPTY result is not a passing result.** A backgrounded suite run
   left a 0-byte output file, and `grep -E "passed|failed"` over it printed nothing —
   which looks exactly like a clean run with a quiet tail. Re-run it in the foreground
   rather than inferring; `wc -l` on the output file is the one-second check that says
   which of the two you have. Same class as the `| head` masking `exit=1` below.
4. **Never count guards off a `-k`-filtered run.** `-k "Reasoning or MarkerUnion"` also
   matched two pre-existing tests, which is how a commit message came to claim "4 of 6
   fire" when all 4 of the new ones did. The suite total is the reliable figure.
5. **`mlx_vlm.generate` is the re-exported *function*, not the module.**
   `from mlx_vlm.generate import ar` works; `import mlx_vlm.generate.ar as m` and
   attribute access both fail.

**Provenance**

6. **"Not an ancestor of `upstream/main`" ≠ fork-authored** — upstream squash-merges.
   Check `git log -1 --format=%an`.
7. **A partially-restored commit is unfinished, and a "RESOLVED" note can be one half
   short.** Restore from `git show --stat`, never from a report's file list. For a
   commit that is mostly `print` -> `logger`, a partial restore leaves **no failing
   test and no audit hit** — `cfcc36d9` was declared whole three separate times.
   Worse: a note naming the symbols it verified *present* treats presence as
   completion, when what is missing can be a **rewrite of a symbol that was there all
   along** (`7fbc7bc9`). Presence of a symbol says nothing about whether its body is
   upstream's.
   **And a PER-FILE byte-identity sweep cannot close a commit either.** The obvious
   way to check one is `for f in $(git show --name-only ...); do git diff
   upstream/main -- $f; done`, and for `29b6c00b` that returned **10 of 11 files at
   0 diff-lines** — which reads as "essentially landed" and is exactly wrong. The
   whole residue was two keyword arguments inside the eleventh. A 0-diff file tells
   you nothing you did not already know; **the diverged files are the only ones worth
   reading, and you must look at the commit's own hunks within them**, not at the
   file's total divergence. `29b6c00b` has now been declared complete twice.
8. **"Not missing" is not "not dropped".** `#1433`'s prefill gate and `#1598`'s
   predicate refactor were *narrowings and refactors upstream had already removed* —
   nothing was absent, so no report could list them and no identifier count could flag
   them. Only comparing the shared body against upstream's finds this shape.
9. **A weakened assertion and a legitimate numerical relaxation look identical in a
   diff.** AST-diff bodies and check `memory.md` for a recorded reason. Worse variant:
   a fork test had been edited to assert a **buggy** value (#1402's cache key), so the
   suite actively certified the bug. `only_ours` being empty proves nothing.

**Measurement**

10. **A line-count delta measures ALIGNMENT, not CONTENT.** The general form of the
    rule above. `apc.py` read as 554 missing lines / 36 sites / 75 hunks and was a
    reordering. Two commits' entire reported residue turned out to be a **local
    variable rename**. Run `dev/check_body_divergence.py --summary`.
11. **The per-commit dropped-hunk report does not tell you how much of a file
    diverges.** It ranks commits it can attribute. Deciding a file was "markable" from
    it produced the wrong call on `turboquant.py` twice. Use a direct
    `git diff upstream/main -- <path>` for that question, and
    `dev/check_body_divergence.py --file <path>` to classify what it shows.
12. **A marker on a whitespace probe line creates a new probe line.** Chasing them in
    `test_generate.py` went 10 -> 8 -> 1 -> 2 sites. **Converge the ordering instead** —
    that took it to 0. And check `git diff -U0` before assuming a residual site IS a
    probe: a deletion-only hunk looks identical and is not fixable that way.
    `check_body_divergence.py` now refuses `# Fork: placement only` outright: all three
    markers in this tree that claimed it were false.

**Proof**

13. **`git stash` reverts to HEAD, which is useless once the fix is committed.** A
    guard written after its own fix landed will pass against HEAD and prove nothing.
    Revert to the commit *before* the fix: `git show <fix>^:<path> > <path>`, run,
    restore.
14. **A restored upstream test can fail for a reason that has nothing to do with
    placement.** `221fe0b3`'s streaming test failed three times for three different
    causes, each hidden behind the last. Do not adjust a restored test to make it pass;
    find the missing half.
15. **For a test-file reorder, compare `pytest --collect-only` counts on both sides.**
    It is the only thing that catches a shadowed definition — which is how the
    doubly-defined `TestLagunaProcessor` hid a missing symbol.

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
