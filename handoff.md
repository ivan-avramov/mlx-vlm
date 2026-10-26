# Handoff

The single handoff for this repo. Next-session prompt: **"Read handoff.md and do the
next open task."** Keep exactly one `handoff.md`: update this file in place, never add a
second one. When every task is done, delete it, along with the "Open handoff" paragraph
in AGENTS.md's "Traps" section, in the same commit.

Last updated 2026-10-25, by the session that did the staged upstream sync to `e9af3766`.

## Context (read first)

- **Repo:** `ivan-avramov/mlx-vlm`, a fork of `Blaizzy/mlx-vlm`. `main` is the only
  branch. Remotes: `origin` (push) and `upstream` (fetch only). Merge, never rebase.
  Push only when the operator says so. Rules live in `AGENTS.md`, history in
  `memory.md` (see the 2026-10-25 block for this sync), and upstream defects found here
  in `docs/upstream-bugs.md` (items 9–13 are new).
- **State:** `main` contains `upstream/main` (`e9af3766`); `git log HEAD..upstream/main`
  was empty on 2026-10-25, and the sync was pushed (`origin/main` at `1669ee72` or later).
- **No hosted CI.** GitHub Actions are disabled on the fork (free-account minutes), and
  `.github/workflows/` is byte-identical to upstream's. The gate is local:
  `.venv/bin/python dev/ci.py` (pre-commit, the eight audits vs `.upstream-sync-ref`,
  the full suite; exit 1 on failure). `dev/ci.py drift` and `dev/ci.py leads` are
  information only.
- **Current gate result (mlx 0.32.4):** style ok, all eight audits ok, suite
  8821 passed / **5 failed**, exactly T1's list. Nothing else fails.
- **Venv** (`.venv`, Python 3.12): mlx 0.32.4 (also run on 0.32.3, see T1),
  transformers 5.14.1, huggingface-hub 1.33.0 (keep it below 2.0), plus datasets,
  pytest-xdist, cryptography and pyflakes. `requirements.txt` needs `mlx>=0.32.4`.
- **Test-suite conventions added this sync:** `tests/conftest.py` has
  `KNOWN_UPSTREAM_FAILURES` (upstream tests that fail on upstream's own tree) and
  `FORK_DIVERGENT_UPSTREAM_TESTS` (upstream tests that assert behaviour the fork
  changes on purpose). Both apply strict xfail, so an entry turns red when it stops
  failing. The three fork-owned test copies (C104: `tests/test_speculative.py`,
  `test_server.py`, `test_generate.py`) now carry slices of upstream's test harness
  verbatim, so most new upstream tests can be ported unchanged.

## T1 — five fork exactness tests fail on mlx 0.32.4 (blocks the stack bump)

The fork now needs mlx 0.32.4. On it these five fail; on mlx 0.32.3 all five pass
(same tree, checked 2026-10-25):

```
tests/test_mtp_exactness.py::test_e1a_exact_joint_law_quantized_hybrid[Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed]
tests/test_p369_phase2_observe.py::test_quantized_e1c_detects_an_in_support_change_of_one_verifier_row
tests/test_p369_phase2_oracle.py::test_the_live_classifier_flags_exactly_the_pre_fix_defects
tests/test_p369_phase2_oracle.py::test_the_pre_fix_tree_shows_each_precise_defect
tests/test_p400_tq_width2.py::test_the_exactness_checks_hold_under_production_numerics
```

Known:
- **Cause class.** MLX 0.32.4 (ml-explore/mlx#4516) widens each input to float
  *before* summing a group in its affine qmv `load_vector`. Kernels that copy MLX's qmv
  drift by about one bf16 ulp unless they do the same. Upstream #2527 (`1a4ddb68`)
  fixed its copies (`models/fast_ops.py`, `models/glm_moe_dsa/speculative_verifier.py`);
  both are byte-identical here. `models/quantized_verifier.py` already widens. A
  tree-wide grep for an unwidened `sum += x[i] + x[i + 1] ...` finds nothing left.
- **Already fixed:** 169 of the original 174 failures on 0.32.4, via #2527 plus the
  Qwen verifier q/k-normalization delegation (`docs/upstream-bugs.md` item 12).
- **The deployed-model case:** `max |P_on - P_off| = 1.3e-05` against a `1e-05` bound
  (`test_mtp_exactness.py:295`). The other four are the fork's P369/P400 research
  harnesses. The two `oracle` tests check *pre-fix defect signatures*, which a one-ulp
  numeric shift can legitimately move.

To do:
1. Find which verify-path kernel or MLX op still differs from decode on 0.32.4 for the
   `Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed` recipe. Suspects: an MLX `qmm`
   (multi-row) call on a verify path whose decode twin is `qmv`, for one of the
   recipe's bit widths; and the TurboQuant "deployed KV codec" paths (P400, P369
   observe).
2. Fix it so that verify equals decode on both 0.32.4 and 0.32.3, as upstream did, keyed
   on the MLX version if the two need different arithmetic. **Do not loosen the 1e-5
   bound.**
3. For each of the two `oracle` tests, decide whether it needs a code fix or a reviewed
   re-baseline of its expected defect signature, and write the reason down.
4. Done when `.venv/bin/python dev/ci.py` is green on 0.32.4 and still green on 0.32.3
   (`uv pip install --python .venv/bin/python "mlx==0.32.3"`, run the gate, reinstall
   0.32.4).

## T2 — make the upstream-copy differential a `dev/` audit

At each stage of the sync, upstream's own copies of the three C104 test files were
run against the merged tree, and only *new* failures were judged. That check found a
real bug (a standalone `glm_moe_dsa_mtp` drafter loaded with every weight silently
dropped). It is not yet in the repo. Make it permanent:

1. `dev/check_upstream_test_copies.py`: for each C104 file, write
   `git show <ref>:<path>` to a temporary `mlx_vlm/tests/test_zz_up_*.py`, run pytest on
   those files from `mlx_vlm/`, and compare the failing node ids (parametrization
   stripped) with a reviewed baseline file, `.upstream-test-copy-failures`: one id per
   line with a reason, and a hard error on a missing reason, like the other exclusion
   files. Always remove the temporary files.
2. Gate on **new** failures only; report baseline entries that now pass, so they can be
   pruned.
3. Seed the baseline from a run against `.upstream-sync-ref`. Expect the fork's
   deliberate divergences: batched-MTP refusal, opt-in chat compaction (upstream's
   copies run without `MLX_VLM_CHAT_COMPACTION=1`), the prefix-cache trim policy, and
   the `KNOWN_UPSTREAM_FAILURES` items. Counts for reference: 31 failures on the
   pre-sync tree, about 49 at `e9af3766`.
4. The `check_*.py` name makes `tests/test_local_ci.py` require it in `dev/ci.py`'s
   `AUDITS`. Give it a test in `mlx_vlm/tests/` covering both directions plus the
   reason contract, validate that it fires, and add its baseline count to AGENTS.md's
   table and cold-start block.

## T3 — notes for the stack bump (done in the `mlx_local_stack` session, not here)

- **Order:** the fork is pushed (`origin/main` at `1669ee72` or later). Bump the stack only after
  T1, because the fork now requires mlx 0.32.4 and T1 is exactly the 0.32.4 breakage.
  Until then the stack stays on 0.32.3 and on its current fork commit.
- **Dependencies:** `datasets>=4.0.0` is new. A full `uv lock` resolves consistently,
  but installing `datasets` on its own into an existing env upgrades `huggingface-hub`
  to 2.x, which `transformers` 5.14 rejects. Expect `transformers`/`huggingface-hub`
  bumps in the lock; run the stack's own tests on them.
- **Renamed runtime knobs:** `spec_draft_model` / `spec_draft_kind` are now
  `draft_model` / `draft_kind` (#2457). CLI flags and `MLX_VLM_DRAFT_*` env vars are
  unchanged; an old-name settings PATCH comes back as `rejected: unknown knob`. The
  stack only uses the CLI flags. Stale doc row:
  `mlx_local_stack/docs/specs/p405-kv-scheme-precedence.md:60`.
- **Sampling defaults (#2545):** Python and CLI generation without an explicit
  temperature now follows the checkpoint's `generation_config.json`. The server is
  unchanged. Current stack probes all pin `temperature=0` or pass a sampler.
- **Nemotron drafter:** the probe-only
  `caslca/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit-mtp-drafter` no longer loads (the
  fork converged on upstream's drafter layout; strict load fails loudly). The target is
  served draft-OFF, so nothing served changes. If the probe is wanted again, re-split
  with `python -m mlx_vlm.split_mtp` from the BF16 source.
