# Handoff

The single handoff for this repo. Next-session prompt: **"Read handoff.md and do the
next open task."** Keep exactly one `handoff.md`: update this file in place, never add a
second one. When every task is done, delete it, along with the "Open handoff" paragraph
in AGENTS.md's "Traps" section, in the same commit.

Last updated 2026-10-26, by the session that closed T1 (the mlx 0.32.4 exactness failures,
`98cfb0b6`). Open: T2, T3.

## Context (read first)

- **Repo:** `ivan-avramov/mlx-vlm`, a fork of `Blaizzy/mlx-vlm`. `main` is the only
  branch. Remotes: `origin` (push) and `upstream` (fetch only). Merge, never rebase.
  Push only when the operator says so. Rules live in `AGENTS.md`, history in
  `memory.md` (see the 2026-10-25 block for this sync), and upstream defects found here
  in `docs/upstream-bugs.md` (items 9–14 are new).
- **State:** `main` contains `upstream/main` (`e9af3766`); `git log HEAD..upstream/main`
  was empty on 2026-10-25, and the sync was pushed (`origin/main` at `1669ee72` or later).
- **No hosted CI.** GitHub Actions are disabled on the fork (free-account minutes), and
  `.github/workflows/` is byte-identical to upstream's. The gate is local:
  `.venv/bin/python dev/ci.py` (pre-commit, the eight audits vs `.upstream-sync-ref`,
  the full suite; exit 1 on failure). `dev/ci.py drift` and `dev/ci.py leads` are
  information only.
- **Current gate result (mlx 0.32.4, 2026-10-26, `98cfb0b6`):** style ok, all eight
  audits ok, suite 9012 passed / 0 failed.
- **mlx 0.32.3 is unsupported.** Since upstream #2527 the exact verifier kernels match
  only 0.32.4's qmv; the suite fails ~190 tests on 0.32.3 (measured 2026-10-26). The
  earlier claim here that T1's five tests passed on 0.32.3 was wrong for this tree.
- **Venv** (`.venv`, Python 3.12): mlx 0.32.4,
  transformers 5.14.1, huggingface-hub 1.33.0 (keep it below 2.0), plus datasets,
  pytest-xdist, cryptography and pyflakes. `requirements.txt` needs `mlx>=0.32.4`.
- **Test-suite conventions added this sync:** `tests/conftest.py` has
  `KNOWN_UPSTREAM_FAILURES` (upstream tests that fail on upstream's own tree) and
  `FORK_DIVERGENT_UPSTREAM_TESTS` (upstream tests that assert behaviour the fork
  changes on purpose). Both apply strict xfail, so an entry turns red when it stops
  failing. The three fork-owned test copies (C104: `tests/test_speculative.py`,
  `test_server.py`, `test_generate.py`) now carry slices of upstream's test harness
  verbatim, so most new upstream tests can be ported unchanged.

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

- **Order:** T1 is fixed in `98cfb0b6` (push pending the operator). The stack must move
  to mlx 0.32.4 in the same step that takes any fork commit from the `e9af3766` sync on:
  from there the fork's exact verifier kernels match only 0.32.4, and ~190 fork tests
  fail on 0.32.3. Until then the stack stays on 0.32.3 and on its current fork commit.
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
