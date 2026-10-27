# Handoff

The single handoff for this repo. Next-session prompt: **"Read handoff.md and do the
next open task."** Keep exactly one `handoff.md`: update this file in place, never add a
second one. When every task is done, delete it, along with the "Open handoff" paragraph
in AGENTS.md's "Traps" section, in the same commit.

Last updated 2026-10-26, by the session that merged upstream `2ceff4c1` (0.7.11). T1
closed in `98cfb0b6`, T2 in `069fab9c`. Open: T3.

## Context (read first)

- **Repo:** `ivan-avramov/mlx-vlm`, a fork of `Blaizzy/mlx-vlm`. `main` is the only
  branch. Remotes: `origin` (push) and `upstream` (fetch only). Merge, never rebase.
  Push only when the operator says so. Rules live in `AGENTS.md`, history in
  `memory.md` (see the 2026-10-25 block for this sync), and upstream defects found here
  in `docs/upstream-bugs.md` (items 9–14 are new).
- **State (2026-10-26):** `main` contains upstream `2ceff4c1` (0.7.11: #2558 native MTP
  in the SFT/ORPO trainers, #2600 diarization and speaker-embedding endpoints; see
  `memory.md`) and the audio-section convergence (`0d80513f`). All of it, T1 and T2
  included, is pushed (`origin/main` = `main`, 2026-10-26). Push only when the operator
  says so. Next weekly sync is due by 2026-11-02.
- **No hosted CI.** GitHub Actions are disabled on the fork (free-account minutes), and
  `.github/workflows/` is byte-identical to upstream's. The gate is local:
  `.venv/bin/python dev/ci.py` (pre-commit, the nine audits vs `.upstream-sync-ref`,
  the full suite; exit 1 on failure). `dev/ci.py drift` and `dev/ci.py leads` are
  information only.
- **Current gate result (mlx 0.32.4, 2026-10-26, after the `2ceff4c1` merge and the audio convergence):** style ok, all
  nine audits ok, suite 9205 passed / 0 failed.
- **mlx 0.32.3 is unsupported.** Since upstream #2527 the exact verifier kernels match
  only 0.32.4's qmv; the suite fails ~190 tests on 0.32.3 (measured 2026-10-26). The
  earlier claim here that T1's five tests passed on 0.32.3 was wrong for this tree.
- **Checking another mlx version without touching `.venv`:** build a throwaway venv
  from `uv pip freeze --python .venv/bin/python` minus `mlx`/`mlx-metal`/`mlx-vlm`, add
  the other mlx, then run tests with `PYTHONPATH=<checkout>` (it takes precedence over
  the editable install). Diff the two freezes first; only mlx may differ. Never
  reinstall mlx in `.venv`.
- **Verify == decode depends on MLX's private dispatch.** `singleton_quantized_linear`
  is exact only where MLX's plain singleton qmv and `gather_qmv` pick the same kernel
  (`mlx/backend/metal/quantized.cpp`, `dispatch_qmv` vs `gather_qmv`). Known
  divergences: K <= 128 (`qmv_quad`) and, from 0.32.4, N % 8 != 0 (`qmv_fast` partial
  rows); both now take a per-row path (`docs/upstream-bugs.md` 14). On every MLX bump,
  diff that dispatch between the two versions: the tests pin only the known geometries.
- **Four-eyes helpers:** the codex CLI sandbox has no Metal device, so codex cannot run
  GPU tests; the designer runs them and relays the output. T1's run is recorded in
  `98cfb0b6`'s message.
- **Venv** (`.venv`, Python 3.12): mlx 0.32.4, mlx-audio 0.6.2,
  transformers 5.14.1, huggingface-hub 1.33.0 (keep it below 2.0), plus datasets,
  pytest-xdist, cryptography and pyflakes. `requirements.txt` needs `mlx>=0.32.4`.
- **Test-suite conventions added this sync:** `tests/conftest.py` has
  `KNOWN_UPSTREAM_FAILURES` (upstream tests that fail on upstream's own tree) and
  `FORK_DIVERGENT_UPSTREAM_TESTS` (upstream tests that assert behaviour the fork
  changes on purpose). Both apply strict xfail, so an entry turns red when it stops
  failing. The three fork-owned test copies (C104: `tests/test_speculative.py`,
  `test_server.py`, `test_generate.py`) now carry slices of upstream's test harness
  verbatim, so most new upstream tests can be ported unchanged.

## T3 — notes for the stack bump (done in the `mlx_local_stack` session, not here)

- **Order:** T1 is fixed in `98cfb0b6` (push pending the operator). The stack must move
  to mlx 0.32.4 in the same step that takes any fork commit from the `e9af3766` sync on:
  from there the fork's exact verifier kernels match only 0.32.4, and ~190 fork tests
  fail on 0.32.3. Until then the stack stays on 0.32.3 and on its current fork commit.
- **Dependencies:** `datasets>=4.0.0` is new. A full `uv lock` resolves consistently,
  but installing `datasets` on its own into an existing env upgrades `huggingface-hub`
  to 2.x, which `transformers` 5.14 rejects. Expect `transformers`/`huggingface-hub`
  bumps in the lock; run the stack's own tests on them. From the `2ceff4c1` merge on,
  `mlx-audio>=0.6.2` (#2600; was 0.5.2). In `.venv` the 0.5.6 -> 0.6.2 upgrade moved only
  `miniaudio` (1.61 -> 1.71).
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
