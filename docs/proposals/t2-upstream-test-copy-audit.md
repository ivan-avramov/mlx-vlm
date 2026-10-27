# T2 — the upstream-test-copy differential as a gating `dev/` audit

Status: specification, four-eyes light. Source: `handoff.md` T2.

## Goal

The fork owns three test files that upstream also has (C104: `mlx_vlm/tests/test_speculative.py`,
`test_server.py`, `test_generate.py`), so upstream's own assertions in those files never run
against the fork's library. During the 2026-10-25 sync, running upstream's copies against the
merged tree and judging only *new* failures found a real bug (a standalone `glm_moe_dsa_mtp`
drafter loaded with every weight silently dropped). That check exists only as a throwaway shell
script in `/tmp`. Solving this means a permanent `dev/check_upstream_test_copies.py` that runs
upstream's copies of the three files (at a given ref) against the working tree, fails when any
test fails that a reviewed baseline does not excuse, reports baseline entries that no longer
excuse anything, never leaves files behind, never reports success when it could not run, and is
wired into `dev/ci.py`'s gate with a test pinning its behaviour.

## Scope

- New: `dev/check_upstream_test_copies.py`.
- New: `.upstream-test-copy-failures` (the baseline; header written by the implementer, entries
  seeded by the designer after the candidate exists).
- New: `mlx_vlm/tests/test_upstream_test_copy_check.py` (the audit's own test).
- Changed: `dev/ci.py` — add the script to `AUDITS` (no extra args); update "eight" in its
  docstring/help text to "nine".
- Designer only (not the implementer): AGENTS.md table, cold-start block and audit counts;
  `handoff.md`; `memory.md`.

## Settled constraints

- (handoff) Copies are written to `mlx_vlm/tests/test_zz_up_<stem>.py`, where `<stem>` is the
  original file's stem (`test_zz_up_test_server.py`), and pytest is run from `mlx_vlm/`. They are
  always removed.
- (handoff) Gate on **new** failures only; report baseline entries that now pass.
- (handoff) Baseline: one entry per line with a `# reason`; a missing reason is a hard error.
- (handoff) Seeded from a run against `.upstream-sync-ref`.
- (handoff) The `check_*.py` name puts it under `tests/test_local_ci.py`'s wiring check.
- (designer) Accepts `--upstream-ref REF` (default `upstream/main`), because `dev/ci.py` passes
  that flag to every audit.
- (designer) Stale baseline entries **warn and do not gate**, the same rule every other audit
  uses for its exclusions file (`check_body_divergence.py:573`, `check_dead_helpers.py:254`, ...).
- (designer, departs from the handoff's "parametrization stripped" — operator to confirm at Go)
  Failures are reported with their **exact** node ids. A baseline entry is a node-id *pattern* in
  which `*` is the only wildcard and brackets are literal (the same rule as
  `tests/conftest.py::_node_pattern`). An entry with no `[` also matches every parametrization of
  that test. Reason: 13 of today's 16 failing test functions fail only for some parameters (e.g.
  `TestCompaction::test_max_output_is_a_ceiling_for_short_prompts` fails only its `chat`
  parameters, `test_nemotron_mtp_generation_matches_autoregressive` only `[0.7-dtype1-video]`);
  stripping would excuse a future regression in the parameters that pass today.

## Assumptions

- The three files exist at every ref the audit is given; absence is an error, not a skip.
- The run takes ~15 s on this machine (measured: 49 failed / 1687 passed / 6 skipped, 14 s, at
  the current tree against `e9af3766`).
- Nothing else collects `mlx_vlm/tests/` while the audit runs (`dev/ci.py` runs stages
  sequentially).

## Supported conditions and formats

- macOS with Metal, the repo's `.venv` (Python 3.12, pytest 8+), invoked as
  `python dev/check_upstream_test_copies.py [...]` from any cwd; paths derive from the script's
  location.
- Baseline file format (UTF-8 text):
  - blank lines and lines whose first non-space character is `#` are ignored;
  - every other line is `<pattern>  # <reason>`; the pattern is the text before the first `#`,
    stripped; the reason is the text after it, stripped and non-empty;
  - a pattern is a repo-relative node id of the *original* file, e.g.
    `mlx_vlm/tests/test_server.py::TestCompaction::test_max_output_is_a_ceiling_for_short_prompts[*-chat-*]`;
  - a collection error of a whole copy is the id `mlx_vlm/tests/<file>.py` (no `::`);
  - a missing reason, a pattern that does not start with one of the three original paths, or a
    duplicate pattern is a hard error naming the line number.
- Reported ids are repo-relative to the original file, with the exact pytest node-id suffix:
  `tests/test_zz_up_test_server.py::A::b[x]` -> `mlx_vlm/tests/test_server.py::A::b[x]`.

## Interfaces

Module constants:

- `COPIES = ("mlx_vlm/tests/test_speculative.py", "mlx_vlm/tests/test_server.py",
  "mlx_vlm/tests/test_generate.py")`.
- `BASELINE_FILE` = repo root / `.upstream-test-copy-failures`.

Functions (importable; the test exercises them directly):

- `load_baseline(path) -> dict[str, str]` — pattern -> reason. Raises `SystemExit` with a message
  naming the file and line for each hard error above.
- `matches(pattern: str, nodeid: str) -> bool` — `*` matches any run of characters including
  none; everything else is literal; the whole id must match. A pattern with no `[` additionally
  matches `nodeid` when `nodeid` with its trailing `[...]` removed matches it.
- `compare(failing: set[str], baseline: dict[str, str]) -> tuple[list[str], list[str]]` —
  `(new, stale)`: `new` = sorted failing ids matched by no pattern; `stale` = sorted patterns
  matching no failing id.
- `run_copies(ref: str, *, repo_root: Path, copies: Sequence[str], python: str) ->
  dict[str, str]` — writes each copy, runs pytest once over all of them, removes everything it
  created, and returns {failing original id -> one-line crash message ('' if none)}. Details:
  - `git -C repo_root show <ref>:<path>` for each path; failure -> `CopyRunError`.
  - Destination `<dir of path>/test_zz_up_<stem>.py`. If that path is tracked by git ->
    `CopyRunError` (never overwrite a real file). An untracked leftover is overwritten.
  - pytest runs as `python -m pytest -q -p no:cacheprovider --continue-on-collection-errors
    -p <plugin> <copies...>` with cwd = the parent of the copies' directory (`mlx_vlm/` for the
    real layout), and a child environment that sets `PYTHONDONTWRITEBYTECODE=1` and removes every
    `MLX_VLM_*` variable (upstream's copies assume upstream's defaults, e.g. compaction off).
  - Outcomes come from a small pytest plugin module the script writes to a temporary directory
    (put on the child's `PYTHONPATH`, ahead of anything inherited). The plugin records, per
    report: the node id, and whether it failed (`report.failed` in any phase, including collect
    reports), plus the crash message; and the number of items collected per copy. It writes
    them as JSON to a temporary file. **Never parse terminal output** (it truncates and node ids
    may contain ` - `).
  - pytest exit codes 0 and 1 are normal. Any other exit code, a missing or unreadable result
    file, or any copy with zero collected items and no collection error -> `CopyRunError` with
    the pytest output tail.
  - Cleanup in `finally`: the copies, any `__pycache__/test_zz_up_*` files, the plugin directory
    and the result file — also on `KeyboardInterrupt`.
- `main(argv=None) -> int`:
  - `--upstream-ref REF` (default `upstream/main`);
  - `--list`: print every failing id (sorted, one per line, ready to paste into the baseline
    without a reason) and return 0 without consulting the baseline;
  - otherwise: load the baseline (hard error -> non-zero), `run_copies`, `compare`; print a
    summary line (`N failing, M excused by K baseline entries, X new vs <ref>`); print each new id
    with its crash message; print stale patterns under a `warning:` heading; return 1 if `new`,
    else 0.
  - `CopyRunError` -> print it to stderr, return 2.

Success path: baseline with entries covering today's 49 failures -> `run_copies` returns 49 ids
-> `compare` returns `([], [])` -> prints the summary, returns 0, tree unchanged.

Failure path: a fork change breaks an upstream assertion in `test_generate.py::test_foo` ->
`run_copies` returns it -> `compare` puts `mlx_vlm/tests/test_generate.py::test_foo` in `new` ->
printed with its crash message, returns 1 -> `dev/ci.py` reports the audit FAIL.

## Acceptance criteria

1. `python dev/check_upstream_test_copies.py --upstream-ref e9af3766` returns 0 with the seeded
   baseline; `git status --porcelain --ignored mlx_vlm/tests` shows no `test_zz_up_` path
   afterwards.
2. Removing one baseline entry makes it return 1 and name exactly the uncovered ids.
3. A fork change that breaks an upstream-copy assertion makes it return 1 naming that test
   (designer validates by a temporary revert, per AGENTS.md "Validate it FIRES").
4. Interrupting it (SIGINT) mid-run leaves no `test_zz_up_` file.
5. `dev/ci.py audits` runs it and passes; `tests/test_local_ci.py` passes.
6. `mlx_vlm/tests/test_upstream_test_copy_check.py` covers: the reason contract and every other
   baseline hard error; `matches` (wildcards, literal brackets, the no-`[` rule, whole-id
   anchoring); `compare` both directions (an unexcused failure is new; an entry matching nothing
   is stale; neither is silently dropped); id mapping; and `run_copies` end to end against a
   throwaway git repo built in `tmp_path` (a committed copy with one passing test, one failing
   parametrized case and, in a second copy, an import error) — including that the working tree
   is clean afterwards, that a tracked destination is refused, and that an unknown ref raises
   `CopyRunError`. It must not run the real three upstream files.
7. The full suite passes as on the baseline; pre-commit clean.

## Non-goals

- Running upstream copies of any other file (the fork's other test files match upstream or are
  fork-only).
- Fixing any of the failures the baseline excuses.
- A worktree-based run against an arbitrary fork commit (the old `/tmp` script did that).

## Open questions (for Go)

1. Exact ids with `*` patterns instead of stripped parametrization — recommended (see Settled).
2. Stale entries warn, not gate — recommended for consistency; strict xfail-style gating would
   turn red on any flaky test.
3. Scrubbing `MLX_VLM_*` from the child env — recommended: makes the result independent of the
   operator's shell.

## Amendments after review (designer, 2026-10-26; within the goal, not re-reviewed)

From the test-writer's review and the panel's opinion. Each one closes a way the audit could report
success without a complete run, or damage or leave residue in the tree.

- **R1 Complete runs only.** The child's environment also drops `PYTEST_ADDOPTS` and
  `PYTEST_PLUGINS`. The plugin records the items left after deselection
  (`pytest_collection_finish`), any deselected items (`pytest_deselected`), whether
  `--collect-only` was in effect, and each item's final outcome: its call report, or a setup
  report that failed or skipped. `CopyRunError` is raised if collect-only was in effect, if any
  item was deselected, if any collected item has no final outcome (`-x`, `--maxfail`, a crash),
  or if every collected item of a copy was skipped. The summary prints, per copy, the collected,
  passed, failed and skipped counts.
- **R2 Destinations.** A destination that exists and is not a regular file (a symlink, a
  directory or anything else, checked with `lstat`) raises `CopyRunError`. An untracked regular
  leftover is unlinked. Each copy is created with `O_CREAT | O_EXCL | O_NOFOLLOW`. Matching
  `__pycache__/test_zz_up_<stem>*` entries are removed **before** the run as well as after, since
  `PYTHONDONTWRITEBYTECODE` stops writes but not reads.
- **R3 Cleanup.** Every artifact is removed independently. If an artifact in the tree cannot be
  removed and no other exception is in flight, `CopyRunError` is raised; otherwise the removal
  error goes to stderr and the original exception propagates.
- **R4 Signals.** While copies may exist, SIGTERM and SIGHUP raise `KeyboardInterrupt`, so the
  cleanup runs. The previous handlers are restored afterwards.
- **R5 One run at a time.** `run_copies` holds an exclusive, non-blocking `flock` on
  `<git dir of repo_root>/upstream-test-copies.lock` for its whole duration. If the lock is
  already held, it raises `CopyRunError`.
- **R6 Matching.** The no-`[` rule strips only a trailing `[...]`: the id must end in `]`, and
  everything from the first `[` to the end is removed.
- **R7 Baseline.** `main` reads the module's `BASELINE_FILE` when it is called. Hard-error
  messages name the path as given, plus the line number. The unreachable empty-pattern branch is
  removed.
- **R8 Messages.** When a report has no `reprcrash` (collection errors), the message is the last
  line beginning `E ` (without that prefix), else the last non-blank line. `test_zz_up_<stem>.py`
  in a message is rewritten to `<stem>.py`.
- **R9 Exit 2 means "could not run".** A blob that is not UTF-8, or an `OSError` while writing a
  copy, raises `CopyRunError`.

Declined, as residuals:
- **`#` inside a node id.** No current id contains one, and the failure mode is loud: the entry
  goes stale and the failure counts as new.
- **A future conftest xfail pattern matching a copy.** No pattern matches a copy today, and
  recording xfails would misreport upstream's own xfail markers.
- **A conftest `collect_ignore` for leftovers.** It touches a shared file and would hide
  leftovers instead of exposing them.
- **A concurrent full-suite run while the audit runs.** This is the spec's assumption: `dev/ci.py`
  runs its stages sequentially.

Scope request for the operator (not applied): a timeout on the child pytest.

## Amendments after verify (designer, 2026-10-26; second fix round approved by the operator)

- **S1** `matches`: the no-`[` rule applies only when the id contains a `[` and ends in `]`.
  Otherwise the result is the literal match, with no exception.
- **S2** A failure to remove any part of the plugin directory is collected as a cleanup problem,
  using `rmtree`'s error callback, and handled like every other R3 problem.
- **S3** SIGINT, SIGTERM and SIGHUP are blocked (`signal.pthread_sigmask`) while a copy is created
  and recorded, and throughout cleanup. A signal that arrives meanwhile is delivered once they are
  unblocked, after cleanup has finished. Cleanup also removes every destination that exists as a
  regular file, not only the recorded ones; the lock and the pre-checks make them the audit's
  own.
- **S4** An xfailed outcome (`report.wasxfail` on a skipped call or setup report) is recorded as
  `xfailed`. It is counted separately in the per-copy summary and does not count as skipped in the
  all-skipped check.
- **S5** The test file's timeouts for subprocess-based tests bound hangs only; they must be
  generous (>= 60 s).

## Amendment after the second fix round (designer; third and final round approved by the operator)

- **S3'** S3's mechanism changes; its guarantee does not. `pthread_sigmask` blocks only the calling
  thread. A process-directed signal reaches any other thread (an xdist worker's execnet thread),
  and Python still runs the handler in the main thread, so the masked section is interrupted.
  Instead, during the two critical sections (creating and recording a copy; all of cleanup),
  SIGINT, SIGTERM and SIGHUP get Python handlers that only record the signal. When the section
  ends, the previous handlers are restored, and if a signal was recorded, `KeyboardInterrupt` is
  raised. `pthread_sigmask` is no longer used. The child pytest inherits ordinary handlers. Must
  hold whether or not the process has other threads.
