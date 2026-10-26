# T1: fork exactness tests on mlx 0.32.4

## Goal

Five fork tests fail on mlx 0.32.4 (`handoff.md` T1):
`test_mtp_exactness.py::test_e1a_exact_joint_law_quantized_hybrid[...-mixed]`,
`test_p369_phase2_observe.py::test_quantized_e1c_detects_an_in_support_change_of_one_verifier_row`,
`test_p369_phase2_oracle.py::test_the_pre_fix_tree_shows_each_precise_defect`,
`test_p369_phase2_oracle.py::test_the_live_classifier_flags_exactly_the_pre_fix_defects`,
`test_p400_tq_width2.py::test_the_exactness_checks_hold_under_production_numerics`.
Solving it means: on mlx 0.32.4 (the fork's floor), verify equals decode bit for bit
for every quantized projection the exact verifier serves, with the 1e-5 law bound
unchanged; and each test harness checks what it claims independent of which token
trajectory the installed MLX's numerics happen to sample. `.venv/bin/python dev/ci.py`
is green on 0.32.4. mlx 0.32.3 is out of scope (operator, 2026-10-26): upstream #2527
made the exact verifier kernels 0.32.4-only, and the baseline suite fails 190 tests on
0.32.3 for that reason.

## Root causes (established by the designer; reproduce, do not re-derive)

1. **Library defect, upstream code.** `mlx_vlm/models/quantized_verifier.py::singleton_quantized_linear`
   (byte-identical to upstream, #2127) computes a `[B, T, K]` block with one
   `mx.gather_qmm` so each row gets "singleton qmv arithmetic". MLX's `dispatch_qmv`
   (`mlx/backend/metal/quantized.cpp`, identical in v0.32.3 and v0.32.4) routes a plain
   `quantized_matmul` with `K == 128 || (K == 64 && bits >= 2)`, power-of-two bits and
   no global scale to `qmv_quad` (quad-group reduction); `gather_qmv` never does (simd
   reduction). Different reduction trees, so for K <= 128 verify != decode by one output
   ulp on some inputs. Measured, 40 seeds per config, `[1, 2, K]` bf16: on 0.32.3 only
   8-bit K in {64, 128} mismatch; on 0.32.4 also 2- and 4-bit. All other K and bits
   match. In the E1a mixed fixture the 8-bit `linear_attn.out_proj` (K = 128) reaches it
   through `speculative/ops/linear.py::_target_verify_quantized_linear` (also
   upstream-identical); `supports_optimized_affine_head` rejects K % 512 != 0 so the
   fork's exact kernel is not used. Upstream's `models/linear.py::native_batch_linear`
   already guards its own call with `K % 512 == 0`; the speculative path does not.
2. **Test fixture, `test_p369_phase2_observe.py`.** The in-support test bumps the top
   canonical row value at hard-coded `position = 3`. On 0.32.4 the sampled trajectory
   differs, and position 3's filtered support is one token (p = 1.0), so a bump cannot
   change the canonical distribution; the comparator is correct. The P400 test re-runs
   the same file under TF32 and fails for the same reason.
3. **Test fixture, `dev/p369_phase2` oracle.** Budget cases hard-code budgets 2/3/4.
   On 0.32.3 the dense fixture's free trajectory begins `61, 61` (two `<think>`, not
   counted), so the first over-budget token falls on round 1's first draft / second
   draft / bonus (positions 4/5/6) and the pre-fix tree's overshoot ladder is 2/1/0. On
   0.32.4 the trajectory begins `4, 57, 9`, the trips move to positions 2/3/4, the ladder
   reads 1/0/2, and in `budget2` the continuation samples `<think>` (61) again, which
   re-opens thinking and trips a second time (a second M3 event the oracle's
   first-forced-token measure does not count).
   `ThinkingBudgetCriteria` (`utils.py:2306`): `<think>` is never counted, `</think>` ends
   thinking, the count is cumulative, and the forced `\n` (63) is emitted at the position
   after the first token that takes the count over the budget.

## Scope

- `mlx_vlm/models/quantized_verifier.py` (`singleton_quantized_linear` only).
- `mlx_vlm/tests/test_p369_phase2_observe.py` (the in-support test only).
- `dev/p369_phase2/oracle_cases.py`, `dev/p369_phase2/oracle.py` (budget cases).
- `mlx_vlm/tests/test_p369_phase2_oracle.py` (`EXPECT_PRE_FIX` budget entries, docstring).
- One new fork regression test file, `mlx_vlm/tests/test_singleton_quantized_narrow.py`.
- `docs/upstream-bugs.md`: a new item for defect 1.

## Settled constraints (designer)

- Do not loosen any tolerance, the 1e-5 law bound included.
- Do not change MLX, `.venv` packages or the five tests' assertions other than as
  specified below.
- A fork hunk in an upstream-shared file carries a `# Fork:` comment saying what
  deviates and why (AGENTS.md "the third direction"); per-site, never a blanket claim.
- Format with the pinned hooks (`uvx black@26.3.1`, `uvx isort@5.13.2 --profile=black`).

## Interfaces and changes

### 1. `singleton_quantized_linear(linear, x) -> Optional[mx.array]`

Unchanged contract: returns `None` for unsupported inputs; otherwise a `[B, T, N]` array
whose every `[b, t]` row is bitwise equal to `linear(x[b:b+1, t:t+1])` (the decode call),
bias included. New behaviour: when `x.shape[-1] <= 128` or the output width
`N = linear.weight.shape[0]` is not a multiple of 8, compute each `[b, t]` row with
exactly that decode call and concatenate (per-row singleton calls are required, because a
`[B, 1, K]` call is a different native shape when B > 1). Otherwise the existing
`gather_qmm` path is unchanged. The two conditions are the dispatch geometries where
MLX's plain singleton qmv and `gather_qmv` choose different kernels: K <= 128 covers
`qmv_quad` (all versions; deliberately wider than MLX's exact condition), and N % 8 != 0
covers mlx 0.32.4's `qmv_fast` with partial rows, which `gather_qmv` (still gated on
`N % 8 == 0`) never takes (fix-round amendment: confirmed on 0.32.4 for bf16 and fp16,
4- and 8-bit, K in {512, 1024}, N in {7, 12, 100}; none on 0.32.3). The per-row path is
exact by construction and these projections are narrow, so its cost is negligible. One
`# Fork:` comment states this; the function's docstring describes both paths.

Success paths: E1a mixed fixture, 8-bit K = 128 `out_proj`, `x = [1, 2, 128]` -> two
decode calls -> rows bitwise equal decode; fp16 4-bit K = 512, N = 7 -> per-row path ->
bitwise equal decode. Failure path: unsupported module or K mismatch ->
`None`, unchanged.

### 2. Observe in-support test

Select the position instead of hard-coding 3: the first emitted position >= 1 that has
an ON verifier row and whose canonical distribution (`compare.canonical(row, params)`)
has >= 2 tokens in support and for which the bumped row (top value + 0.01, cast back
to the row's dtype) has a canonical distribution different from the unbumped row's
(fix-round amendment: in bf16 a +0.01 bump can round away); `assert` such a position
exists. Bump that row as now and assert the comparator reports exactly that position.
Nothing else in the file changes.

### 3. Oracle budget cases

Replace `budget2` / `budget3` / `budget4` with `budget_first_draft`,
`budget_second_draft`, `budget_bonus` (case key `budget_at` naming a `WHERE` entry).
`run_cases` generates the free trajectory with `<think>` appended as now and
`n = 16`, then derives `(budget, n)` with a new `budget_trip(free, where)`:

- for rounds `r = 1, 2, ...`: `p = WHERE[where](r)`; stop when `p + 1 > len(free)`;
  `head = free[:p + 1]`;
- if `head` contains `</think>` (62) or `\n` (63): no usable case (a later round's head
  contains it too) -> return `None`;
- if `head[-1]` is `<think>` (61): try the next round;
- `rest = free[p + 1 : WHERE["bonus"](r) + 1]` (the remainder of round `r`, which the
  pre-fix tree emits verbatim): if `rest` contains 62 or 63, return `None` (round `r + 1`'s
  head contains it) (fix-round amendment);
- `counted` = number of tokens in `head` other than 61; if `counted >= 2`, return the
  pair `(budget, n)` = `(counted - 1, p + 4)`; exactly two values (`run_cases` records
  `trip_position = n - 4`).

`p` is then the first over-budget token, the forced `\n` lands at `p + 1`, the pre-fix
tree (whole block, then `</think>`) emits its closer at the end of round `r` plus one,
and `n = p + 4` is the shortest emission that still contains the pre-fix closer for a
first-draft trip; it leaves no room for a re-opened thinking span to trip again (that
needs `<think>` plus one counted token after `</think>`). With no usable case, append
`dict(name=..., error="no usable budget trip")` as the stop cases do. Record
`budget`, `n` and `trip_position` in the case result. Expected pre-fix signatures, by
construction: overshoot 2 / 1 / 0, `forced_sequence_cut=True`. Update `EXPECT_PRE_FIX` and
`CASES`. The stop and length cases are unchanged.

`oracle.defects` (fix-round amendment): `forced_sequence_cut` compares windows instead of
counting forced-token ids anywhere in the emission, so a naturally sampled `\n` or
`</think>` after the trip cannot change it: with `a`, `b` the first forced index of ON and
OFF and `L = len(FORCED)`, `forced_sequence_cut = a is None or on[a:a + L] != off[b:b + L]`
(only when `b` is not None, as now). Update the docstring's `forced_sequence_cut` line.
`overshoot` is unchanged. The docstring of `budget_trip` must state the counting rules
correctly: `<think>` is not counted, `</think>` ends thinking, `\n` is counted.

### 4. Regression test (`test_singleton_quantized_narrow.py`, fork-only)

On Metal: for bits in {2, 4, 8}, K in {64, 128}, and `B, T` in {(1, 2), (2, 3)}, bf16
affine `nn.QuantizedLinear` (with and without bias) and inputs over enough seeds that the
baseline `gather_qmm` path is shown to mismatch on the installed MLX for 8-bit K = 128:
assert `singleton_quantized_linear` equals per-row decode bitwise. Also one K = 512 case
(gather path) to pin that wide K is unchanged. Fix-round amendment: add fp16, and output
widths N in {7, 12, 100} at K = 512 (4- and 8-bit), shown to mismatch on the baseline on
0.32.4.

### 5. `docs/upstream-bugs.md`

New item: the `singleton_quantized_linear` / `qmv_quad` defect, evidence above, the fork's
fix, and that upstream's `native_batch_linear` guard shows the same knowledge.

## Acceptance criteria

1. The five tests pass on mlx 0.32.4.
2. The regression test fails on the baseline `singleton_quantized_linear` and passes on
   the candidate on 0.32.4; it also passes on the candidate on 0.32.3 (the fix does not
   depend on the MLX version).
3. `.venv/bin/python dev/ci.py` green on 0.32.4 (style, eight audits, suite).
   Fork-marker audit passes with the new hunk.
4. Oracle: `test_the_fixed_tree_shows_none` and the classifier unit tests still pass.
5. Every reported test or gate result states the mlx version it ran on
   (`python -c "import mlx.core as mx; print(mx.__version__)"` in the same environment).
6. The repository `.venv` stays on mlx 0.32.4 and is never reinstalled. The one 0.32.3
   run (criterion 2) uses `/tmp/t1/v323/bin/python`, which matches `.venv` except mlx.

## Supported conditions

Apple Silicon with Metal, `.venv` (Python 3.12), mlx 0.32.4. bf16/fp16
affine quantized projections; nvfp4/mx modes keep their current behaviour except that
K <= 128 or N % 8 != 0 now also takes the per-row path (exact by construction).

## Non-goals

Making the suite green on mlx 0.32.3; changing `speculative/ops/linear.py`, `native_batch_linear`, the fork's exact affine
kernels, MLX, or any tolerance; T2 / T3 of the handoff; filing upstream.

## Open questions

None blocking. Recommendation recorded: fix inside `singleton_quantized_linear` (all
callers, `qwen4_exp` included) rather than guarding the speculative caller, because the
function's own contract is what fails.
