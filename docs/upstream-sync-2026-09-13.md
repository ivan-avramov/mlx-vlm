# Upstream sync — 2026-09-13

Status: merge `c5a6f97b` landed on local main after unit/static/cold review, compatibility screens and both shipped-state KV4 capacity ladders passed. Production activation and publication remain pending.

Baseline `420c01e1`; merged upstream target `45d6e125` (0.7.0). Both isolated test and serving-pinned runtime environments pass the full suite with MLX/MLX-Metal 0.32.2; the latter retains serving Transformers/audio/NumPy pins. The original runtime and stack M40/M41 artifacts remain unchanged.

## Fork disposition

| Area | Disposition | Preserved contract / verification |
|---|---|---|
| Dedicated speculative server loop | Retired in favor of upstream unified BatchGenerator | Explicit full-cap allocation parameter now crosses both batching constructors. Cached-session inline MTP remains separate because it owns session rewind. |
| Tool parsing and policy | Adapted to upstream tools package | Request parser override, protocol streaming, multiple calls and existing compatibility surfaces. |
| Manual APC materialization and old checkpoint gate | Retired in favor of upstream coordinator/checkpoint queue | Full-cap reservation before restore admission sizing. APC remains OFF in deployed settings. |
| Duplicate server sampler | Retired; shared generation sampler imported by identity | Preserve top-p/min-p/top-k filter order and per-request cached-path seeds; explicit compatibility reexports. Optional speculative batching retains upstream shared sampler semantics. |
| Qwen model-local verifier block | Retired in favor of upstream extracted verifier | Quantized slicing and transactions; EpiCache observation/RoPE hooks isolated in shared preparation. |
| TurboQuant attention/storage | Adapted to shared attention helpers and concrete storage owners | Remove shadowed constructor/storage copies, identical fused-quant override, duplicate APC method. Live tests cover factory selection, floors, shrink, snapshots, merge/extract and legacy metadata. |
| EpiCache wrapper snapshots | Adapted to upstream checkpoint protocol | Capture wrapper eviction/absolute-position metadata together with plain-KV storage. Explicitly reject unsupported inner cache classes. |
| Native hybrid MTP verification | Retained | Certified pre-norm hidden and one-pass recurrent snapshots differ from upstream DFlash verifier numerics/rollback layout. Narrow adapter dispatch preserves both contracts pending matched evidence. |
| Repaired sidecar packing and normalization | Retained; duplicate root-type fallback retired | Upstream now provides root-type fallback. Expert packing, normalization corrections and idempotence tests remain. |
| Full-cap preallocation and bounded attention | Retained | Runtime memory gate depends on allocation lifetimes and bounded prefill; no substitution based on short-generation peaks. |
| Test device setup | Adapted to scoped fixtures | Three import-time CPU setters poisoned unrelated tests. Real import/fixture tests prove isolation and restoration, including failure teardown. |
| Legacy regression guards | Retired or adapted with replacement evidence | Remove fixed upstream shadowing tripwire and obsolete private-predicate shape guard; preserve unique fallback/wiring tests and test current runtime behavior. |

## Validation record

- Original source suite crashed on MLX 0.32.0 and 0.32.2; collection-time CPU mutation was diagnosed separately from merge regressions.
- Device isolation: three failing real-import probes before the fix; ten behavioral import/fixture tests pass after it.
- Cache refactor: nine initial live-class failures plus later floor-loss regressions reproduced before repairs. Focused tests and final full-suite results are recorded in the integration evidence directory.
- Legacy guard adaptation: 73 tests pass; unique assertions retained where upstream coverage was incomplete.
- Full suite: 5357 passed, 10 skipped, 149 passing subtests (MLX/MLX-Metal 0.32.2). All eight static audits pass after the final provenance annotation; both supplementary reports reviewed. Independent cold review identified destination-floor overwrite; live regression and full suite verify its repair.
- Matched old/new five-case screens (six requests per arm) pass for `Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed` and `Qwen3.8-27B-mlx-uniform-4bit`, including native tools, vision, convergence and positive MTP counters. JSON reasoning differs while final answers match. Old-source/new-MLX controls reproduce original responses; a source-associated numerical change remains under assessment.
- Third-turn session probe reuses 1839 tokens with a correct answer and MTP active; old/new anchoring and persistence logic match.
- Both shipped-state KV4 capacity ladders pass at nominal 131072/196608/262144 with actual prompts 130783/196115/261449. Largest-rung MLX peaks: 41.1029 GB for `Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed`, 37.7970 GB for `Qwen3.8-27B-mlx-uniform-4bit`. Full cap/preallocation262144 preserved; these bounded memory probes are not quality recertification.
- Source static lead reviews closed 30 dropped-hunk leads and 23 untested-fork leads; lead counts are not proof of exhaustive behavior coverage.
- Matched quality/timing diagnostic proposals remain pending in the stack's C77/C78 decisions. Historical timing differences are not causal runtime estimates. Original stack submodules/environment remain on the prior runtime.
- Smoke success is not statistical recertification. Any numerical behavior change requires scoped assessment of affected certification evidence before activation.

Local logs and inventories: `$STACK_WORKDIR/upstream/2026-09-13/{logs,evidence}/`. Stack result tag `m43on-20260913` preserves the validated capacity records and redacted provenance. The experiment worktree remains frozen at the tested source commit; this checkpoint updates documentation only.
