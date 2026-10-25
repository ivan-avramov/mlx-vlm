# Converging `speculative/drafters/nemotron_h_mtp/` with upstream (#2528)

Status: **adopted, option A** (operator, 2026-10-25: "is A the quality-bias option,
if so, go for A"; decisions 1-3 as recommended). One design-review round (codex
gpt-6-astra, claude-opus-5-5) before adoption. Implemented in the stage C merge of the
2026-10-25 staged sync (`b8bf04c6`). Equivalence check (step 1): **bit-exact**; the
script is the appendix below.

## Goal

Upstream #2528 (`63fc9795`) adds its own `speculative/drafters/nemotron_h_mtp/`
package at the same path the fork created in `d06ed84e`, so the stage C merge has an
add/add conflict on all four files. Solving it means choosing one drafter package
that (a) satisfies upstream's registry, which now maps `nemotron_h` and
`nemotron_h_nano_omni` to `nemotron_h_mtp.split:NemotronHMTPSplitter`; (b) keeps
behaviour the fork actually depends on; and (c) leaves the smallest permanent
divergence from upstream, per AGENTS.md's "prefer upstream implementations; retire
superseded fork code".

## Facts established

1. **The round logic is the shared base's.** The fork's `NemotronHMTPDraftModel`
   (488 lines, standalone `nn.Module`) carries an older copy of
   `mtp_base.py::AutoregressiveMTPDraftModel`'s round methods. `mtp_base.py` is
   byte-identical in both trees. The fork copy uses the same algorithm, lacks the
   later `(logits_hidden, pre_hc_hidden)` split, and adds no-op `int`/array guards.
   The drafter math is identical: `eh_proj(concat(enorm(embed), hnorm(hidden)))`, the
   same `NemotronHBlock` stack, then `final_layernorm`.
2. **Upstream's class adds or changes behaviour beyond the round logic:**
   - `prefill_from_target_hidden` raises if `input_ids.shape[1] != hidden.shape[1]`.
     Upstream pairs this with target-side chunked-prefill gating
     (`supports_chunked_prefill`, which neither drafter defines). The fork's
     chunked-prefill policy differs (`language.py:842-845`).
   - `bind` validates that the target has a `NemotronHModel` backbone and matching
     `hidden_size`/`vocab_size` (`ValueError`, where the fork raised `AttributeError`).
   - Block-type validation moves into config (`ValueError`) and rejects `-` (MLP)
     blocks as well as mamba.
   - `cast_predicate` keeps `e_score_correction_bias` in fp32.
   - **There is no `runtime_block_size` cap.** The fork's config caps the effective
     block at `min(block_size, nextn_depth + 1)`, and `common.py::_dflash_block_total`
     and `mtp.py` read that cap. Upstream deliberately reuses the head recursively for
     more draft tokens. Standard splits write `block_size = depth + 1 = 2`, so the
     default is unchanged. Only an explicit larger `block_size` now drafts more.
3. **Fork-only behaviour, as distinct from fork-only code:**
   - The M29 H1 head profiler in `draft_block` (`MLX_VLM_MTP_PROFILE_HEAD`). The
     round profiler (`MLX_VLM_MTP_PROFILE`, `MTPRoundProfiler`, driven from
     `speculative/mtp.py`) is independent of the drafter and **stays**.
   - `_kv_valid_len` and `_position` are written in `set_shared_kv` and filtered in
     `filter_batch`, but nothing reads them. That is dead state.
4. **Parameter layout differs.** Upstream keeps HF's placement (`layers.0.enorm`,
   `layers.0.hnorm`, `layers.0.eh_proj`, `layers.<last>.final_layernorm`). The fork
   lifts those modules to the top level. A fork-split int4 sidecar therefore differs
   in six keys at least (`eh_proj.{weight,scales,biases}`, `enorm`, `hnorm`,
   `final_layernorm`), and upstream's strict load fails **loudly** on it.
5. **The registry is lazy.** `MTP_SPLITTERS` stores import strings.
   `get_mtp_splitter`, which `load_drafter` and `resolve_drafter_kind` call on
   **every** drafter config, raises when it resolves the `nemotron_h` entry against a
   module with no `NemotronHMTPSplitter`. The fork's own sidecar carries
   `text_config.model_type == "nemotron_h"`, so keeping the fork's `split.py` breaks
   loading the very artifact it was meant to keep.
6. **Splitting.** Upstream's `NemotronHMTPSplitter` plugs into the shared
   `split_mtp`, bundled-drafter loading (#2457) and `load_drafter`'s in-memory split.
   The fork's standalone CLI provides two things that path lacks:
   - It fetches only the config, the index and the `mtp.*` shard. Upstream's
     `split()` calls `get_model_path(source)` with no `allow_patterns`, so a Hub split
     downloads the whole checkpoint, about 60 GB for the BF16 Lightning source.
   - It honours `mtp_file` / `mlx_lm_extra_tensors.mtp_file` sidecars. Upstream
     selects via the index or root-level safetensors.

   Bundled loading does not reach the stack's target: the community 4-bit checkpoint
   drops the `mtp.*` block.
7. **Stack exposure.** `mlx_local_stack/main_models.yaml` serves
   `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit` **draft-OFF**. The only fork-format
   artifact is the probe-only `caslca/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit-mtp-drafter`,
   whose model card says "not certified, not recommended for serving".
8. **The hidden-state contract is target-side and pre-existing.** The fork's
   `nemotron_h` target feeds the drafter the pre-`norm_f` hidden (`d06ed84e`,
   `86f352b6`; markers at `language.py:581,767,818`). Upstream's target feeds the
   post-`norm_f` hidden. This divergence predates the sync and is not introduced or
   changed by any option here: the drafter math is identical (fact 1), so option A
   preserves today's measured pair (about 0.90 acceptance, M14/M29). The
   `nemotron_h_nano_omni` target is upstream's own and gets upstream's pair.
9. **Upstream dependencies outside the package.** Upstream's `config.py` reads
   `num_nextn_predict_layers`, `mtp_hybrid_override_pattern` and
   `mtp_layers_block_type` from `models/nemotron_h/config.py` (#2528, which
   auto-merges), and its `sanitize` calls `models/nemotron_h/language.py::Model.sanitize`
   (#2528's `block_configs` expert count, auto-merged). `language.py` also conflicts on
   one line: the fork's `NemotronHBlock(args, block_type, layer_idx=i)` against
   upstream's `NemotronHBlock(args.for_layer(index), block_type)`.

## Options

- **A. Converge on upstream's package byte-identical (recommended).** Details below.
- **B. A plus a load-time map for fork-format sidecars.** Rejected. It means at
  least six renames plus config validation, held permanently in an otherwise
  byte-identical upstream file, for a probe-only artifact.
- **C. Keep the fork drafter with a shim splitter.** Rejected on maintenance cost:
  about 830 lines duplicating the shared base, plus a second split path.

## Resolution (option A)

1. **Prove equivalence before deleting.** In the stage C worktree, before removing
   the fork class, build the fork drafter and upstream's from one config and one set
   of random weights, mapped between the two layouts. Assert that `_forward_hidden`,
   `prefill_from_target_hidden` followed by `draft_block` (greedy), and
   `accept_verified_tokens` with uniform acceptance produce identical outputs and
   caches. Record the script and the result in the merge commit.
2. Take upstream's `__init__.py`, `config.py`, `nemotron_h_mtp.py` and `split.py`
   byte-identical.
3. In `models/nemotron_h/language.py`, resolve the `NemotronHBlock(...)` line as a
   union: `args.for_layer(index)` plus the fork's `layer_idx=index` (M34). Keep
   #2528's `config.py` and `sanitize` hunks. Leave the pre-norm feed (fact 8) as
   marked fork divergence, and re-word its markers so they no longer refer to a fork
   drafter class.
4. **Tests.**
   - Port upstream's Nemotron drafter tests into the fork-owned `test_speculative.py`
     (C104), adapted to its fixtures where upstream's harness (`model_case`) is
     absent, and re-pin the blob.
   - Move the drafter fixtures that `test_mtp_profile.py` imports out of
     `test_nemotron_h_mtp.py` into a helper rebuilt on upstream's class, so the
     round-profiler tests keep collecting.
   - In `test_nemotron_h_mtp.py`, keep only behaviour upstream's class must also
     have that the ported tests do not cover (kind mapping, bind error, cache only
     for attention sub-layers, uniform-acceptance error), with upstream's exception
     types.
   - Retire the fork-layout, sanitize, split-CLI and `runtime_block_size` tests.
5. **Head profiler: retire.** Delete `MTPHeadProfiler`, `head_profiler_from_env`,
   `reset_head_profiler`, the `_after_report` flush, and the five `TestHeadProfiler`
   tests. Keep `MTPRoundProfiler` and its tests.
6. **Bookkeeping.**
   - Drop the `# Fork:` marker on `drafters/__init__.py`'s `"nemotron_h_mtp": "mtp"`
     line, which upstream now carries too.
   - Update AGENTS.md's note that cites "the nemotron_h_mtp splitter" as untested
     fork code.
   - Prune exclusion entries that referenced the fork splitter.
   - Record the retirements (layout, standalone CLI with lazy shards and `mtp_file`,
     `runtime_block_size` cap, head profiler) in the merge commit and `memory.md`.
7. All eight audits, the full suite, the upstream-copy differential run, and the
   stage change review.

## Acceptance criteria

- The four package files are byte-identical to `b8bf04c6`.
  `check_body_divergence.py --file` on `models/nemotron_h/{config,language}.py` shows
  no #2528 line `gone`.
- The equivalence check in step 1 passes.
- `get_mtp_splitter(source_config=<nemotron_h config>)` returns upstream's splitter.
  `load_drafter(<tiny nemotron_h checkpoint with mtp.*>)` loads, `draft.reset(target)`
  binds the target's embeddings and head, and a draft round runs.
- An MTP round on a `nemotron_h` target with a prompt longer than
  `prefill_step_size` succeeds. This checks fact 2's prefill guard against the fork's
  chunked-prefill policy. If it fails, the fix goes in the fork's target-side policy,
  not in upstream's drafter.
- `test_mtp_profile.py`'s round-profiler tests still collect and pass.
- Every retired fork test is either covered by a test the fork's suite collects or
  listed with the reason it no longer applies.
- All eight audits are green vs `b8bf04c6`, and the full suite is green.

## Non-goals

- Changing the pre-norm hidden contract (fact 8). It is a residual: whether
  upstream's post-norm feed is better for Nemotron-H needs a measurement on real
  weights, outside this merge.
- Serving Nemotron with a drafter (it stays draft-OFF).
- Re-splitting or republishing the probe artifact.

## Decisions for the operator

1. The published probe artifact stops loading on new fork builds, loudly (strict
   load). Recommended: accept. Option B's map is a permanent divergence for a
   probe-only artifact, and a re-split needs the 60 GB BF16 source.
2. Retire the M29 head profiler. Recommended: retire. M29 is concluded at its gate,
   and keeping the hook means overriding `draft_block` in an upstream file or the
   shared base.
3. Retire the standalone split CLI's lazy shard fetch and `mtp_file` support.
   Recommended: retire. No stack consumer uses either, and both are re-addable to the
   shared `MTPSplitter` later if ever needed (as an upstream-shaped change, not a
   Nemotron-only one).

## Appendix: the step-1 equivalence check

Run against the stage C merge tree with the fork's three package files restored from
`9fe943f6` into a temporary `speculative/drafters/_fork_nemotron_h_mtp_tmp/` package
(deleted afterwards). Result, 2026-10-25, mlx 0.32.4: every check `OK`, then
`EQUIVALENT`. Every parameter was re-randomized, so no zero/one init could mask a
mismatch.

```python
"""Fork NemotronHMTPDraftModel (HEAD 9fe943f6) vs upstream's (b8bf04c6): same weights, same inputs."""
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_vlm.models.nemotron_h.config import ModelConfig as NHConfig
from mlx_vlm.models.nemotron_h import Model as NHModel
from mlx_vlm.speculative.drafters import _fork_nemotron_h_mtp_tmp as fork
from mlx_vlm.speculative.drafters import nemotron_h_mtp as up

H, V = 32, 32
base = dict(model_type="nemotron_h", vocab_size=V, hidden_size=H, intermediate_size=32,
    num_hidden_layers=2, max_position_embeddings=64, num_attention_heads=2,
    num_key_value_heads=2, attention_bias=False, mamba_num_heads=2, mamba_head_dim=8,
    mamba_proj_bias=False, ssm_state_size=8, conv_kernel=4, n_groups=1, mlp_bias=False,
    layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True,
    hybrid_override_pattern="*E", moe_intermediate_size=32,
    moe_shared_expert_intermediate_size=32, moe_latent_size=None, n_group=1,
    n_routed_experts=4, n_shared_experts=1, topk_group=1, num_experts_per_tok=2,
    norm_topk_prob=True, routed_scaling_factor=1.0, num_nextn_predict_layers=1,
    mtp_layers_block_type=["attention", "moe"])
text = NHConfig(**base)
mx.random.seed(0)
target = NHModel(text)
fd = fork.Model(fork.ModelConfig(text_config=text, block_size=2, mtp_block_types=["attention", "moe"]))
ud = up.Model(up.ModelConfig(text_config=text, block_size=2))
mx.random.seed(1)
fork_params = dict(tree_flatten(fd.parameters()))
for k in list(fork_params):  # re-randomize every tensor so no zero/one init hides a mismatch
    fork_params[k] = mx.random.normal(fork_params[k].shape).astype(fork_params[k].dtype) * 0.2
fd.load_weights(list(fork_params.items()), strict=True)
last = len(ud.layers) - 1
def to_up(k):
    for n in ("enorm", "hnorm", "eh_proj"):
        if k.startswith(n + "."):
            return f"layers.0.{k}"
    if k.startswith("final_layernorm."):
        return f"layers.{last}.{k}"
    return k
up_params = {to_up(k): v for k, v in fork_params.items()}
assert set(up_params) == set(k for k, _ in tree_flatten(ud.parameters())), sorted(set(up_params) ^ set(k for k, _ in tree_flatten(ud.parameters())))
ud.load_weights(list(up_params.items()), strict=True)

def same(a, b, what):
    assert a.shape == b.shape and mx.array_equal(a, b).item(), what
    print("OK", what, tuple(a.shape))

# 1. single forward
emb = mx.random.normal((1, 3, H)); hid = mx.random.normal((1, 3, H))
same(fd._forward_hidden(emb, hid, fd.make_cache()), ud._forward_hidden(emb, hid, None, ud.make_cache())[0], "_forward_hidden")

# 2. prefill -> draft_block (greedy) -> accept (partial) -> draft_block, on bound drafters
greedy_sampler = lambda logits: mx.argmax(logits, axis=-1)
for d in (fd, ud):
    d.reset(target)
ids = mx.array([[3, 7, 11, 5]]); h = mx.random.normal((1, 4, H))
for d in (fd, ud):
    d.prefill_from_target_hidden(ids, h, 9, greedy_sampler, greedy=True)
same(fd._seed_token, ud._seed_token, "seed token after prefill")
outs = [d.draft_block(mx.array([9]), h[:, -1:], None, 4, greedy_sampler, greedy=True) for d in (fd, ud)]
same(outs[0], outs[1], "draft_block tokens (block 4)")
vh = mx.random.normal((1, 4, H))
for d in (fd, ud):
    d.accept_verified_tokens(vh, outs[0], 1, [int(outs[0][0, 0]), 13], greedy_sampler, greedy=True)
same(fd._seed_token, ud._seed_token, "seed token after accept(1)")
for cf, cu in zip(fd._cache, ud._cache):
    same(cf.keys[..., : cf.offset, :], cu.keys[..., : cu.offset, :], "kv cache keys after accept")
outs2 = [d.draft_block(mx.array([13]), vh[:, 1:2], None, 3, greedy_sampler, greedy=True) for d in (fd, ud)]
same(outs2[0], outs2[1], "second draft_block tokens")
print("EQUIVALENT")
```
