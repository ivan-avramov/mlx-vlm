# Upstream bugs found here — our own record of what comes from where

Bugs that live in `Blaizzy/mlx-vlm` itself, not in this fork. Each was found while
resolving a merge or draining an audit, and each is reproducible against
`upstream/main` with no fork code involved.

**This file is bookkeeping, and that is the whole of its purpose.** Its job is to
stop each of these being re-investigated as a fork problem every time it surfaces in
an audit, to record which `.symbol-exclusions` entries exist *because* upstream has a
defect, and to explain the one divergence we carry *because* upstream is wrong
(item 4). Two consequences, both deliberate:

* **These are not to be filed with upstream.** Engaging upstream's issue tracker is
  out of scope for this fork (decided 2026-08-10). An earlier handoff invented that
  task; it was never asked for and nothing else in `AGENTS.md`, `upstream-gaps.md` or
  `memory.md` ever proposed it. Don't re-propose it.
* **These are not to be patched locally either.** A local fix to upstream code is a
  new permanent conflict site, and for items 1-3 our copy is byte-identical to
  upstream's — which is the state worth keeping. Item 4 is the exception and is
  already diverged deliberately; its marker and guard say so.

So the correct action on every entry below is: read it, understand why our tree looks
the way it does, and move on.

Line numbers are `upstream/main` at base `ffd7aeff` unless stated. Verified
2026-08-10; none had been filed by anyone else as of that date
(`gh search issues --repo Blaizzy/mlx-vlm`), which is recorded only because it
confirms they are genuinely unreported rather than known-and-wontfix.

---

## 1. `BatchTurboQuantKVCache.zero_row_tail` is defined twice; the first copy is dead

`mlx_vlm/turboquant.py`, `class BatchTurboQuantKVCache`:

* lines **6230-6244** — a `_map_state`-based version with a nested `_z(arr, ndim)`
  helper that indexes 3-D and 4-D state differently.
* lines **6271-6273** — a `_zero_state_row_tail`-based version.

Python binds the later definition, so the first is unreachable, and with it the
nested `_z`. Nothing else in the file references either.

Not to be confused with the `state`/`meta_state` pairs in `TurboQuantKVCache`,
`BatchTurboQuantKVCache` and `HybridQuantKVCache`, which are also two defs each and
are legitimate `@property` / `@x.setter` pairs. `zero_row_tail` carries no decorators
on either copy — this is real shadowing.

```python
import ast, subprocess
from collections import Counter
src = subprocess.run(["git","show","upstream/main:mlx_vlm/turboquant.py"],
                     capture_output=True, text=True).stdout
for n in ast.walk(ast.parse(src)):
    if isinstance(n, ast.ClassDef):
        for m in n.body:
            if getattr(m, "name", "") == "zero_row_tail":
                print(n.name, m.lineno, m.end_lineno,
                      [ast.unparse(d) for d in m.decorator_list])
# BatchTurboQuantKVCache 6230 6244 []
# BatchTurboQuantKVCache 6271 6273 []
```

This fork carries only the live copy, byte-identical to upstream's second one, which
is why `.symbol-exclusions` has entries for the dead sibling and its `_z`. Deleting
the dead copy upstream would let both entries be pruned here.

## 2. `TestLagunaProcessor` is defined twice, so one of its tests never runs

`mlx_vlm/tests/test_processors.py`, lines **2323-2330** and **2770-2851**. Our copy
of this file is byte-identical to upstream's, so this is upstream's bug.

The first class holds one test, `test_chat_template_owns_laguna_special_tokens`. The
second shadows it, so that test is never collected — confirmed by
`pytest --collect-only`, which reports only the second class's two tests:

```
tests/test_processors.py::TestLagunaProcessor::test_auto_processor_patch_intercepts_laguna
tests/test_processors.py::TestLagunaProcessor::test_from_pretrained_loads_fast_tokenizer_directly
```

Renaming the first class (say to `TestLagunaChatTemplate`) collects all three.

A green suite is what makes this durable: nothing fails, and the count looks right.
It is also why a symbol the shadowed test exercises can go missing without any test
noticing. (`utils.should_add_special_tokens` was the example when this was first
recorded; it is present in both trees today, so the shadowing is currently costing
coverage rather than hiding a missing symbol.)

## 3. `skip_special_tokens` is inert on the normal text path

Three call sites set it for tool-calling requests:

* `mlx_vlm/server/openai.py:994`
* `mlx_vlm/server/openai.py:1704`
* `mlx_vlm/server/anthropic.py:489`

all of the form `gen_args.skip_special_tokens = False`, so that tool-call and channel
markers survive detokenisation and the tool parser can see them.

But `args.skip_special_tokens` is read in exactly two places in
`mlx_vlm/server/generation.py` — lines **1906** and **1971** — and both are inside
`ResponseGenerator._generate_diffusion`. The autoregressive text path never reads it,
so all three assignments are no-ops for every non-diffusion model. The field defaults
to `True` at `generation.py:734`.

```python
# enclosing scope of every `args.skip_special_tokens` read, upstream/main
# generation.py:1906 -> ResponseGenerator._generate_diffusion
# generation.py:1971 -> ResponseGenerator._generate_diffusion
```

Confirmed identical in this fork (our `anthropic.py:489` is byte-identical to
upstream's), so `gen_args.skip_special_tokens = False` does nothing in **both**
trees. Recorded as `docs/upstream-gaps.md` item 9(b) and deliberately not patched
here.

## 4. Uniform KV quantization raises on any sliding-window model

`mlx_vlm/generate/common.py::maybe_quantize_kv_cache`'s final loop — the uniform,
non-hybrid, non-TurboQuant path — is:

```python
for index, layer_cache in enumerate(prompt_cache):
    if (
        hasattr(layer_cache, "to_quantized")
        and layer_cache.offset >= quantized_kv_start
    ):
        prompt_cache[index] = layer_cache.to_quantized(...)
```

`hasattr(..., "to_quantized")` is the wrong gate. `RotatingKVCache` **has** that
method, and it is a stub:

```python
def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
    raise NotImplementedError("RotatingKVCache Quantization NYI")
```

So the call is made and raises. Every other branch of the same function guards
`RotatingKVCache` explicitly (`if isinstance(entry, cache.RotatingKVCache): return
entry` in both `hybridize` and `quantize_entry`) — only the uniform path does not,
which is what makes this look like an oversight rather than a design choice.

Reachable for any sliding-window model (Gemma-family SWA layers produce a
`RotatingKVCache`) run with `--kv-bits` set on the default `uniform` scheme.
Reproduced directly against upstream's function:

```python
import ast, subprocess, textwrap
import mlx.core as mx
from mlx_vlm.generate import common as ourmod
from mlx_vlm.models import cache

src = subprocess.run(["git", "show", "upstream/main:mlx_vlm/generate/common.py"],
                     capture_output=True, text=True).stdout
tree, lines = ast.parse(src), src.splitlines()
fn = next(textwrap.dedent("\n".join(lines[n.lineno - 1:n.end_lineno]))
          for n in tree.body if getattr(n, "name", "") == "maybe_quantize_kv_cache")
ns = dict(vars(ourmod)); exec(fn, ns)

entry = cache.RotatingKVCache(max_size=32, keep=4)
entry.update_and_fetch(mx.zeros((1, 2, 8, 4)), mx.zeros((1, 2, 8, 4)))
ns["maybe_quantize_kv_cache"]([entry], quantized_kv_start=0, kv_group_size=64, kv_bits=4)
# NotImplementedError: RotatingKVCache Quantization NYI
```

**This fork already does not have the bug**, and that is the one entry here where our
divergence is load-bearing rather than incidental: our uniform loop skips
`RotatingKVCache` with a `continue`. It had no test until 2026-08-10 — see
`TestUniformKvQuantSkipsRotatingCaches` in `tests/test_dropped_upstream_guards.py`,
which also pins the premise (that `to_quantized` still raises), so it cannot become
vacuous if mlx ever implements it.

The fix upstream is one line: the same `isinstance` guard its other two branches
already use.
