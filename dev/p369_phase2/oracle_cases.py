"""P369 phase 2 E7 oracle — the scripted cases (Q30).

Run by ``oracle.py`` INSIDE one isolated checkout: ``python -I oracle_cases.py <checkout> <cases.json>``. Only
``<checkout>`` (an extracted ``mlx_vlm`` tree of one commit) and its tests directory are put on ``sys.path``
(asserted); the cases use that tree's own ``mlx_vlm`` and ``tests/test_mtp_exactness`` fixtures and print one
JSON document: per case, plain (OFF) and MTP (ON) emissions, cache offsets, a valid-state digest and the next
turn's emission. ``run_cases`` is also imported in process by the live-classifier test (current code), with
an ``Observer`` told where each ON request ends.
"""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PARAMS = dict(temperature=0.5, top_p=0.95, top_k=20, min_p=0.0)
BLOCK = 3
THINK, CLOSE, NEWLINE = 61, 62, 63
SPECIAL = {THINK: "<think>", CLOSE: "</think>", NEWLINE: "\n"}
# Emitted positions of a full-accept MTP run (block 3): 0 = the prefill's first token; round r covers the
# drafts at 1 + 3r and 2 + 3r and the verifier bonus at 3 + 3r.
WHERE = {
    "first_token": lambda r: 0,
    "first_draft": lambda r: 1 + BLOCK * r,
    "second_draft": lambda r: 2 + BLOCK * r,
    "bonus": lambda r: 3 + BLOCK * r,
}


class _Detokenizer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.text, self.last_segment = "", ""

    def add_token(self, token, skip_special_token_ids=()):
        self.last_segment = f"<{int(token)}>"
        self.text += self.last_segment

    def finalize(self):
        self.last_segment = ""


def _tokenizer(stop_ids):
    import mlx.core as mx

    from mlx_vlm.utils import StoppingCriteria

    class Tokenizer:
        pad_token, eos_token = "<pad>", "<eos>"

        def __init__(self):
            self.stopping_criteria = StoppingCriteria(
                [], None, additional_eos_token_ids=list(stop_ids)
            )
            self.detokenizer = _Detokenizer()

        def decode(self, ids):
            return " ".join(SPECIAL.get(int(i), str(int(i))) for i in ids)

        def encode(self, text, add_special_tokens=False):
            return [{v: k for k, v in SPECIAL.items()}[text]]

        def __call__(self, prompts, **kwargs):
            text = prompts[0] if isinstance(prompts, list) else prompts
            ids = [int(t) for t in text.split()]
            return SimpleNamespace(
                input_ids=mx.array([ids]), attention_mask=mx.ones((1, len(ids)))
            )

    return Tokenizer()


def _digest(state):
    import mlx.core as mx
    import numpy as np

    h = hashlib.sha256()
    for leaf in state.cache or []:
        h.update(type(leaf).__name__.encode())
        h.update(repr(getattr(leaf, "offset", None)).encode())
        for a in leaf.state:
            if a is None:
                continue
            mx.eval(a)
            h.update(np.array(a.astype(mx.float32)).tobytes())
    return h.hexdigest()


def _snapshot(state):
    if state.cache is None:
        return dict(offset=None, digest=None, token_ids=None)
    offsets = [int(c.offset) for c in state.cache if hasattr(c, "offset")]
    return dict(
        offset=max(offsets),
        digest=_digest(state),
        token_ids=[int(t) for t in state.token_ids or []],
    )


def _run(model, ids, state, *, drafter=None, n, stop_ids=(), budget=None, seed=5):
    import mlx.core as mx

    from mlx_vlm.generate.dispatch import stream_generate

    kwargs = dict(
        input_ids=mx.array([ids], dtype=mx.int32),
        prompt_cache_state=state,
        max_tokens=n,
        prefill_step_size=None,
        seed=seed,
        **PARAMS,
    )
    if drafter is not None:
        kwargs.update(draft_model=drafter, draft_kind="mtp", draft_block_size=BLOCK)
    if budget is not None:
        kwargs.update(thinking_budget=budget, enable_thinking=True)
    final = list(stream_generate(model, _tokenizer(stop_ids), "", **kwargs))[-1]
    return [int(t) for t in final.token_ids], final.finish_reason


def stop_position(free, where):
    """The first position of that kind whose token does not occur earlier (so the stop lands there)."""
    for r in range(len(free)):
        p = WHERE[where](r)
        if p >= len(free) - 1:
            break
        if free[p] not in free[:p]:
            return p
        if where == "first_token":
            break
    return None


def run_cases(cases, observer=None):
    from test_mtp_exactness import PROMPT, fake_model, hybrid_lm, mtp_drafter, scripted

    from mlx_vlm.generate.common import PromptCacheState

    lm = hybrid_lm(logit_scale=4.0)
    model = fake_model(lm)
    out = []
    for case in cases:
        budget = case.get("budget")
        ids = list(PROMPT) + ([THINK] if budget is not None else [])
        n = int(case["n"])
        free, _ = _run(model, ids, PromptCacheState(), n=n + BLOCK + 1)
        position = stop_position(free, case["where"]) if case.get("where") else None
        if case.get("where") and position is None:
            out.append(dict(name=case["name"], error="no usable stop position"))
            continue
        stop_ids = [free[position]] if position is not None else []
        result = dict(name=case["name"], free=free, stop_position=position)
        for arm in ("off", "on"):
            state, drafter = PromptCacheState(), None
            if arm == "on":
                drafter = mtp_drafter(lm)
                scripted(
                    drafter,
                    lambda p: free[p] if p < len(free) else None,
                    high=True,
                    vocab=64,
                )
                if observer is not None:
                    observer.begin_request(case["name"])
            emitted, finish = _run(
                model,
                ids,
                state,
                drafter=drafter,
                n=n,
                stop_ids=stop_ids,
                budget=budget,
            )
            if arm == "on" and observer is not None:
                observer.end_request(len(emitted), finish)
            snap = _snapshot(state)  # BEFORE the next turn mutates the session
            follow = (snap["token_ids"] or list(ids) + emitted) + [21, 22, 23]
            nxt, _ = _run(model, follow, state, drafter=drafter, n=6, seed=9)
            result[arm] = dict(emitted=emitted, finish=finish, next=nxt, **snap)
        out.append(result)
    return out


def main():
    checkout = Path(sys.argv[1]).resolve()
    sys.path[:0] = [str(checkout), str(checkout / "mlx_vlm" / "tests")]
    import mlx_vlm

    assert Path(mlx_vlm.__file__).resolve().is_relative_to(checkout), mlx_vlm.__file__
    print(json.dumps(run_cases(json.loads(Path(sys.argv[2]).read_text()))))


if __name__ == "__main__":
    main()
