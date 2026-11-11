"""M72 1.3 A3 (off is today), A4 (a) / (c) (on is consistent, same process) and A7 (R1 range check) on a synthetic
Qwen3.5-family model with the kernels' attention shape (24 / 4 / 256, bf16; the last full-attention layer native16).

A3 runs the same fixed-seed generation twice in fresh interpreters: once on the PARENT commit's ``mlx_vlm`` (extracted
with ``git archive``: the commit before the one that added ``mlx_vlm/tq4_fast.py``, or HEAD while it is uncommitted) and
once on this tree with the mode absent; tokens and logits must be byte-identical, TQ4 and native16, MTP OFF and ON.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Metal GPU required for the TQ4 kernels"
)

from test_mtp_exactness import (  # noqa: E402
    PICKS,
    deployed_sampling,
    fake_model,
    mtp_drafter,
    scripted,
)
from test_tq4_impl_support import (  # noqa: E402
    bit_equal,
    caches_bit_equal,
    fill_cache,
    lm24,
    prefill,
    prompt,
    tq4_cache,
)

from mlx_vlm import tq4_attention as t4  # noqa: E402
from mlx_vlm import tq4_fast as f  # noqa: E402
from mlx_vlm import turboquant as tq  # noqa: E402
from mlx_vlm.generate import ar  # noqa: E402

FORK = Path(__file__).resolve().parents[2]
PROMPT_LEN = 2300  # > 2048: the 2-pass decode, D2 and d2_verify side
CAP = 4096


class Delta:
    def __enter__(self):
        self.before = f.route_counts()
        return self

    def __exit__(self, *exc):
        after = f.route_counts()
        self.d = {
            k: after[k] - self.before.get(k, 0)
            for k in after
            if after[k] != self.before.get(k, 0)
        }
        return False


def wrap(lm):
    return type("M", (), {"config": lm.config, "language_model": lm})()


# ================================================================== A3: off is today (vs the parent commit)
A3_SCRIPT = textwrap.dedent("""
    import json, sys
    import mlx.core as mx
    import numpy as np
    from mlx_vlm.generate import ar
    from mlx_vlm.generate.common import maybe_quantize_kv_cache
    from mlx_vlm.models.qwen3_5.config import ModelConfig, TextConfig, VisionConfig
    from mlx_vlm.models.qwen3_5.language import LanguageModel
    from mlx_vlm.speculative.drafters.qwen3_5_mtp import ModelConfig as MTPConfig
    from mlx_vlm.speculative.drafters.qwen3_5_mtp import Qwen3_5MTPDraftModel
    import copy
    from types import SimpleNamespace

    out_dir = sys.argv[1]
    mx.random.seed(0)
    cfg = TextConfig(model_type="qwen3_5_text", hidden_size=128, intermediate_size=256,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, num_hidden_layers=4, num_attention_heads=24, rms_norm_eps=1e-6, vocab_size=64,
        num_key_value_heads=4, max_position_embeddings=1 << 20, full_attention_interval=2, head_dim=256,
        tie_word_embeddings=False)
    vision = VisionConfig(model_type="qwen3_5", depth=1, hidden_size=8, intermediate_size=16, out_hidden_size=128,
        num_heads=1, in_channels=3, patch_size=4, temporal_patch_size=2, spatial_merge_size=1,
        num_position_embeddings=4)
    lm = LanguageModel(cfg, config=ModelConfig(text_config=cfg, vision_config=vision, model_type="qwen3_5"))
    lm.model.norm.weight = lm.model.norm.weight * 8.0
    lm.set_dtype(mx.bfloat16)
    mx.eval(lm.parameters())
    tc = copy.deepcopy(lm.config.text_config)
    tc.mtp_num_hidden_layers = 1
    mx.random.seed(1)
    drafter = Qwen3_5MTPDraftModel(MTPConfig(text_config=tc, block_size=3))
    drafter.set_dtype(mx.bfloat16)
    mx.eval(drafter.parameters())
    rng = np.random.default_rng(3)
    ids = mx.array(rng.integers(1, 60, size=(1, %(n)d)), dtype=mx.int32)
    result = {}

    def manual(kv):
        cache = lm.make_cache()
        if kv:
            maybe_quantize_kv_cache(cache, quantized_kv_start=0, kv_group_size=64, kv_bits=4,
                kv_quant_scheme="turboquant", max_kv_size=%(cap)d, kv_prealloc_tokens=%(cap)d)
        for s in range(0, ids.shape[1], 512):
            out = lm(ids[:, s:s + 512], cache=cache)
            mx.eval(out.logits)
        logits = [out.logits[0, -1]]
        tok = int(mx.argmax(out.logits[0, -1]).item())
        toks = [tok]
        for _ in range(24):
            out = lm(mx.array([[tok]], dtype=mx.int32), cache=cache)
            mx.eval(out.logits)
            logits.append(out.logits[0, -1])
            tok = int(mx.argmax(out.logits[0, -1]).item())
            toks.append(tok)
        arr = mx.stack(logits).astype(mx.float32)
        mx.eval(arr)
        return toks, np.array(arr)

    def fake(lm):
        def emb(input_ids, pixel_values, mask=None, **kw):
            return SimpleNamespace(inputs_embeds=lm.model.embed_tokens(input_ids), to_dict=lambda: {})
        config = copy.copy(lm.config)
        config.eos_token_id = None
        return SimpleNamespace(language_model=lm, get_input_embeddings=emb, config=config)

    def stepped(kv, draft):
        extra = dict(draft_model=drafter, draft_kind="mtp", draft_block_size=3) if draft else {}
        kvkw = dict(kv_bits=4, kv_quant_scheme="turboquant", quantized_kv_start=0,
                    max_kv_size=%(cap)d, kv_prealloc_tokens=%(cap)d) if kv else {}
        toks, lps = [], []
        for t, lp in ar.generate_step(ids, fake(lm), None, None, max_tokens=24, prompt_cache=lm.make_cache(),
                                      prefill_step_size=512, temperature=0.0, **kvkw, **extra):
            toks.append(int(t))
            if lp is not None:  # speculative rounds may yield tokens without logprobs
                lps.append(lp.astype(mx.float32))
        if not lps:
            return toks, np.zeros((0,), np.float32)
        arr = mx.stack(lps)
        mx.eval(arr)
        return toks, np.array(arr)

    for name, fn in (("manual_tq4", lambda: manual(True)), ("manual_native16", lambda: manual(False)),
                     ("step_tq4_mtp_off", lambda: stepped(True, False)),
                     ("step_tq4_mtp_on", lambda: stepped(True, True)),
                     ("step_native16_mtp_on", lambda: stepped(False, True))):
        toks, arr = fn()
        np.save(f"{out_dir}/{name}.npy", arr)
        result[name] = toks
    result["tq4_modules_imported"] = sorted(m for m in sys.modules if m.startswith("mlx_vlm.tq4_"))
    print(json.dumps(result))
    """)


def _parent_ref() -> str:
    try:
        added = subprocess.run(
            [
                "git",
                "-C",
                str(FORK),
                "log",
                "--diff-filter=A",
                "--format=%H",
                "--",
                "mlx_vlm/tq4_fast.py",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout.split()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"git history unavailable ({type(exc).__name__}); A3 needs the parent commit"
        )
    return f"{added[-1]}^" if added else "HEAD"


def _run_tree(root: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    script = out / "a3_generate.py"
    script.write_text(A3_SCRIPT % {"n": PROMPT_LEN, "cap": CAP})
    env = dict(os.environ, PYTHONPATH=str(root))
    env.pop("MLX_VLM_TQ4_ATTENTION", None)
    r = subprocess.run(
        [sys.executable, str(script), str(out)],
        cwd=str(out),
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert r.returncode == 0, r.stderr[-4000:]
    import json

    return json.loads(r.stdout.strip().splitlines()[-1])


def test_a3_mode_absent_is_byte_identical_to_the_parent_commit(tmp_path):
    ref = _parent_ref()
    parent = tmp_path / "parent"
    parent.mkdir()
    archive = tmp_path / "parent.tar"
    subprocess.run(
        [
            "git",
            "-C",
            str(FORK),
            "archive",
            "--format=tar",
            "-o",
            str(archive),
            ref,
            "mlx_vlm",
        ],
        check=True,
        timeout=120,
    )
    with tarfile.open(archive) as t:
        t.extractall(parent)  # noqa: S202 - our own repository's archive
    assert not (
        parent / "mlx_vlm" / "tq4_fast.py"
    ).exists(), f"{ref} already has the mode"
    old = _run_tree(parent, tmp_path / "out_parent")
    new = _run_tree(FORK, tmp_path / "out_tree")
    assert new["tq4_modules_imported"] == []  # the mode absent imports nothing of it
    for name in (
        "manual_tq4",
        "manual_native16",
        "step_tq4_mtp_off",
        "step_tq4_mtp_on",
        "step_native16_mtp_on",
    ):
        assert new[name] == old[name], name
        a = np.load(tmp_path / "out_parent" / f"{name}.npy")
        b = np.load(tmp_path / "out_tree" / f"{name}.npy")
        assert a.shape == b.shape and a.tobytes() == b.tobytes(), name


# ================================================================== A4 (a): MTP OFF, mode off vs mode on from one state
def _greedy(lm, cache, first, n):
    tok, toks, logits = first, [], []
    for _ in range(n):
        out = lm(mx.array([[tok]], dtype=mx.int32), cache=cache)
        mx.eval(out.logits)
        logits.append(out.logits[0, -1])
        tok = int(mx.argmax(out.logits[0, -1]).item())
        toks.append(tok)
    arr = mx.stack(logits)
    mx.eval(arr)
    return toks, arr


@pytest.fixture(scope="module")
def target():
    return lm24(logit_scale=8.0)


@pytest.mark.parametrize("name", t4.MODES)
def test_a4a_decode_mode_on_equals_mode_off_from_the_same_state(target, name):
    lm = target
    model = wrap(lm)
    t4.apply_to_model(model, None)
    ids = prompt(PROMPT_LEN)
    snap, restore = tq4_cache(lm, CAP), tq4_cache(lm, CAP)
    out = prefill(lm, snap, ids)
    prefill(
        lm, restore, ids
    )  # the snapshot: an identical mode-off state, verified bit for bit
    assert caches_bit_equal(snap, restore)
    first = int(mx.argmax(out.logits[0, -1]).item())
    try:
        off_toks, off_logits = _greedy(lm, snap, first, 64)
        t4.apply_to_model(model, name)
        with Delta() as d:
            on_toks, on_logits = _greedy(lm, restore, first, 64)
    finally:
        t4.apply_to_model(model, None)
    assert on_toks == off_toks
    assert bit_equal(on_logits, off_logits)
    assert d.d == {"d2": 64, "native:decode": 64}
    assert len(set(off_toks)) > 1  # a non-degenerate continuation


# ================================================================== A4 (c): m72_exact, MTP ON emits MTP OFF's tokens
def _generate(lm, drafter, n, block, **kw):
    extra = {}
    if drafter is not None:
        extra = dict(draft_model=drafter, draft_kind="mtp", draft_block_size=block)
    stream = ar.generate_step(
        prompt(PROMPT_LEN),
        fake_model(lm),
        None,
        None,
        max_tokens=n,
        prompt_cache=lm.make_cache(),
        prefill_step_size=512,
        kv_bits=4,
        kv_quant_scheme="turboquant",
        quantized_kv_start=0,
        max_kv_size=CAP,
        kv_prealloc_tokens=CAP,
        **extra,
        **kw,
    )
    return [int(t) for t, _ in stream]


@pytest.fixture
def drafter(target):
    """A fresh (deterministic) drafter per test: ``scripted`` patches its proposals in place."""
    return mtp_drafter(target, block=3)


@pytest.mark.parametrize("sampling", ("greedy", "seeded"))
@pytest.mark.parametrize("acceptance", ("scripted", "real"))
@pytest.mark.parametrize("block", (3, 4))
def test_a4c_m72_exact_mtp_on_emits_mtp_off_tokens(
    target, drafter, sampling, acceptance, block
):
    lm = target
    model = wrap(lm)
    params = (
        {"temperature": 0.0}
        if sampling == "greedy"
        else dict(deployed_sampling(PICKS[0]), seed=11)
    )
    t4.apply_to_model(model, "m72_exact")
    try:
        off = _generate(lm, None, 256, block, **params)
        script = None
        if (
            acceptance == "scripted"
        ):  # patches the drafter's proposals in place (high acceptance)
            script = scripted(
                drafter, lambda p: off[p] if p < len(off) else None, high=True, vocab=64
            )
        with Delta() as d:
            on = _generate(lm, drafter, 256, block, **params)
    finally:
        t4.apply_to_model(model, None)
    assert len(on) == len(off) == 256
    assert on == off
    assert d.d.get("d2_verify", 0) > 0 and d.d.get("mma", 0) == 0
    assert d.d.get("native:verify", 0) > 0
    assert not any(k.startswith("fallback:") for k in d.d)
    if script is not None:
        assert script.rounds  # verify rounds really ran on the scripted proposals


def test_a4c_m72_v1_routes_verify_through_the_matrix_unit_kernel(target, drafter):
    """m72_v1 is joint-tolerance (not exact): routing only; the divergence is measured on the real model (A6)."""
    lm = target
    model = wrap(lm)
    t4.apply_to_model(model, "m72_v1")
    try:
        with Delta() as d:
            toks = _generate(lm, drafter, 64, 3, temperature=0.0)
    finally:
        t4.apply_to_model(model, None)
    assert len(toks) == 64
    assert d.d.get("mma", 0) > 0 and d.d.get("d2_verify", 0) == 0
    assert d.d.get("p2s", 0) > 0 and d.d.get("native:verify", 0) > 0


# ================================================================== A7: R1 range check
def test_a7_passes_on_the_synthetic_model_and_records_values():
    lm = lm24()
    out = t4.check_fp16_range(wrap(lm))
    assert out["max_q_bound"] == pytest.approx(16.0)  # sqrt(256) * max|ones|
    assert set(out["layers"]) == {
        "model.layers.1.self_attn",
        "model.layers.3.self_attn",
    }
    assert 0.0 < out["max_codebook"] <= 1.0


@pytest.mark.parametrize("gamma,ok", [(1024.0, True), (1032.0, False), (5000.0, False)])
def test_a7_query_bound_refuses_a_planted_violation(gamma, ok):
    lm = lm24()
    attn = lm.model.layers[3].self_attn
    w = mx.ones_like(attn.q_norm.weight)
    attn.q_norm.weight = mx.concatenate([w[:-1], mx.array([gamma], dtype=w.dtype)])
    if ok:
        assert t4.check_fp16_range(wrap(lm))["max_q_bound"] == pytest.approx(16 * gamma)
    else:
        with pytest.raises(t4.Tq4AttentionError, match=r"layers\.3\.self_attn"):
            t4.check_fp16_range(wrap(lm))


def test_a7_codebook_bound_refuses_a_planted_violation_in_a_cache():
    lm = lm24()
    cache = fill_cache(64, 256)
    assert t4.check_fp16_range(wrap(lm), [cache])["max_codebook"] <= 1.0
    cb = cache.value_codec.codebook
    cache.value_codec.codebook = mx.concatenate(
        [cb[:-1], mx.array([1.0001], dtype=cb.dtype)]
    )
    with pytest.raises(t4.Tq4AttentionError, match="codebook"):
        t4.check_fp16_range(wrap(lm), [cache])


def _planted(cb):
    return mx.concatenate([cb[:-1], mx.array([1.5], dtype=cb.dtype)])


def test_a7_codebook_bound_refuses_without_caches(monkeypatch):
    lm = lm24()
    real = tq._codebook
    monkeypatch.setattr(tq, "_codebook", lambda dim, bits: _planted(real(dim, bits)))
    with pytest.raises(t4.Tq4AttentionError, match="codebook"):
        t4.check_fp16_range(wrap(lm))
