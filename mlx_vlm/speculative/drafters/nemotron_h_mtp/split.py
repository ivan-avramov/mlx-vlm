import argparse
import glob
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional

import mlx.core as mx
from safetensors import safe_open

from ....utils import get_model_path
from .config import NemotronHMTPConfig
from .nemotron_h_mtp import NemotronHMTPDraftModel


def _resolve_source_path(
    source: str, revision: Optional[str], force_download: bool
) -> Path:
    """Fetch only what extraction needs from a hub source: config + index
    first (cheap), then just the shard(s) the index says hold `mtp.*` keys.

    `get_model_path`'s own default `allow_patterns` pulls every `*.safetensors`
    in the repo -- correct for loading a full model, wrong for pulling 270
    tensors out of one shard of a 14-shard, ~60 GB checkpoint. A purely local
    `source` is unaffected: `get_model_path` returns it without touching the
    network, and `allow_patterns` is only consulted on the download branch.
    """
    probe_path = get_model_path(
        source,
        revision=revision,
        force_download=force_download,
        allow_patterns=["config.json", "model.safetensors.index.json"],
    )

    weight_map = _weight_map(probe_path)
    needed_shards = sorted(
        {filename for key, filename in weight_map.items() if key.startswith("mtp.")}
    )

    allow_patterns = ["config.json", "model.safetensors.index.json"]
    if needed_shards:
        allow_patterns += needed_shards
    else:
        # No index (unsharded checkpoint) or no `mtp.*` entries in it -- fall
        # back to fetching every shard so `_iter_mtp_keys`'s glob fallback has
        # something to scan; `split_nemotron_h_mtp` still raises cleanly below
        # if no `mtp.*` tensors turn up anywhere.
        allow_patterns.append("*.safetensors")

    return get_model_path(
        source,
        revision=revision,
        force_download=force_download,
        allow_patterns=allow_patterns,
    )


def _safetensor_files(model_path: Path) -> List[Path]:
    return [
        Path(path)
        for path in glob.glob(str(model_path / "*.safetensors"))
        if not path.endswith("consolidated.safetensors")
    ]


def _weight_map(model_path: Path) -> Dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    with open(index_path) as f:
        data = json.load(f)
    return data.get("weight_map", {})


def _config_mtp_file(model_path: Path) -> Optional[Path]:
    # Fork: fork-only, mirrors qwen3_5_mtp.split._config_mtp_file. Upstream has
    # no config.json fallback for locating a relocated MTP sidecar. Some
    # converters (mlx-optiq among them) move the sidecar out of the repo root
    # -- where a non-recursive `*.safetensors` glob would miss it -- and
    # record its real path in config.json instead.
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        config = json.load(f)
    mtp_file = config.get("mtp_file") or (config.get("mlx_lm_extra_tensors") or {}).get(
        "mtp_file"
    )
    if not mtp_file:
        return None
    path = model_path / mtp_file
    return path if path.exists() else None


def _iter_mtp_keys(model_path: Path) -> Iterable[tuple]:
    """Yield (shard, keys) for the shards/files that hold `mtp.*` tensors.

    On the real base checkpoint all 270 tensors sit in one indexed shard
    (`model-00014-of-00014.safetensors`), so the weight-map branch below is
    the one actually exercised end to end; the sidecar/glob fallbacks exist
    so this also works against a relocated or unindexed source.
    """
    weight_map = _weight_map(model_path)
    if weight_map:
        by_file: Dict[str, List[str]] = {}
        for key, filename in weight_map.items():
            if key.startswith("mtp."):
                by_file.setdefault(filename, []).append(key)
        if by_file:
            for filename, keys in by_file.items():
                yield model_path / filename, keys
            return

    sidecar = _config_mtp_file(model_path)
    candidates = [sidecar] if sidecar else _safetensor_files(model_path)

    for file in candidates:
        with safe_open(file, framework="mlx") as f:
            keys = [key for key in f.keys() if key.startswith("mtp.")]
        if keys:
            yield file, keys


def _load_selected_tensors(file: Path, keys: List[str]) -> Dict[str, mx.array]:
    """Lazily pull only `keys` out of `file` -- never loads the whole shard.

    `safe_open(...).get_tensor(key)` reads a single tensor's bytes off disk;
    with 270 of ~7000+ keys selected out of a 14-shard checkpoint, peak
    memory stays at "270 small/medium MoE tensors", not "one 4-5 GB shard".

    Falls back to `mx.load` (mirroring qwen3_5_mtp.split/deepseek_v4_mtp.split)
    when the installed `safetensors` package's `get_tensor` can't handle a
    dtype it hits -- observed on this box's `safetensors` build for bf16
    (`TypeError: data type 'bfloat16' not understood`). `mx.load` is still
    lazy (MLX defers materialization until the returned arrays are actually
    used), so selecting only `keys` out of the loaded dict keeps peak memory
    to the selected tensors, not the whole shard.
    """
    tensors = {}
    try:
        with safe_open(file, framework="mlx") as f:
            for key in keys:
                tensors[key] = mx.array(f.get_tensor(key))
    except (AttributeError, RuntimeError, TypeError):
        shard = mx.load(str(file))
        tensors = {key: shard[key] for key in keys}
    return tensors


def _quantize(
    weights: Dict[str, mx.array], bits: int, group_size: int
) -> Optional[dict]:
    """Affine-quantize projection weights in place, matching mlx-lm convert.

    Skips the MoE router gate (`mixer.gate.weight` stays full precision for
    routing stability -- same rationale as glm4_moe_lite_mtp's `noaux_tc`
    gate), norms (ndim==1), the fp32 router correction bias (no `.weight`
    suffix), and anything whose last dim doesn't divide evenly by
    `group_size`. Returns the quantization config to record, or ``None`` if
    nothing was quantized.
    """
    quantized_any = False
    for key in list(weights):
        if not key.endswith(".weight") or key.endswith("mixer.gate.weight"):
            continue
        weight = weights[key]
        if weight.ndim < 2 or weight.shape[-1] % group_size != 0:
            continue
        wq, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
        weights[key] = wq
        weights[key[: -len(".weight")] + ".scales"] = scales
        weights[key[: -len(".weight")] + ".biases"] = biases
        quantized_any = True

    if not quantized_any:
        return None
    return {"group_size": group_size, "bits": bits, "mode": "affine"}


def split_nemotron_h_mtp(
    source: str,
    output: str,
    *,
    revision: Optional[str] = None,
    block_size: Optional[int] = None,
    force_download: bool = False,
    q_bits: Optional[int] = None,
    q_group_size: int = 64,
) -> Path:
    """Write Nemotron-H native MTP tensors into a standalone drafter sidecar.

    Streams only the `mtp.*` tensors out of the source checkpoint (lazy
    per-tensor safetensors reads -- the source shard is never loaded whole),
    sanitizes them into this drafter's parameter layout, optionally
    affine-quantizes, and writes an `mtp.safetensors` sidecar plus a
    `config.json` that names it via `mtp_file` -- the same discovery contract
    `qwen3_5_mtp.split._config_mtp_file` reads, so a re-split (or any other
    `_iter_mtp_keys`-style reader) that treats this output directory as a
    source for a further extraction would find it too.
    """
    source_path = _resolve_source_path(source, revision, force_download)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(source_path / "config.json") as f:
        source_config = json.load(f)
    text_config = dict(source_config.get("text_config") or source_config)

    selected: Dict[str, mx.array] = {}
    for file, keys in _iter_mtp_keys(source_path):
        selected.update(_load_selected_tensors(file, keys))

    if not selected:
        raise ValueError(f"No mtp.* tensors found in {source_path}.")

    sanitize_context = SimpleNamespace(
        args=SimpleNamespace(n_routed_experts=text_config.get("n_routed_experts"))
    )
    selected = NemotronHMTPDraftModel.sanitize(sanitize_context, selected)

    quantization = None
    if q_bits is not None:
        quantization = _quantize(selected, q_bits, q_group_size)

    mx.eval(list(selected.values()))
    mtp_filename = "mtp.safetensors"
    mx.save_safetensors(
        str(output_path / mtp_filename),
        selected,
        metadata={"format": "mlx"},
    )

    mtp_block_types = text_config.get("mtp_layers_block_type") or ["attention", "moe"]
    depth = int(text_config.get("num_nextn_predict_layers", 1))
    draft_config = {
        "model_type": "nemotron_h_mtp",
        "mtp_file": mtp_filename,
        "text_config": text_config,
        "mtp_block_types": mtp_block_types,
        "block_size": int(block_size or depth + 1),
        "tie_word_embeddings": bool(text_config.get("tie_word_embeddings", False)),
    }
    if quantization is not None:
        draft_config["quantization"] = quantization
        draft_config["quantization_config"] = quantization

    with open(output_path / "config.json", "w") as f:
        json.dump(dict(sorted(draft_config.items())), f, indent=2)

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    ):
        src = source_path / name
        if src.exists():
            shutil.copy(src, output_path / name)

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split Nemotron-H native MTP tensors into a standalone MLX drafter sidecar."
    )
    parser.add_argument("--model", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--q-bits", type=int, default=None)
    parser.add_argument("--q-group-size", type=int, default=64)
    return parser


def main():
    args = build_parser().parse_args()
    output = split_nemotron_h_mtp(**vars(args))
    print(f"Wrote Nemotron-H MTP drafter to {output}")


if __name__ == "__main__":
    main()
