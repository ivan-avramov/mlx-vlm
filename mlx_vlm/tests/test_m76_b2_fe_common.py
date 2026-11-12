"""M76 B2 test-writer helpers (fork-only, dev; CPU only; no server, no model, no GPU). Holds no test.

Written from ``.fe-material/m76-b2-fork-dev-tools.md`` (SPEC v1) before the implementation, by a test writer who did not
write it. Registries are built the way the stack commits them (B1 edit model, campaign ``f45``: X = ``kv_bits: 4`` + an added
``tq4_attention: m72_exact`` line; V = the same with ``m72_v1``; XO = X with the three MTP lines commented). The worker argv is
spelled by mlx-serve's REAL ``process_manager._build_command`` (the sibling ``mlx-serve`` copy), run in its own venv python in a
subprocess with only the vision executable path pointed at the stack root's ``.venv/bin/mlx_vlm.server``.
"""

# isort: skip_file

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

FORK = (
    Path(__file__).resolve().parents[2]
)  # the fork checkout (whatever its directory is named): <fork>/mlx_vlm/tests/this file
DEV = FORK / "dev"
sys.path.insert(0, str(DEV))
SIBLING = (
    FORK.parent
)  # where a sibling ``mlx-serve`` copy would sit (absent in a clean clone: the guard skips)

from p369_phase2 import registry_states  # noqa: E402

from mlx_vlm.tests import test_p475a_kv_arm as kv  # noqa: E402

MLX_SERVE_SRC = SIBLING / "mlx-serve" / "src"
MLX_SERVE_PY = (
    SIBLING / "mlx-serve" / ".venv" / "bin" / "python"
)  # never resolved: the venv symlink is the interpreter

FIXTURE = kv.FIXTURE
PICK = kv.PICK_A  # Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed
UNIFORM = kv.UNIFORM
PICK_D, PICK_C, SERVED_X = kv.PICK_D, kv.PICK_C, kv.SERVED_X
MODES = {"X": "m72_exact", "V": "m72_v1"}
MTP_KEYS = ("draft_kind", "draft_model", "mtp_verify_scan")


# ------------------------------------------------------------------ registries as the stack commits them
def add_mode(text, entry, literal):
    """``text`` with a ``    tq4_attention: <literal>`` line right after ``entry``'s ``kv_bits`` line (B1's ``add`` edit)."""
    lines = text.split("\n")
    start, end = registry_states._span(lines, entry)
    hits = [i for i in range(start, end) if re.match(r"^    kv_bits:( |$)", lines[i])]
    assert len(hits) == 1, (entry, hits)
    lines.insert(hits[0] + 1, f"    tq4_attention: {literal}")
    return "\n".join(lines)


def kv_text(entry, bits, scheme=None, mode=None, base=None):
    """``base`` (default the golden fixture) with ``entry``'s KV keys set; ``mode`` is the RAW yaml literal of the new line."""
    keys = dict(kv_bits=bits)
    if scheme is not None:
        keys["kv_quant_scheme"] = scheme
    text = kv.set_kv(FIXTURE if base is None else base, entry, **keys)
    return text if mode is None else add_mode(text, entry, mode)


def comment_mtp(text, entry):
    """B1's ``comment`` edit: each of the entry's three MTP lines prefixed ``# `` (plain, no note)."""
    lines = text.split("\n")
    start, end = registry_states._span(lines, entry)
    for key in MTP_KEYS:
        hits = [
            i for i in range(start, end) if re.match(rf"^    {key}:( |$)", lines[i])
        ]
        assert len(hits) == 1, (key, hits)
        lines[hits[0]] = "    # " + lines[hits[0]][4:]
    return "\n".join(lines)


X_TEXT = kv_text(PICK, 4, mode="m72_exact")
V_TEXT = kv_text(PICK, 4, mode="m72_v1")
XO_TEXT = comment_mtp(X_TEXT, PICK)  # B1's XO: X with the three MTP lines commented
XO_ENTRY_OFF_TEXT = kv_text(
    PICK, 4, mode="m72_exact", base=registry_states.apply(FIXTURE, {PICK})
)  # as ENTRY_OFF spells it
VO_TEXT = comment_mtp(V_TEXT, PICK)
ARM_TEXTS = {"X": X_TEXT, "V": V_TEXT, "XO": XO_TEXT}
DELTAS = {
    "X": {"kv_bits": (0, 4), "tq4_attention": (None, "m72_exact")},
    "V": {"kv_bits": (0, 4), "tq4_attention": (None, "m72_v1")},
    "XO": {"kv_bits": (0, 4), "tq4_attention": (None, "m72_exact")},
}


def entry_of(text, name=PICK):
    return dict(registry_states.settings(text)[name])


# ------------------------------------------------------------------ mlx-serve's real argv builder
_BUILD = """
import json, os, pathlib, sys
sys.path.insert(0, {src!r})
os.environ['MLX_SERVE_CONFIG'] = {reg!r}
from mlx_serve import config, process_manager as pm
pm._MLX_VLM_SERVER = pathlib.Path({exe!r})
print(json.dumps(pm._build_command(config._load()[0][{name!r}])))
"""


class BuilderRefused(RuntimeError):
    """mlx-serve's own ``_load`` / ``_build_command`` refused the registry entry."""


_SIBLING = []  # the one guard's answer, computed once: [None] runnable, [reason] not


def require_mlx_serve():
    """The ONE guard of every test that needs mlx-serve's real ``_build_command``: skips, with one reason, when the sibling
    ``mlx-serve`` copy is not runnable here (its venv python or ``src`` missing, or ``mlx_serve`` / its dependencies cannot be
    imported there).  Checked once per session; never a failure for that cause."""
    if not _SIBLING:
        reason = None
        if not (MLX_SERVE_PY.exists() and MLX_SERVE_SRC.is_dir()):
            reason = f"the sibling mlx-serve copy is not runnable here: {MLX_SERVE_PY}"
        else:
            probe = (
                f"import sys; sys.path.insert(0, {str(MLX_SERVE_SRC)!r}); "
                "from mlx_serve import config, process_manager"
            )
            try:
                got = subprocess.run(
                    [str(MLX_SERVE_PY), "-I", "-c", probe],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                reason = f"the sibling mlx-serve copy is not runnable here: {type(exc).__name__}: {exc}"
            else:
                if got.returncode != 0:
                    tail = got.stderr.strip().splitlines()[-1:] or [""]
                    reason = f"the sibling mlx-serve copy is not runnable here: cannot import mlx_serve: {tail[0]}"
        _SIBLING.append(reason)
    if _SIBLING[0] is not None:
        pytest.skip(_SIBLING[0])


def real_command(stack, registry_path, name):
    """mlx-serve's ``_build_command`` for ``name`` on ``registry_path``: the executable (``<stack>/.venv/bin/mlx_vlm.server``)
    then the flags, exactly as the stack serves them."""
    require_mlx_serve()
    exe = Path(stack) / ".venv" / "bin" / "mlx_vlm.server"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.touch()  # _build_command only checks that the vision executable exists
    code = _BUILD.format(
        src=str(MLX_SERVE_SRC),
        reg=str(registry_path),
        exe=str(exe),
        name=name,
    )
    got = subprocess.run(
        [str(MLX_SERVE_PY), "-I", "-c", code], capture_output=True, text=True
    )
    if got.returncode != 0:
        tail = got.stderr.strip().splitlines()[-1:] or [""]
        raise BuilderRefused(
            tail[0]
        )  # mlx-serve ran and refused the registry (the guard above passed)
    return json.loads(got.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------ a fake stack root for orchestrate (git repo, fake venvs)
AXES = ("math500", "humanevalplus", "mbppplus", "cjudge")
# the 20 C84 items the native16 bundles written by ``make_stack_root`` hold, as ``<axis>/<row id>``
ITEMS = [dict(id=f"{axis}/{axis}-{i}") for axis in AXES for i in range(5)]


def _script(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def make_stack_root(tmp, label, registry_text):
    """A clean git repo ``tmp/label`` committing ``main_models.yaml`` + the pick's native16 C84 bundle; ``.venv/bin/python``
    runs mlx-serve's real ``_build_command`` (the wrapper only re-points the vision executable) and ``.venv-bench/bin/python``
    answers the C84 reconstruction with ``ITEMS`` and logs its arguments to ``<root>/.bench-calls``.
    """
    require_mlx_serve()  # the fake venv python runs the sibling mlx-serve
    root = Path(tmp) / label
    root.mkdir(parents=True)
    (root / "main_models.yaml").write_bytes(registry_text.encode("utf-8"))
    (root / ".gitignore").write_text(".venv/\n.venv-bench/\n.bench-calls\n")
    (root / "benchmark" / "bench").mkdir(parents=True)
    (root / "benchmark" / "bench" / "__init__.py").write_text("")
    for axis in AXES:  # orchestrate.C84_AXES: one native16 bundle per axis
        bundle = (
            root
            / "benchmark"
            / "results"
            / PICK
            / f"{axis}.c84bundle-after-native16-20260914.jsonl"
        )
        bundle.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            dict(id=f"{axis}-{i}", sampler_seed=1000 + i, prompt_tokens=500 + i)
            for i in range(5)
        ]
        bundle.write_text("".join(json.dumps(r) + "\n" for r in rows))
    wrapper = (
        "#!/bin/sh\n"
        # the caller's code sets MLX_SERVE_CONFIG BEFORE it imports mlx_serve: the executable is patched right after that import
        f"exec '{MLX_SERVE_PY}' -I -c 'import sys\n"
        f'sys.path.insert(0, "{MLX_SERVE_SRC}")\n'
        'imp = "from mlx_serve import config, process_manager\\n"\n'
        f'patch = "process_manager._MLX_VLM_SERVER = __import__(\\"pathlib\\").Path(\\"{root}/.venv/bin/mlx_vlm.server\\")\\n"\n'
        "assert imp in sys.argv[1]\n"
        'exec(sys.argv[1].replace(imp, imp + patch))\' "$2"\n'
    )
    _script(root / ".venv" / "bin" / "python", wrapper)
    (
        root / ".venv" / "bin" / "mlx_vlm.server"
    ).touch()  # _build_command only checks the vision executable exists
    items = root / ".items.json"
    items.write_text(json.dumps(ITEMS))
    bench = (
        "#!/bin/sh\n"
        f"echo \"$3|$4|$5|$6\" >> '{root}/.bench-calls'\n"
        f"cat '{items}'\n"
    )
    _script(root / ".venv-bench" / "bin" / "python", bench)
    (root / ".gitignore").write_text(
        ".venv/\n.venv-bench/\n.bench-calls\n.items.json\n"
    )
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid")
    env.update(GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid")
    for args in (
        ["init", "-q"],
        ["add", "-A"],
        ["-c", "commit.gpgsign=false", "commit", "-q", "--no-verify", "-m", label],
    ):
        subprocess.run(
            ["git", "-C", str(root), *args], check=True, env=env, capture_output=True
        )
    return root
