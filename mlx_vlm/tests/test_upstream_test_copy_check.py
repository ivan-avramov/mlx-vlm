"""Behavioural contract for T2; all executed copies are synthetic, pure Python."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

_DEV = Path(__file__).resolve().parents[2] / "dev/check_upstream_test_copies.py"
_SPECULATIVE = "mlx_vlm/tests/test_speculative.py"
_SERVER = "mlx_vlm/tests/test_server.py"
_GENERATE = "mlx_vlm/tests/test_generate.py"
_CASE = _SERVER + "::TestExample::test_case"
_BAD = _CASE + "[bad - case]"
_GOOD_SOURCE = "def test_passes():\n    assert 2 + 2 == 4\n"
_PARAM_SOURCE = """import pytest

def test_passes():
    assert 2 + 2 == 4

class TestExample:
    @pytest.mark.parametrize('value', [1, 0], ids=['good', 'bad - case'])
    def test_case(self, value):
        assert value == 1, 'T2 assertion marker'
"""


@pytest.fixture(scope="module")
def audit():
    assert _DEV.is_file(), f"missing {_DEV}"
    spec = importlib.util.spec_from_file_location("_upstream_copies_under_test", _DEV)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture
def make_repo(tmp_path, monkeypatch):
    # Only synthetic repositories are committed; the checkout is never committed.
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "")

    def make(files):
        root = tmp_path / "repo"
        root.mkdir()
        _git(root, "init", "-q")
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        _git(root, "add", ".")
        _git(
            root,
            "-c",
            "user.name=T2 fixture",
            "-c",
            "user.email=t2@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "Synthetic upstream copies",
        )
        return root, _git(root, "rev-parse", "HEAD").strip()

    return make


def _assert_clean(root):
    assert _git(root, "status", "--porcelain", "--untracked-files=all") == ""
    assert list(root.rglob("test_zz_up_*")) == []


def _run(audit, root, ref, copies):
    return audit.run_copies(ref, repo_root=root, copies=copies, python=sys.executable)


def test_copies_contract(audit):
    """Preserves an invariant: the gate covers exactly the three owned copies."""
    assert audit.COPIES == (_SPECULATIVE, _SERVER, _GENERATE)


def test_baseline_reads_reasons_comments_and_collection_ids(audit, tmp_path):
    """Exercises new behaviour: UTF-8 reasons and collection-error baselines."""
    path = tmp_path / "baseline"
    path.write_text(
        "\n  # header\n"
        f"  {_CASE}[*-chat-*]  # reviewed: café # additional detail\n"
        f"{_GENERATE} # reviewed collection error\n",
        encoding="utf-8",
    )
    assert audit.load_baseline(path) == {
        _CASE + "[*-chat-*]": "reviewed: café # additional detail",
        _GENERATE: "reviewed collection error",
    }


@pytest.mark.parametrize(
    "entry",
    [
        _CASE,
        _CASE + " #",
        _CASE + " #   ",
        "mlx_vlm/tests/test_unknown.py::test_x # reason",
        "tests/test_server.py::test_x # reason",
        "prefix/" + _CASE + " # reason",
        _CASE + " # a different reason",
    ],
    ids=[
        "no-separator",
        "empty-reason",
        "spaces-reason",
        "unknown-file",
        "wrong-root",
        "prefixed-path",
        "duplicate",
    ],
)
def test_baseline_hard_errors_name_file_and_line(audit, tmp_path, entry):
    """Preserves an invariant: malformed exclusions never silently excuse tests."""
    path = tmp_path / "baseline"
    path.write_text(f"# header\n{_CASE} # valid reason\n{entry}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        audit.load_baseline(path)
    message = str(caught.value)
    assert str(path) in message
    assert re.search(r"\b3\b", message.replace(str(path), "")), message


def test_comment_only_baseline_excuses_nothing(audit, tmp_path):
    """Preserves an invariant: an empty reviewed baseline is valid."""
    path = tmp_path / "baseline"
    path.write_text("\n # comment\n", encoding="utf-8")
    assert audit.load_baseline(path) == {}


@pytest.mark.parametrize(
    "pattern,suffix,expected",
    [
        ("test_*case", "test_case", True),
        ("test_*case", "test_long_case", True),
        ("test_case[*-chat-*]", "test_case[x-chat-y]", True),
        ("test_case[*-chat-*]", "test_case[-chat-]", True),
        ("test_case[*-chat-*]", "test_case[x-completion-y]", False),
        ("test_case[ab]", "test_case[ab]", True),
        ("test_case[ab]", "test_casea", False),
        ("test_case[ab]", "test_case[b]", False),
        ("test_case[]", "test_case[]", True),
        ("test_case", "test_case", True),
        ("test_case", "test_case[chat]", True),
        ("test_case", "test_case[]", True),
        ("test_case", "test_case[chat]suffix", False),
        ("test_case", "test_case[chat", False),
        ("test_case", "test_case_extra[chat]", False),
        ("test_case[x]", "test_case[x][y]", False),
        ("test_case[x]", "test_case", False),
        ("test_case", "prefix_test_case", False),
        ("test_case", "test_case_suffix", False),
        ("test_?.+^$", "test_?.+^$", True),
        ("test_?", "test_a", False),
        ("test_.", "test_a", False),
    ],
)
def test_matches_boundaries(audit, pattern, suffix, expected):
    """Exercises new behaviour: only star is special; parameter widening is bounded."""
    prefix = _SERVER + "::TestExample::"
    assert audit.matches(prefix + pattern, prefix + suffix) is expected
    assert not audit.matches(prefix + pattern, "other/" + prefix + suffix)


def test_compare_reports_sorted_new_ids_and_stale_patterns(audit):
    """Reproduces a failure the spec fixes: an unreviewed parameter failure escapes."""
    known = _CASE + "[*-chat-*]"
    stale_a = _GENERATE + "::test_a"
    stale_z = _SPECULATIVE + "::test_z"
    new_a = _GENERATE + "::test_new"
    new_z = _CASE + "[x-completion-y]"
    failing = {_CASE + "[a-chat-b]", new_z, new_a}
    baseline = {stale_z: "old", known: "reviewed chat", stale_a: "old"}
    assert audit.compare(failing, baseline) == (
        sorted([new_z, new_a]),
        sorted([stale_z, stale_a]),
    )
    assert failing == {_CASE + "[a-chat-b]", new_z, new_a}
    assert baseline == {stale_z: "old", known: "reviewed chat", stale_a: "old"}


def test_compare_empty_sides_and_overlapping_patterns(audit):
    """Preserves an invariant: neither unmatched side nor overlapping matches vanish."""
    baseline = {_CASE: "all parameters", _BAD: "specific parameter"}
    assert audit.compare({_BAD}, baseline) == ([], [])
    assert audit.compare(set(), baseline) == ([], sorted(baseline))
    assert audit.compare({_BAD, _GENERATE}, {}) == (sorted([_BAD, _GENERATE]), [])
    assert audit.compare(set(), {}) == ([], [])


def test_run_copies_maps_exact_failure_and_collection_ids(audit, make_repo):
    """Reproduces a failure the spec fixes: upstream failures must reach the gate."""
    root, ref = make_repo(
        {
            _SERVER: _PARAM_SOURCE,
            _GENERATE: "import t2_deliberately_absent_module\n",
        }
    )
    result = _run(audit, root, ref, (_SERVER, _GENERATE))
    assert set(result) == {_BAD, _GENERATE}
    assert "T2 assertion marker" in result[_BAD]
    assert "t2_deliberately_absent_module" in result[_GENERATE]
    assert all("\n" not in message for message in result.values())
    _assert_clean(root)


def test_run_copies_success_overwrites_leftovers_and_removes_bytecode(audit, make_repo):
    """Preserves an invariant: success cleans both copied source and stale bytecode."""
    root, ref = make_repo({_SERVER: _GOOD_SOURCE})
    directory = root / "mlx_vlm/tests"
    (directory / "test_zz_up_test_server.py").write_text(
        "raise RuntimeError('leftover')\n"
    )
    cache = directory / "__pycache__"
    cache.mkdir()
    (cache / "test_zz_up_test_server.cpython-312.pyc").write_bytes(b"stale")
    assert _run(audit, root, ref, (_SERVER,)) == {}
    _assert_clean(root)


def test_run_copies_uses_ref_tests_against_working_tree(audit, make_repo):
    """Preserves an invariant: committed assertions exercise the working library."""
    root, ref = make_repo(
        {
            _SERVER: (
                "from t2_subject import value\n"
                "def test_value():\n"
                "    assert value == 2\n"
            ),
            "mlx_vlm/t2_subject.py": "value = 1\n",
        }
    )
    (root / "mlx_vlm/t2_subject.py").write_text("value = 2\n")
    (root / _SERVER).write_text("raise RuntimeError('not the ref copy')\n")
    before = _git(root, "diff")
    assert _run(audit, root, ref, (_SERVER,)) == {}
    assert _git(root, "diff") == before
    assert list(root.rglob("test_zz_up_*")) == []


def test_run_copies_refuses_tracked_destination(audit, make_repo):
    """Preserves an invariant: the audit cannot overwrite a tracked destination."""
    destination = "mlx_vlm/tests/test_zz_up_test_server.py"
    sentinel = "# tracked sentinel\n"
    root, ref = make_repo({_SERVER: _GOOD_SOURCE, destination: sentinel})
    with pytest.raises(audit.CopyRunError):
        _run(audit, root, ref, (_SERVER,))
    assert (root / destination).read_text() == sentinel
    assert _git(root, "status", "--porcelain") == ""


def test_run_copies_unknown_ref_is_error(audit, make_repo):
    """Preserves an invariant: an unresolved ref cannot masquerade as success."""
    root, _ = make_repo({_SERVER: _GOOD_SOURCE})
    with pytest.raises(audit.CopyRunError):
        _run(audit, root, "t2-ref-that-does-not-exist", (_SERVER,))
    _assert_clean(root)


def test_run_copies_missing_second_copy_cleans_first(audit, make_repo):
    """Preserves an invariant: partial copy preparation leaves no source behind."""
    root, ref = make_repo({_SERVER: _GOOD_SOURCE})
    with pytest.raises(audit.CopyRunError):
        _run(audit, root, ref, (_SERVER, _GENERATE))
    _assert_clean(root)


@pytest.mark.parametrize("exit_code", [2, 3, 4, 5])
def test_run_copies_abnormal_pytest_exit_is_error(audit, make_repo, exit_code):
    """Preserves an invariant: abnormal runner termination is never green."""
    root, ref = make_repo(
        {
            _SERVER: _GOOD_SOURCE,
            "mlx_vlm/tests/conftest.py": (
                "def pytest_sessionfinish(session, exitstatus):\n"
                "    print('T2 abnormal exit marker')\n"
                f"    session.exitstatus = {exit_code}\n"
            ),
        }
    )
    with pytest.raises(audit.CopyRunError) as caught:
        _run(audit, root, ref, (_SERVER,))
    assert "T2 abnormal exit marker" in str(caught.value)
    _assert_clean(root)


def test_run_copies_zero_items_in_one_copy_is_error(audit, make_repo):
    """Preserves an invariant: a passing sibling cannot hide an uncollected copy."""
    root, ref = make_repo({_SERVER: _GOOD_SOURCE, _GENERATE: "# no tests\n"})
    with pytest.raises(audit.CopyRunError):
        _run(audit, root, ref, (_SERVER, _GENERATE))
    _assert_clean(root)


@pytest.fixture
def cli(audit, tmp_path, monkeypatch):
    baseline = tmp_path / "controlled-baseline"
    baseline.write_text("# initially empty\n", encoding="utf-8")
    monkeypatch.setattr(audit, "BASELINE_FILE", baseline)
    calls = []

    def install(result=None, error=None):
        def run(ref, *, repo_root, copies, python):
            calls.append((ref, repo_root, copies, python))
            if error is not None:
                raise error
            return dict(result or {})

        monkeypatch.setattr(audit, "run_copies", run)

    return baseline, calls, install


def test_main_excuses_failures_and_warns_without_gating_stale(audit, cli, capsys):
    """Exercises new behaviour: reviewed failures pass and stale entries only warn."""
    baseline, calls, install = cli
    stale = _GENERATE + "::test_old"
    baseline.write_text(f"{_CASE} # reviewed\n{stale} # obsolete\n")
    install({_BAD: "boom", _CASE + "[other]": "boom"})
    assert audit.main(["--upstream-ref", "t2-reviewed-ref"]) == 0
    out = capsys.readouterr().out
    assert re.search(
        r"2 failing, 2 excused by \d+ baseline entries, 0 new vs t2-reviewed-ref", out
    )
    assert "warning:" in out.lower()
    assert stale in out
    assert len(calls) == 1
    assert calls[0][0] == "t2-reviewed-ref"
    assert tuple(calls[0][2]) == audit.COPIES


def test_main_new_failure_returns_one_with_exact_id_and_message(audit, cli, capsys):
    """Reproduces a failure the spec fixes: an unexcused assertion must fail CI."""
    _, calls, install = cli
    install({_BAD: "T2 crash message"})
    assert audit.main(["--upstream-ref", "t2-new-ref"]) == 1
    out = capsys.readouterr().out
    assert "1 failing, 0 excused by 0 baseline entries, 1 new vs t2-new-ref" in out
    assert _BAD in out
    assert "T2 crash message" in out
    assert len(calls) == 1


def test_main_run_error_returns_two_on_stderr(audit, cli, capsys):
    """Preserves an invariant: inability to execute has its own nonzero result."""
    _, _, install = cli
    install(error=audit.CopyRunError("T2 cannot execute"))
    assert audit.main(["--upstream-ref", "t2-error-ref"]) == 2
    assert "T2 cannot execute" in capsys.readouterr().err


def test_main_list_is_sorted_and_never_loads_baseline(audit, cli, monkeypatch, capsys):
    """Exercises new behaviour: listing failures works without a valid baseline."""
    baseline, calls, install = cli
    baseline.unlink()
    install({_BAD: "ignored", _GENERATE: "ignored"})

    def forbidden_load(path):
        pytest.fail("--list consulted the baseline")

    monkeypatch.setattr(audit, "load_baseline", forbidden_load)
    assert audit.main(["--upstream-ref", "t2-list-ref", "--list"]) == 0
    assert capsys.readouterr().out.splitlines() == sorted([_BAD, _GENERATE])
    assert calls[0][0] == "t2-list-ref"


def test_main_invalid_baseline_is_nonzero(audit, cli):
    """Preserves an invariant: malformed review evidence never yields success."""
    baseline, _, install = cli
    baseline.write_text(_CASE + " #\n")
    install({})
    try:
        result = audit.main(["--upstream-ref", "t2-invalid-ref"])
    except SystemExit as error:
        assert error.code is not None and error.code != 0
    else:
        assert isinstance(result, int) and result != 0


def test_main_defaults_to_upstream_main(audit, cli):
    """Preserves an invariant: the documented default ref reaches the runner."""
    _, calls, install = cli
    install({})
    assert audit.main([]) == 0
    assert calls[0][0] == "upstream/main"


# Third delivery: the frozen bytes above are preserved; review regressions follow.
_TWO_FAILURES = """def test_a_fail():
    assert False, 'first failure'

def test_b_fail():
    assert False, 'second failure'

def test_pass():
    assert True
"""


@pytest.mark.parametrize("options", ["-x", "--collect-only", "-k test_pass"])
def test_r1_inherited_options_do_not_shorten_run(
    audit, make_repo, monkeypatch, options
):
    """Reproduces a failure the spec fixes: shell options hide upstream failures."""
    root, ref = make_repo({_SERVER: _TWO_FAILURES})
    monkeypatch.setenv("PYTEST_ADDOPTS", options)
    result = _run(audit, root, ref, (_SERVER,))
    assert set(result) == {_SERVER + "::test_a_fail", _SERVER + "::test_b_fail"}
    _assert_clean(root)


def test_r1_inherited_plugins_are_removed(audit, make_repo, monkeypatch):
    """Preserves an invariant: shell plugin injection cannot change the audit run."""
    root, ref = make_repo({_SERVER: _GOOD_SOURCE})
    monkeypatch.setenv("PYTEST_PLUGINS", "t2_nonexistent_injected_plugin")
    assert _run(audit, root, ref, (_SERVER,)) == {}
    _assert_clean(root)


@pytest.mark.parametrize("options", ["-x", "--collect-only", "-k test_pass"])
def test_r1_configured_incomplete_run_is_error(audit, make_repo, options):
    """Reproduces a failure the spec fixes: config can stop, collect or deselect."""
    root, ref = make_repo(
        {_SERVER: _TWO_FAILURES, "pytest.ini": f"[pytest]\naddopts = {options}\n"}
    )
    with pytest.raises(audit.CopyRunError):
        _run(audit, root, ref, (_SERVER,))
    _assert_clean(root)


def test_r1_hook_deselection_is_error(audit, make_repo):
    """Reproduces a failure the spec fixes: deselection need not come from flags."""
    root, ref = make_repo(
        {
            _SERVER: _TWO_FAILURES,
            "mlx_vlm/tests/conftest.py": (
                "def pytest_collection_modifyitems(config, items):\n"
                "    removed = items.pop(0)\n"
                "    config.hook.pytest_deselected(items=[removed])\n"
            ),
        }
    )
    with pytest.raises(audit.CopyRunError):
        _run(audit, root, ref, (_SERVER,))
    _assert_clean(root)


@pytest.mark.parametrize("phase", ["setup", "call"])
def test_r1_all_skipped_copy_is_error(audit, make_repo, phase):
    """Reproduces a failure the spec fixes: collecting only skips proves nothing."""
    source = (
        "import pytest\n@pytest.mark.skip(reason='T2 skip')\ndef test_skip():\n    pass\n"
        if phase == "setup"
        else "import pytest\ndef test_skip():\n    pytest.skip('T2 skip')\n"
    )
    root, ref = make_repo({_SERVER: source, _GENERATE: _GOOD_SOURCE})
    with pytest.raises(audit.CopyRunError):
        _run(audit, root, ref, (_SERVER, _GENERATE))
    _assert_clean(root)


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_r2_nonregular_destination_is_refused_and_preserved(
    audit, make_repo, plugin_storage, kind
):
    """Reproduces a failure the spec fixes: destination reuse damages real content."""
    root, ref = make_repo({_SERVER: _GOOD_SOURCE})
    original = root / _SERVER
    sentinel = "# an uncommitted edit must survive\n"
    original.write_text(sentinel)
    destination = original.with_name("test_zz_up_" + original.name)
    if kind == "symlink":
        destination.symlink_to(original.name)
        preserved = original
    else:
        destination.mkdir()
        preserved = destination / "sentinel"
        preserved.write_text(sentinel)
    try:
        with pytest.raises(audit.CopyRunError):
            _run(audit, root, ref, (_SERVER,))
    finally:
        assert preserved.read_text() == sentinel
        assert original.read_text() == sentinel
    assert destination.is_symlink() if kind == "symlink" else destination.is_dir()


def test_r2_hardlink_leftover_is_replaced_without_changing_target(audit, make_repo):
    """Reproduces a failure the spec fixes: writing a hard link overwrites its target."""
    import os

    root, ref = make_repo({_SERVER: _GOOD_SOURCE})
    original = root / _SERVER
    sentinel = "# preserve this working tree edit\n"
    original.write_text(sentinel)
    destination = original.with_name("test_zz_up_" + original.name)
    os.link(original, destination)
    try:
        assert _run(audit, root, ref, (_SERVER,)) == {}
    finally:
        assert original.read_text() == sentinel
    assert not destination.exists()


def test_r2_valid_stale_bytecode_cannot_replace_ref_assertions(
    audit, make_repo, monkeypatch
):
    """Reproduces a failure the spec fixes: valid cached passing code hides a failure."""
    import os

    failing_source = "def test_cache():\n    assert 1 == 0\n"
    passing_source = failing_source.replace("1 == 0", "1 == 1")
    root, ref = make_repo({_SERVER: failing_source})
    destination = root / "mlx_vlm/tests/test_zz_up_test_server.py"
    destination.write_text(passing_source)
    env = dict(os.environ)
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env.pop("PYTEST_PLUGINS", None)
    generated = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--assert=rewrite",
            "tests/" + destination.name,
        ],
        cwd=root / "mlx_vlm",
        env=env,
        capture_output=True,
        text=True,
    )
    assert generated.returncode == 0, generated.stdout + generated.stderr
    caches = list((destination.parent / "__pycache__").glob("test_zz_up_*.pyc"))
    assert len(caches) == 1
    header = caches[0].read_bytes()[:16]
    timestamp = int.from_bytes(header[8:12], "little")
    assert int.from_bytes(header[12:16], "little") == len(failing_source.encode())
    real_run = subprocess.run

    def same_timestamp_run(command, *args, **kwargs):
        if list(command[:3]) == [sys.executable, "-m", "pytest"]:
            # Reproduce a same-second, same-size overwrite deterministically.
            os.utime(destination, (timestamp, timestamp))
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", same_timestamp_run)
    assert set(_run(audit, root, ref, (_SERVER,))) == {_SERVER + "::test_cache"}
    _assert_clean(root)


@pytest.fixture
def plugin_storage(tmp_path, monkeypatch):
    import tempfile

    directory = tmp_path / "plugin-storage"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    return directory


@pytest.mark.parametrize("original_error", [False, True])
def test_r3_cleanup_reports_failure_and_continues(
    audit, make_repo, monkeypatch, plugin_storage, capsys, original_error
):
    """Reproduces a failure the spec fixes: one unlink failure abandons other cleanup."""
    files = {_SERVER: _GOOD_SOURCE, _GENERATE: _GOOD_SOURCE}
    if original_error:
        files["mlx_vlm/tests/conftest.py"] = (
            "def pytest_sessionfinish(session, exitstatus):\n"
            "    session.exitstatus = 3\n"
        )
    root, ref = make_repo(files)
    blocked = root / "mlx_vlm/tests/test_zz_up_test_server.py"
    other = root / "mlx_vlm/tests/test_zz_up_test_generate.py"
    real_run, real_unlink = subprocess.run, Path.unlink
    child_finished = False

    def record_run(command, *args, **kwargs):
        nonlocal child_finished
        result = real_run(command, *args, **kwargs)
        if list(command[:3]) == [sys.executable, "-m", "pytest"]:
            child_finished = True
        return result

    def failing_unlink(path, *args, **kwargs):
        if path == blocked and child_finished:
            raise PermissionError("T2 injected cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", record_run)
    monkeypatch.setattr(Path, "unlink", failing_unlink)
    try:
        with pytest.raises(audit.CopyRunError) as caught:
            _run(audit, root, ref, (_SERVER, _GENERATE))
        if original_error:
            assert "pytest exited 3" in str(caught.value)
            assert "T2 injected cleanup failure" in capsys.readouterr().err
        else:
            assert "T2 injected cleanup failure" in str(caught.value)
        assert blocked.exists()
        assert not other.exists()
        assert list(plugin_storage.iterdir()) == []
    finally:
        # Remove only the deliberately blocked artifact, after checking the audit.
        real_unlink(blocked, missing_ok=True)


_SLEEPING_COPY = """from pathlib import Path
import time

def test_wait():
    Path('started').write_text('ready')
    while True:
        time.sleep(1)
"""


def _start_audit_process(root, ref, plugin_storage):
    import os

    driver = """import importlib.util
import sys
import tempfile
from pathlib import Path
tempfile.tempdir = sys.argv[4]
spec = importlib.util.spec_from_file_location('t2_child_audit', sys.argv[1])
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)
try:
    audit.run_copies(sys.argv[3], repo_root=Path(sys.argv[2]),
                     copies=('mlx_vlm/tests/test_server.py',), python=sys.executable)
except audit.CopyRunError as error:
    print(str(error), file=sys.stderr)
    raise SystemExit(2)
"""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.Popen(
        [sys.executable, "-c", driver, str(_DEV), str(root), ref, str(plugin_storage)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )


def _wait_for_copy(root, process):
    import time

    deadline = time.monotonic() + 60
    while not (root / "mlx_vlm/started").exists():
        assert process.poll() is None, "audit exited before the synthetic test started"
        assert time.monotonic() < deadline, "synthetic test did not start in time"
        time.sleep(0.025)


def _stop_process_group(process):
    import os
    import signal

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.communicate(timeout=60)


@pytest.mark.parametrize("signal_name", ["SIGTERM", "SIGHUP"])
def test_r4_signal_cleans_copies_and_plugin(
    audit, make_repo, plugin_storage, signal_name
):
    """Reproduces a failure the spec fixes: termination bypasses normal finally cleanup."""
    import os
    import signal

    root, ref = make_repo({_SERVER: _SLEEPING_COPY})
    process = _start_audit_process(root, ref, plugin_storage)
    try:
        _wait_for_copy(root, process)
        os.kill(process.pid, getattr(signal, signal_name))
        process.wait(timeout=60)
        # Candidate 1 leaves a child alive; stop it before inspecting leftovers.
        _stop_process_group(process)
        assert process.returncode != 0
        assert list(root.rglob("test_zz_up_*")) == []
        assert list(plugin_storage.iterdir()) == []
    finally:
        _stop_process_group(process)


def test_r5_concurrent_run_is_refused(audit, make_repo, plugin_storage):
    """Reproduces a failure the spec fixes: simultaneous runs share destination names."""
    import os
    import signal

    root, ref = make_repo({_SERVER: _SLEEPING_COPY})
    first = _start_audit_process(root, ref, plugin_storage)
    second = None
    try:
        _wait_for_copy(root, first)
        second = _start_audit_process(root, ref, plugin_storage)
        try:
            output, _ = second.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            pytest.fail(
                "a second audit was not refused while the first held its copies"
            )
        assert second.returncode == 2, output.decode(errors="replace")
        assert b"lock" in output.lower()
        assert first.poll() is None
        assert (root / "mlx_vlm/tests/test_zz_up_test_server.py").exists()
    finally:
        if second is not None:
            _stop_process_group(second)
        if first.poll() is None:
            os.kill(first.pid, signal.SIGINT)
            first.wait(timeout=60)
        _stop_process_group(first)


def test_r6_unmatched_closing_bracket_remains_literal(audit):
    """Preserves an invariant: a closing bracket alone is not a parameter suffix."""
    assert audit.matches(_CASE, _CASE + "]") is False
    assert audit.matches(_CASE + "]", _CASE + "]") is True


def test_r8_collection_message_names_cause_and_original_file(audit, make_repo):
    """Reproduces a failure the spec fixes: import diagnostics expose only a temp name."""
    root, ref = make_repo(
        {_SERVER: "raise ImportError('T2 collection cause in ' + __file__)\n"}
    )
    result = _run(audit, root, ref, (_SERVER,))
    assert set(result) == {_SERVER}
    assert "T2 collection cause" in result[_SERVER]
    assert "test_server.py" in result[_SERVER]
    assert "test_zz_up_" not in result[_SERVER]
    assert "\n" not in result[_SERVER]
    _assert_clean(root)


def test_r9_non_utf8_blob_is_copy_run_error(audit, make_repo):
    """Reproduces a failure the spec fixes: a decoding error escapes exit-2 handling."""
    root, _ = make_repo({_SERVER: _GOOD_SOURCE})
    (root / _SERVER).write_bytes(b"# coding: latin-1\n# caf\xe9\n")
    _git(root, "add", _SERVER)
    _git(
        root,
        "-c",
        "user.name=T2 fixture",
        "-c",
        "user.email=t2@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-qm",
        "Synthetic non-UTF-8 blob",
    )
    with pytest.raises(audit.CopyRunError):
        _run(audit, root, "HEAD", (_SERVER,))
    _assert_clean(root)


def test_r9_copy_write_failure_is_copy_run_error(audit, make_repo, plugin_storage):
    """Reproduces a failure the spec fixes: a missing directory escapes exit-2 handling."""
    root, ref = make_repo({_SERVER: _GOOD_SOURCE})
    directory = root / "mlx_vlm/tests"
    saved = root / "mlx_vlm/saved-tests"
    directory.rename(saved)
    try:
        with pytest.raises(audit.CopyRunError):
            _run(audit, root, ref, (_SERVER,))
        assert list(plugin_storage.iterdir()) == []
    finally:
        saved.rename(directory)
    _assert_clean(root)


def test_r3_plugin_artifact_cleanup_failure_in_tree_is_reported(
    audit, make_repo, monkeypatch
):
    """Reproduces a failure the spec fixes: plugin cleanup silently leaves tree artifacts."""
    import os
    import shutil
    import tempfile

    root, ref = make_repo({_SERVER: _GOOD_SOURCE})
    storage = root / "audit-temporaries"
    storage.mkdir()
    real_unlink = os.unlink
    refused = []

    def refuse_result_unlink(path, *args, **kwargs):
        if Path(path).suffix == ".json":
            refused.append(str(path))
            raise PermissionError("T2 injected plugin result cleanup failure")
        return real_unlink(path, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(tempfile, "tempdir", str(storage))
            patch.setattr(os, "unlink", refuse_result_unlink)
            with pytest.raises(audit.CopyRunError):
                _run(audit, root, ref, (_SERVER,))
    finally:
        assert refused, "the result cleanup failure was not injected"
        assert list(root.rglob("test_zz_up_*")) == []
        shutil.rmtree(storage)


@pytest.mark.parametrize("signal_name", ["SIGTERM", "SIGINT"])
def test_s3_signal_during_cleanup_removes_every_artifact_before_propagating(
    audit, make_repo, monkeypatch, plugin_storage, signal_name
):
    """Reproduces a failure the spec fixes: a signal interrupts the cleanup itself."""
    import os
    import signal

    root, ref = make_repo({_SERVER: _GOOD_SOURCE, _GENERATE: _GOOD_SOURCE})
    targets = [
        root / "mlx_vlm/tests/test_zz_up_test_server.py",
        root / "mlx_vlm/tests/test_zz_up_test_generate.py",
    ]
    real_run, real_unlink = subprocess.run, Path.unlink
    child_finished = False
    injected = False

    def seed_bytecode_after_run(command, *args, **kwargs):
        nonlocal child_finished
        result = real_run(command, *args, **kwargs)
        if list(command[:3]) == [sys.executable, "-m", "pytest"]:
            for target in targets:
                assert target.is_file()
                cache = target.parent / "__pycache__"
                cache.mkdir(exist_ok=True)
                (cache / (target.stem + ".cleanup-probe.pyc")).write_bytes(b"probe")
            child_finished = True
        return result

    def interrupt_first_removal(path, *args, **kwargs):
        nonlocal injected
        if child_finished and path.name.startswith("test_zz_up_") and not injected:
            injected = True
            os.kill(os.getpid(), getattr(signal, signal_name))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", seed_bytecode_after_run)
    monkeypatch.setattr(Path, "unlink", interrupt_first_removal)
    previous_sigint = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            _run(audit, root, ref, (_SERVER, _GENERATE))
        assert injected, "the signal was not sent during cleanup"
        assert list(root.rglob("test_zz_up_*")) == []
        assert list(plugin_storage.iterdir()) == []
        _assert_clean(root)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)


def test_s3_copy_created_before_open_returns_is_cleaned_after_interrupt(
    audit, make_repo, monkeypatch, plugin_storage
):
    """Reproduces a failure the spec fixes: a created but unrecorded copy survives."""
    import os

    root, ref = make_repo({_SERVER: _GOOD_SOURCE, _GENERATE: _GOOD_SOURCE})
    unrecorded = root / "mlx_vlm/tests/test_zz_up_test_generate.py"
    real_open = os.open
    injected = False

    def interrupt_after_creation(path, flags, *args, **kwargs):
        nonlocal injected
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path) == unrecorded and flags & os.O_CREAT and not injected:
            injected = True
            assert unrecorded.is_file()
            # The caller never receives the descriptor or records its successful open.
            os.close(fd)
            raise KeyboardInterrupt("T2 interrupted after creation, before return")
        return fd

    monkeypatch.setattr(os, "open", interrupt_after_creation)
    with pytest.raises(KeyboardInterrupt):
        _run(audit, root, ref, (_SERVER, _GENERATE))
    assert injected, "the copy-creation boundary was not exercised"
    assert list(root.rglob("test_zz_up_*")) == []
    assert list(plugin_storage.iterdir()) == []
    _assert_clean(root)


@pytest.mark.parametrize("phase", ["call", "setup"])
def test_s4_all_xfailed_copy_is_not_refused_as_all_skipped(audit, make_repo, phase):
    """Reproduces a failure the spec fixes: expected failures are treated as skips."""
    if phase == "call":
        source = (
            "import pytest\n"
            "@pytest.mark.xfail(reason='T2 expected failure')\n"
            "def test_expected_failure():\n"
            "    assert False, 'expected body failure'\n"
        )
    else:
        source = (
            "import pytest\n"
            "@pytest.fixture\n"
            "def broken_setup():\n"
            "    assert False, 'expected setup failure'\n"
            "@pytest.mark.xfail(reason='T2 expected failure')\n"
            "def test_expected_failure(broken_setup):\n"
            "    pass\n"
        )
    root, ref = make_repo({_SERVER: source})
    result = _run(audit, root, ref, (_SERVER,))
    assert result == {}
    assert result.counts[_SERVER]["xfailed"] == 1
    assert result.counts[_SERVER]["skipped"] == 0
    _assert_clean(root)


@pytest.fixture
def idle_signal_thread():
    import threading

    ready = threading.Event()
    stop = threading.Event()

    def wait_until_stopped():
        ready.set()
        stop.wait()

    thread = threading.Thread(target=wait_until_stopped, daemon=True)
    thread.start()
    try:
        assert ready.wait(timeout=60), "the idle thread did not start"
        assert thread.is_alive()
        yield thread
    finally:
        stop.set()
        thread.join(timeout=60)
        assert not thread.is_alive(), "the idle thread did not stop"


@pytest.mark.parametrize("signal_name", ["SIGTERM", "SIGINT"])
def test_s3_prime_cleanup_signal_with_another_thread_leaves_no_artifacts(
    audit, make_repo, monkeypatch, plugin_storage, idle_signal_thread, signal_name
):
    """Reproduces a failure the spec fixes: another thread defeats masked cleanup."""
    import os
    import signal
    import time

    root, ref = make_repo({_SERVER: _GOOD_SOURCE, _GENERATE: _GOOD_SOURCE})
    targets = [
        root / "mlx_vlm/tests/test_zz_up_test_server.py",
        root / "mlx_vlm/tests/test_zz_up_test_generate.py",
    ]
    real_run, real_unlink = subprocess.run, Path.unlink
    child_finished = False
    injected = False

    def seed_bytecode_after_run(command, *args, **kwargs):
        nonlocal child_finished
        result = real_run(command, *args, **kwargs)
        if list(command[:3]) == [sys.executable, "-m", "pytest"]:
            for target in targets:
                assert target.is_file()
                cache = target.parent / "__pycache__"
                cache.mkdir(exist_ok=True)
                (cache / (target.stem + ".thread-probe.pyc")).write_bytes(b"probe")
            child_finished = True
        return result

    def interrupt_first_removal(path, *args, **kwargs):
        nonlocal injected
        if child_finished and path.name.startswith("test_zz_up_") and not injected:
            assert idle_signal_thread.is_alive()
            injected = True
            os.kill(os.getpid(), getattr(signal, signal_name))
            # Allow delivery to the idle thread before leaving this removal boundary.
            time.sleep(0.05)
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", seed_bytecode_after_run)
    monkeypatch.setattr(Path, "unlink", interrupt_first_removal)
    previous_sigint = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            _run(audit, root, ref, (_SERVER, _GENERATE))
        assert injected, "the cleanup signal was not injected"
        assert idle_signal_thread.is_alive()
        assert list(root.rglob("test_zz_up_*")) == []
        assert list(plugin_storage.iterdir()) == []
        _assert_clean(root)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)


@pytest.mark.parametrize("signal_name", ["SIGTERM", "SIGINT"])
def test_s3_prime_creation_signal_with_another_thread_leaves_no_artifacts(
    audit, make_repo, monkeypatch, plugin_storage, idle_signal_thread, signal_name
):
    """Preserves an invariant: a signal before recording a copy cannot leave it behind."""
    import os
    import signal
    import time

    root, ref = make_repo({_SERVER: _GOOD_SOURCE, _GENERATE: _GOOD_SOURCE})
    destination = root / "mlx_vlm/tests/test_zz_up_test_generate.py"
    real_open = os.open
    injected = False

    def signal_after_creation(path, flags, *args, **kwargs):
        nonlocal injected
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path) == destination and flags & os.O_CREAT and not injected:
            try:
                assert destination.is_file()
                assert idle_signal_thread.is_alive()
                injected = True
                os.kill(os.getpid(), getattr(signal, signal_name))
                time.sleep(0.05)
            except BaseException:
                # The audit has not received this descriptor and cannot close it.
                os.close(fd)
                raise
        return fd

    monkeypatch.setattr(os, "open", signal_after_creation)
    previous_sigint = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            _run(audit, root, ref, (_SERVER, _GENERATE))
        assert injected, "the creation signal was not injected"
        assert idle_signal_thread.is_alive()
        assert list(root.rglob("test_zz_up_*")) == []
        assert list(plugin_storage.iterdir()) == []
        _assert_clean(root)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
