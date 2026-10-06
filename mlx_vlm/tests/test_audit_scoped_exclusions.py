"""Fork-only: scoped exclusions and pinned allowlist entries in the dev/ audits.

Added at the v0.7.6 sync after cold review. Two holes:

1. `.symbol-exclusions` / `.deletion-exclusions` excused BARE names. An entry for a
   generic test-helper name (`verify`, `forward`, `start`, ...) also excused any FUTURE
   dropped def of that name anywhere in the file. A qualified entry
   (`Class.method`, `outer.inner`) must excuse only that scope.
2. `.fork-marker-allowlist` exempted whole files forever. An entry pinned to the
   file's git blob (`path@<blob>`) must stop applying once the file changes, so a
   later unmarked edit is reported instead of silently absorbed.
"""

import importlib.util
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    path = _ROOT / "dev" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sym():
    return _load("check_upstream_symbols")


@pytest.fixture(scope="module")
def dele():
    return _load("check_upstream_deletions")


@pytest.fixture(scope="module")
def cfm():
    return _load("check_fork_markers")


SOURCE = '''
class TestA:
    def verify(self):
        pass

class TestB:
    def verify(self):
        pass

def helper():
    def verify():
        pass
'''


class TestQualifiedNames:
    def test_qualnames_cover_every_scope(self, sym, dele):
        expected = {"TestA.verify", "TestB.verify", "helper.verify"}
        assert sym.defined_qualnames(SOURCE)["verify"] == expected
        assert dele.defined_qualnames(SOURCE)["verify"] == expected

    @pytest.mark.parametrize("module", ["sym", "dele"])
    def test_a_qualified_entry_excuses_only_its_scope(self, module, request):
        audit = request.getfixturevalue(module)
        quals = audit.defined_qualnames(SOURCE)["verify"]
        one = [("t.py", "TestA.verify", "r")]
        assert not audit.scoped_excuse("t.py", "verify", quals, one)
        every = [
            ("t.py", "TestA.verify", "r"),
            ("t.py", "TestB.verify", "r"),
            ("t.py", "helper.verify", "r"),
        ]
        assert audit.scoped_excuse("t.py", "verify", quals, every)

    @pytest.mark.parametrize("module", ["sym", "dele"])
    def test_a_bare_entry_still_excuses_every_scope(self, module, request):
        audit = request.getfixturevalue(module)
        quals = audit.defined_qualnames(SOURCE)["verify"]
        assert audit.scoped_excuse("t.py", "verify", quals, [("t.py", "verify", "r")])

    @pytest.mark.parametrize("module", ["sym", "dele"])
    def test_entries_do_not_leak_across_files(self, module, request):
        audit = request.getfixturevalue(module)
        quals = audit.defined_qualnames(SOURCE)["verify"]
        assert not audit.scoped_excuse(
            "u.py", "verify", quals, [("t.py", "verify", "r")]
        )

    @pytest.mark.parametrize("module", ["sym", "dele"])
    def test_a_module_scope_entry_excuses_only_the_top_level_def(self, module, request):
        audit = request.getfixturevalue(module)
        source = "def verify():\n    pass\n\nclass C:\n    def verify(self):\n        pass\n"
        quals = audit.defined_qualnames(source)["verify"]
        assert quals == {"verify", "C.verify"}
        entry = [("t.py", ".verify", "r")]
        assert not audit.scoped_excuse("t.py", "verify", quals, entry)
        assert audit.scoped_excuse("t.py", "verify", {"verify"}, entry)

    def test_fork_marker_case_4_accepts_a_qualified_entry(self, cfm):
        covered, _ = cfm.deletion_of_excused_symbols(
            ["-    def verify(self):"], ["TestA.verify"]
        )
        assert covered

    GENERIC = (
        "verify", "values", "start", "sampler", "image", "forward", "embeddings",
        "language", "criteria", "has_pending_prompts", "unprocessed_prompts",
        "last_segment",
    )

    @pytest.mark.parametrize(
        "exclusions", [".symbol-exclusions", ".deletion-exclusions"]
    )
    def test_generic_test_helper_names_are_qualified(self, exclusions):
        bare = []
        for raw in (_ROOT / exclusions).read_text().splitlines():
            rule = raw.partition("#")[0].strip()
            if "::" in rule and rule.partition("::")[2] in self.GENERIC:
                bare.append(rule)
        assert not bare, bare


class TestPinnedAllowlist:
    def test_rule_parsing(self, cfm):
        assert cfm.parse_allowlist_rule("a/b.py") == ("a/b.py", None)
        assert cfm.parse_allowlist_rule("a/b.py@abc1234") == ("a/b.py", "abc1234")

    def test_unpinned_entry_applies(self, cfm):
        assert cfm.allowlist_entry_applies("a.py", "a.py", None, lambda p: "f00")

    def test_pinned_entry_applies_only_to_that_blob(self, cfm):
        blob = "0123456789abcdef0123456789abcdef01234567"
        assert cfm.allowlist_entry_applies("a.py", "a.py", blob[:12], lambda p: blob)
        assert not cfm.allowlist_entry_applies(
            "a.py", "a.py", blob[:12], lambda p: "f" * 40
        )

    def test_a_short_pin_is_rejected(self, cfm):
        with pytest.raises(SystemExit):
            cfm.parse_allowlist_rule("a.py@abc")

    def test_every_repo_allowlist_entry_is_pinned_and_current(self, cfm):
        entries = cfm.load_allowlist()
        assert entries, "expected the C104 test-file entries"
        for rule, _reason in entries:
            glob, pin = cfm.parse_allowlist_rule(rule)
            assert pin, f"{rule}: whole-file entries must be pinned to a blob"
            blob = subprocess.run(
                ["git", "-C", str(_ROOT), "rev-parse", f":{glob}"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            assert blob.startswith(pin), (
                f"{glob} changed since its allowlist pin {pin}: mark the new fork "
                "sites (or review them) and re-pin"
            )
