"""Tests for `dev/check_upstream_registries.py`'s four detection shapes.

A fork-only file, and the third test of any `dev/` audit after
`test_fork_marker_check.py` and `test_body_divergence_check.py`. Same reason all three
exist: a bug that makes one of these checks MORE PERMISSIVE fails nothing, still prints
OK, and silently stops reporting dropped content.

For this script permissiveness has a specific and dangerous form: **silently skipping a
construct.** Every shape it looks at is an `ast.Assign` / `ast.AnnAssign` /
`ast.Import*`, and the AST offers a dozen ways to write each one. A registry built with
`frozenset([...])` instead of a literal, a field declared with `x: int` instead of
`x = 1`, a re-export written `import a.b as c` — miss any of those and the check keeps
printing OK over an entire category it never examined. So most of what follows is
coverage of *forms*, not of the comparison.

The four shapes each cost a real loss in this fork's history, and every one is
reproduced below against a synthetic pair:

  1. four `MODEL_REMAPPING` entries ("unlimited-ocr", "inkling_mm_model" among them)
  2. a whole module-level container
  3. `models/gemma4/__init__.py`'s `Gemma4VideoProcessor` re-export
  4. `deepseek_v4/config.py`'s `ModelConfig.index_block` / `.index_keep` — which
     `check_upstream_symbols.py`'s docstring claimed to catch and cannot, since a
     dataclass field is an AnnAssign and that script collects only def/class names
"""

import ast
import importlib.util
from pathlib import Path

import pytest

_DEV = Path(__file__).resolve().parents[2] / "dev" / "check_upstream_registries.py"


def _load():
    spec = importlib.util.spec_from_file_location("_cur_under_test", _DEV)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cur():
    assert _DEV.is_file(), f"missing {_DEV}"
    return _load()


def _kinds(findings):
    return sorted({kind for kind, _symbol, _detail in findings})


def _symbols(findings, kind):
    return sorted(symbol for k, symbol, _d in findings if k == kind)


class TestContainerEntries:
    """Shape 1's parser. A form it does not recognise is a registry never compared."""

    @pytest.mark.parametrize(
        ("label", "source", "expected"),
        [
            ("dict", "{'a': 1, 'b': 2}", {"'a'", "'b'"}),
            ("list", "['a', 'b']", {"'a'", "'b'"}),
            ("set", "{'a', 'b'}", {"'a'", "'b'"}),
            ("tuple", "('a', 'b')", {"'a'", "'b'"}),
            ("frozenset call", "frozenset(['a', 'b'])", {"'a'", "'b'"}),
            ("dict call", "dict([('a', 1)])", {"('a', 1)"}),
            ("name keys", "{Foo: 1}", {"Foo"}),
            ("name elements", "[Foo, Bar]", {"Bar", "Foo"}),
            ("empty dict", "{}", set()),
        ],
    )
    def test_recognised_forms(self, cur, label, source, expected):
        node = ast.parse(source, mode="eval").body
        assert cur.container_entries(node) == expected, label

    @pytest.mark.parametrize(
        ("label", "source"),
        [
            ("a plain int", "5"),
            ("a string", "'hello'"),
            ("a comprehension", "[x for x in y]"),
            ("a multi-arg call", "dict(a=1, b=2)"),
            ("an attribute", "mod.THING"),
        ],
    )
    def test_non_containers_return_none(self, cur, label, source):
        """None means "not a registry", which is different from "an empty registry".

        A comprehension is deliberately not a container literal: its entries are not
        statically knowable, and returning an empty set for one would make every
        comprehension-built registry read as "upstream has nothing here".
        """
        node = ast.parse(source, mode="eval").body
        assert cur.container_entries(node) is None, label

    def test_int_and_string_keys_do_not_collide(self, cur):
        """Keyed by repr, so `4` and `"4"` are two entries.

        A registry keyed by both is not unusual, and collapsing them would let a
        dropped string key hide behind a surviving int key of the same digits.
        """
        node = ast.parse("{4: 'a', '4': 'b'}", mode="eval").body
        assert cur.container_entries(node) == {"4", "'4'"}

    def test_a_dict_unpacking_records_a_sentinel(self, cur):
        """`{**other, 'a': 1}` has keys that are not statically knowable.

        Recorded as `**` rather than ignored, so a registry that MERGES another is
        visibly not fully comparable instead of looking exhaustively checked.
        """
        node = ast.parse("{**other, 'a': 1}", mode="eval").body
        assert cur.container_entries(node) == {"**", "'a'"}


class TestModuleRegistries:
    def test_module_level_only(self, cur):
        """A container inside a function is local state, not a registry.

        Including them drowned the signal in the prototype — every helper that builds
        a dict became a comparison.
        """
        source = (
            "TOP = {'a': 1}\n\n\ndef f():\n    local = {'b': 2}\n    return local\n"
        )
        assert cur.module_registries(ast.parse(source)) == {"TOP": {"'a'"}}

    def test_annotated_assignment(self, cur):
        source = "REG: dict = {'a': 1}\n"
        assert cur.module_registries(ast.parse(source)) == {"REG": {"'a'"}}

    def test_an_annotation_without_a_value_is_not_a_registry(self, cur):
        assert cur.module_registries(ast.parse("REG: dict\n")) == {}

    def test_multiple_targets(self, cur):
        source = "A = B = {'x': 1}\n"
        assert cur.module_registries(ast.parse(source)) == {
            "A": {"'x'"},
            "B": {"'x'"},
        }


class TestBoundImportNames:
    @pytest.mark.parametrize(
        ("label", "source", "expected"),
        [
            ("from-import", "from x import y", {"y"}),
            ("from-import as", "from x import y as z", {"z"}),
            ("plain import binds the root", "import a.b.c", {"a"}),
            ("import as binds the alias", "import a.b as c", {"c"}),
            ("multiple names", "from x import y, w", {"w", "y"}),
            ("relative", "from . import y", {"y"}),
        ],
    )
    def test_forms(self, cur, label, source, expected):
        assert cur.bound_import_names(ast.parse(source)) == expected, label

    def test_star_import_is_skipped(self, cur):
        """`from x import *` binds an unknowable set — there is nothing to compare.

        Treating it as binding nothing would report every name in every starred
        module as missing.
        """
        assert cur.bound_import_names(ast.parse("from x import *")) == set()

    def test_imports_inside_functions_count(self, cur):
        """`ast.walk`, not `tree.body` — and that is deliberate here.

        Unlike registries, a function-local import is exactly how this fork's lazy
        re-exports work (`generate/__init__.py`'s `__getattr__` imports inside the
        function), so restricting to module level would make the fork's own
        indirection look like a dropped re-export.
        """
        source = "def f():\n    from x import y\n    return y\n"
        assert cur.bound_import_names(ast.parse(source)) == {"y"}


class TestClassAttributes:
    def test_dataclass_fields_are_found(self, cur):
        """Shape 4, and the reason it exists.

        `check_upstream_symbols.py` collects Function/AsyncFunction/ClassDef names, so
        a dataclass field is invisible to it — its docstring cited
        `ModelConfig.index_block` / `.index_keep` as instances it catches and it cannot
        see either. Every `models/*/config.py` in this tree is dataclass fields, where a
        dropped field silently changes model behaviour rather than raising.
        """
        source = (
            "import dataclasses\n\n\n@dataclasses.dataclass\nclass ModelConfig:\n"
            "    index_block: int = 3\n    index_keep: int = 1\n"
        )
        assert cur.class_attributes(ast.parse(source)) == {
            "ModelConfig": {"index_block", "index_keep"}
        }

    def test_plain_and_annotated_assignments_both_count(self, cur):
        source = "class C:\n    A = 1\n    B: int = 2\n    C_: int\n"
        assert cur.class_attributes(ast.parse(source)) == {"C": {"A", "B", "C_"}}

    def test_methods_are_not_attributes(self, cur):
        """Those are check_upstream_symbols.py's job.

        Two checks reporting one event is how the symbol/deletion pair used to leave
        both halves of a rename unconnected.
        """
        source = "class C:\n    A = 1\n\n    def m(self):\n        pass\n"
        assert cur.class_attributes(ast.parse(source)) == {"C": {"A"}}

    def test_a_class_with_no_attributes_is_omitted(self, cur):
        source = "class C:\n    def m(self):\n        pass\n"
        assert cur.class_attributes(ast.parse(source)) == {}

    def test_nested_classes_are_found(self, cur):
        source = "class Outer:\n    X = 1\n\n    class Inner:\n        Y = 2\n"
        assert cur.class_attributes(ast.parse(source)) == {
            "Outer": {"X"},
            "Inner": {"Y"},
        }


class TestFindingsForFile:
    """The four historical losses, reproduced."""

    UPSTREAM = """\
from .gemma4 import Gemma4Processor, Gemma4VideoProcessor

MODEL_REMAPPING = {
    "llava_qwen2": "fastvlm",
    "unlimited-ocr": "ocr",
    "inkling_mm_model": "inkling",
}
PROMPT_FORMATS = {"gemma": 1}
__all__ = ["a", "b", "c"]


class ModelConfig:
    index_block: int = 3
    index_keep: int = 1
"""

    OURS = """\
from .gemma4 import Gemma4Processor

MODEL_REMAPPING = {"llava_qwen2": "fastvlm"}
__all__ = ["a", "b"]


class ModelConfig:
    index_block: int = 3
"""

    def test_all_four_shapes_are_reported(self, cur):
        findings = cur.findings_for_file("p.py", self.UPSTREAM, self.OURS)

        assert _kinds(findings) == [
            "class-attribute",
            "re-export",
            "registry",
            "registry-entry",
        ]
        assert _symbols(findings, "re-export") == ["Gemma4VideoProcessor"]
        assert _symbols(findings, "registry") == ["PROMPT_FORMATS"]
        assert _symbols(findings, "class-attribute") == ["ModelConfig::index_keep"]
        assert _symbols(findings, "registry-entry") == [
            "MODEL_REMAPPING::'inkling_mm_model'",
            "MODEL_REMAPPING::'unlimited-ocr'",
            "__all__::'c'",
        ]

    def test_an_identical_file_reports_nothing(self, cur):
        assert cur.findings_for_file("p.py", self.UPSTREAM, self.UPSTREAM) == []

    def test_our_own_additions_are_not_reported(self, cur):
        """One direction only. What we add is fork work; what upstream has and we
        lack is the question."""
        ours = self.UPSTREAM.replace(
            '__all__ = ["a", "b", "c"]', '__all__ = ["a", "b", "c", "d"]'
        )
        assert cur.findings_for_file("p.py", self.UPSTREAM, ours) == []

    def test_values_are_not_compared(self, cur):
        """Presence only, deliberately. Values diverge legitimately across this fork
        (every tuned constant); a wrong value is check_body_divergence.py's job."""
        ours = self.UPSTREAM.replace(
            '"llava_qwen2": "fastvlm"', '"llava_qwen2": "other"'
        )
        assert cur.findings_for_file("p.py", self.UPSTREAM, ours) == []

    def test_a_renamed_registry_reports_as_a_missing_registry(self, cur):
        """Which is correct: a rename is exactly the event that loses entries."""
        upstream = 'REG = {"a": 1}\n'
        ours = 'RENAMED = {"a": 1}\n'
        findings = cur.findings_for_file("p.py", upstream, ours)
        assert _symbols(findings, "registry") == ["REG"]

    def test_a_missing_class_is_not_reported_here(self, cur):
        upstream = "class C:\n    A = 1\n"
        assert cur.findings_for_file("p.py", upstream, "") == []

    def test_a_syntax_error_propagates(self, cur):
        """Fail loud — main() turns this into a fatal error listing the paths."""
        with pytest.raises(SyntaxError):
            cur.findings_for_file("p.py", "def f(:\n", "")


class TestExclusionMatching:
    def test_an_empty_list_excuses_nothing(self, cur):
        """Fail closed."""
        assert cur.matching_exclusion("p.py", "X", []) is None

    def test_an_exact_match(self, cur):
        rules = [("p.py", "X", "why")]
        assert cur.matching_exclusion("p.py", "X", rules) == ("p.py", "X")

    def test_it_does_not_leak_across_files(self, cur):
        rules = [("p.py", "X", "why")]
        assert cur.matching_exclusion("q.py", "X", rules) is None

    def test_a_whole_container_can_be_excused_by_one_glob(self, cur):
        """`MODEL_REMAPPING::*` covers every entry, which is why the symbol field
        embeds the container name rather than being the bare entry."""
        rules = [("p.py", "MODEL_REMAPPING::*", "why")]
        assert cur.matching_exclusion("p.py", "MODEL_REMAPPING::'a'", rules)
        assert cur.matching_exclusion("p.py", "OTHER::'a'", rules) is None

    def test_one_entry_can_be_excused_without_the_container(self, cur):
        rules = [("p.py", "MODEL_REMAPPING::'a'", "why")]
        assert cur.matching_exclusion("p.py", "MODEL_REMAPPING::'a'", rules)
        assert cur.matching_exclusion("p.py", "MODEL_REMAPPING::'b'", rules) is None


class TestExclusionsFileParsing:
    def test_the_symbol_field_may_contain_a_double_colon(
        self, cur, tmp_path, monkeypatch
    ):
        """The split is on the FIRST `::` only.

        `path::MODEL_REMAPPING::'foo'` is one path and one symbol, not three fields —
        getting this wrong would silently mis-parse every registry-entry and
        class-attribute exclusion into an unmatchable rule.
        """
        path = tmp_path / ".registry-exclusions"
        path.write_text("mlx_vlm/utils.py::MODEL_REMAPPING::'foo'  # a real reason\n")
        monkeypatch.setattr(cur, "EXCLUSIONS_FILE", path)
        assert cur.load_exclusions() == [
            ("mlx_vlm/utils.py", "MODEL_REMAPPING::'foo'", "a real reason")
        ]

    def test_a_line_without_a_reason_is_fatal(self, cur, tmp_path, monkeypatch):
        path = tmp_path / ".registry-exclusions"
        path.write_text("p.py::X\n")
        monkeypatch.setattr(cur, "EXCLUSIONS_FILE", path)
        with pytest.raises(SystemExit):
            cur.load_exclusions()

    def test_a_malformed_rule_is_fatal(self, cur, tmp_path, monkeypatch):
        path = tmp_path / ".registry-exclusions"
        path.write_text("p.py  # no double colon\n")
        monkeypatch.setattr(cur, "EXCLUSIONS_FILE", path)
        with pytest.raises(SystemExit):
            cur.load_exclusions()

    def test_the_repo_baseline_is_the_five_reviewed_re_exports(self, cur):
        """Pins the baseline, like the other audits' tests do.

        All five are fork replacements that already carry a `# Fork:` marker at their
        import site. A sixth appearing means either a new deliberate replacement or a
        dropped re-export, and it should require editing this test to say which.
        """
        parsed = cur.load_exclusions()

        assert len(parsed) == 5
        assert all(reason.startswith("REVIEWED:") for _p, _s, reason in parsed)
        symbols = sorted(symbol for _p, symbol, _r in parsed)
        assert symbols == [
            "_check_configured_context_budget",
            "generation_stream",
            "generation_stream",
            "generation_stream",
            "top_p_sampling",
        ]
