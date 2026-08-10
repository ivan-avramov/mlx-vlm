"""Tests for `dev/check_body_divergence.py`'s alignment-vs-content rules.

A fork-only file, and the second test of any `dev/` audit script after
`test_fork_marker_check.py`. Same reason that one exists: the audits are the safety
net for every merge, so a bug that makes one MORE PERMISSIVE is the worst shape of
bug available here — nothing fails, the check still prints OK, and divergence stops
being reported. `docs/upstream-gaps.md` is what happens with no check at all; a check
that lies is worse.

For this script "more permissive" has a specific meaning: calling a real CONTENT
difference mere ALIGNMENT. That is the direction most of these tests guard. Two
normalisations look harmless and are not — indentation (semantic in Python) and
comments (one of the ten dropped hunks the marker rollout found was a comment) — so
`normalise` must leave both alone, and several tests say so out loud.

The other direction, calling alignment "content", only produces noise. It is still
tested, because a noisy gate gets switched off.
"""

import importlib.util
from pathlib import Path

import pytest

_DEV = Path(__file__).resolve().parents[2] / "dev" / "check_body_divergence.py"


def _load():
    spec = importlib.util.spec_from_file_location("_cbd_under_test", _DEV)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cbd():
    assert _DEV.is_file(), f"missing {_DEV}"
    return _load()


def _cmp(cbd, upstream, ours, path="mlx_vlm/x.py"):
    return cbd.FileComparison(path, upstream, ours)


UP = """\
import os


def alpha(x):
    return x + 1


def beta(y):
    return y * 2
"""


class TestNormalise:
    """The normalisation is the only place a content difference can be lost."""

    def test_blank_lines_and_trailing_space_are_alignment(self, cbd):
        assert cbd.normalise("def f():\n    return 1\n") == cbd.normalise(
            "def f():   \n\n    return 1\n\n"
        )

    def test_indentation_is_content_not_alignment(self, cbd):
        """In Python an indent change moves a statement between blocks.

        Stripping leading whitespace would make a real behavioural difference read
        as alignment and be silently converged away — the single most dangerous
        over-normalisation available to this script.
        """
        a = "def f():\n    if x:\n        return 1\n    return 2\n"
        b = "def f():\n    if x:\n        return 1\n        return 2\n"
        assert cbd.normalise(a) != cbd.normalise(b)

    def test_comments_are_content_not_alignment(self, cbd):
        """A dropped comment is a dropped hunk.

        One of the ten found while marking was exactly a comment, and it had no
        failing test and no audit hit. Normalising comments away here would put
        that shape back out of reach.
        """
        a = "def f():\n    # explains the clamp\n    return 1\n"
        b = "def f():\n    return 1\n"
        assert cbd.normalise(a) != cbd.normalise(b)

    def test_a_real_statement_change_is_content(self, cbd):
        assert cbd.normalise("def f():\n    return 1\n") != cbd.normalise(
            "def f():\n    return 2\n"
        )


class TestRelocatedDefinitions:
    def test_same_order_reports_nothing(self, cbd):
        assert cbd.relocated_definitions(["a", "b", "c"], ["a", "b", "c"]) == []

    def test_a_single_swap_is_reported(self, cbd):
        moved = cbd.relocated_definitions(["a", "b", "c"], ["a", "c", "b"])
        assert len(moved) == 1
        assert moved[0] in {"b", "c"}

    def test_an_inserted_fork_only_helper_does_not_cascade(self, cbd):
        """This is what makes the signal quiet enough to gate on.

        Comparing raw positions would call every definition after an insertion
        "moved". Dropping one-sided names first and taking the longest common
        subsequence of what remains reports the 2 real relocations in this tree
        rather than hundreds.
        """
        assert cbd.relocated_definitions(["a", "b", "c"], ["a", "fork", "b", "c"]) == []

    def test_a_removed_upstream_symbol_does_not_cascade(self, cbd):
        assert cbd.relocated_definitions(["a", "gone", "b"], ["a", "b"]) == []

    def test_empty_inputs(self, cbd):
        assert cbd.relocated_definitions([], []) == []
        assert cbd.relocated_definitions(["a"], []) == []


class TestDefinitionBodies:
    def test_a_syntax_error_propagates(self, cbd):
        """Fail loud. A file this script cannot parse is a file it is not checking.

        `main` turns this into a fatal error listing the paths, rather than skipping
        them — an audit that quietly covers less than it claims is the bug this whole
        test file exists to prevent.
        """
        with pytest.raises(SyntaxError):
            cbd.definition_bodies("def f(:\n")

    def test_decorators_are_part_of_the_body(self, cbd):
        """A decorator change IS a body change.

        `@pytest.mark.parametrize` data lives entirely in the decorator, so excluding
        it would make a rewritten test case look byte-identical.
        """
        defs, _ = cbd.definition_bodies("@deco(1)\ndef f():\n    pass\n")
        assert "@deco(1)" in defs["f"]

    def test_module_statements_are_separated_from_definitions(self, cbd):
        defs, mods = cbd.definition_bodies("import os\n\n\ndef f():\n    pass\n")
        assert list(defs) == ["f"]
        assert mods == ["import os"]


class TestFileComparison:
    def test_a_pure_reorder_is_alignment_only(self, cbd):
        ours = """\
import os


def beta(y):
    return y * 2


def alpha(x):
    return x + 1
"""
        cmp = _cmp(cbd, UP, ours)
        assert cmp.content_score == 0
        assert cmp.alignment_only_file is True
        assert [k for k, _ in cmp.findings()] == [cbd.FILE_SYMBOL, "beta"]

    def test_an_ours_only_definition_is_content(self, cbd):
        cmp = _cmp(cbd, UP, UP + "\n\ndef fork_helper():\n    pass\n")
        assert cmp.ours_only == ["fork_helper"]
        assert cmp.content_score == 1
        assert cmp.alignment_only_file is False

    def test_an_up_only_definition_is_content(self, cbd):
        ours = UP.replace("def beta(y):\n    return y * 2\n", "")
        cmp = _cmp(cbd, UP, ours)
        assert cmp.up_only == ["beta"]
        assert cmp.alignment_only_file is False

    def test_a_changed_body_is_content(self, cbd):
        cmp = _cmp(cbd, UP, UP.replace("return x + 1", "return x + 42"))
        assert cmp.content_differing == ["alpha"]
        assert cmp.alignment_only == []
        assert cmp.alignment_only_file is False

    def test_a_blank_line_only_change_is_reported_as_alignment(self, cbd):
        """And it names the definition as well as the file.

        A blank-line-only change leaves content_score at 0, so the whole-file
        finding fires too. Both are wanted: `<file>` says converge the file, the
        per-definition finding says which part of it to look at.
        """
        cmp = _cmp(cbd, UP, UP.replace("def alpha(x):\n", "def alpha(x):\n\n"))
        assert cmp.alignment_only == ["alpha"]
        assert cmp.content_differing == []
        assert [name for name, _why in cmp.findings()] == [cbd.FILE_SYMBOL, "alpha"]

    def test_a_new_module_statement_is_content(self, cbd):
        cmp = _cmp(cbd, UP, "import sys\n" + UP)
        assert cmp.module_stmts_same_multiset is False
        assert cmp.content_score == 1
        assert cmp.alignment_only_file is False

    def test_reordered_module_statements_are_not_content(self, cbd):
        up = "import os\nimport sys\n\n\ndef alpha(x):\n    return x\n"
        ours = "import sys\nimport os\n\n\ndef alpha(x):\n    return x\n"
        cmp = _cmp(cbd, up, ours)
        assert cmp.module_stmts_ordered_equal is False
        assert cmp.module_stmts_same_multiset is True
        assert cmp.content_score == 0

    def test_a_definition_that_moved_AND_changed_is_content_not_relocation(self, cbd):
        """Relocation is only claimed for a byte-identical body.

        Otherwise a rewritten function that also moved would be reported as "just
        out of place", inviting someone to move it back and call the file converged
        while its real content divergence went unexamined. That is the permissive
        direction, so it gets its own test.
        """
        ours = """\
import os


def beta(y):
    return y * 2


def alpha(x):
    return x + 999
"""
        cmp = _cmp(cbd, UP, ours)
        assert cmp.content_differing == ["alpha"]
        assert "alpha" not in cmp.relocated
        assert cmp.alignment_only_file is False


class TestExclusionMatching:
    def test_an_empty_list_excuses_nothing(self, cbd):
        """Fail closed."""
        assert cbd.matching_exclusion("mlx_vlm/x.py", "alpha", []) is None

    def test_an_exact_match(self, cbd):
        rules = [("mlx_vlm/x.py", "alpha", "why")]
        assert cbd.matching_exclusion("mlx_vlm/x.py", "alpha", rules) == (
            "mlx_vlm/x.py",
            "alpha",
        )

    def test_it_does_not_leak_across_files(self, cbd):
        rules = [("mlx_vlm/x.py", "alpha", "why")]
        assert cbd.matching_exclusion("mlx_vlm/y.py", "alpha", rules) is None

    def test_it_does_not_leak_across_symbols(self, cbd):
        rules = [("mlx_vlm/x.py", "alpha", "why")]
        assert cbd.matching_exclusion("mlx_vlm/x.py", "beta", rules) is None

    def test_a_path_glob(self, cbd):
        rules = [("mlx_vlm/server/*.py", "alpha", "why")]
        assert cbd.matching_exclusion("mlx_vlm/server/app.py", "alpha", rules)
        assert cbd.matching_exclusion("mlx_vlm/apc.py", "alpha", rules) is None

    def test_the_file_pseudo_symbol(self, cbd):
        """`<file>` names the whole-file finding, and `path::*` must still cover it."""
        by_name = [("mlx_vlm/x.py", cbd.FILE_SYMBOL, "why")]
        assert cbd.matching_exclusion("mlx_vlm/x.py", cbd.FILE_SYMBOL, by_name)
        by_star = [("mlx_vlm/x.py", "*", "why")]
        assert cbd.matching_exclusion("mlx_vlm/x.py", cbd.FILE_SYMBOL, by_star)

    def test_a_real_symbol_entry_does_not_excuse_the_whole_file(self, cbd):
        """Excusing one definition must not excuse the file-level finding.

        The file-level finding says the ENTIRE diff is alignment; a per-definition
        reason cannot support that claim.
        """
        rules = [("mlx_vlm/x.py", "alpha", "why")]
        assert cbd.matching_exclusion("mlx_vlm/x.py", cbd.FILE_SYMBOL, rules) is None

    def test_the_file_pseudo_symbol_is_not_a_legal_identifier(self, cbd):
        """Which is what stops it colliding with a real definition name."""
        assert not cbd.FILE_SYMBOL.isidentifier()


class TestExclusionsFileParsing:
    def test_a_line_without_a_reason_is_fatal(self, cbd, tmp_path, monkeypatch):
        path = tmp_path / ".body-divergence-exclusions"
        path.write_text("mlx_vlm/x.py::alpha\n")
        monkeypatch.setattr(cbd, "EXCLUSIONS_FILE", path)
        with pytest.raises(SystemExit):
            cbd.load_exclusions()

    def test_a_malformed_rule_is_fatal(self, cbd, tmp_path, monkeypatch):
        path = tmp_path / ".body-divergence-exclusions"
        path.write_text("mlx_vlm/x.py  # no double colon\n")
        monkeypatch.setattr(cbd, "EXCLUSIONS_FILE", path)
        with pytest.raises(SystemExit):
            cbd.load_exclusions()

    def test_comments_and_blank_lines_are_skipped(self, cbd, tmp_path, monkeypatch):
        path = tmp_path / ".body-divergence-exclusions"
        path.write_text("# header\n\nmlx_vlm/x.py::alpha  # a real reason\n")
        monkeypatch.setattr(cbd, "EXCLUSIONS_FILE", path)
        assert cbd.load_exclusions() == [("mlx_vlm/x.py", "alpha", "a real reason")]

    def test_a_missing_file_reads_as_empty(self, cbd, tmp_path, monkeypatch):
        monkeypatch.setattr(cbd, "EXCLUSIONS_FILE", tmp_path / "absent")
        assert cbd.load_exclusions() == []

    def test_the_repo_baseline_is_empty(self, cbd):
        """Pins the baseline, the way the other audits' baselines are pinned.

        Every finding this script reports has a mechanical fix, so a non-empty
        baseline means someone claimed a misalignment was deliberate. That should
        require editing this test and saying why.
        """
        assert cbd.load_exclusions() == []


class TestAbsentUpstreamLines:
    """The marker-review view. Its job is to be readable, not to be a proof.

    `check_fork_markers.py` proves a `# Fork:` comment exists; nothing proves it is
    true, and four in this tree were not (`3105b598`, `0670f556`). This is the column a
    reviewer checks a marker's claim against, so the tests here are about it reporting
    the right *set* — a marker review that silently omits a line is a marker review
    that passes a false marker.
    """

    def test_a_strict_superset_reports_nothing(self, cbd):
        """absent=0 is the common case for fork work: we added, we did not replace.

        Note that editing an existing line is a replacement, not an addition — a
        trailing `# noqa` on `return x + 1` makes `absent` report the original. That is
        correct and is the next test.
        """
        ours = UP.replace("    return x + 1", "    log(x)\n    return x + 1")
        cmp = _cmp(cbd, UP, ours)
        assert cmp.absent_upstream_lines("alpha") == []

    def test_a_replaced_line_is_reported(self, cbd):
        cmp = _cmp(cbd, UP, UP.replace("return x + 1", "return x + 42"))
        assert cmp.absent_upstream_lines("alpha") == [(1, "return x + 1")]

    def test_a_dropped_comment_is_reported(self, cbd):
        """One of the ten dropped hunks the marker rollout found was a comment.

        This is the case that caught `0670f556`: upstream's explanatory comment showed
        as upstream-only, which is what proved the construct it describes is upstream's
        and not the fork's, contradicting the marker.
        """
        up = "def alpha(x):\n    # clamp, see #909\n    return x\n"
        ours = "def alpha(x):\n    return x\n"
        cmp = _cmp(cbd, up, ours)
        assert cmp.absent_upstream_lines("alpha") == [(1, "# clamp, see #909")]

    def test_it_counts_multiplicity(self, cbd):
        """Upstream calling something twice where we call it once is a dropped site.

        AGENTS.md's "count the call sites" rule in miniature — `eda1ec4f` changed the
        same line in two endpoints and only one landed.
        """
        up = "def alpha(x):\n    f(x)\n    f(x)\n    return x\n"
        ours = "def alpha(x):\n    f(x)\n    return x\n"
        cmp = _cmp(cbd, up, ours)
        assert cmp.absent_upstream_lines("alpha") == [(1, "f(x)")]

    def test_blank_lines_are_not_reported(self, cbd):
        """Otherwise every reflowed body drowns the real omissions in whitespace."""
        up = "def alpha(x):\n\n\n    return x\n"
        ours = "def alpha(x):\n    return x\n"
        cmp = _cmp(cbd, up, ours)
        assert cmp.absent_upstream_lines("alpha") == []

    def test_reindented_lines_are_not_reported_and_that_is_deliberate(self, cbd):
        """A known, chosen looseness — the one place this view is permissive.

        A fork that wraps upstream's code in a new `if` re-indents all of it, and
        reporting every line as absent would make the view useless for exactly the
        bodies it matters most for. So comparison is on stripped text.

        This does NOT weaken the alignment gate: `normalise` (which decides
        ALIGNMENT findings) leaves leading whitespace alone, because in Python an
        indent change is a semantic change. Two different questions, two different
        normalisations — asserted together here so neither drifts into the other.
        """
        up = "def alpha(x):\n    return x\n"
        ours = "def alpha(x):\n    if y:\n        return x\n"
        cmp = _cmp(cbd, up, ours)
        assert cmp.absent_upstream_lines("alpha") == []
        assert cbd.normalise(up) != cbd.normalise(ours)

    def test_our_own_additions_are_never_reported(self, cbd):
        """One direction only. The other direction is what `# Fork:` markers are for."""
        cmp = _cmp(cbd, UP, UP.replace("return x + 1", "return x + 1\n    extra()"))
        assert cmp.absent_upstream_lines("alpha") == []

    def test_an_unshared_name_reports_nothing(self, cbd):
        cmp = _cmp(cbd, UP, UP + "\n\ndef fork_helper():\n    pass\n")
        assert cmp.absent_upstream_lines("fork_helper") == []
        assert cmp.absent_upstream_lines("never_defined_anywhere") == []

    def test_content_lines_helper(self, cbd):
        assert cbd._content_lines("a\n\n   b   \n\t\n c\n") == ["a", "b", "c"]


class TestAbsentFromFile:
    """The second tier, and the one worth reading. `absent` cannot tell moved from lost.

    `turboquant.py::_fused_mse_decode_2pass_1_kernel` reports 66 absent lines and 2
    file-absent: the fork rewrote the kernel and kept upstream's whole body as an
    ours-only `_fused_mse_decode_2pass_1_kernel_legacy` sibling, so 64 of the 66 never
    left the file. Reading the per-definition number as content loss overstates it by
    33x — the same "a measure that cannot distinguish two situations needing opposite
    responses" error AGENTS.md's central rule is about, one level down.
    """

    UP_TWO = """\
def alpha(x):
    return helper(x) + 1


def beta(y):
    return y
"""

    def test_content_moved_to_another_definition_is_not_gone(self, cbd):
        ours = """\
def alpha(x):
    return rewritten(x)


def alpha_legacy(x):
    return helper(x) + 1


def beta(y):
    return y
"""
        cmp = _cmp(cbd, self.UP_TWO, ours)
        assert cmp.absent_upstream_lines("alpha") == [(1, "return helper(x) + 1")]
        assert cmp.absent_from_file("alpha") == []

    def test_content_moved_to_another_SHARED_definition_is_not_gone(self, cbd):
        """The mover need not be an ours-only helper — a shared sibling counts too.

        The moved text must match exactly, which is worth noting: `helper(y)` is not
        `helper(x)`, so a "move" that also renames a variable still reports. That is
        correct but it is why AGENTS.md records two commits whose entire residue was a
        local variable rename — expect that shape and read it as a rename.
        """
        ours = """\
def alpha(x):
    return rewritten(x)


def beta(y):
    return helper(x) + 1
"""
        cmp = _cmp(cbd, self.UP_TWO, ours)
        assert cmp.absent_upstream_lines("alpha")
        assert cmp.absent_from_file("alpha") == []

    def test_a_move_that_renames_a_variable_still_reports(self, cbd):
        """The flip side of the above, made explicit so the looseness is bounded."""
        ours = """\
def alpha(x):
    return rewritten(x)


def beta(y):
    return helper(y) + 1
"""
        cmp = _cmp(cbd, self.UP_TWO, ours)
        assert cmp.absent_from_file("alpha") == [(1, "return helper(x) + 1")]

    def test_genuinely_missing_content_is_still_reported(self, cbd):
        """The tier must not swallow the signal it exists to sharpen."""
        ours = self.UP_TWO.replace("    return helper(x) + 1", "    return 0")
        cmp = _cmp(cbd, self.UP_TWO, ours)
        assert cmp.absent_from_file("alpha") == [(1, "return helper(x) + 1")]

    def test_it_is_a_subset_of_the_per_definition_view(self, cbd):
        """A structural invariant: the tier can only ever narrow, never add."""
        ours = self.UP_TWO.replace("    return helper(x) + 1", "    return 0")
        cmp = _cmp(cbd, self.UP_TWO, ours)
        assert set(cmp.absent_from_file("alpha")) <= set(
            cmp.absent_upstream_lines("alpha")
        )

    def test_module_level_code_counts_as_the_file(self, cbd):
        """A line hoisted OUT of a definition to module scope has not left the file.

        Comparison is against the whole source text, not against the definitions the
        AST walk collected, precisely so a hoist to module scope does not read as a
        loss. `generate/common.py` has 21 module statements against upstream's 14, so
        this is not a hypothetical shape.
        """
        up = """\
def alpha(x):
    LIMIT = compute(8)
    return LIMIT


def beta(y):
    return y
"""
        ours = """\
LIMIT = compute(8)


def alpha(x):
    return LIMIT


def beta(y):
    return y
"""
        cmp = _cmp(cbd, up, ours)
        assert cmp.absent_upstream_lines("alpha") == [(1, "LIMIT = compute(8)")]
        assert cmp.absent_from_file("alpha") == []

    def test_an_unshared_name_reports_nothing(self, cbd):
        cmp = _cmp(cbd, self.UP_TWO, self.UP_TWO)
        assert cmp.absent_from_file("nope") == []

    def test_the_real_turboquant_kernel_case(self, cbd):
        """Ties the tier to the file that motivated it, the way
        `test_fork_marker_check.py` ties case 4 to its `.symbol-exclusions` entries.

        If the fork ever drops the `_legacy` sibling this fails, which is correct — the
        64 lines would then genuinely be gone and the count should say so.
        """
        import subprocess

        root = Path(__file__).resolve().parents[2]
        path = "mlx_vlm/turboquant.py"
        up = subprocess.run(
            ["git", "-C", str(root), "show", f"upstream/main:{path}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        ours = (root / path).read_text()
        cmp = cbd.FileComparison(path, up, ours)
        name = "_fused_mse_decode_2pass_1_kernel"

        assert f"{name}_legacy" in cmp.ours_only
        per_def = sum(c for c, _ in cmp.absent_upstream_lines(name))
        file_wide = sum(c for c, _ in cmp.absent_from_file(name))
        assert per_def > 50, per_def
        assert file_wide < 5, file_wide


class TestSweep:
    """`--sweep` is the marker-review worklist. Its only job is to rank correctly.

    A report cannot be "more permissive" the way a gate can, so the risk here is
    different: ranking by the wrong number sends the reader to the wrong definition
    first. That already happened once — ranking by `absent` put a rewritten Metal
    kernel on top with 64 of its 66 lines sitting in an ours-only sibling.
    """

    def _comparisons(self, cbd):
        superset = cbd.FileComparison(
            "a.py",
            "def f(x):\n    return x\n",
            "def f(x):\n    log(x)\n    return x\n",
        )
        small_gone = cbd.FileComparison(
            "b.py",
            "def g(x):\n    return helper(x)\n",
            "def g(x):\n    return other(x)\n",
        )
        big_absent_no_gone = cbd.FileComparison(
            "c.py",
            "def h(x):\n    a = 1\n    b = 2\n    c = 3\n    return a\n",
            "def h(x):\n    return rebuilt(x)\n\n\ndef h_legacy(x):\n"
            "    a = 1\n    b = 2\n    c = 3\n    return a\n",
        )
        return superset, small_gone, big_absent_no_gone

    def test_it_ranks_by_gone_not_absent(self, cbd, capsys):
        """The regression this mode was rewritten for."""
        _, small_gone, big_absent_no_gone = self._comparisons(cbd)
        assert sum(c for c, _ in big_absent_no_gone.absent_upstream_lines("h")) > sum(
            c for c, _ in small_gone.absent_upstream_lines("g")
        )
        assert big_absent_no_gone.absent_from_file("h") == []

        cbd.report_sweep([big_absent_no_gone, small_gone])
        out = capsys.readouterr().out
        # ".py::" rather than "::" — the column header is `path::definition`.
        listed = [l for l in out.splitlines() if ".py::" in l]
        assert listed, out
        assert "b.py::g" in listed[0], listed

    def test_strict_supersets_are_counted_but_not_listed(self, cbd, capsys):
        """22 of the tree's 61 are pure fork addition; listing them is noise, and
        omitting them silently would misreport coverage. So: counted, not listed."""
        superset, small_gone, _ = self._comparisons(cbd)
        cbd.report_sweep([superset, small_gone])
        out = capsys.readouterr().out
        assert "a.py::f" not in out
        assert "b.py::g" in out
        assert "2 content-differing shared definition(s)" in out
        assert "The other 1 are strict supersets" in out

    def test_it_survives_a_tree_with_nothing_to_report(self, cbd, capsys):
        identical = cbd.FileComparison(
            "a.py", "def f(x):\n    return x\n", "def f(x):\n    return x\n"
        )
        cbd.report_sweep([identical])
        out = capsys.readouterr().out
        assert "0 content-differing" in out
        assert "::" not in out.split("path::definition")[-1]
