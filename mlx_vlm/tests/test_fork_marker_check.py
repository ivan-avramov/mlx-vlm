"""Tests for `dev/check_fork_markers.py`'s coverage rules.

A fork-only file, and the first test of any `dev/` audit script. The audits are the
safety net for every merge, so a bug that makes one MORE PERMISSIVE is the worst
shape of bug available here: nothing fails, the check still prints OK, and dropped
content stops being reported. `docs/upstream-gaps.md` exists because that happened
without a check at all; it would be worse with one that lies.

These cover "case 4" — a pure-deletion hunk whose removed top-level symbols are all
already excused in `.symbol-exclusions`. That rule was added because a file holding a
reviewed absent symbol was permanently un-drainable: diff attributes the deletion to
whatever line of ours sits at the seam (a blank one), so there is no enclosing
definition to mark and no whitespace to converge. `generate/dispatch.py` and
`server/generation.py` were both stuck at one residual site with every genuine site
marked.

The rule has to stay narrow, which is what most of these assert. It must NOT cover a
whitespace-only deletion (a real alignment artifact worth reporting) and must NOT
cover a hunk that removes any unexcused symbol (that is a dropped hunk).
"""

import importlib.util
from pathlib import Path

import pytest

_DEV = Path(__file__).resolve().parents[2] / "dev" / "check_fork_markers.py"


def _load():
    spec = importlib.util.spec_from_file_location("_cfm_under_test", _DEV)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cfm():
    assert _DEV.is_file(), f"missing {_DEV}"
    return _load()


EXCUSED = ["_cache_fully_retained", "_prefix_cache_trim_amount"]


class TestDeletionOfExcusedSymbols:
    """The narrowness of case 4 is the whole point of the rule."""

    @pytest.mark.parametrize(
        ("label", "body"),
        [
            ("both excused", ["-def _cache_fully_retained(c):", "-    return True"]),
            ("async def", ["-async def _cache_fully_retained(c):"]),
            ("class", ["-class _prefix_cache_trim_amount:"]),
            (
                "two excused in one hunk",
                [
                    "-def _cache_fully_retained(c):",
                    "-    return True",
                    "-",
                    "-def _prefix_cache_trim_amount(a, b):",
                ],
            ),
        ],
    )
    def test_covers_deletions_of_only_excused_symbols(self, cfm, label, body):
        covered, names = cfm.deletion_of_excused_symbols(body, EXCUSED)
        assert covered is True, label
        assert names, "the covered names must be reported, not just counted"

    @pytest.mark.parametrize(
        ("label", "body"),
        [
            ("whitespace only", ["-", "-    "]),
            ("comment only", ["-# a comment"]),
            ("blank plus comment", ["-", "-# ---- section ----"]),
            ("statement but no def", ["-CONSTANT = 3"]),
        ],
    )
    def test_does_not_cover_a_deletion_with_no_top_level_definition(
        self, cfm, label, body
    ):
        """A whitespace/comment-only deletion is a real alignment artifact.

        Covering it would silently swallow the trap-9 whitespace probes the
        convention wants a human to either converge or write down.
        """
        covered, names = cfm.deletion_of_excused_symbols(body, EXCUSED)
        assert covered is False, label
        assert names == []

    @pytest.mark.parametrize(
        ("label", "body"),
        [
            ("unexcused only", ["-def brand_new_upstream_helper(x):"]),
            (
                "one excused, one not",
                [
                    "-def _cache_fully_retained(c):",
                    "-def brand_new_upstream_helper(x):",
                ],
            ),
        ],
    )
    def test_does_not_cover_a_deletion_touching_an_unexcused_symbol(
        self, cfm, label, body
    ):
        """This is the case that must never pass: a genuinely dropped symbol.

        `all()` over the found names, not `any()` — a hunk that removes one excused
        symbol alongside one nobody reviewed is a dropped hunk wearing a reviewed
        symbol as cover.
        """
        covered, _ = cfm.deletion_of_excused_symbols(body, EXCUSED)
        assert covered is False, label

    def test_indented_definitions_are_not_top_level(self, cfm):
        """A method inside a deleted class must not satisfy the rule by itself.

        The regex anchors at column 0 after diff's `-`. A hunk that deletes only a
        method body has no top-level definition, so it falls through to the normal
        enclosing-definition rule, which is correct: the class is still there and
        can carry the marker.
        """
        covered, names = cfm.deletion_of_excused_symbols(
            ["-    def _cache_fully_retained(self):"], EXCUSED
        )
        assert covered is False
        assert names == []

    def test_empty_exclusion_list_covers_nothing(self, cfm):
        """Fail closed: no exclusions means no case-4 coverage at all."""
        covered, _ = cfm.deletion_of_excused_symbols(
            ["-def _cache_fully_retained(c):"], []
        )
        assert covered is False


class TestExcusedSymbols:
    def test_matches_an_exact_path(self, cfm):
        exclusions = [("mlx_vlm/generate/dispatch.py", "_cache_fully_retained")]
        assert cfm.excused_symbols("mlx_vlm/generate/dispatch.py", exclusions) == [
            "_cache_fully_retained"
        ]

    def test_does_not_leak_across_files(self, cfm):
        """An exclusion is per-path; a symbol excused in one file is not in another.

        `.symbol-exclusions` uses path globs, so the containment has to come from
        the glob rather than from the symbol name being globally excused.
        """
        exclusions = [("mlx_vlm/generate/dispatch.py", "_cache_fully_retained")]
        assert cfm.excused_symbols("mlx_vlm/server/openai.py", exclusions) == []

    def test_honours_a_path_glob(self, cfm):
        exclusions = [("mlx_vlm/server/*.py", "_check_configured_context_budget")]
        assert cfm.excused_symbols("mlx_vlm/server/generation.py", exclusions) == [
            "_check_configured_context_budget"
        ]
        assert cfm.excused_symbols("mlx_vlm/apc.py", exclusions) == []


class TestSymbolExclusionsParsing:
    def test_reads_the_real_file_without_raising(self, cfm):
        """Lenient by design — `check_upstream_symbols.py` validates the format.

        Duplicating its `sys.exit` calls here would mean two scripts to fix when the
        format drifts, so this one skips malformed lines. That is only safe while it
        stays a *parser* and never the validator, which this asserts by shape.
        """
        parsed = cfm.load_symbol_exclusions()

        assert parsed, "the repo's .symbol-exclusions should not read as empty"
        for path_glob, symbol_glob in parsed:
            assert path_glob and symbol_glob
            assert "#" not in path_glob and "#" not in symbol_glob
            assert path_glob == path_glob.strip()

    def test_the_dispatch_pair_is_still_what_case_4_rests_on(self, cfm):
        """Ties the rule to the real entries it was built for.

        If either entry is ever pruned, `generate/dispatch.py` starts reporting its
        deletion-only hunk again — which is correct, and this test says so out loud
        rather than leaving the next reader to rediscover the link.
        """
        excused = cfm.excused_symbols(
            "mlx_vlm/generate/dispatch.py", cfm.load_symbol_exclusions()
        )

        assert "_cache_fully_retained" in excused
        assert "_prefix_cache_trim_amount" in excused
