"""Tests for `dev/find_untested_fork_code.py`'s counting.

Fourth test of any `dev/` script. This one does not gate, so the usual "a permissive
bug prints OK forever" argument applies differently: a miscounting *lead generator*
sends people to the wrong place, or — worse — reports everything as untested and gets
ignored as noise.

Which is not hypothetical. The first run of this analysis reported all 95 fork-only
definitions as untested, because it classified files with
`("/tests/" in p and tests or library)[p] = text`. With `tests` still an empty dict on
the first iteration, `True and {}` is `{}`, which is falsy, so `{} or library` selected
`library` — every test file landed in the library corpus and every test count was 0.
`test_test_files_are_classified_as_tests` is that bug.
"""

import importlib.util
from pathlib import Path

import pytest

_DEV = Path(__file__).resolve().parents[2] / "dev" / "find_untested_fork_code.py"


def _load():
    spec = importlib.util.spec_from_file_location("_fufc_under_test", _DEV)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fufc():
    assert _DEV.is_file(), f"missing {_DEV}"
    return _load()


class TestPathClassification:
    @pytest.mark.parametrize(
        "path",
        [
            "mlx_vlm/tests/test_server.py",
            "mlx_vlm/tests/conftest.py",
            "some/nested/tests/test_x.py",
        ],
    )
    def test_test_files_are_classified_as_tests(self, fufc, path):
        """The regression that made the first run report 95/95 untested."""
        assert fufc.is_test_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "mlx_vlm/server/openai.py",
            "mlx_vlm/utils.py",
            "mlx_vlm/models/gemma4/language.py",
        ],
    )
    def test_library_files_are_not(self, fufc, path):
        assert fufc.is_test_path(path) is False


class TestReferenceCount:
    CORPUS = {
        "a.py": "helper()\nhelper()\n",
        "b.py": "other(helper)\n",
        "c.py": "nothing here\n",
    }

    def test_counts_every_occurrence(self, fufc):
        assert fufc.reference_count("helper", self.CORPUS) == 3

    def test_excludes_the_defining_file(self, fufc):
        assert fufc.reference_count("helper", self.CORPUS, exclude="a.py") == 1

    def test_absent_name_is_zero(self, fufc):
        assert fufc.reference_count("absent", self.CORPUS) == 0

    def test_matching_is_whole_word(self, fufc):
        """A substring match would count `_trim_cache` inside `_trim_cache_for_apc`
        and report coverage that does not exist — the permissive direction."""
        corpus = {"a.py": "_trim_cache_for_apc()\nmy_trim_cache = 1\n"}
        assert fufc.reference_count("_trim_cache", corpus) == 0

    def test_an_empty_corpus_is_zero(self, fufc):
        assert fufc.reference_count("helper", {}) == 0


class TestTopLevelNames:
    def test_module_level_only(self, fufc):
        source = (
            "def outer():\n    def inner():\n        pass\n\n\nclass C:\n"
            "    def method(self):\n        pass\n"
        )
        assert fufc.top_level_names(source) == {"outer", "C"}

    def test_async_defs_count(self, fufc):
        assert fufc.top_level_names("async def endpoint():\n    pass\n") == {"endpoint"}

    def test_a_syntax_error_propagates(self, fufc):
        """main() collects these into a visible warning rather than skipping quietly."""
        with pytest.raises(SyntaxError):
            fufc.top_level_names("def f(:\n")
