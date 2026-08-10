"""Tests for `dev/fork_coverage_report.py`'s span arithmetic.

Fifth test of a `dev/` script. This one is a report rather than a gate, so the risk is
misattribution: the whole output rests on mapping coverage's line numbers onto
definition spans, and an off-by-one there moves statements between definitions
silently — a covered neighbour would mask an uncovered one, and the report would read
green over a gap.

The coverage-data half is not unit-tested: it needs a real `.coverage` file, and it is
`coverage`'s own `analysis2` doing the work. What is tested here is the part this script
actually owns.
"""

import importlib.util
from pathlib import Path

import pytest

_DEV = Path(__file__).resolve().parents[2] / "dev" / "fork_coverage_report.py"


def _load():
    spec = importlib.util.spec_from_file_location("_fcr_under_test", _DEV)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fcr():
    assert _DEV.is_file(), f"missing {_DEV}"
    return _load()


class TestTopLevelDefs:
    def test_span_covers_the_whole_definition(self, fcr):
        source = "def f():\n    a = 1\n    return a\n"
        assert fcr.top_level_defs(source) == {"f": (1, 3)}

    def test_decorators_are_included_in_the_span(self, fcr):
        """`@decorator` lines are executable statements that coverage attributes to the
        file. Excluding them from the span drops them from every decorated definition's
        totals — silently understating both the statement count and the misses.
        """
        source = "@deco\n@deco2\ndef f():\n    return 1\n"
        assert fcr.top_level_defs(source) == {"f": (1, 4)}

    def test_adjacent_definitions_do_not_overlap(self, fcr):
        """The property that makes per-definition attribution meaningful at all."""
        source = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
        spans = fcr.top_level_defs(source)
        assert spans == {"a": (1, 2), "b": (5, 6)}
        (a_start, a_end), (b_start, _b_end) = spans["a"], spans["b"]
        assert a_end < b_start

    def test_a_class_span_covers_its_methods(self, fcr):
        """Methods are attributed to the enclosing class, matching the granularity the
        rest of `dev/` uses."""
        source = "class C:\n    def m(self):\n        return 1\n"
        assert fcr.top_level_defs(source) == {"C": (1, 3)}

    def test_nested_definitions_are_not_reported_separately(self, fcr):
        source = "def outer():\n    def inner():\n        return 1\n    return inner\n"
        assert list(fcr.top_level_defs(source)) == ["outer"]

    def test_async_definitions_are_found(self, fcr):
        source = "async def f():\n    return 1\n"
        assert fcr.top_level_defs(source) == {"f": (1, 2)}

    def test_module_level_statements_are_not_definitions(self, fcr):
        source = "X = 1\nimport os\n\n\ndef f():\n    return X\n"
        assert list(fcr.top_level_defs(source)) == ["f"]

    def test_a_syntax_error_propagates(self, fcr):
        with pytest.raises(SyntaxError):
            fcr.top_level_defs("def f(:\n")
