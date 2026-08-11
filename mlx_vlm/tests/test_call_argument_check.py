"""Tests for `dev/check_call_arguments.py`, the dropped-keyword-argument audit.

A fork-only file, and the third test of any `dev/` audit script after
`test_fork_marker_check.py` and `test_body_divergence_check.py`. Same reason those
exist: the audits are the safety net for every merge, so a bug that makes one MORE
PERMISSIVE is the worst shape available here — nothing fails, the check still prints
OK, and dropped content stops being reported.

For this script "more permissive" has a specific meaning: **failing to report a
keyword name upstream passes and we do not.** That is the direction most of these
tests guard, and it is not hypothetical — the founding instance
(`stream_diffusion_generate_from_kwargs` losing `skip_special_tokens=` and
`verbose=`) sat behind a `# Fork:` marker claiming completeness while seven gating
audits and 3001 tests stayed green.

The opposite direction — reporting fork additions, renamed access paths, or differing
argument *values* — only produces noise, but it is tested just as carefully, because
a noisy gate gets switched off and then protects nothing. `.fork-marker-allowlist`
took a months-long rollout to drain for exactly that reason.
"""

import importlib.util
from pathlib import Path

import pytest

_DEV = Path(__file__).resolve().parents[2] / "dev" / "check_call_arguments.py"


def _load():
    spec = importlib.util.spec_from_file_location("_cca_under_test", _DEV)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cca():
    assert _DEV.is_file(), f"missing {_DEV}"
    return _load()


def _findings(cca, upstream: str, ours: str):
    """Findings for one synthetic file pair, keyed as (definition, callee, missing)."""
    up = cca.CallSites(upstream)
    our = cca.CallSites(ours)
    out = []
    for definition, up_calls in up.by_definition.items():
        our_calls = our.by_definition.get(definition)
        if our_calls is None:
            continue
        for callee, (up_names, _) in up_calls.items():
            if callee not in our_calls:
                continue
            our_names, star = our_calls[callee]
            missing = up_names - our_names
            if missing:
                out.append((definition, callee, frozenset(missing), star))
    return out


class TestTheFoundingInstance:
    """The exact shape that motivated the script, reduced to its essentials."""

    UP = """\
def stream_generate(model, **kwargs):
    skip_special_tokens = kwargs.pop("skip_special_tokens", False)
    verbose = kwargs.pop("verbose", False)
    yield from stream_diffusion_generate_from_kwargs(
        model,
        ids,
        kwargs,
        skip_special_tokens=skip_special_tokens,
        verbose=verbose,
    )
"""

    OURS = """\
def stream_generate(model, **kwargs):
    skip_special_tokens = kwargs.pop("skip_special_tokens", False)
    verbose = kwargs.pop("verbose", False)
    yield from stream_diffusion_generate_from_kwargs(
        model,
        ids,
        kwargs,
        )
"""

    def test_both_dropped_kwargs_are_reported(self, cca):
        (finding,) = _findings(cca, self.UP, self.OURS)
        definition, callee, missing, star = finding
        assert definition == "stream_generate"
        assert callee == "stream_diffusion_generate_from_kwargs"
        assert missing == frozenset({"skip_special_tokens", "verbose"})

    def test_a_kwargs_dict_passed_POSITIONALLY_is_not_star_forwarding(self, cca):
        """The founding bug's callee takes `kwargs` as a positional argument.

        That is not `**kwargs`, so the hit must NOT be tagged `[**]` — tagging it
        would have understated the one finding this script exists to make.
        """
        (finding,) = _findings(cca, self.UP, self.OURS)
        assert finding[3] is False

    def test_the_fix_makes_it_silent(self, cca):
        assert _findings(cca, self.UP, self.UP) == []


class TestWhatMustStaySilent:
    """False positives. A noisy gate gets switched off, and then protects nothing."""

    def test_a_fork_ADDED_kwarg_is_not_a_finding(self, cca):
        """`ours > upstream` is fork work. `_make_cache`'s prealloc kwargs are this."""
        up = "def _make_cache(m):\n    return BatchKVCache(lp)\n"
        ours = "def _make_cache(m):\n    return BatchKVCache(lp, prealloc_tokens=8)\n"
        assert _findings(cca, up, ours) == []

    def test_a_differing_VALUE_for_the_same_name_is_not_a_finding(self, cca):
        """Names, never values — the same rule as check_upstream_registries.py.

        `make_cache=_make_cache` against
        `make_cache=functools.partial(_make_cache, kv_prealloc_tokens=...)` is a real
        site in this tree and a fork enhancement. Comparing values would report it.
        """
        up = "def f():\n    g(make_cache=_make_cache)\n"
        ours = (
            "def f():\n"
            "    g(make_cache=functools.partial(_make_cache, kv_prealloc_tokens=0))\n"
        )
        assert _findings(cca, up, ours) == []

    def test_a_changed_ACCESS_PATH_to_the_same_callee_is_not_a_finding(self, cca):
        """The fork re-imports helpers constantly; keying on the dotted path would
        make every one of those a spurious hit and bury the real ones."""
        up = "def f():\n    common.maybe_quantize_kv_cache(c, bits=4)\n"
        ours = "def f():\n    maybe_quantize_kv_cache(c, bits=4)\n"
        assert _findings(cca, up, ours) == []

    def test_a_definition_we_do_not_have_is_not_a_finding(self, cca):
        """That is check_upstream_symbols.py's job, and it has its own exclusions."""
        up = "def only_upstream():\n    g(a=1)\n"
        ours = "def something_else():\n    pass\n"
        assert _findings(cca, up, ours) == []

    def test_a_dropped_CALL_is_not_reported_as_a_dropped_ARGUMENT(self, cca):
        """A vanished call site belongs to check_dead_helpers.py.

        Reporting it here would double-count it and, worse, describe it wrongly:
        "passes fewer keyword arguments" is not what happened.
        """
        up = "def f():\n    g(a=1)\n"
        ours = "def f():\n    pass\n"
        assert _findings(cca, up, ours) == []

    def test_kwargs_are_unioned_across_MULTIPLE_call_sites(self, cca):
        """One definition can call the same helper several ways.

        Upstream passing `a=` at one site and `b=` at another must not report when
        ours passes `a=` at one and `b=` at the other — nothing is missing.
        """
        up = "def f():\n    g(a=1)\n    g(b=2)\n"
        ours = "def f():\n    g(b=2)\n    g(a=1)\n"
        assert _findings(cca, up, ours) == []


class TestStarForwardingIsTaggedNotFiltered:
    """`[**]` hits are ambiguous, and reported anyway — see the module docstring."""

    def test_star_forwarding_is_still_reported(self, cca):
        """The name MAY arrive through the dict. It may also not.

        Filtering these would be the permissive failure this file guards: our
        `basicConfig(**log_kwargs)` really does supply the names, but a call that
        forwards an unrelated dict really does not, and nothing distinguishes them
        mechanically. So: report, tag, and require a written reason.
        """
        up = "def f():\n    g(level=1, format='x')\n"
        ours = "def f():\n    g(**opts)\n"
        (finding,) = _findings(cca, up, ours)
        assert finding[2] == frozenset({"level", "format"})
        assert finding[3] is True, "star forwarding must be tagged"

    def test_explicit_kwargs_alongside_star_still_narrow_the_finding(self, cca):
        up = "def f():\n    g(level=1, format='x')\n"
        ours = "def f():\n    g(level=1, **opts)\n"
        (finding,) = _findings(cca, up, ours)
        assert finding[2] == frozenset({"format"})


class TestModuleScopeIsCovered:
    """Module-scope calls, filed under `<module>`.

    Omitted from the first draft, which would have left a real hole: a registration or
    middleware call at module scope (`app.add_middleware(...)`) is exactly the shape
    that loses a keyword in a merge, and `check_body_divergence.py`'s `gone` cannot see
    module scope either — so nothing at all would have covered it. Measured before
    adding: zero hits tree-wide, so the coverage was free.
    """

    def test_a_dropped_kwarg_at_module_scope_is_reported(self, cca):
        up = "app.add_middleware(CORS, allow_origins=['*'], allow_headers=['x'])\n"
        ours = "app.add_middleware(CORS, allow_origins=['*'])\n"
        (finding,) = _findings(cca, up, ours)
        assert finding[0] == cca.MODULE_SCOPE
        assert finding[2] == frozenset({"allow_headers"})

    def test_module_scope_is_filed_under_a_non_identifier(self, cca):
        """So it cannot collide with a real definition named `module`."""
        assert not cca.MODULE_SCOPE.isidentifier()

    def test_module_scope_calls_are_unioned_across_statements(self, cca):
        """Two statements at module scope are one scope, not two.

        Collected per-statement then merged, so a name passed at one site must excuse
        the other — the same union rule as multiple call sites inside a definition.
        """
        up = "g(a=1)\ng(b=2)\n"
        ours = "g(b=2)\ng(a=1)\n"
        assert _findings(cca, up, ours) == []

    def test_a_file_with_no_module_level_statements_has_no_module_entry(self, cca):
        """Avoids a spurious empty `<module>` row on a pure definitions file."""
        sites = cca.CallSites("def f():\n    pass\n")
        assert cca.MODULE_SCOPE not in sites.by_definition

    def test_definition_scope_and_module_scope_do_not_bleed(self, cca):
        """A kwarg passed inside a function must not excuse the module-scope call."""
        up = "g(a=1)\n\n\ndef f():\n    g(a=1)\n"
        ours = "g()\n\n\ndef f():\n    g(a=1)\n"
        (finding,) = _findings(cca, up, ours)
        assert finding[0] == cca.MODULE_SCOPE


class TestCalleeNameResolution:
    def test_plain_and_dotted_callees_both_resolve_to_the_simple_name(self, cca):
        import ast

        assert cca.callee_name(ast.parse("f()").body[0].value.func) == "f"
        assert cca.callee_name(ast.parse("a.b.f()").body[0].value.func) == "f"

    def test_an_unnameable_callee_is_skipped_rather_than_crashing(self, cca):
        """`funcs[0](...)` and `(lambda: g)()(...)` have no simple name.

        A crash here would take the whole audit down on one exotic call site, which
        for a gating script means it stops running at all.
        """
        import ast

        assert cca.callee_name(ast.parse("funcs[0]()").body[0].value.func) is None
        assert (
            _findings(cca, "def f():\n    h[0](a=1)\n", "def f():\n    h[0]()\n") == []
        )


class TestNestedAndMethodCalls:
    def test_a_call_inside_a_METHOD_is_found(self, cca):
        """`walk` descends the whole definition, so class bodies are covered.

        The founding instance's sibling — `server/generation.py`'s
        `_run_diffusion` — is a method, and #1716's dropped call site was one too.
        """
        up = "class C:\n    def m(self):\n        g(a=1, b=2)\n"
        ours = "class C:\n    def m(self):\n        g(a=1)\n"
        (finding,) = _findings(cca, up, ours)
        assert finding[0] == "C"
        assert finding[2] == frozenset({"b"})


class TestExclusionsContract:
    """Same contract as every other audit: three fields, and a reason is mandatory."""

    def test_a_reasonless_exclusion_is_a_hard_error(self, cca, tmp_path, monkeypatch):
        path = tmp_path / ".call-argument-exclusions"
        path.write_text("a.py::f::g\n")
        monkeypatch.setattr(cca, "EXCLUSIONS_FILE", path)
        with pytest.raises(SystemExit):
            cca.load_exclusions()

    def test_a_two_field_exclusion_is_a_hard_error(self, cca, tmp_path, monkeypatch):
        """Two fields would silently excuse every callee in the definition."""
        path = tmp_path / ".call-argument-exclusions"
        path.write_text("a.py::f  # reason\n")
        monkeypatch.setattr(cca, "EXCLUSIONS_FILE", path)
        with pytest.raises(SystemExit):
            cca.load_exclusions()

    def test_globs_match_on_all_three_fields(self, cca, tmp_path, monkeypatch):
        path = tmp_path / ".call-argument-exclusions"
        path.write_text("mlx_vlm/*.py::main::basicConfig  # reason\n")
        monkeypatch.setattr(cca, "EXCLUSIONS_FILE", path)
        exclusions = cca.load_exclusions()
        assert cca.is_excused(
            cca.Finding("mlx_vlm/x.py", "main", "basicConfig", {"level"}, True),
            exclusions,
        )

    def test_an_exclusion_does_not_leak_to_a_different_callee(
        self, cca, tmp_path, monkeypatch
    ):
        """The reason this file uses three fields instead of two."""
        path = tmp_path / ".call-argument-exclusions"
        path.write_text("mlx_vlm/x.py::main::basicConfig  # reason\n")
        monkeypatch.setattr(cca, "EXCLUSIONS_FILE", path)
        exclusions = cca.load_exclusions()
        assert not cca.is_excused(
            cca.Finding("mlx_vlm/x.py", "main", "something_else", {"level"}, False),
            exclusions,
        )

    def test_an_empty_exclusions_list_excuses_nothing(self, cca):
        assert not cca.is_excused(cca.Finding("a.py", "f", "g", {"x"}, False), [])


class TestUnparseableInputIsReportedNotSwallowed:
    def test_a_syntax_error_surfaces(self, cca):
        """A check that goes quiet on an unparseable file is the failure mode dev/
        exists to prevent — the same contract as `definition_bodies`."""
        with pytest.raises(SyntaxError):
            cca.CallSites("def f(:\n")


class TestTheRealTreeIsGreen:
    """The gate's own state, so a regression in this fork fails HERE too."""

    def test_the_dispatch_diffusion_call_site_passes_both_kwargs(self):
        """Belt-and-braces against the founding instance, read from real source.

        `test_dropped_upstream_guards.py` proves it at runtime; this proves the
        audit's own subject matter did not silently regress.
        """
        import inspect

        from mlx_vlm.generate import dispatch

        source = inspect.getsource(dispatch.stream_generate)
        call = source.split("stream_diffusion_generate_from_kwargs(", 1)
        assert len(call) == 2, "the diffusion dispatch call site is gone"
        body = call[1].split("return", 1)[0]
        assert "skip_special_tokens=skip_special_tokens" in body
        assert "verbose=verbose" in body
