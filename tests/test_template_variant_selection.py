"""The tool must serve the template that was asked for, or say it cannot.

A usability measurement drove `prepare_simulation` over six realistic tasks on
three backends and got a working deck for none of them. The cause was not
missing knowledge. Three call sites read `template_variants[0]` and nothing
else, so:

    prepare_simulation('fenics', '3d linear elasticity')  ->  the 2D template
    prepare_simulation('fourc',  '3d linear elasticity')  ->  linear_2d
    prepare_simulation('fenics', 'heat with time dependent BC') -> 2d_steady

while a `3d` variant and a `2d_transient` variant sat in the catalog, correct
and unreachable by any tool.

Two properties are asserted here, and the second matters more than the first.

  * When a variant satisfying the request exists, serve it.
  * When one does NOT exist, SAY SO. Silent substitution is the failure a weak
    model cannot detect: handed a deck labelled "2D plane stress" for a 3D task,
    it ships the 2D deck. No amount of prose above the template changes that,
    which is why this is a retrieval fix and not a knowledge fix.
"""
from __future__ import annotations

import asyncio
import inspect
import re

import pytest

from tools import consolidated as C
from tools.consolidated import _select_template_variant


@pytest.fixture(scope="module")
def tools():
    from core.registry import load_all_backends
    load_all_backends()
    captured = {}

    class _Recorder:
        def tool(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn
            return deco

    C.register_consolidated_tools(_Recorder())
    return captured


def _call(fn, *a, **k):
    r = fn(*a, **k)
    return asyncio.run(r) if inspect.iscoroutine(r) else r


def _variant(text: str) -> str:
    m = re.search(r"## Template \(([^)]+)\)", text)
    return m.group(1) if m else ""


# ── the selector itself ───────────────────────────────────────────────────
@pytest.mark.parametrize("query, variants, expected", [
    ("3d linear elasticity", ["2d", "3d", "plate_hole"], "3d"),
    ("linear elasticity", ["2d", "3d"], "2d"),                  # no qualifier
    ("transient heat", ["2d_steady", "2d_transient"], "2d_transient"),
    ("steady heat", ["2d_transient", "2d_steady"], "2d_steady"),
    ("nonlinear 3d solid", ["linear_2d", "nonlinear_3d"], "nonlinear_3d"),
    ("three-dimensional elasticity", ["2d", "3d"], "3d"),
    ("time-dependent conduction", ["steady", "transient"], "transient"),
])
def test_a_qualifier_in_the_request_selects_the_variant(query, variants, expected):
    chosen, _ = _select_template_variant(query, variants)
    assert chosen == expected


def test_an_unsatisfiable_qualifier_is_stated_not_swallowed():
    chosen, note = _select_template_variant("3d linear elasticity", ["2d"])
    assert chosen == "2d"
    assert "no template variant provides it" in note
    assert "does NOT satisfy" in note


def test_siblings_are_always_named():
    _, note = _select_template_variant("elasticity", ["2d", "3d", "thick_beam"])
    assert "3d" in note and "thick_beam" in note


def test_no_variants_is_not_a_crash():
    assert _select_template_variant("anything", []) == ("", "")


# ── through the real tool, on the cases that failed ───────────────────────
def test_3d_request_gets_a_3d_template(tools):
    out = _call(tools["prepare_simulation"], "fenics", "3d linear elasticity")
    assert _variant(out) == "3d", _variant(out)
    # and the served deck must actually be 3D
    assert "create_box" in out


def test_transient_request_gets_a_transient_template(tools):
    out = _call(tools["prepare_simulation"], "fenics",
                "heat with time dependent boundary condition")
    assert "transient" in _variant(out), _variant(out)


def test_a_backend_without_the_variant_warns(tools):
    """scikit-fem has no 3D elasticity variant. Serving the 2D one silently is
    how a weak model ships a plane-stress deck for a 3D problem."""
    out = _call(tools["prepare_simulation"], "skfem", "3d linear elasticity")
    assert "⚠" in out
    assert "no template variant provides it" in out


def test_a_variant_can_be_requested_by_name(tools):
    out = _call(tools["examples"], "linear_elasticity", solver="fenics",
                action="template", variant="3d")
    assert "create_box" in out


def test_an_unknown_variant_lists_the_real_ones(tools):
    out = _call(tools["examples"], "linear_elasticity", solver="fenics",
                action="template", variant="9d")
    assert "No variant" in out and "3d" in out


def test_no_call_site_still_reads_variant_zero_blindly():
    """The regression guard. Three sites read `template_variants[0]`; if one
    comes back, the measured failure comes back with it."""
    import ast
    import pathlib
    src = pathlib.Path(C.__file__).read_text()
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        val = node.value
        if getattr(val, "attr", None) != "template_variants":
            continue
        sl = node.slice
        # variants[0] or variants[:1] — both pick blindly.
        if (isinstance(sl, ast.Constant) and sl.value == 0) or isinstance(sl, ast.Slice):
            offenders.append(node.lineno)
    assert not offenders, (
        "a call site picks template_variants[0] without consulting the request, "
        f"at line(s) {offenders}. Use _select_template_variant.")
