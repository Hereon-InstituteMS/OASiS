"""A template must not claim to be a backend it does not use.

OASiS is going to be read and picked apart by the computational-mechanics
community. The single most damaging thing it could contain is a template that
names a solver and quietly solves with something else — a reviewer who finds one
is entitled to distrust everything around it.

There were real instances. DUNE's `linear_elasticity_2d` was literally
`return _poisson_2d(params)` and printed "DUNE-fem Poisson solve complete."
under an elasticity name; its `stokes_2d` was a self-described Poisson "proxy".
Both were repaired on their branch. This guard is what stops the next one.

WHAT IS AND IS NOT A SURROGATE, because the distinction decides whether a thing
is dishonest or merely limited:

  * A template that solves the physics with a DIFFERENT library, while SAYING SO
    in its docstring, is legitimate. Eight Kratos templates assemble with
    numpy/scipy and seven of them say "Kratos (manual assembly)" or
    "(standalone)" on the first line; the server instructions route them to
    `run_simulation` accordingly. The physics is right and the reader is told.
  * The same template with an unqualified "— Kratos Multiphysics" docstring is
    NOT legitimate, and exactly one was in that state. A weak model reads the
    first line and stops; telling it "Kratos Multiphysics" when the code is
    scipy is the opposite of the truth.
  * A reference stub that says "Not a runnable input — the user must supply the
    case-specific mesh" and lists the requirements is legitimate. Twenty 4C
    entries are of that kind. Declining to fabricate a mesh you cannot know is
    honesty, not a gap.

So the rule enforced here is narrow and defensible: if a template does not use
its own backend, its docstring must say so.
"""
from __future__ import annotations

import ast
import functools
import re

import pytest

# How to tell a template really drives its own backend.
_BACKEND_MARKERS = {
    "fenics": ("dolfinx", "ufl"),
    "dune": ("dune",),
    "ngsolve": ("ngsolve", "netgen"),
    "skfem": ("skfem",),
    "kratos": ("KratosMultiphysics",),
    "febio": ("febio_spec",),
    "fourc": ("PROBLEM TYPE", "PROBLEMTYPE"),
    "sparta": ("create_box", "species", "global "),
}

# Wording that tells a reader the template does not use the backend itself.
_DISCLOSES = re.compile(
    r"manual assembly|standalone|without Kratos|does not use|no Kratos"
    # "NOT a runnable 4C input" — allow words between, since the first version
    # required the phrase verbatim and so accused five honest 4C entries whose
    # header says exactly that with the solver name in the middle.
    r"|not\s+(?:a\s+)?(?:\w+\s+){0,3}runnable"
    r"|reference stub|meta-reference|umbrella"
    r"|scipy|numpy-only|proxy|placeholder|surrogate|stub", re.I)


@functools.lru_cache(maxsize=1)
def _templates():
    from core.registry import get_backend, list_backends, load_all_backends

    load_all_backends()
    out = []
    for entry in list_backends():
        be = get_backend(entry["name"])
        if not be:
            continue
        for p in be.supported_physics():
            for v in list(p.template_variants):
                try:
                    out.append((entry["name"], f"{p.name}/{v}",
                                be.generate_input(p.name, v, {})))
                except Exception:
                    continue                      # generator failures: other tests
    return tuple(out)


def _header(text: str) -> str:
    """The docstring, or the leading comment block — what a reader sees first."""
    try:
        doc = ast.get_docstring(ast.parse(text))
        if doc:
            return doc
    except (SyntaxError, ValueError):
        pass
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#") or not s:
            # Skip pure divider rules. A first version returned `# =====` as the
            # header and then accused five honest 4C entries whose very next line
            # reads "This is NOT a runnable 4C input" — a guard that misreads the
            # text it judges is worse than none.
            if s.strip("#= \t-*_"):
                lines.append(s)
            if len(lines) > 25:
                break
        elif lines:
            break
    return "\n".join(lines) or text[:400]


def test_a_template_that_does_not_use_its_backend_says_so():
    """The narrow, defensible rule. Exactly one template failed it: a Kratos
    Poisson script whose docstring read "— Kratos Multiphysics" while the code
    assembled with scipy."""
    offenders = []
    for backend, name, text in _templates():
        markers = _BACKEND_MARKERS.get(backend)
        if not markers or any(m in text for m in markers):
            continue
        if _DISCLOSES.search(_header(text)):
            continue
        offenders.append(f"{backend}/{name}: {_header(text).splitlines()[0][:90]!r}")
    assert not offenders, (
        "these templates do not use their own backend and do not say so, so a "
        "reader is told the opposite of the truth:\n  " + "\n  ".join(offenders)
        + "\n\nEither drive the backend, or state plainly in the first line that "
          "the script assembles the problem itself.")


def test_the_guard_catches_the_shape_it_was_written_for():
    """Calibration on both the real defect and the legitimate cases it must not
    accuse — a guard that flagged all 8 manual-assembly Kratos templates, or the
    20 honest 4C stubs, would be deleted within a week."""
    misleading = '"""Poisson on the unit square — Kratos Multiphysics"""\nimport scipy\n'
    assert not _DISCLOSES.search(_header(misleading))

    for honest in (
        '"""Heat conduction — Kratos (manual assembly)"""\nimport numpy\n',
        '"""MPM — large-deformation solid — Kratos (standalone)"""\n',
        "# 4C reference stub: ssi / monolithic_elch_3d\n"
        "# Not a runnable input — the user must supply the mesh\n",
    ):
        assert _DISCLOSES.search(_header(honest)), honest


def test_no_template_delegates_to_a_different_physics():
    """DUNE's `linear_elasticity_2d` was `return _poisson_2d(params)`. A template
    whose body hands off to another physics' generator is a surrogate however it
    is worded, so this reads the source rather than the output."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "backends"
    offenders = []
    for path in sorted(root.glob("*/generators/*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        all_fns = [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        def _template_shaped(nm: str) -> bool:
            return nm.startswith("_") and (
                "template" in nm or "_2d" in nm or "_3d" in nm or "_1d" in nm)

        # Locally defined AND IMPORTED template functions. The first version
        # collected only local definitions and so missed the one real surrogate
        # in the repo: DUNE's `_elasticity_2d` returns `_poisson_2d(params)`,
        # and `_poisson_2d` is imported `from .poisson`. The test passed and I
        # only noticed because I checked the file by hand instead of trusting it.
        template_fns = {n.name for n in all_fns if _template_shaped(n.name)}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    nm = a.asname or a.name
                    if _template_shaped(nm):
                        template_fns.add(nm)
        for fn in all_fns:
            body = [s for s in fn.body if not isinstance(s, ast.Expr)]
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            call = body[0].value
            if not isinstance(call, ast.Call):
                continue
            callee = str(getattr(call.func, "id", None)
                         or getattr(call.func, "attr", ""))
            # Only a call to ANOTHER TEMPLATE FUNCTION counts. A first version
            # compared name prefixes and flagged `sorted`, `dedent`, `format` and
            # `get` — ordinary one-line helpers, 20-odd false accusations against
            # one real finding. A delegation target must itself be a private
            # template-shaped function defined in this file.
            if not callee.startswith("_"):
                continue
            if callee not in template_fns or callee == fn.name:
                continue
            mine = fn.name.strip("_").split("_")[0]
            theirs = callee.strip("_").split("_")[0]
            if theirs != mine:
                offenders.append(
                    f"{path.parent.parent.name}/{path.name}:{fn.lineno} "
                    f"{fn.name} -> {callee}")
    assert not offenders, (
        "a template generator delegates straight to a different physics:\n  "
        + "\n  ".join(offenders))
