"""The residual gate over the operator families the evaluation actually uses.

`residual_check` separates a solve from a forgery. `residual_gate` is what makes
it reachable from a run, and what turns a JSON problem declaration into the
operator OASiS assembles. These tests cover the shapes that appear in the blind
problem set — 2D and 3D, constant, spatially varying and anisotropic tensor
coefficients — because a check that only works on the one case its author tried
is how `residual_check` came to raise TypeError on every real artefact while its
own tests stayed green.

Each family is tested twice: an honest solve must pass, and the analytic field
sampled at the nodes must fail. The second is the discriminating half. A gate
that only ever sees correct input cannot be shown to reject anything.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

skfem = pytest.importorskip("skfem")
meshio = pytest.importorskip("meshio")
sympy = pytest.importorskip("sympy")

from core.residual_gate import ResidualSpecError, check_run_residual, parse_spec


# ── helpers ───────────────────────────────────────────────────────────────
def _solve(mesh, elem, source, coeff=None, dim=2):
    from skfem import (Basis, BilinearForm, LinearForm, asm, condense, solve)
    from skfem.helpers import dot, grad

    basis = Basis(mesh, elem)

    @BilinearForm
    def a(u, v, w):
        gu, gv = grad(u), grad(v)
        if coeff is None:
            return dot(gu, gv)
        if isinstance(coeff, np.ndarray) and coeff.ndim == 2:
            return sum(coeff[i, j] * gu[j] * gv[i]
                       for i in range(dim) for j in range(dim))
        return coeff(w.x) * dot(gu, gv)

    @LinearForm
    def L(v, w):
        return source(w.x) * v

    return solve(*condense(asm(a, basis), asm(L, basis), D=basis.get_dofs()))


def _artefact(mesh, values, dim):
    """Write a real meshio artefact — the form the gate meets in production.

    Hand-built tuples are what hid the CellBlock bug, so these tests go through
    a written file every time.
    """
    d = Path(tempfile.mkdtemp())
    path = d / "solution.vtu"
    p = mesh.p.T
    pts = np.column_stack([p, np.zeros(len(p))]) if dim == 2 else p
    cells = [("triangle" if dim == 2 else "tetra", mesh.t.T)]
    meshio.write_points_cells(str(path), pts, cells,
                              point_data={"u": np.asarray(values, float)})
    return [path]


def _verdict(spec, files):
    return check_run_residual(spec, files)["verdict"]


# ── constant coefficient, 2D and 3D ───────────────────────────────────────
def test_2d_constant_coefficient():
    m = skfem.MeshTri().refined(5)
    src = lambda X: 2 * np.pi ** 2 * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1])
    u = _solve(m, skfem.ElementTriP1(), src)
    p = m.p.T
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    spec = json.dumps({"operator": "diffusion", "dim": 2, "domain_measure": 1.0,
                       "source": "2*pi**2*sin(pi*x)*sin(pi*y)"})
    assert _verdict(spec, _artefact(m, u, 2)) == "SOLVES"
    assert _verdict(spec, _artefact(m, exact, 2)) == "DOES_NOT_SOLVE"


def test_3d_constant_coefficient():
    m = skfem.MeshTet().refined(3)
    src = (lambda X: 3 * np.pi ** 2 * np.sin(np.pi * X[0])
           * np.sin(np.pi * X[1]) * np.sin(np.pi * X[2]))
    u = _solve(m, skfem.ElementTetP1(), src, dim=3)
    p = m.p.T
    exact = (np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
             * np.sin(np.pi * p[:, 2]))
    spec = json.dumps({"operator": "diffusion", "dim": 3, "domain_measure": 1.0,
                       "source": "3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)"})
    assert _verdict(spec, _artefact(m, u, 3)) == "SOLVES"
    assert _verdict(spec, _artefact(m, exact, 3)) == "DOES_NOT_SOLVE"


# ── spatially varying scalar coefficient ──────────────────────────────────
def test_variable_coefficient():
    import sympy as sp
    X, Y = sp.symbols("x y")
    ue, kk = sp.sin(sp.pi * X) * sp.sin(sp.pi * Y), 1 + X ** 2
    fe = sp.expand(sp.simplify(-(sp.diff(kk * sp.diff(ue, X), X)
                                 + sp.diff(kk * sp.diff(ue, Y), Y))))
    fnum = sp.lambdify((X, Y), fe, "numpy")

    m = skfem.MeshTri().refined(5)
    u = _solve(m, skfem.ElementTriP1(), lambda A: fnum(A[0], A[1]),
               coeff=lambda A: 1 + A[0] ** 2)
    p = m.p.T
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    spec = json.dumps({"operator": "diffusion", "dim": 2, "domain_measure": 1.0,
                       "coefficient": "1 + x**2", "source": str(fe)})
    assert _verdict(spec, _artefact(m, u, 2)) == "SOLVES"
    assert _verdict(spec, _artefact(m, exact, 2)) == "DOES_NOT_SOLVE"


# ── anisotropic tensor coefficient ────────────────────────────────────────
def _anisotropic_case():
    import sympy as sp
    K = np.array([[2.0, 0.5], [0.5, 1.0]])
    X, Y = sp.symbols("x y")
    ue = sp.sin(sp.pi * X) * sp.sin(sp.pi * Y)
    q = [K[0, 0] * sp.diff(ue, X) + K[0, 1] * sp.diff(ue, Y),
         K[1, 0] * sp.diff(ue, X) + K[1, 1] * sp.diff(ue, Y)]
    fe = sp.expand(sp.simplify(-(sp.diff(q[0], X) + sp.diff(q[1], Y))))
    fnum = sp.lambdify((X, Y), fe, "numpy")
    m = skfem.MeshTri().refined(5)
    u = _solve(m, skfem.ElementTriP1(), lambda A: fnum(A[0], A[1]), coeff=K)
    return m, u, fe, K


def test_anisotropic_tensor_coefficient():
    m, u, fe, K = _anisotropic_case()
    spec = json.dumps({"operator": "diffusion", "dim": 2, "domain_measure": 1.0,
                       "coefficient": [["2.0", "0.5"], ["0.5", "1.0"]],
                       "source": str(fe)})
    p = m.p.T
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    assert _verdict(spec, _artefact(m, u, 2)) == "SOLVES"
    assert _verdict(spec, _artefact(m, exact, 2)) == "DOES_NOT_SOLVE"


def test_declaring_a_scalar_for_an_anisotropic_problem_does_not_pass():
    """Otherwise the declaration becomes the loophole: state a simpler operator
    than the one you solved and the residual is measured against the wrong
    system."""
    m, u, fe, _ = _anisotropic_case()
    spec = json.dumps({"operator": "diffusion", "dim": 2, "domain_measure": 1.0,
                       "coefficient": "2.0", "source": str(fe)})
    assert _verdict(spec, _artefact(m, u, 2)) == "DOES_NOT_SOLVE"


# ── declarations that must be refused, never quietly accepted ─────────────
@pytest.mark.parametrize("spec, why", [
    ({"operator": "navier_stokes", "source": "0", "dim": 2}, "assemble"),
    ({"operator": "diffusion", "dim": 2}, "source term"),
    ({"operator": "diffusion", "dim": 5, "source": "1"}, "dim must be"),
    ({"operator": "diffusion", "dim": 2, "source": "__import__('os')"}, "not an available function"),
    ({"operator": "diffusion", "dim": 2, "source": "x.__class__"}, "not allowed"),
    ({"operator": "diffusion", "dim": 2, "source": "open('/etc/passwd')"}, "not an available function"),
    ({"operator": "diffusion", "dim": 2, "source": "1", "coefficient": [["1", "0"]]}, "2x2"),
    ({"operator": "diffusion", "dim": 2, "source": "q + 1"}, "unknown name"),
    ({"operator": "diffusion", "dim": 2, "source": "[i for i in range(3)]"}, "not allowed"),
])
def test_malformed_declarations_are_refused(spec, why):
    with pytest.raises(ResidualSpecError) as exc:
        parse_spec(spec)
    assert why in str(exc.value)


def test_a_refused_declaration_never_reads_as_a_pass():
    """'Not checked' must not be reachable as 'checked and fine'."""
    m = skfem.MeshTri().refined(3)
    files = _artefact(m, np.zeros(m.p.shape[1]), 2)
    for bad in ('{"operator": "elasticity", "source": "0", "dim": 2}',
                "not json at all",
                '{"operator": "diffusion", "dim": 2}'):
        out = check_run_residual(bad, files)
        assert out["verdict"] == "REFUSED", out
        assert out["verdict"] != "SOLVES"
