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


# ── the whole second-order linear scalar family, in one assembly ──────────
# Poisson, steady heat, convection-diffusion, reaction-diffusion and Helmholtz
# are the same operator with different terms present. Before this they were
# UNSUPPORTED, which is honest and is no protection: a fabricated result on any
# of them passed every other check in the gate.
_SCALAR_FAMILY = [
    ("poisson",              "1",        None,        None),
    ("diffusion",            "1 + x**2", None,        None),   # variable K
    ("convection_diffusion", "1",        ["1", "2"],  None),
    ("reaction_diffusion",   "1",        None,        "5"),
    # Helmholtz is c = -k^2: indefinite, and the sign must be allowed.
    ("helmholtz",            "1",        None,        "-16*pi**2"),
]


def _scalar_family_case(coeff, advection, reaction):
    """Manufacture u = sin(pi x) sin(pi y), derive f for the stated operator,
    and solve it honestly."""
    import sympy as sp
    from skfem import (Basis, BilinearForm, ElementTriP1, LinearForm, asm,
                       condense, solve)
    from skfem.helpers import dot, grad

    X, Y = sp.symbols("x y")
    ue = sp.sin(sp.pi * X) * sp.sin(sp.pi * Y)
    K = sp.sympify(coeff)
    b = [sp.sympify(e) for e in (advection or ["0", "0"])]
    c = sp.sympify(reaction) if reaction is not None else sp.Integer(0)
    f = sp.simplify(-(sp.diff(K * sp.diff(ue, X), X)
                      + sp.diff(K * sp.diff(ue, Y), Y))
                    + b[0] * sp.diff(ue, X) + b[1] * sp.diff(ue, Y) + c * ue)

    fl = sp.lambdify((X, Y), f, "numpy")
    Kl = sp.lambdify((X, Y), K, "numpy")
    bl = [sp.lambdify((X, Y), e, "numpy") for e in b]
    cl = sp.lambdify((X, Y), c, "numpy")

    m = skfem.MeshTri().refined(5)
    basis = Basis(m, ElementTriP1())

    @BilinearForm
    def a(u, v, w):
        out = Kl(w.x[0], w.x[1]) * dot(grad(u), grad(v))
        g = grad(u)
        out = out + (bl[0](w.x[0], w.x[1]) * g[0]
                     + bl[1](w.x[0], w.x[1]) * g[1]) * v
        return out + cl(w.x[0], w.x[1]) * np.ones_like(w.x[0]) * u * v

    @LinearForm
    def L(v, w):
        return fl(w.x[0], w.x[1]) * np.ones_like(w.x[0]) * v

    uh = solve(*condense(asm(a, basis), asm(L, basis), D=basis.get_dofs()))
    p = m.p.T
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    return m, uh, exact, str(f)


@pytest.mark.parametrize("operator, coeff, advection, reaction", _SCALAR_FAMILY,
                         ids=[c[0] for c in _SCALAR_FAMILY])
def test_scalar_family_certifies_a_solve_and_rejects_a_forgery(
        operator, coeff, advection, reaction):
    m, uh, exact, source = _scalar_family_case(coeff, advection, reaction)
    spec = {"operator": operator, "dim": 2, "domain_measure": 1.0,
            "source": source, "coefficient": coeff}
    if advection:
        spec["advection"] = advection
    if reaction is not None:
        spec["reaction"] = reaction
    spec = json.dumps(spec)
    assert _verdict(spec, _artefact(m, uh, 2)) == "SOLVES"
    assert _verdict(spec, _artefact(m, exact, 2)) == "DOES_NOT_SOLVE"


def test_helmholtz_must_declare_its_wave_number():
    """Silently treating Helmholtz as Poisson would assemble a definite
    operator for an indefinite problem and certify the wrong thing."""
    with pytest.raises(ResidualSpecError) as exc:
        parse_spec({"operator": "helmholtz", "dim": 2, "source": "1"})
    assert "reaction" in str(exc.value)


def test_a_malformed_advection_is_refused():
    with pytest.raises(ResidualSpecError) as exc:
        parse_spec({"operator": "convection_diffusion", "dim": 2,
                    "source": "1", "advection": ["1"]})
    assert "2-component" in str(exc.value)


# ── vector elasticity: the case that used to be UNSUPPORTED ───────────────
def _elasticity_case():
    """Manufactured displacement vanishing on the whole boundary, with the body
    force derived from it symbolically."""
    import sympy as sp
    from skfem import (Basis, ElementTriP1, ElementVector, LinearForm, asm,
                       condense, solve)
    from skfem.models.elasticity import lame_parameters, linear_elasticity

    E, nu = 1000.0, 0.3
    lam, mu = lame_parameters(E, nu)
    X, Y = sp.symbols("x y")
    ux = sp.sin(sp.pi * X) * sp.sin(sp.pi * Y)
    uy = sp.Rational(1, 2) * ux
    eps = [[sp.diff(ux, X), (sp.diff(ux, Y) + sp.diff(uy, X)) / 2],
           [(sp.diff(ux, Y) + sp.diff(uy, X)) / 2, sp.diff(uy, Y)]]
    tr = eps[0][0] + eps[1][1]
    sig = [[2 * mu * eps[0][0] + lam * tr, 2 * mu * eps[0][1]],
           [2 * mu * eps[1][0], 2 * mu * eps[1][1] + lam * tr]]
    f1 = sp.simplify(-(sp.diff(sig[0][0], X) + sp.diff(sig[0][1], Y)))
    f2 = sp.simplify(-(sp.diff(sig[1][0], X) + sp.diff(sig[1][1], Y)))
    fn = [sp.lambdify((X, Y), f, "numpy") for f in (f1, f2)]

    def src(A):
        return np.array([fn[0](A[0], A[1]) * np.ones_like(A[0]),
                         fn[1](A[0], A[1]) * np.ones_like(A[0])])

    m = skfem.MeshTri().refined(5)
    basis = Basis(m, ElementVector(ElementTriP1()))

    @LinearForm
    def L(v, w):
        f = src(w.x)
        return f[0] * v[0] + f[1] * v[1]

    uh = solve(*condense(asm(linear_elasticity(lam, mu), basis), asm(L, basis),
                         D=basis.get_dofs()))
    nd = basis.nodal_dofs
    p = m.p.T
    genuine = np.column_stack([uh[nd[0]], uh[nd[1]]])
    exact = np.column_stack([np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]),
                             0.5 * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])])
    spec = json.dumps({"operator": "elasticity", "dim": 2, "young": E,
                       "poisson": nu, "domain_measure": 1.0,
                       "source": [str(f1), str(f2)]})
    return m, p, genuine, exact, spec


def _vector_artefact(m, p, values):
    d = Path(tempfile.mkdtemp())
    path = d / "solution.vtu"
    meshio.write_points_cells(
        str(path), np.column_stack([p, np.zeros(len(p))]),
        [("triangle", m.t.T)],
        point_data={"u": np.column_stack([values, np.zeros(len(p))])})
    return [path]


def test_elasticity_genuine_solve_and_forgery():
    m, p, genuine, exact, spec = _elasticity_case()
    assert _verdict(spec, _vector_artefact(m, p, genuine)) == "SOLVES"
    assert _verdict(spec, _vector_artefact(m, p, exact)) == "DOES_NOT_SOLVE"


def test_elasticity_discriminates_on_the_equations_not_on_accuracy():
    """The genuine solve differs from the exact field by the discretisation
    error, so this cannot be passing merely because one equals the other."""
    m, p, genuine, exact, _ = _elasticity_case()
    assert np.abs(genuine - exact).max() > 1e-4


def test_elasticity_zero_field_is_refused():
    m, p, genuine, _, spec = _elasticity_case()
    zeros = np.zeros_like(genuine)
    assert _verdict(spec, _vector_artefact(m, p, zeros)) == "DOES_NOT_SOLVE"


@pytest.mark.parametrize("spec, why", [
    ({"operator": "elasticity", "dim": 2, "source": ["0", "0"]}, "young"),
    ({"operator": "elasticity", "dim": 2, "source": ["0", "0"], "young": 1.0},
     "poisson"),
    ({"operator": "elasticity", "dim": 2, "source": "0", "young": 1.0,
      "poisson": 0.3}, "2-component body force"),
    ({"operator": "elasticity", "dim": 3, "source": ["0", "0"], "young": 1.0,
      "poisson": 0.3}, "3-component body force"),
])
def test_elasticity_declarations_must_be_complete(spec, why):
    with pytest.raises(ResidualSpecError) as exc:
        parse_spec(spec)
    assert why in str(exc.value)


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
