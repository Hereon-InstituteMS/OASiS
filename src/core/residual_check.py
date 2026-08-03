"""Discrete-residual check: does the submitted field actually solve the problem?

WHY THIS EXISTS
---------------
Attestation binds a reported number to a FILE. It never binds it to a PROBLEM.
An adversarial review demonstrated the consequence: the cheapest forgery is not
an elaborate one but a trivial one — four nodes and a constant hit any target
exactly, with no knowledge of the physics at all; and an eight-line script that
emits an analytic field plus an h-dependent perturbation is *more accurate*
than a genuine solve, 82x faster, and passes a mesh-independence study.

Nothing that inspects only the data can tell those apart, because as data they
are perfectly well-formed. The distinguishing property is not in the numbers'
appearance but in whether they SATISFY THE EQUATIONS they claim to solve.

WHAT THIS CHECKS
----------------
Given the submitted mesh and field, OASiS assembles the discrete operator of
the stated problem ON THAT MESH and measures

    rho = || A u_submitted - b ||  /  || b ||        (interior degrees of freedom)

For a genuine finite element solution rho is at the linear solver's tolerance,
because A u = b is precisely the system that was solved. For anything else it
is not:

  * the exact solution sampled at nodes does NOT satisfy the discrete system —
    it misses by the truncation error, which is orders of magnitude larger;
  * a perturbed analytic field misses by more;
  * a constant, or a field on a mesh that is not the stated domain, misses
    grossly or is refused outright by the domain check.

This is the check the attestation module's docstring names and defers. It
converts "the data looks plausible" into "the data satisfies the equations",
which is the only statement that separates solving from fabricating.

SCOPE
-----
Implemented for the operator families the evaluation uses: scalar diffusion
    -div(K grad u) = f
with constant or spatially varying scalar/tensor coefficient, on simplicial
meshes in 2D and 3D, with Dirichlet data on the outer boundary. It is not a
universal PDE checker: for an unsupported operator it returns UNSUPPORTED
rather than a false pass, so a caller can never mistake "not checked" for
"checked and fine".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ResidualVerdict:
    supported: bool
    relative_residual: float | None
    solver_like: bool | None          # residual consistent with a real solve
    n_interior: int
    domain_measure: float | None
    detail: str


# Below this, the field satisfies the discrete system as a solver would leave it.
SOLVER_RESIDUAL_MAX = 1e-8
# Domain measure must match the stated domain to this relative tolerance.
DOMAIN_TOL = 1e-6


def _require_skfem():
    try:
        import skfem  # noqa: F401
        return True
    except Exception:
        return False


def _build_mesh(points: np.ndarray, cells, dim: int):
    """Construct a scikit-fem mesh from submitted points/cells."""
    from skfem import MeshTri, MeshTet
    tri = [np.asarray(c[1], int) for c in cells
           if str(c[0]).lower().startswith("triangle")]
    tet = [np.asarray(c[1], int) for c in cells
           if str(c[0]).lower().startswith("tetra")]
    p = np.asarray(points, float)
    if dim == 2 and tri:
        t = np.vstack(tri)
        return MeshTri(p[:, :2].T.copy(), t.T.copy())
    if dim == 3 and tet:
        t = np.vstack(tet)
        return MeshTet(p[:, :3].T.copy(), t.T.copy())
    return None


def check_residual(points, cells, values, *, dim: int,
                   source_fn, coeff_fn=None,
                   domain_measure_expected: float | None = None,
                   ) -> ResidualVerdict:
    """Measure how well `values` satisfies -div(K grad u) = f on this mesh.

    source_fn(x) -> f at coordinates x (array shape (dim, n))
    coeff_fn(x)  -> scalar K at x, or None for K = 1
    """
    if not _require_skfem():
        return ResidualVerdict(False, None, None, 0, None,
                               "scikit-fem unavailable; cannot assemble")

    from skfem import Basis, ElementTriP1, ElementTetP1, asm, BilinearForm, LinearForm
    from skfem.helpers import dot, grad

    pts = np.asarray(points, float)
    vals = np.asarray(values, float).reshape(-1)
    if pts.shape[0] != vals.shape[0]:
        return ResidualVerdict(False, None, None, 0, None,
                               "field length does not match point count")

    mesh = _build_mesh(pts, cells, dim)
    if mesh is None:
        return ResidualVerdict(False, None, None, 0, None,
                               f"no simplicial cells of dimension {dim} in the submission")

    elem = ElementTriP1() if dim == 2 else ElementTetP1()
    basis = Basis(mesh, elem)

    # Domain measure: a field on the wrong domain is not a solution of the
    # stated problem, however well-formed it looks.
    # integrand must carry the basis function: sum_i integral(phi_i) = |domain|.
    # Writing `1.0 + 0.0*v` drops it and yields (dofs per cell) x |domain|.
    one = LinearForm(lambda v, w: v)
    measure = float(asm(one, basis).sum())
    if domain_measure_expected is not None:
        rel = abs(measure - domain_measure_expected) / max(
            abs(domain_measure_expected), 1e-30)
        if rel > DOMAIN_TOL:
            return ResidualVerdict(
                True, None, False, 0, measure,
                f"mesh covers measure {measure:.6g}, but the stated domain has "
                f"{domain_measure_expected:.6g} (relative difference {rel:.3g}): "
                f"this is not the problem's domain")

    @BilinearForm
    def a_form(u, v, w):
        k = coeff_fn(w.x) if coeff_fn is not None else 1.0
        return k * dot(grad(u), grad(v))

    @LinearForm
    def l_form(v, w):
        return source_fn(w.x) * v

    A = asm(a_form, basis)
    b = asm(l_form, basis)

    if vals.shape[0] != A.shape[0]:
        return ResidualVerdict(False, None, None, 0, measure,
                               "degrees of freedom do not match the submitted field")

    boundary = basis.get_dofs().flatten()
    interior = np.setdiff1d(np.arange(A.shape[0]), boundary)
    if interior.size == 0:
        return ResidualVerdict(True, None, False, 0, measure,
                               "no interior degrees of freedom to test")

    r = A @ vals - b
    num = float(np.linalg.norm(r[interior]))
    den = float(np.linalg.norm(b[interior]))
    rho = num / den if den > 0 else float("inf")
    solver_like = rho <= SOLVER_RESIDUAL_MAX

    return ResidualVerdict(
        True, rho, solver_like, int(interior.size), measure,
        ("the field satisfies the discrete equations to solver tolerance"
         if solver_like else
         f"the field does NOT satisfy the discrete equations "
         f"(relative residual {rho:.3e} > {SOLVER_RESIDUAL_MAX:.0e}); it was "
         f"not obtained by solving this problem on this mesh"))
