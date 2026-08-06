"""Tier-2 for fenics linear_elasticity#2 (and the same physics behind #5): the
2D Lame lambda decides plane strain vs plane stress, silently, and the two
differ by the factor (1 - nu^2) in tip deflection.

Wrong variant: the 3D lambda = E*nu/((1+nu)(1-2nu)) inside a 2D form. That is
PLANE STRAIN. Nothing warns. Plane stress needs
lambda* = 2*lambda*mu/(lambda + 2*mu).

The fixture solves the same end-loaded cantilever twice, once with each lambda,
and checks the ratio of tip deflections against (1 - nu^2) at two Poisson
ratios. The claim's own correction is part of the check: the ratio must NOT be
(1 - nu), and the stiffening must NOT be ~30% at nu = 0.3.

Tolerances are applied inside the fixture; the expectations hold no numbers.

Mutation control: T2_MUTATE=1 uses the plane-stress lambda for BOTH solves, so
the ratio becomes 1 and every verdict flips.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"
E = 1.0e5
LX, LY = 1.0, 0.2


def tip_deflection(nu: float, plane_stress: bool) -> float:
    msh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([LX, LY])], [40, 8])
    gdim = msh.geometry.dim
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 2, (gdim,)))
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    if plane_stress:
        lam = 2.0 * lam * mu / (lam + 2.0 * mu)
    lam_c = dolfinx.fem.Constant(msh, lam)
    mu_c = dolfinx.fem.Constant(msh, mu)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0 * mu_c * eps(w) + lam_c * ufl.tr(eps(w)) * ufl.Identity(gdim)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx

    msh.topology.create_connectivity(gdim - 1, gdim)
    left = dolfinx.mesh.locate_entities_boundary(
        msh, gdim - 1, lambda x: np.isclose(x[0], 0.0))
    right = dolfinx.mesh.locate_entities_boundary(
        msh, gdim - 1, lambda x: np.isclose(x[0], LX))
    facet_tags = dolfinx.mesh.meshtags(
        msh, gdim - 1, np.sort(right),
        np.full(len(right), 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    traction = dolfinx.fem.Constant(msh, np.array([0.0, -10.0]))
    L = ufl.dot(traction, v) * ds(1)

    clamp = dolfinx.fem.Function(V)
    bc = dolfinx.fem.dirichletbc(
        clamp, dolfinx.fem.locate_dofs_topological(V, gdim - 1, left))
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix="t2_ps_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = prob.solve()
    if isinstance(uh, tuple):
        uh = uh[0]
    return float(np.min(uh.x.array[1::gdim]))


def main() -> int:
    ok = True
    for nu in (0.3, 0.45):
        strain = tip_deflection(nu, plane_stress=MUTATE)
        stress = tip_deflection(nu, plane_stress=True)
        ratio = strain / stress
        pred_sq = 1.0 - nu * nu
        pred_lin = 1.0 - nu
        print(f"nu={nu} strain_over_stress={ratio:.5f} "
              f"one_minus_nu2={pred_sq:.5f} one_minus_nu={pred_lin:.5f}")
        near_sq = abs(ratio - pred_sq) < 0.02
        near_lin = abs(ratio - pred_lin) < 0.02
        print(f"nu={nu}_matches_one_minus_nu_squared={near_sq}")
        print(f"nu={nu}_matches_one_minus_nu={near_lin}")
        ok = ok and near_sq and not near_lin
    if ok:
        print("VERDICT=plane_strain_is_one_minus_nu_squared_of_plane_stress")
        return 0
    print("VERDICT=ratio_does_not_follow_one_minus_nu_squared")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
