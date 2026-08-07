"""Tier-2 for fenics biharmonic#1: the claim says the interior-penalty weight
must use ufl.CellDiameter / ufl.FacetArea so it tracks the local element size,
and that hard-coding h degrades the convergence rate from O(h^2) to about O(h).
On a UNIFORM refinement sequence that degradation does not happen, and this
fixture pins what really happens instead.

Wrong variant: h_E hard-coded as fem.Constant(msh, 1/8) in the penalty weight
alpha/h_E while the mesh is refined from 8x8 to 32x32. Right variant:
alpha/avg(ufl.CellDiameter(msh)).

C0-IP MMS on the unit square, u = sin(pi x) sin(pi y), simply supported, P2,
alpha = 8, L2 error at N = 8 / 16 / 32. The fixture also measures the penalty
weight itself, as the facet-average of alpha/h_E, so "does the weight track the
mesh" is checked directly rather than inferred.

Observed on dolfinx 0.10.0: the frozen weight is 64.0 on every mesh, i.e. it
does not move at all, while the CellDiameter weight grows 45.3 -> 90.5 -> 181.0
as it should. And yet the frozen-weight L2 errors are 4.0243e-02 -> 7.3802e-03
-> 1.5895e-03, rates 2.45 and 2.22, against the CellDiameter form's 3.2735e-02
-> 9.1180e-03 -> 2.3612e-03, rates 1.84 and 1.95. The frozen weight is at least
as accurate here and nowhere near O(h), so treat this as a graded-mesh concern
and diagnose it by comparing local penalty magnitudes, not by watching a global
rate.

Mutation control: T2_MUTATE=1 makes the primary weight the CellDiameter one, so
the primary weight does track the mesh and the fixture loses its own
expectation.
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

from dolfinx.fem.petsc import LinearProblem  # noqa: E402

DEGREE = 2
ALPHA = 8.0
FROZEN_H = 1.0 / 8.0
MESHES = (8, 16, 32)


def run(n: int, frozen: bool):
    """Return (L2 error, facet-average of the penalty weight alpha/h_E)."""
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", DEGREE))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    f = 4.0 * np.pi ** 4 * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    nrm = ufl.FacetNormal(msh)
    al = dolfinx.fem.Constant(msh, ALPHA)
    h_pen = (dolfinx.fem.Constant(msh, FROZEN_H) if frozen
             else ufl.avg(ufl.CellDiameter(msh)))
    a = (ufl.inner(ufl.div(ufl.grad(u)), ufl.div(ufl.grad(v))) * ufl.dx
         - ufl.inner(ufl.avg(ufl.div(ufl.grad(u))),
                     ufl.jump(ufl.grad(v), nrm)) * ufl.dS
         - ufl.inner(ufl.jump(ufl.grad(u), nrm),
                     ufl.avg(ufl.div(ufl.grad(v)))) * ufl.dS
         + al / h_pen * ufl.inner(ufl.jump(ufl.grad(u), nrm),
                                  ufl.jump(ufl.grad(v), nrm)) * ufl.dS)
    L = ufl.inner(f, v) * ufl.dx
    bfacets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    bc = dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(msh, 0.0),
        dolfinx.fem.locate_dofs_topological(V, fdim, bfacets), V)
    problem = LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix="t2_bih1_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    uh = problem.solve()
    err = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form((uh - u_exact) ** 2 * ufl.dx))
    weight = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(al / h_pen * ufl.dS))
    area = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(dolfinx.fem.Constant(msh, 1.0) * ufl.dS))
    return float(np.sqrt(abs(err))), float(weight / area)


def rates(errs):
    return [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]


def main() -> int:
    primary_frozen = not MUTATE
    primary = [run(n, primary_frozen) for n in MESHES]
    cell = [run(n, False) for n in MESHES]
    p_err = [e for e, _ in primary]
    p_w = [w for _, w in primary]
    c_err = [e for e, _ in cell]
    c_w = [w for _, w in cell]
    for i, n in enumerate(MESHES):
        print(f"N={n} primary_l2={p_err[i]:.4e} primary_penalty_weight={p_w[i]:.1f} "
              f"celldiam_l2={c_err[i]:.4e} celldiam_penalty_weight={c_w[i]:.1f}")
    r_p, r_c = rates(p_err), rates(c_err)
    print("primary_rates=" + " ".join(f"{r:.2f}" for r in r_p))
    print("celldiam_rates=" + " ".join(f"{r:.2f}" for r in r_c))

    w_ratio = p_w[-1] / p_w[0]
    print(f"primary_penalty_weight_ratio_finest_over_coarsest={w_ratio:.3f}")
    frozen_weight = w_ratio < 1.01
    rate_ok = min(r_p) > 2.0
    not_worse = min(r_p) >= min(r_c) - 0.05
    not_order_one = min(r_p) > 1.5
    print(f"primary_penalty_weight_does_not_track_the_mesh={frozen_weight}")
    print(f"primary_rate_stays_above_two={rate_ok}")
    print(f"primary_rate_at_least_as_good_as_celldiameter={not_worse}")
    print(f"claimed_degradation_to_order_one_not_observed={not_order_one}")
    if frozen_weight and rate_ok and not_worse and not_order_one:
        print("VERDICT=frozen_penalty_h_does_not_degrade_uniform_convergence")
        return 0
    print("VERDICT=frozen_penalty_h_degraded_the_rate")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
