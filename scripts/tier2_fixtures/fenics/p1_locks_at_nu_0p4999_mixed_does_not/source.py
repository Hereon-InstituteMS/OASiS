"""Tier-2 for fenics nearly_incompressible_elasticity#0: a low-order
displacement-only formulation LOCKS as nu -> 0.5, and the mixed (u, p) method is
the robust fix. The claim also carries an explicit CORRECTION: the locking ratio
is NOT ~1/(1-2nu).

Wrong variant: P1 triangles, displacement only, on a 1.0 x 0.2 cantilever with
an end traction at nu = 0.4999. Reference: P2/P1 Herrmann mixed formulation on
the same mesh. Observed on 10x2 and 20x4 the mixed tip deflection is about 20x
and 19x larger than the P1 one, while the same P2 displacement-only space is
within a few percent of the mixed reference; 1/(1-2nu) would predict 5000x.

Mutation control: T2_MUTATE=1 solves the tested case with the mixed (u, p)
formulation, and the tip deflection then matches the reference exactly.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

E_MOD = 1.0e5
TRACTION = -1.0e2
NU = 0.4999


def tip_deflection(nx: int, ny: int, kind: str, nu: float) -> float:
    msh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([1.0, 0.2])],
        [nx, ny], dolfinx.mesh.CellType.triangle)
    tdim, gdim = msh.topology.dim, msh.geometry.dim
    mu = E_MOD / (2.0 * (1.0 + nu))
    lam = E_MOD * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    msh.topology.create_connectivity(tdim - 1, tdim)

    def left(x):
        return np.isclose(x[0], 0.0)

    facets = dolfinx.mesh.locate_entities_boundary(
        msh, tdim - 1, lambda x: np.isclose(x[0], 1.0))
    mt = dolfinx.mesh.meshtags(msh, tdim - 1, np.sort(facets),
                               np.full(len(facets), 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=mt)
    trac = dolfinx.fem.Constant(msh, np.array([0.0, TRACTION]))

    def eps(w):
        return ufl.sym(ufl.grad(w))

    if kind in ("P1", "P2"):
        V = dolfinx.fem.functionspace(
            msh, ("Lagrange", 1 if kind == "P1" else 2, (gdim,)))
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        a = (2 * mu * ufl.inner(eps(u), eps(v))
             + lam * ufl.div(u) * ufl.div(v)) * ufl.dx
        L = ufl.dot(trac, v) * ds(1)
        bc = dolfinx.fem.dirichletbc(
            np.zeros(gdim), dolfinx.fem.locate_dofs_geometrical(V, left), V)
        prob = dolfinx.fem.petsc.LinearProblem(
            a, L, bcs=[bc], petsc_options_prefix=f"t2_nie0_{kind}{nx}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = prob.solve()
    elif kind == "mixed":
        P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(gdim,))
        P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
        W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P2, P1]))
        (uu, pp) = ufl.TrialFunctions(W)
        (vv, qq) = ufl.TestFunctions(W)
        a = (2 * mu * ufl.inner(eps(uu), eps(vv)) + pp * ufl.div(vv)
             + qq * ufl.div(uu) - (1.0 / lam) * pp * qq) * ufl.dx
        L = ufl.dot(trac, vv) * ds(1)
        V0, _ = W.sub(0).collapse()
        clamped = dolfinx.fem.Function(V0)
        bc = dolfinx.fem.dirichletbc(
            clamped,
            dolfinx.fem.locate_dofs_geometrical((W.sub(0), V0), left),
            W.sub(0))
        prob = dolfinx.fem.petsc.LinearProblem(
            a, L, bcs=[bc], petsc_options_prefix=f"t2_nie0_mx{nx}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                           "pc_factor_mat_solver_type": "mumps"})
        uh = prob.solve().sub(0).collapse()
    else:
        raise AssertionError(kind)
    assert prob.solver.getConvergedReason() > 0, f"{kind} solve failed"

    pt = np.array([1.0, 0.5 * 0.2, 0.0])
    tree = dolfinx.geometry.bb_tree(msh, tdim)
    cand = dolfinx.geometry.compute_collisions_points(tree, pt.reshape(1, 3))
    cells = dolfinx.geometry.compute_colliding_cells(
        msh, cand, pt.reshape(1, 3))
    return float(uh.eval(pt, [cells.links(0)[0]])[1])


def main() -> int:
    tested = "mixed" if MUTATE else "P1"
    print(f"displacement_space_under_test={tested}")
    print(f"poisson_ratio={NU}")
    ratios = []
    for nx, ny in ((10, 2), (20, 4)):
        ref = tip_deflection(nx, ny, "mixed", NU)
        got = tip_deflection(nx, ny, tested, NU)
        p2 = tip_deflection(nx, ny, "P2", NU)
        ratios.append((ref / got, ref / p2))
        print(f"mesh={nx}x{ny} mixed_tip={ref:.6e} "
              f"{tested}_tip={got:.6e} P2_tip={p2:.6e} "
              f"mixed_over_tested={ref / got:.3f} mixed_over_P2={ref / p2:.3f}")
    locks = all(r[0] > 5.0 for r in ratios)
    p2_ok = all(0.9 < r[1] < 1.1 for r in ratios)
    rule = 1.0 / (1.0 - 2.0 * NU)
    worst = max(r[0] for r in ratios)
    print(f"one_over_one_minus_two_nu={rule:.1f}")
    print(f"largest_measured_stiffness_ratio={worst:.3f}")
    print(f"locking_ratio_exceeds_5={locks}")
    print(f"p2_is_within_10_percent_of_mixed={p2_ok}")
    print(f"one_over_one_minus_two_nu_overpredicts_by_more_than_10x="
          f"{rule > 10.0 * worst}")
    if locks and p2_ok and rule > 10.0 * worst:
        print("VERDICT=p1_locks_while_p2_and_mixed_agree")
        return 0
    print("VERDICT=no_locking_measured")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
