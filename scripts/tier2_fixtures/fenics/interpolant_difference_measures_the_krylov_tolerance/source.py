"""Tier-2 for fenics matrix_free_poisson#10: do not use the difference against an
interpolated manufactured field as a proxy for discretisation error in this
template. For a polynomial reference field of degree <= the element degree the
Galerkin solution equals the interpolant to machine precision, so what such a
check actually measures is the CG stopping tolerance.

Wrong variant: report max|u_h - I(u_exact)| as "the error" for the matrix-free CG
solve. Right variant: a residual, a flux balance or an energy identity, none of
which needs a reference solution.

Reference fields (both harmonic, so f = 0 and both lie exactly in the space):
degree 1 uses 0.5*x + 0.3*y + 0.2, degree 2 uses x**2 - y**2.

Observed on dolfinx 0.10.0, structured unit square, 8/16/32 cells per side: the
direct-LU solution differs from the interpolant by 6.7e-16 to 2.5e-14 at both
degrees, while the matrix-free CG run at rtol 1e-6 reports 6.4e-08, 4.1e-07,
8.5e-07 at degree 1 and 4.9e-07, 7.6e-07, 1.5e-06 at degree 2 -- the reported
"error" sits at the Krylov tolerance and RISES as the mesh is refined. The
reference-free relative Galerkin residual of the same solutions is at the
tolerance too, which is the honest thing to report.

Mutation control: T2_MUTATE=1 tightens the CG tolerance to 1e-13. The reported
"error" then collapses to order 1e-14, which is what proves it was never a
discretisation error, and the krylov-tolerance token goes False.
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

from dolfinx import fem, la, mesh  # noqa: E402

DTYPE = dolfinx.default_scalar_type
SIZES = (8, 16, 32)
RTOL = 1e-13 if MUTATE else 1e-6


def build(n: int, degree: int):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n)
    V = fem.functionspace(msh, ("Lagrange", degree))
    if degree == 1:
        def uex(x):
            return 0.5 * x[0] + 0.3 * x[1] + 0.2
    else:
        def uex(x):
            return x[0] ** 2 - x[1] ** 2
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    facets = mesh.exterior_facet_indices(msh.topology)
    bdofs = fem.locate_dofs_topological(V, tdim - 1, facets)
    uD = fem.Function(V, dtype=DTYPE)
    uD.interpolate(uex)
    bc = fem.dirichletbc(uD, bdofs)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(fem.Constant(msh, DTYPE(0.0)), v) * ufl.dx
    Iu = fem.Function(V, dtype=DTYPE)
    Iu.interpolate(uex)
    return comm, msh, V, a, L, bc, Iu


def lu_diff(n: int, degree: int) -> float:
    _, _, _, a, L, bc, Iu = build(n, degree)
    lp = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix="t2mfp10_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    out = lp.solve()
    uh = out[0] if isinstance(out, tuple) else out
    return float(np.abs(uh.x.array - Iu.x.array).max())


def matrix_free(n: int, degree: int, rtol: float):
    comm, msh, V, a, L, bc, Iu = build(n, degree)
    ui = fem.Function(V, dtype=DTYPE)
    M_fem = fem.form(ufl.action(a, ui), dtype=DTYPE)
    L_fem = fem.form(L, dtype=DTYPE)

    def action_A(xv, yv):
        ui.x.array[:] = xv.array
        ui.x.scatter_forward()
        yv.array[:] = 0.0
        fem.assemble_vector(yv.array, M_fem)
        yv.scatter_reverse(la.InsertMode.add)
        bc.set(yv.array, alpha=0.0)

    b = fem.assemble_vector(L_fem)
    ui.x.array[:] = 0.0
    bc.set(ui.x.array, alpha=-1.0)
    fem.assemble_vector(b.array, M_fem)
    b.scatter_reverse(la.InsertMode.add)
    bc.set(b.array, alpha=0.0)
    b.scatter_forward()
    nr = b.index_map.size_local

    def gdot(v0, v1):
        return comm.allreduce(np.vdot(v0[:nr], v1[:nr]), MPI.SUM)

    uh = fem.Function(V, dtype=DTYPE)
    yv = la.vector(b.index_map, 1, DTYPE)
    action_A(uh.x, yv)
    r = b.array - yv.array
    p = la.vector(b.index_map, 1, DTYPE)
    p.array[:] = r
    rn0 = rn = gdot(r, r)
    its, conv = 0, True
    for k in range(4000):
        action_A(p, yv)
        alpha = rn / gdot(p.array, yv.array)
        uh.x.array[:] += alpha * p.array
        r -= alpha * yv.array
        rn, rn_old = gdot(r, r), rn
        if rn / rn0 < rtol ** 2:
            its, conv = k + 1, True
            break
        p.array[:] = (rn / rn_old) * p.array + r
    else:
        conv = False
    uh.x.scatter_forward()
    bc.set(uh.x.array, alpha=1.0)
    uh.x.scatter_forward()
    diff = float(np.abs(uh.x.array - Iu.x.array).max())
    # reference-free check on the same solution
    ui.x.array[:] = uh.x.array
    ui.x.scatter_forward()
    g = la.vector(b.index_map, 1, DTYPE)
    g.array[:] = 0.0
    fem.assemble_vector(g.array, M_fem)
    g.scatter_reverse(la.InsertMode.add)
    bc.set(g.array, alpha=0.0)
    # f = 0 here, so the Galerkin residual of the full solution over the free
    # DOFs is just A*u_full; its natural scale is the lifting right-hand side b.
    nrm_A = float(np.sqrt(abs(gdot(g.array, g.array))))
    scale = max(float(np.sqrt(abs(gdot(b.array, b.array)))), 1e-30)
    return diff, its, conv, nrm_A / scale


def main() -> int:
    print(f"cg_rtol={RTOL:.0e}")
    lu, mf, res = {}, {}, {}
    for degree in (1, 2):
        lu[degree] = [lu_diff(n, degree) for n in SIZES]
        rows = [matrix_free(n, degree, RTOL) for n in SIZES]
        mf[degree] = [r[0] for r in rows]
        res[degree] = [r[3] for r in rows]
        conv = all(r[2] for r in rows)
        print(f"degree={degree} n={list(SIZES)} "
              f"lu_minus_interpolant={['%.3e' % e for e in lu[degree]]}")
        print(f"degree={degree} n={list(SIZES)} "
              f"matrix_free_minus_interpolant={['%.3e' % e for e in mf[degree]]} "
              f"all_converged={conv}")
        print(f"degree={degree} n={list(SIZES)} "
              f"reference_free_relative_galerkin_residual="
              f"{['%.3e' % e for e in res[degree]]}")

    lu_exact = all(e < 1e-12 for d in lu.values() for e in d)
    at_tol = all(1e-8 <= e <= 1e-4 for d in mf.values() for e in d)
    not_falling = all(mf[d][-1] >= mf[d][0] for d in mf)
    print(f"lu_minus_interpolant_is_machine_precision={lu_exact}")
    print(f"matrix_free_difference_is_at_the_krylov_tolerance_1em6={at_tol}")
    print(f"matrix_free_difference_did_not_fall_under_refinement={not_falling}")
    if lu_exact and at_tol and not_falling:
        print("VERDICT=interpolant_difference_measures_the_krylov_tolerance")
        return 0
    print("VERDICT=interpolant_difference_behaved_like_a_discretisation_error")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
