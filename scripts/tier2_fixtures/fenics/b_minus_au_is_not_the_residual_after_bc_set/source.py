"""Tier-2 for fenics matrix_free_poisson#3: after the CG loop,
`bc.set(u.x.array, alpha=1.0)` writes the prescribed Dirichlet values into the
solution. From that moment on, `b - A*u` is NO LONGER the residual of the system
that was solved, because b already carries the lifting term -A*u_bc.

Wrong variant: recompute ||b - A*u|| / ||b|| after the bc.set and report it as
"the residual". Right variant: assemble the Galerkin residual of the FULL
solution, action(a, u) - L, zeroed at the constrained DOFs.

Observed on dolfinx 0.10.0 with a converged matrix-free CG solve (16x16, P2,
rtol 1e-10): before the bc.set the same expression is the residual of the reduced
system and comes out at the Krylov tolerance, of order 1e-11; after the bc.set it
jumps to 1.0 -- 9.999e-01 with the constrained rows zeroed and 1.000e+00 without
-- for a solve whose reference-free Galerkin residual is of order 1e-9. The
misleading number is the size of A*u_bc, not an error in the solution.

Mutation control: T2_MUTATE=1 evaluates b - A*u BEFORE the bc.set (the residual
of the system that was actually solved) and reports the Galerkin residual of the
full solution afterwards, so the order-one token is never printed.
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
N = 16
DEGREE = 2
RTOL = 1e-10


def main() -> int:
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N)
    V = fem.functionspace(msh, ("Lagrange", DEGREE))
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    facets = mesh.exterior_facet_indices(msh.topology)
    bdofs = fem.locate_dofs_topological(V, tdim - 1, facets)
    uD = fem.Function(V, dtype=DTYPE)
    uD.interpolate(lambda x: 0.5 * x[0])
    bc = fem.dirichletbc(uD, bdofs)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = 10.0 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_fem = fem.form(ufl.inner(f, v) * ufl.dx, dtype=DTYPE)
    ui = fem.Function(V, dtype=DTYPE)
    M_fem = fem.form(ufl.action(a, ui), dtype=DTYPE)

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

    def ratio_b_minus_Au(uf) -> float:
        yv = la.vector(b.index_map, 1, DTYPE)
        action_A(uf.x, yv)
        res = b.array - yv.array
        return float(np.sqrt(abs(gdot(res, res)) / abs(gdot(b.array, b.array))))

    uh = fem.Function(V, dtype=DTYPE)
    yv = la.vector(b.index_map, 1, DTYPE)
    action_A(uh.x, yv)
    r = b.array - yv.array
    p = la.vector(b.index_map, 1, DTYPE)
    p.array[:] = r
    rn0 = rn = gdot(r, r)
    its, conv = 0, False
    for k in range(2000):
        action_A(p, yv)
        alpha = rn / gdot(p.array, yv.array)
        uh.x.array[:] += alpha * p.array
        r -= alpha * yv.array
        rn, rn_old = gdot(r, r), rn
        if rn / rn0 < RTOL ** 2:
            its, conv = k + 1, True
            break
        p.array[:] = (rn / rn_old) * p.array + r
    uh.x.scatter_forward()
    print(f"cg_converged={conv} iterations={its} rtol={RTOL:.0e}")

    before = ratio_b_minus_Au(uh)
    print(f"b_minus_Au_over_b_BEFORE_bc_set={before:.4e}")

    bc.set(uh.x.array, alpha=1.0)  # write the prescribed values in
    uh.x.scatter_forward()

    after = None
    if MUTATE:
        print("mutation=b_minus_Au_only_evaluated_before_bc_set")
    else:
        after = ratio_b_minus_Au(uh)
        print(f"b_minus_Au_over_b_AFTER_bc_set={after:.4e}")

    # the reference-free check that IS valid for the full solution
    ui.x.array[:] = uh.x.array
    ui.x.scatter_forward()
    gres = la.vector(b.index_map, 1, DTYPE)
    gres.array[:] = 0.0
    fem.assemble_vector(gres.array, M_fem)
    gres.scatter_reverse(la.InsertMode.add)
    rhs = fem.assemble_vector(L_fem)
    rhs.scatter_reverse(la.InsertMode.add)
    gres.array[:] -= rhs.array
    bc.set(gres.array, alpha=0.0)
    galerkin = float(np.sqrt(abs(gdot(gres.array, gres.array))
                             / abs(gdot(rhs.array, rhs.array))))
    print(f"galerkin_residual_of_full_solution={galerkin:.4e}")

    small_before = before < 1e-8
    small_galerkin = galerkin < 1e-6
    print(f"reduced_residual_before_bc_set_is_at_the_krylov_tolerance={small_before}")
    print(f"galerkin_residual_of_full_solution_is_small={small_galerkin}")
    if after is not None:
        order_one = 0.5 < after < 2.0
        print(f"b_minus_Au_after_bc_set_is_order_one={order_one}")
        print(f"b_minus_Au_after_bc_set_is_at_least_1e7_times_the_galerkin_residual="
              f"{after > 1e7 * galerkin}")
    else:
        order_one = False
    if conv and small_before and small_galerkin and order_one \
            and after > 1e7 * galerkin:
        print("VERDICT=b_minus_Au_is_order_one_after_bc_set_while_the_solve_is_good")
        return 0
    print("VERDICT=b_minus_Au_stayed_a_valid_residual")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
