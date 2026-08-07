"""Tier-2 for fenics matrix_free_poisson#2: zero the Dirichlet entries of the
output of EVERY operator application (`bc.set(y.array, alpha=0.0)` at the end of
action_A). Without it the constrained rows feed the unconstrained ones and CG has
no fixed point.

Wrong variant: action_A that assembles the action and returns without zeroing the
constrained entries. Right variant: the same action_A with
bc.set(y.array, alpha=0.0) as its last statement.

Observed on dolfinx 0.10.0, ONE rank (this is a serial failure, it needs no MPI):
with the zeroing removed, rnorm/rnorm0 grows monotonically at every logged
iteration -- of order 1e0 at iteration 20, 1e1 at 60, 1e3 at 100, 1e8 at 160,
1e13 at 200 -- CG never reaches rtol inside 200 iterations, and the returned
field reaches max(u) of order 1e27 while min(u) is still EXACTLY the prescribed
boundary minimum, so a min-only sanity print looks innocent. The identical run
with the zeroing converges in about 100 iterations and stays in the physical
range. Magnitudes depend on mesh and degree (16x16, P2 here); only the monotone
growth and the 1e10-plus overflow are asserted.

Mutation control: T2_MUTATE=1 puts bc.set(y.array, alpha=0.0) back into the
run under test; CG then converges and the divergence tokens go False.
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
MAX_ITER = 200
LOG_AT = (20, 60, 100, 140, 160, 200)


def run(zero_bc_in_action: bool):
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
        if zero_bc_in_action:
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
    hist, its, conv = {}, MAX_ITER, False
    for k in range(MAX_ITER):
        action_A(p, yv)
        alpha = rn / gdot(p.array, yv.array)
        uh.x.array[:] += alpha * p.array
        r -= alpha * yv.array
        rn, rn_old = gdot(r, r), rn
        if (k + 1) in LOG_AT:
            hist[k + 1] = float(np.sqrt(abs(rn / rn0)))
        if rn / rn0 < 1e-20:
            its, conv = k + 1, True
            break
        p.array[:] = (rn / rn_old) * p.array + r
    uh.x.scatter_forward()
    bc.set(uh.x.array, alpha=1.0)
    uh.x.scatter_forward()
    return dict(its=its, conv=conv, hist=hist,
                umin=float(uh.x.array[:nr].min()),
                umax=float(uh.x.array[:nr].max()),
                bc_min=float(uD.x.array[bdofs].min()))


def main() -> int:
    bad = run(zero_bc_in_action=MUTATE)
    good = run(zero_bc_in_action=True)
    tag = "with_zeroing" if MUTATE else "no_zeroing"
    for k in sorted(bad["hist"]):
        print(f"{tag}_rnorm_over_rnorm0_at_iter_{k}={bad['hist'][k]:.3e}")
    print(f"{tag}_converged={bad['conv']} iterations={bad['its']}")
    print(f"{tag}_u_range=[{bad['umin']:.3e}, {bad['umax']:.3e}] "
          f"prescribed_boundary_min={bad['bc_min']:.3e}")
    print(f"reference_run_with_zeroing_converged={good['conv']} "
          f"iterations={good['its']} umax={good['umax']:.3e}")

    seq = [bad["hist"][k] for k in sorted(bad["hist"])]
    grows = len(seq) >= 4 and all(b > a for a, b in zip(seq, seq[1:]))
    overflow = bad["umax"] > 1e10
    min_innocent = abs(bad["umin"] - bad["bc_min"]) <= 1e-14
    print(f"residual_ratio_grows_monotonically={grows}")
    print(f"diverging_run_reached_max_iter_without_rtol={not bad['conv']}")
    print(f"max_u_exceeds_1e10={overflow}")
    print(f"min_u_still_equals_the_prescribed_boundary_minimum={min_innocent}")
    print(f"same_run_with_zeroing_converges={good['conv'] and good['umax'] < 1.0}")
    if grows and not bad["conv"] and overflow and min_innocent \
            and good["conv"] and good["umax"] < 1.0:
        print("VERDICT=unzeroed_action_makes_cg_diverge_while_min_u_looks_innocent")
        return 0
    print("VERDICT=unzeroed_action_converged_anyway")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
