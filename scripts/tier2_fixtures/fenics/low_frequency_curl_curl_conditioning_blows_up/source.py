"""Tier-2 for fenics maxwell#3: the curl-curl + omega^2-mass formulation becomes
ill-conditioned as omega -> 0, because the gradient kernel of curl is no longer
regularised by the mass term. The claim's fix is a mixed (A, phi) formulation
with a Lagrange multiplier on the divergence.

Wrong variant: (curl u, curl v) - omega^2 (u, v) on a 16x16 unit square, N1curl
degree 1, zero tangential trace, divergence-free right-hand side, solved with
GMRES(100)+ILU at omega = 1e0 ... 1e-4. The condition number of the free block
is measured densely at every omega.

Observed: cond = 8.37e3, 8.38e5, 8.38e7, 8.38e9, 8.38e11 - exactly the 1/omega^2
law the claim predicts, a factor 100 per decade of omega. GMRES converges
(reason 2) at omega = 1 and 0.1 and then does not: at omega <= 1e-2 it stops with
reason -5, KSP_DIVERGED_BREAKDOWN, after 200 iterations. The claim's wording
("iteration count explodes") understates it; the Krylov solve breaks down.

Note the claim names GMRES + AMS; hypre AMS needs a discrete-gradient operator
that cannot be requested through petsc_options alone, so ILU is used here and
reported as such.

Mutation control: T2_MUTATE=1 solves the mixed (A, phi) system on N1curl x P1
with the multiplier terms (grad phi, v) + (u, grad psi). Its condition number is
flat in omega (1.57e4) and GMRES converges in about 150 iterations at every
omega.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

N = 16
OMEGAS = (1.0e0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4)
KSP_OPTS = {"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": 1e-8,
            "ksp_max_it": 3000, "ksp_gmres_restart": 100}


def source(msh):
    x = ufl.SpatialCoordinate(msh)
    return ufl.as_vector((ufl.sin(ufl.pi * x[1]), ufl.sin(ufl.pi * x[0])))


def plain(omega: float, tag: str):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    V = dolfinx.fem.functionspace(
        msh, basix.ufl.element("N1curl", msh.basix_cell(), 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = (ufl.inner(ufl.curl(u), ufl.curl(v))
         - omega ** 2 * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(source(msh), v) * ufl.dx
    bdofs = dolfinx.fem.locate_dofs_topological(
        V, tdim - 1, dolfinx.mesh.exterior_facet_indices(msh.topology))
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Function(V), bdofs)
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix=tag, petsc_options=KSP_OPTS)
    prob.solve()
    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a), bcs=[bc])
    A.assemble()
    free = np.setdiff1d(np.arange(V.dofmap.index_map.size_global),
                        np.asarray(bdofs))
    cond = np.linalg.cond(
        A.convert("dense").getDenseArray()[np.ix_(free, free)])
    return (prob.solver.getConvergedReason(),
            prob.solver.getIterationNumber(), float(cond))


def mixed(omega: float, tag: str):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    E = basix.ufl.element("N1curl", msh.basix_cell(), 1)
    P = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([E, P]))
    (u, phi) = ufl.TrialFunctions(W)
    (v, psi) = ufl.TestFunctions(W)
    a = (ufl.inner(ufl.curl(u), ufl.curl(v)) - omega ** 2 * ufl.inner(u, v)
         + ufl.inner(ufl.grad(phi), v) + ufl.inner(u, ufl.grad(psi))) * ufl.dx
    L = ufl.inner(source(msh), v) * ufl.dx
    facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    V0, _ = W.sub(0).collapse()
    Q0, _ = W.sub(1).collapse()
    d0 = dolfinx.fem.locate_dofs_topological((W.sub(0), V0), tdim - 1, facets)
    d1 = dolfinx.fem.locate_dofs_topological((W.sub(1), Q0), tdim - 1, facets)
    bcs = [dolfinx.fem.dirichletbc(dolfinx.fem.Function(V0), d0, W.sub(0)),
           dolfinx.fem.dirichletbc(dolfinx.fem.Function(Q0), d1, W.sub(1))]
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options_prefix=tag, petsc_options=KSP_OPTS)
    prob.solve()
    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a), bcs=bcs)
    A.assemble()
    dense = A.convert("dense").getDenseArray()
    bd = np.union1d(np.asarray(d0[0]), np.asarray(d1[0]))
    free = np.setdiff1d(np.arange(dense.shape[0]), bd)
    cond = np.linalg.cond(dense[np.ix_(free, free)])
    return (prob.solver.getConvergedReason(),
            prob.solver.getIterationNumber(), float(cond))


def main() -> int:
    run = mixed if MUTATE else plain
    print(f"formulation={'mixed_A_phi' if MUTATE else 'plain_curl_curl'}")
    conds, reasons = [], []
    for omega in OMEGAS:
        reason, iters, cond = run(omega, f"t2_mw3_{omega:.0e}_")
        conds.append(cond)
        reasons.append(reason)
        print(f"omega={omega:.0e} ksp_reason={reason} iterations={iters} "
              f"condition_number={cond:.4e}")
    ratios = [conds[i + 1] / conds[i] for i in range(len(conds) - 1)]
    print(f"condition_number_ratio_per_decade="
          + " ".join(f"{r:.1f}" for r in ratios))
    quadratic = all(50.0 < r < 200.0 for r in ratios)
    failed = [r for r in reasons if r < 0]
    print(f"condition_number_scales_like_one_over_omega_squared={quadratic}")
    print(f"condition_number_span={conds[-1] / conds[0]:.3e}")
    print(f"ksp_failed_at_low_omega={bool(failed)}")
    print(f"ksp_failure_reason_is_breakdown={failed[:1] == [-5]}")
    if quadratic and failed:
        print("VERDICT=low_frequency_curl_curl_loses_conditioning_and_the_ksp")
        return 0
    print("VERDICT=conditioning_is_omega_independent")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
