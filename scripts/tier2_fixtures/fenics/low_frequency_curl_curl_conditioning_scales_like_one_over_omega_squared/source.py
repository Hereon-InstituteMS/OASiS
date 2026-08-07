"""Tier-2 for fenics maxwell#3: the curl-curl + omega^2-mass formulation loses
its regularisation as omega -> 0, because the only thing keeping the gradient
kernel of curl out of the operator's null space is the omega^2 mass term.

Wrong variant: (curl A, curl v) + omega^2 (A, v) = (f, v) on N1curl degree 1,
8x8 unit square, natural boundary condition, f = (1, 0) so the right-hand side
has a gradient component. Measured on dolfinx 0.10.0 for
omega = 1, 1e-1, 1e-2, 1e-3: the spectral condition number (dense SVD of the
assembled matrix, no solver involved) is 2.59e+03, 2.59e+05, 2.59e+07,
2.59e+09 -- exactly a factor 100 per decade of omega, i.e. 1/omega^2 -- and
GMRES/Jacobi needs 3, 6, 33, 150 iterations for the same tolerance.

The fix the claim names -- the mixed (A, phi) formulation with a Lagrange
multiplier enforcing (A, grad q) = 0 -- is measured alongside: its condition
number is 7.98e+04, 1.06e+04, 1.02e+04, 1.02e+04 over the same omegas, i.e.
flat.

FINDING against the claim text: the claim quotes iteration counts for
"GMRES + AMS preconditioner", but hypre's AMS cannot be used off the shelf here.
Requesting pc_type hypre / pc_hypre_type ams through LinearProblem fails with
PETSc "Error error code 83" because the discrete gradient operator is never
handed to it, so the iteration counts above are GMRES/Jacobi. The claim also
says the condition number is "printed by PETSc"; nothing prints it, it has to be
computed.

Mutation control: T2_MUTATE=1 puts the mixed (A, phi) formulation under test.
Its condition number no longer tracks 1/omega^2 and the signal disappears.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

N = 8
OMEGAS = (1.0, 1e-1, 1e-2, 1e-3)
KSP = {"ksp_type": "gmres", "pc_type": "jacobi", "ksp_rtol": 1e-8,
       "ksp_max_it": 5000, "ksp_gmres_restart": 100}


def mesh_and_spaces():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    ne = basix.ufl.element("N1curl", msh.basix_cell(), 1)
    p1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    return msh, ne, p1


def unmixed(omega: float):
    msh, ne, _ = mesh_and_spaces()
    V = dolfinx.fem.functionspace(msh, ne)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = (ufl.inner(ufl.curl(u), ufl.curl(v))
         + omega ** 2 * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(ufl.as_vector((1.0, 0.0)), v) * ufl.dx
    A = dolfinx.fem.assemble_matrix(dolfinx.fem.form(a))
    A.scatter_reverse()
    s = np.linalg.svd(A.to_dense(), compute_uv=False)
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[], petsc_options_prefix=f"t2_mx3_u{omega:g}_",
        petsc_options=KSP)
    prob.solve()
    return (float(s[0] / s[-1]), prob.solver.getIterationNumber(),
            prob.solver.getConvergedReason())


def mixed(omega: float):
    msh, ne, p1 = mesh_and_spaces()
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([ne, p1]))
    (u, p), (v, q) = ufl.TrialFunctions(W), ufl.TestFunctions(W)
    a = (ufl.inner(ufl.curl(u), ufl.curl(v)) + omega ** 2 * ufl.inner(u, v)
         + ufl.inner(ufl.grad(p), v) + ufl.inner(u, ufl.grad(q))) * ufl.dx
    L = ufl.inner(ufl.as_vector((1.0, 0.0)), v) * ufl.dx
    A = dolfinx.fem.assemble_matrix(dolfinx.fem.form(a))
    A.scatter_reverse()
    dense = A.to_dense()
    _, qmap = W.sub(1).collapse()
    keep = np.setdiff1d(np.arange(dense.shape[0]), np.array([qmap[0]]))
    s = np.linalg.svd(dense[np.ix_(keep, keep)], compute_uv=False)
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[], petsc_options_prefix=f"t2_mx3_m{omega:g}_",
        petsc_options=KSP)
    prob.solve()
    return (float(s[0] / s[-1]), prob.solver.getIterationNumber(),
            prob.solver.getConvergedReason())


def main() -> int:
    print(f"formulation_under_test={'mixed_A_phi' if MUTATE else 'omega2_mass_only'}")

    ams = ""
    msh, ne, _ = mesh_and_spaces()
    V = dolfinx.fem.functionspace(msh, ne)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    try:
        p = dolfinx.fem.petsc.LinearProblem(
            (ufl.inner(ufl.curl(u), ufl.curl(v))
             + 1e-6 * ufl.inner(u, v)) * ufl.dx,
            ufl.inner(ufl.as_vector((1.0, 0.0)), v) * ufl.dx,
            bcs=[], petsc_options_prefix="t2_mx3_ams_",
            petsc_options={"ksp_type": "gmres", "pc_type": "hypre",
                           "pc_hypre_type": "ams", "ksp_max_it": 200})
        p.solve()
        print(f"ams_ran=True reason={p.solver.getConvergedReason()}")
    except Exception as exc:
        ams = f"{type(exc).__name__} {str(exc).splitlines()[0]}"
        print(f"ams_request_failed={ams}")
    print(f"ams_preconditioner_unusable_from_linearproblem={bool(ams)}")

    conds, iters, mixed_conds = [], [], []
    for w in OMEGAS:
        c, it, reason = mixed(w) if MUTATE else unmixed(w)
        cm, _, _ = mixed(w)
        conds.append(c)
        iters.append(it)
        mixed_conds.append(cm)
        print(f"omega={w:g} condition_number_under_test={c:.4e} "
              f"gmres_iterations={it} reason={reason} "
              f"condition_number_mixed_A_phi={cm:.4e}")

    per_decade = [conds[i + 1] / conds[i] for i in range(len(conds) - 1)]
    print(f"condition_growth_per_decade={[round(x, 1) for x in per_decade]}")
    tracks = all(50.0 < r < 200.0 for r in per_decade)
    print(f"under_test_condition_number_grows_100x_per_decade_of_omega={tracks}")
    grow = iters[-1] > 10 * max(1, iters[0])
    print(f"gmres_iterations_grow_by_more_than_10x={grow}")
    flat = max(mixed_conds[1:]) / min(mixed_conds[1:]) < 2.0
    print(f"mixed_A_phi_condition_number_stays_bounded={flat}")

    if bool(ams) and tracks and grow and flat:
        print("VERDICT=low_frequency_conditioning_degrades_like_one_over_"
              "omega_squared")
        return 0
    print("VERDICT=conditioning_was_omega_independent")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
