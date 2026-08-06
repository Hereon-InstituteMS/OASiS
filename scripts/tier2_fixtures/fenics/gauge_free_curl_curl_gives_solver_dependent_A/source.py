"""Tier-2 for fenics magnetostatics#3: the 3D curl-curl operator without a gauge
is singular - every gradient is in its kernel - and this does NOT show up as a
solver failure. Every solver reports success and returns a DIFFERENT A, while
curl A is the same to eleven digits because B = curl A is well defined even when
A is not.

Wrong variant: (curl A, curl w) = (f, w) on a 6x6x6 unit cube, N1curl degree 1,
zero tangential trace, divergence-free f = (sin(pi y), sin(pi z), sin(pi x)), and
no gauge term. Solved three ways: MUMPS-LU, CG/Jacobi and GMRES/ILU.

Observed: converged reasons 4, 2, 2 - nothing raised, no zero-pivot complaint -
with ||A||_L2 = 0.05694 (CG) and 0.05745 (GMRES) against 5.20 or 14.65 from LU,
which is not even repeatable between runs because MUMPS returns an arbitrary
member of the solution set. ||curl A||_L2 meanwhile is 0.24664765621630 / ...631
/ ...677 from the three solvers. This also falsifies the older wording quoted in
the claim: no solver returns KSP_DIVERGED_BREAKDOWN.

Mutation control: T2_MUTATE=1 adds the mass term + inner(A, w)*dx, which
regularises the kernel; the three solvers then agree on ||A||_L2 to eleven
digits.
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

N = 6
SOLVERS = (
    ("lu", {"ksp_type": "preonly", "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"}),
    ("cg_jacobi", {"ksp_type": "cg", "pc_type": "jacobi",
                   "ksp_rtol": 1e-10, "ksp_max_it": 5000}),
    ("gmres_ilu", {"ksp_type": "gmres", "pc_type": "ilu",
                   "ksp_rtol": 1e-10, "ksp_max_it": 5000}),
)


def solve(name: str, opts: dict, gauge: bool):
    msh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N)
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    V = dolfinx.fem.functionspace(
        msh, basix.ufl.element("N1curl", msh.basix_cell(), 1))
    u, w = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = ufl.as_vector((ufl.sin(ufl.pi * x[1]), ufl.sin(ufl.pi * x[2]),
                       ufl.sin(ufl.pi * x[0])))
    a = ufl.inner(ufl.curl(u), ufl.curl(w)) * ufl.dx
    if gauge:
        a = a + ufl.inner(u, w) * ufl.dx
    L = ufl.inner(f, w) * ufl.dx
    zero = dolfinx.fem.Function(V)
    bc = dolfinx.fem.dirichletbc(zero, dolfinx.fem.locate_dofs_topological(
        V, tdim - 1, dolfinx.mesh.exterior_facet_indices(msh.topology)))
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix=f"t2_ms3_{name}_{int(gauge)}_",
        petsc_options=opts)
    raised = ""
    try:
        A = prob.solve()
    except Exception as exc:  # noqa: BLE001
        return -999, f"{type(exc).__name__}", 0.0, 0.0
    reason = prob.solver.getConvergedReason()
    n_a = np.sqrt(abs(dolfinx.fem.assemble_scalar(dolfinx.fem.form(
        ufl.inner(A, A) * ufl.dx))))
    n_c = np.sqrt(abs(dolfinx.fem.assemble_scalar(dolfinx.fem.form(
        ufl.inner(ufl.curl(A), ufl.curl(A)) * ufl.dx))))
    return reason, raised, float(n_a), float(n_c)


def spread(vals: list[float]) -> float:
    return max(vals) / min(vals)


def main() -> int:
    print(f"gauge_term_added={MUTATE}")
    reasons, norms_a, norms_c = [], [], []
    for name, opts in SOLVERS:
        reason, raised, n_a, n_c = solve(name, opts, MUTATE)
        reasons.append(reason)
        norms_a.append(n_a)
        norms_c.append(n_c)
        print(f"solver={name} converged_reason={reason} raised={raised!r} "
              f"norm_A={n_a:.14f} norm_curl_A={n_c:.14f}")
    all_conv = all(r > 0 for r in reasons)
    a_spread = spread(norms_a)
    c_spread = spread(norms_c)
    print(f"norm_A_max_over_min={a_spread:.6f}")
    print(f"norm_curl_A_max_over_min_minus_one={c_spread - 1.0:.3e}")
    print(f"every_solver_reported_converged={all_conv}")
    print(f"no_solver_reported_breakdown={all(r != -12 for r in reasons)}")
    print(f"A_depends_on_the_solver_by_more_than_10x={a_spread > 10.0}")
    print(f"curl_A_agrees_to_ten_digits={c_spread - 1.0 < 1e-10}")
    if all_conv and a_spread > 10.0 and c_spread - 1.0 < 1e-10:
        print("VERDICT=gauge_free_curl_curl_leaves_A_solver_dependent")
        return 0
    print("VERDICT=A_is_solver_independent")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
