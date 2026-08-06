"""Tier-2 for fenics dg_methods#3: the DG advection flux must be UPWIND — the
value taken from the upstream side. A centred flux 0.5*(u('+') + u('-')) gives
an operator without the upwind dissipation.

Wrong variant: the interior flux written as bn('+')*avg(u), i.e. the upwind
trace with its dissipative half removed. One fem.Constant theta multiplies that
half, so theta = 1 is the upwind flux, theta = 0 the centred flux, and the two
operators come from the same compiled form on the same mesh — the dissipation
is literally the only difference.

Two things are measured. First the operator: A_upwind - A_centred is assembled
and checked to be symmetric, positive semi-definite and non-zero, which is what
"the upwind dissipation" means as a matrix statement. Second the consequence on
a steady pure-advection problem with inflow data in [0, 1].

FINDING against the claim as written. The claim predicts that the centred flux
"oscillates". On the steady problem it does something harsher and quieter: the
centred operator is SINGULAR — PETSc's LU hits a zero pivot, KSPConvergedReason
comes back -11 (KSP_DIVERGED_PC_FAILED), ksp.solve() raises nothing, and every
dof of the answer is inf. Measured with b = (1, 0.5) and again with b = (1, 0),
on 8x8 and 16x16 triangles. The upwind operator on the same mesh solves cleanly
and its solution stays within 5% of the range of the inflow data. So the fixture pins
the real signal: no dissipation -> no invertible operator, not a wobbly answer.

Mutation control: T2_MUTATE=1 puts the upwind flux in the slot where the
centred flux was, so the solve succeeds and the difference operator is zero.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402

from dolfinx import fem, mesh  # noqa: E402
from dolfinx.fem.petsc import assemble_matrix, assemble_vector  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

FCO = {"quadrature_degree": 4}
N = 8


def main() -> int:
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.triangle)
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    V = fem.functionspace(msh, ("DG", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    b = ufl.as_vector([1.0, 0.5])
    n = ufl.FacetNormal(msh)
    bn = ufl.dot(b, n)
    theta = fem.Constant(msh, 1.0)     # 1 -> upwind, 0 -> centred
    flux = (bn("+") * ufl.avg(u)
            + theta * abs(bn("+")) / 2.0 * (u("+") - u("-")))
    bn_out = (bn + abs(bn)) / 2.0
    bn_in = (bn - abs(bn)) / 2.0
    u_D = 16.0 * x[1] ** 2 * (1.0 - x[1]) ** 2      # smooth, in [0, 1]
    a = (-ufl.inner(u * b, ufl.grad(v)) * ufl.dx
         + flux * ufl.jump(v) * ufl.dS
         + bn_out * u * v * ufl.ds)
    L = -bn_in * u_D * v * ufl.ds
    a_form = fem.form(a, form_compiler_options=FCO)
    rhs = assemble_vector(fem.form(L, form_compiler_options=FCO))
    rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    def dense(A):
        ai, aj, av = A.getValuesCSR()
        return sp.csr_matrix((av, aj, ai), shape=A.getSize()).toarray()

    def run(th):
        theta.value = th
        A = assemble_matrix(a_form)
        A.assemble()
        ksp = PETSc.KSP().create(msh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        uh = fem.Function(V)
        raised = ""
        try:
            ksp.solve(rhs, uh.x.petsc_vec)
        except Exception as exc:                       # pragma: no cover
            raised = f"{type(exc).__name__}: {exc}"
        uh.x.scatter_forward()
        return dense(A), ksp.getConvergedReason(), uh.x.array.copy(), raised

    A_slot, r_slot, u_slot, raised = run(1.0 if MUTATE else 0.0)
    A_up, r_up, u_up, _ = run(1.0)

    # The dissipation as a matrix statement.
    D = A_up - A_slot
    asym = float(np.abs(D - D.T).max())
    eig = np.linalg.eigvalsh((D + D.T) / 2.0)
    dnorm = float(np.abs(D).max())
    psd = bool(eig.min() > -1.0e-12)
    print(f"upwind_minus_slot_max_asymmetry={asym:.3e} "
          f"min_eig={eig.min():.3e} max_abs={dnorm:.3e}")
    print(f"upwind_dissipation_is_symmetric_psd_and_nonzero="
          f"{asym < 1.0e-12 and psd and dnorm > 1.0e-6}")

    print(f"slot_ksp_converged_reason={r_slot} "
          f"slot_reason_is_diverged_pc_failed={r_slot == -11}")
    print(f"slot_solve_raised_nothing={raised == ''}")
    print(f"slot_solution_all_nonfinite="
          f"{bool(np.all(~np.isfinite(u_slot)))}")
    print(f"u: min={u_slot.min():.6e}, max={u_slot.max():.6e}")

    in_range = bool(u_up.min() > -5.0e-2 and u_up.max() < 1.0 + 5.0e-2)
    print(f"upwind_reference_reason_is_converged={r_up > 0}")
    print(f"upwind_reference_solution_finite="
          f"{bool(np.all(np.isfinite(u_up)))}")
    print(f"upwind_reference_within_5pc_of_inflow_data_range={in_range}")
    print(f"upwind_reference_range=[{u_up.min():.6e}, {u_up.max():.6e}]")

    if (r_slot == -11 and raised == "" and bool(np.all(~np.isfinite(u_slot)))
            and asym < 1.0e-12 and psd and dnorm > 1.0e-6
            and r_up > 0 and in_range):
        print("VERDICT=centred_flux_loses_the_dissipation_and_the_operator")
        return 0
    print("VERDICT=centred_flux_behaves_like_upwind")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
