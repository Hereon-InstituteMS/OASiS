"""Tier-2 for fenics dg_methods#4: the SIPG penalty must scale with the DG
degree — alpha = 4*(degree+1)**2, not a fixed number.

Wrong variant: alpha = 10, a value that is perfectly safe at degree 1 and 2, on
a degree-3 DG discretisation of the same advection-diffusion problem
(eps = 0.005, b = (1, 0.5), f = 1, u_D = 0 on a 16x16 triangle mesh). alpha is
a fem.Constant, so the two runs share one compiled form and differ in nothing
else.

Observed: at degree 3 with alpha = 10 the solution reaches max|u| of order 1e3
for data of size 1 on the unit square, and KSPConvergedReason is 4 — the solver
reports success. Nothing in the solver output mentions it; only a magnitude
check on uh.x.array does. Raising alpha to 4*(3+1)**2 = 64 brings the same case
back to max|u| ~ 1.2. The degree-1 control run with the same fixed alpha = 10
stays at max|u| ~ 1.3, which is why a fixed penalty survives casual testing.

Mutation control: T2_MUTATE=1 puts the degree-scaled penalty in the slot where
the fixed alpha = 10 was, so the blow-up disappears.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402

from dolfinx import fem, mesh  # noqa: E402
from dolfinx.fem.petsc import assemble_matrix, assemble_vector  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

N = 16
EPS = 0.005
FIXED_ALPHA = 10.0


def solve(degree: int, alphas):
    """Assemble once for `degree`, solve for every alpha in `alphas`."""
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.triangle)
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    V = fem.functionspace(msh, ("DG", degree))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    fco = {"quadrature_degree": 2 * degree + 2}
    b = ufl.as_vector([1.0, 0.5])
    n = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    h_avg = (h("+") + h("-")) / 2.0
    eps = EPS
    alpha = fem.Constant(msh, 1.0)
    f = fem.Constant(msh, 1.0)
    u_D = fem.Constant(msh, 0.0)
    bn = ufl.dot(b, n)
    up = ((bn("+") + abs(bn("+"))) / 2.0 * u("+")
          + (bn("+") - abs(bn("+"))) / 2.0 * u("-"))
    bn_out = (bn + abs(bn)) / 2.0
    bn_in = (bn - abs(bn)) / 2.0
    a = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - eps * ufl.inner(ufl.avg(ufl.grad(u)), ufl.jump(v, n)) * ufl.dS
         - eps * ufl.inner(ufl.jump(u, n), ufl.avg(ufl.grad(v))) * ufl.dS
         + alpha / h_avg * eps * ufl.inner(ufl.jump(u, n),
                                           ufl.jump(v, n)) * ufl.dS
         - ufl.inner(u * b, ufl.grad(v)) * ufl.dx
         + up * ufl.jump(v) * ufl.dS
         + bn_out * u * v * ufl.ds
         - eps * ufl.dot(ufl.grad(u), n) * v * ufl.ds
         - eps * ufl.dot(ufl.grad(v), n) * u * ufl.ds
         + alpha / h * eps * u * v * ufl.ds)
    L = (f * v * ufl.dx - bn_in * u_D * v * ufl.ds
         - eps * ufl.dot(ufl.grad(v), n) * u_D * ufl.ds
         + alpha / h * eps * u_D * v * ufl.ds)
    a_form = fem.form(a, form_compiler_options=fco)
    L_form = fem.form(L, form_compiler_options=fco)
    out = []
    for al in alphas:
        alpha.value = float(al)
        A = assemble_matrix(a_form)
        A.assemble()
        rhs = assemble_vector(L_form)
        rhs.ghostUpdate(addv=PETSc.InsertMode.ADD,
                        mode=PETSc.ScatterMode.REVERSE)
        ksp = PETSc.KSP().create(msh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        uh = fem.Function(V)
        ksp.solve(rhs, uh.x.petsc_vec)
        uh.x.scatter_forward()
        out.append((ksp.getConvergedReason(), float(np.abs(uh.x.array).max())))
    return out


def main() -> int:
    scaled3 = 4.0 * (3 + 1) ** 2
    slot_alpha = scaled3 if MUTATE else FIXED_ALPHA
    (r_slot, m_slot), (r_ref, m_ref) = solve(3, [slot_alpha, scaled3])
    (r_d1, m_d1), = solve(1, [FIXED_ALPHA])

    print(f"deg3_slot_alpha={slot_alpha:.1f} reason={r_slot} "
          f"max_abs_u={m_slot:.6e}")
    print(f"deg3_scaled_alpha={scaled3:.1f} reason={r_ref} "
          f"max_abs_u={m_ref:.6e}")
    print(f"deg1_fixed_alpha={FIXED_ALPHA:.1f} reason={r_d1} "
          f"max_abs_u={m_d1:.6e}")

    blows_up = m_slot > 1.0e2
    still_ok = r_slot > 0
    scaled_ok = m_ref < 1.0e1
    deg1_ok = m_d1 < 1.0e1 and r_d1 > 0
    print(f"deg3_fixed_penalty_blows_up={blows_up}")
    print(f"deg3_ksp_still_reports_converged={still_ok}")
    print(f"deg3_scaled_penalty_is_order_one={scaled_ok}")
    print(f"deg1_same_fixed_penalty_is_fine={deg1_ok}")
    print(f"blowup_ratio_slot_over_scaled={m_slot / m_ref:.3e}")

    if blows_up and still_ok and scaled_ok and deg1_ok:
        print("VERDICT=sipg_penalty_must_scale_with_degree")
        return 0
    print("VERDICT=fixed_penalty_is_safe_at_degree_3")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
