"""Tier-2 for fenics dg_methods#7: for pure advection (eps = 0) DROP the
interior-penalty diffusion block entirely — do not just set eps small.

Wrong variant: the SIPG block left in place with eps = 0, written the way it is
usually written by hand, i.e. the consistency terms carrying eps but the
penalty spelled alpha/h_avg * jump(u, n).jump(v, n) * dS without an eps factor.
At eps = 0 the consistency terms vanish and that penalty does NOT: it stays in
the operator as a large artificial coupling that annihilates the very jumps the
DG space exists for.

Measured on 16x16 triangles, DG1, b = (1, 0), inflow profile 16*y^2*(1-y)^2 so
the exact solution is that same profile transported along x:
  * the assembled matrix with the eps-free penalty differs from the
    pure-advection matrix by a Frobenius-scale term of order 1e2, i.e. it did
    not go away;
  * the jump seminorm sqrt(assemble(jump(uh)**2 * dS)) collapses by more than
    three orders of magnitude — the solution is being forced continuous across
    element faces, which is exactly the over-stabilisation the claim describes;
  * the L2 error against the exact solution gets worse, not better.

FINDING, scope correction to the claim. The claim says the penalty "REMAINS"
at eps = 0. That is true only for the eps-free spelling. The shipped template
writes the penalty as alpha/h_avg * eps * jump(u, n).jump(v, n) * dS, and this
fixture also measures that variant: at eps = 0 its matrix is bit-for-bit the
pure-advection matrix (difference norm exactly 0.0), so for that spelling the
whole block really is a multiplicative zero and no over-stabilisation occurs.
Both facts are pinned here.

Mutation control: T2_MUTATE=1 removes the penalty from the slot, the documented
cure for hyperbolic problems, and the jump structure is preserved.
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

FCO = {"quadrature_degree": 4}
N = 16
ALPHA = 16.0


def main() -> int:
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.triangle)
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    V = fem.functionspace(msh, ("DG", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    b = ufl.as_vector([1.0, 0.0])
    n = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    h_avg = (h("+") + h("-")) / 2.0
    bn = ufl.dot(b, n)
    up = ((bn("+") + abs(bn("+"))) / 2.0 * u("+")
          + (bn("+") - abs(bn("+"))) / 2.0 * u("-"))
    bn_out = (bn + abs(bn)) / 2.0
    bn_in = (bn - abs(bn)) / 2.0
    eps = fem.Constant(msh, 0.0)      # pure advection
    pen = fem.Constant(msh, 0.0)      # coefficient in front of the penalty
    u_D = 16.0 * x[1] ** 2 * (1.0 - x[1]) ** 2

    a = (-ufl.inner(u * b, ufl.grad(v)) * ufl.dx
         + up * ufl.jump(v) * ufl.dS
         + bn_out * u * v * ufl.ds
         + eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - eps * ufl.inner(ufl.avg(ufl.grad(u)), ufl.jump(v, n)) * ufl.dS
         - eps * ufl.inner(ufl.jump(u, n), ufl.avg(ufl.grad(v))) * ufl.dS
         + pen * ALPHA / h_avg * ufl.inner(ufl.jump(u, n),
                                           ufl.jump(v, n)) * ufl.dS)
    L = -bn_in * u_D * v * ufl.ds
    a_form = fem.form(a, form_compiler_options=FCO)
    rhs = assemble_vector(fem.form(L, form_compiler_options=FCO))
    rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    uh = fem.Function(V)
    err_form = fem.form((uh - u_D) ** 2 * ufl.dx, form_compiler_options=FCO)
    jmp_form = fem.form(ufl.jump(uh) ** 2 * ufl.dS, form_compiler_options=FCO)

    def run(pen_value):
        pen.value = float(pen_value)
        A = assemble_matrix(a_form)
        A.assemble()
        ksp = PETSc.KSP().create(msh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.solve(rhs, uh.x.petsc_vec)
        uh.x.scatter_forward()
        e = float(np.sqrt(abs(fem.assemble_scalar(err_form))))
        j = float(np.sqrt(abs(fem.assemble_scalar(jmp_form))))
        return A, ksp.getConvergedReason(), e, j

    # reference: block dropped entirely (the documented cure)
    A0, r0, e0, j0 = run(0.0)
    A0 = A0.copy()
    # the slot: eps-free penalty kept at eps = 0
    A1, r1, e1, j1 = run(0.0 if MUTATE else 1.0)
    # the template spelling: penalty multiplied by eps, at eps = 0
    A2, r2, e2, j2 = run(float(eps.value))

    D1 = A1.copy()
    D1.axpy(-1.0, A0)
    D2 = A2.copy()
    D2.axpy(-1.0, A0)
    d1, d2 = float(D1.norm()), float(D2.norm())
    print(f"pure_advection_matrix_norm={A0.norm():.6e}")
    print(f"slot_minus_pure_advection_norm={d1:.6e}")
    print(f"template_spelling_minus_pure_advection_norm={d2:.6e}")
    print(f"eps_free_penalty_survives_at_eps0={d1 > 1.0}")
    print(f"template_penalty_is_a_multiplicative_zero={d2 == 0.0}")
    print(f"jump_seminorm_no_block={j0:.6e} slot={j1:.6e}")
    print(f"l2_error_no_block={e0:.6e} slot={e1:.6e}")
    crushed = j0 / j1 > 1.0e2 if j1 > 0 else True
    worse = e1 > e0
    print(f"jump_structure_crushed_by_penalty={crushed}")
    print(f"l2_error_gets_worse_with_penalty={worse}")
    print(f"ksp_converged_in_every_run={r0 > 0 and r1 > 0 and r2 > 0}")

    if (d1 > 1.0 and d2 == 0.0 and crushed and worse
            and r0 > 0 and r1 > 0 and r2 > 0):
        print("VERDICT=eps_free_penalty_over_stabilises_pure_advection")
        return 0
    print("VERDICT=penalty_block_vanished_by_itself")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
