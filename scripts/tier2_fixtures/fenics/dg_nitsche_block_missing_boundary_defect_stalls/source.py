"""Tier-2 for fenics dg_methods#1: a DG advection-diffusion form also needs the
DIFFUSIVE Dirichlet value imposed weakly — the Nitsche/SIPG block on ds. With
only the advective inflow term the diffusion operator sees a pure natural
(zero-flux) condition and the boundary value is never imposed.

Wrong variant: the full upwind + SIPG interior form, correct in every other
respect, with the three ds terms
    -eps*dot(grad(u), n)*v*ds - eps*dot(grad(v), n)*u*ds + alpha/h*eps*u*v*ds
and their u_D partners left out. A single fem.Constant switches the whole block
on and off, so both variants share one compiled form.

The diagnostic is the one the claim prescribes, and it is a STALL, not a size:
the boundary defect sqrt(assemble((uh - u_D)**2 * ds)) normalised by the
interior L2 norm is measured on 8x8, 16x16 and 32x32. Observed in the
diffusion-dominated regime (eps = 1, b = (1, 0.5)): without the block the
defect is 1.9245 / 1.9238 / 1.9235 — a relative change below 1e-3 per
refinement, the fingerprint of a condition that is simply not being applied.
With the block the same quantity falls by a factor of about 3.8 per refinement
(0.04517 / 0.01205 / 0.00314). The KSP converges in all six solves, so nothing
in the solver output reveals the difference.

Mutation control: T2_MUTATE=1 switches the Nitsche block on in the slot where
it was missing; the defect then falls with every refinement and the stall
expectation is lost.
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
EPS = 1.0
MESHES = (8, 16, 32)


def defect(n_cells: int, nitsche: bool) -> tuple[int, float]:
    msh = mesh.create_unit_square(MPI.COMM_WORLD, n_cells, n_cells,
                                  mesh.CellType.triangle)
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    V = fem.functionspace(msh, ("DG", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    b = ufl.as_vector([1.0, 0.5])
    n = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    h_avg = (h("+") + h("-")) / 2.0
    alpha = 16.0
    eps = fem.Constant(msh, EPS)
    nit = fem.Constant(msh, 1.0 if nitsche else 0.0)
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
         + nit * (-eps * ufl.dot(ufl.grad(u), n) * v * ufl.ds
                  - eps * ufl.dot(ufl.grad(v), n) * u * ufl.ds
                  + alpha / h * eps * u * v * ufl.ds))
    L = (f * v * ufl.dx
         - bn_in * u_D * v * ufl.ds
         + nit * (-eps * ufl.dot(ufl.grad(v), n) * u_D * ufl.ds
                  + alpha / h * eps * u_D * v * ufl.ds))
    A = assemble_matrix(fem.form(a, form_compiler_options=FCO))
    A.assemble()
    rhs = assemble_vector(fem.form(L, form_compiler_options=FCO))
    rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    uh = fem.Function(V)
    ksp.solve(rhs, uh.x.petsc_vec)
    uh.x.scatter_forward()
    bdry = fem.assemble_scalar(
        fem.form((uh - u_D) ** 2 * ufl.ds, form_compiler_options=FCO))
    interior = fem.assemble_scalar(
        fem.form(uh ** 2 * ufl.dx, form_compiler_options=FCO))
    return ksp.getConvergedReason(), float(np.sqrt(abs(bdry) / abs(interior)))


def main() -> int:
    reasons, bad, good = [], [], []
    for n_cells in MESHES:
        r, d = defect(n_cells, nitsche=MUTATE)
        reasons.append(r)
        bad.append(d)
        print(f"slot_without_nitsche N={n_cells} reason={r} defect={d:.8f}")
    for n_cells in MESHES:
        r, d = defect(n_cells, nitsche=True)
        reasons.append(r)
        good.append(d)
        print(f"with_nitsche_block   N={n_cells} reason={r} defect={d:.8f}")

    rel = [abs(bad[i + 1] - bad[i]) / bad[i] for i in range(len(bad) - 1)]
    fall = [good[i] / good[i + 1] for i in range(len(good) - 1)]
    print(f"slot_relative_change_per_refinement="
          f"{' '.join(f'{r:.3e}' for r in rel)}")
    print(f"with_nitsche_reduction_factor_per_refinement="
          f"{' '.join(f'{r:.3f}' for r in fall)}")
    stalls = max(rel) < 1.0e-3
    falls = min(fall) > 2.0
    all_conv = all(r > 0 for r in reasons)
    print(f"defect_stalls_under_refinement={stalls}")
    print(f"defect_falls_under_refinement_with_nitsche={falls}")
    print(f"ksp_converged_in_every_run={all_conv}")
    if stalls and falls and all_conv:
        print("VERDICT=boundary_value_never_applied_without_nitsche_block")
        return 0
    print("VERDICT=no_stall_detected")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
