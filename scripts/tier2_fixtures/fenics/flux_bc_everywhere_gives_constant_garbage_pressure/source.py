"""Tier-2 for fenics mixed_poisson#0: prescribing sigma.n on the ENTIRE boundary
of a mixed Poisson problem is not a well-posed problem. The pressure is then
determined only up to a constant and, unless the prescribed normal fluxes
integrate to exactly the source, no solution exists at all — yet the script
exits 0 and prints a pressure. The fix is to leave part of the boundary to the
NATURAL pressure condition, -u_D * dot(tau, n) * ds(tag) in L.

Wrong variant: a Dirichlet condition on the RT subspace over every exterior
facet, with a Gaussian source whose integral does not match the zero prescribed
flux. Right variant: flux prescribed on x = 0 and x = 1 only, with the natural
pressure condition on y = 0 and y = 1.

RT1 x DG0 on an 8x8 unit square, solved with a direct LU factorisation
(MUMPS). Observed on dolfinx 0.10.0: the solve reports KSPConvergedReason 4,
the vector is finite, and the pressure comes back as a single constant of
enormous magnitude — min(p) == max(p) == 1.171722e+15, a relative spread of
0.0 across all 128 pressure dofs. The well-posed variant on the same mesh gives
a pressure that actually varies, range [-8.9315e-01, 9.2133e-01].

Mutation control: T2_MUTATE=1 makes the well-posed boundary split the primary
one, the pressure is no longer a huge constant and the fixture loses its own
expectation.
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

import basix.ufl  # noqa: E402
from petsc4py import PETSc  # noqa: E402

N = 8
DEGREE = 1


def solve(flux_everywhere: bool):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    RT = basix.ufl.element("RT", msh.basix_cell(), DEGREE)
    DG = basix.ufl.element("DG", msh.basix_cell(), DEGREE - 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([RT, DG]))
    (sig, u) = ufl.TrialFunctions(W)
    (tau, v) = ufl.TestFunctions(W)
    x = ufl.SpatialCoordinate(msh)
    f = 10.0 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
    n = ufl.FacetNormal(msh)
    a = (ufl.inner(sig, tau) + ufl.div(tau) * u + ufl.div(sig) * v) * ufl.dx
    L = -f * v * ufl.dx

    V0, _ = W.sub(0).collapse()
    g = dolfinx.fem.Function(V0)
    g.x.array[:] = 0.0
    if flux_everywhere:
        facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    else:
        facets = dolfinx.mesh.locate_entities_boundary(
            msh, fdim, lambda X: np.isclose(X[0], 0.0) | np.isclose(X[0], 1.0))
        natural = dolfinx.mesh.locate_entities_boundary(
            msh, fdim, lambda X: np.isclose(X[1], 0.0) | np.isclose(X[1], 1.0))
        tags = dolfinx.mesh.meshtags(
            msh, fdim, np.sort(natural),
            np.full(len(natural), 1, dtype=np.int32))
        ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
        L = L - ufl.sin(5.0 * x[0]) * ufl.dot(tau, n) * ds(1)
    bcs = [dolfinx.fem.dirichletbc(
        g, dolfinx.fem.locate_dofs_topological((W.sub(0), V0), fdim, facets),
        W.sub(0))]

    af, Lf = dolfinx.fem.form(a), dolfinx.fem.form(L)
    A = dolfinx.fem.petsc.assemble_matrix(af, bcs=bcs)
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(Lf)
    dolfinx.fem.petsc.apply_lifting(b, [af], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    w = dolfinx.fem.Function(W)
    ksp.solve(b, w.x.petsc_vec)
    w.x.scatter_forward()
    p = w.sub(1).collapse().x.array
    return int(ksp.getConvergedReason()), np.array(p)


def main() -> int:
    reason, p = solve(flux_everywhere=not MUTATE)
    reason_ok, p_ok = solve(flux_everywhere=False)
    lo, hi = float(np.min(p)), float(np.max(p))
    spread = (hi - lo) / max(abs(hi), 1.0)
    print(f"primary_reason={reason} primary_pressure_dofs={p.size} "
          f"primary_pressure_range=[{lo:.6e}, {hi:.6e}] "
          f"primary_relative_spread={spread:.3e}")
    print(f"wellposed_reason={reason_ok} wellposed_pressure_range="
          f"[{float(np.min(p_ok)):.4e}, {float(np.max(p_ok)):.4e}]")

    converged = reason > 0
    finite = bool(np.all(np.isfinite(p)))
    constant = spread < 1.0e-9
    huge = max(abs(lo), abs(hi)) > 1.0e6
    ok_varies = (float(np.max(p_ok)) - float(np.min(p_ok))) > 1.0e-3 \
        and max(abs(float(np.min(p_ok))), abs(float(np.max(p_ok)))) < 1.0e3
    print(f"solver_reported_converged={converged}")
    print(f"primary_solution_is_finite={finite}")
    print(f"primary_pressure_is_a_single_constant={constant}")
    print(f"primary_pressure_magnitude_exceeds_1e6={huge}")
    print(f"wellposed_variant_pressure_varies_and_is_order_one={ok_varies}")
    if converged and finite and constant and huge and ok_varies:
        print("VERDICT=flux_bc_on_the_whole_boundary_returns_a_converged_constant_garbage_pressure")
        return 0
    print("VERDICT=flux_bc_on_the_whole_boundary_was_well_posed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
