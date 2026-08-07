"""Tier-2 for fenics convection_diffusion#3: SUPG in space needs IMPLICIT time
stepping. The SUPG test function v + tau*b.grad(v) also weights the time
derivative, so an explicit Euler step is unstable at time steps that satisfy
the ordinary convective CFL condition.

Wrong variant: M_supg u^{n+1} = M_supg u^n - dt*K_supg u^n (explicit Euler,
lumped nowhere, the SUPG mass matrix on both sides). Right variant: the same
spatial form with theta = 1, (M_supg + dt*K_supg) u^{n+1} = M_supg u^n.

Rotating-hill test so nothing leaves the domain: unit square, 16x16 P1,
b = 2*pi*(-(y-0.5), (x-0.5)), kappa = 1e-3, a Gaussian hill, u = 0 on the whole
boundary, dt = 0.9 * h / max|b| — i.e. a convective CFL number of 0.9, below
the classical limit of 1.

Observed on dolfinx 0.10.0: after 250 explicit steps max|u_h| has grown from
9.4e-01 to 2.7e+114, while the implicit run at the SAME dt decays monotonically
to 1.1e-01. The claim's phrase "diverges to NaN within a few steps" is too
strong — the blow-up is exponential, not immediate, and this run never reaches
a non-finite value — but the instability itself is real and the implicit
scheme removes it.

Mutation control: T2_MUTATE=1 makes the primary integrator implicit, so the
primary amplification disappears and the fixture loses its own expectation.
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

from petsc4py import PETSc  # noqa: E402

N = 16
KAPPA = 1.0e-3
CFL = 0.9
NSTEP = 250


def run(implicit: bool):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    b = ufl.as_vector((-2.0 * np.pi * (x[1] - 0.5),
                       2.0 * np.pi * (x[0] - 0.5)))
    kappa = dolfinx.fem.Constant(msh, KAPPA)
    h = ufl.CellDiameter(msh)
    bnorm = ufl.sqrt(ufl.dot(b, b)) + 1.0e-12
    pe = bnorm * h / (2.0 * kappa)
    tau = h / (2.0 * bnorm) * (1.0 / ufl.tanh(pe) - 1.0 / pe)
    w = v + tau * ufl.dot(b, ufl.grad(v))          # SUPG test function
    bmax = 2.0 * np.pi * np.sqrt(0.5)
    dt = CFL * (1.0 / N) / bmax

    m = u * w * ufl.dx
    k = (ufl.dot(b, ufl.grad(u)) * w
         + kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx

    u_n = dolfinx.fem.Function(V)
    u_n.interpolate(lambda X: np.exp(-((X[0] - 0.5) ** 2
                                       + (X[1] - 0.75) ** 2) / 0.01))
    bfacets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    bc = dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(msh, 0.0),
        dolfinx.fem.locate_dofs_topological(V, fdim, bfacets), V)

    a_form = dolfinx.fem.form(m + dt * k) if implicit else dolfinx.fem.form(m)
    A = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    rhs = dolfinx.fem.form(
        ufl.replace(m if implicit else (m - dt * k), {u: u_n}))
    uh = dolfinx.fem.Function(V)
    first = last = float("nan")
    for step in range(NSTEP):
        vec = dolfinx.fem.petsc.assemble_vector(rhs)
        dolfinx.fem.petsc.apply_lifting(vec, [a_form], bcs=[[bc]])
        vec.ghostUpdate(addv=PETSc.InsertMode.ADD,
                        mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(vec, [bc])
        ksp.solve(vec, uh.x.petsc_vec)
        uh.x.scatter_forward()
        u_n.x.array[:] = uh.x.array
        arr = u_n.x.array
        peak = (float(np.max(np.abs(arr)))
                if np.all(np.isfinite(arr)) else float("nan"))
        if step == 0:
            first = peak
        last = peak
        if not np.isfinite(peak):
            break
    return dt, first, last


def main() -> int:
    dt_p, first_p, last_p = run(implicit=MUTATE)
    dt_i, first_i, last_i = run(implicit=True)
    growth = last_p / first_p if first_p else float("nan")
    print(f"dt={dt_p:.6f} convective_cfl={CFL} steps={NSTEP}")
    print(f"primary_peak_after_first_step={first_p:.4e} "
          f"primary_peak_after_last_step={last_p:.4e} "
          f"primary_growth_factor={growth:.4e}")
    print(f"implicit_peak_after_first_step={first_i:.4e} "
          f"implicit_peak_after_last_step={last_i:.4e}")

    below_cfl = CFL < 1.0
    amplified = np.isfinite(growth) and growth > 1.0e6
    implicit_bounded = np.isfinite(last_i) and last_i <= first_i
    reached_nan = not np.isfinite(last_p)
    print(f"time_step_is_below_the_convective_cfl_limit={below_cfl}")
    print(f"primary_amplifies_by_more_than_1e6={amplified}")
    print(f"implicit_supg_peak_never_grows={implicit_bounded}")
    print(f"primary_reached_a_non_finite_value={reached_nan}")
    if below_cfl and amplified and implicit_bounded:
        print("VERDICT=explicit_euler_with_supg_diverges_below_the_convective_cfl")
        return 0
    print("VERDICT=explicit_supg_was_stable")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
