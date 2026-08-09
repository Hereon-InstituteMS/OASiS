"""Tier-2 for fenics cahn_hilliard#5: verify a Cahn-Hilliard run with two checks
that need no reference solution -- total mass must be conserved and the total
free energy must decrease monotonically.

The fixture assembles mass = fem.assemble_scalar(fem.form(c*dx)) and
energy = fem.assemble_scalar(fem.form((100*c^2*(1-c)^2
+ 0.5*lmbda*inner(grad(c), grad(c)))*dx)) at every step of two otherwise
identical runs (unit square 24x24, P1 x P1 mixed, lmbda = 1e-2, M = 1,
theta = 0.5, dt = 5e-6, 25 steps), the second one with the wrong variant: a
Dirichlet condition on c over the whole boundary.

Observed on dolfinx 0.10.0: with no Dirichlet condition the mass drift over the
run is at round-off (order 1e-16) and the free energy is strictly
non-increasing; the same run with c pinned on the boundary drifts by order 1e-2
-- thirteen orders of magnitude more -- while every step still converges and the
energy still decreases. A visible drift means a boundary condition is leaking
mass, not that the solver is inaccurate.

Mutation control: T2_MUTATE=1 drops the Dirichlet condition from the second run
as well, so the drift stays at round-off.
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

N, LMBDA, MOB, THETA, DT, STEPS = 24, 1.0e-2, 1.0, 0.5, 5.0e-6, 25


def march(tag: str, with_bc: bool):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    ME = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P1, P1]))
    u, u0 = dolfinx.fem.Function(ME), dolfinx.fem.Function(ME)
    rng = np.random.default_rng(7)
    u.sub(0).interpolate(lambda x: 0.63 + 0.02 * (0.5 - rng.random(x.shape[1])))
    u.sub(1).interpolate(lambda x: np.zeros(x.shape[1]))
    u.x.scatter_forward()
    u0.x.array[:] = u.x.array
    q, v = ufl.TestFunctions(ME)
    c, mu = ufl.split(u)
    c0, mu0 = ufl.split(u0)
    cv = ufl.variable(c)
    dfdc = ufl.diff(100.0 * cv**2 * (1 - cv) ** 2, cv)
    mu_mid = (1.0 - THETA) * mu0 + THETA * mu
    F = ((c - c0) * q * ufl.dx
         + DT * MOB * ufl.dot(ufl.grad(mu_mid), ufl.grad(q)) * ufl.dx
         + mu * v * ufl.dx - dfdc * v * ufl.dx
         - LMBDA * ufl.dot(ufl.grad(c), ufl.grad(v)) * ufl.dx)

    bcs = []
    if with_bc:
        V0, _ = ME.sub(0).collapse()
        msh.topology.create_connectivity(1, 2)
        facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
        dofs = dolfinx.fem.locate_dofs_topological((ME.sub(0), V0), 1, facets)
        g = dolfinx.fem.Function(V0)
        g.x.array[:] = 0.63
        bcs = [dolfinx.fem.dirichletbc(g, dofs, ME.sub(0))]

    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, u, bcs=bcs, petsc_options_prefix=f"t2_ch5_{tag}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_max_it": 40})
    mass = dolfinx.fem.form(c * ufl.dx)
    energy = dolfinx.fem.form(
        (100.0 * c**2 * (1 - c) ** 2
         + 0.5 * LMBDA * ufl.inner(ufl.grad(c), ufl.grad(c))) * ufl.dx)
    m = [dolfinx.fem.assemble_scalar(mass)]
    e = [dolfinx.fem.assemble_scalar(energy)]
    reasons = []
    for _ in range(STEPS):
        u0.x.array[:] = u.x.array
        prob.solve()
        u.x.scatter_forward()
        reasons.append(prob.solver.getConvergedReason())
        m.append(dolfinx.fem.assemble_scalar(mass))
        e.append(dolfinx.fem.assemble_scalar(energy))
    m, e = np.array(m), np.array(e)
    rises = np.diff(e)
    return dict(n_bcs=len(bcs), all_converged=all(r > 0 for r in reasons),
                drift=float(abs(m[-1] - m[0])),
                rel_drift=float(abs(m[-1] - m[0]) / abs(m[0])),
                monotone=bool(np.all(rises <= 1.0e-12 * abs(e[0]))),
                worst_rise=float(rises.max()), e0=float(e[0]),
                e1=float(e[-1]))


def main() -> int:
    free = march("nobc", with_bc=False)
    pinned = march("bc", with_bc=not MUTATE)
    for tag, r in (("no_bc", free), ("dirichlet_on_c", pinned)):
        print(f"{tag}: n_bcs={r['n_bcs']} all_converged={r['all_converged']} "
              f"mass_drift={r['drift']:.3e} rel_drift={r['rel_drift']:.3e} "
              f"energy {r['e0']:.6f} -> {r['e1']:.6f} "
              f"worst_step_rise={r['worst_rise']:.3e} "
              f"monotone={r['monotone']}")

    roundoff = free["drift"] < 1.0e-13
    print(f"no_bc_mass_drift_at_roundoff={roundoff}")
    print(f"no_bc_free_energy_monotonically_decreasing={free['monotone']}")
    leaks = pinned["drift"] > 1.0e-3
    print(f"dirichlet_on_c_leaks_mass={leaks}")
    orders = leaks and roundoff and pinned["drift"] > 1.0e10 * max(
        free["drift"], 1.0e-18)
    print(f"drift_ratio_is_many_orders_of_magnitude={orders}")
    print("dirichlet_run_still_converged_and_still_lost_energy="
          f"{pinned['all_converged'] and pinned['monotone']}")

    if (roundoff and free["monotone"] and free["all_converged"] and leaks
            and orders):
        print("VERDICT=mass_drift_means_a_leaking_bc_not_an_inaccurate_solver")
        return 0
    print("VERDICT=no_leak_detected")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
