"""Tier-2 for fenics cahn_hilliard#6: the previous-step Function must be
refreshed INSIDE the time loop -- u0.x.array[:] = u.x.array before every
problem.solve() -- and nothing in dolfinx does it for you or warns you.

Wrong variant: u0 left at the initial state. Unit square 24x24, P1 x P1 mixed,
lmbda = 1e-2, M = 1, theta = 0.5, dt = 5e-6, 25 steps, run twice: once
refreshing u0, once not.

Observed on dolfinx 0.10.0: the first step advances normally and every later
step re-solves the identical problem whose solution is already in hand, so SNES
converges after a single iteration from the second step onward (mean iteration
count about 1, against about 17 for the correct loop), every step reports a
POSITIVE converged reason, the mass drift stays at round-off so the mass check
cannot see the bug -- and the concentration simply stops evolving: its spread
stays at the initial noise level while the correct run separates past [0, 1].
(The knowledge text says "0 iterations from the second step onward"; on this
install with the default snes tolerances it is exactly 1.)

Mutation control: T2_MUTATE=1 refreshes u0 in the second run too, so the
iteration count and the phase separation both come back.
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


def march(tag: str, refresh: bool):
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
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, u, petsc_options_prefix=f"t2_ch6_{tag}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_max_it": 40})
    _, cdofs = ME.sub(0).collapse()
    cdofs = np.asarray(cdofs, dtype=np.int32)
    mass = dolfinx.fem.form(c * ufl.dx)
    m0 = dolfinx.fem.assemble_scalar(mass)
    spread0 = float(u.x.array[cdofs].max() - u.x.array[cdofs].min())
    its, reasons = [], []
    for _ in range(STEPS):
        if refresh:
            u0.x.array[:] = u.x.array
        prob.solve()
        u.x.scatter_forward()
        its.append(prob.solver.getIterationNumber())
        reasons.append(prob.solver.getConvergedReason())
    c_end = u.x.array[cdofs]
    return dict(its=its, mean_it=float(np.mean(its)),
                tail_max_it=int(max(its[1:])),
                all_converged=all(r > 0 for r in reasons),
                drift=float(abs(dolfinx.fem.assemble_scalar(mass) - m0)),
                spread0=spread0,
                spread=float(c_end.max() - c_end.min()),
                cmin=float(c_end.min()), cmax=float(c_end.max()))


def main() -> int:
    good = march("fresh", refresh=True)
    bad = march("stale", refresh=MUTATE)
    for tag, r in (("refreshed_u0", good), ("stale_u0", bad)):
        print(f"{tag}: mean_newton_its={r['mean_it']:.2f} "
              f"first_its={r['its'][:6]} max_it_after_step_1={r['tail_max_it']} "
              f"all_converged={r['all_converged']} "
              f"mass_drift={r['drift']:.3e} "
              f"c_range=[{r['cmin']:.6f}, {r['cmax']:.6f}] "
              f"spread {r['spread0']:.3e} -> {r['spread']:.3e}")

    collapsed = bad["tail_max_it"] <= 1 and good["mean_it"] > 5.0
    print(f"stale_u0_iteration_count_collapses={collapsed}")
    print(f"every_stale_step_reported_convergence={bad['all_converged']}")
    print("stale_u0_mass_conservation_still_looks_perfect="
          f"{bad['drift'] < 1.0e-13}")
    frozen = bad["spread"] < 3.0 * bad["spread0"]
    print(f"stale_u0_field_stopped_evolving={frozen}")
    moved = good["cmin"] < -0.01 and good["cmax"] > 1.01
    print(f"refreshed_loop_separated={moved}")

    if (collapsed and bad["all_converged"] and bad["drift"] < 1.0e-13
            and frozen and moved):
        print("VERDICT=stale_u0_freezes_the_run_and_every_check_still_passes")
        return 0
    print("VERDICT=stale_u0_was_caught")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
