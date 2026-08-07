"""Tier-2 for fenics cahn_hilliard#7: the concentration legitimately overshoots
the two wells -- the converged solution takes values slightly below 0 and
slightly above 1. Do not treat that as a bug and do not clamp it.

Unit square 24x24, P1 x P1 mixed, lmbda = 1e-2, M = 1, theta = 0.5, dt = 5e-6,
30 steps, run twice: once as written, once with the wrong variant, np.clip of
the concentration dofs into [0, 1] after every step.

Observed on dolfinx 0.10.0: the unclamped run separates to a concentration a few
percent outside [0, 1] while EVERY step converges (positive SNES reason) and the
total mass is conserved to round-off, so the overshoot is not a solver failure.
Early in the same run, before separation, the field is still inside [0, 1] and
nowhere near either well -- a field pinned inside [0, 1] is the signature of a
run that has not separated. Clamping the field into [0, 1]
each step destroys the one exact invariant the scheme has: the mass drifts by
order 1e-2 instead of 1e-16.

Mutation control: T2_MUTATE=1 removes the clamp from the second run, so mass
conservation is restored.
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

N, LMBDA, MOB, THETA, DT, STEPS, EARLY = 24, 1.0e-2, 1.0, 0.5, 5.0e-6, 30, 3


def march(tag: str, clamp: bool):
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
        F, u, petsc_options_prefix=f"t2_ch7_{tag}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_max_it": 40})
    _, cdofs = ME.sub(0).collapse()
    cdofs = np.asarray(cdofs, dtype=np.int32)
    mass = dolfinx.fem.form(c * ufl.dx)
    m0 = dolfinx.fem.assemble_scalar(mass)
    spread0 = float(u.x.array[cdofs].max() - u.x.array[cdofs].min())
    reasons, early = [], None
    for k in range(STEPS):
        u0.x.array[:] = u.x.array
        prob.solve()
        u.x.scatter_forward()
        if clamp:
            u.x.array[cdofs] = np.clip(u.x.array[cdofs], 0.0, 1.0)
            u.x.scatter_forward()
        reasons.append(prob.solver.getConvergedReason())
        if k == EARLY - 1:
            cc = u.x.array[cdofs]
            early = (float(cc.min()), float(cc.max()),
                     float(cc.max() - cc.min()))
    c_end = u.x.array[cdofs]
    return dict(all_converged=all(r > 0 for r in reasons), early=early,
                cmin=float(c_end.min()), cmax=float(c_end.max()),
                spread0=spread0, spread=float(c_end.max() - c_end.min()),
                drift=float(abs(dolfinx.fem.assemble_scalar(mass) - m0)))


def main() -> int:
    free = march("free", clamp=False)
    clamped = march("clamped", clamp=not MUTATE)
    for tag, r in (("as_written", free), ("clamped_to_unit_interval",
                                         clamped)):
        print(f"{tag}: all_converged={r['all_converged']} "
              f"c_range=[{r['cmin']:.6f}, {r['cmax']:.6f}] "
              f"mass_drift={r['drift']:.3e}")
    lo, hi, sp = free["early"]
    print(f"as_written_after_{EARLY}_steps: c_range=[{lo:.6f}, {hi:.6f}] "
          f"spread {free['spread0']:.3e} -> {sp:.3e}")

    below = free["cmin"] < -0.01
    above = free["cmax"] > 1.01
    print(f"converged_solution_overshoots_below_zero={below}")
    print(f"converged_solution_overshoots_above_one={above}")
    print("every_step_converged_so_the_overshoot_is_not_a_failure="
          f"{free['all_converged']}")
    print("unclamped_run_conserves_mass_at_roundoff="
          f"{free['drift'] < 1.0e-13}")
    early_inside = (lo >= 0.0 and hi <= 1.0 and lo > 0.3 and hi < 0.9)
    print("early_field_inside_unit_interval_is_nowhere_near_the_wells="
          f"{early_inside}")
    broke = clamped["drift"] > 1.0e-3
    print(f"clamping_to_the_unit_interval_breaks_mass_conservation={broke}")

    if (below and above and free["all_converged"] and free["drift"] < 1.0e-13
            and early_inside and broke):
        print("VERDICT=the_overshoot_is_physical_the_clamp_is_the_bug")
        return 0
    print("VERDICT=no_overshoot_or_clamp_was_harmless")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
