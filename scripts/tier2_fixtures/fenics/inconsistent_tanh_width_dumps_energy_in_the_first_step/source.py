"""Tier-2 for fenics multiphase#6: initialise phi with the tanh profile that
BELONGS to your eps, phi0 = tanh(d/(eps*sqrt(2))). Any other width makes the
first steps spend themselves re-profiling the interface instead of moving it, so
early-time results are meaningless. Nothing errors; the only visible sign is a
steep knee in the free-energy history, so print the free energy every step.

32x32 unit square, eps = 3h, droplet of radius 0.25, 20 backward-Euler steps of
dt = 1e-3. Wrong variant: the initial profile uses a width of eps/6. Right
variant: the width is eps.

Observed: the too-sharp start begins at free energy 4.1753 and loses 0.70375 in
the FIRST step, while the consistent tanh starts at 1.4819 and loses 0.00288 -
a factor of ~244 in first-step energy release. Twenty steps later the too-sharp
run is still relaxing ~8x faster per step. Every SNES solve in both runs
converged and nothing was raised.

Mutation control: T2_MUTATE=1 selects the consistent width as the checked run;
its first-step release ratio is 1.0 and the knee disappears.
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

N, NSTEP, DT, R, EPS_OVER_H = 32, 20, 1e-3, 0.25, 3.0
SHARP, CONSISTENT = 1.0 / 6.0, 1.0
OPTS = {"snes_type": "newtonls", "ksp_type": "preonly", "pc_type": "lu"}


def run(width_factor: float, tag: str):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    eps = EPS_OVER_H / N
    width = eps * width_factor
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    phi, phi_n = dolfinx.fem.Function(V), dolfinx.fem.Function(V)

    def ic(x):
        d = R - np.sqrt((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)
        return np.tanh(d / (width * np.sqrt(2.0)))

    phi.interpolate(ic)
    phi_n.interpolate(ic)
    v = ufl.TestFunction(V)
    dt_c = dolfinx.fem.Constant(msh, DT)
    eps_c = dolfinx.fem.Constant(msh, eps)
    F = ((phi - phi_n) / dt_c * v * ufl.dx
         + eps_c * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
         + (1.0 / eps_c) * (phi ** 3 - phi) * v * ufl.dx)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, phi, petsc_options_prefix=f"t2_mp6_{tag}_", petsc_options=OPTS)
    energy = dolfinx.fem.form(
        eps_c / 2 * ufl.dot(ufl.grad(phi), ufl.grad(phi)) * ufl.dx
        + (phi ** 2 - 1) ** 2 / (4 * eps_c) * ufl.dx)

    es = [float(dolfinx.fem.assemble_scalar(energy))]
    reasons, raised = [], "none"
    for _ in range(NSTEP):
        try:
            prob.solve()
        except Exception as exc:  # noqa: BLE001
            raised = f"{type(exc).__name__}: {exc}"
        reasons.append(prob.solver.getConvergedReason())
        es.append(float(dolfinx.fem.assemble_scalar(energy)))
        phi_n.x.array[:] = phi.x.array
    return es, reasons, raised


def main() -> int:
    runs = {}
    for name, wf in (("sharp_eps_over_6", SHARP), ("consistent_eps", CONSISTENT)):
        es, reasons, raised = run(wf, name)
        runs[name] = (es, all(r > 0 for r in reasons), raised)
        print(f"width={name} energy_first={es[0]:.4f} "
              f"first_step_release={es[1] - es[0]:+.5f} "
              f"energy_after_{NSTEP}_steps={es[-1]:.4f} "
              f"last_step_release={es[-1] - es[-2]:+.6f} "
              f"all_converged={runs[name][1]} raised={raised}")

    sharp, cons = runs["sharp_eps_over_6"][0], runs["consistent_eps"][0]
    d1_sharp, d1_cons = sharp[0] - sharp[1], cons[0] - cons[1]
    late_sharp, late_cons = sharp[-2] - sharp[-1], cons[-2] - cons[-1]
    ratio_first = d1_sharp / d1_cons
    ratio_late = late_sharp / late_cons
    conv = runs["sharp_eps_over_6"][1] and runs["consistent_eps"][1]
    quiet = (runs["sharp_eps_over_6"][2] == "none"
             and runs["consistent_eps"][2] == "none")
    print(f"first_step_release_ratio={ratio_first:.1f} "
          f"late_step_release_ratio={ratio_late:.1f}")
    print(f"sharp_start_energy_is_higher={sharp[0] > 2 * cons[0]}")
    print(f"first_step_release_ratio_exceeds_50={ratio_first > 50.0}")
    print(f"still_relaxing_faster_after_{NSTEP}_steps={ratio_late > 3.0}")
    print(f"every_step_converged_in_both_runs={conv}")
    print(f"nothing_was_raised={quiet}")

    sel = "consistent_eps" if MUTATE else "sharp_eps_over_6"
    sel_es = runs[sel][0]
    sel_ratio = (sel_es[0] - sel_es[1]) / d1_cons
    print(f"selected_width={sel} selected_first_step_release_ratio={sel_ratio:.1f}")
    print(f"selected_first_step_release_is_at_least_50x_the_consistent_one="
          f"{sel_ratio > 50.0}")

    if (sel_ratio > 50.0 and ratio_first > 50.0 and ratio_late > 3.0
            and conv and quiet and sharp[0] > 2 * cons[0]):
        print("VERDICT=inconsistent_initial_width_spends_the_first_steps_reprofiling")
        return 0
    print("VERDICT=initial_width_made_no_difference")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
