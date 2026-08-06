"""Tier-2 for fenics multiphase#1: the interface width eps must cover at least
about two cells. Under-resolving it does NOT produce NaNs or a solver failure -
it produces a small bounded overshoot of the physical range [-1, 1], and the
only way to catch it is to test max|phi| against 1 explicitly.

32x32 unit square, circular droplet, Allen-Cahn, 10 backward Euler steps of
dt = 1e-4. The fixture sweeps eps/h = 2.56, 1.28, 0.64, 0.32 and records the
extreme values of phi and the SNES converged reason of every step. The
under-resolved configuration the pitfall warns about (eps/h = 0.32) is the one
the verdict is built on: it leaves [-1, 1] by about one percent, stays finite,
and SNES reports CONVERGED_FNORM_RELATIVE at every single step - no
DIVERGED_FNORM_NAN anywhere, and the overshoot is nowhere near the 10-30% that
was previously claimed.

Mutation control: T2_MUTATE=1 selects eps/h = 2.56 for the checked run; the
range violation disappears.
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

N, NSTEP, DT, R = 32, 10, 1e-4, 0.25
SWEEP = (2.56, 1.28, 0.64, 0.32)
BAD_EOH, GOOD_EOH = 0.32, 2.56
REASONS = {v: k for k, v in PETSc.SNES.ConvergedReason.__dict__.items()
           if isinstance(v, int)}


def run(eps_over_h: float):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    eps = eps_over_h / N
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    phi = dolfinx.fem.Function(V)
    phi_n = dolfinx.fem.Function(V)

    def ic(x):
        d = R - np.sqrt((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)
        return np.tanh(d / (eps * np.sqrt(2.0)))

    phi.interpolate(ic)
    phi_n.interpolate(ic)
    v = ufl.TestFunction(V)
    dt_c = dolfinx.fem.Constant(msh, DT)
    eps_c = dolfinx.fem.Constant(msh, eps)
    F = ((phi - phi_n) / dt_c * v * ufl.dx
         + eps_c * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
         + (1.0 / eps_c) * (phi ** 3 - phi) * v * ufl.dx)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, phi, petsc_options_prefix=f"t2_mp1_{int(eps_over_h * 100)}_",
        petsc_options={"snes_type": "newtonls", "ksp_type": "preonly",
                       "pc_type": "lu"})
    lo, hi, reasons = float(phi.x.array.min()), float(phi.x.array.max()), []
    for _ in range(NSTEP):
        prob.solve()
        reasons.append(prob.solver.getConvergedReason())
        lo = min(lo, float(phi.x.array.min()))
        hi = max(hi, float(phi.x.array.max()))
        phi_n.x.array[:] = phi.x.array
    finite = bool(np.all(np.isfinite(phi.x.array)))
    return lo, hi, reasons, finite


def main() -> int:
    table = {}
    for eoh in SWEEP:
        lo, hi, reasons, finite = run(eoh)
        over = max(abs(lo), abs(hi)) - 1.0
        table[eoh] = (lo, hi, over, reasons, finite)
        print(f"eps_over_h={eoh:g} range=[{lo:.6f}, {hi:.6f}] "
              f"overshoot={over:.3e} reasons={sorted(set(reasons))} "
              f"finite={finite}")

    overs = [table[e][2] for e in SWEEP]
    trend = all(overs[i] < overs[i + 1] for i in range(len(overs) - 1))
    print(f"overshoot_grows_as_eps_over_h_falls={trend}")
    print(f"resolved_eps_over_h_2p56_stays_in_physical_range="
          f"{table[2.56][2] <= 0.0}")

    sel = GOOD_EOH if MUTATE else BAD_EOH
    lo, hi, over, reasons, finite = table[sel]
    all_pos = all(r > 0 for r in reasons)
    names = sorted({REASONS.get(r) for r in reasons})
    print(f"selected_eps_over_h={sel:g} selected_reason_names={names}")
    print(f"selected_leaves_physical_range={over > 0.0}")
    print(f"selected_overshoot_is_only_a_few_percent="
          f"{0.0 < over < 0.10}")
    print(f"selected_snes_converged_every_step={all_pos}")
    print(f"selected_solution_is_finite={finite}")
    nan_reason = any(REASONS.get(r) == "DIVERGED_FNORM_NAN" for r in reasons)
    print(f"selected_saw_diverged_fnorm_nan={nan_reason}")

    if (trend and over > 0.0 and over < 0.10 and all_pos and finite
            and not nan_reason and table[2.56][2] <= 0.0):
        print("VERDICT=under_resolved_eps_overshoots_while_snes_reports_converged")
        return 0
    print("VERDICT=resolution_made_no_difference")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
