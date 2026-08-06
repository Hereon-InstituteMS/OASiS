"""Tier-2 for fenics multiphase#5: a time step that is too large for Newton does
not manifest as an instability. Backward Euler is unconditionally stable, so the
failure lands entirely in the nonlinear solve, and the remedy is step-size
control keyed on the converged reason rather than a CFL-style formula.

Allen-Cahn droplet on a 32x32 unit square, eps = 2.56h, 3 backward Euler steps
per dt. dt = 1e-4 and dt = 1e-2 both converge with CONVERGED_FNORM_RELATIVE even
though 1e-2 is several times the explicit diffusive limit h^2/(4 eps), and the
free energy still decreases monotonically. dt = 1.0 and dt = 100.0 both fail at
the very first step with DIVERGED_LINE_SEARCH while phi stays finite and inside
[-1.01, 1.01] - there is no blow-up to see, only a negative converged reason.

Mutation control: T2_MUTATE=1 keeps the well-sized steps only; nothing fails and
the diverged-reason findings are not produced.
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

N, NSTEP, EPS_OVER_H, R = 32, 3, 2.56, 0.25
SMALL = (1e-4, 1e-2)
LARGE = (1.0, 100.0)
REASONS = {v: k for k, v in PETSc.SNES.ConvergedReason.__dict__.items()
           if isinstance(v, int)}


def run(dt: float):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    eps = EPS_OVER_H / N
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    phi, phi_n = dolfinx.fem.Function(V), dolfinx.fem.Function(V)

    def ic(x):
        d = R - np.sqrt((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)
        return np.tanh(d / (eps * np.sqrt(2.0)))

    phi.interpolate(ic)
    phi_n.interpolate(ic)
    v = ufl.TestFunction(V)
    dt_c = dolfinx.fem.Constant(msh, dt)
    eps_c = dolfinx.fem.Constant(msh, eps)
    F = ((phi - phi_n) / dt_c * v * ufl.dx
         + eps_c * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
         + (1.0 / eps_c) * (phi ** 3 - phi) * v * ufl.dx)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, phi, petsc_options_prefix=f"t2_mp5_{abs(int(np.log10(dt)))}_",
        petsc_options={"snes_type": "newtonls", "ksp_type": "preonly",
                       "pc_type": "lu"})
    energy = dolfinx.fem.form(
        eps_c / 2 * ufl.dot(ufl.grad(phi), ufl.grad(phi)) * ufl.dx
        + (phi ** 2 - 1) ** 2 / (4 * eps_c) * ufl.dx)
    es = [float(dolfinx.fem.assemble_scalar(energy))]
    reasons = []
    for _ in range(NSTEP):
        prob.solve()
        reasons.append(prob.solver.getConvergedReason())
        es.append(float(dolfinx.fem.assemble_scalar(energy)))
        phi_n.x.array[:] = phi.x.array
    finite = bool(np.all(np.isfinite(phi.x.array)))
    peak = float(np.max(np.abs(phi.x.array)))
    mono = all(es[i + 1] <= es[i] + 1e-14 for i in range(len(es) - 1))
    return reasons, finite, peak, mono


def main() -> int:
    h = 1.0 / N
    eps = EPS_OVER_H / N
    explicit_limit = h * h / (4.0 * eps)
    print(f"explicit_diffusive_limit_h2_over_4eps={explicit_limit:.3e}")

    ok = True
    for dt in SMALL:
        reasons, finite, peak, mono = run(dt)
        names = sorted({REASONS.get(r) for r in reasons})
        print(f"dt={dt:g} reasons={names} peak_abs_phi={peak:.6f} "
              f"energy_monotone={mono}")
        ok = ok and all(r > 0 for r in reasons) and finite and mono
    print(f"well_sized_steps_all_converged={ok}")
    print(f"dt_1e-2_is_above_the_explicit_limit="
          f"{1e-2 > explicit_limit}")

    if MUTATE:
        print("VERDICT=well_sized_steps_converge_and_stay_stable")
        return 0 if ok else 1

    bad_first, bounded, names_all = True, True, set()
    for dt in LARGE:
        reasons, finite, peak, _ = run(dt)
        names = sorted({REASONS.get(r) for r in reasons})
        names_all.update(names)
        print(f"dt={dt:g} reasons={names} peak_abs_phi={peak:.6f} "
              f"finite={finite}")
        bad_first = bad_first and reasons[0] < 0
        bounded = bounded and finite and peak < 1.01
    print(f"too_large_dt_fails_at_the_very_first_step={bad_first}")
    print(f"too_large_dt_field_stays_finite_and_bounded={bounded}")
    print(f"too_large_dt_reason_names={sorted(names_all)}")
    if ok and bad_first and bounded:
        print("VERDICT=large_dt_fails_in_newton_not_as_an_instability")
        return 0
    print("VERDICT=large_dt_behaved_differently")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
