"""Tier-2 for fenics multiphase#7: a sharp `ufl.conditional` Heaviside for
phase-dependent material properties is differentiable as far as UFL and SNES are
concerned, so it does NOT break Newton. The genuine reason to prefer a smoothed
0.5*(1 + tanh(phi/eps)) is accuracy: a conditional is evaluated at QUADRATURE
POINTS, so the effective location of the material jump depends on the quadrature
degree of the form.

32x32 unit square, eps = 3h, droplet r = 0.25, rho = 1 + 999*conditional(phi>0,
1, 0) multiplying the mass term - a 1000:1 density ratio - 8 backward Euler steps
of dt = 1e-3 with the analytic ufl.derivative Jacobian. Every step converges with
CONVERGED_FNORM_RELATIVE in 2 to 3 Newton iterations and ufl.derivative through
the conditional is a valid ufl Form, so the previously quoted DIVERGED_FNORM_NAN
stall does not exist. What IS real: assembling int H dx with the quadrature
degree set through ufl.Measure metadata moves the answer, and it moves several
times more for the sharp conditional than for the tanh-smoothed Heaviside.

Mutation control: T2_MUTATE=1 selects the smoothed Heaviside; Newton still
converges (it always did) and the quadrature sensitivity drops below the
threshold, so the sharp-conditional finding is lost.
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

N, NSTEP, DT, EPS_OVER_H, R = 32, 8, 1e-3, 3.0, 0.25
DEGREES = (1, 2, 3, 4, 6, 10)
REASONS = {v: k for k, v in PETSc.SNES.ConvergedReason.__dict__.items()
           if isinstance(v, int)}


def build():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    eps = EPS_OVER_H / N
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    phi, phi_n = dolfinx.fem.Function(V), dolfinx.fem.Function(V)

    def ic(x):
        d = R - np.sqrt((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)
        return np.tanh(d / (eps * np.sqrt(2.0)))

    phi.interpolate(ic)
    phi_n.interpolate(ic)
    return msh, phi, phi_n, dolfinx.fem.Constant(msh, eps), ic


def heaviside(kind: str, phi, eps_c):
    if kind == "sharp":
        return ufl.conditional(ufl.gt(phi, 0.0), 1.0, 0.0)
    return 0.5 * (1.0 + ufl.tanh(phi / eps_c))


def newton_history(kind: str):
    msh, phi, phi_n, eps_c, _ = build()
    v = ufl.TestFunction(phi.function_space)
    dt_c = dolfinx.fem.Constant(msh, DT)
    rho = 1.0 + 999.0 * heaviside(kind, phi, eps_c)
    F = (rho * (phi - phi_n) / dt_c * v * ufl.dx
         + eps_c * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
         + (1.0 / eps_c) * (phi ** 3 - phi) * v * ufl.dx)
    J = ufl.derivative(F, phi)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, phi, J=J, petsc_options_prefix=f"t2_mp7_{kind}_",
        petsc_options={"snes_type": "newtonls", "ksp_type": "preonly",
                       "pc_type": "lu"})
    reasons, its = [], []
    for _ in range(NSTEP):
        prob.solve()
        reasons.append(prob.solver.getConvergedReason())
        its.append(prob.solver.getIterationNumber())
        phi_n.x.array[:] = phi.x.array
    return reasons, its, isinstance(J, ufl.form.Form), phi


def quadrature_spread(kind: str) -> tuple[float, list[float]]:
    msh, phi, _, eps_c, _ = build()
    H = heaviside(kind, phi, eps_c)
    vals = []
    for q in DEGREES:
        dx = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": q})
        vals.append(float(dolfinx.fem.assemble_scalar(dolfinx.fem.form(H * dx))))
    return max(vals) - min(vals), vals


def main() -> int:
    kind = "smooth" if MUTATE else "sharp"
    reasons, its, is_form, phi = newton_history(kind)
    names = sorted({REASONS.get(r) for r in reasons})
    print(f"heaviside={kind} density_ratio=1000 reasons={names} "
          f"iterations={its}")
    conv = all(r > 0 for r in reasons)
    small_its = max(its) < 5
    nan_seen = any(REASONS.get(r) == "DIVERGED_FNORM_NAN" for r in reasons)
    print(f"selected_h_newton_converged_every_step={conv}")
    print(f"selected_h_iterations_stay_below_five={small_its}")
    print(f"selected_h_saw_diverged_fnorm_nan={nan_seen}")
    print(f"derivative_through_the_material_law_is_a_ufl_form={is_form}")
    print(f"solution_is_finite={bool(np.all(np.isfinite(phi.x.array)))}")

    sharp_spread, sharp_vals = quadrature_spread("sharp")
    smooth_spread, _ = quadrature_spread("smooth")
    sel_spread = smooth_spread if MUTATE else sharp_spread
    print(f"sharp_quadrature_values={[f'{v:.6f}' for v in sharp_vals]}")
    print(f"sharp_spread={sharp_spread:.3e} smooth_spread={smooth_spread:.3e}")
    moves = sel_spread > 5e-4
    print(f"selected_h_integral_moves_with_quadrature_degree={moves}")
    print(f"sharp_is_more_quadrature_sensitive_than_smooth="
          f"{sharp_spread > 3.0 * smooth_spread}")

    if (conv and small_its and not nan_seen and is_form and moves
            and sharp_spread > 3.0 * smooth_spread):
        print("VERDICT=conditional_is_newton_safe_but_quadrature_dependent")
        return 0
    print("VERDICT=selected_heaviside_is_quadrature_insensitive")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
