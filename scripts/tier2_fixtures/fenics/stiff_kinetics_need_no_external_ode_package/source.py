"""Tier-2 for fenics reaction_diffusion#8: an external stiff-ODE package is NOT
required for high Damkohler numbers in dolfinx. The advice that "for very stiff
systems (Da > 1000) external SUNDIALS coupling is required" is falsified twice
over: plain backward Euler through NonlinearProblem handles it, and the packages
that advice sends the user to are not installed in a standard conda-forge fenics
environment.

Two-species 2A <-> B on a 24x24 unit square, D = 0.01, dt = 0.05, 5 backward-
Euler steps, forward rate constant swept over 1e2, 1e3, 1e4, 1e5, 1e6, 1e8, i.e.
Damkohler = k*L^2/D from 1e4 to 1e10. The default run also tries the external
route the advice recommends.

Observed on dolfinx 0.10.0: every step at every rate returns a positive SNES
converged reason, the field stays finite and below 1.0, and the stoichiometric
invariant A + 2B is unchanged to round-off. import sundials / scikits.odes /
assimulo / cvode all raise ModuleNotFoundError.

Mutation control: T2_MUTATE=1 stays inside dolfinx and never attempts the
external import.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import importlib  # noqa: E402

import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

N, D, KR, DT, NSTEP = 24, 0.01, 1.0, 0.05, 5
RATES = (1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e8)
PACKAGES = ("sundials", "scikits.odes", "assimulo", "cvode")


def run(kf: float):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P1, P1]))
    w = dolfinx.fem.Function(W)
    w_n = dolfinx.fem.Function(W)
    w_n.sub(0).interpolate(lambda x: 1.0 + 0.5 * np.sin(2 * np.pi * x[0]))
    w_n.sub(1).interpolate(lambda x: np.full_like(x[0], 0.2))
    w_n.x.scatter_forward()
    w.x.array[:] = w_n.x.array
    A, B = ufl.split(w)
    An, Bn = ufl.split(w_n)
    va, vb = ufl.TestFunctions(W)
    r = kf * A * A - KR * B
    F = (((A - An) / DT) * va * ufl.dx
         + D * ufl.dot(ufl.grad(A), ufl.grad(va)) * ufl.dx
         + 2 * r * va * ufl.dx
         + ((B - Bn) / DT) * vb * ufl.dx
         + D * ufl.dot(ufl.grad(B), ufl.grad(vb)) * ufl.dx
         - r * vb * ufl.dx)
    inv = dolfinx.fem.form((A + 2 * B) * ufl.dx)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, w, petsc_options_prefix=f"t2_rd8_{int(np.log10(kf))}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    w.x.array[:] = w_n.x.array
    inv0 = float(dolfinx.fem.assemble_scalar(inv))
    reasons = []
    for _ in range(NSTEP):
        prob.solve()
        w.x.scatter_forward()
        reasons.append(int(prob.solver.getConvergedReason()))
        w_n.x.array[:] = w.x.array
    inv1 = float(dolfinx.fem.assemble_scalar(inv))
    return (reasons, float(np.max(np.abs(w.x.array))),
            bool(np.all(np.isfinite(w.x.array))), inv0, inv1)


def main() -> int:
    ok_all, finite_all, peak_hi, drift_hi = True, True, 0.0, 0.0
    for kf in RATES:
        reasons, peak, finite, i0, i1 = run(kf)
        drift = abs(i1 - i0) / abs(i0)
        ok = all(r > 0 for r in reasons) and len(reasons) == NSTEP
        ok_all &= ok
        finite_all &= finite
        peak_hi = max(peak_hi, peak)
        drift_hi = max(drift_hi, drift)
        print(f"forward_rate={kf:.0e} damkohler={kf / D:.0e} "
              f"snes_reasons={reasons} max_abs_c={peak:.4f} "
              f"invariant_drift={drift:.3e} all_steps_converged={ok}")

    missing = []
    if not MUTATE:
        for name in PACKAGES:
            try:
                importlib.import_module(name)
            except ImportError as exc:
                missing.append(name)
                print(f"import_{name.replace('.', '_')}_error: "
                      f"{type(exc).__name__}: {exc}")
        print(f"external_stiff_ode_packages_missing={len(missing)}"
              f"/{len(PACKAGES)}")
        print(f"no_external_stiff_ode_package_is_importable="
              f"{len(missing) == len(PACKAGES)}")
    print(f"backward_euler_converged_at_every_rate_up_to_da_1e10={ok_all}")
    print(f"field_stayed_finite_and_bounded={finite_all and peak_hi < 2.0}")
    print(f"invariant_held_to_roundoff_at_every_rate={drift_hi < 1e-12}")

    # the verdict is about backward Euler; the import evidence is reported
    # separately so that T2_MUTATE=1 (which never touches the external route)
    # still ends on the honest verdict
    if ok_all and finite_all and peak_hi < 2.0 and drift_hi < 1e-12:
        print("VERDICT=high_damkohler_needs_no_external_ode_package")
        return 0
    print("VERDICT=backward_euler_could_not_do_it")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
