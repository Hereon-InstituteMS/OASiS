"""Tier-2 for fenics heat#3: the adiabatic-cell test, and what each mass-term
mistake actually looks like.

All walls natural, uniform T0 = 1, no source: the answer must stay T = 1 for
ever. The claim's IMPORTANT CORRECTION is the point of this fixture — flipping
the sign of the old-time mass term does NOT blow up. max|T| stays exactly 1
while int(T dx) alternates sign every step. A different mistake, dropping the
mass term from the LHS, leaves the singular pure-Neumann stiffness matrix and
does blow up. The two must be distinguishable, otherwise the diagnosis is a
guess.

Mutation control: T2_MUTATE=1 uses the correct positive sign on both sides; the
alternation disappears.
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

NSTEP, DT = 20, 0.01


def run(variant: str):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 16, 16)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    T_n = dolfinx.fem.Function(V)
    T_n.x.array[:] = 1.0
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dt = dolfinx.fem.Constant(msh, DT)
    stiff = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    if variant == "correct":
        a = (u / dt) * v * ufl.dx + stiff
        L = (T_n / dt) * v * ufl.dx
    elif variant == "rhs_sign_flipped":
        a = (u / dt) * v * ufl.dx + stiff
        L = -(T_n / dt) * v * ufl.dx
    elif variant == "no_lhs_mass":
        a = stiff
        L = (T_n / dt) * v * ufl.dx
    else:
        raise ValueError(variant)
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[], petsc_options_prefix="t2_ms_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    peaks, integrals = [], []
    for _ in range(NSTEP):
        T_h = prob.solve()
        if isinstance(T_h, tuple):
            T_h = T_h[0]
        peaks.append(float(np.max(np.abs(T_h.x.array))))
        integrals.append(float(dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(T_h * ufl.dx))))
        T_n.x.array[:] = T_h.x.array
    return peaks, integrals


def main() -> int:
    variant = "correct" if MUTATE else "rhs_sign_flipped"
    peaks, integrals = run(variant)
    ok_peaks, ok_int = run("correct")
    bad_peaks, _ = run("no_lhs_mass")

    print(f"correct_peak_first={ok_peaks[0]:.6f} "
          f"correct_peak_last={ok_peaks[-1]:.6f} "
          f"correct_integral_last={ok_int[-1]:.6f}")
    print(f"variant_peak_first={peaks[0]:.6f} "
          f"variant_peak_last={peaks[-1]:.6f}")
    stays_one = all(abs(p - 1.0) < 1e-9 for p in peaks)
    signs = [1 if s > 0 else -1 for s in integrals]
    alternates = all(signs[i] != signs[i + 1] for i in range(len(signs) - 1))
    print(f"sign_flip_peak_stays_one={stays_one}")
    print(f"sign_flip_integral_alternates={alternates}")
    blows_up = (not np.isfinite(bad_peaks[-1])) or bad_peaks[-1] > 1e10
    print(f"no_lhs_mass_blows_up={blows_up}")
    correct_holds = all(abs(p - 1.0) < 1e-9 for p in ok_peaks) and         abs(ok_int[-1] - 1.0) < 1e-9
    print(f"correct_form_holds_T_equal_one={correct_holds}")
    if stays_one and alternates and blows_up and correct_holds:
        print("VERDICT=sign_flip_alternates_bounded_missing_mass_blows_up")
        return 0
    print("VERDICT=variants_not_distinguished")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
