"""Tier-2: a Neo-Hookean Newton run that goes NaN and still exits 0.

Claim: skfem hyperelasticity#10 -- an aggressive default load ("E=1.0 /
traction=0.1 -- ~10% normalised force") drives det(F) <= 0 in a few quadrature
points after the first iteration, log(J) returns NaN, Newton diverges to
NaN-everywhere displacements, and the exit code is STILL 0 so an rc-only gate
misses it.  Defence: keep the first-iteration strain small via a stiffer
material (E=1000) or a smaller traction, and add explicit 'nan' sentinels to
the gate.

Measured on skfem 12.0.1 (MeshTri().refined(3), ElementVector(ElementTriP1()),
plane strain nu = 0.3, left edge clamped, dead traction on the right facets):

  * THE QUOTED PARAMETER PAIR IS WRONG.  E = 1.0 with an AXIAL traction of
    0.1 converges cleanly in four Newton steps: min det(F) stays above 1, no
    NaN, no warning.  The entry's specific numbers do not reproduce the
    failure.
  * THE MECHANISM IS REAL and reproduces once the load is large enough to
    invert an element -- here a TRANSVERSE traction of 0.5 at E = 1.0.  The
    first Newton step overshoots, min det(F) goes NEGATIVE, log(J) is NaN
    from then on, and every subsequent displacement is NaN.
  * IT IS NOT ENTIRELY SILENT: scipy emits MatrixRankWarning("Matrix is
    exactly singular") once the tangent fills with NaN.  But no exception is
    raised, and a child process running the diverged case exits with
    returncode 0 -- verified here by actually launching one -- so an rc-only
    gate does pass on a NaN answer.
  * THE PRESCRIBED DEFENCE WORKS: at E = 1000 with the same traction the
    first-iteration strain is small, min det(F) stays at 1 to four decimals
    and the run converges.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings

import numpy as np
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementVector,
    FacetBasis,
    LinearForm,
    MeshTri,
    condense,
    solve,
)
from skfem.helpers import ddot, det, grad, inv, transpose

NU = 0.3


def lame(E):
    return E / (2 * (1 + NU)), E * NU / ((1 + NU) * (1 - 2 * NU))


def build(E):
    mu, lam = lame(E)

    def F_of(w):
        du = w["disp"].grad
        ident = np.zeros_like(du)
        ident[0, 0] = 1.0
        ident[1, 1] = 1.0
        return ident + du

    @LinearForm
    def res(v, w):
        F = F_of(w)
        Fit = transpose(inv(F))
        with np.errstate(all="ignore"):
            P = mu * (F - Fit) + lam * np.log(det(F)) * Fit
        return ddot(P, grad(v))

    @BilinearForm
    def tang(u, v, w):
        F = F_of(w)
        Fit = transpose(inv(F))
        with np.errstate(all="ignore"):
            lnJ = np.log(det(F))
        dF = grad(u)
        T = np.einsum("ij...,kj...,kl...->il...", Fit, dF, Fit)
        return ddot(mu * dF + (mu - lam * lnJ) * T
                    + lam * ddot(Fit, dF) * Fit, grad(v))

    return res, tang


def run(E, traction, direction, nit=4, refine=3):
    res, tang = build(E)
    m = MeshTri().refined(refine).with_boundaries({
        "left": lambda x: x[0] < 1e-10,
        "right": lambda x: x[0] > 1.0 - 1e-10,
    })
    e = ElementVector(ElementTriP1())
    ib = Basis(m, e)
    fb = FacetBasis(m, e, facets=m.boundaries["right"])

    @LinearForm
    def load(v, w):
        return traction * v[direction]

    f_ext = load.assemble(fb)
    D = ib.get_dofs("left").all()
    u = ib.zeros()
    jmin, msgs = [], []
    for _ in range(nit):
        w = ib.interpolate(u)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            R = res.assemble(ib, disp=w) - f_ext
            K = tang.assemble(ib, disp=w)
            u = u + solve(*condense(K, -R, D=D))
            msgs += [str(c.message) for c in caught]
        du = ib.interpolate(u).grad
        ident = np.zeros_like(du)
        ident[0, 0] = 1.0
        ident[1, 1] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            jmin.append(float(np.nanmin(np.asarray(det(ident + du)))
                              if np.isfinite(np.asarray(det(ident + du))).any()
                              else float("nan")))
    return u, jmin, sorted(set(msgs))


CHILD = textwrap.dedent('''
    import runpy, sys, warnings, numpy as np
    warnings.simplefilter("ignore")
    mod = runpy.run_path(sys.argv[1])
    u, jmin, msgs = mod["run"](1.0, 0.5, 1)
    print("Max displacement:", np.abs(u).max())
''')


def main() -> int:
    ok = True

    # --- the parameter pair the entry actually names -------------------
    u_a, j_a, m_a = run(1.0, 0.1, 0)
    print(f"quoted_E1_traction0p1_axial_min_detF={j_a[-1]:.4f}")
    print(f"quoted_E1_traction0p1_axial_has_nan={bool(np.isnan(u_a).any())}")
    print(f"quoted_E1_traction0p1_axial_warnings={m_a!r}")
    print(f"quoted_parameter_pair_reproduces_the_failure="
          f"{bool(np.isnan(u_a).any())}")
    if np.isnan(u_a).any():
        print("FAIL: the quoted E=1.0/traction=0.1 pair DID go NaN, so the "
              "entry's numbers are right after all", file=sys.stderr)
        ok = False

    # --- a load large enough to invert an element ----------------------
    u_b, j_b, m_b = run(1.0, 0.5, 1)
    print(f"aggressive_first_iter_min_detF={j_b[0]:.4f}")
    print(f"aggressive_min_detF_went_negative={bool(j_b[0] < 0.0)}")
    print(f"aggressive_solution_has_nan={bool(np.isnan(u_b).any())}")
    print(f"aggressive_warnings={m_b!r}")
    print(f"aggressive_run_is_completely_silent={not m_b}")
    if not (j_b[0] < 0.0):
        print("FAIL: the aggressive load did not invert an element",
              file=sys.stderr)
        ok = False
    if not np.isnan(u_b).any():
        print("FAIL: the aggressive load did not produce NaN displacements",
              file=sys.stderr)
        ok = False

    # --- an rc-only gate really would pass -----------------------------
    child = subprocess.run(
        [sys.executable, "-c", CHILD, __file__],
        capture_output=True, text=True, timeout=180)
    out = (child.stdout or "") + (child.stderr or "")
    print(f"child_returncode={child.returncode}")
    print(f"child_printed_nan={'nan' in out.lower()}")
    print(f"rc_only_gate_would_pass_on_nan="
          f"{child.returncode == 0 and 'nan' in out.lower()}")
    if child.returncode != 0:
        print(f"FAIL: the diverged child exited {child.returncode}, so rc "
              f"alone WOULD have caught it", file=sys.stderr)
        ok = False
    if "nan" not in out.lower():
        print("FAIL: the child did not print a nan displacement",
              file=sys.stderr)
        ok = False

    # --- the prescribed defence ----------------------------------------
    u_c, j_c, m_c = run(1000.0, 0.5, 1)
    print(f"stiff_E1000_min_detF={j_c[-1]:.4f}")
    print(f"stiff_E1000_has_nan={bool(np.isnan(u_c).any())}")
    print(f"stiff_E1000_max_disp={float(np.abs(u_c).max()):.4e}")
    print(f"stiffer_material_defence_works="
          f"{not np.isnan(u_c).any() and j_c[-1] > 0.9}")
    if np.isnan(u_c).any() or j_c[-1] <= 0.9:
        print("FAIL: the stiffer material did not keep det(F) healthy",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
