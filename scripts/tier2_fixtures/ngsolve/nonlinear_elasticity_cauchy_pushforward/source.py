"""Tier-2: pushing PK1 forward with the PK2 formula gives a non-symmetric,
wrong-by-a-factor-of-F Cauchy stress.

Claim: ngsolve nonlinear_elasticity#5 -- sigma = (1/J) F S F^T with S = dW/dE the
2nd Piola-Kirchhoff stress.  Writing sigma = (1/J) F P F^T with P = F S (the 1st
PK / PK1) mixes up the push-forward; correct alternatives are
sigma = (1/J) F S F^T or sigma = (1/J) P F^T.

Wrong variant: assemble sigma_wrong = (1/J) * F * P * F.trans on a genuinely
non-symmetric F (shear + stretch) and compare against the two correct forms.

Observed on NGSolve 6.2.2604: the correct push-forward is symmetric to machine
precision, the PK1-in-PK2-slot version is visibly non-symmetric and differs from
the correct stress by an O(1) relative amount, while (1/J) P F^T reproduces
(1/J) F S F^T to machine precision.  Cof(F) == Det(F) * Inv(F).trans is also
pinned as an independent check that the tensor algebra itself is intact.
"""

from __future__ import annotations

import math
import sys

from ngsolve import (
    Cof,
    CoefficientFunction,
    Det,
    GridFunction,
    Grad,
    Id,
    InnerProduct,
    Integrate,
    Inv,
    Mesh,
    VectorH1,
    log,
    unit_square,
    x,
    y,
)

E, NU = 200.0, 0.3
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))


def l2(mesh, mat) -> float:
    return math.sqrt(abs(Integrate(InnerProduct(mat, mat), mesh)))


def main() -> int:
    ok = True

    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
    fes = VectorH1(mesh, order=2)
    gfu = GridFunction(fes)
    # Shear + stretch: F is NOT symmetric, which is what exposes the mix-up.
    gfu.Set(CoefficientFunction((0.30 * y + 0.15 * x, 0.05 * y)))

    ident = Id(2)
    F = ident + Grad(gfu)
    C = F.trans * F
    J = Det(F)
    print(f"F_type={type(F).__name__}")

    jmin = Integrate(J, mesh)
    print(f"mean_detF_positive={jmin > 0}")

    # Neo-Hookean PK2 and PK1
    S = MU * ident - MU / J**2 * Inv(C) + LAM * log(J) / J**2 * Inv(C)
    P = F * S

    sigma_ok = 1 / J * F * S * F.trans          # documented, correct
    sigma_alt = 1 / J * P * F.trans             # documented alternative
    sigma_bad = 1 / J * F * P * F.trans         # the pitfall

    n_ok = l2(mesh, sigma_ok)
    n_bad = l2(mesh, sigma_bad)
    print(f"norm_correct_positive={n_ok > 1e-6}")

    # --- the two documented forms agree to machine precision ---------------
    alt_rel = l2(mesh, sigma_alt - sigma_ok) / n_ok
    alt_matches = alt_rel < 1e-12
    print(f"pk1_pushforward_matches_pk2_form={alt_matches}")
    if not alt_matches:
        print(f"FAIL: (1/J) P F^T did not reproduce (1/J) F S F^T "
              f"(rel={alt_rel:.3e})", file=sys.stderr)
        ok = False

    # --- WRONG variant: PK1 substituted into the PK2 push-forward -----------
    bad_rel = l2(mesh, sigma_bad - sigma_ok) / n_ok
    bad_is_wrong = bad_rel > 0.1
    print(f"wrong_form_reldiff_gt_0p1={bad_is_wrong}")
    if not bad_is_wrong:
        print(f"FAIL: (1/J) F P F^T agreed with the correct stress "
              f"(rel={bad_rel:.3e}) -- pitfall did not reproduce",
              file=sys.stderr)
        ok = False

    # --- symmetry is the cheap tell ---------------------------------------
    sym_ok = l2(mesh, sigma_ok - sigma_ok.trans) / n_ok
    sym_bad = l2(mesh, sigma_bad - sigma_bad.trans) / n_bad
    print(f"correct_is_symmetric={sym_ok < 1e-12}")
    print(f"wrong_is_symmetric={sym_bad < 1e-12}")
    print(f"wrong_symmetry_defect_gt_1e-3={sym_bad > 1e-3}")
    if not (sym_ok < 1e-12 and sym_bad > 1e-3):
        print(f"FAIL: symmetry signature absent (ok={sym_ok:.3e}, "
              f"bad={sym_bad:.3e})", file=sys.stderr)
        ok = False

    # --- independent sanity on the tensor algebra --------------------------
    cof_rel = l2(mesh, Cof(F) - J * Inv(F).trans) / l2(mesh, Cof(F))
    print(f"cof_equals_det_times_invT={cof_rel < 1e-12}")
    if cof_rel >= 1e-12:
        print(f"FAIL: Cof(F) != Det(F)*Inv(F).trans (rel={cof_rel:.3e})",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
