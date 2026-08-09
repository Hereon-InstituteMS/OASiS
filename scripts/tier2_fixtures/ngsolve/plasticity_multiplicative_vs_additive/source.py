"""Tier-2: at large strain the additive split eps = eps_e + eps_p and the
multiplicative split F = F_e * F_p give measurably different Cauchy stresses on
the SAME GridFunction, and the gap is of order Det(F)^-1.

Claim: ngsolve plasticity#5 -- for large deformations switch to F = F_e * F_p.
"Signal: define F = Id + Grad(gfu) with VectorH1 displacement, compute Det(F) and
Cof(F) on an integration rule; with the small-strain additive split, the Cauchy
stress assembled via Stress(eps_e) and the spatially-pushed-forward Mandel stress
from F_e disagree by a factor on the order of (Det(F))^-1 beyond ~5 %
engineering strain."

Wrong variant: keep the small-strain additive stress
sigma = 2*mu*(Sym(Grad(u)) - eps_p) + lam*tr(Sym(Grad(u)) - eps_p)*I
while the deformation is driven from 1 % to 30 % engineering strain, and compare
against the Neo-Hookean push-forward from F_e = F * Inv(F_p).

Observed on NGSolve 6.2.2604 (unit_square maxh 0.4, VectorH1 order 2, integrals
taken with Integrate(..., order=6)):
    strain  1 %   2 %   5 %   10 %   20 %   30 %
    reldiff 1.0 % 1.9 % 3.9 % 6.9 % 11.6 % 15.3 %
    |1/J-1| 0.7 % 1.4 % 3.3 % 6.3 % 11.4 % 15.5 %
so the disagreement really is of order Det(F)^-1, and it crosses 5 % between 5 %
and 10 % engineering strain.  Det(F_e)*Det(F_p) - Det(F) is 0 to machine
precision and Cof(F) == Det(F)*Inv(F).trans to ~1e-16, so the kinematics behind
the comparison are exact.

Mutation control (re-runnable): T2_MUTATE=1 applies the documented fix at the
pathology site -- USE_ADDITIVE goes False, so small_strain_stress() returns the
multiplicative Neo-Hookean push-forward and the large-strain comparison is
like-for-like.  The relative difference collapses to 0 at every strain level and
the fixture goes red on reldiff_at_20pct_strain_gt_10pct=True,
reldiff_monotonically_increasing=True and gap_is_order_inverse_detF=True.
Re-run: T2_MUTATE=1 python source.py
"""

from __future__ import annotations

import math
import os
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
    Sym,
    Trace,
    VectorH1,
    log,
    unit_square,
    x,
    y,
)

E, NU = 200000.0, 0.3
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))

A_PLASTIC = 0.004          # deviatoric plastic strain magnitude
STRAINS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)
INT_ORDER = 6

MUTATE = os.environ.get("T2_MUTATE") == "1"

# The wrong variant keeps the small-strain additive split.
# T2_MUTATE=1 switches it to the multiplicative split (the documented fix).
USE_ADDITIVE = not MUTATE

mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
AREA = Integrate(CoefficientFunction(1.0), mesh, order=INT_ORDER)
fes = VectorH1(mesh, order=2)
gfu = GridFunction(fes)

EPS_P = CoefficientFunction((A_PLASTIC, 0.0, 0.0, -A_PLASTIC), dims=(2, 2))
F_P = Id(2) + EPS_P


def frob(m) -> float:
    return math.sqrt(abs(Integrate(InnerProduct(m, m), mesh, order=INT_ORDER)) / AREA)


def mean(cf) -> float:
    return float(Integrate(cf, mesh, order=INT_ORDER)) / AREA


def multiplicative_stress(F):
    """Neo-Hookean Cauchy stress pushed forward from F_e = F * Inv(F_p)."""
    F_e = F * Inv(F_P)
    J_e = Det(F_e)
    B_e = F_e * F_e.trans
    return 1 / J_e * (MU * (B_e - Id(2)) + LAM * log(J_e) * Id(2))


def small_strain_stress(F):
    if not USE_ADDITIVE:
        return multiplicative_stress(F)
    eps = 0.5 * (F + F.trans) - Id(2)      # == Sym(Grad(gfu))
    eps_e = eps - EPS_P
    return 2 * MU * eps_e + LAM * Trace(eps_e) * Id(2)


def main() -> int:
    ok = True

    # --- the kinematics the claim tells you to build ------------------------
    gfu.Set(CoefficientFunction((0.2 * x + 0.05 * y, -0.06 * y)))
    F = Id(2) + Grad(gfu)
    print(f"F_type={type(F).__name__} disp_space_type={type(fes).__name__}")
    print(f"sym_grad_matches_F={frob(Sym(Grad(gfu)) - (0.5 * (F + F.trans) - Id(2))) < 1e-14}")
    cof_err = frob(Cof(F) - Det(F) * Inv(F).trans)
    print(f"cof_equals_det_times_invT={cof_err < 1e-14}")
    det_split = abs(mean(Det(F * Inv(F_P)) * Det(F_P) - Det(F)))
    print(f"det_multiplicative_split_exact={det_split < 1e-14}")
    if not (cof_err < 1e-14 and det_split < 1e-14):
        print(f"FAIL: kinematic identities broke (cof={cof_err:.3e}, "
              f"det_split={det_split:.3e})", file=sys.stderr)
        ok = False

    # --- WRONG variant: additive split driven to large strain ---------------
    rel, inv_j = [], []
    for e_eng in STRAINS:
        gfu.Set(CoefficientFunction((e_eng * x, -NU * e_eng * y)))
        F = Id(2) + Grad(gfu)
        sig_add = small_strain_stress(F)
        sig_mul = multiplicative_stress(F)
        rel.append(frob(sig_mul - sig_add) / frob(sig_add))
        inv_j.append(abs(1.0 / mean(Det(F)) - 1.0))

    print(f"detF_stays_positive={all(v < 1.0 for v in inv_j)}")
    print(f"reldiff_at_1pct_strain_lt_2pct={rel[0] < 0.02}")
    print(f"reldiff_at_5pct_strain_lt_5pct={rel[2] < 0.05}")
    print(f"reldiff_at_20pct_strain_gt_10pct={rel[4] > 0.10}")
    print(f"reldiff_at_30pct_strain_gt_reldiff_at_1pct={rel[-1] > rel[0]}")
    monotone = all(rel[i + 1] > rel[i] for i in range(len(rel) - 1))
    print(f"reldiff_monotonically_increasing={monotone}")
    if not (rel[0] < 0.02 and rel[4] > 0.10 and monotone):
        print(f"FAIL: the additive/multiplicative gap did not grow with strain "
              f"(rel={rel})", file=sys.stderr)
        ok = False

    # --- and the gap really is of order Det(F)^-1 ---------------------------
    order_ratio = [rel[i] / inv_j[i] for i in range(len(rel))]
    same_order = all(0.5 < r < 2.0 for r in order_ratio)
    print(f"gap_is_order_inverse_detF={same_order}")
    print(f"gap_ratio_tightens_at_large_strain="
          f"{abs(order_ratio[-1] - 1.0) < abs(order_ratio[0] - 1.0)}")
    if not same_order:
        print(f"FAIL: the gap is not of order Det(F)^-1 "
              f"(ratios={order_ratio})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
