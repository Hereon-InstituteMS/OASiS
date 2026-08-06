"""Tier-2: the post-yield slope of a J2 bar is E*H/(E+H), NOT E/(1 + 3*mu/H).

Claim: ngsolve plasticity#1 -- "Stress(gfu) integrated by Integrate over a
uniaxial-tension VectorH1 cantilever follows a linear elastic line whose slope
differs by a factor of E_t/E = (1 + 3*mu/H)^-1 from the elastic Young modulus at
the first integration point reaching sqrt(3/2 s:s) >= sigma_y.  The post-yield
slope is the observable."

Wrong variant: skip the radial return and keep the elastic modulus past yield.

What is measured: a homogeneous strain state is prescribed on a VectorH1
GridFunction, the J2 radial return with linear isotropic hardening is written as
NGSolve CoefficientFunction algebra (Sym, Deviator, InnerProduct, IfPos) and the
stress components are extracted with Integrate over unit_square.  Two paths:
  * uniaxial STRESS -- the lateral strain is bisected until Integrate(sigma_yy)
    is zero to machine precision (~1e-13 against sigma_y = 250);
  * simple SHEAR.

Observed on NGSolve 6.2.2604 with E = 2e5, nu = 0.3, sigma_y = 250, H = 2000:
  * elastic secant slope / E = 1.0 exactly on both paths;
  * uniaxial post-yield slope / E = 0.00990099009900860, which equals
    H/(H+E) = 0.00990099009900990 to ~1e-13 relative and is 15 % away from the
    claim's H/(H+3*mu) = 0.00859220092531395;
  * the SHEAR post-yield slope / mu is the one that equals H/(H+3*mu).
So the formula in the claim is the deviatoric / shear tangent, not the uniaxial
one; (1 + 3*mu/H)^-1 only coincides with the uniaxial ratio when 3*mu = E, i.e.
nu = 0.5.  Both are pinned below.
"""

from __future__ import annotations

import sys

from ngsolve import (
    CoefficientFunction,
    Deviator,
    GridFunction,
    Grad,
    Id,
    IfPos,
    InnerProduct,
    Integrate,
    Mesh,
    Sym,
    Trace,
    VectorH1,
    sqrt,
    unit_square,
    x,
    y,
)

E, NU, SIG_Y, H_MOD = 200000.0, 0.3, 250.0, 2000.0
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))

N_STEPS = 20
EPS_MAX = 5.0e-3          # 4x the uniaxial yield strain
GAM_MAX = 7.5e-3          # 4x the shear yield strain

# The wrong variant runs without the radial return.
RETURN_MAP_WRONG = False

mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
AREA = Integrate(CoefficientFunction(1.0), mesh)
fes = VectorH1(mesh, order=1)
gfu = GridFunction(fes)


def mat3(v) -> CoefficientFunction:
    return CoefficientFunction(tuple(v[i][j] for i in range(3) for j in range(3)),
                               dims=(3, 3))


def avg(cf) -> float:
    return float(Integrate(cf, mesh)) / AREA


def embed(e_zz: float):
    """3x3 strain from Sym(Grad(gfu)) plus a prescribed out-of-plane component."""
    e2 = Sym(Grad(gfu))
    return CoefficientFunction((e2[0, 0], e2[0, 1], 0.0,
                                e2[1, 0], e2[1, 1], 0.0,
                                0.0, 0.0, e_zz), dims=(3, 3))


def radial_return(eps3, eps_p3, alpha: float, return_map: bool = True):
    eps_e = eps3 - eps_p3
    sig_tr = 2 * MU * eps_e + LAM * Trace(eps_e) * Id(3)
    s_tr = Deviator(sig_tr)
    q_tr = sqrt(1.5 * InnerProduct(s_tr, s_tr))
    f = q_tr - (SIG_Y + H_MOD * alpha)
    if not return_map:
        return sig_tr, CoefficientFunction(0.0), eps_p3, q_tr
    dgamma = IfPos(f, f / (3 * MU + H_MOD), 0.0)
    flow_n = 1.5 * s_tr / (q_tr + 1e-30)
    sig = sig_tr - 2 * MU * dgamma * flow_n
    q = sqrt(1.5 * InnerProduct(Deviator(sig), Deviator(sig)))
    return sig, dgamma, eps_p3 + dgamma * flow_n, q


def uniaxial_path(return_map: bool = True):
    """Drive eps_xx; bisect the lateral strain so that sigma_yy = 0."""
    eps_p = [[0.0] * 3 for _ in range(3)]
    alpha = 0.0
    out = []
    for k in range(1, N_STEPS + 1):
        e_ax = k / N_STEPS * EPS_MAX
        lo, hi = -e_ax, 0.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            gfu.Set(CoefficientFunction((e_ax * x, mid * y)))
            sig, _dg, _ep, _q = radial_return(embed(mid), mat3(eps_p), alpha,
                                              return_map)
            if avg(sig[1, 1]) > 0.0:
                hi = mid
            else:
                lo = mid
        mid = 0.5 * (lo + hi)
        gfu.Set(CoefficientFunction((e_ax * x, mid * y)))
        sig, dg, ep_new, q = radial_return(embed(mid), mat3(eps_p), alpha,
                                           return_map)
        s_xx, s_yy, q_val = avg(sig[0, 0]), avg(sig[1, 1]), avg(q)
        alpha += avg(dg)
        eps_p = [[avg(ep_new[i, j]) for j in range(3)] for i in range(3)]
        out.append((e_ax, s_xx, s_yy, q_val, alpha))
    return out


def shear_path():
    eps_p = [[0.0] * 3 for _ in range(3)]
    alpha = 0.0
    out = []
    for k in range(1, N_STEPS + 1):
        gam = k / N_STEPS * GAM_MAX
        gfu.Set(CoefficientFunction((gam * y, 0.0)))
        sig, dg, ep_new, q = radial_return(embed(0.0), mat3(eps_p), alpha, True)
        tau, q_val = avg(sig[0, 1]), avg(q)
        alpha += avg(dg)
        eps_p = [[avg(ep_new[i, j]) for j in range(3)] for i in range(3)]
        out.append((gam, tau, q_val, alpha))
    return out


def secant(pts, i, j, xi=0, yi=1) -> float:
    return (pts[j][yi] - pts[i][yi]) / (pts[j][xi] - pts[i][xi])


def main() -> int:
    ok = True

    print(f"disp_space_type={type(fes).__name__} "
          f"strain_cf_type={type(Sym(Grad(gfu))).__name__}")
    gfu.Set(CoefficientFunction((0.01 * x + 0.004 * y, -0.003 * y)))
    dev_tr = abs(avg(Trace(Deviator(embed(-0.002)))))
    print(f"deviator_is_traceless={dev_tr < 1e-15}")

    # --- RIGHT variant: full radial return, uniaxial stress -----------------
    uni = uniaxial_path(True)
    max_syy = max(abs(p[2]) for p in uni)
    print(f"uniaxial_sigma_yy_is_zero={max_syy < 1e-9 * SIG_Y}")
    if max_syy >= 1e-9 * SIG_Y:
        print(f"FAIL: the uniaxial-stress condition was not met "
              f"(max|sigma_yy|={max_syy:.3e})", file=sys.stderr)
        ok = False

    el_slope = secant(uni, 1, 2)
    pl_slope = secant(uni, 17, 19)
    ratio = pl_slope / el_slope
    pred_uni = H_MOD / (H_MOD + E)
    pred_shear = H_MOD / (H_MOD + 3 * MU)
    print(f"elastic_secant_equals_E={abs(el_slope / E - 1.0) < 1e-10}")
    print(f"post_yield_slope_far_below_elastic={ratio < 0.05}")
    print(f"uniaxial_ratio_matches_H_over_H_plus_E="
          f"{abs(ratio / pred_uni - 1.0) < 1e-9}")
    print(f"uniaxial_ratio_matches_H_over_H_plus_3mu="
          f"{abs(ratio / pred_shear - 1.0) < 1e-9}")
    print(f"claim_formula_relative_error_gt_10pct="
          f"{abs(ratio / pred_shear - 1.0) > 0.10}")
    if abs(ratio / pred_uni - 1.0) >= 1e-9:
        print(f"FAIL: uniaxial post-yield ratio {ratio!r} does not match "
              f"H/(H+E)={pred_uni!r}", file=sys.stderr)
        ok = False
    if abs(ratio / pred_shear - 1.0) < 1e-9:
        print("FAIL: the claim's (1+3mu/H)^-1 now does match the uniaxial "
              "ratio -- revisit the falsification in this fixture",
              file=sys.stderr)
        ok = False

    # --- where the claim's formula actually belongs: shear ------------------
    sh = shear_path()
    el_shear = secant(sh, 1, 2)
    pl_shear = secant(sh, 17, 19)
    ratio_shear = pl_shear / el_shear
    print(f"shear_elastic_secant_equals_mu={abs(el_shear / MU - 1.0) < 1e-10}")
    print(f"shear_ratio_matches_H_over_H_plus_3mu="
          f"{abs(ratio_shear / pred_shear - 1.0) < 1e-9}")
    if abs(ratio_shear / pred_shear - 1.0) >= 1e-9:
        print(f"FAIL: shear post-yield ratio {ratio_shear!r} does not match "
              f"H/(H+3mu)={pred_shear!r}", file=sys.stderr)
        ok = False

    # --- WRONG variant: no radial return -----------------------------------
    bad = uniaxial_path(RETURN_MAP_WRONG)
    bad_ratio = secant(bad, 17, 19) / secant(bad, 1, 2)
    worst_q = max(p[3] for p in bad)
    yield_now = SIG_Y + H_MOD * bad[-1][4]
    print(f"elastic_only_post_yield_slope_equals_E={abs(bad_ratio - 1.0) < 1e-9}")
    print(f"elastic_only_alpha_stays_zero={bad[-1][4] == 0.0}")
    print(f"elastic_only_overshoots_yield_by_gt_3x={worst_q > 3.0 * yield_now}")
    if not (abs(bad_ratio - 1.0) < 1e-9 and worst_q > 3.0 * yield_now):
        print(f"FAIL: skipping the return map no longer overshoots the yield "
              f"surface (ratio={bad_ratio:.6f}, q={worst_q:.3f}, "
              f"limit={yield_now:.3f})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
