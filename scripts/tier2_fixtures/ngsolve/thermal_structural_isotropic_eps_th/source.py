"""Tier-2: thermal strain eps_th = alpha*T*Id(dim) is isotropic; a non-Id tensor is not.

Claim: ngsolve thermal_structural#0 — eps_th = alpha * T * Id(dim) expands
equally in all directions; applying alpha as a non-Id tensor (e.g. only along x)
produces anisotropic expansion in the VectorH1 GridFunction that does NOT match
the expected uniform-temperature stress-free state — a uniformly heated
unconstrained specimen should give zero Stress and uniform Strain in all
directions.

Wrong variant: eps_th = alpha * T * CF((1,0,0,0), dims=(2,2)), i.e. the
"expansion only along x" tensor, in an otherwise identical solve.

Setup: unit_square maxh=0.3, VectorH1(order=2), the three rigid-body modes
removed with a 3-component NumberSpace so the specimen is genuinely
unconstrained; E=210 GPa, nu=0.3, alpha=1.2e-5, dT=100.

Observed on NGSolve 6.2.2604 (2026-08-03):
    Id(2)          -> eps_xx = eps_yy = 1.560000e-03 = (1+nu) alpha dT
    diag(1,0)      -> eps_xx = 2.730000e-03, eps_yy = -1.170000e-03
The heated specimen CONTRACTS transversely under the non-Id tensor, which is the
unmistakable signature. Note that the total stress relaxes to ~0 in BOTH cases:
a spatially uniform eigenstrain on a free body is always compatible, so "zero
Stress" alone does not discriminate — the strain isotropy does.
"""
from __future__ import annotations

import sys

import ngsolve as ngs
from netgen.geom2d import unit_square

# WRONG: the thermal-strain tensor -- "x_only" is alpha*T*diag(1,0)
EPS_TH_TENSOR = "x_only"

E, NU, ALPHA, DT = 210e3, 0.3, 1.2e-5, 100.0
MU = E / (2.0 * (1.0 + NU))
LAM = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
THREE_K = 3.0 * LAM + 2.0 * MU
EXACT = (1.0 + NU) * ALPHA * DT             # 1.560000e-03


def tensor(kind: str):
    if kind == "identity":
        return ngs.Id(2)
    if kind == "x_only":
        return ngs.CoefficientFunction((1, 0, 0, 0), dims=(2, 2))
    raise ValueError(kind)


def free_expansion(kind: str) -> tuple[float, float, float]:
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.3))
    Vu = ngs.VectorH1(mesh, order=2)
    N = ngs.NumberSpace(mesh)
    X = Vu * N * N * N
    (uu, l1, l2, l3), (vv, m1, m2, m3) = X.TnT()
    x, y = ngs.x, ngs.y

    def strain(w):
        return 0.5 * (ngs.Grad(w) + ngs.Grad(w).trans)

    def stress(w):
        return 2 * MU * strain(w) + LAM * ngs.Trace(strain(w)) * ngs.Id(2)

    a = ngs.BilinearForm(X, symmetric=True)
    a += ngs.InnerProduct(stress(uu), strain(vv)) * ngs.dx
    a += (l1 * vv[0] + l2 * vv[1] + l3 * (y * vv[0] - x * vv[1])) * ngs.dx
    a += (m1 * uu[0] + m2 * uu[1] + m3 * (y * uu[0] - x * uu[1])) * ngs.dx
    a.Assemble()
    eps_th = THREE_K * ALPHA * DT * tensor(kind)
    f = ngs.LinearForm(X)
    f += ngs.InnerProduct(eps_th, strain(vv)) * ngs.dx
    f.Assemble()
    gf = ngs.GridFunction(X)
    gf.vec.data = a.mat.Inverse(X.FreeDofs(), inverse="umfpack") * f.vec
    gu = gf.components[0]
    exx = float(ngs.Integrate(ngs.Grad(gu)[0, 0], mesh))
    eyy = float(ngs.Integrate(ngs.Grad(gu)[1, 1], mesh))
    resid = stress(gu) - eps_th
    smax = float(abs(ngs.Integrate(ngs.InnerProduct(resid, resid), mesh))) ** 0.5
    return exx, eyy, smax


def main() -> int:
    ok = True
    print(f"ngsolve_version={ngs.__version__}")
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.3))
    print(f"structural_space_type={ngs.VectorH1(mesh, order=2).type} "
          f"eps_th_dims={list(tensor('identity').dims)}")
    print(f"expected_uniform_strain={EXACT:.6e}")

    # --- WRONG variant: non-Id (x-only) thermal strain --------------------
    exx_b, eyy_b, sig_b = free_expansion(EPS_TH_TENSOR)
    aniso_b = abs(exx_b - eyy_b) / EXACT
    print(f"wrong_tensor={EPS_TH_TENSOR} eps_xx={exx_b:.6e} eps_yy={eyy_b:.6e} "
          f"anisotropy={aniso_b:.4f} residual_stress={sig_b:.3e}")
    print(f"wrong_tensor_expansion_anisotropic={aniso_b > 0.5}")
    print(f"wrong_tensor_contracts_transversely={eyy_b < 0.0}")
    print(f"wrong_tensor_misses_uniform_strain="
          f"{abs(exx_b - EXACT) / EXACT > 0.1 or abs(eyy_b - EXACT) / EXACT > 0.1}")
    print(f"zero_stress_alone_does_not_discriminate={sig_b < 1e-6}")
    if aniso_b <= 0.5:
        print(f"FAIL: the non-Id thermal strain expanded isotropically anyway "
              f"(eps_xx={exx_b:.6e}, eps_yy={eyy_b:.6e})", file=sys.stderr)
        ok = False
    if eyy_b >= 0.0:
        print("FAIL: the heated specimen did not contract transversely under "
              "the x-only thermal strain", file=sys.stderr)
        ok = False

    # --- RIGHT variant: eps_th = alpha * T * Id(2) ------------------------
    exx_g, eyy_g, sig_g = free_expansion("identity")
    aniso_g = abs(exx_g - eyy_g) / EXACT
    rel = abs(exx_g - EXACT) / EXACT
    print(f"identity_tensor eps_xx={exx_g:.6e} eps_yy={eyy_g:.6e} "
          f"anisotropy={aniso_g:.3e} residual_stress={sig_g:.3e}")
    print(f"identity_expansion_isotropic={aniso_g < 1e-6}")
    print(f"identity_matches_expected_uniform_strain={rel < 1e-6}")
    print(f"identity_leaves_body_stress_free={sig_g < 1e-6}")
    print(f"identity_expands_in_both_directions={exx_g > 0.0 and eyy_g > 0.0}")
    if aniso_g >= 1e-6:
        print(f"FAIL: eps_th = alpha*T*Id(2) is not isotropic "
              f"(eps_xx={exx_g:.6e}, eps_yy={eyy_g:.6e})", file=sys.stderr)
        ok = False
    if rel >= 1e-6:
        print(f"FAIL: the isotropic thermal strain missed the expected uniform "
              f"strain by {rel:.3e}", file=sys.stderr)
        ok = False
    if sig_g >= 1e-6:
        print(f"FAIL: the uniformly heated free specimen is not stress-free "
              f"({sig_g:.3e})", file=sys.stderr)
        ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
