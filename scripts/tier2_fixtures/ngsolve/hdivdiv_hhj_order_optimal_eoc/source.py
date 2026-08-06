"""Tier-2: HHJ is order-optimal — order k gives h^(k+1) in L2 for the deflection.

Claim: ngsolve hdivdiv#3 — an h-refinement study at fixed order=1 must show the
L2 error of the deflection scaling as h^2; if it scales as h the moment space is
mis-typed and an order is lost.

MMS (exact simply-supported solution, so no truncated series is needed):
    w_ex = sin(pi x) sin(pi y)      on the unit square
    q    = D * 4 pi^4 * w_ex        since Laplace^2 w_ex = 4 pi^4 w_ex
w_ex and its full Hessian vanish on the boundary, so w = 0 and M_nn = 0 hold
exactly.

Wrong variant: swap the HDivDiv moment space for the fully C0-continuous
MatrixValued(H1, symmetric=True) tensor space and repeat the same study.

Observed on NGSolve 6.2.2604 (2026-08-03):
  * HDivDiv order=1: L2 err 3.90e-02 -> 8.53e-03 for maxh 0.2 -> 0.1, EOC 2.19.
  * HDivDiv order=2: L2 err 1.83e-03 -> 1.76e-04, EOC 3.38 (= k+1).
  * MatrixValued(H1): L2 err 1.24e+14 -> 9.70e+15 -> 1.29e+26; the error GROWS
    under refinement. So the mis-typed space does not merely "lose an order",
    it loses inf-sup stability outright.
"""
from __future__ import annotations

import math
import sys

import ngsolve as ngs
from netgen.geom2d import unit_square

# WRONG: the mis-typed moment space used for the failing study
MISTYPED_MOMENT_SPACE = "matrixh1"

PI = math.pi
E, NU, T_PLATE = 1.0, 0.3, 1.0
D = E * T_PLATE ** 3 / (12.0 * (1.0 - NU ** 2))
BND = "bottom|right|top|left"
W_EX = ngs.sin(PI * ngs.x) * ngs.sin(PI * ngs.y)
Q_MMS = D * 4.0 * PI ** 4 * W_EX


def hhj_l2_error(maxh: float, order: int, moment: str) -> float:
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=maxh))
    if moment == "hdivdiv":
        Vm = ngs.HDivDiv(mesh, order=order - 1, dirichlet=BND)
    elif moment == "matrixh1":
        Vm = ngs.MatrixValued(ngs.H1(mesh, order=max(order - 1, 1)),
                              symmetric=True)
    else:
        raise ValueError(moment)
    Vw = ngs.H1(mesh, order=order, dirichlet=BND)
    X = Vm * Vw
    (sig, ww), (tau, vv) = X.TnT()
    n = ngs.specialcf.normal(2)

    def compliance(s):
        return (1.0 / (D * (1.0 - NU))) \
            * (s - NU / (1.0 + NU) * ngs.Trace(s) * ngs.Id(2))

    a = ngs.BilinearForm(X, symmetric=True)
    a += ngs.InnerProduct(compliance(sig), tau) * ngs.dx
    a += ngs.InnerProduct(tau, ww.Operator("hesse")) * ngs.dx
    a += ngs.InnerProduct(sig, vv.Operator("hesse")) * ngs.dx
    a += -(tau * n * n) * (ngs.Grad(ww) * n) * ngs.dx(element_boundary=True)
    a += -(sig * n * n) * (ngs.Grad(vv) * n) * ngs.dx(element_boundary=True)
    a.Assemble()
    f = ngs.LinearForm(X)
    f += -Q_MMS * vv * ngs.dx
    f.Assemble()
    gf = ngs.GridFunction(X)
    gf.vec.data = a.mat.Inverse(X.FreeDofs(), inverse="umfpack") * f.vec
    gw = gf.components[1]
    return math.sqrt(abs(ngs.Integrate((gw - W_EX) ** 2, mesh)))


def study(order: int, moment: str) -> tuple[list[float], float]:
    errs = [hhj_l2_error(h, order, moment) for h in (0.2, 0.1)]
    eoc = (math.log(errs[0] / errs[1]) / math.log(2.0)
           if errs[1] > 0 else float("inf"))
    return errs, eoc


def main() -> int:
    ok = True
    print(f"ngsolve_version={ngs.__version__}")

    _mm = ngs.Mesh(unit_square.GenerateMesh(maxh=0.5))
    print(f"moment_space_type={ngs.HDivDiv(_mm, order=1).type} "
          f"deflection_space_type={ngs.H1(_mm, order=2).type}")

    e1, eoc1 = study(1, "hdivdiv")
    print(f"hdivdiv_order1_l2={e1[0]:.5e},{e1[1]:.5e} eoc={eoc1:.3f}")
    print(f"hdivdiv_order1_eoc_in_1p5_2p5={1.5 <= eoc1 <= 2.5}")
    if not 1.5 <= eoc1 <= 2.5:
        print(f"FAIL: order=1 HHJ deflection EOC {eoc1:.3f} is not h^2",
              file=sys.stderr)
        ok = False

    e2, eoc2 = study(2, "hdivdiv")
    print(f"hdivdiv_order2_l2={e2[0]:.5e},{e2[1]:.5e} eoc={eoc2:.3f}")
    print(f"hdivdiv_order2_eoc_ge_2p5={eoc2 >= 2.5}")
    print(f"hdivdiv_order2_beats_order1={e2[1] < e1[1]}")
    if eoc2 < 2.5:
        print(f"FAIL: order=2 HHJ deflection EOC {eoc2:.3f} is below k+1=3",
              file=sys.stderr)
        ok = False

    # --- WRONG variant: mis-typed moment space ----------------------------
    ew, eocw = study(2, MISTYPED_MOMENT_SPACE)
    print(f"mistyped_{MISTYPED_MOMENT_SPACE}_l2={ew[0]:.5e},{ew[1]:.5e} "
          f"eoc={eocw:.3f}")
    print(f"mistyped_moment_space_l2_gt_1e6={min(ew) > 1e6}")
    print(f"mistyped_moment_space_error_grows_under_refinement={ew[1] > ew[0]}")
    print(f"mistyped_moment_space_second_order={1.5 <= eocw <= 2.5}")
    if min(ew) <= 1e6:
        print(f"FAIL: mis-typed moment space stayed accurate "
              f"(L2 {ew[0]:.3e}, {ew[1]:.3e})", file=sys.stderr)
        ok = False
    if not ew[1] > ew[0]:
        print("FAIL: mis-typed moment space did not diverge under refinement",
              file=sys.stderr)
        ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
