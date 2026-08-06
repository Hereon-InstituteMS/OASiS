"""Tier-2: the HHJ moment space must be HDivDiv, not a VectorH1/H1 tensor space.

Claim: ngsolve hdivdiv#0 — HDivDiv enforces normal-normal continuity across
facets, WEAKER than full H2 conformity; a code "claiming to be the HHJ moment
space but using VectorH1 instead of HDivDiv" is non-conforming and the
deflection is wrong.

Wrong variant: two of them.
  (a) literal VectorH1(mesh, order=k) as the moment space — NGSolve refuses to
      build the compliance form at all: NgException 'Trace of non-matrix called'.
  (b) the closest thing that DOES build: MatrixValued(H1(...), symmetric=True),
      i.e. a fully C0-continuous symmetric tensor. It assembles and solves, and
      the deflection blows up past 1e6.

Observed on NGSolve 6.2.2604 (2026-08-03):
  * HDivDiv:     relative nn facet jump ~1e-17 (exact), relative tt jump ~2.3.
  * MatrixValued(H1): BOTH jumps are exactly 0 — it is strictly MORE continuous
    than HDivDiv, so the claim's "allows tangent-tangent jumps" has the sign
    backwards; the real failure is loss of inf-sup stability.
  * MatrixValued(H1) HHJ deflection ~1.8e14 at maxh=0.4 and it DIVERGES under
    refinement — so "off by a constant factor (~1.1-1.5)" is falsified too.
  * HDivDiv + M_nn Dirichlet reproduces the Navier simply-supported value
    0.00406235 q L^4 / D to 4 digits.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import ngsolve as ngs
from netgen.geom2d import unit_square

# which space plays the role of the moment tensor in the WRONG solve
WRONG_MOMENT_SPACE = "matrixh1"

E, NU, T_PLATE, Q = 1.0, 0.3, 1.0, 1.0
D = E * T_PLATE ** 3 / (12.0 * (1.0 - NU ** 2))
NAVIER_COEFF = 0.004062352659370942          # Navier series, SS square plate
W_REF = NAVIER_COEFF * Q / D
BND = "bottom|right|top|left"


def facet_jumps(V: ngs.FESpace, seed: int = 7) -> tuple[float, float]:
    """Relative squared nn / tt facet jump of a fixed pseudo-random member of V.

    A structural property of the SPACE: if every member has zero nn jump the
    space is normal-normal conforming.
    """
    gf = ngs.GridFunction(V)
    gf.vec.FV().NumPy()[:] = np.random.default_rng(seed).standard_normal(V.ndof)
    u, v = V.TnT()
    n = ngs.specialcf.normal(2)
    tg = ngs.specialcf.tangential(2)

    def nn(w):
        return ngs.InnerProduct(w * n, n)

    def tt(w):
        return ngs.InnerProduct(w * tg, tg)

    def quad(f, jump: bool) -> float:
        b = ngs.BilinearForm(V, symmetric=True)
        if jump:
            b += (f(u) - f(u.Other())) * (f(v) - f(v.Other())) \
                * ngs.dx(skeleton=True)
        else:
            b += f(u) * f(v) * ngs.dx(skeleton=True)
        b.Assemble()
        return float(ngs.InnerProduct(gf.vec, b.mat * gf.vec))

    return (abs(quad(nn, True)) / abs(quad(nn, False)),
            abs(quad(tt, True)) / abs(quad(tt, False)))


def hhj_centre_deflection(maxh: float, order: int, moment: str) -> float:
    """HHJ plate solve; `moment` selects the moment-tensor space."""
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=maxh))
    if moment == "hdivdiv":
        Vm = ngs.HDivDiv(mesh, order=order - 1, dirichlet=BND)
    elif moment == "matrixh1":
        Vm = ngs.MatrixValued(ngs.H1(mesh, order=order - 1), symmetric=True)
    elif moment == "vectorh1":
        Vm = ngs.VectorH1(mesh, order=order - 1)
    else:
        raise ValueError(moment)
    Vw = ngs.H1(mesh, order=order, dirichlet=BND)
    X = Vm * Vw
    (sig, ww), (tau, vv) = X.TnT()
    n = ngs.specialcf.normal(2)

    def compliance(s):
        # 1/(D(1-nu)) * (s - nu/(1+nu) tr(s) I) -- inverse plate stiffness
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
    f += Q * vv * ngs.dx
    f.Assemble()
    gf = ngs.GridFunction(X)
    gf.vec.data = a.mat.Inverse(X.FreeDofs(), inverse="umfpack") * f.vec
    return abs(gf.components[1](mesh(0.5, 0.5)))


def main() -> int:
    ok = True
    print(f"ngsolve_version={ngs.__version__}")

    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.4))
    hdd_nn, hdd_tt = facet_jumps(ngs.HDivDiv(mesh, order=1, dgjumps=True))
    mh1_nn, mh1_tt = facet_jumps(
        ngs.MatrixValued(ngs.H1(mesh, order=1, dgjumps=True), symmetric=True))
    print(f"hdivdiv_rel_nn_jump={hdd_nn:.3e}  hdivdiv_rel_tt_jump={hdd_tt:.3e}")
    print(f"matrixh1_rel_nn_jump={mh1_nn:.3e}  matrixh1_rel_tt_jump={mh1_tt:.3e}")
    print(f"hdivdiv_rel_nn_jump_lt_1e-10={hdd_nn < 1e-10}")
    print(f"hdivdiv_rel_tt_jump_gt_0p1={hdd_tt > 0.1}")
    print(f"matrixh1_moment_space_fully_continuous="
          f"{mh1_nn < 1e-10 and mh1_tt < 1e-10}")
    if not (hdd_nn < 1e-10 and hdd_tt > 0.1):
        print("FAIL: HDivDiv is not nn-conforming-with-tt-jumps as claimed",
              file=sys.stderr)
        ok = False
    if not (mh1_nn < 1e-10 and mh1_tt < 1e-10):
        print("FAIL: MatrixValued(H1) is not fully continuous", file=sys.stderr)
        ok = False

    # --- WRONG variant (a): literal VectorH1 moment space -----------------
    msg = ""
    try:
        hhj_centre_deflection(0.4, 2, "vectorh1")
    except Exception as exc:                       # noqa: BLE001
        msg = str(exc)
    print(f"vectorh1_moment_space_raises={bool(msg)}")
    print(f"vectorh1_msg={msg.strip()!r}")
    if "Trace of non-matrix called" not in msg:
        print("FAIL: VectorH1 moment space did not raise the documented "
              f"NgException; got {msg!r}", file=sys.stderr)
        ok = False

    # --- WRONG variant (b): fully continuous tensor moment space ----------
    wc_wrong = hhj_centre_deflection(0.4, 2, WRONG_MOMENT_SPACE)
    print(f"wrong_moment_space={WRONG_MOMENT_SPACE} wrong_centre_deflection="
          f"{wc_wrong:.6e}")
    print(f"wrong_moment_space_wc_gt_1e6={wc_wrong > 1e6}")
    if not wc_wrong > 1e6:
        print(f"FAIL: mis-typed moment space stayed bounded "
              f"(w_centre={wc_wrong:.6e}); the non-conforming pair did not "
              f"exhibit the pathology", file=sys.stderr)
        ok = False
    # the claim's "off by a constant factor 1.1-1.5" band
    print(f"wrong_over_reference_in_1p1_1p5_band="
          f"{1.1 <= wc_wrong / W_REF <= 1.5}")

    # --- RIGHT variant: HDivDiv moments -----------------------------------
    wc_ok = hhj_centre_deflection(0.15, 2, "hdivdiv")
    rel = abs(wc_ok - W_REF) / W_REF
    print(f"hdivdiv_centre_deflection={wc_ok:.6e} navier_reference={W_REF:.6e} "
          f"rel_dev={rel:.3e}")
    print(f"hdivdiv_hhj_matches_navier_within_2pct={rel < 0.02}")
    if rel >= 0.02:
        print(f"FAIL: HDivDiv HHJ missed the Navier reference by {rel:.3%}",
              file=sys.stderr)
        ok = False
    if not math.isfinite(wc_ok):
        print("FAIL: HDivDiv HHJ deflection is not finite", file=sys.stderr)
        ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
