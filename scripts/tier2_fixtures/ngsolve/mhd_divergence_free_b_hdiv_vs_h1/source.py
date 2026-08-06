"""Tier-2: a VectorH1 magnetic field really does carry O(0.1) divergence, HDiv
with the constraint drives it orders lower, and the grad-div penalty at the
tau ~ 1 the claim recommends barely helps while costing an order of magnitude
in accuracy.

Claim: ngsolve mhd#5 -- "Divergence-free B constraint: div(B) = 0 must be
enforced via grad-div penalty OR HDiv elements.  Signal: a VectorH1 B field on a
non-trivial geometry develops max|div(B)| ~ O(0.1) over time (should be ~1e-14);
this drives spurious monopole currents and the kinetic energy balance drifts.
HDiv elements enforce div(B) = 0 pointwise; grad-div penalty tau*(div(B),
div(C)) with tau ~ 1 is the alternative for H1."

Wrong variant: representing B in VectorH1 with no divergence control.

Setup: a target field B = (sin(pi y) cos(pi x), -cos(pi y) sin(pi x)), which is
divergence-free in closed form, is represented three ways on the same mesh: the
plain VectorH1 L2 projection, the same with a grad-div penalty at several tau,
and an HDiv field with div(B) = 0 imposed through an L2 Lagrange multiplier.
Divergence is measured as ||div B||_L2 and accuracy as ||B - B_exact||_L2, so
the two are on the same footing.

CORRECTION this fixture records.  The "tau ~ 1" recommendation is too weak to
be useful and is not free.  At tau = 1 the divergence falls by only about a
factor of five while the field error grows by an order of magnitude; the tau
that actually controls divergence costs more than an order of magnitude of
accuracy.  The trade-off is the point, and the claim states only one side of it.

What this fixture pins, all re-measured on this run:
  * the exact field's divergence is zero to roundoff, so the divergence measured
    afterwards is entirely the discretisation's;
  * the plain VectorH1 representation carries ||div B|| at the claim's O(0.1)
    level, not the 1e-14 the claim says it should be;
  * HDiv with the constraint is orders of magnitude lower on the same mesh;
  * the grad-div penalty reduces divergence monotonically in tau -- and the
    field error grows monotonically with it;
  * at tau = 1 specifically, the divergence reduction is under a factor of ten
    while the error has already grown by more than a factor of five.
"""
from __future__ import annotations

import sys

from netgen.geom2d import unit_square
from ngsolve import (
    BilinearForm,
    BitArray,
    CoefficientFunction,
    GridFunction,
    HDiv,
    InnerProduct,
    Integrate,
    L2,
    LinearForm,
    Mesh,
    VectorH1,
    cos,
    div,
    dx,
    pi,
    sin,
    sqrt,
    x,
    y,
)

MAXH = 0.15


def bexact():
    return CoefficientFunction((sin(pi * y) * cos(pi * x),
                                -cos(pi * y) * sin(pi * x)))


def measure(mesh, Bh, Bex):
    d = float(sqrt(Integrate(div(Bh) ** 2, mesh)))
    e = float(sqrt(Integrate(InnerProduct(Bh - Bex, Bh - Bex), mesh)))
    return d, e


def vector_h1(mesh, Bex, tau):
    fes = VectorH1(mesh, order=1)
    B, C = fes.TnT()
    a = BilinearForm(fes)
    a += (InnerProduct(B, C) + tau * div(B) * div(C)) * dx
    f = LinearForm(fes)
    f += InnerProduct(Bex, C) * dx
    a.Assemble()
    f.Assemble()
    gf = GridFunction(fes)
    gf.vec.data = a.mat.Inverse(inverse="umfpack") * f.vec
    return fes.ndof, gf


def hdiv_constrained(mesh, Bex, order=3):
    S = HDiv(mesh, order=order)
    L = L2(mesh, order=order - 1)
    X = S * L
    (B, lam), (C, mu) = X.TnT()
    a = BilinearForm(X)
    a += (InnerProduct(B, C) + div(C) * lam + div(B) * mu) * dx
    f = LinearForm(X)
    f += InnerProduct(Bex, C) * dx
    a.Assemble()
    f.Assemble()
    fr = BitArray(X.FreeDofs())
    fr[X.Range(1).start] = False               # pin the multiplier's constant
    gf = GridFunction(X)
    gf.vec.data = a.mat.Inverse(fr, inverse="umfpack") * f.vec
    return X.ndof, gf.components[0]


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=MAXH))
    Bex = bexact()
    d_ex = float(sqrt(Integrate(
        (Bex[0].Diff(x) + Bex[1].Diff(y)) ** 2, mesh)))
    print(f"exact_field_divergence={d_ex:.3e}")
    print(f"exact_field_is_divergence_free={d_ex < 1e-12}")

    nd0, gf0 = vector_h1(mesh, Bex, 0.0)
    d0, e0 = measure(mesh, gf0, Bex)
    print(f"vectorh1_ndof={nd0} div_L2={d0:.6e} field_err={e0:.6e}")
    print(f"vectorh1_divergence_is_order_0p1={0.01 < d0 < 3.0}")
    print(f"vectorh1_divergence_is_not_1e-14={d0 > 1e-10}")

    ndh, gfh = hdiv_constrained(mesh, Bex)
    dh, eh = measure(mesh, gfh, Bex)
    print(f"hdiv_ndof={ndh} div_L2={dh:.6e} field_err={eh:.6e}")
    print(f"hdiv_divergence_orders_below_vectorh1={dh < d0 / 1e4}")
    print(f"hdiv_also_more_accurate={eh < e0}")

    taus = (1.0, 1e2, 1e4, 1e6)
    rows = []
    for tau in taus:
        _, gft = vector_h1(mesh, Bex, tau)
        dt, et = measure(mesh, gft, Bex)
        rows.append((tau, dt, et))
        print(f"graddiv tau={tau:g} div_L2={dt:.6e} field_err={et:.6e} "
              f"div_reduction={d0 / dt:.3f} err_growth={et / e0:.3f}")

    div_mono = all(b[1] < a[1] for a, b in zip(rows, rows[1:]))
    err_mono = all(b[2] > a[2] for a, b in zip(rows, rows[1:]))
    print(f"graddiv_divergence_falls_monotonically_in_tau={div_mono}")
    print(f"graddiv_field_error_grows_monotonically_in_tau={err_mono}")

    tau1 = rows[0]
    red1 = d0 / tau1[1]
    grow1 = tau1[2] / e0
    print(f"tau_one_divergence_reduction={red1:.3f}")
    print(f"tau_one_error_growth={grow1:.3f}")
    print(f"tau_one_barely_reduces_divergence={red1 < 10.0}")
    print(f"tau_one_already_costs_accuracy={grow1 > 5.0}")
    print(f"graddiv_at_tau_one_is_a_poor_trade={red1 < 10.0 and grow1 > 5.0}")

    ok = (
        d_ex < 1e-12
        and 0.01 < d0 < 3.0
        and dh < d0 / 1e4
        and div_mono and err_mono
        and red1 < 10.0 and grow1 > 5.0
    )
    if ok:
        return 0
    print("FAIL: divergence-free B invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
