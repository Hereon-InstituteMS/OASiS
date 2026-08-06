"""Tier-2: the rotational convection form has exactly zero energy production --
but the 0.5 the claim puts in front of it makes it a different momentum
equation, not a variant of the same one.

Claims covered: ngsolve time_dependent_ns#1 and navier_stokes#1 -- "Convection
form: Grad(u)*u (non-conservative) vs 0.5*(Grad(u)*u - Grad(u)^T*u) (skew-sym).
Signal: long-time kinetic_energy in a closed periodic box drifts (grows or
decays) with the non_conservative BilinearForm by O(1%) over 1000 steps; the
skew_symmetric variant on the GridFunction velocity preserves it to machine
precision because it makes the convective operator anti-symmetric."

Wrong variant: the non-conservative form, whose energy production is not zero.

TWO CORRECTIONS this fixture records.

  (a) The 0.5.  Grad(u)*u - Grad(u)^T*u is the rotational (Lamb) form, and it
      differs from Grad(u)*u by exactly Grad(u)^T*u = grad(|u|^2/2) -- a
      gradient, which is why it can be absorbed into the pressure and gives the
      SAME momentum equation.  Halve it and the leftover is no longer a
      gradient, so it cannot be absorbed and the equation being solved is not
      Navier-Stokes.  The fixture measures this: the weak curl of the
      difference, which must vanish for a gradient.
  (b) "Machine precision".  The SPATIAL energy production of the rotational
      form is machine zero, but a time-marched run does not conserve energy to
      machine precision -- the residual drift comes from the time
      discretisation.  Over 400 explicit steps the drift is small but far above
      roundoff.  What is true is the RATIO: the non-conservative form drifts by
      two orders of magnitude more.

What this fixture pins, all re-measured on this run:
  * on a closed box with a divergence-free-by-construction field, the energy
    production integral of the rotational form is zero to roundoff while the
    non-conservative form's is not -- and the non-conservative one is not
    merely small, its pointwise L2 density is O(1) against the convection term
    itself;
  * the same holds with the claim's 0.5 in front, so the 0.5 does not break the
    energy property -- only the equation;
  * the weak curl of (Grad(u)*u minus the full rotational form) is at the level
    of the discretisation error, and of (Grad(u)*u minus the halved form) is
    two orders larger -- so the first difference is a gradient and the second
    is not;
  * marched forward at a step both survive, the non-conservative form's kinetic
    energy drifts by more than a hundred times the rotational form's.
"""
from __future__ import annotations

import sys

import numpy
from netgen.geom2d import SplineGeometry
from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    GridFunction,
    Grad,
    H1,
    InnerProduct,
    Integrate,
    LinearForm,
    Mesh,
    VectorH1,
    div,
    dx,
    grad,
    pi,
    sin,
    sqrt,
    x,
    y,
)

MAXH = 0.15
NSTEPS = 400
DT = 1e-4


def _mesh():
    g = SplineGeometry()
    g.AddRectangle((0, 0), (1, 1), bcs=["bot", "right", "top", "left"])
    return Mesh(g.GenerateMesh(maxh=MAXH))


def _initial(space):
    """u = curl(psi) with psi vanishing to second order on the boundary:
    divergence-free by construction and u = 0 on the whole boundary."""
    gf = GridFunction(space)
    psi = sin(pi * x) ** 2 * sin(pi * y) ** 2
    gf.Set(CoefficientFunction((psi.Diff(y), -psi.Diff(x))))
    return gf


def weak_curl_norm(mesh, d):
    """L2 norm of curl(d) taken weakly against H1_0 test functions."""
    S = H1(mesh, order=2, dirichlet=".*")
    phi, psit = S.TnT()
    f = LinearForm(S)
    f += InnerProduct(
        d, CoefficientFunction((grad(psit)[1], -grad(psit)[0]))) * dx
    f.Assemble()
    m = BilinearForm(S)
    m += phi * psit * dx
    m.Assemble()
    g = GridFunction(S)
    g.vec.data = m.mat.Inverse(S.FreeDofs(), inverse="umfpack") * f.vec
    return float(sqrt(Integrate(g * g, mesh)))


def march(mesh, form, nsteps=NSTEPS, dt=DT):
    V = VectorH1(mesh, order=2, dirichlet=".*")
    Q = H1(mesh, order=1)
    X = V * Q
    (u, p), (v, q) = X.TnT()
    stokes = (div(u) * q + div(v) * p) * dx          # nu = 0: closed box
    mstar = BilinearForm(X)
    mstar += InnerProduct(u, v) * dx + dt * stokes
    mstar.Assemble()

    gfu = GridFunction(X)
    gfu.components[0].vec.data = _initial(V).vec
    w = gfu.components[0]
    inv = mstar.mat.Inverse(X.FreeDofs(), "umfpack")
    rhs = gfu.vec.CreateVector()
    ke0 = float(Integrate(InnerProduct(w, w), mesh))
    for _ in range(nsteps):
        c = (Grad(w) * w if form == "nonconservative"
             else Grad(w) * w - Grad(w).trans * w)
        conv = LinearForm(X)
        conv += InnerProduct(c, v) * dx
        conv.Assemble()
        rhs.data = mstar.mat * gfu.vec - dt * conv.vec
        gfu.vec.data = inv * rhs
        cur = float(Integrate(InnerProduct(w, w), mesh))
        if not numpy.isfinite(cur):
            return ke0, cur
    return ke0, float(Integrate(InnerProduct(w, w), mesh))


def main() -> int:
    mesh = _mesh()
    V = VectorH1(mesh, order=3, dirichlet=".*")
    w = _initial(V)

    conv = Grad(w) * w
    rot = Grad(w) * w - Grad(w).trans * w
    half = 0.5 * (Grad(w) * w - Grad(w).trans * w)

    conv_norm = float(sqrt(Integrate(InnerProduct(conv, conv), mesh)))
    print(f"convection_L2_norm={conv_norm:.6f}")

    # --- energy production ---------------------------------------------
    prods = {}
    for name, c in (("nonconservative", conv), ("rotational", rot),
                    ("claim_half", half)):
        integ = float(Integrate(InnerProduct(c, w), mesh))
        dens = float(sqrt(Integrate(InnerProduct(c, w) ** 2, mesh)))
        prods[name] = (integ, dens)
        print(f"{name}_energy_production={integ:+.6e} "
              f"pointwise_L2={dens:.6e}")

    ratio = abs(prods['nonconservative'][0]) / max(
        abs(prods['rotational'][0]), 1e-300)
    print(f"nonconservative_over_rotational_production={ratio:.3e}")
    print(f"nonconservative_produces_energy={ratio > 1e6}")
    print(f"nonconservative_density_is_order_one="
          f"{prods['nonconservative'][1] > 0.01 * conv_norm}")
    print(f"rotational_energy_production_is_roundoff="
          f"{abs(prods['rotational'][0]) < 1e-14}")
    print(f"half_factor_keeps_the_energy_property="
          f"{abs(prods['claim_half'][0]) < 1e-14}")

    # --- is the difference a gradient? ---------------------------------
    c_full = weak_curl_norm(mesh, conv - rot) / conv_norm
    c_half = weak_curl_norm(mesh, conv - half) / conv_norm
    c_ctrl = weak_curl_norm(mesh, conv) / conv_norm
    print(f"weak_curl_conv_minus_rotational={c_full:.6e}")
    print(f"weak_curl_conv_minus_half={c_half:.6e}")
    print(f"weak_curl_of_convection_itself={c_ctrl:.6e}")
    print(f"rotational_difference_is_a_gradient={c_full < 0.05 * c_ctrl}")
    print(f"half_difference_is_not_a_gradient={c_half > 10.0 * c_full}")
    print(f"half_form_solves_a_different_equation={c_half > 10.0 * c_full}")

    # --- marched drift --------------------------------------------------
    ke0_n, ke1_n = march(mesh, "nonconservative")
    ke0_r, ke1_r = march(mesh, "rotational")
    d_n = abs(ke1_n - ke0_n) / ke0_n
    d_r = abs(ke1_r - ke0_r) / ke0_r
    print(f"nonconservative_ke_drift={d_n:.6e}")
    print(f"rotational_ke_drift={d_r:.6e}")
    print(f"both_runs_finite="
          f"{bool(numpy.isfinite(ke1_n) and numpy.isfinite(ke1_r))}")
    print(f"rotational_drift_at_least_100x_smaller={d_n > 100 * d_r}")
    print(f"rotational_drift_is_not_machine_precision={d_r > 1e-12}")

    ok = (
        ratio > 1e6
        and prods["nonconservative"][1] > 0.01 * conv_norm
        and abs(prods["rotational"][0]) < 1e-14
        and abs(prods["claim_half"][0]) < 1e-14
        and c_full < 0.05 * c_ctrl
        and c_half > 10.0 * c_full
        and numpy.isfinite(ke1_n) and numpy.isfinite(ke1_r)
        and d_n > 100 * d_r
        and d_r > 1e-12
    )
    if ok:
        return 0
    print("FAIL: convection-form invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
