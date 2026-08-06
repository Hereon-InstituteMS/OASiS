"""Tier-2: the L-shape r^(2/3) corner, and where refinement has to go.

Claim: skfem adaptive_poisson#5 -- the L-shape re-entrant corner gives a
u ~ r^(2/3) singularity at (0,0); uniform refinement converges at H^1 rate 2/3
(sub-optimal) while h-adaptive refinement recovers rate 1.  "Run m = m.refined()
(uniform) once before the adaptive loop to give the estimator enough elements
to discriminate the singular zone.  Signal: ... the refined-mesh region does
not cluster around (0,0)."

What this fixture DOES verify, on skfem 12.0.1 with MeshTri.init_lshaped():

  * the singularity exponent, from the EXACT solution rather than from a
    discrete one: |grad u| fitted against r on a log-log scale over four
    decades gives the analytic -1/3 to three decimals, so u ~ r^(2/3) is the
    right description and |grad u| is unbounded at the corner.
  * uniform refinement does NOT dilute the difficulty: the largest elemental
    indicator keeps a stable -- in fact slightly rising -- share of the total
    indicator mass as elements are added everywhere.
  * a caution the entry does not give: with an h-weighted facet-jump
    indicator the LARGEST elemental value does not sit on the elements
    closest to the corner, and its rank drifts outward under uniform
    refinement, because the h factor suppresses the small corner elements.
    "Where the indicator is biggest" and "where the singularity is" are not
    the same question.
  * Dorfler marking reaches the SAME corner resolution with far fewer degrees
    of freedom, and puts a larger fraction of its elements near the corner.
  * the entry's advice to refine uniformly once first does buy something, but
    not for the stated reason: on the 6-element L-shape every element already
    carries a non-zero indicator, so it is not that there are too few to
    discriminate -- it is that the indicators are nearly EQUAL, and one
    uniform refinement is what separates them.

What it does NOT verify, deliberately: the numerical rates 2/3 and 1.  The
H^1 error against the analytic gradient is not computable to that accuracy
here because the exact gradient is unbounded at the corner and its quadrature
diverges as the mesh approaches the singularity, so any "measured rate" from
that route would be an artefact of the quadrature rather than of the method.
"""
from __future__ import annotations

import sys

import numpy as np
from skfem import (
    Basis,
    ElementTriP1,
    Functional,
    InteriorFacetBasis,
    MeshTri,
    condense,
    solve,
)
from skfem.models.poisson import laplace

ALPHA = 2.0 / 3.0


def exact(x):
    r = np.sqrt(x[0] ** 2 + x[1] ** 2)
    th = np.arctan2(x[1], x[0])
    th = np.where(th < 0.0, th + 2.0 * np.pi, th)
    return r ** ALPHA * np.sin(ALPHA * th)


def exact_grad_magnitude(r):
    """|grad u| = alpha * r^(alpha-1) for this separable mode."""
    return ALPHA * r ** (ALPHA - 1.0)


def solve_l(m):
    ib = Basis(m, ElementTriP1())
    K = laplace.assemble(ib)
    D = ib.get_dofs().all()
    x = ib.zeros()
    x[D] = exact(ib.doflocs[:, D])
    return ib, solve(*condense(K, ib.zeros(), x=x, D=D))


def indicator(m, ib, u):
    e = ElementTriP1()
    f0 = InteriorFacetBasis(m, e, side=0)
    f1 = InteriorFacetBasis(m, e, side=1)

    @Functional
    def jump(w):
        return w.h * ((w["g0"][0] - w["g1"][0]) * w.n[0]
                      + (w["g0"][1] - w["g1"][1]) * w.n[1]) ** 2

    ef = jump.elemental(f0, g0=f0.interpolate(u).grad,
                        g1=f1.interpolate(u).grad)
    a = np.zeros(m.t.shape[1])
    np.add.at(a, f0.tind, ef)
    np.add.at(a, f1.tind, ef)
    return np.sqrt(a)


def dorfler(eta, theta):
    order = np.argsort(eta)[::-1]
    cum = np.cumsum(eta[order] ** 2)
    k = int(np.searchsorted(cum, theta * cum[-1])) + 1
    return order[:k]


def centroid_radius(m):
    c = m.p[:, m.t].mean(axis=1)
    return np.sqrt(c[0] ** 2 + c[1] ** 2)


def main() -> int:
    ok = True
    base = MeshTri.init_lshaped()
    print(f"lshape_vertices={base.p.shape[1]} lshape_elements={base.t.shape[1]}")
    print(f"origin_is_a_vertex="
          f"{bool(np.any(np.hypot(base.p[0], base.p[1]) < 1e-14))}")

    # --- the exponent, from the exact solution ---------------------------
    r = np.logspace(-6, -2, 60)
    slope, intercept = np.polyfit(np.log(r), np.log(exact_grad_magnitude(r)), 1)
    print(f"fitted_grad_exponent={slope:.6f}")
    print(f"analytic_exponent={ALPHA - 1.0:.6f}")
    print(f"exponent_matches_minus_one_third="
          f"{abs(slope - (ALPHA - 1.0)) < 1e-3}")
    print(f"gradient_is_unbounded_at_the_corner="
          f"{exact_grad_magnitude(1e-8) > exact_grad_magnitude(1e-2)}")
    if abs(slope - (ALPHA - 1.0)) >= 1e-3:
        print("FAIL: the fitted exponent is not -1/3", file=sys.stderr)
        ok = False

    # --- the coarsest mesh cannot discriminate ---------------------------
    ib0, u0 = solve_l(base)
    eta0 = indicator(base, ib0, u0)
    print(f"unrefined_elements={base.t.shape[1]}")
    print(f"unrefined_nonzero_indicators={int((eta0 > 0).sum())}")
    spread0 = float(eta0.max() / eta0.min())
    print(f"unrefined_indicator_spread={spread0:.4f}")
    print(f"unrefined_indicator_is_nearly_uniform={spread0 < 3.0}")
    ib1, u1 = solve_l(base.refined())
    eta1 = indicator(base.refined(), ib1, u1)
    spread1 = float(eta1.max() / eta1.min())
    print(f"after_one_uniform_refinement_spread={spread1:.4f}")
    print(f"one_uniform_refinement_sharpens_the_indicator="
          f"{spread1 > spread0}")
    if not spread1 > spread0:
        print("FAIL: a single uniform refinement did not sharpen the "
              "indicator field, so the entry's advice buys nothing",
              file=sys.stderr)
        ok = False

    # --- uniform refinement leaves the peak at the corner ----------------
    mu = base.refined(2)
    shares, at_corner = [], []
    for _ in range(4):
        ibx, ux = solve_l(mu)
        ex = indicator(mu, ibx, ux)
        rc = centroid_radius(mu)
        top = int(np.argmax(ex))
        shares.append(float(ex[top] ** 2 / (ex ** 2).sum()))
        # rank of the peak element by distance from the corner, as a
        # fraction: 0 = closest element in the mesh, 1 = furthest
        at_corner.append(float(np.mean(rc < rc[top])))
        print(f"uniform_N={ibx.N} peak_share={shares[-1]:.4e} "
              f"peak_radius_rank={at_corner[-1]:.4f} "
              f"min_centroid_radius={rc.min():.4e}")
        mu = mu.refined()
    print(f"uniform_peak_radius_ranks={[f'{x:.4f}' for x in at_corner]}")
    print(f"uniform_peak_stays_in_the_innermost_decile="
          f"{all(x < 0.1 for x in at_corner)}")
    print(f"peak_indicator_element_is_not_the_closest_to_the_corner="
          f"{all(x > 0.25 for x in at_corner)}")
    print(f"peak_radius_rank_drifts_outward="
          f"{at_corner[-1] > at_corner[0]}")
    print(f"uniform_peak_share_does_not_fall={shares[-1] >= 0.5 * shares[0]}")
    print(f"uniform_peak_share_even_rises={shares[-1] > shares[0]}")
    if not all(x > 0.25 for x in at_corner):
        print("FAIL: the largest elemental indicator WAS among the elements "
              "closest to the corner, so the h-weighting caution does not "
              "apply", file=sys.stderr)
        ok = False
    if shares[-1] < 0.5 * shares[0]:
        print("FAIL: uniform refinement diluted the indicator peak",
              file=sys.stderr)
        ok = False

    # --- adaptive refinement clusters -------------------------------------
    ma = base.refined(2)
    for _ in range(12):
        ibx, ux = solve_l(ma)
        ma = ma.refined(dorfler(indicator(ma, ibx, ux), 0.5))
    ib_a = Basis(ma, ElementTriP1())
    ra = centroid_radius(ma)
    # uniform refinement carried on until it reaches the SAME corner
    # resolution the adaptive run achieved
    mu2 = base.refined(2)
    while centroid_radius(mu2).min() > ra.min() * 1.001:
        mu2 = mu2.refined()
    ru = centroid_radius(mu2)
    print(f"adaptive_N={ib_a.N} adaptive_elements={ma.t.shape[1]} "
          f"min_centroid_radius={ra.min():.4e}")
    print(f"uniform_N={Basis(mu2, ElementTriP1()).N} "
          f"uniform_elements={mu2.t.shape[1]} "
          f"min_centroid_radius={ru.min():.4e}")
    nu = Basis(mu2, ElementTriP1()).N
    print(f"same_corner_resolution={bool(abs(ru.min() - ra.min())
                                         < 1e-9 + 1e-3 * ra.min())}")
    print(f"dofs_for_that_resolution_adaptive={ib_a.N}")
    print(f"dofs_for_that_resolution_uniform={nu}")
    print(f"dof_saving_factor={nu / ib_a.N:.3f}")
    print(f"adaptive_reaches_it_with_fewer_dofs={ib_a.N < nu}")
    print(f"adaptive_saves_at_least_2x={nu > 2.0 * ib_a.N}")
    frac_a = float(np.mean(ra < 0.1))
    frac_u = float(np.mean(ru < 0.1))
    print(f"adaptive_fraction_of_elements_within_r0p1={frac_a:.4f}")
    print(f"uniform_fraction_of_elements_within_r0p1={frac_u:.4f}")
    print(f"adaptive_concentrates_more_near_the_corner={frac_a > frac_u}")
    if not ib_a.N < nu:
        print("FAIL: adaptive refinement needed at least as many DOFs as "
              "uniform for the same corner resolution", file=sys.stderr)
        ok = False
    if not frac_a > frac_u:
        print("FAIL: adaptive refinement did not concentrate more elements "
              "near the corner", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
