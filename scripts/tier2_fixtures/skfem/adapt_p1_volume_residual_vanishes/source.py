"""Tier-2: for P1 the interior residual is just h^2 * int f^2 -- and dropping
it does not change the estimator's ORDER.

Claim: skfem adaptive_poisson#0 -- for P1 elements the volume residual
Laplace(u_h) vanishes inside elements, so the eta_K^2 volume term is just
h_K^2 int_K f^2 dx; for P2 and higher the second-derivative residual must be
reinstated.  "Signal: error estimator converges with order h instead of h^2 on
smooth solutions; refinement stalls at a non-zero plateau because the dominant
residual term is missing."

Measured on skfem 12.0.1 on a SMOOTH problem (-Laplace u = 2 pi^2 sin(pi x)
sin(pi y) on the unit square, exact u = sin(pi x) sin(pi y)),
MeshTri.init_tensor at nx = 8, 16, 32, 64, ElementTriP1, with the true H^1
error computed against the analytic gradient:

  * THE P1 FACT IS EXACT.  The interpolated gradient is identical at every
    quadrature point of an element, so u_h is affine there and its interior
    Laplacian is identically zero; on ElementTriP2 the same gradient VARIES
    within an element, which is exactly why the second-derivative term has to
    come back.
  * THE SIGNAL IS FALSE.  Dropping the h^2 int f^2 term does NOT change the
    estimator's convergence order and does NOT produce a plateau: both the
    full estimator and the jump-only one fall at the same rate as the H^1
    error, and their effectivity indices are both stable under refinement.
    What changes is the CONSTANT -- the effectivity drops by about a tenth --
    so an order-based gate cannot see this at all.

Mutation control: T2_MUTATE=1 reinstates the documented fix at the pathology
site -- the "jump-only" estimator is assembled with the h^2 int f^2 volume
term switched back on -- so the two estimators coincide and the effectivity
shortfall vanishes.  Re-run with T2_MUTATE=1 python source.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from skfem import (
    Basis,
    ElementTriP1,
    ElementTriP2,
    Functional,
    InteriorFacetBasis,
    LinearForm,
    MeshTri,
    condense,
    solve,
)
from skfem.models.poisson import laplace

MUTATE = os.environ.get("T2_MUTATE") == "1"
# The pathology: the second estimator drops the h^2 int f^2 volume term.
# T2_MUTATE=1 puts it back -- the documented fix -- at the call site below.
JUMPONLY_WITH_VOLUME = False if not MUTATE else True


def f_smooth(x):
    return 2.0 * np.pi ** 2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


@LinearForm
def load(v, w):
    return f_smooth(w.x) * v


@Functional
def h1_sq(w):
    x = w.x
    gx = np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
    gy = np.pi * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])
    return (w["gu"][0] - gx) ** 2 + (w["gu"][1] - gy) ** 2


def solve_smooth(nx):
    m = MeshTri.init_tensor(np.linspace(0.0, 1.0, nx + 1),
                            np.linspace(0.0, 1.0, nx + 1))
    ib = Basis(m, ElementTriP1())
    K = laplace.assemble(ib)
    b = load.assemble(ib)
    D = ib.get_dofs().all()
    return m, ib, solve(*condense(K, b, D=D))


def estimator(m, ib, u, with_volume):
    e = ElementTriP1()
    fb0 = InteriorFacetBasis(m, e, side=0)
    fb1 = InteriorFacetBasis(m, e, side=1)

    @Functional
    def jump(w):
        return w.h * ((w["g0"][0] - w["g1"][0]) * w.n[0]
                      + (w["g0"][1] - w["g1"][1]) * w.n[1]) ** 2

    ef = jump.elemental(fb0, g0=fb0.interpolate(u).grad,
                        g1=fb1.interpolate(u).grad)
    eta2 = np.zeros(m.t.shape[1])
    np.add.at(eta2, fb0.tind, ef)
    np.add.at(eta2, fb1.tind, ef)
    if with_volume:
        @Functional
        def vol(w):
            return w.h ** 2 * f_smooth(w.x) ** 2
        eta2 = eta2 + vol.elemental(ib)
    return float(np.sqrt(eta2.sum()))


def main() -> int:
    ok = True
    # --- the P1 fact, and its P2 counterpart -----------------------------
    m0 = MeshTri().refined(2)
    for tag, elem in (("p1", ElementTriP1()), ("p2", ElementTriP2())):
        ibx = Basis(m0, elem)
        ux = ibx.project(lambda x: x[0] ** 2 + x[1] ** 2)
        g = np.asarray(ibx.interpolate(ux).grad)
        spread = float(np.abs(g - g[:, :, [0]]).max())
        print(f"{tag}_grad_variation_within_element={spread:.3e}")
        print(f"{tag}_gradient_constant_per_element={spread < 1e-12}")
    ib1 = Basis(m0, ElementTriP1())
    g1 = np.asarray(ib1.interpolate(
        ib1.project(lambda x: x[0] ** 2 + x[1] ** 2)).grad)
    ib2 = Basis(m0, ElementTriP2())
    g2 = np.asarray(ib2.interpolate(
        ib2.project(lambda x: x[0] ** 2 + x[1] ** 2)).grad)
    s1 = float(np.abs(g1 - g1[:, :, [0]]).max())
    s2 = float(np.abs(g2 - g2[:, :, [0]]).max())
    print(f"p1_interior_laplacian_is_identically_zero={s1 < 1e-12}")
    print(f"p2_interior_laplacian_is_not_zero={s2 > 1e-6}")
    if not (s1 < 1e-12 and s2 > 1e-6):
        print("FAIL: the P1/P2 interior-residual distinction did not hold",
              file=sys.stderr)
        ok = False

    # --- the convergence study --------------------------------------------
    errs, full, jumponly = [], [], []
    for nx in (8, 16, 32, 64):
        m, ib, u = solve_smooth(nx)
        e = float(np.sqrt(h1_sq.assemble(ib, gu=ib.interpolate(u).grad)))
        errs.append(e)
        full.append(estimator(m, ib, u, True))
        jumponly.append(estimator(m, ib, u, JUMPONLY_WITH_VOLUME))
        print(f"nx{nx}_h1_error={e:.4e} eta_full={full[-1]:.4e} "
              f"eta_jumponly={jumponly[-1]:.4e}")

    def rates(v):
        return [float(np.log2(v[i] / v[i + 1])) for i in range(len(v) - 1)]

    r_err, r_full, r_jump = rates(errs), rates(full), rates(jumponly)
    print(f"h1_error_rates={[f'{r:.3f}' for r in r_err]}")
    print(f"eta_full_rates={[f'{r:.3f}' for r in r_full]}")
    print(f"eta_jumponly_rates={[f'{r:.3f}' for r in r_jump]}")
    print(f"all_three_converge_at_the_same_order="
          f"{max(abs(r_full[i] - r_jump[i]) for i in range(len(r_full)))
             < 0.05}")
    print(f"dropping_the_volume_term_changes_the_order="
          f"{max(abs(r_full[i] - r_jump[i]) for i in range(len(r_full)))
             > 0.5}")
    eff_full = [full[i] / errs[i] for i in range(len(errs))]
    eff_jump = [jumponly[i] / errs[i] for i in range(len(errs))]
    print(f"effectivity_full={[f'{x:.3f}' for x in eff_full]}")
    print(f"effectivity_jumponly={[f'{x:.3f}' for x in eff_jump]}")
    print(f"both_effectivities_are_stable="
          f"{max(eff_full) - min(eff_full) < 0.4
             and max(eff_jump) - min(eff_jump) < 0.4}")
    print(f"jumponly_effectivity_is_lower="
          f"{all(eff_jump[i] < eff_full[i] for i in range(len(errs)))}")
    print(f"effectivity_shortfall_fraction="
          f"{1.0 - eff_jump[-1] / eff_full[-1]:.4f}")
    print(f"jumponly_estimator_plateaus="
          f"{jumponly[-1] > 0.5 * jumponly[0]}")
    if max(abs(r_full[i] - r_jump[i]) for i in range(len(r_full))) > 0.5:
        print("FAIL: dropping the volume term DID change the order, so the "
              "entry's signal holds after all", file=sys.stderr)
        ok = False
    if jumponly[-1] > 0.5 * jumponly[0]:
        print("FAIL: the jump-only estimator plateaued", file=sys.stderr)
        ok = False
    if not all(eff_jump[i] < eff_full[i] for i in range(len(errs))):
        print("FAIL: dropping the volume term did not lower the effectivity",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
