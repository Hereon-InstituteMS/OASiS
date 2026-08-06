"""Tier-2: the flow-rate error factor is the channel LENGTH, not 2.

Claim: skfem hydraulic_resistance#4 -- ib_fac_out.interpolate(u).value[0] *
ib_fac_out.dx computes int_outlet u_x ds.  "Forgetting to use FacetBasis
(using ib_u directly on a boundary integral) double-counts interior edges and
produces a Q that's ~2x too large.  Signal: Q in summary is 2x the expected
Poiseuille value; R_fe is half of R_exact."

Measured on skfem 12.0.1 with the shipped channel discretisation, sweeping the
channel length:

  * THE FACTOR IS NOT 2 AND THE MECHANISM IS NOT DOUBLE-COUNTING.  Using the
    volumetric Basis computes an AREA integral, int_Omega u_x dOmega, which
    for this flow equals Q times the channel length.  The ratio of the wrong
    Q to the right one therefore tracks L: it is near 1 at L = 1, near 2 at
    L = 2, near 3 at L = 3 and near 5 at L = 5.  It reads as "2x" only
    because the template's default length happens to be 2.
  * so at L = 1 the mistake is nearly INVISIBLE -- the wrong Q is within a
    couple of percent of the right one -- and a gate calibrated on "2x" would
    pass it.  The error grows without bound as the channel lengthens.
  * "R_fe is half of R_exact" inherits the same problem: R_fe is R_exact
    divided by the length, not by two.
  * the failure is silent in every case: no exception, and nothing warns
    about the WRONG integral -- both return a single finite float.  The only
    warning skfem emits here is a style deprecation about the template's own
    `interpolate(u).value[0]` idiom, which fires identically for the right
    and the wrong integral and therefore carries no information about which
    one you ran.
  * the dimensional check that always works: ib_u.dx integrates over area and
    ib_fac_out.dx over a line, so their sums are the domain area and the
    outlet length respectively -- compare those before trusting a flux.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import warnings

import numpy as np
import scipy.sparse as sp
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    FacetBasis,
    LinearForm,
    MeshTri,
    condense,
    solve,
)
from skfem.helpers import ddot, div, sym_grad

MU, H, P_IN = 1.0, 0.1, 1.0


@BilinearForm
def stiffness(u, v, w):
    return 2.0 * MU * ddot(sym_grad(u), sym_grad(v))


@BilinearForm
def neg_div(u, p, w):
    return -div(u) * p


@LinearForm
def inlet_traction(v, w):
    return P_IN * v.value[0]


def run(length):
    m = MeshTri.init_tensor(np.linspace(0.0, length, 41),
                            np.linspace(0.0, H, 7))
    m = m.with_boundaries({
        "inlet": lambda x: np.isclose(x[0], 0.0),
        "outlet": lambda x: np.isclose(x[0], length),
        "wall": lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], H),
    })
    ev = ElementVector(ElementTriP2())
    bu = Basis(m, ev, intorder=4)
    bp = Basis(m, ElementTriP1(), intorder=4)
    fi = FacetBasis(m, ev, intorder=4, facets=m.boundaries["inlet"])
    fo = FacetBasis(m, ev, intorder=4, facets=m.boundaries["outlet"])
    B = neg_div.assemble(bu, bp)
    K = sp.bmat([[stiffness.assemble(bu), B.T], [B, None]], format="csr")
    F = np.concatenate([inlet_traction.assemble(fi), bp.zeros()])
    D = np.concatenate([bu.get_dofs("wall").all(),
                        [bu.N + int(bp.get_dofs("outlet").all()[0])]])
    x = solve(*condense(K, F, x=np.zeros(K.shape[0]), D=D))
    u = x[:bu.N]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        q_facet = float(np.sum(fo.interpolate(u).value[0] * fo.dx))
        q_volume = float(np.sum(bu.interpolate(u).value[0] * bu.dx))
        msgs = sorted({str(c.message) for c in caught})
    return q_facet, q_volume, msgs, float(np.sum(bu.dx)), float(np.sum(fo.dx))


def main() -> int:
    ok = True
    lengths = (1.0, 2.0, 3.0, 5.0)
    ratios, allmsgs = [], []
    for length in lengths:
        qf, qv, msgs, area, line = run(length)
        ratios.append(qv / qf)
        allmsgs += msgs
        print(f"L{length:g}_Q_facet={qf:.6e}")
        print(f"L{length:g}_Q_volume={qv:.6e}")
        print(f"L{length:g}_ratio={qv / qf:.4f} channel_length={length:g}")
        print(f"L{length:g}_ratio_tracks_length="
              f"{abs(qv / qf - length) < 0.15 * length}")
        print(f"L{length:g}_basis_dx_sum={area:.6f} domain_area="
              f"{length * H:.6f}")
        print(f"L{length:g}_facet_dx_sum={line:.6f} outlet_length={H:.6f}")

    print(f"ratios={[f'{r:.4f}' for r in ratios]}")
    print(f"ratio_tracks_channel_length_everywhere="
          f"{all(abs(ratios[i] - lengths[i]) < 0.15 * lengths[i]
                 for i in range(len(lengths)))}")
    print(f"ratio_is_always_about_two="
          f"{all(1.8 < r < 2.2 for r in ratios)}")
    print(f"ratio_at_L1_is_nearly_one={abs(ratios[0] - 1.0) < 0.1}")
    print(f"mistake_is_nearly_invisible_at_L1={abs(ratios[0] - 1.0) < 0.1}")
    print(f"ratio_grows_with_length={ratios[0] < ratios[-1]}")
    if all(1.8 < r < 2.2 for r in ratios):
        print("FAIL: the ratio really was ~2x at every length",
              file=sys.stderr)
        ok = False
    if not all(abs(ratios[i] - lengths[i]) < 0.15 * lengths[i]
               for i in range(len(lengths))):
        print("FAIL: the ratio did not track the channel length",
              file=sys.stderr)
        ok = False

    # --- the resistance side ----------------------------------------------
    qf, qv, _, _, _ = run(2.0)
    r_exact = 12.0 * MU * 2.0 / H ** 3
    print(f"R_from_facet={P_IN / qf:.6e}")
    print(f"R_from_volume={P_IN / qv:.6e}")
    print(f"R_exact={r_exact:.6e}")
    print(f"R_from_volume_is_half_of_exact="
          f"{abs(P_IN / qv - r_exact / 2.0) / r_exact < 0.05}")
    print(f"R_from_volume_is_exact_over_length="
          f"{abs(P_IN / qv - (P_IN / qf) / 2.0) / (P_IN / qf) < 0.05}")

    # --- silence, and the dimensional guard --------------------------------
    uniq = sorted(set(allmsgs))
    deprec = [x for x in uniq if "unnecessary" in x or "deprecat" in x.lower()]
    correctness = [x for x in uniq if x not in deprec]
    print(f"flux_warnings={uniq!r}")
    print(f"only_warning_is_a_style_deprecation="
          f"{bool(deprec) and not correctness}")
    print(f"no_warning_about_the_wrong_integral={not correctness}")
    print(f"template_value_idiom_is_deprecated={bool(deprec)}")
    _, _, _, area, line = run(2.0)
    print(f"basis_dx_integrates_area={abs(area - 2.0 * H) < 1e-9}")
    print(f"facet_dx_integrates_a_line={abs(line - H) < 1e-9}")
    print(f"dimensional_guard_discriminates="
          f"{abs(area - 2.0 * H) < 1e-9 and abs(line - H) < 1e-9
             and abs(area - line) > 1e-6}")
    if correctness:
        print(f"FAIL: a flux computation warned about correctness: "
              f"{correctness!r}", file=sys.stderr)
        ok = False
    if not (abs(area - 2.0 * H) < 1e-9 and abs(line - H) < 1e-9):
        print("FAIL: the dx sums are not the area and the outlet length",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
