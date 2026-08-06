"""Tier-2: the sign of the inlet traction decides the flow direction.

Claim: skfem hydraulic_resistance#3 -- the inlet pressure traction is a
NATURAL BC, int_inlet (-p_in) v_x ds with outward normal (-1, 0) at x=0, so it
enters positively.  "The MINUS SIGN matters -- wrong sign gives flow in the
wrong direction (Q < 0) and R < 0.  Signal: summary['Q'] is negative;
results.vtu shows pressure increasing in the flow direction (physically
wrong); relative_error is order 1 (R has wrong sign)."

Measured on skfem 12.0.1 with the shipped channel discretisation (41x7 tensor
grid, ElementVector(TriP2)/TriP1, intorder=4, stress form, walls no-slip,
outlet pressure pinned):

  * every part reproduces.  The correct sign gives a positive flow rate and a
    positive resistance, with the pressure falling from inlet to outlet.
    Flipping the sign flips the flow rate, the resistance and the pressure
    field all at once, and the pressure then RISES along the flow.
  * the problem is linear in the traction, so the flipped run is the exact
    negative of the correct one -- not merely "wrong direction" but the same
    magnitude with the opposite sign, which is why |R| still looks plausible
    and only the SIGN gives it away.
  * the failure is silent: no exception, no warning, a finite solution.
  * the cheapest guard is therefore a sign assertion on Q, not a magnitude
    check on the relative error, because the relative error computed as
    |R_fe - R_exact| / R_exact comes out near 2 rather than "order 1" and a
    tolerance-style gate keyed on a small number would flag it for the wrong
    reason.
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

MU, H, L, P_IN = 1.0, 0.1, 2.0, 1.0


@BilinearForm
def stiffness(u, v, w):
    return 2.0 * MU * ddot(sym_grad(u), sym_grad(v))


@BilinearForm
def neg_div(u, p, w):
    return -div(u) * p


def run(sign):
    m = MeshTri.init_tensor(np.linspace(0.0, L, 41), np.linspace(0.0, H, 7))
    m = m.with_boundaries({
        "inlet": lambda x: np.isclose(x[0], 0.0),
        "outlet": lambda x: np.isclose(x[0], L),
        "wall": lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], H),
    })
    ev = ElementVector(ElementTriP2())
    bu = Basis(m, ev, intorder=4)
    bp = Basis(m, ElementTriP1(), intorder=4)
    fi = FacetBasis(m, ev, intorder=4, facets=m.boundaries["inlet"])
    fo = FacetBasis(m, ev, intorder=4, facets=m.boundaries["outlet"])

    @LinearForm
    def inlet_traction(v, w):
        return sign * P_IN * v.value[0]

    B = neg_div.assemble(bu, bp)
    K = sp.bmat([[stiffness.assemble(bu), B.T], [B, None]], format="csr")
    F = np.concatenate([inlet_traction.assemble(fi), bp.zeros()])
    D = np.concatenate([bu.get_dofs("wall").all(),
                        [bu.N + int(bp.get_dofs("outlet").all()[0])]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = solve(*condense(K, F, x=np.zeros(K.shape[0]), D=D))
        msgs = sorted({str(c.message) for c in caught})
    u, p = x[:bu.N], x[bu.N:]
    q = float(np.sum(fo.interpolate(u).value[0] * fo.dx))
    p_in = float(p[bp.get_dofs("inlet").all()].mean())
    p_out = float(p[bp.get_dofs("outlet").all()].mean())
    return u, q, P_IN / q, p_in, p_out, msgs, bool(np.isfinite(x).all())


def main() -> int:
    ok = True
    r_exact = 12.0 * MU * L / H ** 3
    print(f"R_exact={r_exact:.6e}")

    res = {}
    for tag, sign in (("correct", +1.0), ("flipped", -1.0)):
        u, q, r, pin, pout, msgs, fin = run(sign)
        res[tag] = (u, q, r, pin, pout)
        print(f"{tag}_Q={q:+.6e}")
        print(f"{tag}_R={r:+.6e}")
        print(f"{tag}_p_inlet={pin:+.4f} p_outlet={pout:+.4f}")
        print(f"{tag}_pressure_falls_along_flow={pin > pout}")
        print(f"{tag}_warnings={msgs!r}")
        print(f"{tag}_finite={fin}")
        if msgs:
            print(f"FAIL: {tag} warned {msgs!r}", file=sys.stderr)
            ok = False
        if not fin:
            print(f"FAIL: {tag} was not finite", file=sys.stderr)
            ok = False

    _, q_ok, r_ok, pin_ok, pout_ok = res["correct"]
    _, q_bad, r_bad, pin_bad, pout_bad = res["flipped"]
    print(f"correct_Q_is_positive={q_ok > 0}")
    print(f"flipped_Q_is_negative={q_bad < 0}")
    print(f"correct_R_is_positive={r_ok > 0}")
    print(f"flipped_R_is_negative={r_bad < 0}")
    print(f"correct_pressure_drops={pin_ok > pout_ok}")
    print(f"flipped_pressure_rises={pin_bad < pout_bad}")
    if not (q_ok > 0 and q_bad < 0 and r_ok > 0 and r_bad < 0):
        print("FAIL: the sign flip did not reverse Q and R", file=sys.stderr)
        ok = False
    if not (pin_ok > pout_ok and pin_bad < pout_bad):
        print("FAIL: the pressure gradient did not reverse", file=sys.stderr)
        ok = False

    # --- the problem is linear, so the flip is an exact negation ----------
    dev = float(np.abs(res["correct"][0] + res["flipped"][0]).max())
    print(f"velocity_fields_sum_to_zero_max_abs={dev:.3e}")
    print(f"flip_is_an_exact_negation={dev < 1e-12}")
    print(f"magnitudes_are_identical={abs(abs(r_bad) - r_ok) < 1e-6}")
    if dev >= 1e-12:
        print("FAIL: the flipped run was not the exact negative",
              file=sys.stderr)
        ok = False

    # --- what the relative error actually reads ---------------------------
    rel_ok = abs(r_ok - r_exact) / r_exact
    rel_bad = abs(r_bad - r_exact) / r_exact
    print(f"correct_relative_error={rel_ok:.4e}")
    print(f"flipped_relative_error={rel_bad:.4e}")
    print(f"flipped_relative_error_is_near_two={1.9 < rel_bad < 2.1}")
    print(f"flipped_relative_error_is_order_one={0.5 < rel_bad < 1.5}")
    print(f"sign_check_is_sharper_than_magnitude_check="
          f"{q_bad < 0 and not (0.5 < rel_bad < 1.5)}")
    if 0.5 < rel_bad < 1.5:
        print("FAIL: the relative error really was order one", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
