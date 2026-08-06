"""Tier-2: ElementLineHermite gives the 4-DOF Euler-Bernoulli beam element.

Claim: skfem biharmonic#3 (previously "claim inherited -- not yet empirically
verified") -- Basis(MeshLine(), ElementLineHermite()).Nbfun == 4: deflection and
slope at each endpoint, C^1 continuous.

Wrong variant: ElementLineP1 (2 DOFs, C^0). A beam bending form needs the second
derivative, and skfem raises rather than quietly returning something.

Right variant: ElementLineHermite. The cantilever with a tip load P is solved and
checked against the analytic tip deflection P L^3 / (3 E I) -- an exact
closed-form value, not a measured one.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology
site -- the element handed to the bending form u.hess * v.hess becomes
ElementLineHermite instead of the C^0 ElementLineP1 -- so the assembly
succeeds and the "it raises" evidence disappears.  Re-run with
T2_MUTATE=1 python source.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from skfem import (
    Basis,
    BilinearForm,
    ElementLineHermite,
    ElementLineP1,
    LinearForm,
    MeshLine,
    condense,
    solve,
)

MUTATE = os.environ.get("T2_MUTATE") == "1"


@BilinearForm
def bending(u, v, w):
    return u.hess[0][0] * v.hess[0][0]


def main() -> int:
    ok = True

    hermite = Basis(MeshLine(), ElementLineHermite())
    linep1 = Basis(MeshLine(), ElementLineP1())
    print(f"hermite_nbfun={hermite.Nbfun}")
    print(f"linep1_nbfun={linep1.Nbfun}")
    if hermite.Nbfun != 4:
        print(f"FAIL: Hermite Nbfun is {hermite.Nbfun}, claim says 4",
              file=sys.stderr)
        ok = False

    # --- WRONG variant: C^0 line element in a bending form --------------
    # The pathology: the C^0 ElementLineP1 is used in a form that reads
    # u.hess.  T2_MUTATE=1 applies the documented fix -- the C^1
    # ElementLineHermite -- at exactly this assembly.
    beam_element = ElementLineP1() if not MUTATE else ElementLineHermite()
    raised = ""
    try:
        bending.assemble(Basis(MeshLine().refined(3), beam_element))
    except Exception as exc:
        raised = f"{type(exc).__name__}: {exc}"
    print(f"linep1_beam_raises={bool(raised)}")
    print(f"linep1_beam_msg={raised!r}")
    if not raised:
        print("FAIL: ElementLineP1 assembled a bending form without complaint",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant: cantilever tip deflection ----------------------
    L, E, I, P = 1.0, 1.0, 1.0, 1.0
    m = MeshLine(np.linspace(0.0, L, 17))
    basis = Basis(m, ElementLineHermite())
    K = (E * I) * bending.assemble(basis)

    # Point load at the free end: the last vertex DOF.
    f = basis.zeros()
    clamped = basis.get_dofs(lambda x: x[0] < 1e-10)
    tip = basis.get_dofs(lambda x: x[0] > L - 1e-10)
    tip_deflection_dof = int(np.min(tip.flatten()))
    f[tip_deflection_dof] = P

    w = solve(*condense(K, f, D=clamped))
    tip_w = float(w[tip_deflection_dof])
    analytic = P * L ** 3 / (3.0 * E * I)
    rel = abs(tip_w - analytic) / abs(analytic)
    print(f"hermite_tip_over_analytic_within_1pct={rel < 1e-2}")
    print(f"hermite_tip_matches_analytic={rel < 1e-6}")
    if rel > 1e-6:
        print(f"FAIL: cantilever tip deflection {tip_w!r} vs analytic "
              f"{analytic!r} (rel {rel:.3e})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
