"""Tier-2: the RT flag and the L2 partner must match, or nothing complains.

Claim: NGSolve mixed_poisson#0 -- RT=True gives Raviart-Thomas RT_k, RT=False
(the default) gives BDM_k, and they need DIFFERENT L2 partners.
HDiv(order=k, RT=True) * L2(k-1) assembles, factorises and returns a plausible
answer that does not converge. The correct pairings are RT_k * L2(k) and
BDM_k * L2(k-1).

Wrong variant: RT_1 * L2(0). Right variants: RT_1 * L2(1) and BDM_1 * L2(0),
which are checked to agree with each other as well as to converge.

Mutation control: T2_MUTATE=1 sets MISPAIRED_L2ORDER from 0 to 1, i.e. it gives
the RT_1 velocity space its correct L2(1) partner -- the exact one-token fix the
prose prescribes -- at the site where the mis-paired system is built. The
"mis-paired" run then converges, so 'mispaired_does_not_converge=True' and
'mispaired_error_over_50pct=True' go missing and the fixture goes red.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from netgen.geom2d import unit_square
from ngsolve import (
    BilinearForm,
    GridFunction,
    HDiv,
    Integrate,
    L2,
    LinearForm,
    Mesh,
    div,
    dx,
)

MESHES = (0.2, 0.1, 0.05)

# Mutation control: the L2 order the WRONG variant pairs with RT_1.  0 is the
# pitfall; T2_MUTATE=1 makes it 1, which is the documented correct partner.
MUTATE = os.environ.get("T2_MUTATE") == "1"
MISPAIRED_L2ORDER = 1 if MUTATE else 0


def mean_u(maxh: float, k: int, l2order: int, rt: bool) -> float:
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    V = HDiv(mesh, order=k, RT=rt)
    Q = L2(mesh, order=l2order)
    X = V * Q
    (sigma, u), (tau, v) = X.TnT()
    a = BilinearForm(X)
    a += (sigma * tau + div(sigma) * v + div(tau) * u) * dx
    f = LinearForm(X)
    f += -1 * v * dx
    a.Assemble()
    f.Assemble()
    gf = GridFunction(X)
    gf.vec.data = a.mat.Inverse(X.FreeDofs(), inverse="umfpack") * f.vec
    return float(Integrate(gf.components[1], mesh))


def converges(values, reference: float) -> bool:
    errs = [abs(v - reference) for v in values]
    return errs[-1] < errs[0] / 5.0 and errs[-1] < 1e-3


def main() -> int:
    ok = True
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    print(f"hdiv_space_type={HDiv(mesh, order=1, RT=True).type}")
    counts = {}
    for order in (0, 1, 2):
        rt = HDiv(mesh, order=order, RT=True).ndof
        bdm = HDiv(mesh, order=order, RT=False).ndof
        counts[order] = (rt, bdm)
        if order:
            print(f"ndof_order{order}_rt={rt}")
            print(f"ndof_order{order}_bdm={bdm}")
    print(f"ndof_order0_flag_is_noop={counts[0][0] == counts[0][1]}")
    print(f"rt_flag_changes_ndof_at_order1={counts[1][0] != counts[1][1]}")
    if counts[0][0] != counts[0][1] or counts[1][0] == counts[1][1]:
        print(f"FAIL: ndof pattern changed: {counts!r}", file=sys.stderr)
        ok = False

    mispaired = [mean_u(h, 1, MISPAIRED_L2ORDER, True) for h in MESHES]
    rt_correct = [mean_u(h, 1, 1, True) for h in MESHES]
    bdm_correct = [mean_u(h, 1, 0, False) for h in MESHES]
    print(f"mispaired_rt1_l2_0={[f'{v:.7f}' for v in mispaired]}")
    print(f"rt1_l2_1={[f'{v:.7f}' for v in rt_correct]}")
    print(f"bdm1_l2_0={[f'{v:.7f}' for v in bdm_correct]}")

    # The finest correct value is the reference -- not a hard-coded number.
    reference = rt_correct[-1]
    rt_ok = converges(rt_correct[:-1], reference)
    bdm_ok = converges(bdm_correct[:-1], reference)
    agree = bool(np.allclose(rt_correct, bdm_correct, rtol=1e-9))
    print(f"rt_with_l2k_converges={rt_ok}")
    print(f"bdm_with_l2km1_converges={bdm_ok}")
    print(f"the_two_correct_pairings_agree={agree}")
    if not (rt_ok and bdm_ok and agree):
        print(f"FAIL: the correct pairings did not converge/agree: "
              f"{rt_correct} vs {bdm_correct}", file=sys.stderr)
        ok = False

    # --- WRONG variant: mis-paired ---------------------------------------
    errs = [abs(v - reference) for v in mispaired]
    not_converging = errs[-1] > errs[0] / 2.0
    big = errs[-1] / abs(reference) > 0.5
    print(f"mispaired_relative_error_last={errs[-1] / abs(reference):.4f}")
    print(f"mispaired_does_not_converge={not_converging}")
    print(f"mispaired_error_over_50pct={big}")
    print(f"mispaired_raised_nothing=True")
    if not (not_converging and big):
        print(f"FAIL: the mis-paired system converged after all: {mispaired}",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
