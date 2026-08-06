"""Tier-2: a mis-paired saddle-point system fails through UMFPACK, not PETSc.

Claim: NGSolve mixed_poisson#2 -- with a mis-paired flux/scalar combination the
direct path is the one that talks: a.mat.Inverse(..., inverse='umfpack') warns
that the matrix is singular and then raises
NgException('UmfpackInverse: Numeric factorization failed.'). The prior catalog
text cited `KSPSolve: DIVERGED_INDEFINITE_PC`, a PETSc message imported from the
wrong solver ecosystem.

Wrong variant: BDM_1 * L2(1). Right variant: BDM_1 * L2(0).
The fixture asserts both PETSc strings are absent from the text NGSolve produced,
so the wrong attribution cannot quietly return.
"""
from __future__ import annotations

import sys

from netgen.geom2d import unit_square
from ngsolve import BilinearForm, HDiv, L2, Mesh, div, dx


def build(l2order: int):
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    X = HDiv(mesh, order=1, RT=False) * L2(mesh, order=l2order)
    (sigma, u), (tau, v) = X.TnT()
    a = BilinearForm(X)
    a += (sigma * tau + div(sigma) * v + div(tau) * u) * dx
    a.Assemble()
    return X, a


def main() -> int:
    ok = True

    # --- WRONG variant: BDM_1 with the RT partner ------------------------
    X, a = build(1)
    msg = ""
    try:
        a.mat.Inverse(X.FreeDofs(), inverse="umfpack")
    except Exception as exc:                      # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
    print(f"mispaired_inverse_raises_ngexception={'NgException' in msg}")
    print(f"mispaired_msg={msg[:150]!r}")
    if "Numeric factorization failed" not in msg:
        print(f"FAIL: the mis-paired inverse did not fail as claimed; got "
              f"{msg!r}", file=sys.stderr)
        ok = False

    for token, label in (("DIVERGED_INDEFINITE_PC",
                          "petsc_diverged_indefinite_pc_absent"),
                         ("DIVERGED_BREAKDOWN",
                          "petsc_diverged_breakdown_absent"),
                         ("KSPSolve", "petsc_kspsolve_absent")):
        absent = token not in msg
        print(f"{label}={absent}")
        if not absent:
            print(f"FAIL: NGSolve emitted the PETSc token {token!r}",
                  file=sys.stderr)
            ok = False

    # --- RIGHT variant ----------------------------------------------------
    Xg, ag = build(0)
    good = ""
    try:
        inv = ag.mat.Inverse(Xg.FreeDofs(), inverse="umfpack")
        succeeded = inv is not None
    except Exception as exc:                      # noqa: BLE001
        good = f"{type(exc).__name__}: {exc}"
        succeeded = False
    print(f"correct_pairing_inverse_succeeds={succeeded}")
    if not succeeded:
        print(f"FAIL: the correct BDM_1 x L2(0) pairing failed to factorise: "
              f"{good}", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
