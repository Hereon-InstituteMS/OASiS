"""Tier-2: `periodic` lives on the DG mesh classes only.

Claim: skfem convection_diffusion#2 -- the periodic(mesh, ix, ix0) classmethod
lives ONLY on MeshTri1DG / MeshQuad1DG / MeshLine1DG, not on MeshTri / MeshQuad /
Mesh, and there is no top-level skfem name containing 'periodic'.

Wrong variant: MeshTri.periodic(...), which raises AttributeError.
Right variant: MeshTri1DG.periodic(m, ix, ix0), which is constructed here and
checked to actually collapse the paired DOFs -- the periodic basis has fewer
DOFs than the plain one on the same mesh, which is the whole point.
"""
from __future__ import annotations

import inspect
import sys

import numpy as np
import skfem
from skfem import (
    Basis,
    ElementTriP1,
    MeshLine1DG,
    MeshQuad,
    MeshQuad1DG,
    MeshTri,
    MeshTri1DG,
)


def main() -> int:
    ok = True

    plain = {"meshtri": MeshTri, "meshquad": MeshQuad}
    dg = {"meshtri1dg": MeshTri1DG, "meshquad1dg": MeshQuad1DG,
          "meshline1dg": MeshLine1DG}
    for label, cls in plain.items():
        present = hasattr(cls, "periodic")
        print(f"{label}_has_periodic={present}")
        if present:
            print(f"FAIL: {label} now has a periodic attribute; the claim is "
                  f"stale", file=sys.stderr)
            ok = False
    for label, cls in dg.items():
        present = hasattr(cls, "periodic")
        print(f"{label}_has_periodic={present}")
        if not present:
            print(f"FAIL: {label} lost its periodic classmethod",
                  file=sys.stderr)
            ok = False

    sig = str(inspect.signature(MeshTri1DG.periodic))
    print(f"periodic_signature={sig}")
    if sig != "(mesh, ix, ix0)":
        print(f"FAIL: periodic signature is {sig!r}, claim says "
              f"'(mesh, ix, ix0)'", file=sys.stderr)
        ok = False

    names = [n for n in dir(skfem) if "periodic" in n.lower()]
    print(f"toplevel_periodic_names={names}")
    if names:
        print(f"FAIL: skfem now exposes {names!r} at top level",
              file=sys.stderr)
        ok = False

    # --- WRONG variant: the plain-class spelling -------------------------
    msg = ""
    try:
        MeshTri.periodic                 # noqa: B018 - deliberate probe
    except AttributeError as exc:
        msg = str(exc)
    print(f"meshtri_periodic_attributeerror={msg!r}")
    if "no attribute 'periodic'" not in msg:
        print(f"FAIL: expected the AttributeError, got {msg!r}",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant: build one and check DOFs actually get paired -----
    m = MeshTri().refined(3)
    basis_plain = Basis(m, ElementTriP1())
    left = m.p[0] < 1e-10
    right = m.p[0] > 1.0 - 1e-10
    ix = np.nonzero(right)[0]
    # pair each right-edge vertex with the left-edge vertex at the same y
    ix0 = np.array([np.nonzero(left & (np.abs(m.p[1] - m.p[1][i]) < 1e-10))[0][0]
                    for i in ix])
    mp = MeshTri1DG.periodic(m, ix, ix0)
    basis_periodic = Basis(mp, ElementTriP1())
    fewer = basis_periodic.N < basis_plain.N
    print(f"plain_basis_N={basis_plain.N}")
    print(f"periodic_basis_N={basis_periodic.N}")
    print(f"n_paired={len(ix)}")
    print(f"periodic_identifies_paired_dofs={fewer}")
    if not fewer:
        print(f"FAIL: the periodic mesh did not reduce the DOF count "
              f"({basis_periodic.N} vs {basis_plain.N})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
