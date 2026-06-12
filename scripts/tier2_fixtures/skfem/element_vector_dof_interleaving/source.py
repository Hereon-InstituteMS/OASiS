"""Tier-2: scikit-fem ElementVector DoF interleaving.

Pitfall (skfem linear_elasticity, PR #20): `ElementVector(ElementQuad1())`
stores DoFs interleaved (x0, y0, x1, y1, ...), NOT blocked
(x0, x1, ..., xN, y0, y1, ..., yN). Naive `u.reshape(2, -1)` therefore
scrambles components across rows — row 0 ends up holding a mix of x
and y values, not pure x.

The correct extraction is via the basis's `nodal_dofs` table:
  nodal_dofs.shape == (n_components, n_nodes)
  u[nodal_dofs[0]] is the x-component
  u[nodal_dofs[1]] is the y-component

This fixture demonstrates the bug by (a) computing the WRONG
component-split via reshape, (b) computing the CORRECT split via
nodal_dofs, and (c) verifying they do not match — printing the
key strings 'ElementVector', 'interleaved', and 'do not match'
that the Tier-2 runner matches against the Signal.
"""
from __future__ import annotations

import sys
import numpy as np
from skfem import Basis, ElementQuad1, ElementVector, MeshQuad


def main() -> int:
    m = MeshQuad.init_tensor(np.linspace(0, 1, 5),
                              np.linspace(0, 1, 5))
    e = ElementVector(ElementQuad1())
    ib = Basis(m, e)
    # Synthesize a solution where x = 1.0 * node_index and
    # y = 100.0 * node_index, so any cross-talk in reshape is
    # screamingly visible.
    n_nodes = m.p.shape[1]
    u = np.empty(ib.N)
    u[ib.nodal_dofs[0]] = np.arange(n_nodes, dtype=float)
    u[ib.nodal_dofs[1]] = 100.0 * np.arange(n_nodes, dtype=float)

    # CORRECT extraction
    ux_correct = u[ib.nodal_dofs[0]]
    uy_correct = u[ib.nodal_dofs[1]]
    # WRONG extraction — naive reshape assumes blocked layout
    ux_wrong, uy_wrong = u.reshape(2, -1)

    print("ElementVector(ElementQuad1()) stores DoFs interleaved.")
    print(f"  n_dofs = {ib.N}, n_nodes = {n_nodes}")
    print(f"  ux_correct (via nodal_dofs[0])[:4] = {ux_correct[:4]}")
    print(f"  uy_correct (via nodal_dofs[1])[:4] = {uy_correct[:4]}")
    print(f"  ux_wrong   (u.reshape(2,-1)[0])[:4] = {ux_wrong[:4]}")
    print(f"  uy_wrong   (u.reshape(2,-1)[1])[:4] = {uy_wrong[:4]}")

    if np.allclose(ux_correct, ux_wrong) \
            and np.allclose(uy_correct, uy_wrong):
        print("FIXTURE WARNING: reshape and nodal_dofs match — "
              "either skfem changed layout or the test mesh is "
              "degenerate", file=sys.stderr)
        return 2

    print("Components extracted via reshape(2,-1) and nodal_dofs "
          "do not match — the reshape path is wrong.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
