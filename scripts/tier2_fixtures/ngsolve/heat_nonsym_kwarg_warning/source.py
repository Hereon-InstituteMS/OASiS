"""Tier-2: BilinearForm(..., nonsym=True) is an undocumented kwarg.

Pitfall (NGSolve heat#0 — drift correction): The prior catalog
recommended `BilinearForm(..., nonsym=True)` for transient mass
matrices to "get compatible sparsity pattern". Empirically, the
kwarg is undocumented; NGSolve emits a warning calling it a
possible typo and silently drops it. Matrix sparsity is
identical with or without.

This fixture verifies the warning text fires — if a future
NGSolve release legitimises the kwarg, the warning goes away
and this fixture fails (signal to revisit the catalog).

Mutation control: T2_MUTATE=1 applies the documented fix at the
pathology site — the BilinearForm is constructed as
BilinearForm(fes) with the bogus nonsym=True kwarg dropped. The
NGSolve warning is then never emitted and every expectation
("kwarg 'nonsym'", "undocumented flags option", "BilinearForm")
disappears from the output. Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

from netgen.geom2d import unit_square
from ngsolve import H1, BilinearForm, Mesh, dx

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    fes = H1(mesh, order=1)
    u, v = fes.TnT()
    # Triggers the warning on stderr; Assemble succeeds.
    a = BilinearForm(fes) if MUTATE else BilinearForm(fes, nonsym=True)
    a += u * v * dx
    a.Assemble()
    print(f"a.mat sparsity (nnz): {a.mat.AsVector().size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
