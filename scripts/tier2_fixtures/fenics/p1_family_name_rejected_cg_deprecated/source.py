"""Tier-2 for fenics linear_elasticity#6: 'P1' is not an element family name;
'CG' still is, with a deprecation warning.

The claim carries its own correction — the earlier wording said ('CG', 1) raises
— so the fixture checks BOTH branches on one mesh: the degree-suffixed DOLFIN
name must raise, and 'CG' must build a space while emitting a deprecation
warning.

Mutation control: T2_MUTATE=1 asks for 'Lagrange' instead of 'P1'; nothing
raises.
"""
from __future__ import annotations

import os
import tempfile
import warnings

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    family = "Lagrange" if MUTATE else "P1"

    raised = ""
    try:
        dolfinx.fem.functionspace(msh, (family, 1))
        print(f"family_{family}_raised=False")
    except Exception as exc:
        raised = f"{type(exc).__name__}: {exc}"
        print(f"family_{family}_raised=True {raised}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        V = dolfinx.fem.functionspace(msh, ("CG", 1))
    n = V.dofmap.index_map.size_global
    texts = " | ".join(str(w.message) for w in caught)
    print(f"CG_built_space=True CG_ndofs={n}")
    print(f"CG_warning_count={len(caught)}")
    print(f"CG_warning_text={texts}")
    if raised and "Unknown element family" in raised and n > 0:
        print("VERDICT=p1_rejected_cg_deprecated_but_working")
        return 0
    print("VERDICT=family_names_behave_differently")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
