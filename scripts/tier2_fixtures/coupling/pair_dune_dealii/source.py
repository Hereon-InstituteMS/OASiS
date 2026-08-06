"""DUNE-fem <-> deal.II, both roles — the pair with no FEniCSx in it.

THE CLAIM UNDER TEST: the sides table names this pair twice, once in each row
("DUNE-fem ... coupled to FEniCSx and deal.II"; "deal.II ... coupled to FEniCSx
and DUNE-fem").

It is worth its own fixture because every other cross-code pair in the table has
FEniCSx on one side. If the shipped FEniCSx participant and the driver happened
to agree on a convention that the knowledge states wrongly, every FEniCSx pair
would still converge to the right answer and the error would be invisible. This
pair removes that common factor: two participants written independently, neither
of them the one the rest of the table is anchored to, meeting the closed form.

Both are also the awkward backends — DUNE JIT-compiles its forms on a fresh
process every iteration, deal.II is a compiled executable driven by a wrapper
through a plain-text input file — so this is the slowest pair in the set.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _couplinglib as L                                    # noqa: E402


def body() -> None:
    L.require_available("dune", "dealii")
    exe = L.dealii_exe()
    print(f"dealii_solver_built={exe.is_file()}")
    L.heat_arrangement("dune_D_dealii_N", "dune", "dealii", "left")
    L.heat_arrangement("dealii_D_dune_N", "dealii", "dune", "left")
    print("pairs_run=2")


L.main(body)
