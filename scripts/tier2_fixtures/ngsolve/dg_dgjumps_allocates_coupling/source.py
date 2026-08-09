"""Tier-2: L2 needs dgjumps=True before a skeleton integral can be assembled.

Claim: ngsolve dg_methods#0 -- "MUST set dgjumps=True on the L2 / DG FE space;
without it the cross-element coupling entries are NOT allocated.  Signal:
assembling a DG BilinearForm with jump terms after fes = L2(mesh, order=k)
raises, at .Assemble(), the literal NgException('SparseMatrixTM::AddElementMatrix:
illegal dnums' + "in Assemble BilinearForm 'biform_from_py'").  The fix is
L2(mesh, order=k, dgjumps=True), which assembles.  'Sparse matrix: entry at
(i,j) does not exist' is NOT a NGSolve 6.2.2604 string."

Wrong variant: the same interior-penalty jump form on a default L2 space.

What this fixture pins, all re-measured on this run:
  * the default space accepts the form at __iadd__ and only fails at Assemble()
    -- the claim's stated failure point, checked by recording the stage;
  * the message carries 'illegal dnums' AND the 'in Assemble BilinearForm'
    tail, both literal;
  * the alternative wording the claim declares absent does not appear;
  * with dgjumps=True the same form assembles and the matrix gains
    off-block-diagonal entries -- ndof is unchanged, nze is strictly larger
    than the block-diagonal-only count, which is the structural content of
    "coupling entries are not allocated".

Mutation control:  T2_MUTATE=1 applies the documented fix at the pathology site
-- the "plain" space is built as L2(mesh, order=1, dgjumps=True), so the space
that is supposed to lack the coupling allocation now has it.  Assemble()
succeeds and the expectations plain_form_built_without_error=True,
plain_assemble_raised=True, plain_msg_has_illegal_dnums=True,
plain_msg_has_assemble_tail=True and coupling_entries_allocated=True all
disappear.  Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

from netgen.geom2d import unit_square
from ngsolve import BilinearForm, L2, Mesh, dx


ABSENT_WORDING = "Sparse matrix: entry at"

# Mutation control: under T2_MUTATE=1 the "plain" L2 space is given
# dgjumps=True, i.e. the documented fix -- the pathology is removed.
MUTATE = os.environ.get("T2_MUTATE") == "1"


def _jump_form(fes):
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += u * v * dx
    a += (u - u.Other()) * (v - v.Other()) * dx(skeleton=True)
    return a


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))

    fes_plain = L2(mesh, order=1, dgjumps=MUTATE)
    fes_dg = L2(mesh, order=1, dgjumps=True)
    print(f"ndof_plain={fes_plain.ndof}")
    print(f"ndof_dgjumps={fes_dg.ndof}")
    print(f"ndof_unchanged_by_dgjumps={fes_plain.ndof == fes_dg.ndof}")

    # --- wrong variant: no dgjumps -------------------------------------
    stage = "constructed"
    msg = ""
    a_plain = _jump_form(fes_plain)          # __iadd__ must NOT raise
    stage = "form_built"
    try:
        a_plain.Assemble()
        stage = "assembled"
    except Exception as exc:                  # noqa: BLE001
        msg = str(exc)
    print(f"plain_form_built_without_error={stage == 'form_built'}")
    print(f"plain_assemble_raised={stage == 'form_built' and bool(msg)}")
    print(f"plain_msg_has_illegal_dnums={'illegal dnums' in msg}")
    print(f"plain_msg_has_assemble_tail={'in Assemble BilinearForm' in msg}")
    print(f"absent_wording_really_absent={ABSENT_WORDING not in msg}")

    # --- right variant: dgjumps=True ------------------------------------
    a_dg = _jump_form(fes_dg)
    a_dg.Assemble()
    nze_dg = a_dg.mat.nze

    # Reference: the SAME mass integral on the default L2 space.  dgjumps is a
    # property of the SPACE, not of the form -- it is what widens the allocated
    # sparsity pattern -- so the block-diagonal count has to be taken on the
    # space that lacks it.
    up, vp = fes_plain.TnT()
    a_mass = BilinearForm(fes_plain)
    a_mass += up * vp * dx
    a_mass.Assemble()
    nze_mass = a_mass.mat.nze

    print(f"dgjumps_assembled=True")
    print(f"dgjumps_nze={nze_dg}")
    print(f"plain_space_blockdiag_nze={nze_mass}")
    print(f"coupling_entries_allocated={nze_dg > nze_mass}")

    ok = (
        stage == "form_built"
        and "illegal dnums" in msg
        and "in Assemble BilinearForm" in msg
        and ABSENT_WORDING not in msg
        and fes_plain.ndof == fes_dg.ndof
        and nze_dg > nze_mass
    )
    if ok:
        return 0
    print("FAIL: dgjumps coupling-allocation invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
