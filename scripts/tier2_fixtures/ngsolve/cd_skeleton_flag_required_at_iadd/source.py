"""Tier-2: the jump integrand over plain dx is rejected -- but at __iadd__, and
the two alternative wordings the claim rules out really are absent.

Claim: ngsolve convection_diffusion#2 -- "dx(skeleton=True) for interior facet
integrals.  Signal: omitting skeleton=True for a jump integrand raises, at
.Assemble(), the literal NgException('DG-facet terms need either skeleton=True
or element_boundary=True').  The prior catalog signal `SymbolicBFI: u.Other()
outside skeleton context` does not exist in NGSolve 6.2.2604, and the
alternative 'silently integrates against bulk dx' branch does not happen
either."

Wrong variant: the jump form over dx with no skeleton flag.

CORRECTION this fixture records, the same one dg_methods#2 carries: the message
text is exact but the stated failure point is not.  The NgException comes out of
BilinearForm.__iadd__, before Assemble() is reached, so a try/except wrapped
around .Assemble() as the wording invites never sees it.  The fixture records
the stage the failure actually happens at.

What this fixture pins, all re-measured on this run:
  * the literal message, as a whole sentence;
  * that it surfaces at __iadd__ and Assemble() is never reached;
  * the prior wording the claim rules out is absent from the message;
  * the "silently integrates against bulk dx" branch really does not happen --
    checked by the failure being raised at all, and by the form object never
    reaching a state where it holds an integrator;
  * element_boundary=True is the second accepted spelling the message names, and
    it assembles -- so the message's advice is actionable, checked rather than
    assumed;
  * dx(skeleton=True) assembles too and produces cross-element entries.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology site
-- the bad form is written with dx(skeleton=True) instead of plain dx -- so
__iadd__ accepts it, the form assembles and the NgException never fires.
Re-run with `T2_MUTATE=1 python source.py`.
"""
from __future__ import annotations

import os
import sys

import numpy
from netgen.geom2d import unit_square
from ngsolve import BilinearForm, L2, Mesh, dx

MESSAGE = "DG-facet terms need either skeleton=True or element_boundary=True"
PRIOR = "SymbolicBFI: u.Other() outside skeleton context"
MUTATE = os.environ.get("T2_MUTATE") == "1"


def written(form, fes, mesh):
    rows, cols, vals = form.mat.COO()
    r = numpy.asarray(rows)
    c = numpy.asarray(cols)
    live = numpy.abs(numpy.asarray(vals)) > 1e-14
    blk = fes.ndof // mesh.ne
    same = (r[live] // blk) == (c[live] // blk)
    return int(live.sum()), int((~same).sum())


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    fes = L2(mesh, order=1, dgjumps=True)
    u, v = fes.TnT()
    jump = (u - u.Other()) * (v - v.Other())

    a_bad = BilinearForm(fes)
    stage = "form_constructed"
    msg = ""
    try:
        # The pathology: no skeleton flag on the facet integrand.
        # T2_MUTATE=1 adds skeleton=True at this __iadd__.
        a_bad += jump * (dx(skeleton=True) if MUTATE else dx)
        stage = "iadd_accepted"
        a_bad.Assemble()
        stage = "assembled"
    except Exception as exc:                                   # noqa: BLE001
        msg = str(exc)
    print(f"plain_dx_stage_reached={stage}")
    print(f"plain_dx_raised={bool(msg)}")
    print(f"plain_dx_message_literal={MESSAGE in msg}")
    print(f"raised_at_iadd_not_assemble={stage == 'form_constructed'}")
    print(f"assemble_never_reached={stage != 'assembled'}")
    print(f"prior_catalog_wording_absent={PRIOR not in msg}")
    print(f"no_silent_bulk_dx_integration={bool(msg)}")

    a_int = BilinearForm(fes)
    a_int += jump * dx(skeleton=True)
    a_int.Assemble()
    n_i, off_i = written(a_int, fes, mesh)
    print(f"skeleton_true_assembles=True")
    print(f"skeleton_entries={n_i} cross_element={off_i}")
    print(f"skeleton_couples_elements={off_i > 0}")

    # The message names a SECOND accepted spelling; check it works.
    eb_ok = True
    eb_err = ""
    try:
        a_eb = BilinearForm(fes)
        a_eb += jump * dx(element_boundary=True)
        a_eb.Assemble()
        n_e, off_e = written(a_eb, fes, mesh)
    except Exception as exc:                                   # noqa: BLE001
        eb_ok = False
        eb_err = f"{type(exc).__name__}: {exc}"
        n_e = off_e = 0
    print(f"element_boundary_accepted={eb_ok} err={eb_err!r}")
    print(f"element_boundary_entries={n_e} cross_element={off_e}")
    print(f"messages_second_spelling_is_actionable={eb_ok and off_e > 0}")

    ok = (
        stage == "form_constructed"
        and MESSAGE in msg
        and PRIOR not in msg
        and off_i > 0
        and eb_ok and off_e > 0
    )
    if ok:
        return 0
    print("FAIL: skeleton-flag invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
