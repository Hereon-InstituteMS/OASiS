"""Tier-2: a jump term written over the plain volume measure is rejected loudly,
and it is rejected at form construction -- NOT at Assemble().

Claim: ngsolve dg_methods#2 -- "dx(skeleton=True) integrates over INTERIOR
facets; ds(skeleton=True) over BOUNDARY facets.  Signal: applying a jump-penalty
term over plain dx is NOT silently dropped: BilinearForm += (u - u.Other())*
(v - v.Other())*dx on an L2(dgjumps=True) space raises at .Assemble() with the
literal NgException('DG-facet terms need either skeleton=True or
element_boundary=True')."

Wrong variant: the jump form over dx with no skeleton flag.

CORRECTION this fixture records.  The message text is exactly right, but the
stated failure point is not: on NGSolve 6.2.2604 the NgException comes out of
BilinearForm.__iadd__, before Assemble() is ever reached.  The fixture pins the
stage it actually happens at, so a guard placed around .Assemble() (which is
what the claim's wording invites) would never see it.

What this fixture pins, all re-measured on this run:
  * the literal message, checked as a whole sentence;
  * that it surfaces at __iadd__ and that Assemble() is never reached;
  * that dx(skeleton=True) accepts the identical integrand and assembles;
  * that dx(skeleton=True) and ds(skeleton=True) are different measures --
    the interior-facet form couples DOFs across elements (wider sparsity),
    the boundary-facet form does not.
"""
from __future__ import annotations

import sys

import numpy
from netgen.geom2d import unit_square
from ngsolve import (
    BilinearForm,
    L2,
    Mesh,
    ds,
    dx,
)


MESSAGE = "DG-facet terms need either skeleton=True or element_boundary=True"


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    fes = L2(mesh, order=1, dgjumps=True)
    u, v = fes.TnT()
    jump = (u - u.Other()) * (v - v.Other())

    # --- wrong variant: plain dx ---------------------------------------
    a_bad = BilinearForm(fes)
    stage = "form_constructed"
    msg = ""
    try:
        a_bad += jump * dx
        stage = "iadd_accepted"
        a_bad.Assemble()
        stage = "assembled"
    except Exception as exc:                       # noqa: BLE001
        msg = str(exc)
    print(f"plain_dx_stage_reached={stage}")
    print(f"plain_dx_raised={bool(msg)}")
    print(f"plain_dx_message_literal={MESSAGE in msg}")
    print(f"raised_at_iadd_not_assemble={stage == 'form_constructed'}")
    print(f"assemble_never_reached={stage != 'assembled'}")

    # --- right variant: dx(skeleton=True) ------------------------------
    a_int = BilinearForm(fes)
    a_int += jump * dx(skeleton=True)
    a_int.Assemble()
    print(f"skeleton_dx_assembled=True")

    # --- ds(skeleton=True) is the BOUNDARY measure ---------------------
    a_bnd = BilinearForm(fes)
    a_bnd += u * v * ds(skeleton=True)
    a_bnd.Assemble()
    print(f"skeleton_ds_assembled=True")

    # The two measures cover disjoint facet sets, and the difference is visible
    # in the assembled matrices.  nze counts ALLOCATED slots -- a property of
    # the dgjumps space, identical for both -- so count entries that actually
    # received a value instead.  The interior-facet mass form writes into
    # off-diagonal element blocks (two elements share every interior facet); the
    # boundary-facet form touches one element per facet and stays strictly
    # block-diagonal.
    def written(form):
        rows, cols, vals = form.mat.COO()
        r = numpy.asarray(rows)
        c = numpy.asarray(cols)
        v = numpy.abs(numpy.asarray(vals))
        live = v > 1e-14
        blk = fes.ndof // mesh.ne          # DOFs per element, L2 is blocked
        same = (r[live] // blk) == (c[live] // blk)
        return int(live.sum()), int((~same).sum())

    n_int, off_int = written(a_int)
    n_bnd, off_bnd = written(a_bnd)
    print(f"interior_entries_written={n_int} cross_element={off_int}")
    print(f"boundary_entries_written={n_bnd} cross_element={off_bnd}")
    print(f"dx_skeleton_couples_neighbouring_elements={off_int > 0}")
    print(f"ds_skeleton_stays_element_local={off_bnd == 0}")

    ok = (
        stage == "form_constructed"
        and MESSAGE in msg
        and off_int > 0
        and off_bnd == 0
    )
    if ok:
        return 0
    print("FAIL: skeleton-measure invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
