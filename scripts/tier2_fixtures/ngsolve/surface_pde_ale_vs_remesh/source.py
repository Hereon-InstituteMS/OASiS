"""Tier-2: for an evolving surface, re-meshing per step loses the solution;
mesh.SetDeformation (ALE) keeps the same DOFs and the same field.

Claim: ngsolve surface_pde#4 - for evolving surfaces use a deformation
mapping + ALE with mesh.SetDeformation; re-meshing every step (new OCC
geometry per time step) loses solution continuity.

Wrong variant: grow the sphere by building a NEW OCCGeometry(Sphere(r=1+s))
  each step, then move the field onto the new space the two ways a user
  actually reaches for - a raw coefficient-vector copy, and
  GridFunction.Set(old_gridfunction).
Right variant : one mesh, one space, mesh.SetDeformation(gfdef) with
  gfdef = s*(x, y, z); the coefficient vector is untouched.

Observed on NGSolve 6.2.2604, OCC sphere maxh=0.4, Curve(3), H1 order 2 on
mesh.Boundaries('.*'), c = x*y + z (2026-08-06):
  ALE, s = 0.05/0.10/0.20: area and Integrate(c*c*ds) both scale by exactly
    the geometric factor (1+s)^2, to 3.1e-05 / 6.0e-05 / 1.1e-04 relative
    (that residue is the Curve(3) geometry error, not dissipation);
    UnsetDeformation restores the undeformed measures.
  REMESH: ndof goes 381 -> 426 / 463 / 509, so the raw copy raises
    NgException 'BaseVector::Set: size of me = 426 != size of other = 381',
    and gf_new.Set(gf_old) returns WITHOUT raising having produced the
    exactly-zero vector - 100% of the solution gone, silently.

CORRECTION to the claim text: the loss is not "~1-3% of L^2 norm per step".
Across a re-mesh the transfer either raises or silently yields zero.
"""
from __future__ import annotations

import sys

from netgen.occ import OCCGeometry, Pnt, Sphere
from ngsolve import (CoefficientFunction, GridFunction, H1, Integrate, Mesh,
                     VectorH1, ds, x, y, z)

STEPS = (0.05, 0.10, 0.20)


def build(radius):
    geo = OCCGeometry(Sphere(Pnt(0, 0, 0), r=radius))
    mesh = Mesh(geo.GenerateMesh(maxh=0.4))
    mesh.Curve(3)
    return mesh


def main() -> int:
    ok = True

    mesh = build(1.0)
    surf = mesh.Boundaries(".*")
    fes = H1(mesh, order=2, definedon=surf)
    gfc = GridFunction(fes)
    gfc.Set(x * y + z, definedon=surf)
    area0 = Integrate(CoefficientFunction(1) * ds, mesh)
    l2sq0 = Integrate(gfc * gfc * ds, mesh)
    ndof0 = fes.ndof
    print(f"base_ndof={ndof0} base_ne={mesh.ne}")

    # ---- RIGHT: ALE, same DOFs, deformed geometry -----------------------
    gfdef = GridFunction(VectorH1(mesh, order=2))
    ale_worst = 0.0
    for s in STEPS:
        gfdef.Set(CoefficientFunction((s * x, s * y, s * z)))
        mesh.SetDeformation(gfdef)
        area = Integrate(CoefficientFunction(1) * ds, mesh)
        l2sq = Integrate(gfc * gfc * ds, mesh)
        mesh.UnsetDeformation()
        fac = (1.0 + s) ** 2
        ale_worst = max(ale_worst,
                        abs(area / (fac * area0) - 1.0),
                        abs(l2sq / (fac * l2sq0) - 1.0))
    print(f"ale_ndof_unchanged={fes.ndof == ndof0}")
    print(f"ale_scales_with_geometric_factor={ale_worst < 1e-3}")
    restored = Integrate(CoefficientFunction(1) * ds, mesh)
    print(f"ale_unset_deformation_restores={abs(restored - area0) < 1e-12}")
    if not (ale_worst < 1e-3 and fes.ndof == ndof0):
        print(f"FAIL: ALE did not transport the field exactly; worst "
              f"relative deviation {ale_worst:.3e}", file=sys.stderr)
        ok = False

    # ---- WRONG: rebuild the geometry and the space every step -----------
    ndof_changed = True
    copy_msg = ""
    set_raised = ""
    worst_kept = 1.0
    for s in STEPS:
        m_new = build(1.0 + s)
        surf_new = m_new.Boundaries(".*")
        fes_new = H1(m_new, order=2, definedon=surf_new)
        ndof_changed = ndof_changed and (fes_new.ndof != ndof0)
        g_new = GridFunction(fes_new)
        try:
            g_new.vec.data = gfc.vec
        except Exception as exc:                   # noqa: BLE001
            copy_msg = str(exc)
        try:
            g_new.Set(gfc, definedon=surf_new)
        except Exception as exc:                   # noqa: BLE001
            set_raised = f"{type(exc).__name__}: {exc}"
        kept = Integrate(g_new * g_new * ds, m_new) / ((1.0 + s) ** 2 * l2sq0)
        worst_kept = min(worst_kept, kept)
        print(f"remesh_step_s={s} new_ndof={fes_new.ndof} "
              f"fraction_of_L2_kept={kept:.6f}")

    print(f"remesh_ndof_changed={ndof_changed}")
    print(f"remesh_raw_dof_copy_raised={bool(copy_msg)}")
    print(f"remesh_copy_msg={copy_msg!r}")
    print(f"remesh_cross_mesh_set_raised={bool(set_raised)}")
    print(f"remesh_cross_mesh_set_lost_everything={worst_kept == 0.0}")
    print("remesh_loss_is_total_not_a_few_percent="
          f"{worst_kept < 0.01}")
    if not ndof_changed:
        print("FAIL: re-meshing the grown sphere kept the same ndof, so the "
              "continuity pitfall cannot arise here", file=sys.stderr)
        ok = False
    if "BaseVector::Set: size of me" not in copy_msg:
        print(f"FAIL: the raw coefficient-vector copy across meshes did not "
              f"raise the documented NgException; got {copy_msg!r}",
              file=sys.stderr)
        ok = False
    if set_raised:
        print(f"FAIL: cross-mesh Set raised instead of silently losing the "
              f"field: {set_raised}", file=sys.stderr)
        ok = False
    if worst_kept >= 0.01:
        print(f"FAIL: re-meshing kept {worst_kept:.4f} of the L2 norm; the "
              f"transfer pathology is absent", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
