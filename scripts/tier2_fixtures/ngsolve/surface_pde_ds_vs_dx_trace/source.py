"""Tier-2: surface integrals need ds AND .Trace(); dx raises, it does not
silently give zero.

Claim: ngsolve surface_pde#0 - on the shipped generator's own geometry
(OCCGeometry(Sphere(...)) -> 3-D volume mesh, H1 restricted to
mesh.Boundaries('.*')) a BilinearForm written with *dx raises
NgException('Trialfunction does not support VOL-forms, maybe a Trace()
operator is missing, type = gradboundary'); the same form with *ds
assembles.  `mesh dim mismatch in BilinearForm` is NOT a string NGSolve
emits.  dx on that mesh is still meaningful for VOLUME quantities.

Wrong variant A: grad(u).Trace() * grad(v).Trace() * dx
Wrong variant B: grad(u) * grad(v) * ds          (ds, but .Trace() omitted)
Right variant  : grad(u).Trace() * grad(v).Trace() * ds

Observed on NGSolve 6.2.2604, OCC sphere maxh=0.4, Curve(4), H1 order 3 on
mesh.Boundaries('.*') (2026-08-06).  Both wrong variants raise at the `+=`
statement, before Assemble() is ever reached:
  A -> NgException: Trialfunction does not support VOL-forms, maybe a
       Trace() operator is missing, type = gradboundary
  B -> NgException: Trialfunction does not support BND-forms, maybe a
       Trace() operator is missing, type = grad
The right variant assembles (nnz = 14384); Integrate(1*dx) = 4.188798
(= 4/3 pi, the ball volume) and Integrate(1*ds) = 12.566386 (= 4 pi).
"""
from __future__ import annotations

import math
import sys

from netgen.occ import OCCGeometry, Pnt, Sphere
from ngsolve import (BilinearForm, CoefficientFunction, H1, Integrate, Mesh,
                     ds, dx, grad)


def main() -> int:
    ok = True

    geo = OCCGeometry(Sphere(Pnt(0, 0, 0), r=1.0))
    mesh = Mesh(geo.GenerateMesh(maxh=0.4))
    mesh.Curve(4)
    fes = H1(mesh, order=3, definedon=mesh.Boundaries(".*"))
    u_c, v_c = fes.TnT()
    print(f"mesh_dim={mesh.dim} has_volume_elements={mesh.ne > 0} "
          f"surface_ndof={fes.ndof}")

    # ---- WRONG A: volume measure on a surface-restricted space ----------
    stage = "iadd"
    msg_dx = ""
    nnz_dx = -1
    try:
        a_dx = BilinearForm(fes)
        a_dx += grad(u_c).Trace() * grad(v_c).Trace() * dx
        stage = "assemble"
        a_dx.Assemble()
        nnz_dx = a_dx.mat.nze
    except Exception as exc:                       # noqa: BLE001
        msg_dx = str(exc)
    print(f"dx_form_raised={bool(msg_dx)} at_stage={stage}")
    print(f"dx_msg={msg_dx!r}")
    print("dx_msg_has_VOL_forms="
          f"{'does not support VOL-forms' in msg_dx}")
    print(f"dx_msg_has_gradboundary={'type = gradboundary' in msg_dx}")
    print("dx_msg_has_fabricated_dim_mismatch="
          f"{'mesh dim mismatch in BilinearForm' in msg_dx}")
    print(f"dx_form_silently_gave_zero_matrix={nnz_dx == 0}")
    if not msg_dx:
        print("FAIL: grad(u).Trace()*grad(v).Trace()*dx did not raise on a "
              "surface-restricted H1 space", file=sys.stderr)
        ok = False
    elif "does not support VOL-forms" not in msg_dx:
        print(f"FAIL: unexpected exception text {msg_dx!r}", file=sys.stderr)
        ok = False

    # ---- WRONG B: ds, but .Trace() omitted ------------------------------
    msg_nt = ""
    try:
        a_nt = BilinearForm(fes)
        a_nt += grad(u_c) * grad(v_c) * ds
        a_nt.Assemble()
    except Exception as exc:                       # noqa: BLE001
        msg_nt = str(exc)
    print(f"no_trace_ds_form_raised={bool(msg_nt)}")
    print(f"no_trace_msg={msg_nt!r}")
    print("no_trace_msg_has_BND_forms="
          f"{'does not support BND-forms' in msg_nt}")
    if "does not support BND-forms" not in msg_nt:
        print(f"FAIL: omitting .Trace() under ds did not raise the expected "
              f"NgException; got {msg_nt!r}", file=sys.stderr)
        ok = False

    # ---- RIGHT: ds with .Trace() ----------------------------------------
    a_ok = BilinearForm(fes)
    a_ok += grad(u_c).Trace() * grad(v_c).Trace() * ds
    a_ok.Assemble()
    print(f"ds_trace_form_assembles=True ds_trace_nnz={a_ok.mat.nze}")
    print(f"ds_trace_nnz_gt_0={a_ok.mat.nze > 0}")
    if a_ok.mat.nze <= 0:
        print("FAIL: the ds + .Trace() form assembled an empty matrix",
              file=sys.stderr)
        ok = False

    # ---- dx is still meaningful for VOLUME quantities on this mesh ------
    vol = Integrate(CoefficientFunction(1) * dx, mesh)
    area = Integrate(CoefficientFunction(1) * ds, mesh)
    vol_ex = 4.0 * math.pi / 3.0
    area_ex = 4.0 * math.pi
    print(f"volume_dx_matches_ball_volume={abs(vol / vol_ex - 1) < 1e-3}")
    print(f"surface_ds_matches_4pi={abs(area / area_ex - 1) < 1e-3}")
    print(f"volume_dx_is_not_the_surface_area={abs(vol / area_ex - 1) > 0.5}")
    if not (abs(vol / vol_ex - 1) < 1e-3 and abs(area / area_ex - 1) < 1e-3):
        print(f"FAIL: dx={vol:.6f} (expected {vol_ex:.6f}), "
              f"ds={area:.6f} (expected {area_ex:.6f})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
