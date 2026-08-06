"""Tier-2: OCCGeometry on a SOLID gives a volume mesh, not a surface mesh -
and dim=2 does not fix it, it flattens the geometry.

Claim: ngsolve surface_pde#5 - feeding OCCGeometry a shape that still
carries volumetric solids builds a 3-D mesh instead of a surface mesh, and
ds then runs over the volume boundary rather than the intended surface.
Select the faces (geo.faces[i] / Glue(faces)).

Wrong variant A: OCCGeometry(Sphere(...))  - what the shipped
  surface_pde_3d template does.  Volume mesh; a plain H1(mesh, order) lives
  on the ball, not on the sphere.
Wrong variant B: "force" a surface mesh with OCCGeometry(face, dim=2).
Right variant  : OCCGeometry(Glue(sphere.faces)) - a surface-only mesh.

Observed on NGSolve 6.2.2604 / netgen OCC, maxh=0.4, Curve(4) (2026-08-06):
  solid      -> mesh.dim=3, mesh.ne=215 volume elements,
                Integrate(1*dx)=4.188798 (the ball volume),
                Integrate(1*ds)=12.566386, plain H1 order 2 ndof=506
  faces      -> mesh.dim=3, mesh.ne=0, Integrate(1*dx)=0.0,
                Integrate(1*ds)=12.566386, plain H1 order 2 ndof=378;
                grad(u)*grad(v)*dx assembles an EMPTY matrix (nnz=0) and
                NGSolve prints "used dof inconsistency"
  faces,dim=2-> mesh.dim=2 but the vertices are 2-D points: the sphere is
                flattened, Integrate(1*dx)=6.066810 against 4*pi=12.566371
  OCCGeometry(sphere.faces, dim=3) raises TypeError
                "incompatible constructor arguments"

CORRECTION to the claim text: selecting the faces does NOT make Mesh.dim
return 2.  It stays 3; the discriminator is mesh.ne == 0 (no volume
elements).  Passing dim=2 is what returns 2, and it destroys the geometry.
"""
from __future__ import annotations

import math
import sys

from netgen.occ import Glue, OCCGeometry, Pnt, Sphere
from ngsolve import (BilinearForm, CoefficientFunction, H1, Integrate, Mesh,
                     ds, dx, grad)

EXACT_AREA = 4.0 * math.pi
EXACT_VOL = 4.0 * math.pi / 3.0


def build(shape, dim=None):
    geo = OCCGeometry(shape) if dim is None else OCCGeometry(shape, dim=dim)
    mesh = Mesh(geo.GenerateMesh(maxh=0.4))
    mesh.Curve(4)
    return mesh


def main() -> int:
    ok = True
    sphere = Sphere(Pnt(0, 0, 0), r=1.0)
    print(f"n_faces={len(sphere.faces)} n_solids={len(sphere.solids)}")

    # ---- WRONG A: the solid, exactly as the shipped template does -------
    m_solid = build(sphere)
    fes_solid = H1(m_solid, order=2)
    print(f"solid_geo_mesh_dim={m_solid.dim}")
    print(f"solid_geo_has_volume_elements={m_solid.ne > 0}")
    vol = Integrate(CoefficientFunction(1) * dx, m_solid)
    are = Integrate(CoefficientFunction(1) * ds, m_solid)
    print(f"solid_geo_dx_is_ball_volume={abs(vol / EXACT_VOL - 1) < 1e-3}")
    print(f"solid_geo_ds_is_sphere_area={abs(are / EXACT_AREA - 1) < 1e-3}")
    if not (m_solid.ne > 0 and abs(vol / EXACT_VOL - 1) < 1e-3):
        print(f"FAIL: OCCGeometry(Sphere(...)) did not build a volume mesh "
              f"(ne={m_solid.ne}, dx={vol:.6f})", file=sys.stderr)
        ok = False

    # ---- RIGHT: select the faces ----------------------------------------
    m_surf = build(Glue(sphere.faces))
    fes_surf = H1(m_surf, order=2)
    print(f"faces_geo_mesh_dim={m_surf.dim}")
    print(f"faces_geo_n_volume_elements={m_surf.ne}")
    svol = Integrate(CoefficientFunction(1) * dx, m_surf)
    sare = Integrate(CoefficientFunction(1) * ds, m_surf)
    print(f"faces_geo_dx_is_zero={svol == 0.0}")
    print(f"faces_geo_ds_matches_4pi={abs(sare / EXACT_AREA - 1) < 1e-3}")
    print("solid_geo_plain_h1_bigger_than_surface="
          f"{fes_solid.ndof > fes_surf.ndof}")
    if not (m_surf.ne == 0 and abs(sare / EXACT_AREA - 1) < 1e-3):
        print(f"FAIL: Glue(sphere.faces) did not build a surface-only mesh "
              f"(ne={m_surf.ne}, ds={sare:.6f})", file=sys.stderr)
        ok = False
    if fes_solid.ndof <= fes_surf.ndof:
        print(f"FAIL: the volume space ({fes_solid.ndof}) is not larger than "
              f"the surface space ({fes_surf.ndof})", file=sys.stderr)
        ok = False

    # on the surface mesh, dx assembles an EMPTY matrix and NGSolve warns
    a_empty = BilinearForm(fes_surf)
    u, v = fes_surf.TnT()
    a_empty += grad(u) * grad(v) * dx
    a_empty.Assemble()
    print(f"faces_geo_grad_dx_nnz_is_zero={a_empty.mat.nze == 0}")
    if a_empty.mat.nze != 0:
        print(f"FAIL: dx on a surface-only mesh assembled {a_empty.mat.nze} "
              f"nonzeros instead of an empty matrix", file=sys.stderr)
        ok = False

    # ---- WRONG B: dim=2 flattens the surface ----------------------------
    m_flat = build(sphere.faces[0], dim=2)
    pt = m_flat.vertices[0].point
    flat_area = Integrate(CoefficientFunction(1) * dx, m_flat)
    print(f"dim2_mesh_dim={m_flat.dim}")
    print(f"dim2_vertex_point_dimension={len(pt)}")
    print("dim2_area_far_from_4pi="
          f"{abs(flat_area / EXACT_AREA - 1) > 0.4}")
    if not (m_flat.dim == 2 and len(pt) == 2
            and abs(flat_area / EXACT_AREA - 1) > 0.4):
        print(f"FAIL: dim=2 did not flatten the sphere "
              f"(dim={m_flat.dim}, ptdim={len(pt)}, "
              f"dx={flat_area:.6f})", file=sys.stderr)
        ok = False

    # ---- the ListOfShapes overload takes no dim kwarg -------------------
    msg = ""
    try:
        OCCGeometry(sphere.faces, dim=3)
    except Exception as exc:                       # noqa: BLE001
        msg = str(exc)
    print(f"faces_list_with_dim_kwarg_raised={bool(msg)}")
    print(f"faces_list_msg_first_line={msg.splitlines()[0] if msg else ''!r}")
    print("faces_list_msg_has_incompatible_ctor="
          f"{'incompatible constructor arguments' in msg}")
    if "incompatible constructor arguments" not in msg:
        print(f"FAIL: OCCGeometry(faces, dim=3) did not raise the pybind11 "
              f"overload error; got {msg!r}", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
