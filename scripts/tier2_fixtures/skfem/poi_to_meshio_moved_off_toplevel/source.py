"""Tier-2: to_meshio moved off the top-level skfem namespace.

Claim: skfem poisson#3 -- VTU output in skfem 12.x goes through
skfem.io.meshio.to_meshio(mesh, point_data=...). The function was NOT removed,
only relocated: hasattr(skfem, 'to_meshio') is False,
hasattr(skfem.io.meshio, 'to_meshio') is True.

Wrong variant: the legacy top-level spelling skfem.to_meshio(...), which raises
AttributeError. This fixture also checks the cell-type translation and a real
.vtu round-trip, so it catches a silent regression in either.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import skfem
import skfem.io.meshio as skfem_meshio
from skfem import Basis, ElementQuad1, MeshQuad, MeshTri


def main() -> int:
    ok = True

    # --- WRONG variant: the legacy top-level spelling -------------------
    print(f"toplevel_has_to_meshio={hasattr(skfem, 'to_meshio')}")
    raised = ""
    try:
        skfem.to_meshio                # noqa: B018 - deliberate probe
    except AttributeError as exc:
        raised = str(exc)
    print(f"toplevel_attributeerror={raised!r}")
    if "no attribute 'to_meshio'" not in raised:
        print(f"FAIL: expected the top-level AttributeError, got {raised!r}",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant --------------------------------------------------
    print(f"io_meshio_has_to_meshio={hasattr(skfem_meshio, 'to_meshio')}")
    if not hasattr(skfem_meshio, "to_meshio"):
        print("FAIL: skfem.io.meshio.to_meshio is gone too", file=sys.stderr)
        ok = False
        return 2

    mq = MeshQuad().refined(2)
    mt = MeshTri().refined(2)
    quad_type = skfem_meshio.to_meshio(mq).cells[0].type
    tri_type = skfem_meshio.to_meshio(mt).cells[0].type
    print(f"quad_cell_type={quad_type}")
    print(f"tri_cell_type={tri_type}")
    if quad_type != "quad" or tri_type != "triangle":
        print(f"FAIL: cell-type translation gave {quad_type!r} / {tri_type!r}",
              file=sys.stderr)
        ok = False

    basis = Basis(mq, ElementQuad1())
    values = basis.project(lambda x: x[0] + 2.0 * x[1])
    mesh_io = skfem_meshio.to_meshio(mq, point_data={"u": values})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "result.vtu"
        mesh_io.write(str(path))
        wrote = path.is_file() and path.stat().st_size > 0
        print(f"vtu_written={wrote}")
        import meshio
        back = meshio.read(str(path))
        preserved = ("u" in back.point_data
                     and np.allclose(back.point_data["u"], values, atol=1e-6))
    print(f"vtu_roundtrip_point_data_preserved={preserved}")
    if not (wrote and preserved):
        print("FAIL: the .vtu round-trip lost the point data", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
