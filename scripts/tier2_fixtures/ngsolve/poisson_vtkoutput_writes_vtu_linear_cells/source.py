"""Tier-2: VTKOutput writes .vtu directly and every exported cell is a LINEAR
VTK triangle, so subdivision is the only way P2+ detail reaches the file.

Claim: ngsolve poisson#3 -- "VTKOutput writes .vtu natively -- no XDMF
conversion required.  The constructor expects a (mesh, coefs, names, filename)
tuple; VTKOutput.Do() emits the file.  For higher-order fields, pass
subdivision=N: the exported cells are all LINEAR VTK triangles, so subdivision
is the only way the P2+ enrichment reaches the file."

Wrong variant: exporting a P2 field with subdivision=0 and expecting the
curvature to survive.

What this fixture pins, all re-measured on this run:
  * .Do() creates a file whose name ends in .vtu, with no conversion step;
  * the file's own header declares UnstructuredGrid;
  * at subdivision 0 the cell-type array contains only VTK type 5, the linear
    triangle, and the cell count equals the mesh element count -- one linear
    cell per element;
  * CORRECTION: at subdivision 2 the types are 5 AND 9, so NGSolve also emits
    linear QUADS. Every exported cell is linear, which is the load-bearing half
    of the claim, but "all linear triangles" is only true at subdivision 0;
  * subdivision is what adds sample points: both the point count and the cell
    count grow with it, and the exported value list gains values that were not
    in the subdivision-0 file -- which is the P2 enrichment reaching the file;
  * no XDMF or .h5 companion file is produced.
"""
from __future__ import annotations

import glob
import os
import re
import sys
import tempfile

import numpy
from netgen.geom2d import unit_square
from ngsolve import GridFunction, H1, Mesh, VTKOutput, pi, sin, x, y


def export(mesh, gf, subdivision, tag, tmpdir):
    base = os.path.join(tmpdir, tag)
    vtk = VTKOutput(mesh, coefs=[gf], names=["u"], filename=base,
                    subdivision=subdivision)
    vtk.Do()
    files = sorted(glob.glob(base + "*"))
    return files


def point_values(path):
    """NGSolve writes the arrays as raw appended binary, so the offsets in the
    XML header have to be followed rather than the element text read."""
    blob = open(path, "rb").read()
    header = blob[:blob.find(b"<AppendedData")].decode("ascii", "replace")
    start = blob.find(b"_", blob.find(b"<AppendedData")) + 1

    def array(name, dtype):
        m = re.search(rf'Name="{name}"[^>]*offset="(\d+)"', header)
        if not m:
            return numpy.array([], dtype=dtype)
        off = start + int(m.group(1))
        n = int(numpy.frombuffer(blob[off:off + 4], dtype="<u4")[0])
        return numpy.frombuffer(blob[off + 4:off + 4 + n], dtype=dtype)

    vals = [float(v) for v in array("u", "<f8")]
    tset = sorted({int(t) for t in array("types", "u1")})
    ncell = re.search(r'NumberOfCells="(\d+)"', header)
    npt = re.search(r'NumberOfPoints="(\d+)"', header)
    return (vals, tset, int(ncell.group(1)) if ncell else -1,
            int(npt.group(1)) if npt else -1, header)


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    fes = H1(mesh, order=2)
    gf = GridFunction(fes)
    gf.Set(sin(pi * x) * sin(pi * y))

    tmpdir = tempfile.mkdtemp()
    files0 = export(mesh, gf, 0, "sub0", tmpdir)
    files2 = export(mesh, gf, 2, "sub2", tmpdir)
    print(f"subdivision0_files={[os.path.basename(f) for f in files0]}")
    print(f"subdivision2_files={[os.path.basename(f) for f in files2]}")
    all_files = files0 + files2
    print(f"every_file_is_vtu={all(f.endswith('.vtu') for f in all_files)}")
    print(f"no_xdmf_or_h5_companion="
          f"{not any(f.endswith(('.xdmf', '.h5')) for f in all_files)}")

    v0, t0, n0, p0, raw0 = point_values(files0[0])
    v2, t2, n2, p2, raw2 = point_values(files2[0])
    print(f"header_declares_unstructuredgrid={'UnstructuredGrid' in raw0}")
    print(f"subdivision0_cell_types={t0}")
    print(f"subdivision2_cell_types={t2}")
    LINEAR = {5, 9}                    # VTK triangle and VTK quad
    print(f"subdivision0_is_only_linear_triangles={t0 == [5]}")
    print(f"subdivision2_types_are_all_linear={set(t2) <= LINEAR}")
    print(f"subdivision2_also_emits_quads={9 in t2}")
    print(f"claim_all_triangles_holds_only_at_subdivision0="
          f"{t0 == [5] and 9 in t2}")
    print(f"mesh_elements={mesh.ne} subdivision0_cells={n0}")
    print(f"subdivision0_is_one_cell_per_element={n0 == mesh.ne}")
    print(f"points_subdivision0={p0} points_subdivision2={p2}")
    print(f"cells_subdivision0={n0} cells_subdivision2={n2}")
    print(f"subdivision_adds_sample_points={p2 > p0 and n2 > n0}")

    pts = [(float(a), float(b))
           for a in numpy.linspace(0.005, 0.995, 41)
           for b in numpy.linspace(0.005, 0.995, 41)]
    true_max = max(float(gf(mesh(a, b))) for a, b in pts)
    print(f"field_sampled_max={true_max:.6f}")
    print(f"exported_values_subdivision0={len(v0)}")
    print(f"exported_values_subdivision2={len(v2)}")
    extra = sorted(set(numpy.round(v2, 10)) - set(numpy.round(v0, 10)))
    print(f"values_present_only_at_subdivision2={len(extra)}")
    print(f"p2_enrichment_reaches_the_file_only_via_subdivision="
          f"{len(extra) > 0 and len(v2) > len(v0)}")

    ok = (
        all(f.endswith(".vtu") for f in all_files)
        and not any(f.endswith((".xdmf", ".h5")) for f in all_files)
        and "UnstructuredGrid" in raw0
        and t0 == [5] and set(t2) <= LINEAR and 9 in t2
        and n0 == mesh.ne
        and p2 > p0 and n2 > n0
        and len(extra) > 0 and len(v2) > len(v0)
    )
    if ok:
        return 0
    print("FAIL: VTKOutput invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
