"""Tier-2: meshio's 2D points and wrong-cell-tag behaviour, both directions.

Claim: skfem wave#6 -- VTK output via meshio.Mesh(...) requires 3D points, so a
2D mesh must be padded with a zero z-column; the cells argument is
[('quad', m.t.T)] for MeshQuad and [('triangle', m.t.T)] for MeshTri, and
"passing the wrong cell-type tag produces a silently malformed .vtu file".
Signal: "meshio.write raises WriteError or ParaView shows a zero-thickness
slab".

Measured on skfem 12.0.1 / meshio 5.3.5.  BOTH halves of the signal point the
wrong way round:

  * 2D POINTS DO NOT RAISE.  meshio.Mesh accepts an (N, 2) array and
    meshio.write completes, printing a plain line on stdout -- "Warning: VTU
    requires 3D points, but 2D points given. Appending 0 third component." --
    and the file reads back with a (N, 3) point array meshio supplied itself.
    That notice is plain text on the process stdout: it is not a Python
    warning (an empty catch_warnings record) and it escapes
    contextlib.redirect_stdout, so a caller cannot capture or filter it the
    usual ways.  The case the entry says raises is the silent one.
  * A WRONG CELL TAG USUALLY RAISES.  meshio validates the nodes-per-cell
    count, so tagging quads as 'triangle', 'quad8' or 'line' raises
    WriteError naming the expected shape.  The case the entry says is silent
    is loud.
  * THE GENUINELY SILENT CASE is a wrong tag with the SAME nodes-per-cell:
    tagging 4-node quads as 'tetra' passes the only check meshio makes,
    writes without a murmur, and reads back as tetrahedra -- a
    three-dimensional cell type on a flat mesh.  That is the mis-tag worth
    guarding against, and no exception will tell you about it.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology
site -- the points handed to meshio.Mesh are padded with a zero z-column,
np.column_stack([m.p.T, np.zeros(m.p.shape[1])]), instead of the raw (N, 2)
m.p.T.  meshio then has nothing to complain about and its stdout notice
'VTU requires 3D points, but 2D points given' -- itself an expect_in_output
string -- disappears from the output.  Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import warnings

import meshio
import numpy as np
from skfem import MeshQuad

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    ok = True
    print(f"meshio_version={meshio.__version__}")
    m = MeshQuad.init_tensor(np.linspace(0.0, 1.0, 4),
                             np.linspace(0.0, 1.0, 4))
    print(f"mesh_p_shape={m.p.shape} mesh_t_shape={m.t.shape}")
    print(f"points_are_2d={m.p.shape[0] == 2}")
    d = tempfile.mkdtemp()

    # --- 2D points --------------------------------------------------------
    err = None
    buf = io.StringIO()
    path2d = os.path.join(d, "twod.vtu")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            # THE PATHOLOGY: the raw (N, 2) point array.  Under T2_MUTATE=1
            # the documented fix is applied here and the points are padded
            # with a zero z-column before meshio ever sees them.
            pts = (np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
                   if MUTATE else m.p.T)
            mio = meshio.Mesh(pts, [("quad", m.t.T)])
            # meshio prints its notice straight to the process stdout, past
            # contextlib.redirect_stdout, so it lands in this run's own
            # output rather than in `buf` -- which is itself the point: it is
            # not a warning object and not an exception, just text.
            meshio.write(path2d, mio)
        except Exception as e:                       # noqa: BLE001
            err = e
        wmsgs = sorted({str(c.message) for c in caught})
    printed = buf.getvalue()
    print(f"twod_points_raise={err is not None}")
    print(f"twod_points_exception={type(err).__name__ if err else None}")
    print(f"twod_points_python_warnings={wmsgs!r}")
    print(f"twod_points_redirect_stdout_captured={printed.strip()!r}")
    print(f"twod_points_notice_escapes_redirect_stdout="
          f"{err is None and printed.strip() == ''}")
    print(f"twod_points_no_exception_and_no_python_warning="
          f"{err is None and not wmsgs}")
    print(f"twod_points_file_written={os.path.isfile(path2d)}")
    if err is not None:
        print(f"FAIL: 2D points raised {type(err).__name__}: {err}",
              file=sys.stderr)
        ok = False
    else:
        back = meshio.read(path2d)
        print(f"twod_points_read_back_shape={back.points.shape}")
        print(f"meshio_supplied_the_third_column="
              f"{back.points.shape[1] == 3}")
        print(f"third_column_is_zero="
              f"{bool(np.allclose(back.points[:, 2], 0.0))}")
        if back.points.shape[1] != 3:
            print("FAIL: meshio did not pad to 3 columns", file=sys.stderr)
            ok = False

    # --- wrong cell tags --------------------------------------------------
    pts3d = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
    loud, silent = [], []
    for tag in ("quad", "triangle", "quad8", "line", "tetra"):
        p = os.path.join(d, f"{tag}.vtu")
        e2 = None
        try:
            meshio.write(p, meshio.Mesh(pts3d, [(tag, m.t.T)]))
        except Exception as exc:                     # noqa: BLE001
            e2 = exc
        if e2 is None:
            back = meshio.read(p)
            types = [c.type for c in back.cells]
            print(f"tag_{tag}_wrote_ok=True read_back_types={types}")
            silent.append(tag)
        else:
            print(f"tag_{tag}_exception={type(e2).__name__}")
            print(f"tag_{tag}_message={str(e2)[:90]!r}")
            loud.append(tag)

    print(f"wrong_tags_that_raise={sorted(t for t in loud)}")
    print(f"wrong_tags_that_are_silent="
          f"{sorted(t for t in silent if t != 'quad')}")
    print(f"mismatched_node_count_is_loud="
          f"{set(loud) == {'triangle', 'quad8', 'line'}}")
    print(f"same_node_count_wrong_tag_is_silent={'tetra' in silent}")
    if "tetra" not in silent:
        print("FAIL: tagging quads as tetra did NOT write silently",
              file=sys.stderr)
        ok = False
    if set(loud) != {"triangle", "quad8", "line"}:
        print(f"FAIL: the loud set was {sorted(loud)}, not the three "
              f"node-count mismatches", file=sys.stderr)
        ok = False

    # --- and the silent mis-tag really is malformed ----------------------
    back = meshio.read(os.path.join(d, "tetra.vtu"))
    types = [c.type for c in back.cells]
    print(f"tetra_file_cell_types={types}")
    print(f"tetra_file_claims_3d_cells_on_flat_mesh="
          f"{'tetra' in types and bool(np.allclose(back.points[:, 2], 0.0))}")
    if "tetra" not in types:
        print("FAIL: the mis-tagged file did not read back as tetra",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
