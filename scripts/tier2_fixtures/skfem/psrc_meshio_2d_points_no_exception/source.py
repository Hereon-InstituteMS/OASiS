"""Tier-2: meshio does not raise on 2D points -- it warns on stdout and writes.

Claim: skfem point_source#5 -- VTK output for a 2D MeshTri wants 3D-padded
points; "do NOT rely on an exception to catch the missing z-column: meshio
5.3.5 does not raise on an (N, 2) points array -- meshio.Mesh constructs fine
and the writer prints 'Warning: VTU requires 3D points, but 2D points given.
Appending 0 third component.' and writes the file anyway.  Signal: that
warning line on stdout (not stderr, not an exception) at the meshio write
call, and a .vtu whose point array has 3 columns you never supplied; there is
no 'expected ndarray of shape (N, 3)' ValueError from the meshio.Mesh
constructor."

Measured on skfem 12.0.1 / meshio 5.3.5, MeshTri().refined(3) carrying a
point-source solution as point data.  The entry is right in every particular,
and this fixture adds the two things that make the failure hard to catch:

  * the notice is not a Python warning -- warnings.catch_warnings records
    nothing -- so `warnings.simplefilter("error")` will not turn it into an
    exception;
  * it is written straight to the process stdout, past
    contextlib.redirect_stdout, so a caller cannot capture it that way
    either.

The padded call produces a byte-identical point array, which is the check
worth writing: compare what came back against what you meant to send.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology
site -- the first meshio.Mesh is built from z-padded (N, 3) points instead of
the raw (N, 2) m.p.T array.  meshio then has nothing to append, so the notice
'VTU requires 3D points, but 2D points given' is never printed and
'meshio_added_a_column_you_never_supplied=True' disappears from the output.
Re-run: T2_MUTATE=1 python source.py
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
from skfem import Basis, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    ok = True
    print(f"meshio_version={meshio.__version__}")
    m = MeshTri().refined(3)
    ib = Basis(m, ElementTriP1())
    K = laplace.assemble(ib)
    D = ib.get_dofs().all()
    f = ib.zeros()
    f[int(np.argmin((ib.doflocs[0] - 0.5) ** 2
                    + (ib.doflocs[1] - 0.5) ** 2))] = 1.0
    u = solve(*condense(K, f, D=D))
    print(f"dofs_N={ib.N} mesh_p_shape={m.p.shape}")
    d = tempfile.mkdtemp()

    # --- the unpadded call ------------------------------------------------
    err = None
    ctor_err = None
    buf = io.StringIO()
    path = os.path.join(d, "unpadded.vtu")
    # THE PATHOLOGY: the raw (N, 2) point array goes straight to meshio.
    # Under T2_MUTATE the documented fix is applied here -- the z column is
    # appended before the mesh is built.
    supplied = (m.p.T if not MUTATE
                else np.column_stack([m.p.T, np.zeros(m.p.shape[1])]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            mio = meshio.Mesh(supplied, [("triangle", m.t.T)],
                              point_data={"u": u})
        except Exception as e:                       # noqa: BLE001
            ctor_err = e
            mio = None
        if mio is not None:
            try:
                with contextlib.redirect_stdout(buf):
                    meshio.write(path, mio)
            except Exception as e:                   # noqa: BLE001
                err = e
        msgs = sorted({str(c.message) for c in caught})
    print(f"mesh_constructor_raises={ctor_err is not None}")
    print(f"writer_raises={err is not None}")
    print(f"python_warnings_recorded={msgs!r}")
    print(f"notice_is_not_a_python_warning={not msgs}")
    print(f"redirect_stdout_captured={buf.getvalue().strip()!r}")
    print(f"notice_escapes_redirect_stdout={buf.getvalue().strip() == ''}")
    print(f"file_written={os.path.isfile(path)}")
    if ctor_err is not None or err is not None:
        print(f"FAIL: the unpadded call raised "
              f"{type(ctor_err or err).__name__}", file=sys.stderr)
        ok = False
    if msgs:
        print(f"FAIL: a Python warning WAS recorded: {msgs!r}",
              file=sys.stderr)
        ok = False

    back = meshio.read(path)
    print(f"unpadded_read_back_points_shape={back.points.shape}")
    print(f"supplied_columns={supplied.shape[1]}")
    print(f"read_back_has_three_columns={back.points.shape[1] == 3}")
    print(f"third_column_is_all_zero="
          f"{bool(np.allclose(back.points[:, 2], 0.0))}")
    print(f"meshio_added_a_column_you_never_supplied="
          f"{back.points.shape[1] > supplied.shape[1]}")
    if back.points.shape[1] != 3:
        print("FAIL: meshio did not append the third component",
              file=sys.stderr)
        ok = False

    # --- the padded call --------------------------------------------------
    pts = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
    path2 = os.path.join(d, "padded.vtu")
    meshio.write(path2, meshio.Mesh(pts, [("triangle", m.t.T)],
                                    point_data={"u": u}))
    back2 = meshio.read(path2)
    print(f"padded_read_back_points_shape={back2.points.shape}")
    print(f"padded_matches_what_was_sent="
          f"{bool(np.allclose(back2.points, pts))}")
    print(f"padded_and_unpadded_agree="
          f"{bool(np.allclose(back.points, back2.points))}")
    print(f"point_data_round_trips="
          f"{bool(np.allclose(back2.point_data['u'], u))}")
    if not np.allclose(back2.points, pts):
        print("FAIL: the padded write did not round-trip its points",
              file=sys.stderr)
        ok = False

    # --- the ValueError the entry says is absent --------------------------
    text = str(ctor_err) if ctor_err else ""
    print(f"expected_ndarray_shape_valueerror_present="
          f"{'expected ndarray of shape' in text}")
    if "expected ndarray of shape" in text:
        print("FAIL: the constructor DID raise the shape ValueError",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
