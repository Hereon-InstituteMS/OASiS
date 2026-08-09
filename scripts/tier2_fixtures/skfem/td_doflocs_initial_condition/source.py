"""Tier-2: initial conditions must be built on ib.doflocs, not on mesh.p.

Claim: skfem time_dependent#6 -- ib.doflocs is the (ndim, N) coordinate array
used for initial-condition assignment.  "Setting u_init from a callable like
u_init = np.sin(np.pi * ib.mesh.p[0]) (using mesh.p directly) only matches
DOFs for P1 elements; for ElementVector / P2 / higher-order the DOFs include
face/edge midpoints and the initial condition is silently zero on those DOFs.
Use ib.doflocs ... and ib.project / ib.interpolator for IC assignment."

Measured on skfem 12.0.1 on a 4x4 MeshTri.init_tensor:

  * SHAPES: doflocs is (2, N) for every element tested.  For ElementTriP1
    N equals the vertex count and doflocs IS mesh.p; for ElementTriP2 N is
    larger than the vertex count, and for ElementVector(ElementTriP1) it is
    dim times larger.
  * doflocs[:, :n_vertices] equals mesh.p exactly, which is what makes the
    silent version possible: a length-n_vertices array assigned into the
    first slots of a zero vector is dimensionally plausible and leaves the
    remaining DOFs at zero.
  * THE FAILURE IS SILENT AND LARGE.  With P2 the mesh.p-based IC leaves the
    majority of DOFs at zero; nothing raises, nothing warns, and the field is
    wrong by an O(1) amount against ib.project of the same callable.
  * A DIRECT ASSIGNMENT of the short array is the loud version: it raises
    ValueError about the shape mismatch.  Only the zero-padded form is
    silent, so the guard has to be a length check against ib.N, not a
    try/except.

Mutation control: T2_MUTATE=1 applies the entry's own fix at the pathology
site -- the zero-padded initial condition is built by evaluating the callable
on ib.doflocs over all ib.N slots instead of writing field(mesh.p) into the
first n_vertices slots.  No DOF is then left at zero by accident and the field
agrees with ib.project.  Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys
import warnings

MUTATE = os.environ.get("T2_MUTATE") == "1"

import numpy as np
from skfem import (
    Basis,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    MeshTri,
)
from skfem.models.poisson import mass


def field(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def main() -> int:
    ok = True
    m = MeshTri.init_tensor(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    nverts = m.p.shape[1]
    print(f"mesh_p_shape={m.p.shape} n_vertices={nverts}")

    bases = {
        "P1": Basis(m, ElementTriP1()),
        "P2": Basis(m, ElementTriP2()),
        "vecP1": Basis(m, ElementVector(ElementTriP1())),
    }
    for tag, ib in bases.items():
        print(f"{tag}_N={ib.N} doflocs_shape={ib.doflocs.shape}")
        print(f"{tag}_doflocs_is_ndim_by_N="
              f"{ib.doflocs.shape == (m.p.shape[0], ib.N)}")
        print(f"{tag}_N_equals_vertex_count={ib.N == nverts}")
        if ib.doflocs.shape != (m.p.shape[0], ib.N):
            print(f"FAIL: {tag} doflocs is not (ndim, N)", file=sys.stderr)
            ok = False

    ib1, ib2 = bases["P1"], bases["P2"]
    print(f"p1_doflocs_equals_mesh_p="
          f"{bool(np.allclose(ib1.doflocs, m.p))}")
    print(f"p2_doflocs_prefix_equals_mesh_p="
          f"{bool(np.allclose(ib2.doflocs[:, :nverts], m.p))}")
    print(f"p2_extra_dofs={ib2.N - nverts}")
    if not np.allclose(ib1.doflocs, m.p):
        print("FAIL: P1 doflocs did not equal mesh.p", file=sys.stderr)
        ok = False
    if not np.allclose(ib2.doflocs[:, :nverts], m.p):
        print("FAIL: the P2 doflocs prefix did not equal mesh.p; the silent "
              "padded assignment could not arise", file=sys.stderr)
        ok = False

    # --- the loud version: a direct assignment ---------------------------
    err = None
    try:
        u = ib2.zeros()
        u[:] = field(m.p)
    except ValueError as e:
        err = e
    print(f"direct_short_assignment_raises={err is not None}")
    print(f"direct_short_assignment_message={str(err)[:70]!r}")
    if err is None:
        print("FAIL: assigning a vertex-length array over ib.N did not raise",
              file=sys.stderr)
        ok = False

    # --- the silent version: pad with zeros ------------------------------
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        u_padded = ib2.zeros()
        if MUTATE:                       # the fix: evaluate on ib.doflocs
            u_padded[:] = field(ib2.doflocs)
        else:                            # the pathology: mesh.p, zero-padded
            u_padded[:nverts] = field(m.p)
        u_project = ib2.project(field)
        msgs = sorted({str(c.message) for c in caught})
    zeros = int((u_padded == 0.0).sum())
    print(f"padded_ic_zero_dofs={zeros} of {ib2.N}")
    print(f"padded_ic_majority_of_dofs_are_zero={zeros > ib2.N // 2}")
    print(f"padded_ic_warnings={msgs!r}")
    print(f"padded_ic_is_silent={not msgs}")
    if msgs:
        print(f"FAIL: the padded assignment warned {msgs!r}", file=sys.stderr)
        ok = False
    if zeros <= ib2.N // 2:
        print("FAIL: the padded IC did not leave most DOFs at zero",
              file=sys.stderr)
        ok = False

    # --- and the field is wrong by an O(1) amount ------------------------
    M = mass.assemble(ib2)
    d = u_padded - u_project
    err_l2 = float(np.sqrt(d @ (M @ d)))
    ref_l2 = float(np.sqrt(u_project @ (M @ u_project)))
    print(f"padded_vs_project_l2_error={err_l2:.4e}")
    print(f"project_l2_norm={ref_l2:.4e}")
    print(f"relative_error={err_l2 / ref_l2:.4f}")
    print(f"padded_ic_is_wrong_by_order_one={err_l2 > 0.2 * ref_l2}")
    print(f"padded_ic_max_abs_deviation={float(np.abs(d).max()):.4f}")
    if err_l2 <= 0.2 * ref_l2:
        print("FAIL: the padded IC was close enough to the projection",
              file=sys.stderr)
        ok = False

    # --- doflocs-based evaluation is the fix ------------------------------
    u_doflocs = field(ib2.doflocs)
    print(f"doflocs_ic_length={u_doflocs.shape[0]} equals_N="
          f"{u_doflocs.shape[0] == ib2.N}")
    d2 = u_doflocs - u_project
    print(f"doflocs_vs_project_l2_error={float(np.sqrt(d2 @ (M @ d2))):.4e}")
    print(f"doflocs_ic_is_close_to_project="
          f"{float(np.sqrt(d2 @ (M @ d2))) < 0.05 * ref_l2}")
    if u_doflocs.shape[0] != ib2.N:
        print("FAIL: evaluating on doflocs did not give an ib.N-long vector",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
