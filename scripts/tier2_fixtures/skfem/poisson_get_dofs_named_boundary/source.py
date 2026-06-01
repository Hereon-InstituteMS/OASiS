"""Tier-2: scikit-fem named-boundary access + to_meshio location.

Two catalog-drift falsifications in one fixture:

  (1) get_dofs(name) does NOT work on a freshly constructed mesh —
      m.boundaries is None and the call raises
      ValueError: Boundary 'left' not found.
      The named-boundary set must first be registered via
      m.with_boundaries({'left': filter}). The catalog hint
      'ib.get_dofs(\"left\") for named' is incomplete without
      mentioning the with_boundaries prerequisite.

  (2) The catalog says 'For v12+: to_meshio removed — use
      meshio.Mesh() directly'. In skfem 12.0.1, to_meshio is
      removed only from the TOP-LEVEL skfem namespace; it is
      still present as skfem.io.meshio.to_meshio and remains
      the recommended path (preserves cell type, ordering,
      and boundary tags).

This fixture is a single python-mode probe that runs all
assertions and prints them as machine-readable lines for the
expect_in_output matcher.
"""
from __future__ import annotations

import sys

import numpy as np
import skfem
import skfem.io.meshio
from skfem import (
    Basis,
    ElementQuad1,
    MeshQuad,
)


def main() -> int:
    print(f"skfem_version={skfem.__version__}")

    # (1) Fresh mesh: boundaries is None
    m = MeshQuad.init_tensor(np.linspace(0, 1, 5),
                             np.linspace(0, 1, 5))
    print(f"fresh_boundaries={m.boundaries}")

    basis = Basis(m, ElementQuad1())
    raised = False
    try:
        basis.get_dofs("left")
    except ValueError as e:
        raised = "Boundary 'left' not found" in str(e)
        print(f"get_dofs_left_unregistered_raise={raised}")
    if not raised:
        print("ERROR: get_dofs('left') did NOT raise on "
              "unregistered boundary", file=sys.stderr)
        return 2

    # (2) After with_boundaries, get_dofs(name) returns the
    # expected DOFs on a 5x5 quad mesh: 5 vertices on the
    # left edge.
    m_named = m.with_boundaries({
        "left": lambda x: np.abs(x[0]) < 1e-10,
    })
    print(f"named_boundaries={sorted(m_named.boundaries.keys())}")
    basis_named = Basis(m_named, ElementQuad1())
    n_left = len(basis_named.get_dofs("left").flatten())
    print(f"left_dof_count={n_left}")
    # Catalog claim verified: get_dofs() returns all 16
    # boundary DOFs of the 5x5 quad mesh.
    n_all = len(basis.get_dofs().flatten())
    print(f"all_boundary_dof_count={n_all}")

    # (3) to_meshio location check
    has_top_level = hasattr(skfem, "to_meshio")
    print(f"top_level_to_meshio={has_top_level}")
    has_submodule = hasattr(skfem.io.meshio, "to_meshio")
    print(f"submodule_to_meshio={has_submodule}")

    if not (has_submodule and callable(skfem.io.meshio.to_meshio)):
        print("ERROR: skfem.io.meshio.to_meshio missing or "
              "not callable", file=sys.stderr)
        return 2

    # (4) to_meshio call returns a meshio.Mesh
    mio = skfem.io.meshio.to_meshio(
        m_named,
        point_data={"phi": np.zeros(m_named.p.shape[1])},
    )
    cell_type = mio.cells[0].type
    n_cells = mio.cells[0].data.shape[0]
    print(f"meshio_cell_type={cell_type}")
    print(f"meshio_n_cells={n_cells}")

    ok = (raised
          and n_left == 5
          and n_all == 16
          and has_submodule
          and (not has_top_level)
          and cell_type == "quad"
          and n_cells == 16)
    if ok:
        return 0
    print("ERROR: one or more assertions failed", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
