"""Tier-2 for fenics contact#0: DOLFINx ships no contact object of any kind —
no solver, no contact boundary, no gap function. The penalty residual has to be
built by hand.

Wrong variant: reaching for a contact class the way a structural-mechanics
package would offer one. dolfinx.fem.ContactBoundary and
dolfinx.fem.ContactProblem both fail with
"AttributeError: module 'dolfinx.fem' has no attribute 'Contact...'", and
filtering dir(dolfinx), dir(dolfinx.fem) and dir(dolfinx.fem.petsc) for names
containing 'ontact' returns the empty list in every case. The separate
dolfinx_contact package is not part of a standard installation either:
importing it raises ModuleNotFoundError.

The fixture also shows the cure is real, not hypothetical: the hand-written
penalty residual gamma*max(phi - u, 0)*v*dx compiles and assembles into a
non-zero vector on the same installation.

Mutation control: T2_MUTATE=1 asks for dolfinx.fem.Function instead, an
attribute that does exist, so no AttributeError is produced.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    wanted = (["Function", "Constant"] if MUTATE
              else ["ContactBoundary", "ContactProblem"])
    n_raised = 0
    for name in wanted:
        try:
            getattr(dolfinx.fem, name)
            print(f"lookup_{name}=found")
        except AttributeError as exc:
            n_raised += 1
            print(f"lookup_{name}_AttributeError: {exc}")
    print(f"contact_class_lookups_that_raised={n_raised}")

    hits = {
        "dolfinx": [x for x in dir(dolfinx) if "ontact" in x],
        "dolfinx.fem": [x for x in dir(dolfinx.fem) if "ontact" in x],
        "dolfinx.fem.petsc": [x for x in dir(dolfinx.fem.petsc)
                              if "ontact" in x],
    }
    for mod, found in hits.items():
        print(f"names_containing_contact_in_{mod}={found}")
    no_contact_names = all(not v for v in hits.values())
    print(f"no_contact_names_anywhere={no_contact_names}")

    try:
        import dolfinx_contact  # noqa: F401
        print("dolfinx_contact_import=succeeded")
        addon_absent = False
    except ModuleNotFoundError as exc:
        print(f"dolfinx_contact_import_ModuleNotFoundError: {exc}")
        addon_absent = True
    print(f"dolfinx_contact_addon_absent={addon_absent}")

    # The documented cure: assemble the penalty residual by hand.
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    phi = dolfinx.fem.Constant(msh, -0.2)
    gamma = dolfinx.fem.Constant(msh, 1.0e4)
    u.x.array[:] = -0.5           # fully penetrated state
    arg = phi - u
    smooth_max = (arg + ufl.sqrt(arg ** 2 + 1.0e-12)) / 2.0
    r = dolfinx.fem.petsc.assemble_vector(
        dolfinx.fem.form(gamma * smooth_max * v * ufl.dx))
    r.ghostUpdate(addv=1, mode=2)
    hand_norm = float(np.linalg.norm(r.array))
    hand_ok = np.isfinite(hand_norm) and hand_norm > 0.0
    print(f"hand_written_penalty_residual_assembles={hand_ok}")

    if n_raised == 2 and no_contact_names and addon_absent and hand_ok:
        print("VERDICT=no_contact_api_build_the_penalty_by_hand")
        return 0
    print("VERDICT=contact_api_present")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
