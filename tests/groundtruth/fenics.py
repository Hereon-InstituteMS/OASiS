"""Ground-truth probes for FEniCSx / dolfinx.

FEniCSx exposes its API as a Python package tree rooted at
``dolfinx``: mesh constructors under ``dolfinx.mesh``, function-space
machinery under ``dolfinx.fem``, IO under ``dolfinx.io``, PETSc
integration under ``dolfinx.fem.petsc``, and so on.  The catalog
references these by their dotted path (``dolfinx.mesh.create_rectangle``,
``dolfinx.fem.functionspace``, ``dolfinx.io.XDMFFile``), so the
verification is a per-path attribute check.

Unlike NGSolve / scikit-fem, dolfinx is rarely pip-installable in a
clean way -- the canonical install is conda, with optional system
wheel availability for some platform/Python combinations.  All probes
here return ``None`` (or ``False`` for individual checks) when
dolfinx is not importable so the test skips cleanly in environments
that lack it.

The catalog references for FEniCSx already exist in
``scripts/fingerprint_solvers.py`` (``fingerprint_fenics()``); this
module exposes the underlying primitives so the catalog-consistency
test can reuse the same API map.
"""

from __future__ import annotations

import importlib


def has_attr(dotted_path: str) -> bool | None:
    """Return ``True`` / ``False`` if the dotted-path attribute can be
    resolved on the installed ``dolfinx`` package; return ``None``
    when ``dolfinx`` is not importable at all so the caller can
    distinguish "missing attribute" from "package missing".

    Walks the dotted path one segment at a time.  ``getattr`` on a
    parent package only sees submodules that have been imported by
    the parent's ``__init__`` -- and dolfinx deliberately does NOT
    side-import every submodule (notably ``dolfinx.fem.petsc`` is
    only available after ``import dolfinx.fem.petsc``).  When
    ``getattr`` raises ``AttributeError`` we therefore try
    ``importlib.import_module`` on the same path before giving up;
    this catches the petsc case and any other lazy-loaded submodule.

    Example::

        has_attr("dolfinx.mesh.create_rectangle")    # True / False / None
        has_attr("dolfinx.fem.petsc.LinearProblem")  # True when petsc-enabled
    """
    parts = dotted_path.split(".")
    if not parts or parts[0] != "dolfinx":
        # Dolfinx-specific; refuse other roots so callers don't
        # accidentally pass an arbitrary path and get a misleading True.
        return None
    try:
        obj: object = importlib.import_module("dolfinx")
    except ImportError:
        return None
    module_path = "dolfinx"
    for segment in parts[1:]:
        next_path = f"{module_path}.{segment}"
        try:
            obj = getattr(obj, segment)
        except AttributeError:
            # Lazy-loaded submodule: try importing it explicitly.
            try:
                obj = importlib.import_module(next_path)
            except ImportError:
                return False
        module_path = next_path
    return True


def is_available() -> bool:
    """``True`` if ``dolfinx`` is importable in the current environment."""
    try:
        importlib.import_module("dolfinx")
    except ImportError:
        return False
    return True


# Hand-curated list of dotted paths the catalog mentions in template
# code and structured knowledge.  These are checked in addition to
# whatever the live source scan picks up, so a path that the catalog
# DOES use but that our regex happens to miss (broken across lines,
# embedded in a docstring fragment, etc.) is still verified.
#
# Partially overlaps with
# ``scripts/fingerprint_solvers.py::fingerprint_fenics`` (~9 entries
# in common); intentionally extends it because the catalog references
# a few names the fingerprint script does not (``create_unit_square``
# / ``create_unit_cube`` / ``locate_dofs_*`` / ``NonlinearProblem``).
# When the two lists drift, update both; there is no programmatic
# single source of truth.
CATALOG_API_PATHS: tuple[str, ...] = (
    "dolfinx.mesh.create_rectangle",
    "dolfinx.mesh.create_box",
    "dolfinx.mesh.create_unit_square",
    "dolfinx.mesh.create_unit_cube",
    "dolfinx.fem.functionspace",
    "dolfinx.fem.Function",
    "dolfinx.fem.dirichletbc",
    "dolfinx.fem.locate_dofs_topological",
    "dolfinx.fem.locate_dofs_geometrical",
    "dolfinx.fem.petsc.LinearProblem",
    "dolfinx.fem.petsc.NonlinearProblem",
    "dolfinx.io.XDMFFile",
    "dolfinx.io.VTKFile",
    "dolfinx.default_scalar_type",
)
