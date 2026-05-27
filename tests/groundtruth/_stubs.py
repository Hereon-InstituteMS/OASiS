"""Per-backend probe stubs.

Each function below should be replaced with a real probe.  Until then they
return ``None`` so the test suite skips the corresponding check rather
than failing or silently passing.

The stub-import dance also keeps these names visible to readers browsing
the package layout: a missing probe is now a TODO item with a clearly
named placeholder, not an absence.
"""

from __future__ import annotations


# deal.II finite-element classes -- implemented in ``dealii.py`` (source-grep
# family).  Future deal.II probes (Dirichlet BC types, quadrature classes,
# Manifold types) can either extend ``dealii.py`` or follow its pattern in
# additional modules.


# ── FEniCSx / dolfinx (Python introspection) ──────────────────────────────
def fenics_element_families() -> None:
    return None


def fenics_solver_types() -> None:
    return None


# ── NGSolve (Python introspection) ────────────────────────────────────────
def ngsolve_finite_element_spaces() -> None:
    return None


# scikit-fem -- implemented in ``skfem.py`` (Python introspection family).
# Use that module directly; the stub below is removed.


# ── Kratos Multiphysics (Python + JSON registries) ────────────────────────
def kratos_constitutive_laws() -> None:
    return None


def kratos_strategies() -> None:
    return None


# ── DUNE-fem (Python + C++) ───────────────────────────────────────────────
def dune_grid_implementations() -> None:
    return None


# ── FEBio (XML schema) ────────────────────────────────────────────────────
def febio_material_types() -> None:
    return None
