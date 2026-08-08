"""Tier-2: which dolfinx module actually reads a Gmsh mesh here.

CORRECTION 2026-08-08, measured.  This fixture used to say:

    "Several fenics generators call dolfinx.io.gmshio.model_to_mesh(...).
     That submodule is optional and ONLY available if the Python 'gmsh'
     package is installed.  The current ofa-fenicsx conda env does NOT
     include gmsh -- catalog use of gmshio raises ImportError at runtime."

Every clause of that is wrong on this host.  gmsh 4.15.1 imports fine.
dolfinx.io.gmshio is missing because dolfinx 0.10.0 RENAMED the submodule to
dolfinx.io.gmsh, which is present and carries model_to_mesh and read_from_msh.
There is no install gap; there was a module rename, and the catalog already
imports the new name (src/backends/fenics/generators/{poisson,navier_stokes,
elasticity}.py all use `from dolfinx.io.gmsh import read_from_msh`, and
generators/__init__.py states "the module is `gmsh`, NOT `gmshio`").

So the diagnostic this fixture prints is the rename, not a gap, and it asserts
values rather than the mere existence of a report.  Its three expectations were
bare `key=` prefixes -- gmshio_present=, gmsh_present=, catalog_uses_gmshio= --
which match True and False alike, which is how it could be the stated evidence
for a claim that is the opposite of what it measured.
"""
from __future__ import annotations

import importlib
import os
import sys

# MUTATION CONTROL: T2_MUTATE=1 applies the documented fix at the site that
# fails -- the import reaches for dolfinx.io.gmsh, the name dolfinx 0.10
# actually ships, instead of the removed dolfinx.io.gmshio. The ImportError
# then does not happen, so gmshio_present=False and
# gmshio_absent_because_renamed=True both disappear and the fixture goes red.
# Re-run: T2_MUTATE=1 python source.py
MUTATE = os.environ.get("T2_MUTATE") == "1"
OLD_NAME = "dolfinx.io.gmsh" if MUTATE else "dolfinx.io.gmshio"


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def main() -> int:
    import dolfinx

    gmshio_present = _importable(OLD_NAME)
    print(f"probed_legacy_module_name={OLD_NAME}")
    gmsh_mod_present = _importable("dolfinx.io.gmsh")
    print(f"dolfinx_version={dolfinx.__version__}")
    print(f"gmshio_present={gmshio_present}")
    print(f"dolfinx_io_gmsh_present={gmsh_mod_present}")

    entrypoints = []
    if gmsh_mod_present:
        mod = importlib.import_module("dolfinx.io.gmsh")
        entrypoints = sorted(n for n in ("model_to_mesh", "read_from_msh")
                             if hasattr(mod, n))
    print(f"dolfinx_io_gmsh_entrypoints={entrypoints}")
    print(f"dolfinx_io_gmsh_has_model_to_mesh="
          f"{'model_to_mesh' in entrypoints}")

    try:
        import gmsh  # type: ignore
        print(f"gmsh_pyver={getattr(gmsh, '__version__', 'not-exposed')}")
        print("gmsh_present=True")
        gmsh_present = True
    except ImportError as exc:
        print(f"gmsh_present=False import_error={exc!r}")
        gmsh_present = False

    # THE POINT.  Absent gmshio + present gmsh + present dolfinx.io.gmsh is a
    # RENAME.  Absent gmshio + absent gmsh would be the install gap the old
    # docstring claimed.  The two are opposite diagnoses and the fixture now
    # tells them apart.
    renamed = (not gmshio_present) and gmsh_mod_present and gmsh_present
    gap = (not gmshio_present) and not gmsh_present
    print(f"gmshio_absent_because_renamed={renamed}")
    print(f"gmshio_absent_because_gmsh_missing={gap}")
    if not (renamed or gap or gmshio_present):
        print("FAIL: neither diagnosis holds -- gmshio is absent, gmsh is "
              "importable, but dolfinx.io.gmsh is not there either",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
