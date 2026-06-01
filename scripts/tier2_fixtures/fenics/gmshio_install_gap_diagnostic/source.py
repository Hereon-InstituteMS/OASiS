"""Tier-2: dolfinx.io.gmshio install-gap diagnostic.

Several fenics generators (notably advanced.py, more
complex mesh examples) call
dolfinx.io.gmshio.model_to_mesh(...) to import a Gmsh
.msh / Python gmsh model. That submodule is optional and
ONLY available if the Python 'gmsh' package is installed.
The current ofa-fenicsx conda env does NOT include gmsh
— catalog use of gmshio raises ImportError at runtime.

This fixture reports the install-gap status. It always
PASSES (the gap is a known limitation tracked in the
campaign); the regression value is the live diagnostic
output in the fixture log.

When the env is rebuilt with gmsh:

  conda install -c conda-forge gmsh python-gmsh

the same fixture will record 'gmshio_present=True' and
'gmsh_pyver=<x>'.
"""
from __future__ import annotations

import importlib
import sys


def main() -> int:
    # Try gmshio import.
    gmshio_present = False
    try:
        importlib.import_module("dolfinx.io.gmshio")
        gmshio_present = True
    except ImportError:
        pass
    print(f"gmshio_present={gmshio_present}")

    # gmsh python module
    try:
        import gmsh  # type: ignore
        gver = getattr(gmsh, "__version__",
                       "version-not-exposed")
        print(f"gmsh_pyver={gver}")
        print("gmsh_present=True")
    except ImportError as exc:
        print(f"gmsh_present=False import_error={exc!r}")

    # The catalog gen function names that name gmshio
    catalog_uses = [
        "src/backends/fenics/generators/advanced.py "
        "(model_to_mesh import for Gmsh-driven workflows)",
    ]
    print(f"catalog_uses_gmshio={catalog_uses}")

    # Always pass — this is a diagnostic.
    return 0


if __name__ == "__main__":
    sys.exit(main())
