"""Tier-2: `dolfinx.io.gmshio` is gone in dolfinx 0.10 — RENAMED, not missing.

WHAT THIS USED TO SAY, AND WHY IT WAS WRONG. The previous version of this
fixture blamed a missing `gmsh` package: "That submodule is optional and ONLY
available if the Python 'gmsh' package is installed. The current ofa-fenicsx
conda env does NOT include gmsh." Both halves are false on this host. gmsh
4.15.1 imports fine in the conda env `fenics`; `dolfinx.io.gmshio` is absent
because dolfinx 0.10.0 RENAMED the submodule to `dolfinx.io.gmsh`, which
imports and carries `model_to_mesh`, `read_from_msh`, `create_mesh` and
`MeshData`. An agent given the old diagnosis would install a package it
already has and still not find the module.

The distinction is the whole point, and the two causes are told apart by ONE
observation: whether `gmsh` itself imports.

    gmsh imports + gmshio missing   -> a RENAME. Use dolfinx.io.gmsh.
    gmsh missing                    -> an install gap. conda install gmsh.

WHAT THE OLD FIXTURE ASSERTED: nothing. Its three expected strings —
'gmshio_present=', 'gmsh_present=', 'catalog_uses_gmshio=' — were labels this
script printed unconditionally, so gmshio_present=True and gmshio_present=False
passed alike. That was measured on 2026-08-07: three staged copies differing
only in which modules the probes imported were run through
scripts/run_tier2_fixtures.py and ALL THREE PASSED. A fixture that cannot go
red is not a check.

So this now asserts the VERDICT, not the labels, and the mutation control
below proves the verdict moves.

Mutation control: T2_MUTATE=1 points the fallback probe at a module that does
not exist, which is what a genuine install gap looks like. The verdict then
flips to VERDICT=gmsh_install_gap, the expected
VERDICT=gmshio_was_renamed_to_dolfinx_io_gmsh string goes missing, and the
fixture fails its own expectations.
"""
from __future__ import annotations

import importlib
import os
import sys

MUTATE = os.environ.get("T2_MUTATE") == "1"

# The module dolfinx 0.10 replaced `dolfinx.io.gmshio` with. Under mutation,
# a name that cannot exist — standing in for the env where gmsh is genuinely
# absent and the replacement therefore cannot be imported either.
REPLACEMENT = "dolfinx.io.definitely_not_installed" if MUTATE \
    else "dolfinx.io.gmsh"


def _imports(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def main() -> int:
    old_name = _imports("dolfinx.io.gmshio")
    new_name = _imports(REPLACEMENT)
    gmsh_pkg = _imports("gmsh")

    print(f"gmshio_present={old_name}")
    print(f"replacement_module={REPLACEMENT}")
    print(f"replacement_present={new_name}")
    print(f"gmsh_present={gmsh_pkg}")

    if gmsh_pkg:
        import gmsh  # noqa: F401  (already imported above; for the version)
        print(f"gmsh_pyver={getattr(gmsh, '__version__', 'not-exposed')}")

    # The API the catalog actually needs, wherever it now lives.
    entry_points: list[str] = []
    if new_name:
        mod = importlib.import_module(REPLACEMENT)
        entry_points = [n for n in ("model_to_mesh", "read_from_msh",
                                    "create_mesh", "MeshData")
                        if hasattr(mod, n)]
    print(f"entry_points_available={entry_points}")

    # THE DISCRIMINATION. An install gap and a rename look identical if you
    # only probe the old name; they differ on whether gmsh itself is there.
    if old_name:
        print("VERDICT=gmshio_still_present")
        return 0
    if gmsh_pkg and new_name and "model_to_mesh" in entry_points:
        print("VERDICT=gmshio_was_renamed_to_dolfinx_io_gmsh")
        return 0
    print("VERDICT=gmsh_install_gap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
