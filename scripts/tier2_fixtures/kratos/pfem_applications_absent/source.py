"""Tier-2: no PFEM application imports on a pip Kratos stack.

The claim is an availability fact. It is checked by import,
not by asking the package index, so it holds
offline and states something about THIS build. The positive
controls rule out a fixture that passes because nothing at all
is installed.
"""
from __future__ import annotations

import importlib
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

# Imported unconditionally: with Kratos absent this raises and the
# process exits non-zero, so the fixture cannot report a pass while
# testing nothing.
import KratosMultiphysics as KM

print(f"kratos_version_present={bool(KM.__file__)}")

# (application_module_suffix, must_import)
CASES = [('PfemFluidDynamicsApplication', False), ('DelaunayMeshingApplication', False), ('PfemSolidMechanicsApplication', False), ('PFEM2Application', False), ('StructuralMechanicsApplication', True), ('FluidDynamicsApplication', True)]


def main() -> int:
    bad = 0
    for app, must in CASES:
        try:
            importlib.import_module("KratosMultiphysics." + app)
            got, err = True, ""
        except BaseException as exc:
            got, err = False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:130]}"
        print(f"importable[{app}]={got}_expected={must}")
        if not got:
            print(f"  message: {err}")
        if got != must:
            bad += 1
            print(f"FAIL: {app} importable={got} expected={must} {err}",
                  file=sys.stderr)
    print(f"app_import_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
