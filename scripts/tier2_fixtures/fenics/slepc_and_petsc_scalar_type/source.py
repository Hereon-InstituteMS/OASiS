"""Tier-2: SLEPc availability and PETSc scalar type.

Pitfalls (fenics eigenvalue#0 + #4 / helmholtz#3):
  - SLEPc must be available for eigenvalue problems. The
    binding is slepc4py.SLEPc.EPS.
  - PETSc scalar type determines whether complex coefficients
    work. dolfinx.default_scalar_type reports float64 in the
    real build, complex128 in the complex build.

Verifies the conda-forge fenics-dolfinx build state so the
agent knows which path is available.

Mutation control: T2_MUTATE=1 asks the WRONG package for the
eigensolver — petsc4py.PETSc instead of slepc4py.SLEPc. petsc4py
imports perfectly well (so slepc_available is unaffected and the
run does not crash), but it does not own EPS: the eigensolver class
lives in SLEPc, which is the whole point of the pitfall. The
measurement flips to EPS_available=False and the expectation
'EPS_available=True' disappears.
"""
from __future__ import annotations

import importlib
import os
import sys

import numpy as np

MUTATE = os.environ.get("T2_MUTATE") == "1"

# Where the EPS eigensolver class is looked for. SLEPc owns it;
# PETSc does not.
EPS_HOST = "petsc4py.PETSc" if MUTATE else "slepc4py.SLEPc"


def main() -> int:
    try:
        import slepc4py  # noqa: F401
        slepc_ok = True
    except ImportError:
        slepc_ok = False

    print(f"eps_probe_module={EPS_HOST}")
    try:
        eps_host = importlib.import_module(EPS_HOST)
        eps_ok = hasattr(eps_host, "EPS")
    except ImportError:
        eps_ok = False

    import dolfinx
    dtype_name = dolfinx.default_scalar_type.__name__
    is_complex = np.issubdtype(dolfinx.default_scalar_type,
                                 np.complexfloating)

    print(f"slepc_available={slepc_ok}")
    print(f"EPS_available={eps_ok}")
    print(f"scalar_type={dtype_name}")
    print(f"is_complex_build={is_complex}")

    # The current ofa-fenicsx env: slepc4py with EPS, real PETSc.
    if slepc_ok and eps_ok and dtype_name == "float64":
        return 0
    print("ERROR: unexpected build state", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
