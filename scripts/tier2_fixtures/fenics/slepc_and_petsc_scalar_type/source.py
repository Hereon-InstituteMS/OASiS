"""Tier-2: SLEPc availability and PETSc scalar type.

Pitfalls (fenics eigenvalue#0 + #4 / helmholtz#3):
  - SLEPc must be available for eigenvalue problems. The
    binding is slepc4py.SLEPc.EPS.
  - PETSc scalar type determines whether complex coefficients
    work. dolfinx.default_scalar_type reports float64 in the
    real build, complex128 in the complex build.

Verifies the conda-forge fenics-dolfinx build state so the
agent knows which path is available.
"""
from __future__ import annotations

import sys

import numpy as np


def main() -> int:
    try:
        from slepc4py import SLEPc
        slepc_ok = True
        eps_ok = hasattr(SLEPc, "EPS")
    except ImportError:
        slepc_ok = False
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
