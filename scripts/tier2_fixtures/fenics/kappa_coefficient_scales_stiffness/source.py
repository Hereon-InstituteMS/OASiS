"""Tier-2 for fenics poisson#4: a forgotten kappa coefficient is silent, and the
whole stiffness matrix is off by exactly kappa.

Wrong variant: assemble the diffusion form WITHOUT the coefficient. Nothing is
raised; the matrix is simply the kappa = 1 matrix. The fixture assembles both
and checks the norm ratio equals the kappa it left out, which is the only
observable the claim offers.

The ratio is computed and compared to kappa inside the fixture, so no measured
number is pinned in the expectations.

Mutation control: T2_MUTATE=1 puts the fem.Constant back into the form; the
ratio becomes 1 and the verdict flips.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"
KAPPA = 7.0


def norm_of(a) -> float:
    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a))
    A.assemble()
    return A.norm()


def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    kappa = dolfinx.fem.Constant(msh, KAPPA)

    reference = norm_of(kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    if MUTATE:
        written = norm_of(kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    else:
        # kappa forgotten — the mistake the claim is about.
        written = norm_of(ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx)

    print(f"kappa={KAPPA}")
    print(f"assembly_raised=False")
    ratio = reference / written
    print(f"norm_ratio={ratio:.9f}")
    off_by_kappa = abs(ratio - KAPPA) < 1e-9 * KAPPA
    print(f"off_by_exactly_kappa={off_by_kappa}")
    if off_by_kappa:
        print("VERDICT=missing_coefficient_scales_stiffness_by_kappa")
        return 0
    print("VERDICT=coefficient_present")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
