"""Tier-2 for fenics hyperelasticity#0: signals phrased around
dolfinx.nls.petsc.NewtonSolver.solve describe a 0.9-era code path. On dolfinx
0.10.0 that pairing is dead, so hyperelastic Newton diagnostics have to be read
off PETSc SNES instead.

The fixture builds a compressible Neo-Hookean NonlinearProblem (the standard
hyperelasticity residual, P = dPsi/dF via ufl.variable/ufl.diff), then tries
NewtonSolver(MPI.COMM_WORLD, problem). Observed signal: a DeprecationWarning
"dolfinx.nls.petsc.NewtonSolver is deprecated. Use
dolfinx.fem.petsc.NonlinearProblem, a high level interface to PETSc SNES." and
then AttributeError: 'NonlinearProblem' object has no attribute 'a'. The
supported replacement is exercised in the same run: problem.solve(), then
problem.solver (a petsc4py SNES) with getConvergedReason() and
getIterationNumber().

Mutation control: T2_MUTATE=1 skips the legacy pairing and only walks the SNES
path.
"""
from __future__ import annotations

import os
import tempfile
import warnings

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402
import dolfinx.nls.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    d = msh.geometry.dim
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (d,)))
    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    f_var = ufl.variable(ufl.Identity(d) + ufl.grad(u))
    j = ufl.det(f_var)
    mu, lmbda = 1.0, 10.0
    psi = (mu / 2) * (ufl.tr(f_var.T * f_var) - d) - mu * ufl.ln(j) \
        + (lmbda / 2) * ufl.ln(j) ** 2
    piola = ufl.diff(psi, f_var)
    body = dolfinx.fem.Constant(msh, (0.0, -0.4))
    res = ufl.inner(piola, ufl.grad(v)) * ufl.dx - ufl.inner(body, v) * ufl.dx

    msh.topology.create_connectivity(d - 1, d)
    left = dolfinx.mesh.locate_entities_boundary(
        msh, d - 1, lambda x: np.isclose(x[0], 0.0))
    bc = dolfinx.fem.dirichletbc(
        np.zeros(d), dolfinx.fem.locate_dofs_topological(V, d - 1, left), V)
    problem = dolfinx.fem.petsc.NonlinearProblem(
        res, u, bcs=[bc], petsc_options_prefix="t2_hy0_",
        petsc_options={"snes_type": "newtonls", "ksp_type": "preonly",
                       "pc_type": "lu", "snes_rtol": 1e-9})

    legacy = ""
    deprecated = False
    if not MUTATE:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
                print("legacy_newtonsolver_pairing_raised=False")
            except Exception as exc:  # noqa: BLE001
                legacy = f"{type(exc).__name__}: {exc}"
                print(f"legacy_newtonsolver_pairing_raised=True {legacy}")
        cats = [w.category.__name__ for w in caught]
        texts = " | ".join(str(w.message) for w in caught)
        deprecated = "DeprecationWarning" in cats
        print(f"legacy_warning_categories={cats}")
        print(f"legacy_warning_text={texts}")

    problem.solve()
    reason = problem.solver.getConvergedReason()
    its = problem.solver.getIterationNumber()
    kind = type(problem.solver).__name__
    tip = float(np.max(np.abs(u.x.array)))
    print(f"snes_object_type={kind}")
    print(f"snes_converged_reason={reason} snes_iterations={its}")
    print(f"max_abs_displacement={tip:.6e}")
    snes_ok = kind == "SNES" and reason > 0 and its >= 1 and tip > 0.0
    print(f"snes_path_works={snes_ok}")
    no_attr = "object has no attribute 'a'" in legacy
    print(f"legacy_pairing_fails_on_missing_attribute_a={no_attr}")
    print(f"legacy_pairing_warned_deprecated={deprecated}")

    if no_attr and deprecated and snes_ok:
        print("VERDICT=read_hyperelastic_newton_signals_off_snes")
        return 0
    print("VERDICT=legacy_newtonsolver_pairing_still_works")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
