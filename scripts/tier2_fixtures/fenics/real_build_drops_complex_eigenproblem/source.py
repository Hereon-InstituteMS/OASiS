"""Tier-2 for fenics eigenvalue#5: a complex-valued eigenproblem needs dolfinx +
PETSc + SLEPc built with --with-scalar-type=complex. The default conda-forge
fenics-dolfinx build is REAL, and the fixture shows the three places that shows
up when you try to set up a complex Helmholtz / Maxwell eigenproblem anyway.

Wrong variant: an imaginary coefficient (a complex damping term, a complex EPS
target) fed to the real build. Right variant: keep every value real —
dolfinx.default_scalar_type — and split into a (re, im) real-pair formulation
if the physics really is complex.

Observed on the conda-forge `fenics` env, dolfinx 0.10.0 / slepc4py 3.24.3:

  * dolfinx.default_scalar_type is float64 and
    numpy.issubdtype(dolfinx.default_scalar_type, numpy.complexfloating) is
    False, as is PETSc.ScalarType;
  * a UFL form carrying an imaginary literal is REJECTED at fem.form with
    ValueError "Unexpected complex value in real expression.";
  * SLEPc rejects a complex target with TypeError "must be real number, not
    complex";
  * but Function.interpolate of a complex-valued expression is NOT rejected —
    numpy raises only a ComplexWarning and the imaginary part is silently
    dropped, leaving a real Function holding just the real part. That is the
    silent-wrong-answer path the claim warns about, and it is the only one of
    the three that does not stop you.

Mutation control: T2_MUTATE=1 uses dolfinx.default_scalar_type values
everywhere, so nothing is rejected and nothing is dropped, and the fixture
loses its own expectations.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

import warnings  # noqa: E402

from petsc4py import PETSc  # noqa: E402
from slepc4py import SLEPc  # noqa: E402


def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    scalar = np.dtype(dolfinx.default_scalar_type).name
    is_complex = bool(np.issubdtype(dolfinx.default_scalar_type,
                                    np.complexfloating))
    print(f"default_scalar_type={scalar}")
    print(f"petsc_scalar_type={np.dtype(PETSc.ScalarType).name}")
    print(f"default_scalar_type_is_complex={is_complex}")

    gamma = dolfinx.default_scalar_type(0.5) if MUTATE else 0.5j

    # 1. an imaginary damping term in the eigen-form
    form_err = ""
    try:
        dolfinx.fem.form(gamma * u * v * ufl.dx)
        print("damping_form_compiled=True")
    except Exception as exc:
        form_err = f"{type(exc).__name__}: {exc}"
        print(f"damping_form_compiled=False {form_err}")

    # 2. a complex shift handed to SLEPc
    A = dolfinx.fem.petsc.assemble_matrix(
        dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx))
    A.assemble()
    eps = SLEPc.EPS().create(msh.comm)
    eps.setOperators(A)
    target = (dolfinx.default_scalar_type(1.0) if MUTATE else 1.0 + 3.0j)
    target_err = ""
    try:
        eps.setTarget(target)
        print("complex_target_accepted=True")
    except Exception as exc:
        target_err = f"{type(exc).__name__}: {exc}"
        print(f"complex_target_accepted=False {target_err}")

    # 3. the silent one: interpolating a complex expression
    coeff = dolfinx.default_scalar_type(1.0) if MUTATE else (1.0 + 2.0j)
    f = dolfinx.fem.Function(V)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        interp_err = ""
        try:
            f.interpolate(lambda x: coeff * np.ones_like(x[0]))
        except Exception as exc:
            interp_err = f"{type(exc).__name__}: {exc}"
        names = sorted({w.category.__name__ for w in caught})
    stored = float(np.max(f.x.array)) if not interp_err else float("nan")
    print(f"interpolate_error={interp_err!r} warnings={names} "
          f"stored_value={stored}")

    dropped = (not interp_err
               and "ComplexWarning" in names
               and abs(stored - float(np.real(coeff))) < 1.0e-12
               and abs(float(np.imag(coeff))) > 0.0)
    print(f"real_build_reports_float64_not_complex={not is_complex}")
    print(f"imaginary_coefficient_rejected_at_form_compilation={bool(form_err)}")
    print(f"slepc_rejected_the_complex_target={bool(target_err)}")
    print(f"interpolate_silently_dropped_the_imaginary_part={dropped}")
    if (not is_complex) and form_err and target_err and dropped:
        print("VERDICT=real_build_rejects_complex_forms_but_silently_drops_interpolated_imaginary_parts")
        return 0
    print("VERDICT=real_build_carried_the_complex_eigenproblem")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
