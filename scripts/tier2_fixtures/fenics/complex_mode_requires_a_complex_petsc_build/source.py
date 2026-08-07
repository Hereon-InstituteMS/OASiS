"""Tier-2 for fenics helmholtz#3: the conda-forge dolfinx build is REAL, and the
three ways that shows up.

  * numpy.issubdtype(default_scalar_type, complexfloating) is the check to make
    BEFORE building the form;
  * fem.form on a form carrying an imaginary coefficient raises ValueError;
  * Function.interpolate of a complex callable does not raise — it drops the
    imaginary part, with a ComplexWarning.

Mutation control: T2_MUTATE=1 uses real coefficients throughout, so nothing is
raised and nothing is discarded.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import basix  # noqa: E402
import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

import warnings  # noqa: E402


def main() -> int:
    is_complex = bool(np.issubdtype(dolfinx.default_scalar_type,
                                    np.complexfloating))
    print(f"build_is_complex={is_complex}")
    print(f"scalar_type={np.dtype(dolfinx.default_scalar_type).name}")

    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    coeff = 1.0 if MUTATE else 1.0j

    # TWO SPELLINGS, and they fail differently — the claim quotes only one.
    # A complex fem.Constant is rejected by the float64 Form binding with a
    # pybind11 TypeError; the documented ValueError appears only when the
    # imaginary value is a UFL literal inside the expression.
    form_err = ""
    try:
        dolfinx.fem.form(dolfinx.fem.Constant(msh, coeff) * v * ufl.dx)
        print("constant_spelling_raised=False")
    except Exception as exc:
        print(f"constant_spelling_raised=True {type(exc).__name__}: "
              f"{str(exc).splitlines()[0]}")
    try:
        dolfinx.fem.form(ufl.as_ufl(coeff) * v * ufl.dx)
        print("literal_spelling_raised=False")
    except Exception as exc:
        form_err = f"{type(exc).__name__}: {exc}"
        print(f"literal_spelling_raised=True {form_err}")
    print(f"form_raised={bool(form_err)}")

    f = dolfinx.fem.Function(V)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f.interpolate(lambda x: np.full(x.shape[1], coeff))
    texts = " | ".join(f"{w.category.__name__}: {w.message}" for w in caught)
    print(f"interpolate_raised=False")
    print(f"interpolate_warnings={texts}")
    imag_kept = bool(np.any(np.imag(np.asarray(f.x.array, dtype=complex))))
    print(f"imaginary_part_kept={imag_kept}")
    if not is_complex and form_err and not imag_kept:
        print("VERDICT=real_build_rejects_forms_and_discards_imaginary_parts")
        return 0
    print("VERDICT=complex_survived")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
