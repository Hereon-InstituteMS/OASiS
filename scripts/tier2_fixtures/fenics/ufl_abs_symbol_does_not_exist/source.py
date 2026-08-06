"""Tier-2 for fenics dg_methods#9: modern UFL has no ufl.Abs symbol — the DG
upwind flux must be built with Python's builtin abs() (which UFL overloads for
Expr operands) or with ufl.algebra.Abs as the explicit fallback.

Wrong variant: writing the upwind trace as
    (bn('+') + ufl.Abs(bn('+')))/2 * u('+') + ...
The failure happens at attribute-access time, before any assembly is
attempted, with the literal text "module 'ufl' has no attribute 'Abs'".

The fixture takes the absolute-value operator from a slot, builds the upwind
flux with it, and reports what came back. It also checks the two documented
replacements: ufl.algebra.Abs is a real class, and the builtin abs() applied
to a UFL Expr returns an instance of exactly that class.

Mutation control: T2_MUTATE=1 puts the builtin abs() in the slot — the correct
idiom — so nothing is raised and the flux is built.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import ufl  # noqa: E402
import ufl.algebra  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(msh, ("DG", 1))
    u = ufl.TrialFunction(V)
    n = ufl.FacetNormal(msh)
    b = ufl.as_vector([1.0, 0.5])
    bn = ufl.dot(b, n)

    # The slot: the absolute value the DG upwind flux is written with.
    raised = ""
    flux = None
    try:
        abs_op = abs if MUTATE else ufl.Abs
        flux = ((bn("+") + abs_op(bn("+"))) / 2.0 * u("+")
                + (bn("+") - abs_op(bn("+"))) / 2.0 * u("-"))
    except AttributeError as exc:
        raised = f"AttributeError: {exc}"

    print(f"ufl_Abs_raised_attributeerror={bool(raised)}")
    if raised:
        print(f"raised_text={raised}")
    print(f"upwind_flux_built={flux is not None}")
    print(f"ufl_has_attribute_Abs={hasattr(ufl, 'Abs')}")

    # Documented replacements.
    algebra_ok = isinstance(ufl.algebra.Abs, type)
    builtin_is_algebra_abs = isinstance(abs(bn("+")), ufl.algebra.Abs)
    print(f"ufl_algebra_Abs_is_a_class={algebra_ok}")
    print(f"builtin_abs_returns_ufl_algebra_Abs={builtin_is_algebra_abs}")

    if raised and algebra_ok and builtin_is_algebra_abs and flux is None:
        print("VERDICT=ufl_Abs_missing_use_builtin_abs")
        return 0
    print("VERDICT=ufl_Abs_usable")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
