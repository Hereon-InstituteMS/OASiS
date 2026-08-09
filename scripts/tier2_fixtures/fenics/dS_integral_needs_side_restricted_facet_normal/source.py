"""Tier-2 for fenics dg_methods#5: FacetNormal is outward and the avg/jump
operators live on a two-sided facet, so anything built from n inside a dS
integral needs a '+'/'-' side.

Wrong variant: bn = ufl.dot(b, n) used directly in a dS integrand, as in
bn * avg(u) * jump(v) * dS. The UFL form object is built without a murmur; the
failure comes later, out of fem.form, as
    ValueError: Discontinuous type Jacobian must be restricted.
The same integrand with ufl.dot(b, n('+')) compiles.

The fixture also checks the correction the claim makes to its own older text:
the string "side specifier required on '+' or '-' for restricted facet
integrals" does not appear anywhere in what UFL raises on this release, and
jump(v, n) is fine unrestricted because the operator restricts n itself.

Mutation control: T2_MUTATE=1 puts n('+') in the slot, so fem.form succeeds.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

from dolfinx import fem, mesh  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

OLD_TEXT = "side specifier required on '+' or '-' for restricted facet integrals"


def main() -> int:
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4,
                                  mesh.CellType.triangle)
    msh.topology.create_connectivity(1, 2)
    V = fem.functionspace(msh, ("DG", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    b = ufl.as_vector([1.0, 0.5])
    n = ufl.FacetNormal(msh)

    # the slot: the normal used to build the advective facet coefficient
    bn = ufl.dot(b, n("+")) if MUTATE else ufl.dot(b, n)
    built, build_err = True, ""
    try:
        a = bn * ufl.avg(u) * ufl.jump(v) * ufl.dS
    except Exception as exc:
        built, build_err = False, f"{type(exc).__name__}: {exc}"
    print(f"ufl_form_object_built_without_error={built} {build_err}")

    compiled, msg = False, ""
    if built:
        try:
            fem.form(a)
            compiled = True
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
    print(f"fem_form_compiled={compiled}")
    if msg:
        print(f"fem_form_raised: {msg}")
    print(f"raised_is_valueerror={msg.startswith('ValueError')}")
    print(f"old_quoted_message_absent={OLD_TEXT.lower() not in msg.lower()}")

    # the same integrand with the side specifier, always run
    ok = False
    try:
        fem.form(ufl.dot(b, n("+")) * ufl.avg(u) * ufl.jump(v) * ufl.dS)
        ok = True
    except Exception as exc:                            # pragma: no cover
        print(f"restricted_variant_raised: {type(exc).__name__}: {exc}")
    print(f"restricted_normal_compiles={ok}")

    # jump(v, n) restricts n internally, so it needs no suffix
    jump_ok = False
    try:
        fem.form(ufl.inner(ufl.avg(ufl.grad(u)), ufl.jump(v, n)) * ufl.dS)
        jump_ok = True
    except Exception as exc:                            # pragma: no cover
        print(f"jump_with_normal_raised: {type(exc).__name__}: {exc}")
    print(f"jump_v_n_needs_no_side_specifier={jump_ok}")

    if (built and not compiled and msg.startswith("ValueError")
            and OLD_TEXT.lower() not in msg.lower() and ok and jump_ok):
        print("VERDICT=dS_integrand_needs_a_restricted_facet_normal")
        return 0
    print("VERDICT=unrestricted_normal_accepted_in_dS")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
