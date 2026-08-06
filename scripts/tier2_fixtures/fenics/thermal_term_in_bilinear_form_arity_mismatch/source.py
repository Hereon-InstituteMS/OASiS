"""Tier-2 for fenics thermal_structural#1: the thermal term carries no trial
function, so it is a linear functional and belongs in L, never in the bilinear
form a.

Wrong variant: a = inner(sigma_elastic(u) - beta*(T - T_ref)*Identity(d),
eps(v))*dx. The subtraction mixes an expression with no form arguments into one
that carries the test function, and UFL refuses it at form-compile time.
Observed signal: ArityMismatch raised out of ufl/algorithms/check_arities.py,
message "Adding expressions with non-matching form arguments () vs ('v_1',)."
The fix is L = beta*(T - T_ref)*div(v)*dx, which uses
inner(c*Identity(d), eps(v)) == c*div(v); the fixture also checks that identity
numerically so the correction is not taken on faith.

Mutation control: T2_MUTATE=1 keeps the thermal term out of a and puts it in L,
so nothing raises.
"""
from __future__ import annotations

import os
import tempfile

# Hard set, not setdefault: the signal here IS a form-compile failure, and a
# cache already holding a stale .c file for this form masks it with a
# TimeoutError instead (see thermal_structural#2).
os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="ffcx_t2_arity_")

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

E, NU, ALPHA = 210e9, 0.3, 1.2e-5
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
BETA = (3 * LAM + 2 * MU) * ALPHA


def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    d = msh.geometry.dim
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (d,)))
    S = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    T = dolfinx.fem.Function(S)
    T.interpolate(lambda x: 300.0 + 100.0 * x[0])
    t_ref = dolfinx.fem.Constant(msh, 300.0)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    eps = lambda w: ufl.sym(ufl.grad(w))  # noqa: E731
    elastic = 2 * MU * eps(u) + LAM * ufl.tr(eps(u)) * ufl.Identity(d)
    thermal = BETA * (T - t_ref) * ufl.Identity(d)

    a = ufl.inner(elastic, eps(v)) * ufl.dx
    if not MUTATE:
        a = ufl.inner(elastic - thermal, eps(v)) * ufl.dx

    raised = ""
    by_exception = False
    try:
        try:
            dolfinx.fem.form(a)
            print("bilinear_form_compiled=True")
        except Exception as exc:  # noqa: BLE001
            by_exception = True
            raised = f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
    except BaseException as exc:  # noqa: BLE001
        raised = f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
    if raised:
        print(f"bilinear_form_compiled=False error={raised}")
    # ufl.algorithms.check_arities.ArityMismatch derives from BaseException,
    # so a plain `except Exception:` around the form compilation misses it.
    print(f"arity_mismatch_caught_by_except_Exception={by_exception}")

    # The fix compiles, and inner(c*I, eps(v)) really is c*div(v).
    fix = BETA * (T - t_ref) * ufl.div(v) * ufl.dx
    dolfinx.fem.form(fix)
    b_fix = dolfinx.fem.assemble_vector(dolfinx.fem.form(fix))
    b_ident = dolfinx.fem.assemble_vector(
        dolfinx.fem.form(ufl.inner(thermal, eps(v)) * ufl.dx))
    same = bool(np.allclose(b_fix.array, b_ident.array, rtol=1e-12, atol=0.0))
    print(f"linear_form_in_L_compiles=True")
    print(f"div_v_identity_holds={same}")

    is_arity = ("ArityMismatch" in raised
                and "non-matching form arguments" in raised)
    print(f"thermal_term_in_a_is_an_arity_mismatch={is_arity}")
    if is_arity and same:
        print("VERDICT=thermal_term_belongs_in_L_not_in_a")
        return 0
    print("VERDICT=no_arity_mismatch")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
