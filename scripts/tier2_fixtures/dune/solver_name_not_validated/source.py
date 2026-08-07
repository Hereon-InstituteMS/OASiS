"""Tier-2: the DUNE-fem `solver=` string is not validated in Python.

dune/fem/discretefunction/_solvers.py::femsolver just forwards whatever
string it is given as the C++ parameter ``linear.method``. The check
happens inside DUNE's parameter reader when the scheme is constructed,
so a plausible-looking name produces a RuntimeError that names a
parameter the caller never wrote:

    RuntimeError: ParameterInvalid [getEnumeration:.../dune/fem/io/
    parameter/reader.hh:300]: Parameter 'fem.solver.linear.method'
    invalid. Valid values are: gmres, cg, bicgstab

That message is also the authoritative list of accepted Krylov methods
for the default (numpy/"fem") storage — this fixture harvests it rather
than trusting documentation. The valid names are asserted to work.

Verified by execution against dune-fem 2.12.0.2 on 2026-08-03.
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

from dune.grid import structuredGrid                       # noqa: E402
from dune.fem.space import lagrange                        # noqa: E402
from dune.fem.scheme import galerkin                       # noqa: E402
from dune.ufl import DirichletBC                           # noqa: E402
from ufl import (TrialFunction, TestFunction,              # noqa: E402
                 dot, grad, dx)


def main() -> int:
    gridView = structuredGrid([0, 0], [1, 1], [4, 4])
    space = lagrange(gridView, order=1)
    u, v = TrialFunction(space), TestFunction(space)
    a = dot(grad(u), grad(v)) * dx
    b = 1.0 * v * dx
    eqn = [a == b, DirichletBC(space, 0)]

    fail = []

    # 1. A plausible but wrong name must be REJECTED, and the message
    #    must name the parameter and enumerate the alternatives.
    msg = ""
    try:
        galerkin(eqn, solver="conjugate_gradient")
        print("bogus_solver_rejected=False")
        fail.append("solver='conjugate_gradient' was ACCEPTED — the "
                    "parameter enumeration no longer guards it")
    except Exception as exc:                          # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        print("bogus_solver_rejected=True")
        print("bogus_solver_error_class=" + type(exc).__name__)
        print("bogus_solver_names_the_parameter="
              + str("fem.solver.linear.method" in msg))
        one_line = " ".join(msg.split())
        print(f"bogus_solver_message={one_line[:300]}")
        if "fem.solver.linear.method" not in msg:
            fail.append("error message no longer names "
                        "'fem.solver.linear.method'")
        for name in ("gmres", "cg", "bicgstab"):
            if name not in msg:
                fail.append(f"error message no longer enumerates "
                            f"'{name}'")

    # 2. Every name the error message advertises must actually work.
    for name in ("cg", "gmres", "bicgstab"):
        try:
            scheme = galerkin(eqn, solver=name)
            uh = space.interpolate(0, name="uh")
            info = scheme.solve(target=uh)
            print(f"solver_{name}_converged={bool(info['converged'])}")
            if not info["converged"]:
                fail.append(f"advertised solver '{name}' did not "
                            f"converge on a 4x4 Poisson problem")
        except Exception as exc:                      # noqa: BLE001
            print(f"solver_{name}_converged=ERROR")
            fail.append(f"advertised solver '{name}' raised "
                        f"{type(exc).__name__}: {exc}")

    if not fail:
        print("dune_solver_enumeration_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
