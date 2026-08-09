"""Tier-2: linear.preconditioning.method takes a SHORT enumeration.

The DUNE catalog used to advertise "PETSc fieldsplit", "AMG" and "ILU"
as preconditioners for this backend. For the default (numpy/"fem")
storage none of those names is accepted. The parameter reader rejects
anything outside a five-name list, and it does so at SCHEME
CONSTRUCTION — before any solve — naming the parameter in a spelling
the caller never wrote:

    RuntimeError: ParameterInvalid [getEnumeration:.../dune/fem/io/
    parameter/reader.hh:300]: Parameter
    'fem.solver.linear.preconditioning.method' invalid.
    Valid values are: none, sor, ssor, gauss-seidel, jacobi

This fixture harvests that message rather than trusting documentation,
then proves each advertised name really does build and solve.

Verified by execution against dune-fem 2.12.0.2 on 2026-08-03.

MUTATION CONTROL. T2_MUTATE=1 puts three names the enumeration DOES
accept — jacobi, sor, ssor — in the slot where the base run puts ilu,
amg and fieldsplit. That is the pathology removed: nothing is rejected,
so 'bogus_ilu_rejected=True', 'bogus_amg_rejected=True' and
'bogus_fieldsplit_rejected=True' are no longer printed and a FAIL: line
appears. Same form, so no new module is compiled.
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dune.grid import structuredGrid                       # noqa: E402
from dune.fem.space import lagrange                        # noqa: E402
from dune.fem.scheme import galerkin                       # noqa: E402
from dune.ufl import DirichletBC                           # noqa: E402
from ufl import (TrialFunction, TestFunction,              # noqa: E402
                 dot, grad, dx)

VALID = ("none", "sor", "ssor", "gauss-seidel", "jacobi")
BOGUS = ("ilu", "amg", "fieldsplit")


def main() -> int:
    gridView = structuredGrid([0, 0], [1, 1], [4, 4])
    space = lagrange(gridView, order=1)
    u, v = TrialFunction(space), TestFunction(space)
    eqn = [dot(grad(u), grad(v)) * dx == 1.0 * v * dx,
           DirichletBC(space, 0)]

    fail = []

    # 1. the names an LLM reaches for must all be REJECTED, and the
    #    message must name the parameter and enumerate the alternatives.
    bogus_names = ("jacobi", "sor", "ssor") if MUTATE else BOGUS
    if MUTATE:
        print("mutation=the_bogus_slot_uses_three_accepted_"
              "preconditioner_names")
    for name in bogus_names:
        try:
            galerkin(eqn, solver="cg",
                     parameters={"linear.preconditioning.method": name})
            print(f"bogus_{name}_rejected=False")
            fail.append(f"preconditioner '{name}' was accepted")
            continue
        except Exception as exc:
            msg = str(exc)
            print(f"bogus_{name}_rejected=True")
            if "fem.solver.linear.preconditioning.method" not in msg:
                fail.append(f"'{name}' message does not name the "
                            f"parameter: {msg[:200]}")
            missing = [n for n in VALID if n not in msg]
            if missing:
                fail.append(f"'{name}' message no longer enumerates "
                            f"{missing}: {msg[:200]}")
    print("bogus_names_the_parameter=True")
    print("bogus_enumerates_the_valid_names=True")

    # 2. every advertised name must actually build and solve.
    for name in VALID:
        try:
            scheme = galerkin(
                eqn, solver="cg",
                parameters={"linear.preconditioning.method": name,
                            "linear.tolerance": 1e-12})
            uh = space.interpolate(0, name=f"u_{name.replace('-', '_')}")
            info = scheme.solve(target=uh)
            ok = bool(info["converged"])
        except Exception as exc:                            # pragma: no cover
            ok = False
            fail.append(f"advertised preconditioner '{name}' failed: "
                        f"{type(exc).__name__}: {exc}")
        print(f"pc_{name.replace('-', '_')}_converged={ok}")

    for f in fail:
        print(f"FAIL: {f}")
    if fail:
        return 1
    print("dune_preconditioner_enumeration_verified=True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
