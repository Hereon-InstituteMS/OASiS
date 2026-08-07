"""Tier-2: a zero right-hand side written with Python zeros has no
integration domain (poisson#18).

inner(as_vector([0, 0]), v)*dx folds to Zero() inside UFL, and a Zero
carries no domain, so the measure has nothing to attach to and __rmul__
raises before dune-fem is involved at all. The two working spellings —
dune.ufl.Constant components, and `a == 0` — are both exercised so the
fixture proves a fix as well as a failure.

Verified by execution against dune-fem 2.12.0.2.
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

from dune.grid import structuredGrid                           # noqa: E402
from dune.fem.space import lagrange                            # noqa: E402
from dune.fem.scheme import galerkin                           # noqa: E402
from dune.ufl import Constant, DirichletBC                     # noqa: E402
from ufl import (TrialFunction, TestFunction, as_vector,        # noqa: E402
                 inner, grad, dx)


def main() -> int:
    fail: list[str] = []
    gridView = structuredGrid([0, 0], [1, 1], [4, 4])
    space = lagrange(gridView, order=1, dimRange=2)
    u, v = TrialFunction(space), TestFunction(space)
    a = inner(grad(u), grad(v)) * dx

    # 1. The broken spelling — pure UFL, no DUNE involvement.
    try:
        inner(as_vector([0, 0]), v) * dx
        print("python_zero_rhs_rejected=False")
        fail.append("inner(as_vector([0, 0]), v)*dx was accepted; UFL "
                    "no longer folds it to a domain-less Zero")
    except ValueError as exc:
        msg = " ".join(str(exc).split())
        print(f"python_zero_rhs_rejected={type(exc).__name__}")
        print(f"python_zero_rhs_message={msg[:160]}")
        if "missing an integration domain" not in msg:
            fail.append(f"the ValueError no longer says 'missing an "
                        f"integration domain': {msg[:160]}")

    # 2. Fix A — dune.ufl.Constant components carry the domain.
    b = inner(as_vector([Constant(0.0, name="fx"),
                         Constant(0.0, name="fy")]), v) * dx
    scheme = galerkin([a == b, DirichletBC(space, [0, 0])], solver="cg")
    uh = space.interpolate([0, 0], name="uh")
    info = scheme.solve(target=uh)
    print(f"constant_zero_rhs_solves={bool(info['converged'])}")
    if not info["converged"]:
        fail.append("the dune.ufl.Constant spelling of a zero rhs did "
                    "not solve")

    # 3. Fix B — `a == 0`, accepted as long as `a` keeps both arguments.
    scheme0 = galerkin([a == 0, DirichletBC(space, [0, 0])], solver="cg")
    uh0 = space.interpolate([0, 0], name="uh0")
    info0 = scheme0.solve(target=uh0)
    print(f"a_equals_zero_solves={bool(info0['converged'])}")
    if not info0["converged"]:
        fail.append("`a == 0` did not solve")

    # 4. And the reason it works: `a` still holds a trial AND a test
    #    function, which is what dune-fem requires.
    n_args = len(a.arguments())
    print(f"lhs_argument_count={n_args}")
    if n_args != 2:
        fail.append(f"the left-hand side has {n_args} arguments; the "
                    f"`a == 0` route depends on it having 2")

    if not fail:
        print("dune_zero_rhs_domain_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
