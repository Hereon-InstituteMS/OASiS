"""Tier-2: component-wise DirichletBCs on edges that MEET lose one
constraint at the shared corner dof.

The roller/symmetry boundary condition of every 2D solid-mechanics
model is two component-wise constraints on two edges — u_x = 0 on
x = 0 and u_y = 0 on y = 0. On dune-fem 2.12.0.2 the dof where the two
edges meet keeps only ONE of the two masks. Nothing is raised, the
solve converges, and the resulting global error is O(h^2), so on a
coarse mesh it is easily mistaken for discretisation error.

Two component-wise BCs on the SAME edge DO merge correctly, which is
asserted here as the control: it rules out "component masks never
merge" as the explanation and pins the failure to the shared corner.

Verified by execution against dune-fem 2.12.0.2 on 2026-08-03.
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np                                         # noqa: E402
from dune.grid import structuredGrid                       # noqa: E402
from dune.fem.space import lagrange                        # noqa: E402
from dune.fem.scheme import galerkin                       # noqa: E402
from dune.ufl import DirichletBC                           # noqa: E402
from ufl import (TrialFunction, TestFunction,              # noqa: E402
                 SpatialCoordinate, Identity, as_vector,
                 grad, inner, sym, tr, dx, ds, conditional, lt)

E, NU, TRACTION = 210e9, 0.3, 1.0e6
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
TOL = 1e-8
PARAMS = {"linear.tolerance": 1e-14, "linear.maxiterations": 100000}


def solve(space, a, L, bcs, name):
    scheme = galerkin([a == L] + bcs, solver="cg", parameters=PARAMS)
    uh = space.interpolate([0, 0], name=name)
    scheme.solve(target=uh)
    return np.array(uh.as_numpy).reshape(-1, 2)


def main() -> int:
    gridView = structuredGrid([0, 0], [1, 1], [4, 4])
    space = lagrange(gridView, dimRange=2, order=1)
    u, v = TrialFunction(space), TestFunction(space)
    x = SpatialCoordinate(space)
    I = Identity(2)

    def eps(w):
        return sym(grad(w))

    def sigma(w):
        return LAM * tr(eps(w)) * I + 2 * MU * eps(w)

    a = inner(sigma(u), eps(v)) * dx
    L = conditional(lt(1.0 - x[0], TOL), TRACTION * v[0], 0.0) * ds
    left = conditional(lt(x[0], TOL), 1, 0)
    bottom = conditional(lt(x[1], TOL), 1, 0)

    # nodal coordinates, so the corner dof can be found and read back
    X = np.array(space.interpolate(as_vector([x[0], x[1]]),
                                   name="X").as_numpy).reshape(-1, 2)
    i00 = int(np.argmin(np.hypot(X[:, 0], X[:, 1])))

    fail = []

    # A. two component-wise BCs on edges that MEET
    vals = solve(space, a, L,
                 [DirichletBC(space, [0, None], left),
                  DirichletBC(space, [None, 0], bottom)], "meet")
    ux00, uy00 = float(vals[i00, 0]), float(vals[i00, 1])
    print(f"corner_ux_should_be_zero={ux00:.3e}")
    print(f"corner_uy_should_be_zero={uy00:.3e}")
    dropped = abs(ux00) > 1e-12 or abs(uy00) > 1e-12
    print(f"corner_constraint_dropped={dropped}")
    if not dropped:
        fail.append("the shared corner honoured BOTH component masks; "
                    "the catalog pitfall is stale and should be "
                    "removed")

    # A'. the list order must NOT matter (rules out "last wins")
    vals_rev = solve(space, a, L,
                     [DirichletBC(space, [None, 0], bottom),
                      DirichletBC(space, [0, None], left)], "meet_rev")
    same = bool(np.allclose(vals, vals_rev, rtol=0, atol=1e-18))
    print(f"corner_result_independent_of_list_order={same}")
    if not same:
        fail.append("swapping the two BCs changed the answer, so the "
                    "rule is list-order dependent after all")

    # B. CONTROL: two component-wise BCs on the SAME edge must merge
    vals_same = solve(space, a, L,
                      [DirichletBC(space, [0, None], left),
                       DirichletBC(space, [None, 0], left)], "same_edge")
    on_left = X[:, 0] < TOL
    mx = float(np.abs(vals_same[on_left, 0]).max())
    my = float(np.abs(vals_same[on_left, 1]).max())
    print(f"same_edge_max_abs_ux={mx:.3e}")
    print(f"same_edge_max_abs_uy={my:.3e}")
    merged = mx < 1e-12 and my < 1e-12
    print(f"same_edge_masks_merge={merged}")
    if not merged:
        fail.append("component masks on the SAME edge did not merge; "
                    "the pitfall is broader than the catalog says")

    for f in fail:
        print(f"FAIL: {f}")
    if fail:
        return 1
    print("dune_componentwise_bc_corner_trap_reproduced=True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
