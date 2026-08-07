"""Tier-2: a DirichletBC indicator is evaluated per BOUNDARY FACET.

  linear_elasticity#2                        an indicator that is true
                                             only at a POINT selects
                                             nothing: adding
                                             DirichletBC(space,[0,0],
                                             And(x[0]<tol, x[1]<tol)) to
                                             pin the corner node changes
                                             the solution by exactly
                                             zero, because no facet
                                             quadrature point satisfies
                                             both tests.
  _general natural_bc_measured.
  Signal_pointwise_indicator                 the same measurement.

The form is character-for-character the one in
componentwise_bc_corner_drops_a_constraint, so this fixture reuses that
compiled module instead of minting another. The controls that make the
null result meaningful are in the same run: an EDGE-sized indicator on
the same corner does change the solution, and the facet-quadrature
explanation is checked directly by evaluating the point indicator at
the facet centres.

Verified by execution against dune-fem 2.12.0.2.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from dune.grid import structuredGrid                            # noqa: E402
from dune.fem.space import lagrange                             # noqa: E402
from dune.fem.scheme import galerkin                            # noqa: E402
from dune.ufl import DirichletBC                                # noqa: E402
from ufl import (TrialFunction, TestFunction,                    # noqa: E402
                 SpatialCoordinate, Identity, as_vector,
                 grad, inner, sym, tr, dx, ds, conditional, lt, And)

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
    fail: list[str] = []
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
    corner_point = conditional(And(lt(x[0], TOL), lt(x[1], TOL)), 1, 0)

    X = np.array(space.interpolate(as_vector([x[0], x[1]]),
                                   name="X").as_numpy).reshape(-1, 2)
    i00 = int(np.argmin(np.hypot(X[:, 0], X[:, 1])))
    print(f"corner_node_coordinates={X[i00].tolist()}")

    base_bcs = [DirichletBC(space, [0, None], left),
                DirichletBC(space, [None, 0], bottom)]
    base = solve(space, a, L, base_bcs, "base")
    print(f"corner_value_without_extra_bc="
          f"{base[i00, 0]:.6e},{base[i00, 1]:.6e}")

    # ── the point indicator: accepted, and completely inert ────────
    with_point = solve(space, a, L,
                       base_bcs + [DirichletBC(space, [0, 0],
                                               corner_point)],
                       "with_point")
    delta_point = float(np.abs(with_point - base).max())
    print(f"point_bc_was_accepted=True")
    print(f"point_bc_max_change={delta_point:.6e}")
    print(f"point_bc_changed_nothing={delta_point == 0.0}")
    print(f"corner_value_with_point_bc="
          f"{with_point[i00, 0]:.6e},{with_point[i00, 1]:.6e}")
    if delta_point != 0.0:
        fail.append(f"the point-sized BC changed the solution by "
                    f"{delta_point:.6e}; the claim is that it changes "
                    f"it by EXACTLY zero because no facet is selected")

    # The decisive assertion: the constraint the caller ASKED for is not
    # satisfied. Without this, an indicator that happens to be redundant
    # with an existing BC would also "change nothing" and the fixture
    # would pass with the pitfall absent — found by mutation testing.
    corner_ux = float(abs(with_point[i00, 0]))
    print(f"corner_ux_with_point_bc={corner_ux:.6e}")
    print(f"corner_is_not_zero_despite_the_bc={corner_ux > 1e-9}")
    if corner_ux <= 1e-9:
        fail.append(f"the corner dof came back at {corner_ux:.6e}, i.e. "
                    f"the BC the caller wrote WAS honoured; the claim "
                    f"is that a point indicator selects no facet and "
                    f"the dof keeps its unconstrained value")

    # ── control: an EDGE-sized indicator DOES act ──────────────────
    # The edge has to be one the base does not already constrain in
    # that component, otherwise the merge rule makes the extra BC
    # redundant and a null result would prove nothing. Pinning u_y on
    # the TOP edge fights the Poisson contraction, so it must move the
    # solution.
    top = conditional(lt(1.0 - x[1], TOL), 1, 0)
    with_edge = solve(space, a, L,
                      base_bcs + [DirichletBC(space, [None, 0], top)],
                      "with_edge")
    delta_edge = float(np.abs(with_edge - base).max())
    print(f"edge_bc_max_change={delta_edge:.6e}")
    print(f"edge_bc_changed_the_solution={delta_edge > 0.0}")
    if delta_edge <= 0.0:
        fail.append("an EDGE-sized BC also changed nothing, so this "
                    "fixture cannot distinguish 'point indicators "
                    "select no facet' from 'extra BCs never matter'")

    # ── why: no boundary facet CENTRE satisfies both tests ─────────
    n_facets = 0
    n_corner_facets = 0
    for element in gridView.elements:
        for intersection in gridView.intersections(element):
            if not intersection.boundary:
                continue
            n_facets += 1
            centre = intersection.geometry.center
            if abs(centre[0]) < TOL and abs(centre[1]) < TOL:
                n_corner_facets += 1
    print(f"boundary_facets={n_facets}")
    print(f"facet_centres_satisfying_the_point_test={n_corner_facets}")
    print(f"no_facet_centre_is_the_corner={n_corner_facets == 0}")
    if n_facets == 0:
        fail.append("no boundary facets were found, so the geometric "
                    "explanation could not be checked")
    if n_corner_facets != 0:
        fail.append(f"{n_corner_facets} facet centres satisfy the "
                    f"point test; the explanation for the null result "
                    f"does not hold on this grid")

    if not fail:
        print("dune_point_indicator_selects_nothing_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
