"""Tier-2: ds(id) works, and the ids are GEOMETRIC and 1-based.

An earlier revision of this catalog claimed that dune-fem has no facet
tagging and that ds(1) is silently empty. That was wrong, and the way
it was wrong is the pitfall worth gating: the test that produced it put
its Dirichlet condition on x=0 and then integrated over ds(1) — which
for a 2D YaspGrid IS x=0. Every test function on that edge had been
eliminated by the constraint, so the term vanished.

dune/fem/misc/boundaryidprovider.hh returns, for YaspGrid,

    intersection.boundary() ? intersection.indexInInside() + 1 : 0

so on a 2D structuredGrid the ids are 1 = x-min, 2 = x-max,
3 = y-min, 4 = y-max.

This fixture measures the partition directly (assemble 1*v*ds(k) and
sum it: each id must carry exactly one unit edge, and the four together
must equal the plain ds), then solves the Neumann problem twice with
the Dirichlet condition on two different edges and shows that the SAME
ds(id) term gives 0.0 or the exact answer depending only on which edge
the id names.

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
from dune.fem import assemble, integrate                   # noqa: E402
from dune.ufl import DirichletBC                           # noqa: E402
from ufl import (TrialFunction, TestFunction,              # noqa: E402
                 SpatialCoordinate, dot, grad, dx, ds,
                 conditional, lt)

TOL = 1e-8


def main() -> int:
    gridView = structuredGrid([0, 0], [1, 1], [8, 8])
    space = lagrange(gridView, order=1)
    u, v = TrialFunction(space), TestFunction(space)
    x = SpatialCoordinate(space)
    fail = []

    # 1. the ids partition the boundary: sum of 1*v*ds(k) is the length
    #    of the edge that id names.
    for k in (1, 2, 3, 4):
        rhs = float(np.array(assemble(1.0 * v * ds(k)).as_numpy).sum())
        print(f"ds{k}_boundary_length={rhs:.4f}")
        if abs(rhs - 1.0) > 1e-9:
            fail.append(f"ds({k}) measured {rhs}, expected one unit edge")
    empty = float(np.array(assemble(1.0 * v * ds(5)).as_numpy).sum())
    total = float(np.array(assemble(1.0 * v * ds).as_numpy).sum())
    print(f"ds5_boundary_length={empty:.4f}")
    print(f"ds_all_boundary_length={total:.4f}")
    if abs(empty) > 1e-12:
        fail.append("ds(5) is not empty; the id range is not 1..2*dim")
    if abs(total - 4.0) > 1e-9:
        fail.append(f"plain ds measured {total}, expected 4.0")

    # 2. the SAME ds(id) term is zero or exact depending only on which
    #    edge carries the Dirichlet condition.
    def solve(dirichlet_indicator, flux_id, name):
        scheme = galerkin(
            [dot(grad(u), grad(v)) * dx == 1.0 * v * ds(flux_id),
             DirichletBC(space, 0, dirichlet_indicator)],
            solver="cg", parameters={"linear.tolerance": 1e-14})
        uh = space.interpolate(0, name=name)
        scheme.solve(target=uh)
        return uh

    left = conditional(lt(x[0], TOL), 1, 0)
    uh_clamped = solve(left, 1, "clamped")     # flux on the CLAMPED edge
    uh_free = solve(left, 2, "free")           # flux on the FREE edge
    m_clamped = float(np.abs(np.array(uh_clamped.as_numpy)).max())
    m_free = float(np.array(uh_free.as_numpy).max())
    err = float(np.sqrt(integrate((uh_free - x[0]) ** 2,
                                  gridView=gridView, order=4)))
    print(f"flux_on_clamped_edge_max_abs={m_clamped:.6f}")
    print(f"flux_on_free_edge_max={m_free:.6f}")
    print(f"flux_on_free_edge_l2_error_vs_x={err:.3e}")
    if m_clamped > 1e-12:
        fail.append("flux on the clamped edge should contribute nothing")
    if abs(m_free - 1.0) > 1e-9 or err > 1e-12:
        fail.append("flux on ds(2) should reproduce u = x exactly")

    # 3. ds(0) is not a wildcard — it is rejected.
    try:
        assemble(1.0 * v * ds(0))
        print("ds0_rejected=False")
        fail.append("ds(0) was accepted; ids are documented as 1-based")
    except Exception as exc:
        print("ds0_rejected=True")
        print(f"ds0_exception={type(exc).__name__}")

    for f in fail:
        print(f"FAIL: {f}")
    if fail:
        return 1
    print("dune_boundary_ids_are_geometric_and_one_based=True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
