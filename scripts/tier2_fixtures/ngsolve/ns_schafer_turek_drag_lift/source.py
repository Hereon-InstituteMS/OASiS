"""Tier-2: the DFG-2D-1 benchmark, run for real -- drag AND lift land inside the
published bounds, and a broken inlet profile takes them straight out.

Claim: ngsolve time_dependent_ns#6 -- "Benchmark: DFG Schafer-Turek (Re=20
steady, Re=100 periodic vortex shedding) and the lid-driven cavity at Re=400,
1000, 5000.  Signal: a transient NS implementation with VectorH1 + H1
BilinearForm should reproduce Schafer-Turek drag/lift (post-processed via
Integrate over the cylinder BND) to within published bounds (Cd ~ 5.57 at Re=20)
and lid-cavity Ghia-streamfunction values computed from the GridFunction at the
chosen Re; values 5%+ off expose either a missing convective term, mis-tuned
theta, or insufficient mesh near walls."

Wrong variant: the same solve with the inlet profile replaced by a uniform plug,
one of the boundary-condition errors the claim says the benchmark exposes.

Setup: DFG-2D-1, the steady Re = 20 case.  Channel 2.2 x 0.41, cylinder radius
0.05 centred at (0.2, 0.2), nu = 1e-3, parabolic inlet with peak 0.3 so the mean
is 0.2 and Re = U_mean * D / nu = 20.  Taylor-Hood VectorH1 order k * H1 order
k-1 on a curved mesh, steady Navier-Stokes solved with ngsolve.solvers.Newton.
Drag and lift come from the residual functional -- the assembled residual tested
against a function that is 1 on the cylinder in the drag or lift direction --
which is the variational route and is more accurate than integrating the stress
directly.  Published bounds: Cd in [5.5700, 5.5900], Cl in [0.0104, 0.0110].

Departure from the claim, recorded: the claim says values 5%+ off expose a
problem.  Cd is far less sensitive than that suggests -- the coarsest mesh tried
here already puts Cd inside the published band while its Cl is off by a factor
of nearly three.  Checking drag alone passes a mesh that is badly wrong.

What this fixture pins, all re-measured on this run:
  * Newton converges (status 0) on every mesh;
  * the finest run's Cd AND Cl both land inside the published bounds --
    this is the benchmark passing, not a self-comparison;
  * Cd is inside the band even on the coarsest mesh, while Cl is not, so drag
    alone is not a sufficient check;
  * replacing the parabolic inlet with a plug of the same mean velocity moves Cd
    outside the published band, so the benchmark does detect the BC error the
    claim says it detects.
"""
from __future__ import annotations

import sys

from netgen.geom2d import SplineGeometry
from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    GridFunction,
    Grad,
    H1,
    InnerProduct,
    Mesh,
    VectorH1,
    VectorH1 as _V,
    div,
    dx,
    solvers,
    y,
)

NU = 1e-3
UMAX = 0.3
UBAR = 2 * UMAX / 3
D = 0.1
CD_BOUNDS = (5.5700, 5.5900)
CL_BOUNDS = (0.0104, 0.0110)


def dfg_mesh(maxh, cylh, order):
    g = SplineGeometry()
    g.AddRectangle((0, 0), (2.2, 0.41),
                   bcs=("wall", "outlet", "wall", "inlet"),
                   leftdomain=1, rightdomain=0)
    g.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1,
                bc="cyl", maxh=cylh)
    mesh = Mesh(g.GenerateMesh(maxh=maxh))
    mesh.Curve(2 * order)
    return mesh


def run(maxh, cylh, order, inlet="parabolic"):
    mesh = dfg_mesh(maxh, cylh, order)
    V = VectorH1(mesh, order=order, dirichlet="wall|inlet|cyl")
    Q = H1(mesh, order=order - 1)
    X = V * Q
    (u, p), (v, q) = X.TnT()
    a = BilinearForm(X)
    a += (NU * InnerProduct(Grad(u), Grad(v))
          + InnerProduct(Grad(u) * u, v)
          - div(u) * q - div(v) * p) * dx
    gfu = GridFunction(X)
    if inlet == "parabolic":
        uin = CoefficientFunction((4 * UMAX * y * (0.41 - y) / 0.41 ** 2, 0))
    else:                                     # plug with the SAME mean
        uin = CoefficientFunction((UBAR, 0))
    gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))
    ret = solvers.Newton(a, gfu, maxit=25, printing=False, inverse="umfpack")

    dts = GridFunction(X)
    dts.components[0].Set(CoefficientFunction((1, 0)),
                          definedon=mesh.Boundaries("cyl"))
    lts = GridFunction(X)
    lts.components[0].Set(CoefficientFunction((0, 1)),
                          definedon=mesh.Boundaries("cyl"))
    res = gfu.vec.CreateVector()
    a.Apply(gfu.vec, res)
    cd = -2 * InnerProduct(res, dts.vec) / (UBAR ** 2 * D)
    cl = -2 * InnerProduct(res, lts.vec) / (UBAR ** 2 * D)
    return mesh.ne, X.ndof, ret, float(cd), float(cl)


def inside(v, lo_hi):
    return lo_hi[0] <= v <= lo_hi[1]


def main() -> int:
    print(f"reynolds_number={UBAR * D / NU:g}")
    print(f"published_cd_bounds={CD_BOUNDS}")
    print(f"published_cl_bounds={CL_BOUNDS}")

    rows = []
    for maxh, cylh, order in ((0.15, 0.06, 2), (0.08, 0.03, 3),
                              (0.05, 0.015, 3)):
        ne, nd, ret, cd, cl = run(maxh, cylh, order)
        rows.append((ne, nd, ret, cd, cl))
        print(f"elements={ne} ndof={nd} order={order} newton_status={ret[0]} "
              f"newton_iters={ret[1]} Cd={cd:.5f} Cl={cl:.6f} "
              f"cd_in_bounds={inside(cd, CD_BOUNDS)} "
              f"cl_in_bounds={inside(cl, CL_BOUNDS)}")

    print(f"newton_converged_on_every_mesh={all(r[2][0] == 0 for r in rows)}")
    fine = rows[-1]
    print(f"finest_cd={fine[3]:.5f}")
    print(f"finest_cl={fine[4]:.6f}")
    print(f"finest_cd_inside_published_bounds={inside(fine[3], CD_BOUNDS)}")
    print(f"finest_cl_inside_published_bounds={inside(fine[4], CL_BOUNDS)}")
    print(f"benchmark_reproduced="
          f"{inside(fine[3], CD_BOUNDS) and inside(fine[4], CL_BOUNDS)}")

    coarse = rows[0]
    print(f"coarse_cd_inside_bounds={inside(coarse[3], CD_BOUNDS)}")
    print(f"coarse_cl_inside_bounds={inside(coarse[4], CL_BOUNDS)}")
    print(f"drag_alone_would_pass_the_coarse_mesh="
          f"{inside(coarse[3], CD_BOUNDS) and not inside(coarse[4], CL_BOUNDS)}")

    ne, nd, ret, cd_p, cl_p = run(0.08, 0.03, 3, inlet="plug")
    print(f"plug_inlet_newton_status={ret[0]}")
    print(f"plug_inlet_Cd={cd_p:.5f}")
    print(f"plug_inlet_Cl={cl_p:.6f}")
    print(f"plug_inlet_cd_leaves_the_band={not inside(cd_p, CD_BOUNDS)}")
    print(f"benchmark_detects_the_bc_error={not inside(cd_p, CD_BOUNDS)}")

    ok = (
        all(r[2][0] == 0 for r in rows)
        and inside(fine[3], CD_BOUNDS) and inside(fine[4], CL_BOUNDS)
        and inside(coarse[3], CD_BOUNDS) and not inside(coarse[4], CL_BOUNDS)
        and not inside(cd_p, CD_BOUNDS)
    )
    if ok:
        return 0
    print("FAIL: Schafer-Turek benchmark invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
