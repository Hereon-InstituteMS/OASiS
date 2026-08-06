"""Tier-2: a converged Newton and an enforced contact constraint are different
things -- the displacement criterion is satisfied to machine precision while the
gap is still violated by a percent of an element edge.

Claim: ngsolve contact#6 -- "Convergence criterion: check both displacement
residual AND contact gap violation.  Signal: a Newton solver that stops when
||du||/||u|| < 1e-6 alone can return with a still-active gap of 1-5%
element-edge size, because the gap residual scales differently from the
displacement residual.  Add an explicit max(min(gap, 0)) check below tol_gap."

Wrong variant: accepting the solve on the displacement criterion alone.

Setup: unit-square linear-elastic block, E = 1000, nu = 0.3, fixed on the top
edge, body force pressing it onto a rigid floor G0 = 0.01 below its bottom edge,
penalty contact through ngsolve.solvers.Newton.  gamma is chosen so the residual
gap lands inside the 1-5%-of-an-element-edge band the claim names.

The displacement criterion is measured, not assumed: from the converged state
one further Newton step is taken by hand and its ||du|| / ||u|| reported.  That
is the number a stopping rule of the claim's shape would look at.

What this fixture pins, all re-measured on this run:
  * ngsolve.solvers.Newton reports status 0 -- converged -- on this solve;
  * the next Newton increment is many orders below 1e-6, so the displacement
    criterion is not merely met but met with enormous margin;
  * the gap violation at that same state is inside the 1-5% band of an element
    edge, i.e. exactly the failure the claim describes;
  * the claim's recommended check, max(min(gap, 0)) against a tolerance, does
    fire on this state while the displacement check does not -- the two are
    computed on the same converged vector and compared;
  * raising gamma pushes the gap under the tolerance without changing the
    displacement verdict at all, so the gap check is carrying information the
    displacement check never had.
"""
from __future__ import annotations

import sys

import numpy
from netgen.geom2d import SplineGeometry
from ngsolve import (
    BilinearForm,
    GridFunction,
    Grad,
    Id,
    IfPos,
    InnerProduct,
    Mesh,
    Trace,
    Variation,
    VectorH1,
    ds,
    dx,
    solvers,
)

E, NU = 1000.0, 0.3
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
G0 = 0.01
FORCE = 150.0
MAXH = 0.2
TOL_DISP = 1e-6
TOL_GAP = 1e-3 * MAXH          # 0.1% of an element edge


def _mesh():
    g = SplineGeometry()
    g.AddRectangle((0, 0), (1, 1), bcs=["bot", "right", "top", "left"])
    return Mesh(g.GenerateMesh(maxh=MAXH))


def _sym(G):
    return 0.5 * (G + G.trans)


def _stress(G):
    return 2 * MU * _sym(G) + LAM * Trace(_sym(G)) * Id(2)


def solve(mesh, gamma):
    fes = VectorH1(mesh, order=1, dirichlet="top")
    u = fes.TrialFunction()
    a = BilinearForm(fes, symmetric=False)
    a += Variation(0.5 * InnerProduct(_stress(Grad(u)), _sym(Grad(u))) * dx)
    a += Variation(FORCE * u[1] * dx)
    p = -(u[1] + G0)
    a += Variation(0.5 * gamma * IfPos(p, p * p, 0) * ds("bot"))

    gfu = GridFunction(fes)
    gfu.vec[:] = 0
    ret = solvers.Newton(a, gfu, maxit=40, printing=False)

    # One more Newton step, by hand, to read off the increment a stopping rule
    # of the claim's shape would be testing.
    res = gfu.vec.CreateVector()
    du = gfu.vec.CreateVector()
    a.Apply(gfu.vec, res)
    a.AssembleLinearization(gfu.vec)
    du.data = a.mat.Inverse(fes.FreeDofs(), inverse="umfpack") * res
    nrm_du = float(numpy.linalg.norm(du.FV().NumPy()))
    nrm_u = float(numpy.linalg.norm(gfu.vec.FV().NumPy()))
    rel_du = nrm_du / max(nrm_u, 1e-30)

    uy = [float(gfu(mesh(float(t), 0.0))[1])
          for t in numpy.linspace(0.005, 0.995, 60)]
    # the claim's own expression: max over the boundary of min(gap, 0), where
    # gap = u_y + G0 is non-negative when the constraint holds
    gap_violation = -min(0.0, min(g + G0 for g in uy))
    return ret, rel_du, gap_violation


def main() -> int:
    mesh = _mesh()
    edge = MAXH

    gamma = 1e4
    ret, rel_du, gap = solve(mesh, gamma)
    pct = 100 * gap / edge
    print(f"gamma={gamma:g}")
    print(f"newton_status={ret[0]} numit={ret[1]}")
    print(f"newton_reports_converged={ret[0] == 0}")
    print(f"next_increment_relative_norm={rel_du:.6e}")
    print(f"displacement_criterion_satisfied={rel_du < TOL_DISP}")
    print(f"displacement_criterion_margin_decades="
          f"{numpy.log10(TOL_DISP / max(rel_du, 1e-300)):.1f}")
    print(f"gap_violation={gap:.6e}")
    print(f"gap_violation_pct_of_element_edge={pct:.4f}")
    print(f"gap_violation_in_claimed_1_to_5_pct_band={1.0 <= pct <= 5.0}")
    print(f"gap_criterion_fires={gap > TOL_GAP}")
    print(f"displacement_ok_but_gap_violated="
          f"{rel_du < TOL_DISP and gap > TOL_GAP}")

    # Raising gamma fixes the gap; the displacement verdict never moved.
    ret2, rel_du2, gap2 = solve(mesh, 1e7)
    print(f"tight_gamma_newton_status={ret2[0]}")
    print(f"tight_gamma_next_increment={rel_du2:.6e}")
    print(f"tight_gamma_gap_violation={gap2:.6e}")
    print(f"tight_gamma_displacement_criterion_satisfied={rel_du2 < TOL_DISP}")
    print(f"tight_gamma_gap_criterion_clear={gap2 < TOL_GAP}")
    print(f"displacement_verdict_identical_both_runs="
          f"{(rel_du < TOL_DISP) == (rel_du2 < TOL_DISP)}")
    print(f"gap_verdict_changed_between_runs="
          f"{(gap > TOL_GAP) != (gap2 > TOL_GAP)}")

    ok = (
        ret[0] == 0
        and rel_du < TOL_DISP
        and 1.0 <= pct <= 5.0
        and gap > TOL_GAP
        and ret2[0] == 0
        and rel_du2 < TOL_DISP
        and gap2 < TOL_GAP
    )
    if ok:
        return 0
    print("FAIL: contact convergence-criterion invariant not held",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
