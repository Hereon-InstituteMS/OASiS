"""Tier-2: get the contact normal's sign wrong and the penalty drives the
bodies together -- and it does so while Newton still reports success.

Claim: ngsolve contact#3 -- "Contact normal must be consistent with mesh
boundary orientation.  Signal: a flipped normal causes penalty to PUSH bodies
INTO each other instead of separating them -- gap goes negative without bound;
check the sign by evaluating specialcf.normal(2 or 3).dot(n_expected) on the
contact boundary."

Wrong variant: the penalty term entered with the opposite sign, which is what a
flipped normal produces -- a reaction along +n instead of -n.

Setup: unit-square linear-elastic block, E = 1000, nu = 0.3, fixed on the top
edge, body force pressing it onto a rigid floor G0 = 0.01 below its bottom edge.
The correct penalty energy is +0.5*gamma*<-(u_y+G0)>_+^2 on the bottom edge; the
flipped one is the same with a minus, which is exactly the residual a
wrong-signed normal gives.

What this fixture pins, all re-measured on this run:
  * the diagnostic the claim recommends actually works: specialcf.normal(2)
    integrated over the bottom edge is (0, -1), the outward normal, and it is
    (0, +1) on the top edge, so the check discriminates between the two;
  * with the correct sign, penetration is smaller than the unconstrained
    deflection -- the penalty separates;
  * with the flipped sign at the SAME gamma, penetration is larger than the
    unconstrained deflection -- the penalty pushes in, which is the claim's
    assertion, measured against the no-contact run rather than asserted;
  * raising gamma makes it worse rather than better: the flipped penetration
    grows by roughly an order of magnitude per decade of gamma, "without bound"
    in the claim's phrasing;
  * and the failure is quiet -- ngsolve.solvers.Newton returns status 0 on the
    flipped runs.  Solver status is not a guard against this.
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
    Integrate,
    Mesh,
    Trace,
    Variation,
    VectorH1,
    ds,
    dx,
    solvers,
    specialcf,
)

E, NU = 1000.0, 0.3
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
G0 = 0.01
FORCE = 150.0
MAXH = 0.2


def _mesh():
    g = SplineGeometry()
    g.AddRectangle((0, 0), (1, 1), bcs=["bot", "right", "top", "left"])
    return Mesh(g.GenerateMesh(maxh=MAXH))


def _sym(G):
    return 0.5 * (G + G.trans)


def _stress(G):
    return 2 * MU * _sym(G) + LAM * Trace(_sym(G)) * Id(2)


def solve(mesh, gamma, sign):
    fes = VectorH1(mesh, order=1, dirichlet="top")
    u = fes.TrialFunction()
    a = BilinearForm(fes, symmetric=False)
    a += Variation(0.5 * InnerProduct(_stress(Grad(u)), _sym(Grad(u))) * dx)
    a += Variation(FORCE * u[1] * dx)
    if sign != 0:
        p = -(u[1] + G0)
        a += Variation(sign * 0.5 * gamma * IfPos(p, p * p, 0) * ds("bot"))
    gfu = GridFunction(fes)
    gfu.vec[:] = 0
    ret = solvers.Newton(a, gfu, maxit=40, printing=False)
    uy = [float(gfu(mesh(float(t), 0.0))[1])
          for t in numpy.linspace(0.005, 0.995, 60)]
    return ret, max(0.0, -(min(uy) + G0))


def main() -> int:
    mesh = _mesh()

    # --- the diagnostic the claim recommends ---------------------------
    n = specialcf.normal(2)
    nb = (float(Integrate(n[0] * ds("bot"), mesh)),
          float(Integrate(n[1] * ds("bot"), mesh)))
    nt = (float(Integrate(n[0] * ds("top"), mesh)),
          float(Integrate(n[1] * ds("top"), mesh)))
    print(f"bottom_normal_integral=({nb[0]:.6f},{nb[1]:.6f})")
    print(f"top_normal_integral=({nt[0]:.6f},{nt[1]:.6f})")
    print(f"bottom_outward_normal_is_minus_y="
          f"{abs(nb[0]) < 1e-10 and abs(nb[1] + 1.0) < 1e-10}")
    print(f"normal_check_discriminates_the_two_edges="
          f"{abs(nb[1] - nt[1]) > 1.9}")

    # --- reference: no contact at all ----------------------------------
    ret_free, pen_free = solve(mesh, 0.0, 0)
    print(f"no_contact_status={ret_free[0]} penetration={pen_free:.6e}")

    gamma = 1e3
    ret_ok, pen_ok = solve(mesh, gamma, +1)
    ret_bad, pen_bad = solve(mesh, gamma, -1)
    print(f"correct_sign status={ret_ok[0]} penetration={pen_ok:.6e}")
    print(f"flipped_sign status={ret_bad[0]} penetration={pen_bad:.6e}")
    print(f"correct_sign_separates={pen_ok < pen_free}")
    print(f"flipped_sign_pushes_in={pen_bad > pen_free}")

    # --- and it gets worse with gamma ----------------------------------
    pens = []
    for gm in (1e2, 1e3):
        r, p = solve(mesh, gm, -1)
        pens.append((gm, r, p))
        print(f"flipped gamma={gm:g} status={r[0]} penetration={p:.6e}")
    worse = pens[1][2] > pens[0][2]
    print(f"flipped_penetration_grows_with_gamma={worse}")
    growth = pens[1][2] / max(pens[0][2], 1e-30)
    print(f"flipped_growth_per_decade={growth:.2f}")
    print(f"flipped_growth_at_least_5x_per_decade={growth > 5.0}")

    quiet = all(r[1][0] == 0 for r in pens) and ret_bad[0] == 0
    print(f"flipped_runs_all_report_newton_status_0={quiet}")
    print(f"solver_status_is_not_a_guard_here={quiet}")

    ok = (
        abs(nb[0]) < 1e-10 and abs(nb[1] + 1.0) < 1e-10
        and abs(nb[1] - nt[1]) > 1.9
        and pen_ok < pen_free
        and pen_bad > pen_free
        and worse and growth > 5.0
        and quiet
    )
    if ok:
        return 0
    print("FAIL: contact-normal orientation invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
