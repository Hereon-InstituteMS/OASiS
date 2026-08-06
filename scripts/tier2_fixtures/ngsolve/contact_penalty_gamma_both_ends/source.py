"""Tier-2: the penalty parameter has a floor and a ceiling -- but the ceiling
does NOT announce itself the way the claim says.

Claim: ngsolve contact#0 -- "Penalty parameter gamma: too small -> contact not
enforced; too large -> ill-conditioning.  Signal: too small gives max
penetration > 5% of element edge; too large produces NewtonMinimization
`DivisionByZero` / cond(K)>1e14 warnings from the sparse solver."

Wrong variant: the same obstacle problem with gamma far below, and far above,
the usable range.

Setup: unit-square linear-elastic block, E = 1000, nu = 0.3, fixed on the top
edge, body force pushing it down onto a rigid floor a distance G0 = 0.01 below
its bottom edge.  Penalty energy 0.5*gamma*<-(u_y+G0)>_+^2 on the bottom edge,
solved with ngsolve.solvers.Newton.  Penetration is read off the deformed
bottom edge at 60 points and reported as a fraction of the element edge length.

CORRECTION this fixture records.  The small-gamma half is right: penetration
does exceed 5% of an element edge at low gamma and falls below it as gamma
rises.  The large-gamma half names messages NGSolve 6.2.2604 never emits.  There
is no `DivisionByZero`, no `NewtonMinimization`, and no condition-number warning
anywhere in a run at gamma = 1e22.  What does happen is that
ngsolve.solvers.Newton stops converging and says so, in different words:
"Warning: Newton might not converge! Error = ".  Its return value is the tell --
(status, numit) with status -1 -- and an agent grepping for the claimed strings
finds nothing and concludes the solve was clean.

The return tuple is (status, numit), status 0 = converged.  Read the other way
round, a converged solve looks like 3 iterations and a diverged one like 0
iterations, which inverts the meaning of the check.

What this fixture pins, all re-measured on this run:
  * Newton's return is a 2-tuple whose FIRST entry is the status, confirmed
    against its own docstring and against a run that is known to converge;
  * at the low end penetration exceeds 5% of the element edge, at the high end
    it does not, so the claim's 5% observable discriminates;
  * across the usable range Newton reports status 0;
  * at gamma = 1e22 Newton reports status -1 and consumes its whole budget;
  * none of DivisionByZero / NewtonMinimization / cond appears in the captured
    output of that run -- checked by capturing stdout and stderr, not asserted.
"""
from __future__ import annotations

import contextlib
import io
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
MAXIT = 40
CLAIMED_STRINGS = ("DivisionByZero", "NewtonMinimization", "cond(")


def _mesh():
    g = SplineGeometry()
    g.AddRectangle((0, 0), (1, 1), bcs=["bot", "right", "top", "left"])
    return Mesh(g.GenerateMesh(maxh=MAXH))


def _strain(G):
    return 0.5 * (G + G.trans)


def _stress(G):
    return 2 * MU * _strain(G) + LAM * Trace(_strain(G)) * Id(2)


def solve(mesh, gamma):
    fes = VectorH1(mesh, order=1, dirichlet="top")
    u = fes.TrialFunction()
    a = BilinearForm(fes, symmetric=False)
    a += Variation(0.5 * InnerProduct(_stress(Grad(u)), _strain(Grad(u))) * dx)
    a += Variation(FORCE * u[1] * dx)
    p = -(u[1] + G0)
    a += Variation(0.5 * gamma * IfPos(p, p * p, 0) * ds("bot"))
    gfu = GridFunction(fes)
    gfu.vec[:] = 0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        ret = solvers.Newton(a, gfu, maxit=MAXIT, printing=False)
    uy = [float(gfu(mesh(float(t), 0.0))[1])
          for t in numpy.linspace(0.005, 0.995, 60)]
    pen = max(0.0, -(min(uy) + G0))
    return ret, pen, buf.getvalue()


def main() -> int:
    mesh = _mesh()

    # --- what does Newton actually return? -----------------------------
    doc = solvers.Newton.__doc__ or ""
    ret_ok, _, _ = solve(mesh, 1e5)
    print(f"newton_return={ret_ok}")
    print(f"newton_return_is_pair={isinstance(ret_ok, tuple) and len(ret_ok) == 2}")
    print(f"newton_first_entry_is_status={ret_ok[0] == 0}")
    print(f"docstring_says_first_is_status="
          f"{'first one is 0 if' in doc.lower().replace(chr(10), ' ')}")

    # --- the small-gamma end -------------------------------------------
    edge = MAXH
    rows = []
    for gamma in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e8):
        ret, pen, _ = solve(mesh, gamma)
        rows.append((gamma, ret, pen))
        print(f"gamma={gamma:g} status={ret[0]} numit={ret[1]} "
              f"penetration={pen:.6e} pct_of_element_edge={100 * pen / edge:.3f}")

    lo_pen = rows[0][2]
    hi_pen = rows[-1][2]
    print(f"low_gamma_penetration_over_5pct_of_edge={lo_pen > 0.05 * edge}")
    print(f"high_gamma_penetration_under_5pct_of_edge={hi_pen < 0.05 * edge}")
    print(f"penetration_monotone_decreasing="
          f"{all(b[2] <= a[2] + 1e-15 for a, b in zip(rows, rows[1:]))}")
    print(f"all_usable_gammas_converged={all(r[1][0] == 0 for r in rows)}")

    # --- the large-gamma end -------------------------------------------
    ret_hi, pen_hi, captured = solve(mesh, 1e22)
    print(f"extreme_gamma_status={ret_hi[0]}")
    print(f"extreme_gamma_numit={ret_hi[1]}")
    print(f"extreme_gamma_newton_failed={ret_hi[0] != 0}")
    print(f"extreme_gamma_used_whole_budget={ret_hi[1] >= MAXIT}")
    found = [s for s in CLAIMED_STRINGS if s.lower() in captured.lower()]
    print(f"claimed_strings_found_in_output={found}")
    print(f"no_divisionbyzero_no_cond_warning={found == []}")
    real = "Newton might not converge" in captured
    print(f"real_warning_wording_present={real}")

    ok = (
        isinstance(ret_ok, tuple) and len(ret_ok) == 2 and ret_ok[0] == 0
        and lo_pen > 0.05 * edge
        and hi_pen < 0.05 * edge
        and all(r[1][0] == 0 for r in rows)
        and ret_hi[0] != 0 and ret_hi[1] >= MAXIT
        and found == [] and real
    )
    if ok:
        return 0
    print("FAIL: penalty-gamma invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
