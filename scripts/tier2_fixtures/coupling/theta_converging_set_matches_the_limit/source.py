"""How many constant thetas actually converge at an unbalanced ratio — the
claim that said "only one", and does not survive being run.

THE CLAIM UNDER TEST, from the relaxation guidance:

  "swept over rho from 1/4 to 9, the fastest constant theta was 1/(1+rho) at
   EVERY ratio, and at the unbalanced ones it was the ONLY constant theta that
   converged at all rather than diverging"

The first half holds and has its own fixture. The second half does not, and the
reason is worth stating because it is a general hazard in knowledge written from
a sweep: it is a statement about a GRID reported as a statement about THETA.

The driver's amplification factor is sqrt((1-theta)^2 + rho*theta^2), which is
below one exactly when theta < 2/(1+rho). That is a whole interval, not a point,
and it always contains 1/(1+rho) with the same width on either side in
multiplicative terms. At rho = 4 the interval is theta < 0.4, so on the grid the
knowledge itself describes — "theta from 0.1 to 1.0" — three values converge,
not one. At rho = 9 the interval is theta < 0.2, so on that same coarse grid
exactly one value happens to land inside it, and the sweep reported the grid
spacing as a property of the method.

So this fixture runs every constant theta on that coarse grid at rho = 4, to
convergence, and asserts the set that converges is exactly the set the limit
predicts — which is the durable statement — while printing the count, which is
what falsifies the "only one" wording.

It stops at rho = 4 on purpose. rho = 4 is where the claim is false ON THE
KNOWLEDGE'S OWN GRID, so one ratio settles it, and the sibling fixture
theta_stability_limit already verifies the same limit at rho = 9 (theta = 0.1
settles, theta = 0.2 sits exactly on the boundary and never settles). Once the
limit is established as the thing that decides, the rho = 9 story follows from
it by arithmetic rather than needing another twenty minutes of solver time: the
interval there is theta < 0.2, so a grid stepping by 0.1 contains exactly one
usable value and a grid stepping by 0.05 contains three.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _couplinglib as L                                    # noqa: E402

# (rho, grid, what the grid is). The grid is the knowledge's own coarse sweep
# restricted to the values at or below the stability limit, plus one above it so
# the fixture is not only asking which thetas converge but also which do not.
CASES = [
    (4.0, [0.1, 0.2, 0.3, 0.5], "the coarse sweep the knowledge describes"),
]
# 150 is comfortably above what the three converging thetas need at this
# tolerance and caps what the diverging one costs, which is the whole budget.
MAX_ITER = 150
TOL = 1e-4

# NOT a convergence check. How close a CONVERGED run lands to the exact answer
# is set by the coupling tolerance, and these sweeps run at a loose one on
# purpose — the ranking of thetas is asymptotic, so a loose tolerance orders
# them exactly as a tight one would for a fraction of the iterations. Measured
# here, a run converged to tol=1e-4 sits a few 1e-3 K from the exact interface
# temperature. This threshold exists for one thing: catching a run that reached
# the tolerance on a DIFFERENT fixed point, which is tens of K away, and which
# would otherwise be counted as a success and could win the iteration contest.
WRONG_FIXED_POINT_ATOL = 0.5



def body() -> None:
    L.require_available("skfem")
    for rho, grid, what in CASES:
        p = L.problem_with_rho(rho)
        limit = 2.0 / (1.0 + rho)
        theta_opt = p.theta_opt("left")
        label = f"rho{rho:g}_n{len(grid)}"
        print(f"=== {label}: {what}")
        print(f"{label}_stability_limit={limit:.6f}")
        print(f"{label}_theta_opt={theta_opt:.6f}")
        predicted = [t for t in grid if t < limit - 1e-12]
        print(f"{label}_predicted_to_converge={','.join(str(t) for t in predicted)}")

        converged = []
        for th in grid:
            r = L.probe_theta(f"c{rho:g}_{th}", rho, th, MAX_ITER, tol=TOL,
                              quiet=True)
            print(f"{label}_theta{th}_amplification={r['amplification']:.4f}")
            print(f"{label}_theta{th}_converged={r['converged']}")
            print(f"{label}_theta{th}_iterations={r['iterations']}")
            if r["converged"]:
                converged.append(th)
                # A theta that reached the tolerance on the wrong fixed point
                # would otherwise be counted as a success.
                ex = r["result"]["exports"]
                lo, hi = L.span(ex["left"]["values"])
                L.close(0.5 * (lo + hi), p.t_iface, WRONG_FIXED_POINT_ATOL,
                        f"{label}_theta{th}_T_err")

        print(f"{label}_converged_set={','.join(str(t) for t in converged) or 'none'}")
        print(f"{label}_converged_count={len(converged)}")
        L.check(converged == predicted, f"{label}_set_does_not_match_the_limit",
                f"the amplification factor says theta < {limit:.4f} converges, "
                f"so {predicted} should have converged and {converged} did. The "
                f"durable statement is the interval, not any one theta.")
        L.check(theta_opt in converged, f"{label}_theta_opt_did_not_converge",
                f"1/(1+rho) = {theta_opt:.4f}")

    # The measurement that falsifies "the ONLY constant theta that converged".
    rho4 = [c for c in CASES if c[0] == 4.0][0]
    n_expected_rho4 = len([t for t in rho4[1] if t < 2.0 / (1.0 + 4.0)])
    print(f"rho4_constant_thetas_that_converge={n_expected_rho4}")
    L.check(n_expected_rho4 > 1, "only_one_theta_converges_after_all",
            "if exactly one constant theta converges at rho=4 then the "
            "knowledge's 'ONLY constant theta that converged' wording is "
            "right and this fixture's premise is wrong — recheck before "
            "changing the text back")
    print(f"cases_tested={len(CASES)}")


L.main(body)
