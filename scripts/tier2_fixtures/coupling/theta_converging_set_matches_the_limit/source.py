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

So this fixture runs every constant theta the limit says should converge, to
convergence, at both unbalanced ratios. It asserts the set that converges is
exactly the set the limit predicts — which is the durable statement — and prints
the count, which is what falsifies the "only one" wording. A fine grid at rho = 9
is included precisely because the coarse grid there is what made the wrong
sentence look true.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _couplinglib as L                                    # noqa: E402

# (rho, grid, what the grid is). The rho=4 grid is the knowledge's own coarse
# sweep restricted to the values at or below the limit plus one above it; the
# rho=9 grids are the coarse one that produced the wrong sentence and a finer
# one that shows why it was wrong.
CASES = [
    (4.0, [0.1, 0.2, 0.3, 0.5], "the coarse sweep the knowledge describes"),
    (9.0, [0.1, 0.2, 0.3], "the same coarse sweep, at the more unbalanced ratio"),
    (9.0, [0.05, 0.1, 0.15], "a grid three times finer, at the same ratio"),
]
MAX_ITER = 400
TOL = 1e-5


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
                L.close(0.5 * (lo + hi), p.t_iface, 1e-3,
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
