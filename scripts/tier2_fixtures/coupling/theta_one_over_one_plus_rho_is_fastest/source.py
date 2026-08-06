"""theta = 1/(1+rho) is the fastest constant theta, at every ratio tested.

THE CLAIM UNDER TEST, and it is the single number the knowledge tells an agent
to compute before running anything:

  "the best theta is **theta ~ 1 / (1 + rho)** — 0.5 when the two sides are
   balanced, smaller when the Dirichlet side is the stiffer one, larger when it
   is the softer one. This is the one number in this section worth computing
   before you run anything: swept over rho from 1/4 to 9, the fastest constant
   theta was 1/(1+rho) at EVERY ratio"

An "optimal" claim cannot be checked by running the optimum. It needs
neighbours, and neighbours that are themselves stable — otherwise the comparison
is against thetas that diverge, which proves only the stability limit that the
sibling fixture theta_stability_limit already covers.

So the bracket here is multiplicative and rho-independent: 0.6x, 1.0x and 1.4x
of 1/(1+rho). Every one of those is below the stability limit 2/(1+rho) at every
rho by construction, so all three converge and the comparison is a genuine
contest between converging choices rather than a contest against divergence.

Each run also has its physics checked against the closed form. A theta that
reached the tolerance fastest by converging to the wrong fixed point would
otherwise win this comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _couplinglib as L                                    # noqa: E402

RHOS = [1.0, 4.0, 9.0]
BRACKET = [0.6, 1.0, 1.4]      # multiples of theta_opt; all < 2x = the limit
MAX_ITER = 400
TOL = 1e-5


def body() -> None:
    L.require_available("skfem")
    wins = 0
    for rho in RHOS:
        p = L.problem_with_rho(rho)
        theta_opt = p.theta_opt("left")
        limit = 2.0 / (1.0 + rho)
        print(f"=== rho={rho:g} theta_opt={theta_opt:.6f} "
              f"stability_limit={limit:.6f}")
        results = []
        for mult in BRACKET:
            th = mult * theta_opt
            L.check(th < limit, f"rho{rho:g}_bracket_{mult}_is_unstable",
                    f"theta={th:.4f} is not below the stability limit "
                    f"{limit:.4f}; the bracket must compare converging choices")
            r = L.probe_theta(f"o{rho:g}_{mult}", rho, th, MAX_ITER, tol=TOL,
                              quiet=True)
            tag = f"rho{rho:g}_mult{mult}"
            print(f"{tag}_theta={th:.6f}")
            print(f"{tag}_amplification={r['amplification']:.4f}")
            print(f"{tag}_converged={r['converged']}")
            print(f"{tag}_iterations={r['iterations']}")
            if not L.check(r["converged"], f"{tag}_did_not_converge",
                           f"residual {r['residual']:.3e} after "
                           f"{r['iterations']} of {MAX_ITER}"):
                continue
            # A run that reached tol by settling on the WRONG fixed point would
            # otherwise win the iteration-count contest.
            ex = r["result"]["exports"]
            lo, hi = L.span(ex["left"]["values"])
            L.close(0.5 * (lo + hi), p.t_iface, 1e-3, f"{tag}_T_err")
            results.append((mult, r["iterations"]))

        if len(results) < len(BRACKET):
            L.check(False, f"rho{rho:g}_incomplete_bracket",
                    "not every theta in the bracket converged, so the "
                    "comparison cannot be made")
            continue
        best_mult, best_iters = min(results, key=lambda t: t[1])
        print(f"rho{rho:g}_fastest_multiple_of_theta_opt={best_mult}")
        print(f"rho{rho:g}_fastest_iterations={best_iters}")
        if L.check(best_mult == 1.0, f"rho{rho:g}_theta_opt_was_not_fastest",
                   f"1/(1+rho) took {dict(results).get(1.0)} iterations; "
                   f"{best_mult}x took {best_iters}. The knowledge says the "
                   f"fastest constant theta was 1/(1+rho) at EVERY ratio."):
            wins += 1

    print(f"ratios_tested={len(RHOS)}")
    print(f"ratios_where_theta_opt_won={wins}")
    L.check(wins == len(RHOS), "theta_opt_did_not_win_everywhere",
            f"{wins} of {len(RHOS)}")


L.main(body)
