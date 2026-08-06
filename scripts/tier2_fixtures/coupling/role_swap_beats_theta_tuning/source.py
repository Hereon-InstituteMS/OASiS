"""Swapping which side is Dirichlet beats every theta you could have picked.

THE CLAIM UNDER TEST, from the relaxation guidance:

  "convergence is fastest when the DIRICHLET side is the SOFTER / LESS
   CONDUCTIVE subdomain. If a coupling converges too slowly to be practical,
   SWAP WHICH SIDE IS DIRICHLET before doing anything else — that replaces rho
   by 1/rho and is usually a bigger win than any theta."

and, in the observations underneath:

  "for one asymmetric split, the SAME problem with the SAME tolerance failed to
   converge inside the iteration budget with the stiff subdomain on the
   Dirichlet side, and converged comfortably inside it once the two roles were
   swapped and theta set to 1/(1+rho) for the new rho."

This is the piece of advice most likely to be ignored, because it asks for an
edit to both participant scripts rather than a change to one argument. It is
also the only tuning advice in the knowledge that claims to beat a SEARCH: not
"this theta is better than that one" but "no theta in the stiff orientation
matches the swapped one".

So the fixture sweeps theta in the stiff orientation rather than testing a
single value. A claim that role-swapping beats theta tuning is only tested if
theta tuning was actually attempted and lost.

THE BUDGET IS NOT TUNED. max_iter=60 with tol=1e-8 is the iteration budget in
the knowledge's own example `couple(...)` call, chosen before any of these runs
and used unchanged, so the fixture cannot be accused of picking the one budget
that makes the claim true.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The shared library lives in a sibling `_lib/` DIRECTORY, not as a bare
# file, because scripts/mutate_tier2_fixtures.py stages a fixture into a
# scratch tree and copies only sibling directories whose name starts with
# `_`. As a bare file it would not be copied, the staged fixture could not
# import it, and every mutation verdict would be VACUOUS_BASELINE.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                     # noqa: E402

RHO = 9.0                # stiff Dirichlet side: the orientation to avoid
BUDGET = 60              # max_iter from the served example call
TOL = 1e-8               # tol from the same call
SWEEP = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


def body() -> None:
    L.require_available("skfem")
    p = L.problem_with_rho(RHO)
    print(f"conductance_left={p.cl:.6f}")
    print(f"conductance_right={p.cr:.6f}")
    print(f"rho_dirichlet_on_stiff_side={p.rho('left'):.6f}")
    print(f"rho_dirichlet_on_soft_side={p.rho('right'):.6f}")
    print(f"theta_opt_stiff={p.theta_opt('left'):.6f}")
    print(f"theta_opt_soft={p.theta_opt('right'):.6f}")
    print(f"budget_max_iter={BUDGET}")

    # ── orientation A: Dirichlet on the STIFF subdomain, theta SWEPT ───────
    converged_any = []
    for th in SWEEP:
        r = L.probe_theta(f"stiff_{th}", RHO, th, BUDGET, tol=TOL,
                          dirichlet="left", quiet=True)
        print(f"stiff_theta={th} converged={r['converged']} "
              f"iters={r['iterations']} residual={r['residual']:.3e} "
              f"amplification={r['amplification']:.4f}")
        if r["converged"]:
            converged_any.append((th, r["iterations"]))

    # ── orientation B: the SAME problem, roles swapped, one theta ──────────
    # No search here on purpose: the swapped orientation gets exactly the theta
    # the rule prescribes, and still wins.
    th_soft = p.theta_opt("right")
    swapped = L.probe_theta("soft", RHO, th_soft, BUDGET, tol=TOL,
                            dirichlet="right", quiet=True)
    print(f"swapped_theta={th_soft:.6f} converged={swapped['converged']} "
          f"iters={swapped['iterations']} residual={swapped['residual']:.3e} "
          f"amplification={swapped['amplification']:.4f}")

    L.check(swapped["converged"], "swap_did_not_converge",
            f"the rule says the swapped orientation converges comfortably "
            f"inside the budget; residual {swapped['residual']:.3e} after "
            f"{swapped['iterations']} of {BUDGET}")
    if swapped["converged"]:
        headroom = BUDGET / max(swapped["iterations"], 1)
        print(f"swapped_headroom_factor={headroom:.1f}")
        L.check(headroom >= 2.0, "swap_only_just_made_it",
                f"'comfortably inside the budget' should not mean using "
                f"{swapped['iterations']} of {BUDGET} iterations")

    print(f"stiff_orientation_thetas_that_converged={len(converged_any)}")
    if converged_any:
        best = min(converged_any, key=lambda t: t[1])
        print(f"stiff_best_theta={best[0]} stiff_best_iters={best[1]}")
        # The claim survives a converging stiff orientation only if swapping
        # still wins outright; say which reading of the claim held.
        L.check(best[1] > swapped["iterations"],
                "theta_tuning_matched_or_beat_the_swap",
                f"a swept theta in the stiff orientation converged in "
                f"{best[1]} iterations against {swapped['iterations']} for the "
                f"swap — the claim that swapping is 'usually a bigger win than "
                f"any theta' does not hold on this problem")
    else:
        print("stiff_orientation_best_theta=none_converged_within_budget")

    # …and the physics of the swapped run is right, not merely convergent.
    if swapped["converged"]:
        ex = swapped["result"]["exports"]
        lo, hi = L.span(ex["left"]["values"])
        L.close(0.5 * (lo + hi), p.t_iface, 1e-3, "swapped_T_err")
        lo, hi = L.span(ex["left"]["normal_fluxes"])
        L.close(0.5 * (lo + hi), p.q, 5e-3, "swapped_q_err")
    print(f"orientations_compared=2")
    print(f"thetas_swept_in_the_bad_orientation={len(SWEEP)}")


L.main(body)
