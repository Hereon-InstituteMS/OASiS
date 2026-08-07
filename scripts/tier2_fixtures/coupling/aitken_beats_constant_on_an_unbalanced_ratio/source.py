"""At a strongly unbalanced ratio the default accelerator lands on the answer
and the naive constant theta does not.

THE CLAIM UNDER TEST is the comparative half of the accelerator advice, the
reason "aitken" is both the default and the recommendation:

  "Measured across conductance ratios rho from 1/4 to 9 and theta from 0.1 to
   1.0 on this driver, Aitken matched or beat a constant theta almost
   everywhere, and in a quarter of those settings it converged to the right
   interface value where the SAME constant theta diverged by tens of orders of
   magnitude. It is the main thing protecting you from a theta chosen too
   large."

A whole sweep is not what a fixture is for, and reproducing one would take
hours of solver time to restate a summary. What a fixture can do is take the
setting the claim is about — one strongly unbalanced ratio, and the theta the
knowledge itself hands an agent who cannot estimate rho ("first try, cannot
estimate rho at all: theta=0.5, keep accelerator='aitken'") — and run the two
accelerators head to head against the closed form under an identical budget.

WHY rho = 6 AND NOT rho = 9. The ratio has to be one where the naive theta is
genuinely bad for the constant accelerator, or there is no contest: theta = 0.5
must sit above the stability limit 2/(1+rho), which needs rho > 3. It also has
to be one the default actually rescues, and that is a real limit measured on
this driver rather than an arbitrary choice — at rho = 9 with theta = 0.5 the
default accelerator does NOT rescue the split. It diverges too, by many orders
of magnitude, just far fewer than the constant arm's. That is the knowledge's
own caveat ("It is not magic: at a strongly unbalanced ratio no accelerator
rescues a bad split") and it is why the comparative claim is worded "almost
everywhere". rho = 6 is inside the swept range, well above the limit that makes
theta = 0.5 a bad constant choice, and inside the region the default still
recovers.

BOTH ARMS USE THE SAME max_iter AND THE SAME tol. Without that the comparison
is between an accelerator and a budget.

WHAT COUNTS AS WINNING is not `converged`. Each arm's interface temperature and
interface flux are checked from BOTH sides against the closed form, with the
two outward normals' opposite signs, plus the conservation balance and the
tool's own validation block — a partitioned scheme that converged to the wrong
fixed point would otherwise be counted a winner. The constant arm is then
required to have FAILED that same check, so the fixture states a contrast and
not a single success.
"""
from __future__ import annotations

import contextlib
import io
import math
import sys
from pathlib import Path

# The shared library lives in a sibling `_lib/` DIRECTORY, not as a bare
# file, because scripts/mutate_tier2_fixtures.py stages a fixture into a
# scratch tree and copies only sibling directories whose name starts with
# `_`. As a bare file it would not be copied, the staged fixture could not
# import it, and every mutation verdict would be VACUOUS_BASELINE.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                     # noqa: E402

RHO = 6.0               # strongly unbalanced: theta=0.5 is far above 2/(1+rho)
THETA = 0.5             # the knowledge's own "cannot estimate rho" default
MAX_ITER = 300          # the SAME budget for both arms
TOL = 1e-4              # and the SAME tolerance

RUNAWAY = 1.0           # deviation above this: the value is not even the right
                        # order of magnitude
SETTLED = 1e-3

# Physics against the closed form. A run converged to tol=1e-4 sits a few
# 1e-3 K and a few 1e-2 W/m^2 away; these thresholds sit an order of magnitude
# above that and orders of magnitude below a wrong fixed point (tens of K) or a
# flux sign error (O(1) relative).
T_ATOL = 0.05
Q_ATOL = 0.2
BALANCE_RTOL = 1e-2


def _quiet_stage(root, name, backend, edits):
    """`L.stage` prints the interpreter it resolved, which is right for a pair
    fixture and noise for a two-arm comparison that stages four participants."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return L.stage(root, name, backend, edits)


def probe(tag: str, accelerator: str, rho: float = RHO, theta: float = THETA,
          max_iter: int = MAX_ITER, tol: float = TOL, dirichlet: str = "left",
          backend: str = "skfem", mesh_l=(16, 16), mesh_r=(14, 12)) -> dict:
    """couplinglib.probe_theta with the ACCELERATOR made a parameter.

    Same problem, same starting guess away from every ratio's answer, same
    non-matching interface meshes, same registered `couple` tool through
    `pair`. The deviation is probe_theta's: how far the interface value ended
    from the exact answer, relative to it — the residual is normalised by the
    export magnitude and SATURATES near a constant of order one on a diverging
    run, so it cannot tell a runaway from a slow iteration.
    """
    p = L.problem_with_rho(rho)
    roles = {"left": "dirichlet" if dirichlet == "left" else "neumann",
             "right": "dirichlet" if dirichlet == "right" else "neumann"}
    root = L.workroot(tag)
    specs = []
    for pos, mesh in (("left", mesh_l), ("right", mesh_r)):
        partner = "right" if pos == "left" else "left"
        e = L.heat_edits(p, pos, roles[pos], partner, mesh)
        e["T_INIT"] = f"{L.SWEEP_T_INIT}"
        specs.append(_quiet_stage(root, pos, backend, e))
    res = L.pair(specs, max_iter=max_iter, tol=tol, accelerator=accelerator,
                 theta=theta)
    worst = 0.0
    for ex in (res.get("exports") or {}).values():
        for v in ex.get("values", []) or []:
            d = abs(float(v) - p.t_iface)
            worst = max(worst, d) if math.isfinite(d) else float("inf")
    deviation = worst / abs(p.t_iface) if math.isfinite(worst) else float("inf")
    return {"converged": bool(res.get("converged")),
            "iterations": int(res.get("iterations", 0)),
            "residual": float(res.get("residual", float("nan"))),
            "deviation": deviation, "result": res, "problem": p,
            "accelerator": accelerator, "budget": (rho, theta, max_iter, tol)}


def shown(dev: float) -> str:
    return "inf" if not math.isfinite(dev) else f"{dev:.3e}"


def reached_closed_form(tag: str, r: dict, loud: bool) -> bool:
    """Did this arm actually land on the analytic interface state?

    `loud` says whether a miss is a FAILURE. The arm that is EXPECTED to miss
    calls this quietly: its failure is the fixture's evidence, not the
    fixture's failure, and a FAIL: line there would contradict the verdict.
    """
    p, res = r["problem"], r["result"]
    ok = bool(r["converged"])
    if not ok and loud:
        L.check(False, f"{tag}_did_not_converge",
                f"residual {r['residual']:.3e} after {r['iterations']} of "
                f"{MAX_ITER}: {str(res.get('error'))[:200]}")
    ex = res.get("exports") or {}
    if not ex:
        if loud:
            L.check(False, f"{tag}_no_exports", "no interface exports at all")
        return False
    nl, nr = len(ex["left"]["coordinates"]), len(ex["right"]["coordinates"])
    print(f"{tag}_n_points={nl}/{nr}")
    if nl == nr:
        ok = False
        if loud:
            L.check(False, f"{tag}_matching_meshes",
                    f"both sides used {nl} interface points, so the "
                    f"non-matching-interface claim was not exercised")
    for side in ("left", "right"):
        lo, hi = L.span(ex[side]["values"])
        print(f"{tag}_{side}_T_span=[{lo:.10g},{hi:.10g}]")
        err = abs(0.5 * (lo + hi) - p.t_iface)
        print(f"{tag}_{side}_T_err={err:.3e}")
        if not (err <= T_ATOL):
            ok = False
            if loud:
                L.check(False, f"{tag}_{side}_T_off_the_closed_form",
                        f"|T - {p.t_iface:.6f}| = {err:.6e} > {T_ATOL:.1e}")
    for side, sign in (("left", +1.0), ("right", -1.0)):
        lo, hi = L.span(ex[side]["normal_fluxes"])
        print(f"{tag}_{side}_q_span=[{lo:.10g},{hi:.10g}]")
        err = abs(0.5 * (lo + hi) - sign * p.q)
        print(f"{tag}_{side}_q_err={err:.3e}")
        if not (err <= Q_ATOL):
            ok = False
            if loud:
                L.check(False, f"{tag}_{side}_q_off_the_closed_form",
                        f"|q - {sign * p.q:.6f}| = {err:.6e} > {Q_ATOL:.1e}")
    net_l, net_r = L.net_flux(ex["left"]), L.net_flux(ex["right"])
    rel = abs(net_l + net_r) / max(abs(net_l), abs(net_r), 1e-30)
    print(f"{tag}_flux_balance_rel={rel:.3e}")
    if not (math.isfinite(rel) and rel < BALANCE_RTOL):
        ok = False
        if loud:
            L.check(False, f"{tag}_flux_not_balanced",
                    f"net(left)={net_l:.6e} net(right)={net_r:.6e}")
    if res.get("validation"):
        ok = False
        if loud:
            L.check(False, f"{tag}_validation_not_empty",
                    "; ".join(res["validation"])[:300])
    return bool(ok)


def body() -> None:
    L.require_available("skfem")
    p = L.problem_with_rho(RHO)
    limit = 2.0 / (1.0 + RHO)
    print(f"rho={RHO:g} theta={THETA} max_iter={MAX_ITER} tol={TOL:g}")
    print(f"k_left={p.kl:.4f} T_exact={p.t_iface:.6f} q_exact={p.q:.6f}")
    print(f"theta_opt_for_this_rho={p.theta_opt('left'):.6f}")
    print(f"stability_limit_for_this_rho={limit:.6f}")
    print(f"constant_amplification_at_this_theta="
          f"{p.amplification('left', THETA):.4f}")

    # The premise: the naive theta must be a BAD constant choice here, or the
    # contest is against nothing. This is arithmetic on the knowledge's own
    # formula, evaluated before anything runs.
    naive_is_bad = THETA > limit
    print(f"naive_theta_above_the_constant_stability_limit={bool(naive_is_bad)}")
    L.check(naive_is_bad, "naive_theta_is_already_stable_here",
            f"theta={THETA} is below the limit {limit:.6f} at rho={RHO:g}, so "
            f"a constant theta converges and there is nothing for the "
            f"accelerator to beat")

    a = probe("ait", "aitken")
    c = probe("cst", "constant")
    for tag, r in (("aitken", a), ("constant", c)):
        print(f"{tag}_converged={r['converged']}")
        print(f"{tag}_iterations={r['iterations']}")
        print(f"{tag}_residual={r['residual']:.3e}")
        print(f"{tag}_deviation_from_exact={shown(r['deviation'])}")

    a_ok = bool(reached_closed_form("aitken", a, loud=True)
                and a["deviation"] < SETTLED)
    print(f"aitken_reached_closed_form={a_ok}")
    L.check(a_ok, "aitken_did_not_reach_the_closed_form",
            f"deviation {shown(a['deviation'])} at rho={RHO:g}, theta={THETA}")

    c_ok = bool(reached_closed_form("constant", c, loud=False)
                and c["deviation"] < SETTLED)
    print(f"constant_reached_closed_form={c_ok}")
    L.check(not c_ok, "the_same_constant_theta_also_worked",
            f"the constant arm reached the closed form in {c['iterations']} "
            f"iterations with deviation {shown(c['deviation'])}, so this ratio "
            f"shows no advantage for the default and the claim is not being "
            f"tested here")

    diverged = (not math.isfinite(c["deviation"])) or c["deviation"] > RUNAWAY
    print(f"constant_theta_diverged={bool(diverged)}")

    won = bool(a_ok and not c_ok)
    print(f"aitken_beat_constant={won}")
    L.check(won, "aitken_did_not_beat_constant",
            f"at rho={RHO:g}, theta={THETA}, max_iter={MAX_ITER}, tol={TOL:g}: "
            f"aitken deviation {shown(a['deviation'])}, constant deviation "
            f"{shown(c['deviation'])}")

    # The two arms differ in ONE argument. Read back what each run was
    # actually given rather than asserting it from the constants: a comparison
    # in which the accelerator arm also got a different budget would prove
    # nothing, however green it looked.
    same_budget = bool(a["budget"] == c["budget"]
                       and a["accelerator"] != c["accelerator"])
    print(f"arms_differed_only_in_the_accelerator={same_budget}")
    L.check(same_budget, "the_two_arms_did_not_share_a_budget",
            f"aitken ran (rho, theta, max_iter, tol)={a['budget']} and "
            f"constant ran {c['budget']}")
    print("ratios_tested=1")


L.main(body)
