"""The default accelerator survives the case the constant one blows up on.

THE CLAIM UNDER TEST is the sentence the knowledge attaches to its own
worked divergence example, and it is the one that decides whether an agent
reading that example panics about the DEFAULT:

  "at rho = 4, theta = 0.5 with a CONSTANT accelerator DIVERGES — the interface
   values run away by many orders of magnitude and the conservation check fires
   — while theta = 0.2 converges. Nothing warns you in advance: a diverging
   coupling looks like a converging one for the first few iterations. Note the
   default 'aitken' survives this particular case; do not read the constant-
   theta stability limit as a property of the tool's default"

"aitken" is the `couple` tool's DEFAULT accelerator, and until this fixture
nothing behind that word had ever been run. The sibling fixtures
theta_stability_limit, theta_one_over_one_plus_rho_is_fastest and
theta_converging_set_matches_the_limit all sweep the CONSTANT accelerator, so
the entire measured basis of the theta advice was about the option the
knowledge tells agents NOT to reach for first.

THE EXPERIMENT is one problem, run twice, with ONE string different. Same rho,
same theta0, same max_iter, same tol, same meshes, same initial guess — only
`accelerator` changes. Anything less than that and the two arms are not
comparable and "survives" means nothing.

WHAT IS MEASURED, and why not the obvious thing. This follows
couplinglib.probe_theta's reasoning exactly, because the trap is the same one.
The residual is normalised by the raw export magnitude, so a diverging run
SATURATES it near a constant of order one instead of sending it to infinity —
the constant arm here ends with a residual of order 1, which is a number a
merely-slow run could also produce. The raw interface value is no better: it
sits near 316 K, so an error amplified fiftyfold still leaves it within a factor
of two of the right answer. What separates a runaway from a settling iteration
is the DEVIATION from the exact interface temperature relative to it, which
grows as the amplification factor to the power of the iteration count with
nothing to hide behind.

And "survives" is not "did not crash". The Aitken arm has its PHYSICS checked
against the closed form: the interface temperature from BOTH sides, the
interface flux from both sides against ±q with the two outward normals'
opposite signs, the net flux balance, and an empty `validation` block. A
partitioned scheme that converged to the wrong fixed point would otherwise pass
for a survivor.

WHY THE COPY OF probe_theta. couplinglib's probe_theta hard-codes
accelerator="constant" and this fixture's whole point is the other value, so the
probe is reproduced here with the accelerator as a parameter. It is otherwise
the same construction — same problem_with_rho, same heat_edits, same
SWEEP_T_INIT, same meshes, same `pair` through the registered `couple` tool —
so the two arms of this fixture differ in exactly one argument.
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

RHO = 4.0               # the ratio the knowledge's own divergence example uses
THETA0 = 0.5            # and its theta
MAX_ITER = 200
TOL = 1e-4
# The two accelerator strings, named so the difference between the arms is one
# identifier and the mutation can move it.
AITKEN = "aitken"
CONSTANT = "constant"

# Deviation from the exact interface temperature, relative to it.
RUNAWAY = 1.0           # above this the interface value is not even the right
                        # order — the run has run away
SETTLED = 1e-3          # below this it has landed on the answer

# Physics, against the closed form. A run converged to tol=1e-4 sits a few
# 1e-3 K and a few 1e-2 W/m^2 from the exact interface values — these
# thresholds sit an order of magnitude above that and orders of magnitude below
# any pathology worth catching: a wrong fixed point is tens of K away and a sign
# error is O(1) in the flux.
T_ATOL = 0.05
Q_ATOL = 0.2
BALANCE_RTOL = 1e-2


def _quiet_stage(root, name, backend, edits):
    """`L.stage` prints the interpreter it resolved, which is right for a pair
    fixture and noise for a two-arm comparison that stages four participants."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return L.stage(root, name, backend, edits)


def probe(tag: str, accelerator: str, rho: float = RHO, theta: float = THETA0,
          max_iter: int = MAX_ITER, tol: float = TOL, dirichlet: str = "left",
          backend: str = "skfem", mesh_l=(16, 16), mesh_r=(14, 12)) -> dict:
    """couplinglib.probe_theta with the ACCELERATOR made a parameter.

    Everything else is identical to it, deliberately: same problem, same
    starting guess away from every ratio's answer, same non-matching interface
    meshes, and the same registered `couple` tool through `pair`.
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
            "accelerator": accelerator}


def shown(dev: float) -> str:
    return "inf" if not math.isfinite(dev) else f"{dev:.3e}"


def report(tag: str, r: dict) -> None:
    print(f"{tag}_accelerator={r['accelerator']}")
    print(f"{tag}_converged={r['converged']}")
    print(f"{tag}_iterations={r['iterations']}")
    print(f"{tag}_residual={r['residual']:.3e}")
    print(f"{tag}_deviation_from_exact={shown(r['deviation'])}")


def reached_closed_form(tag: str, r: dict) -> bool:
    """Not `converged`: the interface VALUES, against the analytic answer.

    A partitioned fixed-point scheme converges to a fixed point, which is the
    solution only if the two participants exchange the right quantity with the
    right sign in the right units. So this checks the temperature from both
    sides, the flux from both sides with their opposite outward normals, the
    conservation balance, and the tool's own validation block.
    """
    p, res = r["problem"], r["result"]
    ok = L.check(r["converged"], f"{tag}_did_not_converge",
                 f"residual {r['residual']:.3e} after {r['iterations']} of "
                 f"{MAX_ITER}: {str(res.get('error'))[:200]}")
    ex = res.get("exports") or {}
    if not ex:
        return bool(L.check(False, f"{tag}_no_exports",
                            "the run returned no interface exports at all"))
    nl, nr = len(ex["left"]["coordinates"]), len(ex["right"]["coordinates"])
    print(f"{tag}_n_points={nl}/{nr}")
    ok = L.check(nl != nr, f"{tag}_matching_meshes",
                 f"both sides used {nl} interface points, so the "
                 f"non-matching-interface claim was not exercised") and ok
    for side in ("left", "right"):
        lo, hi = L.span(ex[side]["values"])
        print(f"{tag}_{side}_T_span=[{lo:.10g},{hi:.10g}]")
        ok = L.close(0.5 * (lo + hi), p.t_iface, T_ATOL,
                     f"{tag}_{side}_T_err") and ok
    for side, sign in (("left", +1.0), ("right", -1.0)):
        lo, hi = L.span(ex[side]["normal_fluxes"])
        print(f"{tag}_{side}_q_span=[{lo:.10g},{hi:.10g}]")
        ok = L.close(0.5 * (lo + hi), sign * p.q, Q_ATOL,
                     f"{tag}_{side}_q_err") and ok
    net_l, net_r = L.net_flux(ex["left"]), L.net_flux(ex["right"])
    rel = abs(net_l + net_r) / max(abs(net_l), abs(net_r), 1e-30)
    print(f"{tag}_flux_balance_rel={rel:.3e}")
    ok = L.check(rel < BALANCE_RTOL, f"{tag}_flux_not_balanced",
                 f"net(left)={net_l:.6e} net(right)={net_r:.6e}") and ok
    ok = L.check(not res.get("validation"), f"{tag}_validation_not_empty",
                 "; ".join(res.get("validation") or [])[:300]) and ok
    return bool(ok)


def body() -> None:
    L.require_available("skfem")
    p = L.problem_with_rho(RHO)
    print(f"rho={RHO:g} theta0={THETA0} max_iter={MAX_ITER} tol={TOL:g}")
    print(f"k_left={p.kl:.4f} T_exact={p.t_iface:.6f} q_exact={p.q:.6f}")
    print(f"constant_amplification={p.amplification('left', THETA0):.4f}")
    print(f"stability_limit_for_this_rho={2.0 / (1.0 + RHO):.6f}")

    # ── arm 1: the constant accelerator, which the knowledge says diverges ──
    c = probe("cst", CONSTANT)
    report("constant", c)

    # ── arm 2: the DEFAULT accelerator, same everything else ───────────────
    a = probe("ait", AITKEN)
    report("aitken", a)

    diverged = (not math.isfinite(c["deviation"])) or c["deviation"] > RUNAWAY
    print(f"constant_theta_diverged={bool(diverged)}")
    L.check(diverged, "constant_theta_did_not_diverge",
            f"deviation from the exact interface temperature ended at "
            f"{shown(c['deviation'])}, which is not a runaway; the knowledge's "
            f"whole divergence example rests on it being one")
    L.check(not c["converged"], "constant_theta_converged",
            "the constant arm reached the tolerance, so there is nothing for "
            "the default to survive")

    settled = L.check(a["deviation"] < SETTLED, "aitken_deviation_too_large",
                      f"deviation ended at {shown(a['deviation'])}, above "
                      f"{SETTLED:.0e}")
    physics = reached_closed_form("aitken", a)
    survived = bool(settled and physics)
    print(f"aitken_reached_closed_form={survived}")
    L.check(survived, "aitken_did_not_survive",
            "the knowledge says the default accelerator survives this exact "
            "case; on this driver it did not")

    # The gap between the two arms, so the run reports its own numbers rather
    # than the fixture pinning one.
    better = (not math.isfinite(c["deviation"])) or (
        a["deviation"] < c["deviation"])
    if math.isfinite(c["deviation"]) and a["deviation"] > 0.0:
        print(f"orders_of_magnitude_between_the_arms="
              f"{math.log10(c['deviation'] / a['deviation']):.1f}")
    print(f"aitken_deviation_below_constant={bool(better)}")
    L.check(better, "aitken_was_not_better_than_constant",
            f"aitken {shown(a['deviation'])} vs constant "
            f"{shown(c['deviation'])}")

    # A two-arm comparison in which both arms ran the same accelerator would
    # be vacuous however green it looked, so say out loud that they did not.
    distinct = bool(AITKEN != CONSTANT and a["accelerator"] != c["accelerator"])
    print(f"the_two_arms_ran_different_accelerators={distinct}")
    L.check(distinct, "both_arms_ran_the_same_accelerator",
            f"arm 1 ran {c['accelerator']!r} and arm 2 ran "
            f"{a['accelerator']!r}; there is no comparison here")
    print("arms=2")


L.main(body)
