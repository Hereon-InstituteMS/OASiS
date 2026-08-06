"""A unit mismatch converges, balances perfectly, and no check inside `couple`
says a word. Only a comparison against an independent answer finds it.

THE CLAIM UNDER TEST. The knowledge's failure table lists, under "converged to a
plausible but wrong answer": "check that both sides use the same units and the
same interface coordinate", and under an unbalanced interface: "Otherwise:
different units on the two sides". The contract states the mechanism plainly —
"the driver does NOT convert units, sign conventions or field names". What none
of that says, and what an agent needs to know before trusting a green run, is
that the resulting wrong answer is INVISIBLE to every guard the tool has.

So this fixture runs the SAME coupling twice, changing exactly one constant: the
right subdomain's outer boundary temperature, written once in kelvin and once as
the same physical temperature in degrees Celsius — the single commonest way two
decks disagree about units. It then asks what `couple` reports about each.

The answer is that `couple` cannot tell them apart. Both converge, in a similar
number of iterations, to a similar residual. Both have an EMPTY validation
block. Both balance the interface flux to roundoff, because conservation is a
property of the fixed point and holds whatever units the boundary data is in.
The difference shows up only against an independent answer, and it is not
subtle: the interface temperature is out by tens of percent and the interface
flux by an order of magnitude.

That is why the pair fixtures assert the closed form rather than `converged`,
and this is the fixture that justifies the expense.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _couplinglib as L                                    # noqa: E402

KELVIN = 300.0          # the right subdomain's outer boundary, in K
CELSIUS = 26.85         # the SAME physical temperature, in degrees C


def run(tag: str, t_outer_right: float) -> dict:
    """The placeholder coupling with the right side's outer BC set to
    `t_outer_right`, whatever unit the author of that deck had in mind."""
    p = L.DEFAULT
    root = L.workroot(tag)
    left = L.stage(root, "left", "skfem",
                   L.heat_edits(p, "left", "dirichlet", "right", (16, 16)))
    right_edits = L.heat_edits(p, "right", "neumann", "left", (14, 12))
    right_edits["T_OUTER"] = f"{t_outer_right}"
    right = L.stage(root, "right", "skfem", right_edits)
    res = L.pair([left, right], max_iter=200, tol=1e-8,
                 accelerator="constant", theta=p.theta_opt("left"))
    return res


def report(tag: str, res: dict) -> dict:
    """Everything `couple` itself is able to say about this run."""
    p = L.DEFAULT
    out = {}
    L.check(bool(res.get("converged")), f"{tag}_did_not_converge",
            str(res.get("error"))[:200])
    print(f"{tag}_converged={bool(res.get('converged'))}")
    print(f"{tag}_iterations={res['iterations']}")
    print(f"{tag}_residual={res['residual']:.3e}")
    print(f"{tag}_validation_entries={len(res['validation'])}")
    if res["validation"]:
        print(f"{tag}_validation_text={' | '.join(res['validation'])[:220]}")
    bal = L.check_balance(res)
    print(f"{tag}_balance_warnings={len(bal)}")
    ex = res["exports"]
    net_l, net_r = L.net_flux(ex["left"]), L.net_flux(ex["right"])
    scale = max(abs(net_l), abs(net_r), 1e-30)
    out["balance_rel"] = abs(net_l + net_r) / scale
    print(f"{tag}_flux_balance_rel={out['balance_rel']:.3e}")

    lo, hi = L.span(ex["left"]["values"])
    out["t"] = 0.5 * (lo + hi)
    lo, hi = L.span(ex["left"]["normal_fluxes"])
    out["q"] = 0.5 * (lo + hi)
    out["t_rel"] = abs(out["t"] - p.t_iface) / abs(p.t_iface)
    out["q_rel"] = abs(out["q"] - p.q) / abs(p.q)
    print(f"{tag}_T_iface={out['t']:.9g}")
    print(f"{tag}_q_iface={out['q']:.9g}")
    print(f"{tag}_T_disagreement_with_closed_form={out['t_rel'] * 100:.1f}pct")
    print(f"{tag}_q_disagreement_with_closed_form={out['q_rel'] * 100:.1f}pct")
    out["mentions_units"] = any(
        "unit" in w.lower() for w in res["validation"])
    return out


def body() -> None:
    L.require_available("skfem")
    p = L.DEFAULT
    print(f"closed_form_T_iface={p.t_iface:.9f}")
    print(f"closed_form_q={p.q:.9f}")
    print(f"right_outer_kelvin={KELVIN}")
    print(f"right_outer_celsius={CELSIUS}")

    ok = report("consistent", run("consistent", KELVIN))
    bad = report("mismatched", run("mismatched", CELSIUS))

    # ── the control agrees with the closed form ───────────────────────────
    L.check(ok["t_rel"] < 1e-5, "control_disagrees_with_closed_form",
            f"the consistent-units run must reproduce the closed form, got "
            f"{ok['t_rel'] * 100:.4f}% off")

    # ── and the mismatched run does NOT — by a lot ────────────────────────
    L.check(bad["t_rel"] > 0.10, "mismatch_did_not_move_the_answer",
            f"the mismatched run agreed with the closed form to "
            f"{bad['t_rel'] * 100:.3f}%, so this fixture is no longer "
            f"demonstrating a silent-wrong result at all")
    L.check(bad["q_rel"] > 1.0, "mismatch_flux_barely_moved",
            f"interface flux off by only {bad['q_rel'] * 100:.1f}%")

    # ── while `couple` reports the two runs identically ───────────────────
    L.check(not bad["mentions_units"], "couple_actually_mentioned_units",
            "the validation block named units — if a check for this was added, "
            "this fixture is out of date and the knowledge should say so")
    print(f"both_converged={ok is not None and bad is not None}")
    print(f"mismatched_validation_is_empty={not bool(bad.get('mentions_units'))}")
    L.check(bad["balance_rel"] < 1e-4, "mismatch_broke_the_balance",
            f"the point of this fixture is that conservation is a property of "
            f"the fixed point and holds in the wrong units too; got "
            f"{bad['balance_rel']:.3e}")

    # ── the detector that DOES catch it, run explicitly ───────────────────
    # An independent answer for the same quantity is the only thing that
    # separates the two runs. Here that is the closed form; in a code that can
    # solve the problem un-split it is a monolithic re-solve.
    from core.quality_checks import check_monolithic_consistency
    w_ok = check_monolithic_consistency(ok["t"], p.t_iface, qoi="T_iface")
    w_bad = check_monolithic_consistency(bad["t"], p.t_iface, qoi="T_iface")
    print(f"monolithic_check_on_control={len(w_ok)}")
    print(f"monolithic_check_on_mismatch={len(w_bad)}")
    L.check(not w_ok, "monolithic_check_fired_on_a_correct_run", str(w_ok)[:200])
    L.check(len(w_bad) == 1, "monolithic_check_missed_the_mismatch",
            f"the one detector that can see this must see it: {w_bad}")
    if w_bad:
        print(f"monolithic_says={w_bad[0][:180]}")
    print("runs_compared=2")


L.main(body)
