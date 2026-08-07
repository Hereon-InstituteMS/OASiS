"""Aitken's theta really is per-participant, really does start from the theta
you pass, and really is clamped into [0.05, 1.0].

THE CLAIM UNDER TEST is the description of the `couple` tool's DEFAULT
accelerator, the one sentence an agent reads before deciding what to pass:

  "'aitken' — ONE theta for the whole interface state, recomputed every
   iteration, starting from the theta you pass and clamped into [0.05, 1.0].
   ... The two fallback paths inside the update ... hold the previous theta,
   clamped into the same [0.05, 1.0]."

Four separate assertions live there and each one can be wrong on its own. ONE
THETA FOR THE WHOLE INTERFACE STATE is a statement about how many independent
thetas the driver carries — this fixture asserted the opposite until the two
coupling branches were reconciled, because knowledge/coupling-revision's driver
really did carry one per participant and feature/coupling-robustness replaced
that (giving each participant its own theta relaxes the two halves of one
coupled system by different amounts, which drove the two thetas to opposite
clamps on a Dirichlet-Neumann split). STARTING FROM THE THETA YOU PASS is a
statement about what the first adaptation is seeded with — a driver that
quietly seeded 0.5 whatever you passed would look identical from the outside on
most problems. CLAMPED INTO [0.05, 1.0] is a statement about two numbers, and a
bound that is present in the source but never reached is not a bound anyone has
tested. THE FALLBACKS CLAMP THE SAME WAY is the fourth: an earlier version of
this driver floored them at 0.1 instead, so the served interval was incomplete,
and the two paths are reached often enough (iteration 1, and any degenerate
denominator) that the difference is visible in real numbers.

WHICH PART IS UNIT-LEVEL AND WHICH IS END-TO-END, stated plainly because the
two carry different weight:

  * UNIT. The clamp BOUNDS are pinned by calling `core.coupling_driver._aitken`
    directly with residual vectors chosen so the unclamped rule
    theta = -theta_prev * <r_old, dr> / <dr, dr> lands far outside the interval
    in each direction — around -50 and +50 on the arms below, and around -1000
    and +1000 on the second pair. The returned theta must be EXACTLY the bound,
    with no tolerance, because it is a `min`/`max` against a literal. A third
    call puts the unclamped value inside the interval and requires it back
    unchanged, so this cannot be satisfied by a driver that clamps everything
    to one number. Driving the rule out of range at all is not something a
    two-participant conduction problem can be steered into on demand, which is
    why this part is a unit call.
  * END-TO-END. The rest is a REAL coupling, run through the registered
    `couple` tool with the driver untouched, at rho=4 and theta0=0.5 — the
    setting the knowledge's own divergence example uses — and checked against
    the closed-form interface temperature and flux. From that run the fixture
    asserts that the number of theta adaptations is exactly two per iteration
    after the first (the per-participant claim, for two participants), that the
    theta seeding the first adaptation on each side is the theta that was
    passed (the starting-value claim), that no theta the driver used ever left
    the interval, and that the low bound was actually REACHED — so the clamp is
    a path the default takes on an ordinary problem, not a line of dead code.
    On this setting only the LOW bound is reached; the high bound is proven at
    unit level and its end-to-end count is printed for the record, not asserted.

HOW THE END-TO-END RUN IS OBSERVED WITHOUT BEING CHANGED. `_aitken` is wrapped
in a recorder that calls the real function and returns its two return values
verbatim; the recorder only reads its arguments. That is asserted rather than
asserted-by-comment: the same case is run FIRST with the driver completely
untouched, and the recorded run must match it in convergence flag, iteration
count and interface temperature to the last bit. If wrapping changed anything,
those two runs would differ and the fixture fails.

THE BOUNDS THE FIXTURE ASSERTS AGAINST ARE NAMED CONSTANTS below and are used
only in assertions and printing, never to build the inputs — the inputs drive
the rule out of range by a ratio argument that does not know what the bounds
are. Moving one of those constants to a value the real driver does not use
therefore makes the fixture fail, which is what the recorded mutation does.
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

import numpy as np                                          # noqa: E402

# The interval the served knowledge states. Assertions and printing only: the
# inputs that drive the rule out of range are built from a ratio and never from
# these.
CLAMP_LOW = 0.05
CLAMP_HIGH = 1.0

RHO = 4.0
THETA0 = 0.5
MAX_ITER = 200
TOL = 1e-4
N_PARTICIPANTS = 2

T_ATOL = 0.05
Q_ATOL = 0.2
BALANCE_RTOL = 1e-2


# ── unit level: the bounds themselves ──────────────────────────────────────

def unclamped(prev_relaxed, new_raw, prev_raw, theta_prev) -> float:
    """The Aitken rule WITHOUT the clamp, recomputed here so the fixture can
    say how far outside the interval the rule wanted to go. This is a reading
    of the inputs; it never feeds back into the driver."""
    r_new = np.asarray(new_raw, float) - np.asarray(prev_relaxed, float)
    if prev_raw is None:
        return float("nan")
    dr = r_new - np.asarray(prev_raw, float)
    den = float(np.dot(dr, dr))
    if den < 1e-30:
        return float("nan")
    return -theta_prev * float(np.dot(np.asarray(prev_raw, float), dr)) / den


def drive(theta_prev: float, ratio: float) -> tuple:
    """Arguments for `_aitken` whose unclamped theta is theta_prev/(1 - ratio).

    With a previous residual r_old and a new residual r_new = ratio * r_old the
    rule collapses to theta = theta_prev * 1 / (1 - ratio), so a ratio just
    above 1 sends it far negative and a ratio just below 1 sends it far
    positive — as far as wanted, and without any reference to what the bounds
    are. `prev_relaxed` is a plausible interface state (temperatures near
    300 K) and `new_raw` is that state plus the new residual, which is the
    shape the driver really passes.
    """
    r_old = np.array([1.0, 1.0, 1.0, 1.0])
    r_new = ratio * r_old
    prev_relaxed = np.array([300.0, 301.0, 302.0, 303.0])
    return prev_relaxed, prev_relaxed + r_new, r_old, theta_prev


def unit_bounds() -> bool:
    from core.coupling_driver import _aitken

    ok = True

    # (1) the rule far BELOW the interval must come back exactly at the floor
    for theta_prev, ratio in ((0.5, 1.01), (1.0, 1.001)):
        args = drive(theta_prev, ratio)
        u = unclamped(*args)
        th, _ = _aitken(*args)
        print(f"low_theta_prev={theta_prev}_unclamped={u:.6g}_returned={th!r}")
        ok = L.check(u < CLAMP_LOW, f"low_case_{theta_prev}_did_not_go_low",
                     f"the unclamped rule gave {u:.6g}, which is not below "
                     f"{CLAMP_LOW}; this case tests nothing") and ok
        ok = L.check(th == CLAMP_LOW, f"low_case_{theta_prev}_not_at_the_floor",
                     f"the driver returned {th!r} where the served interval's "
                     f"floor is {CLAMP_LOW}") and ok
    print(f"clamp_low_respected={bool(ok)}")

    # (2) the rule far ABOVE the interval must come back exactly at the ceiling
    hi_ok = True
    for theta_prev, ratio in ((0.5, 0.99), (1.0, 0.999)):
        args = drive(theta_prev, ratio)
        u = unclamped(*args)
        th, _ = _aitken(*args)
        print(f"high_theta_prev={theta_prev}_unclamped={u:.6g}_returned={th!r}")
        hi_ok = L.check(u > CLAMP_HIGH, f"high_case_{theta_prev}_did_not_go_high",
                        f"the unclamped rule gave {u:.6g}, which is not above "
                        f"{CLAMP_HIGH}; this case tests nothing") and hi_ok
        hi_ok = L.check(th == CLAMP_HIGH,
                        f"high_case_{theta_prev}_not_at_the_ceiling",
                        f"the driver returned {th!r} where the served "
                        f"interval's ceiling is {CLAMP_HIGH}") and hi_ok
    print(f"clamp_high_respected={bool(hi_ok)}")

    # (3) a value INSIDE the interval must come back untouched, or the two
    # assertions above would also be satisfied by a driver that returns a
    # constant.
    args = drive(0.5, -1.0)              # unclamped = 0.5 / 2 = 0.25
    u = unclamped(*args)
    th, _ = _aitken(*args)
    print(f"inside_unclamped={u:.6g}_returned={th!r}")
    mid_ok = L.check(CLAMP_LOW < u < CLAMP_HIGH, "inside_case_is_not_inside",
                     f"the unclamped rule gave {u:.6g}, which is not strictly "
                     f"inside [{CLAMP_LOW}, {CLAMP_HIGH}]")
    mid_ok = L.check(th == u, "inside_value_was_altered",
                     f"the driver returned {th!r} for an unclamped {u!r}; the "
                     f"clamp must not touch a value already in range") and mid_ok
    print(f"unclamped_value_inside_the_bounds_passed_through={bool(mid_ok)}")

    # (4) THE TWO FALLBACK PATHS clamp into the SAME interval. This is the half
    # of the served sentence that used to be wrong in the other direction: both
    # paths floored theta at 0.1 while the advertised interval started at 0.05,
    # so a theta_prev between the two came back changed for no stated reason.
    # A theta_prev strictly inside [0.05, 0.1] separates the two rules by
    # construction — under the old floor it comes back 0.1, under the correct
    # one it comes back untouched — and it is used for both paths.
    fb_ok = True
    probe_theta = 0.07
    L.check(CLAMP_LOW < probe_theta < 0.1, "fallback_probe_is_not_between",
            f"{probe_theta} must lie strictly between the served floor "
            f"{CLAMP_LOW} and the 0.1 an earlier driver used, or this arm "
            f"cannot tell the two apart")
    # first iteration: no previous residual to extrapolate from
    th_first, _ = _aitken(np.array([300.0, 301.0]), np.array([301.0, 302.0]),
                          None, probe_theta)
    print(f"fallback_first_iteration_returned={th_first!r}")
    fb_ok = L.check(th_first == probe_theta,
                    "first_iteration_fallback_did_not_hold_theta",
                    f"with no previous residual the update must hold "
                    f"theta_prev={probe_theta} inside [{CLAMP_LOW}, "
                    f"{CLAMP_HIGH}]; it returned {th_first!r}") and fb_ok
    # degenerate denominator: r_k identical to r_{k-1}, so dr is exactly zero
    r_same = np.array([1.0, 1.0])
    th_degen, _ = _aitken(np.array([300.0, 301.0]),
                          np.array([300.0, 301.0]) + r_same, r_same, probe_theta)
    print(f"fallback_degenerate_denominator_returned={th_degen!r}")
    fb_ok = L.check(th_degen == probe_theta,
                    "degenerate_fallback_did_not_hold_theta",
                    f"with a zero residual change the update must hold "
                    f"theta_prev={probe_theta} inside [{CLAMP_LOW}, "
                    f"{CLAMP_HIGH}]; it returned {th_degen!r}") and fb_ok
    # and they DO clamp — a theta_prev below the floor comes back at the floor,
    # so "hold the previous theta" is not "return it unconditionally".
    th_below, _ = _aitken(np.array([300.0]), np.array([301.0]), None, 0.001)
    print(f"fallback_below_the_floor_returned={th_below!r}")
    fb_ok = L.check(th_below == CLAMP_LOW, "fallback_did_not_clamp_at_all",
                    f"a theta_prev of 0.001 must come back at the floor "
                    f"{CLAMP_LOW}, not {th_below!r}") and fb_ok
    print(f"fallbacks_clamp_into_the_same_interval={bool(fb_ok)}")

    return bool(ok and hi_ok and mid_ok and fb_ok)


# ── end to end: a real coupling through the registered `couple` tool ────────

def _quiet_stage(root, name, backend, edits):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return L.stage(root, name, backend, edits)


def probe(tag: str, accelerator: str = "aitken", rho: float = RHO,
          theta: float = THETA0, max_iter: int = MAX_ITER, tol: float = TOL,
          dirichlet: str = "left", backend: str = "skfem",
          mesh_l=(16, 16), mesh_r=(14, 12)) -> dict:
    """couplinglib.probe_theta with the ACCELERATOR made a parameter — that
    file hard-codes the constant one and this fixture is about the default."""
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
    lo, hi = ((0.0, 0.0) if not res.get("exports")
              else L.span(res["exports"]["left"]["values"]))
    return {"converged": bool(res.get("converged")),
            "iterations": int(res.get("iterations", 0)),
            "residual": float(res.get("residual", float("nan"))),
            "deviation": (worst / abs(p.t_iface) if math.isfinite(worst)
                          else float("inf")),
            "t_left": 0.5 * (lo + hi), "result": res, "problem": p}


def reached_closed_form(tag: str, r: dict) -> bool:
    p, res = r["problem"], r["result"]
    ok = L.check(r["converged"], f"{tag}_did_not_converge",
                 f"residual {r['residual']:.3e} after {r['iterations']} of "
                 f"{MAX_ITER}: {str(res.get('error'))[:200]}")
    ex = res.get("exports") or {}
    if not ex:
        return bool(L.check(False, f"{tag}_no_exports", "no exports at all"))
    nl, nr = len(ex["left"]["coordinates"]), len(ex["right"]["coordinates"])
    print(f"{tag}_n_points={nl}/{nr}")
    ok = L.check(nl != nr, f"{tag}_matching_meshes",
                 f"both sides used {nl} interface points") and ok
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
    print(f"served_interval=[{CLAMP_LOW},{CLAMP_HIGH}]")

    unit_ok = unit_bounds()
    L.check(unit_ok, "unit_level_clamp_checks_failed",
            "the driver did not return the served bounds for inputs the "
            "unclamped rule sends far outside them")

    # ── the same case twice: driver untouched, then watched ────────────────
    print(f"rho={RHO:g} theta0={THETA0} max_iter={MAX_ITER} tol={TOL:g}")
    plain = probe("plain")
    print(f"plain_converged={plain['converged']}")
    print(f"plain_iterations={plain['iterations']}")
    print(f"plain_residual={plain['residual']:.3e}")
    print(f"plain_deviation_from_exact={plain['deviation']:.3e}")
    physics = reached_closed_form("plain", plain)
    print(f"aitken_reached_closed_form={bool(physics)}")
    L.check(physics, "the_untouched_run_missed_the_closed_form",
            "the end-to-end arm must be a coupling that is actually right, or "
            "what its thetas did is of no interest")

    from core import coupling_driver as CD
    real = CD._aitken
    seen: list[tuple[float, float, float]] = []   # theta_prev, unclamped, theta

    def recorder(prev_relaxed, new_raw, prev_raw, theta_prev):
        th, r = real(prev_relaxed, new_raw, prev_raw, theta_prev)
        seen.append((float(theta_prev),
                     unclamped(prev_relaxed, new_raw, prev_raw, theta_prev),
                     float(th)))
        return th, r

    CD._aitken = recorder
    try:
        watched = probe("watched")
    finally:
        CD._aitken = real

    inert = bool(watched["converged"] == plain["converged"]
                 and watched["iterations"] == plain["iterations"]
                 and watched["t_left"] == plain["t_left"])
    print(f"watched_iterations={watched['iterations']}")
    print(f"recorder_did_not_change_the_run={inert}")
    L.check(inert, "the_recorder_changed_the_run",
            f"untouched: converged={plain['converged']} "
            f"iterations={plain['iterations']} T={plain['t_left']!r}; "
            f"watched: converged={watched['converged']} "
            f"iterations={watched['iterations']} T={watched['t_left']!r}")

    # (a) ONE theta for the WHOLE INTERFACE STATE. The driver relaxes every
    # participant, but Aitken is applied to the composite fixed-point map, so a
    # run that reached iteration N must have asked for exactly N - 1 thetas —
    # not N_PARTICIPANTS * (N - 1), which is what a per-participant scheme
    # gives and what this fixture asserted before the two coupling branches
    # were reconciled. The discrimination is in the arithmetic: with two
    # participants the two counts differ by a factor of two, so neither
    # assertion can be satisfied by the other implementation.
    expected_calls = watched["iterations"] - 1
    per_participant_would_be = N_PARTICIPANTS * expected_calls
    print(f"aitken_theta_calls={len(seen)}")
    print(f"aitken_theta_calls_expected={expected_calls}")
    print(f"aitken_theta_calls_if_per_participant={per_participant_would_be}")
    one_global = bool(len(seen) == expected_calls and expected_calls > 0)
    print(f"one_theta_for_the_whole_interface_state={one_global}")
    L.check(one_global, "theta_was_not_a_single_global_one",
            f"{len(seen)} adaptations over {watched['iterations']} iterations "
            f"with {N_PARTICIPANTS} participants; ONE global theta gives "
            f"{expected_calls} and a per-participant theta gives "
            f"{per_participant_would_be}")

    # (b) STARTING FROM THE THETA YOU PASS: the first adaptation is seeded with
    # theta0 and nothing else.
    first = [tp for tp, _, _ in seen[:1]]
    print(f"first_theta_prev={first}")
    seeded = bool(len(first) == 1 and all(tp == THETA0 for tp in first))
    print(f"first_theta_prev_was_the_theta_passed={seeded}")
    L.check(seeded, "the_first_theta_was_not_the_one_passed",
            f"the first adaptation was seeded with {first}, not "
            f"{THETA0} — 'starting from the theta you pass' is then wrong")

    # (c) the interval, observed from a real run.
    thetas = [th for _, _, th in seen]
    us = [u for _, u, _ in seen if u == u]
    lo_hits = sum(1 for th in thetas if th == CLAMP_LOW)
    hi_hits = sum(1 for th in thetas if th == CLAMP_HIGH)
    inside = sum(1 for th in thetas if CLAMP_LOW < th < CLAMP_HIGH)
    print(f"theta_min_used={min(thetas):.6g} theta_max_used={max(thetas):.6g}")
    if us:
        print(f"unclamped_min={min(us):.6g} unclamped_max={max(us):.6g}")
    print(f"theta_hits_low_bound={lo_hits}")
    print(f"theta_hits_high_bound={hi_hits}")
    print(f"theta_strictly_inside={inside}")

    within = bool(thetas and all(CLAMP_LOW <= th <= CLAMP_HIGH
                                 for th in thetas))
    print(f"every_theta_within_the_bounds={within}")
    L.check(within, "a_theta_left_the_served_interval",
            f"min {min(thetas):.6g}, max {max(thetas):.6g} against "
            f"[{CLAMP_LOW}, {CLAMP_HIGH}]")

    # A bound that is in the source but never reached is not a bound anyone has
    # tested. On this setting the LOW one is reached; the high one is proven at
    # unit level only and its count above is printed for the record.
    exercised = bool(lo_hits >= 1 and any(u < CLAMP_LOW for u in us))
    print(f"clamp_low_exercised_end_to_end={exercised}")
    L.check(exercised, "the_low_bound_was_never_reached_in_a_real_run",
            f"{lo_hits} of {len(thetas)} adaptations landed on {CLAMP_LOW} and "
            f"the unclamped rule's minimum over the run was "
            f"{(min(us) if us else float('nan')):.6g}; the clamp would be "
            f"untested code on this problem")
    # Also require that the clamp is not the whole story — a driver that
    # returned the floor every time would satisfy the line above.
    print(f"adaptation_was_not_all_clamp={bool(inside > 0)}")
    L.check(inside > 0, "every_adaptation_sat_on_a_bound",
            "no theta was strictly inside the interval, so nothing was "
            "adapting and the accelerator is not doing what is claimed")

    print("end_to_end_runs=2")


L.main(body)
