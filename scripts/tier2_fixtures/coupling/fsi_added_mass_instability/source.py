"""The added-mass instability of partitioned FSI, measured rather than cited.

THE CLAIM UNDER TEST is the `fsi` pitfall entry that says a partitioned
Dirichlet-Neumann coupling with an incompressible fluid diverges once the fluid
it has to push is not light compared with the structure, and that relaxation
moves that boundary without removing it. A pitfall entry an agent is served has
to be something this install has actually seen, so this fixture drives the same
case across the boundary in both directions.

THE CASE is one backward-Euler step from rest of the transient problem — which
is the only setting in which added mass exists at all. A steady coupling has
none: the interface is a stationary wall and the fluid never has to be
accelerated. With DT > 0 the interface becomes a MOVING wall with u = d/dt, and
that kinematic term is how the structure's acceleration reaches the fluid.

WHAT IS SWEPT is the structure density alone. Geometry, fluid, mesh, time step
and tolerance are held fixed, so the only thing that changes between a run that
converges and a run that does not is the mass the structure brings against the
mass of fluid it has to move — rho_s * h against rho_f * L for this geometry.

NOTHING HERE IS TUNED TO A THRESHOLD. The fixture asserts the SHAPE of the
result — that a heavy structure converges unrelaxed, that a light one diverges
with its residual growing rather than stalling, that Aitken converges a case
constant relaxation cannot, and that a lighter case defeats Aitken too. The
density at which each of those happens is printed, not asserted, because it is
a property of this geometry and would become a number to hit rather than a
number to measure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                        # noqa: E402
import fsilib as F                                             # noqa: E402

DT = 0.01
MAX_ITER = 25
TOL = 1e-7


def run(tag, rho_s, accel, theta, case):
    r = F.run_transient(tag, rho_s, DT, case, accelerator=accel, theta=theta,
                        max_iter=MAX_ITER, tol=TOL)
    h = [v for v in (r.get("history") or []) if v == v]
    conv = bool(r.get("converged"))
    grew = bool(h and max(h) > 1.3 * h[0])
    print(f"{tag}_converged={conv}")
    print(f"{tag}_iterations={r.get('iterations')}")
    print(f"{tag}_residual_grew={grew}")
    print(f"{tag}_history_head={[round(v, 6) for v in h[:4]]}")
    print(f"{tag}_history_tail={[round(v, 9) for v in h[-3:]]}")
    print(f"{tag}_error={(r.get('error') or 'none')[:110]}")
    return r, h, conv, grew


def body() -> None:
    L.require_available("fenics", "skfem")
    case = F.FsiCase(nxf=32, nyf=8)
    mass_f = case.rho_f * case.lx          # the fluid the interface must move
    print(f"added_mass_scale_rho_f_times_L={mass_f:.6g}")

    # ── heavy structure, NO relaxation: the control that the case is solvable
    heavy = 1000.0
    print(f"heavy_rho_s={heavy} heavy_mass_ratio="
          f"{heavy * case.hs / mass_f:.6g}")
    _, hh, hc, hg = run("heavy_unrelaxed", heavy, "constant", 1.0, case)
    L.check(hc, "heavy_unrelaxed_converges",
            "the unrelaxed iteration fails even for a heavy structure, so this "
            "sweep is measuring something other than added mass")
    L.check(not hg, "heavy_unrelaxed_residual_falls",
            f"the residual grew on the heavy case: {hh[:4]}")

    # ── light structure, NO relaxation: the instability itself
    light = 100.0
    print(f"light_rho_s={light} light_mass_ratio="
          f"{light * case.hs / mass_f:.6g}")
    _, lh, lc, lg = run("light_unrelaxed", light, "constant", 1.0, case)
    L.check(not lc, "light_unrelaxed_diverges",
            "the unrelaxed iteration converged for the light structure too, so "
            "on this geometry the boundary is elsewhere and the sweep must be "
            "widened rather than the claim kept")
    L.check(lg, "light_unrelaxed_residual_grows",
            f"the light unrelaxed run failed to converge but its residual did "
            f"NOT grow ({lh[:4]} -> {lh[-3:]}) — that is a stall, which is a "
            f"different failure from added mass and must not be reported as it")

    # ── the SAME light structure with Aitken: relaxation moves the boundary
    _, ah, ac, ag = run("light_aitken", light, "aitken", 0.5, case)
    L.check(not ag, "light_aitken_residual_falls",
            f"Aitken did not stabilise the case constant relaxation diverged "
            f"on: {ah[:4]} -> {ah[-3:]}")
    gain = (lh[-1] / ah[-1]) if ah and ah[-1] > 0 else float("inf")
    print(f"aitken_residual_gain_at_light={gain:.4g}")
    L.check(gain > 100.0, "aitken_beats_constant_at_the_same_density",
            f"Aitken left the residual within a factor {gain:.3g} of the "
            f"unrelaxed run's, so it did not move the boundary here")

    # ── lighter still: Aitken has a limit too, and that is the point
    lighter = 10.0
    print(f"lighter_rho_s={lighter} lighter_mass_ratio="
          f"{lighter * case.hs / mass_f:.6g}")
    _, gh, gc, gg = run("lighter_aitken", lighter, "aitken", 0.5, case)
    L.check(not gc, "lighter_aitken_still_diverges",
            "Aitken converged the lighter case as well, so the claim that "
            "relaxation only MOVES the boundary is not supported here and the "
            "entry must be rewritten")
    L.check(gg, "lighter_aitken_residual_grows",
            f"the lighter Aitken run did not converge but its residual did not "
            f"grow either ({gh[:4]} -> {gh[-3:]}); that is a stall, not added "
            f"mass")

    # ── the observable the pitfall entry promises, checked as a number ─────
    #     A partitioned FSI that is diverging is not distinguished from one
    #     converging slowly by the verdict alone — both come back
    #     converged=False. It is the residual HISTORY that separates them, and
    #     that is what the entry tells an agent to read.
    print(f"divergence_is_visible_in_history_not_verdict="
          f"{bool((not lc) and lg and (not hg))}")
    L.check((not lc) and lg and (not hg),
            "history_separates_divergence_from_slow_convergence",
            "the residual history did not separate the diverging run from the "
            "converging one, so the Signal in the pitfall entry is wrong")

    print("densities_run=4")


L.main(body)
