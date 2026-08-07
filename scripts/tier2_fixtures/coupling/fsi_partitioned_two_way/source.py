"""FSI through the generic `couple` path: cross-code, same-code, and the two
one-way controls that prove both directions carry physics.

WHAT THIS FIXTURE ESTABLISHES, and nothing beyond it:

  1. a partitioned fluid-structure interaction runs end to end through the
     public `couple` tool, with the fluid in FEniCSx and the structure in
     scikit-fem — two different codes under two different interpreters;
  2. the same coupling with the structure ALSO in FEniCSx lands on the same
     interface displacement, so the answer is not an artefact of one structure
     code;
  3. the converged answer is the ROOT of the coupled system, not merely a place
     the iteration stopped: a Newton-Krylov re-solve of the same coupled
     interface equation is handed to `couple(monolithic=)` and agrees;
  4. BOTH DIRECTIONS ARE ACTIVE. Suppressing structure->fluid (the fluid stops
     moving its mesh) changes the converged deflection by a measured amount well
     outside every tolerance here, and `couple`'s own responsiveness and
     interface-sensitivity checks catch the suppression unprompted. Suppressing
     fluid->structure collapses the deflection to zero and breaks the interface
     force balance. A one-way coupling could not produce (4).
  5. interface equilibrium and kinematic continuity hold COMPONENTWISE across
     two non-matching interface discretisations.

WHAT IT DOES NOT ESTABLISH. The Newton-Krylov reference drives the same
participant scripts, so it cannot see a bug inside a participant — both would
be wrong identically. The closed-form handles below are what stands in for that
on the fluid and structure separately, and an independent-code FSI reference is
what stands in for it on the coupled problem.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                        # noqa: E402
import fsilib as F                                             # noqa: E402


def body() -> None:
    L.require_available("fenics", "skfem")
    case = F.FsiCase()

    # ── 1. cross-code two-way, checked against the Newton-Krylov re-solve ──
    x = F.run_pair("cross", "skfem", case, with_reference=True)
    ok = F.report_verdict(x, "cross")
    ok &= F.interface_equilibrium(x, "cross")
    ok &= F.kinematic_continuity(x, "cross")
    mono = x.get("monolithic_check") or {}
    print(f"cross_reference_status={mono.get('status')}")
    L.check(mono.get("status") == "checked", "cross_reference_ran",
            str(mono)[:300])
    rel = float((mono.get("solid") or {}).get("relative_l2", float("nan")))
    print(f"cross_reference_rel_l2={rel:.3e}")
    L.check(rel == rel and rel < 1e-5, "cross_matches_reference",
            f"relative L2 against the Newton-Krylov root = {rel:.3e}")
    d_cross = F.max_dy(x)
    print(f"cross_max_dy={d_cross:.9e}")
    print(f"cross_n_points_fluid={len(x['exports']['fluid']['coordinates'])}")
    print(f"cross_n_points_solid={len(x['exports']['solid']['coordinates'])}")
    L.check(len(x["exports"]["fluid"]["coordinates"])
            != len(x["exports"]["solid"]["coordinates"]),
            "cross_meshes_nonmatching",
            "the two interface discretisations are identical, so the mapping "
            "was never exercised")
    print(f"cross_meshes_nonmatching=True")

    # ── 2. same-code control: the structure in FEniCSx instead ─────────────
    s = F.run_pair("same", "fenics", case, with_reference=True)
    ok &= F.report_verdict(s, "same")
    ok &= F.interface_equilibrium(s, "same")
    ok &= F.kinematic_continuity(s, "same")
    d_same = F.max_dy(s)
    print(f"same_max_dy={d_same:.9e}")
    cs = F.profile_rel_l2(F.signed_profile(x), F.signed_profile(s))
    print(f"crosscode_vs_samecode_rel_l2={cs:.3e}")
    L.check(cs < 5e-3, "crosscode_agrees_with_samecode",
            f"the two structure codes disagree by {cs:.2%} in relative L2 on "
            f"the converged interface displacement")

    # ── 3. suppress structure -> fluid (the fluid stops moving its mesh) ────
    #     This run is ALSO the rigid-wall configuration, where the fluid
    #     problem is plane Poiseuille and both closed forms apply.
    r = F.run_pair("rigid", "skfem", case, move_mesh=False)
    fx, fy = r["_raw_fluid"]["meta"]["net_force"]
    print(f"rigid_net_force_x={fx:.6f} rigid_net_force_y={fy:.6f}")
    L.close(fy, case.rigid_wall_normal_force, 0.02 * case.rigid_wall_normal_force,
            "rigid_normal_force_err")
    L.close(fx, case.rigid_wall_shear_force, 0.02 * case.rigid_wall_shear_force,
            "rigid_shear_force_err")
    d_rigid = F.max_dy(r)
    print(f"rigid_max_dy={d_rigid:.9e}")
    L.close(d_rigid, case.beam_deflection_estimate,
            0.03 * case.beam_deflection_estimate, "rigid_beam_deflection_err")

    #     THE TWO-WAY PROOF, from the numbers.
    swing = abs(d_rigid - d_cross) / d_cross
    print(f"reverse_direction_effect={swing:.6f}")
    L.check(swing > 0.03, "reverse_direction_is_active",
            f"suppressing structure->fluid moved the converged deflection by "
            f"only {swing:.3%}; at that size this case is effectively one-way "
            f"and must not be reported as two-way FSI")

    #     THE TWO-WAY PROOF, from OASiS's own checks: with the fluid ignoring
    #     the displacement its export cannot move when its imports do, so the
    #     interface-sensitivity probe must object without being told.
    #     Asserted on the NUMBER the probe returns, not on the wording of the
    #     message: a check that greps a sentence stops working the day the
    #     sentence is reworded, and would have to be rewritten rather than
    #     re-run to notice.
    print(f"rigid_validation_nonempty={bool(r.get('validation'))}")
    L.check(bool(r.get("validation")), "rigid_suppression_detected",
            "couple reported NO finding on a run whose fluid ignores the "
            "structure entirely — the one-way detector is not working")
    L.check("NOT VERIFIED" in (r.get("verification") or ""),
            "rigid_not_verified",
            f"verdict was {str(r.get('verification'))[:160]}")
    for name, res_ in (("cross", x), ("rigid", r)):
        for who in ("fluid", "solid"):
            sv = ((res_.get("interface_sensitivity") or {}).get(who) or {})
            print(f"{name}_sensitivity_{who}_S={sv.get('S')} "
                  f"noise={sv.get('noise')} signal={sv.get('signal')}")
    s_cross_fluid = ((x.get("interface_sensitivity") or {}).get("fluid") or {}).get("S")
    s_cross_solid = ((x.get("interface_sensitivity") or {}).get("solid") or {}).get("S")
    s_rigid_fluid = ((r.get("interface_sensitivity") or {}).get("fluid") or {}).get("S")
    L.check(s_cross_fluid is not None and s_cross_fluid > 1e-4,
            "twoway_fluid_responds_to_displacement",
            f"the fluid's interface sensitivity is {s_cross_fluid} — its answer "
            f"does not depend on the displacement it is handed, so this is not "
            f"a two-way coupling")
    L.check(s_cross_solid is not None and s_cross_solid > 1e-4,
            "twoway_solid_responds_to_traction",
            f"the structure's interface sensitivity is {s_cross_solid}")
    L.check(s_rigid_fluid is not None and s_rigid_fluid < 1e-9,
            "rigid_fluid_sensitivity_collapses",
            f"with the mesh motion switched off the fluid's sensitivity is "
            f"{s_rigid_fluid}, not ~0 — the suppression did not take, so the "
            f"comparison above is not measuring what it claims")

    # ── 3b. the reference check must be able to FAIL. Stop the iteration
    #     early under a tolerance loose enough to call it converged, and the
    #     Newton-Krylov root must then be visibly far away. Without this the
    #     agreement in step 1 is unfalsifiable.
    #     The loose tolerance is MEASURED, not guessed: run the truncation once
    #     under a tolerance nothing could meet to find out what residual three
    #     under-relaxed iterations actually leave, then set the tolerance just
    #     above it. Guessing a number here made the control fail as
    #     "not converged", which is the driver behaving correctly and NOT the
    #     case this control exists to exhibit — a run that reports CONVERGED and
    #     is nowhere near the root.
    cal = F.run_pair("shortcal", "skfem", case, max_iter=3, tol=1e-14,
                     accelerator="constant", theta=0.3)
    hist = [v for v in (cal.get("history") or []) if v == v]
    loose = 1.5 * hist[-1] if hist else 1.0
    print(f"stopshort_calibrated_tol={loose:.3e}")
    s2 = F.run_pair("short", "skfem", case, with_reference=True,
                    max_iter=3, tol=loose, accelerator="constant", theta=0.3)
    m2 = s2.get("monolithic_check") or {}
    rel2 = float((m2.get("solid") or {}).get("relative_l2", float("nan")))
    print(f"stopshort_converged={bool(s2.get('converged'))}")
    print(f"stopshort_reference_rel_l2={rel2:.3e}")
    L.check(bool(s2.get("converged")), "stopshort_reported_converged",
            "the deliberately-truncated run did not even report convergence, "
            "so it does not exercise the case this control exists for")
    L.check(rel2 == rel2 and rel2 > 1e-2, "reference_check_discriminates",
            f"a run stopped 3 iterations in at tol=2e-1 was still within "
            f"{rel2:.3e} of the Newton-Krylov root, so the agreement reported "
            f"in step 1 does not distinguish a converged coupling from a "
            f"truncated one")
    L.check(bool(s2.get("validation")), "stopshort_flagged",
            "couple verified a coupling that is far from the root of its own "
            "system")

    # ── 4. suppress fluid -> structure (the structure ignores the load) ─────
    n = F.run_pair("noload", "skfem", case, feedback=False)
    d_noload = F.max_dy(n)
    print(f"noload_max_dy={d_noload:.3e}")
    L.check(d_noload < 1e-12 * max(d_cross, 1e-30), "noload_collapses",
            f"the structure still deflected {d_noload:.3e} with the load "
            f"suppressed")
    nv = " | ".join(n.get("validation") or [])
    print(f"noload_validation_nonempty={bool(n.get('validation'))}")
    L.check(bool(n.get("validation")), "noload_suppression_detected",
            "couple reported NO finding on a run whose structure ignores the "
            "fluid entirely")

    # ── 5. THE MODE NOTHING HERE CATCHES, run so that the limit is measured
    #     rather than argued. Flip the WHOLE traction convention in the fluid —
    #     both the load the structure applies and the flux the conservation
    #     check sees — which is what a participant written with the other
    #     normal produces. Every internal check then agrees with itself: the
    #     iteration converges, the interface force balances exactly, and the
    #     Newton-Krylov reference agrees to solver tolerance because it drives
    #     THE SAME flipped scripts. Only an answer from outside separates it,
    #     and here that is the closed form on the undeformed configuration.
    b = F.run_pair("signflip", "skfem", case, with_reference=True,
                   fluid_extra=F.WHOLE_CONVENTION_FLIP)
    print(f"signflip_converged={bool(b.get('converged'))}")
    eq_ok = F.interface_equilibrium(b, "signflip")
    mb = b.get("monolithic_check") or {}
    relb = float((mb.get("solid") or {}).get("relative_l2", float("nan")))
    print(f"signflip_reference_rel_l2={relb:.3e}")
    d_flip = F.max_dy(b)
    fyb = b["_raw_fluid"]["meta"]["net_force"][1]
    print(f"signflip_max_dy={d_flip:.9e} signflip_net_force_y={fyb:.6f}")
    dy_flip = np.asarray(b["exports"]["solid"]["values"], float)[:, 1]
    print(f"signflip_deflects_the_other_way={bool(dy_flip.max() <= 0)}")
    L.check(bool(b.get("converged")), "signflip_still_converges",
            "the flipped run did not converge, so it does not demonstrate the "
            "silent case")
    L.check(eq_ok, "signflip_still_balances",
            "the force balance caught the whole-convention flip after all — "
            "then this limit no longer holds and the note must be rewritten")
    L.check(relb == relb and relb < 1e-5, "signflip_reference_still_agrees",
            f"the shared-script reference disagreed by {relb:.3e} on the "
            f"flipped run; if it can see this, say so instead")
    L.check(bool(dy_flip.max() <= 0), "signflip_deflection_reverses",
            f"the deflection did not reverse (max dy = {dy_flip.max():.3e}), "
            f"so the flip did not take and this control demonstrates nothing")
    #     The separation is a PHYSICS statement, not a tolerance: a channel
    #     carrying a positive pressure drop cannot pull its own wall inwards.
    #     Any answer that says it does is wrong however cleanly it converged.
    L.check(fyb < 0.0, "signflip_caught_by_the_sign_of_the_load",
            f"the exported net normal interface force stayed positive "
            f"({fyb:.4g}) on the flipped run, so the flip is not visible even "
            f"in the load")
    caught_only_by_physics = (bool(b.get("converged")) and eq_ok
                              and relb == relb and relb < 1e-5)
    print(f"signflip_caught_only_by_closed_form={caught_only_by_physics}")

    print("configurations_run=6")


L.main(body)
