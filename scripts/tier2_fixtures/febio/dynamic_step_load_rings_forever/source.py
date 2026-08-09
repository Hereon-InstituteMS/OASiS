"""Tier-2: a step-applied stretch under DYNAMIC rings at the step-size
Nyquist frequency and the Prony terms do not damp it.

Verifies febio::viscoelasticity#0. The shipped relaxation deck is DYNAMIC
with a STEP-interpolated load curve. Its logged sz alternates sign on
almost every step, so there is no plateau to read and the value at t_end
is an arbitrary phase.

Three runs:

  * the shipped deck — the sign of sz must flip on nearly every step, and
    the peak |sz| must EXCEED the final |sz|, i.e. the amplitude does not
    decay,
  * the same deck under STATIC — must give ZERO sign changes and a final
    sz of the correct (compressive, negative) sign, which the DYNAMIC run
    does not even get right,
  * the DYNAMIC deck with the step refined tenfold — must ring WORSE,
    which is what rules out step-size convergence error.

Everything terminates normally with exit 0, so the termination banner
tells you nothing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    # MUTATION CONTROL. The pathology this fixture reproduces is the
    # EDIT that turns a correct deck into the broken one, so removing
    # the pathology means not making that edit. Neutralising L.swap /
    # L.drop does exactly that: every deck built below is the correct
    # one, the pitfall is never triggered, the diagnostic must not
    # appear and the verdict token flips to not_reproduced.
    print("mutation=the_deck_edits_that_introduce_the_pitfall_are_"
          "neutralised")
    L.swap = lambda deck, old, new, **kw: deck
    L.drop = lambda deck, fragment: deck


CSV = "visco_relax.csv"


def stress_history(run):
    blocks = L.parse_log_csv(run.files.get(CSV) or "")
    return [rows[1][0] for _t, rows in blocks if 1 in rows][1:]


def sign_changes(values) -> int:
    return sum(1 for a, b in zip(values, values[1:]) if a * b < 0)


def main() -> int:
    base = L.template("viscoelasticity_3d_stress_relax")
    static = L.swap(base, "<analysis>DYNAMIC</analysis>",
                    "<analysis>STATIC</analysis>")
    fine = L.swap(L.swap(base, "<time_steps>50</time_steps>",
                         "<time_steps>500</time_steps>"),
                  "<step_size>0.2</step_size>", "<step_size>0.02</step_size>")

    rd = L.run(base, collect=(CSV,), timeout=1200)
    rs = L.run(static, collect=(CSV,), timeout=1200)
    rf = L.run(fine, collect=(CSV,), timeout=1800)

    hd, hs, hf = (stress_history(rd), stress_history(rs),
                  stress_history(rf))
    if not (hd and hs and hf):
        print("FAIL: one of the runs logged no element stress")
        return L.report(False, "relaxation_ringing", "reproduced",
                        "not_reproduced")
    cd, cs, cf = sign_changes(hd), sign_changes(hs), sign_changes(hf)
    peak_d, final_d = max(abs(v) for v in hd), abs(hd[-1])
    peak_f = max(abs(v) for v in hf)
    print(f"shipped_DYNAMIC: rc={rd.rc} "
          f"normal={int(rd.normal_termination)} "
          f"steps={rd.steps_completed} samples={len(hd)} "
          f"sign_changes={cd} peak={peak_d:.6g} final={hd[-1]:.6g}")
    print(f"STATIC_control: rc={rs.rc} "
          f"normal={int(rs.normal_termination)} "
          f"samples={len(hs)} sign_changes={cs} final={hs[-1]:.6g}")
    print(f"DYNAMIC_step_refined_10x: rc={rf.rc} "
          f"normal={int(rf.normal_termination)} "
          f"samples={len(hf)} sign_changes={cf} peak={peak_f:.6g}")
    rings = cd > 0.8 * len(hd)
    undamped = peak_d > final_d
    clean_static = cs == 0 and hs[-1] < 0
    worse_when_refined = peak_f > peak_d and cf > cd
    all_normal = all(r.rc == 0 and r.normal_termination
                     for r in (rd, rs, rf))
    print(f"rings_on_almost_every_step={int(rings)} "
          f"amplitude_does_not_decay={int(undamped)} "
          f"static_is_monotone_and_compressive={int(clean_static)} "
          f"refining_makes_it_worse={int(worse_when_refined)} "
          f"all_terminate_normally={int(all_normal)}")
    good = (rings and undamped and clean_static and worse_when_refined
            and all_normal)
    return L.report(good, "relaxation_ringing", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
