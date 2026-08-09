"""Tier-2: a DYNAMIC thermo-fluid run with L_diff < h undershoots the
coldest prescribed temperature, with no diagnostic.

Verifies febio::heat#5 — the one [Numerical] heat claim that is cheap
enough to execute as a fixture, because the artefact tracks the RATIO
L_diff / h and nothing else, so a single mesh at two ratios settles it.

  L_diff = sqrt(K * t_end / (density * cp_physical)),
  cp_physical = cp_normalized * R / M

The fixture picks t_end to hit a chosen ratio, FLATTENS the load
controller first (the shipped deck ramps the prescribed temperature over
t in [0,1]; at a matched ratio t_end drops below 1 s and an unflattened
run measures the unfinished ramp instead of the artefact — that confound
is what an earlier version of this pitfall reported), and asserts:

  * at ratio 0.1 the coldest node falls BELOW the coldest prescribed
    temperature while the run terminates normally with every step
    completed and exit 0,
  * at ratio 3 the same deck comes back with no node below it,
  * neither run prints any warning or error.

The undershoot is the assertion; the fixture computes it and does not
carry a number in the catalogue text.
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


CSV = "heat_bar_T.csv"
R_GAS, MOLAR_MASS, CP_N, K_COND, RHO = 8.31446, 0.029, 3.5, 0.026, 1.2
T_COLD, T_HOT, N_ELEM = 300.0, 400.0, 8


def t_end_for(ratio: float, n: int) -> float:
    """t_end that makes L_diff / h equal `ratio` on an n-element bar."""
    cp_phys = CP_N * R_GAS / MOLAR_MASS
    h = 1.0 / n
    return (ratio * h) ** 2 * RHO * cp_phys / K_COND


def flat_ramp(deck: str) -> str:
    return L.swap(deck, "<points><pt>0,0</pt><pt>1,1</pt></points>",
                  "<points><pt>0,1</pt><pt>1,1</pt></points>")


def field(run) -> list:
    blocks = L.parse_log_csv(run.files.get(CSV) or "")
    return [v[0] for v in blocks[-1][1].values()] if blocks else []


def main() -> int:
    results = {}
    if MUTATE:
        print("mutation=the_under_resolved_case_runs_at_ratio_3")
    for key in (0.1, 3.0):
        ratio = 3.0 if (MUTATE and key == 0.1) else key
        te = t_end_for(ratio, N_ELEM)
        deck = flat_ramp(L.template("heat_3d_bar", n=N_ELEM,
                                    analysis="DYNAMIC", time_steps=10,
                                    step_size=te / 10))
        r = L.run(deck, collect=(CSV,), timeout=900)
        vals = field(r)
        lo = min(vals) if vals else None
        results[key] = (r, lo)
        print(f"ratio={ratio} t_end={te:.6g} rc={r.rc} "
              f"normal={int(r.normal_termination)} "
              f"steps={r.steps_completed} minT={lo} "
              f"below_coldest_prescribed="
              f"{int(lo is not None and lo < T_COLD - 1e-6)} "
              f"warnings={int('WARNING' in r.text)} "
              f"errors={int('ERROR' in r.text)}")

    r_bad, lo_bad = results[0.1]
    r_ok, lo_ok = results[3.0]
    silent = (r_bad.rc == 0 and r_bad.normal_termination
              and r_bad.steps_completed == 10
              and "ERROR" not in r_bad.text and "WARNING" not in r_bad.text)
    undershoots = lo_bad is not None and lo_bad < T_COLD - 1.0
    recovered = (lo_ok is not None and lo_ok >= T_COLD - 1e-6
                 and r_ok.rc == 0 and r_ok.normal_termination)
    print(f"undershoot_kelvin={None if lo_bad is None else T_COLD - lo_bad}")
    print(f"silent_wrong_answer={int(silent and undershoots)} "
          f"gone_at_ratio_3={int(recovered)}")
    good = silent and undershoots and recovered
    return L.report(good, "diffusion_length_undershoot", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
