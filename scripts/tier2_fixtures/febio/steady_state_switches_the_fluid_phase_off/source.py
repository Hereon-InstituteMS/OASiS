"""Tier-2: a flat pore-pressure field is the ANALYSIS WORD, not a missing
drainage BC.

Verifies febio::biphasic#2. The shipped confined-compression deck HAS a
`zero fluid pressure` BC on its loaded face. Under
<analysis>STEADY-STATE</analysis> the element pressure is EXACTLY zero at
every step, because steady state sets div w = 0 and there is no transient
to solve. Change the single word to TRANSIENT and the pressure comes
alive, while the solid stress converges on the same value.

Both runs reach normal termination with exit 0. The fixture asserts:

  * every logged pressure is exactly 0.0 under STEADY-STATE,
  * at least one is not, under TRANSIENT,
  * the final solid stress agrees between the two to a tight tolerance,
    which is what shows the fluid phase and the permeability are live and
    the analysis word had merely switched them off.
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


OUT = ("  <Output>\n    <logfile>\n"
       '      <element_data data="sz;p" delim="," file="e.csv"/>\n'
       "    </logfile>\n  </Output>\n")


def run_with(analysis: str):
    base = L.template("biphasic_3d_confined")
    deck = L.swap(base, "<analysis>STEADY-STATE</analysis>",
                  f"<analysis>{analysis}</analysis>")
    deck = L.swap(deck, "</febio_spec>", OUT + "</febio_spec>")
    r = L.run(deck, collect=("e.csv",), timeout=900)
    blocks = L.parse_log_csv(r.files.get("e.csv") or "")
    pressures = [rows[1][1] for _t, rows in blocks if 1 in rows]
    stresses = [rows[1][0] for _t, rows in blocks if 1 in rows]
    return r, pressures, stresses


def main() -> int:
    ss, p_ss, s_ss = run_with("STEADY-STATE")
    tr, p_tr, s_tr = run_with("TRANSIENT")
    if not (p_ss and p_tr):
        print("FAIL: no element pressure logged")
        return L.report(False, "steady_state_flat_pressure", "reproduced",
                        "not_reproduced")
    flat = all(p == 0.0 for p in p_ss)
    alive = any(p != 0.0 for p in p_tr)
    rel = abs(s_ss[-1] - s_tr[-1]) / max(abs(s_ss[-1]), abs(s_tr[-1]), 1e-30)
    print(f"STEADY-STATE: rc={ss.rc} normal={int(ss.normal_termination)} "
          f"steps={ss.steps_completed} "
          f"every_pressure_exactly_zero={int(flat)} "
          f"max_abs_p={max(abs(p) for p in p_ss):.6g} "
          f"final_sz={s_ss[-1]}")
    print(f"TRANSIENT: rc={tr.rc} normal={int(tr.normal_termination)} "
          f"steps={tr.steps_completed} "
          f"pressure_is_nonzero={int(alive)} "
          f"max_abs_p={max(abs(p) for p in p_tr):.6g} "
          f"final_sz={s_tr[-1]}")
    print(f"same_solid_stress_relative_difference={rel:.3e}")
    good = (flat and alive and rel < 1e-6
            and ss.rc == 0 and tr.rc == 0
            and ss.normal_termination and tr.normal_termination)
    return L.report(good, "steady_state_flat_pressure", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
