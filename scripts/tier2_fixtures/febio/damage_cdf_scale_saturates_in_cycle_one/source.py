"""Tier-2: a CDF scale small enough to saturate in cycle 1 makes the model
look elastic-with-a-knockdown.

Verifies febio::damage#3. The damage-rate parameters on `CDF Weibull` are
<alpha> and <mu>; there is no beta. Two halves:

  * <beta> is rejected by name at parse — nothing is silently ignored,
  * with a small <mu> the second cycle's peak equals the first to a
    fraction of a percent while sitting far BELOW the undamaged
    reference: the knockdown was fully applied during cycle 1. With the
    shipped <mu> the second cycle is markedly softer than the first.

The comparison against the undamaged reference is what distinguishes
"saturated damage" from "no damage at all"; without it, equal peaks would
be equally consistent with a model that never damaged anything.
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


SHIPPED_POINTS = ("<points>\n"
                  "        <pt>0,0</pt><pt>0.5,0.5</pt><pt>1.0,0</pt>\n"
                  "        <pt>1.5,0.75</pt><pt>2.0,0</pt>\n"
                  "        <pt>2.5,1.0</pt><pt>3.0,0</pt>\n"
                  "      </points>")
TWO_EQUAL = ("<points>\n"
             "        <pt>0,0</pt><pt>0.5,1.0</pt><pt>1.0,0</pt>\n"
             "        <pt>1.5,1.0</pt><pt>2.0,0</pt>\n"
             "      </points>")


def cycle_peaks(deck):
    r = L.run(L.add_logfile(deck, ("element_data", "sz", "s.csv")),
              collect=("s.csv",), timeout=1200)
    blocks = L.parse_log_csv(r.files.get("s.csv") or "")
    hist = [(t, rows[1][0]) for t, rows in blocks if 1 in rows]
    first = [abs(v) for t, v in hist if t is not None and t <= 1.0]
    second = [abs(v) for t, v in hist if t is not None and 1.0 < t <= 2.0]
    return r, (max(first) if first else None), (max(second) if second else None)


def main() -> int:
    base = L.template("damage_3d_cycle")
    two_cycle = L.swap(L.swap(base, SHIPPED_POINTS, TWO_EQUAL),
                       "<time_steps>60</time_steps>",
                       "<time_steps>40</time_steps>")

    beta = L.run(L.swap(two_cycle, "<mu>0.5</mu>",
                        "<mu>0.5</mu><beta>2.0</beta>"), timeout=900)
    beta_ok = ('tag "beta"' in beta.text and "unrecognized tag" in beta.text
               and beta.read_failed and beta.rc != 0)
    print(f"beta_parameter: rc={beta.rc} "
          f"read_failed={int(beta.read_failed)} "
          f"rejected_by_name={int(beta_ok)}")

    r_und, u1, _u2 = cycle_peaks(
        L.swap(two_cycle, "<Dmax>0.9</Dmax>", "<Dmax>0.0</Dmax>"))
    r_sat, s1, s2 = cycle_peaks(L.swap(two_cycle, "<mu>0.5</mu>",
                                       "<mu>0.01</mu>"))
    r_liv, l1, l2 = cycle_peaks(two_cycle)
    if None in (u1, s1, s2, l1, l2):
        print("FAIL: a run logged no element stress")
        return L.report(False, "damage_scale", "reproduced",
                        "not_reproduced")
    sat_ratio, live_ratio = s2 / s1, l2 / l1
    print(f"undamaged_reference: rc={r_und.rc} cycle1_peak={u1}")
    print(f"small_scale_mu: rc={r_sat.rc} "
          f"normal={int(r_sat.normal_termination)} cycle1_peak={s1} "
          f"cycle2_peak={s2} ratio={sat_ratio:.6f} "
          f"below_undamaged={int(s1 < 0.5 * u1)}")
    print(f"shipped_scale_mu: rc={r_liv.rc} "
          f"normal={int(r_liv.normal_termination)} cycle1_peak={l1} "
          f"cycle2_peak={l2} ratio={live_ratio:.6f}")
    saturated = abs(sat_ratio - 1.0) < 0.01 and s1 < 0.5 * u1
    still_climbing = live_ratio < 0.9
    all_clean = all(r.rc == 0 and r.normal_termination
                    for r in (r_und, r_sat, r_liv))
    print(f"small_scale_saturates_in_cycle_one={int(saturated)} "
          f"shipped_scale_still_softens={int(still_climbing)}")
    good = beta_ok and saturated and still_climbing and all_clean
    return L.report(good, "damage_scale", "reproduced", "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
