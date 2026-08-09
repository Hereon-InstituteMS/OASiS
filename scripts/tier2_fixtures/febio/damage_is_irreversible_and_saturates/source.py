"""Tier-2: two identical load cycles, and what the damage cap does to
them.

Verifies febio::damage#0 and NARROWS it. The shipped deck's load curve
ramps to three DIFFERENT amplitudes, so it cannot answer a question about
two identical cycles; the fixture replaces it with a 0 -> A -> 0 -> A
history, which is what the claim prescribes, and sweeps Dmax.

Executed at Dmax = 0, 0.3, 0.6 and 0.9: the second cycle's peak equals
the first to within a fraction of a percent at Dmax = 0, 0.3 AND 0.6 —
damage has already saturated at its cap during cycle 1 — and is markedly
softer only at Dmax = 0.9.

WHAT THIS FALSIFIES. The claim says the peaks are "bit-identical" at
Dmax = 0 and 0.3, and that "at Dmax = 0.6 and 0.9 cycle 2 was markedly
softer". Neither holds as stated: the saturated peaks agree closely but
not bit-for-bit, and Dmax = 0.6 sits on the saturated side, not the
softening side. The structure of the claim — irreversible, saturating,
softening only while D is still climbing — survives; the boundary and the
word "bit-identical" do not. The fixture asserts the structure and the
measured ordering, and deliberately does not pin the boundary, which is
deck-dependent.
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
CAPS = ("0.0", "0.3", "0.6", "0.9")


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
    ratios = {}
    all_clean = True
    for cap in CAPS:
        r, p1, p2 = cycle_peaks(
            L.swap(two_cycle, "<Dmax>0.9</Dmax>", f"<Dmax>{cap}</Dmax>"))
        all_clean = all_clean and r.rc == 0 and r.normal_termination
        ratio = None if not p1 else p2 / p1
        ratios[cap] = ratio
        print(f"Dmax={cap}: rc={r.rc} normal={int(r.normal_termination)} "
              f"steps={r.steps_completed} cycle1_peak={p1} "
              f"cycle2_peak={p2} "
              f"ratio={'None' if ratio is None else f'{ratio:.6f}'}")
    if any(v is None for v in ratios.values()):
        print("FAIL: a run logged no element stress")
        return L.report(False, "damage_cycles", "reproduced",
                        "not_reproduced")
    saturated = [c for c in CAPS if abs(ratios[c] - 1.0) < 0.01]
    softening = [c for c in CAPS if ratios[c] < 0.9]
    print(f"caps_whose_second_cycle_repeats_the_first={saturated}")
    print(f"caps_whose_second_cycle_is_markedly_softer={softening}")
    print(f"peaks_at_saturated_caps_are_close_but_not_bit_identical="
          f"{int(all(0 < abs(ratios[c] - 1.0) < 0.01 for c in saturated))}")
    good = (all_clean and len(saturated) >= 2 and len(softening) >= 1
            and max(float(c) for c in saturated)
            < min(float(c) for c in softening))
    return L.report(good, "damage_cycles", "reproduced_with_correction",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
