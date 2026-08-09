"""Tier-2: the compression-side response is INVARIANT to k1 — that
invariance is the proof the fiber term is gated off.

Verifies febio::fiber_reinforced#1. The HGO fiber energy is switched off
once the fiber stretch drops below 1, so a compressed specimen returns
the matrix-only response. The fixture takes the shipped HGO deck, points
the fibers ALONG the load, and runs four cases: +10% and -10% stretch, at
k1/c = 1 and at k1/c = 10.

The assertion that matters is not that tension differs from compression —
that is true of almost any material — but that the two COMPRESSION runs
give the same stress despite a tenfold change in k1, while the two
TENSION runs do not.
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


MAT_AXIS_X = ('      <mat_axis type="vector">\n'
              "        <a>1,0,0</a>\n"
              "        <d>0,1,0</d>\n"
              "      </mat_axis>\n")
MAT_AXIS_Z = ('      <mat_axis type="vector">\n'
              "        <a>0,0,1</a>\n"
              "        <d>1,0,0</d>\n"
              "      </mat_axis>\n")
import re


def final_stress(run):
    blocks = L.parse_log_csv(run.files.get("s.csv") or "")
    return blocks[-1][1][1][0] if blocks and 1 in blocks[-1][1] else None


def run_case(k1: str, stretch: str):
    base = L.template("fiber_reinforced_3d_hgo")
    deck = L.swap(base, MAT_AXIS_X, MAT_AXIS_Z)
    deck = L.swap(deck, "<k1>1.0</k1>", f"<k1>{k1}</k1>")
    if '<value lc="1">0.3</value>' not in deck:
        L.die("the HGO template's prescribed displacement changed")
    deck = deck.replace('<value lc="1">0.3</value>',
                        f'<value lc="1">{stretch}</value>')
    deck = L.add_logfile(deck, ("element_data", "sz", "s.csv"))
    r = L.run(deck, collect=("s.csv",), timeout=900)
    return r, final_stress(r)


def main() -> int:
    results = {}
    for k1 in ("1.0", "10.0"):
        for stretch in ("0.1", "-0.1"):
            r, sz = run_case(k1, stretch)
            results[(k1, stretch)] = (r, sz)
            print(f"k1={k1} stretch={stretch}: rc={r.rc} "
                  f"normal={int(r.normal_termination)} "
                  f"steps={r.steps_completed} final_sz={sz}")
    if any(sz is None for _r, sz in results.values()):
        print("FAIL: a run logged no element stress")
        return L.report(False, "fibers_tension_only", "reproduced",
                        "not_reproduced")
    c1 = results[("1.0", "-0.1")][1]
    c10 = results[("10.0", "-0.1")][1]
    t1 = results[("1.0", "0.1")][1]
    t10 = results[("10.0", "0.1")][1]
    comp_dev = abs(c1 - c10) / max(abs(c1), abs(c10), 1e-30)
    tens_dev = abs(t1 - t10) / max(abs(t1), abs(t10), 1e-30)
    print(f"compression_invariant_to_k1={int(comp_dev < 1e-9)} "
          f"rel_change={comp_dev:.3e}")
    print(f"tension_responds_to_k1={int(tens_dev > 1e-4)} "
          f"rel_change={tens_dev:.3e}")
    print(f"tension_exceeds_compression_in_magnitude="
          f"{int(abs(t10) > abs(c10))}")
    all_clean = all(r.rc == 0 and r.normal_termination
                    for r, _s in results.values())
    good = (all_clean and comp_dev < 1e-9 and tens_dev > 1e-4
            and abs(t10) > abs(c10))
    return L.report(good, "fibers_tension_only", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
