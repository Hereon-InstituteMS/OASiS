"""Tier-2: reversing one contact surface disables the contact without
saying so.

Verifies febio::rigid_body#4. Contact facets must wind so the surface
normal points away from their own body. The fixture reverses the node
order of every quad4 on the block's top surface and compares against the
unmodified deck:

  * correct winding — a small residual penetration,
  * one surface reversed — the FULL contact travel comes back as
    interpenetration, exactly as if no <Contact> section existed, and the
    run is identical to a correct one in every log line.

The fixture asserts normal termination, the same completed-step count,
no error, and that the reversed run emits NO warning the correct run does
not — which is the honest form of the claim.

A PRECISION ON THE CLAIM, which says the reversed run is "identical to a
correct run in every log line". It is not quite: the CORRECT run
additionally warns `Problem is diverging. Stiffness matrix will now be
reformed`, because it is actually resolving contact. The difference
points the wrong way — the broken deck is the QUIETER of the two — so
nothing in the log flags the disabled contact, which is what makes the
pitfall real. The fixture pins the subset relation rather than equality.
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


import re

BLOCK_TOP_Z, PUNCH_BOTTOM_Z = 0.5, 0.55
GAP = PUNCH_BOTTOM_Z - BLOCK_TOP_Z           # 0.05
DRIVE = 0.15                                  # prescribed rigid displacement
TRAVEL = DRIVE - GAP                           # contact travel once closed


def geometry(deck: str):
    """Node ids of the punch's bottom face and of the block-top nodes
    that lie UNDER its footprint.

    The footprint filter is not cosmetic. Taking the extreme block-top z
    over ALL top nodes measures the untouched rim of the block, which
    does not move in any variant — an earlier draft of this fixture did
    exactly that and reported the working contact as inactive.
    """
    coords = {int(m.group(1)): tuple(float(x) for x in m.group(2).split(","))
              for m in re.finditer(r'<node id="(\d+)">([^<]+)</node>', deck)}
    punch = [k for k, c in coords.items()
             if abs(c[2] - PUNCH_BOTTOM_Z) < 1e-9]
    if not punch:
        L.die("no punch-bottom nodes found; the contact template changed")
    x0 = min(coords[k][0] for k in punch)
    x1 = max(coords[k][0] for k in punch)
    y0 = min(coords[k][1] for k in punch)
    y1 = max(coords[k][1] for k in punch)
    under = [k for k, c in coords.items()
             if abs(c[2] - BLOCK_TOP_Z) < 1e-9
             and x0 - 1e-9 <= c[0] <= x1 + 1e-9
             and y0 - 1e-9 <= c[1] <= y1 + 1e-9]
    if not under:
        L.die("no block-top nodes under the punch footprint")
    return punch, under


def penetration_fraction(run, punch, under):
    """(punch bottom - block top under it) / contact travel, at t_end.

    1.0 means the whole prescribed indentation became interpenetration —
    i.e. the contact did nothing.
    """
    blocks = L.parse_log_csv(run.files.get("pos.txt") or "")
    if not blocks:
        return None
    last = blocks[-1][1]
    top = sum(last[k][2] for k in under if k in last) / len(under)
    bottom = min(last[k][2] for k in punch if k in last)
    return (top - bottom) / TRAVEL


def main() -> int:
    base = L.template("rigid_contact_3d_indentation")
    punch, under = geometry(base)
    i = base.find('<Surface name="BlockTop">')
    j = base.find("</Surface>", i)
    if i < 0 or j <= i:
        L.die("the contact template has no BlockTop <Surface>")
    surface = base[i:j]
    reversed_surface = re.sub(
        r'<quad4 id="(\d+)">([\d,]+)</quad4>',
        lambda m: f'<quad4 id="{m.group(1)}">'
        + ",".join(reversed(m.group(2).split(","))) + "</quad4>", surface)
    if reversed_surface == surface:
        L.die("facet reversal did not apply; nothing was triggered")

    good_run = L.run(base, collect=("pos.txt",), timeout=1200)
    bad_run = L.run(L.swap(base, surface, reversed_surface),
                    collect=("pos.txt",), timeout=1200)
    fg = penetration_fraction(good_run, punch, under)
    fb = penetration_fraction(bad_run, punch, under)
    print(f"correct_winding: rc={good_run.rc} "
          f"normal={int(good_run.normal_termination)} "
          f"steps={good_run.steps_completed} "
          f"penetration_fraction_of_travel={fg:.4f}")
    print(f"block_top_reversed: rc={bad_run.rc} "
          f"normal={int(bad_run.normal_termination)} "
          f"steps={bad_run.steps_completed} "
          f"penetration_fraction_of_travel={fb:.4f} "
          f"warnings={int('WARNING' in bad_run.text)} "
          f"errors={int('ERROR' in bad_run.text)}")
    # "Identical in every log line" means the DIAGNOSTICS match: same
    # exit status, same completed-step count, no error, and the same set
    # of distinct warning texts. Both decks declare contact, so both
    # carry the same stiffness-symmetry note.
    def warnings(run):
        lines = run.text.splitlines()
        return {lines[i + 1].strip(" *") for i, l in enumerate(lines)
                if "WARNING" in l and i + 1 < len(lines)}

    wg, wb = warnings(good_run), warnings(bad_run)
    # The difference that does exist points the WRONG WAY: the reversed
    # deck is QUIETER. Nothing appears in it that is not also in the
    # correct run, so no message flags the disabled contact.
    nothing_extra = wb <= wg
    indistinguishable = (bad_run.rc == good_run.rc == 0
                         and bad_run.normal_termination
                         and bad_run.steps_completed
                         == good_run.steps_completed
                         and "ERROR" not in bad_run.text
                         and nothing_extra)
    print(f"reversed_run_says_nothing_extra={int(nothing_extra)} "
          f"only_in_correct={sorted(wg - wb)} "
          f"only_in_reversed={sorted(wb - wg)}")
    print(f"log_cannot_flag_the_problem={int(indistinguishable)}")
    good = (fg is not None and fb is not None
            and fg < 0.2 and fb > 0.99 and indistinguishable)
    return L.report(good, "facet_winding", "reproduced", "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
