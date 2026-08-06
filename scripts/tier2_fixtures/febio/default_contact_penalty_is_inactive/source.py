"""Tier-2: a <Contact> section with the default penalty is silently
inactive.

Verifies febio::rigid_body#3. FEBio's <penalty> default is 1 and is NOT
scaled by the material stiffness, so on a block with E = 1000 it is
orders of magnitude too soft. Four variants of the same deck:

  * penalty 1, auto_penalty off — nearly all of the contact travel is
    absorbed as interpenetration,
  * penalty 10000 (ten times E), auto_penalty off — better, but still
    far from closed,
  * the shipped auto_penalty=1 with a multiplier of 10 — small residual
    penetration,
  * and, for reference, no contact at all.

Every one of them ends in normal termination with exit 0 and all steps
completed. There is nothing in the log that separates working contact
from inactive contact, which is the claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

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


AUTO = ("<penalty>10.0</penalty>\n"
        "      <auto_penalty>1</auto_penalty>")


def main() -> int:
    base = L.template("rigid_contact_3d_indentation")
    punch, under = geometry(base)
    i = base.find("  <Contact>")
    j = base.find("</Contact>") + len("</Contact>\n")
    contact = base[i:j]

    variants = {
        "auto_penalty_10_shipped": base,
        "penalty_1_no_auto": L.swap(
            base, AUTO,
            "<penalty>1.0</penalty>\n      <auto_penalty>0</auto_penalty>"),
        "penalty_10000_no_auto": L.swap(
            base, AUTO,
            "<penalty>10000.0</penalty>\n"
            "      <auto_penalty>0</auto_penalty>"),
        "no_contact_at_all": L.drop(base, contact),
    }
    fractions = {}
    all_normal = True
    for tag, deck in variants.items():
        r = L.run(deck, collect=("pos.txt",), timeout=1200)
        f = penetration_fraction(r, punch, under)
        fractions[tag] = f
        all_normal = all_normal and r.rc == 0 and r.normal_termination
        print(f"{tag}: rc={r.rc} normal={int(r.normal_termination)} "
              f"steps={r.steps_completed} "
              f"penetration_fraction_of_travel="
              f"{'None' if f is None else f'{f:.4f}'} "
              f"warnings={int('WARNING' in r.text)}")
    ok = all(f is not None for f in fractions.values())
    if ok:
        soft = fractions["penalty_1_no_auto"] > 0.99
        stiffer = (fractions["penalty_10000_no_auto"]
                   < fractions["penalty_1_no_auto"])
        auto_best = (fractions["auto_penalty_10_shipped"]
                     < fractions["penalty_10000_no_auto"])
        indistinguishable = (fractions["penalty_1_no_auto"]
                             > 0.9 * fractions["no_contact_at_all"])
        print(f"default_penalty_absorbs_almost_all_travel={int(soft)} "
              f"raising_it_helps={int(stiffer)} "
              f"auto_penalty_is_best={int(auto_best)} "
              f"penalty_1_is_as_bad_as_no_contact={int(indistinguishable)}")
    good = ok and all_normal and soft and stiffer and auto_best \
        and indistinguishable
    return L.report(good, "contact_penalty", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
