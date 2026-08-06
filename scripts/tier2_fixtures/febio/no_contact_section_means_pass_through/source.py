"""Tier-2: without a <Contact> section the rigid punch passes straight
through the deformable block, and the run reports success.

Verifies febio::rigid_body#2. The shipped indentation template is run
twice, identical but for the <Contact> section:

  * with it, the block surface under the punch is driven down and only a
    small fraction of the contact travel remains as interpenetration,
  * without it, the block surface does not move AT ALL and the entire
    travel becomes interpenetration.

Both reach `N O R M A L  T E R M I N A T I O N` with exit 0 and every
step completed, and nothing in the no-contact run's output mentions
contact, an interface, a surface pair or penetration at all — the fixture
asserts that absence by word. (Both runs do warn about other things, so
"no warnings" would be the wrong test.) The only detection is to read the
geometry back out, which is what this fixture does.
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


def main() -> int:
    base = L.template("rigid_contact_3d_indentation")
    i = base.find("  <Contact>")
    j = base.find("</Contact>") + len("</Contact>\n")
    if i < 0 or j <= i:
        L.die("the contact template has no <Contact> section")
    contact = base[i:j]
    punch, under = geometry(base)
    print(f"punch_bottom_nodes={len(punch)} "
          f"block_top_nodes_under_the_punch={len(under)}")

    with_c = L.run(base, collect=("pos.txt",), timeout=1200)
    without = L.run(L.drop(base, contact), collect=("pos.txt",),
                    timeout=1200)
    fw = penetration_fraction(with_c, punch, under)
    fo = penetration_fraction(without, punch, under)
    print(f"with_contact: rc={with_c.rc} "
          f"normal={int(with_c.normal_termination)} "
          f"steps={with_c.steps_completed} "
          f"penetration_fraction_of_travel={fw:.4f}")
    print(f"no_contact: rc={without.rc} "
          f"normal={int(without.normal_termination)} "
          f"steps={without.steps_completed} "
          f"penetration_fraction_of_travel={fo:.4f}")
    # Both runs warn about other things (an unloaded first step, and,
    # where contact IS declared, a note about the stiffness symmetry).
    # What must be absent from the no-contact run is any mention of the
    # interface it is missing.
    quiet_words = ("contact", "interface", "penetrat", "surface pair")
    mentions = [w for w in quiet_words if w in without.text.lower()]
    silent = (without.rc == 0 and without.normal_termination
              and without.steps_completed == with_c.steps_completed
              and "ERROR" not in without.text
              and not mentions)
    print(f"missing_interface_is_never_mentioned={int(silent)} "
          f"words_found={mentions}")
    good = (fw is not None and fo is not None
            and fw < 0.2 and fo > 0.99 and silent
            and with_c.rc == 0 and with_c.normal_termination)
    return L.report(good, "missing_contact", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
