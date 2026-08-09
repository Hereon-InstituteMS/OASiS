"""Tier-2: a COUPLED elastic child of `uncoupled viscoelastic` is reported
as a MISSING property.

Verifies febio::viscoelasticity#1 — the message points at the wrong
problem, and the fixture's job is to prove that it CANNOT distinguish the
two cases. Three variants:

  * a coupled `neo-Hookean` child,
  * a coupled `isotropic elastic` child,
  * no <elastic> child at all,

all three giving the same `Component "Material1" needs to have property
"elastic" defined` message. The fixture compares the message text across
the three runs and requires equality — that is what turns "the message
points at the wrong problem" into an observation.

ONE PRECISION ON THE CLAIM, which says the three messages are
byte-identical. They are identical except for the parenthesised LINE
NUMBER, which necessarily moves when a block is deleted from the deck:
the two coupled-child variants agree byte for byte, the no-child variant
differs only in that number. The fixture therefore compares with the line
number stripped, and prints all three raw messages so the difference is
visible rather than hidden by the comparison.
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

MOONEY = ('      <elastic type="Mooney-Rivlin">\n'
          "        <density>1.0</density>\n"
          "        <c1>1.0</c1>\n"
          "        <c2>0.0</c2>\n"
          "        <k>1000.0</k>\n"
          "      </elastic>\n")


def msg(run) -> str | None:
    # The message is longer than 71 columns, so the ERROR box breaks it
    # across lines; match on the unwrapped text.
    m = re.search(r'Component "[^"]+" needs to have property '
                  r'"[^"]+" defined \(line \d+\)', run.flat)
    return None if m is None else m.group(0)


def without_line(message: str | None) -> str | None:
    if message is None:
        return None
    return re.sub(r"\(line \d+\)", "(line N)", message)


def main() -> int:
    base = L.template("viscoelasticity_3d_stress_relax")
    r = L.run(base, timeout=600)
    variants = {
        "coupled_neo_hookean": L.swap(
            base, MOONEY,
            '      <elastic type="neo-Hookean">\n'
            "        <density>1</density><E>1000</E><v>0.3</v>\n"
            "      </elastic>\n"),
        "coupled_isotropic_elastic": L.swap(
            base, MOONEY,
            '      <elastic type="isotropic elastic">\n'
            "        <density>1</density><E>1000</E><v>0.3</v>\n"
            "      </elastic>\n"),
        "no_child_at_all": L.drop(base, MOONEY),
    }
    messages = {}
    for name, deck in variants.items():
        w = L.run(deck)
        messages[name] = msg(w)
        print(f"{name}: rc={w.rc} read_failed={int(w.read_failed)} "
              f"message={messages[name]}")
        if not (w.read_failed and w.rc != 0 and messages[name]):
            messages[name] = None
    distinct = {without_line(m) for m in messages.values() if m}
    identical = (len(distinct) == 1 and len(messages) == 3
                 and None not in messages.values())
    raw = {m for m in messages.values() if m}
    print(f"all_three_give_the_same_message_modulo_line_number="
          f"{int(identical)}")
    print(f"distinct_raw_messages={len(raw)} "
          f"(the line number moves when a block is deleted)")
    print(f"control_mooney_rivlin: rc={r.rc} "
          f"normal={int(r.normal_termination)} steps={r.steps_completed}")
    good = (identical and r.rc == 0 and r.normal_termination
            and msg(r) is None)
    return L.report(good, "viscoelastic_child_kind", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
