"""Tier-2: <Globals><Solutes> is what creates c1, and its absence is
reported at the BC.

Verifies febio::multiphasic#0. Without the Solutes block the
concentration DOF does not exist, and the failure surfaces on the first
thing that REFERENCES one — so the line number points at the boundary
condition, not at the missing Globals block.

Two variants: the block deleted, and a solute index that was never
declared (c9). Both give the same message shape with a different suffix,
which is the claim.
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


SOLUTES = ("    <Solutes>\n"
           '      <solute id="1" name="Na">\n'
           "        <charge_number>0</charge_number>\n"
           "        <molar_mass>22.99</molar_mass>\n"
           "        <density>1.0</density>\n"
           "      </solute>\n"
           "    </Solutes>\n")


def main() -> int:
    right = L.template("multiphasic_3d_diffusion")
    no_solutes = L.drop(right, SOLUTES)
    undeclared = L.swap(right, "<dof>c1</dof>", "<dof>c9</dof>", count=1)
    wn = L.run(no_solutes)
    wu = L.run(undeclared)
    r = L.run(right)
    n_msg = ('tag "dof"' in wn.text and "invalid value: c1" in wn.text)
    u_msg = ('tag "dof"' in wu.text and "invalid value: c9" in wu.text)
    misdirected = "Solutes" not in wn.text and "Globals" not in wn.text
    print(f"no_solutes: rc={wn.rc} read_failed={int(wn.read_failed)} "
          f"names_the_dof={int(n_msg)} "
          f"never_names_Globals_or_Solutes={int(misdirected)}")
    print(f"undeclared_c9: rc={wu.rc} read_failed={int(wu.read_failed)} "
          f"names_the_dof={int(u_msg)}")
    print(f"right: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    good = (n_msg and u_msg and misdirected
            and wn.read_failed and wu.read_failed
            and wn.rc != 0 and wu.rc != 0
            and r.rc == 0 and r.normal_termination
            and 'tag "dof"' not in r.text)
    return L.report(good, "globals_solutes", "reproduced", "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
