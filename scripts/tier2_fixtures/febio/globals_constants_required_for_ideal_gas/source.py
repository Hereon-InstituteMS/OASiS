"""Tier-2: `ideal gas` reads R, T (and optionally P) out of
<Globals><Constants>.

Verifies febio::heat#6, all four branches of it in one run:

  * no <Globals> block at all,
  * <R>0</R>,
  * <T>0</T>,
  * <P> omitted — which is NOT an error.

The first three read `...SUCCESS!` and only then fail, each with its own
message naming the constant. The fourth runs to normal termination and
prints a WARNING saying FEBio computed P itself. Pinning the fourth is
what stops a wrapper treating the whole block as mandatory.
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


GLOBALS = ("  <Globals>\n"
           "    <Constants>\n"
           "      <R>8.31446</R>\n"
           "      <T>300.0</T>\n"
           "      <P>101325</P>\n"
           "    </Constants>\n"
           "  </Globals>\n")
R_MSG = "A positive universal gas constant R must be defined in Globals"
T_MSG = ("A positive referential absolute temperature T must be defined "
         "in Globals")
P_WARN = "The referential absolute pressure P is calculated internally as"


def main() -> int:
    right = L.template("heat_3d_bar", n=2)
    r = L.run(right)
    variants = {
        "no_globals": (L.drop(right, GLOBALS), R_MSG),
        "R_zero": (L.swap(right, "<R>8.31446</R>", "<R>0</R>"), R_MSG),
        "T_zero": (L.swap(right, "<T>300.0</T>", "<T>0</T>"), T_MSG),
    }
    ok = True
    for name, (deck, msg) in variants.items():
        w = L.run(deck)
        hit = w.has(msg)
        # These fail AFTER the reader says SUCCESS.
        print(f"{name}: rc={w.rc} read_success={int(w.read_success)} "
              f"read_failed={int(w.read_failed)} names_constant={int(hit)}")
        ok = ok and hit and w.read_success and w.rc != 0
    no_p = L.run(L.drop(right, "      <P>101325</P>\n"))
    p_ok = (no_p.rc == 0 and no_p.normal_termination and no_p.has(P_WARN))
    print(f"P_omitted: rc={no_p.rc} normal={int(no_p.normal_termination)} "
          f"steps={no_p.steps_completed} warning={int(no_p.has(P_WARN))}")
    print(f"control: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    good = (ok and p_ok and r.rc == 0 and r.normal_termination
            and not r.has(R_MSG) and not r.has(T_MSG))
    return L.report(good, "globals_constants", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
