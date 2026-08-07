"""Tier-2: with the module left as solid, a biphasic deck fails on the
ANALYSIS WORD.

Verifies febio::biphasic#0. Each module installs its own analysis
vocabulary, and the reader reaches <Control> before <Material>, so the
first thing rejected is `STEADY-STATE` — a word that is perfectly
correct. The message names the analysis word, which reads as if that word
were wrong when the MODULE is.

The fixture pins the misdirection: the message names "analysis" and
never mentions Module or biphasic.
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


def main() -> int:
    right = L.template("biphasic_3d_confined")
    wrong = L.swap(right, '<Module type="biphasic"/>',
                   '<Module type="solid"/>')
    w = L.run(wrong)
    r = L.run(right)
    msg = ('tag "analysis"' in w.text
           and "invalid value: STEADY-STATE" in w.text)
    silent_about_module = not any(
        t in w.text for t in ("Module", "biphasic module", "module type"))
    print(f"wrong: rc={w.rc} read_failed={int(w.read_failed)} "
          f"names_analysis_word={int(msg)} "
          f"never_names_the_module={int(silent_about_module)}")
    print(f"right: rc={r.rc} read_success={int(r.read_success)} "
          f"normal={int(r.normal_termination)} steps={r.steps_completed}")
    good = (msg and silent_about_module and w.read_failed and w.rc != 0
            and r.rc == 0 and r.normal_termination
            and 'tag "analysis"' not in r.text)
    if not good:
        print(w.text[:1000])
    return L.report(good, "module_gates_analysis", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
