"""Tier-2: STATIC is rejected in the biphasic module, and a raw ordinal
is quietly accepted.

Verifies febio::biphasic#6. Two halves, and the second is the dangerous
one:

  * `<analysis>STATIC</analysis>` in a biphasic deck is a clean parse
    error naming the word — not a silent fallback to ordinal 0,
  * `<analysis>1</analysis>` is ACCEPTED and selects TRANSIENT, so a
    deck carrying a number instead of a word runs and gives a different
    analysis from the one its author would read off the file.

The fixture asserts the ordinal run terminates normally AND differs from
the STEADY-STATE run it replaced, which is what makes "quietly selects
TRANSIENT" an observation rather than an inference.
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


OUT = ("  <Output>\n    <logfile>\n"
       '      <element_data data="sz;p" delim="," file="e.csv"/>\n'
       "    </logfile>\n  </Output>\n")


def with_log(deck: str) -> str:
    return L.swap(deck, "</febio_spec>", OUT + "</febio_spec>")


def last(run):
    b = L.parse_log_csv(run.files.get("e.csv") or "")
    return b[-1][1].get(1) if b else None


def main() -> int:
    right = with_log(L.template("biphasic_3d_confined"))
    static = L.swap(right, "<analysis>STEADY-STATE</analysis>",
                    "<analysis>STATIC</analysis>")
    ordinal = L.swap(right, "<analysis>STEADY-STATE</analysis>",
                     "<analysis>1</analysis>")
    transient = L.swap(right, "<analysis>STEADY-STATE</analysis>",
                       "<analysis>TRANSIENT</analysis>")

    w = L.run(static)
    r = L.run(right, collect=("e.csv",))
    o = L.run(ordinal, collect=("e.csv",))
    t = L.run(transient, collect=("e.csv",))

    rejected = ('tag "analysis"' in w.text
                and "invalid value: STATIC" in w.text
                and w.read_failed and w.rc != 0)
    print(f"STATIC_in_biphasic: rc={w.rc} read_failed={int(w.read_failed)} "
          f"names_the_word={int(rejected)}")
    print(f"STEADY-STATE: rc={r.rc} normal={int(r.normal_termination)} "
          f"last={last(r)}")
    print(f"ordinal_1: rc={o.rc} normal={int(o.normal_termination)} "
          f"last={last(o)}")
    print(f"TRANSIENT: rc={t.rc} normal={int(t.normal_termination)} "
          f"last={last(t)}")
    ordinal_accepted = o.rc == 0 and o.normal_termination
    is_transient = (last(o) is not None and last(o) == last(t))
    differs_from_steady = last(o) != last(r)
    print(f"ordinal_accepted={int(ordinal_accepted)} "
          f"ordinal_1_equals_TRANSIENT={int(is_transient)} "
          f"differs_from_STEADY_STATE={int(differs_from_steady)}")
    good = (rejected and ordinal_accepted and is_transient
            and differs_from_steady and r.rc == 0 and t.rc == 0)
    return L.report(good, "analysis_enum_scope", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
