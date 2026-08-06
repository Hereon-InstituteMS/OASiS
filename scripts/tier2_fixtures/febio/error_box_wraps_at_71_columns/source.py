"""Tier-2: FEBio's ERROR box wraps at 71 columns, so whole-sentence greps miss.

Verifies febio::linear_elasticity#9. This is a pitfall about the CAPTURE
of a message rather than about the deck, so the fixture asserts a
negative: the exact sentence the solver printed cannot be found in the
captured text as a fixed string, because the box broke the line after
the word "matrix".

Reproduced by asking the default skyline linear solver for a format it
does not have — <symmetric_stiffness>non-symmetric</symmetric_stiffness>
on a solid deck. The fixture requires:

  * the whole sentence is ABSENT from the raw captured text,
  * the short fragment "does not support the requested" is PRESENT,
  * the box lines appear in the order the pitfall quotes,
  * and the neighbouring mistake is separated: `unsymmetric`, which
    looks plausible, is rejected at parse with
    `tag "symmetric_stiffness" (line N) : invalid value: unsymmetric`.

If FEBio ever stops wrapping, the first assertion fails and the pitfall
needs rewriting — which is the correct outcome, not a false alarm.

MUTATION CONTROL. T2_MUTATE=1 tests the whole-sentence grep against the
UNWRAPPED capture — the world in which FEBio's ERROR box does not break
the line after "matrix". The sentence is then findable as a fixed
string, whole_sentence_absent_from_raw_text goes 0, and
'error_box_wrap=reproduced' is no longer printed.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

WHOLE = ("The selected linear solver does not support the requested "
         "matrix format.")
FRAGMENT = "does not support the requested"


def control(sym: str) -> str:
    return ("  <Control>\n"
            "    <analysis>STATIC</analysis>\n"
            "    <time_steps>2</time_steps>\n"
            "    <step_size>0.5</step_size>\n"
            '    <solver type="solid">\n'
            f"      <symmetric_stiffness>{sym}</symmetric_stiffness>\n"
            "    </solver>\n"
            "  </Control>")


def main() -> int:
    w = L.run(L.solid_deck(control=control("non-symmetric")))
    r = L.run(L.solid_deck(control=control("symmetric")))
    b = L.run(L.solid_deck(control=control("unsymmetric")))

    if MUTATE:
        print("mutation=the_whole_sentence_grep_runs_on_the_"
              "unwrapped_capture")
        _capture = re.sub(r"\s+", " ",
                          re.sub(r"\s*\*?\s*\n\s*\*?\s*", " ", w.text))
    else:
        _capture = w.text
    whole_absent = WHOLE not in _capture
    frag_present = FRAGMENT in w.text
    first_line = ("The selected linear solver does not support the "
                  "requested matrix") in w.text
    second_line = "format." in w.text
    third_line = "Please select a different linear solver." in w.text
    rejoined = w.has(WHOLE)      # only findable after unwrapping

    print(f"wrap: rc={w.rc} read_success={int(w.read_success)} "
          f"error_termination={int(w.error_termination)} "
          f"steps={w.steps_completed}")
    print(f"whole_sentence_absent_from_raw_text={int(whole_absent)} "
          f"fragment_present={int(frag_present)} "
          f"findable_after_unwrapping={int(rejoined)}")
    print(f"box_lines: first={int(first_line)} second={int(second_line)} "
          f"third={int(third_line)}")
    print(f"control_symmetric: rc={r.rc} "
          f"normal={int(r.normal_termination)} "
          f"fragment_absent={int(FRAGMENT not in r.text)}")
    bad_enum = ('tag "symmetric_stiffness"' in b.text
                and "invalid value: unsymmetric" in b.text)
    print(f"unsymmetric_spelling: rc={b.rc} "
          f"read_failed={int(b.read_failed)} rejected_by_name={int(bad_enum)}")

    good = (whole_absent and frag_present and rejoined
            and first_line and second_line and third_line
            and w.rc != 0 and w.read_success and w.error_termination
            and r.rc == 0 and r.normal_termination
            and FRAGMENT not in r.text
            and bad_enum and b.read_failed)
    if not good:
        print(w.text[:1200])
    return L.report(good, "error_box_wrap", "reproduced", "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
