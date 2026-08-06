"""Tier-2: <Control> without a nested <solver type=...> is rejected.

Verifies febio::linear_elasticity#2. The message is
`Component "" needs to have property "solver" defined (line N)` — the
empty quotes are not a bug, the FEAnalysis object has no name, and a
reader looking for a material or BC name in that slot will not find one.

The fixture also checks the type string is module-scoped: type="solid"
runs, and a wrong-module solver name in a solid deck is rejected as an
invalid attribute value rather than as a missing property, so the two
mistakes are distinguishable from the message alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

CTRL_NO_SOLVER = ("  <Control>\n"
                  "    <analysis>STATIC</analysis>\n"
                  "    <time_steps>2</time_steps>\n"
                  "    <step_size>0.5</step_size>\n"
                  "  </Control>")
CTRL_WRONG_TYPE = ("  <Control>\n"
                   "    <analysis>STATIC</analysis>\n"
                   "    <time_steps>2</time_steps>\n"
                   "    <step_size>0.5</step_size>\n"
                   '    <solver type="biphasic"/>\n'
                   "  </Control>")


def main() -> int:
    w = L.run(L.solid_deck(control=CTRL_NO_SOLVER))
    t = L.run(L.solid_deck(control=CTRL_WRONG_TYPE))
    r = L.run(L.solid_deck())

    msg = 'needs to have property "solver" defined' in w.text
    empty_name = 'Component ""' in w.text
    other = 'invalid value for attribute "type"' in t.text
    print(f"no_solver: rc={w.rc} read_failed={int(w.read_failed)} "
          f"message={int(msg)} empty_component_name={int(empty_name)}")
    print(f"out_of_module_solver: rc={t.rc} "
          f"read_failed={int(t.read_failed)} invalid_type={int(other)} "
          f"same_message_as_missing={int(msg and 'needs to have property' in t.text)}")
    print(f"right: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    good = (msg and empty_name and w.read_failed and w.rc != 0
            and other and t.read_failed
            and 'needs to have property "solver"' not in t.text
            and r.rc == 0 and r.normal_termination
            and 'needs to have property "solver"' not in r.text)
    if not good:
        print(w.text[:900])
        print(t.text[:900])
    return L.report(good, "control_no_solver", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
