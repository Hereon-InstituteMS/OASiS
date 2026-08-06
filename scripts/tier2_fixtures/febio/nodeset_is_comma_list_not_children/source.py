"""Tier-2: a <NodeSet> carries a comma-separated text list, not <n id=/>
children.

Verifies febio::heat#8. The message is
`tag "NodeSet" (line N) : invalid value:` with NOTHING after the colon,
because the element's text content is empty — the fixture asserts the
empty tail explicitly, since a reader expecting the offending value to be
quoted there will conclude the message is truncated.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

import re


def main() -> int:
    right = L.template("heat_3d_bar", n=2)
    wrong = L.swap(
        right, '<NodeSet name="cold_face">1,4,7,10</NodeSet>',
        '<NodeSet name="cold_face"><n id="1"/><n id="4"/>'
        '<n id="7"/><n id="10"/></NodeSet>')
    w = L.run(wrong)
    r = L.run(right)
    m = re.search(r'tag "NodeSet" \(line \d+\) : invalid value:(.*)', w.text)
    empty_tail = bool(m) and m.group(1).strip(" *") == ""
    print(f"wrong: rc={w.rc} read_failed={int(w.read_failed)} "
          f"message={int(bool(m))} empty_value_after_colon={int(empty_tail)}")
    print(f"right: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    good = (bool(m) and empty_tail and w.read_failed and w.rc != 0
            and r.rc == 0 and r.normal_termination
            and 'tag "NodeSet"' not in r.text)
    if not good:
        print(w.text[:1000])
    return L.report(good, "nodeset_child_elements", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
