"""Tier-2: a flat parameter and a missing nested property are
DISTINGUISHABLE from the message alone.

Verifies febio::biphasic#1. The `biphasic` material takes nested <solid>
and <permeability> properties, each with its own type=. The pitfall's
useful content is that the two mistakes report differently:

  * a flat parameter is named back verbatim —
    `tag "E" (line N) : unrecognized tag`,
  * a missing nested PROPERTY is reported as
    `Component "Material1" needs to have property "permeability" defined`,
    quoting the material's own name.

The fixture requires BOTH messages and requires each to be ABSENT from
the other run, because it is the discrimination that is being claimed,
not either message on its own.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

PERM = ('      <permeability type="perm-const-iso">\n'
        "        <perm>0.001</perm>\n"
        "      </permeability>\n")


def main() -> int:
    right = L.template("biphasic_3d_confined")
    flat = L.swap(right, "      <phi0>0.2</phi0>\n",
                  "      <phi0>0.2</phi0>\n      <E>1000.0</E>\n")
    missing = L.drop(right, PERM)

    wf = L.run(flat)
    wm = L.run(missing)
    r = L.run(right)

    flat_msg = 'tag "E"' in wf.text and "unrecognized tag" in wf.text
    miss_msg = wm.has('Component "Material1" needs to have property '
                      '"permeability" defined')
    print(f"flat_parameter: rc={wf.rc} read_failed={int(wf.read_failed)} "
          f"names_the_tag={int(flat_msg)} "
          f"not_a_missing_property_message="
          f"{int('needs to have property' not in wf.text)}")
    print(f"missing_property: rc={wm.rc} read_failed={int(wm.read_failed)} "
          f"names_the_property={int(miss_msg)} "
          f"not_an_unrecognized_tag_message="
          f"{int('unrecognized tag' not in wm.text)}")
    print(f"right: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    good = (flat_msg and "needs to have property" not in wf.text
            and miss_msg and "unrecognized tag" not in wm.text
            and wf.read_failed and wm.read_failed
            and wf.rc != 0 and wm.rc != 0
            and r.rc == 0 and r.normal_termination)
    return L.report(good, "biphasic_flat_vs_missing", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
