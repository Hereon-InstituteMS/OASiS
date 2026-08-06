"""Tier-2: the rigid DOF spelling differs per rigid_bc type.

Verifies febio::rigid_body#1. rigid_displacement takes lowercase x/y/z;
rigid_rotation takes capital-R Ru/Rv/Rw. Each wrong spelling is rejected
by VALUE, naming the offending string, so the message tells you exactly
what you wrote — the fixture requires the quoted value to be the one it
supplied, not merely that an error occurred.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

def main() -> int:
    base = L.template("rigid_body_3d_pushdown")
    r = L.run(base, timeout=400)

    cap = L.run(L.swap(base, "      <dof>z</dof>\n",
                       "      <dof>Rz</dof>\n"), timeout=400)
    cap_ok = ('tag "dof"' in cap.text and "invalid value: Rz" in cap.text)

    rot = L.swap(
        base,
        '    <rigid_bc name="impactor_push" type="rigid_displacement">\n'
        "      <rb>2</rb>\n      <dof>z</dof>\n",
        '    <rigid_bc name="impactor_push" type="rigid_rotation">\n'
        "      <rb>2</rb>\n      <dof>u</dof>\n")
    wr = L.run(rot, timeout=400)
    rot_ok = ('tag "dof"' in wr.text and "invalid value: u" in wr.text)

    print(f"capital_R_on_translation: rc={cap.rc} "
          f"read_failed={int(cap.read_failed)} quotes_Rz={int(cap_ok)}")
    print(f"lowercase_on_rotation: rc={wr.rc} "
          f"read_failed={int(wr.read_failed)} quotes_u={int(rot_ok)}")
    print(f"control: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    good = (cap_ok and rot_ok and cap.read_failed and wr.read_failed
            and cap.rc != 0 and wr.rc != 0
            and r.rc == 0 and r.normal_termination)
    return L.report(good, "rigid_dof_spelling", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
