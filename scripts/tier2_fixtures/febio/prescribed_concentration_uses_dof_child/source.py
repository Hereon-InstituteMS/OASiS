"""Tier-2: a `prescribed concentration` BC selects its species with
<dof>c1</dof>, not with <sol>.

Verifies febio::multiphasic#3. Both mistakes are parse-time and both name
the offending tag, and the fixture requires each message absent from the
other run so the two are genuinely distinguished.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

def main() -> int:
    right = L.template("multiphasic_3d_diffusion")
    with_sol = L.swap(right,
                      "      <dof>c1</dof>\n"
                      '      <value lc="1">1.0</value>',
                      "      <sol>1</sol>\n"
                      '      <value lc="1">1.0</value>')
    undeclared = L.swap(right, "<dof>c1</dof>", "<dof>c9</dof>", count=1)
    ws = L.run(with_sol)
    wu = L.run(undeclared)
    r = L.run(right)
    s_msg = 'tag "sol"' in ws.text and "unrecognized tag" in ws.text
    u_msg = 'tag "dof"' in wu.text and "invalid value: c9" in wu.text
    print(f"sol_child: rc={ws.rc} read_failed={int(ws.read_failed)} "
          f"unrecognized_tag={int(s_msg)} "
          f"not_the_dof_message={int('invalid value: c' not in ws.text)}")
    print(f"undeclared_species: rc={wu.rc} "
          f"read_failed={int(wu.read_failed)} invalid_value={int(u_msg)} "
          f"not_an_unrecognized_tag={int('unrecognized tag' not in wu.text)}")
    print(f"right: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    good = (s_msg and u_msg and "invalid value: c" not in ws.text
            and "unrecognized tag" not in wu.text
            and ws.read_failed and wu.read_failed
            and ws.rc != 0 and wu.rc != 0
            and r.rc == 0 and r.normal_termination)
    return L.report(good, "concentration_dof_child", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
