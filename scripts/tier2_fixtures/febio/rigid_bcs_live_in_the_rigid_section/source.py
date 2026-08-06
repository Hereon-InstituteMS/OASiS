"""Tier-2: rigid-body BCs go in <Rigid>, and the two ways of getting it
wrong fail at DIFFERENT stages.

Verifies febio::rigid_body#0:

  * a <bc type="prescribed displacement"> on nodes that belong to a rigid
    material READS cleanly and then dies at model initialisation with
    `Rigid nodes cannot be prescribed.` and
    `Boundary condition N (push) failed to initialize`,
  * a <rigid_bc> placed inside <Boundary> is caught at parse with
    `tag "rigid_bc" (line N) : unrecognized tag`.

The first is the dangerous one: a wrapper that stops at the reader line
believes the deck is fine.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

import re


def main() -> int:
    base = L.template("rigid_body_3d_pushdown")
    i, j = base.find("  <Rigid>"), base.find("</Rigid>") + len("</Rigid>\n")
    if i < 0 or j <= i:
        L.die("the rigid template no longer has a <Rigid> section")
    inner = (base[i:j].replace("  <Rigid>\n", "")
             .replace("  </Rigid>\n", ""))
    no_rigid = base[:i] + base[j:]

    # (a) rigid_bc moved into <Boundary>
    in_boundary = L.swap(no_rigid, "  </Boundary>\n",
                         inner + "  </Boundary>\n")
    wb = L.run(in_boundary, timeout=400)
    parse_ok = ('tag "rigid_bc"' in wb.text
                and "unrecognized tag" in wb.text)

    # (b) a nodal prescribed displacement on the rigid body's own nodes
    block = re.search(r'<Elements type="hex8" name="ImpactorPart">(.*?)'
                      r"</Elements>", base, re.S)
    if block is None:
        L.die("cannot find the rigid part's elements in the template")
    ids = sorted({int(x) for x in re.findall(
        r"\d+", re.sub(r'elem id="\d+"', "", block.group(1)))})
    nodal = L.swap(
        no_rigid, "  </Boundary>\n",
        '    <bc name="push" type="prescribed displacement" '
        'node_set="rigid_nodes"><dof>z</dof>'
        '<value lc="1">-0.1</value></bc>\n  </Boundary>\n')
    nodal = L.swap(nodal, '<NodeSet name="base">',
                   f'<NodeSet name="rigid_nodes">'
                   f'{",".join(str(v) for v in ids[:8])}</NodeSet>\n'
                   f'    <NodeSet name="base">')
    wn = L.run(nodal, timeout=400)
    init_ok = (wn.read_success and not wn.read_failed
               and wn.has("Rigid nodes cannot be prescribed.")
               and wn.has("failed to initialize")
               and wn.has("Model initialization failed"))

    r = L.run(base, timeout=400)
    print(f"rigid_bc_in_Boundary: rc={wb.rc} "
          f"read_failed={int(wb.read_failed)} parse_error={int(parse_ok)}")
    print(f"nodal_bc_on_rigid_nodes: rc={wn.rc} "
          f"read_success={int(wn.read_success)} "
          f"rigid_nodes_message={int(init_ok)}")
    print(f"control: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    good = (parse_ok and wb.read_failed and wb.rc != 0
            and init_ok and wn.rc != 0
            and r.rc == 0 and r.normal_termination)
    if not good:
        print(wn.text[:1000])
    return L.report(good, "rigid_bc_section", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
