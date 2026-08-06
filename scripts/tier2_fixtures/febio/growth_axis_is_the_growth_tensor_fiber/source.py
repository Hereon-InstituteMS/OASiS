"""Tier-2: `kinematic growth` does not accept <mat_axis>, and <MeshData>
placed too early is rejected by name.

Verifies the syntax half of febio::growth_remodeling#3. Three rejections:

  * <mat_axis> on the material — `unrecognized tag`,
  * the old <theta> spelling — likewise,
  * <MeshData> before </MeshDomains> —
    `MeshData must appear after MeshDomain section.`

The last one is the useful one: it is not a `tag "..."` message at all, so
a wrapper matching only that family will not recognise it.

NOT ASSERTED HERE: that the growth axis defaults silently to global x and
that `fiber growth` / `area growth` stretch different edges. That is a
geometric measurement over several growth-tensor types and belongs in a
numerical fixture; it is named as uncovered rather than claimed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

MESHDATA = ('  <MeshData>\n'
            '    <ElementData type="mat_axis" elem_set="allel">'
            '<e lid="1"><a>0,0,1</a><d>1,0,0</d></e></ElementData>\n'
            "  </MeshData>\n")
ANCHOR = "      <density>1.0</density>\n      <elastic"


def main() -> int:
    base = L.template("growth_remodeling_3d_isotropic")
    r = L.run(base, timeout=400)
    rejected = 0
    for tag, xml in (
            ("mat_axis",
             '      <mat_axis type="vector"><a>0,0,1</a>'
             "<d>1,0,0</d></mat_axis>\n"),
            ("theta", "      <theta>30</theta>\n")):
        w = L.run(L.swap(base, ANCHOR,
                         "      <density>1.0</density>\n" + xml
                         + "      <elastic"))
        hit = f'tag "{tag}"' in w.text and "unrecognized tag" in w.text
        print(f"{tag}_on_material: rc={w.rc} "
              f"read_failed={int(w.read_failed)} unrecognized_tag={int(hit)}")
        if hit and w.read_failed and w.rc != 0:
            rejected += 1

    early = L.run(L.swap(base, "  <MeshDomains>", MESHDATA + "  <MeshDomains>"))
    md_ok = early.has("MeshData must appear after MeshDomain section.")
    not_tag_family = 'tag "MeshData"' not in early.text
    print(f"meshdata_before_meshdomains: rc={early.rc} "
          f"read_failed={int(early.read_failed)} own_message={int(md_ok)} "
          f"not_a_tag_message={int(not_tag_family)}")
    print(f"control: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    print(f"rejected_by_name={rejected} of 2")
    good = (rejected == 2 and md_ok and not_tag_family
            and early.rc != 0 and r.rc == 0 and r.normal_termination)
    return L.report(good, "growth_axis_names", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
