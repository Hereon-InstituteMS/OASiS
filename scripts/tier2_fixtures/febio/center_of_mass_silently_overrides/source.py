"""Tier-2: <center_of_mass> silently overrides the computed centre, and
<override_com> changes nothing.

Verifies febio::rigid_body#6. A rigid body is rotated about x. Three
runs:

  * <center_of_mass> omitted — the body pivots about its own geometric
    centre,
  * <center_of_mass>0,0,5</center_of_mass> — the body pivots about that
    point instead and swings a long way,
  * the same plus <override_com>1</override_com> — BIT-IDENTICAL to the
    previous run, i.e. the companion flag is not needed to make the tag
    take effect and does not change anything.

All three end in normal termination with exit 0; the only detection is
reading node positions back out.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

MATERIAL = ('    <material id="2" name="Impactor" type="rigid body">\n'
            "      <density>10.0</density>\n"
            "    </material>")
ROTATION = ("  <Rigid>\n"
            '    <rigid_bc name="lock" type="rigid_fixed"><rb>2</rb>'
            "<Rx_dof>1</Rx_dof><Ry_dof>1</Ry_dof><Rz_dof>1</Rz_dof>"
            "<Rv_dof>1</Rv_dof><Rw_dof>1</Rw_dof></rigid_bc>\n"
            '    <rigid_bc name="spin" type="rigid_rotation"><rb>2</rb>'
            '<dof>Ru</dof><value lc="1">0.5</value></rigid_bc>\n'
            "  </Rigid>\n")


def positions(run):
    blocks = L.parse_log_csv(run.files.get("pos.txt") or "")
    if not blocks:
        return []
    last = blocks[-1][1]
    return [v for k in sorted(last) for v in last[k]]


def main() -> int:
    base = L.template("rigid_body_3d_pushdown")
    i = base.find("  <Rigid>")
    j = base.find("</Rigid>") + len("</Rigid>\n")
    spin = L.swap(base, base[i:j], ROTATION)

    variants = {
        "com_omitted": spin,
        "com_written": L.swap(
            spin, "<density>10.0</density>",
            "<density>10.0</density>"
            "<center_of_mass>0,0,5</center_of_mass>"),
        "com_plus_override_flag": L.swap(
            spin, "<density>10.0</density>",
            "<density>10.0</density>"
            "<center_of_mass>0,0,5</center_of_mass>"
            "<override_com>1</override_com>"),
    }
    data = {}
    for tag, deck in variants.items():
        r = L.run(deck, collect=("pos.txt",), timeout=1200)
        data[tag] = (r, positions(r))
        zs = [v for n, v in enumerate(data[tag][1]) if n % 3 == 2]
        print(f"{tag}: rc={r.rc} normal={int(r.normal_termination)} "
              f"steps={r.steps_completed} "
              f"z_range=[{min(zs):.6f},{max(zs):.6f}]" if zs else
              f"{tag}: rc={r.rc} NO POSITIONS")

    def dev(a, b):
        if not a or len(a) != len(b):
            return float("inf")
        return max(abs(x - y) / max(abs(x), abs(y), 1e-12)
                   for x, y in zip(a, b))

    omitted = data["com_omitted"][1]
    written = data["com_written"][1]
    flagged = data["com_plus_override_flag"][1]
    d_effect = dev(omitted, written)
    d_flag = dev(written, flagged)
    print(f"writing_center_of_mass_changes_the_result={int(d_effect > 0.1)} "
          f"max_rel_dev={d_effect:.3e}")
    print(f"override_com_changes_nothing={int(d_flag == 0.0)} "
          f"max_rel_dev={d_flag:.3e}")
    all_clean = all(r.rc == 0 and r.normal_termination
                    for r, _p in data.values())
    print(f"all_terminate_normally={int(all_clean)}")
    good = all_clean and d_effect > 0.1 and d_flag == 0.0
    return L.report(good, "center_of_mass", "reproduced", "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
