"""Tier-2: <symmetric_stiffness>symmetric</symmetric_stiffness> changes the
Newton TANGENT, not the residual.

Verifies febio::biphasic#7. The claim is that symmetric + skyline is a
legitimate workaround when nothing else is available: it costs
convergence robustness, not accuracy.

Executed on a confined-compression deck with a strain-dependent
perm-Holmes-Mow permeability — where the biphasic tangent really IS
unsymmetric, so the workaround is being tested where it could fail:
symmetric+skyline and non-symmetric+bicgstab reach the same element
stress. The fixture requires agreement to a tight relative tolerance and
requires both runs to complete, so it fails if the workaround ever starts
changing the answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

HOLMES_MOW = ('      <permeability type="perm-Holmes-Mow">\n'
              "        <perm>0.001</perm><M>1.0</M><alpha>2.0</alpha>\n"
              "      </permeability>\n")
TOL = 1e-9


def deck(sym: str, solver: str) -> str:
    mesh, _info = L.hex8_box(1)
    material = (
        "  <Material>\n"
        '    <material id="1" name="Material1" type="biphasic">\n'
        "      <phi0>0.2</phi0>\n"
        '      <solid type="neo-Hookean"><density>1.0</density>'
        "<E>1000.0</E><v>0.0</v></solid>\n" + HOLMES_MOW
        + "    </material>\n  </Material>")
    control = (
        "  <Control>\n    <analysis>TRANSIENT</analysis>\n"
        "    <time_steps>4</time_steps>\n    <step_size>0.1</step_size>\n"
        '    <solver type="biphasic">\n'
        f"      <symmetric_stiffness>{sym}</symmetric_stiffness>\n"
        f"      {solver}\n    </solver>\n  </Control>")
    boundary = (
        "  <Boundary>\n"
        '    <bc name="fix" type="zero displacement" node_set="bottom">'
        "<x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof></bc>\n"
        '    <bc name="confine" type="zero displacement" '
        'node_set="all_nodes"><x_dof>1</x_dof><y_dof>1</y_dof></bc>\n'
        '    <bc name="push" type="prescribed displacement" '
        'node_set="top"><dof>z</dof>'
        '<value lc="1">-0.1</value></bc>\n'
        '    <bc name="drain" type="zero fluid pressure" node_set="top"/>\n'
        "  </Boundary>")
    return L.solid_deck(mesh=mesh, material=material, control=control,
                        boundary=boundary, n=1,
                        module='<Module type="biphasic"/>',
                        output=L.logfile(("element_data", "sz", "e.csv")))


def sz(run):
    b = L.parse_log_csv(run.files.get("e.csv") or "")
    return b[-1][1].get(1, [None])[0] if b else None


def main() -> int:
    a = L.run(deck("symmetric", ""), collect=("e.csv",), timeout=600)
    b = L.run(deck("non-symmetric", '<linear_solver type="bicgstab"/>'),
              collect=("e.csv",), timeout=600)
    sa, sb = sz(a), sz(b)
    print(f"symmetric_skyline: rc={a.rc} normal={int(a.normal_termination)} "
          f"steps={a.steps_completed} sz={sa}")
    print(f"nonsymmetric_bicgstab: rc={b.rc} "
          f"normal={int(b.normal_termination)} "
          f"steps={b.steps_completed} sz={sb}")
    if sa is None or sb is None:
        print("FAIL: one of the runs logged no element stress")
        return L.report(False, "symmetric_workaround", "reproduced",
                        "not_reproduced")
    rel = abs(sa - sb) / max(abs(sa), abs(sb), 1e-30)
    print(f"relative_difference={rel:.3e} tolerance={TOL:.0e}")
    good = (a.rc == 0 and b.rc == 0 and a.normal_termination
            and b.normal_termination and rel < TOL)
    return L.report(good, "symmetric_workaround", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
