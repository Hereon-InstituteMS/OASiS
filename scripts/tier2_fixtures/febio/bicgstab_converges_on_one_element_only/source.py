"""Tier-2: bicgstab is the only solver that CAN run an unsymmetric system
on this build, and not one that reliably does.

Verifies febio::biphasic#4. The same confined-compression biphasic deck,
written here rather than taken from the templates so the mesh can be
refined, converges on a single hex8 and fails on 2x2x2 with
`Linear solver failed to find solution. Aborting run.` — the same message
LU gives, and easy to misread as a modelling problem when it is a solver
problem.

The one-element run is the control: it is what rules out the deck.

MUTATION CONTROL. T2_MUTATE=1 runs the "2x2x2" slot on the SINGLE hex8
— the pathology (a mesh bicgstab cannot get through) removed. That run
then terminates normally without the linear-solver message, so
'bicgstab_mesh_dependence=reproduced' is no longer printed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

def deck(n: int) -> str:
    mesh, _info = L.hex8_box(n)
    material = (
        "  <Material>\n"
        '    <material id="1" name="Material1" type="biphasic">\n'
        "      <phi0>0.2</phi0>\n"
        '      <solid type="neo-Hookean"><density>1.0</density>'
        "<E>1000.0</E><v>0.0</v></solid>\n"
        '      <permeability type="perm-const-iso">'
        "<perm>0.001</perm></permeability>\n"
        "    </material>\n  </Material>")
    control = (
        "  <Control>\n    <analysis>TRANSIENT</analysis>\n"
        "    <time_steps>10</time_steps>\n    <step_size>0.1</step_size>\n"
        '    <solver type="biphasic">\n'
        "      <symmetric_stiffness>non-symmetric</symmetric_stiffness>\n"
        '      <linear_solver type="bicgstab"/>\n'
        "    </solver>\n  </Control>")
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
                        boundary=boundary, n=n,
                        module='<Module type="biphasic"/>',
                        output=L.logfile(("element_data", "sz;p", "e.csv")))


def main() -> int:
    one = L.run(deck(1), collect=("e.csv",), timeout=600)
    if MUTATE:
        print("mutation=the_refined_slot_runs_the_single_hex8")
    many = L.run(deck(1 if MUTATE else 2), collect=("e.csv",),
                 timeout=600)
    msg = "Linear solver failed to find solution"
    b = L.parse_log_csv(one.files.get("e.csv") or "")
    print(f"one_hex8: rc={one.rc} normal={int(one.normal_termination)} "
          f"steps={one.steps_completed} "
          f"solver_message={int(one.has(msg))} "
          f"last_element={b[-1][1].get(1) if b else None}")
    print(f"two_by_two_by_two: rc={many.rc} "
          f"normal={int(many.normal_termination)} "
          f"steps={many.steps_completed} "
          f"solver_message={int(many.has(msg))} "
          f"error_termination={int(many.error_termination)}")
    good = (one.rc == 0 and one.normal_termination and not one.has(msg)
            and many.rc != 0 and not many.normal_termination
            and many.has(msg) and many.steps_completed == 0)
    if not good:
        print(many.text[:1000])
    return L.report(good, "bicgstab_mesh_dependence", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
