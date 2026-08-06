"""Tier-2: KM.VtkOutput rejects the cable/ring Line3DN geometry.

Pitfall (kratos.cable_net #1). Both halves are observed: the
rejection, and the documented workaround (a visualisation
SubModelPart of plain Element3D2N lines) succeeding.

Mutation control: T2_MUTATE=1 hands VtkOutput the supported Element3D2N geometry in the slot where the fixture expects the unsupported Line3DN, removing the geometry that VTK output cannot write; the call then succeeds instead of raising.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import KratosMultiphysics.CableNetApplication  # noqa: F401

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=line3dn_case_replaced_by_the_supported_element3d2n")

_VTK = KM.Parameters("""{
    "file_format": "ascii",
    "output_precision": 7,
    "output_control_type": "step",
    "output_interval": 1,
    "output_sub_model_parts": false,
    "output_path": "vtk_probe",
    "nodal_solution_step_data_variables": ["DISPLACEMENT"]
}""")


def make(element_name: str, n: int):
    model = KM.Model()
    mp = model.CreateModelPart("v" + element_name)
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    mp.SetBufferSize(2)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    for i, (x, y, z) in enumerate([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                   (2.0, 0.5, 0.0)][:n]):
        mp.CreateNewNode(i + 1, x, y, z)
    prop = mp.CreateNewProperties(1)
    prop.SetValue(KM.YOUNG_MODULUS, 2.1e11)
    prop.SetValue(KM.DENSITY, 7850.0)
    prop.SetValue(SMA.CROSS_AREA, 0.01)
    mp.CreateNewElement(element_name, 1, list(range(1, n + 1)), prop)
    return mp


def main() -> int:
    bad = 0
    cable = (make("Element3D2N", 2) if MUTATE
             else make("SlidingCableElement3D3N", 3))
    try:
        KM.VtkOutput(cable, _VTK).PrintOutput()
        print("vtk_on_line3dn_raised=False")
        print("FAIL: VtkOutput accepted the Line3DN geometry", file=sys.stderr)
        bad += 1
    except Exception as exc:
        msg = str(exc).replace("\n", " ")
        print("vtk_on_line3dn_raised=True")
        if "no VTK-output is implemented" in msg:
            print("observed=no_vtk_output_implemented_for_geometry")
        else:
            print(f"FAIL: unexpected message {msg[:150]}", file=sys.stderr)
            bad += 1

    # The documented workaround must actually work, or the claim's
    # second half is unverified.
    line = make("Element3D2N", 2)
    try:
        KM.VtkOutput(line, _VTK).PrintOutput()
        print("vtk_on_element3d2n_raised=False")
    except Exception as exc:
        print("vtk_on_element3d2n_raised=True")
        print(f"FAIL: the workaround geometry is rejected too: "
              f"{str(exc).splitlines()[0][:130]}", file=sys.stderr)
        bad += 1

    print(f"vtk_line3dn_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
