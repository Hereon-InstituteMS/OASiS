"""Tier-2: Line3DN, the geometry every cable_net element is built on,
is a stub that answers some questions, refuses others, and silently
returns nothing for the rest.

The claim (kratos.cable_net #10) walks the source and lists five gaps.
Three of them are reachable from Python and are pinned here:

  * a geometry query raises with the source's own misspelling,
    "not available for arbitrarty noded line";
  * the shape-function API returns an EMPTY table instead of raising,
    even though the class docstring advertises quadratic shape
    functions — the silent half of the same defect;
  * the constructor's PointsNumber validation is commented out, so a
    node list of the wrong length is accepted without a word and the
    geometry reports whatever count it was handed.

Two of them are NOT reachable: the pybind11 Geometry binding exposes
neither ShapeFunctionValue nor PointsLocalCoordinates / LumpingFactors
/ IsInside / InverseOfJacobian, so the claim's headline Signal — a
KRATOS_ERROR out of ShapeFunctionValue — cannot be provoked from
Python at all. The identical message IS reachable through DomainSize,
which is what this fixture asserts on; the fixture also records the
absence so the unreachable half is not mistaken for a passing one.

Mutation control: T2_MUTATE=1 builds the geometry from StructuralMechanics' CableElement3D2N, which carries a real Line3D2N geometry, instead of CableNet's SlidingCableElement3D3N stub. DomainSize() then answers instead of raising the misspelled KRATOS_ERROR and ShapeFunctionsValues() returns a populated table, so the stub-specific assertions collapse.
"""
from __future__ import annotations

import os
import sys

sys.excepthook = sys.__excepthook__
os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import KratosMultiphysics.CableNetApplication as CN                # noqa: F401

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=stub_geometry_replaced_by_a_real_line_geometry")

# The source's own typo. Written split so this file's text cannot be
# mistaken for a spelling mistake of its own.
TYPO = "arbitrar" + "ty noded line"

# Named in the claim's Signal but absent from the Python binding.
CXX_ONLY = ("ShapeFunctionValue", "PointsLocalCoordinates",
            "LumpingFactors", "IsInside", "InverseOfJacobian")


def cable_geometry(node_ids):
    model = KM.Model()
    mp = model.CreateModelPart("cable")
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    for i, (x, y, z) in enumerate([(0, 0, 0), (1, 0, 0), (2, 0, 0),
                                   (2, 1, 0), (3, 0, 0)], 1):
        mp.CreateNewNode(i, float(x), float(y), float(z))
    props = mp.CreateNewProperties(1)
    props.SetValue(KM.YOUNG_MODULUS, 2.1e11)
    props.SetValue(SMA.CROSS_AREA, 0.01)
    props.SetValue(KM.DENSITY, 7850.0)
    props.SetValue(KM.CONSTITUTIVE_LAW, SMA.TrussConstitutiveLaw())
    if MUTATE:
        # Pathology removed: a real Line3D2N-backed element, not the stub.
        el = mp.CreateNewElement("CableElement3D2N", 1,
                                 list(node_ids)[:2], props)
    else:
        el = mp.CreateNewElement("SlidingCableElement3D3N", 1,
                                 list(node_ids), props)
    # Keep the model part alive: the geometry holds references into it.
    return el.GetGeometry(), mp


def main() -> int:
    fail: list[str] = []
    geom, _keep = cable_geometry([1, 2, 3])

    # 1. The misspelled KRATOS_ERROR, reached through DomainSize.
    raised, message = False, ""
    try:
        geom.DomainSize()
    except Exception as exc:                                  # noqa: BLE001
        raised = True
        message = " ".join(str(exc).split())
    print(f"domain_size_raised={raised}")
    print(f"domain_size_message_has_source_typo={TYPO in message}")
    print(f"domain_size_message={message[:150]}")
    if not (raised and TYPO in message):
        fail.append(f"DomainSize() on a cable_net geometry did not raise "
                    f"with the source's own '{TYPO}' text; the claim is "
                    f"that the stub geometry refuses this family of "
                    f"queries and that the typo is real. Got: "
                    f"{message[:140] or 'no exception'}")

    # 2. The shape-function table comes back EMPTY, with no error — the
    #    silent counterpart of (1), and the one a user actually hits
    #    because nothing tells them anything went wrong.
    table = geom.ShapeFunctionsValues()
    empty = (table.Size1() == 0 and table.Size2() == 0)
    print(f"shape_functions_table_rows={table.Size1()}")
    print(f"shape_functions_table_cols={table.Size2()}")
    print(f"shape_functions_returned_empty_without_raising={empty}")
    if not empty:
        fail.append(f"ShapeFunctionsValues() returned a "
                    f"{table.Size1()}x{table.Size2()} table; the claim is "
                    f"that no shape functions are implemented despite the "
                    f"class docstring advertising quadratic ones")

    # 3. EdgesNumber is 0 — correct for a 1D-in-3D line, unusual enough
    #    that a caller probing it gets an answer they do not expect.
    edges = geom.EdgesNumber()
    print(f"edges_number={edges}")
    if edges != 0:
        fail.append(f"EdgesNumber() returned {edges}, not 0")

    # 4. The commented-out PointsNumber check: a five-entry node list
    #    for a THREE-node element is accepted in silence.
    five_ok, five_count = False, -1
    try:
        geom5, _keep5 = cable_geometry([1, 2, 3, 4, 5])
        five_count = geom5.PointsNumber()
        five_ok = True
    except Exception as exc:                                  # noqa: BLE001
        print(f"five_node_create_raised={type(exc).__name__}")
    print(f"five_node_list_accepted_silently={five_ok}")
    print(f"five_node_geometry_points_number={five_count}")
    if not (five_ok and five_count == 5):
        fail.append("a five-entry node list for SlidingCableElement3D3N "
                    "was rejected; the claim is that Line3DN's PointsNumber "
                    "validation is commented out and any N is accepted at "
                    "construction")

    # 5. Record which Signals of this claim cannot be provoked from
    #    Python. Unreachable is a different statement from 'passes'.
    missing = [n for n in CXX_ONLY if not hasattr(geom, n)]
    print(f"cxx_only_signal_methods_absent_from_binding="
          f"{len(missing)}_of_{len(CXX_ONLY)}")
    print(f"cxx_only_signal_methods={','.join(missing)}")
    if len(missing) != len(CXX_ONLY):
        fail.append(f"the Python Geometry binding now exposes "
                    f"{sorted(set(CXX_ONLY) - set(missing))}; this "
                    f"fixture asserts they are C++-only and must be "
                    f"rewritten to provoke them directly")

    if not fail:
        print("cablenet_line3dn_stub_geometry_verified=True")
        return 0
    for reason in fail:
        print(f"FAIL: {reason}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
