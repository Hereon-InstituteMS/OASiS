"""Tier-2: mortar contact entities are Conditions, never Elements.

The claim is that a contact SubModelPart must hold Conditions of a
Mortar type and not Elements. That is enforced by the registry:
the same name resolves as a Condition and is rejected as an
Element.

Mutation control: T2_MUTATE=1 creates the mortar entities with CreateNewCondition in the slot that is supposed to try CreateNewElement, i.e. it stops asking for them as Elements. Every as_element[...] line then reads True and contact_entity_kind_mismatches rises above 0.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication  # noqa: F401
import KratosMultiphysics.ContactStructuralMechanicsApplication  # noqa: F401

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=mortar_entities_created_as_conditions_in_both_slots")


NAMES = ["ALMFrictionlessMortarContactCondition2D2N",
         "ALMFrictionalMortarContactCondition2D2N",
         "PenaltyFrictionlessMortarContactCondition2D2N"]


def main() -> int:
    model = KM.Model()
    mp = model.CreateModelPart("t")
    mp.SetBufferSize(1)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    mp.CreateNewNode(2, 1.0, 0.0, 0.0)
    prop = mp.CreateNewProperties(1)

    bad = 0
    for i, name in enumerate(NAMES):
        try:
            mp.CreateNewCondition(name, 100 + i, [1, 2], prop)
            print(f"as_condition[{name}]=True")
        except Exception as exc:
            print(f"as_condition[{name}]=False")
            print(f"FAIL: {name} is not registered as a Condition: "
                  f"{str(exc).splitlines()[0][:130]}", file=sys.stderr)
            bad += 1
        try:
            if MUTATE:
                # Pathology removed: ask for it as the Condition it is.
                mp.CreateNewCondition(name, 500 + i, [1, 2], prop)
            else:
                mp.CreateNewElement(name, 500 + i, [1, 2], prop)
            print(f"as_element[{name}]=True")
            print(f"FAIL: {name} was ALSO accepted as an Element, so the "
                  f"Condition/Element distinction this pitfall rests on does "
                  f"not hold", file=sys.stderr)
            bad += 1
        except Exception as exc:
            print(f"as_element[{name}]=False")
            print(f"  message: {str(exc).splitlines()[0][:120]}")
    print(f"contact_entity_kind_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
