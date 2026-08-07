"""Tier-2: Kratos DEM is NOT 3D-only -- CylinderParticle2D is a registered element.

Pitfall (kratos.dem): the catalog carried, in two places, the claim "DEM is
always 3D internally: the MDPA must reference SphericParticle3D even for
problems posed in a 2D plane", and advised constraining out-of-plane DOFs.
The premise is half right and the advice is wrong. SphericParticle2D really is
unregistered -- but the 2D particle exists under a different stem:

  CreateNewElement("SphericParticle2D", ...)   -> RuntimeError, not registered
  CreateNewElement("CylinderParticle2D", ...)  -> constructs
  CreateNewElement("CylinderContinuumParticle2D", ...) -> constructs

so the correct fix for a planar DEM problem is CylinderParticle2D (with the 2D
discontinuum law DEM_D_Hertz_viscous_Coulomb2D), not a 3D sphere with pinned
z-DOFs. This fixture pins BOTH halves so neither can drift back.

Mutation control: T2_MUTATE=1 INVERTS the expected verdict for every name --
it asserts SphericParticle2D constructs and the two Cylinder names do not.
The CreateNewElement calls themselves are untouched, so the mutation proves the
verdicts come from real element construction against this build's registry.
Mutated, all three constructs= lines disagree with themselves and the process
exits non-zero.
"""
from __future__ import annotations

import os
import sys

import KratosMultiphysics as KM
import KratosMultiphysics.DEMApplication  # noqa: F401  (registers DEM elements)

MUTATE = os.environ.get("T2_MUTATE") == "1"

# name -> must this element construct on a correct DEMApplication build?
CASES = [
    ("SphericParticle2D", False),
    ("CylinderParticle2D", True),
    ("CylinderContinuumParticle2D", True),
]
if MUTATE:
    print("mutation=expected_registration_verdicts_inverted")
    CASES = [(name, not must) for name, must in CASES]


def main() -> int:
    model = KM.Model()
    mp = model.CreateModelPart("Particles")
    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    prop = mp.CreateNewProperties(1)

    mismatches = 0
    for i, (name, must) in enumerate(CASES, start=1):
        try:
            mp.CreateNewElement(name, i, [1], prop)
            got = True
        except Exception:
            got = False
        print(f"constructs[{name}]={got}_expected={must}")
        if got != must:
            mismatches += 1
            print(f"MISMATCH: {name} constructs={got} expected={must}",
                  file=sys.stderr)

    print(f"element_registration_mismatches={mismatches}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
