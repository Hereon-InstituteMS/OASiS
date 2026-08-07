"""Tier-2: Kratos DEM has no SphericParticle2D.

Pitfall (Kratos dem#3): DEM is always 3D internally. The MDPA
must reference SphericParticle3D even for problems posed in a
2D plane. Calling CreateNewElement('SphericParticle2D', ...)
raises:

  RuntimeError: Error: The Element "SphericParticle2D" is not
  registered!
  Maybe you need to import the application where it is defined?
  The following Elements are registered: ...

The fix is to use SphericParticle3D and constrain DOFs in the
out-of-plane direction.

Mutation control: T2_MUTATE=1 creates the registered SphericParticle3D instead of the unregistered SphericParticle2D, removing the bad element name. The element is then created and the RuntimeError naming SphericParticle2D as 'not registered' disappears.
"""
from __future__ import annotations

import os
import sys
import traceback

import KratosMultiphysics as KM
import KratosMultiphysics.DEMApplication as DEM  # noqa: F401

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=registered_sphericparticle3d_used_instead")


def main() -> int:
    model = KM.Model()
    mp = model.CreateModelPart("Particles")
    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    prop = mp.CreateNewProperties(1)
    try:
        name = "SphericParticle3D" if MUTATE else "SphericParticle2D"
        mp.CreateNewElement(name, 1, [1], prop)
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: SphericParticle2D accepted (catalog claim wrong)",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
