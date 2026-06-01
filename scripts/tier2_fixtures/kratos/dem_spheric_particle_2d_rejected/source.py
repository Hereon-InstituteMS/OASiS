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
"""
from __future__ import annotations

import sys
import traceback

import KratosMultiphysics as KM
import KratosMultiphysics.DEMApplication as DEM  # noqa: F401


def main() -> int:
    model = KM.Model()
    mp = model.CreateModelPart("Particles")
    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    prop = mp.CreateNewProperties(1)
    try:
        mp.CreateNewElement("SphericParticle2D", 1, [1], prop)
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: SphericParticle2D accepted (catalog claim wrong)",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
