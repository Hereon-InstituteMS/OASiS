"""Tier-2: Kratos DEM has no SphericParticle2D.

Pitfall (kratos.dem::13), first half. Calling
CreateNewElement('SphericParticle2D', ...) raises:

  RuntimeError: Error: The Element "SphericParticle2D" is not
  registered!
  Maybe you need to import the application where it is defined?
  The following Elements are registered: ...

WHAT THIS DOCSTRING USED TO SAY, AND WHY IT NO LONGER DOES
----------------------------------------------------------
It read "DEM is always 3D internally. The MDPA must reference
SphericParticle3D even for problems posed in a 2D plane... the fix
is to use SphericParticle3D and constrain DOFs in the out-of-plane
direction." That was retracted on 2026-08-07: SphericParticle2D is
indeed unregistered, but the 2D particle exists under a different
stem, and CreateNewElement('CylinderParticle2D', ...) constructs.
The correct fix for a planar DEM problem is CylinderParticle2D with
the 2D law DEM_D_Hertz_viscous_Coulomb2D, NOT a pinned 3D sphere.
dem_has_real_2d_cylinder_particles pins both halves and is the
fixture of record for dem::13; this one is the redundant remainder,
kept because it still executes.

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
