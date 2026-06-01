"""Tier-2: Kratos FRICTION_ANGLE accessed from wrong module.

Pitfall (Kratos plasticity#4): Plasticity variables (FRICTION_ANGLE,
DILATANCY_ANGLE, YIELD_STRESS_COMPRESSION, HARDENING_CURVE,
COHESION) live in KratosMultiphysics.ConstitutiveLawsApplication.
Core KratosMultiphysics does NOT have them. KM.FRICTION_ANGLE
raises AttributeError at attribute lookup time, before any
properties.SetValue is reached:

  AttributeError: Module KratosMultiphysics has no attribute
  FRICTION_ANGLE.

(YIELD_STRESS / FRACTURE_ENERGY are in core KM; only the
plasticity-specific ones split into CLA.)
"""
from __future__ import annotations

import sys
import traceback

import KratosMultiphysics as KM


def main() -> int:
    try:
        # Bug: FRICTION_ANGLE lives in CLA, not KM.
        KM.FRICTION_ANGLE
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: KM.FRICTION_ANGLE resolved (catalog claim wrong)",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
