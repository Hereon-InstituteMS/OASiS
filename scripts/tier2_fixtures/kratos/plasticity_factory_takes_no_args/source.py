"""Tier-2: SmallStrainIsotropicPlasticityFactory takes no constructor args.

Pitfall (Kratos plasticity#5): The factory class signature is
SmallStrainIsotropicPlasticityFactory() with no arguments.
Passing KM.Parameters to it raises:

  TypeError: __init__(): incompatible constructor arguments.
  The following argument types are supported:
    1. KratosConstitutiveLawsApplication.SmallStrainIsotropicPlasticityFactory()

To configure a plastic law, use the specific pre-combined
class (e.g. SmallStrainIsotropicPlasticityMisesMises3D), not
the factory with Parameters.
"""
from __future__ import annotations

import sys
import traceback

import KratosMultiphysics as KM
import KratosMultiphysics.ConstitutiveLawsApplication as CLA


def main() -> int:
    try:
        CLA.SmallStrainIsotropicPlasticityFactory(KM.Parameters("{}"))
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: factory accepted KM.Parameters (catalog claim wrong)",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
