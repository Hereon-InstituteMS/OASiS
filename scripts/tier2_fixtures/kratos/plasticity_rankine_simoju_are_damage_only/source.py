"""Tier-2: Rankine + SimoJu are damage models, NOT plasticity classes.

Catalog under audit (kratos plasticity):
  yield_surfaces:
    VonMises, Tresca, DruckerPrager, MohrCoulomb,
    ModifiedMohrCoulomb, Rankine, SimoJu

The pattern claim is
  <StrainSize><HardeningType><Dimension><YieldSurface><PlasticPotential>
  example: SmallStrainIsotropicPlasticity3DModifiedMohrCoulombModifiedMohrCoulomb

Combining 'Rankine' or 'SimoJu' as the YieldSurface + PlasticPotential
implies SmallStrainIsotropicPlasticity3DRankineRankine and
SmallStrainIsotropicPlasticity3DSimoJuSimoJu exist. They do NOT in
KratosConstitutiveLawsApplication 10.4.2:

  hasattr(CLA, 'SmallStrainIsotropicPlasticity3DRankineRankine')   False
  hasattr(CLA, 'SmallStrainIsotropicPlasticity3DSimoJuSimoJu')      False

The Rankine and SimoJu surfaces only exist in the *damage* family,
not the plasticity factory:
  hasattr(CLA, 'SmallStrainDplusDminusDamageRankineRankine3D')      True
  hasattr(CLA, 'SmallStrainDplusDminusDamageSimoJuSimoJu3D')        True
  hasattr(CLA, 'SmallStrainIsotropicDamage3DRankine')               True
  hasattr(CLA, 'SmallStrainIsotropicDamage3DSimoJu')                True

The other 5 catalog yield surfaces (VonMises, Tresca, DruckerPrager,
MohrCoulomb, ModifiedMohrCoulomb) DO combine correctly with
SmallStrainIsotropicPlasticity:

  hasattr(CLA, 'SmallStrainIsotropicPlasticity3DVonMisesVonMises')           True
  hasattr(CLA, 'SmallStrainIsotropicPlasticity3DDruckerPragerDruckerPrager') True
  hasattr(CLA, 'SmallStrainIsotropicPlasticity3DMohrCoulombMohrCoulomb')     True
  hasattr(CLA, 'SmallStrainIsotropicPlasticity3DModifiedMohrCoulombModifiedMohrCoulomb') True
  hasattr(CLA, 'SmallStrainIsotropicPlasticity3DTrescaTresca')              True
"""
from __future__ import annotations

import sys

import KratosMultiphysics.ConstitutiveLawsApplication as CLA


def main() -> int:
    # (1) Plasticity-side Rankine/SimoJu do NOT exist.
    nonexistent = [
        "SmallStrainIsotropicPlasticity3DRankineRankine",
        "SmallStrainIsotropicPlasticity3DSimoJuSimoJu",
        "SmallStrainKinematicPlasticity3DRankineRankine",
        "SmallStrainKinematicPlasticity3DSimoJuSimoJu",
        "FiniteStrainIsotropicPlasticity3DRankineRankine",
    ]
    all_missing = True
    for name in nonexistent:
        present = hasattr(CLA, name)
        if present:
            print(f"unexpected_present_{name}=True", file=sys.stderr)
            all_missing = False
    print(f"all_plasticity_rankine_simoju_missing={all_missing}")

    # (2) The other 5 yield surfaces DO combine into plasticity classes.
    valid = [
        "SmallStrainIsotropicPlasticity3DVonMisesVonMises",
        "SmallStrainIsotropicPlasticity3DTrescaTresca",
        "SmallStrainIsotropicPlasticity3DDruckerPragerDruckerPrager",
        "SmallStrainIsotropicPlasticity3DMohrCoulombMohrCoulomb",
        ("SmallStrainIsotropicPlasticity3DModified"
         "MohrCoulombModifiedMohrCoulomb"),
    ]
    all_valid = True
    for name in valid:
        if not hasattr(CLA, name):
            print(f"unexpected_missing_{name}=True", file=sys.stderr)
            all_valid = False
    print(f"all_5_valid_yield_surfaces_present={all_valid}")

    # (3) Rankine + SimoJu DO exist as damage models.
    damage = [
        "SmallStrainDplusDminusDamageRankineRankine3D",
        "SmallStrainDplusDminusDamageSimoJuSimoJu3D",
        "SmallStrainIsotropicDamage3DRankine",
        "SmallStrainIsotropicDamage3DSimoJu",
    ]
    all_damage_present = True
    for name in damage:
        if not hasattr(CLA, name):
            print(f"unexpected_missing_damage_{name}=True",
                  file=sys.stderr)
            all_damage_present = False
    print(f"all_rankine_simoju_damage_models_present="
          f"{all_damage_present}")

    if all_missing and all_valid and all_damage_present:
        return 0
    print("FAIL: invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
