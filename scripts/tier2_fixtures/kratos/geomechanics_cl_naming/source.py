"""Tier-2: Kratos GeoMechanics constitutive law names.

The kratos geomechanics catalog under audit listed 7 CL names:

  LinearElastic2DPlaneStrain                       MISSING
  LinearElastic3DLaw                               MISSING
  ModifiedCamClay                                  MISSING (no any Cam-Clay law)
  MohrCoulomb                                      MISSING (only GeoMohrCoulombWithTensionCutOff*)
  DruckerPrager                                    MISSING (no Drucker-Prager law at all)
  SmallStrainUDSM2DPlaneStrainLaw                  OK
  SmallStrainUDSM3DLaw                             OK

Real Kratos GeoMechanicsApplication registers (per binary string
scan of libKratosGeoMechanicsCore.so):

  GeoLinearElasticPlaneStrain2DLaw       (NOT LinearElastic2D…)
  GeoIncrementalLinearElastic3DLaw       (NOT LinearElastic3DLaw)
  GeoIncrementalLinearElasticInterfaceLaw
  LinearElastic2DInterfaceLaw
  LinearElastic3DInterfaceLaw
  GeoMohrCoulombWithTensionCutOff2D      (NOT plain MohrCoulomb)
  GeoMohrCoulombWithTensionCutOff3D
  MohrCoulombWithTensionCutOff
  SmallStrainUDSM2DPlaneStrainLaw        ✓ catalog matched
  SmallStrainUDSM3DLaw                   ✓ catalog matched
  SmallStrainUDSM2DInterfaceLaw
  SmallStrainUDSM3DInterfaceLaw
  TrussBackboneConstitutiveLaw

Critically: ModifiedCamClay and DruckerPrager are NOT registered
ANYWHERE in GeoMechanicsApplication — those rows in the catalog
have no upstream backing at all in Kratos 10.4.2.

This fixture verifies the registration story via binary string
scan (since Python attributes on the GMA module do not include
these names — CLs are JSON-driven, not Python-imported).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    so_path = (
        Path.home() / "Schreibtisch" / "Open-FEM-agent" / ".venv"
        / "lib" / "python3.12" / "site-packages"
        / "KratosMultiphysics" / ".libs"
        / "libKratosGeoMechanicsCore.so"
    )
    if not so_path.is_file():
        print(f"FAIL: binary not found at {so_path}", file=sys.stderr)
        return 2

    result = subprocess.run(
        ["strings", str(so_path)],
        capture_output=True, text=True, check=True)
    text = result.stdout

    # Names the catalog claimed (audit target).
    catalog_listed = [
        "LinearElastic2DPlaneStrain",
        "LinearElastic3DLaw",
        "ModifiedCamClay",
        "MohrCoulomb",
        "DruckerPrager",
        "SmallStrainUDSM2DPlaneStrainLaw",
        "SmallStrainUDSM3DLaw",
    ]

    # Real registered names (from binary string scan).
    real_names = [
        "GeoLinearElasticPlaneStrain2DLaw",
        "GeoIncrementalLinearElastic3DLaw",
        "GeoMohrCoulombWithTensionCutOff2D",
        "GeoMohrCoulombWithTensionCutOff3D",
        "SmallStrainUDSM2DPlaneStrainLaw",
        "SmallStrainUDSM3DLaw",
    ]

    # Catalog names which catalog-listed and DO exist (subset).
    catalog_correct = {
        "SmallStrainUDSM2DPlaneStrainLaw",
        "SmallStrainUDSM3DLaw",
    }

    # Verify each catalog-listed name's existence as a registered
    # (whole-word) token in the binary.
    catalog_present = {}
    for n in catalog_listed:
        # Match as whole word, not as substring
        pattern = r"\b" + re.escape(n) + r"\b"
        catalog_present[n] = bool(re.search(pattern, text))
    print(f"catalog_present_count="
          f"{sum(catalog_present.values())}/{len(catalog_listed)}")

    # Verify the catalog-correct subset all present.
    catalog_correct_ok = all(
        catalog_present[n] for n in catalog_correct)
    print(f"catalog_correct_subset_ok={catalog_correct_ok}")

    # Verify the catalog-WRONG names absent.
    catalog_wrong = [n for n in catalog_listed
                     if n not in catalog_correct]
    wrong_correctly_missing = all(
        not catalog_present[n] for n in catalog_wrong)
    print(f"all_5_wrong_catalog_names_missing="
          f"{wrong_correctly_missing}")

    # Verify the real names exist.
    real_present = {}
    for n in real_names:
        pattern = r"\b" + re.escape(n) + r"\b"
        real_present[n] = bool(re.search(pattern, text))
    print(f"real_names_present_count="
          f"{sum(real_present.values())}/{len(real_names)}")
    all_real_ok = all(real_present.values())
    print(f"all_real_names_present={all_real_ok}")

    # Critical: ModifiedCamClay and DruckerPrager have NO
    # variant at all (the entire CL families are not present).
    cam_clay_anywhere = bool(re.search(r"CamClay", text))
    drucker_anywhere = bool(re.search(r"DruckerPrager", text))
    print(f"camclay_anywhere_in_geomech={cam_clay_anywhere}")
    print(f"druckerprager_anywhere_in_geomech={drucker_anywhere}")

    ok = (catalog_correct_ok
          and wrong_correctly_missing
          and all_real_ok
          and (not cam_clay_anywhere)
          and (not drucker_anywhere))
    if ok:
        return 0
    print("FAIL: invariants not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
