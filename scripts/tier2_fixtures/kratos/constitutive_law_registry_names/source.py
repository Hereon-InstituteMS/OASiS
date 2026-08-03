"""Tier-2: constitutive-law NAME drift between the OASiS catalog and the
installed Kratos 10.4.0 registry.

Ten law names the catalog listed before 2026-08-03 do not resolve to anything
on this install — neither as a registry string (KratosGlobals.HasConstitutiveLaw)
nor as a Python attribute of ConstitutiveLawsApplication / StructuralMechanics /
MPMApplication. A Materials.json built from them dies with "not registered".

Also asserts the CORRECTED names this change wrote into the catalog, and the
Dam split where the Python class name and the registry string differ.
"""
from __future__ import annotations

import sys

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA  # noqa: F401
import KratosMultiphysics.ConstitutiveLawsApplication as CLA  # noqa: F401
import KratosMultiphysics.MPMApplication as MPM  # noqa: F401
import KratosMultiphysics.DamApplication as DAM  # noqa: F401

H = KM.KratosGlobals.HasConstitutiveLaw
MODULES = {"KM": KM, "SMA": SMA, "CLA": CLA, "MPM": MPM, "DAM": DAM}


def resolvable(name: str) -> bool:
    """A name is usable if it is either a registry string or a Python class."""
    if H(name):
        return True
    return any(hasattr(m, name) for m in MODULES.values())


# ---- names the pre-2026-08-03 catalog listed that resolve to NOTHING --------
STALE = [
    "Ogden", "Yeoh", "Arruda-Boyce", "Blatz-Ko",           # 'hyperelastic'
    "ModifiedCamClay", "CriticalStateLine",                 # 'plasticity'
    "Mazars", "RankineFragile",                             # 'damage'
    "Perzyna", "DruckerPragerViscoplastic",                 # 'viscoplastic'
    "HyperElasticIsotropicNeoHookean3DLaw",                 # linear_elasticity
    "HyperElasticIsotropicNeoHookean2DLaw",
    "LinearElasticAxisymmetric2DLaw",
    "GeneralizedMaxwell", "GeneralizedKelvin",
    "IsotropicDamage", "DplusDminusDamage",
]
# ---- the corrected names written into the catalog by this change -----------
FIXED = [
    "HyperElasticNeoHookean3DLaw", "HyperElasticNeoHookeanPlaneStrain2DLaw",
    "HyperElasticSimoTaylorNeoHookean3DLaw", "KirchhoffSaintVenant3DLaw",
    "LinearElasticAxisym2DLaw",
    "ViscousGeneralizedMaxwell3D", "ViscousGeneralizedKelvin3D",
    "SmallStrainIsotropicDamageFactory", "SmallStrainIsotropicDamage3DVonMises",
    "SmallStrainDplusDminusDamageModifiedMohrCoulombVonMises3D",
    "GenericSmallStrainViscoplasticity3D",
    "SmallStrainIsotropicPlasticity3DVonMisesVonMises",
    "SmallStrainIsotropicPlasticityPlaneStrainVonMisesVonMises",
    "SmallStrainKinematicPlasticityPlaneStrainVonMisesTresca",
    "LinearElasticIsotropic3DLaw", "HenckyMCPlastic3DLaw",
    "HenckyBorjaCamClayPlastic3DLaw", "JohnsonCookThermalPlastic3DLaw",
    "ThermalLinearPlaneStrain",
]

stale_hits = [n for n in STALE if resolvable(n)]
fixed_miss = [n for n in FIXED if not resolvable(n)]
print(f"n_stale_names_checked={len(STALE)}")
print(f"stale_names_still_resolvable={stale_hits}")
print(f"all_stale_names_unresolvable={not stale_hits}")
print(f"n_corrected_names_checked={len(FIXED)}")
print(f"corrected_names_missing={fixed_miss}")
print(f"all_corrected_names_resolvable={not fixed_miss}")

# ---- Dam: Python class name != registry string ----------------------------
dam_py = hasattr(DAM, "ThermalLinearElastic2DPlaneStrain")
dam_registry = H("ThermalLinearElastic2DPlaneStrain")
dam_real = H("ThermalLinearPlaneStrain")
print(f"dam_python_class_exists={dam_py}")
print(f"dam_same_string_in_registry={dam_registry}")
print(f"dam_registry_string_is_ThermalLinearPlaneStrain={dam_real}")

# ---- yield-surface tags are not laws on their own -------------------------
tags = ["VonMises", "Tresca", "DruckerPrager", "MohrCoulomb",
        "ModifiedMohrCoulomb", "Rankine", "SimoJu"]
tag_hits = [t for t in tags if H(t)]
print(f"bare_yield_surface_tags_registered={tag_hits}")
print(f"no_bare_yield_surface_tag_is_a_law={not tag_hits}")

# ---- 23 of 25 3D isotropic plasticity pairings exist ----------------------
surfaces = ["VonMises", "Tresca", "DruckerPrager", "MohrCoulomb", "ModifiedMohrCoulomb"]
missing = [f"{a}{b}" for a in surfaces for b in surfaces
           if not H(f"SmallStrainIsotropicPlasticity3D{a}{b}")]
print(f"missing_3d_isotropic_pairings={sorted(missing)}")
print(f"n_registered_3d_isotropic_pairings={25 - len(missing)}")

# ---- the two CLA Python classes that are NOT in the string registry -------
# These are the only laws where "assign from Python" and "name it in
# Materials.json" disagree, so they are the only ones that can fail one way
# and not the other. SmallStrainKinematicPlasticityPlaneStrainVonMisesVonMises
# additionally SIGSEGVs when reached through the Python route (see the
# constitutive_laws pitfall).
cla_classes = [a for a in dir(CLA)
               if a.startswith("SmallStrain") or a.startswith("FiniteStrain")]
not_registered = sorted(a for a in cla_classes if not H(a))
print(f"n_cla_python_classes={len(cla_classes)}")
print(f"cla_python_classes_not_in_registry={not_registered}")
print(f"n_cla_python_classes_not_in_registry={len(not_registered)}")

ok = (not stale_hits and not fixed_miss and dam_py and not dam_registry
      and dam_real and not tag_hits and len(missing) == 2
      and not_registered == ["FiniteStrainIsotropicPlasticityFactory",
                             "SmallStrainKinematicPlasticityPlaneStrainVonMisesVonMises"])
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
