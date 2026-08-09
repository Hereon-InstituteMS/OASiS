"""Tier-2 (kratos.mpm::9): MPM constitutive-law names are the fully qualified
registered strings, and the short family label is the commonest setup error.

"LinearElasticPlaneStrain2DLaw" reads like the right name and is not one. The
registered spelling carries the "Isotropic" the short name drops:

    LinearElasticIsotropicPlaneStrain2DLaw   registered
    LinearElasticPlaneStrain2DLaw            not registered

and a materials file built from the short name fails inside the reader:

    RuntimeError: Error: Kratos components missing
    "LinearElasticPlaneStrain2DLaw"
    ... kratos/utilities/read_materials_utility.cpp:209

Three probes, because each one is a different way to be wrong about this:

  1. KratosGlobals.HasConstitutiveLaw discriminates the two names.
  2. KratosGlobals.GetConstitutiveLaw does NOT raise on the unregistered name --
     it returns None. Anyone using it as an existence check gets no exception
     and a null law, which is how a bad name survives a hand-rolled validation.
  3. ReadMaterialsUtility, the path an actual deck takes, raises.

MUTATION CONTROL (T2_MUTATE=1): the two names are SWAPPED, so the fixture feeds
the registered name where it expects a rejection and the short one where it
expects success. Every call still really runs against Kratos; only which string
goes in changes. Mutated, all three probes disagree and the process exits 1.
"""
from __future__ import annotations

import json
import sys

import KratosMultiphysics as KM
import KratosMultiphysics.MPMApplication  # noqa: F401  (registers MPM laws)

MUTATE = "1" == __import__("os").environ.get("T2_MUTATE")

REGISTERED = "LinearElasticIsotropicPlaneStrain2DLaw"
SHORT = "LinearElasticPlaneStrain2DLaw"
if MUTATE:
    print("mutation=registered_and_short_law_names_swapped")
    REGISTERED, SHORT = SHORT, REGISTERED


def _read_materials(law_name):
    """Drive the real materials path; return (raised, message)."""
    model = KM.Model()
    part = model.CreateModelPart("Initial_MPM_Material")
    part.CreateSubModelPart("Parts_Body")
    doc = {"properties": [{
        "model_part_name": "Initial_MPM_Material.Parts_Body",
        "properties_id": 1,
        "Material": {"constitutive_law": {"name": law_name},
                     "Variables": {"DENSITY": 1000.0, "YOUNG_MODULUS": 1.0e6,
                                   "POISSON_RATIO": 0.3,
                                   "MATERIAL_POINTS_PER_ELEMENT": 4},
                     "Tables": {}}}]}
    try:
        KM.ReadMaterialsUtility(model).ReadMaterials(
            KM.Parameters(json.dumps(doc)))
        return False, ""
    except Exception as exc:                 # noqa: BLE001 - classifying
        return True, str(exc).replace("\n", " ")


def main() -> int:
    long_raised, long_msg = _read_materials(REGISTERED)
    short_raised, short_msg = _read_materials(SHORT)

    checks = [
        ("registered_name_is_known",
         KM.KratosGlobals.HasConstitutiveLaw(REGISTERED), True),
        ("short_name_is_not_known",
         KM.KratosGlobals.HasConstitutiveLaw(SHORT), False),
        ("GetConstitutiveLaw_returns_None_instead_of_raising",
         KM.KratosGlobals.GetConstitutiveLaw(SHORT) is None, True),
        ("registered_name_reads_materials", long_raised, False),
        ("short_name_fails_the_materials_read", short_raised, True),
        ("message_says_components_missing",
         "Kratos components missing" in short_msg, True),
        ("message_names_the_offending_string",
         ('"%s"' % SHORT) in short_msg, True),
    ]
    return _report(checks, [("first_message", short_msg[:110])],
                   "mpm_constitutive_law_name_check",
                   "the MPM law-naming claim")


def _report(checks, extras, ok_line, what):
    mismatches = 0
    for label, got, must in checks:
        if got != must:
            mismatches += 1
        print("probe[%s]=%s_expected=%s" % (label, got, must))
    for k, v in extras:
        print("%s=%s" % (k, v))
    print("probe_mismatches=%d" % mismatches)
    if mismatches:
        print("FIXTURE_FAILED: %s does not hold on this build" % what,
              file=sys.stderr)
        return 1
    print("%s=ok" % ok_line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
