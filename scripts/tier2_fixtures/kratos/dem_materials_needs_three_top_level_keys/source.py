"""Tier-2: MaterialsDEM.json needs materials + material_relations + assignation table.

Pitfall (kratos.dem): MaterialsDEM.json is NOT the StructuralMechanics materials
schema. The OASiS DEM generator emitted the FEM shape --

    {"properties": [{"model_part_name": ..., "properties_id": ...,
                     "Material": {"Variables": {...},
                                  "constitutive_law": {"name": ...}}}]}

-- which has none of the three keys DEM reads. DEM wants:

    "materials"                  per-particle data (PARTICLE_DENSITY, ...)
    "material_relations"         per-CONTACT-PAIR data (STATIC_FRICTION,
                                 COEFFICIENT_OF_RESTITUTION,
                                 DEM_DISCONTINUUM_CONSTITUTIVE_LAW_NAME, ...)
    "material_assignation_table" [[submodelpart_name, material_name_or_id], ...]

Each is read with an unguarded Parameters lookup, so the first missing one names
itself:

    RuntimeError: Error: Getting a value that does not exist. entry string : materials

This fixture drives MaterialsAssignationUtility directly, so it needs no mesh.

Mutation control: T2_MUTATE=1 REPLACES the FEM-shaped materials document with a
correctly shaped one carrying all three keys, leaving the utility call and the
assertions untouched. Mutated, fem_shaped_materials_rejected flips to False and
the process exits non-zero.
"""
from __future__ import annotations

import json
import os
import sys

import KratosMultiphysics as KM
import KratosMultiphysics.DEMApplication  # noqa: F401
from KratosMultiphysics.DEMApplication.materials_assignation_utility import (
    MaterialsAssignationUtility,
)

MUTATE = os.environ.get("T2_MUTATE") == "1"

FEM_SHAPED = {
    "properties": [{
        "model_part_name": "SpheresPart",
        "properties_id": 1,
        "Material": {
            "Variables": {"PARTICLE_DENSITY": 2650.0, "YOUNG_MODULUS": 1e7,
                          "POISSON_RATIO": 0.25},
            "constitutive_law": {"name": "DEM_D_Hertz_viscous_Coulomb"},
        },
    }],
}

DEM_SHAPED = {
    "materials": [{"material_name": "mat1", "material_id": 1,
                   "Variables": {"PARTICLE_DENSITY": 2650.0,
                                 "YOUNG_MODULUS": 1e7,
                                 "POISSON_RATIO": 0.25}}],
    "material_relations": [{"material_names_list": ["mat1", "mat1"],
                            "material_ids_list": [1, 1],
                            "Variables": {
                                "COEFFICIENT_OF_RESTITUTION": 0.5,
                                "STATIC_FRICTION": 0.4,
                                "DYNAMIC_FRICTION": 0.4,
                                "FRICTION_DECAY": 500.0,
                                "DEM_DISCONTINUUM_CONSTITUTIVE_LAW_NAME":
                                    "DEM_D_Hertz_viscous_Coulomb"}}],
    "material_assignation_table": [["SpheresPart", "mat1"]],
}

DOCUMENT = FEM_SHAPED
if MUTATE:
    print("mutation=fem_shaped_materials_replaced_with_the_dem_schema")
    DOCUMENT = DEM_SHAPED


def main() -> int:
    model = KM.Model()
    spheres = model.CreateModelPart("SpheresPart")
    params = KM.Parameters(json.dumps(DOCUMENT))

    rejected, msg = False, "<no exception>"
    try:
        MaterialsAssignationUtility(
            model, spheres, params).AssignMaterialParametersToProperties()
    except Exception as exc:  # noqa: BLE001 - classifying, not handling
        msg = str(exc).replace("\n", " ")
        rejected = "entry string : materials" in msg

    print(f"fem_shaped_materials_rejected={rejected}_expected=True")
    print(f"first_message={msg[:130]}")

    if not rejected:
        print("MISMATCH: the FEM-shaped materials document was not rejected on "
              "the 'materials' key", file=sys.stderr)
        return 1

    print("dem_materials_schema_check=ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
