"""Tier-2: 4C time-integration enums are SECTION-DEPENDENT.

The earlier catalog used bare 'GenAlpha' / 'OneStepTheta'
across fluid + scatra, but each section has its own
TIMEINTEGR or DYNAMICTYPE enum:

  FLUID DYNAMIC/TIMEINTEGR:
    Af_Gen_Alpha, Np_Gen_Alpha, BDF2,
    One_Step_Theta, Stationary
  SCALAR TRANSPORT DYNAMIC/TIMEINTEGR:
    Gen_Alpha, BDF2, One_Step_Theta, Stationary
  STRUCTURAL DYNAMIC/DYNAMICTYPE:
    GenAlpha, GenAlphaLieGroup, OneStepTheta, Statics,
    CentrDiff, AdamsBashforth2, AdamsBashforth4,
    ExplicitEuler
  THERMAL DYNAMIC/DYNAMICTYPE:
    GenAlpha, OneStepTheta, Statics, Undefined

Naming convention is SECTION-bound: TIMEINTEGR uses
underscored names; DYNAMICTYPE uses CamelCase. Same
conceptual scheme, different spelling.

This fixture walks 4C's compiled JSON schema and asserts:
  * 'GenAlpha' (bare) is NOT in FLUID/SCATRA TIMEINTEGR.
  * 'OneStepTheta' (CamelCase) is NOT in FLUID/SCATRA.
  * 'Gen_Alpha' (underscored) is in SCATRA but NOT in
    STRUCTURAL/THERMAL DYNAMICTYPE.
  * The 5 fluid + 4 scatra + 4 thermal canonical names
    are all present in their respective enums.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def find_schema() -> Path | None:
    p = (Path.home() / "Schreibtisch" / "4C-src" / "4C"
         / "build" / "4C_schema.json")
    return p if p.is_file() else None


def enum_at(schema: dict, section: str, key: str) -> set[str]:
    sec = schema.get("properties", {}).get(section, {})
    pv = sec.get("properties", {}).get(key, {})
    return set(pv.get("enum", []))


def main() -> int:
    schema_path = find_schema()
    if schema_path is None:
        print("FAIL: 4C_schema.json not found", file=sys.stderr)
        return 2

    with schema_path.open() as f:
        schema = json.load(f)

    fluid_ti = enum_at(schema, "FLUID DYNAMIC", "TIMEINTEGR")
    scatra_ti = enum_at(schema, "SCALAR TRANSPORT DYNAMIC",
                        "TIMEINTEGR")
    struct_dt = enum_at(schema, "STRUCTURAL DYNAMIC",
                        "DYNAMICTYPE")
    therm_dt = enum_at(schema, "THERMAL DYNAMIC",
                       "DYNAMICTYPE")
    print(f"fluid_timeintegr={sorted(fluid_ti)}")
    print(f"scatra_timeintegr={sorted(scatra_ti)}")
    print(f"structural_dynamictype={sorted(struct_dt)}")
    print(f"thermal_dynamictype={sorted(therm_dt)}")

    # Bare 'GenAlpha' / 'OneStepTheta' MUST NOT be in
    # fluid or scatra (these use underscored variants):
    bare_in_fluid = {"GenAlpha", "OneStepTheta"} & fluid_ti
    bare_in_scatra = {"GenAlpha", "OneStepTheta"} & scatra_ti
    print(f"bare_camelcase_in_fluid={sorted(bare_in_fluid)}")
    print(f"bare_camelcase_in_scatra={sorted(bare_in_scatra)}")

    # Underscored MUST NOT be in struct/thermal:
    underscored_in_struct = (
        {"Gen_Alpha", "One_Step_Theta", "Af_Gen_Alpha",
         "Np_Gen_Alpha"} & struct_dt)
    underscored_in_thermal = (
        {"Gen_Alpha", "One_Step_Theta", "Af_Gen_Alpha",
         "Np_Gen_Alpha"} & therm_dt)
    print(f"underscored_in_struct={sorted(underscored_in_struct)}")
    print(f"underscored_in_thermal={sorted(underscored_in_thermal)}")

    fluid_required = {"Af_Gen_Alpha", "Np_Gen_Alpha", "BDF2",
                      "One_Step_Theta", "Stationary"}
    scatra_required = {"Gen_Alpha", "BDF2", "One_Step_Theta",
                       "Stationary"}
    struct_required = {"GenAlpha", "OneStepTheta", "Statics"}
    therm_required = {"GenAlpha", "OneStepTheta", "Statics"}

    ok = (
        not bare_in_fluid
        and not bare_in_scatra
        and not underscored_in_struct
        and not underscored_in_thermal
        and fluid_required <= fluid_ti
        and scatra_required <= scatra_ti
        and struct_required <= struct_dt
        and therm_required <= therm_dt
    )
    if ok:
        return 0
    print("FAIL: time-integration enum/section invariant "
          "not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
