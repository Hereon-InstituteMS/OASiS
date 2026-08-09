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
import os
import sys
from pathlib import Path


# MUTATION CONTROL.  4C's enums are what they are, so the pathology cannot be
# injected from outside.  What CAN be removed is the assumption that the probe
# measures: T2_MUTATE=1 SWAPS the two probe sets, asking whether the underscored
# names are in fluid/scatra (they are) and whether the CamelCase names are in
# structural/thermal (they are).  All four `...=[]` expectations then report
# non-empty sets, which is only possible if each intersection is taken against
# the artefact.  The mutant's output is the positive half of the same claim.
MUTATE = os.environ.get("T2_MUTATE") == "1"

BARE_CAMELCASE = {"GenAlpha", "OneStepTheta"}
UNDERSCORED = {"Gen_Alpha", "One_Step_Theta", "Af_Gen_Alpha", "Np_Gen_Alpha"}
if MUTATE:
    BARE_CAMELCASE, UNDERSCORED = UNDERSCORED, BARE_CAMELCASE


def find_schema() -> Path | None:
    # 2026-08-03: search FOURC_SCHEMA_JSON and the deployed build on the
    # current verification host in addition to the original path. NOTE
    # the deployed build does not currently carry the artefact —
    # 4C_schema.json is produced post-build by
    # `create-schema-files 4C_metadata.yaml 4C_schema.json` (see
    # apps/global_full/CMakeLists.txt) and that step needs the build
    # venv, whose interpreter is broken here. The raw metadata IS
    # present as <build>/4C_metadata.yaml (also obtainable from
    # `4C --parameters`), so regenerating the schema is all this
    # fixture needs to become evaluable again.
    candidates = []
    env = os.environ.get("FOURC_SCHEMA_JSON")
    if env:
        candidates.append(Path(env))
    candidates += [
        Path.home() / "Schreibtisch" / "4C-src" / "4C"
        / "build" / "4C_schema.json",
        Path.home() / "4C" / "build" / "4C_schema.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def find_metadata() -> Path | None:
    """4C_metadata.yaml — the artefact 4C_schema.json is GENERATED from.

    The JSON schema is a post-build product of `create-schema-files`, which
    needs the build venv; on a host where that step did not run, this fixture
    reported "4C_schema.json not found" and went red for an environmental
    reason. The metadata YAML carries the same input description (it is the
    generator's input, also obtainable from `4C --parameters`) and IS present
    in the build tree, so read it when the JSON is absent.
    """
    candidates = []
    env = os.environ.get("FOURC_METADATA_YAML")
    if env:
        candidates.append(Path(env))
    candidates += [
        Path.home() / "Schreibtisch" / "4C-src" / "4C"
        / "build" / "4C_metadata.yaml",
        Path.home() / "4C" / "build" / "4C_metadata.yaml",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def enum_at(schema: dict, section: str, key: str) -> set[str]:
    sec = schema.get("properties", {}).get(section, {})
    pv = sec.get("properties", {}).get(key, {})
    return set(pv.get("enum", []))


def _find_named(node, name: str):
    """First spec node in a metadata tree carrying `name`."""
    if isinstance(node, dict):
        if node.get("name") == name:
            return node
        for v in node.values():
            r = _find_named(v, name)
            if r is not None:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _find_named(v, name)
            if r is not None:
                return r
    return None


def enum_at_meta(meta: dict, section: str, key: str) -> set[str]:
    sec = _find_named(meta.get("sections", {}), section)
    if sec is None:
        return set()
    node = _find_named(sec, key)
    if node is None or node.get("type") != "enum":
        return set()
    return {c["name"] for c in node.get("choices", []) if "name" in c}


def main() -> int:
    schema_path = find_schema()
    meta_path = None if schema_path is not None else find_metadata()
    if schema_path is None and meta_path is None:
        print("FAIL: neither 4C_schema.json nor 4C_metadata.yaml found",
              file=sys.stderr)
        return 2

    if schema_path is not None:
        with schema_path.open() as f:
            schema = json.load(f)
        print("artefact=4C_schema.json")
        def look(section: str, key: str) -> set[str]:
            return enum_at(schema, section, key)
    else:
        import yaml
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        with meta_path.open() as f:
            meta = yaml.load(f, Loader=loader)
        print("artefact=4C_metadata.yaml")
        def look(section: str, key: str) -> set[str]:
            return enum_at_meta(meta, section, key)

    fluid_ti = look("FLUID DYNAMIC", "TIMEINTEGR")
    scatra_ti = look("SCALAR TRANSPORT DYNAMIC", "TIMEINTEGR")
    struct_dt = look("STRUCTURAL DYNAMIC", "DYNAMICTYPE")
    therm_dt = look("THERMAL DYNAMIC", "DYNAMICTYPE")
    print(f"fluid_timeintegr={sorted(fluid_ti)}")
    print(f"scatra_timeintegr={sorted(scatra_ti)}")
    print(f"structural_dynamictype={sorted(struct_dt)}")
    print(f"thermal_dynamictype={sorted(therm_dt)}")

    # Bare 'GenAlpha' / 'OneStepTheta' MUST NOT be in
    # fluid or scatra (these use underscored variants):
    bare_in_fluid = BARE_CAMELCASE & fluid_ti
    bare_in_scatra = BARE_CAMELCASE & scatra_ti
    print(f"bare_camelcase_in_fluid={sorted(bare_in_fluid)}")
    print(f"bare_camelcase_in_scatra={sorted(bare_in_scatra)}")

    # Underscored MUST NOT be in struct/thermal:
    underscored_in_struct = UNDERSCORED & struct_dt
    underscored_in_thermal = UNDERSCORED & therm_dt
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
