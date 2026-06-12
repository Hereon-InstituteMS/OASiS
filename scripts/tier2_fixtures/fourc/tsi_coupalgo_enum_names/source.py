"""Tier-2: 4C TSI DYNAMIC/COUPALGO enum names.

The catalog under audit (data/fourc_knowledge.py["tsi"]
.coupling_algorithms) had FOUR wrong names:

  Catalog (wrong)              → 4C schema (right)
  -----------------------------------------------------
  tsi_iterstaggaitken          → tsi_iterstagg_aitken
  tsi_iterstaggaitkenirons     → tsi_iterstagg_aitkenirons
  tsi_iterstaggfixedrel        → tsi_iterstagg_fixedrelax
  monolithic                   → tsi_monolithic

This fixture walks 4C's compiled JSON schema (the canonical
source for valid enum values) and asserts the catalog's
current 7 names are exactly the schema's COUPALGO enum.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


CATALOG_COUPALGO = {
    "tsi_oneway",
    "tsi_sequstagg",
    "tsi_iterstagg",
    "tsi_iterstagg_aitken",
    "tsi_iterstagg_aitkenirons",
    "tsi_iterstagg_fixedrelax",
    "tsi_monolithic",
}


def find_schema() -> Path | None:
    candidates = [
        Path.home() / "Schreibtisch" / "4C-src"
        / "4C" / "build" / "4C_schema.json",
        Path("/home/hermann/Schreibtisch/4C-src/4C/build/"
             "4C_schema.json"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def main() -> int:
    schema_path = find_schema()
    if schema_path is None:
        print("FAIL: 4C_schema.json not found at expected paths",
              file=sys.stderr)
        return 2

    with schema_path.open() as f:
        schema = json.load(f)

    tsi = schema.get("properties", {}).get("TSI DYNAMIC", {})
    coupalgo = tsi.get("properties", {}).get("COUPALGO", {})
    schema_enum = set(coupalgo.get("enum", []))
    print(f"schema_enum={sorted(schema_enum)}")

    if not schema_enum:
        print("FAIL: schema's TSI DYNAMIC/COUPALGO enum empty",
              file=sys.stderr)
        return 2

    in_both = CATALOG_COUPALGO & schema_enum
    catalog_extra = CATALOG_COUPALGO - schema_enum
    schema_extra = schema_enum - CATALOG_COUPALGO
    print(f"catalog_extra={sorted(catalog_extra)}")
    print(f"schema_extra={sorted(schema_extra)}")

    # Old wrong names must NOT be re-introduced into the
    # schema (and definitely not into our catalog).
    historical_wrong = {
        "tsi_iterstaggaitken",
        "tsi_iterstaggaitkenirons",
        "tsi_iterstaggfixedrel",
        "monolithic",
    }
    wrong_in_schema = historical_wrong & schema_enum
    print(f"historical_wrong_in_schema={sorted(wrong_in_schema)}")

    ok = (
        len(catalog_extra) == 0
        and len(schema_extra) == 0
        and len(wrong_in_schema) == 0
        and CATALOG_COUPALGO == schema_enum
    )
    if ok:
        print(f"all_match=True n={len(in_both)}")
        return 0
    print("FAIL: COUPALGO catalog/schema drift",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
