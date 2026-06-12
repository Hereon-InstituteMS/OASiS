"""Tier-2: 4C beam material names + beam LINE coverage.

The catalog under audit (data/fourc_knowledge.py
beam materials AND src/backends/fourc/generators/beams.py
beam_types topologies) had:

(1) Wrong material name delimiters:

    Catalog (wrong)                Schema (right)
    -----------------------------------------------------
    MAT_Beam_Reissner_ElastHyper   MAT_BeamReissnerElastHyper
    MAT_Beam_Kirchhoff_ElastHyper  MAT_BeamKirchhoffElastHyper
    MAT_Beam_Reissner_ElastPlastic MAT_BeamReissnerElastPlastic

(2) Missing higher-order topologies:

    BEAM3R catalog: LINE2 LINE3 LINE4       — schema: + LINE5
    BEAM3K catalog: LINE2 LINE3             — schema: + LINE4

This fixture walks 4C's compiled JSON schema and asserts:

  * 7 real MAT_Beam* material names are all present in
    the schema's MATERIALS section.
  * The 3 historical wrong underscore-separated names
    are absent.
  * BEAM3R cell types include LINE5.
  * BEAM3K cell types include LINE4.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


REQUIRED_BEAM_MATERIALS = {
    "MAT_BeamReissnerElastHyper",
    "MAT_BeamReissnerElastHyper_ByModes",
    "MAT_BeamReissnerElastPlastic",
    "MAT_BeamKirchhoffElastHyper",
    "MAT_BeamKirchhoffElastHyper_ByModes",
    "MAT_BeamKirchhoffTorsionFreeElastHyper",
    "MAT_BeamKirchhoffTorsionFreeElastHyper_ByModes",
}
HISTORICAL_WRONG_MATERIALS = {
    "MAT_Beam_Reissner_ElastHyper",
    "MAT_Beam_Kirchhoff_ElastHyper",
    "MAT_Beam_Reissner_ElastPlastic",
}


def find_schema() -> Path | None:
    p = (Path.home() / "Schreibtisch" / "4C-src" / "4C"
         / "build" / "4C_schema.json")
    return p if p.is_file() else None


def main() -> int:
    schema_path = find_schema()
    if schema_path is None:
        print("FAIL: 4C_schema.json not found", file=sys.stderr)
        return 2

    with schema_path.open() as f:
        schema = json.load(f)

    # (1) Materials section — collect ALL beam material keys.
    mats_section = (schema.get("properties", {})
                    .get("MATERIALS", {})
                    .get("items", {}))
    found_mat_keys: set[str] = set()
    for e in mats_section.get("oneOf", []):
        for k in e.get("properties", {}):
            if k.startswith("MAT_Beam"):
                found_mat_keys.add(k)
    print(f"schema_beam_materials={sorted(found_mat_keys)}")

    missing_required = REQUIRED_BEAM_MATERIALS - found_mat_keys
    wrong_present = HISTORICAL_WRONG_MATERIALS & found_mat_keys
    print(f"missing_required_materials="
          f"{sorted(missing_required)}")
    print(f"historical_wrong_in_schema="
          f"{sorted(wrong_present)}")

    # (2) BEAM3R / BEAM3K cell types.
    sg = (schema["properties"]["STRUCTURE GEOMETRY"]
                ["properties"]["ELEMENT_BLOCKS"]["items"])

    def cells_for(beam: str) -> list[str]:
        for e in sg.get("oneOf", []):
            title = str(e.get("title", ""))
            if title.startswith(f"{beam},"):
                spec = e["properties"][beam]
                cells: list[str] = []
                # Two layouts: 'oneOf' or 'properties'.
                for c in spec.get("oneOf", []):
                    if c.get("title"):
                        cells.append(c["title"])
                for k in spec.get("properties", {}):
                    cells.append(k)
                return cells
        return []

    b3r = cells_for("BEAM3R")
    b3k = cells_for("BEAM3K")
    b3eb = cells_for("BEAM3EB")
    print(f"beam3r_cells={b3r}")
    print(f"beam3k_cells={b3k}")
    print(f"beam3eb_cells={b3eb}")

    line5_in_b3r = "LINE5" in b3r
    line4_in_b3k = "LINE4" in b3k
    print(f"line5_in_b3r={line5_in_b3r}")
    print(f"line4_in_b3k={line4_in_b3k}")

    ok = (
        not missing_required
        and not wrong_present
        and line5_in_b3r
        and line4_in_b3k
        # BEAM3EB only LINE2:
        and b3eb == ["LINE2"]
    )
    if ok:
        return 0
    print("FAIL: beam catalog invariant not held",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
