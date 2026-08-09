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
import os
import sys
from pathlib import Path


# MUTATION CONTROL.  This fixture reads 4C's own compiled input description and
# reports what it finds there, so the pathology cannot be injected into 4C from
# outside.  What CAN be removed is the assumption that the probe measures at
# all: T2_MUTATE=1 asks the SAME artefact for the names the old (wrong) catalog
# used and for cell types the schema does not list.  All four expectations flip
# — missing_required_materials and historical_wrong_in_schema become non-empty,
# line5_in_b3r and line4_in_b3k become False — which is only possible if every
# one of the four is computed against the artefact rather than printed.
MUTATE = os.environ.get("T2_MUTATE") == "1"

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
# The two probe cell types, named so the mutation can move them.
B3R_PROBE_CELL = "LINE5"
B3K_PROBE_CELL = "LINE4"

if MUTATE:
    # Demand the underscore-separated names 4C does NOT define, treat the
    # CamelCase names it DOES define as "historical wrong", and probe for
    # topologies past the end of each beam's list.
    REQUIRED_BEAM_MATERIALS = (REQUIRED_BEAM_MATERIALS
                               | HISTORICAL_WRONG_MATERIALS)
    HISTORICAL_WRONG_MATERIALS = {"MAT_BeamReissnerElastHyper"}
    B3R_PROBE_CELL = "LINE6"
    B3K_PROBE_CELL = "LINE9"


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
    needs the build venv; on a host where that step did not run, the fixture
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


def _walk_named(node, out: set[str]) -> None:
    """Collect every `name` in a metadata spec tree that names a beam material."""
    if isinstance(node, dict):
        n = node.get("name")
        if isinstance(n, str) and n.startswith("MAT_Beam"):
            out.add(n)
        for v in node.values():
            _walk_named(v, out)
    elif isinstance(node, list):
        for v in node:
            _walk_named(v, out)


def read_metadata(path: Path) -> tuple[set[str], dict[str, list[str]]]:
    import yaml
    try:
        loader = yaml.CSafeLoader
    except AttributeError:                                # pragma: no cover
        loader = yaml.SafeLoader
    with path.open() as f:
        meta = yaml.load(f, Loader=loader)
    mats: set[str] = set()
    _walk_named(meta.get("sections", {}), mats)
    cells = {k: [e.get("cell_type") for e in (v or [])]
             for k, v in (meta.get("legacy_element_specs") or {}).items()}
    return mats, cells


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
        # (1) Materials section — collect ALL beam material keys.
        mats_section = (schema.get("properties", {})
                        .get("MATERIALS", {})
                        .get("items", {}))
        found_mat_keys: set[str] = set()
        for e in mats_section.get("oneOf", []):
            for k in e.get("properties", {}):
                if k.startswith("MAT_Beam"):
                    found_mat_keys.add(k)
        meta_cells: dict[str, list[str]] = {}
        print(f"artefact=4C_schema.json")
    else:
        found_mat_keys, meta_cells = read_metadata(meta_path)
        schema = {}
        print(f"artefact=4C_metadata.yaml")
    print(f"schema_beam_materials={sorted(found_mat_keys)}")

    missing_required = REQUIRED_BEAM_MATERIALS - found_mat_keys
    wrong_present = HISTORICAL_WRONG_MATERIALS & found_mat_keys
    print(f"missing_required_materials="
          f"{sorted(missing_required)}")
    print(f"historical_wrong_in_schema="
          f"{sorted(wrong_present)}")

    # (2) BEAM3R / BEAM3K cell types.
    sg = ((schema.get("properties", {}).get("STRUCTURE GEOMETRY", {})
                 .get("properties", {}).get("ELEMENT_BLOCKS", {})
                 .get("items", {})) if schema else {})

    def cells_for(beam: str) -> list[str]:
        if not schema:
            return list(meta_cells.get(beam, []))
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

    line5_in_b3r = B3R_PROBE_CELL in b3r
    line4_in_b3k = B3K_PROBE_CELL in b3k
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
