"""Tier-2: Kratos CoSimulation accelerator / mapper catalog names.

Catalog under audit:

  convergence_accelerators:
    constant_relaxation, aitken, mvqn,
    ibqn               ← imprecise; real names: block_ibqnls,
                          iqnils (and block_mvqn, mvqn)
    anderson

  data_transfer:
    nearest_neighbor, barycentric,
    kratos_mapping     ← that is the FILE name, not a mapper
                          type (real types: nearest_neighbor,
                          nearest_element, barycentric,
                          coupling_geometry, radial_basis_function)
    empire_mapping     ← NOT registered anywhere in
                          KratosMappingApplication 10.4.2
    radial_basis_function

This fixture verifies:
  * The 5 catalog convergence_accelerator file names that ARE
    present under
    KratosMultiphysics/CoSimulationApplication/
    convergence_accelerators/
  * The 5 catalog data_transfer claims against
    libKratosMappingCore.so binary strings.
  * 'empire' is absent from the mapping binary.
  * 'ibqn' as a literal accelerator file is absent (the real
    file is block_ibqnls.py or iqnils.py).

Also: KratosCoSimulationApplication was missing from the
.venv until pip installed during this probe.

Mutation control: the pathology is the CATALOG's two imprecise names -- 'ibqn'
(the real accelerator file is block_ibqnls.py / iqnils.py) and the 'empire'
mapper family (registered nowhere in libKratosMappingCore.so). Removing it
means auditing the CORRECTED catalog, and that is exactly what T2_MUTATE=1
does: the same directory listing and the same whole-word `strings` scan are
re-run against the names the catalog should have carried, block_ibqnls and
nearest_element, both of which really are there. The measurement stays real
and simply comes out the other way, so ibqn_literal_present=False and
empire_anywhere_in_mapping_binary=False disappear.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import KratosMultiphysics  # noqa: F401
import KratosMultiphysics.CoSimulationApplication  # noqa: F401
import KratosMultiphysics.MappingApplication  # noqa: F401

MUTATE = os.environ.get("T2_MUTATE") == "1"
# The two names under audit. They are the catalog's, not Kratos's.
ACCEL_UNDER_AUDIT = "ibqn"
MAPPER_FAMILY_UNDER_AUDIT = "empire"
if MUTATE:
    print("mutation=audited_names_replaced_by_the_corrected_catalog_names")
    ACCEL_UNDER_AUDIT = "block_ibqnls"
    MAPPER_FAMILY_UNDER_AUDIT = "nearest_element"


def main() -> int:
    # (1) Convergence accelerator file presence
    accel_dir = (Path(KratosMultiphysics.__file__).parent
                 / "CoSimulationApplication"
                 / "convergence_accelerators")
    if not accel_dir.is_dir():
        print(f"FAIL: dir not found {accel_dir}", file=sys.stderr)
        return 2
    files = {f.stem for f in accel_dir.iterdir() if f.suffix == ".py"}
    print(f"accel_files={sorted(files)}")

    catalog_accel = {
        "constant_relaxation", "aitken", "mvqn",
        ACCEL_UNDER_AUDIT, "anderson"}
    catalog_present = {n for n in catalog_accel if n in files}
    print(f"catalog_accel_in_files={sorted(catalog_present)}")
    ibqn_present = ACCEL_UNDER_AUDIT in files
    print(f"{ACCEL_UNDER_AUDIT}_literal_present={ibqn_present}")
    block_ibqnls_present = "block_ibqnls" in files
    print(f"block_ibqnls_present={block_ibqnls_present}")

    # (2) Mapping binary string scan for mapper type names.
    # Scan the .libs of the very install that was just imported above. An
    # absolute path to one particular checkout's .venv went stale when that
    # checkout was renamed, and the fixture then reported "mapping .so not
    # found" instead of measuring anything.
    so = (Path(KratosMultiphysics.__file__).parent / ".libs"
          / "libKratosMappingCore.so")
    if not so.is_file():
        print(f"FAIL: mapping .so not found at {so}", file=sys.stderr)
        return 2
    text = subprocess.run(
        ["strings", str(so)], capture_output=True,
        text=True, check=True).stdout

    catalog_mappers = {
        "nearest_neighbor", "barycentric",
        "kratos_mapping", "empire_mapping",
        "radial_basis_function",
    }
    mapper_present = {n for n in catalog_mappers
                      if re.search(r"\b" + re.escape(n) + r"\b",
                                   text)}
    print(f"catalog_mappers_present_in_binary="
          f"{sorted(mapper_present)}")
    # 'empire' base must be absent
    empire_anywhere = bool(re.search(
        r"\b" + re.escape(MAPPER_FAMILY_UNDER_AUDIT) + r"\w*", text))
    print(f"{MAPPER_FAMILY_UNDER_AUDIT}_anywhere_in_mapping_binary="
          f"{empire_anywhere}")
    # nearest_element + coupling_geometry must be present
    nearest_element_present = bool(re.search(
        r"\bnearest_element\b", text))
    coupling_geom_present = bool(re.search(
        r"\bcoupling_geometry\b", text))
    print(f"nearest_element_present={nearest_element_present}")
    print(f"coupling_geometry_present={coupling_geom_present}")

    ok = (
        # Real accel files present:
        {"constant_relaxation", "aitken", "mvqn", "anderson"}
        <= files
        and block_ibqnls_present
        and not ibqn_present
        # Mappers: catalog's 3 right names present:
        and {"nearest_neighbor", "barycentric",
             "radial_basis_function"} <= mapper_present
        # 'empire_mapping' absent:
        and not empire_anywhere
        # Real names the catalog missed are present:
        and nearest_element_present
        and coupling_geom_present
    )
    if ok:
        return 0
    print("FAIL: cosim catalog invariant not held",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
