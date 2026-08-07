"""Tier-2: ProjectParametersDEM.json is FLAT, not the FEM problem_data schema.

Pitfall (kratos.dem): the OASiS DEM generator emitted the StructuralMechanics
ProjectParameters layout -- a "problem_data" block plus
"solver_settings": {"solver_type": "dem_solver", "time_stepping": {...}}.
Kratos DEM does not read any of that. It wants a flat file whose top level
carries problem_name, FinalTime, MaxTimeStep, Dimension, GravityX/Y/Z and
"solver_settings": {"strategy": "<python module basename>"}.

The first thing that breaks is the strategy lookup, and it breaks in the
DEMAnalysisStage constructor, before a single mdpa byte is read:

  RuntimeError: Error: Getting a value that does not exist. entry string : strategy
  in kratos/sources/kratos_parameters.cpp:426

This fixture builds both decks in-process and asserts the FEM-shaped one raises
on 'strategy' while the flat one gets past solver construction.

Mutation control: T2_MUTATE=1 adds the missing "strategy" key to the FEM-shaped
deck, so the deck that is supposed to fail now constructs a solver. The
assertion machinery is untouched; only the input changes. Mutated,
fem_schema_raises_on_strategy=True disappears and the process exits non-zero.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import KratosMultiphysics as KM
from KratosMultiphysics.DEMApplication.DEM_analysis_stage import DEMAnalysisStage

MUTATE = os.environ.get("T2_MUTATE") == "1"

FEM_SHAPED = {
    "problem_data": {
        "problem_name": "particles",
        "parallel_type": "OpenMP",
        "start_time": 0.0,
        "end_time": 0.5,
    },
    "solver_settings": {
        "solver_type": "dem_solver",
        "model_part_name": "SpheresPart",
        "domain_size": 3,
        "model_import_settings": {"input_type": "mdpa", "input_filename": "particles"},
        "material_import_settings": {"materials_filename": "MaterialsDEM.json"},
        "time_stepping": {"time_step": 1e-05},
    },
}

if MUTATE:
    print("mutation=strategy_key_added_to_the_fem_shaped_deck")
    FEM_SHAPED["solver_settings"]["strategy"] = "sphere_strategy"


def strategy_lookup_fails(params: dict) -> tuple[bool, str]:
    """Return (did it fail on the 'strategy' key, message).

    Runs inside a throwaway directory: once the mutation supplies the missing
    'strategy' key, DEMAnalysisStage gets far enough to open GiD output files,
    and those would otherwise be written into the fixture directory itself.
    TemporaryDirectory follows TMPDIR, which must stay on a real filesystem --
    do not point it at exFAT, which cannot store symlinks.
    """
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as workdir:
        try:
            os.chdir(workdir)
            p = KM.Parameters(json.dumps(params))
            DEMAnalysisStage(KM.Model(), p)
        except Exception as exc:  # noqa: BLE001 - we are classifying the failure
            msg = str(exc).replace("\n", " ")
            return ("entry string : strategy" in msg), msg
        finally:
            os.chdir(cwd)
    return False, "<no exception>"


def main() -> int:
    failed_on_strategy, msg = strategy_lookup_fails(FEM_SHAPED)
    print(f"fem_schema_raises_on_strategy={failed_on_strategy}")
    print(f"first_message={msg[:120]}")

    if not failed_on_strategy:
        print("MISMATCH: the FEM-shaped deck did not fail on the 'strategy' key",
              file=sys.stderr)
        return 1

    print("dem_flat_schema_check=ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
