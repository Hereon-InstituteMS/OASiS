"""Kratos eletype scanner — diff catalog element/condition names vs the
canonical registered list.

Motivation (2026-06-01 audit campaign):
  • poisson #4 — LaplacianElement2D3N is registered (eletype string
    factory only, NOT a Python attr on CDA)
  • poisson #5 — LaplacianElement DOES assemble HEAT_FLUX (the
    'does NOT' catalog claim was wrong)
  • contact #10 — ALMFrictionlessMortarContact base name NOT
    registered; needs 'Condition' suffix + shape (e.g.
    ALMFrictionlessMortarContactCondition2D2N)
  • mpm #11 — UpdatedLagrangianPQ2D NOT registered; needs MPM
    prefix (e.g. MPMUpdatedLagrangian2D4N)

Each of those was discovered by hand-probing one physics. This
scanner automates the same diff for every Kratos catalog entry
so future drift surfaces in a single audit run.

Approach: Kratos error machinery prints the canonical list of
registered names when an unknown name is passed to
model_part.CreateNewElement / CreateNewCondition. The scanner
deliberately calls those with a sentinel name, captures the
error text, and parses the registered list out of it. Then it
extracts every <UpperCase>...<lower> token from the catalog
strings and checks each is reachable.

Usage::

    .venv/bin/python scripts/kratos_eletype_scanner.py

Outputs a JSON-ish summary to stdout and exits 0 (informational
only — does not assert).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "data"))


_TOKEN_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]{4,}(?:[0-9]D[0-9]N(?:[0-9]N)?)?)\b")


def _capture_registered(model_part, kind: str,
                         props_id: int) -> list[str]:
    """Probe Kratos for the canonical list of registered names."""
    fn = (model_part.CreateNewElement
          if kind == "Element"
          else model_part.CreateNewCondition)
    props = model_part.CreateNewProperties(props_id)
    sentinel = f"NOT_AN_{kind.upper()}_SENTINEL_XYZ"
    try:
        fn(sentinel, 999_999_999, [1, 2, 3, 4], props)
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        names: list[str] = []
        in_list = False
        for line in text.splitlines():
            line = line.strip()
            if "are registered" in line:
                in_list = True
                continue
            if in_list:
                if not line or line.startswith("in kratos"):
                    break
                names.append(line)
        return sorted(set(names))
    return []


def _gather_catalog_names(text: str) -> set[str]:
    return {m.group(1) for m in _TOKEN_RE.finditer(text)}


def main() -> int:
    import KratosMultiphysics as KM

    # Sub-applications to probe — extend as new ones come up.
    apps_to_import = [
        "KratosMultiphysics.StructuralMechanicsApplication",
        "KratosMultiphysics.FluidDynamicsApplication",
        "KratosMultiphysics.MPMApplication",
        "KratosMultiphysics.DEMApplication",
        "KratosMultiphysics.MeshMovingApplication",
        "KratosMultiphysics.MappingApplication",
        "KratosMultiphysics.LinearSolversApplication",
        "KratosMultiphysics.ContactStructuralMechanicsApplication",
        "KratosMultiphysics.ConvectionDiffusionApplication",
        "KratosMultiphysics.RANSApplication",
        "KratosMultiphysics.ShallowWaterApplication",
        "KratosMultiphysics.PoromechanicsApplication",
        "KratosMultiphysics.CompressiblePotentialFlowApplication",
        "KratosMultiphysics.GeoMechanicsApplication",
        "KratosMultiphysics.IgaApplication",
    ]
    import importlib
    imported: list[str] = []
    for app in apps_to_import:
        try:
            importlib.import_module(app)
            imported.append(app)
        except ImportError:
            pass

    # Set up a probe model part with enough nodes for any geometry.
    mp = KM.Model().CreateModelPart("probe")
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    mp.AddNodalSolutionStepVariable(KM.VELOCITY)
    mp.AddNodalSolutionStepVariable(KM.PRESSURE)
    mp.AddNodalSolutionStepVariable(KM.TEMPERATURE)
    mp.AddNodalSolutionStepVariable(KM.HEAT_FLUX)
    mp.AddNodalSolutionStepVariable(KM.NORMAL)
    mp.AddNodalSolutionStepVariable(KM.CONDUCTIVITY)
    mp.AddNodalSolutionStepVariable(KM.DENSITY)
    mp.AddNodalSolutionStepVariable(KM.SPECIFIC_HEAT)
    mp.AddNodalSolutionStepVariable(KM.BODY_FORCE)
    mp.AddNodalSolutionStepVariable(KM.NODAL_H)
    mp.AddNodalSolutionStepVariable(KM.MESH_VELOCITY)
    mp.AddNodalSolutionStepVariable(KM.ACCELERATION)
    for k in range(1, 11):
        mp.CreateNewNode(k, float(k % 4), float(k // 4), 0.0)

    registered_elements = _capture_registered(mp, "Element", 91)
    registered_conditions = _capture_registered(mp, "Condition", 92)

    # Walk catalog generators
    gen_dir = REPO / "src" / "backends" / "kratos" / "generators"
    catalog_names: dict[str, set[str]] = {}
    for path in sorted(gen_dir.iterdir()):
        if not (path.suffix == ".py" and not path.name.startswith("_")
                and "base" not in path.name):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue
        catalog_names[path.stem] = _gather_catalog_names(text)

    # Determine which catalog names look like element/condition
    # types: heuristic — at least one digit-D-digit-N pattern.
    # Bare "Element" / "Condition" / "Mortar" tokens are
    # excluded as prose nouns (they occur all over Kratos docs).
    # CreateNewElement / CreateNewCondition are API verbs.
    _PROSE_FALSE_POSITIVES = {
        "CreateNewElement", "CreateNewCondition",
        "Element", "Condition", "Mortar",
        "SubModelPart", "SubModelParts",
        "ResidualBasedNewtonRaphsonStrategy",
        "BuilderAndSolver", "LinearSolver",
        "ConstitutiveLaw", "ResidualCriteria",
        "DisplacementCriteria",
        # Generic structural element names that Kratos resolves
        # through aliases at the Application's Register() call
        # rather than direct CreateNewElement keys:
        "SmallDisplacementElement", "TotalLagrangianElement",
        "UpdatedLagrangianElement",
    }

    def looks_like_eletype(n: str) -> bool:
        if n in _PROSE_FALSE_POSITIVES:
            return False
        return bool(re.search(r"[0-9]D[0-9]N", n))

    drift: dict[str, list[str]] = {}
    for physics, names in catalog_names.items():
        suspicious = sorted(n for n in names if looks_like_eletype(n))
        unreachable = sorted(
            n for n in suspicious
            if n not in registered_elements
            and n not in registered_conditions)
        if unreachable:
            drift[physics] = unreachable

    print(json.dumps({
        "apps_imported": imported,
        "n_registered_elements": len(registered_elements),
        "n_registered_conditions": len(registered_conditions),
        "catalog_drift_by_physics": drift,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
