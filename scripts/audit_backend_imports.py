"""Cross-backend import-availability audit.

For each registered backend, identify the underlying library each
physics module depends on, then verify that the library is
importable in the environment that backend uses. Produces
``scripts/scan_results/backend_imports.json`` summarising
available vs unreachable physics.

Background (2026-06-01): empirical probing surfaced two
structural alignment bugs:

  1. ``supported_physics()`` is a static list — it does NOT check
     whether the underlying library imports. The DUNE backend
     reports 15 supported physics even though ``import dune.fem``
     fails with a libibverbs version mismatch in the available
     python interpreter.

  2. Kratos has 31 physics in supported_physics(), but ~20 of them
     depend on applications NOT installed in the repo .venv
     (DEMApplication, MPMApplication, IgaApplication, etc.).
     Calling ``prepare_simulation(solver='kratos', physics='dem')``
     would return a template that immediately fails at runtime
     with ImportError on first ``import KratosMultiphysics.DEMApplication``.

This script produces an honest reachable-vs-unreachable view
that future regression tests can compare against. Does NOT
modify backend registration today — that is a follow-up refactor
(blocked on user decision).

Usage:
    python3 scripts/audit_backend_imports.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()


# ───────────────────────────────────────────────────────────────
# Configuration: which python interpreter each backend uses, and
# which python-import string actually proves the library works
# for that backend × physics.
# ───────────────────────────────────────────────────────────────

PYTHON_PATHS = {
    "venv":        str(REPO_ROOT / ".venv" / "bin" / "python"),
    "ofa-fenicsx": str(HOME / "miniconda3" / "envs" / "ofa-fenicsx" / "bin" / "python"),
    "ofa-dealii":  str(HOME / "miniconda3" / "envs" / "ofa-dealii" / "bin" / "python"),
    "ofa-dune":    str(HOME / "miniconda3" / "envs" / "ofa-dune" / "bin" / "python"),
    "system":      sys.executable,
}

# Which python interpreter to use for each backend.
BACKEND_PYTHON = {
    "kratos":  "venv",
    "skfem":   "venv",
    "ngsolve": "venv",
    "fenics":  "ofa-fenicsx",
    "dune":    "ofa-dune",
    "dealii":  None,   # C++ compile path; check Debug build instead
    "fourc":   None,   # Binary path check, not python import
}

# Kratos physics → required application name (from the bulk-promotion
# audit). When a physics is not in this map, only the core
# KratosMultiphysics import is required.
KRATOS_PHYSICS_APP = {
    "linear_elasticity":      "StructuralMechanicsApplication",
    "structural_dynamics":    "StructuralMechanicsApplication",
    "plasticity":             "ConstitutiveLawsApplication",
    "fluid":                  "FluidDynamicsApplication",
    "fluid_biomedical":       "FluidDynamicsApplication",
    "contact":                "ContactStructuralMechanicsApplication",
    "heat":                   "ConvectionDiffusionApplication",
    "heat_transient":         "ConvectionDiffusionApplication",
    "dem":                    "DEMApplication",
    "thermal_dem":            "DEMApplication",
    "swimming_dem":           "DEMApplication",
    "fem_to_dem":             "FemToDemApplication",
    "dem_structures_coupling":"DEMApplication",
    "mpm":                    "MPMApplication",
    "geomechanics":           "GeoMechanicsApplication",
    "poromechanics":          "GeoMechanicsApplication",
    "pfem_fluid":             "PfemFluidDynamicsApplication",
    "pfem":                   "PfemFluidDynamicsApplication",
    "iga":                    "IgaApplication",
    "rans":                   "RANSApplication",
    "shallow_water":          "ShallowWaterApplication",
    "rom":                    "RomApplication",
    "shape_optimization":     "ShapeOptimizationApplication",
    "topology_optimization":  "TopologyOptimizationApplication",
    "compressible_potential": "CompressiblePotentialFlowApplication",
    "compressible_flow":      "CompressiblePotentialFlowApplication",
    "cosimulation":           "CoSimulationApplication",
    "chimera":                "ChimeraApplication",
    "fsi":                    "MappingApplication",
    "wind_engineering":       "FluidDynamicsApplication",
    "fluid_hydraulics":       "FluidDynamicsApplication",
    "optimization":           "OptimizationApplication",
    "fluid_dynamics":         "FluidDynamicsApplication",
}

# Special checks for non-python backends.
NONPYTHON_CHECKS = {
    "dealii": HOME / "Schreibtisch" / "dealii-debug" / "lib" / "libdeal_II.g.so",
    "fourc":  HOME / "Schreibtisch" / "4C-src" / "4C" / "build" / "4C",
}


def _can_run(python_path: str, import_stmt: str, timeout: int = 30) -> tuple[bool, str]:
    """Subprocess-check whether `import_stmt` succeeds in `python_path`.

    Returns (ok, diagnostic). diagnostic is empty on success and the
    first ~200 chars of the import error otherwise.
    """
    if not Path(python_path).is_file():
        return False, f"interpreter not found at {python_path}"
    try:
        r = subprocess.run(
            [python_path, "-c", import_stmt],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except Exception as e:
        return False, f"subprocess error: {e}"
    if r.returncode == 0:
        return True, ""
    err = (r.stderr or r.stdout or "").strip()
    # Filter MPI noise lines (shmem warnings etc.)
    err_lines = [l for l in err.splitlines()
                 if "shmem" not in l and "create_and_attach" not in l]
    return False, "\n".join(err_lines)[:240]


def audit_kratos() -> dict:
    """Kratos: core import + each physics app."""
    py = PYTHON_PATHS["venv"]
    core_ok, core_err = _can_run(py, "import KratosMultiphysics")
    if not core_ok:
        return {
            "interpreter": py,
            "core_importable": False,
            "core_error": core_err,
            "physics_available": [],
            "physics_unreachable": [],
        }
    available = ["StructuralMechanicsApplication",
                 "FluidDynamicsApplication",
                 "ConvectionDiffusionApplication"]  # checked separately below
    available_apps = set()
    for app in set(KRATOS_PHYSICS_APP.values()):
        ok, _ = _can_run(py, f"import KratosMultiphysics.{app}")
        if ok:
            available_apps.add(app)
    physics_available = []
    physics_unreachable = []
    for physics, app in KRATOS_PHYSICS_APP.items():
        if app in available_apps:
            physics_available.append({"physics": physics, "app": app})
        else:
            physics_unreachable.append({"physics": physics, "app": app})
    return {
        "interpreter": py,
        "core_importable": True,
        "core_error": "",
        "available_apps": sorted(available_apps),
        "physics_available": sorted(physics_available,
                                      key=lambda x: x["physics"]),
        "physics_unreachable": sorted(physics_unreachable,
                                        key=lambda x: x["physics"]),
    }


def audit_simple_backend(name: str, import_stmt: str) -> dict:
    py_key = BACKEND_PYTHON.get(name)
    py = PYTHON_PATHS.get(py_key) if py_key else None
    if py is None:
        return {"backend": name, "interpreter": None,
                "importable": None, "error": "no python interpreter mapped"}
    ok, err = _can_run(py, import_stmt)
    return {"backend": name, "interpreter": py,
            "importable": ok, "error": err}


def audit_nonpython_backend(name: str) -> dict:
    target = NONPYTHON_CHECKS.get(name)
    if target is None:
        return {"backend": name, "target": None, "available": None,
                "error": "no nonpython check configured"}
    available = target.exists()
    return {"backend": name, "target": str(target),
            "available": available,
            "error": "" if available else f"file not found: {target}"}


def main():
    report = {
        "_comment": "Backend import-availability audit. Surfaces "
                    "the gap between supported_physics() and what "
                    "the underlying library can actually do. Use "
                    "as data for follow-up registry-validation work.",
        "kratos":  audit_kratos(),
        "skfem":   audit_simple_backend("skfem", "import skfem"),
        "ngsolve": audit_simple_backend("ngsolve", "import ngsolve"),
        "fenics":  audit_simple_backend("fenics", "import dolfinx"),
        "dune":    audit_simple_backend("dune", "import dune.fem"),
        "dealii":  audit_nonpython_backend("dealii"),
        "fourc":   audit_nonpython_backend("fourc"),
    }

    # Headline summary line.
    summary = {}
    k = report["kratos"]
    summary["kratos_physics_available"] = len(k.get("physics_available", []))
    summary["kratos_physics_unreachable"] = len(k.get("physics_unreachable", []))
    for be in ("skfem", "ngsolve", "fenics", "dune"):
        summary[f"{be}_importable"] = bool(report[be].get("importable"))
    for be in ("dealii", "fourc"):
        summary[f"{be}_available"] = bool(report[be].get("available"))
    report["summary"] = summary

    out = REPO_ROOT / "scripts" / "scan_results" / "backend_imports.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"audit written: {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
