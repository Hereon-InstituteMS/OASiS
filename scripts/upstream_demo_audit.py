"""Upstream-demo coverage audit.

For each backend, locate the installed package's demo / example /
tutorial directory and produce a precise inventory of physics topics
present upstream. Cross-reference against the backend's
supported_physics() to surface gaps.

Output: data/upstream_coverage.json + a human-readable report.

This is the infrastructure backbone for the user-stated goal: "the
knowledge and the generators need to cover ALL of the upstream
current backend source capabilities". By regenerating the JSON each
session you can see which physics are still missing — concretely,
file-by-file — rather than relying on hand-rolled estimates.

For each backend the script:
  1. Locates the install-time demos directory (conda envs, site-
     packages, source-tree tutorials).
  2. Walks all demo/example/tutorial files.
  3. Normalises each filename to a canonical physics topic via a
     simple substring map (poisson, stokes, etc.).
  4. Records: file path, canonical topic, raw stem.
  5. Cross-references against the backend's supported_physics().
  6. Reports per-backend: (a) topics covered both upstream and in
     our catalog (b) topics upstream but NOT in our catalog (the
     gap) (c) topics in our catalog but NOT in upstream demos
     (these are usually fine — we extend the catalog, but worth
     auditing for phantoms).

This is *NOT* a runtime verification — it's a topic-name audit. A
topic appearing in both the upstream demo dir and our catalog does
NOT prove the catalog content is correct; that's what Tier-0/1/2
gates do. This audit just ensures we're not silently MISSING
upstream-supported physics.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


# Canonical-topic substring map — strings on the LEFT are looked up
# against the lowercased filename stem (with dashes/underscores
# normalised to underscores). First match wins. The RHS is the
# canonical physics name the catalog uses.
#
# When upstream evolves (new demo added), this map may need new
# entries; the audit will surface the un-matched stems in the
# "unmapped" bucket so you can extend the map deliberately.
_TOPIC_MAP: dict[str, str] = {
    "poisson_matrix_free":   "matrix_free_poisson",
    "poisson":               "poisson",
    "elasticity":            "linear_elasticity",
    "hyperelasticity":       "hyperelasticity",
    "biharmonic":            "biharmonic",
    "stokes":                "stokes",
    "navier_stokes":         "navier_stokes",
    "navier-stokes":         "navier_stokes",
    "advection_dg":          "advection_dg",
    "advection-dg":          "advection_dg",
    "advection":             "advection_dg",
    "helmholtz":             "helmholtz",
    "wave":                  "wave",
    "eigenvalue":            "eigenvalue",
    "heat":                  "heat",
    "cahn_hilliard":         "cahn_hilliard",
    "cahn-hilliard":         "cahn_hilliard",
    "maxwell":               "maxwell",
    "phase_field":           "phase_field",
    "phase-field":           "phase_field",
    "hdg":                   "hdg",
    "dg":                    "dg_methods",
    "mixed_poisson":         "mixed_poisson",
    "mixed-poisson":         "mixed_poisson",
    "mixed":                 "mixed_methods",
    "static_condensation":   "static_condensation",
    "static-condensation":   "static_condensation",
    "waveguide":             "waveguide",
    "scattering":            "scattering",
    "pml":                   "pml",
    "axis":                  "axisymmetric",
    "axisymmetric":          "axisymmetric",
    "contact":               "contact",
    "obstacle":              "obstacle_problem",
    "multigrid":             "multigrid",
    "matrix_free":           "matrix_free",
    "matrix-free":           "matrix_free",
    "lagrange_variants":     "lagrange_variants",
    "tnt":                   "tnt_elements",
    "compressible_euler":    "compressible_euler",
    "topology":              "topology_opt",
    "optimal_control":       "optimal_control",
    "stokes_darcy":          "stokes_darcy",
    "darcy":                 "stokes_darcy",
    "biphasic":              "biphasic",
    "multiphasic":           "multiphasic",
    "plate":                 "biharmonic",
    "shell":                 "shell",
    "beam":                  "beam",
    "membrane":              "membrane",
    "fluid":                 "navier_stokes",
    "thermal":               "heat",
    "thermomechanical":      "thermal_structural",
    "thermo":                "thermal_structural",
    "fsi":                   "fsi",
    "tsi":                   "tsi",
    "plasticity":            "plasticity",
    "viscoelastic":          "viscoelasticity",
    "reaction_diffusion":    "reaction_diffusion",
    "reaction-diffusion":    "reaction_diffusion",
    "convection_diffusion":  "convection_diffusion",
    "convection-diffusion":  "convection_diffusion",
    "mhd":                   "mhd",
    "magnetostatic":         "magnetostatics",
    "particle":              "particle_methods",
    "level_set":             "level_set",
    "level-set":             "level_set",
    "fracture":              "fracture",
    "growth":                "growth_remodeling",
    "remodel":               "growth_remodeling",
    "nonlinear":             "nonlinear",
    "adaptive":              "adaptive_poisson",
    "amr":                   "adaptive_poisson",
}

# Stems to skip — utility / infrastructure / IO demos that don't
# represent a distinct physics. The audit notes them but doesn't
# count them as gaps.
_NON_PHYSICS_STEMS: set[str] = {
    "comm_pattern", "gmsh", "interpolation", "interpolation_io",
    "mixed_topology", "types", "pyvista", "pyamg", "data",
    "conftest", "test", "p_mesh_tags", "mesh_tags", "custom_mesh",
    "custom_assembler", "io", "performance", "demo_io",
    "interpolation-io", "demo_interpolation",
}


def _backend_demo_dirs(backend: str) -> list[Path]:
    """Return all candidate demo/example/tutorial directories for
    the named backend. Empty list = not installed / nothing to scan."""
    home = Path.home()
    envs = home / "miniconda3" / "envs"
    candidates: list[Path] = []
    if backend == "fenics":
        candidates += [
            envs / "ofa-fenicsx" / "etc" / "conda"
            / "test-files" / "fenics-dolfinx" / "0"
            / "python" / "demo",
        ]
    elif backend == "skfem":
        # scikit-fem ships examples in its source tree, but the
        # pip wheel doesn't include docs/examples.  Hardcode the
        # known upstream list from the repo (ex01..ex50) — this is
        # what `pip install scikit-fem-docs` would unpack.
        return []
    elif backend == "ngsolve":
        for env in ("ofa-ngsolve", "ofa-fenicsx"):
            for sub in ("ngsolve", "netgen"):
                pat = envs / env / "lib"
                if pat.is_dir():
                    for d in pat.glob("python*/site-packages/" + sub):
                        candidates.append(d)
    elif backend == "dealii":
        for env in ("ofa-dealii",):
            for d in (envs / env / "share").rglob("examples"):
                candidates.append(d)
            for d in (envs / env / "share").rglob("tutorial"):
                candidates.append(d)
    elif backend == "dune":
        candidates += [
            envs / "ofa-dune" / "lib" / "python3.12"
            / "site-packages" / "dune" / "grid" / "tutorial",
        ]
    elif backend == "kratos":
        # Kratos publishes its applications as separate dirs; rely
        # on the per-app source scans done elsewhere.  This audit
        # focuses on demo/tutorial dirs that bundle physics.
        return []
    elif backend == "fourc":
        # 4C tests live in tests/input_files in the source tree;
        # build path is /home/hermann/.../4C/tests/input_files.
        # Treat as available if path exists.
        for cand in [
            home / "Schreibtisch" / "4C" / "tests" / "input_files",
            home / "4C" / "tests" / "input_files",
        ]:
            if cand.is_dir():
                candidates.append(cand)
    elif backend == "febio":
        # FEBio ships verification cases; no Python pip layout.
        return []
    return [c for c in candidates if c.is_dir()]


def _normalize_stem(stem: str) -> str:
    """Normalise a demo filename stem to the lookup form."""
    s = stem.lower()
    if s.startswith("demo_"):
        s = s[5:]
    if s.startswith("ex_"):
        s = s[3:]
    s = s.replace("-", "_")
    return s


def _classify(stem: str) -> str | None:
    """Return canonical topic name for the stem, or None if it
    cannot be classified. Returns the sentinel "__skip__" for
    explicitly non-physics demos.

    Matches LONGEST substring key first — so e.g. `half_loaded_
    waveguide` matches `waveguide` (len 9), not `wave` (len 4).
    Without this discipline `wave` shadows every demo containing
    the letters w-a-v-e."""
    norm = _normalize_stem(stem)
    if norm in _NON_PHYSICS_STEMS:
        return "__skip__"
    for needle in sorted(_TOPIC_MAP.keys(), key=len, reverse=True):
        if needle in norm:
            return _TOPIC_MAP[needle]
    return None


def _scan_one(backend: str) -> dict:
    dirs = _backend_demo_dirs(backend)
    found: list[dict] = []
    unmapped: list[dict] = []
    skipped: list[str] = []
    for d in dirs:
        for f in sorted(d.rglob("*.py")):
            stem = f.stem
            cls = _classify(stem)
            if cls == "__skip__":
                skipped.append(stem)
                continue
            if cls is None:
                unmapped.append({"file": str(f.relative_to(d)),
                                 "stem": stem})
                continue
            found.append({"file": str(f.relative_to(d)),
                          "stem": stem, "topic": cls})
        # Also pick up .prm / .cc / .feb (per-backend extensions).
        for ext in (".cc", ".prm", ".cpp", ".feb"):
            for f in sorted(d.rglob(f"*{ext}")):
                stem = f.stem
                cls = _classify(stem)
                if cls == "__skip__":
                    skipped.append(stem)
                    continue
                if cls is None:
                    unmapped.append({"file": str(f.relative_to(d)),
                                     "stem": stem})
                    continue
                found.append({"file": str(f.relative_to(d)),
                              "stem": stem, "topic": cls})

    topics_upstream = sorted({entry["topic"] for entry in found})
    return {
        "demo_dirs": [str(d) for d in dirs],
        "found": found,
        "unmapped": unmapped,
        "skipped": sorted(set(skipped)),
        "topics_upstream": topics_upstream,
    }


def main() -> int:
    from core.registry import load_all_backends, get_backend
    load_all_backends()

    backends = ("fenics", "dealii", "ngsolve", "skfem", "kratos",
                "dune", "fourc", "febio")
    out: dict[str, dict] = {}
    for be in backends:
        scan = _scan_one(be)
        # Compare with the live backend catalog.
        backend_obj = get_backend(be)
        catalog_topics = (
            sorted(p.name for p in backend_obj.supported_physics())
            if backend_obj else []
        )
        upstream_set = set(scan["topics_upstream"])
        catalog_set = set(catalog_topics)
        scan["catalog_topics"] = catalog_topics
        scan["gap_upstream_not_in_catalog"] = sorted(
            upstream_set - catalog_set)
        scan["extra_in_catalog_not_upstream"] = sorted(
            catalog_set - upstream_set)
        scan["coverage_pct"] = (
            round(100.0 * len(upstream_set & catalog_set)
                  / max(1, len(upstream_set)), 1)
            if upstream_set else None)
        out[be] = scan

    output_path = _REPO / "data" / "upstream_coverage.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))

    # Human-readable report.
    print(f"\nUpstream coverage audit — written to {output_path}\n")
    print(f"{'backend':<10} {'demos':>6} {'topics':>7} "
          f"{'catalog':>8} {'gap':>5}  {'cov%':>6}")
    print("-" * 60)
    for be, scan in out.items():
        n_demos = len(scan["found"])
        n_topics = len(scan["topics_upstream"])
        n_catalog = len(scan["catalog_topics"])
        n_gap = len(scan["gap_upstream_not_in_catalog"])
        cov = (f"{scan['coverage_pct']:5.1f}%"
               if scan["coverage_pct"] is not None else "  N/A ")
        print(f"{be:<10} {n_demos:>6} {n_topics:>7} "
              f"{n_catalog:>8} {n_gap:>5}  {cov}")

    # Detail gaps per backend.
    print("\n--- gaps (upstream demos not in catalog) ---\n")
    for be, scan in out.items():
        if scan["gap_upstream_not_in_catalog"]:
            print(f"  {be}: {scan['gap_upstream_not_in_catalog']}")
        elif scan["demo_dirs"]:
            print(f"  {be}: 0 gaps")
        else:
            print(f"  {be}: no demo dir found / not scanned")

    print("\n--- unmapped upstream stems ---\n")
    for be, scan in out.items():
        if scan["unmapped"]:
            print(f"  {be}: "
                  f"{[u['stem'] for u in scan['unmapped'][:8]]}"
                  + (" ..." if len(scan["unmapped"]) > 8 else ""))

    return 0


if __name__ == "__main__":
    sys.exit(main())
