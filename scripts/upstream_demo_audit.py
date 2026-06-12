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
    # Infrastructure / educational tutorials — not physics.
    "step_1", "step_2", "step_10", "step_13", "step_49",
    "step_53", "step_54", "step_65",
    "step_85",  # generic matrix-free intro
}


# ─────────────────────────────────────────────────────────────────
# CURATED UPSTREAM INVENTORIES
#
# Some backends ship their demos/examples in their SOURCE tree, not
# in the binary install. For those, hand-curate the upstream
# inventory here. Keep these dicts in sync with upstream:
#
#   * skfem: https://github.com/kinnala/scikit-fem/tree/master/docs/examples
#   * dealii: https://www.dealii.org/current/doxygen/deal.II/Tutorial.html
#   * ngsolve: https://docu.ngsolve.org/latest/i-tutorials/
#   * kratos: kratos/applications/<App>/README.md
#   * dune-fem: dune-fem/dune/fem/examples/
#   * fourc: 4C/tests/input_files/ (no public listing — derive from
#     local checkout if present)
#   * febio: FEBio User's Manual modules
#
# Each entry maps a stem-like key (e.g. "ex02") to the canonical
# physics topic it demonstrates. Same TOPIC_MAP semantics apply
# downstream.
# ─────────────────────────────────────────────────────────────────

_SKFEM_UPSTREAM: dict[str, str] = {
    # scikit-fem v12 docs/examples/, sourced from upstream README.
    # When upstream adds ex51+, append here.
    "ex01": "poisson",
    "ex02": "biharmonic",         # Kirchhoff plate via Morley
    "ex03": "linear_elasticity",  # 3D elasticity
    "ex04": "contact",            # Linearized contact
    "ex05": "biharmonic",         # Argyris plate
    "ex06": "stokes",
    "ex07": "dg_methods",
    "ex08": "hyperelasticity",
    "ex09": "wave",               # 3D wave
    "ex10": "nonlinear",          # Minimal surface
    "ex11": "adaptive_poisson",
    "ex12": "post_processing",    # edge / facet I/O
    "ex13": "mixed_poisson",
    "ex14": "poisson",            # inhomogeneous BC
    "ex15": "poisson",            # 1D smooth solution
    "ex16": "linear_elasticity",  # VTK output
    "ex17": "point_source",
    "ex18": "stokes",             # stabilized
    "ex19": "heat_transient",
    "ex20": "stokes",             # creeping flow Taylor-Hood
    "ex21": "eigenvalue",         # structural eigenvalues
    "ex22": "adaptive_poisson",   # residual estimator
    "ex23": "nonlinear",          # Bratu-Gelfand
    "ex24": "stokes",             # driven cavity
    "ex25": "convection_diffusion",
    "ex26": "restriction_matrix",
    "ex27": "navier_stokes",      # backward-facing step
    "ex28": "navier_stokes",      # lid-driven cavity penalty
    "ex29": "hydraulic_resistance",
    "ex30": "stokes",             # Krylov-Uzawa
    "ex31": "curved_elements",
    "ex32": "preconditioning",
    "ex33": "linear_elasticity",  # 3D sphere with cube hole
    "ex34": "biharmonic",         # Euler-Bernoulli beam
    "ex35": "transmission_line",  # characteristic impedance
    "ex36": "wave",
    "ex37": "mixed_methods",
    "ex38": "point_source",
    "ex39": "schrodinger",
    "ex40": "stokes",             # Hagen-Poiseuille
    "ex41": "biharmonic",         # Mindlin-Reissner plate
    "ex42": "periodic_mesh",
    "ex43": "hyperelasticity",    # torsion
    "ex44": "wave",
    "ex45": "poisson",            # 1D manufactured
    "ex46": "hyperelasticity",    # 3D incompressible
    "ex47": "stokes",             # 3D
    "ex48": "p2_meshes",
    "ex49": "biharmonic",         # Reissner-Mindlin mixed
    "ex50": "operator_splitting",
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


_DEALII_UPSTREAM: dict[str, str] = {
    # deal.II tutorial steps from the published Tutorial.html.
    # Coverage: step-1..step-50 (the canonical 50). step-51..step-97
    # are advanced / research-grade; encode incrementally as you
    # validate.
    # step-1 / step-2 are infrastructure (mesh + dof_handler intro)
    # — not encoded here as physics; remove if upstream tutorials
    # docs add new step-N entries that are pure infrastructure.
    "step-3":  "poisson",
    "step-4":  "poisson",                # dimension-independent
    "step-5":  "poisson",                # variable coefficient
    "step-6":  "adaptive_poisson",       # h-adaptive
    "step-7":  "helmholtz",
    "step-8":  "linear_elasticity",
    "step-9":  "advection_dg",           # advection upwind
    # step-10 / step-13 / step-49 / step-53 / step-54 / step-65 are
    # manifold/design infrastructure — not physics.
    "step-11": "poisson",                # mean-value constraint
    "step-12": "advection_dg",           # DG upwind
    "step-14": "error_estimation",
    "step-15": "nonlinear",              # Newton min-surface
    "step-16": "multigrid",
    "step-17": "linear_elasticity",      # MPI parallel
    "step-18": "linear_elasticity",      # nonlinear large-def
    "step-19": "particle_methods",
    "step-20": "mixed_poisson",          # Raviart-Thomas
    "step-21": "darcy",                  # two-phase porous
    "step-22": "stokes",                 # Taylor-Hood
    "step-23": "wave",                   # Newmark
    "step-24": "wave",                   # absorbing BC
    "step-25": "schrodinger",            # Sine-Gordon
    "step-26": "heat",                   # AMR in time
    "step-27": "hp_adaptive",
    "step-28": "neutron_transport",
    "step-29": "helmholtz",              # complex
    "step-30": "advection_dg",           # anisotropic refinement
    "step-31": "navier_stokes",          # Boussinesq
    "step-32": "navier_stokes",          # parallel mantle convection
    "step-33": "compressible_euler",
    "step-34": "boundary_integral",
    "step-35": "stokes_tangential_bc",
    "step-36": "eigenvalue",             # SLEPc
    "step-37": "matrix_free",            # CG matrix-free
    "step-38": "surface_pde",            # Laplace-Beltrami
    "step-39": "advection_dg",           # interior penalty
    "step-40": "linear_elasticity",      # parallel large
    "step-41": "obstacle_problem",
    "step-42": "obstacle_problem",       # elastoplastic
    "step-43": "navier_stokes",          # porous + DG
    "step-44": "hyperelasticity",        # quasi-static
    "step-45": "periodic_mesh",
    "step-46": "fsi",
    "step-47": "biharmonic",             # C0-IPG
    "step-48": "wave",                   # parallel + matrix-free
    "step-50": "multigrid",              # parallel + matrix-free
    "step-51": "hdg",
    "step-52": "heat_transient",         # method-of-lines
    "step-55": "stokes",                 # parallel block
    "step-56": "multigrid",              # Stokes block
    "step-57": "navier_stokes",
    "step-58": "schrodinger",
    "step-59": "matrix_free",            # DG matrix-free
    "step-60": "non_matching_grids",
    "step-61": "hdg",
    "step-62": "scattering",
    "step-63": "multigrid",
    "step-64": "matrix_free",            # CUDA matrix-free
    "step-66": "fsi",                    # parallel matrix-free
    "step-67": "compressible_euler",
    "step-68": "particle_methods",       # advected particles
    "step-69": "shallow_water",
    "step-70": "particle_methods",       # immersed
    "step-71": "plasticity",             # constitutive update
    "step-72": "plasticity",             # AD
    "step-73": "scattering",             # FE-DGQ
    "step-74": "advection_dg",           # SIPG
    "step-75": "hp_adaptive",            # parallel
    "step-76": "linear_elasticity",      # MPI shared-mem
    "step-77": "hyperelasticity",        # Trilinos
    "step-78": "black_scholes",          # finance / parabolic
    "step-79": "topology_opt",
    "step-80": "particle_methods",
    "step-81": "maxwell",                # HDG Maxwell
    "step-82": "level_set",
    "step-83": "checkpoint_restart",
    "step-84": "non_matching_grids",
    "step-85": "matrix_free",
    "step-86": "evolution_pde",
    "step-87": "shape_optimization",
    # step-88..step-97 are recent additions; encode as you verify.
}


_NGSOLVE_UPSTREAM: dict[str, str] = {
    # NGSolve i-tutorials from docu.ngsolve.org.
    # Stable subset covering the canonical physics.
    "unit-1.1":  "poisson",
    "unit-1.2":  "advection_dg",
    "unit-1.3":  "convection_diffusion",
    "unit-1.4":  "helmholtz",
    "unit-1.5":  "linear_elasticity",
    "unit-1.6":  "stokes",
    "unit-1.7":  "navier_stokes",
    "unit-1.8":  "maxwell",
    "unit-2.1":  "linear_elasticity",  # symbolic
    "unit-2.2":  "stokes",             # mixed
    "unit-2.3":  "mixed_methods",
    "unit-2.4":  "heat",
    "unit-2.5":  "hyperelasticity",
    "unit-2.6":  "phase_field",
    "unit-2.7":  "hdg",
    "unit-2.8":  "eigenvalue",
    "unit-2.9":  "navier_stokes",
    "unit-2.10": "plasticity",
    "unit-3.1":  "dg_methods",         # CR / nonconforming
    "unit-3.2":  "hdivdiv",
    "unit-3.3":  "surface_pde",
    "unit-3.4":  "shells",
    "unit-3.5":  "contact",
    "unit-3.6":  "trefftz",
    "unit-3.7":  "vem",
    "unit-3.8":  "time_dependent",
    "unit-3.9":  "time_dependent_ns",
}


_CURATED_INVENTORIES: dict[str, dict[str, str]] = {
    "skfem":   _SKFEM_UPSTREAM,
    "dealii":  _DEALII_UPSTREAM,
    "ngsolve": _NGSOLVE_UPSTREAM,
}


def _scan_one(backend: str) -> dict:
    dirs = _backend_demo_dirs(backend)
    found: list[dict] = []
    unmapped: list[dict] = []
    skipped: list[str] = []

    # Merge curated upstream entries if this backend has one. Mark
    # the source as "curated:" + key so the gap report stays
    # traceable.
    curated = _CURATED_INVENTORIES.get(backend, {})
    for key, topic in curated.items():
        found.append({"file": f"curated:{key}", "stem": key,
                      "topic": topic, "source": "curated"})
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
