#!/usr/bin/env python3
"""
Per-physics layer-3 sweep across every available backend.

Drives each backend through its `generate_input` + `run` path with the same
canonical reference problem per physics row, captures pass/fail, elapsed
wall-clock, and a key scalar (max(u), tip displacement, ...) from the VTU
output.  Emits a markdown table at the end and a JSON dump for downstream
plotting.

Env vars required for the full sweep (set automatically if matching install
exists on disk):
    DEAL_II_DIR     deal.II cmake config dir (.../lib/cmake/deal.II)
    FOURC_BINARY    Path to the built 4C binary
    FOURC_ROOT      Source root for FOURC (some templates depend on it)

Run:
    python benchmarks/sweep_layer3.py [--only POISSON]

`--only` accepts a comma-separated list of row names that exist in
`MATRIX` below.  Additional physics rows are added in follow-up PRs;
keep this docstring in sync with the actual matrix.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

# ── repo paths ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "data"))

# ── env discovery (so users do not need to pre-export anything) ───────────
def _discover_env():
    """Best-effort discovery of solver locations on this machine."""
    home = Path.home()
    # deal.II — both the deal.II-style underscore env var (DEAL_II_DIR, used by
    # the dealii backend's HINTS line) and CMAKE_PREFIX_PATH (which is what
    # find_package(deal.II) actually consumes — the env var spelling
    # `deal.II_DIR` with a dot is not portable across shells).
    dealii_env_root = home / "miniconda3/envs/ofa-dealii"
    if "DEAL_II_DIR" not in os.environ:
        for c in [
            dealii_env_root / "lib/cmake/deal.II",
            Path("/usr/lib/x86_64-linux-gnu/cmake/deal.II"),
        ]:
            if c.is_dir():
                os.environ["DEAL_II_DIR"] = str(c)
                break
    if dealii_env_root.is_dir():
        # Best-effort discovery: append the conda env as a fallback, not as
        # the highest-priority entry, so an explicit user-provided
        # CMAKE_PREFIX_PATH still resolves first.  Use os.pathsep so this
        # remains correct on Windows (`;`) too.
        existing = os.environ.get("CMAKE_PREFIX_PATH", "")
        prefix_parts = [p for p in existing.split(os.pathsep) if p]
        if str(dealii_env_root) not in prefix_parts:
            prefix_parts.append(str(dealii_env_root))
            os.environ["CMAKE_PREFIX_PATH"] = os.pathsep.join(prefix_parts)
    # The conda-forge deal.II 9.1.1 package bakes a feedstock-only
    # compiler path into deal.IIConfig.cmake (/home/conda/feedstock_root/...)
    # which does not exist at runtime.  Override CC/CXX so CMake uses the
    # host's working compiler instead.
    import shutil as _shutil
    if "CXX" not in os.environ:
        for cxx in ("g++", "c++", "clang++"):
            if _shutil.which(cxx):
                os.environ["CXX"] = cxx
                break
    if "CC" not in os.environ:
        for cc in ("gcc", "cc", "clang"):
            if _shutil.which(cc):
                os.environ["CC"] = cc
                break
    # 4C
    if "FOURC_BINARY" not in os.environ:
        for c in [
            home / "Schreibtisch/4C-src/4C/build/4C",
            home / "4C/build/4C",
        ]:
            if c.is_file():
                os.environ["FOURC_BINARY"] = str(c)
                os.environ.setdefault("FOURC_ROOT", str(c.parent.parent))
                break


# NOTE: `_discover_env()` mutates os.environ.  It is invoked at the top of
# main() (not at import time) so that importing this module as a library —
# e.g. for unit-testing the matrix definition or `Cell` dataclass — does not
# silently rewrite CC, CXX, CMAKE_PREFIX_PATH, DEAL_II_DIR, FOURC_BINARY in
# the calling process.

# ── imports that need the registry available at module scope ──────────────
from core.registry import load_all_backends, get_backend  # noqa: E402
from core.backend import BackendStatus  # noqa: E402
from core.post_processing import post_process_file  # noqa: E402


RESULTS_DIR = REPO_ROOT / "benchmarks" / "sweep_results"


@dataclass
class Cell:
    """One (backend, physics, variant, params) cell of the matrix."""

    backend: str
    physics: str
    variant: str
    params: dict
    # The scalar field whose max(abs) we extract from the VTU output for
    # cross-cell comparison.  None ⇒ no scalar extraction (just pass/fail).
    field: str | None = None
    # Human-readable reference if there is one.
    expected: float | None = None
    # Relative tolerance for the expected value.
    rtol: float = 0.05


@dataclass
class CellResult:
    cell: Cell
    status: str
    elapsed_s: float | None
    scalar: float | None = None
    field_used: str | None = None
    error: str | None = None
    note: str = ""


# ═══════════════════════════════════════════════════════════════════════════
#   MATRIX
# ═══════════════════════════════════════════════════════════════════════════
# Reference problem per row:
#
#   POISSON           -Δu = 1 on [0,1]², u = 0 on ∂Ω.  max(u) ≈ 0.0737.
#   HEAT              Steady-state heat conduction.  Each backend's
#                     `heat` template ships with a different canonical
#                     problem: the working FEniCSx and 4C templates
#                     both solve a Dirichlet hot/cold-wall problem
#                     (T=0 on one side, T=100 on the other, no source)
#                     so max(T) = 100 by boundary data, *not* the
#                     source-driven 0.0737 from POISSON.  Variant
#                     names are not shared across backends (FEniCSx
#                     and deal.II use `2d_steady`; skfem / NGSolve /
#                     Kratos / 4C use a plain `2d`); the cell entries
#                     below pick the right variant per backend.  As
#                     with POISSON, the comparison scalar is max(T)
#                     of whichever scalar field the template emits
#                     (the field name itself is per-backend: FEniCSx
#                     `temperature`, 4C `phi_1`, ...).
#
#   ELASTICITY        Each backend's `linear_elasticity/2d` (or
#                     equivalent) template on the rectangular domain
#                     it ships with.  Material: E=1000, ν=0.3.  The
#                     2D plane-strain-vs-plane-stress choice is a
#                     per-template detail and is not enforced here —
#                     the agent has to read each generator to see
#                     which it uses (FEniCSx and skfem both compute
#                     λ = E·ν / ((1+ν)·(1-2ν)), i.e. the standard
#                     plane-strain / 3D form, so for those two the
#                     numbers are directly comparable; other backends
#                     have not been audited here).  As a consequence,
#                     the cross-backend numbers are NOT guaranteed to
#                     agree as absolute values across all six cells.  Comparison metric per cell depends on
#                     the field type:
#                       - vector-valued displacement output (NGSolve,
#                         FEniCSx, 4C, Kratos): the scalar reported is
#                         the maximum L2 magnitude of the displacement
#                         across nodes — `core.post_processing.post_process_file`
#                         collapses multi-component point arrays via
#                         `np.linalg.norm(arr, axis=1)` before
#                         min/max/mean/std.
#                       - per-component fields (deal.II writes "ux",
#                         "uy" separately): the cell asks for one
#                         component (typically the dominant one) so
#                         the scalar there is the maximum of that
#                         component, not a magnitude.
#                     Standardising the elasticity templates so a
#                     single closed-form reference is usable across all
#                     backends is a follow-up.
#
# More rows (heat, stokes, …) are added in follow-up PRs.
# Sweep accepts --only to filter to a comma-separated subset.

_KAPPA = 1.0

MATRIX: dict[str, list[Cell]] = {
    "POISSON": [
        # Reference: -Δu = 1 on [0,1]², u = 0 on ∂Ω.
        # Analytic max(u) ≈ 0.07367 (Fourier series, ~5-digit accuracy).
        Cell("skfem",   "poisson", "2d",         {"kappa": _KAPPA, "nx": 32, "ny": 32},
             field="phi", expected=0.0737, rtol=0.05),
        Cell("ngsolve", "poisson", "2d",         {"kappa": _KAPPA, "nx": 32, "ny": 32},
             field="phi", expected=0.0737, rtol=0.05),
        Cell("fenics",  "poisson", "2d",         {"kappa": _KAPPA, "nx": 32, "ny": 32},
             field="u", expected=0.0737, rtol=0.05),
        # Kratos's `poisson_2d` template is a scipy-only stub: it does not
        # import KratosMultiphysics and runs a hand-rolled scipy.sparse
        # solve instead of touching Kratos at all.  The backend's
        # validate_input rule "Script should import KratosMultiphysics"
        # correctly rejects it.  The cell is kept here so that catalog
        # drift is visible in every sweep run; replacing the template
        # with a real KratosConvectionDiffusionApplication template is
        # tracked separately.
        Cell("kratos",  "poisson", "2d",         {"kappa": _KAPPA, "nx": 32, "ny": 32},
             field="TEMPERATURE", expected=0.0737, rtol=0.05),
        # deal.II 9.1.1 (the version conda-forge ships against python>=3.12)
        # requires the legacy <tbb/task.h> header that was removed in TBB
        # 2020.x; on this machine the conda env supplies TBB 2020.2, so
        # the build of the generated main.cpp fails.  conda-forge's next
        # deal.II (9.3.2) is built against oneTBB but pins python<=3.10
        # and is not co-installable with the current python 3.12 env.
        # The cell stays in the matrix so the gap is visible; resolving
        # it needs a parallel env or a source build of deal.II.
        # NB the deal.II Poisson template writes the result vector as
        # "solution" (data_out.add_data_vector(solution, "solution")), so
        # the expected field name is "solution", not "u".
        Cell("dealii",  "poisson", "2d",         {"refinements": 5},
             field="solution", expected=0.0737, rtol=0.05),
        Cell("fourc",   "poisson", "poisson_2d", {},
             field="phi_1", expected=0.0737, rtol=0.05),
    ],
    "HEAT": [
        # Steady-state heat conduction.  The FEniCSx and 4C `heat/*`
        # templates both ship a Dirichlet hot-cold-wall problem
        # (T=0 on cold side, T=100 on hot side, no volumetric source),
        # so the working backends agree on max(T) = 100.  The other
        # backends fail in different ways (template bugs / unknown
        # variants / scipy stubs), all surfaced by the cells below.

        # skfem's `heat/2d` template subscripts a DofsView with a
        # string (`dofs["left"]`); DofsView is not subscriptable that
        # way in the current scikit-fem release, so the template
        # raises TypeError at execution.  Template bug.
        Cell("skfem",   "heat", "2d",        {"kappa": _KAPPA, "nx": 32, "ny": 32},
             field="phi", expected=100.0, rtol=0.05),
        # NGSolve's `heat/2d` template constructs the RHS via
        # `LinearForm(0*v*dx)` which raises NgException("Linearform
        # must have TestFunction") because the zero coefficient
        # strips the TestFunction.  Template bug — should use either
        # `LinearForm(1*v*dx)` for f=1 or build the form from `v*dx`
        # directly.
        Cell("ngsolve", "heat", "2d",        {"kappa": _KAPPA, "nx": 32, "ny": 32},
             field="phi", expected=100.0, rtol=0.05),
        # FEniCSx exposes explicit steady-vs-transient variants; we
        # pick `2d_steady`.  Output field is named `temperature`
        # (not `u`); max = 100 by Dirichlet data.
        Cell("fenics",  "heat", "2d_steady", {"kappa": _KAPPA, "nx": 32, "ny": 32},
             field="temperature", expected=100.0, rtol=0.05),
        # Kratos `heat/2d` is a scipy stub (one of the 8 known fake
        # templates); validate_input correctly rejects it.  The stub,
        # if it ever ran, writes its point data under the lowercase
        # name `"temperature"` (see
        # src/backends/kratos/generators/heat.py: `point_data={"temperature": u}`).
        # Use the exact emitted name so the cell stays consistent with
        # every other entry in the matrix (the harness has a
        # case-insensitive fallback but relying on it is sloppy).
        Cell("kratos",  "heat", "2d",        {"kappa": _KAPPA, "nx": 32, "ny": 32},
             field="temperature", expected=100.0, rtol=0.05),
        # deal.II steady-state heat — same build-environment caveats
        # as the POISSON cell apply.  Output field is named
        # `temperature` (NOT `solution` like POISSON): the heat
        # template emits `data_out.add_data_vector(solution, "temperature");`,
        # see src/backends/dealii/generators/heat.py.
        Cell("dealii",  "heat", "2d_steady", {"refinements": 5},
             field="temperature", expected=100.0, rtol=0.05),
        # 4C scatra produces a `phi_1` field; the template's BC
        # values match the FEniCSx hot-cold-wall problem so max ≈ 100.
        Cell("fourc",   "heat", "2d",        {},
             field="phi_1", expected=100.0, rtol=0.05),
    ],
    "ELASTICITY": [
        # skfem's `linear_elasticity/2d` template has two stacked
        # problems: (a) it constructs a tensor-product mesh with no
        # boundary tags and then calls `ib.get_dofs("left")`, which
        # raises; and (b) even if (a) is fixed, the template writes a
        # `results_summary.json` only, never a .vtu, so the sweep would
        # then transition from `failed` to `no_output_file`.  Both are
        # template bugs to address upstream.  The cell is kept so the
        # matrix surfaces the gap on every run.
        Cell("skfem",   "linear_elasticity", "2d", {"E": 1000, "nu": 0.3},
             field="displacement", expected=None, rtol=0.5),
        # ngsolve and fenics write the displacement field as
        # "displacement" (vector).  `core.post_processing.post_process_file`
        # collapses multi-component point arrays via the elementwise L2
        # magnitude (`np.linalg.norm(arr, axis=1)`) before computing
        # min/max/mean/std, so the scalar reported below is the
        # maximum of the displacement magnitude across nodes — *not* a
        # per-component max.  Cross-backend consistency is checked
        # qualitatively here (both reporting numbers of comparable
        # order of magnitude); a shared geometry that admits a closed
        # form is a follow-up.
        Cell("ngsolve", "linear_elasticity", "2d", {"E": 1000, "nu": 0.3},
             field="displacement", expected=None, rtol=0.5),
        Cell("fenics",  "linear_elasticity", "2d", {"E": 1000, "nu": 0.3},
             field="displacement", expected=None, rtol=0.5),
        # Kratos `linear_elasticity/2d_nonlinear` was a placeholder
        # stub until the same PR that widened this harness to accept
        # legacy `.vtk` — it is now a real KratosMultiphysics
        # TotalLagrangianElement2D4N cantilever with
        # LinearElasticPlaneStrain2DLaw, prescribed-displacement BC at
        # the tip mid-node, and `KM.VtkOutput` writing a real
        # `Structure_0_*.vtk` containing `DISPLACEMENT` + `REACTION`
        # node fields.  The cell scores max|u| = 0.50884 (the
        # prescribed tip displacement is 0.5 in -y; the magnitude is
        # slightly larger because the Newton solve picks up some
        # bending-induced x-displacement near the tip).  Plain
        # `linear_elasticity/2d` has also been rewritten as a real
        # SmallDisplacement Newton solve.
        Cell("kratos",  "linear_elasticity", "2d_nonlinear", {"E": 1000, "nu": 0.3},
             field="DISPLACEMENT", expected=None, rtol=0.5),
        # deal.II's `linear_elasticity/2d` template needs the same
        # compiler + TBB setup as POISSON (see the deal.II POISSON
        # cell for the env explanation).  Output fields in the
        # generated main.cpp are component-named via
        # `std::vector<std::string> names = {"ux", "uy"};
        #  data_out.add_data_vector(solution, names);`
        # — i.e. there is no single vector field called "u" to extract
        # from.  Use "uy" (the dominant component for a beam under a
        # transverse load); when this build path is restored the cell
        # can be extended with a second variant for "ux".  Pass the
        # same explicit E/ν as the other cells so the matrix is
        # self-contained instead of relying on template defaults.
        Cell("dealii",  "linear_elasticity", "2d", {"E": 1000, "nu": 0.3},
             field="uy", expected=None, rtol=0.5),
        # 4C's `linear_elasticity/2d` template emits an input that
        # references the legacy "WALL" element type, which is not
        # registered in the build of 4C produced by this project's
        # build script (4C builds it conditionally and our build did
        # not enable it).  Run aborts at mesh-read time with
        # "Unknown type 'WALL' of finite element".  Either re-build 4C
        # with the wall element enabled, or update the template to use
        # the modern `SOLID` 2D element — follow-up.  Pass the same
        # explicit material parameters as the other cells so the matrix
        # entry remains stable if the template's default E/nu change.
        Cell("fourc",   "linear_elasticity", "2d", {"E": 1000, "nu": 0.3},
             field="displacement", expected=None, rtol=0.5),
    ],

    # ── HARD PHYSICS rows ────────────────────────────────────────────
    # These exercise the saddle-point / nonlinear / mixed-formulation
    # parts of each backend's catalog.  Far fewer cells reach a useful
    # scalar than the linear-elliptic rows above; the value of these
    # rows is precisely that the failure modes are now visible on
    # every sweep run.

    "STOKES": [
        # 2D Stokes flow.  No standardised canonical problem across
        # the templates yet, so the cross-cell comparison is
        # qualitative.  Expected scalar (when reachable) is the maximum
        # velocity magnitude, which `core.post_processing.post_process_file`
        # produces via `np.linalg.norm(arr, axis=1)` for vector fields.

        # skfem's `stokes/2d` template has two stacked defects:
        # (a) it raises `ValueError: Quadrature mismatch: trial and
        # test functions should have same number of integration
        # points` because the Taylor-Hood mixed basis is assembled
        # with mismatched quadrature orders between velocity P2 and
        # pressure P1; and (b) even with (a) fixed, the template
        # writes only a `results_summary.json` and never a `.vtu`,
        # so the cell would still report `no_vtu_output` until VTU
        # export is added.  Both are template-side gaps.
        Cell("skfem",   "stokes", "2d", {"nx": 32, "ny": 32},
             field="velocity", expected=None, rtol=0.5),
        # NGSolve's `stokes/2d` template runs to the solver step then
        # the MKL Pardiso direct solver raises
        # `RuntimeError: MKL Pardiso error in phase 33: -4` (a
        # numerical singularity / inertia mismatch from the
        # saddle-point system).  Template needs either a different
        # preconditioner / Schur-complement strategy or a stabilised
        # mixed element pair.
        Cell("ngsolve", "stokes", "2d", {"nx": 32, "ny": 32},
             field="velocity", expected=None, rtol=0.5),
        # FEniCSx's `stokes/2d` template still uses
        # `ufl.VectorElement(...)`, which was removed from UFL in the
        # dolfinx-0.10 stack on this machine.  The new path is
        # `basix.ufl.element(...)` + a single mixed `functionspace`.
        # Template targets an older dolfinx API.
        Cell("fenics",  "stokes", "2d", {"nx": 32, "ny": 32},
             field="velocity", expected=None, rtol=0.5),
        # deal.II Stokes — same conda-forge dealii 9.1.1 + TBB
        # header-drift situation as the elliptic rows.  Compile fails
        # before reaching the assembly.
        Cell("dealii",  "stokes", "2d", {},
             field="velocity", expected=None, rtol=0.5),
    ],

    "NAVIER_STOKES": [
        # Lid-driven cavity at Re=100 — the canonical CFD validation
        # benchmark (Ghia, Ghia & Shin 1982).  At the walls the
        # no-slip condition fixes u = 0; the well-known Ghia
        # tabulated reference is the velocity profile *along the
        # vertical centre-line* x = 0.5 (e.g. min u_x ≈ -0.21
        # near mid-height) — NOT a value at the lower wall.  max |u|
        # over the whole domain is order unity by construction (lid
        # velocity = 1) and is what the sweep harness extracts for
        # cross-cell comparison; matching the Ghia centre-line
        # profile would need a dedicated probe in a follow-up.

        # skfem's `navier_stokes/2d` template has two stacked defects:
        # (a) it raises a TypeError / AttributeError at solve.py
        # runtime (same untagged-boundary pattern the linear_elasticity
        # template had before its `with_boundaries({...})` fix); and
        # (b) even with (a) fixed, the template writes a `result.vtu`
        # whose `point_data` is mesh-only — no `velocity` or
        # `pressure` arrays — so the cell would still report
        # `field_not_found` until the VTU export is extended.
        Cell("skfem",   "navier_stokes", "2d", {"Re": 100, "nx": 32},
             field="velocity", expected=None, rtol=0.5),
        # NGSolve's NS template runs cleanly on the lid-driven cavity
        # and writes `velocity` and `pressure` fields; the working
        # cell on this machine reports max |u| ~ 1.0 (lid speed).
        Cell("ngsolve", "navier_stokes", "2d", {"Re": 100, "nx": 32},
             field="velocity", expected=None, rtol=0.5),
        # FEniCSx NS template runs and writes `velocity`; same
        # qualitative expectation (max |u| around the lid).
        Cell("fenics",  "navier_stokes", "2d", {"Re": 100, "nx": 32},
             field="velocity", expected=None, rtol=0.5),
        # deal.II NS — two stacked gaps: (a) the same conda-forge
        # dealii 9.1.1 + TBB build-fail situation as the other
        # deal.II rows; and (b) the deal.II `navier_stokes/2d`
        # template itself is a placeholder `main()` that prints a
        # status message and exits 0 — no solve, no `data_out`,
        # no `.vtu`.  Even once the env is rebuilt the cell will
        # report `no_vtu_output` until a real NS template is
        # written.  Pass explicit Re for matrix self-containment.
        Cell("dealii",  "navier_stokes", "2d", {"Re": 100},
             field="velocity", expected=None, rtol=0.5),
    ],

    "HYPERELASTICITY": [
        # Large-deformation Neo-Hookean elasticity.  No standardised
        # canonical problem across backends — each template ships its
        # own boundary conditions / geometry.  Comparison metric is
        # the max displacement magnitude where extractable.

        # skfem `hyperelasticity/2d` raises during the Newton loop.
        # Root cause is not investigated here — the cell records the
        # observed failure mode (Newton-loop exception, no .vtu);
        # template-side fix tracked separately.
        Cell("skfem",   "hyperelasticity", "2d", {"E": 1000, "nu": 0.3},
             field="displacement", expected=None, rtol=0.5),
        # NGSolve `hyperelasticity/2d` raises during the Newton loop
        # — needs review.
        Cell("ngsolve", "hyperelasticity", "2d", {"E": 1000, "nu": 0.3},
             field="displacement", expected=None, rtol=0.5),
        # FEniCSx `hyperelasticity/3d` runs (max displacement order
        # of magnitude depends on the demo's applied displacement BC).
        # The 2D variant is not in the catalog, so we use 3D here.
        Cell("fenics",  "hyperelasticity", "3d", {"E": 1000, "nu": 0.3},
             field="displacement", expected=None, rtol=0.5),
        # deal.II hyperelasticity — same conda-forge build-fail
        # situation as the other deal.II rows.  Pass explicit
        # material parameters so the matrix entry is self-contained
        # and immune to silent drift if the template's defaults
        # change, matching the policy used for the other cells in
        # this row.
        Cell("dealii",  "hyperelasticity", "3d", {"E": 1000, "nu": 0.3},
             field="displacement", expected=None, rtol=0.5),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
#   CELL RUNNER
# ═══════════════════════════════════════════════════════════════════════════


async def run_cell(cell: Cell, work_dir: Path) -> CellResult:
    backend = get_backend(cell.backend)
    if backend is None:
        return CellResult(cell, status="not_registered", elapsed_s=None,
                          error=f"no backend registered with name {cell.backend!r}")

    status, msg = backend.check_availability()
    if status != BackendStatus.AVAILABLE:
        return CellResult(cell, status=f"unavailable", elapsed_s=None,
                          error=msg[:200] if msg else "unavailable",
                          note=str(status))

    t0 = time.time()
    try:
        content = backend.generate_input(cell.physics, cell.variant, cell.params)
    except Exception as e:
        return CellResult(cell, status="generate_failed", elapsed_s=None,
                          error=f"{type(e).__name__}: {e!s:.300}")

    errors = backend.validate_input(content)
    if errors:
        return CellResult(cell, status="validation_failed", elapsed_s=None,
                          error="; ".join(errors)[:300])

    # Clear any artefacts from a previous sweep run in this cell's directory
    # before we ask the backend to run — otherwise stale .vtu files from a
    # different revision could be picked up by `backend.get_result_files(job)`
    # and a non-VTU-producing run would silently report the previous run's
    # scalar as a pass.
    if work_dir.exists():
        import shutil
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        job = await backend.run(content, work_dir, np=1, timeout=600)
    except Exception as e:
        return CellResult(cell, status="run_threw", elapsed_s=time.time() - t0,
                          error=f"{type(e).__name__}: {e!s:.300}")

    elapsed = time.time() - t0

    if job.status != "completed":
        return CellResult(cell, status=job.status, elapsed_s=elapsed,
                          error=(job.error or "")[:300])

    # ── extract scalar from the LAST output snapshot (final time step /
    # converged state).  Accept every format
    # `core.post_processing.read_mesh()` knows how to load *except*
    # `.pvtu`:
    #   .vtu    — modern XML unstructured (FEniCSx, skfem, fourc, ...)
    #   .vtk    — legacy unstructured (KratosMultiphysics `VtkOutput`)
    #   .pvd    — ParaView collection / time-series index
    #   .xdmf   — XDMF (dolfinx alternative)
    # `.pvtu` (parallel-partitioned VTU wrapper) is intentionally
    # excluded — it can hang PyVista when the per-rank `.vtu`
    # partials are accessed (see the matching policy at
    # src/tools/consolidated.py: "skip .pvtu (parallel wrappers
    # that can hang PyVista)").  The per-rank `.vtu` partials
    # themselves are accepted directly.
    #
    # Suffix matching is case-insensitive: some backends and
    # filesystems emit upper-case (`.VTU`) and `read_mesh` itself
    # lower-cases the suffix before dispatching.
    #
    # Sort numerically by trailing step index, NOT lexicographically:
    # backends like Kratos emit `Structure_0_1.vtk, Structure_0_2.vtk,
    # ..., Structure_0_10.vtk`, where lexicographic sort would place
    # `_10` *before* `_9` and pick the wrong snapshot as "the last".
    # Within the same step, prefer the most-PyVista-stable format
    # (`.vtu` > `.vtk` > `.xdmf` > `.pvd`) so two formats emitted
    # for the same step never pick a less-supported container.
    import re as _re
    _OUTPUT_SUFFIXES = (".vtu", ".vtk", ".pvd", ".xdmf")
    _SUFFIX_PRIORITY = {".vtu": 0, ".vtk": 1, ".xdmf": 2, ".pvd": 3}

    def _step_key(p):
        m = _re.findall(r"\d+", p.stem)
        return (
            int(m[-1]) if m else 0,
            # Negate so higher-priority sorts AFTER lower-priority on
            # the same step (sorted() ascending → key[-1] is "last").
            -_SUFFIX_PRIORITY.get(p.suffix.lower(), 99),
            p.stem,
        )

    output_files = sorted(
        (f for f in backend.get_result_files(job)
         if f.suffix.lower() in _OUTPUT_SUFFIXES),
        key=_step_key,
    )
    scalar = None
    field_used = None
    if cell.field is not None:
        # A cell that asked for a scalar but received no VTU/VTK is a
        # failure of the run even if the backend exited 0 — surface it
        # instead of quietly returning status=completed with scalar=None.
        if not output_files:
            return CellResult(
                cell, status="no_output_file", elapsed_s=elapsed,
                error=("backend reported completed but produced no "
                       f"{'/'.join(_OUTPUT_SUFFIXES)} output in "
                       f"{work_dir}; expected field {cell.field!r}"),
            )
        try:
            pp = post_process_file(output_files[-1], plot_dir=work_dir, plot_fields=False)
            available = [f.name for f in pp.fields]
            for f in pp.fields:
                if f.name == cell.field or f.name.lower() == cell.field.lower():
                    scalar = float(f.max)
                    field_used = f.name
                    break
            # No fallback.  A silent fallback to `pp.fields[0]` would happily
            # report max(Owner) (a cell-ID integer) as if it were `u`, which
            # is exactly the kind of stealth-wrong result this sweep exists
            # to catch.  If the expected field is absent we report the gap
            # explicitly so the matrix shows it, leaving the cell scalar=None.
            if scalar is None:
                return CellResult(
                    cell, status="field_not_found", elapsed_s=elapsed,
                    error=f"expected field {cell.field!r} absent; available: {available[:8]}",
                )
        except Exception as e:
            return CellResult(cell, status="postproc_failed", elapsed_s=elapsed,
                              error=f"{type(e).__name__}: {e!s:.300}")

    return CellResult(cell, status="completed", elapsed_s=elapsed,
                      scalar=scalar, field_used=field_used)


# ═══════════════════════════════════════════════════════════════════════════
#   ROW / MATRIX DRIVERS
# ═══════════════════════════════════════════════════════════════════════════


async def run_row(row_name: str, cells: list[Cell]) -> list[CellResult]:
    print(f"\n=== {row_name} ({len(cells)} cells) ===")
    out: list[CellResult] = []
    for cell in cells:
        work_dir = RESULTS_DIR / row_name.lower() / cell.backend
        result = await run_cell(cell, work_dir)
        out.append(result)
        # one-line summary per cell
        s = result.status
        e = f"{result.elapsed_s:.1f}s" if result.elapsed_s else "    "
        sc = f"{result.scalar:.5f}" if result.scalar is not None else "       "
        f_ = result.field_used or ""
        ref = f" ref={cell.expected:.5f}" if cell.expected is not None else ""
        print(f"  {cell.backend:8s}  {s:18s}  {e:8s}  {sc} {f_:18s}{ref}")
        if result.error:
            print(f"           -> {result.error[:200]}")
    return out


def emit_markdown_table(rows: dict[str, list[CellResult]]) -> str:
    lines: list[str] = []
    lines.append("# Per-physics layer-3 sweep — results\n")
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} on this machine._\n")
    for row_name, results in rows.items():
        lines.append(f"\n## {row_name}\n")
        lines.append("| Backend | Status | Elapsed | Scalar | Field | Ref | Within tol |")
        lines.append("|---------|--------|---------|--------|-------|-----|------------|")
        for r in results:
            ok_tol = ""
            if r.cell.expected is not None and r.scalar is not None:
                rel = abs(r.scalar - r.cell.expected) / max(abs(r.cell.expected), 1e-30)
                ok_tol = "✓" if rel <= r.cell.rtol else f"✗ ({rel:.2%})"
            elif r.scalar is None and r.status == "completed":
                ok_tol = "—"
            lines.append(
                f"| {r.cell.backend} | {r.status} | "
                f"{f'{r.elapsed_s:.1f}s' if r.elapsed_s else '—'} | "
                f"{f'{r.scalar:.5f}' if r.scalar is not None else '—'} | "
                f"{r.field_used or '—'} | "
                f"{f'{r.cell.expected:.5f}' if r.cell.expected is not None else '—'} | "
                f"{ok_tol} |"
            )
    return "\n".join(lines)


def cell_result_to_dict(r: CellResult) -> dict:
    d = asdict(r)
    d["cell"] = asdict(r.cell)
    return d


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="",
                   help="comma-separated row names to run "
                        "(must exist in MATRIX; e.g. POISSON)")
    args = p.parse_args()

    _discover_env()  # mutate os.environ here, not at module-import time
    load_all_backends()
    print("loaded backends:")
    from core.registry import available_backends
    for b in available_backends():
        print(f"  - {b.name():8s}  {b.display_name()}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    requested = set(s.strip().upper() for s in args.only.split(",") if s.strip())
    # Fail fast on a typo: an empty result set (silently writing an empty
    # sweep.json) would look like a successful validation run.
    unknown = requested - MATRIX.keys()
    if unknown:
        raise SystemExit(
            f"--only references rows that are not in the matrix: "
            f"{sorted(unknown)} — known rows: {sorted(MATRIX.keys())}"
        )
    rows_to_run = {n: c for n, c in MATRIX.items() if not requested or n in requested}

    results: dict[str, list[CellResult]] = {}
    for row_name, cells in rows_to_run.items():
        results[row_name] = await run_row(row_name, cells)

    # ── persist
    json_path = RESULTS_DIR / "sweep.json"
    json_path.write_text(json.dumps(
        {row: [cell_result_to_dict(r) for r in res] for row, res in results.items()},
        indent=2, default=str,
    ))

    md = emit_markdown_table(results)
    md_path = RESULTS_DIR / "sweep.md"
    md_path.write_text(md)

    print("\n" + "=" * 72)
    print(md)
    print("=" * 72)
    print(f"\nJSON: {json_path}")
    print(f"  MD: {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
