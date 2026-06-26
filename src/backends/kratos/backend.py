"""
Kratos Multiphysics solver backend.

Kratos is a framework for building multi-disciplinary simulation software.
It uses a three-file system:
  - MainKratos.py: Python driver script
  - ProjectParameters.json: solver settings, BCs, materials
  - mesh.mdpa: mesh data (nodes, elements, conditions)

This backend generates all three files and executes MainKratos.py.
VTK output is configured via ProjectParameters.json.
"""

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from core.backend import (
    sorted_by_step,
    SolverBackend, BackendStatus, InputFormat,
    PhysicsCapability, JobHandle, get_python_executable,
)
from core.registry import register_backend
from .generators import GENERATORS, KNOWLEDGE

logger = logging.getLogger("oasis.kratos")


class KratosBackend(SolverBackend):

    def name(self) -> str:
        return "kratos"

    def display_name(self) -> str:
        return "Kratos Multiphysics"

    def check_availability(self) -> tuple[BackendStatus, str]:
        python = get_python_executable()
        if not python:
            return BackendStatus.NOT_INSTALLED, "No Python found"
        import subprocess
        try:
            result = subprocess.run(
                [python, "-c",
                 "import KratosMultiphysics as KM; "
                 "print(KM.KratosGlobals.Kernel.Version())"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                ver = result.stdout.strip().split('\n')[0]
                return BackendStatus.AVAILABLE, f"Kratos {ver}"
            return BackendStatus.NOT_INSTALLED, f"Kratos import failed: {result.stderr.strip()[:200]}"
        except Exception as e:
            return BackendStatus.NOT_INSTALLED, f"Check failed: {e}"

    def input_format(self) -> InputFormat:
        return InputFormat.JSON

    def supported_physics(self) -> list[PhysicsCapability]:
        # ---- 2026-06-26 honesty audit -----------------------------------
        # Every physics listed here now maps to a generator that runs a REAL
        # solve (model + mesh + solve + output). All former
        # "availability-probe" stub generators — scripts whose only action
        # was to import-check a Kratos sub-application and write
        # {"note": ...} with no solver run — and their PhysicsCapability rows
        # were REMOVED: rom, topology_optimization, iga, wind_engineering,
        # thermal_dem, swimming_dem, fem_to_dem, chimera, droplet_dynamics,
        # free_surface, fluid_biomedical, fluid_hydraulics, fluid, fsi,
        # geomechanics, compressible_potential, rans, pfem_fluid, pfem_solid,
        # pfem2.
        #
        # Importability tiers (verified on this host, python3 / Kratos 10.4):
        #   * The minimal pip stack importable here ships only
        #     StructuralMechanicsApplication, ConvectionDiffusionApplication,
        #     ContactStructuralMechanicsApplication and LinearSolversApplication.
        #     The generators that run end-to-end on that stack (verified
        #     rc=0 with physical output) are: poisson, heat, heat_transient,
        #     linear_elasticity (incl. the 2d_nonlinear Total-Lagrangian
        #     Newton solve), contact (penalty Signorini), structural_dynamics,
        #     plasticity / constitutive_laws (need ConstitutiveLawsApplication).
        #   * The remaining entries below (poromechanics, shallow_water, dam,
        #     dem, mpm, shape_optimization, cosimulation, dem_structures_coupling,
        #     cable_net, optimization) are GENUINE parameterized solves — NOT
        #     probe stubs — that require a fuller Kratos build whose
        #     application is not in the minimal pip stack. They are retained
        #     because they actually solve; they are not silent no-ops.
        return [
            PhysicsCapability(
                name="poisson",
                description="Poisson / convection-diffusion (LaplacianElement, EulerianConvDiff)",
                spatial_dims=[2, 3],
                element_types=["LaplacianElement2D3N", "EulerianConvDiff2D3N"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="linear_elasticity",
                description="Structural mechanics: linear/nonlinear, static/dynamic (StructuralMechanicsApplication)",
                spatial_dims=[2, 3],
                element_types=["SmallDisplacementElement2D3N/3D4N", "TotalLagrangianElement2D3N/3D4N",
                               "ShellThinElement3D3N", "CrBeamElement3D2N", "TrussElement3D2N"],
                template_variants=["2d", "2d_nonlinear"],
            ),
            PhysicsCapability(
                name="heat",
                description="Thermal convection-diffusion: steady and transient (ConvectionDiffusionApplication)",
                spatial_dims=[2, 3],
                element_types=["EulerianConvDiff2D3N", "LaplacianElement2D3N"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="contact",
                description="Contact mechanics: frictionless Signorini (penalty active-set) — real solve via ContactStructuralMechanicsApplication",
                spatial_dims=[2],
                element_types=["Element2D3N (CST) + penalty contact"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="structural_dynamics",
                description="Dynamic structural analysis with Newmark/Bossak time integration",
                spatial_dims=[2],
                element_types=["SmallDisplacementElement2D3N"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="heat_transient",
                description="Transient heat conduction with backward Euler time integration",
                spatial_dims=[2],
                element_types=["EulerianConvDiff2D3N"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="dem",
                description="Discrete Element Method for granular/particle simulations (DEMApplication)",
                spatial_dims=[2, 3],
                element_types=["SphericParticle3D", "CylinderParticle2D"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="mpm",
                description="Material Point Method for large-deformation solid mechanics (MPMApplication)",
                spatial_dims=[2, 3],
                element_types=["UpdatedLagrangianPQ2D", "UpdatedLagrangianAxisym"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="shape_optimization",
                description="Shape optimization with gradient-based methods (ShapeOptimizationApplication)",
                spatial_dims=[2, 3],
                element_types=["SmallDisplacementElement2D3N"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="cosimulation",
                description="CoSimulation framework for multi-solver coupling (CoSimulationApplication)",
                spatial_dims=[2, 3],
                element_types=["Generic (wraps sub-solvers)"],
                template_variants=["2d"],
            ),
            # New applications
            PhysicsCapability("poromechanics", "Poromechanics: fracture in porous media, dam/tunnel (PoromechanicsApplication)", [2, 3],
                              ["SmallStrainUPwDiffOrderElement2D6N"], ["2d"]),
            PhysicsCapability("shallow_water", "Shallow water equations: floods, dam breaks, coastal (ShallowWaterApplication)", [2],
                              ["ShallowWaterElement2D3N"], ["2d"]),
            PhysicsCapability("dam", "Dam engineering: thermal-mechanical, seepage, cracking", [2, 3],
                              ["SmallStrainElement2D3N"], ["2d"]),
            PhysicsCapability("plasticity", "Elasto-plasticity: MC, DP, VonMises, Tresca + 6 hardening laws (ConstitutiveLawsApplication)", [2, 3],
                              ["SmallDisplacementElement3D8N", "SmallDisplacementElement2D4N",
                               "TotalLagrangianElement3D8N"], ["3d"]),
            PhysicsCapability("constitutive_laws", "Extended constitutive laws: hyperelastic, plasticity, damage, viscoplastic", [2, 3],
                              ["SmallDisplacementElement2D3N"], ["2d"]),
            PhysicsCapability("dem_structures_coupling", "DEM-FEM coupling: impact on structures, blast", [2, 3],
                              ["SphericParticle3D + SmallDisplacement"], ["2d"]),
            PhysicsCapability("cable_net", "Cable and net structures: cables, membranes, form-finding", [3],
                              ["CableElement3D2N", "MembraneElement3D3N"], ["2d"]),
            PhysicsCapability("optimization", "General optimization: gradient-based, adjoint, multi-objective", [2, 3],
                              ["Generic"], ["2d"]),
            # ── 2026-06-01 (task #70): _auxiliary_overview in
            #    src/backends/kratos/generators/auxiliary_applications.py
            #    holds 6 substantive integration / API / numerical
            #    pitfalls about Kratos infrastructure apps
            #    (TrilinosApplication PyPI gap, MappingApplication
            #    FSI requirement, MapperFactory deprecation, DEM
            #    3D-only, contact apps PyPI absence). The leading
            #    underscore in the key historically kept this out
            #    of supported_physics; expose it now as
            #    'auxiliary_overview' so users browsing
            #    discover(physics, kratos) see the meta-catalog.
            PhysicsCapability(
                "auxiliary_overview",
                "[Reference] Kratos auxiliary applications "
                "(TrilinosApplication, MetisApplication, "
                "MappingApplication, MeshMovingApplication, "
                "HDF5Application, ...) — infrastructure that "
                "other Kratos analyses depend on. Catalog "
                "contains PyPI-publication status, deprecation "
                "notes, and FSI hidden-dependency warnings. "
                "Not a PDE physics — this is a meta-reference "
                "entry; the underlying KNOWLEDGE key is "
                "'_auxiliary_overview' (with the leading "
                "underscore preserved for backward "
                "compatibility).",
                [2, 3], ["N/A — meta-reference"], ["N/A"]),
        ]

    def get_knowledge(self, physics: str) -> dict:
        # Alias: PhysicsCapability surfaces 'auxiliary_overview'
        # (no leading underscore) but the on-disk KNOWLEDGE key
        # is '_auxiliary_overview'. Map back here so the public
        # name resolves.
        if physics == "auxiliary_overview":
            physics = "_auxiliary_overview"
        # Source-of-truth (audit 2026-06-02):
        # The kratos catalog has exactly ONE per-physics
        # source: backends.kratos.generators.KNOWLEDGE. A
        # previous version of this method tried
        # `from kratos_knowledge import KRATOS_KNOWLEDGE` —
        # but data/kratos_knowledge.py exports per-application
        # constants (KRATOS_APPLICATIONS, STRUCTURAL_MECHANICS,
        # FLUID_DYNAMICS, FSI, ...) and NOT a unified
        # `KRATOS_KNOWLEDGE` dict, so the import always raised
        # ImportError and the lookup was silent dead code.
        # Removed in 2026-06-02 audit; if cross-application
        # catalog access is needed in future, plumb it through
        # explicitly (e.g. a knowledge('kratos_application',
        # 'StructuralMechanicsApplication') surface) rather
        # than re-introducing a phantom flat-dict import.
        return KNOWLEDGE.get(physics, {})

    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        # Meta-reference physics: catalog declares
        # 'auxiliary_overview' so it appears in discover() and
        # knowledge() surfaces, but it has no PDE template —
        # it's a documentation entry for Kratos infrastructure
        # apps (TrilinosApplication, MetisApplication, etc.).
        # Return a commentary script that points to the
        # underlying KNOWLEDGE key. Without this early-return,
        # calling generate_input('auxiliary_overview', 'N/A',
        # {}) would raise ValueError.
        if physics == "auxiliary_overview" or variant == "N/A":
            return (
                '"""Kratos auxiliary_overview — meta-reference"""\n'
                '# This is NOT a runnable Kratos analysis script.\n'
                '# The catalog advertises auxiliary_overview so '
                'it appears in\n'
                '# discover() and knowledge() results — it '
                'documents Kratos\n'
                '# infrastructure applications (TrilinosApplication,\n'
                '# MetisApplication, MappingApplication, '
                'MeshMovingApplication,\n'
                '# HDF5Application, ...) and their hidden FSI '
                'dependencies.\n'
                '#\n'
                '# Use knowledge(kratos, auxiliary_overview) for the\n'
                '# full reference. For a runnable Kratos analysis '
                'pick a\n'
                '# concrete physics: poisson, linear_elasticity, '
                'heat, fluid,\n'
                '# fsi, contact, mpm, dem, geomechanics, '
                'cosimulation, ...\n'
                'import KratosMultiphysics  # placeholder — see '
                'comment above\n'
                'print("auxiliary_overview is a meta-reference; '
                'see knowledge() output")\n'
            )
        key = f"{physics}_{variant}"
        gen = GENERATORS.get(key)
        if not gen:
            raise ValueError(f"No Kratos template for {key}")
        # Return the MainKratos.py script which embeds the JSON and mdpa inline
        return gen(params)

    def validate_input(self, content: str) -> list[str]:
        errors = []
        try:
            compile(content, "<kratos_input>", "exec")
        except SyntaxError as e:
            errors.append(f"Python syntax error: {e}")

        # ---- Honesty guard (2026-06-26 audit) ----------------------------
        # Reject "availability-probe" scripts: ones whose ONLY action is to
        # import-check a Kratos (sub-)application and write {"note": "..."}
        # without ever running a solve. Such stubs produce no solution yet
        # return rc=0, making the catalog silently wrong. A legitimate Kratos
        # analysis MUST actually solve something — via a Kratos strategy /
        # AnalysisStage, or (as several generators in this backend do) by
        # assembling the system and solving it with scipy/numpy while using
        # KratosMultiphysics for I/O.
        lowered = content.lower()

        # Markers that a genuine solve is present (any one suffices). Covers
        # both Kratos-native strategies and the scipy/numpy assemble-and-solve
        # pattern used by the poisson / heat / elasticity / contact / dynamics
        # generators in this backend.
        solve_markers = (
            ".Run()",                 # AnalysisStage.Run()
            ".RunSolutionLoop(",
            ".Solve()",               # strategy / solver .Solve()
            ".SolveSolutionStep(",
            "AnalysisStage",
            "SolvingStrategy",
            "CreateSolver",
            "ResidualBasedNewtonRaphsonStrategy",
            "ResidualBasedLinearStrategy",
            "spsolve",                # scipy sparse direct solve
            "scipy.sparse.linalg",
            "factorized(",            # scipy prefactored solve (dynamics)
            "np.linalg.solve",
            "numpy.linalg.solve",
        )
        has_solve = any(m in content for m in solve_markers)

        # A "note"-only summary is the tell-tale signature of the old probe
        # stubs ({"note": "... available"} / {"note": "not installed"}).
        note_only_summary = (
            ('"note"' in content or "'note'" in content)
            and ("not installed" in lowered
                 or "available" in lowered
                 or "not pip-installable" in lowered)
        )
        imports_subapp = "import KratosMultiphysics." in content

        if not has_solve and (note_only_summary or imports_subapp):
            errors.append(
                "Script appears to be an availability-probe stub: it "
                "import-checks Kratos and/or writes a {\"note\": ...} "
                "summary but never runs a solve (no Kratos strategy / "
                "AnalysisStage and no scipy/numpy linear solve). A Kratos "
                "analysis must build a model and actually solve it, not just "
                "report availability."
            )

        # A valid Kratos input must EITHER use KratosMultiphysics (for the
        # model / output / strategy) OR perform a genuine solve (the
        # scipy-assembly generators in this backend import only numpy/scipy
        # and use KratosMultiphysics for VTK output). If it does neither it is
        # not a runnable analysis.
        if "KratosMultiphysics" not in content and not has_solve:
            errors.append(
                "Script neither uses KratosMultiphysics nor runs a solve — "
                "it is not a runnable Kratos analysis."
            )
        return errors

    async def run(self, input_content: str, work_dir: Path,
                  np: int = 1, timeout=None) -> JobHandle:
        python = get_python_executable()
        if not python:
            return JobHandle(
                job_id=str(uuid.uuid4())[:8],
                backend_name="kratos",
                work_dir=work_dir,
                status="failed",
                error="Python not found",
            )

        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        script_path = work_dir / "MainKratos.py"
        script_path.write_text(input_content)

        job = JobHandle(
            job_id=str(uuid.uuid4())[:8],
            backend_name="kratos",
            work_dir=work_dir,
            status="running",
        )

        cmd = [python, str(script_path)]

        # If KRATOS_ROOT has a source build, use it over pip-installed version
        from core.backend import get_env_with_source_root
        env = get_env_with_source_root("KRATOS_ROOT")

        start = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(work_dir),
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            job.elapsed = time.time() - start
            job.return_code = proc.returncode
            job.pid = proc.pid

            if proc.returncode == 0:
                job.status = "completed"
            else:
                job.status = "failed"
                job.error = stderr.decode(errors="replace")[-2000:]

            (work_dir / "stdout.log").write_text(stdout.decode(errors="replace"))
            (work_dir / "stderr.log").write_text(stderr.decode(errors="replace"))
        except asyncio.TimeoutError:
            job.status = "failed"
            job.elapsed = timeout
            job.error = f"Timed out after {timeout}s"
        except Exception as e:
            job.status = "failed"
            job.elapsed = time.time() - start
            job.error = str(e)

        return job

    def get_result_files(self, job: JobHandle) -> list[Path]:
        results = []
        for ext in ["*.vtu", "*.vtk", "*.pvd"]:
            results.extend(job.work_dir.rglob(ext))
        return sorted_by_step(results)


def register():
    backend = KratosBackend()
    register_backend(backend, aliases=["kratos", "kratosmp"])
    logger.info("Kratos Multiphysics backend registered")
