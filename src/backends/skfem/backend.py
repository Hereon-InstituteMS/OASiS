"""
scikit-fem solver backend.

scikit-fem is a pure-Python FEM assembly library with zero compilation
dependencies (only numpy, scipy, meshio). It generates system matrices
from weak forms and solves with scipy sparse solvers.

This backend demonstrates the agent handling an assembly-level library,
not just turnkey solvers.
"""

import asyncio
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from core.backend import (
    SolverBackend, BackendStatus, InputFormat,
    PhysicsCapability, JobHandle, get_python_executable,
)
from core.registry import register_backend
from .generators import GENERATORS, KNOWLEDGE

logger = logging.getLogger("open-fem-agent.skfem")


class SkfemBackend(SolverBackend):

    def name(self) -> str:
        return "skfem"

    def display_name(self) -> str:
        return "scikit-fem"

    def check_availability(self) -> tuple[BackendStatus, str]:
        import subprocess
        python = get_python_executable()
        if not python:
            return BackendStatus.NOT_INSTALLED, "No Python found"
        try:
            result = subprocess.run(
                [python, "-c", "import skfem; print(skfem.__version__)"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                ver = result.stdout.strip()
                return BackendStatus.AVAILABLE, f"scikit-fem {ver}"
            return BackendStatus.NOT_INSTALLED, f"skfem import failed: {result.stderr.strip()}"
        except Exception as e:
            return BackendStatus.NOT_INSTALLED, f"Check failed: {e}"

    def input_format(self) -> InputFormat:
        return InputFormat.PYTHON

    def supported_physics(self) -> list[PhysicsCapability]:
        return [
            PhysicsCapability(
                name="poisson",
                description="Poisson equation -Δu = f (assembly-level)",
                spatial_dims=[2, 3],
                element_types=["Q1-quad", "P1-tri", "Hex1-hex", "Tet1-tet"],
                template_variants=["2d", "2d_tri", "3d"],
            ),
            PhysicsCapability(
                name="linear_elasticity",
                description="Linear elasticity (plane strain)",
                spatial_dims=[2],
                element_types=["Q1-quad-vec", "P1-tri-vec"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="heat",
                description="Steady heat conduction",
                spatial_dims=[2],
                element_types=["Q1-quad", "P1-tri"],
                template_variants=["2d", "2d_steady"],
            ),
            PhysicsCapability(
                name="stokes",
                description="Stokes flow with Taylor-Hood P2/P1 or Mini element",
                spatial_dims=[2],
                element_types=["P2-P1 Taylor-Hood", "Mini"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="eigenvalue",
                description="Eigenvalue problems (Laplace, elasticity) via scipy eigsh",
                spatial_dims=[2],
                element_types=["P1-tri", "Q1-quad"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="mixed_poisson",
                description="Mixed Poisson with Raviart-Thomas + DG (flux-conservative)",
                spatial_dims=[2],
                element_types=["RT1-P0"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="convection_diffusion",
                description="Convection-diffusion (SUPG or DG interior penalty)",
                spatial_dims=[2],
                element_types=["Q1-quad", "P1-DG"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="biharmonic",
                description="Biharmonic / Kirchhoff plate bending (Morley element)",
                spatial_dims=[2],
                element_types=["Morley", "Argyris"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="nonlinear",
                description="Nonlinear PDE with Newton iteration (manual Newton loop)",
                spatial_dims=[2],
                element_types=["Q1-quad", "P1-tri"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="heat_transient",
                description="Time-dependent heat equation with backward Euler",
                spatial_dims=[2],
                element_types=["Q1-quad"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="navier_stokes",
                description="Navier-Stokes flow with Newton iteration (Taylor-Hood P2/P1)",
                spatial_dims=[2],
                element_types=["P2-P1 Taylor-Hood"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="hyperelasticity",
                description="Neo-Hookean hyperelasticity with Newton iteration (manual assembly)",
                spatial_dims=[2],
                element_types=["P1-tri-vec", "P2-tri-vec"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="dg_methods",
                description="Discontinuous Galerkin for advection using ElementDG and InteriorFacetBasis",
                spatial_dims=[2],
                element_types=["DG-P1-tri", "DG-P2-tri"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="time_dependent",
                description="General time-dependent PDE with theta-method (backward Euler / Crank-Nicolson)",
                spatial_dims=[2],
                element_types=["Q1-quad", "P1-tri"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="helmholtz",
                description="Helmholtz equation -Δu - k²u = f with complex arithmetic and absorbing BC",
                spatial_dims=[2],
                element_types=["Q1-quad", "P1-tri"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="reaction_diffusion",
                description="Reaction-diffusion system (Schnakenberg/Turing patterns) with Newton time-stepping",
                spatial_dims=[2],
                element_types=["Q1-quad"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="wave",
                description="2D scalar wave equation u_tt - c^2 Δu = 0 with explicit central-difference time integration and lumped mass (no per-step linear solve)",
                spatial_dims=[2],
                element_types=["Q1-quad"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="adaptive_poisson",
                description="h-adaptive Poisson with Babuška-Rheinboldt residual estimator on L-shape (canonical re-entrant-corner test)",
                spatial_dims=[2],
                element_types=["P1-tri"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="point_source",
                description="Poisson with Dirac-delta point source — discrete RHS is N_i(x0) (Kronecker e_node for mesh-coincident source). Matches scikit-fem ex17 + ex38.",
                spatial_dims=[2],
                element_types=["P1-tri"],
                template_variants=["2d"],
            ),
        ]

    def get_knowledge(self, physics: str) -> dict:
        return KNOWLEDGE.get(physics, {})

    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        key = f"{physics}_{variant}"
        gen = GENERATORS.get(key)
        if not gen:
            available = ", ".join(GENERATORS.keys())
            raise ValueError(f"Unknown variant '{key}'. Available: {available}")
        return gen(params)

    def validate_input(self, content: str) -> list[str]:
        errors = []
        if "skfem" not in content:
            errors.append("Script should import from skfem")
        try:
            compile(content, "<skfem_input>", "exec")
        except SyntaxError as e:
            errors.append(f"Python syntax error: {e}")
        return errors

    async def run(self, input_content: str, work_dir: Path,
                  np: int = 1, timeout=None) -> JobHandle:
        python = get_python_executable()
        if not python:
            return JobHandle(
                job_id=str(uuid.uuid4())[:8],
                backend_name="skfem",
                work_dir=work_dir,
                status="failed",
                error="Python not found",
            )

        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        script_path = work_dir / "solve.py"
        script_path.write_text(input_content)

        job = JobHandle(
            job_id=str(uuid.uuid4())[:8],
            backend_name="skfem",
            work_dir=work_dir,
            status="running",
        )

        start = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                python, str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(work_dir),
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
        return sorted(results)


def register():
    backend = SkfemBackend()
    register_backend(backend, aliases=["skfem", "scikit-fem", "scikitfem"])
    logger.info("scikit-fem backend registered")
