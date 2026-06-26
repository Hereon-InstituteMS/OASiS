"""
NGSolve solver backend.

Generates Python scripts using the NGSolve/Netgen API, executes them,
and collects VTK output files. NGSolve produces VTK natively via
VTKOutput — no post-conversion needed (unlike FEniCS XDMF).
"""

import asyncio
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
    PhysicsCapability, JobHandle,
)
from core.registry import register_backend
from .generators import GENERATORS, KNOWLEDGE

logger = logging.getLogger("oasis.ngsolve")


def _find_ngsolve_python() -> Optional[Path]:
    """Locate the Python binary with ngsolve installed."""
    import sys
    # 1. The Python running this server (venv-aware)
    return Path(sys.executable)


class NgsolveBackend(SolverBackend):

    def name(self) -> str:
        return "ngsolve"

    def display_name(self) -> str:
        return "NGSolve"

    def check_availability(self) -> tuple[BackendStatus, str]:
        python = _find_ngsolve_python()
        if not python:
            return BackendStatus.NOT_INSTALLED, "No Python with ngsolve found"

        import subprocess
        try:
            result = subprocess.run(
                [str(python), "-c", "import ngsolve; print(ngsolve.__version__)"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                ver = result.stdout.strip()
                return BackendStatus.AVAILABLE, f"NGSolve {ver} at {python}"
            else:
                return BackendStatus.NOT_INSTALLED, f"ngsolve import failed: {result.stderr.strip()}"
        except Exception as e:
            return BackendStatus.NOT_INSTALLED, f"Check failed: {e}"

    def input_format(self) -> InputFormat:
        return InputFormat.PYTHON

    def supported_physics(self) -> list[PhysicsCapability]:
        return [
            PhysicsCapability(
                name="poisson",
                description="Poisson equation -Δu = f with arbitrary-order H1 elements",
                spatial_dims=[2, 3],
                element_types=["H1-P1", "H1-P2", "H1-Pk"],
                template_variants=["2d", "3d"],
            ),
            PhysicsCapability(
                name="linear_elasticity",
                description="Linear elasticity (plane strain / 3D) with VectorH1",
                spatial_dims=[2, 3],
                element_types=["VectorH1-P1", "VectorH1-P2"],
                template_variants=["2d", "3d"],
            ),
            PhysicsCapability(
                name="heat",
                description="Heat conduction (steady and transient with implicit Euler)",
                spatial_dims=[2, 3],
                element_types=["H1-P1", "H1-Pk"],
                template_variants=["2d", "2d_steady", "2d_transient"],
            ),
            PhysicsCapability(
                name="stokes",
                description="Stokes flow with Taylor-Hood P2/P1 or HDG",
                spatial_dims=[2, 3],
                element_types=["VectorH1-P2 + H1-P1", "HDG"],
                template_variants=["2d", "2d_hdg"],
            ),
            PhysicsCapability(
                name="navier_stokes",
                description="Incompressible Navier-Stokes (IMEX time-stepping)",
                spatial_dims=[2],
                element_types=["VectorH1-P2 + H1-P1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="maxwell",
                description="Maxwell's equations with HCurl (Nedelec) elements",
                spatial_dims=[3],
                element_types=["HCurl-Nedelec"],
                template_variants=["3d_magnetostatics"],
            ),
            PhysicsCapability(
                name="helmholtz",
                description="Helmholtz equation with PML (complex-valued)",
                spatial_dims=[2],
                element_types=["H1-Pk-complex"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="hyperelasticity",
                description="Nonlinear hyperelasticity (Neo-Hookean) via SymbolicEnergy",
                spatial_dims=[2, 3],
                element_types=["VectorH1-P2"],
                template_variants=["2d", "3d"],
            ),
            PhysicsCapability(
                name="eigenvalue",
                description="Eigenvalue problems (Laplace, elasticity) via ArnoldiSolver",
                spatial_dims=[2],
                element_types=["H1-Pk"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="convection_diffusion",
                description="Convection-diffusion with DG upwind stabilization",
                spatial_dims=[2],
                element_types=["L2-DG"],
                template_variants=["2d_dg"],
            ),
            PhysicsCapability(
                name="mixed_poisson",
                description="Mixed Poisson with H(div)/L2 (flux recovery)",
                spatial_dims=[2],
                element_types=["HDiv-RT + L2"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="thermal_structural",
                description="Coupled thermal-structural (heat -> elasticity with thermal strain)",
                spatial_dims=[2],
                element_types=["H1 + VectorH1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="surface_pde",
                description="PDE on curved surface manifold (Laplace-Beltrami)",
                spatial_dims=[3],
                element_types=["H1-surface"],
                template_variants=["3d"],
            ),
            PhysicsCapability(
                name="plasticity",
                description="Elasto-plasticity with isotropic hardening (J2/von Mises)",
                spatial_dims=[2],
                element_types=["VectorH1-P2"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="dg_methods",
                description="Interior-penalty DG (SIPG) for advection-diffusion using L2 dglagrange space",
                spatial_dims=[2],
                element_types=["L2-DG-Pk"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="contact",
                description="Unilateral contact / obstacle problem via penalty method on elastic domain",
                spatial_dims=[2],
                element_types=["VectorH1-P2"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="time_dependent_ns",
                description="Transient incompressible Navier-Stokes with IMEX splitting (full channel/cavity)",
                spatial_dims=[2],
                element_types=["VectorH1-P2 + H1-P1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="mhd",
                description="Magnetohydrodynamics: coupled NS + Maxwell, 2.5-D low-Rm Hartmann problem",
                spatial_dims=[2],
                element_types=["VectorH1-P2 + H1-P1", "H1-P2 (magnetic scalar)"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="hdivdiv",
                description="HDivDiv Hellan-Herrmann-Johnson for Kirchhoff plate bending / biharmonic",
                spatial_dims=[2],
                element_types=["HDivDiv + H1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="nonlinear_elasticity",
                description="Large-deformation Neo-Hookean hyperelasticity with load stepping and Cauchy stress output",
                spatial_dims=[2, 3],
                element_types=["VectorH1-P2"],
                template_variants=["2d", "3d"],
            ),
            PhysicsCapability(
                name="phase_field",
                description="Phase-field: Allen-Cahn (interface motion) and fracture (Bourdin staggered scheme)",
                spatial_dims=[2],
                element_types=["H1-Pk", "VectorH1 + H1 (fracture)"],
                template_variants=["2d", "fracture_2d"],
            ),
        ]

    def get_knowledge(self, physics: str) -> dict:
        return KNOWLEDGE.get(physics, {})

    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        key = f"{physics}_{variant}"
        gen = GENERATORS.get(key)
        if not gen:
            available = ", ".join(sorted(GENERATORS.keys()))
            raise ValueError(f"Unknown variant '{key}'. Available: {available}")
        return gen(params)

    def validate_input(self, content: str) -> list[str]:
        errors = []
        if "ngsolve" not in content and "from ngsolve" not in content:
            errors.append("Script should import from ngsolve")
        try:
            compile(content, "<ngsolve_input>", "exec")
        except SyntaxError as e:
            errors.append(f"Python syntax error: {e}")
        return errors

    async def run(self, input_content: str, work_dir: Path,
                  np: int = 1, timeout=None) -> JobHandle:
        python = _find_ngsolve_python()
        if not python:
            return JobHandle(
                job_id=str(uuid.uuid4())[:8],
                backend_name="ngsolve",
                work_dir=work_dir,
                status="failed",
                error="ngsolve not found",
            )

        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        script_path = work_dir / "solve.py"
        script_path.write_text(input_content)

        job_id = str(uuid.uuid4())[:8]
        job = JobHandle(
            job_id=job_id,
            backend_name="ngsolve",
            work_dir=work_dir,
            status="running",
        )

        # NGSolve supports MPI via mpi4py but most problems run serial
        cmd = [str(python), str(script_path)]

        start = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
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
        return sorted_by_step(results)


# ─── Registration ────────────────────────────────────────────────────────

def register():
    backend = NgsolveBackend()
    register_backend(backend, aliases=["ngsolve", "ngs"])
    logger.info("NGSolve backend registered")
