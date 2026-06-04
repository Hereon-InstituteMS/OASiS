"""
DUNE-fem solver backend.

DUNE-fem uses UFL (same form language as FEniCS) with Netgen/ALUGrid meshes.
Generates Python scripts, executes them, collects VTK output.
VTK output is native via gridView.writeVTK() — no conversion needed.

Key advantage: shares UFL with FEniCS, so physics descriptions are nearly
interchangeable. Different backend (DUNE C++ infrastructure vs PETSc).
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

logger = logging.getLogger("oasis.dune")


class DuneBackend(SolverBackend):

    def name(self) -> str:
        return "dune"

    def display_name(self) -> str:
        return "DUNE-fem"

    def check_availability(self) -> tuple[BackendStatus, str]:
        # DUNE-fem is heavy enough that users typically install it
        # into a dedicated conda env (ofa-dune is the convention
        # mirroring ofa-fenicsx / ofa-dealii). The MCP server runs
        # in the oasis .venv which usually does NOT have
        # dune.fem.
        #
        # Search order (priority lowest-first):
        #   0 — any conda env with "dune" in its name (ofa-dune,
        #       dune, dune-fem, etc.) — the most likely place.
        #   1 — any other conda env (covers the case where dune.fem
        #       was installed alongside fenicsx in a single shared
        #       env).
        #   99 — the active interpreter (.venv) — checked last
        #       because dune.fem in the harness's own .venv is rare
        #       and probing it first would slow down the common
        #       case. Still checked so an unusual install layout
        #       gets picked up.
        # (Audit 2026-06-01; comment corrected 2026-06-02 — the
        # earlier "Check the active interpreter first" was wrong.)
        import subprocess
        candidates = []
        active = get_python_executable()
        if active:
            candidates.append(active)
        for conda_base in (
                Path.home() / "miniconda3" / "envs",
                Path.home() / "anaconda3" / "envs",
                Path.home() / "miniforge3" / "envs"):
            if not conda_base.is_dir():
                continue
            for env_dir in conda_base.iterdir():
                py = env_dir / "bin" / "python"
                if py.is_file():
                    # Prioritise *dune* envs but accept any
                    name = env_dir.name.lower()
                    priority = 0 if "dune" in name else 1
                    candidates.append((priority, str(py)))
        # De-dup while preserving order; sort by priority where annotated.
        seen = set()
        ordered = []
        for c in candidates:
            py = c[1] if isinstance(c, tuple) else c
            if py in seen:
                continue
            seen.add(py)
            ordered.append((c[0] if isinstance(c, tuple) else 99, py))
        ordered.sort(key=lambda x: x[0])
        for _, python in ordered:
            try:
                result = subprocess.run(
                    [python, "-c", "import dune.fem; print('OK')"],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    return BackendStatus.AVAILABLE, f"DUNE-fem at {python}"
            except Exception:
                continue
        return BackendStatus.NOT_INSTALLED, (
            "dune.fem not importable in any candidate Python. "
            "Try: conda create -n ofa-dune -c conda-forge dune-fem")

    def input_format(self) -> InputFormat:
        return InputFormat.PYTHON

    def supported_physics(self) -> list[PhysicsCapability]:
        return [
            PhysicsCapability(
                name="poisson",
                description="Poisson equation -Δu = f (UFL forms, DUNE backend)",
                spatial_dims=[2, 3],
                element_types=["Lagrange-P1", "Lagrange-P2"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="heat",
                description="Steady heat conduction (UFL)",
                spatial_dims=[2],
                element_types=["Lagrange-P1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="linear_elasticity",
                description="Linear elasticity (UFL vector space)",
                spatial_dims=[2],
                element_types=["Lagrange-P1-vec"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="stokes",
                description="Stokes flow with Uzawa iteration (UFL)",
                spatial_dims=[2],
                element_types=["Lagrange-P2 + Lagrange-P1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="reaction_diffusion",
                description="Reaction-diffusion (transient)",
                spatial_dims=[2],
                element_types=["Lagrange-P1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="nonlinear",
                description="Nonlinear PDE via Newton method (UFL)",
                spatial_dims=[2],
                element_types=["Lagrange-P1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="dg_advection",
                description="DG method for pure advection equation (upwind flux)",
                spatial_dims=[2],
                element_types=["DG-Lagrange-P1", "DG-Lagrange-P2"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="adaptive_poisson",
                description="h-adaptive Poisson with residual error estimator",
                spatial_dims=[2],
                element_types=["Lagrange-P1", "Lagrange-P2"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="maxwell",
                description="Maxwell equations: 2-D TE-mode scalar Helmholtz proxy (-Δu - k²u = f); full H(curl) via NGSolve",
                spatial_dims=[2],
                element_types=["Lagrange-P2"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="eigenvalue",
                description="Eigenvalue problem -Δu = λu via shift-invert inverse iteration",
                spatial_dims=[2],
                element_types=["Lagrange-P1", "Lagrange-P2"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="hyperelasticity",
                description="Neo-Hookean finite-strain hyperelasticity with automatic Newton via UFL",
                spatial_dims=[2],
                element_types=["Lagrange-P1-vec", "Lagrange-P2-vec"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="navier_stokes",
                description="Incompressible Navier-Stokes via Picard iteration (lid-driven cavity)",
                spatial_dims=[2],
                element_types=["Lagrange-P2-vec + Lagrange-P1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="helmholtz",
                description="Helmholtz equation -Δu - k²u = f with MMS verification",
                spatial_dims=[2],
                element_types=["Lagrange-P2", "Lagrange-P3"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="time_dependent_heat",
                description="Transient heat du/dt - alpha*Δu = f via implicit Euler time-stepping",
                spatial_dims=[2],
                element_types=["Lagrange-P1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="mixed_methods",
                description="Mixed Poisson with Raviart-Thomas RT0 flux and DG-P0 pressure (H(div) x L²)",
                spatial_dims=[2],
                element_types=["RT0 + DG-P0", "RT1 + DG-P1"],
                template_variants=["2d"],
            ),
        ]

    def get_knowledge(self, physics: str) -> dict:
        return KNOWLEDGE.get(physics, {})

    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        key = f"{physics}_{variant}"
        gen = GENERATORS.get(key)
        if not gen:
            raise ValueError(f"No DUNE template for {key}")
        return gen(params)

    def validate_input(self, content: str) -> list[str]:
        errors = []
        if "dune" not in content:
            errors.append("Script should import from dune")
        try:
            compile(content, "<dune_input>", "exec")
        except SyntaxError as e:
            errors.append(f"Python syntax error: {e}")
        return errors

    async def run(self, input_content: str, work_dir: Path,
                  np: int = 1, timeout=None) -> JobHandle:
        python = get_python_executable()
        if not python:
            return JobHandle(
                job_id=str(uuid.uuid4())[:8],
                backend_name="dune",
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
            backend_name="dune",
            work_dir=work_dir,
            status="running",
        )

        cmd = [python, str(script_path)]
        start = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(work_dir),
            )
            # DUNE JIT compiles on first run — can be slow
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
            job.error = f"Timed out after {timeout}s (DUNE JIT compilation can be slow on first run)"
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
    backend = DuneBackend()
    register_backend(backend, aliases=["dune", "dune-fem", "dunefem"])
    logger.info("DUNE-fem backend registered")
