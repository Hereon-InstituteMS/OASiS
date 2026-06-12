"""
FEBio solver backend.

FEBio is an open-source FEM code for biomechanics. Uses XML input files (.feb).
Specialized for soft tissue mechanics, biphasic/multiphasic problems, and
biological applications.

FEBio website: https://febio.org
GitHub: https://github.com/febiosoftware/FEBio
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
    SolverBackend, BackendStatus, InputFormat,
    PhysicsCapability, JobHandle,
)
from core.registry import register_backend
from .generators import GENERATORS as _TEMPLATES, KNOWLEDGE as _FEBIO_KNOWLEDGE

logger = logging.getLogger("oasis.febio")


def _find_febio_binary() -> Optional[Path]:
    """Locate the FEBio binary."""
    env_path = os.environ.get("FEBIO_BINARY")
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    # Common locations
    candidates = [
        Path.home() / "FEBio" / "bin" / "febio4",
        Path.home() / "FEBioStudio" / "bin" / "febio4",
        Path("/opt/febio/bin/febio4"),
        Path("/usr/local/bin/febio4"),
    ]
    for c in candidates:
        if c.is_file():
            return c

    p = shutil.which("febio4") or shutil.which("febio3") or shutil.which("febio")
    return Path(p) if p else None


class FebioBackend(SolverBackend):

    def name(self) -> str:
        return "febio"

    def display_name(self) -> str:
        return "FEBio"

    def check_availability(self) -> tuple[BackendStatus, str]:
        binary = _find_febio_binary()
        if not binary:
            return BackendStatus.NOT_INSTALLED, (
                "FEBio binary not found. Install from https://febio.org/downloads/ "
                "or set FEBIO_BINARY env var."
            )
        return BackendStatus.AVAILABLE, f"FEBio at {binary}"

    def input_format(self) -> InputFormat:
        return InputFormat.XML

    def get_version(self) -> Optional[str]:
        binary = _find_febio_binary()
        if not binary:
            return None
        import subprocess
        try:
            r = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=5)
            return r.stdout.strip() or r.stderr.strip()
        except Exception:
            return None

    def supported_physics(self) -> list[PhysicsCapability]:
        return [
            PhysicsCapability(
                name="linear_elasticity",
                description="Linear elasticity (small strain solid mechanics)",
                spatial_dims=[3],
                element_types=["hex8", "tet4", "tet10"],
                template_variants=["3d_cube"],
            ),
            PhysicsCapability(
                name="hyperelasticity",
                description="Nonlinear hyperelasticity (Neo-Hookean, Mooney-Rivlin)",
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_cube"],
            ),
            PhysicsCapability(
                name="biphasic",
                description="Biphasic poroelasticity (solid + fluid phases)",
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_confined"],
            ),
            PhysicsCapability(
                name="heat",
                description="Heat conduction (steady-state)",
                spatial_dims=[3],
                element_types=["hex8"],
                template_variants=["3d_bar"],
            ),
            PhysicsCapability(
                name="multiphasic",
                description=("Biphasic poroelasticity + solute transport "
                             "(charged-hydrated cartilage, electrolyte "
                             "diffusion, drug delivery)"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_diffusion"],
            ),
            PhysicsCapability(
                name="fluid",
                description=("Incompressible Newtonian fluid via FEBio's "
                             "pressure-velocity fluid solver "
                             "(cardiovascular CFD)"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_channel"],
            ),
            PhysicsCapability(
                name="fluid_fsi",
                description=("Strongly-coupled monolithic FSI "
                             "(arterial wall hemodynamics, cardiac "
                             "chamber dynamics, valve modeling)"),
                spatial_dims=[3],
                element_types=["hex8"],
                template_variants=["3d_block"],
            ),
            PhysicsCapability(
                name="rigid_body",
                description=("Rigid-body material (impactors, fixtures, "
                             "articulating joints, contact prescription)"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_pushdown"],
            ),
            PhysicsCapability(
                name="viscoelasticity",
                description=("Prony-series viscoelastic stress relaxation / "
                             "creep response (cartilage, ligament, tendon)"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_stress_relax"],
            ),
            PhysicsCapability(
                name="plasticity",
                description=("Rate-independent plasticity (J2 / Hill / "
                             "user-curve hardening) — cortical bone, "
                             "metal implants, surgical tools"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_uniaxial"],
            ),
            PhysicsCapability(
                name="fiber_reinforced",
                description=("Anisotropic fiber-reinforced hyperelasticity "
                             "(HGO, transversely isotropic) — arterial "
                             "wall, ligament, tendon, myocardium"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_hgo"],
            ),
            PhysicsCapability(
                name="active_contraction",
                description=("Active contractile fibers on a passive "
                             "elastic base (cardiac chamber, skeletal "
                             "muscle, peristalsis)"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_fiber"],
            ),
            PhysicsCapability(
                name="biphasic_fsi",
                description=("Coupled biphasic tissue + free-fluid FSI "
                             "(blood-tissue perfusion, drug elution, "
                             "cartilage-synovial fluid)"),
                spatial_dims=[3],
                element_types=["hex8"],
                template_variants=["3d_block"],
            ),
            PhysicsCapability(
                name="polar_fluid",
                description=("Micropolar (Cosserat) fluid with "
                             "independent micro-rotation DOFs "
                             "(blood-rheology, polymer suspensions, "
                             "near-wall turbulence corrections)"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_channel"],
            ),
            PhysicsCapability(
                name="damage",
                description=("Continuum damage mechanics — progressive "
                             "stiffness degradation under repeated "
                             "loading (tissue tearing, cartilage "
                             "wear, elastomer fatigue)"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_cycle"],
            ),
            PhysicsCapability(
                name="growth_remodeling",
                description=("Multiplicative growth-and-remodeling "
                             "F = F_e * F_g (vascular adaptation, "
                             "tissue scaffolds, muscle hypertrophy, "
                             "tumor mechanobiology)"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_isotropic"],
            ),
        ]

    def get_knowledge(self, physics: str) -> dict:
        return _FEBIO_KNOWLEDGE.get(physics, {})

    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        key = f"{physics}_{variant}"
        generator = _TEMPLATES.get(key)
        if not generator:
            raise ValueError(f"No FEBio template for {key}. "
                             f"Available: {list(_TEMPLATES.keys())}")
        return generator(params)

    def validate_input(self, content: str) -> list[str]:
        errors = []
        if "<febio_spec" not in content:
            errors.append("Missing <febio_spec> root element")
        if "<Material>" not in content and "<Material " not in content:
            errors.append("Missing Material section")
        if "<Geometry>" not in content and "<Mesh>" not in content:
            errors.append("Missing Geometry/Mesh section")
        return errors

    async def run(self, input_content: str, work_dir: Path,
                  np: int = 1, timeout=None) -> JobHandle:
        binary = _find_febio_binary()
        if not binary:
            return JobHandle(
                job_id=str(uuid.uuid4())[:8],
                backend_name="febio",
                work_dir=work_dir,
                status="failed",
                error="FEBio binary not found",
            )

        work_dir.mkdir(parents=True, exist_ok=True)
        input_file = work_dir / "input.feb"
        input_file.write_text(input_content)

        cmd = [str(binary), "-i", str(input_file)]
        job_id = str(uuid.uuid4())[:8]
        job = JobHandle(job_id=job_id, backend_name="febio", work_dir=work_dir, status="running")

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
            job.status = "completed" if proc.returncode == 0 else "failed"
            if proc.returncode != 0:
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
        for ext in ["*.xplt", "*.vtk", "*.vtu", "*.log"]:
            results.extend(job.work_dir.rglob(ext))
        return sorted(results)



def register():
    register_backend(
        FebioBackend(),
        aliases=["febio", "FEBio", "febio4"],
    )
