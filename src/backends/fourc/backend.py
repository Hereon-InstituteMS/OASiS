"""
4C Multiphysics solver backend.

Self-contained 4C interface with 10 physics generators, domain knowledge,
and input validation. Uses YAML input files (.4C.yaml).
Generators are at backends/fourc/generators/ (10 physics modules).
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

logger = logging.getLogger("open-fem-agent.fourc")

# Path resolution
FOURC_ROOT = Path(os.environ["FOURC_ROOT"]) if os.environ.get("FOURC_ROOT") else None


def _find_fourc_binary() -> Optional[Path]:
    """Locate the 4C binary."""
    env_path = os.environ.get("FOURC_BINARY")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    if FOURC_ROOT:
        for d in ["build", "build/release", "build/debug"]:
            p = FOURC_ROOT / d / "4C"
            if p.is_file():
                return p
    # Fall back to the same search paths used by the
    # autodiscovery scanner (src/core/autodiscovery.py). Without
    # this, the discover MCP tool reports 4C as installed via the
    # scanner, but get_backend('fourc').check_availability() still
    # returns NOT_INSTALLED — the two surfaces drift out of sync.
    for cand in (
        "~/4C/build/4C",
        "~/4c/build/4C",
        "/opt/4c/build/4C",
        "/opt/4C/build/4C",
        "~/Schreibtisch/4C-src/4C/build/4C",
        "~/4C-src/4C/build/4C",
    ):
        p = Path(cand).expanduser()
        if p.is_file():
            return p
    p = shutil.which("4C")
    return Path(p) if p else None


def _get_generators():
    """Import the 4C generators — self-contained in open-fem-agent."""
    # The generators package is at backends/fourc/generators/ (copied from 4c-ai-interface)
    from backends.fourc.generators import get_generator, list_generators
    return get_generator, list_generators


class FourcBackend(SolverBackend):

    def name(self) -> str:
        return "fourc"

    def display_name(self) -> str:
        return "4C Multiphysics"

    def check_availability(self) -> tuple[BackendStatus, str]:
        binary = _find_fourc_binary()
        if not binary:
            return BackendStatus.NOT_INSTALLED, "4C binary not found (set FOURC_BINARY)"
        # Check that local generators are present (self-contained)
        local_gen = Path(__file__).parent / "generators" / "__init__.py"
        if not local_gen.exists():
            return BackendStatus.MISCONFIGURED, "4C generators not found in open-fem-agent"
        return BackendStatus.AVAILABLE, f"4C at {binary}"

    def input_format(self) -> InputFormat:
        return InputFormat.YAML

    def get_version(self) -> Optional[str]:
        binary = _find_fourc_binary()
        if not binary:
            return None
        import subprocess
        try:
            r = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=5)
            for line in r.stdout.splitlines():
                if "version" in line.lower():
                    return line.strip()
        except Exception:
            pass
        return None

    def supported_physics(self) -> list[PhysicsCapability]:
        return [
            PhysicsCapability("poisson", "Poisson / scalar transport", [2, 3],
                              ["QUAD4", "HEX8", "TRI3", "TET4"],
                              ["poisson_2d", "heat_2d", "poisson_3d"]),
            PhysicsCapability("linear_elasticity", "Linear elasticity", [2, 3],
                              ["QUAD4", "HEX8"],
                              ["linear_2d", "nonlinear_3d"]),
            PhysicsCapability("plasticity", "Elasto-plasticity: J2/von Mises, Drucker-Prager, GTN damage, crystal plasticity", [2, 3],
                              ["QUAD4", "HEX8"],
                              ["linear_2d", "nonlinear_3d"]),
            PhysicsCapability("heat", "Heat conduction", [2, 3],
                              ["QUAD4", "HEX8"],
                              ["heat_2d"]),
            PhysicsCapability("fluid", "Incompressible Navier-Stokes", [2, 3],
                              ["QUAD4", "HEX8"],
                              ["channel_2d", "cavity_2d"]),
            PhysicsCapability("fsi", "Fluid-structure interaction", [2, 3],
                              ["QUAD4", "HEX8"],
                              ["fsi_2d"]),
            PhysicsCapability("structural_dynamics", "Structural dynamics", [2, 3],
                              ["QUAD4", "HEX8"],
                              ["genalpha_2d"]),
            PhysicsCapability("beams", "Beam elements", [2, 3],
                              ["BEAM3R", "BEAM3EB"],
                              ["cantilever_static", "cantilever_dynamic"]),
            PhysicsCapability("contact", "Contact mechanics", [3],
                              ["HEX8"],
                              ["penalty_3d"]),
            PhysicsCapability("particle_pd", "Peridynamics (bond-based)", [2],
                              ["particle"],
                              ["plate_2d", "impact_2d"]),
            PhysicsCapability("particle_sph", "Smoothed particle hydrodynamics", [2],
                              ["particle"],
                              ["poiseuille_2d", "dam_break_2d"]),
            PhysicsCapability("tsi", "Thermo-structure interaction", [3],
                              ["SOLIDSCATRA HEX8"],
                              ["monolithic_3d"]),
            PhysicsCapability("ssi", "Structure-scalar interaction (battery/electrode)", [3],
                              ["SOLIDSCATRA HEX8"],
                              ["monolithic_elch_3d"]),
            PhysicsCapability("ale", "ALE mesh movement", [2, 3],
                              ["ALE2", "ALE3"],
                              ["ale_2d"]),
            PhysicsCapability("electrochemistry", "Electrochemistry (Nernst-Planck)", [2, 3],
                              ["TRANSP QUAD4", "TRANSP HEX8"],
                              ["nernst_planck_3d"]),
            PhysicsCapability("level_set", "Level-set interface tracking", [2, 3],
                              ["TRANSP QUAD4"],
                              ["advection_2d"]),
            PhysicsCapability("low_mach", "Low Mach number flow (buoyancy)", [2, 3],
                              ["FLUID QUAD4"],
                              ["heated_channel_2d"]),
            PhysicsCapability("ssti", "Structure-scalar-thermo interaction (3-field)", [3],
                              ["SOLIDSCATRA HEX8"],
                              ["monolithic_3d"]),
            PhysicsCapability("sti", "Scalar-thermo interaction", [3],
                              ["TRANSP HEX8"],
                              ["monolithic_3d"]),
            PhysicsCapability("fbi", "Fluid-beam interaction (immersed)", [3],
                              ["FLUID HEX8", "BEAM3R LINE2"],
                              ["penalty_3d"]),
            PhysicsCapability("fpsi", "Fluid-porous-structure interaction", [3],
                              ["FLUID HEX8", "SOLIDPORO HEX8"],
                              ["monolithic_3d"]),
            PhysicsCapability("pasi", "Particle-structure interaction", [3],
                              ["SOLID HEX8", "particle"],
                              ["dem_impact_3d"]),
            PhysicsCapability("lubrication", "Lubrication (Reynolds equation)", [2],
                              ["LUBRICATION QUAD4"],
                              ["slider_bearing_2d"]),
            PhysicsCapability("cardiac_monodomain", "Cardiac monodomain (electrophysiology)", [3],
                              ["TRANSP HEX8"],
                              ["monodomain_3d"]),
            PhysicsCapability("arterial_network", "Arterial network (1-D blood flow)", [1],
                              ["ARTERY LINE2"],
                              ["single_artery_1d"]),
            PhysicsCapability("xfem_fluid", "XFEM fluid (embedded interfaces)", [3],
                              ["FLUID HEX8"],
                              ["xfem_3d"]),
            PhysicsCapability("fsi_xfem", "FSI XFEM (fixed-grid fluid-structure)", [3],
                              ["FLUID HEX8", "SOLID HEX8"],
                              ["xfem_fsi_3d"]),
            PhysicsCapability("fs3i", "FS3I (fluid-structure-scalar-scalar, 5-field)", [3],
                              ["FLUID HEX8", "SOLID HEX8", "TRANSP HEX8"],
                              ["fs3i_3d"]),
            PhysicsCapability("ehl", "Elastohydrodynamic lubrication", [3],
                              ["LUBRICATION QUAD4", "SOLID HEX8"],
                              ["ehl_3d"]),
            PhysicsCapability("reduced_airways", "Reduced-dimensional airways (lung)", [1],
                              ["REDAIRWAY LINE2"],
                              ["airways_1d"]),
            PhysicsCapability("beam_interaction", "Beam interaction (contact/meshtying)", [3],
                              ["BEAM3R LINE2", "SOLID HEX8"],
                              ["beam_contact_3d", "beam_solid_meshtying_3d"]),
            PhysicsCapability("multiscale", "Multiscale FE-squared (computational homogenisation)", [3],
                              ["SOLID HEX8"],
                              ["fe2_3d"]),
            PhysicsCapability("porous_media", "Poroelasticity (Biot/mixture theory, consolidation)", [2, 3],
                              ["WALLQ4PORO", "WALLQ9PORO", "SOLIDH8PORO", "SOLIDT4PORO"],
                              ["terzaghi_2d", "consolidation_3d"]),
            # New physics
            PhysicsCapability("membrane", "Membrane elements (inflatable, fabric, tissue)", [2, 3],
                              ["MEMBRANE TRI3", "MEMBRANE QUAD4"], ["membrane_2d"]),
            PhysicsCapability("shell", "Shell elements (Kirchhoff-Love, Reissner-Mindlin)", [3],
                              ["SHELL REISSNER QUAD4", "SHELL KIRCHHOFF TRI3", "SOLIDSHELL HEX8"], ["shell_3d"]),
            PhysicsCapability("thermo", "Pure thermal analysis (standalone heat conduction)", [2, 3],
                              ["THERMO QUAD4", "THERMO HEX8"], ["thermo_2d", "thermo_3d"]),
            PhysicsCapability("mixture", "Mixture/composite materials (fiber-reinforced, biological)", [3],
                              ["SOLID HEX8 with MAT_Mixture"], ["mixture_3d"]),
            PhysicsCapability("constraint", "Constraints: MPC, rigid body, periodic BCs, mortar coupling", [2, 3],
                              ["Generic"], ["constraint_3d"]),
            PhysicsCapability("brownian_dynamics", "Brownian dynamics of fiber/biopolymer networks", [3],
                              ["BEAM3R LINE2"], ["brownian_3d"]),
            PhysicsCapability("cardiovascular0d", "0-D cardiovascular: windkessel, closed-loop circulation, heart models", [3],
                              ["coupled to 3D fluid/structure"], ["windkessel_3d"]),
            PhysicsCapability("reduced_lung", "Reduced lung model: 1D airways + 0D alveoli + optional 3D parenchyma", [1, 3],
                              ["REDAIRWAY LINE2 + 0D acini"], ["lung_1d"]),
            PhysicsCapability("fluid_turbulence", "Fluid turbulence: LES (Smagorinsky, dynamic, WALE) and DNS", [2, 3],
                              ["FLUID QUAD4", "FLUID HEX8"], ["les_channel_3d"]),
            # ── 2026-06-01: umbrella catalogs from data/fourc_knowledge.py
            #    that aggregate pitfalls across families of specific
            #    physics. Previously orphaned (catalog reachable via
            #    knowledge(physics=...) but not listed in
            #    discover(physics, fourc)). Exposed here so users see
            #    the umbrella name alongside the specific ones.
            PhysicsCapability(
                "scalar_transport",
                "[Umbrella] Scalar-transport family pitfalls "
                "(applies to poisson, heat, electrochemistry, "
                "level-set, low-mach scalars). For specific "
                "physics use poisson/heat/electrochemistry "
                "directly.",
                [2, 3], ["TRANSP QUAD4", "TRANSP HEX8"],
                ["umbrella"]),
            PhysicsCapability(
                "structural_mechanics",
                "[Umbrella] Structural-mechanics family pitfalls "
                "(applies to linear_elasticity, plasticity, "
                "structural_dynamics, beams, contact). For "
                "specific physics use linear_elasticity / "
                "plasticity / structural_dynamics directly.",
                [2, 3], ["SOLID HEX8", "WALL QUAD4"],
                ["umbrella"]),
            PhysicsCapability(
                "thermal",
                "[Umbrella] Thermal-analysis family pitfalls "
                "(applies to heat, thermo, tsi). For specific "
                "physics use heat / thermo / tsi directly.",
                [2, 3], ["THERMO QUAD4", "THERMO HEX8"],
                ["umbrella"]),
            PhysicsCapability(
                "particles",
                "[Umbrella] Particle-methods family pitfalls "
                "(applies to particle_pd, particle_sph, "
                "pasi, dem). For specific physics use "
                "particle_pd / particle_sph / pasi directly.",
                [2, 3], ["particle"],
                ["umbrella"]),
        ]

    def get_knowledge(self, physics: str) -> dict:
        # Try deep knowledge from data file first
        try:
            import sys
            data_dir = str(Path(__file__).resolve().parents[3] / "data")
            if data_dir not in sys.path:
                sys.path.insert(0, data_dir)
            from fourc_knowledge import FOURC_KNOWLEDGE
            if physics in FOURC_KNOWLEDGE:
                return FOURC_KNOWLEDGE[physics]
        except ImportError:
            pass
        # Fall back to generator-based knowledge
        try:
            get_gen, _ = _get_generators()
            gen = get_gen(physics)
            return gen.get_knowledge()
        except Exception as e:
            return {"error": str(e)}

    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        # First try inline mesh generators (self-contained, no external files)
        try:
            return self._generate_inline(physics, variant, params)
        except ValueError:
            pass

        # Then try tutorial-based templates (these include mesh files)
        try:
            return self._generate_from_tutorial(physics, variant, params)
        except ValueError:
            pass

        # Fallback: try generator-based templates
        try:
            get_gen, _ = _get_generators()
            gen = get_gen(physics)
            content = gen.get_template(variant)
            content = self._resolve_mesh_references(content)
            return content
        except Exception as e:
            raise ValueError(f"No 4C template for {physics}/{variant}: {e}")

    def _generate_inline(self, physics: str, variant: str, params: dict) -> str:
        """Generate self-contained input with inline mesh (no external files)."""
        from backends.fourc.inline_mesh import (
            matched_poisson_input, matched_heat_input,
            matched_elasticity_input, matched_poisson_3d_input,
            matched_l_domain_poisson_input,
        )
        key = f"{physics}_{variant}"
        inline_generators = {
            "poisson_2d": lambda p: matched_poisson_input(
                nx=p.get("nx", 32), ny=p.get("ny", 32)),
            "poisson_poisson_2d": lambda p: matched_poisson_input(
                nx=p.get("nx", 32), ny=p.get("ny", 32)),
            "heat_2d": lambda p: matched_heat_input(
                nx=p.get("nx", 32), ny=p.get("ny", 32),
                T_left=p.get("T_left", 100.0), T_right=p.get("T_right", 0.0)),
            "heat_heat_2d": lambda p: matched_heat_input(
                nx=p.get("nx", 32), ny=p.get("ny", 32),
                T_left=p.get("T_left", 100.0), T_right=p.get("T_right", 0.0)),
            "linear_elasticity_linear_2d": lambda p: matched_elasticity_input(
                nx=p.get("nx", 40), ny=p.get("ny", 4),
                E=p.get("E", 1000.0), nu=p.get("nu", 0.3),
                lx=p.get("lx", 10.0), ly=p.get("ly", 1.0)),
            "linear_elasticity_2d": lambda p: matched_elasticity_input(
                nx=p.get("nx", 40), ny=p.get("ny", 4),
                E=p.get("E", 1000.0), nu=p.get("nu", 0.3),
                lx=p.get("lx", 10.0), ly=p.get("ly", 1.0)),
            "poisson_3d": lambda p: matched_poisson_3d_input(n=p.get("n", 8)),
            "poisson_poisson_3d": lambda p: matched_poisson_3d_input(n=p.get("n", 8)),
            "poisson_l_domain": lambda p: matched_l_domain_poisson_input(
                n=p.get("n", 16)),
        }
        gen = inline_generators.get(key)
        if gen is None:
            raise ValueError(f"No inline generator for {key}")
        return gen(params)

    def _generate_from_tutorial(self, physics: str, variant: str, params: dict) -> str:
        """Generate input from 4C tutorial examples (with mesh files)."""
        tutorials = {
            # Poisson / scalar transport
            "poisson_poisson_2d": ("tutorials/poisson/tutorial_poisson_scatra.4C.yaml",
                                    "tutorials/poisson/tutorial_poisson_geo.e"),
            "poisson_heat_2d": ("tutorials/poisson/tutorial_poisson_thermo.4C.yaml",
                                 "tutorials/poisson/tutorial_poisson_geo.e"),
            "heat_heat_2d": ("tutorials/poisson/tutorial_poisson_thermo.4C.yaml",
                              "tutorials/poisson/tutorial_poisson_geo.e"),
            # Solid mechanics
            "linear_elasticity_linear_2d": ("tutorials/solid/tutorial_solid.4C.yaml",
                                             "tutorials/solid/tutorial_solid_geo_coarse.e"),
            "linear_elasticity_solid_tutorial": ("tutorials/solid/tutorial_solid.4C.yaml",
                                                  "tutorials/solid/tutorial_solid_geo_coarse.e"),
            # Fluid
            "fluid_channel_2d": ("tutorials/fluid/tutorial_fluid.4C.yaml",
                                  "tutorials/fluid/tutorial_fluid.e"),
            "fluid_cavity_2d": ("tutorials/fluid/tutorial_fluid.4C.yaml",
                                 "tutorials/fluid/tutorial_fluid.e"),
            # FSI
            "fsi_fsi_2d": ("tutorials/fsi/tutorial_fsi_2d.4C.yaml",
                            "tutorials/fsi/tutorial_fsi_2d.e"),
            "fsi_fsi_monolithic": ("tutorials/fsi/tutorial_fsi_monolithic.4C.yaml",
                                    "tutorials/fsi/tutorial_fsi_2d.e"),
            "fsi_fsi_3d": ("tutorials/fsi/tutorial_fsi_3d.4C.yaml",
                            "tutorials/fsi/tutorial_fsi_3d.e"),
            # Contact
            "contact_penalty_3d": ("tutorials/contact/tutorial_contact_3d.4C.yaml",
                                    "tutorials/contact/tutorial_contact_3d.e"),
        }
        key = f"{physics}_{variant}"
        if key not in tutorials:
            raise ValueError(f"No 4C tutorial for {key}")

        if not FOURC_ROOT:
            raise ValueError("FOURC_ROOT not set")

        yaml_path = FOURC_ROOT / "tests" / tutorials[key][0]
        if not yaml_path.exists():
            raise ValueError(f"Tutorial file not found: {yaml_path}")

        content = yaml_path.read_text()
        mesh_rel = tutorials[key][1]
        mesh_path = FOURC_ROOT / "tests" / mesh_rel
        if mesh_path.exists():
            content = f"# MESH_FILE: {mesh_path}\n" + content
        return content

    def _resolve_mesh_references(self, content: str) -> str:
        """Find mesh file references in YAML and add MESH_FILE metadata."""
        import re
        # Look for FILE: xxx.e pattern
        match = re.search(r'FILE:\s*(\S+\.e)\b', content)
        if match and FOURC_ROOT:
            mesh_name = match.group(1)
            # Search for the mesh in tests/
            for mesh_path in FOURC_ROOT.rglob(mesh_name):
                content = f"# MESH_FILE: {mesh_path}\n" + content
                break
        return content

    def validate_input(self, content: str) -> list[str]:
        import yaml
        errors = []
        try:
            data = yaml.safe_load(content)
            if not isinstance(data, dict):
                errors.append("Input is not a YAML dictionary")
                return errors
            if "PROBLEM TYPE" not in data:
                errors.append("Missing PROBLEM TYPE section")
            if "MATERIALS" not in data:
                errors.append("Missing MATERIALS section")
        except yaml.YAMLError as e:
            errors.append(f"YAML parse error: {e}")
        return errors

    async def run(self, input_content: str, work_dir: Path,
                  np: int = 1, timeout=None) -> JobHandle:
        binary = _find_fourc_binary()
        if not binary:
            return JobHandle(
                job_id=str(uuid.uuid4())[:8],
                backend_name="fourc",
                work_dir=work_dir,
                status="failed",
                error="4C binary not found",
            )

        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        # Extract mesh file path if embedded in content
        mesh_src = None
        lines = input_content.splitlines()
        if lines and lines[0].startswith("# MESH_FILE: "):
            mesh_src = Path(lines[0].split(": ", 1)[1].strip())
            input_content = "\n".join(lines[1:])

        input_file = work_dir / "input.4C.yaml"
        input_file.write_text(input_content)

        # Copy mesh file if referenced
        if mesh_src and mesh_src.exists():
            import shutil as _shutil
            _shutil.copy2(mesh_src, work_dir / mesh_src.name)

        output_prefix = str(work_dir / "output")

        mpirun = shutil.which("mpirun")
        max_procs = int(os.environ.get("FOURC_MAX_PROCS", "4"))
        np = min(np, max_procs)

        # Wrap with stdbuf -oL to force line-buffered stdout.
        # 4C writes errors to stdout (buffered) then calls MPI_Abort which
        # kills the process before flushing — stdbuf prevents lost messages.
        stdbuf = shutil.which("stdbuf")

        if np > 1 and mpirun:
            base_cmd = [mpirun, "-np", str(np), str(binary), str(input_file), output_prefix]
        else:
            base_cmd = [str(binary), str(input_file), output_prefix]

        cmd = [stdbuf, "-oL"] + base_cmd if stdbuf else base_cmd

        job_id = str(uuid.uuid4())[:8]
        job = JobHandle(job_id=job_id, backend_name="fourc", work_dir=work_dir, status="running")

        start = time.time()
        try:
            env = os.environ.copy()
            # Ensure 4C dependencies are on the library path
            ld_path = env.get("LD_LIBRARY_PATH", "")
            dep_lib = "/opt/4C-dependencies/lib"
            if dep_lib not in ld_path:
                ld_path = f"{dep_lib}:{ld_path}" if ld_path else dep_lib
            env["LD_LIBRARY_PATH"] = ld_path

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
            job.status = "completed" if proc.returncode == 0 else "failed"
            if proc.returncode != 0:
                # 4C often writes the real error to stdout, not stderr
                stdout_text = stdout.decode(errors="replace")
                stderr_text = stderr.decode(errors="replace")
                job.error = (stderr_text[-1000:] + "\n--- stdout tail ---\n" + stdout_text[-1000:])[-2000:]
            else:
                # Skip post_vtu — 4C writes VTU directly via IO/RUNTIME VTK OUTPUT.
                # post_vtu is only needed for legacy .control/.result files and
                # has caused server hangs/bottlenecks. All our templates include
                # the VTK output sections, so post_vtu is unnecessary.
                pass
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

    async def _run_post_vtu(self, work_dir: Path):
        """Launch post_vtu in the background (fire-and-forget).

        Does NOT block the MCP server. VTU files from IO/RUNTIME VTK OUTPUT
        are usually already written during the simulation — post_vtu is a
        best-effort fallback for additional field conversion.

        The process runs independently; if it finishes, VTU files appear.
        If it hangs or fails, no harm done — the simulation result is already
        returned to the agent.
        """
        post_vtu = None
        if FOURC_ROOT:
            for d in ["build", "build/release"]:
                p = FOURC_ROOT / d / "post_vtu"
                if p.is_file():
                    post_vtu = p
                    break
        if not post_vtu:
            post_vtu_path = shutil.which("post_vtu")
            if post_vtu_path:
                post_vtu = Path(post_vtu_path)

        if not post_vtu:
            return

        control_files = list(work_dir.glob("*.control"))
        if not control_files:
            return

        for ctrl in control_files:
            prefix = str(ctrl).replace(".control", "")
            try:
                env = os.environ.copy()
                ld = env.get("LD_LIBRARY_PATH", "")
                dep_lib = "/opt/4C-dependencies/lib"
                if dep_lib not in ld:
                    ld = f"{dep_lib}:{ld}" if ld else dep_lib
                env["LD_LIBRARY_PATH"] = ld

                # Fire-and-forget: launch post_vtu without waiting
                proc = await asyncio.create_subprocess_exec(
                    str(post_vtu), f"--file={prefix}",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    cwd=str(work_dir),
                    env=env,
                )
                logger.info(f"post_vtu launched for {ctrl.name} (PID {proc.pid}, background)")
            except Exception as e:
                logger.warning(f"post_vtu launch failed for {ctrl.name}: {e}")

    def get_result_files(self, job: JobHandle) -> list[Path]:
        results = []
        for ext in ["*.vtu", "*.pvd", "*.pvtu"]:
            results.extend(job.work_dir.rglob(ext))
        return sorted(results)


def register():
    register_backend(
        FourcBackend(),
        aliases=["4c", "4C", "fourc", "four_c"],
    )
