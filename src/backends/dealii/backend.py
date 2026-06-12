"""
deal.ii solver backend.

Generates C++ source files based on deal.ii tutorial step patterns,
compiles them with CMake, and runs the resulting executables.

deal.ii tutorials used:
  - step-3/4/5: Poisson / Laplace equation
  - step-8/17:  Linear elasticity
  - step-26:    Heat equation (transient)
  - step-22:    Stokes flow
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

logger = logging.getLogger("oasis.dealii")


def _find_dealii() -> Optional[Path]:
    """Locate a deal.II installation root.

    Discovery order (first hit wins):
      1. ``DEAL_II_DIR`` env variable (explicit override).
      2. ``DEALII_ROOT`` env variable (alternate spelling).
      3. Conda envs at ``~/miniconda3/envs/*`` and
         ``~/anaconda3/envs/*`` that contain ``include/deal.II/``.
      4. User-source dirs: ``~/dealii``, ``~/deal.II``,
         ``~/Schreibtisch/dealii``, ``~/Schreibtisch/deal.II``,
         ``~/src/dealii``, ``~/src/deal.II``.
      5. System paths: ``/opt/dealii``,
         ``/usr/lib/x86_64-linux-gnu/cmake/deal.II``,
         ``/usr/share/cmake/deal.II``.
      6. ``cmake --find-package`` for system-installed deal.II.

    Returns the install ROOT (the path that contains
    ``include/deal.II/`` or ``share/deal.II/cmake/``). Callers
    pass this to ``find_package(deal.II HINTS ...)``.
    """
    # 1. Explicit env override
    for env_var in ("DEAL_II_DIR", "DEALII_ROOT"):
        env_dir = os.environ.get(env_var)
        if env_dir and Path(env_dir).is_dir():
            return Path(env_dir)

    def _looks_like_dealii_root(p: Path) -> bool:
        """A path is a deal.II install root if it has either
        include/deal.II/ headers or share/deal.II/cmake/ macros
        or lib/cmake/deal.II/ config."""
        if not p.is_dir():
            return False
        return ((p / "include" / "deal.II").is_dir()
                or (p / "share" / "deal.II" / "cmake").is_dir()
                or (p / "lib" / "cmake" / "deal.II").is_dir())

    # 2 + 3. Conda envs (deal.II often lives in a dedicated env).
    # When several envs contain deal.II, prefer the HIGHEST version:
    # iterdir() order is arbitrary, and on this machine an old
    # ofa-dealii (9.1.1, serial, missing fe_interface_values.h /
    # hp/refinement.h / count_dofs_per_fe_block) used to shadow the
    # newer ofa-dealii-93 (9.3.2) depending on directory order —
    # 10 of 39 catalog templates failed to compile purely from
    # losing that race (probe 2026-06-12).
    def _dealii_version_of(root: Path) -> tuple:
        cfg = root / "include" / "deal.II" / "base" / "config.h"
        try:
            for line in cfg.read_text().splitlines():
                if "DEAL_II_PACKAGE_VERSION" in line and '"' in line:
                    ver = line.split('"')[1]
                    return tuple(int(x) for x in ver.split(".")[:3])
        except (OSError, ValueError):
            pass
        return (0, 0, 0)

    candidates: list[Path] = []
    for conda_base in (Path.home() / "miniconda3" / "envs",
                       Path.home() / "anaconda3" / "envs",
                       Path.home() / "miniforge3" / "envs"):
        if not conda_base.is_dir():
            continue
        for env_dir in conda_base.iterdir():
            if _looks_like_dealii_root(env_dir):
                candidates.append(env_dir)
    if candidates:
        return max(candidates, key=_dealii_version_of)

    # 4. User-source dirs (in case the user built from source).
    for sub in ("dealii", "deal.II", "src/dealii", "src/deal.II",
                "Schreibtisch/dealii", "Schreibtisch/deal.II"):
        candidate = Path.home() / sub
        if _looks_like_dealii_root(candidate):
            return candidate
        # Also try common build-subdir layouts.
        for build_sub in ("install", "build/install"):
            inner = candidate / build_sub
            if _looks_like_dealii_root(inner):
                return inner

    # 5. System paths.
    for cand in (Path("/opt/dealii"), Path("/opt/deal.II"),
                 Path("/usr/local/dealii"),
                 Path("/usr/local/deal.II"),
                 Path("/usr")):
        if _looks_like_dealii_root(cand):
            return cand

    # 6. CMake fall-back.
    cmake = shutil.which("cmake")
    if cmake:
        import subprocess
        try:
            r = subprocess.run(
                [cmake, "--find-package", "-DNAME=deal.II",
                 "-DCOMPILER_ID=GNU", "-DLANGUAGE=CXX",
                 "-DMODE=COMPILE"],
                capture_output=True, text=True, timeout=10
            )
            if r.returncode == 0:
                return Path("/usr")  # system-installed
        except Exception:
            pass

    return None


class DealiiBackend(SolverBackend):

    def name(self) -> str:
        return "dealii"

    def display_name(self) -> str:
        return "deal.II"

    def check_availability(self) -> tuple[BackendStatus, str]:
        # Check for cmake
        cmake = shutil.which("cmake")
        if not cmake:
            return BackendStatus.NOT_INSTALLED, "CMake not found"

        # Check for deal.II headers/library
        dealii = _find_dealii()
        if not dealii:
            # Try a test compile
            return self._check_via_compile()

        return BackendStatus.AVAILABLE, f"deal.II found at {dealii}"

    def _check_via_compile(self) -> tuple[BackendStatus, str]:
        """Try to compile a minimal deal.II program to check availability."""
        import subprocess
        import tempfile

        test_cpp = '#include <deal.II/base/utilities.h>\nint main(){return 0;}\n'
        test_cmake = (
            'cmake_minimum_required(VERSION 3.1)\n'
            'find_package(deal.II REQUIRED)\n'
            'deal_ii_initialize_cached_variables()\n'
            'project(test)\n'
            'deal_ii_setup_target(test)\n'
            'add_executable(test test.cpp)\n'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test.cpp").write_text(test_cpp)
            Path(tmpdir, "CMakeLists.txt").write_text(test_cmake)
            try:
                r = subprocess.run(
                    ["cmake", "."], capture_output=True, text=True,
                    cwd=tmpdir, timeout=30
                )
                if r.returncode == 0:
                    return BackendStatus.AVAILABLE, "deal.II found via CMake"
                else:
                    return BackendStatus.NOT_INSTALLED, f"CMake cannot find deal.II: {r.stderr[:200]}"
            except Exception as e:
                return BackendStatus.NOT_INSTALLED, f"Check failed: {e}"

    def input_format(self) -> InputFormat:
        return InputFormat.CPP

    def get_version(self) -> Optional[str]:
        dealii = _find_dealii()
        if not dealii:
            return None
        # Try to read version from cmake config
        for f in dealii.rglob("deal.IIConfig.cmake"):
            text = f.read_text()
            for line in text.splitlines():
                if "DEAL_II_VERSION" in line and "SET" in line:
                    parts = line.split('"')
                    if len(parts) >= 2:
                        return parts[1]
        return "9.x (version detection failed)"

    def supported_physics(self) -> list[PhysicsCapability]:
        return [
            PhysicsCapability(
                name="poisson",
                description="Poisson / Laplace equation (step-3/6, with AMR, L-domain, rectangle)",
                spatial_dims=[2, 3],
                element_types=["Q1", "Q2"],
                template_variants=["2d", "3d", "l_domain", "rectangle", "2d_adaptive"],
            ),
            PhysicsCapability(
                name="linear_elasticity",
                description="Linear elasticity (step-8, with thick beam variant)",
                spatial_dims=[2],
                element_types=["Q1"],
                template_variants=["2d", "thick_beam"],
            ),
            PhysicsCapability(
                name="heat",
                description="Heat equation (transient step-26 and steady-state, with rectangle)",
                spatial_dims=[2],
                element_types=["Q1"],
                template_variants=["2d_transient", "2d_steady", "rectangle"],
            ),
            PhysicsCapability(
                name="stokes",
                description="Stokes flow (step-22, Taylor-Hood Q2/Q1, block preconditioner)",
                spatial_dims=[2],
                element_types=["Q2-Q1 (Taylor-Hood)"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="convection_diffusion",
                description="Convection-diffusion with SUPG stabilization (step-9 based)",
                spatial_dims=[2],
                element_types=["Q1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="nonlinear",
                description="Nonlinear PDE (minimal surface, step-15, Newton method)",
                spatial_dims=[2],
                element_types=["Q1"],
                template_variants=["2d_minimal_surface"],
            ),
            PhysicsCapability(
                name="helmholtz",
                description="Helmholtz equation (complex-valued, step-29 inspired)",
                spatial_dims=[2],
                element_types=["Q1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="eigenvalue",
                description="Eigenvalue problems via SLEPc (step-36 inspired)",
                spatial_dims=[2],
                element_types=["Q1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="wave",
                description="Wave equation with Newmark time integration (step-23 inspired)",
                spatial_dims=[2],
                element_types=["Q1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="hp_adaptive",
                description="hp-adaptive FEM with automatic smoothness estimation (step-27 pattern)",
                spatial_dims=[2],
                element_types=["FE_Q(1..7)", "hp::FECollection"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="dg_transport",
                description="Discontinuous Galerkin for advection problems (step-12 pattern)",
                spatial_dims=[2],
                element_types=["FE_DGQ(1)"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="hyperelasticity",
                description="Finite-strain hyperelasticity with Neo-Hookean material (step-44 pattern)",
                spatial_dims=[3],
                element_types=["Q1"],
                template_variants=["3d"],
            ),
            PhysicsCapability(
                name="parallel_poisson",
                description="MPI-parallel Poisson solver with p4est (step-40 pattern)",
                spatial_dims=[2],
                element_types=["Q2"],
                template_variants=["2d"],
            ),
            # New physics
            PhysicsCapability("navier_stokes", "Navier-Stokes: stationary + transient (step-57, step-35)", [2, 3],
                              ["Q2-Q1 (Taylor-Hood)"], ["2d"]),
            PhysicsCapability("mixed_laplacian", "Mixed Laplacian with Raviart-Thomas H(div) (step-20)", [2],
                              ["FE_RaviartThomas + FE_DGQ"], ["2d"]),
            PhysicsCapability("compressible_euler", "Compressible Euler with shock capturing (step-33, step-69)", [2, 3],
                              ["FE_DGQ"], ["2d"]),
            PhysicsCapability("time_dependent_heat", "Transient heat with AMR (step-26)", [2],
                              ["Q1"], ["2d"]),
            PhysicsCapability("time_dependent_wave", "Wave equation (step-23, step-48)", [2, 3],
                              ["Q1"], ["2d"]),
            PhysicsCapability("time_dependent_ns", "Transient Boussinesq flow (step-35)", [2],
                              ["Q2-Q1"], ["2d"]),
            PhysicsCapability("matrix_free", "Matrix-free high-performance FEM (step-37, step-59)", [2, 3],
                              ["Q1-Q4 (tensor product)"], ["2d"]),
            PhysicsCapability("multigrid", "Geometric multigrid preconditioner (step-16, step-50)", [2, 3],
                              ["Q1-Q2"], ["2d"]),
            PhysicsCapability("multiphysics_dealii", "Two-phase flow / multi-physics (step-21, step-43)", [2],
                              ["Q1"], ["2d"]),
            PhysicsCapability("obstacle_problem", "Variational inequality / contact (step-41)", [2],
                              ["Q1"], ["2d"]),
            PhysicsCapability("topology_opt_dealii", "SIMP topology optimization (step-79)", [2],
                              ["Q1"], ["2d"]),
            PhysicsCapability("error_estimation", "Dual-weighted residual error estimation (step-14)", [2],
                              ["Q1-Q2"], ["2d"]),
            PhysicsCapability("phase_field", "Phase-field / ADR with SUPG (step-63)", [2],
                              ["Q1"], ["2d"]),
            PhysicsCapability("dg_advection_reaction", "DG advection-reaction (step-12, step-39)", [2],
                              ["FE_DGQ"], ["2d"]),
            PhysicsCapability("cg_dg_coupled", "Mixed CG-DG methods (step-46)", [2],
                              ["FE_Q + FE_DGQ"], ["2d"]),
            PhysicsCapability("optimal_control", "Automatic differentiation / optimal control (step-72)", [2, 3],
                              ["Q1"], ["2d"]),
            # ── 2026-06-01: three _DEALII_KNOWLEDGE keys had
            #    detailed pitfalls but no PhysicsCapability entry,
            #    so users browsing discover never saw them.
            #    Catalog content is distinct from the nearby
            #    similarly-named entries (dg_advection_reaction /
            #    obstacle_problem / hyperelasticity) — keep both
            #    surfaces. Closes task #69.
            PhysicsCapability(
                "advection_dg",
                "Pure DG advection (step-9, step-12). Distinct "
                "from dg_advection_reaction (step-12, step-39) — "
                "advection_dg covers step-9 transport without "
                "reaction term. DoFTools::make_flux_sparsity_"
                "pattern required for face coupling.",
                [2], ["FE_DGQ"], ["2d"]),
            PhysicsCapability(
                "contact",
                "Contact / variational inequalities (step-41, "
                "step-42). Active-set strategy. Related to "
                "obstacle_problem (the dealii backend's primary "
                "name for this class) — distinct deep_knowledge "
                "entry kept for active-set-strategy specifics.",
                [2, 3], ["Q1", "Q2"], ["2d"]),
            PhysicsCapability(
                "nonlinear_elasticity",
                "Nonlinear solid mechanics (step-44). Neo-"
                "Hookean three-field (u, p, J) formulation for "
                "quasi-incompressible materials. Distinct from "
                "hyperelasticity (broader catalog) — this entry "
                "focuses on the step-44 three-field method.",
                [3], ["Q1", "Q2"], ["3d"]),
        ]

    def get_knowledge(self, physics: str) -> dict:
        # Resolution order (2026-06-01 audit closes task #69):
        #
        #   1. data/dealii_knowledge.py:DEALII_KNOWLEDGE — the
        #      course-level catalog (overview/tutorials/etc.).
        #      Usually does NOT hold per-physics keys, but some
        #      entries do live here.
        #   2. generator-embedded KNOWLEDGE — the primary 96-pitfall
        #      source-of-truth that the dealii Tier-2 fixtures
        #      were built against. This is the catalog the
        #      cross-backend signal-verification test scores
        #      against.
        #   3. tools.deep_knowledge._DEALII_KNOWLEDGE — fallback
        #      ONLY for keys NOT in either of the above. This is
        #      where {advection_dg, contact, nonlinear_elasticity}
        #      live; without this fallback they were orphaned.
        try:
            import sys
            data_dir = str(Path(__file__).resolve().parents[3] / "data")
            if data_dir not in sys.path:
                sys.path.insert(0, data_dir)
            from dealii_knowledge import DEALII_KNOWLEDGE as deep
            if physics in deep:
                return deep[physics]
        except ImportError:
            pass
        # Primary fallback: generator-embedded knowledge.
        from backends.dealii.generators import get_knowledge
        gen_k = get_knowledge(physics)
        if isinstance(gen_k, dict) and gen_k.get("pitfalls"):
            return gen_k
        # Last fallback: tools.deep_knowledge per-physics catalog,
        # for entries (advection_dg / contact / nonlinear_
        # elasticity) that ONLY live in _DEALII_KNOWLEDGE.
        try:
            from tools.deep_knowledge import _DEALII_KNOWLEDGE
            if physics in _DEALII_KNOWLEDGE:
                return _DEALII_KNOWLEDGE[physics]
        except ImportError:
            pass
        return gen_k

    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        from backends.dealii.generators import get_template
        key = f"{physics}_{variant}"
        generator = get_template(key)
        return generator(params)

    def validate_input(self, content: str) -> list[str]:
        errors = []
        if "#include" not in content:
            errors.append("C++ source does not contain any #include directives")
        if "deal.II" not in content and "deal_II" not in content:
            errors.append("Source does not include deal.II headers")
        if "int main" not in content:
            errors.append("Source does not contain main()")
        return errors

    async def run(self, input_content: str, work_dir: Path,
                  np: int = 1, timeout=None) -> JobHandle:
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        # Write source and CMakeLists
        src_path = work_dir / "main.cpp"
        src_path.write_text(input_content)

        cmake_content = _generate_cmakelists("fem_solve")
        (work_dir / "CMakeLists.txt").write_text(cmake_content)

        job_id = str(uuid.uuid4())[:8]
        job = JobHandle(job_id=job_id, backend_name="dealii", work_dir=work_dir, status="running")

        start = time.time()

        # Step 1: CMake configure
        try:
            proc = await asyncio.create_subprocess_exec(
                "cmake", ".",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(work_dir),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            if proc.returncode != 0:
                job.status = "failed"
                job.error = f"CMake configure failed:\n{stderr.decode(errors='replace')}"
                job.elapsed = time.time() - start
                return job
        except asyncio.TimeoutError:
            job.status = "failed"
            job.error = "CMake configure timed out"
            job.elapsed = time.time() - start
            return job

        # Step 2: Make
        nproc = os.cpu_count() or 4
        try:
            proc = await asyncio.create_subprocess_exec(
                "make", f"-j{min(nproc, 8)}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(work_dir),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            if proc.returncode != 0:
                job.status = "failed"
                job.error = f"Compilation failed:\n{stderr.decode(errors='replace')[-2000:]}"
                job.elapsed = time.time() - start
                return job
        except asyncio.TimeoutError:
            job.status = "failed"
            job.error = "Compilation timed out"
            job.elapsed = time.time() - start
            return job

        # Step 3: Run
        executable = work_dir / "fem_solve"
        if not executable.is_file():
            job.status = "failed"
            job.error = "Executable not found after compilation"
            job.elapsed = time.time() - start
            return job

        mpirun = shutil.which("mpirun")
        if np > 1 and mpirun:
            cmd = [mpirun, "-np", str(np), str(executable)]
        else:
            cmd = [str(executable)]

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
            job.error = f"Execution timed out after {timeout}s"
        except Exception as e:
            job.status = "failed"
            job.elapsed = time.time() - start
            job.error = str(e)

        return job

    def get_result_files(self, job: JobHandle) -> list[Path]:
        results = []
        for ext in ["*.vtu", "*.pvd", "*.vtk", "*.gnuplot", "*.gpl"]:
            results.extend(job.work_dir.rglob(ext))
        return sorted(results)


def _generate_cmakelists(target_name: str) -> str:
    # If DEALII_ROOT points to source with a build dir, use that build
    dealii_root = os.environ.get("DEALII_ROOT", "")
    extra_hints = ""
    if dealii_root:
        for build_dir in ["build/lib/cmake/deal.II", "build", "build/release",
                          "build/Release", "install/lib/cmake/deal.II"]:
            candidate = Path(dealii_root) / build_dir
            if (candidate / "deal.IIConfig.cmake").exists():
                extra_hints = f" {candidate}"
                break
            elif candidate.is_dir():
                extra_hints = f" {candidate}"
                break
        if not extra_hints and Path(dealii_root).is_dir():
            extra_hints = f" {dealii_root}"

    # Fall back to whatever _find_dealii() returns (conda env,
    # /usr, /opt, etc.). Without this, cmake aborts with
    # "Could not find a package configuration file provided by
    # 'deal.II' (requested version 9.0)" on conda-forge installs
    # where the binary isn't on PATH and DEALII_ROOT isn't set
    # — discover('list') correctly reports deal.II AVAILABLE
    # (the dealii backend's check_availability walks conda envs)
    # but the cmake configure step doesn't see the env's
    # lib/cmake/deal.II dir. Audit 2026-06-01.
    if not extra_hints:
        discovered = _find_dealii()
        if discovered is not None:
            # Prefer the lib/cmake/deal.II sub-path if present —
            # find_package picks up the Config file from there.
            cmake_cfg = discovered / "lib" / "cmake" / "deal.II"
            if (cmake_cfg / "deal.IIConfig.cmake").exists():
                extra_hints = f" {cmake_cfg}"
            else:
                extra_hints = f" {discovered}"

    # Honour CC/CXX from the environment so that conda-forge deal.II
    # packages (whose deal.IIConfig.cmake bakes in a feedstock-only
    # compiler path like
    # /home/conda/feedstock_root/build_artifacts/.../x86_64-conda_cos6-linux-...)
    # do not force the build to use a non-existent toolchain when
    # `deal_ii_initialize_cached_variables()` runs (that macro
    # unconditionally writes the cache without FORCE, so any pre-seeded
    # cache entry wins).
    #
    # Pre-seed without FORCE and only when not already defined so an
    # explicit `-DCMAKE_CXX_COMPILER=…` on the user's cmake command line
    # still wins.  Use CACHE STRING to match deal.II's own macro type so
    # CMake does not emit a type-mismatch warning.  Use plain
    # `if(NOT DEFINED CMAKE_*_COMPILER)` rather than the `CACHE{}`
    # operand form, which only works on CMake >= 3.14 (the file declares
    # a minimum of 3.1).  A `-D` from the command line is visible as a
    # regular variable too, so this still respects user overrides.
    cc = os.environ.get("CC", "")
    cxx = os.environ.get("CXX", "")
    compiler_cache = ""
    if cc:
        compiler_cache += (
            f'if(NOT DEFINED CMAKE_C_COMPILER)\n'
            f'  set(CMAKE_C_COMPILER "{cc}" CACHE STRING "C compiler")\n'
            f'endif()\n'
        )
    if cxx:
        compiler_cache += (
            f'if(NOT DEFINED CMAKE_CXX_COMPILER)\n'
            f'  set(CMAKE_CXX_COMPILER "{cxx}" CACHE STRING "C++ compiler")\n'
            f'endif()\n'
        )

    return f"""\
cmake_minimum_required(VERSION 3.1)
{compiler_cache}find_package(deal.II 9.0 REQUIRED
  HINTS ${{DEAL_II_DIR}} ${{deal.II_DIR}}{extra_hints} /usr /usr/local
)
deal_ii_initialize_cached_variables()
project({target_name})
add_executable({target_name} main.cpp)
deal_ii_setup_target({target_name})
"""


def register():
    register_backend(
        DealiiBackend(),
        aliases=["deal.ii", "deal_ii", "dealii", "deal"],
    )
