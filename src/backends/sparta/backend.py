"""
SPARTA backend — Stochastic PArallel Rarefied-gas Time-accurate Analyzer (Sandia).

SPARTA is a Direct Simulation Monte Carlo (DSMC) particle code for rarefied gas
dynamics — a fundamentally different paradigm from the FEM backends: it solves the
Boltzmann equation stochastically with simulator particles + probabilistic collisions,
which no continuum FEM solver can do. This makes SPARTA the particle half of genuinely
forced multi-paradigm couplings (e.g. DSMC gas <-> FEM solid conjugate heat transfer via
preCICE).

The full command knowledge (121 commands: syntax, examples, descriptions, all categories)
is distilled verbatim from the SPARTA documentation into sparta_knowledge.json and served
through get_knowledge(); the 37 worked example decks are bundled as input templates.
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
    PhysicsCapability, JobHandle,
)
from core.registry import register_backend

logger = logging.getLogger("oasis.sparta")

_KNOWLEDGE_FILE = Path(__file__).parent / "sparta_knowledge.json"
_PRECICE_LIB = "/opt/precice/lib"   # libprecice.so.3 — needed for coupled runs


def _load_knowledge() -> dict:
    try:
        return json.loads(_KNOWLEDGE_FILE.read_text())
    except Exception as e:
        logger.warning(f"SPARTA knowledge load failed: {e}")
        return {"commands": {}, "example_templates": {}}


_KB = _load_knowledge()


def _find_sparta_binary() -> Optional[str]:
    """Locate the SPARTA executable: env override, PATH, then known build dirs."""
    env = os.environ.get("SPARTA_BINARY")
    if env and Path(env).exists():
        return env
    for name in ("spa_serial", "spa_mpi", "sparta"):
        p = shutil.which(name)
        if p:
            return p
    for cand in (
        "/home/alexander/Schreibtisch/sparta/src/spa_serial",
        "/home/alexander/Schreibtisch/sparta/src/spa_mpi",
    ):
        if Path(cand).exists():
            return cand
    return None


# ── physics capability -> {relevant commands, example template dir, pitfalls} ──
# Each maps DSMC physics to the SPARTA commands and a verified worked example deck.
_PHYSICS = {
    "rarefied_flow": dict(
        desc="Rarefied / free-molecular gas flow (high Knudsen) via DSMC particles",
        dims=[2, 3], example="free",
        commands=["global", "species", "mixture", "create_box", "create_grid",
                  "create_particles", "collide", "fix", "run", "stats"],
        pitfalls="Grid cell size must be smaller than the local mean free path; "
                 "timestep must be a fraction of the mean collision time."),
    "collision_relaxation": dict(
        desc="Particle-particle collisions with VSS/VHS model + internal energy relaxation",
        dims=[2, 3], example="collide",
        commands=["collide", "species", "mixture", "collide_modify"],
        pitfalls="VSS parameters come from a species .vss file; without 'collide vss' the "
                 "gas is collisionless (free-molecular only)."),
    "hypersonic_flow": dict(
        desc="Hypersonic rarefied flow over a body (shock, surface heat flux) — DSMC",
        dims=[2, 3], example="adjust_temp",
        commands=["read_surf", "surf_collide", "surf_react", "compute", "fix",
                  "bound_modify", "create_particles", "fix emit/face"],
        pitfalls="Resolve the shock/boundary layer with fine cells near the surface; "
                 "run to statistical steady state before sampling surface heat flux."),
    "surface_interaction": dict(
        desc="Gas-surface interaction: diffuse/specular/CLL collision + surface reactions",
        dims=[2, 3], example="adjust_temp",
        commands=["read_surf", "surf_collide", "surf_react", "surf_modify",
                  "compute surf", "fix surf/temp"],
        pitfalls="surf_collide diffuse needs a wall temperature + accommodation; for "
                 "conjugate heat transfer the wall T is updated each coupling window."),
    "chemistry": dict(
        desc="Gas-phase chemical reactions (TCE / QK) during DSMC collisions",
        dims=[2, 3], example="chem",
        commands=["react", "react_modify", "species", "collide", "mixture"],
        pitfalls="Reaction file format (tce/qk) must match the 'react' style; ensure all "
                 "product species are declared in the species file."),
    "axisymmetric": dict(
        desc="2D axisymmetric DSMC (revolved geometry, radial weighting)",
        dims=[2], example="axi",
        commands=["dimension", "global ... axisymmetric", "fix", "global weight"],
        pitfalls="Use 'global ... axisymmetric yes' and radial particle weighting; the "
                 "y=0 axis needs the correct boundary condition."),
    "particle_emission": dict(
        desc="Particle injection / emission from faces or surfaces (inflow boundary)",
        dims=[2, 3], example="emit",
        commands=["fix emit/face", "fix emit/surf", "mixture", "create_particles"],
        pitfalls="Emission rate is set by the mixture number density + face area; mismatch "
                 "with the freestream causes a non-physical inflow."),
    "adaptive_grid": dict(
        desc="Static/dynamic grid adaptation to resolve gradients (refine near shocks)",
        dims=[2, 3], example="adapt",
        commands=["adapt_grid", "fix adapt", "balance_grid", "compute"],
        pitfalls="Over-refinement explodes particle count; cap refinement levels and "
                 "rebalance the grid across MPI ranks after adaptation."),
    "ambipolar_plasma": dict(
        desc="Weakly-ionized (ambipolar) flow: electrons follow ions (DSMC plasma)",
        dims=[2, 3], example="ambi",
        commands=["fix ambipolar", "species", "collide", "react"],
        pitfalls="Ambipolar electrons are attached to ions; the species file must define "
                 "the electron and ion species consistently."),
    "conjugate_heat_transfer": dict(
        desc="DSMC gas <-> FEM solid conjugate heat transfer (the forced two-code coupling; "
             "SPARTA writes surface heat flux, reads back wall temperature via preCICE)",
        dims=[2, 3], example="adjust_temp",
        commands=["surf_collide diffuse", "compute surf ... etot", "fix surf/temp",
                  "fix field/surf", "read_surf"],
        pitfalls="The wall temperature is a coupling unknown updated each preCICE window; "
                 "the DSMC heat flux is statistically noisy — average over the (long) solid "
                 "thermal timescale. Explicit serial coupling is stable because solid "
                 "thermal inertia damps DSMC fluctuations."),
}


class SpartaBackend(SolverBackend):

    def name(self) -> str:
        return "sparta"

    def display_name(self) -> str:
        return "SPARTA (DSMC)"

    def check_availability(self) -> tuple[BackendStatus, str]:
        binpath = _find_sparta_binary()
        if not binpath:
            return (BackendStatus.NOT_INSTALLED,
                    "SPARTA binary not found (set SPARTA_BINARY or build spa_serial)")
        import subprocess
        try:
            subprocess.run([binpath, "-h"], capture_output=True, text=True, timeout=10)
            # SPARTA prints usage/version; rc may be nonzero for -h, that's fine
            tag = "with knowledge" if _KB.get("commands") else "no knowledge file"
            return BackendStatus.AVAILABLE, f"SPARTA at {binpath} ({tag}, {_KB.get('n_commands',0)} commands)"
        except Exception as e:
            return BackendStatus.MISCONFIGURED, f"SPARTA found but check failed: {e}"

    def input_format(self) -> InputFormat:
        return InputFormat.SPARTA

    def supported_physics(self) -> list[PhysicsCapability]:
        out = []
        for name, info in _PHYSICS.items():
            out.append(PhysicsCapability(
                name=name,
                description=info["desc"],
                spatial_dims=info["dims"],
                element_types=["DSMC-particles", "cartesian-grid"],
                template_variants=[info["example"]],
            ))
        return out

    def get_knowledge(self, physics: str) -> dict:
        info = _PHYSICS.get(physics)
        if not info:
            # unknown physics: return the raw command index so the model can still look up
            return {"error": f"unknown physics '{physics}'",
                    "available_physics": sorted(_PHYSICS.keys()),
                    "all_commands": sorted(_KB.get("commands", {}).keys())}
        cmds = _KB.get("commands", {})
        # resolve the relevant command docs (verbatim syntax+examples+description)
        relevant = {}
        for c in info["commands"]:
            base = c.split()[0].replace("/", "_")
            for key in (c, base, c.replace(" ", "_")):
                if key in cmds:
                    relevant[key] = cmds[key]
                    break
        tmpl = _KB.get("example_templates", {}).get(info["example"], {})
        return {
            "description": info["desc"],
            "spatial_dims": info["dims"],
            "pitfalls": info["pitfalls"],
            "relevant_commands": relevant,
            "worked_example": {"dir": info["example"], "decks": tmpl},
            "solver": "SPARTA DSMC; run: spa_serial -in <script>",
            "unit_systems": "SI (global ... gridcut ... ; fnum sets real-particles-per-simulator)",
        }

    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        info = _PHYSICS.get(physics)
        if not info:
            raise ValueError(f"Unknown physics '{physics}'. "
                             f"Available: {', '.join(sorted(_PHYSICS))}")
        decks = _KB.get("example_templates", {}).get(variant or info["example"], {})
        if not decks:
            raise ValueError(f"No example template for '{variant or info['example']}'")
        # pick the primary input deck (in.<name>), apply simple param substitution
        primary = sorted(decks, key=lambda k: (("in." not in k), len(k)))[0]
        deck = decks[primary]
        for k, v in (params or {}).items():
            deck = deck.replace(f"${{{k}}}", str(v))
        return deck

    def validate_input(self, content: str) -> list[str]:
        errors = []
        cmds = _KB.get("commands", {})
        nonblank = [l for l in content.splitlines()
                    if l.strip() and not l.strip().startswith("#")]
        if not nonblank:
            errors.append("Empty SPARTA input script")
            return errors
        # a valid DSMC deck needs a run/grid; check first tokens are known commands
        known = set(cmds.keys()) | {c.split("_")[0] for c in cmds}
        first_tokens = {l.split()[0] for l in nonblank}
        unknown = [t for t in first_tokens if t not in known and t not in
                   {"variable", "label", "next", "jump", "if", "echo", "log", "shell",
                    "print", "include", "clear", "partition", "uncompute", "unfix",
                    "undump", "boundary", "global", "seed", "units", "package"}]
        if unknown:
            errors.append(f"Unrecognized SPARTA command(s): {', '.join(sorted(unknown)[:6])}")
        if "run" not in first_tokens and "run_file" not in first_tokens:
            errors.append("Script has no 'run' command (DSMC will not advance)")
        return errors

    async def run(self, input_content: str, work_dir: Path,
                  np: int = 1, timeout=None) -> JobHandle:
        binary = _find_sparta_binary()
        job_id = str(uuid.uuid4())[:8]
        if not binary:
            return JobHandle(job_id=job_id, backend_name="sparta", work_dir=work_dir,
                             status="failed", error="SPARTA binary not found")
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        script_path = work_dir / "in.sparta"
        script_path.write_text(input_content)

        job = JobHandle(job_id=job_id, backend_name="sparta",
                        work_dir=work_dir, status="running")

        env = os.environ.copy()
        # make libprecice visible for coupled runs (harmless otherwise)
        env["LD_LIBRARY_PATH"] = _PRECICE_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
        if "mpi" in Path(binary).name and np > 1:
            cmd = ["mpirun", "-np", str(np), binary, "-in", str(script_path)]
        else:
            cmd = [binary, "-in", str(script_path)]

        start = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=str(work_dir), env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            job.elapsed = time.time() - start
            job.return_code = proc.returncode
            job.pid = proc.pid
            out = stdout.decode(errors="replace")
            err = stderr.decode(errors="replace")
            (work_dir / "log.sparta").write_text(out)
            (work_dir / "stderr.log").write_text(err)
            # SPARTA prints "ERROR:" to screen even with rc 0 sometimes
            if proc.returncode == 0 and "ERROR" not in out:
                job.status = "completed"
            else:
                job.status = "failed"
                job.error = (err or out)[-2000:]
        except asyncio.TimeoutError:
            job.status = "failed"; job.elapsed = timeout
            job.error = f"Timed out after {timeout}s"
        except Exception as e:
            job.status = "failed"; job.elapsed = time.time() - start
            job.error = str(e)
        return job

    def get_result_files(self, job: JobHandle) -> list[Path]:
        results = []
        # SPARTA dumps: dump.*, *.surf, surf temp/flux files, plus log
        for pat in ["dump.*", "*.dump", "*.surf", "tmp.*", "*.vtk", "log.sparta"]:
            results.extend(job.work_dir.rglob(pat))
        return sorted_by_step(results)

    def get_version(self) -> Optional[str]:
        return f"SPARTA ({_KB.get('n_commands', 0)} commands distilled)"

    def precice_participant(self) -> dict:
        """SPARTA as a preCICE participant — drive it via its Python library, exchange a
        surface quantity (e.g. heat flux out, wall temperature in). Verified pattern."""
        return {
            "description": "SPARTA DSMC preCICE participant (driven via libsparta Python library)",
            "exchange_loop": (
                "import precice, numpy as np\n"
                "from sparta import sparta                 # PYTHONPATH=<sparta>/python\n"
                "spa = sparta(name='serial')               # loads libsparta_serial.so\n"
                "for line in setup_deck.splitlines(): spa.command(line)\n"
                "# setup must include: compute <id> surf all all etot ; fix ave/surf ;\n"
                "#   variable twall equal <T0> ; surf_collide <sc> diffuse v_twall 1.0 ;\n"
                "#   compute totflux reduce sum f_<avesurf>\n"
                "p = precice.Participant('Gas','precice-config.xml',0,1)\n"
                "vid = p.set_mesh_vertices('Gas-Mesh', np.array([[0.0,0.0]]))\n"
                "p.initialize()\n"
                "while p.is_coupling_ongoing():\n"
                "    dt = p.get_max_time_step_size()\n"
                "    T = p.read_data('Gas-Mesh','Wall-Temperature',vid,dt)\n"
                "    spa.command('variable twall delete'); spa.command(f'variable twall equal {float(T[0])}')\n"
                "    spa.command('run 500')                 # advance DSMC, re-average flux\n"
                "    q = spa.extract_compute('totflux',0,0) / WALL_AREA\n"
                "    p.write_data('Gas-Mesh','Heat-Flux',vid,np.array([q])); p.advance(dt)\n"
                "p.finalize()"
            ),
            "notes": ("Build libsparta_serial.so via 'make mode=shlib serial'. The Python library "
                      "exposes command/extract_global/extract_compute/extract_variable only (no surf "
                      "scatter) -> couple a SCALAR (total flux <-> uniform wall temp) by re-issuing a "
                      "SPARTA equal-variable for the wall temperature each window. Set LD_LIBRARY_PATH "
                      "to <sparta>/src and /opt/precice/lib."),
        }


# ─── Registration ────────────────────────────────────────────────────────

def register():
    backend = SpartaBackend()
    register_backend(backend, aliases=["sparta", "dsmc"])
    logger.info("SPARTA (DSMC) backend registered")
