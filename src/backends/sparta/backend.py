"""
SPARTA backend — Stochastic PArallel Rarefied-gas Time-accurate Analyzer (Sandia).

SPARTA is a Direct Simulation Monte Carlo (DSMC) particle code for rarefied gas
dynamics — a fundamentally different paradigm from the FEM backends: it solves the
Boltzmann equation stochastically with simulator particles + probabilistic collisions,
which no continuum FEM solver can do. This makes SPARTA the particle half of genuinely
forced multi-paradigm couplings (e.g. DSMC gas <-> FEM solid conjugate heat transfer via
preCICE).

Knowledge layout (restructured): the agent-facing knowledge lives in
``generators/``, one module per physics, each exposing a KNOWLEDGE dict whose
pitfalls carry an observable ``Signal:`` clause plus parameterised deck
generators — the same shape the FEM backends use. ``sparta_knowledge.json``
remains the verbatim SPARTA documentation index (121 doc pages) and the bundled
upstream example decks; it is a LOOKUP table for validate_input and for
on-demand syntax queries, not something pushed at the agent on every call.
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

from .generators import (  # noqa: F401
    BUILD_FACTS, GENERATORS, HARD_ORDERING_ERRORS, KNOWLEDGE,
    READING_OUTPUT, SILENTLY_ACCEPTED,
)

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
        str(Path.home() / "sparta" / "src" / "spa_serial"),
        str(Path.home() / "sparta" / "src" / "spa_mpi"),
        "/home/alexander/Schreibtisch/sparta/src/spa_serial",
        "/home/alexander/Schreibtisch/sparta/src/spa_mpi",
    ):
        if Path(cand).exists():
            return cand
    return None


def _sparta_data_dirs(binary: Optional[str] = None,
                      extra_dirs: tuple | list = ()) -> list[Path]:
    """Candidate dirs holding SPARTA data files (*.species, *.vss, data.*).

    The bundled example decks reference data files (`species ar.species Ar`,
    `read_surf data.circle`, `collide vss air air.vss`) that live in the
    SPARTA distribution's data/ and examples/ trees, NOT in the knowledge
    JSON. Without staging them the deck dies with e.g.
    'Cannot open species file ar.species' (verified on a macOS build).
    We locate the distribution relative to the binary
    (src/spa_serial -> repo root) plus SPARTA_ROOT.

    Search ORDER matters: explicit per-call ``extra_dirs`` (a task's own
    data dir) come first, then SPARTA_DATA_DIR (colon-separated env, same
    precedence idea), then the distribution. A task-specific circle.surf
    must win over the distribution's example circle.surf of the same name
    — the two are different geometries, and losing that race is silent:
    the deck reads the example surface and runs to completion on the
    wrong body (observed 2026-07)."""
    dirs: list[Path] = []
    for d in extra_dirs:
        p = Path(d).expanduser()
        if p.is_dir() and p not in dirs:
            dirs.append(p)
    data_env = os.environ.get("SPARTA_DATA_DIR", "")
    for d in data_env.split(os.pathsep):
        if d and Path(d).is_dir() and Path(d) not in dirs:
            dirs.append(Path(d))
    root_env = os.environ.get("SPARTA_ROOT", "")
    if root_env and Path(root_env).is_dir():
        dirs.append(Path(root_env))
    if binary:
        try:
            # <repo>/src/spa_serial -> <repo>
            repo = Path(binary).resolve().parent.parent
            if ((repo / "data").is_dir() or (repo / "examples").is_dir()) \
                    and repo not in dirs:
                dirs.append(repo)
        except OSError:
            pass
    return dirs


def _deck_data_refs(deck: str) -> set[str]:
    """File references in a SPARTA deck that must exist in the run dir.

    Handles the reference styles used across the bundled decks:
    `species <file> ...`, `collide vss <mix> <file>`, `read_surf <file>`,
    `read_grid <file>`, `read_isurf <file>`, `react <style> <file>`."""
    wanted: set[str] = set()
    for line in deck.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        if toks[0] == "species" and len(toks) >= 2:
            wanted.add(toks[1])
        elif toks[0] == "collide" and len(toks) >= 4 and toks[1].startswith("vss"):
            wanted.add(toks[3])
        elif toks[0] in ("read_surf", "read_grid", "read_isurf") and len(toks) >= 2:
            wanted.add(toks[1])
        elif toks[0] == "react" and len(toks) >= 3:
            wanted.add(toks[2])
    return wanted


def stage_deck_data_files(deck: str, work_dir: Path,
                          binary: Optional[str] = None,
                          extra_dirs: tuple | list = ()) -> dict:
    """Copy every data file the deck references into ``work_dir``.

    Public staging entry point — used by the single-run path
    (SpartaBackend.run) AND by the coupled path (the couple() tool),
    which previously did NO staging at all: a SPARTA participant deck
    run by the coupling driver in a fresh work_dir died with
    'Cannot open species file ar.species' (../particle.cpp:711).

    Resolution per reference:
      1. already present in work_dir -> keep (never overwrite),
      2. the reference itself is an existing path (absolute or relative
         to a search dir) -> copy it,
      3. basename lookup through ``extra_dirs`` (task data dir),
         SPARTA_DATA_DIR, SPARTA_ROOT, then the distribution's data/ and
         examples/ trees.

    Returns {"staged": {name: source_path}, "missing": [refs]}.
    """
    if binary is None:
        binary = _find_sparta_binary()
    staged: dict[str, str] = {}
    missing: list[str] = []
    wanted = _deck_data_refs(deck)
    if not wanted:
        return {"staged": staged, "missing": missing}
    search_dirs = _sparta_data_dirs(binary, extra_dirs=extra_dirs)
    for ref in wanted:
        name = Path(ref).name
        dest = work_dir / name
        if dest.exists():
            continue
        found = None
        # the deck may reference an explicit path (../data/ar.species,
        # /abs/path/circle.surf) — honor it before basename search
        refp = Path(ref).expanduser()
        if refp.is_absolute() and refp.is_file():
            found = refp
        else:
            for base in search_dirs:
                cand = (base / ref)
                if cand.is_file():
                    found = cand
                    break
        if not found:
            for base in search_dirs:
                for cand_dir in (base, base / "data", base / "examples"):
                    cand = cand_dir / name
                    if cand.is_file():
                        found = cand
                        break
                if not found:
                    # examples/* subdirs (data.circle lives in examples/circle)
                    hits = sorted((base / "examples").glob(f"*/{name}")) \
                        if (base / "examples").is_dir() else []
                    if hits:
                        found = hits[0]
                if found:
                    break
        if found:
            shutil.copy(found, dest)
            staged[name] = str(found)
            logger.info(f"Staged SPARTA data file {name} from {found}")
        else:
            missing.append(ref)
            logger.warning(f"SPARTA data file referenced by deck not found: {ref}")
    return {"staged": staged, "missing": missing}


def _stage_sparta_data_files(deck: str, work_dir: Path, binary: str):
    """Backward-compatible wrapper kept for the single-run path."""
    stage_deck_data_files(deck, work_dir, binary=binary)


# ── physics capability -> deck family, worked example and default variant ──
# The agent-facing knowledge (descriptions, key commands, Signal-carrying
# pitfalls) lives in generators/<physics>.py. This table only records what the
# backend itself needs: the spatial dims, the upstream example directory that
# demonstrates the physics, and the generator variant to offer by default.
_PHYSICS = {
    "rarefied_flow": dict(dims=[2, 3], example="free",
                          variants=["box_2d", "channel_2d"]),
    "collision_relaxation": dict(dims=[2, 3], example="collide",
                                 variants=["box_2d", "internal_energy_2d"]),
    "hypersonic_flow": dict(dims=[2, 3], example="adjust_temp",
                            variants=["circle_2d"]),
    "surface_interaction": dict(dims=[2, 3], example="circle",
                                variants=["circle_2d"]),
    "chemistry": dict(dims=[2, 3], example="chem", variants=["box_3d"]),
    "axisymmetric": dict(dims=[2], example="axi", variants=["body_2d"]),
    "particle_emission": dict(dims=[2, 3], example="emit",
                              variants=["channel_2d"]),
    "adaptive_grid": dict(dims=[2, 3], example="adapt", variants=["circle_2d"]),
    "ambipolar_plasma": dict(dims=[2, 3], example="ambi",
                             variants=["circle_2d"]),
    "conjugate_heat_transfer": dict(dims=[2, 3], example="adjust_temp",
                                    variants=["circle_2d"]),
}

# Cross-cutting reference blocks, trimmed to what an agent needs while writing
# a deck. Deliberately EXCLUDES anything host-specific: no binary paths, no
# distribution paths, no machine-local data directories. Those belong to
# _find_sparta_binary()/_sparta_data_dirs(), not to knowledge served to a model.
_CROSS_CUTTING = {
    "hard_ordering_errors": HARD_ORDERING_ERRORS,
    "more": "build facts (compiled styles, accelerator status), the full "
            "output-reading reference and the list of things SPARTA accepts "
            "silently are in knowledge(topic='overview', solver='sparta')",
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
        # This ran `-h` and then THREW THE RESULT AWAY, returning available
        # unless the call itself raised — so `SPARTA_BINARY=/bin/true` reported
        # "available ... with knowledge, 121 commands". The code read as though
        # it validated, which is worse than an obviously absent check.
        #
        # `-h` prints `SPARTA (<date>)` as its first line, measured on this
        # build. stdin is closed: under an MCP stdio server an inherited stdin is
        # the JSON-RPC stream, and a probed program that reads it consumes the
        # protocol.
        try:
            r = subprocess.run([binpath, "-h"], capture_output=True, timeout=15,
                               stdin=subprocess.DEVNULL)
            blob = (r.stdout + r.stderr).decode("utf-8", errors="replace")
        except (subprocess.TimeoutExpired, OSError) as e:
            # Could not look — do not condemn a possibly-working install.
            blob = None
            _unchecked = f"identity not checked ({type(e).__name__}: {e})"
        if blob is not None and "SPARTA (" not in blob:
            return BackendStatus.MISCONFIGURED, (
                f"the binary at {binpath} does not identify itself as SPARTA "
                f"(its -h output does not begin `SPARTA (`): "
                + " ".join(blob.split())[:120])
        tag = "with knowledge" if _KB.get("commands") else "no knowledge file"
        return BackendStatus.AVAILABLE, f"SPARTA at {binpath} ({tag}, {_KB.get('n_commands',0)} commands)"

    def input_format(self) -> InputFormat:
        return InputFormat.SPARTA

    def supported_physics(self) -> list[PhysicsCapability]:
        out = []
        for name, info in _PHYSICS.items():
            kn = KNOWLEDGE.get(name, {})
            out.append(PhysicsCapability(
                name=name,
                description=kn.get("description", name),
                spatial_dims=info["dims"],
                element_types=["DSMC-particles", "cartesian-grid"],
                template_variants=list(info["variants"]),
            ))
        return out

    def get_knowledge(self, physics: str) -> dict:
        """Structured, physics-scoped knowledge.

        Shape and size deliberately match the FEM backends: a short
        description, the commands THIS physics needs (one line each, not the
        verbatim manual page), the ordered deck skeleton, and the pitfalls —
        each with an observable ``Signal:``. The 121-page command index and the
        37 bundled example decks stay in sparta_knowledge.json and are reached
        on demand (``examples``/``get_command_reference``), because pushing
        them here cost tens of kilobytes per call and pushed the pitfalls past
        the client-side truncation point.
        """
        if physics == "_general":
            return KNOWLEDGE["_general"]
        kn = KNOWLEDGE.get(physics)
        if not kn:
            return {"error": f"unknown physics '{physics}'",
                    "available_physics": sorted(_PHYSICS.keys()),
                    "all_commands": sorted(
                        _KB.get("command_surface", {}).get("true_commands")
                        or _KB.get("commands", {}).keys())}
        info = _PHYSICS[physics]
        out = dict(kn)
        out["variants"] = list(info["variants"])
        out["worked_example"] = (
            f"upstream SPARTA example directory '{info['example']}' — read the "
            f"decks from the installed distribution at examples/"
            f"{info['example']}/in.*. NOTE: the examples() MCP tool does not "
            f"index SPARTA (every keyword returns 'No examples found for ... "
            f"in sparta', and action='template' returns 'Unknown solver: "
            f"sparta'), so do not route this through it. The 37 bundled decks "
            f"are in sparta_knowledge.json['example_templates'] but no MCP "
            f"tool currently reads them.")
        out["command_reference"] = (
            "one-line syntax for the commands this physics needs is in "
            "'key_commands' above. The full SPARTA doc page for any command is "
            "in sparta_knowledge.json['commands'] and is returned by the "
            "python method SpartaBackend.get_command_reference('<command>') — "
            "which is NOT exposed as an MCP tool on this build, so an MCP "
            "client cannot call it. Until it is wired up, treat 'key_commands' "
            "plus knowledge(topic='input_guide', solver='sparta') as the whole "
            "of the syntax you have.")
        out.update(_CROSS_CUTTING)
        return out

    def get_command_reference(self, command: str) -> dict:
        """Verbatim SPARTA doc entry for one command, fetched on demand.

        This is the escape hatch that lets get_knowledge() stay small: the
        121-entry documentation index is still shipped in
        sparta_knowledge.json, it is just no longer pushed at the agent
        wholesale on every knowledge() call.
        """
        cmds = _KB.get("commands", {})
        key = command.strip()
        # ' '->'_' and '/'->'_' must also be tried TOGETHER. Applying them only
        # separately made every real slash form fall through to the bare
        # key.split()[0] page: 'fix ave/surf', 'fix emit/face', 'fix surf/temp'
        # and 'fix ave/time' all returned the generic 'fix' page and
        # 'compute thermal/grid' the generic 'compute' page, silently, even
        # though fix_ave_surf / fix_emit_face / fix_surf_temp /
        # compute_thermal_grid all exist as keys. Slashes are the real deck
        # syntax (upstream decks use 'fix emit/face' 25 times).
        for cand in (key, key.replace(" ", "_"), key.replace("/", "_"),
                     key.replace(" ", "_").replace("/", "_"),
                     key.replace("_", " "),
                     key.split()[0] if key.split() else key):
            if cand in cmds:
                # the entry carries its own "command" field (the deck form,
                # e.g. "compute grid"); "doc_page" is the index key we matched.
                return {"doc_page": cand, **cmds[cand]}
        return {"error": f"no documentation entry for '{command}'",
                "note": "compute/fix/dump STYLES are documented under their "
                        "page name, e.g. 'compute_grid' for 'compute <ID> "
                        "grid ...'",
                "available": sorted(cmds)[:40]}

    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        info = _PHYSICS.get(physics)
        if not info:
            raise ValueError(f"Unknown physics '{physics}'. "
                             f"Available: {', '.join(sorted(_PHYSICS))}")
        variant = variant or info["variants"][0]
        # Resolve the EXACT variant only. A second, unconditional lookup of
        # f"{physics}_{info['variants'][0]}" used to sit here, which meant any
        # unrecognised variant silently returned the default deck for that
        # physics — generate_input('chemistry', 'box_2d') handed back the 3d
        # chemistry deck and generate_input('hypersonic_flow', 'box_2d') handed
        # back the circle deck, with no error and nothing in the deck saying so.
        # It also made the ValueError at the end of this method unreachable.
        gen = GENERATORS.get(f"{physics}_{variant}")
        if gen:
            return gen(params or {})
        # fall back to the bundled upstream example deck for this variant
        decks = _KB.get("example_templates", {}).get(variant, {})
        if decks:
            primary = sorted(decks, key=lambda k: (("in." not in k), len(k)))[0]
            deck = decks[primary]
            for k, v in (params or {}).items():
                deck = deck.replace(f"${{{k}}}", str(v))
            return deck
        available = ", ".join(sorted(
            k for k in GENERATORS if k.startswith(physics + "_")))
        raise ValueError(f"Unknown variant '{variant}' for physics "
                         f"'{physics}'. Available: {available}")

    def validate_input(self, content: str) -> list[str]:
        errors = []
        # SPARTA continues a command onto the next line when the line ends in
        # '&' (src/input.cpp). Joining those first is not cosmetic: without it
        # the continuation line's first word ('combine', 'maxlevel', ...) is
        # read as a command name and every multi-line fix adapt / adapt_grid
        # deck is falsely rejected.
        joined, buf = [], ""
        for raw in content.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith("&"):
                buf += line[:-1].rstrip() + " "
                continue
            joined.append(buf + line)
            buf = ""
        if buf:
            # A '&' with no line after it leaves the last command incomplete.
            # SPARTA does NOT wave this through: 'run 10 &' as the final line
            # of an otherwise valid deck aborts with
            # 'ERROR: Illegal run command (../run.cpp:103)'. Joining without
            # this check turned that abort into a silent validate_input pass.
            errors.append(
                "Deck ends with a dangling '&' line continuation — the last "
                "command is never completed. SPARTA aborts on this (e.g. "
                "'run 10 &' as the final line gives 'ERROR: Illegal run "
                "command (../run.cpp:103)').")
            joined.append(buf.rstrip())
        nonblank = joined
        if not nonblank:
            errors.append("Empty SPARTA input script")
            return errors
        # The parser surface is the 66 commands the BUILD accepts, NOT the 121
        # documentation-page names in _KB['commands'] — 55 of those (compute_grid,
        # fix_ave_surf, dump_image, surf_react_adsorb, suffix, ...) are doc filenames
        # that the binary rejects with "ERROR: Unknown command: ... (../input.cpp:244)".
        # Validating against the doc index waved those through. (Fixed 2026-08-03 after
        # feeding each form to spa_serial.)
        surface = _KB.get("command_surface", {})
        known = set(surface.get("true_commands") or [])
        if not known:  # knowledge file predates the command_surface block
            cmds = _KB.get("commands", {})
            known = set(cmds) | {c.split("_")[0] for c in cmds}
        first_tokens = {l.split()[0] for l in nonblank}
        unknown = sorted(t for t in first_tokens if t not in known)
        if unknown:
            errors.append(
                f"Unrecognized SPARTA command(s): {', '.join(unknown[:6])} — "
                f"note that compute/fix/dump/surf_react STYLES are written "
                f"'compute <ID> <style> ...', not 'compute_<style> ...'")
        if "run" not in first_tokens:
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
        try:
            _stage_sparta_data_files(input_content, work_dir, binary)
        except Exception as e:
            logger.warning(f"SPARTA data-file staging failed (continuing): {e}")

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
        """SPARTA as a preCICE participant, in the load order that actually runs.

        THIS USED TO SERVE THE OPPOSITE OF THE TRUTH. The snippet here began
        `import precice` and then `from sparta import sparta`, under the words
        "Verified pattern", while knowledge(topic='precice', solver='sparta')
        said that exact ordering segfaults. Both went to agents. Running all
        four orderings settles it in favour of the knowledge module: with
        preCICE imported first, `sparta(name='serial')` dies inside
        `PMPI_Type_size`, because libsparta defines its own MPI_* stubs, links
        no real MPI, and a real libmpi in the global symbol namespace
        interposes them. The fixture that runs them is
        scripts/tier2_fixtures/coupling/sparta_precice_load_order_and_coupled_run.
        """
        return {
            "description": ("SPARTA DSMC preCICE participant, in-process via libsparta — "
                            "libsparta MUST be loaded first, with RTLD_DEEPBIND|RTLD_LOCAL"),
            "exchange_loop": (
                "# LOAD ORDER IS NOT OPTIONAL. `import precice` first segfaults; loading\n"
                "# SPARTA first through the stock sparta.py wrapper (RTLD_GLOBAL) makes\n"
                "# `import precice` fail with 'libmpi.so.12: cannot open shared object\n"
                "# file'; mpi4py first does not help. Deep binding is the one that works.\n"
                "import ctypes, os\n"
                "mode = os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_DEEPBIND\n"
                "lib = ctypes.CDLL('<sparta>/src/libsparta_serial.so', mode=mode)\n"
                "import precice, numpy as np              # only now\n"
                "spa = ctypes.c_void_p()\n"
                "lib.sparta_open_no_mpi(0, None, ctypes.byref(spa))\n"
                "cmd = lambda s: lib.sparta_command(spa, s.encode())\n"
                "for line in setup_deck.splitlines(): cmd(line)\n"
                "# setup must include: compute <id> surf all all etot ; fix ave/surf ;\n"
                "#   variable twall equal <T0> ; surf_collide <sc> diffuse v_twall 1.0 ;\n"
                "#   compute totflux reduce sum f_<avesurf>\n"
                "p = precice.Participant('Gas','precice-config.xml',0,1)\n"
                "vid = p.set_mesh_vertices('Gas-Mesh', np.array([[0.0,0.0]]))\n"
                "p.initialize()\n"
                "while p.is_coupling_ongoing():\n"
                "    dt = p.get_max_time_step_size()\n"
                "    T = p.read_data('Gas-Mesh','Wall-Temperature',vid,dt)\n"
                "    cmd('variable twall delete'); cmd(f'variable twall equal {float(T[0])}')\n"
                "    cmd('run 500')                       # advance DSMC, re-average flux\n"
                "    q = <read the averaged flux back> / WALL_AREA\n"
                "    p.write_data('Gas-Mesh','Heat-Flux',vid,np.array([q])); p.advance(dt)\n"
                "p.finalize()"
            ),
            "notes": ("Build libsparta_serial.so via 'make mode=shlib serial'. Set "
                      "LD_LIBRARY_PATH to <sparta>/src and /opt/precice/lib.\n"
                      "PREFER THE SUBPROCESS ROUTE. The in-process library exposes "
                      "command/extract_global/extract_compute/extract_variable only (no surf "
                      "scatter), so it can exchange a SCALAR and nothing more — total flux "
                      "against a uniform wall temperature, re-issued as a SPARTA "
                      "equal-variable each window. Running `spa_serial -in <deck>` once per "
                      "time window instead carries a PER-ELEMENT wall temperature through "
                      "`custom surf ... file` and reads a per-element flux out of a surf "
                      "dump, and gives SPARTA its own address space so the MPI symbol "
                      "clash cannot arise at all. That is what the shipped participant "
                      "(data/coupling_participants/participant_sparta.py) does and what "
                      "the coupled run behind SPARTA's preCICE verdict used."),
        }


# ─── Registration ────────────────────────────────────────────────────────

def register():
    backend = SpartaBackend()
    register_backend(backend, aliases=["sparta", "dsmc"])
    logger.info("SPARTA (DSMC) backend registered")
