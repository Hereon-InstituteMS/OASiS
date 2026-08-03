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
    'Cannot open species file ar.species' (verified on macOS build
    2026-07-14). We locate the distribution relative to the binary
    (src/spa_serial -> repo root) plus SPARTA_ROOT.

    Search ORDER matters: explicit per-call ``extra_dirs`` (a task's own
    data dir) come first, then SPARTA_DATA_DIR (colon-separated env, same
    precedence idea), then the distribution. A task-specific circle.surf
    must win over the distribution's example circle.surf of the same name
    — the two are different geometries (T15 campaign, 2026-07)."""
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
    'Cannot open species file ar.species' (../particle.cpp:711)
    (T15 campaign 2026-07, reproduced 2026-08-01).

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


# ── Execution-verified DSMC pitfalls (campaign 2026-08-03, SPARTA 24 Sep 2025) ──
# Every entry below was produced by running BOTH a wrong variant and a right
# variant of a small case with spa_serial and reading the numbers out of
# log.sparta. These are the failures that leave rc=0 and a plausible-looking
# output while the physics is wrong — the ones an AI driving SPARTA cannot
# catch by checking the exit code.
#
# Verification case for the transport numbers: 2d argon Fourier channel,
# boundary p ss p, box 2e-5 x 1e-4 m, 4x60 grid, nrho 7.07043e23, fnum 1e11,
# ylo wall diffuse 300 K, yhi wall diffuse 1000 K, collide vss, dt 1e-9 s,
# 3000 steps; wall energy flux read via compute boundary + fix ave/time.
_VERIFIED_DSMC_PITFALLS = [
    "[Physics] Omitting the collide command is NEVER an error — the gas is simply "
    "collisionless (free-molecular), and in a wall-bounded conduction problem the wall "
    "heat flux comes out 8.4x too large. Measured: with 'collide vss' cold-wall flux "
    "2.092e5 W/m2, hot-wall -2.037e5 W/m2, mid-gas T = 660.3 K; with the collide line "
    "deleted, 1.7615e6 and -1.7495e6 W/m2 and T = 538.5 K. Both rc=0. "
    "Signal: Ncoll is identically 0 in every stats line. Put ncoll in stats_style and "
    "check it is nonzero before believing any transport number. "
    "(Verified by execution 2026-08-03.)",

    "[Physics] Without a collide command, create_particles also gives every particle "
    "ZERO rotational and vibrational energy even when the mixture asks for trot/tvib, "
    "so internal-energy diagnostics silently read 0. Measured with 'mixture ... trot "
    "100': with collide, step-0 Trot = 99.02 K; without collide, Trot = 0.000 at step 0 "
    "and at every later step. "
    "Signal: a Trot/Tvib column that is exactly 0.000 rather than noisy. "
    "(Verified by execution 2026-08-03.)",

    "[Numerical] 'create_particles <mix> n <N>' with a NONZERO N creates exactly N "
    "simulation particles and SILENTLY overrides 'global nrho' — the realised density "
    "becomes N*fnum/V, not the nrho you set. Only 'n 0' honours nrho and fnum. Measured "
    "against a requested nrho of 7.07043e23: n 0 -> realised 7.0704e23 (exact); "
    "n 50000 -> 5.0e23 (1.41x low); n 5000 -> 5.0e22, i.e. 14x TOO LOW. All rc=0, no "
    "warning. Most upstream example decks use an explicit n, so copying one silently "
    "changes your density. "
    "Signal: compute the realised nrho as Np*fnum/V and compare it with the nrho you "
    "asked for. (Verified by execution 2026-08-03.)",

    "[Numerical] In a 2d run the grid-cell volume is dx*dy PER 1 METRE OF DEPTH and the "
    "z extent given to create_box is IGNORED (src/grid.cpp:307). Sizing fnum for a thin "
    "slab therefore overshoots the particle count by 1/(zhi-zlo). Measured: identical "
    "decks with z = +/-0.5 and z = +/-0.5e-4 both report 'Created 70704 particles'. A "
    "deck sized for a 1e-4-thick slab asked for 1e8 particles and never finished, "
    "leaving an EMPTY log.sparta. "
    "Signal: a 2d run that hangs with no stats table, or a particle count 1/(zhi-zlo) "
    "times your hand calculation. (Verified by execution 2026-08-03.)",

    "[Numerical] An fnum that leaves well under one simulation particle per cell still "
    "runs cleanly and still reports a plausible collision rate, but every PER-CELL "
    "diagnostic becomes garbage. Measured on a 60x60 argon box: fnum 1e11 -> 19.64 "
    "particles/cell, lambda = 1.8865e-6 m, Kn_cell = 1.132; fnum 2e13 -> 0.098 "
    "particles/cell, lambda = 3.33e17 m, Kn_cell = 2.0e23 — 23 orders of magnitude "
    "wrong, rc=0, silent. fnum 2e12 (0.98 particles/cell) is still fine at "
    "lambda = 1.94e-6 m, so the collapse is specifically at sub-particle-per-cell "
    "occupancy. "
    "Signal: 'compute <c> grid all all n' + 'compute reduce ave/min c_c[1]' — require "
    "the cell minimum to be >= 1, not just the average. (Verified by execution "
    "2026-08-03.)",

    "[Numerical] Grid cells much larger than the local mean free path run cleanly and "
    "give an over-diffusive answer. Measured on the Fourier channel: 4x60 grid (cells "
    "~1 mean free path) -> cold-wall flux 2.0917e5 W/m2, mid-gas T = 660.3 K; 4x3 grid "
    "(cells 17.7 mean free paths) -> 4.6193e5 W/m2 (2.21x too high) and T = 606.1 K. "
    "Both rc=0, no warning, and the coarse run even has MORE particles per cell "
    "(1178 vs 59), so a particle-count check does not catch it. "
    "Signal: compute Kn_cell yourself with 'compute <C> lambda/grid ... knall' and "
    "require it >= 1. (Verified by execution 2026-08-03.)",

    "[Numerical] A timestep larger than the mean collision time runs cleanly and "
    "inflates transport. Measured on the Fourier channel (tau = 4.96e-9 s): dt = 1e-9 "
    "(0.2 tau) -> 2.0917e5 W/m2; dt = 1e-8 (2.0 tau) -> 2.2042e5 (+5.4%); dt = 5e-8 "
    "(10.1 tau) -> 4.3788e5 (+109%). All rc=0. 'compute <D> dt/grid all 0.25 0.1 ...' "
    "reports a recommended dt of 4.955e-10 s (cell minimum 4.60e-10 s) in the good AND "
    "the bad run — SPARTA never applies or even compares it for you. "
    "Signal: reduce compute dt/grid with compute reduce min and assert your timestep is "
    "below it. (Verified by execution 2026-08-03.)",

    "[Numerical] A homogeneous equilibrium box is WORTHLESS as a timestep or grid "
    "convergence test — this was a candidate pitfall the campaign had to reject. At "
    "dt = 10 tau the inferred collision frequency 2*Ncollave/Np/dt = 2.0186e8 /s is "
    "indistinguishable from 1/tau = 2.0187e8 /s at dt = 0.2 tau; a 6x6 grid gives "
    "Ncollave = 7133.07 against 7123.80 for 60x60 (0.13%, within noise). Only Kn_cell "
    "moves (1.132 vs 0.113). Both errors show up ONLY in a gradient-driven quantity, so "
    "verify dt and cell size on a wall-bounded or flow-driven case. "
    "Signal: a dt/grid study whose collision statistics are flat is measuring nothing. "
    "(Established by execution 2026-08-03.)",

    "[Physics] Wall fluxes in a driven DSMC run need several flow-through times; the "
    "first stats block is not an answer. Measured cold-wall energy flux by step: "
    "250 -> 5.858e5, 500 -> 3.162e5, 750 -> 2.392e5, 1000 -> 2.079e5, steady mean over "
    "2250-3000 -> 2.092e5 W/m2. Reading at step 250 overestimates by 2.80x, at step 500 "
    "by 1.51x, and the run looks perfectly healthy throughout. "
    "Signal: the two opposed wall fluxes are equal and opposite only after convergence "
    "(5.858e5 vs -5.064e5 at step 250; 2.016e5 vs -1.977e5 at step 3000) — use that "
    "balance as the steady-state test. (Verified by execution 2026-08-03.)",

    "[Physics] surf_collide specular transfers EXACTLY ZERO energy: modelling a thermal "
    "wall with 'specular' gives a wall flux of 0.000 at every output for the whole run "
    "while the hit counts stay healthy, and the gas simply never equilibrates with the "
    "wall (frozen at 638 K between a nominal 300 K and 1000 K pair of walls). specular "
    "takes no temperature argument at all, which is the tell. "
    "Signal: a wall energy-flux column that is exactly 0.000 while Nscoll is large. "
    "Use 'surf_collide <ID> diffuse <Tsurf> <acc>' for a thermal wall. "
    "(Verified by execution 2026-08-03.)",

    "[Syntax] The react command does not check that the file it is handed is a reaction "
    "file: pointing it at a VSS parameter file loads ZERO reactions and the run "
    "proceeds with chemistry silently disabled. Measured: rc=0, no warning, Nreact = 0 "
    "at every stats output while Ncoll = 7114. "
    "Signal: the run header line 'Gas reaction tallies: style tce #-of-reactions 0' and "
    "'Reactions = 0 (0K)' — the ONLY place the problem is visible. "
    "(Verified by execution 2026-08-03.)",

    "[Syntax] The doc-page names in this catalog's 'commands' index are NOT input-script "
    "commands. 55 of the 121 keys (every compute_* and fix_* page, dump_image, "
    "surf_react_adsorb) are documentation filenames; in a deck you write 'compute <ID> "
    "grid ...', 'fix <ID> ave/surf ...', 'dump <ID> image ...', 'surf_react <ID> adsorb "
    "...'. The installed build accepts exactly 66 commands (see "
    "sparta_knowledge.json -> command_surface.true_commands). "
    "Signal: 'ERROR: Unknown command: compute_grid 1 all all nrho (../input.cpp:244)'. "
    "(Verified by execution 2026-08-03.)",
]


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
        pitfalls=[
            "The wall temperature is a coupling unknown updated each preCICE window; "
            "the DSMC heat flux is statistically noisy — average over the (long) solid "
            "thermal timescale. Explicit serial coupling is stable because solid "
            "thermal inertia damps DSMC fluctuations.",
            "Data files (ar.species, *.vss, *.surf) must be IN the run directory — "
            "SPARTA opens them relative to cwd and dies with 'Cannot open species "
            "file ...' (particle.cpp). The couple() tool auto-stages files referenced "
            "by any in.* deck in a participant work_dir (searching the participant's "
            "data_dir, then SPARTA_DATA_DIR, then the distribution); pass explicit "
            "task files via the participant's data_files list.",
            "Half-body surface files (e.g. a half-cylinder arc for a symmetric 2D "
            "case) are OPEN curves: read_surf aborts with 'Watertight check failed "
            "with N unmatched points' (surf.cpp). Place the open endpoints exactly ON "
            "a simulation-box face (e.g. box ylo = 0 for an arc ending at y=0) and "
            "use 'read_surf <file> clip' — verified 2026-08-01.",
            "compute reduce on a fix ave/surf with a SINGLE input column: reference "
            "it as f_ID (per-surf vector), not f_ID[1] — the [1] form aborts with "
            "'Compute reduce fix does not calculate a per-surf array' "
            "(compute_reduce.cpp).",
        ]),
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
        own = ([info["pitfalls"]] if isinstance(info["pitfalls"], str)
               else list(info["pitfalls"]))
        return {
            "description": info["desc"],
            "spatial_dims": info["dims"],
            # pitfalls as a LIST (the physics-specific ones first, then the
            # execution-verified DSMC set that applies to every SPARTA run) —
            # a bare string made catalog tests see it as "no pitfalls" (and the
            # signal counter count characters).
            "pitfalls": own + _VERIFIED_DSMC_PITFALLS,
            "relevant_commands": relevant,
            "worked_example": {"dir": info["example"], "decks": tmpl},
            "solver": "SPARTA DSMC; run: spa_serial -in <script>",
            "unit_systems": "SI (global ... gridcut ... ; fnum sets real-particles-per-simulator)",
            # Cross-cutting, execution-verified against the installed build
            # (campaign 2026-08-03). See sparta_knowledge.json for the full text.
            "installed_build": _KB.get("installed_build", {}),
            "command_surface": _KB.get("command_surface", {}),
            "required_setup_sequence": _KB.get("required_setup_sequence", {}),
            "surfaces": _KB.get("surfaces", {}),
            "reading_output": _KB.get("reading_output", {}),
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
        nonblank = [l for l in content.splitlines()
                    if l.strip() and not l.strip().startswith("#")]
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
