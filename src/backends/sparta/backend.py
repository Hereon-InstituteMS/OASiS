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
    — the two are different geometries."""
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


# ── physics capability -> {relevant commands, example template dir, pitfalls} ──
# Each maps DSMC physics to the SPARTA commands and a verified worked example deck.
_PHYSICS = {
    "rarefied_flow": dict(
        desc="Rarefied / free-molecular gas flow (high Knudsen) via DSMC particles",
        dims=[2, 3], example="free",
        commands=["global", "species", "mixture", "create_box", "create_grid",
                  "create_particles", "collide", "fix", "run", "stats",
                  "compute lambda/grid", "compute dt/grid"],
        pitfalls=[
            "[Physics] 'global fnum' and 'global nrho' both DEFAULT TO 1.0 "
            "(update.cpp sets fnum = 1.0 and nrho = 1.0). A deck that forgets them "
            "is not rejected and is not warned about: it builds the grid, creates "
            "the particles you asked for, runs every timestep and exits 0, but at "
            "nrho = 1 molecule per m^3 the collision rate is numerically zero, so "
            "the gas is silently free-molecular and the temperature never relaxes "
            "off its initial value. This is the single most expensive SPARTA "
            "mistake because nothing in the output looks broken. Signal: the Natt "
            "and Ncoll columns of the stats table are 0 on EVERY row while Np is "
            "large, and the end-of-run summary prints 'Collisions/particle/step: 0' "
            "— with return code 0 and not one ERROR or WARNING line. Compare "
            "against a correctly-scaled run, where the same deck gives Natt and "
            "Ncoll in the hundreds per step. (Verified 2026-08-07)",
            "[Numerical] The DSMC cell-size requirement is that a cell be smaller "
            "than the local mean free path, and SPARTA will not check it for you. "
            "Measure it rather than assuming it: 'compute grid all species nrho' + "
            "'compute thermal/grid all all temp' feeding 'compute lambda/grid "
            "c_nrho[*] c_temp[1] lambda tau knall' gives a per-cell Knudsen number "
            "knall = mean free path / cell size, so knall must be at least ~1 and "
            "Bird's rule of thumb (cell <= lambda/3) wants ~3. Signal: knall well "
            "below 1 on the cells that matter — on a box whose lambda is about "
            "2e-5 m, a 4x4x4 grid over a 1e-4 m box reports knall about 0.76 and a "
            "2x2x2 grid about 0.38, while the 10x10x10 grid the same case is "
            "normally run at reports about 2.0. Every one of those runs exits 0. "
            "(Verified 2026-08-07)",
            "[Output] compute lambda/grid returns a SENTINEL, not a physical value, "
            "for any cell that holds no particles: compute_lambda_grid.cpp defines "
            "BIG 1.0e20 and assigns lambda = BIG and tau = BIG when the cell's "
            "number density is zero. A single empty cell therefore destroys any "
            "spatial average taken over the field — 'compute reduce ave' over 1000 "
            "cells with one empty cell returns 1e17, and the poisoning appears "
            "mid-run, not at step 0, because the particles start uniformly placed "
            "and only later fluctuate a cell empty. Signal: a mean free path that "
            "is a round power of ten (1e20, or 1e20 divided by your cell count) and "
            "is numerically EQUAL to the mean collision time, which cannot both be "
            "true since one is a length and the other a time. Take 'compute reduce "
            "min' alongside the average, or raise the particle count, before "
            "believing any lambda/grid average. (Verified 2026-08-07)",
            "[Numerical] The timestep must resolve BOTH the mean collision time and "
            "the cell transit time, and SPARTA hands you the criterion rather than "
            "enforcing it: 'compute dt/grid <group> <tfraction> <cfraction> <tau> "
            "<temp> <usq> <vsq> <wsq>' returns a per-cell recommended timestep, and "
            "the doc cites Bird's recommendation of 0.2 for the collision fraction. "
            "The recommendation is computed from the gas, so it does NOT move when "
            "you change the timestep — a deck running 100x too coarse gets the same "
            "advice as a correct one and no warning either way. Signal: two things "
            "together — 'compute dt/grid' reduced with 'compute reduce min' sits far "
            "below the value on your 'timestep' line (about 6.9e-9 recommended while "
            "the deck ran 7.0e-7), and the end-of-run summary line "
            "'Cell-touches/particle/step' climbs well above 1 (about 1.4 at the "
            "recommended dt, about 4.6 at 10x, about 37 at 100x). Note that an "
            "equilibrium box will NOT show you this in its collision rate: the "
            "collisions per unit time are flat to about 1% across a 1000x change in "
            "dt, so a timestep study run on a box at rest measures nothing. "
            "(Verified 2026-08-07)",
        ]),
    "collision_relaxation": dict(
        desc="Particle-particle collisions with VSS/VHS model + internal energy relaxation",
        dims=[2, 3], example="collide",
        commands=["collide", "species", "mixture", "collide_modify"],
        pitfalls=[
            "[Physics] SPARTA has exactly TWO gas collision styles, 'none' and "
            "'vss' — collide_vss.h holds the only CollideStyle registration in the "
            "tree, so there is no 'collide vhs' and no 'collide hs' command. VHS "
            "and hard-sphere are the SAME style with alpha = 1.0 in the parameter "
            "file (the distribution ships ar.vss with alpha 1.4, ar.vhs with alpha "
            "1.0, and ar.hs with alpha 1.0 and omega 0.5). Omitting the collide "
            "command altogether is legal and means free-molecular flow. Signal: no "
            "collide line gives a run whose Natt and Ncoll stats columns are 0 on "
            "every row with no diagnostic at all; a fabricated style name gives "
            "'ERROR: Unrecognized collision style'. (Verified 2026-08-07)",
            "[Input] The .vss file's column count is set by the 'relax' keyword and "
            "is checked in only one direction. The default 'relax constant' wants 4 "
            "parameters after the species ID (diam, omega, tref, alpha); 'relax "
            "variable' wants 9. A 9-column file read under the default is accepted "
            "and the extra columns are ignored, so air.vss works with 'collide vss "
            "air air.vss'; the reverse is fatal. Signal: 'ERROR on proc 0: Incorrect "
            "line format in VSS parameter file (../collide_vss.cpp:869)' when a "
            "4-column file is read with 'relax variable'. (Verified 2026-08-07)",
            "[Input] Every species the simulation defines must appear in the VSS "
            "file — species present in the file but unused are silently skipped, "
            "but the reverse aborts, and the message names the species so it is "
            "easy to act on. Signal: 'ERROR on proc 0: Species N2 did not appear in "
            "VSS parameter file (../collide_vss.cpp:924)', or, if the counts "
            "disagree for another reason, 'VSS parameters do not match current "
            "species' (collide_vss.cpp:97). The matching failure one level up is "
            "'ERROR: Species ID does not appear in species file "
            "(../particle.cpp:802)' and a missing file is 'ERROR on proc 0: Cannot "
            "open species file <name> (../particle.cpp:711)'. (Verified 2026-08-07)",
            "[Numerical] Collisions are skipped entirely in any cell holding one "
            "particle or none — collide.cpp guards its NTC loop with 'if (np <= 1) "
            "continue'. What this does and does not cost is worth being precise "
            "about, because the obvious test does not show it: in a uniform "
            "equilibrium box the collision rate scaled to physical units (mean "
            "Ncoll per step times fnum) is recovered to about 0.1% even at 1 "
            "particle per cell, and only falls about 4% at 0.1 particles per cell. "
            "The real cost of a thin cell is variance, not a biased collision rate. "
            "Signal: divide the Np column by your grid cell count; below roughly 10 "
            "particles per cell, treat every PER-CELL quantity as noise even though "
            "the global collision counters still look right. (Verified 2026-08-07)",
        ]),
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
                  "compute surf", "fix surf/temp", "fix ave/surf"],
        pitfalls=[
            "[Setup] Reading a surface is only half the job — every element must "
            "also be bound to a collision model with 'surf_modify', and this one "
            "does abort rather than run wrong. The nine surf_collide styles in the "
            "tree are diffuse, specular, cll, td, impulsive, piston, adiabatic, "
            "transparent and vanish. 'diffuse' takes a wall temperature and an "
            "accommodation coefficient in [0,1]. Signal: a missing binding gives "
            "'ERROR: 50 surface elements not assigned to a collision model "
            "(../surf.cpp:343)'; an accommodation outside [0,1] gives 'ERROR: "
            "Illegal surf_collide diffuse command "
            "(../surf_collide_diffuse.cpp:50)'. (Verified 2026-08-07)",
            "[Validation] A DSMC surface quantity is a stochastic estimate, and the "
            "log gives no indication of its uncertainty — this is the failure mode "
            "that costs the most, because an under-sampled answer and a converged "
            "one look identical. The mean is unbiased; only the scatter moves. On a "
            "2D flow past a cylinder the summed surface pressure came out at the "
            "same value to three digits in every configuration tried, while the "
            "seed-to-seed scatter over six seeds went from about 1% at roughly "
            "45000 particles sampled over 20 steps, to about 0.4% over 200 steps, "
            "to about 0.1% over 2000 steps — and, holding the window at 20 steps "
            "but coarsening the particle weight 50-fold to roughly 900 particles, "
            "up to about 15%. Signal: there is no signal in a single run, so make "
            "one — repeat the case with two or three different 'seed' values and "
            "treat the spread across them as your error bar, and lengthen the "
            "'fix ave/surf' window until that spread is small next to the effect "
            "you are claiming. Scatter falls as roughly 1/sqrt(samples), so a 10x "
            "longer window buys about 3x. (Verified 2026-08-07)",
        ]),
    "chemistry": dict(
        desc="Gas-phase chemical reactions (TCE / QK) during DSMC collisions",
        dims=[2, 3], example="chem",
        commands=["react", "react_modify", "species", "collide", "mixture"],
        pitfalls=[
            "[Physics] A reaction whose species are not all declared is discarded "
            "SILENTLY, and a file can be discarded in its entirety this way. "
            "'react tce air.tce' on a deck declaring only N2 and O2 loads and runs "
            "to completion with rc 0 and no diagnostic, having activated nothing; "
            "the same file on a deck declaring N2 O2 NO N O activates 1486 "
            "reactions. Every product species has to appear in the species command, "
            "not just the reactants. Signal: the end-of-run summary line "
            "'Gas reactions = 0' (or a count far below what the file contains) "
            "while a react command is present, and "
            "'Gas-reactions/particle/step: 0'. (Verified 2026-08-07)",
            "[Input] The react styles registered in the tree are tce, qk, tce/qk, "
            "prob, global and adsorb — note that 'tce/qk' is its own style, not a "
            "way of asking for tce plus qk. The file format is tied to the style, "
            "and a react file also needs a 'collide' command in place, since "
            "reactions are evaluated during collisions: with no collide line there "
            "are no collisions and therefore no chemistry, again with no "
            "diagnostic. Signal: 'Gas reactions' nonzero but "
            "'Collision-attempts/particle/step: 0' means the chemistry is loaded "
            "and can never fire. (Verified 2026-08-07)",
        ]),
    "axisymmetric": dict(
        desc="2D axisymmetric DSMC (revolved geometry, radial weighting)",
        dims=[2], example="axi",
        commands=["dimension", "boundary", "create_box", "global weight", "fix"],
        pitfalls=[
            "[Syntax] Axisymmetry is a BOUNDARY style, not a global keyword. "
            "'axisymmetric' is not in the global command's parser at all — the "
            "accepted keywords are fnum, nrho, vstream, temp, field, surfs, "
            "surfgrid, surfmax, splitmax, surftally, gridcut, comm/sort, "
            "comm/style, weight, particle/reorder, mem/limit and optmove. Write "
            "'dimension 2' plus 'boundary o ar p': the 'a' is the axisymmetric "
            "style on the LOWER y face, and create_box must give ylo exactly 0.0. "
            "Signal: 'ERROR: Illegal global command (../update.cpp:1805)'. "
            "(Verified 2026-08-07)",
            "[Setup] Radial particle weighting is 'global weight cell radius', and "
            "it has two ordering constraints — the model must already be "
            "axisymmetric (so the boundary command comes first) and the grid must "
            "already exist (so create_grid comes first too). Signal: 'ERROR: Cannot "
            "use weight cell radius unless axisymmetric (../grid.cpp:1997)' or "
            "'ERROR: Cannot weight cells before grid is defined (../grid.cpp:1983)'. "
            "(Verified 2026-08-07)",
        ]),
    "particle_emission": dict(
        desc="Particle injection / emission from faces or surfaces (inflow boundary)",
        dims=[2, 3], example="emit",
        commands=["fix emit/face", "fix emit/surf", "mixture", "create_particles",
                  "boundary"],
        pitfalls=[
            "[Physics] Mixture number fractions that sum to LESS than 1.0 are not "
            "an error and are not normalised — the entire deficit is absorbed by "
            "the last species in the mixture. mixture.cpp's init_fraction returns "
            "an error only when the sum EXCEEDS 1.0, and then forcibly sets the "
            "cumulative array's final entry to 1.0, which is the bucket the species "
            "sampler draws against. So 'mixture m N2 frac 0.25' plus 'mixture m O2 "
            "frac 0.25' asks for 50/50 and silently delivers 25/75. The documented "
            "redistribution rule, (1 - sum)/M shared among the M species whose frac "
            "was never set, does work: three species with only N2 pinned at 0.5 "
            "gives 497/252/251 out of 1000. Signal: count the species you actually "
            "got with 'compute count <species>' and compare against what you asked "
            "for — the asymmetry lands entirely on the LAST-listed species. The "
            "opposite mistake is loud: 'ERROR: Mixture m fractions exceed 1.0 "
            "(../mixture.cpp:225)'. (Verified 2026-08-07)",
            "[BC] An inflow needs a matching outflow, and SPARTA will not tell you "
            "otherwise. 'fix emit/face' is accepted on a face whose 'boundary' style "
            "is reflecting, so particles are injected into a box they cannot leave "
            "and the population grows without bound until memory does. Signal: the "
            "Np column rises linearly and never plateaus (0, then 35000, then 70000 "
            "over the first 100 steps) with no error; a correctly-vented case "
            "settles to a steady Np within a few flow-through times. Set the "
            "downstream face to 'o' in the boundary command. (Verified 2026-08-07)",
        ]),
    "adaptive_grid": dict(
        desc="Static/dynamic grid adaptation to resolve gradients (refine near shocks)",
        dims=[2, 3], example="adapt",
        commands=["adapt_grid", "fix adapt", "balance_grid", "compute"],
        pitfalls=[
            "[Numerical] Refinement is UNBOUNDED by default — the adapt_grid and "
            "fix adapt keyword default is maxlevel = 0, which means no limit. A "
            "'fix adapt 50 all refine particle 40 20' on a 10x10 grid took it to "
            "1090 cells within 100 steps; the same command with 'maxlevel 2' "
            "appended stopped at 400. Signal: put Ngrid in your stats_style and "
            "watch it — a cell count that keeps climbing between adapt intervals "
            "rather than settling is the tell, and there is no warning at any "
            "point. (Verified 2026-08-07)",
            "[Numerical] Refining does not add particles, so it makes the per-cell "
            "statistics WORSE at the exact moment you were trying to resolve "
            "something. The run above went from 20000 particles in 100 cells (200 "
            "per cell) to about 16000 in 1090 cells (under 15 per cell) — a "
            "thirteenfold loss of per-cell sample size bought with the refinement. "
            "Signal: track Np and Ngrid together and keep their ratio above roughly "
            "10; if refinement drives it down, lower fnum in step with the "
            "refinement so the particle count grows too. (Verified 2026-08-07)",
            "[Syntax] The adapt_grid styles take different argument shapes and "
            "mixing them up produces a parser error rather than anything "
            "informative: 'particle' takes two bare numbers (refine and coarsen "
            "particle-count thresholds), while a compute reference belongs to the "
            "'value' style, which takes c_ID or c_ID[N] followed by two thresholds. "
            "Signal: 'ERROR: Expected floating point parameter in input script or "
            "data file (../adapt_grid.cpp:248)' when a c_ID is passed where "
            "'particle' expected a number. (Verified 2026-08-07)",
        ]),
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
                  "fix ave/surf", "compute reduce", "read_surf"],
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
            "[Mesh] Half-body surface files (e.g. a half-cylinder arc for a "
            "symmetric 2D case) are OPEN curves, and read_surf runs a watertight "
            "check on them. The check has exactly one exemption, in surf.cpp: an "
            "unmatched endpoint is forgiven if Geometry::point_on_hex puts it on a "
            "face of the simulation box. So the fix is to place both open endpoints "
            "exactly ON a box face (e.g. box ylo = 0 for an arc ending at y=0); the "
            "'clip' keyword is NOT what rescues this case — an arc anchored on ylo=0 "
            "is accepted with or without it, and clip on an arc whose ends float in "
            "the interior still aborts. Signal: 'ERROR: Watertight check failed with "
            "2 unmatched points (../surf.cpp:1168)'. (Verified 2026-08-07)",
            "[API] The per-surf plumbing changes shape at each hop, and each hop has "
            "its own message. 'compute surf' emits a per-surf ARRAY even for one "
            "value, so 'fix ave/surf' must take c_ID[1]; 'fix ave/surf' with a single "
            "input then emits a per-surf VECTOR, so 'compute reduce' must take f_ID "
            "with no bracket. Signal: passing c_ID gives 'ERROR: Fix ave/surf compute "
            "does not calculate a per-surf vector (../fix_ave_surf.cpp:150)', and "
            "passing f_ID[1] gives 'Compute reduce fix does not calculate a per-surf "
            "array' (compute_reduce.cpp). (Verified 2026-08-07)",
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
        return {
            "description": info["desc"],
            "spatial_dims": info["dims"],
            # pitfalls as a LIST (one curated DSMC pitfall per physics), matching
            # every other backend — a bare string made catalog tests see it as
            # "no pitfalls" (and the signal counter count characters).
            "pitfalls": [info["pitfalls"]] if isinstance(info["pitfalls"], str)
                        else list(info["pitfalls"]),
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
