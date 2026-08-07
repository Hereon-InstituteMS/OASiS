"""Runnable 4C deck templates — every one executed on the installed binary.

WHY THIS MODULE EXISTS
----------------------
`prepare_simulation(fourc, <physics>)` used to answer 27 of 4C's 49 catalog
physics with a comment block that said, in as many words, "Not a runnable
input — the user must supply the case-specific mesh + material parameters."
Every other backend in this project ships a fillable skeleton for every physics
it advertises (deal.II 27/27, scikit-fem 22/22, FEBio 17/17, SPARTA 10/10). 4C,
the backend the project is named around, was the exception — and it is the one
whose input format a model is least able to reconstruct from prose: 478
sections, 7383 paths, 2728 distinct keys.

WHAT IS IN HERE, AND WHAT "RUNNABLE" MEANS
------------------------------------------
Each entry is a complete deck, not a skeleton with `<...>` holes. Every one was
executed with `/home/alexander/4C/build/4C` at the rank count recorded in
`np` and exited 0. A deck that 4C rejects is worse than no deck at all,
because it looks like help; so the rule for adding an entry here is that the
exact bytes shipped are the exact bytes that ran.

Each deck is also checked for *physics*, not only for exit status — a run that
completes without the coupling doing anything is a silent failure. The evidence
per deck lives in the `evidence` field and was measured from that deck's own
output (VTU fields, reaction forces, interface fluxes), never assumed.

HOW THE DECKS WERE BUILT
------------------------
From the upstream corpus, not from the grammar. `/home/alexander/4C/tests/
input_files` holds 1978 parseable decks that 4C's own CI runs; each template
starts from a named one of those (`upstream`) and is reduced — mesh shrunk,
regression `RESULT DESCRIPTION` deleted, Belos+XML solvers replaced by direct
UMFPACK — until it is self-contained and small enough to render whole. Two
(`particle_pd_*`) had no upstream deck to start from and were built from the
grammar dump plus the deck shape used by the author of 4C's PD module.

THE TWO DEPENDENCIES THAT COULD NOT BE REMOVED
-----------------------------------------------
`fs3i` and `multiscale` name a file outside the deck. Both were established by
execution, not assumed:

  * FS3I refuses any direct solver — `4C_fs3i_partitioned.cpp:604` throws
    "Iterative solver expected" unless COUPLED_LINEAR_SOLVER is Belos, `:610`
    demands AZPREC Teko, and `TekoPreconditioner::setup` then demands
    TEKO_XML_FILE. The deck therefore names 4C's own recommended block
    preconditioner by absolute path.
  * FE2 multiscale reads the RVE from a second input file through
    `Global::read_micro_fields`; `MICROFILE` cannot be inlined.

Both resolve through `FOURC_ROOT`, the same environment variable this backend
already uses to resolve Exodus meshes. When it is unset the deck still renders
and still teaches the layout, but it will not run — `requires_fourc_root` marks
those two so callers can say so instead of letting a user discover it at
MPI_Abort time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_DECK_DIR = Path(__file__).resolve().parent / "decks"

# Placeholder that the loader rewrites to an absolute path under FOURC_ROOT.
FOURC_ROOT_TOKEN = "@FOURC_ROOT@"


@dataclass(frozen=True)
class Deck:
    """One executed template."""

    physics: str
    variant: str
    filename: str
    np: int                      # rank count the deck was verified at
    upstream: str                # the tests/input_files deck it derives from
    summary: str                 # one line, what it simulates
    evidence: str                # what was measured to show the physics is live
    requires_fourc_root: bool = False
    pitfalls: tuple[str, ...] = field(default_factory=tuple)

    def path(self) -> Path:
        return _DECK_DIR / self.filename

    def text(self) -> str:
        raw = self.path().read_text()
        return _resolve_root(raw)


def _resolve_root(text: str) -> str:
    """Rewrite @FOURC_ROOT@ to the configured 4C source tree.

    Left verbatim when FOURC_ROOT is unset: a wrong absolute path is a worse
    failure than a visible placeholder, because 4C reports the missing file
    and not the missing configuration.
    """
    if FOURC_ROOT_TOKEN not in text:
        return text
    root = os.environ.get("FOURC_ROOT")
    if not root:
        return text
    return text.replace(FOURC_ROOT_TOKEN, str(Path(root).resolve()))


# ── the catalog ────────────────────────────────────────────────────────────
# Ordered by physics name. `np` is the rank count the deck was RUN at, not a
# guess: two decks genuinely need more than one rank (see their pitfalls).

DECKS: tuple[Deck, ...] = (
    Deck(
        physics="fluid_turbulence", variant="les_channel_3d",
        filename="fluid_turbulence.4C.yaml", np=2,
        upstream="f3_cha_8x8x8_recongradl2.4C.yaml + "
                 "f3_stokes_residualbased_rotboxgeom.4C.yaml",
        summary="Large-eddy simulation of turbulent channel flow of height 2: "
                "Smagorinsky subgrid model, periodic in x and z, driven by a "
                "constant streamwise body force.",
        evidence="4C reports 'Turbulence model : Smagorinsky with Smagorinsky "
                 "constant Cs= 0.1' and opens plane-and-time averaged channel "
                 "statistics over the sampling window.",
        pitfalls=(
            "On ONE MPI rank this deck aborts inside "
            "Core::Conditions::PeriodicBoundaryConditions::balance_load with "
            "'terminate called after throwing an instance of int' (SIGABRT, "
            "rc 134). It runs clean on 2 and 4 ranks; reproduced three times. "
            "Periodic boundary conditions need np > 1 on this build.",
            "Every boundary here is periodic or Dirichlet, so the pressure has "
            "a null space. Without DESIGN VOL MODE FOR KRYLOV SPACE PROJECTION "
            "the run stops with 'Nullspace check for sysmat_ failed'.",
            "8x8x8 is a demonstration resolution. An LES on an under-resolved "
            "mesh runs and produces wrong statistics silently.",
        ),
    ),
    Deck(
        physics="fs3i", variant="fs3i_3d",
        filename="fs3i.4C.yaml", np=1,
        upstream="fsi_fp_mono_fs_ga_ga.4C.yaml + fs3i_part_1wc_finperm.4C.yaml",
        summary="Fluid-structure-scalar interaction: a scalar transported in "
                "the fluid and in the solid, exchanging mass across the FSI "
                "interface through a permeability condition.",
        evidence="Fluid scalar 1.000 -> 0.934/0.983 and solid scalar 0.000 -> "
                 "0.054/0.089 over 10 steps, measured from the scatra1/scatra2 "
                 "VTU output; ALE displacement reaches 1.0.",
        requires_fourc_root=True,
        pitfalls=(
            "FS3I rejects direct solvers. 4C_fs3i_partitioned.cpp:604 throws "
            "'Iterative solver expected' unless COUPLED_LINEAR_SOLVER names a "
            "Belos solver, :610 demands AZPREC Teko, and "
            "4C_linear_solver_preconditioner_teko.cpp:48 then throws "
            "'TEKO_XML_FILE parameter not set!'. This is the one deck here "
            "that cannot use UMFPACK and cannot be a single file.",
            "A plain SOLID element aborts in 4C_ssi_clonestrategy.cpp:97 — the "
            "structure elements must be SOLIDSCATRA / WALLSCATRA / SHELLSCATRA "
            "/ TRUSS3SCATRA carrying a meaningful TYPE.",
            "Np_Gen_Alpha for the fluid aborts in 4C_fs3i.cpp:204; BDF2 and "
            "Stationary are rejected as well. One_Step_Theta in all three "
            "fields works.",
        ),
    ),
    Deck(
        physics="fsi", variant="fsi_2d",
        filename="fsi_2d.4C.yaml", np=1,
        upstream="volmortar2D_fsi.4C.yaml + fsi_fp_mono_fs_ga_ga.4C.yaml",
        summary="Partitioned (Dirichlet-Neumann) 2-D fluid-structure "
                "interaction: an elastic block pushed on its far edge drives "
                "an incompressible channel flow on a deforming ALE mesh.",
        evidence="10 coupled steps, FSI outer loop converging with a non-zero "
                 "interface increment (dx 7.7e-05); structure, fluid and ALE "
                 "result files all written.",
        pitfalls=(
            "A Dirichlet FUNCT must be a SYMBOLIC_FUNCTION_OF_SPACE_TIME. "
            "Giving it a SYMBOLIC_FUNCTION_OF_TIME aborts in "
            "4C_utils_function_manager.hpp:143 with 'You tried to query "
            "function 1 as a function of type FunctionOfSpaceTime. Actually, "
            "it has type FunctionOfTime.'",
            "With COUPALGO iter_monolithicfluidsplit the FLUID is the slave "
            "field and may carry no Dirichlet condition on interface dofs — "
            "4C_fsi_monolithicfluidsplit.cpp:135 prints a boxed diagnostic "
            "naming master and slave.",
        ),
    ),
    Deck(
        physics="particle_pd", variant="plate_2d",
        filename="particle_pd_plate.4C.yaml", np=1,
        upstream="none — no bond-based PD deck exists upstream; built from the "
                 "grammar dump and 4C's own PD generator script",
        summary="Bond-based peridynamics: a pre-cracked plate pulled in "
                "tension until the crack runs from the notch tip.",
        evidence="pd_damage_phi mean 0.0668 (the pre-crack alone) at step 0 "
                 "rising to 0.1428 at step 200, max 0.41 -> 0.70 — bonds break "
                 "beyond the notch.",
        pitfalls=(
            "PD is not a separate interaction mode. INTERACTION must be SPH "
            "and PD_BODY_INTERACTION true; the PD parameters then live in "
            "PARTICLE DYNAMIC/PD.",
            "A 2-D PD_DIMENSION requires PARTICLE DYNAMIC/INITIAL AND BOUNDARY "
            "CONDITIONS CONSTRAINT: Projection2D. Without it 4C aborts in "
            "4C_particle_interaction_sph_peridynamic.cpp:92 with 'Plane stress "
            "or plane strain for peridynamic requested. CONSTRAINT must be set "
            "to Projection2D!'",
            "PDFIXED 1 pins a particle at its reference position; PDFIXED 2 "
            "makes it part of a rigid body moved at IMPACTOR_VELOCITY.",
        ),
    ),
    Deck(
        physics="particle_pd", variant="impact_2d",
        filename="particle_pd_impact.4C.yaml", np=1,
        upstream="none — see plate_2d",
        summary="Kalthoff-Winkler edge impact: a rigid impactor strikes a "
                "doubly-notched plate between the notches.",
        evidence="Two peridynamic bodies present; damage mean 0.0825 -> 0.2070 "
                 "over 300 steps and particle speeds reaching 4.6e4 mm/s.",
        pitfalls=(
            "The impactor is a second PDBODYID whose particles carry PDFIXED 2; "
            "contact between bodies is the NORMALCONTACTLAW / NORMAL_STIFF "
            "penalty pair in PARTICLE DYNAMIC/PD.",
            "dx = 12.5 mm here is a teaching resolution. The published "
            "Kalthoff-Winkler study resolves the same specimen at dx = 0.5 mm; "
            "crack paths at this spacing are indicative only.",
        ),
    ),
)


_BY_KEY = {(d.physics, d.variant): d for d in DECKS}


def get(physics: str, variant: str) -> Deck | None:
    return _BY_KEY.get((physics, variant))


def render(physics: str, variant: str) -> str | None:
    """Return the deck text with a short provenance header, or None."""
    d = get(physics, variant)
    if d is None:
        return None
    head = [
        f"# 4C {d.physics} / {d.variant} — runnable template",
        f"# {d.summary}",
        f"# Verified: executed on the installed 4C binary with "
        f"{d.np} MPI rank{'s' if d.np > 1 else ''}, exit 0.",
        f"# Evidence the physics is live: {d.evidence}",
        f"# Derived from upstream deck(s): {d.upstream}",
    ]
    if d.requires_fourc_root:
        head.append(
            "# NOTE: this deck names a file from the 4C source tree; set "
            "FOURC_ROOT so the path resolves.")
    for p in d.pitfalls:
        head.append(f"# Pitfall: {p}")
    return "\n".join(head) + "\n" + d.text()


def variants_for(physics: str) -> list[str]:
    return [d.variant for d in DECKS if d.physics == physics]


def physics_covered() -> list[str]:
    seen: list[str] = []
    for d in DECKS:
        if d.physics not in seen:
            seen.append(d.physics)
    return seen
