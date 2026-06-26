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
    sorted_by_step,
    SolverBackend, BackendStatus, InputFormat,
    PhysicsCapability, JobHandle,
)
from core.registry import register_backend

logger = logging.getLogger("oasis.fourc")

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
    """Import the 4C generators — self-contained in oasis."""
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
            return BackendStatus.MISCONFIGURED, "4C generators not found in oasis"
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
                [2, 3], ["SOLID HEX8", "SOLID QUAD4"],
                ["umbrella"]),
            PhysicsCapability(
                "thermal",
                "[Umbrella] Thermal-analysis family pitfalls "
                "(applies to heat, thermo, tsi). For specific "
                "physics use heat / thermo / tsi directly.",
                [2, 3], ["THERMO QUAD4", "THERMO HEX8"],
                ["umbrella"]),
            PhysicsCapability(
                "input_format",
                "[Reference] Cross-physics general 4C input "
                "pitfalls (ExodusII 1-indexed block IDs, "
                "SYMBOLIC_FUNCTION_OF_SPACE_TIME COMPONENT "
                "requirement, NUMDOF conflicts on shared "
                "FSI/TSI nodes, .yaml-only extension, "
                "post_vtu vs IO/RUNTIME VTK OUTPUT, WALL→SOLID "
                "rename, etc.). Not a PDE physics — meta-"
                "reference entry. Underlying KNOWLEDGE key in "
                "data/fourc_knowledge.py is 'input_format'.",
                [2, 3], ["N/A — meta-reference"], ["N/A"]),
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
        # Resolution: merge data/fourc_knowledge.py (rich
        # course-level dict — description / methods / variants /
        # constitutive_laws / etc.) with the generator's
        # per-physics pitfalls list. Previously the data file
        # SHADOWED the generator: if FOURC_KNOWLEDGE had an
        # entry without a 'pitfalls' field, get_knowledge
        # returned that entry and the generator's pitfalls
        # were unreachable. Critic-audit 2026-06-01 finding #14
        # (fourc::contact had 0 pitfalls reachable; the actual
        # 8 contact.py pitfalls were silently shadowed).
        data_entry: dict = {}
        try:
            import sys
            data_dir = str(Path(__file__).resolve().parents[3] / "data")
            if data_dir not in sys.path:
                sys.path.insert(0, data_dir)
            from fourc_knowledge import FOURC_KNOWLEDGE
            data_entry = FOURC_KNOWLEDGE.get(physics, {})
        except ImportError:
            pass

        gen_entry: dict = {}
        try:
            get_gen, _ = _get_generators()
            gen = get_gen(physics)
            gen_entry = gen.get_knowledge()
        except Exception:  # noqa: BLE001
            pass

        if data_entry and gen_entry:
            # Merge: data_entry wins for shared keys (its
            # description / methods / variants are richer);
            # gen_entry's pitfalls list is preserved unless
            # data_entry has its own.
            merged = dict(data_entry)
            if not data_entry.get("pitfalls") and gen_entry.get(
                    "pitfalls"):
                merged["pitfalls"] = gen_entry["pitfalls"]
            # Carry over any other gen-only keys.
            for k, v in gen_entry.items():
                if k not in merged:
                    merged[k] = v
            return merged
        if data_entry:
            return data_entry
        if gen_entry:
            return gen_entry
        return {"error": f"no knowledge for {physics!r} in fourc"}

    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        # Umbrella / meta-reference physics: catalog declares
        # these so they appear in discover() and knowledge()
        # surfaces (e.g. scalar_transport groups poisson + heat +
        # electrochemistry + level-set + low-mach scalars). They
        # are documentation-only — generate_input returns a YAML
        # commentary block pointing to the concrete physics names
        # in the same family. Without this early-return, calling
        # generate_input('scalar_transport', 'umbrella', {}) would
        # cascade through the inline / tutorial / generator chain
        # and raise ValueError.
        if variant in ("umbrella", "N/A"):
            return self._umbrella_template(physics, variant)

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

        # Honest reference stub BEFORE the generator fallback. Deep
        # multiphysics rows (xfem, fs3i, fpsi, ehl, fbi, pasi, ssi/
        # ssti/sti, cardiac_monodomain, arterial/airway/lung 1-D,
        # multiscale fe2, beam_interaction, particle_pd/sph, LES,
        # brownian) DO have a generator template, but it is a
        # placeholder full of literal <...> scalars + external mesh
        # references that aborts 4C in MatchTree (probe 2026-06-12).
        # A documented stub the user can read beats a guaranteed
        # MPI_Abort, so the stub catalog takes precedence over the
        # broken placeholder. The stub omits MATERIALS on purpose:
        # validate_input() flags it as non-runnable, so the probe
        # never reports it as a completed run — it is honestly
        # "documented, not runnable".
        stub = self._reference_stub_template(physics, variant)
        if stub is not None:
            return stub

        # Fallback: try generator-based templates
        try:
            get_gen, _ = _get_generators()
            gen = get_gen(physics)
            content = gen.get_template(variant)
            content = self._resolve_mesh_references(content)
            return content
        except Exception as e:
            # Last-resort reference stub. The catalog advertises
            # (physics, variant) in supported_physics() so it
            # appears in discover() — without something runnable
            # here, calling generate_input on that pair raised
            # ValueError unconditionally. Many 4C problems
            # (plasticity, particle_pd impact, particle_sph
            # dam_break, porous_media terzaghi/consolidation) need
            # case-specific mesh + parameters that cannot be
            # baked into a generic template. The stub is a
            # valid YAML reference that documents what's
            # required so an LLM agent or human user knows
            # what to fill in. See _reference_stub_template for
            # the list of stub-eligible (physics, variant) pairs.
            stub = self._reference_stub_template(
                physics, variant)
            if stub is not None:
                return stub
            raise ValueError(f"No 4C template for {physics}/{variant}: {e}")

    def _reference_stub_template(self, physics: str,
                                  variant: str) -> str | None:
        """Reference-stub fallback for catalog-advertised
        (physics, variant) pairs that need case-specific
        mesh + parameters (and thus can't be baked into a
        generic generator). Returns a YAML commentary
        block that documents what the user must fill in.

        Returns None for pairs not in the stub catalog —
        the caller falls through to its original
        ValueError.
        """
        # Map (physics, variant) → (problemtype, description,
        # required sections, pitfalls).
        stubs: dict[tuple[str, str], dict] = {
            ("plasticity", "linear_2d"): {
                "problemtype": "Structure",
                "summary": ("Elasto-plasticity (small-strain "
                            "J2 / von Mises with isotropic "
                            "hardening) on a 2D QUAD4 mesh."),
                "needs": ["MAT_Struct_PlasticLinElast or "
                          "MAT_Struct_J2Plast with parameters "
                          "YIELD, ISOHARD",
                          "STRUCTURE GEOMETRY with WALL→SOLID "
                          "QUAD4 elements (post-2026.3 rename)",
                          "STRUCTURAL DYNAMIC with "
                          "DYNAMICTYPE: Statics or GenAlpha",
                          "Solver section with appropriate "
                          "Newton tolerance for plastic step"],
                "pitfalls": ["YIELD too low → instant plastic "
                             "yielding; mesh-dependent",
                             "ISOHARD = 0 with finite YIELD → "
                             "perfectly plastic; non-unique "
                             "solution at limit load"],
            },
            ("plasticity", "nonlinear_3d"): {
                "problemtype": "Structure",
                "summary": ("Finite-strain plasticity (J2 or "
                            "GTN damage) on a 3D HEX8 mesh."),
                "needs": ["MAT_Struct_PlasticNlnLogNeoHooke "
                          "or MAT_Struct_PlasticGTND2 (with "
                          "GTN-damage params f0, fcr, fF)",
                          "STRUCTURE GEOMETRY with SOLID HEX8 "
                          "elements + KINEM nonlinear",
                          "STRUCTURAL DYNAMIC with DYNAMICTYPE: "
                          "Statics (quasi-static) or GenAlpha"],
                "pitfalls": ["KINEM 'linear' kills geometric "
                             "nonlinearity → wrong necking",
                             "GTN nucleation (eN, sN, fN) "
                             "tuning critical for ductile "
                             "fracture initiation"],
            },
            ("particle_pd", "impact_2d"): {
                "problemtype": "Particle",
                "summary": ("Bond-based peridynamics impact "
                            "problem with two PD bodies: a "
                            "pdphase target and a boundaryphase "
                            "impactor with prescribed velocity."),
                "needs": ["MAT_PD_ElastBondbased with bulk "
                          "modulus K and critical_stretch s0",
                          "Two PARTICLE_PHASE sections "
                          "(target pdphase + impactor "
                          "boundaryphase)",
                          "BINNING STRATEGY with appropriate "
                          "BIN_SIZE_LOWER_BOUND",
                          "PARTICLE DYNAMIC with explicit time "
                          "integration; CFL: dt < c_safety * "
                          "dx / wave speed"],
                "pitfalls": ["BIN_SIZE_LOWER_BOUND too large → "
                             "neighbor search slow / OOM",
                             "critical_stretch too low → "
                             "spurious bond breakage at "
                             "boundaries"],
            },
            ("particle_sph", "dam_break_2d"): {
                "problemtype": "Particle",
                "summary": ("SPH dam-break: 2D rectangular "
                            "column of fluid collapsing onto a "
                            "rigid floor under gravity."),
                "needs": ["MAT_PARTICLE with SPH_FLUID "
                          "particle type (density, "
                          "DYN_VISCOSITY, BULK_MODULUS, "
                          "SOUNDSPEED)",
                          "PARTICLE_PHASE for the fluid "
                          "column + a boundaryphase for the "
                          "floor/walls",
                          "PARTICLE DYNAMIC with explicit "
                          "time integration + appropriate "
                          "CFL"],
                "pitfalls": ["SOUNDSPEED too low → fluid "
                             "compresses unrealistically; "
                             "rule of thumb c >= 10 * v_max",
                             "SMOOTHING_LENGTH too small → "
                             "spurious tensile-instability "
                             "voids; rule of thumb h ~ 1.3 * "
                             "particle spacing"],
            },
            ("porous_media", "terzaghi_2d"): {
                "problemtype": "Poroelasticity",
                "summary": ("Terzaghi 1-D consolidation "
                            "benchmark — saturated soil "
                            "column under instantaneously "
                            "applied surface load, pore "
                            "pressure dissipates over time."),
                "needs": ["MAT_FluidPoro (for the fluid "
                          "phase) + MAT_Struct_StVenantKirchhoff "
                          "or PLN_ELASTIC (for the solid "
                          "skeleton)",
                          "STRUCTURE GEOMETRY with "
                          "WALLQ4PORO elements (NOT plain "
                          "WALL — the poro suffix is "
                          "required)",
                          "POROELASTICITY DYNAMIC with "
                          "monolithic coupling (NOT "
                          "partitioned for the consolidation "
                          "stage)",
                          "Drainage BC at the top surface "
                          "(zero pore pressure)"],
                "pitfalls": ["Time scale: poro is "
                             "DYNAMIC formulation — slow "
                             "load ramp >>10 * H/sqrt(E/rho) "
                             "to avoid elastic waves",
                             "Permeability k too small → "
                             "no consolidation in run time; "
                             "rule of thumb t_final >> H^2 "
                             "/ (c_v) where c_v = k*E/mu_f"],
            },
            ("porous_media", "consolidation_3d"): {
                "problemtype": "Poroelasticity",
                "summary": ("3-D consolidation under "
                            "distributed surface load — "
                            "axisymmetric or rectangular "
                            "footprint; HEX8 SOLIDH8PORO "
                            "elements."),
                "needs": ["Same MAT_FluidPoro + solid "
                          "skeleton as terzaghi_2d",
                          "STRUCTURE GEOMETRY with "
                          "SOLIDH8PORO (3D variant)",
                          "POROELASTICITY DYNAMIC with "
                          "monolithic coupling",
                          "Drainage BC on the loaded "
                          "surface"],
                "pitfalls": ["Element-locking: at low "
                             "permeability, the standard "
                             "displacement-based formulation "
                             "locks volumetrically; use "
                             "u-p mixed (SOLIDH8PORO is "
                             "p1-p1 stabilised) or check "
                             "for incompressibility "
                             "locking",
                             "Slow load ramp same as "
                             "terzaghi_2d"],
            },
            # ── Deep multiphysics rows that genuinely need a
            #    case-specific mesh (often TWO meshes), a second
            #    input file, patient-derived topology, an explicit
            #    particle cloud, or a build feature this 4C lacks.
            #    A generic inline QUAD4/HEX8 mesh cannot carry them,
            #    so they are honest reference stubs instead of a
            #    guaranteed MPI_Abort from the placeholder template
            #    (probe 2026-06-12).
            ("fsi_xfem", "xfem_fsi_3d"): {
                "problemtype": "Fluid_Structure_Interaction_XFEM",
                "summary": ("XFEM fluid-structure interaction: a "
                            "structure is immersed in a fixed "
                            "background fluid mesh; the interface "
                            "is captured by XFEM enrichment with "
                            "Nitsche coupling (no ALE)."),
                "needs": ["TWO meshes: a background fluid HEX8 "
                          "block + a separate immersed structure "
                          "HEX8 body (cannot be one inline grid)",
                          "XFLUID DYNAMIC + XFLUID DYNAMIC/GHOST "
                          "PENALTY sections with a Nitsche "
                          "penalty parameter",
                          "STRUCTURAL DYNAMIC (GenAlpha) + FLUID "
                          "DYNAMIC (Np_Gen_Alpha) + two solvers"],
                "pitfalls": ["BUILD LIMITATION: this 4C build "
                             "appears to lack Qhull — the Cut "
                             "library's tessellation backend. "
                             "Interface cutting aborts in "
                             "Cut::TetMesh::call_q_hull (observed "
                             "for level-set on this build). "
                             "Rebuild 4C with Qhull enabled before "
                             "expecting XFEM cut cases to run.",
                             "Nitsche penalty too small → unstable "
                             "interface traction; too large → "
                             "ill-conditioned system"],
            },
            ("xfem_fluid", "xfem_3d"): {
                "problemtype": "Fluid_XFEM",
                "summary": ("XFEM fluid with an embedded interface "
                            "(void, obstacle, or two-phase) on a "
                            "non-conforming background mesh; "
                            "Nitsche coupling enforces interface "
                            "conditions weakly."),
                "needs": ["A background fluid HEX8 mesh PLUS a "
                          "separate cutter mesh or level-set "
                          "function defining the interface",
                          "XFLUID DYNAMIC + XFLUID DYNAMIC/GHOST "
                          "PENALTY sections",
                          "MAT_fluid + appropriate stabilisation"],
                "pitfalls": ["BUILD LIMITATION: this 4C build "
                             "appears to lack Qhull (Cut-library "
                             "tessellation). A cut element aborts "
                             "in Cut::TetMesh::call_q_hull "
                             "(observed for level-set here). "
                             "Rebuild with Qhull before running "
                             "XFEM cut cases.",
                             "Ghost-penalty factor controls "
                             "stability of small cut fractions"],
            },
            ("fs3i", "fs3i_3d"): {
                "problemtype": ("Fluid_Porous_Structure_Scalar_"
                                "Scalar_Interaction"),
                "summary": ("FS3I: 5-field coupling — fluid + "
                            "structure + ALE + a fluid-side scalar "
                            "+ a structure-side scalar (e.g. drug "
                            "mass transfer through a vessel wall)."),
                "needs": ["Separate fluid and structure HEX8 "
                          "meshes sharing a matched FSI interface "
                          "(two node sets) — not a single grid",
                          "STRUCTURAL + FLUID + ALE + FSI DYNAMIC + "
                          "two SCALAR TRANSPORT fields with S2I "
                          "interface coupling",
                          "Monolithic or partitioned FSI solver "
                          "block with shape derivatives"],
                "pitfalls": ["Interface meshes must be conforming "
                             "(matching nodes) or a mortar "
                             "projection is required",
                             "Scalar interface permeability sets "
                             "the transfer rate — easy to make "
                             "the transport trivially zero"],
            },
            ("fpsi", "monolithic_3d"): {
                "problemtype": "Fluid_Porous_Structure_Interaction",
                "summary": ("FPSI: free Navier-Stokes fluid over a "
                            "deformable Biot porous bed, monolithic "
                            "coupling, with ALE tracking the free-"
                            "fluid boundary."),
                "needs": ["TWO meshes: a free-fluid HEX8 domain and "
                          "a porous HEX8 domain sharing the FPSI "
                          "interface (fluid-side + poro-side node "
                          "sets)",
                          "FLUID + POROELASTICITY + ALE + FPSI "
                          "DYNAMIC sections",
                          "MAT_FluidPoro skeleton + pore-fluid "
                          "materials on the porous block"],
                "pitfalls": ["Beavers-Joseph-Saffman interface "
                             "condition parameters strongly affect "
                             "the slip velocity",
                             "Monolithic FPSI needs a block "
                             "preconditioner tuned per field — the "
                             "default may not converge"],
            },
            ("ehl", "ehl_3d"): {
                "problemtype": "Elastohydrodynamic_Lubrication",
                "summary": ("Elastohydrodynamic lubrication: a 2-D "
                            "Reynolds lubrication film coupled to a "
                            "3-D elastic body — film pressure "
                            "deforms the body, which changes the "
                            "film geometry."),
                "needs": ["TWO meshes of different dimension: a 2-D "
                          "QUAD4 lubrication film + a 3-D HEX8 "
                          "elastic body whose contact surface "
                          "receives the film pressure",
                          "STRUCTURAL DYNAMIC + LUBRICATION DYNAMIC "
                          "+ EHL DYNAMIC coupling sections",
                          "MAT_lubrication + a structural material"],
                "pitfalls": ["The film-to-surface projection "
                             "(matching or mortar) must be set up "
                             "per geometry",
                             "Cavitation handling in the Reynolds "
                             "solver is needed for diverging gaps"],
            },
            ("fbi", "penalty_3d"): {
                "problemtype": "Fluid_Beam_Interaction",
                "summary": ("Fluid-beam interaction: a flexible "
                            "beam immersed in a 3-D fluid channel, "
                            "coupled by a penalty drag term."),
                "needs": ["TWO meshes: a fluid HEX8 channel + a "
                          "separate beam LINE2 mesh embedded in it",
                          "FBI DYNAMIC (penalty parameter) + "
                          "STRUCTURAL + FLUID DYNAMIC + a BINNING "
                          "STRATEGY for the beam-fluid search",
                          "MAT_BeamReissnerElastHyper + MAT_fluid"],
                "pitfalls": ["Penalty parameter trades coupling "
                             "accuracy against conditioning",
                             "BIN_SIZE_LOWER_BOUND must cover the "
                             "beam-fluid search radius or pairs are "
                             "missed"],
            },
            ("pasi", "dem_impact_3d"): {
                "problemtype": "Particle_Structure_Interaction",
                "summary": ("Particle-structure interaction: DEM "
                            "granular particles impact a deformable "
                            "structural plate via Hertz contact."),
                "needs": ["A structural HEX8 plate mesh PLUS an "
                          "explicit cloud of DEM particles (a "
                          "PARTICLES section with per-particle "
                          "positions, radii, phases)",
                          "PARTICLE DYNAMIC (INTERACTION DEM) + "
                          "PASI DYNAMIC + STRUCTURAL DYNAMIC + a "
                          "BINNING STRATEGY with the domain box",
                          "MAT_ParticleMaterialDEM + a structural "
                          "material"],
                "pitfalls": ["Explicit DEM time step must satisfy "
                             "the Rayleigh/contact-stiffness CFL or "
                             "it explodes",
                             "DOMAINBOUNDINGBOX must enclose all "
                             "particle motion"],
            },
            ("ssi", "monolithic_elch_3d"): {
                "problemtype": "Structure_Scalar_Interaction",
                "summary": ("Monolithic structure-scalar (electro"
                            "chemistry) interaction: two electrode "
                            "blocks with a scatra-scatra interface "
                            "(S2I) using Butler-Volmer kinetics; "
                            "lithium intercalation swells the "
                            "structure."),
                "needs": ["A mesh with TWO electrode blocks sharing "
                          "an S2I interface (matching node sets on "
                          "each side) — not a single block",
                          "SSI CONTROL (COUPALGO ssi_Monolithic, "
                          "SCATRATIMINTTYPE Elch) + SCALAR "
                          "TRANSPORT/S2I COUPLING + ELCH CONTROL",
                          "SOLIDSCATRA elements + electrode/"
                          "electrolyte materials with kinetics"],
                "pitfalls": ["S2I requires matching (or mortar) "
                             "interface meshes; mismatched nodes "
                             "silently drop the coupling",
                             "Butler-Volmer exchange current "
                             "density sets the interface "
                             "overpotential — easy to stall"],
            },
            ("ssti", "monolithic_3d"): {
                "problemtype": "Structure_Scalar_Thermo_Interaction",
                "summary": ("Monolithic structure-scalar-thermo "
                            "interaction: structural mechanics + "
                            "electrochemical scalar transport + "
                            "thermal field, with S2I interface "
                            "coupling and thermal expansion."),
                "needs": ["A multi-block electrode mesh with an S2I "
                          "interface (as for ssi) plus a cloned "
                          "thermal field",
                          "SSI CONTROL + TSI DYNAMIC + SCALAR "
                          "TRANSPORT + THERMAL DYNAMIC monolithic "
                          "sub-couplings",
                          "SOLIDSCATRA elements + electrode + "
                          "Fourier materials via mesh cloning"],
                "pitfalls": ["Three coupled fields: the monolithic "
                             "block preconditioner must be tuned or "
                             "Newton stalls",
                             "Same S2I matching-mesh requirement as "
                             "ssi"],
            },
            ("sti", "monolithic_3d"): {
                "problemtype": "Scalar_Thermo_Interaction",
                "summary": ("Monolithic scalar-thermo interaction: "
                            "coupled scalar transport and thermal "
                            "fields with the Soret effect (thermo"
                            "diffusion) and reaction heat sources."),
                "needs": ["A single HEX8 domain is enough, BUT the "
                          "scatra field must be cloned to a thermo "
                          "field with matching DOFs and the Soret "
                          "coupling material set up",
                          "STI DYNAMIC (COUPALGO sti_Monolithic) + "
                          "SCALAR TRANSPORT + THERMAL DYNAMIC + the "
                          "monolithic solver block",
                          "MAT_soret (couples concentration and "
                          "temperature) + MAT_Fourier + MAT_scatra"],
                "pitfalls": ["The Soret coefficient links the two "
                             "fields; with it zero the problem "
                             "decouples and the 'interaction' is "
                             "meaningless",
                             "Initial fields for both temperature "
                             "and concentration must be consistent"],
            },
            ("cardiac_monodomain", "monodomain_3d"): {
                "problemtype": "Cardiac_Monodomain",
                "summary": ("Cardiac electrophysiology: action-"
                            "potential propagation through a tissue "
                            "slab via the monodomain reaction-"
                            "diffusion equation with an ionic cell "
                            "model."),
                "needs": ["A tissue HEX8/TET4 mesh WITH per-element "
                          "fiber directions (anisotropic "
                          "conductivity) — fibers are case-specific",
                          "SCALAR TRANSPORT DYNAMIC (nonlinear, "
                          "monodomain) + a stimulation Neumann "
                          "condition with a pulse function",
                          "MAT_myocard with an ionic cell MODEL "
                          "(e.g. TenTusscher, MinimalModel) and "
                          "DIFF1/2/3 fiber/sheet/normal conduction"],
                "pitfalls": ["Mesh must resolve the depolarisation "
                             "wavefront (~0.2 mm) or conduction "
                             "velocity is wrong",
                             "Ionic model + time step must match "
                             "(stiff models need small dt)"],
            },
            ("arterial_network", "single_artery_1d"): {
                "problemtype": "ArterialNetwork",
                "summary": ("1-D arterial pulse-wave propagation: a "
                            "compliant artery with a prescribed "
                            "inlet flow and a 3-element Windkessel "
                            "(RCR) outlet."),
                "needs": ["A 1-D ARTERY line mesh (NODE COORDS + "
                          "ARTERY LINE2 elements) — the network "
                          "topology is case-specific",
                          "ARTERIAL DYNAMIC + an inflow waveform "
                          "FUNCT + DESIGN POINT WINDKESSEL "
                          "CONDITIONS at the outlet",
                          "MAT_cnst_art (wall Young, thickness, "
                          "reference area, blood density/"
                          "viscosity)"],
                "pitfalls": ["Windkessel R/C values set the "
                             "reflection coefficient — wrong values "
                             "give nonphysical pressure",
                             "Time step must resolve the pulse "
                             "wave-speed CFL along the segment"],
            },
            ("reduced_airways", "airways_1d"): {
                "problemtype": "ReducedDimensionalAirWays",
                "summary": ("1-D reduced-dimensional airway tree: a "
                            "branching network of compliant airway "
                            "segments from trachea to terminal "
                            "bronchioles, terminating in acinar "
                            "compartments."),
                "needs": ["An airway-tree line mesh (REDAIRWAY "
                          "LINE2 elements) with physiologically "
                          "ordered branching — derived per patient, "
                          "not generic",
                          "REDUCED DIMENSIONAL AIRWAYS DYNAMIC + "
                          "ACINUS sub-section + a tracheal pressure "
                          "(breathing) waveform",
                          "MAT_redairway_material (proximal + "
                          "distal) + MAT_redairway_acinus_material"],
                "pitfalls": ["Tree topology must follow a Horsfield/"
                             "Strahler ordering or the resistance "
                             "network is unphysical",
                             "Acinar compliance varies with disease "
                             "(emphysema, fibrosis, ARDS)"],
            },
            ("reduced_lung", "lung_1d"): {
                "problemtype": "ReducedLung",
                "summary": ("Reduced-dimensional lung: a 1-D airway "
                            "tree coupled to 0-D alveolar acini and "
                            "optionally a 3-D parenchyma."),
                "needs": ["A patient-derived 1-D airway tree (NODE "
                          "COORDS + ARTERY/REDAIRWAY elements) plus "
                          "per-acinus compliance distribution",
                          "REDUCED DIMENSIONAL AIRWAYS DYNAMIC and "
                          "the 1D-0D (or 3D-0D mortar) coupling "
                          "setup",
                          "Airway wall + acinar materials"],
                "pitfalls": ["1D-0D coupling matches flow/pressure "
                             "at outlets — inconsistent units stall "
                             "the solve",
                             "Tree must be physiologically "
                             "reasonable (see reduced_airways)"],
            },
            ("multiscale", "fe2_3d"): {
                "problemtype": "Structure",
                "summary": ("FE^2 computational homogenisation: each "
                            "macro Gauss point evaluates its "
                            "constitutive response by solving a "
                            "micro-scale RVE boundary-value "
                            "problem."),
                "needs": ["TWO input files: this MACRO file PLUS a "
                          "separate MICRO RVE input file referenced "
                          "by MICRO_INPUT_FILE",
                          "MAT_Struct_Multiscale on the macro mesh "
                          "(points to the micro file + micro solver "
                          "id)",
                          "A macro structural mesh + the full RVE "
                          "definition (mesh, material, periodic "
                          "BCs) in the micro file"],
                "pitfalls": ["The RVE must be large enough to be a "
                             "representative volume or the "
                             "homogenised response is mesh-"
                             "dependent",
                             "FE^2 is expensive: one micro solve per "
                             "macro Gauss point per Newton step"],
            },
            ("beam_interaction", "beam_contact_3d"): {
                "problemtype": "Structure",
                "summary": ("Beam-to-beam contact: two or more beams "
                            "interacting via penalty contact "
                            "(crossing, sliding)."),
                "needs": ["A mesh with TWO (or more) separate beam "
                          "LINE2 element blocks positioned to come "
                          "into contact",
                          "BEAM INTERACTION (STRATEGY beam_to_beam_"
                          "contact) + BEAM TO BEAM CONTACT penalty "
                          "+ a BINNING STRATEGY",
                          "MAT_BeamReissnerElastHyper per beam"],
                "pitfalls": ["SEARCH_RADIUS / BIN_SIZE_LOWER_BOUND "
                             "must cover the closest approach or "
                             "contact pairs are missed",
                             "Penalty parameter trades penetration "
                             "against conditioning"],
            },
            ("beam_interaction", "beam_solid_meshtying_3d"): {
                "problemtype": "Structure",
                "summary": ("Beam-to-solid volume meshtying: a beam "
                            "embedded in a solid block as a "
                            "reinforcement, coupled by penalty or "
                            "mortar volume meshtying."),
                "needs": ["TWO overlapping meshes: a solid HEX8 "
                          "block + a beam LINE2 mesh threaded "
                          "through its interior",
                          "BEAM INTERACTION (STRATEGY beam_to_solid_"
                          "volume_meshtying) + the meshtying penalty "
                          "+ Gauss-point settings + a BINNING "
                          "STRATEGY",
                          "Beam material + solid material"],
                "pitfalls": ["GAUSS_POINTS on the beam controls "
                             "coupling accuracy vs. cost",
                             "Beam must actually lie inside the "
                             "solid volume or no pairs are found"],
            },
            ("particle_pd", "plate_2d"): {
                "problemtype": "Particle",
                "summary": ("2-D peridynamics: a pre-cracked plate "
                            "fragmenting under a prescribed boundary "
                            "velocity, with bond-based PD "
                            "interaction."),
                "needs": ["An explicit particle cloud (a PARTICLES "
                          "section listing every PD point position "
                          "+ phase) — generated by a meshing "
                          "script, not an inline grid helper",
                          "PARTICLE DYNAMIC/SPH and PARTICLE "
                          "DYNAMIC/PD sub-sections (the SPH block is "
                          "required even for pure PD) + a BINNING "
                          "STRATEGY",
                          "MAT_ParticlePD (Young, critical stretch) "
                          "+ MAT_ParticleSPHBoundary"],
                "pitfalls": ["INITIALPARTICLESPACING must match the "
                             "PD grid spacing and the horizon "
                             "(typically 3·dx)",
                             "Explicit time step from the CFL: dt < "
                             "0.5·dx/sqrt(E/rho)"],
            },
            ("particle_sph", "poiseuille_2d"): {
                "problemtype": "Particle",
                "summary": ("2-D SPH Poiseuille flow: pressure-"
                            "driven flow between parallel plates "
                            "developing the parabolic profile."),
                "needs": ["An explicit SPH particle cloud (a "
                          "PARTICLES section with fluid + boundary "
                          "particle positions) generated by a "
                          "script — not an inline FE grid",
                          "PARTICLE DYNAMIC (INTERACTION SPH) + "
                          "PARTICLE DYNAMIC/SPH (kernel, spacing) + "
                          "a driving body force + a BINNING "
                          "STRATEGY",
                          "MAT_ParticleSPHFluid + "
                          "MAT_ParticleSPHBoundary"],
                "pitfalls": ["INITRADIUS must equal the kernel "
                             "support (≈3·dx for a quintic spline)",
                             "Bulk modulus sets the artificial "
                             "speed of sound; too low over-"
                             "compresses the fluid"],
            },
            ("fluid_turbulence", "les_channel_3d"): {
                "problemtype": "Fluid",
                "summary": ("Large-eddy simulation of turbulent "
                            "channel flow with a sub-grid model and "
                            "periodic streamwise/spanwise "
                            "directions."),
                "needs": ["A wall-resolved 3-D HEX8 channel mesh "
                          "graded to the wall (y+ ~ 1) with periodic "
                          "boundary surfaces — a coarse inline grid "
                          "is not a meaningful LES",
                          "FLUID DYNAMIC with a TURBULENCE MODEL "
                          "section (e.g. Smagorinsky/dynamic) + "
                          "periodic boundary conditions + turbulence "
                          "statistics sampling",
                          "MAT_fluid at the target Reynolds number"],
                "pitfalls": ["LES on an under-resolved mesh is "
                             "physically meaningless — it 'runs' but "
                             "the statistics are wrong",
                             "Needs a long sampling time to "
                             "converge mean/Reynolds-stress "
                             "profiles"],
            },
            ("brownian_dynamics", "brownian_3d"): {
                "problemtype": "Structure",
                "summary": ("Brownian dynamics of semiflexible "
                            "polymer filaments: BEAM3R beams under "
                            "thermal-fluctuation (stochastic) "
                            "forcing, optionally with crosslinking."),
                "needs": ["A beam LINE2/LINE3 filament mesh inside a "
                          "periodic box + a BROWNIAN DYNAMICS "
                          "section (thermal energy KT, damping, "
                          "seed)",
                          "STRUCTURAL DYNAMIC with a stochastic "
                          "(statmech) integrator + a BINNING "
                          "STRATEGY for crosslinker search",
                          "MAT_BeamReissnerElastHyper with the "
                          "filament cross-section"],
                "pitfalls": ["The stochastic time step couples to "
                             "KT and damping — too large loses the "
                             "fluctuation-dissipation balance",
                             "Results are statistical: a single "
                             "short run is not representative"],
            },
        }
        spec = stubs.get((physics, variant))
        if spec is None:
            return None
        problemtype = spec["problemtype"]
        summary = spec["summary"]
        needs = "\n".join(
            f"#   {i+1}. {n}" for i, n in enumerate(
                spec["needs"]))
        pitfalls = "\n".join(
            f"#   * {p}" for p in spec["pitfalls"])
        return (
            f"# ============================================\n"
            f"# 4C reference stub: {physics} / {variant}\n"
            f"# ============================================\n"
            f"# {summary}\n"
            f"#\n"
            f"# Not a runnable input — the user must supply\n"
            f"# the case-specific mesh + material parameters.\n"
            f"# This stub lists what's required:\n"
            f"#\n"
            f"{needs}\n"
            f"#\n"
            f"# Pitfalls (see knowledge() for the full set):\n"
            f"{pitfalls}\n"
            f"# ============================================\n"
            f"TITLE:\n"
            f'  - "4C {physics}/{variant} reference stub"\n'
            f"PROBLEM TYPE:\n"
            f'  PROBLEMTYPE: "{problemtype}"\n'
        )

    def _umbrella_template(self, physics: str, variant: str) -> str:
        """Return a YAML-commentary template for umbrella /
        meta-reference physics (scalar_transport,
        structural_mechanics, thermal, particles, input_format).
        These aren't runnable physics inputs — they're a
        catalog cross-reference. The returned YAML is parseable
        and validates against 4C 2026.3 (no PROBLEM TYPE means
        4C reports a 'PROBLEMTYPE missing' diagnostic, but the
        file itself is valid YAML)."""
        family_redirects = {
            "scalar_transport": ("poisson, heat, "
                                 "electrochemistry, level_set, "
                                 "low_mach"),
            "structural_mechanics": ("linear_elasticity, "
                                     "plasticity, "
                                     "structural_dynamics, "
                                     "beams, contact"),
            "thermal": "heat, thermo, tsi",
            "particles": ("particle_pd, particle_sph, pasi, "
                          "dem (use kratos for dem instead)"),
            "input_format": ("meta-reference only — see "
                             "data/fourc_knowledge.py['input_format']"),
        }
        family = family_redirects.get(physics,
                                       "<unknown umbrella>")
        return (
            f"# =====================================================\n"
            f"# 4C umbrella / meta-reference physics: '{physics}'\n"
            f"# variant: '{variant}'\n"
            f"# =====================================================\n"
            f"# This is NOT a runnable 4C input. The catalog\n"
            f"# advertises '{physics}' so it appears in discover()\n"
            f"# and knowledge() results, where it groups related\n"
            f"# physics under a shared documentation umbrella.\n"
            f"#\n"
            f"# For a RUNNABLE input pick one of the concrete\n"
            f"# physics names in the same family:\n"
            f"#\n"
            f"#   {family}\n"
            f"#\n"
            f"# Example: prepare_simulation(fourc, "
            f"{family.split(',')[0].strip()})\n"
            f"# returns a real template, knowledge dict, and\n"
            f"# pitfall list for the first concrete child.\n"
            f"# =====================================================\n"
            f"TITLE:\n"
            f"  - \"4C umbrella reference for {physics}\"\n"
        )

    def _generate_inline(self, physics: str, variant: str, params: dict) -> str:
        """Generate self-contained input with inline mesh (no external files)."""
        from backends.fourc.inline_mesh import (
            matched_poisson_input, matched_heat_input,
            matched_elasticity_input, matched_poisson_3d_input,
            matched_l_domain_poisson_input,
            matched_heat_transient_input,
            matched_elasticity_genalpha_input,
            matched_elasticity_3d_nonlinear_input,
            matched_level_set_advection_input,
            matched_ale_2d_input,
            matched_nernst_planck_3d_input,
            matched_low_mach_heated_channel_input,
            matched_porofluid_single_phase_3d_input,
            matched_tsi_monolithic_3d_input,
            matched_beam_cantilever_static_input,
            matched_beam_cantilever_dynamic_input,
            matched_thermo_2d_input,
            matched_thermo_3d_input,
            matched_lubrication_slider_bearing_input,
            matched_mixture_3d_input,
            matched_constraint_3d_input,
            matched_membrane_2d_input,
            matched_shell_3d_input,
            matched_cardiovascular0d_windkessel_input,
        )
        key = f"{physics}_{variant}"

        def _elasticity(p):
            return matched_elasticity_input(
                nx=p.get("nx", 40), ny=p.get("ny", 4),
                E=p.get("E", 1000.0), nu=p.get("nu", 0.3),
                lx=p.get("lx", 10.0), ly=p.get("ly", 1.0))

        inline_generators = {
            "poisson_2d": lambda p: matched_poisson_input(
                nx=p.get("nx", 32), ny=p.get("ny", 32)),
            "poisson_poisson_2d": lambda p: matched_poisson_input(
                nx=p.get("nx", 32), ny=p.get("ny", 32)),
            # scalar_transport is the catalog umbrella for the same
            # physics — route its concrete variants to the proven
            # matched inputs instead of the placeholder generator
            # templates (probe 2026-06-12: those abort in 4C's
            # MatchTree with un-substituted <...> placeholders).
            "scalar_transport_poisson_2d": lambda p: matched_poisson_input(
                nx=p.get("nx", 32), ny=p.get("ny", 32)),
            "heat_2d": lambda p: matched_heat_input(
                nx=p.get("nx", 32), ny=p.get("ny", 32),
                T_left=p.get("T_left", 100.0), T_right=p.get("T_right", 0.0)),
            "heat_heat_2d": lambda p: matched_heat_input(
                nx=p.get("nx", 32), ny=p.get("ny", 32),
                T_left=p.get("T_left", 100.0), T_right=p.get("T_right", 0.0)),
            "poisson_heat_2d": lambda p: matched_heat_input(
                nx=p.get("nx", 32), ny=p.get("ny", 32),
                T_left=p.get("T_left", 100.0),
                T_right=p.get("T_right", 0.0)),
            "scalar_transport_heat_transient_2d":
                lambda p: matched_heat_transient_input(
                    nx=p.get("nx", 16), ny=p.get("ny", 16),
                    T_left=p.get("T_left", 100.0),
                    T_right=p.get("T_right", 0.0),
                    numstep=p.get("numstep", 10),
                    timestep=p.get("timestep", 0.01)),
            "heat_heat_transient_2d":
                lambda p: matched_heat_transient_input(
                    nx=p.get("nx", 16), ny=p.get("ny", 16),
                    T_left=p.get("T_left", 100.0),
                    T_right=p.get("T_right", 0.0),
                    numstep=p.get("numstep", 10),
                    timestep=p.get("timestep", 0.01)),
            "linear_elasticity_linear_2d": _elasticity,
            "linear_elasticity_2d": _elasticity,
            # solid_mechanics is the structural umbrella physics;
            # its linear_2d variant is the same cantilever the
            # linear_elasticity row uses (probe 2026-06-12).
            "solid_mechanics_linear_2d": _elasticity,
            "solid_mechanics_nonlinear_3d":
                lambda p: matched_elasticity_3d_nonlinear_input(
                    n=p.get("n", 4),
                    E=p.get("E", 1000.0), nu=p.get("nu", 0.3)),
            "structural_dynamics_genalpha_2d":
                lambda p: matched_elasticity_genalpha_input(
                    nx=p.get("nx", 20), ny=p.get("ny", 4),
                    E=p.get("E", 1000.0), nu=p.get("nu", 0.3),
                    dens=p.get("dens", 1.0),
                    numstep=p.get("numstep", 10),
                    timestep=p.get("timestep", 0.05)),
            # low_mach/heated_channel_2d fell through to the generator
            # template with <placeholder> scalars + an external Exodus
            # mesh (probe 2026-06-12: MatchTree abort). Route to the
            # self-contained inline heated-channel Loma input.
            "low_mach_heated_channel_2d":
                lambda p: matched_low_mach_heated_channel_input(
                    nx=min(int(p.get("nx", 32)), 64),
                    ny=min(int(p.get("ny", 8)), 32),
                    u_max=p.get("u_max", 0.3),
                    T_in=p.get("T_in", 293.0),
                    T_wall=p.get("T_wall", 350.0),
                    numstep=p.get("numstep", 5),
                    timestep=p.get("timestep", 0.1)),
            "poisson_3d": lambda p: matched_poisson_3d_input(n=p.get("n", 8)),
            "poisson_poisson_3d": lambda p: matched_poisson_3d_input(n=p.get("n", 8)),
            "poisson_l_domain": lambda p: matched_l_domain_poisson_input(
                n=p.get("n", 16)),
            # electrochemistry/nernst_planck_3d previously fell through
            # to the generator template with <placeholder> scalars + an
            # external Exodus mesh reference (probe 2026-06-12:
            # MatchTree abort). Route to the self-contained inline-mesh
            # Nernst-Planck input. Resolution uses "n" (not nx/ny/nz)
            # so the probe's nz=16 cannot inflate the 3-species
            # nonlinear 3D solve.
            "electrochemistry_nernst_planck_3d":
                lambda p: matched_nernst_planck_3d_input(
                    n=p.get("n", 4),
                    c_left=p.get("c_left", 2.0),
                    c_right=p.get("c_right", 1.0),
                    d_cation=p.get("d_cation", 2.0),
                    d_anion=p.get("d_anion", 1.0),
                    numstep=p.get("numstep", 10),
                    timestep=p.get("dt", 0.001)),
            # ale/ale_2d previously fell through to the generator
            # template with <placeholder> scalars + external Exodus
            # mesh (probe 2026-06-12: MatchTree abort). Inline 2D
            # mesh-motion problem instead.
            "ale_ale_2d": lambda p: matched_ale_2d_input(
                nx=min(int(p.get("nx", 16)), 32),
                ny=min(int(p.get("ny", 16)), 32),
                E=p.get("E", 1.0), nu=p.get("nu", 0.3),
                dens=p.get("rho", 1.0),
                numstep=max(1, round(p.get("T_end", 0.01)
                                     / p.get("dt", 0.001))),
                timestep=p.get("dt", 0.001)),
            # level_set/advection_2d previously fell through to the
            # placeholder generator template (literal <...> scalars +
            # external Exodus mesh → 4C MatchTree abort, probe
            # 2026-06-12). Route to the self-contained inline input.
            "level_set_advection_2d":
                lambda p: matched_level_set_advection_input(
                    nx=p.get("nx", 16), ny=p.get("ny", 16),
                    numstep=p.get("numstep", 10),
                    timestep=p.get("timestep", 0.01),
                    radius=p.get("radius", 0.25)),
            # porous_media/single_phase_3d previously used the generator
            # template with "TRANSPORT ELEMENTS" + a "TYPE
            # PoroFluidMultiPhase" element suffix that 4C's input
            # matcher rejects (probe 2026-06-12: MPI_Abort). Route to
            # the corpus-matched inline input (FLUID ELEMENTS /
            # POROFLUIDMULTIPHASE HEX8 ... MAT 1). Resolution uses "n"
            # (not nx/ny/nz) so the probe's nz=16 cannot inflate the
            # 3D mesh.
            "porous_media_single_phase_3d":
                lambda p: matched_porofluid_single_phase_3d_input(
                    n=p.get("n", 4),
                    permeability=p.get("kappa", 1.0),
                    viscosity=p.get("mu", 0.01),
                    density=p.get("rho", 1.0),
                    numstep=p.get("numstep", 10),
                    timestep=p.get("timestep", 0.01)),
            # tsi/monolithic_3d previously fell through to the generator
            # template with <placeholder> scalars + external Exodus mesh
            # (probe 2026-06-12: MatchTree abort). Route to the inline
            # SOLIDSCATRA cube with genuinely MONOLITHIC two-way coupling
            # (COUPALGO tsi_monolithic, merged TSI block matrix +
            # UMFPACK). Mesh capped at 8^3: the probe passes nx=ny=nz=16
            # and a 16^3 monolithic SOLIDSCATRA solve is too big.
            "tsi_monolithic_3d":
                lambda p: matched_tsi_monolithic_3d_input(
                    nx=min(int(p.get("nx", 4)), 8),
                    ny=min(int(p.get("ny", 4)), 8),
                    nz=min(int(p.get("nz", 4)), 8),
                    E=p.get("E", 200e3), nu=p.get("nu", 0.3),
                    density=p.get("rho", 1.0),
                    conductivity=p.get("kappa", 1.0),
                    numstep=max(1, round(p.get("T_end", 0.01)
                                         / p.get("dt", 0.001))),
                    timestep=p.get("dt", 0.001)),
            # beams/cantilever_* previously fell through to the
            # generator templates with <placeholder> scalars (probe
            # 2026-06-12: MatchTree abort). Route to corpus-matched
            # inline BEAM3R cantilevers; the tip load scales with E,
            # so the probe's E=1000 override converges like the
            # default E=1e7.
            "beams_cantilever_static":
                lambda p: matched_beam_cantilever_static_input(
                    n_elem=p.get("n_elem", 10),
                    length=p.get("length", 10.0),
                    radius=p.get("radius", 0.1),
                    E=p.get("E", 1.0e7), nu=p.get("nu", 0.3),
                    load_factor=p.get("load_factor", 1.0),
                    numstep=p.get("numstep", 5)),
            "beams_cantilever_dynamic":
                lambda p: matched_beam_cantilever_dynamic_input(
                    n_elem=p.get("n_elem", 10),
                    length=p.get("length", 10.0),
                    radius=p.get("radius", 0.1),
                    E=p.get("E", 1.0e7), nu=p.get("nu", 0.3),
                    dens=p.get("rho", 1.0),
                    moment_factor=p.get("moment_factor", 0.2),
                    numstep=p.get("numstep", 5),
                    timestep=p.get("timestep", 0.01)),
            # thermo/thermo_2d + thermo/thermo_3d previously fell
            # through to a one-line comment template ("# Thermal
            # template ...") that is not even a YAML dict, so
            # validate_input failed before the run stage (probe
            # 2026-06-12). Route to genuine PROBLEMTYPE "Thermo"
            # inline-mesh inputs (THERMO QUAD4/HEX8 + MAT_Fourier).
            # The 3D row keys resolution off "n" (NOT nx/ny/nz) so
            # the probe's nz=16 cannot inflate the cube mesh.
            "thermo_thermo_2d": lambda p: matched_thermo_2d_input(
                nx=min(int(p.get("nx", 16)), 32),
                ny=min(int(p.get("ny", 16)), 32),
                T_left=p.get("T_left", 100.0),
                T_right=p.get("T_right", 0.0),
                conductivity=p.get("kappa", 1.0)),
            "thermo_thermo_3d": lambda p: matched_thermo_3d_input(
                n=min(int(p.get("n", 6)), 8),
                T_left=p.get("T_left", 100.0),
                T_right=p.get("T_right", 0.0),
                conductivity=p.get("kappa", 1.0),
                capacity=p.get("capacity", 1.0),
                numstep=max(1, min(20, round(p.get("T_end", 0.5)
                                             / p.get("dt", 0.1)))),
                timestep=p.get("dt", 0.1)),
            # Lubrication (Reynolds eq.) slider bearing: the placeholder
            # generator template emitted literal <...> scalars + an
            # external Exodus mesh, aborting 4C's MatchTree (probe
            # 2026-06-12). Route to the inline-mesh port of the corpus
            # case lubrication_sb_2d.4C.yaml (PURE_LUB, LUBRICATION
            # QUAD4, MAT_lubrication). Mesh capped small for < 30 s.
            "lubrication_slider_bearing_2d":
                lambda p: matched_lubrication_slider_bearing_input(
                    nx=min(int(p.get("nx", 16)), 32),
                    ny=min(int(p.get("ny", 1)), 4)),
            # mixture/mixture_3d previously returned a one-line comment
            # ("# Mixture template ...") — not a YAML dict, so
            # validate_input failed with "Input is not a YAML
            # dictionary" before the run stage (probe 2026-06-12). Route
            # to a self-contained inline HEX8 cube whose material is the
            # 4C Mixture toolbox (MAT_Mixture -> MIX_Rule_Simple ->
            # MIX_Constituent_ElastHyper -> ELAST_CoupLogNeoHooke).
            # Resolution keyed off "n" (NOT nx/ny/nz) so the probe's
            # nz=16 cannot inflate the cube.
            "mixture_mixture_3d": lambda p: matched_mixture_3d_input(
                n=min(int(p.get("n", 4)), 6),
                E=p.get("E", 1000.0), nu=p.get("nu", 0.3),
                density=p.get("rho", 0.1)),
            # constraint/constraint_3d previously returned a one-line
            # comment ("# Constraint template ...") — not a YAML dict, so
            # validate_input failed with "Input is not a YAML
            # dictionary" before the run stage (probe 2026-06-12). Route
            # to a self-contained inline HEX8 cube with a real
            # DESIGN POINT COUPLING CONDITION (multi-point coupling) that
            # ties the loaded face's transverse DOFs together.
            # Resolution keyed off "n" (NOT nx/ny/nz) so the probe's
            # nz=16 cannot inflate the cube.
            "constraint_constraint_3d":
                lambda p: matched_constraint_3d_input(
                    n=min(int(p.get("n", 4)), 6),
                    E=p.get("E", 1000.0), nu=p.get("nu", 0.3)),
            # membrane/membrane_2d + shell/shell_3d previously returned a
            # one-line comment from generators/membrane.py & shell.py
            # ("# Membrane template ...", "# Shell template ...") — not a
            # YAML dict, so validate_input failed with "Input is not a
            # YAML dictionary" before the run stage (probe 2026-06-12).
            # Route to self-contained inline structural inputs: a flat
            # MEMBRANE4 QUAD4 patch under a prescribed uniaxial stretch
            # (membranes are singular without prestress / full Dirichlet),
            # and a flat SHELL7P QUAD4 clamped cantilever under transverse
            # orthopressure. nx,ny capped <=16 for sub-30 s runtime; the
            # shell load scales with E so Newton converges at probe E.
            "membrane_membrane_2d":
                lambda p: matched_membrane_2d_input(
                    nx=min(int(p.get("nx", 8)), 16),
                    ny=min(int(p.get("ny", 8)), 16),
                    E=p.get("E", 1000.0), nu=p.get("nu", 0.3)),
            "shell_shell_3d":
                lambda p: matched_shell_3d_input(
                    nx=min(int(p.get("nx", 8)), 16),
                    ny=min(int(p.get("ny", 4)), 16),
                    E=p.get("E", 1000.0), nu=p.get("nu", 0.3)),
            # cardiovascular0d/windkessel_3d previously fell through to a
            # one-line comment generator template that is not even a YAML
            # dict, so validate_input failed before the run stage (probe
            # 2026-06-12). Route to the corpus-matched inline 0D-3D input:
            # a structural HEX8 cube coupled to a 4-element Windkessel via
            # DESIGN SURF CARDIOVASCULAR 0D conditions. Resolution keys off
            # "n" (NOT nx/ny/nz) so the probe's nz=16 cannot inflate the
            # monolithic 0D-3D solve; n is capped <=4 inside the helper.
            "cardiovascular0d_windkessel_3d":
                lambda p: matched_cardiovascular0d_windkessel_input(
                    n=min(int(p.get("n", 2)), 4),
                    E=p.get("E", 10.0), nu=p.get("nu", 0.3),
                    density=p.get("rho", 2e-6),
                    numstep=max(1, min(10, round(p.get("T_end", 0.3)
                                                 / p.get("dt", 0.1)))),
                    timestep=p.get("dt", 0.1)),
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
        return sorted_by_step(results)


def register():
    register_backend(
        FourcBackend(),
        aliases=["4c", "4C", "fourc", "four_c"],
    )
