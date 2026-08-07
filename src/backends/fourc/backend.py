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


_FOURC_IDENT_CACHE: dict[str, tuple[bool, str]] = {}


def _identifies_as_fourc(binary) -> tuple[bool, str]:
    """Does this executable actually identify itself as 4C?

    Cheap and conservative. 4C prints its usage to stderr and exits non-zero
    when run with no arguments, and that text names the program — so the check
    is "run it with no arguments and look for 4C's own vocabulary". Anything
    that produces neither is not 4C.

    Deliberately fails OPEN on an inability to look (timeout, permission,
    OSError): a check that cannot run must not condemn a working install. It
    fails CLOSED only when the program ran and said something that is not 4C,
    which is the case that matters — `/bin/true` runs, says nothing, and used to
    be reported as an available solver.

    Cached per path, because `check_availability` is called repeatedly by
    `discover` and every knowledge surface, and this spawns a process.
    """
    import subprocess

    key = str(binary)
    if key in _FOURC_IDENT_CACHE:
        return _FOURC_IDENT_CACHE[key]

    verdict: tuple[bool, str]
    try:
        # stdin MUST be closed. An audit pointed this at `/bin/cat`, which
        # consumed the PARENT's entire stdin and made the verdict a function of
        # that text; pointed at a real solver it hung the full timeout and then
        # failed open. Under an MCP stdio server the parent's stdin is the
        # JSON-RPC stream, so an identity probe could eat the protocol.
        r = subprocess.run([key], capture_output=True, timeout=20,
                           stdin=subprocess.DEVNULL)
        blob = (r.stdout + r.stderr).decode("utf-8", errors="replace").lower()
        # 4C's banner names itself in full. The first version matched "4c",
        # "dat file" and "input file", all of which are far too weak: "4c" is
        # two characters, so `/bin/pwd` and `/bin/ls` pass whenever the working
        # directory contains it, and `/usr/bin/env` passes whenever ANY
        # environment variable does — which is the normal state for a 4C user,
        # since LD_LIBRARY_PATH=/opt/4C-dependencies/lib contains it. And
        # "input file" passes `/usr/bin/gcc`, whose no-argument output is "no
        # input files". Four measured false positives from three sloppy tokens.
        # What 4C ACTUALLY emits with no arguments, measured rather than
        # assumed: it does not print a banner at all. It throws
        # `FourC::Core::Exception` with "Please provide both <input> and
        # <output> arguments." and aborts. An audit recommended matching the
        # project's full name, "Comprehensive Computational Community Code" —
        # that appears in the banner on a successful start, NOT on this path, so
        # matching it refused the real binary. Both the first token set and its
        # proposed replacement were wrong in opposite directions, which is why
        # this now uses strings taken from the observed output.
        #
        # `fourc::` is the C++ namespace and is specific enough on its own: no
        # ordinary executable emits it. The argument message is a second,
        # independent witness.
        markers = ("fourc::", "provide both <input> and <output>", "lib4c.so")
        if not blob.strip():
            verdict = (False, "produced no output at all when run with no "
                              "arguments; 4C aborts with a named exception")
        elif any(t in blob for t in markers):
            verdict = (True, "identified itself")
        else:
            verdict = (False, "its output carries none of 4C's own markers: "
                              + " ".join(blob.split())[:120])
    except (subprocess.TimeoutExpired, OSError) as exc:
        # Could not look — do not accuse a possibly-working install.
        # ValueError is deliberately NOT caught here: UnicodeDecodeError is a
        # ValueError, so catching it sent every non-UTF-8 binary down the
        # fail-open path and ACCEPTED it (`/usr/bin/gzip` passed). Decoding is
        # now explicit with errors="replace", so there is nothing left for a
        # ValueError to mean except a real bug, which should surface.
        verdict = (True, f"identity not checked ({type(exc).__name__})")

    _FOURC_IDENT_CACHE[key] = verdict
    return verdict


def _find_fourc_binary() -> Optional[Path]:
    """Locate the 4C binary."""
    # An explicit override that does not resolve must not fall through to the
    # search path — see the FEBio equivalent for what that costs. Kept as a
    # warning-and-None here rather than an exception, because this finder is
    # called from more places than FEBio's and a raise would change behaviour
    # in paths I have not tested.
    env_path = os.environ.get("FOURC_BINARY")
    if env_path and not Path(env_path).is_file():
        logger.warning(
            "FOURC_BINARY is set to %r, which is not a file; NOT falling back "
            "to the search path, because the binary OASiS tests must be the one "
            "you named", env_path)
        return None
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

        # Confirm the binary IS 4C, not merely that a file exists and is
        # executable. `FOURC_BINARY=/bin/true` used to report
        # "available — 4C at /bin/true", so a stale path, a wrong build, or a
        # same-named program on PATH was indistinguishable from a working
        # install. An agent consults `discover`, believes it, and every run then
        # fails for a reason the availability report has already ruled out —
        # which is the worst place to be wrong.
        ident_ok, ident_why = _identifies_as_fourc(binary)
        if not ident_ok:
            return BackendStatus.MISCONFIGURED, (
                f"the binary at {binary} does not identify itself as 4C "
                f"({ident_why}). Point FOURC_BINARY at a real 4C build; a file "
                f"that merely exists and is executable is not a solver.")
        return BackendStatus.AVAILABLE, f"4C at {binary}"

    def input_format(self) -> InputFormat:
        return InputFormat.YAML

    def get_version(self) -> Optional[str]:
        binary = _find_fourc_binary()
        if not binary:
            return None
        import subprocess
        try:
            r = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL)
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
            # heat_transient_2d was reachable through generate_input()
            # and produced a running One-Step-Theta deck, but was absent
            # from template_variants, so no tool could select it and no
            # caller could discover it. Registered 2026-08-03 after
            # executing both variants on the installed 4C (both rc=0).
            PhysicsCapability("heat", "Heat conduction", [2, 3],
                              ["QUAD4", "HEX8"],
                              ["heat_2d", "heat_transient_2d"]),
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
            # inline_penalty_3d FIRST: it is the only contact variant
            # that is self-contained (inline nodes + elements) and runs
            # without FOURC_ROOT or an external Exodus mesh. penalty_3d
            # is the tutorial/format-template route and needs both.
            PhysicsCapability("contact", "Contact mechanics", [3],
                              ["HEX8"],
                              ["inline_penalty_3d", "penalty_3d"]),
            PhysicsCapability("particle_pd", "Peridynamics (bond-based)", [2],
                              ["particle"],
                              ["plate_2d", "impact_2d"]),
            # Variants renamed 2026-08-07 to match the deck that now ships.
            # "poiseuille_2d" served a hydrostatic column and
            # "normal_impact_1d" a 3-D settling pack; a name that describes a
            # different experiment than the deck is the same class of defect
            # as a wrong key, and harder to notice because nothing errors.
            PhysicsCapability("particle_sph", "Smoothed particle hydrodynamics", [2],
                              ["particle"],
                              ["hydrostatic_2d", "dam_break_2d"]),
            PhysicsCapability("particle_dem",
                              "Discrete element method (granular contact, "
                              "friction, rolling, adhesion, walls)",
                              [1, 2, 3], ["particle"],
                              ["settling_3d"]),
            PhysicsCapability("tsi", "Thermo-structure interaction", [2, 3],
                              ["SOLIDSCATRA HEX8"],
                              ["monolithic_3d", "oneway_3d",
                               "plane_strain_2d"]),
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
            # single_phase_3d listed FIRST because it is the only one of
            # the three that generates a complete deck and runs (rc=0 on
            # the installed 4C, 2026-08-03); terzaghi_2d and
            # consolidation_3d fall through to the ~1 kB reference-stub
            # template and are documentation, not runnable input.
            PhysicsCapability("porous_media", "Poroelasticity (Biot/mixture theory, consolidation)", [2, 3],
                              ["WALLQ4PORO", "WALLQ9PORO", "SOLIDH8PORO", "SOLIDT4PORO"],
                              ["single_phase_3d", "terzaghi_2d", "consolidation_3d"]),
            # New physics
            PhysicsCapability("membrane", "Membrane elements (inflatable, fabric, tissue)", [2, 3],
                              ["MEMBRANE TRI3", "MEMBRANE QUAD4"], ["membrane_2d"]),
            PhysicsCapability("shell", "Shell elements (Kirchhoff-Love, Reissner-Mindlin)", [3],
                              ["SHELL REISSNER QUAD4", "SHELL KIRCHHOFF TRI3", "SOLIDSHELL HEX8"], ["shell_3d"]),
            PhysicsCapability("thermo", "Pure thermal analysis (standalone heat conduction)", [2, 3],
                              ["THERMO QUAD4", "THERMO HEX8"], ["thermo_2d", "thermo_3d"]),
            PhysicsCapability("thermo_transient_mms",
                              "Transient thermal MMS on a fixed mesh for "
                              "TEMPORAL-order dt-halving studies "
                              "(One-Step-Theta: order 2 at theta=0.5, "
                              "order 1 at theta=1)", [2],
                              ["THERMO QUAD4"], ["temporal_mms_2d"]),
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

        # Executed deck templates come first. Every entry in
        # backends.fourc.decks was run on the installed binary and exited 0,
        # which is a stronger guarantee than any other branch below offers,
        # and the pairs it covers are exactly the ones that used to fall
        # through to the "Not a runnable input" stub.
        from backends.fourc import decks as _decks
        _deck = _decks.render(physics, variant)
        if _deck is not None:
            return _deck

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
        }
        # A stub must never shadow an executed deck. generate_input already
        # consults backends.fourc.decks first, so a pair present in both would
        # be dead code that silently rots out of date — and the whole point of
        # the deck catalog is that its content is the content that ran.
        from backends.fourc import decks as _decks
        if _decks.get(physics, variant) is not None:
            raise AssertionError(
                f"stub catalog still carries {physics}/{variant}, which now "
                f"has an executed deck in backends.fourc.decks")
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
            matched_tsi_oneway_input,
            matched_tsi_plane_strain_input,
            matched_beam_cantilever_static_input,
            matched_beam_cantilever_dynamic_input,
            matched_thermo_2d_input,
            matched_thermo_3d_input,
            matched_thermo_transient_mms_input,
            matched_lubrication_slider_bearing_input,
            matched_mixture_3d_input,
            matched_constraint_3d_input,
            matched_membrane_2d_input,
            matched_shell_3d_input,
            matched_cardiovascular0d_windkessel_input,
            matched_fluid_cavity_input,
            matched_fluid_channel_input,
            matched_reduced_airways_input,
            matched_contact_3d_input,
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
            # linear_elasticity/nonlinear_3d previously fell through to
            # the generator template with <placeholder> scalars + an
            # external Exodus mesh (probe 2026-06-26: MatchTree abort).
            # It is the same finite-strain HEX8 cube the solid_mechanics
            # umbrella uses; route it to the proven inline input.
            "linear_elasticity_nonlinear_3d":
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
            # tsi/plane_strain_2d: pseudo-2D thin-slab route for 2D
            # plane-strain thermo-mechanics. 4C has NO 2D TSI elements
            # (module solid_scatra_3D_ele; every TSI corpus test is 3D)
            # and the 2D structural eletypes both dead-end with the
            # thermo material on current builds (WALL QUAD4 -> "Invalid
            # type of material law for wall element"; SOLID QUAD4 ->
            # "Element 'SOLID' does not seem to know cell type
            # 'quad4'"). One SOLIDSCATRA HEX8
            # layer with u_z fixed everywhere is exact plane strain;
            # temp_expr imposes a (partner-computed) temperature field.
            "tsi_plane_strain_2d":
                lambda p: matched_tsi_plane_strain_input(
                    nx=min(int(p.get("nx", 16)), 64),
                    ny=min(int(p.get("ny", 4)), 32),
                    lx=p.get("lx", 2.0), ly=p.get("ly", 0.25),
                    thickness=p.get("thickness"),
                    E=p.get("E", 200e9), nu=p.get("nu", 0.3),
                    alpha=p.get("alpha", 12e-6),
                    T_ref=p.get("T_ref", 293.0),
                    T_left=p.get("T_left", 293.0),
                    T_right=p.get("T_right", 450.0),
                    temp_expr=p.get("temp_expr"),
                    density=p.get("rho", 7850.0),
                    conductivity=p.get("kappa", 1.0)),
            # tsi/oneway_3d: the corrected one-way (thermo->structure)
            # heated-beam input. Existed in inline_mesh since the
            # coupled_solve era but was never exposed as a variant —
            # and carried the silent COUPVARIABLE default bug (ran
            # rc=0, zero displacement) until that was fixed.
            "tsi_oneway_3d":
                lambda p: matched_tsi_oneway_input(
                    nx=min(int(p.get("nx", 4)), 8),
                    ny=min(int(p.get("ny", 4)), 8),
                    nz=min(int(p.get("nz", 4)), 8),
                    E=p.get("E", 200e3), nu=p.get("nu", 0.3),
                    alpha=p.get("alpha", 12e-6),
                    T_left=p.get("T_left", 100.0),
                    T_right=p.get("T_right", 0.0),
                    T_ref=p.get("T_ref", 0.0),
                    density=p.get("rho", 1.0),
                    conductivity=p.get("kappa", 1.0)),
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
            # contact/inline_penalty_3d: the ONLY self-contained contact
            # variant. contact/penalty_3d needs FOURC_ROOT and an external
            # Exodus mesh; without them it falls through to the
            # <placeholder> format template, which cannot run (4C's
            # MatchTree rejects "TIMESTEP: <load_step_size>"). This one
            # carries its own nodes and elements and completes every load
            # step on the deployed 4C (verified by execution 2026-08-03).
            "contact_inline_penalty_3d":
                lambda p: matched_contact_3d_input(
                    nx=int(p.get("nx", 2)), ny=int(p.get("ny", 2)),
                    nz=int(p.get("nz", 2)),
                    gap=p.get("gap", 0.1),
                    indent=p.get("indent", 0.3),
                    penalty=p.get("penalty", 1.0e4),
                    E=p.get("E", 1000.0), nu=p.get("nu", 0.3),
                    n_steps=int(p.get("n_steps", 10))),
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
            # thermo_transient_mms/temporal_mms_2d: unsteady-heat MMS
            # family graded on TEMPORAL convergence order. Fixed fine
            # mesh keyed off "n" (capped in the inline builder),
            # dt-halving to the same T_end, so the only thing varying
            # is dt. One-Step-Theta is 2nd-order in time at theta=0.5
            # and 1st-order at theta=1; grade theta=0.5 from Richardson
            # differences of consecutive-dt solutions rather than from
            # an error-vs-exact table, which saturates at the fixed
            # mesh's spatial floor. The volumetric
            # MMS source goes through the PLAIN "DESIGN SURF NEUMANN
            # CONDITIONS" — the THERMO-prefixed Neumann sections are
            # silently ignored in standalone Thermo (see the
            # thermo_transient_mms generator's pitfalls).
            "thermo_transient_mms_temporal_mms_2d":
                lambda p: matched_thermo_transient_mms_input(
                    n=int(p.get("n", 48)),
                    lx=p.get("lx", 1.0), ly=p.get("ly", 1.0),
                    kappa=p.get("kappa", 1.0),
                    rho=p.get("rho", 1.0), c=p.get("c", 1.0),
                    theta=p.get("theta", 0.5),
                    dt=p.get("dt", 0.02),
                    t_end=p.get("T_end", p.get("t_end", 0.4)),
                    temp_offset=p.get("temp_offset", 1.0),
                    amp=p.get("amp", 1.0),
                    grad_amp=p.get("grad_amp", 0.5),
                    mode_x=int(p.get("mode_x", 1)),
                    mode_y=int(p.get("mode_y", 1)),
                    omega=p.get("omega", 6.283185307179586),
                    time_profile=p.get("time_profile", "cos")),
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
            # fluid/{channel_2d,cavity_2d} previously fell through to the
            # generator template with <placeholder> scalars + an external
            # Exodus mesh (FILE: "channel_2d.e", never produced) — 4C
            # aborted in MatchTree (probe 2026-06-26). Route to the
            # self-contained inline incompressible Navier-Stokes inputs
            # (FLUID QUAD4 + MAT_fluid + Np_Gen_Alpha + UMFPACK), based on
            # the corpus case f2_channel20x20_drt_weak.4C.yaml. nx,ny
            # capped for a sub-30 s monolithic solve.
            "fluid_cavity_2d": lambda p: matched_fluid_cavity_input(
                nx=min(int(p.get("nx", 16)), 32),
                ny=min(int(p.get("ny", 16)), 32),
                u_lid=p.get("u_lid", p.get("u_max", 1.0)),
                viscosity=p.get("mu", p.get("viscosity", 0.01)),
                density=p.get("rho", p.get("density", 1.0)),
                numstep=p.get("numstep", 10),
                timestep=p.get("dt", p.get("timestep", 0.1))),
            "fluid_channel_2d": lambda p: matched_fluid_channel_input(
                nx=min(int(p.get("nx", 24)), 48),
                ny=min(int(p.get("ny", 8)), 24),
                u_max=p.get("u_max", 1.0),
                viscosity=p.get("mu", p.get("viscosity", 0.01)),
                density=p.get("rho", p.get("density", 1.0)),
                numstep=p.get("numstep", 10),
                timestep=p.get("dt", p.get("timestep", 0.1))),
            # reduced_airways/airways_1d previously returned a reference
            # stub (no MATERIALS — validate_input flagged it not
            # runnable). The corpus case red_airway_3airway_2acinus_
            # awacinter.4C.yaml is small and fully self-contained (6
            # nodes, 5 elements, inline mesh, UMFPACK), so it ports to a
            # genuinely runnable inline input. numstep capped for a
            # sub-30 s solve.
            "reduced_airways_airways_1d":
                lambda p: matched_reduced_airways_input(
                    peak_pressure=p.get("peak_pressure", 30.0),
                    numstep=min(int(p.get("numstep", 200)), 2000),
                    period=p.get("period", 100.0)),
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
            # Give the RELATIVE location as well as the resolved one. The
            # absolute path is what this host needs to run the deck, but it is
            # also what an agent copies — and a served deck naming
            # `/home/<someone>/4C/tests/...` is a dead end on every other
            # machine. These meshes ship WITH 4C, so anyone who has 4C has them;
            # only the prefix differs. Naming FOURC_ROOT turns an unusable
            # absolute path into a locatable one.
            content = (f"# MESH_FILE: {mesh_path}\n"
                       f"# MESH_FILE_RELATIVE: $FOURC_ROOT/tests/{mesh_rel}"
                       f"  (ships with 4C; the absolute path above is this "
                       f"host's — resolve it against your own FOURC_ROOT)\n"
                       + content)
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
                try:
                    rel = mesh_path.relative_to(FOURC_ROOT)
                except ValueError:
                    rel = mesh_path.name
                content = (f"# MESH_FILE: {mesh_path}\n"
                           f"# MESH_FILE_RELATIVE: $FOURC_ROOT/{rel}"
                           f"  (the absolute path above is this host's — "
                           f"resolve it against your own FOURC_ROOT)\n"
                           + content)
                break
        return content

    def validate_input(self, content: str) -> list[str]:
        import re
        import yaml
        errors = []

        # ── Hard rejections that would make 4C abort ──────────────────
        # 1. Un-substituted <...> placeholders (e.g. <number_of_steps>,
        #    <dynamic_viscosity>). 4C's input matcher aborts on these.
        #    Skip matches inside comment lines so prose like "<...>" in a
        #    reference stub is not flagged; scan only non-comment text.
        non_comment = "\n".join(
            ln for ln in content.splitlines()
            if not ln.lstrip().startswith("#"))
        placeholders = re.findall(
            r"<[A-Za-z_][A-Za-z0-9_ ]*>", non_comment)
        if placeholders:
            uniq = sorted(set(placeholders))
            errors.append(
                "Un-substituted placeholder(s) found "
                f"({', '.join(uniq[:5])}): the deck is not runnable — "
                "every <...> must be replaced with a concrete value.")

        # 2. External mesh FILE: references (e.g. FILE: \"channel_2d.e\").
        #    These abort 4C unless the mesh travels with the deck. A
        #    legitimate tutorial deck carries a leading
        #    '# MESH_FILE: <path>' header that run() copies into the work
        #    dir; allow the FILE: ref only when that header resolves to an
        #    existing file.
        mesh_refs = re.findall(
            r'FILE:\s*["\']?([^\s"\']+\.(?:e|exo|dat|bin))\b',
            non_comment)
        if mesh_refs:
            header_path = None
            first_line = content.splitlines()[0] if content else ""
            if first_line.startswith("# MESH_FILE: "):
                header_path = Path(
                    first_line.split(": ", 1)[1].strip())
            if header_path is None or not header_path.exists():
                errors.append(
                    "External mesh FILE: reference(s) "
                    f"({', '.join(sorted(set(mesh_refs))[:3])}) with no "
                    "bundled/resolvable mesh — the deck references a mesh "
                    "file that is not produced. Use an inline NODE COORDS "
                    "mesh or a '# MESH_FILE:' header pointing at an "
                    "existing file.")

        # ── Structural YAML checks ────────────────────────────────────
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
