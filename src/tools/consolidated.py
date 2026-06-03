"""
Consolidated MCP tools for the Open FEM Agent.

Reduces 48 tools → ~12 tools by combining related functionality.
Fewer tools = faster schema loading = faster agent response.
"""

import json
import os
import time
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Context
from core.backend import detect_template_language
from core.registry import get_backend, available_backends, all_backends

_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "simulation_outputs"
_COUPLING_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "coupling"
FOURC_ROOT = Path(os.environ.get("FOURC_ROOT", ""))
_jobs: dict = {}


async def _run_with_progress(ctx: Context, coro, message_prefix: str = "Running"):
    """Run a coroutine while sending periodic MCP progress keepalives.

    This prevents the MCP client from timing out on long-running simulations
    (DUNE JIT compilation, 4C FSI, deal.II builds can take minutes).
    """
    import asyncio

    task = asyncio.create_task(coro)
    elapsed = 0
    try:
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except asyncio.TimeoutError:
                elapsed += 5
                try:
                    await ctx.report_progress(
                        elapsed, total=None,
                        message=f"{message_prefix} ({elapsed}s elapsed)"
                    )
                except Exception:
                    pass  # progress reporting is best-effort
    except Exception:
        if not task.done():
            task.cancel()
        raise
    return task.result()


def _stub_template_tag(content: str, fmt: str) -> str:
    """Return a `" — ⚠ STUB"` marker if `content` looks like a
    placeholder template (a single comment line, or fewer than
    ~150 chars of non-comment body), otherwise empty.

    The catalog ships 9 fourc physics rows whose generators
    return only a one-line comment (`# Foo template — use ...`)
    because no full template has been written yet. Surfacing
    those as plain `## Template` sections in prepare_simulation
    output misleads the LLM: the heading promises a runnable
    template, but the body is a 50-80 char placeholder.

    Detection rule: strip every line that begins with `#`
    (YAML / Python comment) or is whitespace-only; if what
    remains is shorter than 150 chars, treat as a stub. The
    `fmt` argument tells us which comment character to honour
    — for the (rare) non-comment-character formats (`json`,
    `cpp`), we still apply the size heuristic but skip the
    comment-stripping step. (Audit 2026-06-02.)
    """
    if not isinstance(content, str):
        return ""
    if fmt in ("yaml", "yml", "python", "py"):
        non_comment_lines = [
            ln for ln in content.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        body = "\n".join(non_comment_lines)
    else:
        body = content
    if len(body.strip()) < 150:
        return " — ⚠ STUB (catalog placeholder — no full template yet)"
    return ""


_PHYSICS_SYNONYMS = {
    # ── Heat / thermal conduction ──────────────────────────────────
    # canonical key 'heat' exists in: fourc, fenics, ngsolve, kratos,
    # dealii, dune, skfem, febio (all 8 backends)
    "thermal": "heat",
    "conduction": "heat",
    "temperature": "heat",
    "heat_transfer": "heat",
    "heat_conduction": "heat",
    "heat_flow": "heat",
    "thermal_conduction": "heat",
    "thermal_diffusion": "heat",
    "fourier": "heat",
    # transient flavour (canonical: heat_transient OR time_dependent_heat)
    "unsteady_heat": "heat_transient",
    "transient_heat": "heat_transient",
    "time_heat": "heat_transient",
    "dynamic_thermal": "heat_transient",
    "transient_thermal": "heat_transient",
    "time_dependent_thermal": "time_dependent_heat",

    # ── Linear elasticity / small-strain mechanics ─────────────────
    "elasticity": "linear_elasticity",
    "structural": "linear_elasticity",
    "structural_mechanics": "linear_elasticity",
    "structural_2d": "linear_elasticity",
    "structural_3d": "linear_elasticity",
    "solid": "linear_elasticity",
    "solid_mechanics": "linear_elasticity",
    "mechanics": "linear_elasticity",
    "small_strain": "linear_elasticity",
    "hooke": "linear_elasticity",
    "hookean": "linear_elasticity",
    "linear_solid": "linear_elasticity",
    "plane_strain": "linear_elasticity",
    "plane_stress": "linear_elasticity",
    "elastic": "linear_elasticity",
    "fea": "linear_elasticity",
    "statics": "linear_elasticity",
    "elasticity_2d": "linear_elasticity",
    "elasticity_3d": "linear_elasticity",

    # ── Hyperelasticity / large-deformation solid ──────────────────
    "nonlinear_elasticity": "hyperelasticity",
    "large_deformation": "hyperelasticity",
    "large_strain": "hyperelasticity",
    "neo_hookean": "hyperelasticity",
    "neohookean": "hyperelasticity",
    "mooney_rivlin": "hyperelasticity",
    "ogden": "hyperelasticity",
    "finite_strain": "hyperelasticity",
    "finite_deformation": "hyperelasticity",
    "hyperelastic": "hyperelasticity",
    "hyperelastic_solid": "hyperelasticity",
    "geometric_nonlinearity": "hyperelasticity",
    "nonlinear_solid": "hyperelasticity",

    # ── Plasticity / elasto-plastic ────────────────────────────────
    "elasto_plasticity": "plasticity",
    "elastoplasticity": "plasticity",
    "elasto_plastic": "plasticity",
    "yield": "plasticity",
    "yielding": "plasticity",
    "mohr_coulomb": "plasticity",
    "drucker_prager": "plasticity",
    "von_mises": "plasticity",
    "j2_plasticity": "plasticity",
    "soil_plasticity": "plasticity",
    "metal_plasticity": "plasticity",
    "return_mapping": "plasticity",
    "plastic_flow": "plasticity",

    # ── Stokes (creeping / mixed) flow ─────────────────────────────
    "stokes_flow": "stokes",
    "creeping_flow": "stokes",
    "mixed_stokes": "stokes",
    "taylor_hood": "stokes",
    "low_reynolds": "stokes",

    # ── Navier-Stokes / CFD ────────────────────────────────────────
    "cfd": "navier_stokes",
    "flow": "navier_stokes",
    "fluid_dynamics": "navier_stokes",
    "ns": "navier_stokes",
    "incompressible": "navier_stokes",
    "incompressible_flow": "navier_stokes",
    "viscous_flow": "navier_stokes",
    "fluid_flow": "navier_stokes",
    "fluid_mechanics": "navier_stokes",
    "internal_flow": "navier_stokes",
    "channel_flow": "navier_stokes",
    "external_flow": "navier_stokes",
    "laminar_flow": "navier_stokes",
    # transient flavour
    "transient_ns": "time_dependent_ns",
    "unsteady_ns": "time_dependent_ns",
    "unsteady_navier_stokes": "time_dependent_ns",
    "vortex_shedding": "time_dependent_ns",

    # ── Maxwell / electromagnetism ─────────────────────────────────
    "magnetostatics": "maxwell",
    "electromagnetics": "maxwell",
    "em": "maxwell",
    "magnetic": "maxwell",
    "eddy_current": "maxwell",
    "eddy_current_problem": "maxwell",
    "nedelec": "maxwell",
    "electromagnetic": "maxwell",
    "h_curl": "maxwell",
    "electric_field": "maxwell",
    "magnetic_field": "maxwell",
    "electrostatics": "maxwell",
    "electrodynamics": "maxwell",

    # ── Helmholtz / time-harmonic acoustics ────────────────────────
    "acoustics": "helmholtz",
    "acoustic": "helmholtz",
    "sound": "helmholtz",
    "frequency_domain": "helmholtz",
    "time_harmonic": "helmholtz",
    "scattering": "helmholtz",

    # ── Wave equation (second-order, time-domain) ──────────────────
    "wave_equation": "wave",
    "second_order_wave": "wave",
    "elastic_wave": "wave",
    "transient_wave": "time_dependent_wave",
    "unsteady_wave": "time_dependent_wave",

    # ── Eigenvalue / modal analysis ────────────────────────────────
    "vibration": "eigenvalue",
    "modal": "eigenvalue",
    "frequencies": "eigenvalue",
    "modes": "eigenvalue",
    "natural_frequencies": "eigenvalue",
    "eigenmode": "eigenvalue",
    "eigenfrequency": "eigenvalue",
    "buckling": "eigenvalue",
    "linear_buckling": "eigenvalue",

    # ── Poisson / Laplace / scalar elliptic ────────────────────────
    "diffusion": "poisson",
    "laplace": "poisson",
    "scalar": "poisson",
    "scalar_pde": "poisson",
    "steady_diffusion": "poisson",
    "electrostatic_field": "poisson",
    "elliptic": "poisson",

    # ── Convection-diffusion / scalar transport ────────────────────
    "transport": "convection_diffusion",
    "advection": "convection_diffusion",
    "advection_diffusion": "convection_diffusion",
    "scalar_transport": "convection_diffusion",
    "mass_transport": "convection_diffusion",
    "contaminant_transport": "convection_diffusion",
    "cd": "convection_diffusion",

    # ── DG (discontinuous Galerkin) ────────────────────────────────
    "discontinuous_galerkin": "dg_methods",
    "dg": "dg_methods",
    "ipdg": "dg_methods",
    "sipg": "dg_methods",
    "nipg": "dg_methods",
    "interior_penalty": "dg_methods",

    # ── Biharmonic / plate bending ─────────────────────────────────
    "plate": "biharmonic",
    "kirchhoff": "biharmonic",
    "kirchhoff_love": "biharmonic",
    "fourth_order": "biharmonic",
    "bending": "biharmonic",
    "kirchhoff_plate": "biharmonic",

    # ── Adaptive refinement (dealii / skfem / dune) ────────────────
    # canonical varies: adaptive_refinement (dealii), adaptive_poisson
    # (skfem, dune), hp_adaptive (dealii). _fuzzy_match_physics routes
    # the synonym only if it exists in this backend's catalog, so
    # mapping to adaptive_refinement first is safe — fall-through
    # picks the right one per backend.
    "amr": "hp_adaptive",
    "refinement": "hp_adaptive",
    "adaptive": "hp_adaptive",
    "h_refinement": "hp_adaptive",
    "p_refinement": "hp_adaptive",
    "hp_refinement": "hp_adaptive",
    "hp": "hp_adaptive",
    "error_estimator": "error_estimation",
    "kelly_estimator": "error_estimation",
    "kelly": "error_estimation",
    "adaptive_mesh": "hp_adaptive",
    "mesh_refinement": "hp_adaptive",

    # ── Phase field / fracture / damage ────────────────────────────
    "cahn_hilliard": "phase_field",
    "allen_cahn": "phase_field",
    "phase_field_fracture": "phase_field",
    "brittle_fracture": "fracture",
    "crack": "fracture",
    "crack_propagation": "fracture",
    "fracture_mechanics": "fracture",
    "damage_mechanics": "damage",
    "continuum_damage": "damage",

    # ── Topology / shape optimization ──────────────────────────────
    "topopt": "topology_optimization",
    "topology": "topology_optimization",
    "topology_opt": "topology_optimization",
    "shape_opt": "shape_optimization",
    "shape_optimisation": "shape_optimization",
    "structural_optimization": "topology_optimization",
    "compliance_minimization": "topology_optimization",

    # ── Contact / friction ─────────────────────────────────────────
    "friction": "contact",
    "contact_mechanics": "contact",
    "frictional_contact": "contact",
    "hertz": "contact",
    "mortar_contact": "contact",
    "node_to_surface": "contact",
    "surface_to_surface": "contact",

    # ── FSI / TSI / multiphysics coupling ──────────────────────────
    "fluid_structure": "fsi",
    "fluid_structure_interaction": "fsi",
    "thermo_structural": "thermal_structural",
    "thermomechanical": "thermal_structural",
    "multiphysics": "fsi",
    "coupling": "fsi",
    "thermal_solid_interaction": "tsi",
    "structural_thermal_interaction": "tsi",
    "soil_structure": "ssi",
    "structure_soil_interaction": "ssi",

    # ── Porous media / geomechanics ────────────────────────────────
    "poroelasticity": "porous_media",
    "poro": "porous_media",
    "consolidation": "porous_media",
    "terzaghi": "porous_media",
    "biot": "porous_media",
    "geomechanics": "porous_media",
    "saturated_porous": "porous_media",
    "unsaturated_porous": "porous_media",

    # ── Particle methods: peridynamics, SPH, DEM, MPM ──────────────
    "peridynamics": "particle_pd",
    "pd": "particle_pd",
    "bond_based": "particle_pd",
    "state_based": "particle_pd",
    "ordinary_state_based": "particle_pd",
    "non_ordinary_state_based": "particle_pd",
    "nosbpd": "particle_pd",
    "sph": "particle_sph",
    "smoothed_particle": "particle_sph",
    "smoothed_particle_hydrodynamics": "particle_sph",
    "discrete_element": "dem",
    "discrete_element_method": "dem",
    "granular": "dem",
    "material_point": "mpm",
    "material_point_method": "mpm",
    "particle_in_cell": "mpm",
    "pic": "mpm",
    "lagrangian_particles": "particle_sph",

    # ── Multiphase / free surface / VOF / level-set ────────────────
    "two_phase": "multiphase",
    "multi_phase": "multiphase",
    "vof": "multiphase",
    "volume_of_fluid": "multiphase",
    "immiscible": "multiphase",
    "interface": "multiphase",
    "free_surface_flow": "free_surface",
    "level_set_method": "level_set",
    "droplet": "droplet_dynamics",

    # ── Reaction-diffusion / chemical kinetics ─────────────────────
    "rd": "reaction_diffusion",
    "reaction_diffusion_system": "reaction_diffusion",
    "fitzhugh_nagumo": "reaction_diffusion",
    "gray_scott": "reaction_diffusion",
    "schnakenberg": "reaction_diffusion",
    "chemical_kinetics": "reaction_diffusion",

    # ── Structural dynamics / transient solid ──────────────────────
    "dynamics": "structural_dynamics",
    "transient_structural": "structural_dynamics",
    "dynamic_analysis": "structural_dynamics",
    "time_domain_structural": "structural_dynamics",
    "implicit_dynamics": "structural_dynamics",
    "explicit_dynamics": "structural_dynamics",
    "structural_transient": "structural_dynamics",

    # ── Schrödinger / quantum ──────────────────────────────────────
    "quantum": "schrodinger",
    "quantum_mechanics": "schrodinger",
    "wavefunction": "schrodinger",
    "eigenstate": "schrodinger",

    # ── MHD ────────────────────────────────────────────────────────
    "magnetohydrodynamics": "mhd",
    "magneto_hydrodynamics": "mhd",
    "plasma": "mhd",

    # ── Beams / shells / membranes ─────────────────────────────────
    "beam": "beams",
    "beam_element": "beams",
    "timoshenko": "beams",
    "euler_bernoulli": "beams",
    "shell_element": "shell",
    "kirchhoff_love_shell": "shell",
    "reissner_mindlin": "shell",
    "membrane_element": "membrane",

    # ── Cardiac / cardiovascular ───────────────────────────────────
    "cardiac": "cardiac_monodomain",
    "electrophysiology": "cardiac_monodomain",
    "monodomain": "cardiac_monodomain",
    "bidomain": "cardiac_monodomain",
    "cardiovascular": "cardiovascular0d",
    "windkessel": "cardiovascular0d",
    "lumped_parameter": "cardiovascular0d",
    "0d_model": "cardiovascular0d",

    # ── XFEM ───────────────────────────────────────────────────────
    "extended_fem": "xfem_fluid",
    "xfem": "xfem_fluid",
    "level_set_fem": "xfem_fluid",
    "embedded_interface": "xfem_fluid",

    # ── Reduced-order / multiscale ─────────────────────────────────
    "rom": "rom",
    "reduced_order": "rom",
    "reduced_order_modeling": "rom",
    "pod": "rom",
    "homogenization": "multiscale",
    "fe_squared": "multiscale",
    "fe2": "multiscale",

    # ── Optimization ───────────────────────────────────────────────
    "optimal_control": "optimal_control",
    "adjoint": "optimal_control",
    "inverse_problem": "optimal_control",

    # ── Matrix-free / multigrid (solver-level not physics, but
    #     dealii exposes them as physics keys) ─────────────────────
    "matrix_free_fe": "matrix_free",
    "geometric_multigrid": "multigrid",
    "algebraic_multigrid": "multigrid",
    "amg": "multigrid",
    "gmg": "multigrid",

    # ── HDG / HDivDiv / mixed methods ──────────────────────────────
    "hdg": "hdivdiv",
    "hybridizable_dg": "hdivdiv",
    "hellinger_reissner": "hdivdiv",
    "raviart_thomas": "mixed_poisson",
    "rt": "mixed_poisson",
    "bdm": "mixed_poisson",
    "mixed_finite_element": "mixed_poisson",
    "mixed_method": "mixed_poisson",
    "h_div_conforming": "mixed_poisson",

    # ── Hydraulics / shallow water ─────────────────────────────────
    "shallow_water_equations": "shallow_water",
    "saint_venant": "shallow_water",
    "swe": "shallow_water",

    # ── ALE ────────────────────────────────────────────────────────
    "arbitrary_lagrangian_eulerian": "ale",
    "moving_mesh": "ale",
}


# Queries shorter than this never participate in loose substring
# matches — short tokens collide with too many physics names /
# descriptions ('ns' is a substring of 'transient', 'em' of
# 'eigenvalue', 'pd' of 'pde'). For short tokens we trust ONLY
# exact-name and synonym-map matches. (Audit 2026-06-02.)
_MIN_LOOSE_MATCH_LEN = 4


def _fuzzy_match_physics(backend, query: str) -> str:
    """Fuzzy-match a physics query to an actual physics name in a backend.

    Resolution order (audit 2026-06-02):
      1. Empty -> return empty so caller can surface availables list.
      2. Exact physics-name match.
      3. Synonym map (e.g. 'ns' -> 'navier_stokes', 'em' -> 'maxwell',
         'thermal' -> 'heat'). Synonyms run BEFORE substring matching
         because short-token substrings collide constantly:
         'ns' is a substring of 'transient', 'em' of 'eigenvalue',
         'pd' of 'nonlinear_pde'. Without this ordering, LLMs that
         type the canonical shorthand silently got the wrong physics.
      4. Query is substring of a physics name (only if len >= 4).
      5. Physics name is substring of the query (only if the
         physics name is itself >= 4 chars — otherwise tiny names
         like 'pd' match every query containing those letters).
      6. Query is substring of a physics description (last resort,
         len >= 4).
      7. Fallthrough: return original so caller can produce a
         "no information found" message.
    """
    query_lower = query.lower().strip()

    # Empty / whitespace-only query — return verbatim so the
    # caller's "no information found" path can surface the
    # full available-physics list. Without this guard, the
    # next substring check matches "" to the FIRST physics
    # in the catalog (because "" is a substring of every
    # string) and the LLM silently sees prepare_simulation
    # output for a physics it never asked for. (Audit
    # 2026-06-01.)
    if not query_lower:
        return query_lower

    # 1. Direct match.
    for p in backend.supported_physics():
        if p.name == query_lower:
            return p.name

    # 2. Synonym map — BEFORE the substring scan so short
    # canonical shorthands ('ns', 'em', 'pd') route to the
    # right physics. Only return the synonym if it actually
    # exists in this backend's catalog; otherwise fall through
    # to the loose matchers (a backend that has 'maxwell' but
    # not the synonym should still match via substring).
    mapped = _PHYSICS_SYNONYMS.get(query_lower)
    if mapped:
        for p in backend.supported_physics():
            if p.name == mapped:
                return p.name

    # 3. Loose substring of physics name (only for non-short
    # queries — see _MIN_LOOSE_MATCH_LEN rationale above).
    if len(query_lower) >= _MIN_LOOSE_MATCH_LEN:
        for p in backend.supported_physics():
            if query_lower in p.name.lower():
                return p.name

    # 4. Physics name is substring of query (only when the
    # physics name itself is non-trivial). Without the length
    # guard, a 2-char catalog entry like 'pd' matches every
    # query containing those letters, which is the same
    # collision class we just guarded the other direction
    # against.
    for p in backend.supported_physics():
        if (len(p.name) >= _MIN_LOOSE_MATCH_LEN
                and p.name.lower() in query_lower):
            return p.name

    # 5. Loose substring of physics description (last resort,
    # same length guard).
    if len(query_lower) >= _MIN_LOOSE_MATCH_LEN:
        for p in backend.supported_physics():
            if query_lower in p.description.lower():
                return p.name

    # Nothing matched — return original so the caller can
    # surface the "no information found" message with the
    # available-physics list.
    return query_lower


def _list_alternative_solvers(current_solver: str, physics: str) -> str:
    """List other backends that also support this physics (informational).

    This helps the agent know what alternatives exist if the chosen solver
    runs into issues, without being prescriptive about which to use.
    """
    alternatives = []
    for b in all_backends():
        if b.name() == current_solver:
            continue
        status, _ = b.check_availability()
        for p in b.supported_physics():
            if p.name == physics or physics in p.name or p.name in physics:
                # Tag unavailable backends so the LLM knows
                # they would need to be installed first. Hiding
                # them silently (the old available_backends()
                # behaviour) made dune-fem and febio
                # alternatives invisible. (Audit 2026-06-02.)
                tag = "" if status.value == "available" else f" *[{status.value}]*"
                alternatives.append(
                    f"- **{b.display_name()}**{tag}: {p.description}")
                break
    if not alternatives:
        return ""
    return "Other solvers that support this physics:\n" + "\n".join(alternatives)


def _load_matching_postmortems(solver: str = "", physics: str = "",
                               signal: str = "") -> list[dict]:
    """Load post-mortem JSONs from data/postmortems/, filtered.

    The post-mortems directory is the audit trail behind the pitfall
    DB. Each record explains WHY a pitfall was added — the surface
    symptom that was observed, the root cause, the Table-1 category,
    the exact pitfall entries shipped, and the detection path the
    agent now has. The Open-FEM-Agent paper's §3.2 / §5
    self-correction loop depends on the agent being able to retrieve
    these at planning time.

    Filters (any can be empty, treated as "match all"):
      * solver  — exact match against the post-mortem's `backend`
                  field (case-insensitive). NOT a fuzzy match because
                  the post-mortem's audit value depends on knowing
                  it's about THIS backend, not a similar one.
      * physics — substring match against the `physics` field. A
                  batch post-mortem like
                  "poisson, heat, helmholtz, eigenvalue" matches any
                  of its members.
      * signal  — substring match across each `pitfall_db_entries`
                  string. Useful when the post-execution critic
                  sees a specific error and wants to find the
                  matching post-mortem.

    Returns the post-mortems as parsed dicts. Sorted by `date`
    descending so the most-recent record comes first — typically
    the most-relevant for the current agent session.

    Files under ``data/postmortems/candidates/`` are NOT included
    here. Candidates are the pre-review staging area
    (Open-FEM-Agent §3.2 autonomous-growth path) — promotion to a
    formal post-mortem is a deliberate review step (#46).
    """
    pm_dir = Path(__file__).resolve().parents[2] / "data" / "postmortems"
    if not pm_dir.is_dir():
        return []
    solver_l = solver.lower().strip()
    physics_l = physics.lower().strip()
    signal_l = signal.lower().strip()
    out: list[dict] = []
    for path in pm_dir.glob("*.json"):
        if path.name.startswith("_"):
            # Skip schema / index files.
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        if solver_l and str(doc.get("backend", "")).lower() != solver_l:
            continue
        if physics_l and physics_l not in \
                str(doc.get("physics", "")).lower():
            continue
        if signal_l:
            entries = doc.get("pitfall_db_entries", []) or []
            if not any(signal_l in str(e).lower() for e in entries):
                continue
        out.append(doc)
    out.sort(key=lambda d: str(d.get("date", "")), reverse=True)
    return out


def _make_input_snapshot(input_content: str, solver: str = "",
                         extra: dict | None = None) -> dict:
    """Create a sanitised snapshot of simulation input for diff capture.

    Captures structure (length, line count, key patterns) without leaking content.
    """
    import hashlib
    snap = {
        "solver": solver,
        "input_length": len(input_content),
        "input_lines": input_content.count("\n") + 1,
        "input_hash": hashlib.sha256(input_content.encode()).hexdigest()[:12],
    }
    if extra:
        snap.update(extra)
    return snap


def register_consolidated_tools(mcp: FastMCP):
    """Register all consolidated tools — ~12 tools instead of 48."""

    # Session journal — records events for knowledge capture
    from core.session_journal import get_journal as _get_journal

    # ═══════════════════════════════════════════════════════════
    # 1. KNOWLEDGE (replaces 13 separate knowledge tools)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def knowledge(topic: str, solver: str = "", physics: str = "",
                  signal: str = "") -> str:
        """Get knowledge about solvers, physics, materials, coupling,
        post-mortems, or input formats.

        This is the single entry point for ALL domain knowledge — the
        catalog, the pitfall database, AND the post-mortem record
        store. Wiring post-mortems through this same tool closes the
        self-improvement loop: every prepare_simulation call also
        surfaces the relevant post-mortems so the critic-gate can
        retrieve them at planning time (Open-FEM-Agent §3.2 / §5
        self-correction loop).

        Args:
            topic: What you want to know. Options:
                - "physics" — physics-specific knowledge + matching
                  post-mortems (needs solver + physics)
                - "pitfalls" — all known pitfalls for a solver
                - "postmortems" — formal post-mortem records under
                  data/postmortems/*.json, filtered by solver +
                  physics + optional signal pattern. These are the
                  audit-trail entries that record WHY each pitfall
                  exists; the critic-gate should retrieve them when
                  the agent's plan touches the matching (solver,
                  physics) area.
                - "materials" — material catalog for a solver
                - "coupling" — cross-solver coupling knowledge
                - "tsi" — thermo-structural interaction patterns
                - "precice" — preCICE comparison
                - "input_guide" — how to write input files for a solver
                - "solver_guidance" — which solver to use for a physics type
                - "hardware" — parallelism, GPU, and hardware acceleration capabilities
                - "overview" — backend-level reference catalog (element
                  families, mesh types, solver catalogue, unique
                  features). The content under the special "_general"
                  knowledge key — for dealii ~5 KB, fenics / ngsolve /
                  skfem / kratos / dune ~1-2 KB each. Needs solver=...
            solver: Backend name (e.g. 'fenics', 'fourc', 'dealii', 'ngsolve')
            physics: Physics type (e.g. 'poisson', 'linear_elasticity', 'navier_stokes')
            signal: Optional substring to filter post-mortem
                pitfall_db_entries Signal: clauses against — useful
                when the post-execution critic sees a specific error
                text and wants to find the matching post-mortem.
        """
        _get_journal().record("knowledge_lookup", "knowledge",
                              solver=solver, physics=physics,
                              notes=f"topic={topic}")
        if topic == "physics" and solver and physics:
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            k = backend.get_knowledge(physics)
            if not k:
                return f"No knowledge for '{physics}' in {solver}"
            result = json.dumps(k, indent=2, default=str)
            # Append real test file references
            from tools.knowledge import _find_reference_test_files
            ref = _find_reference_test_files(solver, physics)
            if ref:
                result += f"\n\n{ref}"
            # Append post-mortem BREADCRUMBS (ids only) — not full
            # records — at plan time. Rationale (senior-AI-scientist
            # critic, 2026-05-31): full post-mortems include
            # surface_symptom / root_cause / agent_detection_after_fix,
            # which are diagnostic fields for human review (#46), not
            # pre-execution guidance. Auto-including them at plan
            # time produces linear token bloat in N_postmortems and
            # competes with the catalog for the agent's attention.
            # The pitfall_db_entries the catalog already exposes ARE
            # the pre-execution actionable content; the full
            # post-mortem belongs to the post-execution critic when
            # it has a Signal: to match. Agent can fetch the full
            # record explicitly via
            # `knowledge(topic="postmortems", solver=..., signal=...)`.
            postmortems = _load_matching_postmortems(solver, physics, "")
            if postmortems:
                breadcrumbs = [
                    {"id": pm.get("id", "?"),
                     "categories": pm.get("categories", []),
                     "date": pm.get("date", "")}
                    for pm in postmortems
                ]
                result += (
                    f"\n\n## Post-mortem breadcrumbs "
                    f"({len(postmortems)} record"
                    f"{'' if len(postmortems) == 1 else 's'} — "
                    f"fetch full records via knowledge"
                    f"(topic='postmortems', solver=..., signal=...)"
                    f" when a post-execution Signal needs lookup):\n"
                    + json.dumps(breadcrumbs, indent=2))
            return result

        elif topic == "postmortems":
            postmortems = _load_matching_postmortems(solver, physics, signal)
            if not postmortems:
                what = ", ".join(
                    f"{k}={v!r}" for k, v in
                    {"solver": solver, "physics": physics,
                     "signal": signal}.items() if v)
                return (f"No post-mortems found"
                        f"{' for ' + what if what else ''}. "
                        f"data/postmortems/*.json is the canonical "
                        f"store; absence here means the failure mode "
                        f"has not yet been audited.")
            return json.dumps(postmortems, indent=2)

        elif topic == "pitfalls" and solver:
            # Backend is the source of truth for pitfalls (Table-1
            # promoted, post-execution-critic-actionable). The
            # deep_knowledge fallback was inverted historically —
            # it returned prose entries even for backends whose
            # generators had been Table-1 promoted, breaking the
            # alignment between prepare_simulation and discover.
            # Now backend is consulted FIRST; deep_knowledge is
            # only used as a supplement for physics the backend
            # does not enumerate (rare in practice).
            backend = get_backend(solver)
            all_pitfalls = {}
            if backend:
                for p in backend.supported_physics():
                    k = backend.get_knowledge(p.name)
                    if k and "pitfalls" in k:
                        all_pitfalls[p.name] = k["pitfalls"]
            try:
                from tools.deep_knowledge import _4C_KNOWLEDGE, _FENICS_KNOWLEDGE
                dicts = {"fourc": _4C_KNOWLEDGE, "4c": _4C_KNOWLEDGE,
                         "fenics": _FENICS_KNOWLEDGE, "fenicsx": _FENICS_KNOWLEDGE}
                d = dicts.get(solver.lower(), {})
                for k, v in d.items():
                    if (isinstance(v, dict) and "pitfalls" in v
                            and k not in all_pitfalls):
                        all_pitfalls[k] = v["pitfalls"]
            except ImportError:
                pass
            if not backend:
                if all_pitfalls:
                    return json.dumps(all_pitfalls, indent=2)
            if backend:
                # Also include general input-format pitfalls (e.g., ExodusII
                # block IDs, FUNCT syntax, shared-node NUMDOF conflict)
                general_k = backend.get_knowledge("input_format")
                if isinstance(general_k, dict):
                    gp = general_k.get("general_pitfalls")
                    if gp:
                        all_pitfalls["general_input_format"] = gp
                    et = general_k.get("element_type_per_physics")
                    if et:
                        all_pitfalls["element_types"] = et
                # Include community-contributed knowledge
                community = _load_community_knowledge(solver)
                if community:
                    all_pitfalls["community_contributed"] = [
                        {"title": c["title"], "description": c.get("description", ""),
                         "category": c.get("category", ""), "confidence": c.get("confidence", 0)}
                        for c in community
                    ]
                return json.dumps(all_pitfalls, indent=2)
            return f"No pitfalls found for {solver}"

        elif topic == "materials" and solver:
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            materials = {}
            for p in backend.supported_physics():
                k = backend.get_knowledge(p.name)
                if k and "materials" in k:
                    materials[p.name] = k["materials"]
            return json.dumps(materials, indent=2) if materials else f"No material catalog for {solver}"

        elif topic == "overview" and solver:
            # Surface the backend-level "_general" reference catalog
            # (element families, mesh types, solver catalogue, unique
            # features). Discovered 2026-06-02: get_knowledge('_general')
            # returns substantive reference content for 6 of 8 backends
            # (dealii 5.2 KB; fenics/ngsolve/skfem/kratos/dune 1-2 KB
            # each) but was NOT exposed via any `knowledge(topic=...)`
            # surface — LLMs had no way to discover it existed.
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            general = backend.get_knowledge("_general")
            if not isinstance(general, dict) or not general or "error" in general:
                return (f"No backend-level overview catalog for "
                        f"{solver} (get_knowledge('_general') is "
                        "empty or returned an error).")
            return json.dumps({solver: general}, indent=2)

        elif topic == "coupling":
            from tools.knowledge import register_knowledge_tools
            # Return coupling knowledge directly
            return _get_coupling_knowledge()

        elif topic == "tsi":
            return _get_tsi_knowledge()

        elif topic == "precice":
            return _get_precice_knowledge()

        elif topic == "input_guide" and solver:
            from tools.examples_search import (
                _4C_INPUT_GUIDE, _FENICS_INPUT_GUIDE, _DEALII_INPUT_GUIDE,
                _FEBIO_INPUT_GUIDE, _DUNE_INPUT_GUIDE,
            )
            guides = {"fourc": _4C_INPUT_GUIDE, "4c": _4C_INPUT_GUIDE,
                      "fenics": _FENICS_INPUT_GUIDE, "dealii": _DEALII_INPUT_GUIDE,
                      "febio": _FEBIO_INPUT_GUIDE,
                      "dune": _DUNE_INPUT_GUIDE, "dune-fem": _DUNE_INPUT_GUIDE,
                      "dunefem": _DUNE_INPUT_GUIDE}
            return guides.get(solver.lower(), f"No input guide for {solver}")

        elif topic == "solver_guidance" and physics:
            # Show ALL registered backends so the LLM can learn
            # which solvers offer the physics in principle —
            # even when not installed yet — and decide whether
            # to install one. Tag unavailable backends so the
            # LLM does not try to run_simulation on them.
            # (Audit 2026-06-02; same hide-unavailable bug as
            # discover('list').)
            results = {}
            for b in all_backends():
                for p in b.supported_physics():
                    if p.name == physics:
                        status, _ = b.check_availability()
                        key = (b.display_name() if status.value == "available"
                               else f"{b.display_name()} [{status.value}]")
                        results[key] = {
                            "variants": p.template_variants,
                            "elements": p.element_types,
                            "dims": p.spatial_dims,
                        }
            return json.dumps(results, indent=2) if results else f"No solver supports '{physics}'"

        elif topic == "hardware":
            hw = {
                "FEniCSx (dolfinx)": {
                    "parallelism": "MPI (first-class, domain decomposition via PETSc)",
                    "gpu": "No native GPU. PETSc can use GPU backends (CUDA/HIP) for linear algebra if compiled with Kokkos/CUDA support, but this is not standard.",
                    "threading": "Limited — PETSc threading for assembly",
                    "typical_scale": "Millions of DOFs on HPC clusters",
                },
                "deal.II": {
                    "parallelism": "MPI (p4est for distributed meshes) + threading (TBB/std::thread)",
                    "gpu": "Yes — matrix-free GPU kernels via CUDA and portable backends. GPU support for matrix-free operators is a key feature (step-64 tutorial).",
                    "threading": "SharedMemory::TBB or std::thread for assembly",
                    "typical_scale": "Billions of DOFs demonstrated (matrix-free, GPU)",
                },
                "4C Multiphysics": {
                    "parallelism": "MPI (domain decomposition) + OpenMP threading",
                    "gpu": "No GPU for linear algebra (Epetra-based, CPU-only). Optional ArborX (Kokkos) for GPU-accelerated geometric search only. Tpetra (GPU-capable) not yet integrated.",
                    "threading": "OpenMP (set OMP_NUM_THREADS)",
                    "typical_scale": "Millions of DOFs on MPI clusters",
                    "note": "Trilinos 16.2.0 is the last supported version due to Epetra dependency",
                },
                "NGSolve": {
                    "parallelism": "MPI (via NGSolve's own parallel framework) + shared-memory task parallelism",
                    "gpu": "Experimental CUDA support for some operations. Not production-ready for most users.",
                    "threading": "Task-based parallelism (Netgen's built-in scheduler)",
                    "typical_scale": "Millions of DOFs",
                },
                "scikit-fem": {
                    "parallelism": "Serial only (no MPI). NumPy/SciPy vectorisation for assembly.",
                    "gpu": "No GPU support. Pure Python/NumPy.",
                    "threading": "NumPy BLAS threading only",
                    "typical_scale": "Tens of thousands of DOFs (prototyping)",
                },
                "Kratos Multiphysics": {
                    "parallelism": "MPI (Trilinos-based) + OpenMP for shared memory",
                    "gpu": "Limited — some GPU acceleration via Trilinos/Kokkos for linear algebra. Not all applications support it.",
                    "threading": "OpenMP",
                    "typical_scale": "Millions of DOFs",
                },
                "DUNE-fem": {
                    "parallelism": "MPI (DUNE grid parallelism via ALUGrid/YaspGrid)",
                    "gpu": "No native GPU support in DUNE-fem. DUNE-copasi has experimental GPU work.",
                    "threading": "Limited",
                    "typical_scale": "Moderate (research scale)",
                },
                "FEBio": {
                    "parallelism": (
                        "Shared-memory only (OpenMP). No MPI domain "
                        "decomposition: a single FEBio process drives "
                        "the whole simulation. Multi-physics coupling "
                        "(biphasic, multiphasic, fluid-solid mixture) "
                        "is monolithic in the solver, not via "
                        "subdomain decomposition."),
                    "gpu": (
                        "No GPU support. FEBio's linear-algebra "
                        "back-end is CPU only (PARDISO / MUMPS / "
                        "Skyline). GPU acceleration is on the wishlist "
                        "but not implemented as of FEBio 4.x."),
                    "threading": (
                        "OpenMP across element assembly + PARDISO's "
                        "internal threading. Set OMP_NUM_THREADS for "
                        "assembly; the linear solver uses its own "
                        "OMP_NUM_THREADS or MKL_NUM_THREADS pool."),
                    "typical_scale": (
                        "Hundreds of thousands of DOFs on a workstation; "
                        "millions are routinely run but FEBio targets "
                        "biomechanical models (single bones, soft "
                        "tissue, biphasic cartilage) rather than HPC "
                        "scale."),
                    "note": (
                        "FEBio's strength is biomechanics-specific "
                        "physics (biphasic / multiphasic mixtures, "
                        "active contraction, fiber materials, "
                        "growth-remodeling). It is NOT a general-"
                        "purpose FEM code; do not pick it for "
                        "Navier-Stokes / electromagnetics / "
                        "geomechanics."),
                },
            }
            if solver:
                key_map = {"fourc": "4C Multiphysics", "4c": "4C Multiphysics",
                           "fenics": "FEniCSx (dolfinx)", "fenicsx": "FEniCSx (dolfinx)",
                           "dealii": "deal.II", "deal.ii": "deal.II",
                           "ngsolve": "NGSolve", "skfem": "scikit-fem", "scikit-fem": "scikit-fem",
                           "kratos": "Kratos Multiphysics", "dune": "DUNE-fem", "dune-fem": "DUNE-fem",
                           "febio": "FEBio"}
                name = key_map.get(solver.lower(), solver)
                if name in hw:
                    return json.dumps({name: hw[name]}, indent=2)
                return f"No hardware info for {solver}"
            return json.dumps(hw, indent=2)

        else:
            # Topics list must match the docstring + dispatch
            # branches. Audit 2026-06-01: 'postmortems' was
            # documented in the docstring and implemented at
            # line 326 but missing from this usage hint, so
            # LLMs hitting an invalid topic never learned that
            # postmortems exists. (Same drift class as
            # session_insights' missing 'ingest'.)
            return (
                "Usage: knowledge(topic, solver, physics, signal='')\n"
                "Topics: physics, pitfalls, postmortems, materials, "
                "overview, coupling, tsi, precice, input_guide, "
                "solver_guidance, hardware"
            )

    # ═══════════════════════════════════════════════════════════
    # 2. DISCOVER (replaces 6 discovery tools)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def discover(query: str = "list", solver: str = "") -> str:
        """Discover available solvers and their capabilities.

        Args:
            query: What to discover. Options:
                - "list" — list all solvers with status
                - "physics" — list all physics types per solver
                - "capabilities" — full capabilities matrix
                - "recommend" — recommend solver for a physics (set solver= to physics name)
            solver: Filter by solver name, or physics name for "recommend"
        """
        if query == "list":
            # Show ALL registered backends, not only the
            # installed ones. The MCP server instructions
            # advertise 8 backends; if discover('list') hides
            # the ones the user has not installed yet, an LLM
            # asking for (say) DUNE-fem or FEBio gets no entry,
            # no status, and no install hint — total dead end.
            # Surface every backend with its actual availability
            # status and the install hint that
            # check_availability() returns. (Audit 2026-06-02.)
            lines = []
            for b in all_backends():
                status, msg = b.check_availability()
                core = (f"- **{b.display_name()}** ({b.name()}): "
                        f"{status.value} — "
                        f"{b.input_format().value} input")
                if status.value != "available" and msg:
                    # Inline the install/troubleshoot hint so the
                    # LLM does not have to call a second tool.
                    core += f"\n  *{msg.strip()}*"
                lines.append(core)
            return "\n".join(lines) if lines else "No backends registered."

        elif query == "physics":
            # Show physics for ALL registered backends (same
            # rationale as discover('list')) so an LLM can
            # learn what dune-fem or febio offer even before
            # installing them. Tag unavailable backends with
            # their status so the LLM does not call
            # run_simulation against a backend that will
            # error out on availability. (Audit 2026-06-02.)
            lines = []
            backends = [get_backend(solver)] if solver else all_backends()
            backends = [b for b in backends if b]
            for b in backends:
                status, _ = b.check_availability()
                tag = "" if status.value == "available" else f" *[{status.value}]*"
                lines.append(f"## {b.display_name()}{tag}")
                for p in b.supported_physics():
                    lines.append(f"- **{p.name}**: {p.description} (variants: {', '.join(p.template_variants)})")
                lines.append("")
            return "\n".join(lines)

        elif query == "capabilities":
            # Show ALL registered backends (see discover('list')
            # rationale above) so an LLM sees the full
            # capabilities matrix including not-yet-installed
            # backends. (Audit 2026-06-02.)
            lines = ["| Solver | Physics Count | Input | Status |",
                     "|--------|--------------|-------|--------|"]
            for b in all_backends():
                status, _ = b.check_availability()
                lines.append(f"| {b.display_name()} | {len(b.supported_physics())} | {b.input_format().value} | {status.value} |")
            return "\n".join(lines)

        elif query == "recommend":
            physics = solver  # in this case solver param holds the physics name
            # Empty / whitespace-only physics matches every
            # backend's first physics (substring-of-everything).
            # Same class of bug as the empty-physics
            # prepare_simulation match — reject it explicitly so
            # the LLM gets a clear usage hint instead of a fake
            # "all backends recommend this" result. Audit
            # 2026-06-01.
            if not physics or not physics.strip():
                return ("Empty physics name. Pass the physics "
                        "as the 'solver' parameter, e.g. "
                        "discover(query='recommend', "
                        "solver='poisson').")
            # Route the physics query through the canonical
            # fuzzy resolver per backend so short shorthands
            # ('ns', 'em', 'cfd', ...) hit the synonym map
            # before a loose substring scan. The raw
            # substring-on-name-or-description recommendation
            # silently matched 'ns' to heat in fenics; 'em' to
            # eigenvalue in fenics; 'pd' to nonlinear_pde in
            # fenics. (Audit 2026-06-02; same drift class as
            # the prepare_simulation fix.)
            #
            # Iterate ALL registered backends, not just the
            # installed ones, so the recommendation includes
            # backends the user has not installed yet. Tag
            # unavailable backends inline so the LLM knows
            # they need an install step. (Audit 2026-06-02;
            # same hide-unavailable bug as discover('list').)
            results = []
            for b in all_backends():
                matched = _fuzzy_match_physics(b, physics)
                if not matched:
                    continue
                for p in b.supported_physics():
                    if p.name == matched:
                        status, _ = b.check_availability()
                        tag = "" if status.value == "available" else f" *[{status.value}]*"
                        results.append(
                            f"- **{b.display_name()}**{tag}: {p.description}")
                        break
            return "\n".join(results) if results else f"No solver found for '{physics}'"

        return "Usage: discover(query='list'|'physics'|'capabilities'|'recommend', solver='')"

    # ═══════════════════════════════════════════════════════════
    # 3. EXAMPLES (replaces 7 example/search tools)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def examples(keyword: str, solver: str = "fourc", action: str = "search",
                 max_results: int = 3) -> str:
        """Find and retrieve example input files from solver test suites.

        IMPORTANT: Always call this before writing new input files to study
        real, validated configurations.

        Args:
            keyword: Search term (e.g. 'peridynamic', 'fsi', 'poisson', 'heat')
            solver: Backend name (default: 'fourc')
            action: What to do. Options:
                - "search" — find matching test files with content preview
                - "template" — get a generated template for this physics
                - "tutorials" — list available tutorials
            max_results: Maximum results (default 3)
        """
        if action == "search":
            # Empty / whitespace-only keyword matches every
            # filename (substring-of-everything) and silently
            # returns the first few random files in the test
            # tree. Surface a usage hint instead. Audit
            # 2026-06-01 (mirror of the empty-physics fix in
            # prepare_simulation).
            if not keyword or not keyword.strip():
                return ("Empty keyword. Provide a substring "
                        "to match, e.g. 'poisson', 'fluid', "
                        "'contact'.")
            from tools.examples_search import register_example_tools
            # Shared discovery with prepare_simulation —
            # discover_test_dirs returns local demo paths for all
            # backends; resolve_search_keywords applies the same
            # alias map (ngsolve hyperelasticity -> nonlin,
            # fenics navier_stokes -> navier-stokes, ...) so the
            # two LLM-facing tools surface the same content for
            # the same (solver, keyword) pair. Audit 2026-06-01.
            from tools.knowledge import (discover_test_dirs,
                                         resolve_search_keywords)
            results = []
            test_dirs = discover_test_dirs()
            solver_key = solver.lower()
            test_dir = test_dirs.get(solver_key)
            ext = "*.4C.yaml" if solver_key in ("fourc", "4c") else "*.cc" if solver_key == "dealii" else "*.py"

            if test_dir and test_dir.is_dir():
                # Apply solver-aware aliases on top of the raw
                # keyword (the LLM may have used the catalog
                # physics name verbatim, e.g. 'hyperelasticity'
                # — which doesn't match NGSolve's nonlin.py
                # demo without aliasing).
                kw_candidates = list(dict.fromkeys(
                    [keyword] + resolve_search_keywords(solver, keyword)))
                seen: set = set()
                for kw in kw_candidates:
                    for f in sorted(test_dir.rglob(ext)):
                        if kw.lower() not in f.name.lower():
                            continue
                        if f in seen:
                            continue
                        seen.add(f)
                        try:
                            content = f.read_text()[:5000]
                            rel = f.relative_to(test_dir)
                            results.append(f"### `{rel}`\n```\n{content}\n```\n")
                        except Exception:
                            pass
                        if len(results) >= max_results:
                            break
                    if len(results) >= max_results:
                        break

            # Also search templates. Route the keyword through the
            # canonical fuzzy resolver so short shorthands ('ns',
            # 'em', 'cfd', ...) resolve to the right physics via
            # the synonym map BEFORE a loose substring scan. The
            # old code did a raw substring match on name OR
            # description; keyword='ns' matched heat /
            # thermal_structural / reaction_diffusion /
            # multiphase / time_dependent_heat (all contain "ns"
            # somewhere) — five wrong templates and never a
            # navier_stokes one. (Audit 2026-06-02; same drift
            # class as the prepare_simulation fix.)
            #
            # Same 12000-char limit as prepare_simulation — the
            # harder Layer F templates (ngsolve hdivdiv /
            # nonlinear_elasticity, fenics navier_stokes) exceed
            # 3000 chars and lose their solver/output blocks if
            # truncated lower. (Audit 2026-06-01.)
            backend = get_backend(solver)
            if backend:
                EX_TEMPLATE_LIMIT = 12000
                matched = _fuzzy_match_physics(backend, keyword)
                for p in backend.supported_physics():
                    if p.name == matched:
                        for v in p.template_variants[:1]:
                            try:
                                content = backend.generate_input(p.name, v, {})
                                truncated = len(content) > EX_TEMPLATE_LIMIT
                                body = content[:EX_TEMPLATE_LIMIT]
                                suffix = (f"\n... [truncated {len(content) - EX_TEMPLATE_LIMIT} chars]"
                                          if truncated else "")
                                results.append(f"### Template: `{p.name}/{v}`\n```\n{body}{suffix}\n```\n")
                            except Exception as exc:
                                # Same rationale as the
                                # prepare_simulation generator-
                                # failure surfacing: a silent
                                # except: pass made
                                # examples('search') return a
                                # "no template, no error" reply
                                # for any catalog regression.
                                # Now the failure is visible.
                                # (Audit 2026-06-02.)
                                results.append(
                                    f"### Template: `{p.name}/{v}`\n"
                                    f"⚠ Template generation FAILED: "
                                    f"`{type(exc).__name__}: {exc}`\n")
                        break

            if not results:
                return f"No examples found for '{keyword}' in {solver}"
            return f"## {len(results)} example(s) for '{keyword}' from {solver}\n\n" + "\n---\n".join(results)

        elif action == "template":
            if not keyword or not keyword.strip():
                return ("Empty keyword. Provide a physics name "
                        "(or substring), e.g. 'poisson', 'fluid', "
                        "'contact'.")
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            # Route the keyword through the canonical fuzzy
            # resolver so short shorthands ('ns' -> navier_stokes,
            # 'em' -> maxwell, ...) route via the synonym map
            # first. The raw substring path here matched 'ns'
            # against 'transient_heat' and never against
            # 'navier_stokes' (no adjacent 'ns' substring in
            # 'navier_stokes' itself). (Audit 2026-06-02.)
            matched = _fuzzy_match_physics(backend, keyword)
            for p in backend.supported_physics():
                if p.name == matched:
                    variant = p.template_variants[0] if p.template_variants else "2d"
                    try:
                        content = backend.generate_input(p.name, variant, {})
                        fmt = detect_template_language(content, backend.input_format().value)
                        return f"```{fmt}\n{content}\n```"
                    except Exception as e:
                        return f"Error generating template: {e}"
            return f"No template for '{keyword}' in {solver}"

        elif action == "tutorials":
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            lines = [f"## {backend.display_name()} Templates\n"]
            for p in backend.supported_physics():
                lines.append(f"- **{p.name}**: {', '.join(p.template_variants)} — {p.description}")
            return "\n".join(lines)

        return "Usage: examples(keyword, solver, action='search'|'template'|'tutorials')"

    # ═══════════════════════════════════════════════════════════
    # 4. SIMULATE (replaces run_simulation + run_with_generator)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    async def run_with_generator(solver: str, generator_script: str,
                                  job_name: str = "", np: int = 1,
                                  critic_approved: bool = False,
                                  ctx: Context = None) -> str:
        """Run a generator script that creates an input file, then execute the solver.

        Use this for solvers that need a COMPILED binary or separate input files:
        - 4C: generator creates .4C.yaml + mesh, then 4C binary runs on them
        - deal.II: generator creates main.cpp, then cmake + make + ./fem_solve
        - Kratos (with real binary): generator creates ProjectParameters.json +
          .mdpa + MainKratos.py, then Kratos Python runs MainKratos.py

        DO NOT use this for:
        - FEniCS, NGSolve, scikit-fem, DUNE-fem: use run_simulation() instead
        - Kratos manual-assembly scripts (numpy/scipy): use run_simulation()
          since those are standalone Python scripts, not input-file generators

        The generator script runs in the server's Python. It must produce an
        input file matching one of: *.4C.yaml, *.yaml, input.*, solve.py,
        MainKratos.py

        Args:
            solver: Backend name (fourc, dealii, kratos)
            generator_script: Python script that creates the input file
            job_name: Optional job directory name
            np: MPI processes (default 1)
            critic_approved: Set True only after critic agent approved setup
        """
        import subprocess
        import sys

        _journal = _get_journal()
        _snap = _make_input_snapshot(generator_script, solver, {"type": "generator"})
        _journal.record("tool_call", "run_with_generator", solver=solver,
                        input_snapshot=_snap)

        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        status, msg = backend.check_availability()
        if status.value != "available":
            _journal.record("tool_error", "run_with_generator", solver=solver,
                            error_message=f"Not available: {msg}",
                            input_snapshot=_snap)
            return f"Solver {solver} not available: {msg}"

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = job_name or f"{solver}_gen_{ts}"
        work_dir = _OUTPUT_DIR / name
        work_dir.mkdir(parents=True, exist_ok=True)

        gen_path = work_dir / "generate_input.py"
        gen_path.write_text(generator_script)

        python = sys.executable
        gen_result = subprocess.run(
            [python, str(gen_path)],
            capture_output=True, text=True,
            cwd=str(work_dir),
        )

        if gen_result.returncode != 0:
            _journal.record("tool_error", "run_with_generator", solver=solver,
                            error_message=f"Generator failed: {gen_result.stderr[-200:]}",
                            input_snapshot=_snap)
            return json.dumps({
                "status": "failed", "phase": "generator",
                "error": gen_result.stderr[-500:],
                "work_dir": str(work_dir),
            }, indent=2)

        input_file = None
        for pattern in ["*.4C.yaml", "*.yaml", "input.*", "solve.py", "MainKratos.py"]:
            matches = list(work_dir.glob(pattern))
            if matches:
                input_file = matches[0]
                break

        if not input_file:
            _journal.record("tool_error", "run_with_generator", solver=solver,
                            error_message="Generator did not produce an input file",
                            input_snapshot=_snap)
            return json.dumps({
                "status": "failed", "phase": "generator",
                "error": "Generator did not produce an input file",
                "work_dir": str(work_dir),
            }, indent=2)

        input_content = input_file.read_text()
        # Update snapshot with the generated input's shape
        _snap_run = _make_input_snapshot(input_content, solver,
                                         {"type": "generated_input", "input_file": input_file.name})
        from core.backend import JobHandle
        run_coro = backend.run(input_content, work_dir, np=np, timeout=None)
        if ctx is not None:
            job = await _run_with_progress(ctx, run_coro, f"Running {solver}")
        else:
            job = await run_coro
        _jobs[job.job_id] = job

        if job.error:
            _journal.record("tool_error", "run_with_generator", solver=solver,
                            error_message=job.error[:300],
                            input_snapshot=_snap_run)
        else:
            _journal.record("tool_success", "run_with_generator", solver=solver,
                            input_snapshot=_snap_run)

        result = {
            "job_id": job.job_id, "solver": solver,
            "status": job.status, "work_dir": str(job.work_dir),
            "elapsed": f"{job.elapsed:.2f}s" if job.elapsed else None,
            "input_file": input_file.name,
            "critic_review": "approved" if critic_approved else "SKIPPED",
        }
        if job.error:
            result["error"] = job.error[:500]
        if job.status == "completed":
            result["output_files"] = [f.name for f in backend.get_result_files(job)]
            stdout_log = work_dir / "stdout.log"
            if stdout_log.exists():
                text = stdout_log.read_text()
                result["stdout_tail"] = text[-2000:] if len(text) > 2000 else text
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def run_simulation(solver: str, input_content: str,
                             job_name: str = "", np: int = 1,
                             critic_approved: bool = False,
                             ctx: Context = None) -> str:
        """Run a simulation directly with input content.

        Use this for Python-based solvers (FEniCS, NGSolve, scikit-fem, DUNE-fem)
        where the input IS a Python script. The tool routes through the correct
        Python environment automatically (e.g., conda env for FEniCS).

        For 4C/deal.II/Kratos where a separate input file must be generated
        first, use run_with_generator() instead.

        Args:
            solver: Backend name (best for: fenics, ngsolve, skfem, dune)
            input_content: The input content (Python script / YAML / C++ / XML)
            job_name: Optional job name
            np: MPI processes
            critic_approved: Set True only after critic agent approved
        """
        _journal = _get_journal()
        _snap = _make_input_snapshot(input_content, solver)
        _journal.record("tool_call", "run_simulation", solver=solver,
                        input_snapshot=_snap)

        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        status, msg = backend.check_availability()
        if status.value != "available":
            _journal.record("tool_error", "run_simulation", solver=solver,
                            error_message=f"Not available: {msg}",
                            input_snapshot=_snap)
            return f"Solver {solver} not available: {msg}"

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = job_name or f"{solver}_{ts}"
        work_dir = _OUTPUT_DIR / name
        work_dir.mkdir(parents=True, exist_ok=True)

        run_coro = backend.run(input_content, work_dir, np=np, timeout=None)
        if ctx is not None:
            job = await _run_with_progress(ctx, run_coro, f"Running {solver}")
        else:
            job = await run_coro
        _jobs[job.job_id] = job

        if job.error:
            _journal.record("tool_error", "run_simulation", solver=solver,
                            error_message=job.error[:300],
                            input_snapshot=_snap)
        else:
            _journal.record("tool_success", "run_simulation", solver=solver,
                            input_snapshot=_snap)

        result = {
            "job_id": job.job_id, "solver": solver,
            "status": job.status, "work_dir": str(job.work_dir),
            "elapsed": f"{job.elapsed:.2f}s" if job.elapsed else None,
            "critic_review": "approved" if critic_approved else "SKIPPED",
        }
        if job.error:
            result["error"] = job.error[:500]
        if job.status == "completed":
            result["output_files"] = [f.name for f in backend.get_result_files(job)]
            stdout_log = work_dir / "stdout.log"
            if stdout_log.exists():
                text = stdout_log.read_text()
                result["stdout_tail"] = text[-2000:] if len(text) > 2000 else text
        return json.dumps(result, indent=2)

    # ═══════════════════════════════════════════════════════════
    # 5. COUPLING (keep as-is — core workflow)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    async def coupled_solve(
        problem: str = "heat_dd", solver_a: str = "fenics",
        solver_b: str = "fourc", nx: int = 32, ny: int = 32,
        max_iter: int = 20, tol: float = 1e-6,
        relaxation: float = 1.0, params: str = "{}",
        critic_approved: bool = False,
    ) -> str:
        """Cross-solver domain decomposition coupling.

        Domain A (Dirichlet at interface) supports: fenics, ngsolve, skfem, dune.
        Domain B (Neumann at interface) supports: fenics, fourc, ngsolve, skfem, dune.
        Any combination of these works for heat_dd and poisson_dd problems.

        Args:
            problem: 'heat_dd', 'poisson_dd', 'one_way', 'tsi_dd',
                     'poisson_dd_study', 'l_bracket_tsi', 'heat_dd_precice'
            solver_a, solver_b: Backend names
            nx, ny: Elements per direction
            max_iter: Max iterations
            tol: Convergence tolerance
            relaxation: Under-relaxation parameter
            params: JSON with additional parameters
            critic_approved: Set True after critic review
        """
        _get_journal().record("tool_call", "coupled_solve",
                              solver=f"{solver_a}->{solver_b}",
                              physics=problem)
        # Import and delegate to the full coupling implementation
        from tools.coupling import register_coupling_tools
        # The coupling tools are complex — delegate to the original implementation
        from tools.coupling import (
            _heat_domain_decomposition, _poisson_domain_decomposition,
            _oneway_thermal_structural, _twoway_tsi_coupling,
            _relaxation_parameter_study, _l_bracket_tsi,
            _heat_dd_precice_comparison,
        )

        param_dict = json.loads(params)
        backend_a = get_backend(solver_a)
        backend_b = get_backend(solver_b)
        if not backend_a or not backend_b:
            return f"Backend not found: {solver_a} or {solver_b}"

        dispatch = {
            "heat_dd": lambda: _heat_domain_decomposition(backend_a, backend_b, nx, ny, max_iter, tol, relaxation, param_dict),
            "poisson_dd": lambda: _poisson_domain_decomposition(backend_a, backend_b, nx, ny, max_iter, tol, relaxation, param_dict),
            "one_way": lambda: _oneway_thermal_structural(backend_a, backend_b, nx, ny, param_dict),
            "tsi_dd": lambda: _twoway_tsi_coupling(backend_a, backend_b, nx, ny, max_iter, tol, relaxation, param_dict),
            "poisson_dd_study": lambda: _relaxation_parameter_study(backend_a, backend_b, nx, ny, max_iter, tol, param_dict),
            "l_bracket_tsi": lambda: _l_bracket_tsi(backend_a, backend_b, nx, ny, param_dict),
            "heat_dd_precice": lambda: _heat_dd_precice_comparison(backend_a, backend_b, nx, ny, max_iter, tol, relaxation, param_dict),
        }

        if problem not in dispatch:
            return f"Unknown problem: {problem}. Available: {list(dispatch.keys())}"

        return await dispatch[problem]()

    # ═══════════════════════════════════════════════════════════
    # 6. VISUALIZE (replaces 4 visualization tools)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    async def visualize(job_id: str = "", work_dir: str = "",
                        action: str = "summary", field: str = "",
                        ctx: Context = None) -> str:
        """Post-process and visualize simulation results.

        Args:
            job_id: Job ID from run_simulation (or leave empty and set work_dir)
            work_dir: Direct path to results directory
            action: What to do. Options:
                - "summary" — field statistics + results_summary.json content
                  (default; the fastest pulse-check on a finished run)
                - "list" — list every result file under the work dir
                - "plot" — generate a PNG of the named field (needs field=)
                - "validate" — automated sanity checks across the first
                  3 result files: NaN/Inf detection, constant-field
                  detection, suspiciously-large-magnitude detection
                  (>1e15). Use after summary when a field looks wrong.
            field: Specific field name to plot (e.g. 'temperature', 'displacement')
        """
        from core.backend import JobHandle

        # Find work directory
        if job_id and job_id in _jobs:
            wd = _jobs[job_id].work_dir
        elif work_dir:
            wd = Path(work_dir)
        else:
            return "Provide job_id or work_dir"

        if not wd.is_dir():
            return f"Directory not found: {wd}"

        # Collect result files — skip .pvtu (parallel wrappers that can hang PyVista)
        vtu_files = [f for f in sorted(wd.rglob("*.vtu")) if not f.name.endswith(".pvtu")]
        vtu_files += sorted(wd.rglob("*.vtk"))
        vtu_files += sorted(wd.rglob("*.vtp"))
        vtu_files += sorted(wd.rglob("*.xdmf"))
        vtu_files += sorted(wd.rglob("*.bp"))  # ADIOS2/VTX output from dolfinx 0.10+

        if action == "list":
            return "\n".join(f"- {f.relative_to(wd)}" for f in vtu_files) or "No VTU/VTP files found"

        elif action == "summary":
            try:
                from core.post_processing import read_mesh
                import numpy as np
                import re

                # Layer F catalog templates (fenics / ngsolve /
                # skfem / kratos) write a per-run summary at
                # results_summary.json: max field values, dof
                # counts, convergence metrics. Without surfacing
                # this, visualize('summary') returns '[]' when
                # only the JSON summary exists — even though the
                # template printed exactly the info the LLM wants.
                # Audit 2026-06-01.
                summary_artifacts = []
                for js in sorted(wd.rglob("results_summary.json")):
                    try:
                        with open(js) as _f:
                            summary_artifacts.append({
                                "file": str(js.relative_to(wd)),
                                "summary": json.load(_f),
                            })
                    except Exception as e:
                        summary_artifacts.append({
                            "file": str(js.relative_to(wd)),
                            "error": f"unreadable: {e}",
                        })

                # Group VTU files by field type (structure, fluid, ale, etc.)
                # 4C multi-physics outputs separate files per field
                field_groups: dict[str, list] = {}
                for vtu in vtu_files:
                    name = vtu.stem
                    # Detect field type from filename patterns like
                    # structure-00-0, fluid-05-0, ale-03-0
                    match = re.match(r'^(.*?)(?:-\d+)?(?:-\d+)?$', name)
                    group_name = match.group(1) if match else name
                    # Also strip trailing -0 (processor rank)
                    group_name = re.sub(r'-\d+$', '', group_name)
                    field_groups.setdefault(group_name, []).append(vtu)

                def _safe_float(v):
                    """Convert to float, replacing NaN/Inf with string markers."""
                    f = float(v)
                    if np.isnan(f):
                        return "NaN"
                    if np.isinf(f):
                        return "Inf" if f > 0 else "-Inf"
                    return f

                results = []
                # Show summary per field group, using the last timestep
                # Limit to 10 groups to avoid extremely long responses
                group_idx = 0
                for group, files in sorted(field_groups.items())[:10]:
                    group_idx += 1
                    if ctx is not None:
                        try:
                            await ctx.report_progress(
                                group_idx, len(field_groups),
                                f"Reading {group} ({len(files)} timesteps)")
                        except Exception:
                            pass
                    # Use the last file in each group (latest timestep)
                    last_vtu = sorted(files)[-1]
                    try:
                        mesh = read_mesh(last_vtu)
                        fields = {}
                        for fname in mesh.point_data:
                            arr = np.asarray(mesh.point_data[fname])
                            n_nan = int(np.isnan(arr).sum())
                            n_inf = int(np.isinf(arr).sum())
                            finite = arr[np.isfinite(arr)]
                            stats = {
                                "shape": list(arr.shape),
                            }
                            if len(finite) > 0:
                                stats["min"] = _safe_float(finite.min())
                                stats["max"] = _safe_float(finite.max())
                                stats["mean"] = _safe_float(finite.mean())
                            if n_nan > 0:
                                stats["WARNING_NaN"] = f"{n_nan} values"
                            if n_inf > 0:
                                stats["WARNING_Inf"] = f"{n_inf} values"
                            fields[fname] = stats
                        results.append({
                            "field_group": group,
                            "timesteps": len(files),
                            "latest_file": last_vtu.name,
                            "points": mesh.n_points,
                            "fields": fields,
                        })
                    except Exception as e:
                        results.append({
                            "field_group": group,
                            "timesteps": len(files),
                            "error": str(e),
                        })
                # Prepend the JSON-summary artifacts (if any)
                # so the LLM sees them first.
                output = {
                    "results_summary_json": summary_artifacts,
                    "vtu_field_groups": results,
                }
                # If neither populated, drop the wrapper to keep
                # the legacy '[]' empty signal for "nothing here".
                if not summary_artifacts and not results:
                    return "[]"
                return json.dumps(output, indent=2)
            except Exception as e:
                return f"Error reading results: {e}"

        elif action == "plot" and field:
            try:
                from core.post_processing import read_mesh, plot_field
                vtu = vtu_files[-1] if vtu_files else None
                if not vtu:
                    return "No VTU files to plot"
                mesh = read_mesh(vtu)
                plot_path = wd / f"plot_{field}.png"
                plot_field(mesh, field, plot_path, title=field, spatial_dim=2)
                return f"Plot saved: {plot_path}"
            except Exception as e:
                return f"Error plotting: {e}"

        elif action == "validate":
            # Automated sanity checks on results
            try:
                from core.post_processing import read_mesh
                import numpy as np
                checks = []
                for vtu in vtu_files[:3]:
                    mesh = read_mesh(vtu)
                    for name in mesh.point_data:
                        arr = np.asarray(mesh.point_data[name])
                        issues = []
                        if np.any(np.isnan(arr)):
                            issues.append(f"CONTAINS NaN ({np.isnan(arr).sum()} values)")
                        if np.any(np.isinf(arr)):
                            issues.append(f"CONTAINS Inf ({np.isinf(arr).sum()} values)")
                        if arr.max() == arr.min() and len(arr) > 1:
                            issues.append(f"CONSTANT FIELD (all values = {arr.max():.6e})")
                        if arr.max() > 1e15:
                            issues.append(f"SUSPICIOUSLY LARGE max = {arr.max():.2e}")
                        status = "PASS" if not issues else "ISSUES FOUND"
                        checks.append(f"- {name} in {vtu.name}: {status}" +
                                     (f"\n  " + "\n  ".join(issues) if issues else ""))
                return "## Results Validation\n\n" + "\n".join(checks)
            except Exception as e:
                return f"Validation error: {e}"

        return "Usage: visualize(job_id, action='summary'|'plot'|'list'|'validate', field='')"

    # ═══════════════════════════════════════════════════════════
    # 7. DEVELOPER (replaces 3 developer tools)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def developer(action: str, solver: str = "", keyword: str = "") -> str:
        """Developer tools: architecture, source files, capabilities matrix.

        Args:
            action: What to surface. Options:
                - "architecture" — extension points + source-tree
                  layout for the requested solver
                - "files" — source-file listing filtered by keyword
                - "capabilities" — full backend × physics × variant
                  matrix dump
            solver: Backend name
            keyword: File pattern for "files" action
        """
        if action == "files":
            _get_journal().record("source_read", "developer",
                                  solver=solver, notes=f"keyword={keyword}")
        if action == "architecture" and solver:
            from tools.developer import _SOURCE_LOCATIONS
            info = _SOURCE_LOCATIONS.get(solver, {})
            if not info:
                return f"Unknown solver: {solver}"
            return json.dumps(info, indent=2)

        elif action == "capabilities":
            # All registered backends so the developer-side
            # capabilities listing matches discover('capabilities'):
            # consistent visibility across both surfaces.
            # (Audit 2026-06-02.)
            lines = []
            for b in all_backends():
                status, _ = b.check_availability()
                tag = "" if status.value == "available" else f" *[{status.value}]*"
                physics = [p.name for p in b.supported_physics()]
                lines.append(f"**{b.display_name()}**{tag}: {', '.join(physics)}")
            return "\n".join(lines)

        elif action == "files" and solver:
            # Check if solver has a source root set via env var
            from tools.developer import _SOURCE_LOCATIONS
            info = _SOURCE_LOCATIONS.get(solver, {})
            source_root = info.get("root", "")
            source_env = info.get("source_env_var", "")

            # If keyword starts with "src/" or similar, search the solver source tree
            if keyword and source_root and Path(source_root).is_dir():
                base = Path(source_root)
                pattern = keyword
                files = sorted(base.rglob(pattern))[:30]
                if files:
                    return "\n".join(f"- {f.relative_to(base)} ({f.stat().st_size}b)" for f in files)

            # Default: search the MCP backend files
            base = Path(__file__).resolve().parents[1] / "backends" / solver
            if not base.exists():
                hint = f"\nTo browse {solver} source code, set {source_env} in .claude/settings.json" if source_env else ""
                return f"No source directory for {solver}{hint}"
            pattern = keyword or "*.py"
            files = sorted(base.rglob(pattern))
            result = "\n".join(f"- {f.relative_to(base)} ({f.stat().st_size}b)" for f in files[:20])
            if source_env and not (source_root and Path(source_root).is_dir()):
                result += f"\n\nNote: Set {source_env} env var to browse the full {solver} source tree"
            return result

        return "Usage: developer(action='architecture'|'capabilities'|'files', solver='')"

    # ═══════════════════════════════════════════════════════════
    # 8. PREPARE (meta-tool: knowledge + examples + template in one call)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def prepare_simulation(solver: str, physics: str) -> str:
        """Prepare everything needed to set up a simulation — in ONE call.

        Returns: knowledge + real test file examples + generated template.
        This eliminates 3 separate tool calls before every simulation.

        Supports fuzzy matching: e.g. 'magnetostatics' finds 'maxwell',
        'thermal' finds 'heat', 'elasticity' finds 'linear_elasticity'.

        Args:
            solver: Backend name (e.g. 'fourc', 'fenics', 'ngsolve')
            physics: Physics type (e.g. 'poisson', 'particle_pd', 'navier_stokes',
                     'magnetostatics', 'thermal', 'elasticity')
        """
        _get_journal().record("knowledge_lookup", "prepare_simulation",
                              solver=solver, physics=physics)
        parts = []

        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        # Fuzzy match: find the best matching physics name
        matched_physics = _fuzzy_match_physics(backend, physics)
        if not matched_physics:
            # Empty / whitespace-only query — surface the
            # available-physics list so the LLM can pick a real
            # name. Without this guard prepare_simulation silently
            # builds a half-empty response for a physics it never
            # had. Audit 2026-06-01.
            available = ", ".join(
                p.name for p in backend.supported_physics())
            return (f"Empty physics query. Available physics in "
                    f"{backend.display_name()}: {available}")
        if matched_physics != physics:
            parts.append(f"*Note: '{physics}' matched to '{matched_physics}'*\n")

        # 0. Also available on — show which other backends support this physics (informational)
        alternatives = _list_alternative_solvers(solver, matched_physics)
        if alternatives:
            parts.append("## Also available on\n" + alternatives + "\n")

        # 1. Knowledge
        # Render pitfalls OUTSIDE the JSON dump so the 3000-char
        # truncation does not silently hide them. Audited
        # 2026-06-01: large KNOWLEDGE blocks like ngsolve::
        # hyperelasticity (7 pitfalls / ~4.4 KB) and skfem::
        # poisson (6 / ~4 KB) showed 0/N pitfalls fully visible
        # to the LLM client; every Layer F fix landed but never
        # reached the prepare_simulation surface that's meant to
        # teach the agent.
        k = backend.get_knowledge(matched_physics)
        if k:
            pitfalls_separate = None
            json_payload = k
            if isinstance(k, dict) and isinstance(k.get("pitfalls"), list):
                pitfalls_separate = k["pitfalls"]
                json_payload = {kk: vv for kk, vv in k.items() if kk != "pitfalls"}
            # After the pitfalls carve-out the remaining JSON
            # is description / spaces / solver / elements /
            # materials / time_integration / typical_experiments.
            # Most backends sit < 1.5 KB but fourc::solid_mechanics
            # is ~12 KB (rich plasticity_models + materials dict).
            # The old 3000-char cap silently dropped most of that.
            # Match the TEMPLATE_LIMIT of 12000 set above so the
            # LLM gets the full materials table. Audit 2026-06-01.
            KNOWLEDGE_LIMIT = 16000
            payload_text = json.dumps(json_payload, indent=2, default=str)
            payload_truncated = len(payload_text) > KNOWLEDGE_LIMIT
            payload_body = payload_text[:KNOWLEDGE_LIMIT]
            payload_suffix = (f"\n... [truncated {len(payload_text) - KNOWLEDGE_LIMIT} chars]"
                              if payload_truncated else "")
            parts.append("## Knowledge\n```json\n"
                         + payload_body + payload_suffix
                         + "\n```\n")
            if pitfalls_separate:
                bullets = "\n".join(f"- {p}" for p in pitfalls_separate)
                parts.append(
                    f"### Pitfalls ({len(pitfalls_separate)})\n{bullets}\n")

        # 1b. General input-format pitfalls (ExodusII IDs, FUNCT syntax, etc.)
        # These apply to ALL physics in this solver, not just the current one
        general_k = backend.get_knowledge("input_format")
        if isinstance(general_k, dict):
            gp = general_k.get("general_pitfalls")
            if gp:
                pitfall_text = "\n".join(f"- {p}" for p in gp)
                parts.append(f"## General Input Pitfalls\n{pitfall_text}\n")

        # 2. Real test file examples
        from tools.knowledge import _find_reference_test_files
        ref = _find_reference_test_files(solver, matched_physics)
        if ref:
            parts.append(ref)

        # 3. Generated template
        # Templates can exceed 3000 chars on the harder physics
        # (ngsolve hdivdiv 3.2KB, nonlinear_elasticity 3.4KB,
        # fenics navier_stokes 3.8KB ...) — truncating at 3000
        # cuts off the trailing solver / output / summary
        # blocks the LLM needs to actually run the template.
        # Raise to 12000 chars so the standard Layer F-class
        # templates (typically 2-5KB) render in full. Audit
        # 2026-06-01.
        TEMPLATE_LIMIT = 12000
        for p in backend.supported_physics():
            if p.name == matched_physics and p.template_variants:
                try:
                    content = backend.generate_input(matched_physics, p.template_variants[0], {})
                    fmt = backend.input_format().value
                    truncated = len(content) > TEMPLATE_LIMIT
                    body = content[:TEMPLATE_LIMIT]
                    suffix = (f"\n... [truncated {len(content) - TEMPLATE_LIMIT} chars]"
                              if truncated else "")
                    stub_tag = _stub_template_tag(content, fmt)
                    parts.append(f"## Template ({p.template_variants[0]}){stub_tag}\n```{fmt}\n{body}{suffix}\n```\n")
                except Exception as exc:
                    # Surface the failure: the catalog claims a
                    # template exists (p.template_variants is
                    # non-empty) but the generator raised. The
                    # old `except Exception: pass` silently
                    # produced an LLM-visible "successful" reply
                    # with no template and no hint that the
                    # generator was broken — masking Layer-F
                    # class regressions both from the LLM and
                    # the developer running it. (Audit 2026-06-02.)
                    parts.append(
                        f"## Template ({p.template_variants[0]})\n"
                        f"⚠ Template generation FAILED for "
                        f"`{matched_physics}/{p.template_variants[0]}`: "
                        f"`{type(exc).__name__}: {exc}`\n\n"
                        f"This is a catalog generator bug — the "
                        f"physics is advertised in "
                        f"`{backend.display_name()}.supported_physics()` "
                        f"but `generate_input` raised. The other "
                        f"sections of this response (knowledge, "
                        f"pitfalls, real-test references) are still "
                        f"valid; only the auto-generated template "
                        f"is missing.\n")
                break

        if not parts:
            # List available physics as hint
            avail = [p.name for p in backend.supported_physics()]
            return f"No information found for '{physics}' in {solver}. Available physics: {', '.join(avail)}"

        return f"# Preparation for {matched_physics} on {solver}\n\n" + "\n---\n".join(parts)

    # ═══════════════════════════════════════════════════════════
    # 9. TRANSFER FIELD (keep — needed for coupling)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    async def transfer_field(
        source_vtu: str, field_name: str,
        interface_coord: float, interface_axis: int = 0,
        target_format: str = "json", output_path: str = "",
    ) -> str:
        """Extract field values at an interface from a VTU file and format for transfer.

        Universal data connector for cross-solver coupling. Reads VTU output
        from any solver, extracts values at the interface plane, and formats
        them for the target solver's expected input shape.

        Args:
            source_vtu: Path to VTU result file from the source solver.
            field_name: Field to extract (e.g. 'temperature', 'displacement').
            interface_coord: Coordinate value defining the interface plane.
            interface_axis: Axis perpendicular to interface (0=x, 1=y, 2=z).
            target_format: Output format. Options:
                - "json"        — interface coordinates + values (default)
                - "fenics"      — Python BoundaryCondition snippet (Dirichlet
                                  at this interface), saved as .py
                - "4c_neumann"  — 4C-format YAML snippet for a Neumann
                                  boundary condition, saved as .yaml
            output_path: Where to save the formatted output. If empty,
                auto-generated next to the source VTU as
                'interface_<field_name>.<ext>'.

        Returns:
            A summary string with the interface min/max/mean and the path
            of the saved file.
        """
        from core.field_transfer import extract_interface_from_vtu

        vtu_path = Path(source_vtu)
        if not vtu_path.exists():
            return f"VTU file not found: {source_vtu}"

        try:
            iface = extract_interface_from_vtu(
                vtu_path, field_name, interface_coord, interface_axis)
        except Exception as e:
            return f"Error extracting interface: {e}"

        if not output_path:
            output_path = str(
                vtu_path.parent / f"interface_{field_name}.json")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if target_format == "json":
            iface.to_json(out)
        elif target_format == "fenics":
            from core.field_transfer import format_for_fenics
            code = format_for_fenics(
                iface, "dirichlet", interface_axis, interface_coord)
            out = out.with_suffix(".py")
            out.write_text(code)
        elif target_format == "4c_neumann":
            from core.field_transfer import format_for_4c_neumann
            yaml_snippet = format_for_4c_neumann(iface)
            out = out.with_suffix(".yaml")
            out.write_text(yaml_snippet)
        else:
            return (f"Unknown format: {target_format}. Use 'json', "
                    "'fenics', or '4c_neumann'.")

        vals = iface.values
        summary = (
            f"## Field Transfer: {field_name}\n\n"
            f"- Source: {vtu_path.name}\n"
            f"- Interface: {'xyz'[interface_axis]}={interface_coord}\n"
            f"- Nodes: {len(iface.coordinates)}\n"
            f"- Values: [{vals.min():.6e}, {vals.max():.6e}], "
            f"mean={vals.mean():.6e}\n"
            f"- Output: {out}\n"
        )
        if iface.normal_fluxes is not None:
            fl = iface.normal_fluxes
            summary += f"- Fluxes: [{fl.min():.6e}, {fl.max():.6e}]\n"
        return summary

    # ═══════════════════════════════════════════════════════════
    # 10. MESH (keep — needed for Gmsh)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def generate_mesh(geometry: str, mesh_size: float = 0.1,
                      output_dir: str = "") -> str:
        """Generate a mesh using Gmsh for non-trivial geometries.

        Args:
            geometry: One of the built-in geometries:
                - "l_domain"          — 2D L-shaped domain
                - "plate_with_hole"   — 2D plate with circular hole
                - "channel_cylinder"  — 2D channel with cylindrical obstacle
                (No "custom" passthrough yet — passing any other name
                returns a 'Unknown geometry' message with this list.)
            mesh_size: Target element size
            output_dir: Where to save (auto if empty)
        """
        from tools.mesh_generation import register_mesh_tools
        # Delegate to original. Importer names must match the
        # functions in tools.mesh_generation EXACTLY — the prior
        # _generate_channel_cylinder_2d (without 'with') did not
        # exist there (actual name is _generate_channel_with_
        # cylinder_2d) and the ImportError short-circuited the
        # dispatch dict for ALL three geometries, including
        # l_domain and plate_with_hole. (Audit 2026-06-01.)
        try:
            from tools.mesh_generation import (
                _generate_l_domain_2d,
                _generate_plate_with_hole_2d,
                _generate_channel_with_cylinder_2d,
            )
            generators = {
                "l_domain": _generate_l_domain_2d,
                "plate_with_hole": _generate_plate_with_hole_2d,
                "channel_cylinder": _generate_channel_with_cylinder_2d,
            }
            gen = generators.get(geometry)
            if gen:
                # The three generators have DIFFERENT positional
                # signatures (l_domain: (mesh_size, output_path);
                # plate_with_hole: (mesh_size, radius, width,
                # height, output_path); channel_with_cylinder:
                # (mesh_size, cyl_radius, center, length, height,
                # output_path)). Always pass output_path via
                # keyword. The functions expect a FULL FILE path
                # (gmsh.write needs an extension to pick the
                # output format) — append "<geom>.msh" to the
                # directory the user passed in. (Audit 2026-06-01.)
                out_dir = Path(output_dir or str(_OUTPUT_DIR / "meshes"))
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{geometry}.msh"
                result = gen(mesh_size, output_path=out_file)
                # generators return either Path or (Path, n_nodes,
                # n_elements). Surface a friendly summary.
                if isinstance(result, tuple):
                    path, *meta = result
                    return f"mesh: {path} (nodes={meta[0]}, elements={meta[1]})"
                return str(result) if result is not None else "ok"
            return f"Unknown geometry: {geometry}. Available: {list(generators.keys())}"
        except Exception as e:
            return f"Error: {e}"

    # ═══════════════════════════════════════════════════════════
    # 12. BACKEND DISCOVERY
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def reload_catalog() -> str:
        """Hot-reload the per-backend KNOWLEDGE dicts from disk.

        Closes the gap identified by the
        mcp-catalog-staleness-runtime-isolation post-mortem
        (2026-06-01): the MCP server normally imports
        src/backends/<be>/generators/<physics>.py modules ONCE at
        startup and never refreshes them, so catalog edits made
        during a long-running session are invisible. Postmortems
        in data/postmortems/ are scanned on every request (already
        hot), but pitfall dicts are not.

        This tool walks every imported `backends.<be>.generators.*`
        and `backends.<be>.backend` module, runs importlib.reload
        on each, and re-runs load_all_backends() so the registry
        re-binds the backend objects to the refreshed module
        attributes. After the call, the very next
        mcp__open-fem-agent__knowledge call returns the on-disk
        catalog without having to restart Claude Code.

        Returns a one-line summary of which modules were
        successfully reloaded vs which raised, so the caller can
        tell when a syntax error in a newly-edited generator
        prevented its module from re-importing (in that case the
        OLD dict is still served from the previous import).
        """
        import importlib
        import sys
        reload_ok: list[str] = []
        reload_fail: list[tuple[str, str]] = []

        # Reload data/*_knowledge.py first (sourced by some
        # backends).
        for mod_name in list(sys.modules.keys()):
            if (mod_name.endswith("_knowledge")
                    and not mod_name.startswith("backends.")):
                try:
                    importlib.reload(sys.modules[mod_name])
                    reload_ok.append(mod_name)
                except Exception as exc:  # noqa: BLE001
                    reload_fail.append((mod_name, str(exc)))

        # Reload every imported backends.* submodule.
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("backends.") and "." in mod_name:
                try:
                    importlib.reload(sys.modules[mod_name])
                    reload_ok.append(mod_name)
                except Exception as exc:  # noqa: BLE001
                    reload_fail.append((mod_name, str(exc)))

        # Re-bind backend objects to refreshed modules.
        try:
            from core.registry import load_all_backends
            load_all_backends()
            re_register = "ok"
        except Exception as exc:  # noqa: BLE001
            re_register = f"FAILED: {exc}"

        msg = (f"reload_catalog: {len(reload_ok)} modules reloaded, "
               f"{len(reload_fail)} failed; "
               f"re-register backends: {re_register}.")
        if reload_fail:
            msg += "\n\nFailures:\n" + "\n".join(
                f"  - {n}: {e[:200]}" for n, e in reload_fail[:10])
        return msg

    @mcp.tool()
    def rediscover_backends(confirm: bool = False) -> str:
        """Probe the system for solver backends and report findings.

        Searches pip packages, conda environments, common build directories,
        and source roots. In developer mode, reports git branch and status.

        Args:
            confirm: If True, save the discovered config for future sessions.
                     If False (default), just report what was found.
        """
        from core.autodiscovery import (
            discover_backends as _discover,
            format_discovery,
            save_discovered_config,
        )

        results = _discover()
        report = format_discovery(results)

        if confirm:
            path = save_discovered_config(results)
            report += f"\n\nConfig saved to `{path}`. Will be used on next restart."
        else:
            found_count = sum(1 for r in results if r.found)
            if found_count > 0:
                report += (
                    f"\n\nCall `rediscover_backends(confirm=True)` to save this "
                    f"config for future sessions."
                )

        return report

    # ═══════════════════════════════════════════════════════════
    # 13. SESSION INSIGHTS (knowledge capture)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def session_insights(action: str = "review", path: str = "") -> str:
        """Review knowledge discovered during this session or from saved
        journals on disk.

        Two flows are supported:

        * In-session flow: call ``review`` -> ``approve_all`` /
          ``reject_all`` during the live MCP session to surface
          candidates from the current journal and save approved ones
          to ``data/community_knowledge/pending/``.
        * Ingest flow: call ``ingest`` with ``path`` pointing at a
          previously-saved session journal (``data/sessions/session_*.json``,
          which the server writes on shutdown) or at a directory of
          such files.  Candidates are surfaced just like ``review``
          and can be approved with ``approve_all``.

        Args:
            action:
                - "review" — show candidate knowledge from the current
                  session for approval
                - "ingest" — load saved journal(s) from ``path`` and
                  analyse them; requires ``path``
                - "approve_all" — approve all pending candidates and
                  save to community_knowledge/pending/
                - "reject_all" — dismiss all pending candidates
                - "stats" — current session statistics
            path: file or directory used by the ``ingest`` action;
                ignored otherwise.  Directories are scanned for
                ``session_*.json``.
        """
        from pathlib import Path as _Path

        from core.session_journal import get_journal
        from core.session_analyzer import (
            CandidateKnowledge,
            analyze_journal,
            analyze_journal_file,
            filter_against_existing,
            format_candidates,
        )

        journal = get_journal()

        if action == "stats":
            return json.dumps({
                "session_id": journal.session_id,
                "events": len(journal.events),
                "errors": journal.error_count,
                "solvers_used": sorted(journal.solvers_used),
                "physics_used": sorted(journal.physics_used),
                "duration_seconds": round(journal.duration_seconds, 1),
            }, indent=2)

        if action == "review":
            if len(journal.events) < 3:
                return "Session too short for knowledge extraction (< 3 tool calls)."
            candidates = analyze_journal(journal)
            # Filter against existing knowledge
            existing = _collect_existing_pitfalls()
            candidates = filter_against_existing(candidates, existing)
            if not candidates:
                return "No new knowledge candidates discovered in this session."
            # Store candidates for potential approval
            _pending_candidates.clear()
            _pending_candidates.extend(candidates)
            return format_candidates(candidates)

        if action == "ingest":
            if not path:
                return (
                    "Usage: session_insights('ingest', path='<file_or_dir>')\n"
                    "Point at a session journal saved by the MCP server "
                    "(data/sessions/session_*.json) or a directory of "
                    "such files."
                )
            p = _Path(path)
            if not p.exists():
                return f"Path not found: {p}"
            sources: list[_Path] = (
                sorted(p.glob("session_*.json")) if p.is_dir() else [p]
            )
            if not sources:
                return f"No session_*.json files found in {p}"
            all_candidates: list = []
            errors: list[str] = []
            for s in sources:
                try:
                    all_candidates.extend(analyze_journal_file(s))
                except Exception as e:
                    # repr(e) keeps the exception type so a contributor
                    # can tell `KeyError('events')` from a `FileNotFoundError`.
                    errors.append(f"{s.name}: {e!r}")
            # Cross-source de-duplication on a normalised key (the in-file
            # analyzer runs fuzzy dedup already; cross-file dedup needs to
            # match that contract or near-identical entries from N journals
            # all survive as separate candidates).
            import re as _re
            _retry_re = _re.compile(r"\s*\(retry \d+\)\s*$", _re.IGNORECASE)
            def _norm_title(t: str) -> str:
                return " ".join(_retry_re.sub("", t).lower().split())
            best: dict[tuple[str, str, str], CandidateKnowledge] = {}
            for c in all_candidates:
                key = (
                    c.category.strip().lower(),
                    (c.solver or "").strip().lower(),
                    _norm_title(c.title),
                )
                if key not in best or c.confidence > best[key].confidence:
                    best[key] = c
            candidates = list(best.values())
            existing = _collect_existing_pitfalls()
            candidates = filter_against_existing(candidates, existing)
            _pending_candidates.clear()
            _pending_candidates.extend(candidates)
            header = (
                f"Ingested {len(sources)} journal file(s); "
                f"{len(all_candidates)} raw candidates -> "
                f"{len(candidates)} novel after dedup + filter.\n"
            )
            if errors:
                header += "Errors:\n  " + "\n  ".join(errors) + "\n"
            if not candidates:
                return header + "No new candidates."
            return header + format_candidates(candidates)

        if action == "approve_all":
            if not _pending_candidates:
                return "No pending candidates. Call session_insights('review') first."
            saved = _save_candidates(_pending_candidates, journal.session_id)
            count = len(_pending_candidates)
            _pending_candidates.clear()
            return f"Approved {count} candidate(s). Saved to: {saved}"

        if action == "reject_all":
            count = len(_pending_candidates)
            _pending_candidates.clear()
            return f"Rejected {count} candidate(s)."

        # The Actions list must match the docstring + the
        # actual dispatch branches in this function. Audit
        # 2026-06-01: 'ingest' was documented but missing
        # from this usage hint, so LLMs that hit an invalid
        # action never learned that ingest exists.
        return (
            "Usage: session_insights(action, path='')\n"
            "Actions: review, ingest, approve_all, reject_all, stats\n"
            "Use ingest with path=<session.json|dir> to "
            "analyse saved session journals."
        )

    # Storage for pending candidates between review and approve
    _pending_candidates: list = []


def _collect_existing_pitfalls() -> list[str]:
    """Gather all existing pitfall strings for novelty checking.

    Includes both built-in knowledge AND community contributions.

    Uses all_backends() (not available_backends): the pitfall
    library is a static catalog and the novelty check should
    compare against EVERY known pitfall, including those of
    backends the user has not installed locally. Filtering by
    availability would let a candidate that duplicates a
    dune-fem pitfall slip through as "novel" on any host
    without dune-fem. (Audit 2026-06-02.)
    """
    pitfalls = []
    try:
        for b in all_backends():
            for p in b.supported_physics():
                k = b.get_knowledge(p.name)
                if k and isinstance(k, dict) and "pitfalls" in k:
                    for pit in k["pitfalls"]:
                        if isinstance(pit, str):
                            pitfalls.append(pit)
                        elif isinstance(pit, dict) and "text" in pit:
                            pitfalls.append(pit["text"])
    except Exception:
        pass
    # Also include community contributions
    for c in _load_community_knowledge():
        pitfalls.append(c.get("title", ""))
    return pitfalls


def _load_community_knowledge(solver: str = "") -> list[dict]:
    """Load approved community knowledge from pending/ directory.

    Returns list of candidate dicts. Optionally filter by solver.
    """
    from pathlib import Path
    pending_dir = Path(__file__).parent.parent.parent / "data" / "community_knowledge" / "pending"
    if not pending_dir.exists():
        return []
    entries = []
    for f in sorted(pending_dir.glob("session_*.json")):
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list):
                for entry in data:
                    if solver and entry.get("solver", "") != solver:
                        continue
                    entries.append(entry)
        except (json.JSONDecodeError, OSError):
            continue
    return entries


def _save_candidates(candidates: list, session_id: str) -> str:
    """Save approved candidates to community_knowledge/pending/."""
    from pathlib import Path
    pending_dir = Path(__file__).parent.parent.parent / "data" / "community_knowledge" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for c in candidates:
        entries.append(c.to_dict())

    path = pending_dir / f"session_{session_id}.json"
    path.write_text(json.dumps(entries, indent=2, default=str))
    return str(path)


# ═══════════════════════════════════════════════════════════════
# Helper functions for knowledge (copied from original tools)
# ═══════════════════════════════════════════════════════════════

def _capture_knowledge_fn(fn_name: str) -> str:
    """Reach into tools.knowledge.register_knowledge_tools to pull
    out one of the inline get_*_knowledge closure bodies.

    The three knowledge providers (coupling / TSI / preCICE) live
    inside register_knowledge_tools as nested @mcp.tool() closures,
    not as module-level functions. The consolidated tool surface
    needs to call them outside FastMCP's tool-dispatch path, so
    this helper builds a throwaway FastMCP instance, monkey-
    patches its `tool` decorator to capture every registered
    function by name, runs register_knowledge_tools, then calls
    the requested one.

    Failures here used to be wrapped in `except Exception: pass`
    and produced a bare "...knowledge not available" string to
    the LLM — silent degradation that hid genuine breakage of the
    capture trick (FastMCP API change, register_knowledge_tools
    refactor, missing tools.knowledge module, ...). The wrapper
    now surfaces the exception so the LLM and the developer can
    diagnose. (Audit 2026-06-02.)
    """
    try:
        from tools.knowledge import register_knowledge_tools
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        return (f"⚠ Cannot load knowledge subsystem: "
                f"`{type(exc).__name__}: {exc}`")
    mcp = FastMCP("tmp")
    captured: dict = {}
    orig = mcp.tool

    def cap(*a, **kw):
        d = orig(*a, **kw)

        def w(fn):
            r = d(fn)
            captured[fn.__name__] = fn
            return r
        return w

    mcp.tool = cap
    try:
        register_knowledge_tools(mcp)
    except Exception as exc:
        return (f"⚠ register_knowledge_tools failed while capturing "
                f"`{fn_name}`: `{type(exc).__name__}: {exc}`")
    if fn_name not in captured:
        return (f"⚠ `{fn_name}` was not registered by "
                f"register_knowledge_tools. Captured: "
                f"{sorted(captured.keys())}")
    try:
        return captured[fn_name]()
    except Exception as exc:
        return (f"⚠ `{fn_name}()` raised: "
                f"`{type(exc).__name__}: {exc}`")


def _get_coupling_knowledge():
    """Return coupling knowledge string (or a visible error block)."""
    return _capture_knowledge_fn("get_coupling_knowledge")


def _get_tsi_knowledge():
    return _capture_knowledge_fn("get_tsi_knowledge")


def _get_precice_knowledge():
    return _capture_knowledge_fn("get_precice_knowledge")


