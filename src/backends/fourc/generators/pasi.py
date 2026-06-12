"""Particle-Structure Interaction (PASI) generator for 4C.

Covers coupling of a particle method (DEM, SPH, or peridynamics) with
structural finite elements.  Particles interact with the structural
surface via contact or coupling conditions.  Applications include
impact of granular material on structures, sloshing of particle-laden
fluids against deformable containers, and blast loading simulations.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class PASIGenerator(BaseGenerator):
    """Generator for Particle-Structure Interaction problems in 4C."""

    module_key = "pasi"
    display_name = "Particle-Structure Interaction (PASI)"
    problem_type = "Particle_Structure_Interaction"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Particle-Structure Interaction (PASI) couples a particle "
                "method (DEM, SPH, or peridynamics) with a structural finite "
                "element discretisation.  The particles exert forces on the "
                "structural surface (impact, pressure) and the structure "
                "provides a moving boundary for the particles.  The PROBLEM "
                "TYPE is 'Particle_Structure_Interaction'.  Required sections "
                "include STRUCTURAL DYNAMIC for the FEM field, PARTICLE "
                "DYNAMIC for the particle field, and PASI DYNAMIC for the "
                "coupling parameters.  A BINNING STRATEGY section is needed "
                "for efficient particle-structure contact detection."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "STRUCTURAL DYNAMIC",
                "PARTICLE DYNAMIC",
                "PASI DYNAMIC",
                "BINNING STRATEGY",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "PARTICLE DYNAMIC/INITIAL AND BOUNDARY CONDITIONS",
                "PARTICLE DYNAMIC/SPH",
                "IO",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/STRUCTURE",
                "IO/RUNTIME VTK OUTPUT/PARTICLES",
            ],
            "materials": {
                "MAT_ParticleMaterialDEM": {
                    "description": (
                        "Discrete element method (DEM) particle material.  "
                        "Defines particle density, radius, and contact "
                        "stiffness for granular particles."
                    ),
                    "parameters": {
                        "DENS": {
                            "description": "Particle density",
                            "range": "> 0",
                        },
                        "RADIUS": {
                            "description": "Particle radius",
                            "range": "> 0",
                        },
                        "YOUNG": {
                            "description": "Contact Young's modulus",
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Contact Poisson's ratio",
                            "range": "[0, 0.5)",
                        },
                    },
                },
                "MAT_ParticleMaterialSPH": {
                    "description": (
                        "SPH particle material for fluid-like particles "
                        "interacting with structural surfaces."
                    ),
                    "parameters": {
                        "DENS": {
                            "description": "Particle density",
                            "range": "> 0",
                        },
                        "DYNAMICVISCOSITY": {
                            "description": "Dynamic viscosity",
                            "range": "> 0",
                        },
                        "BULK_MODULUS": {
                            "description": "Bulk modulus for equation of state",
                            "range": "> 0",
                        },
                    },
                },
                "MAT_ElastHyper / MAT_Struct_StVenantKirchhoff": {
                    "description": (
                        "Standard structural material for the FEM "
                        "domain.  Any structural material supported by "
                        "4C can be used."
                    ),
                },
            },
            "solver": {
                "structure_solver": {
                    "type": "UMFPACK or Belos",
                    "notes": "Solver for the structural FEM equations.",
                },
            },
            "time_integration": {
                "PASI_DYNAMIC": (
                    "Controls the coupled time stepping.  TIMESTEP, "
                    "NUMSTEP, MAXTIME define the time loop.  The "
                    "particle and structure fields are advanced together."
                ),
                "PARTICLE_DYNAMIC": (
                    "Controls particle-specific settings: time integrator "
                    "(e.g. VelocityVerlet), particle type, interaction "
                    "model."
                ),
                "STRUCTURAL_DYNAMIC": (
                    "Standard structural time integration.  For impact "
                    "problems, use explicit dynamics (GenAlpha with "
                    "appropriate spectral radius)."
                ),
            },
            "pitfalls": [
                (
                    "[Input] BINNING STRATEGY is MANDATORY "
                    "for PASI — controls spatial partitioning "
                    "for particle-structure contact search. "
                    "Set BIN_SIZE_LOWER_BOUND to roughly the "
                    "particle diameter. Signal: omitting "
                    "BINNING STRATEGY aborts with 'no "
                    "binning strategy defined for PASI' at "
                    "setup; the contact-search algorithm "
                    "needs the spatial grid. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] Particle interaction model (DEM "
                    "contact, SPH, etc.) must be COMPATIBLE "
                    "with the structural surface. DEM "
                    "particles use Hertz/hooke contact; SPH "
                    "uses boundary particles. Signal: "
                    "configuring DEM particles with an SPH "
                    "boundary surface (or vice versa) "
                    "produces 'interaction type mismatch' "
                    "or silent zero-force coupling. Match "
                    "particle TYPE to structural surface "
                    "treatment. (Audit 2026-06-02.)"
                ),
                (
                    "[Mesh] Structural elements at the "
                    "particle-structure interface need "
                    "sufficiently fine mesh resolution to "
                    "resolve individual particle contacts. "
                    "Signal: h_structure >> particle "
                    "diameter spreads each particle's "
                    "contact force over too few "
                    "structural DOFs — stress field looks "
                    "uniform instead of showing distinct "
                    "contact patches. Target h_structure "
                    "~ 0.5 * particle_diameter near "
                    "interfaces. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Time step must satisfy "
                    "BOTH the structural CFL AND the "
                    "particle stability limit (depends on "
                    "contact stiffness k_c and particle "
                    "mass m_p): dt < pi * sqrt(m_p/k_c) / "
                    "10. Signal: a coupled simulation that "
                    "respects structural CFL but exceeds "
                    "particle stability shows particle "
                    "positions oscillating wildly within "
                    "each structural step. Take "
                    "min(dt_struct, dt_particle). (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] PASI requires SEPARATE geometry "
                    "sections for structural mesh AND "
                    "particle initial positions. The "
                    "PARTICLES section defines positions, "
                    "types, and initial velocities. "
                    "Signal: putting particles inside "
                    "STRUCTURE GEOMETRY raises 'unexpected "
                    "particle entry in structural mesh' at "
                    "parse; both fields need their own "
                    "geometry blocks. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] For SPH-structure coupling, "
                    "boundary handling at the structural "
                    "surface requires SPECIAL attention — "
                    "ghost particles or repulsive boundary "
                    "conditions. Signal: a plain SPH-"
                    "structure coupling without boundary "
                    "treatment lets SPH particles "
                    "PENETRATE the structure (kernel "
                    "support extends across the wall, "
                    "computing density from non-existent "
                    "neighbours); add Adami boundary "
                    "particles or a repulsive wall force. "
                    "(Audit 2026-06-02.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "dem_impact_3d",
                    "description": (
                        "DEM granular particles impacting a deformable "
                        "structural plate.  Tests particle-structure "
                        "contact force transfer, structural response to "
                        "distributed impact loads, and time step "
                        "stability."
                    ),
                    "template_variant": "dem_impact_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "dem_impact_3d",
                "description": (
                    "3-D DEM-structure interaction: granular particles "
                    "impacting a deformable plate.  Hertz contact, "
                    "SOLID HEX8 structure, UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "dem_impact_3d") -> str:
        templates = {
            "dem_impact_3d": self._template_dem_impact_3d,
        }
        if variant == "default":
            variant = "dem_impact_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_dem_impact_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D Particle-Structure Interaction (PASI) -- DEM Impact
            #
            # Discrete element particles (granular) impact a deformable
            # structural plate.  Particles use Hertz contact with the
            # structural surface.
            #
            # Mesh: exodus file with:
            #   element_block 1 = structural plate (HEX8)
            #   node_set 1 = plate fixed boundary
            # Particles: defined in PARTICLES section
            # ---------------------------------------------------------------
            TITLE:
              - "3-D particle-structure interaction -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Particle_Structure_Interaction"
            IO:
              STDOUTEVERY: <stdout_interval>
              STRUCT_STRESS: "Cauchy"
              STRUCT_STRAIN: "GL"
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/STRUCTURE:
              OUTPUT_STRUCTURE: true
              DISPLACEMENT: true
            IO/RUNTIME VTK OUTPUT/PARTICLES:
              OUTPUT_PARTICLES: true

            # == Structure (deformable plate) ==================================
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "GenAlpha"
              TIMESTEP: <structure_timestep>
              NUMSTEP: <structure_num_steps>
              LINEAR_SOLVER: 1
              PREDICT: "ConstDisVelAcc"
              TOLRES: <structure_residual_tolerance>
              TOLDISP: <structure_displacement_tolerance>
            STRUCTURAL DYNAMIC/GENALPHA:
              RHO_INF: <genalpha_rho_inf>

            # == Particle ======================================================
            PARTICLE DYNAMIC:
              INTERACTION: "DEM"
              TIMESTEP: <particle_timestep>
              NUMSTEP: <particle_num_steps>
              RESULTSEVERY: <particle_results_interval>
              RESTARTEVERY: <particle_restart_interval>
              WRITE_PARTICLE_RUNTIME_VTK: true

            # == PASI coupling =================================================
            PASI DYNAMIC:
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              RESULTSEVERY: <results_output_interval>

            # == Binning strategy (spatial search) =============================
            BINNING STRATEGY:
              BIN_SIZE_LOWER_BOUND: <bin_size_lower_bound>
              DOMAINBOUNDINGBOX_LOWER: [<domain_lower_x>, <domain_lower_y>, <domain_lower_z>]
              DOMAINBOUNDINGBOX_UPPER: [<domain_upper_x>, <domain_upper_y>, <domain_upper_z>]

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "structure_solver"

            # == Materials =====================================================
            MATERIALS:
              # DEM particle material
              - MAT: 1
                MAT_ParticleMaterialDEM:
                  DENS: <particle_density>
                  RADIUS: <particle_radius>
                  YOUNG: <particle_contact_Young_modulus>
                  NUE: <particle_contact_Poisson_ratio>
              # Structural plate material
              - MAT: 2
                MAT_ElastHyper:
                  NUMMAT: 1
                  MATIDS: [3]
                  DENS: <structure_density>
              - MAT: 3
                ELAST_CoupNeoHooke:
                  YOUNG: <structure_Young_modulus>

            # == Boundary Conditions ===========================================

            # Structural: fixed boundary
            DESIGN SURF DIRICH CONDITIONS:
              - E: <fixed_face_id>
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]

            # == Particles =====================================================
            PARTICLES:
              - TYPE: "DEM"
                MAT: 1
                POSITION: [<particle_position_x>, <particle_position_y>, <particle_position_z>]
                VELOCITY: [<particle_velocity_x>, <particle_velocity_y>, <particle_velocity_z>]
                RADIUS: <particle_radius>

            # == Geometry ======================================================
            STRUCTURE GEOMETRY:
              FILE: "<mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  SOLID:
                    HEX8:
                      MAT: 2
                      KINEM: <kinematics>

            RESULT DESCRIPTION:
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_node_id>
                  QUANTITY: "dispz"
                  VALUE: <expected_displacement>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check particle density
        dens = params.get("particle_DENS") or params.get("DENS")
        if dens is not None:
            try:
                d = float(dens)
                if d <= 0:
                    issues.append(f"Particle DENS must be > 0, got {d}.")
            except (TypeError, ValueError):
                issues.append(
                    f"Particle DENS must be a positive number, "
                    f"got {dens!r}."
                )

        # Check particle radius
        radius = params.get("RADIUS")
        if radius is not None:
            try:
                r = float(radius)
                if r <= 0:
                    issues.append(
                        f"Particle RADIUS must be > 0, got {r}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"RADIUS must be a positive number, got {radius!r}."
                )

        # Check contact Young's modulus
        young = params.get("YOUNG")
        if young is not None:
            try:
                e = float(young)
                if e <= 0:
                    issues.append(f"YOUNG must be > 0, got {e}.")
            except (TypeError, ValueError):
                issues.append(
                    f"YOUNG must be a positive number, got {young!r}."
                )

        # Check BIN_SIZE_LOWER_BOUND
        bin_size = params.get("BIN_SIZE_LOWER_BOUND")
        if bin_size is not None:
            try:
                bs = float(bin_size)
                if bs <= 0:
                    issues.append(
                        f"BIN_SIZE_LOWER_BOUND must be > 0, got {bs}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"BIN_SIZE_LOWER_BOUND must be a positive number, "
                    f"got {bin_size!r}."
                )

        # Check timestep
        timestep = params.get("TIMESTEP")
        if timestep is not None:
            try:
                dt = float(timestep)
                if dt <= 0:
                    issues.append(
                        f"TIMESTEP must be > 0, got {dt}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"TIMESTEP must be a positive number, "
                    f"got {timestep!r}."
                )

        return issues
