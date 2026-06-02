"""Fluid-Beam Interaction (FBI) generator for 4C.

Covers coupling of a 3-D fluid field (incompressible Navier-Stokes) with
embedded 1-D beam elements.  The beams are immersed in the fluid domain
and interact via penalty or mortar coupling.  Applications include flow
around fibers, stent deployment in blood vessels, and fiber-reinforced
composites with fluid flow.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class FBIGenerator(BaseGenerator):
    """Generator for Fluid-Beam Interaction problems in 4C."""

    module_key = "fbi"
    display_name = "Fluid-Beam Interaction (FBI)"
    problem_type = "Fluid_Beam_Interaction"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Fluid-Beam Interaction (FBI) couples a 3-D incompressible "
                "Navier-Stokes fluid field with embedded 1-D beam elements.  "
                "The beams are immersed in the fluid volume and do not "
                "require a body-fitted mesh.  Coupling is achieved via "
                "penalty or mortar methods that transfer fluid drag forces "
                "to the beams and impose the beam velocity as a constraint "
                "on the fluid.  The PROBLEM TYPE is "
                "'Fluid_Beam_Interaction'.  Required dynamics sections "
                "include STRUCTURAL DYNAMIC (for beams), FLUID DYNAMIC, "
                "and FBI DYNAMIC for the coupling parameters.  The fluid "
                "mesh uses standard FLUID elements while beams use BEAM3R "
                "or BEAM3EB elements.  No ALE mesh motion is needed since "
                "the coupling is immersed (non-body-fitted)."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "STRUCTURAL DYNAMIC",
                "FLUID DYNAMIC",
                "FBI DYNAMIC",
                "SOLVER 1",
                "SOLVER 2",
                "MATERIALS",
                "STRUCTURE GEOMETRY",
                "FLUID GEOMETRY",
            ],
            "optional_sections": [
                "FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION",
                "FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/BEAMS",
                "IO/RUNTIME VTK OUTPUT/FLUID",
                "BINNING STRATEGY",
            ],
            "materials": {
                "MAT_BeamReissnerElastHyper": {
                    "description": (
                        "Geometrically exact Reissner beam material.  "
                        "Defines axial, shear, bending, and torsional "
                        "stiffness for 1-D beam elements embedded in "
                        "the fluid."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus of beam material",
                            "range": "> 0",
                        },
                        "POISSONRATIO": {
                            "description": "Poisson's ratio of beam material",
                            "range": "(0, 0.5)",
                        },
                        "DENS": {
                            "description": "Mass density of beam material",
                            "range": "> 0",
                        },
                        "CROSSAREA": {
                            "description": "Cross-sectional area of beam",
                            "range": "> 0",
                        },
                        "SHEARCORR": {
                            "description": "Shear correction factor",
                            "range": "> 0 (typically 1.0)",
                        },
                        "MOMINPOL": {
                            "description": "Polar moment of inertia",
                            "range": "> 0",
                        },
                        "MOMIN2": {
                            "description": "Second moment of area (axis 2)",
                            "range": "> 0",
                        },
                        "MOMIN3": {
                            "description": "Second moment of area (axis 3)",
                            "range": "> 0",
                        },
                    },
                },
                "MAT_fluid": {
                    "description": (
                        "Newtonian fluid material for the background "
                        "fluid field."
                    ),
                    "parameters": {
                        "DYNVISCOSITY": {
                            "description": "Dynamic viscosity [Pa s]",
                            "range": "> 0",
                        },
                        "DENSITY": {
                            "description": "Fluid density [kg/m^3]",
                            "range": "> 0",
                        },
                    },
                },
            },
            "solver": {
                "structure_solver": {
                    "type": "UMFPACK (direct)",
                    "notes": (
                        "Beam problems are typically small and well-suited "
                        "to direct solvers."
                    ),
                },
                "fluid_solver": {
                    "type": "Belos or UMFPACK",
                    "notes": (
                        "Fluid solver for the background Navier-Stokes "
                        "equations.  Iterative for large 3-D problems."
                    ),
                },
            },
            "coupling_parameters": {
                "PENALTY_PARAMETER": (
                    "Penalty stiffness for the immersed coupling.  "
                    "Controls how strongly the beam velocity constraint "
                    "is enforced on the fluid.  Too small leads to "
                    "fluid penetrating the beam; too large causes "
                    "ill-conditioning."
                ),
                "COUPLING_TYPE": (
                    "Type of FBI coupling: 'penalty' (penalty method) "
                    "or 'mortar' (mortar-based).  Penalty is simpler; "
                    "mortar is more accurate but more expensive."
                ),
            },
            "pitfalls": [
                (
                    "[Input] FBI does NOT use ALE mesh motion. "
                    "The fluid mesh is FIXED (Eulerian); the "
                    "beams move through the fluid via "
                    "immersed coupling. Signal: including ALE "
                    "DYNAMIC in an FBI input aborts with "
                    "'FBI is incompatible with ALE' from "
                    "4C_fbi_factory.cpp; the immersed-"
                    "coupling assumption breaks if the fluid "
                    "mesh also moves. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] FBI penalty parameter must "
                    "be carefully tuned. Signal: too LARGE "
                    "values give cond(K) > 1e15 (sparse "
                    "solver stagnates); too SMALL values "
                    "let the fluid 'slip through' the beam "
                    "— visible non-zero velocity inside "
                    "the beam region in ParaView, "
                    "decreasing rapidly as penalty "
                    "increases. Typical penalty range "
                    "(beam-axial-stiffness equivalent): "
                    "1e3 - 1e6 N/m. (Audit 2026-06-02.)"
                ),
                (
                    "[Mesh] Beam elements (BEAM3R, BEAM3EB) "
                    "are 1D line elements. The fluid mesh "
                    "must cover the ENTIRE domain including "
                    "the region occupied by beams. Signal: "
                    "an open fluid domain that ends at the "
                    "beam region produces a 'beam outside "
                    "fluid domain' error from "
                    "4C_fbi_partitioner.cpp — the immersed "
                    "search algorithm cannot find host "
                    "fluid elements for beam Gauss points. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] BINNING STRATEGY may be "
                    "needed to efficiently search for beam-"
                    "fluid element pairs. Set "
                    "BIN_SIZE_LOWER_BOUND appropriately for "
                    "element sizes. Signal: without "
                    "binning, the beam-fluid pair search "
                    "is O(N_beams * N_fluid) — wall-clock "
                    "explodes for large meshes. With "
                    "BIN_SIZE_LOWER_BOUND ~ "
                    "max(h_beam, h_fluid), search is O(N) "
                    "amortised. (Audit 2026-06-02.)"
                ),
                (
                    "[Output] Structural output uses the "
                    "BEAM discretisation, NOT the standard "
                    "STRUCTURE output. Use IO/RUNTIME VTK "
                    "OUTPUT/BEAMS for beam visualisation. "
                    "Signal: configuring IO/RUNTIME VTK "
                    "OUTPUT/STRUCTURE for an FBI problem "
                    "produces empty output (no beam DOFs "
                    "in the structure section); the "
                    "BEAMS subsection has BeamCenterline, "
                    "BeamCrossSection, etc. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] FBI beam material uses "
                    "DEDICATED beam material types (e.g. "
                    "MAT_BeamReissnerElastHyper) — NOT "
                    "standard solid materials. Cross-"
                    "sectional properties (CROSSAREA, "
                    "MOMINPOL, etc.) must be specified. "
                    "Signal: using MAT_ElastHyper for "
                    "beams raises 'beam element requires "
                    "beam material' from "
                    "4C_mat_beam_base.cpp at setup; the "
                    "beam factory does not accept generic "
                    "solid materials. (Audit 2026-06-02.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "fiber_in_channel_3d",
                    "description": (
                        "A flexible fiber immersed in a 3-D channel flow.  "
                        "The fluid drag causes the fiber to deform and the "
                        "fiber in turn disturbs the flow field.  Tests "
                        "penalty coupling, beam large deformation, and "
                        "fluid stabilization."
                    ),
                    "template_variant": "penalty_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "penalty_3d",
                "description": (
                    "3-D FBI with penalty coupling: flexible beam in "
                    "channel flow.  BEAM3R elements immersed in FLUID "
                    "HEX8 elements, penalty-based velocity coupling, "
                    "UMFPACK solvers."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "penalty_3d") -> str:
        templates = {
            "penalty_3d": self._template_penalty_3d,
        }
        if variant == "default":
            variant = "penalty_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_penalty_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D Fluid-Beam Interaction (FBI) with Penalty Coupling
            #
            # A flexible beam is immersed in a 3-D fluid channel.  The
            # fluid exerts drag on the beam, causing it to deform, and
            # the beam displaces the fluid via the coupling penalty term.
            #
            # Mesh: requires TWO exodus files:
            #   Fluid mesh: "fluid.e" with
            #     element_block 1 = fluid domain (HEX8)
            #     node_set 1 = inlet
            #     node_set 2 = outlet
            #     node_set 3 = walls (no-slip)
            #   Beam mesh: "beams.e" with
            #     element_block 1 = beam elements (LINE2)
            #     node_set 1 = beam clamped end
            # ---------------------------------------------------------------
            TITLE:
              - "3-D fluid-beam interaction -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Fluid_Beam_Interaction"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/BEAMS:
              OUTPUT_BEAMS: true
              DISPLACEMENT: true
              USE_ABSOLUTE_POSITIONS: true
              TRIAD_VISUALISATION_POINT: true
            IO/RUNTIME VTK OUTPUT/FLUID:
              OUTPUT_FLUID: true
              VELOCITY: true
              PRESSURE: true

            # == Structure (beams) =============================================
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

            # == Fluid =========================================================
            FLUID DYNAMIC:
              TIMEINTEGR: "Np_Gen_Alpha"
              TIMESTEP: <fluid_timestep>
              NUMSTEP: <fluid_num_steps>
              LINEAR_SOLVER: 2
              ITEMAX: <fluid_max_iterations>
            FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES:
              TOL_VEL_RES: <fluid_velocity_residual_tolerance>
              TOL_VEL_INC: <fluid_velocity_increment_tolerance>
              TOL_PRES_RES: <fluid_pressure_residual_tolerance>
              TOL_PRES_INC: <fluid_pressure_increment_tolerance>
            FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION:
              CHARELELENGTH_PC: "root_of_volume"

            # == FBI coupling ==================================================
            FBI DYNAMIC:
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              COUPLING_TYPE: "<coupling_type>"
              PENALTY_PARAMETER: <penalty_parameter>
              RESULTSEVERY: <results_output_interval>

            # == Binning (for beam-fluid search) ===============================
            BINNING STRATEGY:
              BIN_SIZE_LOWER_BOUND: <bin_size_lower_bound>

            # == Solvers =======================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "beam_solver"
            SOLVER 2:
              SOLVER: "UMFPACK"
              NAME: "fluid_solver"

            # == Materials =====================================================
            MATERIALS:
              # Beam material (Reissner elastohyper)
              - MAT: 1
                MAT_BeamReissnerElastHyper:
                  YOUNG: <beam_Young_modulus>
                  POISSONRATIO: <beam_Poisson_ratio>
                  DENS: <beam_density>
                  CROSSAREA: <beam_cross_section_area>
                  SHEARCORR: <beam_shear_correction_factor>
                  MOMINPOL: <beam_polar_moment_of_inertia>
                  MOMIN2: <beam_moment_of_inertia_2>
                  MOMIN3: <beam_moment_of_inertia_3>
              # Fluid material
              - MAT: 2
                MAT_fluid:
                  DYNVISCOSITY: <fluid_dynamic_viscosity>
                  DENSITY: <fluid_density>

            # == Boundary Conditions ===========================================

            # Beam: clamped end
            DESIGN POINT DIRICH CONDITIONS:
              - E: <beam_clamped_node_set_id>
                NUMDOF: 6
                ONOFF: [1, 1, 1, 1, 1, 1]
                VAL: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0, 0, 0, 0]

            # Fluid: inlet
            DESIGN SURF DIRICH CONDITIONS:
              - E: <inlet_face_id>
                NUMDOF: 4
                ONOFF: [1, 1, 1, 0]
                VAL: [<inlet_velocity_x>, <inlet_velocity_y>, <inlet_velocity_z>, 0.0]
                FUNCT: [<inlet_ramp_function>, 0, 0, 0]
              # Fluid: no-slip walls
              - E: <wall_face_id>
                NUMDOF: 4
                ONOFF: [1, 1, 1, 0]
                VAL: [0.0, 0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0, 0]

            # Inlet ramp-up function
            FUNCT<inlet_ramp_function>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<inlet_ramp_expression>"

            # == Geometry ======================================================
            STRUCTURE GEOMETRY:
              FILE: "<beam_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  BEAM3R:
                    LINE2:
                      MAT: 1

            FLUID GEOMETRY:
              FILE: "<fluid_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  FLUID:
                    HEX8:
                      MAT: 2
                      NA: Euler

            RESULT DESCRIPTION:
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_beam_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_beam_displacement>
                  TOLERANCE: <result_tolerance>
              - FLUID:
                  DIS: "fluid"
                  NODE: <result_fluid_node_id>
                  QUANTITY: "velx"
                  VALUE: <expected_fluid_velocity>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check beam Young's modulus
        young = params.get("YOUNG") or params.get("beam_YOUNG")
        if young is not None:
            try:
                e = float(young)
                if e <= 0:
                    issues.append(f"Beam YOUNG must be > 0, got {e}.")
            except (TypeError, ValueError):
                issues.append(
                    f"Beam YOUNG must be a positive number, got {young!r}."
                )

        # Check penalty parameter
        penalty = params.get("PENALTY_PARAMETER")
        if penalty is not None:
            try:
                p = float(penalty)
                if p <= 0:
                    issues.append(
                        f"PENALTY_PARAMETER must be > 0, got {p}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"PENALTY_PARAMETER must be a positive number, "
                    f"got {penalty!r}."
                )

        # Check fluid viscosity
        viscosity = params.get("DYNVISCOSITY")
        if viscosity is not None:
            try:
                mu = float(viscosity)
                if mu <= 0:
                    issues.append(
                        f"Fluid DYNVISCOSITY must be > 0, got {mu}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DYNVISCOSITY must be a positive number, "
                    f"got {viscosity!r}."
                )

        # Check fluid density
        density = params.get("DENSITY")
        if density is not None:
            try:
                rho = float(density)
                if rho <= 0:
                    issues.append(
                        f"Fluid DENSITY must be > 0, got {rho}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DENSITY must be a positive number, got {density!r}."
                )

        # Check cross-sectional area
        crossarea = params.get("CROSSAREA")
        if crossarea is not None:
            try:
                a = float(crossarea)
                if a <= 0:
                    issues.append(
                        f"CROSSAREA must be > 0, got {a}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"CROSSAREA must be a positive number, "
                    f"got {crossarea!r}."
                )

        # Check coupling type
        coupling_type = params.get("COUPLING_TYPE")
        if coupling_type is not None and coupling_type not in (
            "penalty", "mortar",
        ):
            issues.append(
                f"COUPLING_TYPE must be 'penalty' or 'mortar', "
                f"got {coupling_type!r}."
            )

        return issues
