"""FS3I (Fluid-Structure-Scalar-Scalar Interaction) generator for 4C.

Covers the 5-field coupling of fluid, structure, and two scalar transport
fields (one in the fluid domain, one in the structure domain).  The
fluid-structure interaction is handled by standard FSI (ALE-based), and
each domain carries an additional scalar transport field.  Applications
include drug delivery through arterial walls, mass transfer across
deformable membranes, and nutrient transport in biological tissues.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class FS3IGenerator(BaseGenerator):
    """Generator for FS3I (5-field coupling) problems in 4C."""

    module_key = "fs3i"
    display_name = "FS3I (Fluid-Porous-Structure-Scalar-Scalar Interaction)"
    problem_type = "Fluid_Porous_Structure_Scalar_Scalar_Interaction"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "FS3I is a 5-field coupling framework combining "
                "Fluid-Structure Interaction (FSI) with two scalar "
                "transport fields: one in the fluid domain and one in "
                "the structural (porous) domain.  The fluid field solves "
                "Navier-Stokes, the structure field solves momentum "
                "balance, and each domain has its own advection-diffusion "
                "scalar transport field.  At the FSI interface, both "
                "velocity continuity and scalar concentration continuity "
                "(or flux balance) are enforced.  The PROBLEM TYPE is "
                "'Fluid_Porous_Structure_Scalar_Scalar_Interaction'.  "
                "Required dynamics sections include FSI DYNAMIC, "
                "SCALAR TRANSPORT DYNAMIC (for the fluid-side scalar), "
                "SCALAR TRANSPORT DYNAMIC 2 or equivalent structure-side "
                "scalar config, and FS3I DYNAMIC for overall coupling "
                "parameters.  Typical applications include drug elution "
                "from stents, oxygen transport through vessel walls, and "
                "mass transfer in filtration membranes."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "STRUCTURAL DYNAMIC",
                "FLUID DYNAMIC",
                "ALE DYNAMIC",
                "FSI DYNAMIC",
                "SCALAR TRANSPORT DYNAMIC",
                "FS3I DYNAMIC",
                "SOLVER 1",
                "SOLVER 2",
                "SOLVER 3",
                "MATERIALS",
                "CLONING MATERIAL MAP",
                "STRUCTURE GEOMETRY",
                "FLUID GEOMETRY",
            ],
            "optional_sections": [
                "FSI DYNAMIC/MONOLITHIC SOLVER",
                "FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION",
                "FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/STRUCTURE",
                "IO/RUNTIME VTK OUTPUT/FLUID",
            ],
            "materials": {
                "MAT_fluid": {
                    "description": (
                        "Newtonian fluid for the free fluid domain."
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
                "MAT_scatra": {
                    "description": (
                        "Scalar transport material for the fluid-side "
                        "concentration field.  Defines diffusivity."
                    ),
                    "parameters": {
                        "DIFFUSIVITY": {
                            "description": (
                                "Molecular diffusion coefficient [m^2/s]"
                            ),
                            "range": "> 0",
                        },
                    },
                },
                "MAT_scatra_reaction": {
                    "description": (
                        "Scalar transport material for the structure-side "
                        "concentration field.  May include reaction terms "
                        "for drug metabolism or nutrient consumption."
                    ),
                    "parameters": {
                        "DIFFUSIVITY": {
                            "description": (
                                "Effective diffusion coefficient in "
                                "structure/porous domain [m^2/s]"
                            ),
                            "range": "> 0",
                        },
                        "REACOEFF": {
                            "description": (
                                "First-order reaction rate coefficient [1/s]"
                            ),
                            "range": ">= 0",
                        },
                    },
                },
                "MAT_ElastHyper": {
                    "description": (
                        "Hyperelastic structural material (same as FSI)."
                    ),
                    "parameters": {
                        "NUMMAT": {
                            "description": "Number of sub-materials",
                            "range": "1",
                        },
                        "MATIDS": {
                            "description": "Sub-material IDs",
                            "range": "",
                        },
                        "DENS": {
                            "description": "Structural density [kg/m^3]",
                            "range": "> 0",
                        },
                        "POLYCONVEX": {
                            "description": "Polyconvexity check flag (wrapper)",
                            "range": "0 | 1",
                        },
                    },
                },
            },
            "solver": {
                "fsi_solver": {
                    "type": "UMFPACK or block solver",
                    "notes": (
                        "Solver for the FSI sub-problem (fluid + structure "
                        "+ ALE)."
                    ),
                },
                "scatra_solver": {
                    "type": "UMFPACK",
                    "notes": (
                        "Solver for the scalar transport fields.  Scalar "
                        "transport systems are typically well-conditioned."
                    ),
                },
            },
            "coupling_parameters": {
                "FS3I_APPROACH": (
                    "Overall FS3I coupling approach: 'sequential' "
                    "(solve FSI first, then scalar transport) or "
                    "'fully_coupled' (monolithic or iterative coupling "
                    "of all 5 fields)."
                ),
                "SCATRA_COUPLING": (
                    "Interface condition for the scalar: 'Dirichlet-"
                    "Neumann' (concentration match on one side, flux on "
                    "the other) or 'Dirichlet-Dirichlet' (concentration "
                    "continuity on both sides)."
                ),
            },
            "pitfalls": [
                (
                    "[Input] FS3I is a FIVE-field problem: "
                    "fluid + structure + ALE + fluid-side "
                    "scalar + structure-side scalar. All "
                    "five must be configured. Signal: "
                    "missing any field section aborts at "
                    "setup with 'FS3I field N not found' "
                    "from 4C_fs3i_factory.cpp; the failure "
                    "message names the missing field. Use "
                    "an FS3I tutorial as the starting "
                    "template — greenfield is too error-"
                    "prone for 5-field problems. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] CLONING MATERIAL MAP must "
                    "specify TWO mappings: (1) fluid -> "
                    "ALE (mesh motion), (2) fluid scatra "
                    "-> structure scatra (scalar cloning). "
                    "Signal: missing either mapping raises "
                    "'cannot clone material for <field>' "
                    "at setup. Both required — they sit as "
                    "two SRC/TAR entries in the CLONING "
                    "MATERIAL MAP list. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Numerical] Scalar transport in the "
                    "fluid domain is ADVECTION-DOMINATED. "
                    "Signal: SCATRA STABILIZATION block set "
                    "to no_stabilization on a fluid-scatra "
                    "field (PROBLEMTYPE: Fluid_Structure_"
                    "Scalar_Scalar_Interaction, FLUID3 + "
                    "TRANSP) at Pe > 10 shows visible "
                    "oscillations downstream of source/sink "
                    "locations that do NOT damp with "
                    "refinement; switching STABTYPE to "
                    "residual_based (SUPG/GLS) removes them. "
                    "Use DEFINITION_TAU: Codina as the "
                    "typical choice. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Structure-side scalar "
                    "typically has MUCH LOWER diffusivity "
                    "than fluid-side (~1e-9 vs 1e-3 m^2/s "
                    "for drug transport). Signal: coupling "
                    "without under-relaxation iterates "
                    "between two non-converging states "
                    "because the diffusivity contrast "
                    "drives the interface concentration "
                    "outside the stable basin. Use Aitken "
                    "or fixed-relaxation (omega ~ 0.5) on "
                    "the scalar coupling. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] FS3I fluid elements need "
                    "NA: ALE (same as FSI). Scalar "
                    "transport on the FLUID mesh ALSO uses "
                    "the ALE velocity. Signal: setting NA: "
                    "Euler on FS3I FLUID3 elements raises "
                    "'fluid kinematic type incompatible "
                    "with moving-mesh scalar transport' at "
                    "setup; without it, the FLUID3 + TRANSP "
                    "convective term sees Eulerian velocity "
                    "but the mesh is moving, producing a "
                    "spurious O(|v_mesh|) error in the "
                    "SCATRA DYNAMIC transport. 4C handles "
                    "ALE-scatra automatically when NA: ALE "
                    "is set on the fluid elements. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] FS3I DYNAMIC controls the "
                    "OVERALL coupling loop (max iterations, "
                    "convergence tolerances for the scalar "
                    "coupling). FSI DYNAMIC controls the "
                    "FSI sub-loop. Signal: setting "
                    "ITEMAX in FSI DYNAMIC but not in "
                    "FS3I DYNAMIC limits the inner FSI "
                    "iterations but lets the outer scalar "
                    "coupling iterate indefinitely; "
                    "match both. The scalar coupling "
                    "typically needs more iterations (5-10) "
                    "than FSI (3-5). (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] For SEQUENTIAL FS3I, "
                    "ensure FSI time step and scalar "
                    "transport time step are COMPATIBLE — "
                    "typically both use the SAME time "
                    "step. Signal: different dt's in FSI "
                    "DYNAMIC vs SCALAR TRANSPORT DYNAMIC "
                    "produces sub-stepping that "
                    "accumulates O(dt) splitting error "
                    "per step; result drifts from a "
                    "fully-coupled monolithic reference "
                    "by 5-15% on the scalar field. Set "
                    "TIMESTEP equal in both fields. "
                    "(Audit 2026-06-02.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "drug_eluting_stent_3d",
                    "description": (
                        "A drug-eluting stent in a pulsatile flow vessel.  "
                        "The FSI captures wall motion, the fluid-side "
                        "scalar tracks drug concentration in blood, and "
                        "the structure-side scalar models drug diffusion "
                        "through the vessel wall.  Tests the full 5-field "
                        "FS3I coupling."
                    ),
                    "template_variant": "fs3i_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "fs3i_3d",
                "description": (
                    "3-D FS3I: fluid-structure interaction with scalar "
                    "transport in both domains.  Neo-Hookean structure, "
                    "Newtonian fluid, ALE mesh motion, advection-"
                    "diffusion scalar transport.  UMFPACK solvers."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "fs3i_3d") -> str:
        templates = {
            "fs3i_3d": self._template_fs3i_3d,
        }
        if variant == "default":
            variant = "fs3i_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_fs3i_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D FS3I (Fluid-Porous-Structure-Scalar-Scalar Interaction)
            #
            # 5-field coupling: fluid + structure + ALE + fluid-side scalar
            # + structure-side scalar.  The FSI sub-problem handles flow and
            # wall motion; the scalar transport fields model mass transfer
            # (e.g. drug concentration) in both domains.
            #
            # Mesh: requires exodus files with:
            #   "fsi.e" or separate fluid/structure meshes
            #   element_block 1 = structure (HEX8)
            #   element_block 2 = fluid (HEX8)
            #   node_set 1 = structure fixed end
            #   node_set 2 = FSI interface (structure side)
            #   node_set 3 = fluid inlet
            #   node_set 4 = fluid walls
            #   node_set 5 = FSI interface (fluid side)
            # ---------------------------------------------------------------
            TITLE:
              - "3-D FS3I -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Fluid_Porous_Structure_Scalar_Scalar_Interaction"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/STRUCTURE:
              OUTPUT_STRUCTURE: true
              DISPLACEMENT: true
            IO/RUNTIME VTK OUTPUT/FLUID:
              OUTPUT_FLUID: true
              VELOCITY: true
              PRESSURE: true

            # == Structure =====================================================
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "GenAlpha"
              LINEAR_SOLVER: 1
              PREDICT: "ConstDisVelAcc"
              TOLRES: <structure_residual_tolerance>
              TOLDISP: <structure_displacement_tolerance>
            STRUCTURAL DYNAMIC/GENALPHA:
              RHO_INF: <genalpha_rho_inf>

            # == Fluid =========================================================
            FLUID DYNAMIC:
              TIMEINTEGR: "Np_Gen_Alpha"
              LINEAR_SOLVER: 2
              ITEMAX: <fluid_max_iterations>
            FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES:
              TOL_VEL_RES: <fluid_velocity_residual_tolerance>
              TOL_VEL_INC: <fluid_velocity_increment_tolerance>
              TOL_PRES_RES: <fluid_pressure_residual_tolerance>
              TOL_PRES_INC: <fluid_pressure_increment_tolerance>
            FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION:
              CHARELELENGTH_PC: "root_of_volume"

            # == ALE mesh motion ===============================================
            ALE DYNAMIC:
              ALE_TYPE: "springs_spatial"
              LINEAR_SOLVER: 1
              MAXITER: <ale_max_iterations>

            # == FSI coupling ==================================================
            FSI DYNAMIC:
              MAXTIME: <end_time>
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              SECONDORDER: true
            FSI DYNAMIC/MONOLITHIC SOLVER:
              SHAPEDERIVATIVES: true

            # == Scalar transport ==============================================
            SCALAR TRANSPORT DYNAMIC:
              SOLVERTYPE: "nonlinear"
              TIMEINTEGR: "OneStepTheta"
              THETA: <scatra_theta>
              TIMESTEP: <scatra_timestep>
              NUMSTEP: <scatra_num_steps>
              LINEAR_SOLVER: 3
              VELOCITYFIELD: "Navier_Stokes"

            # == FS3I coupling =================================================
            FS3I DYNAMIC:
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              FS3I_APPROACH: "<fs3i_approach>"
              SCATRA_COUPLING: "<scatra_coupling_type>"
              ITEMAX: <fs3i_max_coupling_iterations>
              CONVTOL: <fs3i_convergence_tolerance>
              RESULTSEVERY: <results_output_interval>

            # == Solvers =======================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "structure_ale_solver"
            SOLVER 2:
              SOLVER: "UMFPACK"
              NAME: "fluid_solver"
            SOLVER 3:
              SOLVER: "UMFPACK"
              NAME: "scatra_solver"

            # == Materials =====================================================
            MATERIALS:
              # Fluid material
              - MAT: 1
                MAT_fluid:
                  DYNVISCOSITY: <fluid_dynamic_viscosity>
                  DENSITY: <fluid_density>
              # Structure material (Neo-Hookean)
              - MAT: 2
                MAT_ElastHyper:
                  NUMMAT: 1
                  MATIDS: [3]
                  DENS: <structure_density>
              - MAT: 3
                ELAST_CoupNeoHooke:
                  YOUNG: <structure_Young_modulus>
              # ALE pseudo-material
              - MAT: 4
                MAT_Struct_StVenantKirchhoff:
                  YOUNG: <ale_Young_modulus>
                  NUE: <ale_Poisson_ratio>
                  DENS: <ale_density>
              # Fluid-side scalar transport material
              - MAT: 5
                MAT_scatra:
                  DIFFUSIVITY: <fluid_scalar_diffusivity>
              # Structure-side scalar transport material
              - MAT: 6
                MAT_scatra:
                  DIFFUSIVITY: <structure_scalar_diffusivity>

            # Clone fluid -> ALE, fluid-scatra -> structure-scatra
            CLONING MATERIAL MAP:
              - SRC_FIELD: "fluid"
                SRC_MAT: 1
                TAR_FIELD: "ale"
                TAR_MAT: 4
              - SRC_FIELD: "scatra1"
                SRC_MAT: 5
                TAR_FIELD: "scatra2"
                TAR_MAT: 6

            # == Boundary Conditions ===========================================

            # Structure: fixed support
            DESIGN SURF DIRICH CONDITIONS:
              - E: <structure_fixed_face_id>
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]

            # Fluid: inlet velocity
            DESIGN SURF FLUID DIRICH CONDITIONS:
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

            # ALE: fix outer boundaries
            DESIGN SURF ALE DIRICH CONDITIONS:
              - E: <ale_fixed_face_id>
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]

            # Scalar transport: inlet concentration
            DESIGN SURF TRANSPORT DIRICH CONDITIONS:
              - E: <scatra_inlet_face_id>
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<inlet_scalar_concentration>]
                FUNCT: [0]

            # FSI coupling interface
            DESIGN FSI COUPLING SURF CONDITIONS:
              - E: <fsi_interface_structure_id>
                coupling_id: 1
              - E: <fsi_interface_fluid_id>
                coupling_id: 1

            # Inlet ramp function
            FUNCT<inlet_ramp_function>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<inlet_ramp_expression>"

            # == Geometry ======================================================
            STRUCTURE GEOMETRY:
              FILE: "<structure_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  SOLID:
                    HEX8:
                      MAT: 2
                      KINEM: <kinematics>

            FLUID GEOMETRY:
              FILE: "<fluid_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 2
                  FLUID:
                    HEX8:
                      MAT: 1
                      NA: ALE

            RESULT DESCRIPTION:
              - FLUID:
                  DIS: "fluid"
                  NODE: <result_fluid_node_id>
                  QUANTITY: "velx"
                  VALUE: <expected_fluid_velocity>
                  TOLERANCE: <result_tolerance>
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_structure_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_displacement>
                  TOLERANCE: <result_tolerance>
              - SCATRA:
                  DIS: "scatra1"
                  NODE: <result_scatra_node_id>
                  QUANTITY: "phi1"
                  VALUE: <expected_concentration>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check fluid viscosity
        viscosity = params.get("DYNVISCOSITY")
        if viscosity is not None:
            try:
                mu = float(viscosity)
                if mu <= 0:
                    issues.append(
                        f"DYNVISCOSITY must be > 0, got {mu}."
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
                        f"DENSITY must be > 0, got {rho}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DENSITY must be a positive number, got {density!r}."
                )

        # Check diffusivity
        for key in ("DIFFUSIVITY", "fluid_scalar_diffusivity",
                     "structure_scalar_diffusivity"):
            diff = params.get(key)
            if diff is not None:
                try:
                    d = float(diff)
                    if d <= 0:
                        issues.append(
                            f"{key} must be > 0, got {d}."
                        )
                except (TypeError, ValueError):
                    issues.append(
                        f"{key} must be a positive number, got {diff!r}."
                    )

        # Check Young's modulus
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

        # Check CLONING MATERIAL MAP
        has_cloning = params.get("has_cloning_material_map")
        if has_cloning is not None and not has_cloning:
            issues.append(
                "CLONING MATERIAL MAP is required for FS3I.  "
                "It maps fluid -> ALE and fluid-scatra -> "
                "structure-scatra."
            )

        # Check fluid NA mode
        fluid_na = params.get("fluid_NA") or params.get("NA")
        if fluid_na is not None:
            if str(fluid_na).upper() != "ALE":
                issues.append(
                    f"Fluid elements MUST use NA: ALE for FS3I, "
                    f"got {fluid_na!r}."
                )

        return issues
