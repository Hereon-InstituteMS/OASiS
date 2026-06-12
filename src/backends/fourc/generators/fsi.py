"""Fluid-Structure Interaction (FSI) generator for 4C.

Covers monolithic and partitioned FSI coupling of an incompressible
Navier-Stokes fluid with a hyperelastic or St. Venant-Kirchhoff structure.
Includes ALE mesh motion handling and CLONING MATERIAL MAP.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class FSIGenerator(BaseGenerator):
    """Generator for Fluid-Structure Interaction problems in 4C."""

    module_key = "fsi"
    display_name = "Fluid-Structure Interaction (FSI)"
    problem_type = "Fluid_Structure_Interaction"

    # ── Knowledge ─────────────────────────────────────────────────────

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Monolithic or partitioned coupling of incompressible "
                "Navier-Stokes fluid with geometrically nonlinear structures.  "
                "The fluid domain moves with the structure via ALE (Arbitrary "
                "Lagrangian-Eulerian) mesh motion.  This is the most complex "
                "problem type in 4C, requiring coordinated setup of three "
                "fields (structure, fluid, ALE) plus coupling conditions."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "STRUCTURAL DYNAMIC",
                "STRUCTURAL DYNAMIC/GENALPHA",
                "FLUID DYNAMIC",
                "ALE DYNAMIC",
                "FSI DYNAMIC",
                "FSI DYNAMIC/MONOLITHIC SOLVER",
                "SOLVER 1",
                "SOLVER 2",
                "SOLVER 3",
                "MATERIALS",
                "STRUCTURE GEOMETRY",
                "FLUID GEOMETRY",
                "CLONING MATERIAL MAP",
                "DESIGN FSI COUPLING LINE CONDITIONS",  # 2-D
                # or "DESIGN FSI COUPLING SURF CONDITIONS" for 3-D
            ],
            "optional_sections": [
                "FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION",
                "FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES",
                "FSI DYNAMIC/PARTITIONED SOLVER",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/STRUCTURE",
                "IO/RUNTIME VTK OUTPUT/FLUID",
            ],
            "materials": {
                "MAT_fluid": {
                    "description": (
                        "Newtonian fluid for the fluid field."
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
                "MAT_ElastHyper (Neo-Hooke)": {
                    "description": (
                        "Hyperelastic material for the structure.  Uses a "
                        "nested ELAST_CoupNeoHooke sub-material."
                    ),
                    "parameters": {
                        "NUMMAT": {"description": "Number of sub-materials", "range": "1"},
                        "MATIDS": {"description": "List of sub-material IDs", "range": ""},
                        "DENS": {"description": "Structural density [kg/m^3]", "range": "> 0"},
                    },
                },
                "MAT_Struct_StVenantKirchhoff": {
                    "description": (
                        "Linear elastic structural material (small-strain "
                        "approximation, but used with nonlinear kinematics)."
                    ),
                    "parameters": {
                        "YOUNG": {"description": "Young's modulus [Pa]", "range": "> 0"},
                        "NUE": {"description": "Poisson's ratio", "range": "[0, 0.5)"},
                        "DENS": {"description": "Structural density [kg/m^3]", "range": "> 0"},
                    },
                },
                "ALE material (clone)": {
                    "description": (
                        "ALE mesh motion material.  Cloned from the fluid "
                        "material via CLONING MATERIAL MAP.  Typically use "
                        "MAT_Struct_StVenantKirchhoff with YOUNG=1, NUE=0."
                    ),
                },
            },
            "solver": {
                "ALE_solver": {
                    "type": "UMFPACK (direct)",
                    "notes": "ALE system is small relative to fluid/structure.",
                },
                "Fluid_solver": {
                    "type": "Belos (iterative) or UMFPACK for small problems",
                    "notes": "AZTOL ~1e-12 for FSI accuracy.",
                },
                "Structure_solver": {
                    "type": "UMFPACK (direct) for small problems",
                },
                "Monolithic_FSI_solver": {
                    "type": "Belos with MueLu block preconditioner",
                    "notes": (
                        "Required for iter_mortar_monolithicfluidsplit. "
                        "Set LINEARBLOCKSOLVER: LinalgSolver. "
                        "Alternatively use UMFPACK for small 2-D demos."
                    ),
                },
            },
            "coupling_algorithms": {
                "recommended": "iter_mortar_monolithicfluidsplit",
                "alternatives": [
                    "iter_monolithicfluidsplit",
                    "iter_monolithicstructuresplit",
                    "iter_stagg_AITKEN_rel_force",
                    "iter_stagg_fixed_rel_force",
                ],
                "key_settings": {
                    "COUPALGO": "Coupling algorithm selector in FSI DYNAMIC",
                    "SHAPEDERIVATIVES": "Must be true for monolithic FSI",
                    "SECONDORDER": "Enable second-order time integration coupling",
                },
            },
            "ale_settings": {
                "ALE_TYPE": {
                    "springs_spatial": "Spring-based ALE (spatial formulation, recommended for 2-D)",
                    "springs_material": "Spring-based ALE (material formulation, good for 3-D)",
                },
                "important": (
                    "ALE Dirichlet BCs must be set on all outer fluid "
                    "boundaries (except the FSI interface) to keep the "
                    "ALE mesh fixed there."
                ),
            },
            "cloning_material_map": {
                "purpose": (
                    "Maps the fluid material to an ALE field material.  "
                    "4C internally clones the fluid mesh to create the ALE "
                    "discretisation; this map tells it which material to use."
                ),
                "format": (
                    "SRC_FIELD: fluid, SRC_MAT: <fluid_mat_id>, "
                    "TAR_FIELD: ale, TAR_MAT: <ale_mat_id>"
                ),
            },
            "pitfalls": [
                "FSI is the most complex problem type in 4C -- read carefully.",
                "Fluid elements MUST use NA: ALE (not Euler!) for FSI problems.",
                "ALE Dirichlet BCs must be applied on all outer fluid boundaries "
                "that are not FSI coupling surfaces.  If missing, the ALE mesh "
                "distorts freely and the simulation diverges.",
                "Coupling surfaces must have matching node sets in both the "
                "structure and fluid meshes (or use mortar coupling).",
                "CLONING MATERIAL MAP is required: it maps fluid -> ALE material.",
                "SHAPEDERIVATIVES must be true in FSI DYNAMIC/MONOLITHIC SOLVER "
                "for monolithic schemes.",
                "SECONDORDER: true couples the time integration at second order -- "
                "recommended for accuracy.",
                "Each field (structure, fluid, ALE) needs its own SOLVER N entry.",
                "For 2-D: use DESIGN FSI COUPLING LINE CONDITIONS.  "
                "For 3-D: use DESIGN FSI COUPLING SURF CONDITIONS.",
                "Structure uses NUMDOF matching dimension (2 or 3), "
                "fluid uses NUMDOF = dim + 1 (includes pressure).",
                # Shared-node NUMDOF conflict (applies to ALL multi-physics)
                "CRITICAL: DESIGN LINE DIRICH CONDITIONS applies to ALL "
                "discretizations containing a node, not just the intended one.  "
                "If a node exists in both structure (NUMDOF=2) and fluid (NUMDOF=3) "
                "discretizations, a Dirichlet with the wrong NUMDOF will fail.  "
                "Workarounds: (a) offset the structural mesh slightly to avoid "
                "shared nodes at the FSI-Dirichlet boundary, (b) use mortar "
                "coupling with non-conforming meshes, (c) remove the structural "
                "Dirichlet and rely on the FSI coupling constraint.",
                # Invalid section names
                "DESIGN FLUID LINE LIFT&DRAG does NOT exist in 4C for 2-D.  "
                "Only DESIGN FLUID SURF LIFT&DRAG exists (for 3-D).  "
                "For 2-D lift/drag, set LIFTDRAG: true in FLUID DYNAMIC and "
                "4C computes it automatically from the no-slip boundaries.",
                # IO section
                "The IO section does NOT have EVERY_ITERATION -- that is not a "
                "valid parameter.  Use RESULTSEVERY in each field's DYNAMIC section "
                "to control output frequency.",
            ],
            "ale_boundary_conditions": {
                "description": (
                    "ALE Dirichlet BCs pin the ALE mesh at fixed fluid boundaries.  "
                    "The FSI interface is free (ALE moves with the structure there)."
                ),
                "rules": [
                    "ALL walls with no-slip fluid BC: apply ALE Dirichlet (fix mesh)",
                    "Inflow boundary: apply ALE Dirichlet (fix mesh)",
                    "Outflow boundary: apply ALE Dirichlet (fix mesh)",
                    "Cylinder/obstacle surfaces: apply ALE Dirichlet (fix mesh)",
                    "FSI interface: do NOT apply ALE Dirichlet (mesh moves with structure)",
                ],
                "common_mistake": (
                    "Forgetting ALE Dirichlet on some outer boundary causes the "
                    "ALE mesh to distort freely there, leading to inverted elements "
                    "and solver divergence."
                ),
            },
            "valid_2d_elements": {
                "FLUID": ["QUAD4", "QUAD9", "TRI3", "TRI6"],
                "SOLID (structure)": ["QUAD4", "QUAD9", "TRI3", "TRI6"],
                "notes": (
                    "QUAD4 is most commonly used and best validated for FSI.  "
                    "TRI3 works but is less accurate for pressure.  "
                    "For 3-D FSI: FLUID HEX8/TET4, SOLID HEX8/TET4."
                ),
            },
            "typical_experiments": [
                {
                    "name": "channel_with_flap_2d",
                    "description": (
                        "Flow past a flexible flap (Turek-Hron benchmark).  "
                        "A channel with a cylinder and an attached elastic "
                        "flag.  Tests vortex shedding, large structural "
                        "deformation, and ALE mesh quality."
                    ),
                },
            ],
        }

    # ── Variants ──────────────────────────────────────────────────────

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "fsi_2d",
                "description": (
                    "2-D monolithic FSI: channel flow past a deformable wall "
                    "segment.  Neo-Hookean structure, Newtonian fluid, ALE "
                    "mesh motion.  UMFPACK solvers for simplicity."
                ),
            },
        ]

    # ── Templates ─────────────────────────────────────────────────────

    def get_template(self, variant: str = "fsi_2d") -> str:
        templates = {
            "fsi_2d": self._template_fsi_2d,
        }
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_fsi_2d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 2-D Monolithic FSI -- Channel with Deformable Wall
            #
            # Structure: Neo-Hookean elastic wall on the upper channel boundary
            # Fluid:     Incompressible Navier-Stokes (ALE)
            # Coupling:  Monolithic fluid-split with mortar
            #
            # Mesh: requires "fsi_2d.e" with:
            #   element_block 1 = structure (QUAD4)
            #   element_block 2 = fluid (QUAD4)
            #   node_set 1 = structure fixed end (Dirichlet)
            #   node_set 2 = FSI interface (structure side)
            #   node_set 3 = structure fixed end (other)
            #   node_set 5 = fluid inlet
            #   node_set 6 = fluid bottom wall
            #   node_set 7 = fluid outlet (or left open for do-nothing)
            #   node_set 8 = fluid outer boundary (ALE Dirichlet)
            #   node_set 9 = FSI interface (fluid side)
            # ---------------------------------------------------------------
            PROBLEM SIZE:
              DIM: 2
            PROBLEM TYPE:
              PROBLEMTYPE: "Fluid_Structure_Interaction"

            # == Structure =================================================
            STRUCTURAL DYNAMIC:
              INT_STRATEGY: "Standard"
              LINEAR_SOLVER: 3
              PREDICT: "ConstDisVelAcc"
              M_DAMP: <structure_mass_damping>
              K_DAMP: <structure_stiffness_damping>
              TOLDISP: <structure_displacement_tolerance>
              TOLRES: <structure_residual_tolerance>
            STRUCTURAL DYNAMIC/GENALPHA:
              BETA: <genalpha_beta>
              GAMMA: <genalpha_gamma>
              ALPHA_M: <genalpha_alpha_m>
              ALPHA_F: <genalpha_alpha_f>
              RHO_INF: <genalpha_rho_inf>

            # == Fluid =====================================================
            FLUID DYNAMIC:
              LINEAR_SOLVER: 2
              TIMEINTEGR: "Np_Gen_Alpha"
              GRIDVEL: BDF2
              ADAPTCONV: true
              ITEMAX: <fluid_max_iterations>
            FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES:
              TOL_VEL_RES: <fluid_velocity_residual_tolerance>
              TOL_VEL_INC: <fluid_velocity_increment_tolerance>
              TOL_PRES_RES: <fluid_pressure_residual_tolerance>
              TOL_PRES_INC: <fluid_pressure_increment_tolerance>
            FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION:
              CHARELELENGTH_PC: "root_of_volume"

            # == ALE mesh motion ===========================================
            ALE DYNAMIC:
              ALE_TYPE: springs_spatial
              MAXITER: <ale_max_iterations>
              TOLRES: <ale_residual_tolerance>
              TOLDISP: <ale_displacement_tolerance>
              LINEAR_SOLVER: 1

            # == FSI coupling ==============================================
            FSI DYNAMIC:
              MAXTIME: <end_time>
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              SECONDORDER: true
            FSI DYNAMIC/MONOLITHIC SOLVER:
              SHAPEDERIVATIVES: true

            # == Solvers ===================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "ALE solver"
            SOLVER 2:
              SOLVER: "UMFPACK"
              NAME: "Fluid solver"
            SOLVER 3:
              SOLVER: "UMFPACK"
              NAME: "Structure solver"

            # == Materials =================================================
            MATERIALS:
              # Fluid material
              - MAT: 1
                MAT_fluid:
                  DYNVISCOSITY: <fluid_dynamic_viscosity>
                  DENSITY: <fluid_density>
              # Structure material (Neo-Hookean hyperelastic)
              - MAT: 2
                MAT_ElastHyper:
                  NUMMAT: 1
                  MATIDS: [3]
                  DENS: <structure_density>
              - MAT: 3
                ELAST_CoupNeoHooke:
                  YOUNG: <structure_Young_modulus>
              # ALE pseudo-material (cloned from fluid)
              - MAT: 4
                MAT_Struct_StVenantKirchhoff:
                  YOUNG: <ale_Young_modulus>
                  NUE: <ale_Poisson_ratio>
                  DENS: <ale_density>

            # Map fluid material -> ALE material
            CLONING MATERIAL MAP:
              - SRC_FIELD: "fluid"
                SRC_MAT: 1
                TAR_FIELD: "ale"
                TAR_MAT: 4

            # == Geometry ==================================================
            STRUCTURE GEOMETRY:
              FILE: "fsi_2d.e"
              ELEMENT_BLOCKS:
                - ID: 1
                  SOLID:
                    QUAD4:
                      MAT: 2
                      KINEM: nonlinear
                      THICKNESS: <wall_thickness>
                      STRESS_STRAIN: plane_strain
                      GP: [2, 2]

            FLUID GEOMETRY:
              FILE: "fsi_2d.e"
              ELEMENT_BLOCKS:
                - ID: 2
                  FLUID:
                    QUAD4:
                      MAT: 1
                      NA: ALE

            # == Boundary Conditions =======================================

            # Structure: fixed supports
            DESIGN POINT DIRICH CONDITIONS:
              - E: 1
                ENTITY_TYPE: node_set_id
                NUMDOF: 2
                ONOFF: [1, 1]
                VAL: [0.0, 0.0]
                FUNCT: [0, 0]
              - E: 3
                ENTITY_TYPE: node_set_id
                NUMDOF: 2
                ONOFF: [1, 1]
                VAL: [0.0, 0.0]
                FUNCT: [0, 0]

            # Fluid: inlet parabolic, walls no-slip
            DESIGN LINE DIRICH CONDITIONS:
              # Structure fixed edge (line)
              - E: 1
                ENTITY_TYPE: node_set_id
                NUMDOF: 2
                ONOFF: [1, 1]
                VAL: [0.0, 0.0]
                FUNCT: [0, 0]
              # Fluid inlet (parabolic ramp-up)
              - E: 5
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [<inlet_velocity>, 0.0, 0.0]
                FUNCT: [1, 0, 0]
              # Fluid bottom wall (no-slip)
              - E: 6
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]

            # ALE: fix outer fluid boundaries
            DESIGN LINE ALE DIRICH CONDITIONS:
              - E: 6
                ENTITY_TYPE: node_set_id
                NUMDOF: 2
                ONOFF: [1, 1]
                VAL: [0.0, 0.0]
                FUNCT: [0, 0]
              - E: 8
                ENTITY_TYPE: node_set_id
                NUMDOF: 2
                ONOFF: [1, 1]
                VAL: [0.0, 0.0]
                FUNCT: [0, 0]

            # FSI coupling interface
            DESIGN FSI COUPLING LINE CONDITIONS:
              - E: 2
                ENTITY_TYPE: node_set_id
                coupling_id: 1
              - E: 9
                ENTITY_TYPE: node_set_id
                coupling_id: 1

            # Smooth ramp-up function for inlet
            # IMPORTANT: When using VARIABLE/multifunction with
            # SYMBOLIC_FUNCTION_OF_SPACE_TIME, you MUST include COMPONENT: 0.
            # Without it, the VARIABLE definition is not parsed correctly.
            #
            # Example with ramp-up variable:
            #   FUNCT1:
            #     - COMPONENT: 0
            #       SYMBOLIC_FUNCTION_OF_SPACE_TIME: "6*U_bar*y*(H-y)/(H*H)*a"
            #     - VARIABLE: 0
            #       NAME: "a"
            #       TYPE: "multifunction"
            #       NUMPOINTS: 3
            #       TIMES: [0, 2, 10000]
            #       DESCRIPTION: ["0.5*(1-cos(pi*t/2))", "1.0"]
            #
            # Simpler alternative (no VARIABLE needed, bake time into expression):
            #   FUNCT1:
            #     - COMPONENT: 0
            #       SYMBOLIC_FUNCTION_OF_SPACE_TIME: "6*y*(H-y)/(H*H)*(t<T_ramp?0.5*(1-cos(pi*t/T_ramp)):1)"
            FUNCT1:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<inlet_ramp_expression>"

            # == VTK output ================================================
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/STRUCTURE:
              OUTPUT_STRUCTURE: true
              DISPLACEMENT: true
            IO/RUNTIME VTK OUTPUT/FLUID:
              OUTPUT_FLUID: true
              VELOCITY: true
              PRESSURE: true
        """)

    # ── Validation ────────────────────────────────────────────────────

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        """Validate FSI-specific parameters.

        Checks that all required sections are present, fluid uses ALE,
        and CLONING MATERIAL MAP exists.
        """
        issues: list[str] = []

        # Check required sections if a full input dict is provided
        sections = params.get("sections") or params.get("input_sections")
        if sections is not None:
            if isinstance(sections, (list, set)):
                section_set = set(s.upper() if isinstance(s, str) else s for s in sections)
            elif isinstance(sections, dict):
                section_set = set(
                    k.upper() if isinstance(k, str) else k for k in sections.keys()
                )
            else:
                section_set = set()

            required = {
                "PROBLEM TYPE",
                "STRUCTURAL DYNAMIC",
                "FLUID DYNAMIC",
                "ALE DYNAMIC",
                "FSI DYNAMIC",
                "MATERIALS",
                "STRUCTURE GEOMETRY",
                "FLUID GEOMETRY",
                "CLONING MATERIAL MAP",
            }
            missing = required - section_set
            if missing:
                issues.append(
                    f"Missing required FSI sections: {sorted(missing)}"
                )

        # Check fluid NA mode
        fluid_na = params.get("fluid_NA") or params.get("NA")
        if fluid_na is not None:
            if str(fluid_na).upper() != "ALE":
                issues.append(
                    f"Fluid elements MUST use NA: ALE for FSI, got {fluid_na!r}.  "
                    f"Euler grid does not support mesh motion."
                )

        # Check SHAPEDERIVATIVES
        shape_deriv = params.get("SHAPEDERIVATIVES")
        if shape_deriv is not None and not shape_deriv:
            issues.append(
                "SHAPEDERIVATIVES must be true in FSI DYNAMIC/MONOLITHIC SOLVER "
                "for monolithic coupling schemes."
            )

        # Validate materials
        viscosity = params.get("viscosity") or params.get("DYNVISCOSITY")
        if viscosity is not None:
            try:
                mu = float(viscosity)
                if mu <= 0:
                    issues.append(f"Fluid DYNVISCOSITY must be > 0, got {mu}.")
            except (TypeError, ValueError):
                issues.append(f"Fluid DYNVISCOSITY must be numeric, got {viscosity!r}.")

        density = params.get("fluid_density") or params.get("DENSITY")
        if density is not None:
            try:
                rho = float(density)
                if rho <= 0:
                    issues.append(f"Fluid DENSITY must be > 0, got {rho}.")
            except (TypeError, ValueError):
                issues.append(f"Fluid DENSITY must be numeric, got {density!r}.")

        young = params.get("YOUNG") or params.get("young")
        if young is not None:
            try:
                E = float(young)
                if E <= 0:
                    issues.append(f"Structure YOUNG must be > 0, got {E}.")
            except (TypeError, ValueError):
                issues.append(f"Structure YOUNG must be numeric, got {young!r}.")

        poisson = params.get("NUE") or params.get("poisson_ratio")
        if poisson is not None:
            try:
                nu = float(poisson)
                if nu < 0 or nu >= 0.5:
                    issues.append(
                        f"Poisson's ratio must be in [0, 0.5), got {nu}.  "
                        f"nu=0.5 is incompressible (not supported for solid)."
                    )
            except (TypeError, ValueError):
                issues.append(f"Poisson's ratio must be numeric, got {poisson!r}.")

        # Check CLONING MATERIAL MAP presence
        has_cloning = params.get("has_cloning_material_map")
        if has_cloning is not None and not has_cloning:
            issues.append(
                "CLONING MATERIAL MAP is required for FSI.  It maps the fluid "
                "material to the ALE pseudo-material."
            )

        return issues
