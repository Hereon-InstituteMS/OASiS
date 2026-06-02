"""Structure-Scalar Interaction (SSI) generator for 4C.

Covers monolithic and partitioned coupling of structural mechanics with
scalar transport (including electrochemistry).  Key application: electrode
mechanics in battery simulations where lithium intercalation causes
volumetric expansion, and mechanical stress affects diffusion and
electrochemical kinetics.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class SSIGenerator(BaseGenerator):
    """Generator for Structure-Scalar Interaction problems in 4C."""

    module_key = "ssi"
    display_name = "Structure-Scalar Interaction (SSI / Electrode Mechanics)"
    problem_type = "Structure_Scalar_Interaction"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Structure-Scalar Interaction (SSI) couples a structural "
                "mechanics field with a scalar transport field.  The primary "
                "application is electrode mechanics in lithium-ion batteries, "
                "where intercalation of lithium ions causes volumetric "
                "swelling of the electrode particles, and mechanical stress "
                "in turn affects ion diffusion and electrochemical reaction "
                "kinetics (Butler-Volmer).  4C supports monolithic "
                "(ssi_Monolithic) and partitioned (ssi_IterStagg) coupling.  "
                "The PROBLEM TYPE is 'Structure_Scalar_Interaction'.  The "
                "coupling is controlled via the 'SSI CONTROL' section.  "
                "When electrochemistry is involved, set SCATRATIMINTTYPE: "
                "'Elch' and include ELCH CONTROL and S2I COUPLING sections."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "STRUCTURAL DYNAMIC",
                "SCALAR TRANSPORT DYNAMIC",
                "SSI CONTROL",
                "SSI CONTROL/MONOLITHIC",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "ELCH CONTROL",
                "SSI CONTROL/ELCH",
                "SCALAR TRANSPORT DYNAMIC/S2I COUPLING",
                "SCALAR TRANSPORT DYNAMIC/STABILIZATION",
                "SCALAR TRANSPORT DYNAMIC/NONLINEAR",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/STRUCTURE",
            ],
            "materials": {
                "MAT_MultiplicativeSplitDefgradElastHyper": {
                    "description": (
                        "Multiplicative split of the deformation gradient "
                        "into elastic and inelastic parts.  Used for "
                        "electrode mechanics where lithium intercalation "
                        "causes inelastic volumetric growth.  References "
                        "an elastic sub-material and an inelastic "
                        "deformation gradient factor."
                    ),
                    "parameters": {
                        "NUMMATEL": {
                            "description": "Number of elastic sub-materials",
                            "range": ">= 1",
                        },
                        "MATIDSEL": {
                            "description": (
                                "List of elastic sub-material IDs "
                                "(e.g. ELAST_CoupSVK)"
                            ),
                            "range": "valid MAT IDs",
                        },
                        "NUMFACINEL": {
                            "description": (
                                "Number of inelastic deformation gradient "
                                "factors"
                            ),
                            "range": ">= 1",
                        },
                        "INELDEFGRADFACIDS": {
                            "description": (
                                "List of inelastic deformation gradient "
                                "factor material IDs"
                            ),
                            "range": "valid MAT IDs",
                        },
                        "DENS": {
                            "description": "Mass density",
                            "range": "> 0",
                        },
                    },
                },
                "MAT_electrode": {
                    "description": (
                        "Electrode material for lithium-ion batteries.  "
                        "Defines concentration-dependent diffusion "
                        "coefficient, electronic conductivity, maximum "
                        "concentration, and open-circuit potential (OCP) "
                        "model.  Used in the scalar transport field."
                    ),
                    "parameters": {
                        "DIFF_PARA_NUM": {
                            "description": (
                                "Number of diffusion coefficient parameters"
                            ),
                            "range": ">= 1",
                        },
                        "DIFF_PARA": {
                            "description": "Diffusion coefficient parameter(s)",
                            "range": "> 0",
                        },
                        "COND_PARA_NUM": {
                            "description": (
                                "Number of conductivity parameters"
                            ),
                            "range": ">= 1",
                        },
                        "COND_PARA": {
                            "description": "Electronic conductivity parameter(s)",
                            "range": "> 0",
                        },
                        "C_MAX": {
                            "description": (
                                "Maximum lithium concentration in the "
                                "electrode material"
                            ),
                            "range": "> 0",
                        },
                        "CHI_MAX": {
                            "description": (
                                "Maximum state of charge (stoichiometry)"
                            ),
                            "range": "> 0",
                        },
                    },
                },
                "ELAST_CoupSVK": {
                    "description": (
                        "Coupled St. Venant-Kirchhoff elastic material.  "
                        "Used as the elastic sub-material within "
                        "MAT_MultiplicativeSplitDefgradElastHyper."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus",
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Poisson's ratio",
                            "range": "(0, 0.5)",
                        },
                    },
                },
                "MAT_InelasticDefgradNoGrowth": {
                    "description": (
                        "Trivial inelastic deformation gradient factor "
                        "that applies no growth.  Used as a placeholder "
                        "or for problems where only elastic deformation "
                        "is desired."
                    ),
                },
            },
            "solver": {
                "monolithic": {
                    "type": "UMFPACK for small problems, Belos + block prec for large",
                    "notes": (
                        "The monolithic SSI solver handles the coupled "
                        "displacement-concentration system.  For electrode "
                        "mechanics with electrochemistry, the system can be "
                        "large and may benefit from iterative solvers."
                    ),
                },
            },
            "coupling_algorithms": {
                "ssi_Monolithic": (
                    "Fully coupled monolithic solve.  Both fields are "
                    "assembled into a single block system and solved "
                    "simultaneously.  Most robust for strong coupling."
                ),
                "ssi_IterStagg": (
                    "Iterative staggered (partitioned) approach.  Fields "
                    "are solved alternately with relaxation until "
                    "convergence.  Cheaper per iteration but may need "
                    "more iterations for strongly coupled problems."
                ),
            },
            "electrochemistry_settings": {
                "SCATRATIMINTTYPE": (
                    "Set to 'Elch' in SSI CONTROL to enable the "
                    "electrochemistry scalar transport formulation."
                ),
                "EQUPOT": (
                    "Electroneutrality condition in ELCH CONTROL.  Options: "
                    "'divi' (divergence-based), 'ENC' (electroneutrality "
                    "constraint)."
                ),
                "DIFFCOND_FORMULATION": (
                    "Set to true in ELCH CONTROL for diffusion-conduction "
                    "formulation (concentrated solution theory)."
                ),
                "INITPOTCALC": (
                    "Set to true in SSI CONTROL/ELCH to compute a "
                    "consistent initial electric potential field."
                ),
            },
            "pitfalls": [
                (
                    "[Input] SCATRATIMINTTYPE must be set to "
                    "'Elch' in SSI CONTROL when "
                    "electrochemistry is involved. Signal: "
                    "omitting it defaults to plain scalar "
                    "transport without electrochemical "
                    "source terms — the BV (Butler-Volmer) "
                    "current at electrode interfaces is "
                    "ZERO and lithium intercalation produces "
                    "no concentration change. Set "
                    "SCATRATIMINTTYPE: Elch for any "
                    "electrode problem. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] For electrode problems, "
                    "structural elements MUST use "
                    "MAT_MultiplicativeSplitDefgradElastHyper "
                    "with appropriate inelastic growth "
                    "factors. Signal: a standard elastic "
                    "material (MAT_ElastHyper) cannot "
                    "capture swelling — lithium intercalation "
                    "produces zero deformation even at high "
                    "concentration. The multiplicative split "
                    "F = F_e * F_g separates elastic and "
                    "growth contributions; F_g is driven by "
                    "concentration. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] S2I (scatra-scatra interface) "
                    "coupling conditions are needed for "
                    "electrode-electrolyte interfaces. Set "
                    "COUPLINGTYPE in SCALAR TRANSPORT "
                    "DYNAMIC/S2I COUPLING. Signal: omitting "
                    "S2I COUPLING gives zero current "
                    "across the interface; the catalog has "
                    "the structure connected mechanically "
                    "but the SCATRA fields decoupled. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Input] VELOCITYFIELD must be set to "
                    "'Navier_Stokes' in SCALAR TRANSPORT "
                    "DYNAMIC for SSI — enables coupling "
                    "with the structural velocity field. "
                    "Signal: default VELOCITYFIELD: zero "
                    "makes the scatra ignore the structural "
                    "motion; transport of species in a "
                    "deforming domain is wrong by O(|v_s|) "
                    "— visible as a stationary "
                    "concentration field even with large "
                    "structural deformation. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Numerical] CONVFORM should typically "
                    "be 'conservative' for SSI to ensure "
                    "mass conservation in the deforming "
                    "domain. Signal: CONVFORM: convective on "
                    "a moving-mesh scatra produces O(dv_s/"
                    "dt) mass-balance drift over time — "
                    "integral of concentration drifts by "
                    "1-5% per cycle, increasing with "
                    "deformation amplitude. The "
                    "conservative form integrates the "
                    "divergence-form transport equation "
                    "and preserves mass to machine "
                    "precision. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] Monolithic SSI (SSI CONTROL/"
                    "MONOLITHIC) needs MATRIXTYPE set "
                    "appropriately: 'sparse' for direct "
                    "solvers, 'block' for block-"
                    "preconditioned iterative solvers. "
                    "Signal: 'sparse' with a Belos GMRES + "
                    "block-Teko preconditioner ignores the "
                    "block structure — preconditioner "
                    "iterations explode; 'block' with a "
                    "direct UMFPACK solver wastes memory "
                    "constructing block sub-blocks. Match "
                    "MATRIXTYPE to the SOLVER type. (Audit "
                    "2026-06-02.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "electrode_intercalation_3d",
                    "description": (
                        "Two electrode particles separated by a membrane "
                        "with Butler-Volmer kinetics.  Lithium intercalates "
                        "from one side causing swelling.  Tests "
                        "concentration-dependent OCP, S2I interface "
                        "coupling, and structural deformation."
                    ),
                    "template_variant": "monolithic_elch_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "monolithic_elch_3d",
                "description": (
                    "3-D monolithic SSI with electrochemistry: two electrode "
                    "blocks with Butler-Volmer S2I interface.  "
                    "MAT_MultiplicativeSplitDefgradElastHyper for structure, "
                    "MAT_electrode for scalar transport, UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "monolithic_elch_3d") -> str:
        templates = {
            "monolithic_elch_3d": self._template_monolithic_elch_3d,
        }
        if variant == "default":
            variant = "monolithic_elch_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_monolithic_elch_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D Monolithic Structure-Scalar Interaction (Electrochemistry)
            #
            # Two electrode blocks with scatra-scatra interface (S2I)
            # coupling using Butler-Volmer kinetics.  The structural field
            # deforms due to lithium intercalation swelling.
            #
            # Mesh: requires an exodus file with:
            #   element_block 1 = left electrode (HEX8)
            #   element_block 2 = right electrode (HEX8)
            #   node_set 1 = left structural Dirichlet (fixed face)
            #   node_set 2 = right structural Dirichlet (prescribed displacement)
            #   node_set 3 = S2I interface (left side)
            #   node_set 4 = S2I interface (right side)
            #   node_set 5 = potential Dirichlet BC face
            #   node_set 6 = potential Neumann BC face
            # ---------------------------------------------------------------
            TITLE:
              - "3-D SSI with electrochemistry -- generated template"
            PROBLEM TYPE:
              PROBLEMTYPE: "Structure_Scalar_Interaction"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/STRUCTURE:
              OUTPUT_STRUCTURE: true
              DISPLACEMENT: true

            # == Structure =====================================================
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "OneStepTheta"
              LINEAR_SOLVER: 1

            # == Scalar Transport (Electrochemistry) ===========================
            SCALAR TRANSPORT DYNAMIC:
              SOLVERTYPE: "nonlinear"
              VELOCITYFIELD: "Navier_Stokes"
              INITIALFIELD: "field_by_condition"
              CONVFORM: "conservative"
              SKIPINITDER: true
              LINEAR_SOLVER: 1
            SCALAR TRANSPORT DYNAMIC/STABILIZATION:
              STABTYPE: "no_stabilization"
              DEFINITION_TAU: "Zero"
              EVALUATION_TAU: "integration_point"
              EVALUATION_MAT: "integration_point"
            SCALAR TRANSPORT DYNAMIC/S2I COUPLING:
              COUPLINGTYPE: "MatchingNodes"

            # == Electrochemistry control ======================================
            ELCH CONTROL:
              EQUPOT: "<electroneutrality_method>"
              DIFFCOND_FORMULATION: <diffcond_flag>
              COUPLE_BOUNDARY_FLUXES: <couple_boundary_fluxes_flag>

            # == SSI coupling ==================================================
            SSI CONTROL:
              RESTARTEVERY: <restart_interval>
              NUMSTEP: <number_of_steps>
              TIMESTEP: <timestep>
              RESULTSEVERY: <results_output_interval>
              COUPALGO: ssi_Monolithic
              SCATRATIMINTTYPE: "Elch"
            SSI CONTROL/MONOLITHIC:
              ABSTOLRES: <absolute_residual_tolerance>
              LINEAR_SOLVER: 1
              MATRIXTYPE: "sparse"
            SSI CONTROL/ELCH:
              INITPOTCALC: <compute_initial_potential>

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "direct_solver"

            # == Materials =====================================================
            MATERIALS:
              # Structural material: multiplicative decomposition F = F_e * F_i
              - MAT: 1
                MAT_MultiplicativeSplitDefgradElastHyper:
                  NUMMATEL: 1
                  MATIDSEL: [<elastic_sub_material_id>]
                  NUMFACINEL: 1
                  INELDEFGRADFACIDS: [<inelastic_factor_material_id>]
                  DENS: <structural_density>
              # Second block structural material (same formulation)
              - MAT: 2
                MAT_MultiplicativeSplitDefgradElastHyper:
                  NUMMATEL: 1
                  MATIDSEL: [<elastic_sub_material_id>]
                  NUMFACINEL: 1
                  INELDEFGRADFACIDS: [<inelastic_factor_material_id>]
                  DENS: <structural_density>
              # Elastic sub-material (St. Venant-Kirchhoff)
              - MAT: 3
                ELAST_CoupSVK:
                  YOUNG: <Young_modulus>
                  NUE: <Poisson_ratio>
              # Inelastic growth factor (no growth placeholder)
              - MAT: 4
                MAT_InelasticDefgradNoGrowth: {}
              # Electrode material (scalar transport)
              - MAT: 5
                MAT_electrode:
                  DIFF_COEF_CONC_DEP_FUNCT: <diff_coef_concentration_function>
                  DIFF_COEF_TEMP_SCALE_FUNCT: <diff_coef_temperature_function>
                  COND_CONC_DEP_FUNCT: <cond_concentration_function>
                  COND_TEMP_SCALE_FUNCT: <cond_temperature_function>
                  DIFF_PARA_NUM: <num_diffusion_parameters>
                  DIFF_PARA: [<diffusion_coefficient>]
                  COND_PARA_NUM: <num_conductivity_parameters>
                  COND_PARA: [<electronic_conductivity>]
                  C_MAX: <max_lithium_concentration>
                  CHI_MAX: <max_stoichiometry>
                  OCP_MODEL:
                    Function:
                      OCP_FUNCT_NUM: <ocp_function_id>
                    X_MIN: <ocp_x_min>

            # == Boundary Conditions ===========================================

            # Structural Dirichlet: fixed face
            DESIGN SURF DIRICH CONDITIONS:
              - E: <fixed_face_id>
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]
              # Structural: prescribed displacement face
              - E: <loaded_face_id>
                NUMDOF: 3
                ONOFF: [<active_displacement_dofs>]
                VAL: [<prescribed_displacement_values>]
                FUNCT: [<displacement_time_functions>]

            # Potential Dirichlet BC
            DESIGN SURF TRANSPORT DIRICH CONDITIONS:
              - E: <potential_dirichlet_face_id>
                NUMDOF: <num_scalar_dofs>
                ONOFF: [<active_scalar_dofs>]
                VAL: [<potential_values>]
                FUNCT: [<potential_time_functions>]

            # S2I interface conditions
            DESIGN SURF S2I COUPLING CONDITIONS:
              - E: <s2i_face_left>
                S2I_KINETICS_ID: <kinetics_condition_id>
                INTERFACE_SIDE: "slave"
              - E: <s2i_face_right>
                S2I_KINETICS_ID: <kinetics_condition_id>
                INTERFACE_SIDE: "master"

            # OCP function
            FUNCT<ocp_function_id>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<ocp_expression>"

            # == Geometry ======================================================
            STRUCTURE GEOMETRY:
              FILE: "<mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  SOLIDSCATRA:
                    HEX8:
                      MAT: 1
                      KINEM: <kinematics>
                      TYPE: Undefined
                - ID: 2
                  SOLIDSCATRA:
                    HEX8:
                      MAT: 2
                      KINEM: <kinematics>
                      TYPE: Undefined

            RESULT DESCRIPTION:
              - SCATRA:
                  DIS: "scatra"
                  NODE: <result_node_id>
                  QUANTITY: "phi"
                  VALUE: <expected_concentration>
                  TOLERANCE: <result_tolerance>
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_displacement>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

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

        # Check Poisson's ratio
        nue = params.get("NUE")
        if nue is not None:
            try:
                nu = float(nue)
                if nu <= 0 or nu >= 0.5:
                    issues.append(
                        f"NUE must be in (0, 0.5), got {nu}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"NUE must be a number in (0, 0.5), got {nue!r}."
                )

        # Check C_MAX
        c_max = params.get("C_MAX")
        if c_max is not None:
            try:
                cm = float(c_max)
                if cm <= 0:
                    issues.append(
                        f"C_MAX (max concentration) must be > 0, got {cm}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"C_MAX must be a positive number, got {c_max!r}."
                )

        # Check coupling algorithm
        coupalgo = params.get("COUPALGO")
        if coupalgo is not None and coupalgo not in (
            "ssi_Monolithic", "ssi_IterStagg",
            "ssi_IterStagg_FixedRel_ScatraToSolid",
            "ssi_IterStagg_FixedRel_SolidToScatra",
        ):
            issues.append(
                f"COUPALGO should be 'ssi_Monolithic' or 'ssi_IterStagg' "
                f"(or a variant), got {coupalgo!r}."
            )

        # Check SCATRATIMINTTYPE
        scatra_type = params.get("SCATRATIMINTTYPE")
        if scatra_type is not None and scatra_type not in ("Elch", "Standard"):
            issues.append(
                f"SCATRATIMINTTYPE must be 'Elch' or 'Standard', "
                f"got {scatra_type!r}."
            )

        # Check density
        dens = params.get("DENS")
        if dens is not None:
            try:
                d = float(dens)
                if d <= 0:
                    issues.append(f"DENS must be > 0, got {d}.")
            except (TypeError, ValueError):
                issues.append(
                    f"DENS must be a positive number, got {dens!r}."
                )

        return issues
