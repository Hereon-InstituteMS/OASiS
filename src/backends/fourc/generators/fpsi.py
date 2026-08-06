"""Fluid-Porous-Structure Interaction (FPSI) generator for 4C.

Covers coupling of a free fluid domain (Navier-Stokes) with a porous medium
(Darcy/Biot) and a deformable solid structure.  The fluid-porous interface
uses Beavers-Joseph-Saffman conditions while the porous-structure interface
ensures kinematic compatibility.  Applications include geomechanics,
biological tissue perfusion, and filtration systems.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class FPSIGenerator(BaseGenerator):
    """Generator for Fluid-Porous-Structure Interaction problems in 4C."""

    module_key = "fpsi"
    display_name = "Fluid-Porous-Structure Interaction (FPSI)"
    problem_type = "Fluid_Porous_Structure_Interaction"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Fluid-Porous-Structure Interaction (FPSI) couples three "
                "domains: a free fluid (incompressible Navier-Stokes), a "
                "porous medium (saturated or partially saturated, governed "
                "by Biot poroelasticity), and optionally a purely structural "
                "domain.  At the fluid-porous interface, "
                "Beavers-Joseph-Saffman slip conditions govern the tangential "
                "velocity and pressure continuity is enforced.  The porous "
                "skeleton deforms and ALE mesh motion tracks the fluid "
                "domain boundary.  The PROBLEM TYPE is "
                "'Fluid_Porous_Structure_Interaction'.  The problem "
                "requires FLUID DYNAMIC, POROUS DYNAMIC (or "
                "POROELASTICITY DYNAMIC), ALE DYNAMIC, and "
                "FPSI DYNAMIC sections.  Materials include MAT_fluid "
                "for the free fluid, MAT_StructPoro for the porous "
                "skeleton, and MAT_FluidPoro for the pore fluid."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "STRUCTURAL DYNAMIC",
                "FLUID DYNAMIC",
                "POROELASTICITY DYNAMIC",
                "ALE DYNAMIC",
                "FPSI DYNAMIC",
                "SOLVER 1",
                "SOLVER 2",
                "MATERIALS",
                "CLONING MATERIAL MAP",
            ],
            "optional_sections": [
                "FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION",
                "FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/STRUCTURE",
                "IO/RUNTIME VTK OUTPUT/FLUID",
            ],
            "materials": {
                "MAT_StructPoro": {
                    "description": (
                        "Porous solid skeleton material.  Wraps an "
                        "underlying elastic material and adds porosity "
                        "and permeability for the porous medium."
                    ),
                    "parameters": {
                        "MATID": {
                            "description": (
                                "ID of the underlying structural material "
                                "(e.g. MAT_ElastHyper)"
                            ),
                            "range": "valid MAT ID",
                        },
                        "POROSITYLAW": {
                            "description": (
                                "Porosity evolution law (e.g. 'constant', "
                                "'linear')"
                            ),
                            "range": "string",
                        },
                        "INITPOROSITY": {
                            "description": "Initial porosity (volume fraction of pores)",
                            "range": "(0, 1)",
                        },
                    },
                },
                "MAT_FluidPoro": {
                    "description": (
                        "Pore fluid material within the porous medium.  "
                        "Defines fluid viscosity and density for Darcy "
                        "flow within the porous domain."
                    ),
                    "parameters": {
                        "DYNVISCOSITY": {
                            "description": "Dynamic viscosity of pore fluid [Pa s]",
                            "range": "> 0",
                        },
                        "DENSITY": {
                            "description": "Density of pore fluid [kg/m^3]",
                            "range": "> 0",
                        },
                        "PERMEABILITY": {
                            "description": "Intrinsic permeability of porous medium [m^2]",
                            "range": "> 0",
                        },
                    },
                },
                "MAT_fluid": {
                    "description": (
                        "Newtonian fluid material for the free fluid domain."
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
                "fluid_solver": {
                    "type": "UMFPACK or Belos",
                    "notes": "Solver for the free fluid Navier-Stokes equations.",
                },
                "poro_solver": {
                    "type": "UMFPACK",
                    "notes": (
                        "Solver for the coupled poroelasticity system "
                        "(displacement + pore pressure)."
                    ),
                },
            },
            "interface_conditions": {
                "Beavers_Joseph_Saffman": (
                    "Tangential slip condition at the fluid-porous "
                    "interface.  The slip coefficient alpha_BJS controls "
                    "the tangential velocity jump.  Normal velocity and "
                    "pressure are continuous."
                ),
                "FPSI_COUPLING": (
                    "Coupling conditions are applied on the fluid-porous "
                    "interface surfaces.  Both sides must be defined "
                    "in DESIGN SURF FPSI COUPLING CONDITIONS."
                ),
            },
            "pitfalls": [
                (
                    "[Input] FPSI REQUIRES ALE mesh motion "
                    "for the free fluid domain (the porous "
                    "boundary deforms). Include an ALE "
                    "DYNAMIC section AND a fluid -> ale entry "
                    "in CLONING MATERIAL MAP. Signal: the two "
                    "omissions fail very differently. Missing "
                    "ALE DYNAMIC gives a clean, actionable "
                    "abort from adapter/4C_adapter_ale.cpp — "
                    "'No linear solver defined for ALE "
                    "problems. Please set LINEAR_SOLVER in "
                    "ALE DYNAMIC to a valid number!' — which "
                    "never mentions FPSI. Missing the "
                    "fluid -> ale clone gives NO 4C "
                    "diagnostic at all: an uncaught "
                    "std::out_of_range terminates the "
                    "process, with no 'PROC 0 ERROR in' line "
                    "and no occurrence of the word 'cloning' "
                    "anywhere in the log. Neither 'FPSI: ALE "
                    "field not configured' nor 'cannot clone "
                    "material for ALE' exists. (Corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Input] The fluid-porous interface "
                    "must be declared from BOTH sides — the "
                    "DESIGN FPSI COUPLING SURF CONDITIONS "
                    "list needs an entry on the fluid side "
                    "and one on the porous side, sharing a "
                    "coupling_id. Signal: a one-sided "
                    "declaration does not run and transfer "
                    "nothing; it aborts before the first time "
                    "step with 'got N master nodes but 0 "
                    "slave nodes for coupling' from "
                    "coupling/src/adapter/"
                    "4C_coupling_adapter.cpp. The message "
                    "names neither FPSI nor the condition, so "
                    "read 'slave nodes' as 'the side you "
                    "forgot'. (Corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Input] MAT_StructPoro wraps an "
                    "UNDERLYING structural material (e.g. "
                    "MAT_ElastHyper). MATID MUST point to a "
                    "valid structural material ID. Signal: "
                    "the two ways of getting it wrong abort "
                    "in different files. Pointing MATID at a "
                    "fluid material gives 'Mat::StructPoro: "
                    "underlying material should be of type "
                    "Mat::So3Material' from "
                    "mat/4C_mat_structporo.cpp; pointing it "
                    "at an undefined id gives \"Material 'MAT "
                    "N' could not be found\" from "
                    "mat/4C_mat_par_bundle.cpp, so grepping "
                    "for the structporo file after a typo "
                    "finds nothing. The phrase 'StructPoro "
                    "inner material must be elastic' does not "
                    "exist. (Corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Numerical] Porosity INITPOROSITY must "
                    "be in (0, 1) — open interval, endpoints "
                    "excluded — and the endpoints behave "
                    "OPPOSITELY. Signal: INITPOROSITY = 0 "
                    "kills the process with SIGFPE inside "
                    "Mat::PAR::PoroLawNeoHooke::"
                    "compute_porosity — in the porous "
                    "constitutive law, not in Darcy flow — "
                    "with no 4C error line and no NaN ever "
                    "printed, because nothing gets far enough "
                    "to print one. INITPOROSITY = 1 does not "
                    "crash at all: it completes the run, says "
                    "nothing, and returns a different answer. "
                    "The dangerous endpoint is 1. Typical "
                    "range for biological / geological "
                    "problems: 0.1 - 0.5. (Corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Numerical] Permeability in "
                    "MAT_FluidPoro controls Darcy flow "
                    "resistance and both ends bite, but not "
                    "as solver behaviour. Signal: a very low "
                    "permeability produces no iterative-"
                    "solver stagnation — it aborts with 'zero "
                    "porosity!' from "
                    "fluid_ele/4C_fluid_ele_calc_poro.cpp, "
                    "raised inside FluidEleCalcPoro::"
                    "compute_stabilization_parameters, "
                    "because the permeability enters the "
                    "porous stabilisation parameter and the "
                    "porosity check there fires first. A very "
                    "high permeability runs to the end and "
                    "returns a different answer with no "
                    "remark about the Darcy regime; compare "
                    "against full Navier-Stokes to check it "
                    "still applies. (Corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Input] CLONING MATERIAL MAP must map "
                    "FLUID -> ALE material. The porous "
                    "domain does NOT need ALE cloning (it "
                    "moves with the solid skeleton, "
                    "Lagrangian formulation). Signal: none — "
                    "this rule is unenforced. Adding a "
                    "porous-structure -> ale entry as well is "
                    "accepted in silence: the run exits 0, "
                    "passes its result tests, and builds the "
                    "same discretisations as without it. "
                    "Nothing named 4C_fpsi_factory.cpp exists "
                    "and 'porous field already has Lagrangian "
                    "motion' is never printed, so the rule "
                    "has to be followed deliberately rather "
                    "than learned from a diagnostic. "
                    "(Corrected by execution 2026-08-06.)"
                ),
                (
                    "[Input] The Beavers-Joseph slip "
                    "coefficient at the fluid-porous "
                    "interface is FPSI DYNAMIC/ALPHABJ, "
                    "described in fpsi/4C_fpsi_input.cpp as "
                    "the 'Beavers-Joseph-Coefficient for "
                    "Slip-Boundary-Condition at Fluid-Porous-"
                    "Interface (0.1-4)'. It is a scalar on "
                    "the FPSI DYNAMIC section, NOT a "
                    "SLIP_COEFF component of DESIGN FPSI "
                    "COUPLING SURF CONDITIONS — that "
                    "condition accepts only E / ENTITY_TYPE / "
                    "NODE_SET_NAME / coupling_id, and any "
                    "other key is rejected at parse time with "
                    "'Could not match this input' from "
                    "core/fem/src/condition/"
                    "4C_fem_condition_definition.cpp. Signal: "
                    "omitting ALPHABJ does NOT give no-slip; "
                    "it defaults to 1.0, in the middle of the "
                    "documented range, and moving it within "
                    "that range visibly changes the answer. "
                    "(Corrected by execution 2026-08-06.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "channel_over_porous_bed_3d",
                    "description": (
                        "Free fluid channel flow over a deformable porous "
                        "bed.  The fluid exerts shear on the porous "
                        "surface (BJS slip), fluid seeps into the porous "
                        "medium via Darcy flow, and the porous skeleton "
                        "deforms under fluid pressure.  Tests FPSI "
                        "coupling, ALE mesh motion, and poroelasticity."
                    ),
                    "template_variant": "monolithic_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "monolithic_3d",
                "description": (
                    "3-D monolithic FPSI: free fluid channel over a "
                    "deformable porous bed.  Navier-Stokes + Biot "
                    "poroelasticity + ALE, BJS interface conditions, "
                    "UMFPACK solvers."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "monolithic_3d") -> str:
        templates = {
            "monolithic_3d": self._template_monolithic_3d,
        }
        if variant == "default":
            variant = "monolithic_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_monolithic_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D Monolithic Fluid-Porous-Structure Interaction (FPSI)
            #
            # Free fluid channel over a deformable porous bed.  The fluid
            # domain uses Navier-Stokes, the porous domain uses Biot
            # poroelasticity (coupled skeleton + pore fluid).  ALE mesh
            # motion tracks the fluid domain boundary deformation.
            #
            # Mesh: requires exodus files with:
            #   Fluid mesh: "fluid.e"
            #     element_block 1 = fluid domain (HEX8)
            #     node_set 1 = inlet
            #     node_set 2 = outlet
            #     node_set 3 = walls
            #     node_set 4 = FPSI interface (fluid side)
            #   Poro mesh: "poro.e"
            #     element_block 1 = porous domain (HEX8)
            #     node_set 1 = FPSI interface (poro side)
            #     node_set 2 = poro bottom (fixed)
            # ---------------------------------------------------------------
            TITLE:
              - "3-D FPSI (fluid-porous-structure) -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Fluid_Porous_Structure_Interaction"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/FLUID:
              OUTPUT_FLUID: true
              VELOCITY: true
              PRESSURE: true
            IO/RUNTIME VTK OUTPUT/STRUCTURE:
              OUTPUT_STRUCTURE: true
              DISPLACEMENT: true

            # == Structure (porous skeleton) ===================================
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "Statics"
              LINEAR_SOLVER: 1
              TOLRES: <structure_residual_tolerance>
              TOLDISP: <structure_displacement_tolerance>

            # == Fluid (free fluid) ============================================
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

            # == Poroelasticity ================================================
            POROELASTICITY DYNAMIC:
              TIMESTEP: <poro_timestep>
              NUMSTEP: <poro_num_steps>
              MAXTIME: <poro_max_time>
              COUPALGO: "poro_monolithic"
              LINEAR_SOLVER: 1

            # == ALE mesh motion ===============================================
            ALE DYNAMIC:
              ALE_TYPE: "springs_spatial"
              LINEAR_SOLVER: 1
              MAXITER: <ale_max_iterations>

            # == FPSI coupling =================================================
            FPSI DYNAMIC:
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              RESULTSEVERY: <results_output_interval>
              COUPALGO: "fpsi_monolithic"

            # == Solvers =======================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "structure_poro_solver"
            SOLVER 2:
              SOLVER: "UMFPACK"
              NAME: "fluid_solver"

            # == Materials =====================================================
            MATERIALS:
              # Free fluid
              - MAT: 1
                MAT_fluid:
                  DYNVISCOSITY: <fluid_dynamic_viscosity>
                  DENSITY: <fluid_density>
              # Porous skeleton (wraps elastic sub-material)
              - MAT: 2
                MAT_StructPoro:
                  MATID: 3
                  POROSITYLAW: "<porosity_law>"
                  INITPOROSITY: <initial_porosity>
              # Elastic sub-material for skeleton
              - MAT: 3
                MAT_ElastHyper:
                  NUMMAT: 1
                  MATIDS: [4]
                  DENS: <skeleton_density>
              - MAT: 4
                ELAST_CoupNeoHooke:
                  YOUNG: <skeleton_Young_modulus>
              # Pore fluid in porous domain
              - MAT: 5
                MAT_FluidPoro:
                  DYNVISCOSITY: <pore_fluid_viscosity>
                  DENSITY: <pore_fluid_density>
                  PERMEABILITY: <intrinsic_permeability>
              # ALE pseudo-material (cloned from fluid)
              - MAT: 6
                MAT_Struct_StVenantKirchhoff:
                  YOUNG: <ale_Young_modulus>
                  NUE: <ale_Poisson_ratio>
                  DENS: <ale_density>

            # Clone fluid mesh -> ALE mesh
            CLONING MATERIAL MAP:
              - SRC_FIELD: "fluid"
                SRC_MAT: 1
                TAR_FIELD: "ale"
                TAR_MAT: 6

            # == Boundary Conditions ===========================================

            # Porous skeleton: fixed bottom
            DESIGN SURF DIRICH CONDITIONS:
              - E: <poro_bottom_face_id>
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]

            # Fluid: inlet
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

            # ALE: fixed outer fluid boundaries
            DESIGN SURF ALE DIRICH CONDITIONS:
              - E: <ale_fixed_face_id>
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]

            # FPSI coupling interface
            DESIGN SURF FPSI COUPLING CONDITIONS:
              - E: <fpsi_interface_fluid_id>
                coupling_id: 1
                INTERFACE_SIDE: "fluid"
              - E: <fpsi_interface_poro_id>
                coupling_id: 1
                INTERFACE_SIDE: "poro"

            # Inlet ramp function
            FUNCT<inlet_ramp_function>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<inlet_ramp_expression>"

            # == Geometry ======================================================
            FLUID GEOMETRY:
              FILE: "<fluid_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  FLUID:
                    HEX8:
                      MAT: 1
                      NA: ALE

            PORO GEOMETRY:
              FILE: "<poro_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  SOLIDPORO:
                    HEX8:
                      MAT: 2
                      KINEM: <kinematics>
                      POROFLUIDMAT: 5

            RESULT DESCRIPTION:
              - FLUID:
                  DIS: "fluid"
                  NODE: <result_fluid_node_id>
                  QUANTITY: "velx"
                  VALUE: <expected_fluid_velocity>
                  TOLERANCE: <result_tolerance>
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_poro_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_skeleton_displacement>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check porosity
        porosity = params.get("INITPOROSITY")
        if porosity is not None:
            try:
                phi = float(porosity)
                if phi <= 0 or phi >= 1:
                    issues.append(
                        f"INITPOROSITY must be in (0, 1), got {phi}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"INITPOROSITY must be a number in (0, 1), "
                    f"got {porosity!r}."
                )

        # Check permeability
        perm = params.get("PERMEABILITY")
        if perm is not None:
            try:
                k = float(perm)
                if k <= 0:
                    issues.append(
                        f"PERMEABILITY must be > 0, got {k}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"PERMEABILITY must be a positive number, "
                    f"got {perm!r}."
                )

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

        # Check Young's modulus
        young = params.get("YOUNG")
        if young is not None:
            try:
                e = float(young)
                if e <= 0:
                    issues.append(
                        f"YOUNG must be > 0, got {e}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"YOUNG must be a positive number, got {young!r}."
                )

        # Check CLONING MATERIAL MAP
        has_cloning = params.get("has_cloning_material_map")
        if has_cloning is not None and not has_cloning:
            issues.append(
                "CLONING MATERIAL MAP is required for FPSI.  "
                "It maps fluid material to the ALE pseudo-material."
            )

        return issues
