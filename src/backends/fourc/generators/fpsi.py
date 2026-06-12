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
                    "boundary deforms). Include ALE DYNAMIC "
                    "section + CLONING MATERIAL MAP for "
                    "fluid -> ALE. Signal: missing ALE "
                    "DYNAMIC raises 'FPSI: ALE field not "
                    "configured' at setup; missing CLONING "
                    "MATERIAL MAP raises 'cannot clone "
                    "material for ALE'. Both are needed "
                    "even if mesh motion is small. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] The fluid-porous interface "
                    "must be defined from BOTH sides — FPSI "
                    "coupling conditions reference node "
                    "sets on the fluid AND porous meshes. "
                    "Signal: defining only the fluid-side "
                    "FPSI coupling condition gives zero "
                    "force transfer to the porous skeleton "
                    "(it does not deform); defining only "
                    "the porous-side gives zero fluid "
                    "feedback. Both required for two-way "
                    "coupling. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] MAT_StructPoro wraps an "
                    "UNDERLYING elastic material (e.g. "
                    "MAT_ElastHyper). MATID parameter MUST "
                    "point to a valid structural material "
                    "ID. Signal: wrong MATID (pointing to a "
                    "fluid material or undefined ID) "
                    "aborts with 'StructPoro inner material "
                    "must be elastic' from "
                    "4C_mat_structporo.cpp. Use the nested-"
                    "material pattern: MAT_StructPoro with "
                    "MATID -> MAT_ElastHyper. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Numerical] Porosity INITPOROSITY must "
                    "be in (0, 1) — open interval, "
                    "endpoints excluded. Signal: "
                    "INITPOROSITY = 0 (no pores) or "
                    "INITPOROSITY = 1 (all pores, no "
                    "skeleton) triggers DIVISION BY ZERO in "
                    "Darcy flow (NaN appears in fluid "
                    "velocity at the first iteration). "
                    "Typical range for biological / "
                    "geological problems: 0.1 - 0.5. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Numerical] Permeability in MAT_"
                    "FluidPoro controls Darcy flow "
                    "resistance. VERY LOW permeability "
                    "(< 1e-15 m^2) makes the system "
                    "extremely stiff — iterative solver "
                    "iteration counts explode; VERY HIGH "
                    "permeability (> 1e-6 m^2) approaches "
                    "free flow and the Darcy approximation "
                    "becomes invalid (Reynolds-pore "
                    "criterion). Signal: stagnating CG/"
                    "GMRES at very low K; for very high K, "
                    "compare against full NS to verify the "
                    "Darcy regime applies. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] CLONING MATERIAL MAP must map "
                    "FLUID -> ALE material. The porous "
                    "domain does NOT need ALE cloning (it "
                    "moves with the solid skeleton, "
                    "Lagrangian formulation). Signal: "
                    "trying to clone porous -> ALE raises "
                    "'porous field already has Lagrangian "
                    "motion' from 4C_fpsi_factory.cpp; only "
                    "the free-fluid Eulerian field needs an "
                    "ALE pseudo-discretisation. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] Beavers-Joseph-Saffman slip "
                    "coefficient must be specified at the "
                    "fluid-porous interface. Signal: "
                    "omitting BJS defaults to no-slip — "
                    "the fluid sees the porous wall as a "
                    "solid wall, which over-predicts wall "
                    "shear by 2-5x for permeable beds and "
                    "under-predicts seepage. The published "
                    "BJS coefficient is sqrt(K) / "
                    "sqrt(viscosity) — set the SLIP_COEFF "
                    "in DESIGN FPSI COUPLING SURF "
                    "CONDITIONS. (Audit 2026-06-02.)"
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
