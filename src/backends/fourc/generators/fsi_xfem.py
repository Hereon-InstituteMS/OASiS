"""XFEM Fluid-Structure Interaction (FSI XFEM) generator for 4C.

Covers FSI problems where the fluid-structure interface is captured via
XFEM instead of body-fitted (ALE) mesh motion.  The structural mesh moves
through a fixed background fluid mesh, which is enriched with XFEM
discontinuities at the interface.  This approach avoids ALE mesh
distortion issues for large structural deformations.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class FSIXFEMGenerator(BaseGenerator):
    """Generator for XFEM-based FSI problems in 4C."""

    module_key = "fsi_xfem"
    display_name = "FSI XFEM (Fluid-Structure Interaction with XFEM)"
    problem_type = "Fluid_Structure_Interaction_XFEM"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "XFEM-based Fluid-Structure Interaction couples an "
                "incompressible Navier-Stokes fluid with a deformable "
                "structure using the eXtended Finite Element Method for "
                "the fluid field.  Unlike classical ALE-based FSI, the "
                "fluid mesh is FIXED (Eulerian) and the structural mesh "
                "moves through it.  The fluid approximation space is "
                "enriched with XFEM basis functions at the interface to "
                "capture the velocity and pressure discontinuities.  "
                "Nitsche's method enforces the interface kinematic and "
                "traction conditions weakly.  The PROBLEM TYPE is "
                "'Fluid_Structure_Interaction_XFEM'.  Required dynamics "
                "sections include STRUCTURAL DYNAMIC, XFLUID DYNAMIC, "
                "and FSI DYNAMIC.  No ALE DYNAMIC section is needed "
                "(this is a key advantage over standard FSI).  Ghost-"
                "penalty stabilisation prevents ill-conditioning from "
                "small cut elements."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "STRUCTURAL DYNAMIC",
                "FLUID DYNAMIC",
                "XFLUID DYNAMIC",
                "FSI DYNAMIC",
                "SOLVER 1",
                "SOLVER 2",
                "MATERIALS",
                "STRUCTURE GEOMETRY",
                "FLUID GEOMETRY",
            ],
            "optional_sections": [
                "XFLUID DYNAMIC/GHOST PENALTY",
                "XFLUID DYNAMIC/STABILIZATION",
                "FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION",
                "FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES",
                "STRUCTURAL DYNAMIC/GENALPHA",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/STRUCTURE",
                "IO/RUNTIME VTK OUTPUT/FLUID",
            ],
            "materials": {
                "MAT_fluid": {
                    "description": (
                        "Newtonian fluid for the background fluid domain."
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
                        "Hyperelastic material for the structure.  "
                        "Suitable for large-deformation FSI where ALE "
                        "would fail."
                    ),
                    "parameters": {
                        "NUMMAT": {
                            "description": "Number of sub-materials",
                            "range": "1",
                        },
                        "MATIDS": {
                            "description": "List of sub-material IDs",
                            "range": "",
                        },
                        "DENS": {
                            "description": "Structural density [kg/m^3]",
                            "range": "> 0",
                        },
                    },
                },
                "MAT_Struct_StVenantKirchhoff": {
                    "description": (
                        "St. Venant-Kirchhoff material for small-strain "
                        "structural response."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus [Pa]",
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Poisson's ratio",
                            "range": "[0, 0.5)",
                        },
                        "DENS": {
                            "description": "Structural density [kg/m^3]",
                            "range": "> 0",
                        },
                    },
                },
            },
            "solver": {
                "fluid_solver": {
                    "type": "UMFPACK or Belos",
                    "notes": (
                        "Direct solver recommended for XFEM due to "
                        "variable system size from cut elements."
                    ),
                },
                "structure_solver": {
                    "type": "UMFPACK",
                    "notes": "Standard structural solver.",
                },
            },
            "xfem_fsi_parameters": {
                "COUPLING_METHOD": (
                    "Nitsche (recommended) for weak enforcement of "
                    "velocity continuity and traction equilibrium at "
                    "the FSI interface."
                ),
                "GHOST_PENALTY": (
                    "Stabilisation for small cut elements.  Essential "
                    "for robustness when the structural mesh passes "
                    "close to fluid element boundaries."
                ),
                "XFEM_FSI_COUPALGO": (
                    "Coupling algorithm: 'xfem_monolithic' for full "
                    "monolithic coupling or 'xfem_partitioned' for "
                    "partitioned (staggered) approach."
                ),
            },
            "pitfalls": [
                (
                    '[Input] FSI-XFEM does not use ALE mesh motion: the fluid mesh '
                    'is fixed and the structure interface cuts through it via XFEM '
                    'enrichment. But 4C does NOT object to leftover ALE plumbing. '
                    'Signal: a deck carrying BOTH an ALE DYNAMIC section and a '
                    "CLONING MATERIAL MAP runs to 'processor 0 finished normally' "
                    "and reproduces the reference results; there is no 'XFEM and "
                    "ALE are mutually exclusive' message and no "
                    '4C_xfem_fluid_setup.cpp in the source. Note also that this '
                    "problem type creates a discretisation named 'ale' regardless, "
                    "so grepping the log for 'ale' proves nothing either way. "
                    'Remove the sections because they are meaningless, not because '
                    '4C will tell you. (Audit 2026-06-02; corrected by execution '
                    '2026-08-06.)'
                ),
                (
                    '[Numerical] Ghost-penalty stabilisation controls the '
                    'conditioning of cut elements with small volume fractions, but '
                    'on a monolithic XFSI problem its absence shows up as a wrong '
                    'answer, not a solver error. Signal: with GHOST_PENALTY_STAB: '
                    'false, or GHOST_PENALTY_FAC: 0.0, the Newton loop converges, '
                    'the run reaches its result test and the pinned values move; no '
                    'condition number, singular-matrix or factorisation message is '
                    'printed, and a direct solver such as UMFPACK factorises '
                    'without complaint. Compare against a reference run rather than '
                    'waiting for the solver to object. (Audit 2026-06-02; corrected '
                    'by execution 2026-08-06.)'
                ),
                (
                    '[Mesh] The structural mesh acts as the CUTTER MESH for the '
                    'fluid XFEM enrichment and its coupled surface must be closed. '
                    'An open surface is caught, but by the quadrature, not by a '
                    'classification check. Signal: dropping one side of the '
                    "coupling surface aborts with 'negative volume predicted by the "
                    "DirectDivergence integration rule;' from "
                    '4C_cut_direct_divergence.cpp, usually alongside a '
                    "Cut::Mesh::DebugDump line. There is no 'inside/outside "
                    "classification inconsistent' message and no silent permeation: "
                    'the run stops. Check that every face of the cutter body '
                    'carries the coupling condition. (Audit 2026-06-02; corrected '
                    'by execution 2026-08-06.)'
                ),
                (
                    '[Numerical] The Nitsche penalty is NIT_STAB_FAC in XFLUID '
                    'DYNAMIC/STABILIZATION (default 35); 4C applies the viscosity '
                    'and cut-size scaling itself via VISC_STAB_TRACE_ESTIMATE and '
                    'VISC_STAB_HK, so the value is a bare dimensionless factor. '
                    'Signal: setting it far too low or far too high does NOT stall '
                    'Newton and does not print a condition number. The run '
                    'converges, reaches the result test, and the coupled '
                    'displacements and velocities are simply wrong. Leave it at the '
                    'default unless a reference solution tells you otherwise. '
                    '(Audit 2026-06-02; corrected by execution 2026-08-06.)'
                ),
                (
                    '[Numerical] Choose the time step so the structure does not '
                    'traverse more than one fluid element per step; cut-topology '
                    'changes are reconstructed by a semi-Lagrangean search for '
                    'nodes the interface has just uncovered. Signal: too large a '
                    "step aborts with 'Initial point for node N for finding the "
                    "Lagrangean origin not in domain!' from "
                    '4C_xfem_xfluid_timeInt_std_SemiLagrange.cpp. Despite being '
                    'printed with a WARNING prefix it is thrown, not logged: the '
                    'run stops and never reaches its result test. Reduce TIMESTEP '
                    'until the message disappears. (Audit 2026-06-02; corrected by '
                    'execution 2026-08-06.)'
                ),
                (
                    '[Output] You cannot inspect an XFSI fluid field in ParaView '
                    'through runtime VTK output, because 4C refuses to produce it. '
                    'Signal: adding IO/RUNTIME VTK OUTPUT with the FLUID and '
                    'STRUCTURE sub-sections aborts before the first step with '
                    "'Runtime output is not available in the old structure time "
                    'integration! You need to take the new one, i.e. set '
                    "INT_STRATEGY: Standard!' from 4C_structure_timint.cpp, and "
                    'setting INT_STRATEGY: Standard does not help, because the XFSI '
                    'adapter builds the legacy integrator regardless and the same '
                    'throw reappears. No .vtu is written either way. Use the '
                    'Ensight .result files or the Gmsh .pos output instead. (Audit '
                    '2026-06-02; corrected by execution 2026-08-06.)'
                ),
                (
                    '[Input] Fluid element blocks must use NA: Euler; there is no '
                    'mesh motion in the fluid domain under XFEM. Signal: NA: ALE '
                    'parses cleanly, builds the discretisations, and then aborts '
                    "with 'Cannot find state dispnp in discretization fluid' from "
                    '4C_fem_discretization.hpp. The message names neither XFEM nor '
                    'ALE nor the NA keyword that was mis-set, so it is easy to '
                    'misread as an output or restart problem; there is no '
                    "4C_fluid_xfem_factory.cpp and no 'kinematic type incompatible' "
                    'string in 4C. (Audit 2026-06-02; corrected by execution '
                    '2026-08-06.)'
                ),
            ],
            "typical_experiments": [
                {
                    "name": "falling_sphere_xfem_3d",
                    "description": (
                        "A rigid or elastic sphere falling through a "
                        "viscous fluid.  The sphere surface cuts through "
                        "the fixed fluid mesh via XFEM.  Tests Nitsche "
                        "coupling, ghost penalty, and structural motion "
                        "through the fluid."
                    ),
                    "template_variant": "xfem_fsi_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "xfem_fsi_3d",
                "description": (
                    "3-D XFEM FSI: structure moving through fixed "
                    "fluid mesh.  Nitsche interface coupling, "
                    "ghost-penalty stabilisation, UMFPACK solvers."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "xfem_fsi_3d") -> str:
        templates = {
            "xfem_fsi_3d": self._template_xfem_fsi_3d,
        }
        if variant == "default":
            variant = "xfem_fsi_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_xfem_fsi_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D XFEM-Based Fluid-Structure Interaction
            #
            # A deformable structure is immersed in a fixed background fluid
            # mesh.  The fluid-structure interface is captured via XFEM
            # enrichment (no ALE mesh motion needed).  Nitsche's method
            # enforces kinematic and traction coupling at the interface.
            #
            # Mesh: requires TWO exodus files:
            #   Fluid mesh: "fluid_bg.e" with
            #     element_block 1 = background fluid (HEX8)
            #     node_set 1 = inlet
            #     node_set 2 = outlet
            #     node_set 3 = walls (no-slip)
            #   Structure mesh: "structure.e" with
            #     element_block 1 = structure (HEX8)
            #     node_set 1 = structure Dirichlet (if any)
            # ---------------------------------------------------------------
            TITLE:
              - "3-D XFEM FSI -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Fluid_Structure_Interaction_XFEM"
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
              TIMESTEP: <structure_timestep>
              NUMSTEP: <structure_num_steps>
              LINEAR_SOLVER: 2
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
              LINEAR_SOLVER: 1
              ITEMAX: <fluid_max_iterations>
            FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES:
              TOL_VEL_RES: <fluid_velocity_residual_tolerance>
              TOL_VEL_INC: <fluid_velocity_increment_tolerance>
              TOL_PRES_RES: <fluid_pressure_residual_tolerance>
              TOL_PRES_INC: <fluid_pressure_increment_tolerance>
            FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION:
              CHARELELENGTH_PC: "root_of_volume"

            # == XFEM settings =================================================
            XFLUID DYNAMIC:
              COUPLING_METHOD: "<coupling_method>"
              VOLUME_GAUSS_POINTS_BY: "<volume_integration_scheme>"
              BOUNDARY_GAUSS_POINTS_BY: "<boundary_integration_scheme>"
              NITSCHE_PENALTY_PARAMETER: <nitsche_penalty_parameter>
              MAXITER_XFEM: <xfem_max_iterations>
            XFLUID DYNAMIC/GHOST PENALTY:
              GHOST_PENALTY_STAB: true
              GHOST_PENALTY_FAC: <ghost_penalty_factor>
              GHOST_PENALTY_TRANSIENT: true
              GHOST_PENALTY_TRANSIENT_FAC: <ghost_penalty_transient_factor>

            # == FSI coupling ==================================================
            FSI DYNAMIC:
              MAXTIME: <end_time>
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              COUPALGO: "<xfem_fsi_coupling_algorithm>"
              RESULTSEVERY: <results_output_interval>

            # == Solvers =======================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "fluid_solver"
            SOLVER 2:
              SOLVER: "UMFPACK"
              NAME: "structure_solver"

            # == Materials =====================================================
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

            # == Boundary Conditions ===========================================

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

            # Structure Dirichlet (optional: constrain motion)
            DESIGN SURF STRUCT DIRICH CONDITIONS:
              - E: <structure_dirichlet_face_id>
                NUMDOF: 3
                ONOFF: [<dof1_fix>, <dof2_fix>, <dof3_fix>]
                VAL: [<dof1_val>, <dof2_val>, <dof3_val>]
                FUNCT: [<dof1_funct>, <dof2_funct>, <dof3_funct>]

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
                      NA: Euler

            STRUCTURE GEOMETRY:
              FILE: "<structure_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  SOLID:
                    HEX8:
                      MAT: 2
                      KINEM: <kinematics>

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

        # Check structural density
        dens = params.get("DENS") or params.get("structure_density")
        if dens is not None:
            try:
                d = float(dens)
                if d <= 0:
                    issues.append(
                        f"Structural DENS must be > 0, got {d}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DENS must be a positive number, got {dens!r}."
                )

        # Check Nitsche penalty
        nitsche = params.get("NITSCHE_PENALTY_PARAMETER")
        if nitsche is not None:
            try:
                n = float(nitsche)
                if n <= 0:
                    issues.append(
                        f"NITSCHE_PENALTY_PARAMETER must be > 0, got {n}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"NITSCHE_PENALTY_PARAMETER must be a positive number, "
                    f"got {nitsche!r}."
                )

        # Warn about ALE sections
        has_ale = params.get("has_ale_dynamic")
        if has_ale:
            issues.append(
                "FSI XFEM does NOT use ALE mesh motion.  Remove the "
                "ALE DYNAMIC section and CLONING MATERIAL MAP."
            )

        # Check fluid NA mode
        fluid_na = params.get("fluid_NA") or params.get("NA")
        if fluid_na is not None:
            if str(fluid_na).upper() != "EULER":
                issues.append(
                    f"Fluid elements MUST use NA: Euler for XFEM FSI "
                    f"(no ALE), got {fluid_na!r}."
                )

        return issues
