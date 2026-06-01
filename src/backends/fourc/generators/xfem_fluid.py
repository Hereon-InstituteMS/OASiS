"""XFEM Fluid generator for 4C.

Covers fluid problems with XFEM (eXtended Finite Element Method) interfaces.
The XFEM approach enriches the fluid approximation space to capture
discontinuities (e.g. two-phase interfaces, embedded boundaries, or void
regions) without requiring the mesh to conform to the interface.  The
interface geometry is typically described by a level-set field or a boundary
mesh.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class XFEMFluidGenerator(BaseGenerator):
    """Generator for XFEM fluid problems in 4C."""

    module_key = "xfem_fluid"
    display_name = "XFEM Fluid (Fluid with XFEM Interfaces)"
    problem_type = "Fluid_XFEM"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The XFEM Fluid module solves incompressible Navier-Stokes "
                "problems with discontinuities captured via the eXtended "
                "Finite Element Method.  The fluid mesh does NOT need to "
                "conform to embedded interfaces, void boundaries, or "
                "two-phase fronts.  Instead, the approximation space is "
                "enriched with discontinuous shape functions along the "
                "interface.  The interface geometry is defined either by a "
                "level-set function (XFLUID DYNAMIC/LEVEL SET) or by an "
                "embedded boundary mesh (cutter mesh).  The PROBLEM TYPE is "
                "'Fluid_XFEM'.  The main dynamics section is 'XFLUID DYNAMIC' "
                "which contains XFEM-specific parameters such as the "
                "integration scheme for cut elements, ghost-penalty "
                "stabilisation, and the interface coupling method "
                "(Nitsche, penalty).  Standard FLUID DYNAMIC settings "
                "(time integration, stabilisation) are also required.  "
                "Elements use FLUID HEX8 or FLUID TET4 with NA: Euler."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "FLUID DYNAMIC",
                "XFLUID DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "XFLUID DYNAMIC/STABILIZATION",
                "XFLUID DYNAMIC/GHOST PENALTY",
                "FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION",
                "FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES",
                "IO",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/FLUID",
                "LEVEL SET GEOMETRY",
            ],
            "materials": {
                "MAT_fluid": {
                    "description": (
                        "Newtonian fluid material for the background "
                        "fluid domain."
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
                "MAT_fluid (second phase)": {
                    "description": (
                        "Second fluid material for two-phase XFEM "
                        "problems.  Used on the other side of the "
                        "interface."
                    ),
                    "parameters": {
                        "DYNVISCOSITY": {
                            "description": "Dynamic viscosity of second phase [Pa s]",
                            "range": "> 0",
                        },
                        "DENSITY": {
                            "description": "Density of second phase [kg/m^3]",
                            "range": "> 0",
                        },
                    },
                },
            },
            "solver": {
                "fluid_solver": {
                    "type": "UMFPACK or Belos",
                    "notes": (
                        "Direct solver (UMFPACK) is robust for XFEM since "
                        "the enriched system may have variable size.  "
                        "For large problems use Belos with ILU "
                        "preconditioner."
                    ),
                },
            },
            "xfem_parameters": {
                "COUPLING_METHOD": (
                    "Interface coupling: 'Nitsche' (weakly enforced, "
                    "recommended) or 'penalty'.  Nitsche preserves "
                    "consistency; penalty is simpler but less accurate."
                ),
                "GHOST_PENALTY": (
                    "Ghost-penalty stabilisation for small cut elements.  "
                    "Controls ill-conditioning caused by elements with "
                    "very small cut volumes.  GHOST_PENALTY_FAC sets the "
                    "scaling."
                ),
                "VOLUME_GAUSS_POINTS_BY": (
                    "Integration scheme for cut elements: "
                    "'Tessellation' (subdivide into sub-cells) or "
                    "'MomentFitting' (moment-fitting quadrature)."
                ),
            },
            "pitfalls": [
                (
                    "[Input] XFEM fluid does NOT use ALE mesh motion. "
                    " The mesh is fixed (Eulerian) and the interface "
                    "cuts through elements.  Do not include ALE "
                    "DYNAMIC. Signal: parser warns `ALE DYNAMIC "
                    "ignored under XFEM` or runtime "
                    "`incompatible ALE+XFEM combination`; if ALE is "
                    "active, the velocity field includes a "
                    "non-physical mesh-displacement contribution. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Ghost-penalty stabilisation is "
                    "critical for cut elements with very small volume "
                    "fractions.  Without it, the system matrix "
                    "becomes severely ill-conditioned or singular. "
                    "Signal: linear solver fails on cut-cell time "
                    "steps with `Belos: condition number > 1e16` or "
                    "`solver diverged after 0 iterations`; the cut "
                    "fraction at the failing step is < 1e-3. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] The interface must be described either "
                    "by a level-set field or a cutter boundary mesh. "
                    " If neither is provided, no XFEM enrichment is "
                    "applied and the solution reverts to standard "
                    "FEM. Signal: result matches a standard "
                    "(non-enriched) reference exactly, with no jump "
                    "in pressure/velocity at the expected interface "
                    "location; XFEM diagnostic prints `0 enriched "
                    "elements`. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] For two-phase problems, two fluid "
                    "materials must be defined -- one for each side "
                    "of the interface.  The XFEM framework selects "
                    "the correct material based on the sign of the "
                    "level-set function. Signal: 4C aborts with "
                    "`XFEM: material map needs MAT_NEGATIVE and "
                    "MAT_POSITIVE` or the two-phase problem behaves "
                    "as single-phase (density/viscosity uniform "
                    "across the interface). (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Cut elements require special "
                    "integration rules.  The VOLUME_GAUSS_POINTS_BY "
                    "parameter controls this.  'Tessellation' is "
                    "more robust; 'MomentFitting' is more efficient "
                    "but may fail for complex cut geometries. "
                    "Signal: MomentFitting failures show runtime "
                    "warning `MomentFitting did not converge for "
                    "element X — falling back to Tessellation` or "
                    "NaN residual on a cut step; switching to "
                    "Tessellation removes the warning. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Numerical] The Nitsche penalty parameter "
                    "(NITSCHE_PENALTY_PARAMETER) must be large "
                    "enough to enforce the interface condition but "
                    "not so large that it causes ill-conditioning. "
                    " Typical values are O(10)--O(100). Signal: too "
                    "low -> interface jump condition violation > 5% "
                    "(visible discontinuity offset in post-"
                    "processing); too high -> linear-solver "
                    "condition number > 1e14, Newton stalls. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Output] Output for XFEM problems may produce "
                    "multiple VTU files (one per sub-domain).  Use "
                    "appropriate post-processing to visualise the "
                    "enriched fields. Signal: visualize('list') "
                    "shows fluid_subdomain_0-*.vtu, "
                    "fluid_subdomain_1-*.vtu rather than a single "
                    "fluid-*.vtu; loading just one of them shows "
                    "only half the domain. (Audit 2026-06-02.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "embedded_cylinder_3d",
                    "description": (
                        "Flow past a cylinder embedded in a background "
                        "fluid mesh via XFEM.  The cylinder surface is "
                        "described by a level-set or boundary mesh.  "
                        "Tests Nitsche coupling, ghost penalty, and "
                        "enriched integration."
                    ),
                    "template_variant": "xfem_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "xfem_3d",
                "description": (
                    "3-D XFEM fluid: flow with an embedded interface "
                    "(void or two-phase).  FLUID HEX8 elements, "
                    "Nitsche coupling, ghost-penalty stabilisation, "
                    "UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "xfem_3d") -> str:
        templates = {
            "xfem_3d": self._template_xfem_3d,
        }
        if variant == "default":
            variant = "xfem_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_xfem_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D XFEM Fluid — Flow with Embedded Interface
            #
            # A fluid domain with an embedded boundary (void, obstacle, or
            # two-phase interface) captured via XFEM.  The background mesh
            # does not conform to the interface.  Nitsche coupling enforces
            # the interface conditions weakly.
            #
            # Mesh: requires exodus file with:
            #   element_block 1 = background fluid (HEX8)
            #   node_set 1 = inlet
            #   node_set 2 = outlet
            #   node_set 3 = walls (no-slip)
            #
            # Cutter/level-set: a separate boundary mesh or level-set
            #   function defining the interface geometry.
            # ---------------------------------------------------------------
            TITLE:
              - "3-D XFEM fluid -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Fluid_XFEM"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/FLUID:
              OUTPUT_FLUID: true
              VELOCITY: true
              PRESSURE: true

            # == Fluid dynamics =================================================
            FLUID DYNAMIC:
              TIMEINTEGR: "Np_Gen_Alpha"
              TIMESTEP: <fluid_timestep>
              NUMSTEP: <fluid_num_steps>
              MAXTIME: <fluid_max_time>
              LINEAR_SOLVER: 1
              ITEMAX: <fluid_max_iterations>
            FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES:
              TOL_VEL_RES: <fluid_velocity_residual_tolerance>
              TOL_VEL_INC: <fluid_velocity_increment_tolerance>
              TOL_PRES_RES: <fluid_pressure_residual_tolerance>
              TOL_PRES_INC: <fluid_pressure_increment_tolerance>
            FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION:
              CHARELELENGTH_PC: "root_of_volume"

            # == XFEM-specific settings =========================================
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

            # == Solver =========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "xfem_fluid_solver"

            # == Materials ======================================================
            MATERIALS:
              # Background fluid material
              - MAT: 1
                MAT_fluid:
                  DYNVISCOSITY: <fluid_dynamic_viscosity>
                  DENSITY: <fluid_density>

            # == Boundary Conditions ============================================

            # Fluid: inlet velocity
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

            # Inlet ramp function
            FUNCT<inlet_ramp_function>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<inlet_ramp_expression>"

            # == Geometry =======================================================
            FLUID GEOMETRY:
              FILE: "<fluid_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  FLUID:
                    HEX8:
                      MAT: 1
                      NA: Euler

            # Cutter boundary mesh (defines the embedded interface)
            XFEM BOUNDARY GEOMETRY:
              FILE: "<cutter_mesh_file>"

            RESULT DESCRIPTION:
              - FLUID:
                  DIS: "fluid"
                  NODE: <result_node_id>
                  QUANTITY: "velx"
                  VALUE: <expected_velocity>
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

        # Check Nitsche penalty parameter
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

        # Check ghost penalty factor
        gpf = params.get("GHOST_PENALTY_FAC")
        if gpf is not None:
            try:
                g = float(gpf)
                if g <= 0:
                    issues.append(
                        f"GHOST_PENALTY_FAC must be > 0, got {g}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"GHOST_PENALTY_FAC must be a positive number, "
                    f"got {gpf!r}."
                )

        # Check coupling method
        coupling = params.get("COUPLING_METHOD")
        if coupling is not None and coupling not in (
            "Nitsche", "penalty",
        ):
            issues.append(
                f"COUPLING_METHOD should be 'Nitsche' or 'penalty', "
                f"got {coupling!r}."
            )

        return issues
