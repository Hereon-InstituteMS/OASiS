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
                    '[Input] XFEM fluid does NOT use ALE mesh motion. The mesh is '
                    'fixed (Eulerian) and the interface cuts through elements. An '
                    'ALE DYNAMIC section is NOT rejected, though: 4C reads it, '
                    'ignores it, and says nothing, so the deck runs and reproduces '
                    'the Eulerian answer. What does break is asking the fluid '
                    'ELEMENT block for ALE kinematics. Signal: with NA: ALE the run '
                    "parses and then aborts at the first assembly with 'Cannot find "
                    "state dispnp in discretization fluid' from "
                    '4C_fem_discretization.hpp -- a missing-state message that '
                    'names neither XFEM nor ALE. Keep NA: Euler; do not expect a '
                    'warning about a leftover ALE DYNAMIC block. (Audit 2026-06-02; '
                    'corrected by execution 2026-08-06.)'
                ),
                (
                    '[Numerical] Ghost-penalty stabilisation matters for cut '
                    'elements with very small volume fractions, but its absence is '
                    'an ACCURACY failure, not a solver failure. Signal: with '
                    'GHOST_PENALTY_STAB: false (or GHOST_PENALTY_FAC: 0.0) the run '
                    "completes its Newton loop, reaches 'Checking results of N "
                    "tests' and reports different values; no condition number is "
                    'printed, no factorisation fails, and a direct solver such as '
                    "UMFPACK is untroubled. There is no 'Belos: condition number' "
                    "or 'solver diverged after 0 iterations' message in 4C. Detect "
                    'this by comparing against a reference, not by watching the '
                    'solver. (Audit 2026-06-02; corrected by execution 2026-08-06.)'
                ),
                (
                    '[Input] The interface must be described by a level-set field '
                    'or a cutter boundary mesh. If no XFEM coupling condition is '
                    'present, 4C does NOT fall back to standard FEM. Signal: it '
                    "aborts in Cut::CutWizard::safety_checks with 'You have to call "
                    "PrepareCut() before you can call the Cut-routine' from "
                    '4C_cut_cutwizard.cpp, before any time step completes, so no '
                    'result test is reached and there is no non-enriched answer to '
                    'compare with. 4C prints no count of enriched elements at any '
                    'point. (Audit 2026-06-02; corrected by execution 2026-08-06.)'
                ),
                (
                    '[Input] Two-phase level-set XFEM is not configured through a '
                    'material map. DESIGN XFEM LEVELSET TWOPHASE VOL CONDITIONS '
                    'takes only E, COUPLINGID, LEVELSETFIELDNO, BOOLEANTYPE and '
                    'COMPLEMENTARY -- there are no MAT_NEGATIVE or MAT_POSITIVE '
                    'keys anywhere in 4C, and adding them is rejected as unmatched '
                    'condition input. Signal: a correctly spelled two-phase '
                    'condition parses and then dies in Element::location_vector '
                    "with 'wrong number of nodes' from 4C_fem_general_element.cpp; "
                    'the only upstream decks that mention the condition leave it as '
                    'an empty list. Treat two-phase XFEM as unavailable rather than '
                    'mis-specified. (Audit 2026-06-02; corrected by execution '
                    '2026-08-06.)'
                ),
                (
                    '[Numerical] Cut elements need a special integration rule, '
                    'chosen with VOLUME_GAUSS_POINTS_BY. Tessellation and '
                    'DirectDivergence are the usable values. Signal: MomentFitting '
                    'does not warn and does not fall back -- it terminates the '
                    'process with SIGSEGV inside '
                    'Core::FE::GaussPointsComposite::num_points, so the log ends '
                    "with 'Signal: Segmentation fault (11)' and no 4C-level "
                    'diagnostic at all. If a cut run dies without a PROC 0 ERROR '
                    'block, check this key first. (Audit 2026-06-02; corrected by '
                    'execution 2026-08-06.)'
                ),
                (
                    '[Numerical] The Nitsche penalty knob is NIT_STAB_FAC in XFLUID '
                    'DYNAMIC/STABILIZATION (default 35), with NIT_STAB_FAC_TANG for '
                    'the tangential term; the viscous scaling by element size is '
                    'applied internally through VISC_STAB_TRACE_ESTIMATE and '
                    'VISC_STAB_HK. There is no NITSCHE_PENALTY_PARAMETER key. '
                    "Signal: the wrong name fails with 'Could not match this input' "
                    "and 4C lists the section's real keys, including NIT_STAB_FAC. "
                    'Mis-setting the real key gives wrong results rather than a '
                    'solver stall, and on problems the space reproduces exactly it '
                    "changes nothing, because Nitsche's method is consistent. "
                    '(Audit 2026-06-02; corrected by execution 2026-08-06.)'
                ),
                (
                    '[Output] An XFEM fluid writes NO VTU files, per sub-domain or '
                    'otherwise. Signal: with IO/RUNTIME VTK OUTPUT and IO/RUNTIME '
                    'VTK OUTPUT/FLUID enabled, a plain Fluid problem produces '
                    'fluid-*.vtu plus a .pvd, while the same configuration on a '
                    'Fluid_XFEM problem produces neither, exits 0, and never '
                    'mentions that the requested output was skipped. Read XFEM '
                    'results from the legacy Ensight .result file, or set '
                    'OUTPUT_GMSH with GMSH_SOL_OUT and read the Gmsh .pos files. '
                    '(Audit 2026-06-02; corrected by execution 2026-08-06.)'
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
