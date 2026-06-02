"""Level-set interface tracking generator for 4C.

Covers level-set methods for tracking interfaces in multi-phase or
free-surface problems.  The level-set field phi is advected by a prescribed
or computed velocity field.  Reinitialization (signed distance function)
maintains phi as a distance function for accurate interface location.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class LevelSetGenerator(BaseGenerator):
    """Generator for level-set interface tracking problems in 4C."""

    module_key = "level_set"
    display_name = "Level-Set Interface Tracking"
    problem_type = "Level_Set"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The level-set module tracks an interface as the zero "
                "contour of a scalar field phi.  phi > 0 represents one "
                "phase, phi < 0 the other.  The level-set field is "
                "advected by the advection equation "
                "d(phi)/dt + u . grad(phi) = 0 using the scalar transport "
                "framework.  Reinitialization periodically restores phi to "
                "a signed distance function to prevent numerical "
                "degradation.  The PROBLEM TYPE is 'Level_Set'.  The module "
                "uses SCALAR TRANSPORT DYNAMIC for the advection equation "
                "and LEVEL-SET CONTROL for reinitialization and other "
                "level-set-specific settings."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "LEVEL-SET CONTROL",
                "LEVEL-SET CONTROL/REINITIALIZATION",
                "SCALAR TRANSPORT DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "SCALAR TRANSPORT DYNAMIC/STABILIZATION",
                "SCALAR TRANSPORT DYNAMIC/NONLINEAR",
                "IO/RUNTIME VTK OUTPUT",
            ],
            "materials": {
                "MAT_scatra": {
                    "description": (
                        "Scalar transport material for the level-set field.  "
                        "DIFFUSIVITY is typically set to 0 (pure advection) "
                        "or a very small value for numerical regularisation."
                    ),
                    "parameters": {
                        "DIFFUSIVITY": {
                            "description": (
                                "Diffusion coefficient for the level-set "
                                "field.  0 for pure advection (standard); "
                                "small positive value for regularisation."
                            ),
                            "range": ">= 0 (typically 0)",
                        },
                    },
                },
            },
            "solver": {
                "direct": {
                    "type": "UMFPACK",
                    "notes": (
                        "Level-set problems are typically moderate in size.  "
                        "Direct solvers work well."
                    ),
                },
            },
            "time_integration": {
                "TIMESTEP": (
                    "Time step size.  Must satisfy CFL condition: "
                    "dt <= h / |u|_max where h is the mesh size."
                ),
                "NUMSTEP": "Total number of time steps.",
                "MAXTIME": "Maximum simulation time.",
            },
            "level_set_control": {
                "REINITIALIZATION": (
                    "Reinitialization method in LEVEL-SET CONTROL/"
                    "REINITIALIZATION.  Options: "
                    "'Signed_Distance_Function' (geometric, exact), "
                    "'Elliptic' (PDE-based, smoother), "
                    "'none' (no reinitialization)."
                ),
                "REINIT_INITIAL": (
                    "Set to true to reinitialise the level-set field at "
                    "the start of the simulation.  Useful when the initial "
                    "condition is not a signed distance function."
                ),
            },
            "velocity_field": {
                "VELOCITYFIELD": (
                    "Velocity field for advection.  Options: "
                    "'function' -- prescribed via VELFUNCNO referencing a "
                    "FUNCT section.  'Navier_Stokes' -- coupled with a "
                    "fluid solve."
                ),
                "VELFUNCNO": (
                    "Function number for the prescribed velocity field "
                    "(used when VELOCITYFIELD: 'function')."
                ),
            },
            "pitfalls": [
                (
                    "[Input] DIFFUSIVITY should be 0 for a "
                    "STANDARD level-set advection. Signal: "
                    "adding any diffusion (DIFFUSIVITY > 0) "
                    "smears the interface — visible "
                    "thickening of the |phi| = 0 isoline "
                    "over time, with smearing width "
                    "proportional to sqrt(D*t). The level-"
                    "set physics is pure advection, so "
                    "diffusion changes the physics (not "
                    "just numerics). (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Reinitialization is "
                    "ESSENTIAL for long-time simulations — "
                    "without it the level-set degrades and "
                    "loses its signed-distance property. "
                    "Signal: |grad(phi)| at the interface "
                    "drifts away from 1 (>2 or <0.5) "
                    "within ~50 time steps; interface "
                    "location shows visible deformation "
                    "errors > 5%. Apply Hamilton-Jacobi "
                    "reinitialisation every ~10 steps to "
                    "restore the SDF. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] VELOCITYFIELD MUST be "
                    "specified. For prescribed velocity use "
                    "'function' with VELFUNCNO pointing to "
                    "the velocity FUNCT. Signal: omitting "
                    "VELOCITYFIELD defaults to ZERO — no "
                    "advection, the interface STAYS PUT "
                    "even with non-zero time stepping; "
                    "level-set field appears time-"
                    "independent. Set VELOCITYFIELD: "
                    "function + VELFUNCNO: N. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Numerical] Stabilisation (SUPG) is "
                    "important for advection-dominated "
                    "level-set transport. Use DEFINITION_TAU: "
                    "'Taylor_Hughes_Zarins' for robust "
                    "stabilisation. Signal: STABTYPE: "
                    "no_stabilization on a high-Pe level-"
                    "set transport produces ringing across "
                    "element edges (5-10% amplitude that "
                    "does not damp). Taylor-Hughes-Zarins "
                    "is the de-facto standard for "
                    "advection-dominated scatra. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] Initial level-set field should "
                    "ideally be a SIGNED DISTANCE FUNCTION. "
                    "If not, set REINIT_INITIAL: true to fix "
                    "it before the first time step. Signal: "
                    "an initial phi = (x - 0.5)^2 (not a "
                    "SDF) gives wrong |grad(phi)| at the "
                    "interface from t=0; interface velocity "
                    "is wrong by factor of |grad(phi)|. "
                    "Either initialise as a true SDF or "
                    "enable REINIT_INITIAL. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Numerical] CFL condition: time step "
                    "must be small relative to mesh size "
                    "and velocity magnitude to prevent "
                    "oscillations. Signal: dt > h / "
                    "max(|u|) gives O(1) overshoots at "
                    "the interface even with SUPG — the "
                    "interface velocity exceeds one "
                    "element per step, breaking upwind "
                    "stability. Safety factor 0.5 * h / "
                    "max(|u|) is conservative; CFL=1 is "
                    "the strict bound. (Audit 2026-06-02.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "zalesak_disc",
                    "description": (
                        "Zalesak's slotted disc rotation benchmark.  A "
                        "notched disc is advected by a solid-body rotation "
                        "velocity field.  After one full revolution the disc "
                        "should return to its original position.  Tests "
                        "advection accuracy and reinitialization quality."
                    ),
                    "template_variant": "advection_2d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "advection_2d",
                "description": (
                    "2-D level-set advection with prescribed velocity field "
                    "and geometric reinitialization.  Uses MAT_scatra with "
                    "DIFFUSIVITY: 0, SUPG stabilisation, UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "advection_2d") -> str:
        templates = {
            "advection_2d": self._template_advection_2d,
        }
        if variant == "default":
            variant = "advection_2d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_advection_2d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 2-D Level-Set Advection with Reinitialization
            #
            # A level-set field is advected by a prescribed velocity field.
            # Periodic reinitialization restores the signed distance property.
            #
            # Mesh: exodus file with:
            #   element_block 1 = computational domain (QUAD4)
            # ---------------------------------------------------------------
            TITLE:
              - "2-D level-set advection -- generated template"
            PROBLEM TYPE:
              PROBLEMTYPE: "Level_Set"

            # == Level-Set Control =============================================
            LEVEL-SET CONTROL:
              NUMSTEP: <number_of_steps>
              TIMESTEP: <timestep>
              MAXTIME: <end_time>
              RESTARTEVERY: <restart_interval>
            LEVEL-SET CONTROL/REINITIALIZATION:
              REINITIALIZATION: "<reinitialization_method>"
              REINIT_INITIAL: <reinitialize_at_start>

            # == Scalar Transport (level-set advection) ========================
            SCALAR TRANSPORT DYNAMIC:
              SOLVERTYPE: "nonlinear"
              MAXTIME: <end_time>
              NUMSTEP: <number_of_steps>
              TIMESTEP: <timestep>
              RESTARTEVERY: <restart_interval>
              MATID: 1
              VELOCITYFIELD: "function"
              VELFUNCNO: <velocity_function_id>
              INITIALFIELD: "field_by_function"
              INITFUNCNO: <initial_levelset_function_id>
              LINEAR_SOLVER: 1
            SCALAR TRANSPORT DYNAMIC/NONLINEAR:
              ITEMAX: <max_nonlinear_iterations>
              CONVTOL: <nonlinear_convergence_tolerance>
            SCALAR TRANSPORT DYNAMIC/STABILIZATION:
              DEFINITION_TAU: "<stabilization_tau_definition>"

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "levelset_solver"

            # == Materials =====================================================
            MATERIALS:
              - MAT: 1
                MAT_scatra:
                  DIFFUSIVITY: <levelset_diffusivity>

            # == Velocity field function =======================================
            FUNCT<velocity_function_id>:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<velocity_x_expression>"
              - COMPONENT: 1
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<velocity_y_expression>"

            # == Initial level-set field =======================================
            FUNCT<initial_levelset_function_id>:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_levelset_expression>"

            # == Geometry ======================================================
            TRANSPORT GEOMETRY:
              FILE: "<mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  TRANSP:
                    QUAD4:
                      MAT: 1
                      TYPE: Std

            RESULT DESCRIPTION:
              - SCATRA:
                  DIS: "scatra"
                  NODE: <result_node_id>
                  QUANTITY: "phi"
                  VALUE: <expected_levelset_value>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check diffusivity
        diffusivity = params.get("DIFFUSIVITY")
        if diffusivity is not None:
            try:
                d = float(diffusivity)
                if d < 0:
                    issues.append(
                        f"DIFFUSIVITY must be >= 0, got {d}."
                    )
                if d > 0:
                    issues.append(
                        f"DIFFUSIVITY = {d} > 0.  For standard level-set "
                        f"advection, DIFFUSIVITY should be 0.  Non-zero "
                        f"diffusion smears the interface."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DIFFUSIVITY must be a non-negative number, "
                    f"got {diffusivity!r}."
                )

        # Check reinitialization method
        reinit = params.get("REINITIALIZATION")
        if reinit is not None and reinit not in (
            "Signed_Distance_Function", "Elliptic", "none"
        ):
            issues.append(
                f"REINITIALIZATION must be 'Signed_Distance_Function', "
                f"'Elliptic', or 'none', got {reinit!r}."
            )

        # Check velocity field
        velfield = params.get("VELOCITYFIELD")
        if velfield is not None and velfield not in (
            "function", "Navier_Stokes", "zero"
        ):
            issues.append(
                f"VELOCITYFIELD must be 'function', 'Navier_Stokes', or "
                f"'zero', got {velfield!r}."
            )

        # Check TIMESTEP
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
                    f"TIMESTEP must be a positive number, got {timestep!r}."
                )

        # Check CONVTOL
        convtol = params.get("CONVTOL")
        if convtol is not None:
            try:
                ct = float(convtol)
                if ct <= 0:
                    issues.append(
                        f"CONVTOL must be > 0, got {ct}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"CONVTOL must be a positive number, got {convtol!r}."
                )

        return issues
