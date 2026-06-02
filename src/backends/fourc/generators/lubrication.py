"""Lubrication generator for 4C.

Covers thin film lubrication problems governed by the Reynolds equation.
Solves for pressure distribution in a thin fluid film between two surfaces.
Applications include journal bearings, squeeze films, slider bearings,
elastohydrodynamic lubrication (EHL), and MEMS devices.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class LubricationGenerator(BaseGenerator):
    """Generator for thin film lubrication (Reynolds equation) problems in 4C."""

    module_key = "lubrication"
    display_name = "Lubrication (Reynolds Equation)"
    problem_type = "Lubrication"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The lubrication module solves the Reynolds equation for "
                "thin film flows.  The Reynolds equation is a 2-D PDE "
                "derived from the Navier-Stokes equations under the thin "
                "film approximation (gap height << lateral dimensions).  "
                "It governs the pressure distribution in a lubricant film "
                "between two surfaces.  The PROBLEM TYPE is 'Lubrication'.  "
                "The dynamics section is 'LUBRICATION DYNAMIC'.  Elements "
                "use the LUBRICATION element type (2-D surface elements "
                "in the film plane).  The film height can be prescribed "
                "or coupled to structural deformation for "
                "elastohydrodynamic lubrication (EHL).  Materials use "
                "MAT_lubrication which defines the lubricant viscosity "
                "and density."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "LUBRICATION DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "IO",
                "IO/RUNTIME VTK OUTPUT",
                "RESULT DESCRIPTION",
            ],
            "materials": {
                "MAT_lubrication": {
                    "description": (
                        "Lubricant material for the Reynolds equation.  "
                        "Defines the dynamic viscosity and density of "
                        "the lubricant fluid."
                    ),
                    "parameters": {
                        "DYNVISCOSITY": {
                            "description": (
                                "Dynamic viscosity of the lubricant [Pa s]"
                            ),
                            "range": "> 0",
                        },
                        "DENSITY": {
                            "description": "Lubricant density [kg/m^3]",
                            "range": "> 0",
                        },
                    },
                },
            },
            "solver": {
                "direct": {
                    "type": "UMFPACK",
                    "notes": (
                        "The Reynolds equation leads to a well-conditioned "
                        "system that is efficiently solved by direct "
                        "solvers.  For large 2-D meshes, iterative solvers "
                        "with AMG can be used."
                    ),
                },
            },
            "time_integration": {
                "TIMESTEP": "Time step size for transient lubrication.",
                "NUMSTEP": "Total number of time steps.",
                "MAXTIME": "Maximum simulation time.",
                "SOLVERTYPE": (
                    "'nonlinear' for variable-viscosity or "
                    "cavitation problems; 'linear' for constant-"
                    "viscosity without cavitation."
                ),
            },
            "film_height": {
                "prescribed": (
                    "Film height can be prescribed as a function of "
                    "space and time via HEIGHT_FUNCTION.  Suitable for "
                    "slider bearings with known geometry."
                ),
                "coupled": (
                    "For EHL, the film height comes from structural "
                    "deformation.  This requires coupling with a "
                    "structural field (not covered in stand-alone "
                    "lubrication)."
                ),
            },
            "pitfalls": [
                (
                    "[Numerical] Reynolds equation is valid "
                    "ONLY for thin films (gap << lateral "
                    "dimension). Signal: PROBLEMTYPE: "
                    "Lubrication applied to a gap comparable "
                    "to lateral extent (h/L > 0.1) under-"
                    "predicts pressure by factor ~2-10x vs a "
                    "full FLUID_DYNAMIC Navier-Stokes run — "
                    "the lubrication assumption neglects "
                    "inertia and convective acceleration. "
                    "Switch to PROBLEMTYPE: Fluid + FLUID3 "
                    "for thicker gaps. (Audit 2026-06-02.)"
                ),
                (
                    "[API] Lubrication elements are 2D "
                    "SURFACE elements (QUAD4, TRI3) in the "
                    "film plane — NOT 3D volume elements. "
                    "Signal: writing 3D HEX8 elements for "
                    "lubrication raises 'unsupported element "
                    "type for lubrication' from "
                    "4C_lubrication_factory.cpp; the "
                    "Reynolds discretisation lives on a "
                    "(d-1)-manifold. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] Film height MUST be specified — "
                    "either via prescribed function or "
                    "coupling to structural deformation. "
                    "Zero film height causes SINGULARITY in "
                    "Reynolds equation. Signal: h(x) -> 0 "
                    "gives pressure NaN at the first time "
                    "step (1/h^3 in the viscous term); set "
                    "a minimum cutoff h_min ~ 1e-9 or "
                    "regularise. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Cavitation (sub-ambient "
                    "pressure) requires SPECIAL treatment. "
                    "Signal: without cavitation model, the "
                    "Reynolds solve can return p < p_ambient "
                    "in divergence regions — physically "
                    "impossible (liquid would vaporise). "
                    "Either enable a cavitation algorithm "
                    "(Elrod-Adams) or clamp p >= p_ambient "
                    "in post-processing if pressures stay "
                    "near ambient. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] Reynolds equation BCs are "
                    "pressure DIRICHLET conditions. At "
                    "least one boundary must have a "
                    "prescribed pressure for well-posedness. "
                    "Signal: a pure-Neumann setup makes the "
                    "Reynolds operator singular (pressure "
                    "determined only up to a constant); "
                    "direct LU solver reports 'zero pivot'. "
                    "Pin pressure at one boundary (typically "
                    "outflow). (Audit 2026-06-02.)"
                ),
                (
                    "[Input] Lubrication units must be "
                    "consistent. Reynolds uses viscosity "
                    "[Pa s], film height [m], surface "
                    "velocity [m/s]. Signal: mixing SI "
                    "(Pa s) and CGS (poise) viscosity "
                    "gives pressures wrong by exactly 10x; "
                    "h in mm vs m gives factor 1e9 off — "
                    "the 1/h^3 in the viscous term amplifies "
                    "unit mismatches. Verify by computing "
                    "h^3 * dp/dx vs 6 * mu * U for a "
                    "Couette-flow sanity. (Audit "
                    "2026-06-02.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "slider_bearing_2d",
                    "description": (
                        "A 2-D slider bearing with a linearly varying "
                        "film height.  An analytic solution exists for "
                        "the pressure distribution.  Tests the Reynolds "
                        "equation solver, prescribed film height, and "
                        "pressure boundary conditions."
                    ),
                    "template_variant": "slider_bearing_2d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "slider_bearing_2d",
                "description": (
                    "2-D slider bearing with prescribed linear film "
                    "height.  QUAD4 lubrication elements, constant "
                    "viscosity lubricant, UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "slider_bearing_2d") -> str:
        templates = {
            "slider_bearing_2d": self._template_slider_bearing_2d,
        }
        if variant == "default":
            variant = "slider_bearing_2d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_slider_bearing_2d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 2-D Slider Bearing (Lubrication / Reynolds Equation)
            #
            # A slider bearing with a linearly varying film height.
            # The lower surface moves at constant velocity, the upper
            # surface is stationary.  Pressure builds up due to the
            # converging gap.
            #
            # Mesh: exodus file with:
            #   element_block 1 = lubrication domain (QUAD4)
            #   node_set 1 = inlet boundary (pressure Dirichlet)
            #   node_set 2 = outlet boundary (pressure Dirichlet)
            # ---------------------------------------------------------------
            TITLE:
              - "2-D slider bearing (Reynolds equation) -- generated template"
            PROBLEM SIZE:
              DIM: 2
            PROBLEM TYPE:
              PROBLEMTYPE: "Lubrication"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>

            # == Lubrication dynamics ==========================================
            LUBRICATION DYNAMIC:
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              SOLVERTYPE: "<solver_type>"
              LINEAR_SOLVER: 1
              RESULTSEVERY: <results_output_interval>
              RESTARTEVERY: <restart_interval>
              SURFACE_VELOCITY: <surface_velocity>
              HEIGHT_FUNCTION: <height_function_id>

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "lubrication_solver"

            # == Materials =====================================================
            MATERIALS:
              - MAT: 1
                MAT_lubrication:
                  DYNVISCOSITY: <lubricant_dynamic_viscosity>
                  DENSITY: <lubricant_density>

            # Film height function (linear: h = h_in - (h_in - h_out) * x / L)
            FUNCT<height_function_id>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<film_height_expression>"

            # == Boundary Conditions ===========================================

            # Pressure Dirichlet at inlet and outlet
            DESIGN LINE LUBRICATION DIRICH CONDITIONS:
              - E: <inlet_boundary_id>
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<inlet_pressure>]
                FUNCT: [0]
              - E: <outlet_boundary_id>
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<outlet_pressure>]
                FUNCT: [0]

            # == Geometry ======================================================
            LUBRICATION GEOMETRY:
              FILE: "<mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  LUBRICATION:
                    QUAD4:
                      MAT: 1

            RESULT DESCRIPTION:
              - LUBRICATION:
                  DIS: "lubrication"
                  NODE: <result_node_id>
                  QUANTITY: "pre"
                  VALUE: <expected_pressure>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check viscosity
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

        # Check density
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

        # Check surface velocity
        velocity = params.get("SURFACE_VELOCITY")
        if velocity is not None:
            try:
                float(velocity)
            except (TypeError, ValueError):
                issues.append(
                    f"SURFACE_VELOCITY must be a number, "
                    f"got {velocity!r}."
                )

        # Check timestep
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
                    f"TIMESTEP must be a positive number, "
                    f"got {timestep!r}."
                )

        return issues
