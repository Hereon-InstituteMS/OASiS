"""Reduced-Dimensional Airways generator for 4C.

Covers 1-D reduced-dimensional modeling of the pulmonary airway tree.
The airways are represented as a branching network of 1-D elements
where each element models a compliant tube segment with airflow governed
by pressure-flow relationships derived from the Navier-Stokes equations
under the long-wavelength approximation.  Applications include
respiratory mechanics, ventilator design, and patient-specific lung
modeling.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class ReducedAirwaysGenerator(BaseGenerator):
    """Generator for reduced-dimensional airways problems in 4C."""

    module_key = "reduced_airways"
    display_name = "Reduced-Dimensional Airways (Lung Airway Tree)"
    problem_type = "ReducedDimensionalAirWays"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The reduced-dimensional airways module models the "
                "pulmonary airway tree as a branching 1-D network.  "
                "Each airway segment is represented by a compliant tube "
                "element with pressure-flow governing equations derived "
                "from the incompressible Navier-Stokes equations under "
                "the long-wavelength (Womersley) approximation.  The "
                "network branches according to the anatomical airway "
                "tree (from trachea down to terminal bronchioles).  "
                "Boundary conditions include flow or pressure at the "
                "tracheal inlet and tissue compliance/resistance at "
                "the terminal ends (acini).  The PROBLEM TYPE is "
                "'ReducedDimensionalAirWays'.  The dynamics section is "
                "'REDUCED DIMENSIONAL AIRWAYS DYNAMIC'.  Elements use "
                "the REDAIRWAY element type (1-D line elements).  "
                "Materials use MAT_redairway_material which defines "
                "airway wall compliance, fluid viscosity, and geometric "
                "parameters (length, radius)."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "REDUCED DIMENSIONAL AIRWAYS DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "IO",
                "RESULT DESCRIPTION",
                "REDUCED DIMENSIONAL AIRWAYS DYNAMIC/ACINUS",
                "REDUCED DIMENSIONAL AIRWAYS DYNAMIC/TISSUE",
            ],
            "materials": {
                "MAT_redairway_material": {
                    "description": (
                        "Reduced-dimensional airway material.  Defines "
                        "the properties of a single airway segment: "
                        "viscosity, wall compliance, and generation-"
                        "dependent parameters."
                    ),
                    "parameters": {
                        "VISCOSITY": {
                            "description": (
                                "Dynamic viscosity of air [Pa s] "
                                "(~1.8e-5 for air at 37C)"
                            ),
                            "range": "> 0",
                        },
                        "WALL_COMPLIANCE": {
                            "description": (
                                "Airway wall compliance factor.  "
                                "Controls how much the airway radius "
                                "changes with transmural pressure."
                            ),
                            "range": ">= 0 (0 for rigid airways)",
                        },
                        "EXPONENT": {
                            "description": (
                                "Exponent in the pressure-area "
                                "relationship for compliant walls."
                            ),
                            "range": "> 0",
                        },
                    },
                },
                "MAT_redairway_acinus_material": {
                    "description": (
                        "Acinar (terminal) material.  Represents the "
                        "compliant alveolar compartments at the end of "
                        "the airway tree.  Defines tissue compliance and "
                        "volume parameters."
                    ),
                    "parameters": {
                        "COMPLIANCE": {
                            "description": (
                                "Alveolar tissue compliance [m^3/Pa]"
                            ),
                            "range": "> 0",
                        },
                        "VOLUME_RELAXED": {
                            "description": (
                                "Relaxed (functional residual capacity) "
                                "volume of the acinus [m^3]"
                            ),
                            "range": "> 0",
                        },
                    },
                },
            },
            "solver": {
                "airways_solver": {
                    "type": "UMFPACK",
                    "notes": (
                        "The reduced airways system is a small 1-D "
                        "network and is efficiently solved with a "
                        "direct solver."
                    ),
                },
            },
            "time_integration": {
                "scheme": (
                    "Implicit time integration for the 1-D network.  "
                    "Time step should resolve the breathing cycle "
                    "(typical period 3--5 seconds) with sufficient "
                    "resolution (dt ~ 0.001--0.01 s)."
                ),
                "TIMESTEP": "Time step for the airway network solver.",
                "NUMSTEP": "Total number of time steps.",
                "MAXTIME": "Maximum simulation time.",
            },
            "pitfalls": [
                (
                    "[Input] The airway tree network must be "
                    "topologically consistent: each parent branch "
                    "bifurcates into exactly two daughter branches. "
                    " Dangling nodes or loops are not supported. "
                    "Signal: 4C aborts during topology check with "
                    "`node X has degree 1 (dangling)` or "
                    "`cycle detected in airway tree`. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] Boundary conditions at the trachea "
                    "define the driving force: typically a pressure "
                    "or flow waveform representing breathing effort. "
                    " Without it, no flow occurs. Signal: simulation "
                    "completes but the flow / volume result fields "
                    "are uniformly zero; OR runtime warning "
                    "`no DESIGN POINT 1D DBC defined at trachea "
                    "node`. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Terminal acinar elements must be "
                    "specified with appropriate compliance.  Missing "
                    "acinar conditions give open-ended branches with "
                    "unphysical behaviour. Signal: terminal-branch "
                    "pressure equals the upstream pressure at all "
                    "times (no impedance), and flow into the "
                    "terminal segment grows unbounded; tidal volume "
                    "diverges with cycle count. (Audit 2026-06-02.)"
                ),
                (
                    "Airway wall compliance strongly affects pressure "
                    "wave propagation.  Rigid airways (compliance = "
                    "0) give a simpler problem but miss physiological "
                    "airway collapse and reopening."
                ),
                (
                    "The reduced model assumes long wavelength "
                    "(Womersley number restrictions).  For high-"
                    "frequency ventilation, a full 3-D CFD approach "
                    "may be needed instead."
                ),
                (
                    "Material properties vary by airway generation.  "
                    "Proximal airways (trachea, bronchi) have stiffer "
                    "walls and larger radii than distal airways "
                    "(bronchioles).  Use generation-specific "
                    "materials."
                ),
                (
                    "[Input] The element type is REDAIRWAY, not "
                    "standard beam or fluid elements.  Using "
                    "incorrect element types will cause failure. "
                    "Signal: input parser abort `unknown element "
                    "type for ELEMENT_BLOCK X` or runtime "
                    "`elementType BEAM3R has no RedAirway "
                    "implementation`. (Audit 2026-06-02.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "single_breath_cycle",
                    "description": (
                        "A single inspiration-expiration cycle through "
                        "a multi-generation airway tree (5-10 "
                        "generations).  Tracheal pressure drives "
                        "breathing, acinar compliance stores lung "
                        "volume.  Tests network topology, acinar BCs, "
                        "and time-dependent flow distribution."
                    ),
                    "template_variant": "airways_1d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "airways_1d",
                "description": (
                    "1-D reduced-dimensional airway tree: branching "
                    "network with REDAIRWAY line elements, acinar "
                    "terminals, pressure-driven breathing, UMFPACK "
                    "solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "airways_1d") -> str:
        templates = {
            "airways_1d": self._template_airways_1d,
        }
        if variant == "default":
            variant = "airways_1d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_airways_1d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 1-D Reduced-Dimensional Airway Tree
            #
            # A branching network of compliant airway segments modelling
            # the pulmonary airways from trachea to terminal bronchioles.
            # Each segment is a 1-D REDAIRWAY line element.  Terminal
            # branches connect to acinar compartments (compliant volumes).
            #
            # Mesh: requires an airway network mesh with:
            #   element_block 1 = airway segments (LINE2)
            #   node_set 1 = tracheal inlet node
            #   node_set 2 = acinar terminal nodes
            # ---------------------------------------------------------------
            TITLE:
              - "1-D reduced-dimensional airway tree -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "ReducedDimensionalAirWays"
            IO:
              STDOUTEVERY: <stdout_interval>

            # == Reduced airways dynamics ======================================
            REDUCED DIMENSIONAL AIRWAYS DYNAMIC:
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              LINEAR_SOLVER: 1
              RESULTSEVERY: <results_output_interval>
              RESTARTEVERY: <restart_interval>
              MAXITER: <nonlinear_max_iterations>
              TOLERANCE: <nonlinear_tolerance>

            # Acinar properties
            REDUCED DIMENSIONAL AIRWAYS DYNAMIC/ACINUS:
              ACINAR_COMPLIANCE: <acinar_compliance>
              ACINAR_VOLUME_RELAXED: <acinar_relaxed_volume>

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "airways_solver"

            # == Materials =====================================================
            MATERIALS:
              # Airway wall material (proximal generations)
              - MAT: 1
                MAT_redairway_material:
                  VISCOSITY: <air_dynamic_viscosity>
                  WALL_COMPLIANCE: <proximal_wall_compliance>
                  EXPONENT: <compliance_exponent>
              # Airway wall material (distal generations)
              - MAT: 2
                MAT_redairway_material:
                  VISCOSITY: <air_dynamic_viscosity>
                  WALL_COMPLIANCE: <distal_wall_compliance>
                  EXPONENT: <compliance_exponent>
              # Acinar material
              - MAT: 3
                MAT_redairway_acinus_material:
                  COMPLIANCE: <acinar_compliance>
                  VOLUME_RELAXED: <acinar_relaxed_volume>

            # == Boundary Conditions ===========================================

            # Tracheal inlet: prescribed pressure waveform
            DESIGN POINT REDAIRWAY PRESCRIBED CONDITIONS:
              - E: <tracheal_inlet_node_id>
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<tracheal_pressure_amplitude>]
                FUNCT: [<breathing_waveform_function>]

            # Breathing waveform (sinusoidal)
            FUNCT<breathing_waveform_function>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<breathing_waveform_expression>"

            # == Geometry ======================================================
            REDAIRWAY GEOMETRY:
              FILE: "<airway_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  REDAIRWAY:
                    LINE2:
                      MAT: 1

            RESULT DESCRIPTION:
              - REDAIRWAY:
                  DIS: "red_airway"
                  NODE: <result_node_id>
                  QUANTITY: "pre"
                  VALUE: <expected_pressure>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check air viscosity
        viscosity = params.get("VISCOSITY")
        if viscosity is not None:
            try:
                mu = float(viscosity)
                if mu <= 0:
                    issues.append(
                        f"Air VISCOSITY must be > 0, got {mu}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"VISCOSITY must be a positive number, "
                    f"got {viscosity!r}."
                )

        # Check wall compliance
        compliance = params.get("WALL_COMPLIANCE")
        if compliance is not None:
            try:
                c = float(compliance)
                if c < 0:
                    issues.append(
                        f"WALL_COMPLIANCE must be >= 0, got {c}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"WALL_COMPLIANCE must be a non-negative number, "
                    f"got {compliance!r}."
                )

        # Check acinar compliance
        acinar_comp = params.get("ACINAR_COMPLIANCE") or params.get("COMPLIANCE")
        if acinar_comp is not None:
            try:
                ac = float(acinar_comp)
                if ac <= 0:
                    issues.append(
                        f"Acinar COMPLIANCE must be > 0, got {ac}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"COMPLIANCE must be a positive number, "
                    f"got {acinar_comp!r}."
                )

        # Check acinar volume
        volume = params.get("VOLUME_RELAXED") or params.get("ACINAR_VOLUME_RELAXED")
        if volume is not None:
            try:
                v = float(volume)
                if v <= 0:
                    issues.append(
                        f"VOLUME_RELAXED must be > 0, got {v}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"VOLUME_RELAXED must be a positive number, "
                    f"got {volume!r}."
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
