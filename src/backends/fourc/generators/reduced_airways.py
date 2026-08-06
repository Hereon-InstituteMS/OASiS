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
                    "[Input] 4C does NOT validate reduced-airway tree topology. "
                    "A loop, or a branch ending on a node with no boundary "
                    "condition, is accepted and solved. Signal: none - there is "
                    "no topology check and no message. Neither `node X has "
                    "degree 1 (dangling)` nor `cycle detected in airway tree` "
                    "exists in 4C; the run completes its time loop and only a "
                    "RESULT DESCRIPTION entry reveals that the answer moved. "
                    "Check the tree yourself before trusting it. (Audit "
                    "2026-08-06, verified by execution.)"
                ),
                (
                    "[Input] Boundary conditions at the trachea define the "
                    "driving force: a DESIGN NODE Reduced D AIRWAYS PRESCRIBED "
                    "CONDITIONS entry on the inlet DNODE, flow by default or "
                    "boundarycond: pressure. Without it nothing drives the "
                    "tree. Signal: the run converges and every step prints "
                    "|Pressure|_max: 0.000E+00 and |Q|_max: 0.000E+00. There is "
                    "NO warning about the missing condition; `no DESIGN POINT "
                    "1D DBC defined at trachea node` does not exist in 4C. "
                    "(Audit 2026-08-06, verified by execution.)"
                ),
                (
                    "[Numerical] The terminal impedance of a reduced-airway "
                    "tree is the acinus element's MAT_0D_MAXWELL_ACINUS_* "
                    "stiffness and viscosity, not a separate acinar condition. "
                    "Signal: make the acinus much softer and the inlet pressure "
                    "collapses onto the prescribed downstream pressure - no "
                    "impedance - with no diagnostic. The acinar volume does NOT "
                    "diverge when the inlet carries a prescribed flow: it is "
                    "fixed by that flow, so watch the pressure, not the volume. "
                    "(Audit 2026-08-06, verified by execution.)"
                ),
                (
                    "[Numerical] Airway wall behaviour is an ELEMENT-line "
                    "option and it depends on the airway TYPE. There is no "
                    "MAT_redAirway material in 4C and no COMPLIANCE material "
                    "parameter; the airway material is MAT_fluid. Collapse and "
                    "reopening come from AirwayColl with Pcrit_Open / "
                    "Pcrit_Close / S_Open / S_Close on the RED_AIRWAY line. "
                    "Signal: on a TYPE Resistive airway, WallElasticity is "
                    "parsed and then silently ignored - the result is bit-for- "
                    "bit unchanged - while it does act on "
                    "ConvectiveViscoElasticRLC. (Audit 2026-08-06, verified by "
                    "execution.)"
                ),
                (
                    "[Numerical] Reduced model assumes LONG WAVELENGTH "
                    "(Womersley number restrictions, Wo < O(1)). For high- "
                    "frequency ventilation a full 3D CFD approach may be needed "
                    "instead. Signal: none from 4C. The reduced-airway and "
                    "artery modules never compute or print a Womersley number - "
                    "the word appears in the source only for 3D fluid inflow "
                    "profiles - so an HFOV-rate drive runs to completion and "
                    "silently returns different numbers. Compute Wo = R * "
                    "sqrt(omega / nu) yourself. (Audit 2026-08-06, verified by "
                    "execution.)"
                ),
                (
                    "[Input] Properties that vary by airway GENERATION are "
                    "element-line tokens, not materials: Area, WallElasticity, "
                    "WallThickness, PoissonsRatio and Generation sit on each "
                    "RED_AIRWAY line, and 4C's own multi-generation decks drive "
                    "every airway from one MAT_fluid. There is no "
                    "MAT_redAirway. Signal: none - a uniform tree is accepted "
                    "without comment; taper Area and set the wall properties "
                    "per element, and check the total resistance yourself. "
                    "(Audit 2026-08-06, verified by execution.)"
                ),
                (
                    "[Input] The element keywords are RED_AIRWAY and "
                    "RED_ACINUS, and each has its own TYPE enum. Signal: the "
                    "spelling REDAIRWAY is rejected by the ParObject factory "
                    "with \"Unknown type 'REDAIRWAY' of finite element\" from "
                    "4C_comm_parobjectfactory.cpp; a beam or fluid element "
                    "keyword dies earlier still, in the element reader, with "
                    "\"Required 'one_of' not found in input line\"; and using "
                    "RED_AIRWAY with an acinus TYPE prints the list of valid "
                    "airway TYPEs. Neither `unknown element type for "
                    "ELEMENT_BLOCK X` nor `elementType BEAM3R has no RedAirway "
                    "implementation` exists in 4C. (Audit 2026-08-06, verified "
                    "by execution.)"
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
