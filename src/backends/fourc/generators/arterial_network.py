"""Arterial Network (1-D blood flow) generator for 4C.

Covers 1-D blood flow simulation in arterial networks using the reduced-order
model derived from cross-sectional averaging of the Navier-Stokes equations.
Solves for pressure, flow rate, and vessel cross-sectional area along
arterial segments connected at junctions.  Applications include pulse wave
propagation, arterial hemodynamics, and cardiovascular system modeling.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class ArterialNetworkGenerator(BaseGenerator):
    """Generator for 1-D arterial network blood flow problems in 4C."""

    module_key = "arterial_network"
    display_name = "Arterial Network (1-D Blood Flow)"
    problem_type = "ArterialNetwork"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The arterial network module solves 1-D blood flow equations "
                "in networks of compliant arterial segments.  The governing "
                "equations are derived from cross-sectional averaging of the "
                "Navier-Stokes equations, yielding a hyperbolic system for "
                "cross-sectional area A, flow rate Q, and pressure p.  "
                "Arterial segments are connected at junction nodes where "
                "mass conservation and pressure continuity are enforced.  "
                "Terminal (outlet) boundaries use Windkessel (lumped "
                "parameter) models to represent the downstream vasculature.  "
                "The PROBLEM TYPE is 'ArterialNetwork'.  The dynamics "
                "section is 'ARTERIAL DYNAMIC'.  Elements use the ARTERY "
                "element type (1-D line elements).  Materials use "
                "MAT_cnst_art which defines vessel wall properties "
                "(compliance, reference area, wall thickness)."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "ARTERIAL DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "IO",
                "IO/RUNTIME VTK OUTPUT",
                "WINDKESSEL CONDITIONS",
                "RESULT DESCRIPTION",
            ],
            "materials": {
                "MAT_cnst_art": {
                    "description": (
                        "Constant arterial wall material.  Defines the "
                        "mechanical properties of the arterial wall for "
                        "the 1-D model: Young's modulus, wall thickness, "
                        "reference cross-sectional area, and blood "
                        "viscosity."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": (
                                "Young's modulus of arterial wall "
                                "[dyn/cm^2 or Pa]"
                            ),
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Poisson's ratio of arterial wall",
                            "range": "[0, 0.5)",
                        },
                        "TH": {
                            "description": "Wall thickness [cm or m]",
                            "range": "> 0",
                        },
                        "DENS": {
                            "description": "Blood density [g/cm^3 or kg/m^3]",
                            "range": "> 0",
                        },
                        "VISCOSITY": {
                            "description": (
                                "Dynamic blood viscosity [Poise or Pa s]"
                            ),
                            "range": "> 0",
                        },
                        "AREA0": {
                            "description": (
                                "Reference (unstressed) cross-sectional "
                                "area [cm^2 or m^2]"
                            ),
                            "range": "> 0",
                        },
                    },
                },
            },
            "solver": {
                "direct": {
                    "type": "UMFPACK",
                    "notes": (
                        "1-D arterial network systems are small and "
                        "efficiently solved by direct solvers."
                    ),
                },
            },
            "time_integration": {
                "TIMESTEP": (
                    "Time step size.  Must satisfy CFL condition for "
                    "the pulse wave speed c = sqrt(E*h/(2*rho*A0)).  "
                    "Typical: 1e-4 to 1e-3 s."
                ),
                "NUMSTEP": "Total number of time steps.",
                "MAXTIME": (
                    "Maximum simulation time.  One cardiac cycle is "
                    "approximately 0.8-1.0 s."
                ),
            },
            "boundary_conditions": {
                "inlet": (
                    "Inflow boundary condition at the aortic root.  "
                    "Typically a prescribed flow rate waveform Q(t) "
                    "or pressure waveform p(t)."
                ),
                "outlet_windkessel": (
                    "Windkessel (RCR) model at terminal outlets.  "
                    "Represents downstream resistance, compliance, "
                    "and venous pressure.  Parameters: R_proximal, "
                    "R_distal, C."
                ),
                "junction": (
                    "Junction conditions at bifurcation points.  "
                    "Conservation of mass and continuity of total "
                    "pressure are enforced automatically."
                ),
            },
            "pitfalls": [
                (
                    "[Numerical] CFL condition for arterial-"
                    "network 1D: dt <= dx / c_max, where "
                    "c_max is the maximum pulse-wave speed. "
                    "Signal: dt > dx/c_max gives NaN within "
                    "~10 steps (typical 'energy not "
                    "conserved' message); for blood at "
                    "c_max ~ 8-12 m/s and segment dx = 1 cm, "
                    "use dt < 1 ms. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] AREA0 (reference cross-section "
                    "area) must MATCH the actual vessel "
                    "geometry. The 1D model uses area "
                    "perturbation from AREA0 to compute "
                    "pressure (p = beta*(sqrt(A) - "
                    "sqrt(A0))). Signal: AREA0 set to half "
                    "the physiological value gives pressure "
                    "off by ~2x because the (A - A0) drives "
                    "the pressure scaling. Match AREA0 to "
                    "the diameter at the chosen vessel. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Windkessel parameters (R, "
                    "C) at terminal outlets strongly affect "
                    "REFLECTED waves. Signal: arbitrary "
                    "R and C give non-physiological "
                    "pressure waveforms — a too-high R "
                    "creates strong end-systolic reflection "
                    "(double-peak in central artery "
                    "pressure); too-low R leaks pressure "
                    "and gives a damped waveform. Tune R "
                    "from physiological impedance "
                    "Z_terminal = rho*c/A_terminal. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] Junction connectivity must be "
                    "CORRECTLY defined. Each junction node "
                    "connects exactly the expected number "
                    "of arterial segments (typically 3 for "
                    "a bifurcation). Signal: a mis-"
                    "connected junction node (4 segments "
                    "when 3 expected, or shared node IDs "
                    "across non-adjacent segments) raises "
                    "'inconsistent junction connectivity' "
                    "at network setup. Verify each junction "
                    "has the right degree. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] Blood is typically Newtonian "
                    "with VISCOSITY ~ 0.03-0.04 Poise = "
                    "3-4 mPa s = 3-4e-3 Pa s. Non-"
                    "Newtonian effects (shear-thinning) "
                    "are NOT captured by the 1D model. "
                    "Signal: using VISCOSITY = 1e-3 Pa s "
                    "(water) gives wave speed off by "
                    "~20% from physiological value; use "
                    "blood viscosity and validate by "
                    "comparing pulse-transit time against "
                    "in-vivo data. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] 1D model ASSUMES axisymmetric "
                    "flow and small wall displacements. "
                    "Cannot capture flow separation, "
                    "secondary flows, or aneurysm rupture "
                    "mechanics. Signal: applying 1D to a "
                    "saccular aneurysm gives smoothly-"
                    "varying pressure but misses the "
                    "stagnation region and wall stress "
                    "concentration — use a 3D FSI model "
                    "for rupture-risk analysis. The 1D "
                    "result is correct for global "
                    "hemodynamics (pulse propagation) "
                    "only. (Audit 2026-06-02.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "single_artery_1d",
                    "description": (
                        "A single arterial segment with prescribed "
                        "inflow at one end and a Windkessel outlet "
                        "at the other.  Tests pulse wave propagation, "
                        "wave reflection, and pressure-flow "
                        "relationship."
                    ),
                    "template_variant": "single_artery_1d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "single_artery_1d",
                "description": (
                    "Single arterial segment with prescribed inflow "
                    "and Windkessel outlet.  MAT_cnst_art material, "
                    "1-D ARTERY elements, UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "single_artery_1d") -> str:
        templates = {
            "single_artery_1d": self._template_single_artery_1d,
        }
        if variant == "default":
            variant = "single_artery_1d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_single_artery_1d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 1-D Arterial Network -- Single Artery Segment
            #
            # A single compliant artery with a prescribed flow rate at the
            # inlet and a three-element Windkessel (RCR) at the outlet.
            # The pulse wave propagates along the artery and reflects from
            # the Windkessel boundary.
            #
            # Mesh: 1-D line mesh with:
            #   element_block 1 = artery segment (LINE2)
            #   node_set 1 = inlet node
            #   node_set 2 = outlet node
            # ---------------------------------------------------------------
            TITLE:
              - "1-D arterial network -- generated template"
            PROBLEM SIZE:
              DIM: 1
            PROBLEM TYPE:
              PROBLEMTYPE: "ArterialNetwork"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>

            # == Arterial dynamics =============================================
            ARTERIAL DYNAMIC:
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              LINEAR_SOLVER: 1
              RESULTSEVERY: <results_output_interval>
              RESTARTEVERY: <restart_interval>

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "artery_solver"

            # == Materials =====================================================
            MATERIALS:
              - MAT: 1
                MAT_cnst_art:
                  YOUNG: <arterial_wall_Young_modulus>
                  NUE: <arterial_wall_Poisson_ratio>
                  TH: <wall_thickness>
                  DENS: <blood_density>
                  VISCOSITY: <blood_viscosity>
                  AREA0: <reference_cross_sectional_area>

            # == Inflow waveform function ======================================
            FUNCT<inflow_function_id>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<inflow_waveform_expression>"

            # == Boundary Conditions ===========================================

            # Inlet: prescribed flow rate
            DESIGN POINT ARTERY DIRICH CONDITIONS:
              - E: <inlet_node_id>
                NUMDOF: <num_artery_dofs>
                ONOFF: [<active_inlet_dofs>]
                VAL: [<inlet_flow_rate>]
                FUNCT: [<inflow_function_id>]

            # Outlet: Windkessel (3-element RCR)
            DESIGN POINT WINDKESSEL CONDITIONS:
              - E: <outlet_node_id>
                R_PROXIMAL: <windkessel_proximal_resistance>
                R_DISTAL: <windkessel_distal_resistance>
                C: <windkessel_compliance>
                P_VENOUS: <venous_pressure>

            # == Geometry ======================================================
            ARTERY GEOMETRY:
              FILE: "<mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  ARTERY:
                    LINE2:
                      MAT: 1

            RESULT DESCRIPTION:
              - ARTERY:
                  DIS: "artery"
                  NODE: <result_node_id>
                  QUANTITY: "one_d_artery_pressure"
                  VALUE: <expected_pressure>
                  TOLERANCE: <result_tolerance>
              - ARTERY:
                  DIS: "artery"
                  NODE: <result_node_id>
                  QUANTITY: "one_d_artery_flowrate"
                  VALUE: <expected_flow_rate>
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
                    issues.append(
                        f"Arterial wall YOUNG must be > 0, got {e}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"YOUNG must be a positive number, got {young!r}."
                )

        # Check wall thickness
        th = params.get("TH")
        if th is not None:
            try:
                t = float(th)
                if t <= 0:
                    issues.append(f"Wall thickness TH must be > 0, got {t}.")
            except (TypeError, ValueError):
                issues.append(
                    f"TH must be a positive number, got {th!r}."
                )

        # Check reference area
        area0 = params.get("AREA0")
        if area0 is not None:
            try:
                a = float(area0)
                if a <= 0:
                    issues.append(
                        f"AREA0 (reference area) must be > 0, got {a}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"AREA0 must be a positive number, got {area0!r}."
                )

        # Check blood density
        dens = params.get("DENS")
        if dens is not None:
            try:
                d = float(dens)
                if d <= 0:
                    issues.append(f"Blood DENS must be > 0, got {d}.")
            except (TypeError, ValueError):
                issues.append(
                    f"DENS must be a positive number, got {dens!r}."
                )

        # Check viscosity
        visc = params.get("VISCOSITY")
        if visc is not None:
            try:
                mu = float(visc)
                if mu <= 0:
                    issues.append(
                        f"Blood VISCOSITY must be > 0, got {mu}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"VISCOSITY must be a positive number, got {visc!r}."
                )

        # Check Windkessel parameters
        for wk_param in ("R_PROXIMAL", "R_DISTAL", "C"):
            val = params.get(wk_param)
            if val is not None:
                try:
                    v = float(val)
                    if v <= 0:
                        issues.append(
                            f"Windkessel {wk_param} must be > 0, got {v}."
                        )
                except (TypeError, ValueError):
                    issues.append(
                        f"{wk_param} must be a positive number, "
                        f"got {val!r}."
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
