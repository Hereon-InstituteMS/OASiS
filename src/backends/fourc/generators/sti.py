"""Scalar-Thermo Interaction (STI) generator for 4C.

Covers coupling of scalar transport with thermal fields.  The thermal field
provides a temperature distribution that affects scalar transport coefficients
(temperature-dependent diffusion, reaction rates), while the scalar field can
generate heat through exothermic/endothermic reactions.  Application domains
include reactive transport, heat-generating chemical processes, and
temperature-dependent species diffusion.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class STIGenerator(BaseGenerator):
    """Generator for Scalar-Thermo Interaction problems in 4C."""

    module_key = "sti"
    display_name = "Scalar-Thermo Interaction (STI)"
    problem_type = "Scalar_Thermo_Interaction"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Scalar-Thermo Interaction (STI) couples a scalar transport "
                "field (diffusion-reaction or electrochemistry) with a thermal "
                "field.  The temperature distribution from the thermal field "
                "modifies transport properties (diffusion coefficients, "
                "reaction rates) in the scalar field via Arrhenius-type "
                "temperature dependencies.  Conversely, exothermic or "
                "endothermic reactions in the scalar field and Joule heating "
                "generate volumetric heat sources for the thermal field.  "
                "The PROBLEM TYPE is 'Scalar_Thermo_Interaction'.  Required "
                "dynamics sections are SCALAR TRANSPORT DYNAMIC and "
                "THERMAL DYNAMIC, plus the coupling section STI DYNAMIC.  "
                "Both fields share the same mesh (via mesh cloning from "
                "a common discretisation)."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "SCALAR TRANSPORT DYNAMIC",
                "THERMAL DYNAMIC",
                "STI DYNAMIC",
                "STI DYNAMIC/MONOLITHIC",
                "SOLVER 1",
                "MATERIALS",
                "CLONING MATERIAL MAP",
            ],
            "optional_sections": [
                "ELCH CONTROL",
                "SCALAR TRANSPORT DYNAMIC/STABILIZATION",
                "SCALAR TRANSPORT DYNAMIC/NONLINEAR",
                "IO/RUNTIME VTK OUTPUT",
                "THERMAL DYNAMIC/RUNTIME VTK OUTPUT",
            ],
            "materials": {
                "MAT_soret": {
                    "description": (
                        "Soret material for scalar transport with "
                        "thermodiffusion (Soret effect).  Captures the "
                        "species flux driven by temperature gradients."
                    ),
                    "parameters": {
                        "DIFFUSIVITY": {
                            "description": "Diffusion coefficient [m^2/s]",
                            "range": "> 0",
                        },
                        "SORET_COEFFICIENT": {
                            "description": (
                                "Soret coefficient S_T [1/K] controlling "
                                "thermodiffusion"
                            ),
                            "range": "any (positive or negative)",
                        },
                    },
                },
                "MAT_Fourier": {
                    "description": (
                        "Fourier heat conduction material for the thermal "
                        "field.  Defines volumetric heat capacity and "
                        "thermal conductivity."
                    ),
                    "parameters": {
                        "CAPA": {
                            "description": "Volumetric heat capacity [J/(m^3 K)]",
                            "range": "> 0",
                        },
                        "CONDUCT": {
                            "description": "Thermal conductivity [W/(m K)]",
                            "range": "> 0",
                        },
                    },
                },
                "MAT_scatra_reaction_thermo": {
                    "description": (
                        "Temperature-dependent reaction material for the "
                        "scalar transport field.  Reaction rate follows "
                        "Arrhenius kinetics with activation energy."
                    ),
                },
            },
            "solver": {
                "monolithic": {
                    "type": "UMFPACK for small problems",
                    "notes": (
                        "The monolithic STI solver handles the coupled "
                        "scalar-thermal block system.  For small to medium "
                        "problems UMFPACK is robust."
                    ),
                },
            },
            "time_integration": {
                "STI_DYNAMIC": (
                    "Controls the coupled STI time stepping.  NUMSTEP, "
                    "TIMESTEP, MAXTIME define the global time loop.  "
                    "COUPALGO selects monolithic or partitioned coupling."
                ),
                "SCALAR_TRANSPORT_DYNAMIC": (
                    "SOLVERTYPE: 'nonlinear' for temperature-dependent "
                    "coefficients.  TIMESTEP should match the STI global "
                    "time step."
                ),
                "THERMAL_DYNAMIC": (
                    "DYNAMICTYPE: 'OneStepTheta' for transient thermal "
                    "analysis or 'Statics' for steady-state thermal."
                ),
            },
            "pitfalls": [
                (
                    "[Input] CLONING MATERIAL MAP is required: "
                    "STI builds the thermo field by cloning "
                    "the scatra mesh, so the pairings run "
                    "scatra -> thermo (one per material "
                    "group).  There is no structural field to "
                    "map from. Signal: omitting it gives the "
                    "shared cloning utility's generic message, "
                    "'At least one material pairing required "
                    "in --CLONING MATERIAL MAP.' from "
                    "core/fem/src/general/utils/"
                    "4C_fem_general_utils_createdis.hpp, exit "
                    "1 -- note it names neither STI nor thermo "
                    "and still spells the section in the "
                    "retired --SECTION dat form. (Verified by "
                    "execution 2026-08-06; 'cannot clone "
                    "material for thermo' is not in the "
                    "binary.)"
                ),
                (
                    "[Input] The thermo -> scatra half of the "
                    "coupling is carried by MAT_soret's SORET "
                    "coefficient.  A temperature-scaling "
                    "function on the diffusion coefficient is "
                    "NOT needed for two-way coupling: 4C's own "
                    "monolithic STI decks leave every "
                    "DIFF_COEF_TEMP_SCALE_FUNCT and "
                    "COND_TEMP_SCALE_FUNCT at 0 and are still "
                    "two-way. Signal: set SORET to 0 and the "
                    "SPECIES field moves, not just the "
                    "temperature -- that is the test for "
                    "whether your coupling is live.  Use the "
                    "temperature-scaling functions for "
                    "Arrhenius electrode kinetics, which is a "
                    "different effect. (Verified by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Numerical] STI has ONE time step, "
                    "SCALAR TRANSPORT DYNAMIC's; the thermo "
                    "field inherits it.  There is no second "
                    "DYNAMIC section to keep in step. Signal: "
                    "STI DYNAMIC has no TIMESTEP parameter at "
                    "all -- adding one is a parse abort, "
                    "'Could not match this input' from "
                    "core/io/src/4C_io_input_spec_builders.cpp "
                    "-- while a THERMAL DYNAMIC section with "
                    "its own TIMESTEP and NUMSTEP is accepted, "
                    "silently ignored, and changes nothing. "
                    "The real hazard is that second case: a "
                    "thermal time step you believe you set. "
                    "(Verified by execution 2026-08-06; the "
                    "predicted interpolation error from "
                    "mismatched steps cannot occur.)"
                ),
                (
                    "[Input] The FIELD PREFIX in a condition "
                    "section name silently decides which field "
                    "gets the load.  DESIGN SURF NEUMANN, "
                    "DESIGN SURF THERMO NEUMANN and DESIGN "
                    "SURF TRANSPORT NEUMANN CONDITIONS are ALL "
                    "valid section names, so nothing is "
                    "checked when the file is read. Signal: "
                    "moving a block between them either kills "
                    "the run mid-solve with 'The NUMDOF you "
                    "have entered in your TRANSPORT NEUMANN "
                    "CONDITION does not equal the number of "
                    "scalars.' from scatra_ele/"
                    "4C_scatra_ele_boundary_calc.cpp -- which "
                    "says TRANSPORT even when your section "
                    "said THERMO, because the thermo field is "
                    "itself a scatra discretisation -- or, "
                    "worse, completes with NO diagnostic and a "
                    "different answer.  Check the section "
                    "prefix against the field you meant. "
                    "(Verified by execution 2026-08-06; "
                    "'condition does not apply to this field' "
                    "is not in the binary and there is no "
                    "parse-time field check.)"
                ),
                (
                    "[Input] For STI with electrochemistry, "
                    "include ELCH CONTROL and set STI "
                    "DYNAMIC/SCATRATIMINTTYPE: 'Elch'.  "
                    "Neither omission degrades gracefully. "
                    "Signal: without ELCH CONTROL, 'Invalid "
                    "type of closing equation for electric "
                    "potential!' from scatra_ele/"
                    "4C_scatra_ele_parameter_elch.cpp, exit 1. "
                    "With SCATRATIMINTTYPE left at its default "
                    "'Standard', the run parses, builds both "
                    "fields, writes its t=0 output and THEN "
                    "dies on a raw 'Signal: Floating point "
                    "exception (8)' while constructing "
                    "ScaTraEleCalcElchElectrodeSTIThermo -- no "
                    "PROC 0 ERROR banner, no source file, and "
                    "SCATRATIMINTTYPE is never mentioned. "
                    "(Verified by execution 2026-08-06; there "
                    "is no quiet zero-Joule-heating run.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "thermodiffusion_3d",
                    "description": (
                        "A domain with a temperature gradient driving "
                        "species transport via the Soret effect.  Tests "
                        "two-way coupling between thermal and scalar "
                        "fields.  Uses monolithic STI with MAT_soret "
                        "and MAT_Fourier."
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
                    "3-D monolithic STI: coupled scalar transport and "
                    "thermal fields.  Temperature-dependent diffusion "
                    "in the scalar field, heat generation feedback.  "
                    "UMFPACK solver."
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
            # 3-D Monolithic Scalar-Thermo Interaction (STI)
            #
            # Coupled scalar transport and thermal fields.  Temperature
            # gradients drive thermodiffusion (Soret effect) in the scalar
            # field, and scalar reactions generate volumetric heat sources
            # in the thermal field.
            #
            # Mesh: requires an exodus file with:
            #   element_block 1 = domain (HEX8 or TET4)
            #   node_set 1 = hot face (thermal Dirichlet)
            #   node_set 2 = cold face (thermal Dirichlet)
            #   node_set 3 = scalar Dirichlet face
            # ---------------------------------------------------------------
            TITLE:
              - "3-D scalar-thermo interaction -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Scalar_Thermo_Interaction"
            IO:
              STDOUTEVERY: <stdout_interval>
              THERM_HEATFLUX: "Initial"
              THERM_TEMPGRAD: "Initial"
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>

            # == Scalar Transport ==============================================
            SCALAR TRANSPORT DYNAMIC:
              SOLVERTYPE: "nonlinear"
              TIMESTEP: <scalar_timestep>
              NUMSTEP: <scalar_num_steps>
              MAXTIME: <scalar_max_time>
              MATID: <scalar_material_id>
              INITIALFIELD: "field_by_function"
              INITFUNCNO: <initial_scalar_function_id>
              LINEAR_SOLVER: 1
            SCALAR TRANSPORT DYNAMIC/STABILIZATION:
              STABTYPE: "no_stabilization"
            SCALAR TRANSPORT DYNAMIC/NONLINEAR:
              ITEMAX: <max_nonlinear_iterations>
              CONVTOL: <nonlinear_convergence_tolerance>

            # == Thermal =======================================================
            THERMAL DYNAMIC:
              DYNAMICTYPE: "OneStepTheta"
              TIMESTEP: <thermal_timestep>
              NUMSTEP: <thermal_num_steps>
              MAXTIME: <thermal_max_time>
              LINEAR_SOLVER: 1
            THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
              OUTPUT_THERMO: true
              TEMPERATURE: true

            # == STI coupling ==================================================
            STI DYNAMIC:
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              TIMESTEP: <timestep>
              RESULTSEVERY: <results_output_interval>
              RESTARTEVERY: <restart_interval>
              COUPALGO: "sti_Monolithic"
            STI DYNAMIC/MONOLITHIC:
              LINEAR_SOLVER: 1

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "sti_solver"

            # == Materials =====================================================
            MATERIALS:
              # Scalar transport material
              - MAT: 1
                MAT_scatra:
                  DIFFUSIVITY: <diffusion_coefficient>
              # Fourier heat conduction (thermal field)
              - MAT: 2
                MAT_Fourier:
                  CAPA: <volumetric_heat_capacity>
                  CONDUCT:
                    constant: [<thermal_conductivity>]

            # Clone scalar mesh -> thermal mesh
            CLONING MATERIAL MAP:
              - SRC_FIELD: "scatra"
                SRC_MAT: 1
                TAR_FIELD: "thermo"
                TAR_MAT: 2

            # == Initial condition function ====================================
            FUNCT<initial_scalar_function_id>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_scalar_expression>"

            # == Boundary Conditions ===========================================

            # Scalar transport: Dirichlet
            DESIGN SURF TRANSPORT DIRICH CONDITIONS:
              - E: <scalar_dirichlet_face_id>
                NUMDOF: <num_scalar_dofs>
                ONOFF: [<active_scalar_dofs>]
                VAL: [<scalar_boundary_values>]
                FUNCT: [<scalar_time_functions>]

            # Thermal: hot face
            DESIGN SURF THERMO DIRICH CONDITIONS:
              - E: <hot_face_id>
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<hot_face_temperature>]
                FUNCT: [0]
              # Thermal: cold face
              - E: <cold_face_id>
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<cold_face_temperature>]
                FUNCT: [0]

            # == Geometry ======================================================
            TRANSPORT GEOMETRY:
              FILE: "<mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  TRANSP:
                    HEX8:
                      MAT: 1
                      TYPE: Std

            RESULT DESCRIPTION:
              - SCATRA:
                  DIS: "scatra"
                  NODE: <result_node_id>
                  QUANTITY: "phi"
                  VALUE: <expected_scalar_value>
                  TOLERANCE: <result_tolerance>
              - THERMAL:
                  DIS: "thermo"
                  NODE: <result_node_id>
                  QUANTITY: "temp"
                  VALUE: <expected_temperature>
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
                if d <= 0:
                    issues.append(
                        f"DIFFUSIVITY must be > 0, got {d}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DIFFUSIVITY must be a positive number, "
                    f"got {diffusivity!r}."
                )

        # Check heat capacity
        capa = params.get("CAPA")
        if capa is not None:
            try:
                c = float(capa)
                if c <= 0:
                    issues.append(
                        f"CAPA must be > 0, got {c}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"CAPA must be a positive number, got {capa!r}."
                )

        # Check conductivity
        conduct = params.get("CONDUCT")
        if conduct is not None:
            if isinstance(conduct, dict):
                vals = conduct.get("constant", [])
                if isinstance(vals, list):
                    for v in vals:
                        try:
                            if float(v) <= 0:
                                issues.append(
                                    f"CONDUCT values must be > 0, got {v}."
                                )
                        except (TypeError, ValueError):
                            issues.append(
                                f"CONDUCT values must be positive, "
                                f"got {v!r}."
                            )

        # Check CLONING MATERIAL MAP presence
        has_cloning = params.get("has_cloning_material_map")
        if has_cloning is not None and not has_cloning:
            issues.append(
                "CLONING MATERIAL MAP is required for STI.  It maps "
                "the scalar transport material to the thermal material."
            )

        # Check convergence tolerance
        convtol = params.get("CONVTOL")
        if convtol is not None:
            try:
                ct = float(convtol)
                if ct <= 0:
                    issues.append(f"CONVTOL must be > 0, got {ct}.")
            except (TypeError, ValueError):
                issues.append(
                    f"CONVTOL must be a positive number, got {convtol!r}."
                )

        return issues
