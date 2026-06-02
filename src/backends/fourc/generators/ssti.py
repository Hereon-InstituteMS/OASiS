"""Structure-Scalar-Thermo Interaction (SSTI) generator for 4C.

Covers monolithic three-field coupling of structural mechanics, scalar
transport, and thermal fields.  This extends SSI by adding a thermal field
that influences both the structural (thermal expansion) and scalar transport
(temperature-dependent diffusion) fields.  Key application: battery cell
modeling where mechanical, electrochemical, and thermal effects interact.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class SSTIGenerator(BaseGenerator):
    """Generator for Structure-Scalar-Thermo Interaction problems in 4C."""

    module_key = "ssti"
    display_name = "Structure-Scalar-Thermo Interaction (SSTI)"
    problem_type = "Structure_Scalar_Thermo_Interaction"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Structure-Scalar-Thermo Interaction (SSTI) is a three-field "
                "coupled problem combining structural mechanics, scalar "
                "transport (optionally with electrochemistry), and thermal "
                "analysis.  The thermal field drives thermal expansion in "
                "the structure and modifies transport coefficients in the "
                "scalar field.  The scalar field (e.g. lithium concentration) "
                "causes volumetric swelling in the structure.  Heat "
                "generation from electrochemical reactions and mechanical "
                "dissipation feeds back into the thermal field.  The "
                "PROBLEM TYPE is 'Structure_Scalar_Thermo_Interaction'.  "
                "This problem requires dynamics sections for all three "
                "fields: STRUCTURAL DYNAMIC, SCALAR TRANSPORT DYNAMIC, "
                "and THERMAL DYNAMIC, plus the coupling section "
                "SSTI CONTROL.  Elements typically use SOLIDSCATRA to "
                "carry both displacement and scalar DOFs, with mesh "
                "cloning for the thermal field."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "STRUCTURAL DYNAMIC",
                "SCALAR TRANSPORT DYNAMIC",
                "THERMAL DYNAMIC",
                "SSTI CONTROL",
                "SSTI CONTROL/MONOLITHIC",
                "SSI CONTROL",
                "SSI CONTROL/MONOLITHIC",
                "TSI DYNAMIC",
                "SOLVER 1",
                "SOLVER 2",
                "MATERIALS",
                "CLONING MATERIAL MAP",
            ],
            "optional_sections": [
                "ELCH CONTROL",
                "SSI CONTROL/ELCH",
                "SCALAR TRANSPORT DYNAMIC/STABILIZATION",
                "SCALAR TRANSPORT DYNAMIC/S2I COUPLING",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/STRUCTURE",
            ],
            "materials": {
                "MAT_Struct_ThermoStVenantK": {
                    "description": (
                        "Thermo-elastic St. Venant-Kirchhoff material for "
                        "the structural field.  Supports thermal expansion "
                        "and links to a thermal material via THERMOMAT.  "
                        "Used in combination with scalar-dependent "
                        "inelastic growth for full SSTI coupling."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus array",
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Poisson's ratio",
                            "range": "(0, 0.5)",
                        },
                        "DENS": {
                            "description": "Mass density",
                            "range": "> 0",
                        },
                        "THEXPANS": {
                            "description": "Thermal expansion coefficient [1/K]",
                            "range": "> 0",
                        },
                        "INITTEMP": {
                            "description": "Reference temperature for zero thermal strain",
                            "range": "any (often 293 K)",
                        },
                        "THERMOMAT": {
                            "description": "Material ID of the thermal material (MAT_Fourier)",
                            "range": "valid MAT ID",
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
                "MAT_electrode": {
                    "description": (
                        "Electrode material for the scalar transport field "
                        "when electrochemistry is active.  Defines "
                        "concentration-dependent diffusion, conductivity, "
                        "and open-circuit potential."
                    ),
                },
            },
            "solver": {
                "field_solvers": {
                    "type": "UMFPACK (direct)",
                    "notes": (
                        "Individual field solvers for structure, scalar "
                        "transport, and thermal fields."
                    ),
                },
                "monolithic_solver": {
                    "type": "UMFPACK or Belos with block preconditioner",
                    "notes": (
                        "The monolithic SSTI solver handles the coupled "
                        "three-field block system.  For small problems "
                        "UMFPACK is sufficient; for large problems use "
                        "iterative solvers with appropriate "
                        "block preconditioning."
                    ),
                },
            },
            "time_integration": {
                "SSTI_CONTROL": (
                    "Controls the overall three-field coupling loop.  "
                    "NUMSTEP, TIMESTEP, MAXTIME define the global time "
                    "stepping.  COUPALGO selects the coupling strategy."
                ),
                "SSI_CONTROL": (
                    "Sub-coupling between structure and scalar transport "
                    "within each SSTI step."
                ),
                "TSI_DYNAMIC": (
                    "Sub-coupling between thermal and structural fields "
                    "within each SSTI step."
                ),
            },
            "pitfalls": [
                (
                    "[Input] SSTI requires CLONING MATERIAL "
                    "MAP entries for BOTH the scalar (structure"
                    " -> scatra) AND thermal (structure -> "
                    "thermo) field cloning. Signal: missing "
                    "either CLONING entry aborts at setup "
                    "with 'cannot clone material for "
                    "<field>' — the SSTI factory needs all "
                    "three fields fully connected. The "
                    "CLONING MATERIAL MAP must contain "
                    "exactly two SRC/TAR entries. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] Elements must be SOLIDSCATRA "
                    "(not plain SOLID) to carry BOTH "
                    "displacement and scalar DOFs. The "
                    "thermal DOF comes from mesh cloning. "
                    "Signal: SOLID elements in an SSTI input "
                    "aborts with 'no SCATRA discretisation "
                    "found' at setup; the structural mesh "
                    "has no SCATRA-side DOFs to clone "
                    "into a thermal field. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] All three DYNAMIC sections "
                    "(STRUCTURAL, SCALAR TRANSPORT, "
                    "THERMAL) must be consistently "
                    "configured with MATCHING time step "
                    "sizes. Signal: different TIMESTEP "
                    "values produce sub-stepping which "
                    "accumulates O(dt) splitting error per "
                    "outer iteration; result diverges from "
                    "a uniform-dt monolithic reference by "
                    "5-15% on the slowest-evolving field. "
                    "Match TIMESTEP across all three. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Input] When electrochemistry is "
                    "involved, set SCATRATIMINTTYPE: 'Elch' "
                    "in SSI CONTROL and include the ELCH "
                    "CONTROL section. Signal: omitting "
                    "either gives plain scalar transport "
                    "without the electrochemical source "
                    "terms — BV current at electrode "
                    "interfaces is ZERO regardless of "
                    "applied potential. Both flags are "
                    "needed even for SSTI. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] SSTI CONTROL must reference "
                    "the monolithic solver via "
                    "LINEAR_SOLVER. The solver must handle "
                    "the three-field BLOCK system. Signal: "
                    "pointing LINEAR_SOLVER at UMFPACK "
                    "works for small problems but is "
                    "memory-prohibitive for any realistic "
                    "battery / TSI-electrode mesh; for "
                    "production use Belos GMRES + Teko "
                    "block preconditioner. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] INITTEMP in the structural "
                    "material MUST MATCH the initial "
                    "temperature field. Signal: INITTEMP = "
                    "300 (room) but THERMAL DYNAMIC's "
                    "INITIALFIELD prescribes T(x,0) = 0 "
                    "produces spurious thermal strain "
                    "alpha * (-300) = large compression at "
                    "t=0; the structure visibly contracts "
                    "before any heating. Set INITTEMP "
                    "equal to the prescribed initial T. "
                    "(Audit 2026-06-02.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "battery_cell_3d",
                    "description": (
                        "A battery electrode undergoing lithium "
                        "intercalation with coupled thermal, mechanical, "
                        "and electrochemical effects.  Temperature affects "
                        "diffusion kinetics, intercalation causes swelling, "
                        "and Joule heating raises the temperature."
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
                    "3-D monolithic SSTI: coupled structure-scalar-thermo "
                    "problem.  SOLIDSCATRA HEX8 elements, thermo-elastic "
                    "structural material, Fourier thermal material, "
                    "electrode scalar material, UMFPACK solvers."
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
            # 3-D Monolithic Structure-Scalar-Thermo Interaction (SSTI)
            #
            # Three-field coupling: structural mechanics + scalar transport
            # (electrochemistry) + thermal.  The thermal field drives
            # thermal expansion and modifies transport coefficients.
            # Scalar concentration causes structural swelling.  Heat
            # generation from reactions feeds back into the thermal field.
            #
            # Elements: SOLIDSCATRA HEX8 (displacement + scalar DOFs;
            #           thermal DOF via mesh cloning)
            #
            # Mesh: requires an exodus file with:
            #   element_block 1 = domain (HEX8)
            #   node_set 1 = fixed structural face
            #   node_set 2 = scalar Dirichlet face
            #   node_set 3 = thermal Dirichlet face
            # ---------------------------------------------------------------
            TITLE:
              - "3-D SSTI (structure-scalar-thermo) -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Structure_Scalar_Thermo_Interaction"
            IO:
              STDOUTEVERY: <stdout_interval>
              STRUCT_STRESS: "2PK"
              STRUCT_STRAIN: "GL"
              THERM_HEATFLUX: "Initial"
              THERM_TEMPGRAD: "Initial"
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/STRUCTURE:
              OUTPUT_STRUCTURE: true
              DISPLACEMENT: true

            # == Structure =====================================================
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "Statics"
              LINEAR_SOLVER: 1

            # == Scalar Transport ==============================================
            SCALAR TRANSPORT DYNAMIC:
              SOLVERTYPE: "nonlinear"
              VELOCITYFIELD: "Navier_Stokes"
              CONVFORM: "conservative"
              INITIALFIELD: "field_by_condition"
              SKIPINITDER: true
              LINEAR_SOLVER: 1
            SCALAR TRANSPORT DYNAMIC/STABILIZATION:
              STABTYPE: "no_stabilization"

            # == Thermal =======================================================
            THERMAL DYNAMIC:
              DYNAMICTYPE: Statics
              TIMESTEP: <thermal_timestep>
              NUMSTEP: <thermal_num_steps>
              LINEAR_SOLVER: 1
            THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
              OUTPUT_THERMO: true
              TEMPERATURE: true

            # == SSI sub-coupling ==============================================
            SSI CONTROL:
              COUPALGO: ssi_Monolithic
              SCATRATIMINTTYPE: "<scatra_time_integration_type>"
            SSI CONTROL/MONOLITHIC:
              LINEAR_SOLVER: 2
              MATRIXTYPE: "sparse"

            # == TSI sub-coupling ==============================================
            TSI DYNAMIC:
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              TIMESTEP: <timestep>
              ITEMAX: <max_tsi_coupling_iterations>
              RESULTSEVERY: <results_output_interval>

            # == SSTI master coupling ==========================================
            SSTI CONTROL:
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              TIMESTEP: <timestep>
              RESULTSEVERY: <results_output_interval>
              RESTARTEVERY: <restart_interval>
              COUPALGO: "ssti_Monolithic"
            SSTI CONTROL/MONOLITHIC:
              LINEAR_SOLVER: 2

            # == Solvers =======================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "field_solver"
            SOLVER 2:
              SOLVER: "UMFPACK"
              NAME: "monolithic_solver"

            # == Materials =====================================================
            MATERIALS:
              # Thermo-elastic structural material
              - MAT: 1
                MAT_Struct_ThermoStVenantK:
                  YOUNGNUM: 1
                  YOUNG: [<Young_modulus>]
                  NUE: <Poisson_ratio>
                  DENS: <density>
                  THEXPANS: <thermal_expansion_coefficient>
                  INITTEMP: <reference_temperature>
                  THERMOMAT: 2
              # Fourier heat conduction (thermal field)
              - MAT: 2
                MAT_Fourier:
                  CAPA: <volumetric_heat_capacity>
                  CONDUCT:
                    constant: [<thermal_conductivity>]
              # Electrode / scalar transport material
              - MAT: 3
                MAT_electrode:
                  DIFF_PARA_NUM: <num_diffusion_parameters>
                  DIFF_PARA: [<diffusion_coefficient>]
                  COND_PARA_NUM: <num_conductivity_parameters>
                  COND_PARA: [<electronic_conductivity>]
                  C_MAX: <max_concentration>
                  CHI_MAX: <max_stoichiometry>

            # Clone structure mesh -> thermo mesh and scatra mesh
            CLONING MATERIAL MAP:
              - SRC_FIELD: "structure"
                SRC_MAT: 1
                TAR_FIELD: "thermo"
                TAR_MAT: 2
              - SRC_FIELD: "structure"
                SRC_MAT: 1
                TAR_FIELD: "scatra"
                TAR_MAT: 3

            # == Boundary Conditions ===========================================

            # Structural: fixed face
            DESIGN SURF DIRICH CONDITIONS:
              - E: <fixed_face_id>
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]

            # Scalar transport: Dirichlet
            DESIGN SURF TRANSPORT DIRICH CONDITIONS:
              - E: <scalar_dirichlet_face_id>
                NUMDOF: <num_scalar_dofs>
                ONOFF: [<active_scalar_dofs>]
                VAL: [<scalar_boundary_values>]
                FUNCT: [<scalar_time_functions>]

            # Thermal: prescribed temperature
            DESIGN SURF THERMO DIRICH CONDITIONS:
              - E: <thermal_dirichlet_face_id>
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<boundary_temperature>]
                FUNCT: [0]

            # == Geometry ======================================================
            STRUCTURE GEOMETRY:
              FILE: "<mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  SOLIDSCATRA:
                    HEX8:
                      MAT: 1
                      KINEM: <kinematics>
                      TYPE: Undefined

            RESULT DESCRIPTION:
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_displacement>
                  TOLERANCE: <result_tolerance>
              - SCATRA:
                  DIS: "scatra"
                  NODE: <result_node_id>
                  QUANTITY: "phi"
                  VALUE: <expected_concentration>
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

        # Check Young's modulus
        young = params.get("YOUNG")
        if young is not None:
            vals = young if isinstance(young, list) else [young]
            for v in vals:
                try:
                    e = float(v)
                    if e <= 0:
                        issues.append(f"YOUNG must be > 0, got {e}.")
                except (TypeError, ValueError):
                    issues.append(
                        f"YOUNG must be a positive number, got {v!r}."
                    )

        # Check Poisson's ratio
        nue = params.get("NUE")
        if nue is not None:
            try:
                nu = float(nue)
                if nu <= 0 or nu >= 0.5:
                    issues.append(f"NUE must be in (0, 0.5), got {nu}.")
            except (TypeError, ValueError):
                issues.append(
                    f"NUE must be a number in (0, 0.5), got {nue!r}."
                )

        # Check thermal expansion coefficient
        thexpans = params.get("THEXPANS")
        if thexpans is not None:
            try:
                alpha = float(thexpans)
                if alpha <= 0:
                    issues.append(
                        f"THEXPANS must be > 0, got {alpha}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"THEXPANS must be a positive number, got {thexpans!r}."
                )

        # Check heat capacity
        capa = params.get("CAPA")
        if capa is not None:
            try:
                c = float(capa)
                if c <= 0:
                    issues.append(
                        f"CAPA (volumetric heat capacity) must be > 0, "
                        f"got {c}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"CAPA must be a positive number, got {capa!r}."
                )

        # Check max concentration
        c_max = params.get("C_MAX")
        if c_max is not None:
            try:
                cm = float(c_max)
                if cm <= 0:
                    issues.append(
                        f"C_MAX must be > 0, got {cm}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"C_MAX must be a positive number, got {c_max!r}."
                )

        # Check element type
        elem_type = params.get("element_type", "")
        if elem_type and "SOLIDSCATRA" not in str(elem_type).upper():
            issues.append(
                f"SSTI requires SOLIDSCATRA elements (not {elem_type}).  "
                f"Plain SOLID elements do not carry scalar/thermal DOFs."
            )

        return issues
