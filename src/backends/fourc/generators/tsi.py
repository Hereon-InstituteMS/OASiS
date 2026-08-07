"""Thermo-Structure Interaction (TSI) generator for 4C.

Covers monolithic and partitioned coupling of thermal and structural fields.
The thermal field solves the heat equation while the structural field solves
the momentum equation with thermal expansion.  The two fields are coupled
through the thermal stress (THR -> STR) and deformation-dependent heat
conduction (STR -> THR).
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class TSIGenerator(BaseGenerator):
    """Generator for Thermo-Structure Interaction problems in 4C."""

    module_key = "tsi"
    display_name = "Thermo-Structure Interaction (TSI)"
    problem_type = "Thermo_Structure_Interaction"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Thermo-Structure Interaction couples a thermal field (heat "
                "equation) with a structural mechanics field (momentum balance "
                "with thermal strain).  The thermal field produces temperature "
                "gradients that cause thermal expansion in the structure, and "
                "the deforming structure can modify the heat conduction path.  "
                "4C supports monolithic (fully coupled, simultaneous solve) and "
                "partitioned (iterative staggered) coupling strategies.  The "
                "PROBLEM TYPE is 'Thermo_Structure_Interaction'.  Three "
                "dynamics sections are required: STRUCTURAL DYNAMIC, "
                "THERMAL DYNAMIC, and TSI DYNAMIC.  The structural mesh uses "
                "SOLIDSCATRA elements (not plain SOLID) to carry the thermal "
                "DOF.  A CLONING MATERIAL MAP maps the structural material to "
                "the thermal material."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "STRUCTURAL DYNAMIC",
                "THERMAL DYNAMIC",
                "TSI DYNAMIC",
                "TSI DYNAMIC/MONOLITHIC",
                "SOLVER 1",
                "SOLVER 2",
                "MATERIALS",
                "CLONING MATERIAL MAP",
            ],
            "optional_sections": [
                "IO",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/STRUCTURE",
                "THERMAL DYNAMIC/RUNTIME VTK OUTPUT",
                "STRUCTURAL DYNAMIC/GENALPHA",
            ],
            "materials": {
                "MAT_Struct_ThermoStVenantK": {
                    "description": (
                        "Thermo-elastic St. Venant-Kirchhoff material.  Extends "
                        "the standard SVK material with thermal expansion "
                        "coefficient and reference temperature.  Links to a "
                        "thermal material via THERMOMAT."
                    ),
                    "parameters": {
                        "YOUNGNUM": {
                            "description": (
                                "Number of Young's modulus entries (typically 1 "
                                "for isotropic)"
                            ),
                            "range": ">= 1",
                        },
                        "YOUNG": {
                            "description": (
                                "Young's modulus array [E] (one entry per "
                                "YOUNGNUM)"
                            ),
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
                            "description": (
                                "Coefficient of thermal expansion alpha_T "
                                "[1/K]"
                            ),
                            "range": "> 0 (typical metals: 1e-6 to 3e-5)",
                        },
                        "INITTEMP": {
                            "description": (
                                "Reference temperature at which thermal strain "
                                "is zero"
                            ),
                            "range": "any (often 0 or 293 K)",
                        },
                        "THERMOMAT": {
                            "description": (
                                "Material ID of the associated thermal material "
                                "(MAT_Fourier)"
                            ),
                            "range": "valid MAT ID",
                        },
                    },
                },
                "MAT_Fourier": {
                    "description": (
                        "Fourier heat conduction material.  Defines volumetric "
                        "heat capacity and thermal conductivity for the "
                        "thermal field."
                    ),
                    "parameters": {
                        "CAPA": {
                            "description": (
                                "Volumetric heat capacity rho * c_p "
                                "[J/(m^3 K)]"
                            ),
                            "range": "> 0",
                        },
                        "CONDUCT": {
                            "description": (
                                "Thermal conductivity k [W/(m K)].  Specified "
                                "as a YAML mapping with 'constant: [value]'."
                            ),
                            "range": "> 0",
                        },
                    },
                },
            },
            "solver": {
                "field_solvers": {
                    "type": "UMFPACK (direct)",
                    "notes": (
                        "Individual field solvers (SOLVER 1) for structure "
                        "and thermal used by the partitioned approach or as "
                        "sub-solvers."
                    ),
                },
                "monolithic_solver": {
                    "type": "Belos with Teko block preconditioner",
                    "notes": (
                        "The monolithic TSI solver (SOLVER 2) uses Belos "
                        "iterative solver with AZPREC: Teko and a "
                        "thermo_solid block preconditioner XML file.  "
                        "For small problems UMFPACK can also be used."
                    ),
                },
            },
            "time_integration": {
                "STRUCTURAL DYNAMIC": (
                    "DYNAMICTYPE: 'Statics' for quasi-static thermo-mechanical "
                    "problems.  'GenAlpha' for transient dynamics with thermal "
                    "coupling."
                ),
                "THERMAL DYNAMIC": (
                    "DYNAMICTYPE: 'Statics' for steady-state thermal field "
                    "within each TSI step.  'OneStepTheta' or 'GenAlpha' for "
                    "transient thermal analysis."
                ),
                "TSI DYNAMIC": (
                    "Controls the overall coupled TSI time stepping.  "
                    "TIMESTEP, NUMSTEP, MAXTIME define the global time loop.  "
                    "ITEMAX sets max coupling iterations per step."
                ),
            },
            "cloning_material_map": {
                "purpose": (
                    "Maps the structural material to the thermal field "
                    "material.  4C clones the structure mesh to create the "
                    "thermo discretisation; this map tells it which material "
                    "to assign.  SRC_FIELD: structure -> TAR_FIELD: thermo."
                ),
            },
            "pitfalls": [
                (
                    "4C has NO 2D TSI elements (the element module is "
                    "solid_scatra_3D_ele; every TSI test in the 4C corpus is "
                    "3D).  Do NOT attempt a 2D thermo-mechanical deck: "
                    "WALL QUAD4 + MAT_Struct_ThermoStVenantK aborts with "
                    "'Invalid type of material law for wall element' "
                    "(4C_w1_mat.cpp), and on current builds SOLID QUAD4 "
                    "aborts with \"Element 'SOLID' does not seem to know "
                    "cell type 'quad4'\".  For a 2D plane-strain problem "
                    "build a pseudo-2D deck from the 'monolithic_3d' "
                    "template: a one-element-thick 3D SOLIDSCATRA HEX8 "
                    "slab with u_z fixed on all nodes (exact plane strain) "
                    "and the temperature field imposed volume-wide via "
                    "DESIGN VOL THERMO DIRICH + a symbolic FUNCT — this "
                    "recipe was verified against the 4C binary 2026-08-01, "
                    "tip displacement within 0.5% of the analytic "
                    "plane-strain thermal-expansion value.  (There is no "
                    "ready-made 'plane_strain_2d' template variant; "
                    "requesting one raises ValueError.)"
                ),
                (
                    "One-way thermo->structure TSI REQUIRES 'TSI DYNAMIC/"
                    "PARTITIONED: COUPVARIABLE: Temperature'.  The 4C "
                    "default is Displacement (structure->thermo): a "
                    "tsi_oneway deck with the default runs to rc=0 but "
                    "produces exactly ZERO displacement — silently wrong "
                    "physics (verified 2026-08-01)."
                ),
                (
                    "Structure elements MUST be SOLIDSCATRA (not plain SOLID) "
                    "to carry the thermal DOF.  Using SOLID elements will "
                    "silently omit the thermal coupling."
                ),
                (
                    "CLONING MATERIAL MAP is required: it maps "
                    "SRC_FIELD: structure, SRC_MAT: <struct_mat_id> to "
                    "TAR_FIELD: thermo, TAR_MAT: <thermo_mat_id>."
                ),
                (
                    "The thermo-elastic material (MAT_Struct_ThermoStVenantK) "
                    "must reference the thermal material via THERMOMAT.  If "
                    "this link is missing, thermal strains are not computed."
                ),
                (
                    "For monolithic TSI, the monolithic solver must be "
                    "specified in TSI DYNAMIC/MONOLITHIC with LINEAR_SOLVER "
                    "pointing to a solver that can handle the coupled block "
                    "system."
                ),
                (
                    "Thermal Neumann BCs use 'DESIGN SURF THERMO NEUMANN "
                    "CONDITIONS' (not regular NEUMANN).  Thermal Dirichlet "
                    "BCs use 'DESIGN SURF THERMO DIRICH CONDITIONS'."
                ),
                (
                    "INITTEMP in the structural material must match the "
                    "actual initial temperature field; otherwise a spurious "
                    "thermal strain offset appears at t=0."
                ),
                (
                    "IO section: use THERM_HEATFLUX and THERM_TEMPGRAD to "
                    "request thermal output quantities."
                ),
            ],
            "typical_experiments": [
                {
                    "name": "heated_bar_3d",
                    "description": (
                        "A bar heated from one end with the other end held at "
                        "reference temperature.  The temperature gradient "
                        "causes axial elongation.  Tests one-way and two-way "
                        "thermo-mechanical coupling.  Uses monolithic TSI "
                        "with MAT_Struct_ThermoStVenantK + MAT_Fourier."
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
                    "3-D monolithic TSI: heated bar with thermal expansion.  "
                    "SOLIDSCATRA HEX8 elements, MAT_Struct_ThermoStVenantK "
                    "material, MAT_Fourier thermal material, monolithic "
                    "Belos/Teko solver."
                ),
            },
        ]

    # NOTE: 'oneway_3d' and 'plane_strain_2d' were advertised here but no
    # template was ever wired into get_template(), so asking for either
    # raised ValueError.  Both recipes are fully documented in the pitfalls
    # above and are derived from 'monolithic_3d':
    #   oneway_3d      - TSI DYNAMIC: COUPALGO tsi_oneway, plus
    #                    TSI DYNAMIC/PARTITIONED: COUPVARIABLE: Temperature
    #                    (the default Displacement silently gives zero
    #                    displacement).
    #   plane_strain_2d - one-element-thick SOLIDSCATRA HEX8 slab, u_z
    #                    fixed on all nodes, temperature imposed volume-wide
    #                    via DESIGN VOL THERMO DIRICH + a symbolic FUNCT.
    # They are not listed until a template exists and has been verified by
    # execution; advertising a variant that raises is worse than omitting it.

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
            # 3-D Monolithic Thermo-Structure Interaction
            #
            # A body is subjected to a thermal load (heat flux or prescribed
            # temperature).  The resulting temperature field drives thermal
            # expansion in the structural field.
            #
            # Elements: SOLIDSCATRA HEX8 (carries both displacement and
            #           thermal DOFs via mesh cloning)
            #
            # Mesh: requires an exodus file with:
            #   element_block 1 = structure (HEX8)
            #   node_set 1 = fixed face (structural Dirichlet + thermal Dirichlet)
            #   node_set 2 = heated face (thermal Neumann or Dirichlet)
            # ---------------------------------------------------------------
            TITLE:
              - "3-D thermo-structure interaction -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Thermo_Structure_Interaction"
            IO:
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
              TIMESTEP: <structure_timestep>
              MAXTIME: <structure_max_time>
              LINEAR_SOLVER: 1

            # == Thermal =======================================================
            THERMAL DYNAMIC:
              DYNAMICTYPE: Statics
              TIMESTEP: <thermal_timestep>
              NUMSTEP: <thermal_num_steps>
              LINEAR_SOLVER: 1
            THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
              OUTPUT_THERMO: true
              TEMPERATURE: true
              TEMPERATURE_RATE: true

            # == TSI coupling ==================================================
            TSI DYNAMIC:
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              TIMESTEP: <timestep>
              ITEMAX: <max_coupling_iterations>
              RESULTSEVERY: <results_output_interval>
            TSI DYNAMIC/MONOLITHIC:
              NORM_RESF: "Rel"
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
              # Fourier heat conduction material (for thermal field)
              - MAT: 2
                MAT_Fourier:
                  CAPA: <volumetric_heat_capacity>
                  CONDUCT:
                    constant: [<thermal_conductivity>]

            # Clone structure mesh -> thermo mesh
            CLONING MATERIAL MAP:
              - SRC_FIELD: "structure"
                SRC_MAT: 1
                TAR_FIELD: "thermo"
                TAR_MAT: 2

            # == Boundary Conditions ===========================================

            # Structural: fixed face
            DESIGN SURF DIRICH CONDITIONS:
              - E: <fixed_face_id>
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]

            # Thermal: prescribed temperature on fixed face
            DESIGN SURF THERMO DIRICH CONDITIONS:
              - E: <cold_face_id>
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<cold_face_temperature>]
                FUNCT: [0]

            # Thermal: heat flux on heated face
            DESIGN SURF THERMO NEUMANN CONDITIONS:
              - E: <heated_face_id>
                NUMDOF: 6
                ONOFF: [1, 0, 0, 0, 0, 0]
                VAL: [<heat_flux_value>, 0, 0, 0, 0, 0]
                FUNCT: [<heat_flux_time_function>, 0, 0, 0, 0, 0]

            # Time-dependent heat flux ramp
            FUNCT1:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<heat_flux_ramp_expression>"

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
              - THERMAL:
                  DIS: "thermo"
                  NODE: <result_node_id>
                  QUANTITY: "temp"
                  VALUE: <expected_temperature>
                  TOLERANCE: <result_tolerance>
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_displacement>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check thermal expansion coefficient
        thexpans = params.get("THEXPANS")
        if thexpans is not None:
            try:
                alpha = float(thexpans)
                if alpha <= 0:
                    issues.append(
                        f"THEXPANS (thermal expansion coeff) must be > 0, "
                        f"got {alpha}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"THEXPANS must be a positive number, got {thexpans!r}."
                )

        # Check Young's modulus
        young = params.get("YOUNG")
        if young is not None:
            vals = young if isinstance(young, list) else [young]
            for v in vals:
                try:
                    e = float(v)
                    if e <= 0:
                        issues.append(
                            f"YOUNG must be > 0, got {e}."
                        )
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
                    issues.append(
                        f"NUE must be in (0, 0.5), got {nu}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"NUE must be a number in (0, 0.5), got {nue!r}."
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
                                f"CONDUCT values must be positive numbers, "
                                f"got {v!r}."
                            )

        # Check CLONING MATERIAL MAP presence
        has_cloning = params.get("has_cloning_material_map")
        if has_cloning is not None and not has_cloning:
            issues.append(
                "CLONING MATERIAL MAP is required for TSI.  It maps the "
                "structural material to the thermal material."
            )

        # Check element type
        elem_type = params.get("element_type", "")
        if elem_type and "SOLIDSCATRA" not in str(elem_type).upper():
            et = str(elem_type).upper()
            if "WALL" in et or "QUAD" in et or "TRI" in et:
                issues.append(
                    f"TSI cannot use 2D elements ({elem_type}): 4C has no "
                    f"2D TSI elements, and MAT_Struct_ThermoStVenantK is "
                    f"rejected by WALL ('Invalid type of material law for "
                    f"wall element').  For 2D plane-strain "
                    f"thermo-mechanics build a pseudo-2D deck from the "
                    f"'monolithic_3d' template: a one-element-thick "
                    f"SOLIDSCATRA HEX8 slab with u_z fixed everywhere."
                )
            else:
                issues.append(
                    f"TSI requires SOLIDSCATRA elements (not {elem_type}).  "
                    f"Plain SOLID elements do not carry the thermal DOF."
                )

        return issues
