"""Elastohydrodynamic Lubrication (EHL) generator for 4C.

Covers coupled lubrication + structure problems where the lubricant
pressure deforms the bounding surfaces and the deformation changes the
film geometry (two-way coupling).  The lubrication field solves the
Reynolds equation for pressure, the structural field solves for elastic
deformation, and the two are coupled through the film height (structure
-> lubrication) and pressure loads (lubrication -> structure).
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class EHLGenerator(BaseGenerator):
    """Generator for Elastohydrodynamic Lubrication problems in 4C."""

    module_key = "ehl"
    display_name = "Elastohydrodynamic Lubrication (EHL)"
    problem_type = "Elastohydrodynamic_Lubrication"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Elastohydrodynamic Lubrication (EHL) couples the "
                "Reynolds equation (thin-film lubrication) with "
                "structural mechanics.  The lubricant pressure from "
                "the Reynolds equation is applied as a surface load "
                "on the structural bodies, and the resulting elastic "
                "deformation changes the film thickness, which feeds "
                "back into the Reynolds equation.  This two-way "
                "coupling is essential when the lubricant pressure is "
                "comparable to the elastic modulus of the contact "
                "surfaces (e.g. rolling element bearings, gear tooth "
                "contacts, bio-tribology).  The PROBLEM TYPE is "
                "'Elastohydrodynamic_Lubrication'.  The dynamics "
                "sections include LUBRICATION DYNAMIC, STRUCTURAL "
                "DYNAMIC, and EHL DYNAMIC for the coupling parameters.  "
                "The lubrication mesh represents the 2-D film domain "
                "and the structural mesh represents the 3-D elastic "
                "bodies.  Materials include MAT_lubrication for the "
                "lubricant and a structural material (e.g. "
                "MAT_Struct_StVenantKirchhoff) for the elastic bodies."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "STRUCTURAL DYNAMIC",
                "LUBRICATION DYNAMIC",
                "EHL DYNAMIC",
                "SOLVER 1",
                "SOLVER 2",
                "MATERIALS",
            ],
            "optional_sections": [
                "IO",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/STRUCTURE",
                "CLONING MATERIAL MAP",
                "RESULT DESCRIPTION",
            ],
            "materials": {
                "MAT_lubrication": {
                    "description": (
                        "Lubricant material for the Reynolds equation.  "
                        "May include pressure-dependent viscosity "
                        "(piezoviscous) for EHL."
                    ),
                    "parameters": {
                        "DYNVISCOSITY": {
                            "description": (
                                "Dynamic viscosity of lubricant at "
                                "reference pressure [Pa s]"
                            ),
                            "range": "> 0",
                        },
                        "DENSITY": {
                            "description": "Lubricant density [kg/m^3]",
                            "range": "> 0",
                        },
                    },
                },
                "MAT_Struct_StVenantKirchhoff": {
                    "description": (
                        "Linear elastic material for the structural "
                        "bodies in contact."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus [Pa]",
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Poisson's ratio",
                            "range": "[0, 0.5)",
                        },
                        "DENS": {
                            "description": "Structural density [kg/m^3]",
                            "range": "> 0",
                        },
                    },
                },
            },
            "solver": {
                "lubrication_solver": {
                    "type": "UMFPACK",
                    "notes": (
                        "The Reynolds equation system is small (2-D) "
                        "and well suited for direct solvers."
                    ),
                },
                "structure_solver": {
                    "type": "UMFPACK or Belos",
                    "notes": (
                        "Structural solver for the 3-D elastic bodies.  "
                        "Direct solver for small problems; iterative "
                        "with AMG for large meshes."
                    ),
                },
            },
            "coupling_parameters": {
                "COUPALGO": (
                    "EHL coupling algorithm: 'ehl_monolithic' for "
                    "simultaneous solution of lubrication + structure, "
                    "or 'ehl_partitioned' for staggered iteration."
                ),
                "FILM_HEIGHT_FROM": (
                    "How the film height is computed: 'structure' "
                    "(from structural deformation, standard for EHL) "
                    "or 'function' (prescribed, for testing)."
                ),
            },
            "pitfalls": [
                (
                    "[Mesh] The lubrication mesh (2D) and "
                    "the structural mesh (3D) must be "
                    "geometrically compatible at the "
                    "contact surface, and the DESIGN SURF "
                    "EHL MORTAR COUPLING CONDITIONS 3D "
                    "Slave/Master pair must both be there. "
                    "Signal: none, in either direction — "
                    "this is a silent-wrong failure. Giving "
                    "the Master side a different InterfaceID "
                    "from the Slave is accepted and changes "
                    "nothing (the id is not used to pair "
                    "them), and deleting the Master "
                    "condition outright still runs to "
                    "completion with the contact "
                    "displacement collapsing to ~1e-15. "
                    "Nothing named 4C_ehl_factory.cpp "
                    "exists and 'no matching lubrication "
                    "interface' is never printed; check the "
                    "condition pair by hand. (Corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Numerical] The minimum-film-height "
                    "regularisation for h -> 0 is a real key: "
                    "LUBRICATION DYNAMIC/GAP_OFFSET, whose "
                    "default is 0. Signal: none — with it at "
                    "0 the run does NOT produce NaN and does "
                    "NOT report a singular matrix; it "
                    "completes every time step and hands back "
                    "finite displacements and a finite "
                    "pressure that are simply wrong. Set "
                    "GAP_OFFSET to a physical h_min and "
                    "compare the two runs; near-contact will "
                    "not announce itself. (Corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Numerical] EHL is HIGHLY nonlinear "
                    "due to the pressure-viscosity "
                    "(piezoviscous) effect: mu = "
                    "mu_0 * exp(alpha * p). Signal: too "
                    "large an alpha does not oscillate "
                    "between two states — the Newton "
                    "residual explodes in a single step and "
                    "the PROCESS IS KILLED by SIGFPE "
                    "(floating-point divide-by-zero inside "
                    "LubricationEleCalc::calc_mat_psl). The "
                    "shell reports 128+8 = 136; 4C prints no "
                    "error line, no 'did not converge' and "
                    "no result-test verdict, because "
                    "MPI_Abort is never reached, so a caller "
                    "that only greps for 4C diagnostics sees "
                    "nothing. Under-relaxation cannot help "
                    "by default: MAXOMEGA / MINOMEGA / "
                    "STARTOMEGA live in ELASTO HYDRO "
                    "DYNAMIC/PARTITIONED and are only "
                    "consulted when ELASTO HYDRO "
                    "DYNAMIC/COUPALGO is ehl_IterStagg, "
                    "while the default is ehl_Monolithic — "
                    "adding them to a monolithic run is "
                    "accepted and inert. (Corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Numerical] Pressure-dependent "
                    "viscosity (Barus exp or Roelands) "
                    "dramatically affects convergence, so "
                    "start with constant viscosity "
                    "(alpha = 0) to verify the setup and "
                    "then ramp alpha over several "
                    "pseudo-load-steps. Signal: the alpha = 0 "
                    "and moderately-ramped runs complete "
                    "every time step and reach the result-"
                    "test manager; the run that has gone too "
                    "far is killed by SIGFPE part-way "
                    "through the first step, so the usable "
                    "marker is the ABSENCE of a 'Checking "
                    "results of' line rather than any 'Newton "
                    "diverged' message. Bisect alpha on that "
                    "marker. (Corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Input] The structural load driving the "
                    "contact MUST sit on the CORRECT "
                    "structural face; a coupling surface is "
                    "not a load surface. Signal: none — "
                    "moving the condition onto the mortar "
                    "Slave face is accepted without comment, "
                    "runs every time step, and gives a "
                    "deformation that LOOKS reasonable "
                    "(the driven nodes take exactly the "
                    "prescribed value, so a plot shows a "
                    "deflected body) while the tangential "
                    "displacement collapses to exactly zero "
                    "and the contact-pressure profile is "
                    "wrong. Compare against Hertzian to "
                    "verify face selection. (Corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Numerical] For TRANSIENT EHL the "
                    "squeeze-film (dh/dt) term must be "
                    "switched on, and the key is LUBRICATION "
                    "DYNAMIC/ADD_SQUEEZE_TERM (bool, default "
                    "false). There is NO 'TRANSIENT' key. "
                    "Signal: writing TRANSIENT: true aborts "
                    "at parse time in "
                    "core/io/src/4C_io_input_spec_builders.cpp "
                    "with 'Could not match this input', the "
                    "section echoed back and the offending "
                    "key listed under '[!] The following "
                    "data remains unused:' — before any "
                    "field is built. ADD_SQUEEZE_TERM needs "
                    "the height rate, so it requires the EHL "
                    "height field; a function-prescribed "
                    "height cannot supply it. (Corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Input] Units must be consistent: "
                    "viscosity in Pa s, pressure in Pa, "
                    "lengths in m, Young's modulus in Pa. "
                    "Signal: on a monolithic EHL a viscosity "
                    "unit slip is not a 'silent scaling "
                    "error' — it destroys the linear algebra "
                    "and the process is KILLED by SIGFPE "
                    "(shell status 136) with no 4C error "
                    "line, no MPI_ABORT block and no result-"
                    "test verdict. Where it dies depends on "
                    "the very scaling that is supposed to "
                    "help: with 4C's default INFNORMSCALING "
                    "it is Epetra_CrsMatrix::InvRowSums, "
                    "called from EHL::Monolithic::"
                    "scale_system, that overflows; with "
                    "INFNORMSCALING off the same deck dies "
                    "inside UMFPACK's umfdi_kernel_init. "
                    "Non-dimensionalising pressure by "
                    "p_Hertz and length by the contact "
                    "half-width is the real remedy. "
                    "(Corrected by execution 2026-08-06.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "line_contact_ehl",
                    "description": (
                        "EHL line contact: a cylinder rolling on a "
                        "flat surface with a lubricant film.  The "
                        "classical Hertzian-EHL benchmark.  Tests "
                        "pressure-film-height coupling and elastic "
                        "flattening of the contact zone."
                    ),
                    "template_variant": "ehl_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "ehl_3d",
                "description": (
                    "3-D EHL: lubrication + elastic structure coupling.  "
                    "Reynolds equation on 2-D film mesh, linear elastic "
                    "3-D structure.  UMFPACK solvers."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "ehl_3d") -> str:
        templates = {
            "ehl_3d": self._template_ehl_3d,
        }
        if variant == "default":
            variant = "ehl_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_ehl_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D Elastohydrodynamic Lubrication (EHL)
            #
            # Coupled lubrication (Reynolds equation) + structural mechanics.
            # The lubricant pressure deforms the elastic bodies, which
            # changes the film geometry and feeds back into the Reynolds
            # equation.
            #
            # Mesh: requires:
            #   Lubrication mesh: "lub.e" with
            #     element_block 1 = lubrication film (QUAD4, 2-D)
            #     node_set 1 = inlet boundary (pressure Dirichlet)
            #     node_set 2 = outlet boundary (pressure Dirichlet)
            #   Structure mesh: "structure.e" with
            #     element_block 1 = elastic body (HEX8, 3-D)
            #     node_set 1 = bottom face (fixed)
            #     node_set 2 = contact surface (receives lubricant pressure)
            # ---------------------------------------------------------------
            TITLE:
              - "3-D elastohydrodynamic lubrication -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Elastohydrodynamic_Lubrication"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/STRUCTURE:
              OUTPUT_STRUCTURE: true
              DISPLACEMENT: true

            # == Structure =====================================================
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "Statics"
              TIMESTEP: <structure_timestep>
              NUMSTEP: <structure_num_steps>
              MAXTIME: <structure_max_time>
              LINEAR_SOLVER: 1
              TOLRES: <structure_residual_tolerance>
              TOLDISP: <structure_displacement_tolerance>

            # == Lubrication ===================================================
            LUBRICATION DYNAMIC:
              TIMESTEP: <lubrication_timestep>
              NUMSTEP: <lubrication_num_steps>
              MAXTIME: <lubrication_max_time>
              SOLVERTYPE: "<lubrication_solver_type>"
              LINEAR_SOLVER: 2
              RESULTSEVERY: <results_output_interval>
              SURFACE_VELOCITY: <surface_velocity>

            # == EHL coupling ==================================================
            EHL DYNAMIC:
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              COUPALGO: "<ehl_coupling_algorithm>"
              ITEMAX: <ehl_max_coupling_iterations>
              CONVTOL: <ehl_convergence_tolerance>
              RESULTSEVERY: <results_output_interval>

            # == Solvers =======================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "structure_solver"
            SOLVER 2:
              SOLVER: "UMFPACK"
              NAME: "lubrication_solver"

            # == Materials =====================================================
            MATERIALS:
              # Structural material
              - MAT: 1
                MAT_Struct_StVenantKirchhoff:
                  YOUNG: <Young_modulus>
                  NUE: <Poisson_ratio>
                  DENS: <density>
              # Lubricant material
              - MAT: 2
                MAT_lubrication:
                  DYNVISCOSITY: <lubricant_dynamic_viscosity>
                  DENSITY: <lubricant_density>

            # == Boundary Conditions ===========================================

            # Structure: fixed bottom
            DESIGN SURF DIRICH CONDITIONS:
              - E: <structure_fixed_face_id>
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]

            # Lubrication: pressure BCs
            DESIGN LINE LUBRICATION DIRICH CONDITIONS:
              - E: <lub_inlet_boundary_id>
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<inlet_pressure>]
                FUNCT: [0]
              - E: <lub_outlet_boundary_id>
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<outlet_pressure>]
                FUNCT: [0]

            # == Geometry ======================================================
            STRUCTURE GEOMETRY:
              FILE: "<structure_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  SOLID:
                    HEX8:
                      MAT: 1
                      KINEM: <kinematics>

            LUBRICATION GEOMETRY:
              FILE: "<lubrication_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  LUBRICATION:
                    QUAD4:
                      MAT: 2

            RESULT DESCRIPTION:
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_structure_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_displacement>
                  TOLERANCE: <result_tolerance>
              - LUBRICATION:
                  DIS: "lubrication"
                  NODE: <result_lubrication_node_id>
                  QUANTITY: "pre"
                  VALUE: <expected_pressure>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check lubricant viscosity
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

        # Check lubricant density
        density = params.get("DENSITY") or params.get("lubricant_density")
        if density is not None:
            try:
                rho = float(density)
                if rho <= 0:
                    issues.append(
                        f"Lubricant DENSITY must be > 0, got {rho}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DENSITY must be a positive number, got {density!r}."
                )

        # Check Young's modulus
        young = params.get("YOUNG")
        if young is not None:
            try:
                e = float(young)
                if e <= 0:
                    issues.append(f"YOUNG must be > 0, got {e}.")
            except (TypeError, ValueError):
                issues.append(
                    f"YOUNG must be a positive number, got {young!r}."
                )

        # Check Poisson's ratio
        nue = params.get("NUE")
        if nue is not None:
            try:
                nu = float(nue)
                if nu < 0 or nu >= 0.5:
                    issues.append(
                        f"NUE must be in [0, 0.5), got {nu}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"NUE must be a number in [0, 0.5), got {nue!r}."
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

        # Check coupling algorithm
        coupalgo = params.get("COUPALGO")
        if coupalgo is not None and coupalgo not in (
            "ehl_monolithic", "ehl_partitioned",
        ):
            issues.append(
                f"EHL COUPALGO should be 'ehl_monolithic' or "
                f"'ehl_partitioned', got {coupalgo!r}."
            )

        return issues
