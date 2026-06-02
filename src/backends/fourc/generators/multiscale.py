"""Multiscale (FE-squared) generator for 4C.

Covers computational homogenisation using nested FE simulations (FE^2).
A macroscopic structural problem is solved with standard finite elements,
but the constitutive response at each Gauss point is computed by solving
a microscale boundary value problem (Representative Volume Element, RVE)
instead of using a closed-form material law.  This enables capturing
complex microstructural effects (heterogeneity, damage, plasticity) at
the macroscale without explicit homogenisation assumptions.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class MultiscaleGenerator(BaseGenerator):
    """Generator for FE^2 nested multiscale problems in 4C."""

    module_key = "multiscale"
    display_name = "Multiscale (FE-squared Computational Homogenisation)"
    problem_type = "Structure"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The FE^2 (FE-squared) multiscale module performs "
                "computational homogenisation by nesting a microscale "
                "FE simulation within each macroscale Gauss point.  "
                "At the macro level, a standard structural problem is "
                "solved.  At each integration point, instead of "
                "evaluating a closed-form constitutive law, a "
                "Representative Volume Element (RVE) boundary value "
                "problem is solved.  The macro deformation gradient is "
                "imposed on the RVE via boundary conditions, and the "
                "homogenised stress and tangent are returned to the "
                "macro solver.  The PROBLEM TYPE is 'Structure' with a "
                "special MAT_Struct_Multiscale material that references "
                "a micro-scale input file.  The dynamics section is "
                "STRUCTURAL DYNAMIC.  The micro-scale input file is a "
                "complete 4C input file describing the RVE (geometry, "
                "materials, solver, boundary conditions).  The macro "
                "material uses MAT_Struct_Multiscale which points to "
                "the micro input file and specifies the microscale "
                "solver ID."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "STRUCTURAL DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
                "STRUCTURE GEOMETRY",
            ],
            "optional_sections": [
                "STRUCTURAL DYNAMIC/GENALPHA",
                "IO",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/STRUCTURE",
                "RESULT DESCRIPTION",
            ],
            "materials": {
                "MAT_Struct_Multiscale": {
                    "description": (
                        "Macro-scale material that triggers FE^2 "
                        "homogenisation.  Each Gauss point evaluates the "
                        "constitutive response by solving a microscale "
                        "RVE problem defined in a separate input file."
                    ),
                    "parameters": {
                        "MICRO_INPUT_FILE": {
                            "description": (
                                "Path to the micro-scale 4C input file "
                                "that defines the RVE problem."
                            ),
                            "range": "valid file path",
                        },
                        "MICRO_SOLVER_ID": {
                            "description": (
                                "SOLVER N ID used for the micro-scale "
                                "problem.  Must reference a valid SOLVER "
                                "section in the MACRO input file."
                            ),
                            "range": ">= 1",
                        },
                        "DENS": {
                            "description": (
                                "Homogenised macroscopic density [kg/m^3]."
                            ),
                            "range": "> 0",
                        },
                    },
                },
                "Micro-scale materials": {
                    "description": (
                        "Materials for the RVE are defined in the micro "
                        "input file.  Any standard 4C material can be "
                        "used (MAT_ElastHyper, MAT_Struct_StVenantKirchhoff, "
                        "damage models, plasticity models, etc.)."
                    ),
                },
            },
            "solver": {
                "macro_solver": {
                    "type": "UMFPACK or Belos",
                    "notes": (
                        "Macro-scale structural solver.  Direct solver "
                        "works for small problems; iterative needed for "
                        "large macro meshes."
                    ),
                },
                "micro_solver": {
                    "type": "UMFPACK",
                    "notes": (
                        "Micro-scale RVE solver.  Direct solver is "
                        "recommended since each RVE is typically small.  "
                        "Must be defined in the MACRO input file via "
                        "SOLVER N (referenced by MICRO_SOLVER_ID)."
                    ),
                },
            },
            "rve_setup": {
                "geometry": (
                    "The RVE is typically a unit cell of the "
                    "microstructure.  It must be a cube/rectangle with "
                    "periodic boundary conditions (PBC) to ensure "
                    "proper homogenisation."
                ),
                "boundary_conditions": (
                    "Periodic boundary conditions (PBC) are standard "
                    "for RVEs.  The macro deformation gradient is "
                    "imposed via tied DOFs on opposite faces."
                ),
                "size": (
                    "The RVE must be statistically representative of "
                    "the microstructure.  Too small -> artificial size "
                    "effects.  Too large -> excessive computation cost "
                    "(each macro Gauss point solves one RVE)."
                ),
            },
            "pitfalls": [
                (
                    "[Performance] FE^2 is extremely expensive: each "
                    "macro Gauss point requires solving a full micro "
                    "FE problem.  For a macro mesh with N elements "
                    "and G Gauss points per element, N*G micro "
                    "problems are solved per macro Newton iteration. "
                    " Use coarse macro meshes and efficient micro "
                    "solvers. Signal: wall-clock per macro Newton "
                    "iteration grows linearly with N*G; profile log "
                    "shows >95% of time in MicroSolver::Solve; for a "
                    "10x10x10 macro mesh expect minutes per "
                    "iteration even with a trivial micro RVE. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Input] The micro input file must be a complete, "
                    "valid 4C input file with its own geometry, "
                    "materials, and boundary conditions.  It is NOT "
                    "merged with the macro input file. Signal: 4C "
                    "aborts during multiscale-init with `failed to "
                    "load micro input file X` or `micro input file "
                    "missing MATERIALS section`. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] MICRO_SOLVER_ID in MAT_Struct_Multiscale "
                    "must reference a SOLVER N section defined in "
                    "the MACRO input file (not the micro file).  "
                    "This solver is used for all micro RVE solves. "
                    "Signal: parser warns `MICRO_SOLVER_ID X not "
                    "found among macro SOLVER definitions` or runtime "
                    "abort `null pointer to micro Belos solver`. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] The RVE boundary conditions must be "
                    "periodic for standard computational "
                    "homogenisation.  Non-periodic BCs (Dirichlet or "
                    "Neumann) lead to over-stiff or over-compliant "
                    "homogenised response. Signal: homogenised "
                    "tangent C* differs from a reference simulation "
                    "with periodic BCs by ~10-20% in the off-"
                    "diagonal entries; uniaxial-tension homogenised "
                    "stress/strain curve sits ~10-30% above a "
                    "periodic reference (Dirichlet RVE over-"
                    "constrains). (Audit 2026-06-02.)"
                ),
                (
                    "[Input] Macro PROBLEM TYPE is "
                    "'Structure' (NOT a dedicated multiscale "
                    "type). The multiscale behaviour is "
                    "activated purely through the "
                    "MAT_Struct_Multiscale material. Signal: "
                    "writing PROBLEMTYPE: 'Multiscale' "
                    "raises 'unknown problem type' — there "
                    "is no such enum. Use PROBLEMTYPE: "
                    "Structure with at least one element "
                    "assigned a MAT_Struct_Multiscale "
                    "material referencing a micro input "
                    "file. (Audit 2026-06-02.)"
                ),
                (
                    "[Performance] MPI parallelism interacts "
                    "with multiscale: macro problem "
                    "distributed across processors, each "
                    "processor's Gauss points solve their "
                    "RVEs INDEPENDENTLY. Signal: imbalanced "
                    "macro decomposition (RVEs differ in "
                    "solve cost by 5-10x) gives wall-clock "
                    "dominated by the slowest rank's RVE "
                    "queue — 4C's load-balancer cannot "
                    "redistribute mid-step. Pre-balance "
                    "the macro mesh based on expected RVE "
                    "complexity. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Convergence at the macro level "
                    "depends on the quality of the micro tangent.  "
                    "Numerical differentiation of the micro response "
                    "can produce inaccurate tangents; algorithmic "
                    "tangent from the micro solver is preferred. "
                    "Signal: macro NOX residual halves slowly "
                    "(linear-rate, not quadratic Newton) and "
                    "MAXITER is hit on every macro step; switching "
                    "MICRO_TANGENT to ALGORITHMIC restores quadratic "
                    "convergence. (Audit 2026-06-02.)"
                ),
                (
                    "[Output] Macro-scale results show "
                    "HOMOGENISED stress/strain. Micro-scale "
                    "fields (damage, plasticity) are NOT "
                    "automatically output. Signal: opening "
                    "the macro IO/RUNTIME VTK OUTPUT "
                    "STRUCTURE VTU in ParaView shows "
                    "smooth homogenised stress from "
                    "MAT_Multiscale; the micro-scale "
                    "fluctuations (e.g. damage localisation "
                    "in one RVE) are invisible in the macro "
                    "output. Use the MULTISCALE micro-"
                    "output writer (separate VTU per Gauss "
                    "point) or post-process selected RVEs "
                    "manually. (Audit 2026-06-02.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "rve_composite_3d",
                    "description": (
                        "A 3-D composite structure (fiber-reinforced "
                        "matrix) where the micro RVE contains a fiber "
                        "inclusion in a matrix.  The macro problem is a "
                        "tensile test.  Tests the full FE^2 loop: "
                        "macro deformation -> RVE solve -> homogenised "
                        "stress/tangent."
                    ),
                    "template_variant": "fe2_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "fe2_3d",
                "description": (
                    "3-D FE^2 multiscale: macro structural problem "
                    "with MAT_Struct_Multiscale material referencing "
                    "a micro RVE input file.  SOLID HEX8 elements, "
                    "UMFPACK solvers for both macro and micro."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "fe2_3d") -> str:
        templates = {
            "fe2_3d": self._template_fe2_3d,
        }
        if variant == "default":
            variant = "fe2_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_fe2_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D FE^2 Multiscale (Macro Input File)
            #
            # Macro-scale structural problem where each Gauss point
            # evaluates its constitutive response by solving a micro-scale
            # RVE (Representative Volume Element) problem.
            #
            # This file defines the MACRO problem.  A separate file defines
            # the MICRO (RVE) problem.
            #
            # Macro mesh: "macro.e" with
            #   element_block 1 = macro structure (HEX8)
            #   node_set 1 = fixed face
            #   node_set 2 = loaded face
            # ---------------------------------------------------------------
            TITLE:
              - "3-D FE^2 multiscale (macro) -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Structure"
            IO:
              STDOUTEVERY: <stdout_interval>
              STRUCT_STRESS: "Cauchy"
              STRUCT_STRAIN: "GL"
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/STRUCTURE:
              OUTPUT_STRUCTURE: true
              DISPLACEMENT: true

            # == Structural dynamics (macro) ===================================
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "Statics"
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              LINEAR_SOLVER: 1
              PREDICT: "ConstDisVelAcc"
              TOLRES: <macro_residual_tolerance>
              TOLDISP: <macro_displacement_tolerance>
              RESULTSEVERY: <results_output_interval>

            # == Solvers (both macro AND micro solvers in this file) ===========
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "macro_solver"
            SOLVER 2:
              SOLVER: "UMFPACK"
              NAME: "micro_rve_solver"

            # == Materials =====================================================
            MATERIALS:
              # Multiscale material: triggers FE^2
              - MAT: 1
                MAT_Struct_Multiscale:
                  MICRO_INPUT_FILE: "<micro_input_file_path>"
                  MICRO_SOLVER_ID: <micro_solver_id>
                  DENS: <homogenised_density>

            # == Boundary Conditions ===========================================

            # Fixed face
            DESIGN SURF DIRICH CONDITIONS:
              - E: <fixed_face_id>
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]

            # Load: prescribed displacement on loaded face
            DESIGN SURF DIRICH CONDITIONS:
              - E: <loaded_face_id>
                NUMDOF: 3
                ONOFF: [<dof1_fix>, <dof2_fix>, <dof3_fix>]
                VAL: [<prescribed_disp_1>, <prescribed_disp_2>, <prescribed_disp_3>]
                FUNCT: [<load_ramp_function>, <load_ramp_function>, <load_ramp_function>]

            # Load ramp
            FUNCT<load_ramp_function>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<load_ramp_expression>"

            # == Geometry ======================================================
            STRUCTURE GEOMETRY:
              FILE: "<macro_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  SOLID:
                    HEX8:
                      MAT: 1
                      KINEM: <kinematics>

            RESULT DESCRIPTION:
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_displacement>
                  TOLERANCE: <result_tolerance>

            # ---------------------------------------------------------------
            # NOTE: The micro-scale RVE input file is SEPARATE.
            # It should define:
            #   - PROBLEM TYPE: Structure
            #   - RVE geometry (unit cell with periodic mesh)
            #   - Micro-scale materials (e.g. fiber + matrix)
            #   - Periodic boundary conditions
            #   - Its own SOLVER section
            # The micro file path is specified in MICRO_INPUT_FILE above.
            # ---------------------------------------------------------------
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check micro input file
        micro_file = params.get("MICRO_INPUT_FILE")
        if micro_file is not None:
            if not micro_file or micro_file == "":
                issues.append(
                    "MICRO_INPUT_FILE must be a non-empty file path."
                )

        # Check micro solver ID
        micro_solver = params.get("MICRO_SOLVER_ID")
        if micro_solver is not None:
            try:
                sid = int(micro_solver)
                if sid < 1:
                    issues.append(
                        f"MICRO_SOLVER_ID must be >= 1, got {sid}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"MICRO_SOLVER_ID must be a positive integer, "
                    f"got {micro_solver!r}."
                )

        # Check density
        density = params.get("DENS") or params.get("homogenised_density")
        if density is not None:
            try:
                rho = float(density)
                if rho <= 0:
                    issues.append(
                        f"DENS (homogenised density) must be > 0, "
                        f"got {rho}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DENS must be a positive number, got {density!r}."
                )

        # Check tolerances
        for tol_key in ("TOLRES", "TOLDISP",
                         "macro_residual_tolerance",
                         "macro_displacement_tolerance"):
            tol = params.get(tol_key)
            if tol is not None:
                try:
                    t = float(tol)
                    if t <= 0:
                        issues.append(
                            f"{tol_key} must be > 0, got {t}."
                        )
                except (TypeError, ValueError):
                    issues.append(
                        f"{tol_key} must be a positive number, "
                        f"got {tol!r}."
                    )

        return issues
