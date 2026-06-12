"""Generator for structural dynamics physics module (GenAlpha, explicit, damping).

Covers time-dependent structural problems with inertia effects, using
DYNAMICTYPE: GenAlpha (or OneStepTheta, ExplEuler) in the STRUCTURAL
DYNAMIC section of 4C.  Produces validated, working .4C.yaml templates.
"""

from __future__ import annotations

from typing import Any

from .base import BaseGenerator


class StructuralDynamicsGenerator(BaseGenerator):
    """Generator for structural dynamics problems in 4C.

    Supports implicit time integration via Generalised-Alpha (GenAlpha),
    OneStepTheta, and explicit Euler.  Includes Rayleigh damping,
    GenAlpha spectral-radius tuning, and beam-specific Lie-group
    integration.
    """

    module_key = "structural_dynamics"
    display_name = "Structural Dynamics (GenAlpha / Explicit / Damping)"
    problem_type = "Structure"

    # ── Knowledge ─────────────────────────────────────────────────────

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The structural dynamics module solves time-dependent structural "
                "problems where inertia effects are important: impact, vibration, "
                "wave propagation.  The PROBLEM TYPE is 'Structure' (same as "
                "quasi-static), but DYNAMICTYPE is set to a transient integrator "
                "instead of 'Statics'.  The recommended integrator is GenAlpha "
                "(Generalised-Alpha), which provides controllable numerical "
                "dissipation of high-frequency modes via the spectral radius "
                "parameter RHO_INF.  The mass matrix is assembled from element "
                "densities (DENS in the material), which is therefore MANDATORY."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "STRUCTURAL DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
                "STRUCTURE GEOMETRY",
            ],
            "materials": {
                "MAT_Struct_StVenantKirchhoff": {
                    "description": (
                        "Standard structural material.  DENS is CRITICAL for "
                        "dynamics because it determines the mass matrix.  "
                        "Zero density means zero inertia."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus E",
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Poisson's ratio nu",
                            "range": "0 < nu < 0.5",
                        },
                        "DENS": {
                            "description": (
                                "Mass density -- MANDATORY for dynamics.  "
                                "Determines the mass matrix M."
                            ),
                            "range": "> 0  (e.g. steel: 7.85e-9 t/mm^3 in mm-t-s units)",
                        },
                    },
                },
                "MAT_ElastHyper + ELAST_CoupNeoHooke": {
                    "description": (
                        "Neo-Hookean hyperelastic for large-deformation dynamics.  "
                        "DENS / NUMMAT / MATIDS / POLYCONVEX live on the "
                        "MAT_ElastHyper wrapper; YOUNG / NUE live on the "
                        "ELAST_CoupNeoHooke sub-material."
                    ),
                    "parameters": {
                        "YOUNG": {"description": "Young's modulus (sub-material)", "range": "> 0"},
                        "NUE": {"description": "Poisson's ratio (sub-material)", "range": "0 < nu < 0.5"},
                        "DENS": {"description": "Mass density (wrapper)", "range": "> 0"},
                        "NUMMAT": {"description": "Number of sub-materials (wrapper)", "range": "1"},
                        "MATIDS": {"description": "List of sub-material IDs (wrapper)", "range": "[<id>]"},
                        "POLYCONVEX": {
                            "description": "Polyconvexity check flag (wrapper)",
                            "range": "0 | 1",
                        },
                    },
                },
            },
            "time_integration": {
                "DYNAMICTYPE": (
                    "Time integrator selection.  Keyword strings below match the "
                    "4C input parser (src/inpar/4C_inpar_structure.cpp).\n"
                    "Implicit options:\n"
                    "  'GenAlpha' -- Generalised-Alpha (RECOMMENDED for most problems).  "
                    "Implicit, second-order accurate, controllable high-frequency "
                    "damping via RHO_INF.  Best all-round choice.\n"
                    "  'GenAlphaLieGroup' -- Lie-group variant for beam elements "
                    "with rotational DOFs.  Required for BEAM3R/BEAM3K dynamics.\n"
                    "  'OneStepTheta' -- Theta-method (theta=0.5: Newmark, "
                    "theta=1.0: backward Euler).  Simpler but less control "
                    "over numerical dissipation.\n"
                    "Explicit options (matrix-free, CFL-constrained, no global Newton):\n"
                    "  'CentrDiff' -- Central differences.  Second-order accurate, "
                    "the standard choice for explicit solid dynamics (impact, "
                    "wave propagation, high strain rates).\n"
                    "  'AdamsBashforth2' -- Two-step Adams-Bashforth, second-order.\n"
                    "  'AdamsBashforth4' -- Four-step Adams-Bashforth, fourth-order "
                    "(higher accuracy but larger startup cost and stricter stability).\n"
                    "  'ExplicitEuler' -- Forward Euler, first-order.  Only for "
                    "diagnostic / very short transients -- prefer CentrDiff.\n"
                    "All explicit schemes require dt < h/c (CFL).  Whether a "
                    "given plasticity / damage material is wired to the explicit "
                    "path depends on its evaluate() interface -- verify with a "
                    "small benchmark before relying on it."
                ),
                "GenAlpha_parameters": (
                    "Configured in the 'STRUCTURAL DYNAMIC/GENALPHA' sub-section.  "
                    "Key parameters:\n"
                    "  RHO_INF -- Spectral radius at infinite frequency [0, 1].  "
                    "RHO_INF=1.0: no numerical damping (energy-conserving).  "
                    "RHO_INF=0.0: maximum high-frequency damping.  "
                    "Typical: 0.8--0.9 for moderate damping.\n"
                    "  BETA, GAMMA, ALPHA_M, ALPHA_F -- Newmark/GenAlpha "
                    "coefficients.  Usually derived from RHO_INF automatically; "
                    "only override for advanced use."
                ),
                "TIMESTEP": (
                    "Time step size.  CRITICAL: must be small enough to resolve "
                    "the highest relevant frequency in the response.  Rule of "
                    "thumb: dt < T_min / 10 where T_min is the period of the "
                    "highest mode of interest.  For explicit methods dt must "
                    "satisfy the CFL condition (dt < h / c where h is element "
                    "size and c is wave speed)."
                ),
                "NUMSTEP": "Total number of time steps.",
                "MAXTIME": "Maximum simulation time.",
            },
            "damping": {
                "DAMPING": (
                    "Damping model.  Options:\n"
                    "  'None' -- No physical damping (numerical damping from "
                    "GenAlpha RHO_INF < 1 still applies).\n"
                    "  'Rayleigh' -- Classical Rayleigh damping: "
                    "C = alpha_M * M + alpha_K * K.  Set M_DAMP and K_DAMP.\n"
                    "  'Material' -- Damping defined at the material level."
                ),
                "M_DAMP": (
                    "Mass-proportional Rayleigh damping coefficient alpha_M.  "
                    "Damps low-frequency modes.  Typical: 0.0 -- 1.0."
                ),
                "K_DAMP": (
                    "Stiffness-proportional Rayleigh damping coefficient alpha_K.  "
                    "Damps high-frequency modes.  Typical: 1e-5 -- 1e-3.  "
                    "Large K_DAMP can make the problem very stiff."
                ),
            },
            "solver": {
                "small_problems": {
                    "SOLVER": "UMFPACK",
                    "description": "Direct solver, robust for small dynamic problems.",
                },
                "large_problems": {
                    "SOLVER": "Belos",
                    "AZPREC": "MueLu",
                    "description": (
                        "Iterative solver with AMG.  For dynamics the system "
                        "matrix includes mass contributions and is often better "
                        "conditioned than pure stiffness problems."
                    ),
                },
            },
            "pitfalls": [
                (
                    "[Physics] DENS (density) in the MAT material is "
                    "CRITICAL for dynamics. DENS = 0 or missing makes the "
                    "mass matrix zero and the dynamic operator singular. "
                    "Signal: M = 0 makes (M + dt*c1*K) reduce to dt*c1*K — "
                    "for explicit schemes the step-update divides by M and "
                    "blows up; for implicit schemes the inertia term "
                    "disappears and the problem effectively becomes quasi-"
                    "static while the user expected transient. 4C ERROR "
                    "from the Newmark integrator may not flag a literal "
                    "'density zero' message; verify DENS > 0 in the MAT "
                    "block by inspection. (Claim inherited — not yet "
                    "fully empirically verified.)"
                ),
                (
                    "[Numerical] Explicit time stepping (DYNAMICTYPE: "
                    "ExplEuler / CentrDiff) requires CFL: dt < h_min / c "
                    "where c = sqrt(E/rho) is the elastic wave speed. "
                    "Violating CFL gives immediate exponential growth. "
                    "Signal: the displacement norm increases by orders of "
                    "magnitude within ~10 time steps; 4C output shows "
                    "exploding solution-norm without converging the time "
                    "step. (Claim inherited.)"
                ),
                (
                    "[API] For beam elements (BEAM3R, BEAM3K) the recommended "
                    "DYNAMICTYPE is GenAlphaLieGroup with MASSLIN: rotations "
                    "— handles rotational DOFs on SO(3) correctly. Using a "
                    "non-Lie-group scheme on beams accumulates rotation "
                    "interpolation errors. Signal: beam rotation field "
                    "drifts away from rigid-body modes under dynamic loading "
                    "with a non-Lie-group scheme; switching to "
                    "GenAlphaLieGroup recovers rigid-body conservation. "
                    "(Claim inherited.)"
                ),
                (
                    "[Numerical] RHO_INF is the high-frequency dissipation "
                    "parameter for generalised-alpha time integration. "
                    "RHO_INF = 1.0 is energy-conserving but admits spurious "
                    "high-frequency ringing; reduce to 0.8 (or lower) if "
                    "the solution shows unphysical ringing. Signal: VTK "
                    "output shows high-frequency velocity oscillation that "
                    "is not present in the analytic / reference solution; "
                    "reducing RHO_INF damps it out. (Claim inherited.)"
                ),
                (
                    "[Numerical] Rayleigh damping (M_DAMP, K_DAMP in the "
                    "STRUCTURAL DYNAMIC section) is frequency-dependent: "
                    "M_DAMP damps low frequencies, K_DAMP damps high. "
                    "Match damping ratios at two target frequencies "
                    "f1, f2 to solve for the (M_DAMP, K_DAMP) pair. "
                    "Signal: monitoring kinetic energy decay rate of "
                    "different modes shows the (omega_1, omega_2) used in "
                    "the calibration are damped at the target ratio; "
                    "other modes deviate. (Claim inherited.)"
                ),
                (
                    "[API] PREDICT: ConstDisVelAcc is the recommended "
                    "predictor for STRUCTURAL DYNAMIC. PREDICT: TangDis "
                    "(tangent displacement) may cause issues in highly "
                    "dynamic problems. Signal: Newton iteration count "
                    "per time step is higher with TangDis on impact-type "
                    "loads; ConstDisVelAcc gives a flatter iteration "
                    "profile. (Claim inherited.)"
                ),
                (
                    "[Syntax] STRUCTURAL DYNAMIC.DYNAMICTYPE validates "
                    "against an allowed enum at YAML parse — values like "
                    "Statics, GenAlpha, OneStepTheta, ExplEuler, "
                    "CentrDiff, GenAlphaLieGroup. A typo or made-up name "
                    "(e.g. 'TotallyMadeUpScheme') is rejected with "
                    "'PROC 0 ERROR ... Could not match this input' from "
                    "core/io/src/4C_io_input_spec_builders.cpp, with the "
                    "STRUCTURAL DYNAMIC block echoed. Signal: stderr "
                    "contains 'Could not match this input' + 'STRUCTURAL "
                    "DYNAMIC' + 'DYNAMICTYPE'. (Verified empirically "
                    "2026-06-01.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "impact_2d",
                    "description": (
                        "2D impact / sudden-load problem.  A block is fixed on "
                        "one side and a sudden pressure is applied on the "
                        "opposite side.  Uses GenAlpha with RHO_INF: 0.9 to "
                        "damp spurious high-frequency modes."
                    ),
                    "template_variant": "genalpha_2d",
                },
                {
                    "name": "vibration_3d",
                    "description": (
                        "Free vibration of a 3D beam or block after an initial "
                        "displacement.  Uses GenAlpha with RHO_INF: 1.0 "
                        "(energy-conserving) to study natural frequencies."
                    ),
                    "template_variant": "genalpha_2d",
                },
            ],
        }

    # ── Templates ─────────────────────────────────────────────────────

    _TEMPLATES: dict[str, str] = {
        "genalpha_2d": """\
# FORMAT TEMPLATE — all numerical values are placeholders.
TITLE:
  - "Structural dynamics -- 2D impact with GenAlpha time integration"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
IO:
  STRUCT_STRESS: "Cauchy"
  STRUCT_STRAIN: "GL"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: <output_interval_steps>
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
  STRESS_STRAIN: true
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "GenAlpha"
  PREDICT: "ConstDisVelAcc"
  TIMESTEP: <timestep>
  NUMSTEP: <number_of_steps>
  MAXTIME: <end_time>
  TOLDISP: <displacement_tolerance>
  TOLRES: <residual_tolerance>
  MAXITER: <max_iterations>
  DAMPING: "Rayleigh"
  M_DAMP: <mass_damping_coefficient>
  K_DAMP: <stiffness_damping_coefficient>
  LINEAR_SOLVER: 1
STRUCTURAL DYNAMIC/GENALPHA:
  RHO_INF: <spectral_radius_rho_inf>
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct_solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: <Young_modulus>
      NUE: <Poisson_ratio>
      DENS: <density>
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    ENTITY_TYPE: node_set_id
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0.0, 0.0]
    FUNCT: [0, 0]
DESIGN LINE NEUMANN CONDITIONS:
  - E: 2
    ENTITY_TYPE: node_set_id
    NUMDOF: 6
    ONOFF: [1, 0, 0, 0, 0, 0]
    VAL: [<applied_load>, 0.0, 0.0, 0.0, 0.0, 0.0]
    FUNCT: [1, 0, 0, 0, 0, 0]
STRUCTURE GEOMETRY:
  ELEMENT_BLOCKS:
    - ID: 1
      WALL:
        QUAD4:
          MAT: 1
          KINEM: linear
          THICK: <thickness>
          STRESS_STRAIN: plane_strain
  FILE: mesh.e
  SHOW_INFO: detailed_summary
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 1
      QUANTITY: "dispx"
      VALUE: <expected_result_value>
      TOLERANCE: <result_tolerance>
""",
    }

    def get_template(self, variant: str = "default") -> str:
        if variant == "default":
            variant = "genalpha_2d"
        if variant not in self._TEMPLATES:
            available = ", ".join(sorted(self._TEMPLATES))
            raise ValueError(
                f"Unknown template variant {variant!r} for {self.module_key}. "
                f"Available: {available}"
            )
        return self._TEMPLATES[variant]

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "genalpha_2d",
                "description": (
                    "2D structural dynamics with GenAlpha time integration, "
                    "Rayleigh damping, SOLID QUAD4 plane-strain elements, "
                    "St. Venant-Kirchhoff material.  Suitable as starting "
                    "point for impact and vibration problems."
                ),
            },
        ]

    # ── Validation ────────────────────────────────────────────────────

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        errors: list[str] = []

        # Check TIMESTEP
        timestep = params.get("TIMESTEP")
        if timestep is not None:
            try:
                dt = float(timestep)
                if dt <= 0:
                    errors.append(
                        f"TIMESTEP must be > 0 for dynamic analysis, got {dt}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"TIMESTEP must be a positive number, got {timestep!r}."
                )

        # Check DENS -- mandatory for dynamics
        dens = params.get("DENS")
        if dens is not None:
            try:
                d = float(dens)
                if d <= 0:
                    errors.append(
                        f"DENS (density) must be > 0 for structural dynamics, "
                        f"got {d}.  Zero or negative density means zero or "
                        f"invalid mass matrix -- the dynamic solve will fail."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"DENS must be a positive number, got {dens!r}."
                )
        else:
            errors.append(
                "DENS (density) is not specified.  It is MANDATORY for "
                "structural dynamics -- the mass matrix cannot be assembled "
                "without it."
            )

        # Check RHO_INF (GenAlpha spectral radius)
        rho_inf = params.get("RHO_INF")
        if rho_inf is not None:
            try:
                r = float(rho_inf)
                if r < 0 or r > 1:
                    errors.append(
                        f"RHO_INF must be in [0, 1], got {r}.  "
                        f"0 = maximum high-frequency damping, "
                        f"1 = energy-conserving (no numerical damping)."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"RHO_INF must be a number in [0, 1], got {rho_inf!r}."
                )

        # Check GenAlpha sub-parameters if provided
        for coeff_name in ("BETA", "GAMMA", "ALPHA_M", "ALPHA_F"):
            val = params.get(coeff_name)
            if val is not None:
                try:
                    c = float(val)
                    if c < 0 or c > 1:
                        errors.append(
                            f"{coeff_name} should be in [0, 1], got {c}."
                        )
                except (TypeError, ValueError):
                    errors.append(
                        f"{coeff_name} must be a number, got {val!r}."
                    )

        # Check Rayleigh damping coefficients
        m_damp = params.get("M_DAMP")
        if m_damp is not None:
            try:
                md = float(m_damp)
                if md < 0:
                    errors.append(
                        f"M_DAMP (mass-proportional damping) must be >= 0, "
                        f"got {md}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"M_DAMP must be a non-negative number, got {m_damp!r}."
                )

        k_damp = params.get("K_DAMP")
        if k_damp is not None:
            try:
                kd = float(k_damp)
                if kd < 0:
                    errors.append(
                        f"K_DAMP (stiffness-proportional damping) must be >= 0, "
                        f"got {kd}."
                    )
                elif kd > 0.01:
                    errors.append(
                        f"K_DAMP = {kd} is unusually large.  High stiffness-"
                        f"proportional damping over-damps high frequencies and "
                        f"makes the problem very stiff.  Typical range: "
                        f"1e-6 to 1e-3."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"K_DAMP must be a non-negative number, got {k_damp!r}."
                )

        # Check Poisson's ratio if provided
        nue = params.get("NUE")
        if nue is not None:
            try:
                nu = float(nue)
                if nu <= 0 or nu >= 0.5:
                    errors.append(
                        f"NUE (Poisson's ratio) must be in (0, 0.5), got {nu}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"NUE must be a number in (0, 0.5), got {nue!r}."
                )

        # Check Young's modulus if provided
        young = params.get("YOUNG")
        if young is not None:
            try:
                e = float(young)
                if e <= 0:
                    errors.append(
                        f"YOUNG (Young's modulus) must be > 0, got {e}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"YOUNG must be a positive number, got {young!r}."
                )

        # Check DYNAMICTYPE
        dyntype = params.get("DYNAMICTYPE")
        valid_types = {"GenAlpha", "GenAlphaLieGroup", "OneStepTheta", "ExplEuler"}
        if dyntype is not None and dyntype not in valid_types:
            errors.append(
                f"DYNAMICTYPE must be one of {sorted(valid_types)}, "
                f"got {dyntype!r}.  Use 'GenAlpha' for implicit dynamics "
                f"(recommended) or 'ExplEuler' for explicit."
            )

        # Warn about CFL for explicit
        if dyntype == "ExplEuler" and timestep is not None:
            try:
                dt = float(timestep)
                if dt > 1e-6:
                    errors.append(
                        f"ExplEuler with TIMESTEP = {dt}: explicit methods "
                        f"typically require very small time steps to satisfy "
                        f"the CFL condition (dt < h_min / c_wave).  Verify "
                        f"that this time step is small enough for your mesh "
                        f"and material."
                    )
            except (TypeError, ValueError):
                pass  # Already caught above

        return errors
