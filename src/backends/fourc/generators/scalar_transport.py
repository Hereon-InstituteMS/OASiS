"""Generator for scalar transport physics module (Poisson, heat conduction, advection-diffusion).

Covers stationary and transient scalar transport problems using the
SCALAR TRANSPORT DYNAMIC section in 4C.  Produces validated, working
.4C.yaml templates that an LLM can adapt to specific user requests.
"""

from __future__ import annotations

from typing import Any

from .base import BaseGenerator


class ScalarTransportGenerator(BaseGenerator):
    """Generator for scalar transport problems in 4C.

    Supports Poisson equation, steady-state / transient heat conduction,
    and advection-diffusion with prescribed velocity fields.
    """

    module_key = "scalar_transport"
    display_name = "Scalar Transport (Poisson / Heat / Advection-Diffusion)"
    problem_type = "Scalar_Transport"

    # ── Knowledge ─────────────────────────────────────────────────────

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The scalar transport module solves the generic advection-diffusion "
                "equation  d(phi)/dt + u . grad(phi) - div(kappa * grad(phi)) = f  "
                "for a scalar field phi.  Special cases include the Poisson equation "
                "(stationary, zero velocity), transient heat conduction, and "
                "advection-dominated transport with stabilisation (SUPG).  "
                "The PROBLEM TYPE is 'Scalar_Transport' and the dynamics section "
                "is 'SCALAR TRANSPORT DYNAMIC' (NOT 'SCATRA DYNAMIC').  "
                "Geometry is specified in 'TRANSPORT GEOMETRY' and elements use "
                "the TRANSP category."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "SCALAR TRANSPORT DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
                "TRANSPORT GEOMETRY",
            ],
            "materials": {
                "MAT_scatra": {
                    "description": (
                        "Standard scalar transport material.  Defines the "
                        "diffusivity of the transported quantity."
                    ),
                    "parameters": {
                        "DIFFUSIVITY": {
                            "description": "Isotropic diffusion coefficient kappa",
                            "range": "> 0  (typical: 0.01 -- 100 depending on units)",
                        },
                    },
                },
                "MAT_Fourier": {
                    "description": (
                        "Fourier heat-conduction material used with PROBLEMTYPE "
                        "Thermo (alternative formulation).  Can also be used for "
                        "heat conduction within the scalar transport framework when "
                        "the thermal capacity and conductivity are needed."
                    ),
                    "parameters": {
                        "CAPA": {
                            "description": "Volumetric heat capacity (rho * c_p)",
                            "range": "> 0",
                        },
                        "CONDUCT": {
                            "description": (
                                "Thermal conductivity.  Specified as a YAML mapping "
                                "with 'constant: [value]'."
                            ),
                            "range": "> 0",
                        },
                    },
                },
            },
            "time_integration": {
                "TIMEINTEGR": (
                    "Time integration scheme.  Options: "
                    "'Stationary' (steady-state solve, single step), "
                    "'BDF2' (second-order backward differentiation, robust for stiff problems), "
                    "'OneStepTheta' (theta-method; theta=1.0 gives implicit Euler, "
                    "theta=0.5 gives Crank-Nicolson)."
                ),
                "SOLVERTYPE": (
                    "Use 'linear_full' for linear problems (Poisson, linear heat). "
                    "Use 'nonlinear' only when the equation contains nonlinear terms "
                    "(e.g. radiation, concentration-dependent diffusivity)."
                ),
                "VELOCITYFIELD": (
                    "Velocity field for advective transport.  "
                    "'zero' -- pure diffusion (Poisson / heat conduction).  "
                    "'function' -- prescribed via VELFUNCNO referencing a FUNCT section.  "
                    "'Navier_Stokes' -- coupled with a fluid solve."
                ),
                "TIMESTEP": "Time step size (only relevant for transient problems).",
                "NUMSTEP": "Number of time steps.",
                "MAXTIME": "Maximum simulation time.",
            },
            "solver": {
                "small_problems": {
                    "SOLVER": "UMFPACK",
                    "description": (
                        "Direct solver, robust for problems up to ~50k DOFs.  "
                        "No preconditioner needed."
                    ),
                },
                "large_problems": {
                    "SOLVER": "Belos",
                    "AZPREC": "MueLu",
                    "description": (
                        "Iterative Krylov solver with algebraic multigrid "
                        "preconditioner.  Scalable to millions of DOFs."
                    ),
                },
            },
            "pitfalls": [
                (
                    "[Input] The dynamics section name is 'SCALAR TRANSPORT "
                    "DYNAMIC', NOT 'SCATRA DYNAMIC'.  Using the wrong name "
                    "silently falls back to defaults and produces wrong "
                    "results. Signal: simulation completes but result fields "
                    "are uniformly zero / equal to the initial condition, "
                    "or 4C reports `unknown section: SCATRA DYNAMIC` in the "
                    "input-parser banner. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] VELOCITYFIELD must be 'zero' (not omitted) "
                    "for pure diffusion.  If omitted, 4C may assume a "
                    "velocity field from a coupled problem that does not "
                    "exist, leading to errors. Signal: runtime error "
                    "`requested velocity field not found` or "
                    "`Vector velnp uninitialized`. (Audit 2026-06-02.)"
                ),
                (
                    "[Output] VTK output for scalar transport uses "
                    "'SCALAR TRANSPORT DYNAMIC/RUNTIME VTK OUTPUT' with "
                    "'OUTPUT_SCATRA: true' and 'PHI: true'.  Do NOT use "
                    "'IO/RUNTIME VTK OUTPUT/SCATRA' -- that path is "
                    "invalid. Signal: simulation completes successfully "
                    "but no scatra-*.vtu files appear under the output "
                    "directory; visualize('list') shows only structure "
                    "fields. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] For Exodus-based meshes the geometry section "
                    "is 'TRANSPORT GEOMETRY' with ELEMENT_BLOCKS using "
                    "the TRANSP element category.  Do not use STRUCTURE "
                    "GEOMETRY. Signal: 4C aborts with `expected element "
                    "category TRANSP but got SOLID` or similar mismatch "
                    "during the geometry-parsing phase. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] NUMDOF for scalar transport boundary "
                    "conditions is 1 (single scalar field).  ONOFF, VAL, "
                    "and FUNCT arrays must each have exactly one entry. "
                    "Signal: input parser aborts with `array size "
                    "mismatch` or `expected NUMDOF entries`, citing the "
                    "DESIGN ... DIRICH CONDITIONS block. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Numerical] When using BDF2 time "
                    "integration, the first step is "
                    "AUTOMATICALLY handled with a lower-order "
                    "scheme (backward Euler) — no special "
                    "start-up needed. Signal: a manual two-"
                    "step BDF2 start with non-physical IC "
                    "guesses doubles the first-step error "
                    "vs default 4C handling; rely on the "
                    "built-in start-up. (Audit "
                    "2026-06-02.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "poisson_2d",
                    "description": (
                        "Steady-state Poisson equation on a 2D square domain "
                        "[0,1]x[0,1] with homogeneous Dirichlet BCs on all sides "
                        "and a constant source term.  Uses TIMEINTEGR: Stationary, "
                        "VELOCITYFIELD: zero, MAT_scatra with DIFFUSIVITY: 1.0."
                    ),
                    "template_variant": "poisson_2d",
                },
                {
                    "name": "heat_transient_2d",
                    "description": (
                        "Transient heat conduction on a 2D plate.  One edge is "
                        "heated (Dirichlet T=100), opposite edge held at T=0.  "
                        "Uses TIMEINTEGR: BDF2, MAT_scatra with DIFFUSIVITY "
                        "representing thermal diffusivity alpha = k/(rho*c_p)."
                    ),
                    "template_variant": "heat_transient_2d",
                },
            ],
        }

    # ── Templates ─────────────────────────────────────────────────────

    _TEMPLATES: dict[str, str] = {
        "poisson_2d": """\
# FORMAT TEMPLATE — all numerical values are placeholders.
TITLE:
  - "Poisson equation on unit square -- stationary diffusion"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: <timestep>
  MAXTIME: <end_time>
  NUMSTEP: <number_of_steps>
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct_solver"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: <diffusivity>
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    ENTITY_TYPE: node_set_id
    NUMDOF: 1
    ONOFF: [1]
    VAL: [<boundary_value_edge1>]
    FUNCT: [0]
  - E: 2
    ENTITY_TYPE: node_set_id
    NUMDOF: 1
    ONOFF: [1]
    VAL: [<boundary_value_edge2>]
    FUNCT: [0]
  - E: 3
    ENTITY_TYPE: node_set_id
    NUMDOF: 1
    ONOFF: [1]
    VAL: [<boundary_value_edge3>]
    FUNCT: [0]
  - E: 4
    ENTITY_TYPE: node_set_id
    NUMDOF: 1
    ONOFF: [1]
    VAL: [<boundary_value_edge4>]
    FUNCT: [0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 5
    ENTITY_TYPE: node_set_id
    NUMDOF: 1
    ONOFF: [1]
    VAL: [<source_term_value>]
    FUNCT: [0]
TRANSPORT GEOMETRY:
  ELEMENT_BLOCKS:
    - ID: 1
      TRANSP:
        QUAD4:
          MAT: 1
          TYPE: Std
  FILE: mesh.e
  SHOW_INFO: detailed_summary
RESULT DESCRIPTION:
  - SCATRA:
      DIS: "scatra"
      NODE: 1
      QUANTITY: "phi"
      VALUE: <expected_result_value>
      TOLERANCE: <result_tolerance>
""",
        "heat_transient_2d": """\
# FORMAT TEMPLATE — all numerical values are placeholders.
TITLE:
  - "Transient heat conduction on 2D plate -- BDF2 time integration"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "BDF2"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: <timestep>
  MAXTIME: <end_time>
  NUMSTEP: <number_of_steps>
  LINEAR_SOLVER: 1
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: <output_interval_steps>
SCALAR TRANSPORT DYNAMIC/RUNTIME VTK OUTPUT:
  OUTPUT_SCATRA: true
  PHI: true
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct_solver"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: <diffusivity>
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    ENTITY_TYPE: node_set_id
    NUMDOF: 1
    ONOFF: [1]
    VAL: [<hot_boundary_temperature>]
    FUNCT: [0]
  - E: 2
    ENTITY_TYPE: node_set_id
    NUMDOF: 1
    ONOFF: [1]
    VAL: [<cold_boundary_temperature>]
    FUNCT: [0]
TRANSPORT GEOMETRY:
  ELEMENT_BLOCKS:
    - ID: 1
      TRANSP:
        QUAD4:
          MAT: 1
          TYPE: Std
  FILE: mesh.e
  SHOW_INFO: detailed_summary
RESULT DESCRIPTION:
  - SCATRA:
      DIS: "scatra"
      NODE: 1
      QUANTITY: "phi"
      VALUE: <expected_result_value>
      TOLERANCE: <result_tolerance>
""",
    }

    def get_template(self, variant: str = "default") -> str:
        if variant == "default":
            variant = "poisson_2d"
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
                "name": "poisson_2d",
                "description": (
                    "Stationary Poisson equation on a 2D unit square with "
                    "homogeneous Dirichlet BCs and a constant Neumann source."
                ),
            },
            {
                "name": "heat_transient_2d",
                "description": (
                    "Transient heat conduction on a 2D plate with BDF2 time "
                    "integration.  Hot edge (T=100) and cold edge (T=0)."
                ),
            },
        ]

    # ── Validation ────────────────────────────────────────────────────

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        errors: list[str] = []

        # Check DIFFUSIVITY
        diffusivity = params.get("DIFFUSIVITY")
        if diffusivity is not None:
            try:
                d = float(diffusivity)
                if d <= 0:
                    errors.append(
                        f"DIFFUSIVITY must be > 0, got {d}.  A non-positive "
                        f"diffusivity is unphysical and will cause divergence."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"DIFFUSIVITY must be a positive number, got {diffusivity!r}."
                )

        # Check CAPA (if Fourier material)
        capa = params.get("CAPA")
        if capa is not None:
            try:
                c = float(capa)
                if c <= 0:
                    errors.append(
                        f"CAPA (heat capacity) must be > 0, got {c}."
                    )
            except (TypeError, ValueError):
                errors.append(f"CAPA must be a positive number, got {capa!r}.")

        # Check CONDUCT (if Fourier material)
        conduct = params.get("CONDUCT")
        if conduct is not None:
            if isinstance(conduct, dict):
                vals = conduct.get("constant", [])
                if isinstance(vals, list):
                    for v in vals:
                        try:
                            if float(v) <= 0:
                                errors.append(
                                    f"CONDUCT values must be > 0, got {v}."
                                )
                        except (TypeError, ValueError):
                            errors.append(
                                f"CONDUCT values must be positive numbers, "
                                f"got {v!r}."
                            )
            else:
                try:
                    if float(conduct) <= 0:
                        errors.append(
                            f"CONDUCT must be > 0, got {conduct}."
                        )
                except (TypeError, ValueError):
                    errors.append(
                        f"CONDUCT must be a positive number or dict with "
                        f"'constant: [value]', got {conduct!r}."
                    )

        # Check TIMESTEP for transient problems
        timestep = params.get("TIMESTEP")
        timeintegr = params.get("TIMEINTEGR", "Stationary")
        if timeintegr != "Stationary" and timestep is not None:
            try:
                dt = float(timestep)
                if dt <= 0:
                    errors.append(
                        f"TIMESTEP must be > 0 for transient problems, got {dt}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"TIMESTEP must be a positive number, got {timestep!r}."
                )

        # Warn about VELOCITYFIELD
        velocityfield = params.get("VELOCITYFIELD")
        if velocityfield is not None and velocityfield not in (
            "zero", "function", "Navier_Stokes"
        ):
            errors.append(
                f"VELOCITYFIELD must be 'zero', 'function', or 'Navier_Stokes', "
                f"got {velocityfield!r}."
            )

        return errors
