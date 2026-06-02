"""Generator for solid mechanics physics module (linear/nonlinear elasticity, hyperelastic).

Covers quasi-static structural problems solved with DYNAMICTYPE: Statics
in the STRUCTURAL DYNAMIC section of 4C.  Produces validated, working
.4C.yaml templates for 2D and 3D solid mechanics analyses.
"""

from __future__ import annotations

from typing import Any

from .base import BaseGenerator


class SolidMechanicsGenerator(BaseGenerator):
    """Generator for solid mechanics problems in 4C.

    Supports linear elasticity (small deformation), geometrically nonlinear
    elasticity, hyperelastic materials (Neo-Hookean), and elastoplasticity.
    """

    module_key = "solid_mechanics"
    display_name = "Solid Mechanics (Linear / Nonlinear Elasticity)"
    problem_type = "Structure"

    # ── Knowledge ─────────────────────────────────────────────────────

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The solid mechanics module solves quasi-static structural "
                "problems (no inertia effects) using DYNAMICTYPE: Statics.  "
                "Supports small-deformation linear elasticity (KINEM: linear, "
                "MAXITER: 1) and large-deformation nonlinear analysis "
                "(KINEM: nonlinear, Newton-Raphson iteration).  "
                "The PROBLEM TYPE is 'Structure', the dynamics section is "
                "'STRUCTURAL DYNAMIC', and geometry goes into "
                "'STRUCTURE GEOMETRY'.  In 4C 2026.3 BOTH 2D "
                "and 3D problems use the SOLID eletype "
                "(e.g. 'SOLID QUAD4' in 2D with "
                "'PLANE_ASSUMPTION plane_strain' / 'plane_stress'; "
                "'SOLID HEX8' in 3D, no PLANE_ASSUMPTION). The "
                "legacy 'WALL' / 'THICK' / 'STRESS_STRAIN' "
                "keywords were renamed to 'SOLID' / 'THICKNESS' "
                "/ 'PLANE_ASSUMPTION' — writing the legacy form "
                "raises 'Unknown type WALL of finite element'."
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
                        "St. Venant-Kirchhoff hyperelastic material.  Valid for "
                        "small strains (both linear and nonlinear kinematics).  "
                        "Most common starting-point material for structural analysis."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus E",
                            "range": "> 0  (e.g. steel: 210000 MPa, aluminium: 70000 MPa)",
                        },
                        "NUE": {
                            "description": "Poisson's ratio nu",
                            "range": "0 < nu < 0.5  (0.3 typical for metals; approaching 0.5 = incompressible)",
                        },
                        "DENS": {
                            "description": "Mass density (only needed for dynamics or gravity loads)",
                            "range": "> 0  (e.g. steel: 7.85e-9 t/mm^3)",
                        },
                    },
                },
                "MAT_ElastHyper + ELAST_CoupNeoHooke": {
                    "description": (
                        "Compressible Neo-Hookean hyperelastic material for large "
                        "deformations.  Uses a two-material definition: "
                        "MAT_ElastHyper (wrapper with NUMMAT, MATIDS, DENS) "
                        "referencing an ELAST_CoupNeoHooke sub-material with "
                        "YOUNG and NUE.  Always use with KINEM: nonlinear."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus E of Neo-Hookean model",
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Poisson's ratio nu",
                            "range": "0 < nu < 0.5",
                        },
                        "DENS": {
                            "description": "Mass density (in MAT_ElastHyper)",
                            "range": "> 0",
                        },
                        "NUMMAT": {
                            "description": "Number of sub-materials (always 1 for simple Neo-Hookean)",
                            "range": "1",
                        },
                        "MATIDS": {
                            "description": "List of sub-material IDs referencing ELAST_CoupNeoHooke",
                            "range": "[<id>]",
                        },
                        "POLYCONVEX": {
                            "description": (
                                "MAT_ElastHyper wrapper flag: enable a runtime "
                                "polyconvexity check on the strain-energy "
                                "function (0 = off, 1 = on)."
                            ),
                            "range": "0 | 1",
                        },
                    },
                },
                "MAT_Struct_PlasticNlnLogNeoHooke": {
                    "description": (
                        "Finite-strain J2 elastoplasticity with logarithmic "
                        "Neo-Hookean elastic response.  Supports isotropic "
                        "exponential saturation hardening and optional viscoplasticity."
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
                            "description": "Mass density",
                            "range": "> 0",
                        },
                        "YIELD": {
                            "description": "Initial yield stress sigma_y0",
                            "range": "> 0",
                        },
                        "SATHARDENING": {
                            "description": "Saturation hardening stress (sigma_y_inf - sigma_y0)",
                            "range": ">= 0",
                        },
                        "HARDEXPO": {
                            "description": "Hardening exponent delta (controls rate of saturation)",
                            "range": "> 0",
                        },
                        "ISOHARD": {
                            "description": "Linear isotropic hardening modulus added on top of the Voce saturation law.",
                            "range": ">= 0",
                        },
                        "VISC": {
                            "description": (
                                "Perzyna-type viscoplastic viscosity eta.  "
                                "Set 0 for rate-independent plasticity."
                            ),
                            "range": ">= 0",
                        },
                        "RATE_DEPENDENCY": {
                            "description": (
                                "Perzyna rate exponent n.  Ignored when VISC = 0."
                            ),
                            "range": "> 0",
                        },
                        "TOL": {
                            "description": "Local return-mapping convergence tolerance.",
                            "range": "typical 1e-10",
                        },
                        "HARDENING_FUNC": {
                            "description": (
                                "ID of a user-defined hardening function in the "
                                "FUNCT section.  Set 0 to use the analytic "
                                "Voce+linear law parameterised by YIELD / "
                                "SATHARDENING / HARDEXPO / ISOHARD."
                            ),
                            "range": ">= 0",
                        },
                    },
                },
            },
            "time_integration": {
                "DYNAMICTYPE": (
                    "'Statics' for quasi-static analysis (load is applied "
                    "incrementally via time-stepping without inertia).  "
                    "For dynamic problems use the structural_dynamics generator."
                ),
                "KINEM": (
                    "'linear' -- small-deformation assumption (geometrically linear).  "
                    "'nonlinear' -- large deformation / finite strain.  "
                    "CRITICAL: KINEM must be consistent with the material model.  "
                    "St. Venant-Kirchhoff works with both; Neo-Hookean and "
                    "plasticity models REQUIRE nonlinear."
                ),
                "MAXITER": (
                    "Maximum Newton-Raphson iterations per load step.  "
                    "Set MAXITER: 1 for truly linear problems (linear material + "
                    "KINEM: linear) to avoid unnecessary iterations.  "
                    "Typical: 20--50 for nonlinear problems."
                ),
                "TOLDISP": "Displacement convergence tolerance (typical: 1e-6 to 1e-10).",
                "TOLRES": "Residual force convergence tolerance (typical: 1e-6 to 1e-10).",
            },
            "solver": {
                "small_problems": {
                    "SOLVER": "UMFPACK",
                    "description": (
                        "Direct solver, very robust.  Best for problems up to "
                        "~50k DOFs or for debugging."
                    ),
                },
                "large_problems": {
                    "SOLVER": "Belos",
                    "AZPREC": "MueLu",
                    "SOLVER_XML_FILE": "iterative_gmres_template.xml",
                    "MUELU_XML_FILE": "elasticity_template.xml",
                    "description": (
                        "Iterative Krylov solver (GMRES) with MueLu AMG "
                        "preconditioner.  Scalable to millions of DOFs.  "
                        "Requires XML configuration files for solver and "
                        "preconditioner."
                    ),
                },
            },
            "plasticity_models": {
                "MAT_Struct_PlasticLinElast": {
                    "description": "Small-strain J2 (von Mises) with linear isotropic+kinematic hardening",
                    "parameters": "YOUNG, NUE, DENS, ISOHARD, KINHARD, YIELD, TOL",
                    "kinematics": "linear only",
                },
                "MAT_Struct_DruckerPrager": {
                    "description": (
                        "Small-strain Drucker-Prager (pressure-dependent, smooth cone). "
                        "C is the cohesion parameter; ETA/XI/ETABAR are pre-computed "
                        "from friction phi and dilatancy psi -- see eta_xi_formulas."
                    ),
                    "parameters": "YOUNG, NUE, DENS, ISOHARD, TOL, C, ETA, XI, ETABAR, TANG, MAXITER",
                    "kinematics": "linear only",
                    "eta_xi_formulas": (
                        "For outer cone (circumscribed, matches MC at compression meridian): "
                        "ETA = 6*sin(phi)/(3-sin(phi)), XI = 6*cos(phi)/(3-sin(phi)), "
                        "ETABAR = 6*sin(psi)/(3-sin(psi)). "
                        "For inner cone (inscribed, matches MC at extension meridian): "
                        "ETA = 6*sin(phi)/(3+sin(phi)), XI = 6*cos(phi)/(3+sin(phi)). "
                        "For middle cone (Lode-angle independent best fit): "
                        "ETA = 3*tan(phi)/sqrt(9+12*tan^2(phi)), XI = 3/sqrt(9+12*tan^2(phi))."
                    ),
                },
                "MAT_PlasticElastHyper": {
                    "description": (
                        "Finite-strain J2/Hill with nonlinear isotropic+kinematic hardening, "
                        "Perzyna viscoplasticity, optional thermal softening.  Wraps an "
                        "ELAST_* hyperelastic sub-material; elastic moduli (YOUNG/NUE) "
                        "live in that sub-material, not on this wrapper."
                    ),
                    "parameters": (
                        "INITYIELD, ISOHARD, KINHARD, EXPISOHARD, INFYIELD, VISC, "
                        "RATE_DEPENDENCY, VISC_SOFT, YIELDSOFT, HARDSOFT, "
                        "CTE, INITTEMP, TAYLOR_QUINNEY, PL_SPIN_CHI"
                    ),
                    "kinematics": "nonlinear",
                    "features": "Hill anisotropy, Perzyna viscoplasticity, thermal softening (TSI)",
                },
                "MAT_Struct_PlasticGTN": {
                    "description": (
                        "Gurson-Tvergaard-Needleman ductile damage with void growth, "
                        "nucleation and coalescence (KINEM: linear only in current 4C release)."
                    ),
                    "parameters": (
                        "YOUNG, NUE, DENS, YIELD, ISOHARD, HARDENING_FUNC, "
                        "F0, FN, SN, EN, FC, KAPPA, EF, K1, K2, K3, MAXITER, TOL"
                    ),
                    "kinematics": "linear only",
                },
                "MAT_crystal_plasticity": {
                    "description": (
                        "Single-crystal plasticity with dislocation-density based hardening and "
                        "optional deformation twinning.  Lattice families accepted by the runtime: "
                        "FCC, BCC, D019, L10 (LAT key, default FCC).  The input parser's own "
                        "description string advertises HCP as well, but the constructor sanity "
                        "check rejects 'HCP' with FOUR_C_THROW — use D019 for hexagonal lattices."
                    ),
                    "parameters": (
                        # elastic + Newton tolerance
                        "TOL, YOUNG, NUE, DENS, "
                        # crystal lattice
                        "LAT, CTOA, ABASE, "
                        # slip-system definition
                        "NUMSLIPSYS, NUMSLIPSETS, SLIPSETMEMBERS, SLIPRATEEXP, GAMMADOTSLIPREF, "
                        "DISDENSINIT, DISGENCOEFF, DISDYNRECCOEFF, TAUY0, MFPSLIP, SLIPHPCOEFF, "
                        "SLIPBYTWIN, "
                        # twin-system definition (all optional; NUMTWINSYS/SETS default 0)
                        "NUMTWINSYS, NUMTWINSETS, TWINSETMEMBERS, TWINRATEEXP, GAMMADOTTWINREF, "
                        "TAUT0, MFPTWIN, TWINHPCOEFF, TWINBYSLIP, TWINBYTWIN"
                    ),
                    "kinematics": "nonlinear",
                    "features": (
                        "Dislocation-density evolution (generation + dynamic recovery), "
                        "Hall-Petch via MFPSLIP/MFPTWIN with HP coefficients, "
                        "slip-twin and twin-twin coupling, multiple lattice types"
                    ),
                    "pitfalls": (
                        "Vector-size rule depends on the parameter (two distinct sizing axes):\n"
                        "  - one entry per physical *system* (size = NUMSLIPSYS or NUMTWINSYS): "
                        "SLIPSETMEMBERS, TWINSETMEMBERS — these are 1-based indices into the "
                        "set table (range 1..NUMSLIPSETS or 1..NUMTWINSETS) saying which set "
                        "each individual slip/twin system belongs to.\n"
                        "  - one entry per *set* (size = NUMSLIPSETS or NUMTWINSETS): every "
                        "other slip/twin vector (SLIPRATEEXP, GAMMADOTSLIPREF, DISDENSINIT, "
                        "DISGENCOEFF, DISDYNRECCOEFF, TAUY0, MFPSLIP, SLIPHPCOEFF, SLIPBYTWIN "
                        "and their TWIN-side counterparts TWINRATEEXP, GAMMADOTTWINREF, TAUT0, "
                        "MFPTWIN, TWINHPCOEFF, TWINBYSLIP, TWINBYTWIN).\n"
                        "Mixing these two sizes is a common error — NUMSLIPSYS usually exceeds "
                        "NUMSLIPSETS (e.g. FCC has 12 systems often in 1 set).  Twinning is "
                        "optional: leave NUMTWINSYS=NUMTWINSETS=0 and omit every TWIN-side "
                        "vector (they all have parser defaults) for pure slip plasticity."
                    ),
                },
            },
            "plasticity_pitfalls": [
                "Drucker-Prager uses pre-computed constants (ETA, XI, ETABAR) that map from friction/dilatancy "
                "angles. ETA = 6*sin(phi)/(3-sin(phi)), XI = 6*cos(phi)/(3-sin(phi)), ETABAR = 6*sin(psi)/(3-sin(psi)) "
                "for the circumscribed outer cone. Getting these wrong silently gives wrong yield stress.",
                "PlasticLinElast and DruckerPrager are small-strain only (KINEM: linear). "
                "PlasticElastHyper requires KINEM: nonlinear. Mismatch produces FOUR_C_THROW.",
                "For quasi-static plasticity, use many small load steps (NUMSTEP >= 100). "
                "Too few steps causes the Newton iteration to diverge when crossing the yield surface.",
                "Accumulated plastic strain output is available via 'accumulated_plastic_strain' "
                "in IO/RUNTIME VTK OUTPUT/STRUCTURE with STRESS_STRAIN: true.",
                "TANG parameter controls the material tangent: 'consistent' uses the algorithmic "
                "elastoplastic tangent (required for global Newton convergence in load-controlled problems). "
                "'elastic' uses the elastic tangent as a fallback (poor convergence near yield, but robust "
                "for debugging). Always use 'consistent' for production plasticity simulations.",
                "TESTING PITFALL: Fully displacement-controlled single-element tests (all DOFs prescribed "
                "via Dirichlet BCs) bypass the global Newton iteration — the return mapping is called but "
                "its tangent is irrelevant. For benchmarking plasticity, use Neumann BCs on at least one "
                "face (e.g., confining pressure) so the tangent is actually exercised.",
                "NEUMANN SIGN CONVENTION for structural problems: negative values in the normal direction "
                "produce compressive traction. For a face with outward normal in +x, setting "
                "VAL: [-100e3, 0, 0, 0, 0, 0] applies 100 kPa compression on that face.",
            ],
            "pitfalls": [
                (
                    "[Numerical] KINEM must match the physical assumption.  "
                    "Using 'KINEM: linear' with a hyperelastic material "
                    "(Neo-Hookean, Mooney-Rivlin) is WRONG and produces "
                    "meaningless results. Signal: deflection / strain "
                    "results stay linear with load and disagree sharply "
                    "with the equivalent KINEM: nonlinear run for the same "
                    "geometry — typical divergence ~40% at large strain. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Performance] For truly linear problems set MAXITER: 1. "
                    " 4C will waste time iterating if MAXITER > 1 because "
                    "the solution is already converged after one step. "
                    "Signal: NOX convergence log shows residual already "
                    "below TOLRES at iteration 1 yet a second Newton step "
                    "is taken on every time step. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] DENS (density) is only needed for dynamics or "
                    "body-force loads (gravity).  For quasi-static problems "
                    "without gravity it can be omitted, but it is MANDATORY "
                    "for structural dynamics. Signal: a transient run with "
                    "DYNAMICTYPE: GenAlpha aborts with `mass matrix needs "
                    "DENS in material X` or produces all-zero displacements "
                    "with no inertia effect. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] HEX8 elements suffer from volumetric and "
                    "shear locking in bending-dominated or nearly-"
                    "incompressible problems.  Use TECH: eas_full or "
                    "TECH: fbar to mitigate locking, or use higher-order "
                    "elements (HEX27, TET10). Signal: deflection in a "
                    "cantilever / plate-bending benchmark is much smaller "
                    "than analytical (often 10-100x too stiff), or Poisson "
                    "ratio nu approaching 0.5 makes the response "
                    "approach rigid. (Audit 2026-06-02.)"
                ),
                (
                    "[API] In 4C 2026.3 BOTH 2D and 3D "
                    "structural elements use the SOLID eletype "
                    "factory. 2D elements (SOLID QUAD4 / TRI3 "
                    "etc.) require THICKNESS and "
                    "PLANE_ASSUMPTION: 'THICKNESS 1.0 "
                    "PLANE_ASSUMPTION plane_strain' or "
                    "'plane_stress'. Signal: writing the "
                    "legacy 'WALL' category + 'THICK' / "
                    "'STRESS_STRAIN' keywords raises 'Unknown "
                    "type WALL of finite element' from "
                    "parobjectfactory.cpp:153 at problem setup. "
                    "In ELEMENT_BLOCKS with Exodus meshes use "
                    "the SOLID: sub-key for both 2D and 3D. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Input] Neumann conditions for structures "
                    "have NUMDOF: 6 (3 forces + 3 moments in "
                    "3D) or NUMDOF: 6 in 2D (2 forces + 1 "
                    "moment + 3 unused). The first entries "
                    "are force components. Signal: NUMDOF: 3 "
                    "on a structural Neumann silently drops "
                    "the moment components — applied "
                    "concentrated moments are zero, but no "
                    "error message; the displacement field "
                    "looks like a force-only result. Use "
                    "NUMDOF: 6 even when only forces are "
                    "needed and zero the moment slots. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Input] INT_STRATEGY: Standard is the "
                    "default and works for most problems. Only "
                    "change to 'Old' for legacy compatibility. "
                    "Signal: explicitly setting INT_STRATEGY: "
                    "Old in a new problem can disable modern "
                    "FSI hooks or contact accelerations — the "
                    "simulation runs but features added since "
                    "the rewrite (e.g. specific contact "
                    "formulations) are unavailable. Default "
                    "(omit) is the safe choice. (Audit "
                    "2026-06-02.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "cantilever_2d",
                    "description": (
                        "2D cantilever beam under tip load.  Fixed left edge, "
                        "point or line load on right edge.  Uses SOLID QUAD4 "
                        "elements with plane_strain, MAT_Struct_StVenantKirchhoff, "
                        "KINEM: linear, MAXITER: 1."
                    ),
                    "template_variant": "linear_2d",
                },
                {
                    "name": "compression_3d",
                    "description": (
                        "3D block under uniaxial compression.  Fixed bottom face, "
                        "prescribed displacement on top face.  Uses SOLID HEX8 "
                        "elements with Neo-Hookean material, KINEM: nonlinear."
                    ),
                    "template_variant": "nonlinear_3d",
                },
            ],
        }

    # ── Templates ─────────────────────────────────────────────────────

    _TEMPLATES: dict[str, str] = {
        "linear_2d": """\
# FORMAT TEMPLATE — 2D linear elasticity (plane strain)
# All numerical values are placeholders — determine from your specific problem.
# Check 4C test files via examples(keyword, solver='fourc', action='search') for reference setups.
TITLE:
  - "Linear elastic 2D — plane strain"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
IO:
  STRUCT_STRESS: "Cauchy"
  STRUCT_STRAIN: "GL"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
  STRESS_STRAIN: true
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "Statics"
  TIMESTEP: <timestep>
  NUMSTEP: <number_of_steps>
  MAXTIME: <end_time>
  TOLDISP: <displacement_tolerance>
  TOLRES: <residual_tolerance>
  MAXITER: <max_newton_iterations — use 1 for linear, 10+ for nonlinear>
  LINEAR_SOLVER: 1
  PREDICT: TangDis
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct_solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: <Young_modulus>
      NUE: <Poisson_ratio>
      DENS: <density>
DESIGN LINE DIRICH CONDITIONS:
  - E: <boundary_id>
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0.0, 0.0]
    FUNCT: [0, 0]
DESIGN LINE NEUMANN CONDITIONS:
  - E: <boundary_id>
    NUMDOF: 6
    ONOFF: [<active_dofs>]
    VAL: [<force_values>]
    FUNCT: [0, 0, 0, 0, 0, 0]
STRUCTURE GEOMETRY:
  ELEMENT_BLOCKS:
    - ID: 1
      WALL:
        QUAD4:
          MAT: 1
          KINEM: linear
          THICK: <thickness>
          STRESS_STRAIN: plane_strain
  FILE: <mesh_file.e>
  SHOW_INFO: detailed_summary
""",
        "nonlinear_3d": """\
TITLE:
  - "Nonlinear 3D block compression -- Neo-Hookean, large deformation"
PROBLEM SIZE:
  DIM: 3
# FORMAT TEMPLATE — 3D nonlinear elasticity (large deformation)
# All numerical values are placeholders.
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
IO:
  STRUCT_STRESS: "Cauchy"
  STRUCT_STRAIN: "GL"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
  STRESS_STRAIN: true
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "Statics"
  TIMESTEP: <load_step_size>
  NUMSTEP: <number_of_load_steps>
  MAXTIME: <total_load_parameter>
  TOLDISP: <displacement_tolerance>
  TOLRES: <residual_tolerance>
  MAXITER: <max_newton_iterations>
  PREDICT: TangDis
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct_solver"
MATERIALS:
  - MAT: 1
    MAT_ElastHyper:
      NUMMAT: 1
      MATIDS: [10]
      DENS: <density>
  - MAT: 10
    ELAST_CoupNeoHooke:
      YOUNG: <Young_modulus>
      NUE: <Poisson_ratio>
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN SURF DIRICH CONDITIONS:
  - E: <fixed_boundary_id>
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: <loaded_boundary_id>
    NUMDOF: 3
    ONOFF: [<active_dofs>]
    VAL: [<prescribed_displacement>]
    FUNCT: [0, 1, 0]
STRUCTURE GEOMETRY:
  ELEMENT_BLOCKS:
    - ID: 1
      SOLID:
        HEX8:
          MAT: 1
          KINEM: nonlinear
  FILE: <mesh_file.e>
  SHOW_INFO: detailed_summary
""",
    }

    def get_template(self, variant: str = "default") -> str:
        if variant == "default":
            variant = "linear_2d"
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
                "name": "linear_2d",
                "description": (
                    "Linear elastic 2D cantilever with plane strain, SOLID QUAD4 "
                    "elements, St. Venant-Kirchhoff material, tip load."
                ),
            },
            {
                "name": "nonlinear_3d",
                "description": (
                    "Geometrically nonlinear 3D block compression with "
                    "Neo-Hookean hyperelastic material, HEX8 elements, "
                    "prescribed displacement loading."
                ),
            },
        ]

    # ── Validation ────────────────────────────────────────────────────

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        errors: list[str] = []

        # Check Poisson's ratio
        nue = params.get("NUE")
        if nue is not None:
            try:
                nu = float(nue)
                if nu <= 0:
                    errors.append(
                        f"NUE (Poisson's ratio) must be > 0, got {nu}.  "
                        f"A non-positive Poisson's ratio is unphysical for "
                        f"standard structural materials."
                    )
                elif nu >= 0.5:
                    errors.append(
                        f"NUE (Poisson's ratio) must be < 0.5, got {nu}.  "
                        f"nu = 0.5 means perfectly incompressible, which "
                        f"causes a singular stiffness matrix with standard "
                        f"displacement elements.  Use nu <= 0.499."
                    )
                elif nu > 0.49:
                    errors.append(
                        f"NUE = {nu} is very close to 0.5 (incompressible "
                        f"limit).  Standard HEX8/QUAD4 elements will exhibit "
                        f"severe volumetric locking.  Consider using "
                        f"TECH: fbar or higher-order elements."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"NUE must be a number in (0, 0.5), got {nue!r}."
                )

        # Check Young's modulus
        young = params.get("YOUNG")
        if young is not None:
            try:
                e = float(young)
                if e <= 0:
                    errors.append(
                        f"YOUNG (Young's modulus) must be > 0, got {e}.  "
                        f"A non-positive modulus is unphysical."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"YOUNG must be a positive number, got {young!r}."
                )

        # Check density (warn if zero in dynamics context)
        dens = params.get("DENS")
        dynamictype = params.get("DYNAMICTYPE", "Statics")
        if dens is not None:
            try:
                d = float(dens)
                if d < 0:
                    errors.append(
                        f"DENS (density) must be >= 0, got {d}."
                    )
                if d == 0 and dynamictype != "Statics":
                    errors.append(
                        f"DENS = 0 with DYNAMICTYPE = {dynamictype!r}: "
                        f"zero density means zero mass matrix.  Dynamics "
                        f"requires DENS > 0."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"DENS must be a non-negative number, got {dens!r}."
                )

        # Check KINEM vs material consistency
        kinem = params.get("KINEM")
        material = params.get("material_type", "")
        if kinem == "linear" and material in (
            "MAT_ElastHyper", "MAT_Struct_PlasticNlnLogNeoHooke"
        ):
            errors.append(
                f"KINEM: linear with {material} is inconsistent.  "
                f"Hyperelastic and plasticity materials require "
                f"KINEM: nonlinear."
            )

        # Check yield stress for plastic material
        yield_stress = params.get("YIELD")
        if yield_stress is not None:
            try:
                ys = float(yield_stress)
                if ys <= 0:
                    errors.append(
                        f"YIELD (initial yield stress) must be > 0, got {ys}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"YIELD must be a positive number, got {yield_stress!r}."
                )

        # Check SATHARDENING
        sathard = params.get("SATHARDENING")
        if sathard is not None:
            try:
                sh = float(sathard)
                if sh < 0:
                    errors.append(
                        f"SATHARDENING must be >= 0, got {sh}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"SATHARDENING must be a non-negative number, "
                    f"got {sathard!r}."
                )

        # Check HARDEXPO
        hardexpo = params.get("HARDEXPO")
        if hardexpo is not None:
            try:
                he = float(hardexpo)
                if he <= 0:
                    errors.append(
                        f"HARDEXPO (hardening exponent) must be > 0, got {he}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"HARDEXPO must be a positive number, got {hardexpo!r}."
                )

        return errors
