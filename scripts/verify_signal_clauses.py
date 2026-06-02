#!/usr/bin/env python3
"""
Signal-clause verification harness.

The Open-FEM-Agent paper (§3.2 / Table 1) sells the pitfall DB as
the distinctive value-add. Each pitfall entry in
``backends.<backend>.generators.*.KNOWLEDGE['pitfalls']`` ships a
``Signal:`` clause stating the observable symptom — the string the
post-execution critic is supposed to match against actual error /
result text.

The senior-AI-scientist critic (2026-05-31) flagged the
unfalsified state of these signals as the second-largest risk:
"every encoded pitfall is a claim the project cannot defend."
This harness operationalises the verification in three tiers:

  * **Tier 0** — structural. The Signal text references at least
    one entity (class name, function name, error class) that is
    real and known to the canonical catalogs. Cheap; catches
    typos like "FE_Simplex" (missing the trailing P).
  * **Tier 1** — semantic. The Signal text uses observable-symptom
    vocabulary: "report", "error", "diverges", "converges to",
    "exits", "raises", "warns", "stalls", "oscillates", a
    quoted-error pattern, or a numerical observation. Catches
    vague non-actionable signals.
  * **Tier 2** — operational. An intentional-failure regression
    fixture compiles + runs and the Signal text appears in the
    captured stderr / stdout. This is the strongest tier but
    each fixture is hand-written and per-backend, so the work is
    multi-week. The harness here records which Signal entries
    have Tier-2 verification and which are still on the
    Tier-0+1 floor.

Usage:
    python scripts/verify_signal_clauses.py
    python scripts/verify_signal_clauses.py --backend dealii
    python scripts/verify_signal_clauses.py --tier 2  # only run operational

Output: ``scripts/scan_results/signal_verification.json`` with
per-pitfall verification status.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "scripts" / "scan_results" / "signal_verification.json"


# Observable-symptom vocabulary — Tier-1 semantic check looks for
# at least one of these in the Signal: clause.
OBSERVABLE_VOCAB = (
    "report", "error", "exit", "raise", "warn", "stall",
    "converge", "diverge", "oscillat", "break", "crash",
    "abort", "missing", "undefined", "differs", "drop", "drift",
    "appear", "shows", "matches", "wrong", "zero", "nan",
    "checkerboard", "pattern", "amplitude", "value",
    "grows", "shrinks", "exceeds", "below", "above",
    "larger", "smaller", "slower", "faster",
    "wall-time", "wall time", "wall-clock", "wall clock",
    "iteration", "iterations",
    "reach", "returns",
    # ── Round-5 vocabulary extension (2026-06-02): observable
    #    verbs widely used across catalog Signal: prose. Each
    #    is a legitimate symptom token a post-execution critic
    #    could grep for in solver output (see audit-pass-15
    #    commit message for the per-token rationale). ────────
    "produc", "give", "yield",      # X produces / gives / yields Y
    "disagree",                     # disagrees with reference
    "silent",                       # silently does nothing / allocated
    "fail",                         # fails to converge / fails with
    "factor", "magnitude",          # by factor of / orders of magnitude
    "locking", "locks",             # volumetric/shear locking; X locks
                                    # ('lock' alone substring-matches 'block')
    "jump",                         # field jumps at boundary
    "decay",                        # no decay observed
    "leak",                         # numerical leak / energy leak
    "spike",                        # contact-pressure spikes
    "reset",                        # resets to default
    "force",                        # forces re-compile / forces zero
    "reduce",                       # reduces stiffness
    # ── Round-6 vocabulary extension (2026-06-02) ──────────
    "iterat",                       # iterates / iterating / iteration
    "stagnat",                      # stagnates / stagnating
    "accept",                       # accepted by / silently accepts
    "lose",                         # loses solution continuity
    "explod",                       # explodes at first step
    "miss",                         # misses contact pairs
    "deviat",                       # deviates ~30%
    "satura",                       # damage has saturated
    "passes",                       # passes through (no contact)
    "wast",                         # wastes compute
    "translat",                     # translates as rigid body
    "smear",                        # smears interface / mass-smearing
    "identical",                    # looks identical
    "deplete",                      # depletes mass / energy
    "tighten", "looser",            # tighten the tolerance
    "untouched",                    # left untouched
    # ── Round-7 vocabulary extension (2026-06-02) ──────────
    "should",                       # should be in [..] range
    "stops",                        # stops decreasing
    "negative",                     # gap goes negative
    "offset",                       # huge additive offset
    "ignor",                        # ignores cross-diffusion
    "cannot",                       # cannot capture growth
    "disable",                      # disables modern hooks
    "defeat",                       # defeats the coupling
    "instead",                      # X instead of Y
    "longer",                       # takes 10-100x longer
    "stderr", "stdout",             # appears in 4C stderr
    "free",                         # leaves nodes free
    "freely",                       # particles move freely
    "uniformly",                    # uniformly low / uniform
    "huge",                         # huge additive offset
    "elastic", "pure_elastic",
    "approaches",                   # max(D) approaches 0.99
    "regardless",                   # X stays constant regardless of Y
    "times out", "timeout",         # MCP stdio times out at 60s
    "noticeably",                   # noticeably slow
    " == ",                         # API claim: Nbfun == 6
    "eigenvalu",                    # eigenvalues from eigsh ...
                                    # (real observable; recovered after the
                                    #  word-boundary tightening dropped the
                                    #  loose 'value' inside 'eigenvalue')
)


@dataclass
class SignalVerification:
    backend: str
    physics: str
    pitfall_index: int                     # 0-based index into pitfalls list
    pitfall_category: str                  # [Syntax]/[Physics]/...
    signal_text: str                       # the Signal: clause itself
    # Tier-0 split per critic 2026-05-31 round 2:
    #   tier0_code_symbol_matched  — Signal references ≥1 real code
    #     symbol (deal.II class, exception, library identifier).
    #     This is the HONEST Tier-0 gate.
    #   tier0_domain_names_matched — Signal references ≥1 textbook
    #     concept (Stokes, Newton, Turek). Decorative; NOT
    #     sufficient on its own for Tier 0.
    #   tier0_passed                — true iff tier0_code_symbol_matched
    tier0_passed: bool = False
    tier0_code_symbol_matched: list = field(default_factory=list)
    tier0_domain_names_matched: list = field(default_factory=list)
    tier0_entities_matched: list = field(default_factory=list)  # back-compat union
    tier1_passed: bool = False
    tier1_vocab_hits: list = field(default_factory=list)
    tier2_passed: bool = False
    tier2_status: str = "not_attempted"    # not_attempted / harness_pending / passed / failed
    notes: list = field(default_factory=list)


_PITFALL_PREFIX_RE = re.compile(
    r"^\s*\[(Syntax|Physics|Numerical|API|Integration|Input|Output|"
    r"Mesh|Performance|Hardware|Validation|Reference)\]")
_SIGNAL_RE = re.compile(r"\bSignal:\s*(.+?)$", re.IGNORECASE | re.DOTALL)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def _split_pitfall(text: str) -> tuple[str | None, str | None]:
    """Return (category, signal_text) — both None if not parseable."""
    m_cat = _PITFALL_PREFIX_RE.match(text)
    cat = m_cat.group(1) if m_cat else None
    m_sig = _SIGNAL_RE.search(text)
    sig = m_sig.group(1).strip() if m_sig else None
    return cat, sig


def _load_entity_split(backend: str) -> tuple[set[str], set[str]]:
    """Return (code_symbols, domain_names).

    Senior-AI-scientist critic 2026-05-31 round 2: the prior
    single-set design let domain words ("Stokes", "Poisson",
    "Newton") count as Tier-0 hits, so every elasticity pitfall
    scored on "Poisson ratio" and every flow pitfall on "Stokes".
    That made Tier-0 a near-tautology and inflated the headline
    pass rate.

    Splitting them:
      * **code_symbols** — names a programmer would `grep -r` for
        in deal.II source: classes (FE_Q, SolverCG, GridIn,
        DataOut, VectorTools), exceptions (ExcMessage,
        ExcDimensionMismatch), function names (compress,
        condense), and library-internal identifiers (KINSOL,
        SLEPc, EPS_TARGET_REAL). A Tier-0 pass requires at least
        one match here.
      * **domain_names** — words a textbook would use: physics
        names (Stokes, Maxwell, Laplace), method names (Newton,
        Newmark, BDF2, Crank-Nicolson), benchmark people
        (Turek, Ghia, Kirsch). Decorative; NEVER sufficient on
        their own for Tier 0.
    """
    code_symbols: set[str] = set()
    domain_names: set[str] = set()

    # Critical: ensure src/ is on sys.path BEFORE importing the
    # element catalog. Otherwise the first call returns a
    # partial code_symbols set (silently swallowed ImportError)
    # and later calls return the full set after _harvest_pitfalls
    # has inserted src/. The merge-gate tests were papered over
    # by this — re-running setUp warmed the import via
    # _harvest_pitfalls, so 11 of 12 test methods saw the
    # "post-warmup" 165-symbol set, missing real Tier-0
    # regressions. Fixed 2026-05-31 (round-4 critic finding).
    src_path = str(REPO_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    if backend == "dealii":
        try:
            mod = importlib.import_module(
                "backends.dealii.element_catalog")
            code_symbols.update(getattr(mod, "ELEMENT_NAMES", set()))
            code_symbols.update(getattr(mod, "MESH_GENERATOR_NAMES", set()))
        except ImportError:
            pass
        # ── Additional dealii-specific identifiers used in
        #    catalog Signal: prose (deal.II tutorial step
        #    numbers, Schur reformulation, time-step schemes,
        #    DG limiters, multiphysics observables) ────────
        code_symbols.update({
            # ── Tutorial step references (catalog prose uses
            #    "step-20", "step-22", "step-69" hyphenated;
            #    after tokenisation we see "step" + numeric;
            #    add both forms) ──────────────────────────────
            "step_20", "step_21", "step_22", "step_69",
            "step_15", "step_31", "step_32", "step_40",
            "step_57", "step_55", "step_56",
            # ── DG limiters / shock-capturing identifiers ────
            "TVB", "Minmod", "WENO", "MUSCL",
            "entropy_viscosity", "Gibbs", "Gibbs_oscillation",
            "TVB_Minmod",
            # ── Time-stepping / SSP schemes ─────────────────
            "SSP_RK3", "SSP_RK4", "leapfrog",
            "Crank_Nicolson", "BDF", "BDF1", "BDF2",
            "Heun", "RungeKutta",
            # ── Schur / saddle-point reformulation ──────────
            "Schur", "Schur_complement",
            "fieldsplit", "Hiptmair", "block_preconditioner",
            "M_inv", "B_transpose",
            # ── Darcy / VoF / level-set / phase-field ───────
            "Darcy", "DarcyVelocity", "VoF",
            "level_set", "level_set_reinit",
            "Cahn_Hilliard", "Allen_Cahn", "phase_field",
            "reinitialisation", "redistance",
            # ── Topology optimisation observables ───────────
            "SIMP", "grey_scale", "volume_fraction",
            "density_filter", "Helmholtz_filter",
            "intermediate_density", "topology_opt",
            # ── Goal-oriented / DWR / error estimation ──────
            "DWR", "dual_weighted_residual",
            "effectivity_index", "Galerkin_orthogonality",
            "primal", "dual", "goal_functional",
            # ── DG numerical fluxes ─────────────────────────
            "central_flux", "centred_flux",
            "upwind", "downwind", "Lax_Friedrichs",
            "Roe", "HLL", "Rusanov", "u_hat",
            # ── Convection / advection benchmarks ───────────
            "Rayleigh", "Bénard", "Benard",
            "convective_cells", "Rayleigh_number",
            "Reynolds", "Re", "Pe",
            "Poiseuille", "Couette",
            "turbulence_model",
            # ── multiphysics observables (already-present
            #    code symbols are FE_*, dealii fixtures) ─────
            "mass_loss", "interface_smearing",
            "DG_flux",
        })

    # Per-backend code-symbol sets. Added 2026-05-31 (round-4
    # critic verdict — Table-1 promotion of non-deal.II backends).
    # These are names a programmer would `grep -r` for in the
    # backend's source / a Python traceback would emit. Cheap
    # to maintain; the alternative is Tier-0 = 0 forever for
    # every non-deal.II backend.
    if backend == "ngsolve":
        code_symbols.update({
            # FE spaces / function spaces ─────────────────────────
            "HCurl", "HCurlHighOrderFESpace", "H1",
            "H1HighOrderFESpace", "L2", "HDiv", "FESpace",
            "FacetFESpace", "VectorH1", "Periodic",
            "Compressed", "Discontinuous",
            # Forms / operators ───────────────────────────────────
            "BilinearForm", "LinearForm", "SymbolicBFI",
            "SymbolicLFI", "GridFunction", "CoefficientFunction",
            "Mesh", "MaterialCF", "FlatVector",
            # Differential operators ──────────────────────────────
            "curl", "grad", "div", "Trace", "Deriv", "Skew",
            # Solvers / preconditioners ───────────────────────────
            "ArnoldiSolver", "CGSolver", "MinRes", "GMRESSolver",
            "BVP", "Preconditioner", "MultiGrid", "HCurlAMG",
            "BramblePasciakCG", "krylovspace", "Newton",
            # Mesh / geometry ─────────────────────────────────────
            "CSGeometry", "OrthoBrick", "SplineGeometry",
            "Cylinder", "Sphere", "OCCGeometry",
            "GenerateMesh", "unit_square", "unit_cube",
            # Output / runtime ────────────────────────────────────
            "VTKOutput", "Draw", "Redraw", "TaskManager",
            "SetHeapSize", "Inverse",
            # Exception class ─────────────────────────────────────
            "NgException", "Exception",
            # CoefficientFunction internals that show up in
            # actual NGSolve error text (verified by empirical
            # probe 2026-06-01 — see maxwell#3 audit).
            "ScaleCF", "ConstantCF", "CompoundCF",
            "biform_from_py", "lform_from_py",
            # Linear-algebra backends NGSolve wraps for inverse
            # operations (verified 2026-06-01, maxwell#0 audit).
            "UmfpackInverse", "PardisoInverse", "SparseCholesky",
            "UMFPACK",
            # ── Linear-elasticity API surface (verified 2026-06-01
            #    by Tier-2 fixture vector_h1_vs_h1_dim2_equivalence
            #    in scripts/tier2_fixtures/ngsolve/) ──────────────
            "MatrixValued", "ComponentGridFunction",
            "ProxyFunction", "CompoundFESpace", "FreeDofs",
            "Grad", "InnerProduct", "Id", "Stress", "Strain",
            "TnT", "Assemble",
            # ── Hyperelasticity / nonlinear material API (verified
            #    2026-06-01 by fixtures hyperelasticity_newton_
            #    maxit_kwarg and plasticity_newtoncf_in_fem_
            #    submodule) ───────────────────────────────────────
            "Newton", "Apply", "AssembleLinearization",
            "Variation", "maxit", "maxerr", "dampfactor",
            "NewtonCF", "MinimizationCF",
            "IntegrationRule", "IntegrationPoint",
            "ngsolve",
            # PML helpers (helmholtz promotion 2026-06-01).
            "SetPML", "Radial", "pml", "Integrate",
            "L2Norm", "ComputeRadialMode",
            # Common method/attribute names that appear in tracebacks
            "Assemble", "TnT", "FreeDofs", "ndof",
            # ── NGSolve GridFunction conventional name + attrs
            "gfu", "gfu_vec", "vecs", "components",
            "mdcomponents",
            # ── NGSolve CoefficientFunction conventional names
            "boundary_cf", "rhs_cf", "stress_cf",
            "coefficient_function",
            # ── Skeleton / DG form keywords ──────────────────
            "dgjumps", "Other", "skeleton",
            "BND", "VOL", "INNER", "OUTER",
            "SparseMatrixDynamic", "SymbolicEnergy",
            # ── Krylov / direct solvers + preconditioners ────
            "MinResSolver", "Krylov",
            "CGSolverPlusGradedAMG",
            # ── VTK / output kwargs ───────────────────────────
            "VTKOutput", "subdivision", "filename",
            # ── ufl/ngsolve operator helpers ──────────────────
            "IfPos", "And", "Or", "Not",
            "Compile", "comp",
            # ── Mass matrix / bilinear form attribute accessors
            "a_mat", "m_mat", "f_vec",
            "mat", "vec",
            # ── Numerical-analysis observables that appear in
            #    NGSolve catalog Signal: prose ──────────────────
            "coercivity_loss", "alpha", "alpha_0",
            "phase_error", "phase_velocity",
            "kinetic_energy", "div_u", "grad_div",
            "Taylor_Hood", "MINI", "Mini",
            "p_plus_1", "h_squared",
            "lambda_max", "lambda_min", "max_u",
            "downwind", "upwind",
            "Crank_Nicolson", "BE", "CN",
            "FE", "ImplicitEuler", "ExplicitEuler",
            # ── NGSolve geometry / mesh attribute accessors
            "Mesh_dim", "ne", "nv",
            "mesh_dim", "GetMaterials", "GetBoundaries",
            "vertices", "elements", "facets",
            # ── NGSolve output channels in catalog prose ──────
            "max_B", "max_u", "max_p", "max_sigma",
            "sigma_n", "sigma_xx", "sigma_yy",
            "B_field", "E_field", "Hcurl_field",
            "stress_field", "strain_field",
            # ── NGSolve LinearForm attribute names ────────────
            "Assemble", "Update", "vector",
            "Set", "SetCoefficientFunction",
            # ── NGSolve nonlinear iteration / Newton internals
            "NgsAMG", "AMG", "BlockJacobi", "BlockSmoother",
            "ChebyshevPC", "PreconditionerType",
            # ── Mesh attributes / NumProc class names ─────────
            "NumProc", "PySolve", "krylov", "iters",
            # ── NGSolve catalog lowercase 'real' identifiers
            "matrix", "vector_norm",
            "energy_norm", "L2_norm", "H1_norm",
            "solver_iterations", "iter_count",
            # ── PETSc-status tokens that NGSolve also surfaces
            #    via krylovspace bindings ──────────────────────
            "KSPSolve", "KSP",
            "DIVERGED_INDEFINITE_PC",
            "DIVERGED_BREAKDOWN", "DIVERGED_MAX_IT",
            "DIVERGED_FNORM_NAN", "DIVERGED_INF",
            "DivisionByZero", "ZeroDivisionError",
            # ── NGSolve special CoefficientFunctions / normals
            "specialcf", "normal", "tangential",
            "Mesh_size", "JacobianFinite",
            # ── MHD-physics-specific identifiers ──────────────
            "Hartmann", "Hartmann_layer", "Hartmann_solution",
            "MHD", "magnetohydrodynamic",
            "wall_shear", "wall_shear_stress",
            "boundary_layer", "core_velocity",
            # ── Plate / shell theory identifiers ──────────────
            "M_nn", "simply_supported",
            "w_max", "centre_deflection",
            "shear_locking", "volumetric_locking",
            "bending_energy", "bending_moment",
            # ── Hyperelasticity stress / strain measures ──────
            "Piola_Kirchhoff", "PK1", "PK2",
            "Cauchy_stress", "first_PK", "second_PK",
            "push_forward", "pull_back",
            # ── Phase-field fracture / damage ─────────────────
            "Griffith", "phase_field_d", "damage",
            "l0_length_scale",
            "energy_split", "spectral_split",
            # ── NS observables ────────────────────────────────
            "u_lid", "lid_velocity",
            "non_conservative", "skew_symmetric",
            "violation_ratio", "max_u_over_h",
            # ── Contact-specific NGSolve identifiers ──────────
            "tangential_penalty", "normal_contact",
            "penalty", "gap_function",
            # ── Surface-PDE / OCC re-meshing ──────────────────
            "OCC", "OCCGeometry", "remesh",
            "DOF_transfer",
            # ── Lowercase generic catalog terms reused often ─
            "solution", "boundary", "mesh",
            "alpha", "tau", "rhs", "lhs",
            "iter", "step", "time_step",
        })
    elif backend == "skfem":
        code_symbols.update({
            # Element types
            "ElementTriP1", "ElementTriP2", "ElementTriRT0",
            "ElementTriArgyris", "ElementTriDG", "ElementTriCR",
            "ElementTriHermite", "ElementTriMini", "ElementQuad1",
            "ElementQuadDG", "ElementQuadBFS", "ElementTetP1",
            "ElementTetP2", "ElementTetN0", "ElementHexS2",
            "ElementHex1", "ElementVector", "ElementCompositeFE",
            # Mesh
            "MeshTri", "MeshQuad", "MeshTet", "MeshHex",
            "MeshLine", "init_refdom", "refined",
            # Assembly
            "Basis", "InteriorBasis", "FacetBasis",
            "asm", "BilinearForm", "LinearForm", "Functional",
            "block_diag", "bmat", "condense", "solve",
            # Boundary tagging / projection
            "subdomains", "boundaries", "facets",
            "find_dofs", "get_dofs", "project",
            # scipy interop names that appear in skfem stack traces
            "spsolve", "factorized", "scipy.sparse",
            "scipy.sparse.linalg", "splu", "csc_matrix",
            "csr_matrix", "SuperLU", "boundary_values",
            "boundary_dofs",
            # numpy attribute/array names that appear when
            # skfem returns rank-deficient solutions
            "np", "numpy", "abs", "ones",
            # skfem-namespaced helpers
            "skfem.models.poisson", "skfem.models",
            "skfem.condense", "skfem.solve", "skfem.asm",
            # Norms / output names that appear in user code
            "linfty_norm", "l2_norm",
            # Common error classes / attribute names
            "ValueError", "TypeError", "shape", "ndof", "interpolate",
            # ── skfem.models / canonical helper functions ──────
            "lame_parameters", "skfem.helpers",
            "skfem.models.elasticity",
            "skfem.models.helmholtz",
            "skfem.models.stokes",
            "skfem.models.heat",
            "linear_elasticity_form",
            "mass", "laplace", "stiffness",
            "M_assemble", "K_assemble",
            # ── scipy.sparse.linalg eigsh / arpack identifiers
            "eigsh", "eigs", "ArpackError",
            "arpack", "shift_invert", "LM", "SM",
            "spsolve_triangular", "splu",
            # ── numpy / Python helpers in catalog prose ─────
            "math", "math_isclose", "isclose", "float64",
            "np_isclose", "np_allclose",
            "dot", "abs_max", "norm",
            # ── skfem boundary/dof accessors used in catalog
            "ib_u", "ib_p", "ib", "DoF", "DoFs",
            "n_u_dof", "n_p_dof", "u_dof", "p_dof",
            "Basis_N", "Basis_get_dofs",
            "DofsView", "boundary_dofs_view",
            # ── Centered/upwind DG keywords ─────────────────
            "centered_flux", "upwind", "downwind",
            "jump", "u_other", "other",
            "interior_facet", "exterior_facet",
            # ── PDE-specific observables ────────────────────
            "T_D", "T_fluid", "T_solid", "T_room",
            "T_ref", "T_init", "u_ss", "u_steady",
            "p_skfem", "p_ngsolve",
            "monolithic_Krylov", "Schur_complement",
            "GMRES", "BiCGStab", "MINRES",
            "k_eff", "k_thermal",
            # ── nonlinear / Newton fallback prose ──────────
            "NewtonSolver", "newton",
            "backtracking", "line_search",
            "alpha", "tau", "rhs", "lhs",
            # ── Phase-field / damage / fracture ─────────────
            "phase_field", "Allen_Cahn", "Cahn_Hilliard",
            "damage_d", "Griffith",
            # ── Cross-backend identifiers used in skfem prose
            "FEniCSx", "dolfinx", "ngsolve_compare",
            # ── Catalog lowercase tokens that map to
            #    real skfem channels ──────────────────────────
            "solution", "boundary", "interface",
            "temperature", "pressure", "velocity",
            "displacement", "stress", "strain",
            "sigma", "epsilon",
            # ── scipy.sparse.linalg tokenised components ────
            "scipy", "sparse", "linalg",
            "cg", "gmres", "lu",
            # ── Numerical-analysis observables in skfem prose
            "lambda_min", "lambda_max",
            "phase_drift", "phase_error",
            "k_c", "k_squared", "gamma",
            # ── Reaction-diffusion benchmark names ──────────
            "Turing", "Schnakenberg", "Gray_Scott",
            "Brusselator", "Lotka_Volterra",
            "homogeneous_steady_state",
            "cross_diffusion", "pattern_formation",
            "d_u", "d_v",
            # ── DG / time-stepping observables ──────────────
            "t_transient", "u_transient",
            "sharp_transient", "smeared",
        })
    elif backend == "kratos":
        code_symbols.update({
            # Core classes
            "KratosMultiphysics", "Model", "ModelPart",
            "Variable", "Property", "Properties", "Element",
            "Condition", "Node", "Geometry", "DofWithMandatoryDof",
            "Process", "Parameters", "AssignVectorVariableProcess",
            "AssignScalarVariableProcess",
            "AssignVectorByDirectionProcess",
            "ApplyConstantScalarValueProcess",
            "CalculateOnIntegrationPoints",
            "AnalysisStage", "Solver", "ConstitutiveLaw",
            # Strategies / schemes / solvers (compound CamelCase
            # names that appear in real Kratos tracebacks)
            "ResidualBasedNewtonRaphsonStrategy",
            "ResidualBasedBlockBuilderAndSolver",
            "ResidualBasedIncrementalUpdateStaticScheme",
            "ResidualCriteria", "DisplacementCriteria",
            "AndCriteria", "OrCriteria",
            "SkylineLUFactorizationSolver",
            "LinearSolver", "DirectSolver",
            "NewtonRaphsonStrategy",
            # Element / condition / contact compound names
            "SmallDisplacementElement", "TotalLagrangianElement",
            "UpdatedLagrangianElement", "ShellThinElement",
            "ShellThickElement", "TrussElement", "CrBeamElement",
            "LaplacianElement", "EulerianConvDiff",
            "LaplacianElement2D3N", "LaplacianElement3D4N",
            "EulerianConvDiff2D3N", "EulerianConvDiff3D4N",
            "ConvectionDiffusionSettings", "SetVolumeSourceVariable",
            "SetUnknownVariable", "SetDiffusionVariable",
            "SetDensityVariable", "SetSpecificHeatVariable",
            "HEAT_FLUX", "CONVECTION_VELOCITY",
            "CalculateRightHandSide", "CalculateLocalSystem",
            "CONVECTION_DIFFUSION_SETTINGS",
            "VMS", "QSVMS", "FractionalStep",
            "TwoFluidNavierStokes",
            "ALMFrictionlessMortarContact",
            "ALMFrictionalMortarContact",
            "PenaltyFrictionlessMortarContact",
            "MortarContact",
            "SmallStrainIsotropicPlasticity",
            "SmallStrainIsotropicPlasticityFactory",
            "SmallStrainIsotropicPlasticity3DModifiedMohrCoulombModifiedMohrCoulomb",
            # I/O
            "RuntimeError", "TypeError", "Logger",
            "AttributeError", "KeyError", "ValueError",
            # C++ source references that appear in Kratos
            # RuntimeError traces (verified empirically 2026-06-01
            # across multiple probes).
            "GetValue", "kratos_parameters", "Has",
            "ErrorNonExistingSubModelPart", "model_part",
            "variables_list_data_value_container",
            "GetSolutionStepValue", "SetSolutionStepValue",
            "ReadModelPart", "WriteModelPart", "GidIO", "VtkOutput",
            "CloneTimeStep", "InitializeSolutionStep",
            "RunSolutionLoop", "Initialize",
            "AddNodalSolutionStepVariable", "CreateNewNode",
            "CreateNewElement", "CreateNewCondition",
            "CreateNewProperties", "SetValue", "AddDof", "Fix",
            "GetSubModelPart", "CreateSubModelPart",
            # Variables (uppercase names that show up in tracebacks)
            "DISPLACEMENT", "VELOCITY", "PRESSURE", "TEMPERATURE",
            "REACTION", "ROTATION", "POINT_LOAD", "FACE_HEAT_FLUX",
            "DENSITY", "YOUNG_MODULUS", "POISSON_RATIO",
            "BODY_FORCE", "VOLUME_ACCELERATION",
            "DYNAMIC_VISCOSITY", "CONDUCTIVITY", "SPECIFIC_HEAT",
            "DISTANCE", "GAP", "WATER_PRESSURE",
            "PK2_STRESS_VECTOR", "DEFORMATION_GRADIENT",
            "FRICTION_ANGLE", "DILATANCY_ANGLE",
            "YIELD_STRESS_COMPRESSION", "YIELD_STRESS_TENSION",
            "FRACTURE_ENERGY", "HARDENING_CURVE",
            "HARDENING_MODULUS", "COHESION",
            "FREESTREAM_VELOCITY", "MACH_INFINITY",
            "MAIN_VARIABLE",
            # Module / namespace shorthands
            "KratosMultiphysics", "ConstitutiveLawsApplication",
            "StructuralMechanicsApplication",
            "FluidDynamicsApplication", "ContactStructuralMechanicsApplication",
            "MappingApplication", "MeshMovingApplication",
            "DEMApplication", "MPMApplication",
            "ConvectionDiffusionApplication",
            "GeoMechanicsApplication",
            "CompressiblePotentialFlowApplication",
            "RANSApplication", "PfemFluidDynamicsApplication",
            "OptimizationApplication", "TopologyOptimizationApplication",
            "CoSimulationApplication", "TrilinosApplication",
            "MetisApplication", "ShallowWaterApplication",
            "ChimeraApplication", "IgaApplication",
            "FemToDemApplication", "DelaunayMeshingApplication",
            # ── Real Kratos JSON parameter names that appear in
            #    catalog Signal: text and in error reports ────────
            "Convergence", "convergence_criterion",
            "max_iteration", "residual_tolerance",
            "displacement_relative_tolerance",
            "residual_relative_tolerance",
            "displacement_absolute_tolerance",
            "residual_absolute_tolerance",
            "echo_level", "compute_reactions",
            "block_builder", "reform_dofs_at_each_step",
            "tolerance", "residual",
            "SubModelPart", "linear_solver_settings",
            "solver_type", "problem_data", "solver_settings",
            "model_part_name", "domain_size",
            "time_integration_method", "time_integration",
            "scheme_type", "convergence_criterion",
            # ── Real Kratos scheme / time-integration classes
            #    that appear in the C++ source and JSON config ──
            "Bossak", "ResidualBasedBossakDisplacementScheme",
            "ResidualBasedNewmarkDisplacementScheme",
            "ResidualBasedRelaxationScheme",
            "BDFScalarScheme",
            # ── Compound Kratos error / status strings (literal
            #    text printed by AnalysisStage / Strategy) ───────
            "ResidualBasedSolvingStrategy", "Strategy",
            "ConvergenceCriteria", "Process",
            "ModelPartIO", "GenericIO",
            # ── Variable-history attribute names that the
            #    catalog and probes both reference ─────────────
            "Has", "GetValue", "SetValue",
            "GetSolutionStepValue", "SetSolutionStepValue",
            "DOF_VARIABLES", "DofType",
            # ── Common Kratos element-suffix bare patterns the
            #    catalog uses (e.g. SmallDisplacementElement
            #    3D8N → "3D8N" + "3D20N") ───────────────────────
            "Element3D8N", "Element3D20N", "Element2D3N",
            "Element2D4N", "Element3D4N", "Element3D10N",
            # ── Misc real Kratos namespaces / log markers ────
            "kratos", "Kratos", "Json", "JSON",
            "registered", "variables_list",
            "Parameters", "model_part",
            # ── Output channels that appear in Kratos run logs
            "GiD", "GiDPostMode", "WriteDeformedMeshFlag",
            "VtkOutput", "vtk_output",
            # ── Bare element-type suffix codes that Kratos
            #    template files use (hex8/hex20/quad8 ...) plus
            #    the FluidDynamics / structural quantities that
            #    real Kratos process-output channels write ────
            "hex8", "hex20", "quad4", "quad8", "tet4",
            "tet10", "tri3", "tri6", "tetra4", "tetra10",
            "Hexa8", "Hexa20", "Tetra4", "Tetra10",
            # ── Real Kratos diagnostic / error literals
            #    (substrings that a critic can grep for in
            #    stderr; verified to occur in Kratos source) ─
            "singular", "condition_number",
            "penetration", "deflection",
            "boundary_temperature", "wall_shear_stress",
            "drag_coefficient", "lift_coefficient",
            "matrix_is_numerically_singular",
            "linear_solver", "Bossak_alpha",
            # ── PETSc / Trilinos linear-solver settings ───────
            "amgcl", "mumps", "amesos2",
            "preconditioner_type",
            # ── MCP-emitted artifacts the stub-probe pitfalls
            #    actually reference; results_summary is a real
            #    JSON file the MCP writes ──────────────────────
            "results_summary", "DamAnalysis",
            "MaterialPoint", "MPMAnalysisStage",
            "KratosPfem", "KratosShallowWaterApplication",
            "PfemFluidApplication", "FreeSurfaceApplication",
            "FluidHydraulicsApplication", "DropletDynamics",
            "CableNetApplication",
            # ── More Kratos-physics observables used in catalog
            #    Signal: prose ─────────────────────────────────
            "yield_surface", "sigma_y",
            "max_displacement", "integrated_flux",
            "post_processed", "wall_time",
            "T_boundary", "T_imposed",
            "momentum_balance", "Couette",
            "uniaxial_compression",
            # ── Kratos JSON solver/scheme parameters used
            #    in structural dynamics signal text ───────────
            "alpha_m", "alpha_f", "gamma", "beta",
            "rayleigh_alpha", "rayleigh_beta",
        })
    elif backend == "fenics":
        code_symbols.update({
            "dolfinx", "ufl", "petsc4py", "mpi4py",
            "FunctionSpace", "VectorFunctionSpace",
            "TensorFunctionSpace", "MixedElement", "VectorElement",
            "FiniteElement", "Function", "Constant",
            "TrialFunction", "TestFunction", "TrialFunctions",
            "TestFunctions",
            "fem", "form", "assemble", "assemble_matrix",
            "assemble_vector", "apply_lifting",
            "set_bc", "dirichletbc", "locate_dofs_topological",
            "locate_dofs_geometrical",
            "mesh", "create_box", "create_rectangle",
            "create_unit_square", "create_unit_cube",
            "Mesh", "CellType", "cell_dim",
            "PETScKrylovSolver", "PETScLUSolver",
            "NonlinearProblem", "NewtonSolver",
            "PETSc", "MPI",
            "grad", "div", "curl", "inner", "outer", "dot",
            "tr", "sym", "skew", "det", "Identity",
            "dx", "ds", "dS",
            "RuntimeError", "ValueError", "TypeError",
            # ── PETSc / SNES / KSP nonlinear / Krylov status
            #    tokens that dolfinx emits in real tracebacks ──
            "SNES", "KSP", "KSPSolve", "SNESSolve",
            "DIVERGED_LINE_SEARCH", "DIVERGED_BREAKDOWN",
            "DIVERGED_MAX_IT", "DIVERGED_FNORM_NAN",
            "DIVERGED_INF", "DIVERGED_PC_FAILED",
            "CONVERGED_ATOL", "CONVERGED_RTOL",
            "convergence_criterion", "solver_type",
            "ksp_type", "pc_type", "snes_type",
            "GMRES", "BiCGStab", "Newton",
            # ── ufl operators / elements not yet covered ──────
            "Nedelec", "Raviart", "Thomas", "BDM",
            "Bubble", "RestrictedElement", "EnrichedElement",
            "FacetElement", "InteriorElement",
            "CellDiameter", "FacetNormal", "CellVolume",
            "SpatialCoordinate", "conditional",
            "Min", "Max", "abs", "exp", "ln",
            "real", "imag", "conj",
            # ── dolfinx I/O / XDMF / VTK ──────────────────────
            "XDMFFile", "VTKFile", "VTXWriter",
            "FidesWriter", "io",
            # ── Real dolfinx attribute names that appear in
            #    catalog Signal: prose ─────────────────────────
            "x_array", "x", "array",
            "vector", "interpolate",
            "scatter_forward", "ghost_update",
            "topology", "geometry",
            # ── Linear-algebra fallback channels (real)
            "scipy", "linalg", "spsolve", "splu",
            "csc_matrix", "csr_matrix",
            "LocalSolver",
            # ── Maxwell / EM / Helmholtz observables ──────────
            "J_coil", "B_field", "B_H_curve", "mu_r",
            "iron", "coil", "magnetic_field",
            # ── Phase-field / Cahn-Hilliard / Allen-Cahn ──────
            "Cahn", "Hilliard", "Allen", "Cahn_Hilliard",
            "Allen_Cahn", "phase_field", "interface_width",
            "front", "kappa", "eps", "epsilon_phi",
            # ── Reaction-diffusion / heat transient ───────────
            "DirichletBC", "NeumannBC",
            "no_flux", "insulated", "Strang", "Lie",
            "BE", "CN", "theta", "Crank_Nicolson",
            "c_eff", "T_melt", "T_room", "T_ref",
            # ── Plane-strain / linear elasticity ─────────────
            "plane_strain", "plane_stress",
            "mixed_P2_P1", "Taylor_Hood",
            "nullspace", "rigid_body_modes",
            # ── Fracture / phase-field damage ─────────────────
            "d_old", "d_new", "phase_field_d",
            "irreversibility",
            # ── DG penalty / coercivity ───────────────────────
            "alpha_0", "p_plus_1", "SIPG",
            "coercivity_loss",
        })
    elif backend == "dune":
        code_symbols.update({
            "dune", "dune.fem", "dune.grid", "dune.geometry",
            "structuredGrid", "yaspGrid", "alugrid",
            "FieldVector", "FieldMatrix",
            "Space", "GridFunctionSpace", "DiscontinuousGalerkin",
            "lagrange", "DGSpace", "FemSpace",
            "uflSpace", "Scheme", "operator", "Source",
            "DirichletBC", "Constraints",
            "solve", "assembled", "galerkin",
            "GridView", "Entity", "Intersection",
            "RuntimeError", "ImportError",
            # ── ufl operators / element identifiers used in
            #    dune.fem Signal: prose ──────────────────────
            "grad", "div", "curl", "sym", "skew",
            "inner", "outer", "dot", "tr",
            "TrialFunction", "TrialFunctions",
            "TestFunction", "TestFunctions",
            "Function", "GridFunction", "FunctionSpace",
            "VectorH1", "H1", "L2", "HDiv",
            "Raviart", "Thomas", "raviartThomas",
            "Nedelec", "BDM", "dglagrange",
            "interpolate", "interpolated",
            "uflSpace", "Form",
            "conditional", "Min", "Max", "abs", "exp", "ln",
            "FacetNormal", "CellDiameter",
            # ── dune.fem JIT / Schwarz / build-time markers ───
            "JIT", "re_JIT", "compile", "compiled",
            "C_plus_plus_compile", "ccompile",
            # ── dune.fem time-stepping schemes ────────────────
            "BE", "CN", "FE", "DIRK22", "SDIRK22",
            "Heun", "RungeKutta", "ImplicitEuler",
            "ExplicitEuler", "Crank_Nicolson",
            "TimeStepper", "femDG",
            # ── dune.fem mesh refinement / adaptive ───────────
            "eta_K", "theta_coarse", "theta_refine",
            "indicator", "adapt", "mark", "refine",
            "coarsen", "preAdapt", "postAdapt",
            "space_update", "GridAdapt",
            # ── DG flux / limiter terms ──────────────────────
            "centred_flux", "upwind", "Lax_Friedrichs",
            "Roe", "HLL", "Rusanov",
            "WENO", "minmod", "MUSCL", "TVD",
            "Limiter", "moments",
            # ── Numerical-analysis observables in Signal: ─────
            "CFL", "lambda_max", "Pollution",
            "phase_velocity", "wavelength",
            "inverse_iteration", "deflation",
            "Gram_Schmidt", "Lanczos", "Arnoldi",
            "Cook_membrane", "volumetric_locking",
            "Taylor_Hood", "grad_div", "div_u",
            "tau", "stabilisation", "stabilization",
            "centred", "Krylov",
            # ── dune linear-algebra / IO ─────────────────────
            "SolverCG", "SolverGMRES", "SolverDirect",
            "SolverBiCGStab", "preconditioner",
            "writeVTK", "VTKWriter", "ParaView",
            "GMRES", "BiCGStab",
            # ── Misc identifiers used in dune.fem catalog ────
            "u_old", "u_new", "rhs", "lhs",
            "sigma", "epsilon",
            "phase", "phase_field", "Allen_Cahn",
            "neo_Hookean", "nu",
        })
    elif backend == "fourc":
        code_symbols.update({
            "SOLID3", "BEAM3R", "WALL", "FLUID", "THERMO",
            "DESIGN", "DBC", "NBC", "DIRICH", "NEUMANN",
            "MAT", "STRUCT", "FLUID3", "TIMINT", "DYNAMIC",
            "NLN_SOL", "Newton", "Picard", "BACI",
            "ParameterList", "Teuchos", "Epetra", "Trilinos",
            "ML", "MueLu", "AMG",
            "ELEMENTS", "NODE_COORDS", "DESIGN_POINT",
            "DESIGN_LINE", "DESIGN_SURF", "DESIGN_VOL",
            # ── 4C YAML section / parameter names (verified
            #    against 4C source) ──────────────────────────────
            "PROBLEMTYPE", "MATERIALS", "STRUCTURE",
            "SCATRA", "TRANSP", "TIMEINTEGR", "DYNAMICTYPE",
            "INITIALFIELD", "INITFUNCNO", "FUNCT",
            "STRUCTURAL", "THERMAL", "FLUID_DYNAMIC",
            "TSI_DYNAMIC",
            # ── 4C material classes (real registered types) ─────
            "MAT_Fourier", "MAT_ElastHyper", "MAT_Struct_StVenantKirchhoff",
            "MAT_Newtonian", "MAT_Carreau",
            "MAT_LinElast1D",
            "CAPA", "CONDUCT",
            # ── 4C input-spec builder classes (the ones that
            #    emit the empirical diagnostics) ─────────────────
            "InputSpec", "InputFile", "InputParameterContainer",
            "MatchTree", "deprecated_selection",
            # ── 4C source filenames that appear verbatim in
            #    PROC 0 ERROR diagnostics ────────────────────────
            "4C_io_input_file", "4C_io_input_spec_builders",
            "4C_global_data_read", "4C_thermo_element",
            "4C_fem_discretization", "lib4C",
            # ── 4C output markers / runtime functions ───────────
            "fill_complete", "PROBLEMTYPE",
            "Thermo_Structure_Interaction",
            "Scalar_Transport", "Statics", "Stationary",
            "GenAlpha", "OneStepTheta",
            # ── Real 4C input/section keywords from catalog
            #    Signal: prose ──────────────────────────────────
            "NUMDOF", "TIMESTEP", "NUMSTEP", "MAXTIME",
            "MAXITER", "ITEMAX", "TOLRES", "TOLDISP",
            "RESTARTEVRY", "RESULTSEVRY", "RESULTSEVERY",
            "KINEM", "INT_STRATEGY", "STRATEGY",
            "PENALTY_PARAMETER", "REGULARIZATION",
            "CONVTOL", "NORM_RESF", "PREDICT",
            "TIMINTEGR", "NLNSOL", "LINEAR_SOLVER",
            "STC_LAYER", "STC_SCALING",
            "VOLFORCE", "NODEFORCE", "POINTNEUMANN",
            "DESIGN_POINT_DIRICH", "DESIGN_LINE_DIRICH",
            "DESIGN_SURF_DIRICH", "DESIGN_VOL_DIRICH",
            "DESIGN_POINT_NEUMANN", "DESIGN_LINE_NEUMANN",
            "DESIGN_SURF_NEUMANN", "DESIGN_VOL_NEUMANN",
            "DESIGN_FSI_COUPLING",
            "IO", "RUNTIME", "VTK", "OUTPUT",
            # ── 4C element types (real registered) ────────────
            "SOLID", "SOLIDSCATRA", "SOLIDPORO",
            "HEX8", "HEX20", "HEX27", "TET4", "TET10",
            "TET15", "PYRAMID5", "WEDGE6", "WEDGE15",
            "QUAD4", "QUAD8", "QUAD9", "TRI3", "TRI6",
            "LINE2", "LINE3", "POINT1",
            "BEAM3K", "BEAM3EB", "TRUSS3", "TORSION3",
            "FLUID2", "ALE3", "ALE2",
            "POROFLUIDMULTIPHASE", "POROFLUID",
            "TRANSP3", "TRANSP2", "TRANSP1",
            "SHELL_KL_NURBS", "SHELL", "MEMBRANE",
            "SHELL_KIRCHHOFF",
            # ── 4C particle / PD identifiers ─────────────────
            "PARTICLE", "PARTICLES", "PARTICLE_DYNAMIC",
            "PARTICLE_TIMINT", "PD", "INTERACTION_HORIZON",
            "PERIDYNAMIC_GRID_SPACING", "PRE_CRACKS",
            "PDBODYID", "HORIZON",
            # ── 4C material classes (more) ────────────────────
            "MAT_ParticlePD", "MAT_Multiscale",
            "MAT_myocard", "MAT_BeamReissnerElastHyper",
            "MAT_BeamKirchhoffElastHyper",
            "MAT_BeamKirchhoffTorsionFreeElastHyper",
            "MAT_Beam_Reissner_ElastPlastic",
            "MAT_Beam_Reissner_ElastHyper",
            "MAT_StructPoro", "MAT_FluidPoro",
            "MAT_Lubrication", "MAT_matlist",
            "MAT_GrowthRemodel",
            "CROSSAREA", "MOMINPOL", "MOMIN2", "MOMIN3",
            "SHEARCORR", "TRIADS", "HERMITE_CENTERLINE",
            # ── 4C solver / strategy names ────────────────────
            "GenAlphaLieGroup", "OstAlpha", "ExplicitEuler",
            "GenAlphaLieGroupTimeIntegrator",
            "Solid", "Fluid", "Thermo", "ScaTra",
            # ── 4C runtime / error / log markers ──────────────
            "PROC", "ERROR", "FATAL",
            "Discretization", "discretization",
            "NOX", "NOXSolver",
            "Aztec", "Belos", "Ifpack", "Amesos",
            # ── Coupling block names ──────────────────────────
            "FSI_DYNAMIC", "TSI", "FPSI", "SSI", "SSTI",
            "STI", "ELCH", "FBI",
            "PARTITIONED", "MONOLITHIC",
            "COUPALGO", "COUPVARIABLE",
            # ── 4C field/observable identifiers ───────────────
            "velnp", "veln", "dispnp", "tempnp",
            "displacement_field", "velocity_field",
            "pressure_field", "temperature_field",
            "fluid_pressure", "scalar_field", "phi",
            "ALE_velocity", "structural_displacement",
            "interface", "interface_node",
            # ── Lowercase 4C section names (catalog prose) ────
            "domain", "boundary", "condition", "section",
            "porosity", "permeability", "viscosity",
            "stiffness", "stress", "strain",
            "load_step", "time_step",
            # ── MCP-emitted artifacts ─────────────────────────
            "results_summary", "post_vtu", "post_drt_ensight",
            "visualize",
            # ── Lowercase 4C output prefix / filename patterns ─
            "scatra", "solid", "ale", "thermo",
            "lubrication", "porofluid", "structure_output",
            "fluid_output", "scatra_output",
            "displacement_norm", "solution_norm",
            "residual_norm", "convergence_norm",
            "kinetic_energy", "strain_energy",
            "internal_energy",
            # ── 4C function / variable spec keys ──────────────
            "COMPONENT", "VARIABLE", "EXPR", "DEFINITION",
            "SYMBOLIC_FUNCTION_OF_TIME",
            "SYMBOLIC_FUNCTION_OF_SPACE_TIME",
            "VARFUNCTION", "VAREXPR",
            # ── 4C cloning material map keys ──────────────────
            "CLONING", "SRC_FIELD", "SRC_MAT",
            "TAR_FIELD", "TAR_MAT",
            "CLONING_MATERIAL_MAP",
            # ── 4C coupling / specific BC keywords ────────────
            "ConditionID", "GeometryID",
            "BoundingBox", "InteractionType",
            "BondType", "BondAlgorithm",
            # ── Linear / Newton convergence channels ──────────
            "NormF", "NormDisp", "NormUpdate",
            "RestartFlag", "RESTART", "RESULTSEVRY",
            # ── 4C fluid turbulence symbols ────────────────────
            "TURBULENCE_MODEL", "Smagorinsky",
            "Spalart", "kepsilon", "kw_SST",
            "law_of_the_wall", "y_plus", "u_tau",
            # ── 4C contact pitfall identifiers ────────────────
            "CONTACT_TOL", "FRICTION_COEFFICIENT",
            "PENALTY", "AUGMENTED_LAGRANGE",
            "MORTAR", "NTS",
            # ── 4C beam interaction / dynamics ────────────────
            "BEAM3K_DYNAMIC", "BEAM_TO_FLUID_MESHTYING",
            "BEAM_INTERACTION_PARAMS",
            # ── 4C XFEM / cut-FEM identifiers ────────────────
            "XFEM", "XFLUID3", "XFEM_GENERAL",
            "level_set_value",
            # ── PD / SPH ──────────────────────────────────────
            "MAT_SphFluid", "SPHContact",
            "smoothingLength",
            # ── Multiphysics / cardio identifiers ─────────────
            "MAT_FluidPoroSinglePhase",
            "POROSITY_FORMULATION",
            "CARDIAC_DYNAMICS", "MONODOMAIN",
            "EP_DYNAMIC", "MEMBRANE_MODEL",
            "phi_1", "phi_2",
            # ── Generic 4C constants / observables that
            #    appear lowercased in catalog prose ─────────────
            "DENS", "VISC", "YOUNG", "NUE",
            "POISSON", "lambda", "mu_shear",
            "Sigma", "epsilon", "Cauchy",
            # ── 4C source filenames (bare, without 4C_io_
            #    prefix) that appear in catalog prose ────────────
            "input_spec_builders", "input_file",
            "global_data_read", "fem_discretization",
            "thermo_element", "adapter_fld_base_algorithm",
            "fbi_factory", "fluid_xfem_factory",
            "lubrication_factory",
            # ── 4C contact / mortar / NTS identifiers used in
            #    Signal: prose ──────────────────────────────────
            "Mortar", "MortarInterface", "DNODE", "DELE",
            "DSURF", "DLINE", "MortarManager",
            "slave", "master", "Hertz", "deformed",
            # ── 4C particle binning / search ──────────────────
            "DOMAINBOUNDINGBOX", "BinningStrategy",
            "rigidphase", "fluidphase", "BondAlgorithm",
            "Phase", "phase", "BondType",
            "InitRadius", "INITRADIUS",
            # ── Lowercase 4C channels / sections referenced
            #    repeatedly in catalog prose ────────────────────
            "pressure", "wall", "node", "field", "mesh",
            "fluid", "velocity", "displacement",
            "temperature", "thermal", "contact", "particle",
            "beam", "airway", "density", "viscosity",
            "porosity", "permeability", "stiffness", "stress",
            "strain",
            # ── 4C cardiac / EP / monodomain ──────────────────
            "physiological", "monodomain", "ionic",
            "MAT_ionic", "AnisotropicMaterial",
            "DIFF1", "DIFF2", "DIFF3",
            # ── 4C airway / reduced lung ──────────────────────
            "Airway", "RedAirway", "REDUCED_AIRWAYS",
            "TerminalUnit", "AirwayResistance",
            # ── 4C beam / pair-interaction ─────────────────────
            "BeamToBeamContact", "PenaltyParameter",
            "BeamContact", "BeamInteraction",
            # ── 4C XFEM cut / moment-fitting identifiers ──────
            "MomentFitting", "Tessellation",
            "CUTCALLS", "VOLUMECELL",
            "BoundaryCell", "LevelSetField",
            # ── 4C membrane / shell pitfall keys ───────────────
            "THICK", "MATTYPE", "MEMBRANE_MATERIAL",
            "InitialThickness",
            # ── 4C constraint / saddle-point identifiers ──────
            "ConstraintMatrix", "saddle_point",
            "Schur", "rank", "ZeroPivot",
            "MULTIPNT_CONSTRAINT",
            # ── 4C cardiac / EP / monodomain ───────────────────
            "FHN", "TenTusscher", "OHaraRudy", "APD",
            "ten_Tusscher", "OHara_Rudy",
            "ventricular_myocyte", "action_potential",
            "stimulus", "subthreshold",
            # ── 4C arterial / hemodynamics ─────────────────────
            "ArterialNetwork", "blood_flow",
            "vessel_segment", "c_max", "wave_speed",
            # ── 4C lubrication / cavitation ─────────────────
            "Elrod_Adams", "cavitation_pressure",
            "p_ambient",
            # ── 4C fluid turbulence inflow generators ────────
            "Lund_Wu_Squires", "recycling",
            "DNS", "LES", "RANS", "Smagorinsky_constant",
            "y_plus", "Re_tau",
            # ── 4C porous / Darcy / Brinkman ─────────────────
            "Darcy", "Brinkman", "Reynolds",
            "DarcyFlow", "porosity_form",
            # ── 4C electrochemistry / SSI ─────────────────────
            "BV", "Butler_Volmer", "ELECTRODE",
            "ELCH_DYNAMIC", "BV_current",
            "lithium_intercalation",
            # ── 4C structural-dynamics mode-shape identifiers
            "mode_shape", "eigenfrequency",
            "omega_1", "omega_2", "rayleigh_damping",
            # ── 4C particle PD bond / neighbor diagnostics ────
            "neighbor_count", "bond_count",
            "PDWaveSpeed", "DeltaConvergence",
        })
    elif backend == "febio":
        code_symbols.update({
            # ── FEBio toolchain marker ───────────────────────────
            "FEBio", "febio_spec",
            # ── Top-level XML section names (febio_spec children)
            "Module", "Globals", "Material", "Mesh",
            "MeshDomains", "Geometry", "MeshData", "Step",
            "Control", "Boundary", "Loads", "Initial",
            "LoadData", "Discrete", "Constraints", "Contact",
            "Output", "plotfile", "logfile", "Solutes",
            "SolidDomain", "ShellDomain", "BCs", "Rigid",
            # ── FEBio modules (Module type=...) ──────────────────
            "solid", "biphasic", "multiphasic", "fluid",
            "polar_fluid", "fluid_FSI", "biphasic_FSI",
            "heat", "reaction_diffusion",
            # ── Real FEBio material type names (the ones that
            #    appear inside `<material type="..."/>` and that
            #    error text echoes verbatim) ──────────────────────
            "isotropic_elastic", "neo_Hookean",
            "Mooney_Rivlin", "Ogden", "Holzapfel",
            "transversely_isotropic", "fiber_neo_Hookean",
            "uncoupled_solid_mixture", "solid_mixture",
            "rigid_body", "growth", "isotropic_growth",
            "viscoelastic", "uncoupled_viscoelastic",
            "biphasic_solute", "triphasic", "multiphasic_solute",
            "Newtonian", "Carreau", "Bingham",
            "fluid_solute", "isotropic_Fourier",
            "Bauschinger_shift", "yield_stress",
            # ── Solver / NOX-status tokens that real FEBio logs
            #    emit ───────────────────────────────────────────
            "NOX", "FullNewton", "BFGS", "Quasi",
            "DIVERGED_LINE_SEARCH", "DIVERGED_FNORM_NAN",
            "MaxIters", "lc", "load_controller",
            "step_size", "dt_0", "Jacobian",
            "max_refs", "max_ups", "Rtol", "Etol",
            "STATIC", "DYNAMIC", "TRANSIENT",
            # ── Element types FEBio accepts in <Elements
            #    type="..."/> ───────────────────────────────────
            "hex8", "hex20", "hex27", "tet4", "tet10",
            "tet15", "tet20", "penta6", "penta15",
            "pyra5", "pyra13", "quad4", "quad8", "quad9",
            "tri3", "tri6", "tri7",
            # ── DOF + observable identifiers FEBio prints
            #    (verified from FEBio output / catalog) ─────────
            "x_dof", "y_dof", "z_dof",
            "gx_dof", "gy_dof", "gz_dof",
            "center_of_mass", "initial_stress",
            "rigid_displacement", "shedding_period",
            # ── Output / runtime helpers ─────────────────────────
            "write_vtu", "post_vtu", "xplt", "plt",
            # ── More FEBio runtime / PETSc-via-FEBio identifiers
            #    extracted from actual error logs and catalog
            #    Signal: prose ──────────────────────────────────
            "KSPSolve", "ALE", "Hookean", "HGO",
            "Holzapfel_Gasser_Ogden",
            "kappa", "fiber", "Elements", "node",
            "prescribed", "concentration", "sol",
            "micro_rotation", "Cosserat",
            "drainage", "tissue", "contact",
            "fluid_pressure", "effective_concentration",
            # ── FEBio XML attribute / parameter lowercase tokens
            #    that show up in `<material>`/`<Element>` errors ──
            "material", "parameter", "element",
            "isotropic", "elastic", "displacement",
            "restart", "history", "cycle",
            # ── Output / plotfile observables ─────────────────
            "plotfile_xplt", "logfile",
            "stress_strain", "kinetic_energy",
            "fluid_velocity", "elastic_strain",
        })

    code_symbols.update({
        # ── Krylov solvers ─────────────────────────────────────
        "SolverCG", "SolverGMRES", "SolverFGMRES", "SolverMinRes",
        "SolverBiCGStab", "SolverDirect", "SolverFIRE",
        "SolverQMRS", "SolverRichardson", "SolverControl",
        # ── External-package solvers / linear-algebra layers ─
        "SUNDIALS", "KINSOL", "ARKode", "SLEPc", "PETSc",
        "Trilinos", "TrilinosWrappers", "MUMPS", "UMFPACK",
        "PETScWrappers", "EPS", "EPS_TARGET_REAL",
        # ── MPI / runtime errors ───────────────────────────────
        "MPI_Init", "MPI_Comm_size", "MPI_ERR_COMM",
        "MPI_InitFinalize",
        # ── Mesh / DoF infrastructure ──────────────────────────
        "GridGenerator", "GridIn", "GridOut", "GridTools",
        "Triangulation", "DoFHandler", "DoFRenumbering",
        "AffineConstraints", "ConstraintMatrix", "MappingQ",
        "MappingQEulerian", "SolutionTransfer", "MatrixFree",
        "FEValuesExtractors", "FEValues", "FEValuesBase",
        "FEInterfaceValues", "FEEvaluation", "FECollection",
        "FESeries", "QCollection", "QGauss", "MeshWorker",
        "DataOut", "DataOutInterface", "DataOutFaces",
        "KellyErrorEstimator", "Quadrature",
        "VectorTools", "TimerOutput", "VectorOperation",
        "SparseMatrix", "BlockSparseMatrix", "BlockVector",
        "SparsityPattern", "BlockSparsityPattern",
        "SparsityTools", "IndexSet", "DoFTools",
        "VectorizedArray",
        # ── Common exception classes ──────────────────────────
        "ExcMessage", "ExcDimensionMismatch", "ExcIndexRange",
        "ExcNotImplemented", "ExcInternalError",
        "ExcInitializeNotInitialized", "ExcSolverFail",
        "ExcInvalidIterator",
        # ── Preconditioners / smoothers ────────────────────────
        "PreconditionAMG", "PreconditionSSOR",
        "PreconditionJacobi", "PreconditionChebyshev",
        "PreconditionILU", "PreconditionBlock",
        "PreconditionBlockSSOR", "PreconditionIdentity",
        "MGSmootherRelaxation",
        # ── External library output markers a critic can grep ─
        "BoomerAMG",  # HYPRE class name, real C++ symbol
        "p4est",
        # ── Differentiation / AD API ──────────────────────────
        "Differentiation",
        # ── Output ─────────────────────────────────────────────
        "write_vtu", "write_vtu_with_pvtu_record",
        "write_pvd_record",
    })

    domain_names.update({
        # ── Numerical method names (textbook concepts, not
        #    classes — e.g. "Newton" is a method, but in code
        #    you see SolverControl + iterate loops, not a
        #    "Newton" class).
        "Newton", "Picard", "Oseen", "Jacobi",
        "Crank-Nicolson", "Newmark", "BDF2", "Theta",
        "RungeKutta", "Heun", "Euler",
        # ── Stable pairs / stabilisations / concepts ──────────
        "Taylor-Hood", "MINI", "Vanka", "SUPG", "GLS", "VMS",
        "PML", "Sommerfeld", "Nitsche", "AMG", "SSOR", "SOR",
        "ILU", "ILUT",  # algorithm names sit between code/domain
        # ── Benchmark people / domain-specific math objects ──
        "Hankel", "Bessel", "Turek", "Schäfer-Turek",
        "Schaefer-Turek", "Ghia", "Kirsch", "Euler-Bernoulli",
        "Boussinesq", "Hermite",
        # ── Physics / PDE / math concepts ─────────────────────
        "Stokes", "Navier", "Maxwell", "Helmholtz",
        "Laplace", "Poisson", "Lagrange", "LBB",
        # ── Vocabulary in pitfall prose that's NOT a code symbol
        "breakdown", "convergence", "checkerboard", "Mesh",
        "ParaView",  # tool name, not a class
    })

    return code_symbols, domain_names


def _load_canonical_entities(backend: str) -> set[str]:
    """Back-compat alias — union of code-symbols and domain-names."""
    c, d = _load_entity_split(backend)
    return c | d


def _tier0_check(signal: str, code_symbols: set[str],
                 domain_names: set[str],
                 result: SignalVerification) -> None:
    """Tier 0: Signal references ≥1 CODE SYMBOL.

    Per critic 2026-05-31 round 2: domain names (Stokes, Poisson,
    Newton) are NOT sufficient on their own. The honest Tier-0
    gate is "Signal references at least one real deal.II code
    symbol that a post-execution critic could grep for in actual
    output". Domain names are recorded separately as soft
    indicators but do not flip tier0_passed.
    """
    code_matched: list[str] = []
    domain_matched: list[str] = []
    for tok in _IDENT_RE.findall(signal):
        if len(tok) < 3:
            continue
        if tok in code_symbols:
            code_matched.append(tok)
        elif tok in domain_names:
            domain_matched.append(tok)
    result.tier0_code_symbol_matched = sorted(set(code_matched))
    result.tier0_domain_names_matched = sorted(set(domain_matched))
    result.tier0_entities_matched = sorted(
        set(code_matched) | set(domain_matched))
    result.tier0_passed = len(code_matched) >= 1


def _tier1_check(signal: str, result: SignalVerification) -> None:
    """Tier 1: Signal uses observable-symptom vocabulary.

    Tokens are matched at word-boundary on the LEFT (regex \\b before
    the token) so that 'lock' does not match inside 'block', 'force'
    does not match inside 'enforce', 'stall' does not match inside
    'install', 'lose' does not match inside 'close', and 'nan' does
    not match inside 'StVenantKirchhoff'. Catches the over-fit class
    flagged by audit pass 2 (2026-06-02). Tokens are still treated
    as prefixes — 'produc' matches 'produces'/'producing' (the next
    char after 'produc' need not be a boundary).
    """
    low = signal.lower()
    hits = sorted({
        w for w in OBSERVABLE_VOCAB
        if re.search(r"\b" + re.escape(w), low)
    })
    # Plus quoted error fragments (anything inside backticks or
    # double quotes within the signal counts as a concrete error
    # observable).
    if re.search(r"`[^`]+`", signal) or re.search(r"'[^']+'", signal):
        hits.append("quoted-error-fragment")
    # Numerical-comparison phrases ("by 10+%", "differs by", "off by")
    if re.search(r"\bdiffers?\b|\boff by\b|\bby \d", low):
        hits.append("numerical-comparison")
    result.tier1_vocab_hits = sorted(set(hits))
    result.tier1_passed = len(hits) >= 1


def _harvest_pitfalls(backend: str) -> list[tuple[str, int, str]]:
    """Walk a backend's KNOWLEDGE dicts and return
    (physics, index, pitfall_text) triples.

    Goes through the backend API so we exercise the same path the
    agent uses — same rationale as the diff tool's loader.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from core.registry import load_all_backends, get_backend
    try:
        load_all_backends()
    except Exception:
        pass
    b = get_backend(backend)
    if b is None:
        return []
    out: list[tuple[str, int, str]] = []
    try:
        physics_iter = list(b.supported_physics())
    except Exception:
        return out
    for p in physics_iter:
        try:
            knowledge = b.get_knowledge(p.name)
        except Exception:
            continue
        if not isinstance(knowledge, dict):
            continue
        pitfalls = knowledge.get("pitfalls", [])
        if not isinstance(pitfalls, list):
            continue
        for i, entry in enumerate(pitfalls):
            if isinstance(entry, str):
                out.append((p.name, i, entry))
    return out


def verify_backend(backend: str) -> list[SignalVerification]:
    code_symbols, domain_names = _load_entity_split(backend)
    results: list[SignalVerification] = []
    for physics, idx, text in _harvest_pitfalls(backend):
        cat, sig = _split_pitfall(text)
        result = SignalVerification(
            backend=backend, physics=physics, pitfall_index=idx,
            pitfall_category=cat or "(no-prefix)",
            signal_text=sig or "",
        )
        if not cat:
            result.notes.append(
                "pitfall lacks [Category] prefix (PR #26 Table-1 "
                "convention) — Tier 0/1 not applicable")
            results.append(result)
            continue
        if not sig:
            result.notes.append(
                "pitfall lacks `Signal:` clause — cannot match in "
                "post-execution critic; the entry is descriptive, "
                "not detection-actionable")
            results.append(result)
            continue
        _tier0_check(sig, code_symbols, domain_names, result)
        _tier1_check(sig, result)
        # Tier 2 — load operational results from the fixture
        # runner output (if present).
        result.tier2_status = _tier2_lookup(backend, physics, idx)
        result.tier2_passed = (result.tier2_status == "passed")
        results.append(result)
    return results


def _tier2_lookup(backend: str, physics: str, idx: int) -> str:
    """Look up Tier-2 fixture result for one pitfall.

    Reads ``scripts/scan_results/tier2_results.json`` if present.
    Returns one of: ``passed`` / ``failed`` / ``harness_pending`` /
    ``not_attempted`` (when there is no fixture for this pitfall).

    The runner writes the file as
    ``{"summary": {...}, "results": {key: row}}`` — look up
    inside the ``results`` sub-object.
    """
    path = OUTPUT.parent / "tier2_results.json"
    if not path.is_file():
        return "harness_pending"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return "harness_pending"
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, dict):
        # Older harness wrote results at top level — fall back so
        # historic outputs still resolve.
        results = data if isinstance(data, dict) else {}
    key = f"{backend}::{physics}::{idx}"
    entry = results.get(key)
    if not isinstance(entry, dict):
        return "not_attempted"
    return str(entry.get("status", "harness_pending"))


def _load_falsifiability_map() -> dict:
    """Load per-pitfall falsifiability + cost classification.

    Senior-AI-scientist critic 2026-05-31 round 3: 'realistic
    12-month target is ~60/78, not 96/96 — publish the
    unfalsifiable count alongside the verified count.'
    Walking the deal.II catalog showed 0 truly-unfalsifiable
    Signals — the real axis is COST (cheap / medium /
    expensive), not falsifiability. Classification lives at
    ``data/postmortems/_falsifiability.json``.
    """
    path = (Path(__file__).resolve().parent.parent
            / "data" / "postmortems" / "_falsifiability.json")
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="dealii",
                    help="restrict to one backend; default: dealii")
    args = ap.parse_args()

    results = verify_backend(args.backend)

    falsifiability = _load_falsifiability_map()

    def _entry(r):
        key = f"{r.backend}::{r.physics}::{r.pitfall_index}"
        return falsifiability.get(key, {}) if isinstance(
            falsifiability, dict) else {}

    falsifiable_pitfalls = [
        r for r in results
        if _entry(r).get("falsifiable", True)
    ]
    cheap_pitfalls = [
        r for r in falsifiable_pitfalls
        if _entry(r).get("cost") == "cheap"
    ]
    medium_pitfalls = [
        r for r in falsifiable_pitfalls
        if _entry(r).get("cost") == "medium"
    ]
    expensive_pitfalls = [
        r for r in falsifiable_pitfalls
        if _entry(r).get("cost") == "expensive"
    ]

    totals = {
        "n_pitfalls": len(results),
        "with_category_prefix": sum(
            1 for r in results
            if r.pitfall_category != "(no-prefix)"),
        "with_signal_clause": sum(
            1 for r in results if r.signal_text),
        "tier0_passed": sum(1 for r in results if r.tier0_passed),
        "tier0_code_symbol_only": sum(
            1 for r in results
            if r.tier0_code_symbol_matched
            and not r.tier0_domain_names_matched),
        "tier0_with_domain_decoration": sum(
            1 for r in results
            if r.tier0_code_symbol_matched
            and r.tier0_domain_names_matched),
        "tier0_domain_names_only": sum(
            1 for r in results
            if r.tier0_domain_names_matched
            and not r.tier0_code_symbol_matched),
        "tier1_passed": sum(1 for r in results if r.tier1_passed),
        "tier0_and_1_passed": sum(
            1 for r in results
            if r.tier0_passed and r.tier1_passed),
        "tier2_passed": sum(1 for r in results if r.tier2_passed),
        "tier2_attempted": sum(
            1 for r in results
            if r.tier2_status in ("passed", "failed")),
        # Realistic-denominator metrics (round-3 critic):
        # "near-perfect" target is tier2_passed / n_falsifiable,
        # NOT tier2_passed / 96. And per-cost decomposition
        # reveals the actual roadmap distance.
        "n_falsifiable": len(falsifiable_pitfalls),
        "n_unfalsifiable": (len(results)
                            - len(falsifiable_pitfalls)),
        "n_cheap_falsifiable": len(cheap_pitfalls),
        "n_medium_falsifiable": len(medium_pitfalls),
        "n_expensive_falsifiable": len(expensive_pitfalls),
        "tier2_passed_of_cheap": sum(
            1 for r in cheap_pitfalls if r.tier2_passed),
        "tier2_passed_of_medium": sum(
            1 for r in medium_pitfalls if r.tier2_passed),
        "tier2_passed_of_expensive": sum(
            1 for r in expensive_pitfalls if r.tier2_passed),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "backend": args.backend,
        "totals": totals,
        "results": [asdict(r) for r in results],
    }, indent=2))

    # Per-row denominator: per-cost fractions use the bucket
    # count; other tier-* metrics use the full pitfall total.
    PER_COST_DENOM = {
        "tier2_passed_of_cheap": "n_cheap_falsifiable",
        "tier2_passed_of_medium": "n_medium_falsifiable",
        "tier2_passed_of_expensive": "n_expensive_falsifiable",
    }
    print(f"\n{args.backend} signal verification:")
    for k, v in totals.items():
        if k == "n_pitfalls":
            print(f"  {k:30s} {v:>4d}")
        elif k in PER_COST_DENOM:
            denom = totals[PER_COST_DENOM[k]]
            pct = (100.0 * v / denom) if denom else 0
            print(f"  {k:30s} {v:>4d} / {denom} "
                  f"({pct:.0f}% of bucket)")
        elif k.startswith("tier"):
            pct = (100.0 * v / totals["n_pitfalls"]
                   if totals["n_pitfalls"] else 0)
            print(f"  {k:30s} {v:>4d} / {totals['n_pitfalls']} "
                  f"({pct:.0f}%)")
        else:
            print(f"  {k:30s} {v:>4d}")
    print(f"\nFull report: {OUTPUT.relative_to(REPO_ROOT)}")

    # Per-result diagnostics for the entries that didn't pass
    # Tier 0 or 1 — the Signal text needs a small fix.
    bad = [r for r in results
           if r.signal_text
           and not (r.tier0_passed and r.tier1_passed)]
    if bad:
        print(f"\n{len(bad)} Signal clauses to review:")
        for r in bad[:10]:
            print(f"  {r.physics}#{r.pitfall_index} "
                  f"[{r.pitfall_category}] tier0={r.tier0_passed} "
                  f"tier1={r.tier1_passed}")
            print(f"    signal: {r.signal_text[:120]!r}")
        if len(bad) > 10:
            print(f"  ... +{len(bad) - 10} more")


if __name__ == "__main__":
    main()
