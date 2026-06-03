"""scikit-fem generator registry — maps physics_variant -> generator function."""

from .poisson import GENERATORS as _poisson_gen, KNOWLEDGE as _poisson_kn
from .heat import GENERATORS as _heat_gen, KNOWLEDGE as _heat_kn
from .linear_elasticity import GENERATORS as _elast_gen, KNOWLEDGE as _elast_kn
from .stokes import GENERATORS as _stokes_gen, KNOWLEDGE as _stokes_kn
from .eigenvalue import GENERATORS as _eigen_gen, KNOWLEDGE as _eigen_kn
from .mixed_poisson import GENERATORS as _mixed_gen, KNOWLEDGE as _mixed_kn
from .convection_diffusion import GENERATORS as _convdiff_gen, KNOWLEDGE as _convdiff_kn
from .biharmonic import GENERATORS as _biharmonic_gen, KNOWLEDGE as _biharmonic_kn
from .nonlinear import GENERATORS as _nonlinear_gen, KNOWLEDGE as _nonlinear_kn
from .wave import GENERATORS as _wave_gen, KNOWLEDGE as _wave_kn
from .adaptive_poisson import GENERATORS as _adapt_gen, KNOWLEDGE as _adapt_kn
from .point_source import GENERATORS as _ps_gen, KNOWLEDGE as _ps_kn
from .schrodinger import GENERATORS as _sch_gen, KNOWLEDGE as _sch_kn
from .contact import GENERATORS as _ct_gen, KNOWLEDGE as _ct_kn
from .hydraulic_resistance import GENERATORS as _hr_gen, KNOWLEDGE as _hr_kn
from .advanced import GENERATORS as _advanced_gen, KNOWLEDGE as _advanced_kn

# Merged generator registry: physics_variant -> callable(params) -> str
GENERATORS: dict[str, callable] = {}
for _g in [
    _poisson_gen, _heat_gen, _elast_gen, _stokes_gen,
    _eigen_gen, _mixed_gen, _convdiff_gen, _biharmonic_gen,
    _nonlinear_gen, _wave_gen, _adapt_gen, _ps_gen, _sch_gen,
    _ct_gen, _hr_gen, _advanced_gen,
]:
    GENERATORS.update(_g)

# Merged knowledge registry: physics_name -> dict
KNOWLEDGE: dict[str, dict] = {}
for _k in [
    _poisson_kn, _heat_kn, _elast_kn, _stokes_kn,
    _eigen_kn, _mixed_kn, _convdiff_kn, _biharmonic_kn,
    _nonlinear_kn, _wave_kn, _adapt_kn, _ps_kn, _sch_kn,
    _ct_kn, _hr_kn, _advanced_kn,
]:
    KNOWLEDGE.update(_k)

# General knowledge (not physics-specific)
KNOWLEDGE["_general"] = {
    "description": "scikit-fem general capabilities",
    "element_catalog": {
        "triangular": "P0, P1, P2, P3, P4, Mini, CR (Crouzeix-Raviart), CCR, Morley, Argyris, Hermite, RT0/1/2, BDM1, N1/2/3 (Nedelec), HHJ0/1",
        "quadrilateral": "Q0, Q1, Q2, S2 (Serendipity), BFS (Bogner-Fox-Schmit), RT0/1, N1",
        "tetrahedral": "P0, P1, P2, RT0, N1, Mini, CR, CCR",
        "hexahedral": "H0, H1, H2, S2, RT1, C1",
        "line": "P0, P1, P2, Pp, Hermite, Mini",
    },
    "assembly_types": [
        "@BilinearForm: bilinear forms with u, v (trial/test)",
        "@LinearForm: linear forms with v (test only)",
        "@Functional: scalar integrals/functionals",
        "CellBasis: element interior assembly",
        "FacetBasis: boundary facet assembly",
        "InteriorFacetBasis: interior facet assembly (DG, error estimators)",
        "MortarFacetBasis: mortar mesh assembly (domain decomposition)",
    ],
    "mesh_types": [
        "MeshTri: init_symmetric, init_sqsymmetric, init_tensor, init_circle, init_lshaped",
        "MeshQuad: init_tensor",
        "MeshTet: init_tensor, init_ball",
        "MeshHex: init_tensor",
        "Mesh.load(): import from meshio (Gmsh, VTK, XDMF, any format)",
        "mesh.refined(n): uniform or adaptive (element index array)",
    ],
    "unique_features": [
        "Pure Python — zero compilation, zero external dependencies beyond numpy/scipy",
        "Assembly-level control — you build the matrices, you choose the solver",
        "50+ element types including Argyris, Morley, Nedelec, Raviart-Thomas",
        "meshio integration for any mesh format",
        "JAX-based automatic differentiation via skfem.autodiff",
        "Mortar methods for domain decomposition (MortarFacetBasis)",
        "Adaptive refinement: mesh.refined(element_indices)",
    ],
    "assembly_module_asm_shorthand": {
        "description": (
            "skfem.assembly.asm(form, *bases) is a convenience "
            "shorthand around Form.assemble that auto-wraps bare "
            "callables into the right form subclass by inspecting "
            "form.__code__.co_argcount. Source: "
            "skfem/assembly/__init__.py."),
        "auto_wrap_table": {
            1: "Functional   — (w,)",
            2: "LinearForm   — (v, w)",
            3: "BilinearForm — (u, v, w)",
            4: "TrilinearForm — (u, v, w, p)",
        },
        "backwards_compat_aliases": {
            "InteriorBasis": "alias for CellBasis (no DeprecationWarning)",
            "ExteriorFacetBasis": ("alias for BoundaryFacetBasis "
                                   "(no DeprecationWarning)"),
        },
        "Signal": (
            "[API] skfem.asm()'s auto-wrap based on "
            "form.__code__.co_argcount has TWO unguarded edge "
            "cases at the wrapper-lookup site "
            "[Functional, LinearForm, BilinearForm, "
            "TrilinearForm][nargs - 1]: "
            "(1) Bare-callable with nargs >= 5 raises "
            "IndexError('list index out of range') instead of a "
            "helpful error — common when a user adds an extra "
            "positional arg by mistake or forgets to decorate "
            "with @BilinearForm. "
            "(2) Bare-callable with nargs == 0 SILENTLY wraps as "
            "TrilinearForm via Python negative indexing "
            "(nargs - 1 == -1 hits the last list element). The "
            "wrap succeeds; the error comes at call time as "
            "TypeError('<lambda>() takes 0 positional arguments "
            "but 4 were given') — opaque because the user thinks "
            "they passed a Functional. "
            "Also: skfem.InteriorBasis and skfem.ExteriorFacetBasis "
            "are still active aliases for CellBasis and "
            "BoundaryFacetBasis with NO DeprecationWarning. "
            "Tutorials and StackOverflow answers from <=2022 use "
            "the old names interchangeably; new users see them "
            "work without realizing they're shadows of the "
            "modern names. Plus: skfem.asm uses an internal "
            "_sum(blocks) reducer that asserts `not isinstance("
            "out, int)` to catch the silent-zero case when the "
            "assembly iterator is empty (no bases passed, or "
            "empty product). AssertionError there means 'you "
            "passed nothing assemblable', not a bug in skfem. "
            "(File walk skfem/assembly/__init__.py 2026-06-03.)"
        ),
    },
    "linear_system_utils": {
        "description": (
            "skfem.utils linear-system entry points (enforce / "
            "condense / penalize / solve / mpc / "
            "solver_iter_krylov / solver_iter_pcg). Source: "
            "skfem/utils.py."
        ),
        "boundary_dof_input_modes": (
            "enforce/condense/penalize all use _init_bc which "
            "requires EXACTLY ONE of I (free DOFs) or D "
            "(Dirichlet DOFs). Passing BOTH raises "
            "Exception('Give only I or only D!'); passing NEITHER "
            "raises Exception('Either I or D must be given!'). "
            "The unspecified set is derived as the setdiff against "
            "np.arange(A.shape[0])."),
        "solver_iter_pcg_is_an_alias": (
            "solver_iter_pcg(**kwargs) is a one-line forwarder to "
            "solver_iter_krylov(**kwargs) with NO specialization "
            "beyond the docstring claim ('Conjugate gradient "
            "solver, specialized from solver_iter_krylov') — the "
            "default krylov of solver_iter_krylov is already "
            "scipy.sparse.linalg.cg, so the two are functionally "
            "identical."),
        "auto_injected_diagonal_preconditioner": (
            "solver_iter_krylov(...) auto-injects M=build_pc_diag(A) "
            "iff the literal string 'M' is NOT in kwargs at call "
            "time. Calling solver_iter_krylov(M=None) suppresses "
            "the injection and passes M=None straight to scipy "
            "(== no preconditioner). Convergence diagnostics that "
            "miss this misattribute speedups to the iterative "
            "method when the diagonal PC is actually doing the "
            "work."),
        "Signal": (
            "[API] Three sharp edges in skfem.utils that bite "
            "users on first contact: "
            "(1) Passing both I=... and D=... to enforce / "
            "condense / penalize raises Exception('Give only I "
            "or only D!') — the two are mutually exclusive "
            "complements (setdiff against arange(A.shape[0])), "
            "not intersectable filter sets. "
            "(2) solver_iter_pcg is a synonym for "
            "solver_iter_krylov, NOT a specialized PCG — both "
            "default to scipy.sparse.linalg.cg with auto-injected "
            "diagonal preconditioner. "
            "(3) The diagonal-PC auto-injection in "
            "solver_iter_krylov is gated on `'M' not in kwargs` "
            "— solver_iter_krylov(M=None) ≠ solver_iter_krylov(); "
            "the former gets no preconditioner (M=None forwarded "
            "to scipy), the latter gets a Jacobi PC silently. "
            "(File walk skfem/utils.py 2026-06-03.)"
        ),
    },
    "mortar_workflow": {
        "description": (
            "Nonmatching-mesh (mortar / supermesh) workflow via "
            "skfem.supermeshing — for domain-decomposition or "
            "fluid-structure interface coupling. Source: "
            "skfem/supermeshing.py."
        ),
        "functions": {
            "skfem.supermeshing.intersect(m1, m2)": (
                "Returns (supermesh, ix1, ix2) — supermesh + element-index "
                "arrays into each input mesh. ONLY 1D-1D or 2D-2D; 3D-3D "
                "or 1D-vs-2D raises NotImplementedError('The given mesh "
                "types not supported.')."),
            "skfem.supermeshing.elementwise_quadrature(mesh, supermesh, tind, intorder)": (
                "Build per-element quadrature rules from a supermesh. "
                "REQUIRES supermesh= kwarg explicitly (raises Exception "
                "if missing). intorder defaults to 4."),
        },
        "Signal": (
            "[API] skfem.supermeshing has three hard failure modes "
            "users hit on first use: "
            "(1) Calling intersect(m1, m2) with 3D meshes raises "
            "NotImplementedError('The given mesh types not supported.') "
            "— mortar in 3D is unsupported; "
            "(2) Calling intersect on 2D meshes without shapely>=2 "
            "installed raises Exception('2D supermeshing requires the "
            "package shapely>=2.') — silent until first call; "
            "(3) Calling elementwise_quadrature(mesh) without the "
            "supermesh= kwarg raises Exception('elementwise_quadrature: "
            "User must provide supermesh keyword argument which has been "
            "created using skfem.supermeshing.intersect.') — Python "
            "won't catch the missing kwarg at signature level because "
            "supermesh has default None and the check is at runtime. "
            "(File walk skfem/supermeshing.py 2026-06-03.)"
        ),
    },
}
