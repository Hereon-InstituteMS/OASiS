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
from .advanced import GENERATORS as _advanced_gen, KNOWLEDGE as _advanced_kn

# Merged generator registry: physics_variant -> callable(params) -> str
GENERATORS: dict[str, callable] = {}
for _g in [
    _poisson_gen, _heat_gen, _elast_gen, _stokes_gen,
    _eigen_gen, _mixed_gen, _convdiff_gen, _biharmonic_gen,
    _nonlinear_gen, _wave_gen, _adapt_gen, _ps_gen, _sch_gen,
    _advanced_gen,
]:
    GENERATORS.update(_g)

# Merged knowledge registry: physics_name -> dict
KNOWLEDGE: dict[str, dict] = {}
for _k in [
    _poisson_kn, _heat_kn, _elast_kn, _stokes_kn,
    _eigen_kn, _mixed_kn, _convdiff_kn, _biharmonic_kn,
    _nonlinear_kn, _wave_kn, _adapt_kn, _ps_kn, _sch_kn,
    _advanced_kn,
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
}
