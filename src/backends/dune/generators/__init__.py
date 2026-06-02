"""DUNE-fem generator registry — maps physics_variant -> generator function."""

from .poisson import GENERATORS as _poisson_gen, KNOWLEDGE as _poisson_kn
from .heat import GENERATORS as _heat_gen, KNOWLEDGE as _heat_kn
from .linear_elasticity import GENERATORS as _elast_gen, KNOWLEDGE as _elast_kn
from .stokes import GENERATORS as _stokes_gen, KNOWLEDGE as _stokes_kn
from .reaction_diffusion import GENERATORS as _rxn_gen, KNOWLEDGE as _rxn_kn
from .nonlinear import GENERATORS as _nonlinear_gen, KNOWLEDGE as _nonlinear_kn
from .dg_advection import GENERATORS as _dg_gen, KNOWLEDGE as _dg_kn
from .adaptive_poisson import GENERATORS as _adaptive_gen, KNOWLEDGE as _adaptive_kn
from .advanced import GENERATORS as _advanced_gen, KNOWLEDGE as _advanced_kn

# Merged generator registry: physics_variant -> callable(params) -> str
GENERATORS: dict[str, callable] = {}
for _g in [
    _poisson_gen, _heat_gen, _elast_gen, _stokes_gen,
    _rxn_gen, _nonlinear_gen, _dg_gen, _adaptive_gen,
    _advanced_gen,
]:
    GENERATORS.update(_g)

# Merged knowledge registry: physics_name -> dict
KNOWLEDGE: dict[str, dict] = {}
for _k in [
    _poisson_kn, _heat_kn, _elast_kn, _stokes_kn,
    _rxn_kn, _nonlinear_kn, _dg_kn, _adaptive_kn,
    _advanced_kn,
]:
    KNOWLEDGE.update(_k)

# General knowledge (not physics-specific)
KNOWLEDGE["_general"] = {
    "description": "DUNE-fem general capabilities",
    "form_language": "UFL (shared with FEniCS) — weak forms are directly interchangeable",
    "spaces": {
        "lagrange": "Continuous Lagrange (any order)",
        "dglagrange": "Discontinuous Lagrange",
        "dglegendre": "DG with Legendre basis",
        "dgonb": "DG with orthonormal basis",
        "raviartThomas": "H(div) conforming (camelCase Python wrapper; the C++ header is lowercase but the Python factory is camelCase)",
        "bdm": "Brezzi-Douglas-Marini H(div) elements",
        "bdfm": "Brezzi-Douglas-Fortin-Marini H(div) elements",
        "finiteVolume": "Piecewise-constant FV space",
        "rannacherTurek": "Rannacher-Turek non-conforming",
        "p1Bubble": "Mini element (P1 + bubble)",
        "composite": "Multi-field composite spaces",
        "product": "Product of multiple spaces",
    },
    "grid_types": {
        "structuredGrid": "YaspGrid — structured Cartesian, supports periodic BCs",
        "ALUGrid": "Unstructured adaptive (simplices or cubes), 2D/3D",
        "Gmsh": "Import .msh files via dune.grid.reader.gmsh",
        "geometryGridView": "Deforming/moving meshes",
        "adaptiveLeafGridView": "Optimized for frequent local adaptivity",
    },
    "dg_methods": {
        "module": "dune-fem-dg (pip install dune-fem-dg)",
        "methods": ["Interior Penalty (SIPG, NIPG)", "Bassi-Rebay 1/2 (BR1, BR2)",
                   "LDG (Local DG)", "CDG, CDG2 (Compact DG)"],
        "time_stepping": "SSP Runge-Kutta (explicit/implicit/IMEX)",
    },
    "solvers": {
        "builtin": "Newton-Krylov (CG, GMRES) via galerkin scheme",
        "petsc": "Full PETSc access via as_petsc backend (CG, GMRES, AMG, fieldsplit)",
        "istl": "DUNE native iterative solver library (AMG, ILU)",
        "scipy": "Direct access to CSR matrices via as_numpy for scipy solvers",
    },
    "adaptivity": "mark/adapt/balance cycle with residual error estimators",
    "parallel": "MPI distributed + OpenMP shared memory",
    "vem": "Virtual Element Method via dune-vem module (conforming, nonconforming)",
    "surface_fem": "PDEs on static and evolving surfaces (mean curvature flow)",
    "unique_features": [
        "Shares UFL with FEniCS — physics descriptions are interchangeable",
        "JIT compilation: prototype in Python, performance of C++",
        "Deep h/p-adaptivity with ALUGrid",
        "Comprehensive DG methods via dune-fem-dg",
        "VEM (Virtual Element Method) support",
        "Surface FEM for PDEs on manifolds",
        "Multiple storage backends: numpy, istl, petsc",
    ],
    "cmake_user_macros": {
        "description": (
            "User-callable CMake helpers defined in dune-fem's "
            "cmake/modules/FemShort.cmake — downstream dune-fem-based "
            "projects invoke these in their CMakeLists.txt."
        ),
        "dune_add_subdirs": {
            "signature": "dune_add_subdirs(<dir1> [<dir2>...] [EXCLUDE <str> | NOEXCLUDE])",
            "purpose": "Add multiple subdirectories with one call. Subdirs "
                       "matching EXCLUDE (default: 'test') are added with "
                       "EXCLUDE_FROM_ALL; NOEXCLUDE adds everything.",
        },
        "dune_fem_add_test": {
            "signature": (
                "dune_fem_add_test(<test1> [<test2>...] "
                "[FAILTEST <ftest>...] [COMPILEFAILTEST <cftest>...] "
                "[DEPENDENCY_ONLY <dep>...] [NO_DEPENDENCY <ntest>...])"),
            "purpose": "Register tests with CMake testing framework.",
            "kwargs": {
                "FAILTEST":        "Tests EXPECTED to fail "
                                   "(WILL_FAIL property set to true)",
                "COMPILEFAILTEST": "Tests EXPECTED to fail at compile time "
                                   "(WILL_FAIL too, runs via "
                                   "cmake --build . --target <test>)",
                "DEPENDENCY_ONLY": "Add targets as test dependencies but "
                                   "NOT as runnable tests",
                "NO_DEPENDENCY":   "Add target as test but NOT as dependency",
            },
        },
        "dune_install": {
            "signature": "dune_install(<files>...)",
            "purpose": "Install given files into the current source dir's "
                       "include-directory location "
                       "(CMAKE_CURRENT_SOURCE_DIR with CMAKE_SOURCE_DIR "
                       "replaced by CMAKE_INSTALL_INCLUDEDIR).",
        },
        "Signal": (
            "[API] FAILTEST and COMPILEFAILTEST tests have "
            "WILL_FAIL=true. A test that passes in those buckets is "
            "REPORTED AS FAILING by ctest — the inversion is intentional. "
            "Don't list a working test under FAILTEST expecting it to "
            "pass cleanly; ctest will report it as a regression. (File "
            "walk dune-fem/cmake/modules/FemShort.cmake 2026-06-02.)"
        ),
    },
}
