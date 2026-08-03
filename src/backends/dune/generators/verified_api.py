"""DUNE-fem API surface established BY EXECUTION on the installed package.

Every statement in this module was produced by running a script against
the installed DUNE-fem and reading the output — not by reading upstream
documentation and not by reading the C++ sources alone. Where a source
file is cited it is cited as the EXPLANATION of an observed behaviour,
never as the evidence for it.

Install under test (recorded so a future reader can tell whether this
still applies):

    dune-fem / dune-common / dune-grid / dune-geometry / dune-istl /
    dune-localfunctions / dune-alugrid  all at 2.12.0.2
    (conda env dune-fem-env, CPython 3.12.13, Linux x86-64)

WHY THIS MODULE EXISTS. DUNE-fem shares UFL with FEniCS/dolfinx, so a
model that knows dolfinx writes DUNE-fem weak forms correctly on the
first try — and then gets everything AROUND the form wrong, because the
space, grid, boundary-condition, solver and output APIs have nothing to
do with dolfinx. The three failure classes that cost the most are
collected here:

  * transfer errors — a dolfinx idiom that raises an unhelpful
    exception in DUNE-fem (cheap: you see it immediately);
  * phantom APIs — names that appear in tutorials, older releases or
    prior versions of this catalog and do not exist in 2.12
    (cheap once you know, expensive while you are guessing);
  * silent-wrong traps — the run completes, ``scheme.solve()`` reports
    ``converged: True``, and the answer is wrong by 15 orders of
    magnitude (ruinous, and the reason this module is worth its size).

Provenance convention: every entry ends with what was run and what was
observed, dated. "Executed 2026-08-03" means a script was run on the
install above on that date and the quoted output is verbatim.
"""

from __future__ import annotations


# Dotted `dune.<path>` names this module spells out precisely BECAUSE
# they do NOT resolve on the installed package. They are the phantom
# APIs a model would otherwise reach for, so the catalog has to be
# allowed to name them.
#
# tests/test_catalog_consistency.py::TestDuneFemAPIPaths subtracts this
# set from its resolve-check AND asserts that every entry really is
# unresolvable — so the exemption cannot be used to smuggle in a
# genuinely broken reference; it flips into an extra positive check.
KNOWN_ABSENT_DUNE_PATHS: frozenset[str] = frozenset({
    # `import dune.fem.solver` -> ModuleNotFoundError; solvers are
    # selected with the scheme's `solver=` argument.
    "dune.fem.solver",
    # the multi-field factory is dune.fem.space.product
    "dune.fem.space.product_space",
    # the Python factory is raviartThomas (camelCase)
    "dune.fem.space.raviartthomas",
})

# JIT-generated module names (dune.generated.<kind>_<md5>) appear inside
# quoted error messages. The hash is machine-specific and the module
# only exists after that particular build, so they can never be
# resolve-checked.
JIT_GENERATED_PATH_PREFIXES: tuple[str, ...] = ("dune.generated",)


# ── Reference catalogue merged into KNOWLEDGE["_general"] ────────────

EXECUTED_API: dict = {
    "install_under_test": (
        "dune-fem 2.12.0.2 (dune-common/grid/geometry/istl/"
        "localfunctions/alugrid all 2.12.0.2), CPython 3.12.13, conda "
        "env dune-fem-env, Linux x86-64. All numbers below were "
        "measured on this install between 2026-08-03 and 2026-08-03. "
        "NOTE for readers of older catalog text: entries dated "
        "2026-08-01 in this backend say 'dune-fem 2.10'; the env has "
        "since been rebuilt at 2.12.0.2 and the API facts below were "
        "re-established against 2.12."),

    # ── grids ───────────────────────────────────────────────────────
    "grid_cell_types_measured": {
        "description": (
            "Cell geometry actually produced by each grid factory, "
            "counted with gridView.size(0) and gridView.size(dim) and "
            "read off entity.type. Executed 2026-08-03."),
        "structuredGrid([0,0],[1,1],[4,4])": (
            "16 elements, 25 vertices, geometry type 'quadrilateral' "
            "— YaspGrid produces CUBES, never simplices"),
        "structuredGrid([0,0,0],[1,1,1],[2,2,2])": (
            "8 elements, 27 vertices, geometry type 'hexahedron'"),
        "aluConformGrid(cartesianDomain([0,0],[1,1],[4,4]), dimgrid=2)": (
            "32 elements, 25 vertices, geometry type 'triangle' — each "
            "Cartesian cell is split into 2 triangles"),
        "aluSimplexGrid(same domain, dimgrid=2)": (
            "32 elements, 25 vertices, 'triangle'"),
        "aluCubeGrid(same domain, dimgrid=2)": (
            "16 elements, 25 vertices, 'quadrilateral'"),
        "aluSimplexGrid(cartesianDomain([0,0,0],[1,1,1],[2,2,2]), dimgrid=3)": (
            "48 elements, 27 vertices, 'tetrahedron' — 6 tets per hex"),
        "aluCubeGrid(same 3D domain, dimgrid=3)": (
            "8 elements, 27 vertices, 'hexahedron'"),
        "Signal": (
            "[API] structuredGrid is a YaspGrid and is ALWAYS made of "
            "cubes; the simplicial route measured here is "
            "dune.alugrid (aluConformGrid / aluSimplexGrid). The "
            "element count is the cheapest check: an n-by-n "
            "cartesianDomain gives n^2 cells through "
            "aluCubeGrid/structuredGrid and 2*n^2 cells through "
            "aluConformGrid/aluSimplexGrid. (Executed 2026-08-03: "
            "16 vs 32 elements for n=4. dune.grid also exposes "
            "ugGrid, albertaGrid and onedGrid, which were NOT "
            "exercised in this campaign.)"),
    },

    # ── the single most misleading thing in the whole API ───────────
    "ufl_cell_is_always_a_simplex": {
        "description": (
            "A dune-fem space reports a SIMPLEX UFL cell no matter "
            "what the grid is really made of. On the 16-quadrilateral "
            "structuredGrid above, space.cell() returns 'triangle' and "
            "space.ufl_element() prints '<Lagrange1 on a triangle>'; "
            "on the 8-hexahedron 3D grid it returns 'tetrahedron'. "
            "Meanwhile gridView.type is 'quadrilateral' / "
            "'hexahedron'. Executed 2026-08-03."),
        "why": (
            "dune/ufl/__init__.py::_cell(dimWorld, dimDomain) maps the "
            "dimension straight onto ufl.interval / ufl.triangle / "
            "ufl.tetrahedron / ufl.pentatope. The UFL cell exists only "
            "so UFL can type-check the form; the real basis functions "
            "come from the C++ space and the real quadrature from the "
            "real geometry type. The two are simply not connected."),
        "consequences": [
            "space.cell() / space.ufl_element() are NOT a way to find "
            "out what the mesh is made of — use gridView.type, or "
            "{e.type for e in gridView.elements}, or compare "
            "gridView.size(0) against the expected cube count.",
            "Q1/Q2 elements on a cube grid are reported as 'Lagrange1 "
            "/ Lagrange2 on a triangle'. NOTE that dof counts do NOT "
            "distinguish the two: continuous Lagrange on the 16-cube "
            "grid and on the 32-triangle aluConformGrid over the same "
            "cartesianDomain both have 25 dofs at order 1 and 81 at "
            "order 2, because they share the same vertices (measured "
            "2026-08-03). Only the ELEMENT COUNT and the geometry "
            "type tell them apart.",
            "Any dolfinx-shaped logic that branches on mesh.ufl_cell() "
            "to pick an element family, a quadrature degree or a "
            "basix element will branch on a lie.",
        ],
        "Signal": (
            "[API] repr(space) / space.ufl_element() says 'on a "
            "triangle' while gridView.type says 'quadrilateral'. This "
            "is normal and is NOT a bug to fix — it means the UFL cell "
            "carries no cell-shape information in dune-fem. Read the "
            "geometry from the gridView, never from the UFL element. "
            "(Executed 2026-08-03: structuredGrid([0,0],[1,1],[4,4]) "
            "-> 16 'quadrilateral' elements, space.cell() == "
            "'triangle', space.size == 25.)"),
    },

    # ── spaces ──────────────────────────────────────────────────────
    "space_construction_measured": {
        "description": (
            "What the space factories accept and reject, executed "
            "2026-08-03 on structuredGrid([0,0],[1,1],[4,4])."),
        "works": {
            "lagrange(gridView, order=k)":
                "scalar CG space; .size == 25 for k=1 on the 4x4 grid",
            "lagrange(gridView, order=k, dimRange=d)":
                "vector-valued space; dimRange=2 gives .size == 50 — "
                ".size counts SCALAR dofs, not vector blocks, and "
                "len(uh.as_numpy) == 50 too",
            "dglagrange(gridView, order=0)":
                ".size == 16 == cell count — this is the DG0 space",
            "finiteVolume(gridView)":
                ".size == 16 — the other piecewise-constant space",
        },
        "rejected_with_the_exact_message": {
            "lagrange(gridView, ('Lagrange', 1))":
                "TypeError: '<' not supported between instances of "
                "'tuple' and 'int' — the dolfinx "
                "functionspace(domain, ('Lagrange', k)) idiom lands on "
                "dune-fem's `if order < 1` guard and produces an "
                "error that says nothing about the real mistake",
            "lagrange(gridView, order=1, shape=(2,))":
                "TypeError: lagrange() got an unexpected keyword "
                "argument 'shape' — dolfinx spells vector spaces with "
                "a shape tuple, dune-fem spells them dimRange=d",
            "lagrange(gridView, degree=2)":
                "TypeError: lagrange() got an unexpected keyword "
                "argument 'degree' — the kwarg is `order`, not "
                "`degree`",
            "lagrange(gridView, order=0)":
                "KeyError: 'Parameter error in LagrangeSpace with "
                "order=0: order has to be greater or equal to 1' — "
                "there is no order-0 Lagrange space; the DG0 / P0 "
                "space is dglagrange(gridView, order=0) or "
                "finiteVolume(gridView)",
        },
        "Signal": (
            "[API] Every dolfinx space-construction idiom fails on "
            "dune-fem with an error that does not name the real "
            "problem: the positional element tuple becomes a "
            "tuple-vs-int comparison TypeError, `shape=` and `degree=` "
            "become unexpected-keyword TypeErrors, and order=0 becomes "
            "a KeyError about LagrangeSpace. The correct spelling is "
            "always lagrange(gridView, order=k[, dimRange=d]). "
            "(Executed 2026-08-03, all four messages quoted "
            "verbatim.)"),
    },

    # ── building UFL objects ────────────────────────────────────────
    "ufl_objects_come_from_the_SPACE": {
        "description": (
            "In dolfinx you build UFL objects from the MESH "
            "(ufl.SpatialCoordinate(domain), ufl.FacetNormal(domain)) "
            "and the function space separately. In dune-fem the SPACE "
            "is the ufl.FunctionSpace, and the grid view is not a UFL "
            "object at all. Executed 2026-08-03."),
        "measured": (
            "SpatialCoordinate(space) -> a length-2 vector on a 2D "
            "grid, ufl_shape (2,). SpatialCoordinate(gridView) -> "
            "AttributeError: "
            "'dune.generated.hierarchicalgrid_<hash>.LeafGrid' object "
            "has no attribute 'ufl_domain'."),
        "Signal": (
            "[API] AttributeError \"...LeafGrid object has no "
            "attribute 'ufl_domain'\" means a UFL constructor was "
            "handed the grid view instead of the space. Pass the "
            "space: SpatialCoordinate(space), TrialFunction(space), "
            "TestFunction(space). (Executed 2026-08-03.)"),
    },

    # ── mesh import + dof layout ────────────────────────────────────
    "mesh_import_and_dof_layout_measured": {
        "description": "Executed 2026-08-03.",
        "gmsh_import": (
            "The reader must be named explicitly as a TUPLE: "
            "aluConformGrid((reader.gmsh, 'mesh.msh'), dimgrid=2) — "
            "with `from dune.grid import reader`. Measured on a "
            "4-triangle msh2 file: 4 elements, 5 vertices, all "
            "'triangle'. Passing the path as a bare string, "
            "aluConformGrid('mesh.msh', dimgrid=2), raises "
            "RuntimeError: \"IOError [checkMacroGridFile:.../dune/"
            "alugrid/3d/grid_imp.cc:343]: Wrong file format!\" — "
            "ALUGrid interprets a bare string as its own DGF macro "
            "grid file, not as Gmsh. reader also has .dgf, "
            ".dgfString, .meshio and .structured."),
        "dof_ordering_for_dimRange_gt_1": (
            "INTERLEAVED by component, i.e. (u0, v0, u1, v1, ...). "
            "Measured by interpolating as_vector([x, 100 + y]) into "
            "lagrange(2x2 grid, order=1, dimRange=2): the dof array "
            "had length 18 (9 vertices x 2) and read "
            "[0.0, 100.0, 0.5, 100.0, 1.0, 100.0, 0.0, 100.5, ...] — "
            "every odd index >= 100, every even index < 100. It is "
            "NOT blocked (all of component 0, then all of component "
            "1). Slice uh.as_numpy[i::dimRange] to get component i."),
    },

    # ── boundary conditions ─────────────────────────────────────────
    "dirichlet_bc_measured": {
        "description": (
            "dune.ufl.DirichletBC(functionSpace, value, subDomain=None). "
            "The CONSTRUCTION rows below were measured on a 4x4 "
            "structuredGrid; the solve behaviour in "
            "silent_wrong_traps_measured was measured on a 16x16 one "
            "with the MMS u* = sin(pi x) sin(pi y). Executed "
            "2026-08-03."),
        "construction": {
            "DirichletBC(scalar_space, 0)": "accepted",
            "DirichletBC(vector_space_dimRange2, 0)":
                "AttributeError: 'int' object has no attribute "
                "'ufl_shape' — a bare scalar is only auto-wrapped for "
                "SCALAR spaces; a vector space needs a list",
            "DirichletBC(vector_space_dimRange2, [0, 0])": "accepted",
            "DirichletBC(vector_space_dimRange2, [0, 0, 0])":
                "bare AssertionError with an EMPTY message (the "
                "`assert ufl_value.ufl_shape[0] == dimRange` in "
                "dune/ufl/__init__.py carries no text) — component "
                "count mismatches are the least self-explanatory "
                "failure in the API",
            "DirichletBC(space, TrialFunction(space))":
                "AssertionError: 'the ufl expression used for "
                "Dirichlet conditions should not use a `Test` or "
                "`Trial` function'",
        },
        "how_it_is_applied": (
            "A DirichletBC only does something if it is an ELEMENT OF "
            "THE LIST handed to the scheme: galerkin([a == b, dbc], "
            "...). There is no apply()/assemble-time hook and nothing "
            "to call afterwards — dune-fem decides at scheme-"
            "construction time whether to wrap the operator in a "
            "DirichletWrapper (see integrands.hasDirichletBoundary in "
            "dune/fem/scheme/_schemes.py)."),
        "subdomain_argument": (
            "The optional third argument is a UFL CONDITIONAL over "
            "SpatialCoordinate(space), e.g. "
            "DirichletBC(space, g, conditional(x[0] < 1e-8, 1, 0)). "
            "dune.ufl.BoxDirichletBC(space, value, xL, xR) builds such "
            "a conditional for an axis-aligned box. Nothing checks "
            "that the conditional selects any facet at all."),
    },

    # ── the expensive one ───────────────────────────────────────────
    "silent_wrong_traps_measured": {
        "description": (
            "Measured 2026-08-03. Same problem in every row: "
            "-Delta u = 2 pi^2 sin(pi x) sin(pi y) on a 16x16 "
            "structuredGrid, Lagrange order 1, solver='cg', L2 error "
            "against the exact solution computed with "
            "dune.fem.integrate(..., order=6)."),
        "rows": [
            "CORRECT — galerkin([a == b, DirichletBC(space, 0)]): "
            "converged=True, linear_iterations=1, L2 err 1.899705e-03, "
            "max(u_h)=1.0032 (exact max is 1).",
            "BC OMITTED FROM THE LIST — galerkin(a == b, space=space): "
            "converged=True, linear_iterations=23935, L2 err "
            "7.510512e+14, max(u_h) = -7.5e+14. The scheme silently "
            "solved a PURE NEUMANN problem, which is singular; CG "
            "ground through 23935 iterations and the returned info "
            "dict still says converged: True.",
            "BC PRESENT BUT ITS SUBDOMAIN NEVER FIRES — "
            "DirichletBC(space, 0, conditional(x[0] < -1.0, 1, 0)) on "
            "[0,1]^2: byte-identical to the omitted case — "
            "converged=True, 23935 iterations, L2 err 7.510512e+14. A "
            "wrong sign or a wrong threshold in the conditional is "
            "indistinguishable from having written no BC at all.",
            "PARTIAL BC (left edge only, conditional(x[0] < 1e-8)): "
            "converged=True, 45 iterations, L2 err 2.676075e+00 — a "
            "well-posed but DIFFERENT boundary-value problem. The "
            "iteration count is the tell: 45 (well-posed) vs 23935 "
            "(singular) vs 1 (fully constrained).",
        ],
        "Signal": (
            "[Numerical] scheme.solve() returning "
            "{'converged': True, ...} does NOT mean the boundary "
            "conditions were applied. A DirichletBC that was never put "
            "in the galerkin([...]) list, or whose subDomain "
            "conditional selects no facet, leaves a singular pure-"
            "Neumann system that CG reports as converged. The "
            "observable symptoms are (a) linear_iterations in the "
            "thousands where the constrained problem needs tens, and "
            "(b) max(uh.as_numpy) exploding by many orders of "
            "magnitude. Assert on both. (Executed 2026-08-03: "
            "23935 iterations, L2 error 7.51e+14, converged=True.)"),
    },

    # ── solver control ──────────────────────────────────────────────
    "solver_control_measured": {
        "description": "Executed 2026-08-03 on the same 16x16 problem.",
        "valid_solver_strings": (
            "For the default (numpy/'fem') storage exactly three: "
            "'cg', 'gmres', 'bicgstab'. This is not a documentation "
            "claim — galerkin(..., solver='conjugate_gradient') raises "
            "RuntimeError: \"ParameterInvalid [getEnumeration:.../"
            "dune/fem/io/parameter/reader.hh:300]: Parameter "
            "'fem.solver.linear.method' invalid. Valid values are: "
            "gmres, cg, bicgstab\". The name is NOT validated in "
            "Python — dune/fem/discretefunction/_solvers.py just "
            "forwards it as the parameter 'linear.method' — so the "
            "error surfaces at scheme construction, from C++."),
        "direct_solver_executed": (
            "solver=('suitesparse', 'umfpack') WORKS on this install: "
            "the scheme built (62.6 s, one new JIT module — a direct "
            "solver is a different C++ type, so it costs its own "
            "build) and solved with info "
            "{'converged': True, 'iterations': 0, "
            "'linear_iterations': 1}, reaching exactly the same "
            "max(u_h) = 0.07459830 as the CG solve. "
            "linear_iterations == 1 is the tell that it was direct. "
            "Executed 2026-08-03."),
        "other_backends_NOT_EXECUTED": (
            "SOURCE-READ ONLY (dune/fem/discretefunction/_solvers.py, "
            "2026-08-03) — unlike the umfpack row above, none of "
            "these were run. solver may also be "
            "('suitesparse', 'ldl'|'spqr_symmetric'|"
            "'spqr_nonsymmetric'), ('istl', ...), ('petsc', ...), "
            "('eigen', 'cg'|'bicgstab'), "
            "('viennacl', 'cg'|'gmres'|'bicgstab'). Only the "
            "suitesparse and eigen/viennacl method names are validated "
            "in Python (ValueError listing the alternatives); the rest "
            "become C++ parameters. Whether the underlying library is "
            "actually linked into a given conda build is a separate "
            "question — check before relying on one."),
        "convergence_flag_semantics": (
            "The flag IS meaningful for iteration starvation: with "
            "parameters={'linear.maxiterations': 1} the info dict came "
            "back {'converged': False, 'linear_iterations': -1} and "
            "the L2 error was 5.0e-01; maxiterations 3, 10 and 10000 "
            "all gave converged=True and the correct 1.899705e-03. It "
            "is NOT meaningful for a singular system — see "
            "silent_wrong_traps_measured. Always check "
            "info['converged'] AND a physical sanity bound."),
        "parameter_key_deprecation": (
            "MEASURED: parameters={'newton.tolerance': 1e-10} still "
            "works but emits UserWarning \"the parameter key 'newton' "
            "is deprecated. Replace with 'nonlinear'\", and the key is "
            "rewritten — scheme.parameters came back as "
            "{'nonlinear.tolerance': 1e-10, 'linear.method': 'cg'}. "
            "Passing BOTH 'newton.tolerance' and "
            "'nonlinear.tolerance' was accepted with NO error at all. "
            "SOURCE-READ (dune/fem/scheme/_schemes.py::"
            "_checkNewtonInParameters, not executed): on that "
            "collision the newton.* entry is popped and the "
            "nonlinear.* value survives; the KeyError "
            "\"Mixing new and old parameter keys is not allowed\" is "
            "raised only for the NESTED spellings, when a key "
            "containing 'newton.linear'/'nonlinear.linear' collides "
            "with the already-present de-prefixed 'linear.*' key. "
            "Executed 2026-08-03."),
        "unrecognised_keys_are_silently_ignored": (
            "MEASURED: parameters={'nonlinear.maxiter': 3, "
            "'totally.bogus.key': 42} was accepted with no exception "
            "and NO warning, and the solve ran to the normal answer. "
            "So a mistyped key ('maxiter' for 'maxiterations') leaves "
            "the default silently in place — the parameter dict is "
            "not a checked schema. Executed 2026-08-03."),
        "useful_keys_SOURCE_READ": (
            "From dune/fem/solver/{parameter,newtoninverseoperator}.hh "
            "(read 2026-08-03, only linear.tolerance / "
            "linear.maxiterations / linear.method were actually "
            "exercised): 'linear.tolerance', 'linear.maxiterations', "
            "'linear.verbose', 'linear.preconditioning.method', "
            "'nonlinear.tolerance', 'nonlinear.maxiterations', "
            "'nonlinear.verbose', 'nonlinear.linesearch'. The C++ "
            "reader prefixes them with 'fem.solver.' — that prefix is "
            "what appears in error messages, not what you pass."),
    },

    # ── integrate / error norms ─────────────────────────────────────
    "integrate_measured": {
        "description": (
            "dune.fem.integrate(expr, gridView=None, order=None) — the "
            "2.12 spelling. Executed 2026-08-03 on a 4x4 "
            "structuredGrid, integrand sin(pi x) sin(pi y) whose exact "
            "integral over [0,1]^2 is (2/pi)^2 = 0.405284734569."),
        "default_order_is_safe": (
            "order=None is NOT a low-order default: the measured "
            "relative error was 1.7e-10 for the trig integrand and "
            "1.7e-16 (machine precision) for x^4 y^4. Explicitly "
            "PASSING a small order is what silently under-integrates: "
            "order=1 gave 5.3e-02 relative error, order=2 and order=3 "
            "both gave 1.8e-04 (bit-identical, so those two requests "
            "resolve to the same rule), "
            "order=5 gave 2.4e-07, order=8 gave 7.2e-14."),
        "call_forms": (
            "gridView is REQUIRED unless the expression already "
            "contains a grid function: integrate(expr) on a pure-UFL "
            "expression raises AttributeError \"a 'gridView' has to be "
            "provided or the expression must contain a grid "
            "function.\". A vector-valued integrand returns a "
            "FieldVector, not a float — integrate(as_vector([e, 2*e])) "
            "returned (0.405285, 0.810569)."),
        "deprecated_entry_point": (
            "dune.fem.function.integrate(gridView, expr, order) still "
            "runs and returns the right number but warns "
            "\"dune.fem.function.integrate is deprecated use "
            "dune.fem.integrate instead. New signature is (expr, "
            "gridView, order)\". Note the ARGUMENT ORDER FLIPPED "
            "between the two."),
        "Signal": (
            "[Numerical] An L2/H1 error norm computed with an "
            "explicitly low quadrature order is smaller than the true "
            "error and flatters the convergence table. On this "
            "install the default (order=None) is generous, so the "
            "damage comes from hand-setting order=1 or order=2: the "
            "measured relative quadrature error was 5.3e-02 at "
            "order=1. Use order >= 2k+2 for a Lagrange-order-k error "
            "norm, or leave order unset. (Executed 2026-08-03.)"),
    },

    # ── JIT ─────────────────────────────────────────────────────────
    "jit_compilation_measured": {
        "description": (
            "dune-fem generates and compiles a C++ pybind11 module for "
            "every distinct grid type, space type, UFL integrands "
            "object and scheme type. Executed 2026-08-03."),
        "cache_location": (
            "dune.packagemetadata.getDunePyDir() returned "
            "'<sys.prefix>/.cache/dune-py' — on this install "
            "'/home/.../envs/dune-fem-env/.cache/dune-py', with the "
            "compiled modules in "
            "'<that>/python/dune/generated/*.so' — it grows without "
            "bound: measured 164 .so files / 580 MB of generated "
            "artefacts at the end of this campaign (2026-08-03). "
            "Resolution order is "
            "$DUNE_PY_DIR/dune-py, then <sys.prefix>/.cache/dune-py "
            "if inVirtualEnvironment(), then ~/.cache/dune-py. "
            "'~/.dune/dune-py' — asserted by earlier revisions of "
            "this catalog — DOES NOT EXIST on this install and is not "
            "a path the 2.12 code can produce."),
        "cache_location_depends_on_CONDA_DEFAULT_ENV": (
            "THE SAME INTERPRETER RESOLVES TO TWO DIFFERENT CACHES. "
            "packagemetadata.inVirtualEnvironment() returns 1 as soon "
            "as CONDA_DEFAULT_ENV is present in os.environ, and only "
            "falls back to the sys.prefix != sys.base_prefix test "
            "otherwise. In a conda env those two prefixes are EQUAL "
            "(measured here: both "
            "/home/.../envs/dune-fem-env), so CONDA_DEFAULT_ENV is "
            "the only thing putting the cache inside the env. "
            "Measured 2026-08-03 with the identical interpreter: "
            "CONDA_DEFAULT_ENV set -> "
            "'<env>/.cache/dune-py'; `env -u CONDA_DEFAULT_ENV` -> "
            "'~/.cache/dune-py'; DUNE_PY_DIR=/tmp/xyz -> "
            "'/tmp/xyz/dune-py'. A harness that spawns "
            "<env>/bin/python WITHOUT conda activation therefore "
            "starts from a cold cache in $HOME and pays the full "
            "build again, while an interactive `conda activate` "
            "session reuses the env cache — two caches, silently. "
            "OASiS's own _dune_subprocess_env() in "
            "src/backends/dune/backend.py already sets "
            "CONDA_DEFAULT_ENV, which is what keeps the two paths "
            "in agreement; pin DUNE_PY_DIR if you want certainty."),
        "measured_costs": (
            "Cold cache, machine under heavy concurrent load: a script "
            "that built structuredGrid in 2D and 3D plus three "
            "Lagrange spaces and then aluConformGrid + aluSimplexGrid "
            "+ aluCubeGrid in 2D and aluSimplexGrid + aluCubeGrid in "
            "3D took 439 s wall END TO END, of which the four "
            "'DUNE-INFO: Compiling HierarchicalGrid (new)' builds "
            "dominated (the structuredGrid parts were already warm). "
            "With everything warm, a complete 8x8 Poisson run — grid, "
            "space, scheme, solve — took 0.89 s. The "
            "boundary-condition/solver battery (4 'Compiling "
            "Integrands (new)' lines and 1 'Compiling Scheme (new)') "
            "took 515 s; the quadrature battery (no new scheme) "
            "136 s; a single new scheme for a direct "
            "('suitesparse','umfpack') solver 62.6 s. ALUGrid "
            "hierarchical-grid modules are by far the most expensive "
            "single item."),
        "what_triggers_a_rebuild": (
            "Module names are md5 hashes of the generated C++ (the "
            "type name for spaces/schemes, the emitted integrands for "
            "a form). MEASURED to trigger a new module: a different "
            "Lagrange order (each of k=1,2,3 built its own Space and "
            "Scheme), a different grid class (each ALUGrid variant "
            "built its own HierarchicalGrid), and ANY change to the "
            "form — including changing a bare float literal in it "
            "(7.3125 -> 9.8125 cost 25.204 s and added one .so). "
            "MEASURED not to trigger one: re-running the identical "
            "form (0.051 s, no new .so) and changing the .value of a "
            "dune.ufl.Constant (0.00056 s, no new .so). Storage "
            "backends other than the default numpy were not "
            "exercised."),
        "Signal": (
            "[Performance] 'DUNE-INFO: Compiling <X> (new)' on stderr "
            "means a C++ module is being built right now; the process "
            "will sit there for tens of seconds to minutes with no "
            "further output. A first run that appears hung is almost "
            "always this. Budget >= 600 s of timeout for any first "
            "run, and >= 900 s if ALUGrid is involved. (Executed "
            "2026-08-03: 439 s for five ALUGrid variants.)"),
    },

    # ── runtime constants ───────────────────────────────────────────
    "runtime_constants_measured": {
        "description": (
            "dune.ufl.Constant(value, name=...) is a runtime-"
            "updatable parameter, not a compile-time literal. "
            "Executed 2026-08-03."),
        "measured": (
            "c = Constant(1.0, name='src'); scheme built with "
            "b = c*v*dx on an 8x8 grid. First solve 25.01 s wall "
            "(JIT included), max(u_h) = 0.07459830. Then "
            "c.value = 2.0 and scheme.solve() again: 0.0003 s and "
            "max(u_h) = 0.14919660 — exactly 2.000000x, as a linear "
            "problem must give. No recompilation, no new scheme."),
        "measured_against_the_float_alternative": (
            "The same experiment run both ways on an 8x8 grid, "
            "counting the .so files under "
            "<dunePyDir>/python/dune/generated before and after each "
            "scheme construction:\n"
            "  b = 7.3125*v*dx  (first time)   27.424 s, .so 161->162\n"
            "  b = 7.3125*v*dx  (again)         0.051 s, .so 162->162\n"
            "  b = 9.8125*v*dx  (VALUE CHANGED) 25.204 s, .so "
            "162->163  <-- full rebuild\n"
            "  b = Constant(7.3125)*v*dx        25.518 s, .so "
            "163->164\n"
            "  c.value = 9.8125; scheme.solve() 0.00056 s, .so "
            "164->164  <-- no rebuild\n"
            "Both routes gave the identical answer "
            "(max(u_h) = 0.73199583 for the 9.8125 right-hand side), "
            "so this is a pure cost difference: about 25 s of C++ "
            "compilation per changed literal versus half a "
            "millisecond. Measured 2026-08-03."),
        "plain_ufl_constant_fails": (
            "ufl.Constant(1.0) (the non-dune one) raises "
            "AttributeError: 'float' object has no attribute "
            "'ufl_domain' — its first positional argument is the "
            "domain, not the value. Use dune.ufl.Constant."),
        "Signal": (
            "[Performance] Baking a changing coefficient into the form "
            "as a Python float forces a full JIT rebuild on EVERY new "
            "value — a parameter sweep spends ~25 s of C++ "
            "compilation per sample and the generated/*.so count grows "
            "by one each time. Wrap it in dune.ufl.Constant and assign "
            "to .value between solves instead: same answer, no new "
            ".so, 0.00056 s. (Executed 2026-08-03: changing 7.3125 to "
            "9.8125 as a literal cost 25.204 s and .so 162->163; as a "
            "Constant it cost 0.00056 s and .so 164->164, both "
            "reaching max(u_h) = 0.73199583.)"),
    },

    # ── output ──────────────────────────────────────────────────────
    "vtk_output_measured": {
        "description": (
            "gridView.writeVTK(name, celldata=None, pointdata=None, "
            "cellvector=None, pointvector=None, number=None, "
            "subsampling=None, outputType=OutputType.appendedraw, "
            "write=True, nonconforming=False) — signature read from "
            "dune/grid/grid_generator.py and exercised 2026-08-03."),
        "higher_order_is_sampled_at_VERTICES_by_default": (
            "Measured on a 4x4 structuredGrid (25 vertices, 16 cells) "
            "carrying a Lagrange ORDER-2 function (space.size 81): "
            "writeVTK(pointdata=...) produced a .vtu with "
            "NumberOfPoints=25 — i.e. the file carries one value per grid "
            "VERTEX and the 56 remaining P2 dofs are not written at "
            "all, so any viewer sees a vertex-interpolated version "
            "of a quadratic field (the rendering itself was not "
            "inspected). writeVTK(..., subsampling=2) gave "
            "NumberOfPoints=400 and writeVTK(..., "
            "nonconforming=True) gave 64 (4 per cell, per-cell "
            "discontinuous). Nothing warns. Use subsampling >= order "
            "when you are looking at anything above P1, and remember "
            "the file is then a RESAMPLING, not the discrete "
            "function."),
        "boundary_ids": (
            "dune.fem.utility.inspectBoundaryIds(gridView) projects "
            "the boundary ids onto a finiteVolume function. On a 4x4 "
            "structuredGrid the SET of ids present was {0, 1, 2, 3, 4} "
            "(which id belongs to which side was not checked — read "
            "them off this function rather than assuming). "
            "dune.fem.utility.gridWidth(gridView) returned "
            "0.25 on the same grid. Use these instead of guessing at "
            "the ids in a dune.ufl.BoundaryId conditional."),
        "Signal": (
            "[API] There is no XDMFFile and no VTXWriter on a dune "
            "gridView — hasattr(gridView, 'XDMFFile') is False. VTK "
            "output goes through gridView.writeVTK(...), which writes "
            "<name>.vtu directly (measured: writeVTK('out_plain', "
            "pointdata=...) produced exactly ['out_plain.vtu']). "
            "(Executed 2026-08-03.)"),
    },

    # ── adaptation ──────────────────────────────────────────────────
    "adaptation_measured": {
        "description": (
            "The 2.12 mark/adapt cycle, executed 2026-08-03."),
        "mark_returns_STATISTICS_not_a_marker": (
            "dune.fem.mark(indicator, tol) builds a GridMarker, CALLS "
            "it immediately, and returns what the call returns: the "
            "(nRefined, nCoarsened) tuple — measured as (-1, -1) "
            "because statistics defaults to False. It is NOT a marker "
            "object. Feeding the return value back in, "
            "dune.fem.adapt(marker, [uh]), therefore fails with "
            "AssertionError: 'only one list of discrete functions can "
            "be passed into the adaptation method' (the tuple is not "
            "callable, so gridAdapt re-dispatches it as the first "
            "discrete function). The correct sequence is two "
            "statements with nothing passed between them: "
            "dune.fem.mark(ind, tol) then dune.fem.adapt([uh, ...]). "
            "If you want a reusable marker object, construct "
            "dune.fem.GridMarker(indicator, tol) yourself and hand "
            "THAT to dune.fem.adapt(marker, [uh]) — measured to work, "
            "32 -> 48 elements."),
        "adaptivity_requires_adaptiveLeafGridView": (
            "dune.fem.adapt([uh, ...]) on a plain aluConformGrid leaf "
            "view raises AssertionError: 'the grid views for all "
            "discrete functions need to support adaptivity'. Reading "
            "gridView.canAdapt on that same plain view raises "
            "AttributeError; on "
            "adaptiveLeafGridView(aluConformGrid(...)) it is True. So "
            "an ALUGrid alone is not enough — wrap it in "
            "dune.fem.view.adaptiveLeafGridView before building the "
            "spaces. CAVEAT worth knowing: that check lives in "
            "_adaptArguments and only runs for the LIST form; passing "
            "a single discrete function returns early and skips it, "
            "which is exactly why globalRefine(level, uh) can fail "
            "silently (see global_refinement_measured)."),
        "upstream_bug_gridView_kwarg": (
            "dune.fem.mark(indicator, tol, gridView=gv) RAISES "
            "AttributeError: \"'GridMarker' object has no attribute "
            "'gridView'\" — UNCONDITIONALLY, on a YaspGrid and on an "
            "ALUGrid alike (both measured). The cause is a typo in "
            "dune/fem/_adaptation.py: GridMarker.__init__ stores "
            "self._gridView and then validates "
            "GridMarker.checkGridView(self.gridView), but the class "
            "exposes no `gridView` attribute (only `indicator` is a "
            "property). dune.fem.markNeighbors inherits the same "
            "break because it forwards gridView=. WORKAROUND: omit "
            "gridView= entirely and let the marker take the grid view "
            "from the indicator — which means the indicator must be a "
            "DISCRETE FUNCTION (e.g. finiteVolume(gv).interpolate("
            "...)), not a bare UFL expression. dune.fem.markNeighbors "
            "forwards the same kwarg and was measured to raise the "
            "identical AttributeError."),
        "working_cycle": (
            "gv = adaptiveLeafGridView(aluConformGrid(...)); "
            "ind = finiteVolume(gv).interpolate(<estimator>, "
            "name='ind'); dune.fem.mark(ind, tol); "
            "dune.fem.adapt([uh]). Measured: 32 -> 48 elements, "
            "lagrange(order=1).size 25 -> 33, and uh was prolonged "
            "onto the new space. No space.update() call is needed or "
            "possible — hasattr(space, 'update') is False."),
        "global_refinement_measured": (
            "dune.fem.globalRefine has TWO call forms and they behave "
            "very differently. Passing the HIERARCHICAL GRID refined "
            "in every configuration tried: 16 -> 64 on structuredGrid "
            "4x4 (plain and adaptiveLeaf views) and 32 -> 64 on "
            "aluConformGrid, both with and without a lagrange space "
            "already built on the view (with a space alive the space "
            "resized too: structuredGrid 25 -> 81 dofs, "
            "aluConformGrid 25 -> 41). "
            "Passing a DISCRETE FUNCTION — the form that also "
            "prolongs it — only works on an adaptive ALUGrid view: "
            "  adaptiveLeafGridView(aluConformGrid): "
            "(32 elems, 25 dofs) -> (64, 41). WORKS.\n"
            "  aluConformGrid plain leaf view: RuntimeError "
            "'NotImplemented [numBlocks:.../dune/fem/space/mapper/"
            "indexsetdofmapper.hh:228]: Method numBlocks() called on "
            "non...' — loud, at least.\n"
            "  structuredGrid, PLAIN view: (16, 25, 25) -> "
            "(16, 25, 25). SILENT NO-OP.\n"
            "  structuredGrid wrapped in adaptiveLeafGridView "
            "(canAdapt reads True!): (16, 25, 25) -> (16, 25, 25). "
            "ALSO A SILENT NO-OP.\n"
            "Passing the hierarchical grid AND functions together is "
            "deprecated and warns."),
        "Signal_globalRefine": (
            "[Numerical] dune.fem.globalRefine(level, uh) is a SILENT "
            "NO-OP on a YaspGrid — the element count, the space size "
            "and len(uh.as_numpy) all come back unchanged, with no "
            "exception and no warning, even when "
            "gridView.canAdapt reports True. A refinement study "
            "written that way produces the same numbers at every "
            "'level'. Assert that gridView.size(0) actually grew "
            "after each refinement, or refine via "
            "globalRefine(level, gridView.hierarchicalGrid) and "
            "rebuild the space. (Executed 2026-08-03 on dune-fem "
            "2.12.0.2, all four grid/view combinations measured.)"),
        "mark_lives_on_the_grid_not_the_leaf_view": (
            "hasattr(structuredGrid(...), 'mark') is False while "
            "hasattr(gridView.hierarchicalGrid, 'mark') is True. The "
            "per-element mark() the marker calls internally is on the "
            "ADAPTIVE grid view (dune.fem.view.adaptiveLeafGridView) "
            "or on the hierarchical grid — not on a plain leaf view."),
        "Signal": (
            "[API] An adaptation loop copied from a tutorial that "
            "passes gridView= to dune.fem.mark dies immediately with "
            "AttributeError: 'GridMarker' object has no attribute "
            "'gridView'. That is an upstream defect in dune-fem "
            "2.12.0.2, not a mistake in your indicator — drop the "
            "kwarg and pass a discrete-function indicator instead. "
            "(Executed 2026-08-03 on both YaspGrid and ALUGrid.)"),
    },

    # ── process exit code is not a verdict ──────────────────────────
    "intermittent_teardown_abort_measured": {
        "description": (
            "OBSERVED, NOT FULLY CHARACTERISED — recorded because the "
            "operational consequence is large and the symptom is easy "
            "to misread. A script that builds several ALUGrid and "
            "YaspGrid objects (plain and adaptiveLeafGridView) and "
            "refines them printed all of its output correctly and "
            "then ABORTED DURING INTERPRETER TEARDOWN with a PETSc "
            "signal handler dump ('Caught signal number 11 SEGV', "
            "then 'terminate called after throwing an instance of "
            "Dune::ExceptionStream<Dune::Petsc::Exception>') and exit "
            "code 134. The backtrace is inside "
            "ALUGrid::TetraTop<...>::~TetraTop via the shared_ptr "
            "release of the generated hierarchicalgrid module, i.e. "
            "grid destruction after main."),
        "reproducibility": (
            "INTERMITTENT: the identical script run three times gave "
            "exit codes 134, 0, 134 with byte-identical stdout. A "
            "minimal script (one adaptiveLeafGridView(aluConformGrid) "
            "plus globalRefine, or two of them) exited 0 on every "
            "attempt, so this is NOT 'ALUGrid teardown always "
            "aborts' — it needs the bigger mix and it does not fire "
            "every time. Measured 2026-08-03."),
        "Signal": (
            "[Integration] A DUNE-fem run can produce every correct "
            "result and STILL exit non-zero, because the abort "
            "happens after the last print, during grid destruction. "
            "Judging a DUNE job by returncode alone will therefore "
            "mark a good run as failed intermittently. Have the "
            "script write its results to a file (or print a terminal "
            "sentinel line) and treat THAT as the success criterion, "
            "with the returncode as a secondary signal. (Executed "
            "2026-08-03: same script, exit codes 134 / 0 / 134, "
            "identical stdout.)"),
    },

    # ── threading ───────────────────────────────────────────────────
    "threading_measured": {
        "description": (
            "dune.fem.threading.max and .use are ATTRIBUTES (read and "
            "assign them); .useMax is a CALLABLE "
            "(<built-in method useMax of PyCapsule object>, "
            "callable() is True). Executed 2026-08-03 on a 32-core "
            "box: threading.max == 32 but threading.use == 1 by "
            "DEFAULT. Assigning dune.fem.threading.use = 2 took "
            "effect immediately (read back as 2), and calling "
            "dune.fem.threading.useMax() set use to 32."),
        "Signal": (
            "[Performance] dune-fem assembles and solves on ONE "
            "thread unless you say otherwise — threading.use defaults "
            "to 1 even when threading.max reports 32. If a DUNE run "
            "pegs a single core while the machine idles, that is why. "
            "(Executed 2026-08-03.)"),
    },

    # ── the strongest evidence available: measured orders ───────────
    "measured_convergence_2d_poisson": {
        "description": (
            "Manufactured-solution convergence study run on this "
            "install 2026-08-03. -div(grad u) = f on [0,1]^2, "
            "u* = sin(pi x) sin(pi y) + 0.3 x^2 y, f = -div(grad(u*)) "
            "built symbolically by UFL (so no hand-derived source can "
            "be mistranscribed), exact Dirichlet data on the whole "
            "boundary via DirichletBC(space, u_exact), solver='cg' "
            "with linear.tolerance 1e-12, uniform refinement "
            "n = 4, 8, 16, 32, error norms via "
            "dune.fem.integrate(..., order=2k+4)."),
        "cube_grid_structuredGrid": {
            "k=1": ("L2 EOC 1.989 / 1.997 / 1.999 (theory 2); "
                    "H1 EOC 0.995 / 0.999 / 1.000 (theory 1); "
                    "finest L2 4.556318e-04 at 1089 dofs"),
            "k=2": ("L2 EOC 2.979 / 2.995 / 2.999 (theory 3); "
                    "H1 EOC 1.998 / 2.000 / 2.000 (theory 2); "
                    "finest L2 3.846536e-06 at 4225 dofs"),
            "k=3": ("L2 EOC 3.985 / 3.996 / 3.999 (theory 4); "
                    "H1 EOC 2.996 / 2.999 / 3.000 (theory 3); "
                    "finest L2 2.180413e-08 at 9409 dofs"),
        },
        "simplex_grid_aluConformGrid": {
            "k=1": ("L2 EOC 1.894 / 1.972 / 1.993 (theory 2); "
                    "H1 EOC 0.955 / 0.988 / 0.997 (theory 1); "
                    "finest L2 1.312335e-03 at 1089 dofs"),
            "k=2": ("L2 EOC 2.981 / 2.995 / 2.999 (theory 3); "
                    "H1 EOC 1.954 / 1.988 / 1.997 (theory 2); "
                    "finest L2 8.602468e-06 at 4225 dofs"),
        },
        "reading": (
            "Both element families hit theory. At equal dof count the "
            "cube grid is the more accurate of the two here (k=1, "
            "1089 dofs: 4.56e-04 on cubes vs 1.31e-03 on triangles) "
            "— note the aluConformGrid built from the same "
            "cartesianDomain has twice the elements but the SAME "
            "vertices, hence the same dof count. Why Q1 on cubes "
            "beats P1 on those triangles for this solution was not "
            "investigated, and it is certainly not a general "
            "ranking — it is this MMS on this pair of meshes."),
        "cost": (
            "The whole study (5 order/grid combinations x 4 levels = "
            "20 solves, including the JIT builds for the simplex "
            "spaces and schemes) took 914 s wall on a loaded "
            "32-core box."),
        "Signal": (
            "[Numerical] If a DUNE-fem convergence table shows an "
            "observed L2 order near 0 instead of k+1, and the errors "
            "sit at an O(1) mesh-independent plateau, suspect the "
            "boundary conditions before the discretisation — the "
            "silent Dirichlet trap produces exactly that curve. If "
            "the order is right but the level-to-level ratio flattens "
            "at the finest level, suspect linear.tolerance (set it "
            "well below the finest discretisation error; 1e-12 was "
            "enough down to L2 2.2e-08 here). (Executed 2026-08-03.)"),
    },

    # ── phantom APIs the catalog used to name ───────────────────────
    "phantom_apis_checked": {
        "description": (
            "Names that appear in tutorials, older releases or earlier "
            "revisions of this catalog and DO NOT EXIST in the "
            "installed package. Checked 2026-08-03 by hasattr on the "
            "imported modules and by grepping the whole install tree "
            "(python sources AND the installed C++ headers)."),
        "dune.fem.space.product_space": (
            "ABSENT. hasattr(dune.fem.space, 'product_space') is False "
            "and the string 'product_space' does not occur anywhere "
            "under site-packages/dune or include/dune. The real "
            "multi-field factories are dune.fem.space.product, "
            ".composite and .combined (all with signature "
            "(*spaces, **kwargs)). The shipped mixed-methods template "
            "imports it as `from dune.fem.space import product as "
            "product_space`, which is why the alias looked like an "
            "API."),
        "dune.fem.space.raviartthomas": (
            "ABSENT (lowercase). The Python factory is raviartThomas "
            "(camelCase); only the C++ header is lowercase. "
            "Re-confirmed at runtime 2026-08-03: "
            "hasattr(dune.fem.space, 'raviartThomas') is True, "
            "hasattr(..., 'raviartthomas') is False."),
        "space.update()": (
            "ABSENT. hasattr(space, 'update') is False and no "
            "\"update\" pybind11 binding exists under "
            "include/dune/fempy/. After adaptation the space resizes "
            "itself; the discrete functions passed to "
            "dune.fem.adapt([uh, ...]) are what gets prolonged."),
    },

    # ── module-level API inventory ──────────────────────────────────
    "module_inventory_2_12": {
        "description": (
            "Names CONFIRMED PRESENT by dir()/hasattr on the installed "
            "modules, 2026-08-03. Use this to check a name exists "
            "BEFORE writing a script around it. These are curated "
            "SUBSETS of the real dir() output, so absence from this "
            "list is NOT evidence that a name is missing — for that, "
            "see phantom_apis_checked or just run hasattr yourself."),
        "dune.fem": (
            "GridFunction, GridMarker, Parameter, SpaceMarker, adapt, "
            "assemble, comm, discretefunction, doerflerMark, function, "
            "globalRefine, gridAdapt, integrate, loadBalance, mark, "
            "markNeighbors, model, operator, parameter, plotting, "
            "scheme, setVerbosity, space, spaceAdapt, threading, "
            "utility, view"),
        "dune.fem.space": (
            "bdfm, bdm, combined, composite, dganisotropic, "
            "dglagrange, dglagrangelobatto, dglegendre, dglegendrehp, "
            "dgonb, dgonbhp, finiteVolume, interpolate, lagrange, "
            "lagrangehp, p1Bubble, product, project, rannacherTurek, "
            "raviartThomas"),
        "dune.fem.scheme": (
            "dg, dgGalerkin, galerkin, h1, h1Galerkin, linearized, "
            "molGalerkin, solve"),
        "dune.fem.view": (
            "adaptiveLeafGridView, filteredGridView, geometryGridView"),
        "dune.grid": (
            "albertaGrid, cartesianDomain, gridFunction, onedGrid, "
            "reader, structuredGrid, tensorProductCoordinates, "
            "ugGrid, yaspGrid; reader is an enum with members dgf, "
            "dgfString, gmsh, meshio, structured"),
        "dune.alugrid": (
            "aluConformGrid, aluCubeGrid, aluGrid, aluSimplexGrid, "
            "reader"),
        "dune.ufl": (
            "BoundaryId, BoxDirichletBC, Constant, DirichletBC, "
            "GridFunction, MixedFunctionSpace, NamedConstant, Space, "
            "cell"),
        "dune.fem.threading": (
            "max and use are attributes; useMax is a callable"),
        "dune.fem.parameter": "append, exists, get, log, write",
        "dune.fem.utility": (
            "gridWidth, inspectBoundaryIds, lineSample, pointSample, "
            "boundarySample, Sampler"),
        "there_is_no_dune.fem.solver_module": (
            "import dune.fem.solver raises ModuleNotFoundError — "
            "solvers are selected by the `solver=` argument to the "
            "scheme, not by importing a solver module."),
        "some_of_these_are_ATTRIBUTES_not_modules": (
            "dune.fem.parameter and dune.fem.threading come from the "
            "compiled dune/fem/_fem.so and are attributes of "
            "dune.fem, NOT importable modules: `import "
            "dune.fem.parameter` raises ModuleNotFoundError while "
            "`import dune.fem; dune.fem.parameter.get(...)` works. "
            "Measured 2026-08-03."),
    },
}


# ── Cross-cutting pitfalls appended to KNOWLEDGE["poisson"] ──────────
#
# These live under `poisson` (rather than only under `_general`) because
# knowledge(topic="pitfalls", solver="dune") walks supported_physics()
# and never sees `_general`, and poisson is the physics every model
# touches first.

EXECUTED_PITFALLS: list[str] = [
    (
        "[Numerical] A dune.ufl.DirichletBC only takes effect if it "
        "is IN THE LIST passed to the scheme — galerkin([a == b, dbc], "
        "...). There is no separate apply step, and building the "
        "scheme as galerkin(a == b, space=space) after constructing a "
        "dbc object silently drops it. Signal: the run COMPLETES and "
        "scheme.solve() returns {'converged': True, ...}; the tells "
        "are linear_iterations in the thousands where the constrained "
        "problem needs tens, and max(uh.as_numpy) exploding. "
        "(Executed 2026-08-03, dune-fem 2.12.0.2, 16x16 "
        "structuredGrid, -Delta u = 2 pi^2 sin(pi x) sin(pi y): with "
        "the BC in the list, converged=True, linear_iterations=1, L2 "
        "err 1.899705e-03; with the SAME form and the BC omitted, "
        "converged=True, linear_iterations=23935, L2 err "
        "7.510512e+14.)"
    ),
    (
        "[Numerical] The optional third argument of DirichletBC is a "
        "UFL conditional over SpatialCoordinate(space), and nothing "
        "checks that it selects any facet. A wrong sign or threshold "
        "degrades to no boundary condition at all. Signal: identical "
        "symptoms to omitting the BC — converged=True with thousands "
        "of linear iterations and a solution many orders of magnitude "
        "too large. (Executed 2026-08-03: "
        "DirichletBC(space, 0, conditional(x[0] < -1.0, 1, 0)) on "
        "[0,1]^2 gave converged=True, linear_iterations=23935, L2 err "
        "7.510512e+14 — byte-identical to the omitted-BC run. The "
        "same BC with conditional(x[0] < 1e-8, 1, 0) gave 45 "
        "iterations and a well-posed L2 err of 2.676075e+00.)"
    ),
    (
        "[API] The `solver=` string is not validated in Python; it is "
        "forwarded as the C++ parameter 'fem.solver.linear.method' and "
        "checked when the scheme is constructed. For the default "
        "storage the accepted values are exactly cg, gmres and "
        "bicgstab. Signal: solver='conjugate_gradient' raises "
        "RuntimeError \"ParameterInvalid [getEnumeration:.../dune/fem/"
        "io/parameter/reader.hh:300]: Parameter "
        "'fem.solver.linear.method' invalid. Valid values are: gmres, "
        "cg, bicgstab\" — an error that names a parameter you never "
        "wrote. Direct solves use a tuple instead: "
        "solver=('suitesparse', 'umfpack'). (Executed 2026-08-03.)"
    ),
    (
        "[API] Parameter keys beginning with 'newton.' are deprecated "
        "in favour of 'nonlinear.', and dune-fem REWRITES them for "
        "you. Signal: parameters={'newton.tolerance': 1e-10} emits "
        "UserWarning \"the parameter key 'newton' is deprecated. "
        "Replace with 'nonlinear'\" and scheme.parameters comes back "
        "as {'nonlinear.tolerance': 1e-10, ...}. Passing BOTH "
        "'newton.tolerance' and 'nonlinear.tolerance' is accepted with "
        "no error at all and the newton.* value is silently dropped — "
        "only the nested 'newton.linear.*'/'nonlinear.linear.*' "
        "collision raises KeyError. (Executed 2026-08-03.)"
    ),
    (
        "[API] A dune-fem space reports a SIMPLEX UFL cell whatever "
        "the grid is really made of, because "
        "dune/ufl/__init__.py::_cell maps dimension straight to "
        "ufl.triangle / ufl.tetrahedron. Signal: on "
        "structuredGrid([0,0],[1,1],[4,4]) — 16 elements of type "
        "'quadrilateral' — space.cell() returns 'triangle' and "
        "repr(space) prints '<Lagrange1 on a triangle>'. Read the cell "
        "shape from gridView.type or from "
        "{e.type for e in gridView.elements}, never from the UFL "
        "element — and NOT from the dof count either: continuous "
        "Lagrange has 25 dofs at order 1 on both the 16-cube grid and "
        "the 32-triangle aluConformGrid over the same domain, since "
        "they share the same vertices. Only the element count (16 vs "
        "32) and the geometry type separate them. (Executed "
        "2026-08-03.)"
    ),
    (
        "[API] There is no order-0 Lagrange space: "
        "lagrange(gridView, order=0) raises KeyError 'Parameter error "
        "in LagrangeSpace with order=0: order has to be greater or "
        "equal to 1'. Signal: the FEniCS habit of asking for a "
        "('DG', 0) / order-0 space for a piecewise-constant field "
        "dies here; use dglagrange(gridView, order=0) or "
        "finiteVolume(gridView), both of which gave .size == 16 == "
        "the cell count on a 4x4 grid. (Executed 2026-08-03.)"
    ),
    (
        "[Numerical] dune.fem.globalRefine(level, uh) — the call form "
        "that is supposed to refine AND prolong — is a SILENT NO-OP "
        "on a YaspGrid (structuredGrid). Signal: gridView.size(0), "
        "space.size and len(uh.as_numpy) all come back UNCHANGED with "
        "no exception and no warning, so every 'level' of a "
        "refinement study reports the same error. It works only on "
        "adaptiveLeafGridView(aluConformGrid(...)); on a plain "
        "ALUGrid leaf view it at least raises RuntimeError "
        "'NotImplemented [numBlocks:...indexsetdofmapper.hh:228]'. "
        "Assert the element count grew, or refine with "
        "globalRefine(level, gridView.hierarchicalGrid) — which does "
        "work everywhere (16 -> 64 on a 4x4 YaspGrid) but does not "
        "prolong your functions. (Executed 2026-08-03, all four "
        "grid/view combinations measured: yasp plain and yasp "
        "adaptiveLeaf both (16, 25, 25) -> (16, 25, 25); alu "
        "adaptiveLeaf (32, 25, 25) -> (64, 41, 41).)"
    ),
    (
        "[Integration] A DUNE-fem process can print every correct "
        "result and still exit non-zero, because ALUGrid teardown "
        "after the last statement can trip PETSc's signal handler. "
        "Signal: stdout is complete and correct, then 'PETSc ERROR: "
        "Caught signal number 11 SEGV' and 'terminate called after "
        "throwing an instance of "
        "Dune::ExceptionStream<Dune::Petsc::Exception>', exit code "
        "134. It is INTERMITTENT — the same script gave 134 / 0 / 134 "
        "across three runs with byte-identical stdout — so treat a "
        "written results file or a terminal sentinel line as the "
        "success criterion and the returncode as secondary. "
        "(Executed 2026-08-03; a minimal one-grid script never "
        "reproduced it, so this is not a blanket ALUGrid claim.)"
    ),
    (
        "[Performance] The JIT cache is at <sys.prefix>/.cache/dune-py "
        "(inside a venv or conda env) or ~/.cache/dune-py otherwise, "
        "overridable with $DUNE_PY_DIR; the compiled modules are "
        "<that>/python/dune/generated/*.so. Signal: 'DUNE-INFO: "
        "Compiling <X> (new)' on stderr means a C++ build is running "
        "and the process will produce no further output for tens of "
        "seconds to minutes. Deleting the cache directory resets the "
        "cost. (Executed 2026-08-03, dune-fem 2.12.0.2: "
        "getDunePyDir() returned the conda env's .cache/dune-py, "
        "~/.dune does not exist, and building five ALUGrid variants "
        "cold took 439 s.)"
    ),
]
