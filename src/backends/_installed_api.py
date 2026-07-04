"""Installed-version API references — VERIFIED by actually running on this machine.

Models repeatedly fail not on the physics but on writing API calls for the WRONG
version of the installed code (e.g. NGSolve Integrate() signature, deal.II pointing
DEAL_II_DIR at the source instead of the build tree, the 4C YAML schema). Each entry
below is a minimal smoke test that was WRITTEN, RUN, and FIXED until it executed
cleanly on the version installed here, plus the version-specific gotchas observed.

These are GENERAL API references (a trivial -Laplacian(u)=1 / single-cube run), NOT
solutions to any benchmark — they teach the correct installed-version API so an agent
adapts a known-good call instead of guessing from (outdated) memory.

Surfaced to agents via prepare_simulation()/knowledge() for the matching backend.
Keyed by the backend registry name.
"""

import os as _os
from pathlib import Path as _Path

# Machine-specific prefixes, resolved at import time so the served references
# match THIS install. Override via env vars if your layout differs.
_REPO = _os.environ.get("OASIS_ROOT", str(_Path(__file__).resolve().parents[2]))
_HOME = str(_Path.home())
_VENV_PY = _REPO + "/.venv/bin/python"
_FENICS_PY = _os.environ.get("FENICS_PYTHON", _HOME + "/miniconda3/envs/fenics/bin/python")
_DEALII_BUILD = _os.environ.get("DEALII_BUILD_DIR", _HOME + "/dealii/build")
_FOURC_ROOT = _os.environ.get("FOURC_ROOT", _HOME + "/4C")
_FOURC_BIN = _os.environ.get("FOURC_BINARY", _FOURC_ROOT + "/build/4C")

INSTALLED_API = {
 "dune": {
  "version": "dune-fem 2.12.0.0 (ufl 2024.2.0, python 3.12)",
  "run": _VENV_PY + " <script>.py",
  "verified_smoke_test": (
    "import numpy as np\n"
    "from dune.grid import structuredGrid\n"
    "from dune.fem.space import lagrange\n"
    "from dune.fem.scheme import galerkin\n"
    "from dune.ufl import DirichletBC\n"
    "from ufl import TrialFunction, TestFunction, SpatialCoordinate, dx, grad, dot, sin, pi\n"
    "import dune.fem\n"
    "gridView = structuredGrid([0, 0], [1, 1], [8, 8])      # leaf view directly (quads)\n"
    "space = lagrange(gridView, order=1)\n"
    "u, v = TrialFunction(space), TestFunction(space)\n"
    "x = SpatialCoordinate(space)\n"
    "exact = sin(pi*x[0]) * sin(pi*x[1])\n"
    "a = dot(grad(u), grad(v)) * dx\n"
    "l = 2*pi*pi * exact * v * dx\n"
    "scheme = galerkin([a == l, DirichletBC(space, 0)], solver='cg')  # BC goes IN the eq list\n"
    "uh = space.interpolate(0, name='uh')\n"
    "scheme.solve(target=uh)                                # solves IN PLACE into uh\n"
    "e2 = dune.fem.integrate((uh - exact)**2, gridView=gridView, order=5)\n"
    "print('L2 error =', np.sqrt(e2))                       # -> 7.588e-03 on 8x8; rate 2.0 on refine\n"),
  "gotchas": [
    "Dirichlet BCs go INTO the equation list: `galerkin([a == l, DirichletBC(space, value)])` — NOT attached to the solve call as in FEniCS. The value may be a constant OR a UFL expression (inhomogeneous BCs verified).",
    "`scheme.solve(target=uh)` solves IN PLACE into a pre-created `uh = space.interpolate(0, name='uh')` and returns an info dict ({'converged', 'iterations' (=Newton steps), 'linear_iterations', ...}). It always wraps a Newton loop (0 extra steps for linear problems).",
    "Error norms: `dune.fem.integrate(expr, gridView=gv, order=5)` -> float. (The old `dune.fem.function.integrate(gv, expr, order)` still works but is deprecated and has a DIFFERENT argument order.)",
    "Global refinement: `gridView.hierarchicalGrid.globalRefine(1)` — the existing gridView/space/scheme all track it automatically (just re-solve; no re-creation). `dune.fem.globalRefine(1, hgrid)` is the discrete-function-aware variant (prolongs existing DFs).",
    "CRITICAL: point-evaluating a discrete function at a coordinate, `uh((0.5, 0.5))`, HANGS at 100% CPU (infinite float()/complex() recursion in this dune-2.12/ufl-2024.2 combo). Use `from dune.fem.utility import pointSample; pointSample(uh, [0.5, 0.5])` (verified) or `lineSample(uh, p0, p1, n)`.",
    "JIT: every new form/scheme signature triggers a C++ compile into ~/.cache/dune-py (minutes on FIRST use, instant when warm), serialized on a global cmake.lock — NEVER run two dune scripts concurrently on first compile; a silent zero-output 'hang' usually means it is compiling (or waiting on the lock), not a code bug.",
    "Simplex meshes: `from dune.alugrid import aluConformGrid; aluConformGrid(dune.grid.cartesianDomain([0,0],[1,1],[4,4]))`. Dof vector: `uh.as_numpy`; VTK: `gridView.writeVTK('name', pointdata=[uh])`.",
    "Cosmetic: an 'Invalid MIT-MAGIC-COOKIE-1 key' X11 warning prints at import WITHOUT a trailing newline (it concatenates with your first stdout line — beware when parsing logs).",
  ],
 },
 "fenics": {
  "version": "dolfinx 0.10.0",
  "run": "LD_LIBRARY_PATH=/opt/precice/lib " + _FENICS_PY + " <script>.py",
  "verified_smoke_test": (
    "from mpi4py import MPI\n"
    "import numpy as np, dolfinx, dolfinx.fem.petsc, ufl   # fem.petsc import is MANDATORY\n"
    "mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)\n"
    "V = dolfinx.fem.functionspace(mesh, ('Lagrange', 1))   # lowercase factory\n"
    "u, v = ufl.TrialFunction(V), ufl.TestFunction(V)\n"
    "a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx\n"
    "L = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0)) * v * ufl.dx\n"
    "def bnd(x): return np.isclose(x[0],0)|np.isclose(x[0],1)|np.isclose(x[1],0)|np.isclose(x[1],1)\n"
    "bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0),\n"
    "                             dolfinx.fem.locate_dofs_geometrical(V, bnd), V)\n"
    "problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc],\n"
    "    petsc_options_prefix='poisson_',                    # REQUIRED kwarg in 0.10\n"
    "    petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'})\n"
    "uh = problem.solve()                                    # returns the Function directly\n"
    "e2 = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(uh, uh) * ufl.dx))\n"
    "print('max(u) =', uh.x.array.max())                     # -> ~0.0736\n"),
  "gotchas": [
    "`petsc_options_prefix` is a REQUIRED keyword-only arg of LinearProblem AND NonlinearProblem in 0.10 — omitting it raises TypeError. Any distinct string works.",
    "`import dolfinx.fem.petsc` (and `dolfinx.nls.petsc` for the legacy Newton path) must be EXPLICIT — `import dolfinx` alone gives AttributeError on fem.petsc.",
    "Function spaces: lowercase `dolfinx.fem.functionspace(mesh, ('Lagrange', 1))`. The capitalized FunctionSpace class cannot be built from a (family, degree) tuple.",
    "`interpolate()` DOES accept plain python callables of x (shape (3, npts); vector fields return np.vstack of components). It does NOT accept raw UFL expressions — wrap those in `dolfinx.fem.Expression(expr, V.element.interpolation_points)`; NOTE `interpolation_points` is a PROPERTY in 0.10 (no parentheses).",
    "`uh.x.petsc_vec`, never `uh.vector` (gone in 0.10). Raw dofs: `uh.x.array`; after parallel writes: `uh.x.scatter_forward()`.",
    "`tabulate_dof_coordinates()` returns (ndofs, 3) even in 2D — slice [:, :2]. Point-evaluation coordinates must be 3D too: np.array([[x, y, 0.0]]).",
    "Mixed elements: `basix.ufl.element('Lagrange', mesh.basix_cell(), 2, shape=(2,))` + `dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([P2, P1]))`; split with ufl.TrialFunctions/TestFunctions (PLURAL). Collapse: `V, V_to_W = W.sub(0).collapse()` (returns a tuple).",
    "Subspace BC: value as a Function on the COLLAPSED space + dofs from the two-space form `locate_dofs_geometrical((W.sub(0), V), marker)` + `dirichletbc(func, dofs_pair, W.sub(0))`. Extract components: `wh.sub(0).collapse()`.",
    "SADDLE-POINT (Stokes/mixed) with all-Dirichlet velocity: pressure is only defined up to a constant -> PIN ONE PRESSURE DOF as a subspace BC on W.sub(1) at a corner (verified to machine precision with LU/mumps). Without the pin the system is singular and the answer (and any convergence rate from it) is garbage.",
    "NONLINEAR — BIGGEST 0.10 BREAK: `dolfinx.fem.petsc.NonlinearProblem(F, u, bcs=..., petsc_options_prefix=..., petsc_options={...})` is now SNES-BASED with its own `.solve()` (configure Newton via petsc_options: {'snes_type':'newtonls','snes_rtol':1e-10,'ksp_type':'preonly','pc_type':'lu'}). The old Newton-companion class was RENAMED `NewtonSolverNonlinearProblem` (for the legacy `dolfinx.nls.petsc.NewtonSolver` path, deprecated). The nonlinear unknown is a fem.Function (NOT a TrialFunction) used directly in the residual. Prefer the SNES path.",
    "Scalar norms: `assemble_scalar(form(...))` is the LOCAL contribution — always `mesh.comm.allreduce(val, op=MPI.SUM)` before sqrt.",
    "Use `dolfinx.default_scalar_type` for Constants/BC scalars to avoid dtype mismatches.",
  ],
 },
 "ngsolve": {
  "version": "6.2.2604",
  "run": _VENV_PY + " <script>.py",
  "verified_smoke_test": (
    "from ngsolve import *\n"
    "from netgen.geom2d import unit_square\n"
    "mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))\n"
    "fes = H1(mesh, order=1, dirichlet='.*')\n"
    "u, v = fes.TnT()\n"
    "a = BilinearForm(fes); a += grad(u)*grad(v)*dx; a.Assemble()\n"
    "f = LinearForm(fes); f += 1*v*dx; f.Assemble()\n"
    "gfu = GridFunction(fes)\n"
    "gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse='sparsecholesky') * f.vec\n"
    "print('center value:', gfu(mesh(0.5, 0.5)))   # -> ~0.069\n"),
  "gotchas": [
    "Mesh: do NOT call Mesh() on a unit square; use `from netgen.geom2d import unit_square; Mesh(unit_square.GenerateMesh(maxh=0.2))`.",
    "Dirichlet BC goes on the SPACE: H1(mesh, order=1, dirichlet='.*'); '.*' matches all (unnamed) boundaries robustly.",
    "Trial/test: `u, v = fes.TnT()`. Forms: `a += grad(u)*grad(v)*dx` then `a.Assemble()` (explicit).",
    "Solve: `gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse='sparsecholesky') * f.vec` — FreeDofs() enforces the Dirichlet constraint.",
    "INHOMOGENEOUS Dirichlet (u=g on the boundary, e.g. any MMS): do NOT guess `fes.Vec()`/`fes.GetBaseFunction()`/`sol.FV()` — none of those exist. Verified pattern: set boundary values then solve on the free DOFs with the lifted RHS:\n"
    "    gfu = GridFunction(fes)\n"
    "    gfu.Set(g_exact, BND)                       # impose u=g on the boundary\n"
    "    r = f.vec - a.mat * gfu.vec\n"
    "    gfu.vec.data += a.mat.Inverse(fes.FreeDofs()) * r\n"
    "  Here x,y,exp,sin,cos are NGSolve CoefficientFunctions (NOT numpy: `cf.exp` fails — use exp(cf)); take symbolic derivatives for the MMS source via `uex.Diff(x)` (e.g. f = -(uex.Diff(x).Diff(x)+uex.Diff(y).Diff(y))). Verified: gives optimal O(h^2) for P1.",
    "Point eval: `gfu(mesh(0.5,0.5))` — the coords MUST go through mesh(...); `gfu(0.5,0.5)` does NOT work.",
    "Integrate(cf*dx, mesh) is a SEPARATE functional, not how you assemble forms.",
    "DIRECT SOLVER: `inverse='pardiso'` is NOT available in this build (MKL Pardiso missing) — use `inverse='umfpack'` (also handles indefinite saddle-point systems). Options seen: 'sparsecholesky' (SPD), 'umfpack' (general).",
    "STEADY NAVIER-STOKES / saddle-point (Taylor-Hood = VectorH1(order=2) * H1(order=1)): if velocity is Dirichlet on the WHOLE boundary, pressure is only defined up to a constant -> the saddle matrix is SINGULAR and Newton 'diverges' (garbage increments, residual oscillates). FIX = pin the null space: add a NumberSpace Lagrange multiplier for zero-mean pressure (X = V*Q*N; form has `+ lam*q + mu*p`) OR fix one pressure dof. NEWTON via symbolic linearization (no hand Jacobian): build residual BilinearForm `a` in the unknown (convection term `InnerProduct(grad(u)*u, v)`), then loop: `a.Apply(gfu.vec,res); a.AssembleLinearization(gfu.vec); du.data = a.mat.Inverse(X.FreeDofs(), inverse='umfpack')*res; gfu.vec.data -= du`. Inhom velocity BC on the component: `gfu.components[0].Set(uex, definedon=mesh.Boundaries('.*'))`. Converges quadratically (~4 its) at nu=0.1.",
  ],
 },
 "skfem": {
  "version": "12.0.1",
  "run": _VENV_PY + " <script>.py",
  "verified_smoke_test": (
    "import numpy as np\n"
    "from skfem import MeshTri, ElementTriP1, Basis, BilinearForm, LinearForm, asm, condense, solve\n"
    "from skfem.helpers import dot, grad\n"
    "mesh = MeshTri().refined(4)          # MeshTri() IS the unit square; .refined returns a NEW mesh\n"
    "basis = Basis(mesh, ElementTriP1())\n"
    "@BilinearForm\n"
    "def stiffness(u, v, w): return dot(grad(u), grad(v))\n"
    "@LinearForm\n"
    "def load(v, w): return 1.0 * v\n"
    "A = stiffness.assemble(basis); b = load.assemble(basis)\n"
    "D = basis.get_dofs()                  # all boundary dofs\n"
    "x = solve(*condense(A, b, D=D))\n"
    "print('max u =', x.max())            # -> ~0.073\n"),
  "gotchas": [
    "MeshTri() already IS the unit square; refine with `.refined(n)` (returns a new mesh, not in-place).",
    "Use generic `Basis(mesh, ElementTriP1())` (element instance required).",
    "Decorator arities are mandatory: `@BilinearForm def f(u, v, w)` and `@LinearForm def f(v, w)` — the trailing `w` (global params) is required even if unused. Wrong arity is the #1 error.",
    "Integrand uses skfem.helpers (dot, grad) and returns the pointwise integrand, not an assembled value.",
    "Assemble via `form.assemble(basis)` or `asm(form, basis)` (equivalent).",
    "Dirichlet: `D = basis.get_dofs()` (no args = all boundary dofs) then `solve(*condense(A, b, D=D))` (homogeneous by default).",
  ],
 },
 "dealii": {
  "version": "9.8.0-pre  (build tree at " + _DEALII_BUILD + ")",
  "run": "cmake -DDEAL_II_DIR=" + _DEALII_BUILD + " . && make && LD_LIBRARY_PATH=/opt/4C-dependencies/lib ./<exe>",
  "verified_smoke_test": (
    "// CMakeLists.txt:\n"
    "//   CMAKE_MINIMUM_REQUIRED(VERSION 3.13.4)\n"
    "//   FIND_PACKAGE(deal.II 9.0 REQUIRED HINTS ${DEAL_II_DIR})\n"
    "//   DEAL_II_INITIALIZE_CACHED_VARIABLES()   # BEFORE project()\n"
    "//   PROJECT(poisson CXX)\n"
    "//   ADD_EXECUTABLE(poisson poisson.cc); DEAL_II_SETUP_TARGET(poisson)\n"
    "// poisson.cc: standard step-3 Poisson, Q1, refine_global(5), -Laplacian(u)=1, u=0 on bdry.\n"
    "//   Functions::ZeroFunction<dim>() for the BC; VectorTools::point_value(dof_handler, solution,\n"
    "//   Point<dim>(0.5,0.5)) -> ~0.0737 ; SolverCG + PreconditionIdentity.\n"),
  "gotchas": [
    "CRITICAL: DEAL_II_DIR must be the BUILD tree `" + _DEALII_BUILD + "` (config at .../build/lib/cmake/deal.II). Pointing at its parent source dir SILENTLY falls back to the OLD system install (9.1.1 at /usr) with no error — check cmake's `-- Using the deal.II-X found at ...` line.",
    "Runtime needs `LD_LIBRARY_PATH=/opt/4C-dependencies/lib` (shared TBB/etc).",
    "CMake order: FIND_PACKAGE(deal.II 9.0 REQUIRED HINTS ${DEAL_II_DIR}) -> DEAL_II_INITIALIZE_CACHED_VARIABLES() -> PROJECT() -> DEAL_II_SETUP_TARGET(<tgt>). INITIALIZE must precede PROJECT().",
    "Use modern idioms: fe_values.quadrature_point_indices(), fe_values.dof_indices(), fe.n_dofs_per_cell() (data member fe.dofs_per_cell is deprecated).",
    "Functions::ZeroFunction<dim>() (namespaced; bare ZeroFunction removed). Header <deal.II/base/function.h>.",
    "Sparsity needs both <.../dynamic_sparsity_pattern.h> and <.../sparsity_pattern.h>; BCs need <.../numerics/vector_tools.h> + <.../numerics/matrix_tools.h>.",
    "Evaluate: VectorTools::point_value(dof_handler, solution, Point<dim>(...)); solution.linfty_norm() for the max.",
    "NONLINEAR (Newton) — deal.II nonlinear WORKS here (verified clean O(h^2)); a segfault is YOUR code, not the install. Pattern: `current_solution` is a Vector sized to dof_handler.n_dofs() holding the ITERATE (not the update). Each step assembles system_matrix=Jacobian and system_rhs = MINUS residual, linearized about current_solution: per cell `fe_values.reinit(cell)` then `fe_values.get_function_values/get_function_gradients(current_solution, ...)` for the current state (e.g. k=1+u^2, k'=2u -> J_ij=(k*grad_phi_j.grad_phi_i + k'*phi_j*(grad_u.grad_phi_i))*JxW, R_i=(k*(grad_u.grad_phi_i)-f*phi_i)*JxW). Solve J*delta=-R; `current_solution += delta`; loop until residual l2_norm small.",
    "Newton BCs: set Dirichlet values on current_solution ONCE before iterating, then every Newton UPDATE is homogeneous (ZeroFunction boundary values applied to matrix+rhs each assembly) so the boundary stays fixed.",
    "SEGFAULT in assembly is almost always: (1) `fe_values.reinit(cell)` missing/mis-scoped; (2) update flag missing — need `update_values|update_gradients|update_quadrature_points|update_JxW_values` (reading grads/points without the flag crashes); (3) current_solution not sized to n_dofs() before get_function_*; (4) DoFs not distributed; (5) cell_matrix/local_dof_indices sized with wrong dofs_per_cell.",
    "THIS deal.II IS A RELEASE BUILD: all Assert checks are compiled OUT, so beginner mistakes segfault SILENTLY or give garbage instead of a friendly assertion. Absence of a diagnostic is expected — it is NOT evidence of a broken library.",
    "VARIABLE-COEFFICIENT / custom `Function<dim>` classes (verified): exact override `double value(const Point<dim> &p, const unsigned int component = 0) const override` — ALWAYS write `override` (a wrong signature without it silently hides the base and yields NaN -> 'residual was nan' at solve, not a helpful error). Evaluate at q-points via `coefficient.value(fe_values.quadrature_point(q))` — which REQUIRES `update_quadrature_points` in the FEValues flags (missing flag = the #1 verified cause of 'compiles fine, segfaults in the cell loop').",
    "SparsityPattern LIFETIME: the SparsityPattern must OUTLIVE the SparseMatrix (declare it before the matrix, same scope — never a local in setup()). A destroyed pattern = verified segfault at the first system_matrix.add(), i.e. 'after distribute_dofs, before assembly'.",
    "MISDIAGNOSIS GUARD: before blaming the installation, build+run a known-good minimal case (step-3 pattern) with the same CMake + LD_LIBRARY_PATH — if that runs (it does on this machine, rate 2), the install is FINE and the segfault is YOUR code: check update_quadrature_points, SparsityPattern lifetime, `override` on every Function::value.",
  ],
 },
 "fourc": {
  "version": "build at " + _FOURC_BIN,
  "run": "LD_LIBRARY_PATH=/opt/4C-dependencies/lib " + _FOURC_BIN + " <input>.4C.yaml <output_prefix>",
  "verified_smoke_test": (
    "# Minimal single HEX8 linear-elastic cube (fixed at x=0, pulled at x=1), Statics, 2 steps.\n"
    "# Started from " + _FOURC_ROOT + "/tests/input_files/solid_runtime_material_element_id.4C.yaml\n"
    "# Runs to completion: stdout ends 'processor 0 finished normally' / EXIT:0; writes <prefix>.control + VTK.\n"
    "PROBLEM TYPE: {PROBLEMTYPE: 'Structure'}\n"
    "SOLVER 1: {SOLVER: 'Superlu', NAME: 'Structure_Solver'}\n"
    "STRUCTURAL DYNAMIC: {DYNAMICTYPE: 'Statics', TIMESTEP: 0.5, NUMSTEP: 2, MAXTIME: 1,\n"
    "                     TOLDISP: 1e-9, TOLRES: 1e-9, LINEAR_SOLVER: 1}\n"
    "MATERIALS: [{MAT: 1, MAT_Struct_StVenantKirchhoff: {YOUNG: 100, NUE: 0.3, DENS: 0}}]\n"
    "FUNCT1: [{SYMBOLIC_FUNCTION_OF_SPACE_TIME: 't'}]\n"
    "DESIGN SURF DIRICH CONDITIONS: [{E: 1, NUMDOF: 3, ONOFF: [1,1,1], VAL: [0,0,0], FUNCT: [0,0,0]}]\n"
    "DESIGN SURF NEUMANN CONDITIONS: [{E: 2, NUMDOF: 3, ONOFF: [1,0,0], VAL: [10,0,0], FUNCT: [1,0,0]}]\n"
    "DSURF-NODE TOPOLOGY: ['NODE 1 DSURFACE 1', ... 'NODE 5 DSURFACE 2', ...]\n"
    "NODE COORDS: ['NODE 1 COORD 0.0 0.0 0.0', ...8 nodes...]\n"
    "STRUCTURE ELEMENTS: ['1 SOLID HEX8 1 5 6 2 3 7 8 4 MAT 1 KINEM nonlinear']\n"),
  "gotchas": [
    "Run: `LD_LIBRARY_PATH=/opt/4C-dependencies/lib " + _FOURC_BIN + " <in>.4C.yaml <output_prefix>` — the output prefix is MANDATORY.",
    "File is one YAML map; keys are section names with spaces/slashes (e.g. 'STRUCTURAL DYNAMIC', 'IO/RUNTIME VTK OUTPUT/STRUCTURE').",
    "Required minimal: PROBLEM TYPE, SOLVER 1, STRUCTURAL DYNAMIC, MATERIALS, mesh sections, conditions.",
    "Time integrator references the linear solver via LINEAR_SOLVER: 1 (-> 'SOLVER 1').",
    "Materials: list, each {MAT: <id>, MAT_Struct_<Model>: {...}}; elements reference it by `MAT <id>`.",
    "Mesh is INLINE strings: NODE COORDS = ['NODE <id> COORD x y z', ...]; STRUCTURE ELEMENTS = ['<id> SOLID HEX8 <8 ids> MAT <id> KINEM nonlinear', ...] (raw strings, not maps).",
    "BCs attach to DESIGN entities, not nodes: DSURF-NODE TOPOLOGY maps nodes->surface ids ('NODE n DSURFACE d'); conditions reference d via E: d.",
    "Dirichlet/Neumann: lists of {E, NUMDOF, ONOFF: [...], VAL: [...], FUNCT: [...]}; ONOFF flags active dofs, FUNCT indexes a FUNCTn (0 = constant).",
    "Omit the optional RESULT DESCRIPTION section for a pure smoke run so regression asserts can't fail the job.",
    "Success = stdout 'processor 0 finished normally' / EXIT:0 and a <prefix>.control output file.",
  ],
  # Additional VERIFIED capability references (each written-run-fixed on this 4C, generic
  # problems that teach the schema — NOT any benchmark solution).
  "capabilities": [
   {"name": "Near-incompressible / locking-free structural analysis (what 4C can and cannot do)",
    "facts": [
     "4C has NO geometrically-LINEAR locking-free 2D element. The only 2D structural element is WALL/W1; its EAS is NONLINEAR-ONLY (source hard-throws 'No EAS for geometrically linear WALL element'), and reduced integration (GP 1 1) is unstabilized -> singular system. So a KINEM-linear locking-free nearly-incompressible run in 2D is NOT constructible — do not waste time trying; use nonlinear kinematics or go 3D.",
     "Locking-free / anti-volumetric-locking primitives (FBAR, richer EAS) exist for the 3D SOLID element (hex/tet/wedge); there is no 2D SOLID element. Use `KINEM nonlinear` + EAS on 2D WALL, or FBAR on 3D SOLID.",
     "Because those are FINITE-STRAIN, a valid MMS for them needs the finite-strain manufactured body force `f = -Div P` (P = 1st Piola-Kirchhoff), NOT the linear `-div(sigma)`. Keep the manufactured displacement amplitude MODERATE (O(1) finite-strain fields at near-incompressible lambda make Newton diverge).",
     "2D WALL Q1E4 EAS relieves shear/bending locking but does NOT cure VOLUMETRIC locking as nu->0.5 (verified: at nu=0.4999 it still stalls; at nu=0.3 it works and improves accuracy).",
     "Near-incompressible K is severely ill-conditioned: with a direct solver the residual floors around ~1e-9, so a tight TOLRES (1e-10) spuriously fails at max-iter even though the displacement update is ~1e-13. RELAX TOLRES (e.g. 1e-5) and rely on TOLDISP.",
    ]},
   {"name": "Spatially-varying body force + inhomogeneous Dirichlet via FUNCT (analytic / manufactured loads)",
    "facts": [
     "Define top-level FUNCTn blocks with a SPATIAL expression: `FUNCT1: [{COMPONENT: 0, SYMBOLIC_FUNCTION_OF_SPACE_TIME: 'x + 2.0*y'}]` (x,y,z are spatial coords, t is time; one block per scalar function). Reference a block by its 1-BASED NUMBER (FUNCT1 -> 1).",
     "FUNCT IS AN INDEX, NOT A VALUE. The applied value per dof = VAL[i] * FUNCT[i] where FUNCT[i] is the block index (0 = constant, i.e. just VAL[i]). Putting the load magnitude in the FUNCT field (or vice-versa) is THE classic bug.",
     "Body force over the domain: in 2D the domain is a SURFACE -> `DESIGN SURF NEUMANN CONDITIONS: [{E:1, NUMDOF:2, ONOFF:[1,1], VAL:[1,1], FUNCT:[1,2], TYPE:'Live'}]` with a DSURF-NODE TOPOLOGY listing ALL domain nodes under DSURFACE 1. (3D analog: DESIGN VOL NEUMANN + DVOL.)",
     "Inhomogeneous Dirichlet g(x,y): on a 2D edge `DESIGN LINE DIRICH CONDITIONS: [{E:1, NUMDOF:2, ONOFF:[1,1], VAL:[1,1], FUNCT:[3,3]}]` (value = VAL*FUNCT3) with a DLINE-NODE TOPOLOGY of exactly the boundary nodes. (3D: DESIGN SURF DIRICH.)",
     "MPI_ABORT at 4C_fem_discretization_conditions.cpp:55 means a condition's `E: n` has NO matching D{LINE,SURF,VOL}-NODE TOPOLOGY (wrong/missing number, or wrong topology keyword for the condition's dimension). Also: wrong dimension (no VOL in DIM:2 — body=SURF, BC=LINE), ONOFF/VAL/FUNCT lengths must all equal NUMDOF, and every referenced FUNCTn must be defined.",
     "2D element line: `STRUCTURE ELEMENTS: ['1 WALL QUAD4 1 2 5 4 MAT 1 KINEM linear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2']`. Verified runs to 'processor 0 finished normally'.",
    ]},
   {"name": "Elastoplasticity (von Mises / J2, isotropic hardening)",
    "facts": [
     "Material: MAT_Struct_PlasticLinElast {YOUNG, NUE, DENS, YIELD (initial yield stress), ISOHARD (linear isotropic hardening modulus, stress units), KINHARD (kinematic; 0 for pure isotropic), TOL (return-mapping Newton tol)}. This is a SMALL-STRAIN J2 model -> pair it with `KINEM linear` on the element. (Finite-strain: MAT_Struct_PlasticNlnLogNeoHooke; ductile damage: MAT_Struct_PlasticGTN.)",
     "CONVERGENCE GOTCHA: a single load jump well past yield stalls the local return-mapping Newton. RAMP the load over several steps (e.g. NUMSTEP 5, FUNCT 't') so only ~1% plastic strain accrues per step -> nlniter jumps 2->4 at yield and converges.",
     "Use a robust direct linear solver for tiny meshes (SOLVER 1: {SOLVER: 'UMFPACK'}). Apply load by prescribed displacement (DESIGN SURF DIRICH), fully clamp the opposite face to kill rigid-body modes.",
     "Confirm yielding: IO {STRUCT_STRAIN/STRUCT_PLASTIC_STRAIN: 'EA', STRUCT_STRESS: 'Cauchy'} + runtime VTK STRESS_STRAIN:true; compare von Mises stress to YIELD.",
    ]},
   {"name": "2D structural element + EDGE (line) traction",
    "facts": [
     "2D element: `<id> WALL QUAD4 <4 CCW node ids> MAT <id> KINEM linear EAS none THICK <t> STRESS_STRAIN plane_stress GP 2 2`. Missing/garbled tail params abort at input parse.",
     "EDGE traction in 2D attaches to LINES, not surfaces: use `DESIGN LINE NEUMANN CONDITIONS` (E: d) + a `DLINE-NODE TOPOLOGY` block ('NODE n DLINE d') for the loaded edge; clamp via `DESIGN LINE DIRICH CONDITIONS` + its own DLINE. NUMDOF 6, ONOFF[0]=1 turns on x-traction, VAL is traction per unit edge length (x THICK).",
     "ZERO-DISPLACEMENT GOTCHA (the classic 2D trap): if you attach the load to a DSURFACE in 2D, or DLINE-NODE TOPOLOGY is missing/points at the wrong nodes, 4C STILL runs and exits 0 but the load set is EMPTY -> displacement is zero everywhere. Always confirm every loaded-edge node appears under the Neumann DLINE.",
     "Dirichlet on the clamped edge must constrain BOTH dofs (ONOFF [1,1]) or the body is under-constrained.",
    ]},
   {"name": "Reading results out of 4C output (extract a nodal value)",
    "facts": [
     "By DEFAULT 4C writes only a BINARY <prefix>.control + binary result files — NOT human-readable. Do NOT try to hex-decode them by hand.",
     "To get readable output, add: `IO/RUNTIME VTK OUTPUT: {INTERVAL_STEPS: 1, OUTPUT_DATA_FORMAT: ascii}` and `IO/RUNTIME VTK OUTPUT/STRUCTURE: {OUTPUT_STRUCTURE: true, DISPLACEMENT: true}` (add `STRESS_STRAIN: true` for stresses). This writes `<prefix>-vtk-files/structure-0000N-0.vtu`.",
     "Extract with pyvista (available in " + _VENV_PY + "): `import pyvista as pv, numpy as np; m = pv.read(LAST_vtu); d = np.asarray(m.point_data['displacement']); pts = m.points`. Find your node by coordinate: `i = np.argmin(np.linalg.norm(pts - target_xyz, axis=1))`, then `d[i]` is its displacement (stress/strain are cell or point arrays too).",
     "GOTCHA: read the LAST timestep file (highest number, e.g. structure-00005-0.vtu), NOT structure-00000-0.vtu which is the INITIAL zero state -> reading step 0 gives displacement 0 everywhere and looks like the load did nothing.",
     "STRESS output: set `IO: {STRUCT_STRESS: 'Cauchy'}` (this is what actually enables stress) AND `IO/RUNTIME VTK OUTPUT/STRUCTURE: {..., STRESS_STRAIN: true}`. Then stress is in BOTH point_data['nodal_cauchy_stresses_xyz'] and cell_data['element_cauchy_stresses_xyz'], shape (n,6) Voigt [xx,yy,zz,xy,yz,xz] -> index 0 = sigma_xx. e.g. `pv.read(LAST_pvtu).cell_data['element_cauchy_stresses_xyz'][cell,0]`.",
     "Read the explicit last `.pvtu`/`.vtu` (e.g. sorted(glob('...structure-*.pvtu'))[-1]); pv.read('<prefix>-structure.pvd') is unreliable (its MultiBlock may expose only the t=0 zero block).",
    ]},
   {"name": "Locking-free 2D element (avoid volumetric locking at nu -> 0.5)",
    "facts": [
     "Standard QUAD4 (`EAS none`) volumetrically LOCKS at nearly-incompressible nu (e.g. 0.4999) and in bending -> displacement grossly under-predicted (verified: tip uy was 75x too small with EAS none).",
     "Fix: use Enhanced Assumed Strain -> element line `... WALL QUAD4 <nodes> MAT <id> KINEM nonlinear EAS full THICK <t> STRESS_STRAIN plane_strain GP 2 2`. `EAS full` (Q1E4, 4 enhanced modes) recovers the correct flexible response.",
     "GOTCHA: EAS REQUIRES `KINEM nonlinear`. `KINEM linear EAS full` errors 'No EAS for geometrically linear WALL element'. For small-strain use KINEM nonlinear with a small load (geometrically ~linear).",
     "WALL exposes only `EAS none|full` (no mild/F-bar); EAS is QUAD4-only (rejected for TRI6). The 3D SOLID element family offers additional F-bar/EAS variants.",
     "Use EAS full whenever nu -> 0.5 (incompressible: rubber, J2 plasticity flow, biomechanics) or for thin/bending-dominated low-order meshes.",
    ]},
  ],
 },
}


def render(backend_name: str) -> str:
    """Markdown block of the verified installed-version API reference for a backend, or ''."""
    e = INSTALLED_API.get(backend_name)
    if not e:
        return ""
    out = [f"## Installed-version API reference (VERIFIED by running here) — {backend_name} {e['version']}",
           f"Run: `{e['run']}`",
           "Minimal smoke test that ACTUALLY RUNS on this install (adapt this API; do not guess from memory):",
           "```", e["verified_smoke_test"].rstrip(), "```",
           "Version-specific gotchas:"]
    out += [f"- {g}" for g in e["gotchas"]]
    for cap in e.get("capabilities", []):
        out.append(f"\n### Verified capability: {cap['name']}")
        out += [f"- {f}" for f in cap["facts"]]
    return "\n".join(out)
