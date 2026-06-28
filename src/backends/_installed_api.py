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

INSTALLED_API = {
 "ngsolve": {
  "version": "6.2.2604",
  "run": "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python <script>.py",
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
    "Point eval: `gfu(mesh(0.5,0.5))` — the coords MUST go through mesh(...); `gfu(0.5,0.5)` does NOT work.",
    "Integrate(cf*dx, mesh) is a SEPARATE functional, not how you assemble forms.",
  ],
 },
 "skfem": {
  "version": "12.0.1",
  "run": "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python <script>.py",
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
  "version": "9.8.0-pre  (build tree at /home/alexander/dealii/build)",
  "run": "cmake -DDEAL_II_DIR=/home/alexander/dealii/build . && make && LD_LIBRARY_PATH=/opt/4C-dependencies/lib ./<exe>",
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
    "CRITICAL: DEAL_II_DIR must be the BUILD tree `/home/alexander/dealii/build` (config at .../build/lib/cmake/deal.II). Pointing at `/home/alexander/dealii` SILENTLY falls back to the OLD system install (9.1.1 at /usr) with no error — check cmake's `-- Using the deal.II-X found at ...` line.",
    "Runtime needs `LD_LIBRARY_PATH=/opt/4C-dependencies/lib` (shared TBB/etc).",
    "CMake order: FIND_PACKAGE(deal.II 9.0 REQUIRED HINTS ${DEAL_II_DIR}) -> DEAL_II_INITIALIZE_CACHED_VARIABLES() -> PROJECT() -> DEAL_II_SETUP_TARGET(<tgt>). INITIALIZE must precede PROJECT().",
    "Use modern idioms: fe_values.quadrature_point_indices(), fe_values.dof_indices(), fe.n_dofs_per_cell() (data member fe.dofs_per_cell is deprecated).",
    "Functions::ZeroFunction<dim>() (namespaced; bare ZeroFunction removed). Header <deal.II/base/function.h>.",
    "Sparsity needs both <.../dynamic_sparsity_pattern.h> and <.../sparsity_pattern.h>; BCs need <.../numerics/vector_tools.h> + <.../numerics/matrix_tools.h>.",
    "Evaluate: VectorTools::point_value(dof_handler, solution, Point<dim>(...)); solution.linfty_norm() for the max.",
  ],
 },
 "fourc": {
  "version": "build at /home/alexander/4C/build/4C",
  "run": "LD_LIBRARY_PATH=/opt/4C-dependencies/lib /home/alexander/4C/build/4C <input>.4C.yaml <output_prefix>",
  "verified_smoke_test": (
    "# Minimal single HEX8 linear-elastic cube (fixed at x=0, pulled at x=1), Statics, 2 steps.\n"
    "# Started from /home/alexander/4C/tests/input_files/solid_runtime_material_element_id.4C.yaml\n"
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
    "Run: `LD_LIBRARY_PATH=/opt/4C-dependencies/lib /home/alexander/4C/build/4C <in>.4C.yaml <output_prefix>` — the output prefix is MANDATORY.",
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
     "Extract with pyvista (available in /home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python): `import pyvista as pv, numpy as np; m = pv.read(LAST_vtu); d = np.asarray(m.point_data['displacement']); pts = m.points`. Find your node by coordinate: `i = np.argmin(np.linalg.norm(pts - target_xyz, axis=1))`, then `d[i]` is its displacement (stress/strain are cell or point arrays too).",
     "GOTCHA: read the LAST timestep file (highest number, e.g. structure-00005-0.vtu), NOT structure-00000-0.vtu which is the INITIAL zero state -> reading step 0 gives displacement 0 everywhere and looks like the load did nothing.",
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
