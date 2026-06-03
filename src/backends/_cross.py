"""Cross-backend collation pitfalls.

When a simulation engineer ports the same problem between two backends —
e.g. validating a fenics result against a kratos run — the failure modes
are NOT in any single backend's catalog. They live in the *delta* between
two backends that both claim to solve the same problem.

This module collects those delta-pitfalls. Each entry follows the standard
[Category]+Signal format used elsewhere in the catalog, with the extra
constraint that the Signal clause MUST name at least two backends
side-by-side. A pitfall that's "just a fenics issue" or "just a kratos
issue" belongs in src/backends/<be>/, not here.

Surfaced via `knowledge(topic='cross_backend')` — see
src/tools/consolidated.py.
"""
from __future__ import annotations


_UNITS_DESC = (
    "All 8 backends are UNIT-AGNOSTIC: they perform arithmetic on numbers "
    "and assume the user has fed them a self-consistent set. None validates "
    "that your inputs form a coherent unit system. This is the single most "
    "common bug when porting a problem between backends."
)

_UNITS_PITFALLS = [
    "[Cross-Backend][Units] FEBio's documented user-manual default is "
    "mm-tonne-s (mass = tonne, length = mm, time = s, derived force = N, "
    "derived stress = MPa, derived density = tonne/mm^3 = 1e-9 SI density). "
    "Most fenics / kratos / dealii / ngsolve / skfem / dune / 4C examples "
    "use SI (m, kg, s, derived Pa, derived kg/m^3). Signal: same problem "
    "in two backends returns results differing by exactly 1e3, 1e6, or "
    "1e9 — that factor is the unit-system mismatch, not a numerical bug. "
    "Defense: pick ONE unit system upfront before touching any backend, "
    "put it in a single header constant (E_PA = 210e9 OR E_MPa = 210e3, "
    "never both), and convert ALL backend inputs to match. Verify with a "
    "dimensional sanity check on the FIRST timestep: max displacement "
    "should match analytical expectation to within 10%.",

    "[Cross-Backend][Units] Density conversion is the most-missed step. "
    "Steel in SI is 7850 kg/m^3; in mm-tonne-s it is 7.85e-9 tonne/mm^3 "
    "(NOT 7850, NOT 7.85e-6). Signal: explicit dynamics or modal analysis "
    "ported between SI fenics and mm-tonne-s FEBio gives natural "
    "frequencies off by a factor of sqrt(1e6) ~ 1000. Defense: when "
    "porting from SI to mm-tonne-s, divide density by 1e9 (NOT 1e6, NOT "
    "1e3); when porting from mm-tonne-s to SI, multiply density by 1e9.",

    "[Cross-Backend][Units] Material model parameters embedded in physics: "
    "e.g., a Neo-Hookean shear modulus mu=80 GPa is 8e10 Pa in SI but "
    "8e4 MPa in mm-tonne-s. Signal: hyperelastic problem ported between "
    "fenics (SI) and FEBio (mm-tonne-s) shows max stress off by 1e6 (Pa "
    "vs MPa) — easy to miss because both backends report numbers without "
    "units. Defense: write a units-comment block at the top of every "
    "input file that lists the unit of EACH material parameter, not just "
    "E and nu.",

    "[Cross-Backend][Units] Loads and BCs: an applied traction of 100 in "
    "SI (fenics / kratos / dealii defaults) means 100 Pa, in mm-tonne-s "
    "(FEBio defaults) means 100 MPa = 100e6 Pa — a 1e6 difference. "
    "Signal: contact / hyperelastic / plasticity problem produces "
    "unphysically large deformations in one backend (e.g. FEBio reading "
    "100 as 100 MPa), near-zero in another (e.g. fenics reading 100 as "
    "100 Pa), for the 'same' load value. Defense: tractions and "
    "pressures are where the mismatch bites hardest because every "
    "backend accepts a bare number. Always paste the unit-system "
    "constant next to the load definition in code review.",
]

_UNITS_SIGNAL = (
    "[Cross-Backend][Units] Aggregate: cross-backend numerical "
    "discrepancies that are CLEAN powers of 1000 (1e3, 1e6, 1e9, 1e12) "
    "are almost always unit-system mismatch, NOT a numerical bug. If you "
    "see exactly 1000x, 1e6x, or 1e9x between backends, fix the unit "
    "system before debugging anything else."
)


_NODE_ORDER_DESC = (
    "Mesh-format converters (Gmsh, VTK, ABAQUS .inp) and backend internal "
    "element orderings do NOT agree on the local numbering of element "
    "nodes. A quadratic tet has 10 nodes and the on-edge midpoint nodes "
    "appear in different positions in Gmsh order vs VTK order vs ABAQUS "
    "order vs each backend's internal order."
)

_NODE_ORDER_PITFALLS = [
    "[Cross-Backend][Mesh] Quadratic-tet (Tet10) midpoint-node ordering "
    "differs: Gmsh has midpoints in order (0,1)(1,2)(0,2)(0,3)(1,3)(2,3); "
    "ABAQUS uses (0,1)(1,2)(0,2)(0,3)(2,3)(1,3) — note swapped 5th/6th. "
    "VTK uses yet another ordering. fenics/dolfinx imports Gmsh order "
    "natively via dolfinx.io.gmshio.read_from_msh; dealii's GridIn "
    "re-orders to its own internal scheme; FEBio reads ABAQUS order. "
    "Signal: an MMS convergence study on a 10-node tet shows the right "
    "rate in one backend but a polluted rate (1.5 instead of 3) in "
    "another — caused by midpoints being mis-located after import. "
    "Defense: for Tet10 / Hex20 / Quad9, never trust a converter; verify "
    "with a single Gauss-point test: place a known polynomial f(x,y,z) "
    "on the nodes, evaluate at the cell centroid, compare across "
    "backends; mismatch reveals the ordering bug.",

    "[Cross-Backend][Mesh] Linear-quad face orientation: a 2D quad mesh "
    "has each face winding either CCW (mathematics convention, "
    "fenics/dolfinx default) or CW (some FEBio .feb files via legacy "
    "converters). Signal: a Stokes lid-driven cavity problem solved on "
    "the SAME .msh file by fenics and skfem gives velocity fields that "
    "look mirrored or flipped, because one backend interpreted the mesh "
    "winding opposite to the other. Defense: always pass meshes through "
    "one canonical converter (gmsh+meshio) before feeding any backend; "
    "never bridge ABAQUS->FEBio->fenics in one pipeline. Anchor with a "
    "single-element sanity test: apply a known body force, check that "
    "the centroid displacement matches analytical to 5 digits.",
]

_NODE_ORDER_SIGNAL = (
    "[Cross-Backend][Mesh] If a converged MMS rate in one backend "
    "becomes non-converged in another on the SAME mesh file, suspect "
    "node ordering BEFORE suspecting the discretisation. Validate with "
    "a sentinel: a single high-order element with a known analytic "
    "field, compared node-by-node."
)


_LE_SEMANTICS_DESC = (
    "The phrase 'linear elastic' is overloaded across backends. Some "
    "backends interpret it as small-strain (Cauchy stress = D * sym(grad "
    "u), Green strain tensor LINEARISED); some use the same template "
    "name but actually wire up Total Lagrangian Green-Lagrange / 2nd "
    "Piola-Kirchhoff. Both produce identical results in the small-strain "
    "limit but diverge when applied strain exceeds ~5%."
)

_LE_SEMANTICS_PITFALLS = [
    "[Cross-Backend][Physics] 'linear_elasticity' across backends: "
    "fenics/dolfinx + skfem + ngsolve + dune use small-strain (Cauchy "
    "stress = lambda*tr(eps)*I + 2*mu*eps with eps = sym(grad u), no F "
    "= I + grad u, no PK1 / PK2 machinery). FEBio's 'isotropic elastic' "
    "material (close name) uses the same small-strain form ONLY when "
    "the <analysis>STATIC</analysis> block is 'small-strain'; if the "
    "analysis is 'dynamic' or any kinematic_type other than "
    "small_strain, FEBio silently switches to a finite-strain PK1 "
    "evaluation of the SAME Hooke law. Kratos's LinearElasticIsotropic3D "
    "is always small-strain. 4C's MAT_ELAST_HOOKE in a NONLINEAR "
    "analysis is a geometrically-nonlinear St-Venant-Kirchhoff (finite "
    "strain). Signal: a uniaxial tension test stretched to 10% strain "
    "gives the SAME stress in fenics-LE and "
    "kratos-LinearElasticIsotropic3D (small-strain), but FEBio in "
    "dynamic mode and 4C in nonlinear mode give a slightly DIFFERENT "
    "stress because they silently switch to St-Venant-Kirchhoff. "
    "Defense: when validating a 'linear elastic' problem across "
    "backends, always check the analysis-type setting of EACH backend "
    "and force kinematics='small_strain' / 'linear' / KINEM linear in "
    "the input file even when you THINK it's the default.",
]

_LE_SEMANTICS_SIGNAL = (
    "[Cross-Backend][Physics] 'Linear elastic' is not the same material "
    "law in every backend at >5% strain. The small-strain / "
    "Green-Lagrange branch depends on the BACKEND's analysis-type "
    "metadata, not on the material name. Force the kinematics tag "
    "explicitly in every backend's input."
)


_DIRICHLET_DESC = (
    "Backends impose Dirichlet boundary conditions through different "
    "mechanisms: row-elimination (strong) vs penalty vs Nitsche vs "
    "Lagrange multiplier. Each affects matrix structure, condition "
    "number, and the spurious stress/reaction values at the boundary."
)

_DIRICHLET_PITFALLS = [
    "[Cross-Backend][BC] Dirichlet enforcement: dolfinx's "
    "fem.dirichletbc uses strong row-elimination (diagonal = 1, "
    "off-diagonal = 0, RHS = bc_value); skfem's condense() does the "
    "same. NGSolve's fes.Update() with Dirichlet='boundary' flags the "
    "DOFs and the linear-form assembler skips them. Kratos applies a "
    "PENALTY by default for AssignVectorByDirectionProcess at the "
    "modeler/constructor level; the penalty coefficient is 1e30 unless "
    "overridden. 4C uses strong elimination by default but switches to "
    "penalty/Nitsche for some contact / TSI use cases. FEBio uses "
    "penalty for prescribed displacements on contact surfaces. Signal: "
    "cross-backend reaction-force comparison on the SAME Dirichlet "
    "boundary shows agreement on interior fields to 1e-8 but "
    "reaction-force agreement only to 1e-3 — the discrepancy is the "
    "penalty residual in the backend that uses penalty. Defense: when "
    "porting a problem, force STRONG elimination where each backend "
    "supports it (penalty_coefficient=None or explicit elimination "
    "flag); accept that reaction forces computed from penalty methods "
    "carry an O(1/penalty) bias.",
]

_DIRICHLET_SIGNAL = (
    "[Cross-Backend][BC] If two backends agree on interior fields but "
    "disagree on Dirichlet-boundary reaction forces, check the BC "
    "enforcement mechanism (strong vs penalty) BEFORE re-meshing or "
    "refining."
)


_RESTART_DESC = (
    "Backend-internal checkpoint/restart files are NEVER cross-portable. "
    "Each backend writes its own binary format with backend-specific "
    "assumptions about mesh layout, DOF ordering, time-step metadata, "
    "and solver state."
)

_RESTART_PITFALLS = [
    "[Cross-Backend][Output] Restart files are PER-BACKEND. FEBio's "
    ".restart binary, fenics's dolfinx checkpoint .bp/.xdmf (h5 "
    "backend), 4C's .restart YAML+binary pair, ngsolve's .pickle dump, "
    "kratos's .post.restart all use incompatible layouts. There is no "
    "'open restart format'. Signal: trying to chain a 4C structural "
    "pre-stress computation into a fenics dynamic analysis via "
    "checkpoint+restart fails immediately with a binary-incompatible "
    "parse error, or — worse — succeeds at loading garbage and runs to "
    "NaN. Defense: always exchange FIELD DATA (displacement, velocity, "
    "stress) between backends via a NEUTRAL format: XDMF+HDF5 for "
    "time-dependent, .vtu for snapshots, .npz for raw arrays. Never "
    "exchange solver state. Re-initialise the solver from the field at "
    "the new backend's t0.",
]

_RESTART_SIGNAL = (
    "[Cross-Backend][Output] Cross-backend workflows must exchange "
    "FIELDS, not SOLVER STATE. The neutral exchange format is XDMF+HDF5 "
    "(time-dependent) or .vtu / .npz (snapshots). Reinitialise solver "
    "state on import."
)


_MPI_DESC = (
    "Each backend has its own MPI bootstrapping convention. Mixing "
    "them — or wrapping the wrong one in mpirun — produces silent "
    "serial runs masquerading as parallel."
)

_MPI_PITFALLS = [
    "[Cross-Backend][Performance] MPI launch: dolfinx / 4C use "
    "MPI_COMM_WORLD natively and require `mpirun -n N python script.py` "
    "(or the 4C binary). NGSolve uses TaskManager for shared-memory "
    "parallel (single-process threads) by default and ngs-petsc-mpi for "
    "MPI; mixing `mpirun -n N ngspy script.py` with TaskManager-using "
    "scripts produces N independent runs, each computing the FULL "
    "problem (wasted compute, results identical, no parallel speedup). "
    "Kratos uses MPI via its own KratosTrilinosApplication; ditto "
    "wrapping a non-Trilinos Kratos script in mpirun is a no-op. "
    "Signal: a 4-process mpirun gives speedup ~0.95x (slightly slower "
    "than serial) on a problem you expect to scale linearly — almost "
    "certain you wrapped a backend that doesn't natively use "
    "MPI_COMM_WORLD. Defense: verify MPI is actually used by adding a "
    "MPI.COMM_WORLD.Get_rank()/Get_size() print at the start of every "
    "backend-MPI script; if rank=0 and size=1 on all N processes, your "
    "mpirun did nothing useful.",
]

_MPI_SIGNAL = (
    "[Cross-Backend][Performance] A backend that doesn't natively use "
    "MPI_COMM_WORLD will not parallelise via `mpirun -n N`. Verify with "
    "a Get_rank()/Get_size() print before debugging scalability."
)


_ELEMENT_TYPE_DESC = (
    "Element-type names like 'hex8' / 'HEX8' / 'C3D8' / 'Element3D8N' / "
    "'ElementHex8' refer to nominally the same 8-node hexahedral "
    "element across all 8 backends — but each backend's API exposes a "
    "DIFFERENT string for the same shape, and a copy-pasted element "
    "name from one backend's example silently fails in another."
)

_ELEMENT_TYPE_PITFALLS = [
    "[Cross-Backend][API] Element-type naming: kratos uses "
    "Element3D8N (8-node hex), Element3D4N (4-node tet), "
    "Element3D6N (6-node wedge), Element3D10N (10-node tet). 4C uses "
    "SOLIDHEX8 / SOLIDTET4 / SOLIDTET10 / SOLIDPYRAMID5 / SOLIDWEDGE6, "
    "but EARLIER 4C versions used HEX8 / TET4 directly without the "
    "SOLID prefix — the prefix was added in the migration to the "
    "unified solid namespace. fenics/dolfinx uses cell type enums "
    "mesh.CellType.hexahedron / tetrahedron / prism (lowercase), "
    "passed as the cell_type kwarg to mesh.create_box. skfem uses "
    "Python classes ElementHex1 / ElementTetP1 / ElementHex2 / "
    "ElementTetP2 (with explicit polynomial order in the class name). "
    "FEBio's .feb XML uses elem type='hex8' / 'tet4' / 'tet10' / "
    "'penta6' (note: 'penta' not 'wedge' or 'prism'). dealii uses C++ "
    "enum ReferenceCells::Hexahedron / Tetrahedron / Pyramid / Wedge. "
    "Signal: copy-pasting an element name string from one backend's "
    "example into another's input file silently fails. Kratos throws "
    "KratosError 'element type 'HEX8' is not registered' (uses "
    "Element3D8N); 4C silently emits 'unrecognized element name HEX8' "
    "in older versions vs 'SOLIDHEX8 expected' in newer; FEBio's XML "
    "parser raises XMLSyntaxError on 'hexahedron'. Defense: never "
    "copy an element string between backends. Look up the target "
    "backend's exact spelling in its own catalog (knowledge tool, "
    "topic='overview') before writing the input.",

    "[Cross-Backend][API] Wedge / prism / pyramid element naming "
    "is the most-confused: 4C calls it SOLIDWEDGE6, kratos calls it "
    "Element3D6N (the dimensionality-and-node-count name; no shape "
    "indicator at all), fenics calls it mesh.CellType.prism, dealii "
    "uses ReferenceCells::Wedge, FEBio uses 'penta6'. Five different "
    "words for the same 6-node element. Signal: a Gmsh-generated mesh "
    "containing wedge cells imports correctly to fenics (prism) and "
    "dealii (Wedge) but fails to load in kratos unless the .mdpa "
    "element block names them Element3D6N, and fails in FEBio unless "
    "the .feb element type='penta6'. Defense: pre-process Gmsh "
    "output through meshio's per-backend writers (meshio.write with "
    "the target format), not through a single-format intermediate.",
]

_ELEMENT_TYPE_SIGNAL = (
    "[Cross-Backend][API] Element-type strings (hex8 / HEX8 / "
    "SOLIDHEX8 / Element3D8N / ElementHex1 / mesh.CellType.hexahedron) "
    "are backend-specific spellings of the same shape. Never copy "
    "between backends. Look up the target backend's exact spelling "
    "in its own catalog before writing input."
)


_TIME_INTEGRATION_DESC = (
    "Implicit-dynamics time integration scheme defaults differ across "
    "backends. The SAME problem (mass-spring oscillator, structural "
    "dynamics, transient heat) ported across backends gives different "
    "trajectories because the default beta / gamma / alpha parameters "
    "are not standardised."
)

_TIME_INTEGRATION_PITFALLS = [
    "[Cross-Backend][Numerical] Newmark-beta default parameters: "
    "the 'average acceleration' / 'undamped trapezoidal' scheme is "
    "beta=1/4, gamma=1/2 (unconditionally stable, no algorithmic "
    "damping). 4C's default for structural_dynamics is beta=0.25, "
    "gamma=0.5 (matches average-acceleration). FEBio's default is "
    "ALSO beta=0.25, gamma=0.5 (matches). NGSolve's "
    "TimeIntegrationNewmark and skfem helper default to "
    "beta=0.25, gamma=0.5. Kratos's StructuralMechanicsApplication "
    "ResidualBasedNewmarkDisplacementScheme uses beta=0.25, "
    "gamma=0.5 as DEFAULT but exposes alpha_m / alpha_f for "
    "generalised-alpha; if the user passes alpha_m != 0 it switches "
    "implicitly to the Hilber-Hughes-Taylor (HHT) family and beta / "
    "gamma get RECOMPUTED from alpha, silently. dealii's "
    "step-23/step-25 wave examples use a fixed Newmark variant with "
    "theta=0.5 (which is NOT the same as gamma=0.5 — theta there is "
    "the time-discretisation parameter for the velocity, not the "
    "Newmark gamma). Signal: a 4C linear oscillator with damping "
    "ported to a Kratos generalised-alpha scheme produces a "
    "displacement amplitude that decays faster than expected even "
    "though both backends report 'Newmark beta=0.25 gamma=0.5' — the "
    "Kratos implementation silently added alpha_m=0.05 algorithmic "
    "damping. Defense: when porting a transient structural problem, "
    "explicitly set beta, gamma, alpha_m, alpha_f in EVERY backend's "
    "input (do not rely on defaults); verify with a no-damping "
    "single-DOF reference: period error after 100 cycles should be "
    "< 1% if and only if the schemes match.",
]

_TIME_INTEGRATION_SIGNAL = (
    "[Cross-Backend][Numerical] If a transient solid/structural "
    "problem matches across backends at t=0 but the displacement "
    "envelope decays at different rates, the cause is almost always "
    "an algorithmic-damping parameter (alpha_m / alpha_f) silently "
    "added by one backend's default but not the other's. Always "
    "explicitly set ALL Newmark/generalised-alpha parameters in the "
    "input, do not trust 'default'."
)


CROSS_BACKEND_PITFALLS = {
    "units": {
        "description": _UNITS_DESC,
        "pitfalls": _UNITS_PITFALLS,
        "Signal": _UNITS_SIGNAL,
    },
    "element_node_ordering": {
        "description": _NODE_ORDER_DESC,
        "pitfalls": _NODE_ORDER_PITFALLS,
        "Signal": _NODE_ORDER_SIGNAL,
    },
    "linear_elastic_semantics": {
        "description": _LE_SEMANTICS_DESC,
        "pitfalls": _LE_SEMANTICS_PITFALLS,
        "Signal": _LE_SEMANTICS_SIGNAL,
    },
    "dirichlet_bc_enforcement": {
        "description": _DIRICHLET_DESC,
        "pitfalls": _DIRICHLET_PITFALLS,
        "Signal": _DIRICHLET_SIGNAL,
    },
    "restart_checkpoint_compatibility": {
        "description": _RESTART_DESC,
        "pitfalls": _RESTART_PITFALLS,
        "Signal": _RESTART_SIGNAL,
    },
    "mpi_launch_idioms": {
        "description": _MPI_DESC,
        "pitfalls": _MPI_PITFALLS,
        "Signal": _MPI_SIGNAL,
    },
    "element_type_naming": {
        "description": _ELEMENT_TYPE_DESC,
        "pitfalls": _ELEMENT_TYPE_PITFALLS,
        "Signal": _ELEMENT_TYPE_SIGNAL,
    },
    "time_integration_defaults": {
        "description": _TIME_INTEGRATION_DESC,
        "pitfalls": _TIME_INTEGRATION_PITFALLS,
        "Signal": _TIME_INTEGRATION_SIGNAL,
    },
}


def get_cross_backend_pitfalls(topic: str | None = None) -> dict:
    """Return the cross-backend pitfalls structure.

    If `topic` is provided (e.g. 'units', 'mesh', 'bc'), filter to
    matching entries. Topic matching is a case-insensitive substring
    against the entry key or description.
    """
    if not topic:
        return CROSS_BACKEND_PITFALLS
    t = topic.lower()
    matched = {}
    for key, entry in CROSS_BACKEND_PITFALLS.items():
        if t in key.lower() or t in entry.get("description", "").lower():
            matched[key] = entry
    return matched if matched else CROSS_BACKEND_PITFALLS
