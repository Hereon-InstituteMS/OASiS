"""Matrix-free Poisson generator for FEniCSx/dolfinx.

Encodes the matrix-free conjugate-gradient pattern from upstream
dolfinx ``demo_poisson_matrix_free.py``. The matrix-free formulation
never assembles the global stiffness operator A explicitly — instead
it provides a callable ``action_A(x, y)`` that computes y = A·x by
re-assembling the linear form ``M = ufl.action(a, ui)`` with ui = x.

Why it matters: for very large 3D / high-order problems, the
explicit sparse matrix becomes the memory bottleneck; matrix-free
keeps memory ~O(n_dofs) per process. Pairs naturally with high-
performance solvers (custom CG, Krylov + matrix-free preconditioner).

Source: upstream dolfinx
  conda envs/ofa-fenicsx/etc/conda/test-files/fenics-dolfinx/0/
  python/demo/demo_poisson_matrix_free.py
"""

VARIANTS = ["2d"]

KNOWLEDGE = {
    "description": (
        "Matrix-free conjugate-gradient Poisson solver. Builds the "
        "stiffness operator A as a callable `action_A(x, y)` that "
        "re-assembles `M = ufl.action(a, ui)` on the fly, avoiding "
        "explicit assembly of the global sparse matrix. Mirrors "
        "dolfinx upstream demo_poisson_matrix_free.py — the "
        "canonical matrix-free pattern in FEniCSx 0.10."
    ),
    "weak_form": (
        "a(u, v) = (grad(u), grad(v))_dx;  "
        "L(v) = (f, v)_dx;  "
        "matrix-free: M(v) = a(ui, v) reassembled per CG iteration "
        "with ui = current trial vector."
    ),
    "function_space": "Lagrange order 2 (degree-1 also works)",
    "solver": {
        "type": "custom matrix-free CG, no preconditioner",
        "rtol": "1e-6 default",
        "max_iter": "200 default",
    },
    "elements": ["Lagrange P1 / P2 / P3"],
    "variants": ["2d"],
    "materials": {},
    "pitfalls": [
        "[API] The matrix-free action is built with "
        "`ufl.action(a, ui)` — NOT `a.action(ui)`. The latter is "
        "not a UFL form attribute. After construction, compile via "
        "`fem.form(M, dtype=dtype)` so `fem.assemble_vector` can "
        "consume it. "
        "Signal: AttributeError 'Form' object has no attribute "
        "'action' at the line constructing M; or "
        "RuntimeError 'cannot assemble: form has not been "
        "compiled' from fem.assemble_vector if fem.form(M) is "
        "skipped.",

        "[API] In dolfinx 0.10 the assembly target is the "
        "underlying numpy array, NOT the la.Vector itself: "
        "`fem.assemble_vector(b.array, L_fem)` (note `b.array`, "
        "not `b`). Old code shape `b = fem.assemble_vector(L_fem)` "
        "returns a fresh la.Vector but the in-place form requires "
        "passing the array. "
        "Signal: TypeError 'assemble_vector() takes positional "
        "argument' or 'expected ndarray, got Vector' from "
        "dolfinx.fem.assemble_vector when the first arg is a "
        "Vector instead of a Vector.array.",

        "[Parallel] After in-place assembly into a distributed "
        "array, you MUST call `b.scatter_reverse(la.InsertMode.add)` "
        "to gather ghost contributions onto the owning rank. "
        "Skipping this gives a vector that looks right on rank 0 "
        "but is incomplete at interface DOFs across MPI ranks. "
        "Signal: serial run (mpirun -n 1) converges to the "
        "correct L2 error ~1e-6, but mpirun -n 2 converges to "
        "a residual that plateaus several orders of magnitude "
        "above the serial result. The discrepancy grows with the "
        "number of ranks because `scatter_reverse` was omitted "
        "and ghost-DOF contributions from `la.InsertMode.add` "
        "are silently dropped.",

        "[API] Dirichlet lifting in the matrix-free path needs "
        "`bc.set(ui.x.array, alpha=-1.0)` to set ui to -1 * "
        "Dirichlet value at constrained DOFs, then "
        "`fem.assemble_vector(b.array, M_fem)` adds -A * x_bc to "
        "b. After the action you must zero out BC DOFs in the "
        "RHS via `bc.set(b.array, alpha=0.0)` so the iterative "
        "solver doesn't try to update them. "
        "Signal: the CG iteration count balloons (>200) without "
        "convergence; the final L2 error against the exact "
        "solution (1 + x^2 + 2*y^2 on the unit square with f=-6) "
        "is ~1e-1 instead of ~1e-6. Or: the boundary values in "
        "the result are not equal to uD after `bc.set(u.x.array, "
        "alpha=1.0)`.",

        "[Numerical] Custom CG without a preconditioner is the "
        "demonstration baseline — for ill-conditioned problems "
        "(large mesh, high polynomial degree, anisotropic "
        "coefficients) iteration counts grow as O(h^-1) or worse. "
        "Production matrix-free runs use a Jacobi or AMG "
        "preconditioner via `petsc4py.PETSc.KSP` with "
        "`pc_type='hypre'` or `pc_type='gamg'`. "
        "Signal: CG reports 200 iterations without converging "
        "(raises RuntimeError 'Solver exceeded max iterations'); "
        "the rnorm/rnorm0 trace shows a slow logarithmic decay "
        "rather than the geometric drop seen with a preconditioner.",

        "[Parallel] The inner product in custom CG must use "
        "ONLY owned DOFs, not the ghost-padded array. Use "
        "`v0[:nr]` where `nr = b.index_map.size_local`, then "
        "`comm.allreduce(np.vdot(v0[:nr], v1[:nr]), MPI.SUM)`. "
        "Iterating over the whole array double-counts ghost "
        "DOFs (each owned in one rank but also resident in "
        "neighbours) so the dot product is too large. "
        "Signal: CG diverges (rnorm grows) after the first few "
        "iterations on multi-rank runs but converges in serial; "
        "alpha and beta coefficients computed from the corrupted "
        "dot products become wrong.",

        "[Output] L2 error norm in parallel: "
        "`comm.allreduce(fem.assemble_scalar(fem.form(...)), "
        "op=MPI.SUM)` then sqrt. Each rank contributes its local "
        "integral; the allreduce sums across ranks before the "
        "sqrt. Forgetting the allreduce gives only the rank-0 "
        "local L2 contribution. "
        "Signal: serial L2 error ~1e-6, parallel L2 error "
        "underestimated by a factor of sqrt(nranks). Caused by "
        "summing only the local `fem.assemble_scalar` result "
        "without `comm.allreduce(..., MPI.SUM)`; the bug is "
        "silent because both values look 'small'.",
    ],
    "references": [
        "dolfinx demo: demo_poisson_matrix_free.py (verified by "
        "upstream-demo audit 2026-06-02)",
        "Saad, Y. (2003) — 'Iterative Methods for Sparse Linear "
        "Systems', Ch. 6: Krylov Subspace Methods (CG)",
    ],
}


def generate(variant: str, params: dict) -> str:
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown matrix_free_poisson variant: {variant!r}. "
            f"Available: {VARIANTS}")
    return _matrix_free_poisson_2d(params)


def _matrix_free_poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate
    values for your specific problem.

    Matrix-free CG Poisson on a unit-square mesh. Default exact
    solution: u_D = 1 + x^2 + 2*y^2 with f = -6 (so -Δu = f and
    u|_∂Ω = u_D). The L2 error is reported."""
    nx = params.get("nx", 12)
    ny = params.get("ny", 12)
    degree = params.get("degree", 2)
    rtol = params.get("rtol", 1e-6)
    max_iter = params.get("max_iter", 200)
    return f'''\
"""Matrix-free conjugate-gradient Poisson — FEniCSx/dolfinx
Mirrors upstream demo_poisson_matrix_free.py.
"""
from mpi4py import MPI
import numpy as np
import dolfinx
import ufl
from dolfinx import fem, la
import json

dtype = dolfinx.default_scalar_type
real_type = np.real(dtype(0.0)).dtype
comm = MPI.COMM_WORLD
mesh = dolfinx.mesh.create_rectangle(
    comm, [[0.0, 0.0], [1.0, 1.0]], ({nx}, {ny}),
    dtype=real_type,
)
degree = {degree}
V = fem.functionspace(mesh, ("Lagrange", degree))

tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim - 1, tdim)
facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
dofs = fem.locate_dofs_topological(V=V, entity_dim=tdim - 1, entities=facets)

# Exact solution u_D = 1 + x^2 + 2*y^2 ⇒ -Δu = -6, so f = -6.
uD = fem.Function(V, dtype=dtype)
uD.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)
bc = fem.dirichletbc(value=uD, dofs=dofs)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(mesh, dtype(-6.0))
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx
L_fem = fem.form(L, dtype=dtype)

# Matrix-free action: M(v) = a(ui, v) with ui playing the role of
# the trial vector x. fem.assemble_vector(y, M_fem) gives y = A * ui.
ui = fem.Function(V, dtype=dtype)
M = ufl.action(a, ui)
M_fem = fem.form(M, dtype=dtype)

# RHS: b - A * x_bc via lifting.
b = fem.assemble_vector(L_fem)
ui.x.array[:] = 0.0
bc.set(ui.x.array, alpha=-1.0)
fem.assemble_vector(b.array, M_fem)
b.scatter_reverse(la.InsertMode.add)
bc.set(b.array, alpha=0.0)
b.scatter_forward()


def action_A(x, y):
    ui.x.array[:] = x.array
    ui.x.scatter_forward()
    y.array[:] = 0.0
    fem.assemble_vector(y.array, M_fem)
    y.scatter_reverse(la.InsertMode.add)
    bc.set(y.array, alpha=0.0)


def cg(comm, action_A, x, b, max_iter={max_iter}, rtol={rtol}):
    rtol2 = rtol ** 2
    nr = b.index_map.size_local

    def _gdot(v0, v1):
        return comm.allreduce(np.vdot(v0[:nr], v1[:nr]), MPI.SUM)

    y = la.vector(b.index_map, 1, dtype)
    action_A(x, y)
    r = b.array - y.array
    p = la.vector(b.index_map, 1, dtype)
    p.array[:] = r

    rnorm0 = _gdot(r, r)
    rnorm = rnorm0
    for k in range(max_iter):
        action_A(p, y)
        alpha = rnorm / _gdot(p.array, y.array)
        x.array[:] += alpha * p.array
        r -= alpha * y.array
        rnorm_new = _gdot(r, r)
        beta = rnorm_new / rnorm
        rnorm = rnorm_new
        if rnorm / rnorm0 < rtol2:
            x.scatter_forward()
            return k
        p.array[:] = beta * p.array + r
    raise RuntimeError(
        f"matrix-free CG exceeded max iterations ({{max_iter}})")


# Solve.
u = fem.Function(V, dtype=dtype)
iters = cg(mesh.comm, action_A, u.x, b)
bc.set(u.x.array, alpha=1.0)

# L2 error vs exact solution.
def L2Norm(w):
    val = fem.assemble_scalar(
        fem.form(ufl.inner(w, w) * ufl.dx, dtype=dtype))
    return float(np.sqrt(comm.allreduce(val, op=MPI.SUM)))

err = L2Norm(u - uD)
print(f"Matrix-free CG: iters={{iters}} rtol={rtol} "
      f"L2_error={{err:.4e}}")

# Output: .vtu via dolfinx VTXWriter (Lagrange OK).
with dolfinx.io.VTXWriter(mesh.comm, "result.bp", [u], "BP4") as vtx:
    vtx.write(0.0)

summary = {{
    "iters": int(iters),
    "L2_error": err,
    "n_dofs": int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs),
    "degree": int(degree),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''
