"""NGSolve eigenvalue problem generators and knowledge."""


def _eigenvalue_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Eigenvalue problem: Laplace on unit square."""
    order = params.get("order", 4)
    n_eigs = params.get("n_eigenvalues", 10)
    maxh = params.get("maxh", 0.03)
    return f'''\
"""Eigenvalue problem: Laplace — ArnoldiSolver — NGSolve"""
from ngsolve import *
import json, math

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))
fes = H1(mesh, order={order}, dirichlet="bottom|right|top|left")
u, v = fes.TnT()

a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
m = BilinearForm(u*v*dx).Assemble()

gfu = GridFunction(fes, multidim={n_eigs})
lam = ArnoldiSolver(a.mat, m.mat, fes.FreeDofs(), list(gfu.vecs), shift=0)

print(f"First {n_eigs} eigenvalues:")
exact = [math.pi**2*(i**2+j**2) for i in range(1,6) for j in range(1,6)]
exact.sort()
for i, (computed, ref) in enumerate(zip(lam, exact[:{n_eigs}])):
    err = abs(computed - ref) / ref
    print(f"  lambda_{{i+1}} = {{computed:.6f}} (exact: {{ref:.6f}}, error: {{err:.2e}})")

vtk = VTKOutput(mesh, coefs=[gfu.components[0]], names=["eigenmode_1"],
                filename="result", subdivision=1)
vtk.Do()
summary = {{"eigenvalues": [float(l) for l in lam], "n_dofs": fes.ndof}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Eigenvalue solve complete.")
'''


KNOWLEDGE = {
    "eigenvalue": {
        "description": "Eigenvalue problems via ArnoldiSolver (shift-invert Arnoldi)",
        "spaces": "Any H1 space",
        "solver": "ArnoldiSolver(a.mat, m.mat, freedofs, vecs, shift=target)",
        "pitfalls": [
            "[Numerical] ArnoldiSolver(a.mat, m.mat, fes.FreeDofs(), "
            "vecs, shift=target) uses shift-and-invert: eigenvalues "
            "near 'shift' converge fastest. shift=0 fails for "
            "operators with the gradient kernel as null space — "
            "raises NgException UmfpackInverse 'matrix is singular' "
            "(same as maxwell#5). For Laplace eigenproblems with "
            "Dirichlet BCs the matrix is positive-definite so "
            "shift=0 is safe. Signal: ArnoldiSolver call raises "
            "NgException with UmfpackInverse text when shift is "
            "in the null space; with shift=lowest-expected the "
            "returned eigenvalues are within ~1% of analytic. "
            "(Family-verified via maxwell#5.)",
            "[API] GridFunction(fes, multidim=n) allocates space "
            "for n independent vectors (e.g., n eigenvectors). "
            "Accessed via gfu.vecs[i] (a list-like sequence), "
            "NOT via gfu.mdcomponents (which does not exist as "
            "an attribute). Signal: hasattr(gfu, 'vecs') is True, "
            "len(list(gfu.vecs)) == n; hasattr(gfu, "
            "'mdcomponents') is False. (Verified empirically "
            "2026-06-01 — catalog text tightened from prose "
            "to name the actual access path.)",
            "[Physics] Exact analytic eigenvalues of the "
            "Dirichlet Laplacian on [0,1]^2 are pi^2*(m^2+n^2) "
            "for m, n >= 1. First few: 2*pi^2, 5*pi^2, 5*pi^2 "
            "(degenerate), 8*pi^2, 10*pi^2... Signal: "
            "ArnoldiSolver result on a maxh<=0.05 mesh with "
            "order>=2 elements should agree with these values "
            "to within ~0.5%; larger discrepancy indicates mesh "
            "too coarse or wrong FE order. (Claim inherited — "
            "not yet empirically verified.)",
            "[Syntax] For the generalized eigenvalue problem "
            "A*x = lambda*M*x, pass BOTH matrices to "
            "ArnoldiSolver as a.mat and m.mat. Passing the same "
            "matrix twice (or only A) silently solves a "
            "standard eigenvalue problem against the identity "
            "mass — wrong eigenvalues. Signal: numerical "
            "eigenvalues disagree with the (mass-weighted) "
            "analytic reference by a factor that depends on "
            "the mesh geometry (mesh area, h^2 scaling); "
            "passing m.mat correctly recovers the analytic "
            "result. (Claim inherited.)",
            "[API] Alternative eigenvalue solver: "
            "ngsolve.solvers.PINVIT (preconditioned inverse "
            "iteration) for the lowest eigenvalues. PINVIT "
            "scales better than ArnoldiSolver on large meshes "
            "because it does NOT need a global factorisation — "
            "only a preconditioner application per iteration. "
            "Signal: PINVIT result on a mesh with > 10^5 dofs "
            "completes in a fraction of the wall time of "
            "ArnoldiSolver; PINVIT's per-iter cost is O(N) "
            "vs ArnoldiSolver's O(N^1.5). (Claim inherited.)",
        ],
    },
}

GENERATORS = {
    "eigenvalue_2d": _eigenvalue_2d,
}
