"""scikit-fem eigenvalue problem generators and knowledge."""


def _eigenvalue_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Eigenvalue problem: Laplace on unit square."""
    nx = params.get("nx", 32)
    n_eigs = params.get("n_eigenvalues", 5)
    return f'''\
"""Eigenvalue problem: Laplace eigenvalues — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, mass
import numpy as np
from scipy.sparse.linalg import eigsh
import json, math

m = MeshQuad.init_tensor(np.linspace(0, 1, {nx+1}), np.linspace(0, 1, {nx+1}))
e = ElementQuad1()
ib = Basis(m, e)

K = laplace.assemble(ib)
M = mass.assemble(ib)

D = ib.get_dofs().flatten()
I = ib.complement_dofs(D)

# Solve generalized eigenvalue: K*x = lambda*M*x (restrict to interior DOFs)
eigenvalues, eigenvectors = eigsh(K[I][:, I], k={n_eigs}, M=M[I][:, I], sigma=0, which='LM')

# Exact eigenvalues: pi^2*(m^2+n^2)
exact = sorted([math.pi**2*(i**2+j**2) for i in range(1,6) for j in range(1,6)])[:{n_eigs}]

print(f"Computed eigenvalues: {{eigenvalues}}")
print(f"Exact eigenvalues:    {{exact}}")
for i, (c, e_val) in enumerate(zip(eigenvalues, exact)):
    err = abs(c - e_val) / e_val
    print(f"  lambda_{{i+1}} = {{c:.6f}} (exact: {{e_val:.6f}}, err: {{err:.2e}})")

summary = {{"eigenvalues": eigenvalues.tolist(), "exact": exact, "n_dofs": K.shape[0]}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Eigenvalue solve complete.")
'''


KNOWLEDGE = {
    "eigenvalue": {
        "description": "Eigenvalue problems — Laplace, elasticity vibration (examples 02, 03, 16, 21)",
        "solver": "scipy.sparse.linalg.eigsh (Lanczos for symmetric generalized eigenvalue)",
        "pitfalls": [
            "[API] For Dirichlet eigenvalue problems, restrict "
            "matrices to INTERIOR DOFs: I = basis.complement_dofs("
            "D) where D = basis.get_dofs() (boundary). Then "
            "K_I = K[I][:, I]; M_I = M[I][:, I]; eigsh(K_I, M=M_I, "
            "...). Skipping this leaves the boundary DOFs in "
            "the matrices and the Dirichlet eigenvalues come "
            "back wrong. Signal: len(I) + len(D.nodal['u']) == "
            "basis.N (verified empirically — N=25 on MeshTri."
            "refined(2) splits as 16 boundary + 9 interior). "
            "(Verified empirically 2026-06-01.)",
            "[Numerical] scipy.sparse.linalg.eigsh with sigma=0 "
            "uses shift-and-invert targeting the SMALLEST "
            "eigenvalues. For a Dirichlet-restricted Laplacian "
            "(SPD), sigma=0 is safe (no null space). Signal: "
            "eigsh(K_I, M=M_I, k=5, sigma=0, which='LM') returns "
            "the 5 smallest eigenvalues; switching to "
            "which='SM' (no sigma) is much slower per iteration. "
            "(Verified empirically — see laplace_eigenvalue_basics "
            "fixture.)",
            "[Physics] Analytic eigenvalues of the Dirichlet "
            "Laplacian on [0,1]^2 are pi^2*(m^2+n^2) for m,n>=1. "
            "First few: 19.74, 49.35, 49.35 (degenerate), 78.96, "
            "98.70. A MeshTri refined(4) P1 mesh recovers them "
            "with ~1-4% relative error (expected for P1; "
            "refinement decreases the error). Signal: computed "
            "eigenvalues from eigsh match the analytic sequence "
            "within ~5% relative on MeshTri.refined(4) P1. "
            "(Verified empirically 2026-06-01.)",
            "[Numerical] Structural vibration eigenproblem "
            "K*x = omega^2*M*x — eigenvalues are squared "
            "angular frequencies. Take omega = sqrt(eig) to get "
            "physical natural frequencies in rad/s. Signal: "
            "passing both K and M to eigsh as eigsh(K, M=M, ...) "
            "solves the generalised problem; passing only K "
            "solves the standard problem (against identity), "
            "giving wrong frequency values. (Claim inherited — "
            "not yet empirically separated.)",
        ],
    },
}

GENERATORS = {
    "eigenvalue_2d": _eigenvalue_2d,
}
