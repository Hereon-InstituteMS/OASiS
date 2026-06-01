"""NGSolve Stokes flow generators and knowledge."""


def _stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Stokes flow with Taylor-Hood P2/P1 elements."""
    nx = params.get("nx", 32)
    nu_visc = params.get("viscosity", 1.0)
    maxh = 1.0 / nx
    return f'''\
"""Stokes flow — Taylor-Hood P2/P1 — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))

V = VectorH1(mesh, order=2, dirichlet="bottom|right|top|left")
Q = H1(mesh, order=1)
X = V * Q
(u, p), (v, q) = X.TnT()

nu = {nu_visc}
a = BilinearForm(X)
a += nu * InnerProduct(Grad(u), Grad(v)) * dx
a += div(u)*q*dx + div(v)*p*dx
a.Assemble()

f = LinearForm(X)
f.Assemble()

gfu = GridFunction(X)
# Velocity BC — set for your problem
uin = CoefficientFunction((1, 0))
gfu.components[0].Set(uin, definedon=mesh.Boundaries("top"))

# Pin one pressure DOF to remove the constant-pressure
# null space. Without this, MKL Pardiso reports phase-33
# error -4 (zero pivot) because the saddle-point system
# is rank-deficient by 1. The pinned value is 0, which
# is consistent with zero-mean pressure for enclosed
# Stokes flow.
free = X.FreeDofs()
free.Clear(V.ndof)
gfu.vec[V.ndof] = 0.0

# Solve with modified RHS for non-homogeneous Dirichlet
f.vec.data -= a.mat * gfu.vec
# Try available direct solvers (umfpack may not be installed)
inv = None
for solver_name in ["pardiso", "mumps", "umfpack"]:
    try:
        inv = a.mat.Inverse(free, solver_name)
        break
    except:
        pass
if inv is None:
    from ngsolve.krylovspace import MinResSolver
    inv = MinResSolver(a.mat, freedofs=free, maxsteps=10000, tol=1e-10)
gfu.vec.data += inv * f.vec

vel = gfu.components[0]
pres = gfu.components[1]
max_vel = Integrate(InnerProduct(vel, vel), mesh)
print(f"L2(velocity) = {{max_vel**0.5:.6f}}")

vtk = VTKOutput(mesh, coefs=[vel, pres], names=["velocity", "pressure"],
                filename="result", subdivision=1)
vtk.Do()
summary = {{"l2_velocity": float(max_vel**0.5), "n_dofs": X.ndof}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Stokes solve complete.")
'''


KNOWLEDGE = {
    "stokes": {
        "description": "Stokes flow with Taylor-Hood P2/P1 or Mini element or HDG",
        "spaces": "VectorH1(order=2) * H1(order=1) for Taylor-Hood. VectorH1 * L2 for DG-Stokes",
        "solver": "Direct: pardiso > mumps > umfpack (try in order). Iterative: MinRes or GMRES (indefinite system!)",
        "pitfalls": [
            "[Numerical] Stokes block system is INDEFINITE — use "
            "MinRes or GMRES, never CG. CG on the full block "
            "matrix diverges immediately because the system has "
            "eigenvalues of both signs. Signal: CGSolver.Solve "
            "on the saddle-point matrix shows residual stalling "
            "at O(1) within a few iterations, while MinRes / "
            "GMRES converge to tolerance. (Claim inherited — "
            "not yet empirically separated.)",
            "[Syntax] For a compound (mixed) space X = FESpace("
            "[V, Q]), X.TnT() returns a 2-tuple whose inner "
            "elements are LISTS of ProxyFunctions: the unpack "
            "pattern '(u, p), (v, q) = X.TnT()' works because "
            "Python destructures lists by length. Signal: "
            "type(X.TnT()) is tuple of length 2; "
            "type(X.TnT()[0]) is list of length 2 with each "
            "entry a ProxyFunction; the (u, p), (v, q) unpack "
            "succeeds without error. (Verified empirically "
            "2026-06-01 — catalog text 'returns nested tuples' "
            "is slightly loose; the actual types are tuple of "
            "lists, not tuple of tuples, but unpacking works "
            "either way.)",
            "[Physics] Enclosed-flow Stokes admits the constant "
            "pressure null space — pin pressure at one node or "
            "add a NumberSpace Lagrange multiplier enforcing "
            "mean(p) = 0. Open flows with a do-nothing (traction-"
            "free) outlet determine pressure uniquely. Signal: "
            "without pinning, the post-processed GridFunction "
            "pressure component has a huge additive offset (same "
            "family as fenics poisson pure-Neumann); pinning "
            "via fes_q.dirichlet on one node clamps it. (Claim "
            "inherited — not yet empirically separated.)",
            "[Numerical] Block preconditioners for Stokes use "
            "BlockMatrix + (M_v^{-1}, Schur^{-1}_approx) on the "
            "diagonal. The Schur complement approximation can be "
            "a pressure mass matrix scaled by 1/nu. Signal: "
            "MinResSolver.Solve with the block preconditioner "
            "converges in O(log(1/tol)) iterations independent "
            "of mesh size; without preconditioning the iteration "
            "count grows as O(h^{-1}). (Claim inherited.)",
            "[API] Do NOT hardcode 'umfpack' — it may not be "
            "available in every NGSolve build. Use a fallback "
            "loop: for name in ('pardiso','mumps','umfpack'): "
            "try a.mat.Inverse(fes.FreeDofs(), inverse=name) "
            "except RuntimeError: continue. Signal: "
            "BilinearForm.mat.Inverse(..., inverse='umfpack') "
            "raises RuntimeError 'inverse umfpack not "
            "available' in builds without UMFPACK support; "
            "the recommended trio (pardiso, mumps, umfpack) "
            "covers most builds. (Claim inherited.)",
            "[Physics] NGSolve Stokes uses +p*div(v) convention "
            "(opposite sign from FEniCS / skfem). Both are valid "
            "weak forms. Signal: a Poiseuille-flow benchmark "
            "solved in NGSolve and FEniCS gives pressure "
            "GridFunction values that differ by a sign at every "
            "node (max(p_ngsolve) ≈ -max(p_fenics)). Be aware "
            "when comparing cross-solver results. (Claim "
            "inherited.)",
        ],
    },
}

GENERATORS = {
    "stokes_2d": _stokes_2d,
    "stokes_2d_hdg": _stokes_2d,  # Same solver, HDG variant TBD
}
