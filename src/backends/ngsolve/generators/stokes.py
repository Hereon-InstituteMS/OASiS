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
            "matrix has eigenvalues of BOTH signs, so CG carries "
            "no convergence guarantee and can break down; MinRes / "
            "GMRES are the safe choice. Do NOT expect a loud "
            "failure from CG, though. Signal: on unit_square, "
            "Taylor-Hood VectorH1(order=2)*H1(order=1), the "
            "restricted saddle matrix at maxh=0.2 (n=216) has 37 "
            "negative and 179 positive eigenvalues (min -2.27e-02, "
            "max 5.31e+00) — genuinely indefinite. Yet "
            "scipy cg CONVERGED in every configuration tried: "
            "unpreconditioned 153 / 657 / 1317 iterations at "
            "maxh = 0.3 / 0.15 / 0.08 (rel. residual < 1e-8), and "
            "31 / 31 / 35 iterations with a block-diagonal "
            "(Laplacian, pressure-mass) preconditioner. MinRes "
            "took 130 / 318 / 604 and 28 / 25 / 30. The prior "
            "catalog claim that 'CG diverges immediately' / "
            "'stalls at O(1) within a few iterations' did NOT "
            "reproduce — the observable difference is cost and "
            "robustness, not divergence. (Verified empirically "
            "2026-08-03 on NGSolve 6.2.2604 + scipy 1.15.3 — "
            "catalog-drift correction.)",
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
            "free) outlet determine pressure uniquely. The "
            "singular system does NOT announce itself: on "
            "unit_square maxh=0.15 with velocity Dirichlet on all "
            "four edges and an unpinned H1 pressure, "
            "a.mat.Inverse(..., inverse='umfpack') returned "
            "without raising and yielded mean(p) = 1.06e+00, "
            "|p|_inf = 1.56e+00, versus mean(p) = 5.00e-01, "
            "|p|_inf = 1.00e+00 once one edge of the pressure "
            "space is pinned — i.e. an arbitrary additive "
            "constant, silently. Signal: solve the same enclosed "
            "problem twice with different meshes or solvers and "
            "compare mean(p); if it moves, the constant mode is "
            "unpinned. (Verified empirically 2026-08-03 on "
            "NGSolve 6.2.2604.)",
            "[Numerical] Block preconditioners for Stokes use "
            "BlockMatrix + (M_v^{-1}, Schur^{-1}_approx) on the "
            "diagonal. The Schur complement approximation can be "
            "a pressure mass matrix scaled by 1/nu. Signal: "
            "measured on unit_square Taylor-Hood at maxh = "
            "0.3 / 0.15 / 0.08 (n = 93 / 402 / 1524), MinRes with "
            "the block-diagonal (Laplacian, pressure-mass) "
            "preconditioner took 28 / 25 / 30 iterations — flat "
            "in h — while unpreconditioned MinRes took "
            "130 / 318 / 604, i.e. roughly doubling as h halves, "
            "consistent with O(h^{-1}). (Verified empirically "
            "2026-08-03 on NGSolve 6.2.2604 + scipy 1.15.3.)",
            "[API] Do NOT hardcode a direct solver name — "
            "availability is build-dependent — but the naive "
            "fallback loop 'except RuntimeError: continue' is "
            "BROKEN, because the different unavailable solvers "
            "raise DIFFERENT exception types and NgException is "
            "NOT a subclass of RuntimeError. Measured on this "
            "install (NGSolve 6.2.2604): inverse='sparsecholesky' "
            "OK, 'umfpack' OK, 'superlu' OK, 'pardiso' raises "
            "RuntimeError('MKL Pardiso is not available. Ensure "
            "that MKL is installed'), 'mumps' raises "
            "NgException('SparseMatrix::InverseMatrix: "
            "MumpsInverse not available') whose MRO is "
            "(NgException, Exception, BaseException, object) — so "
            "'except RuntimeError' lets the mumps attempt escape "
            "and kill the script. Signal: "
            "isinstance(e, RuntimeError) is True for the pardiso "
            "failure and False for the mumps failure. Catch bare "
            "Exception in the fallback loop. (Verified "
            "empirically 2026-08-03 — catalog-drift correction.)",
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
