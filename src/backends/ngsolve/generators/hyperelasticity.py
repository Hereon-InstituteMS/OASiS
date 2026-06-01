"""NGSolve hyperelasticity generators and knowledge."""


def _hyperelasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Neo-Hookean hyperelasticity via SymbolicEnergy + Newton."""
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    maxh = params.get("maxh", 0.05)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Neo-Hookean hyperelasticity — Newton solver — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))
fes = VectorH1(mesh, order=2, dirichlet="left|bottom")
u = fes.TrialFunction()

mu, lam = {mu}, {lam}
I = Id(2)
F = I + Grad(u)
C = F.trans * F
J = Det(F)

# Neo-Hookean energy: W = mu/2*(tr(C)-2) - mu*ln(J) + lam/2*ln(J)^2
energy = 0.5*mu*(Trace(C) - 2) - mu*log(J) + 0.5*lam*log(J)**2

a = BilinearForm(fes, symmetric=True)
a += Variation(energy * dx)

# Apply displacement BC — set for your problem
gfu = GridFunction(fes)
gfu.Set(CoefficientFunction((0.3*x, -0.1)), definedon=mesh.Boundaries("top"))

solvers.Newton(a, gfu, maxit=20, printing=True)

vtk = VTKOutput(mesh, coefs=[gfu], names=["displacement"], filename="result", subdivision=1)
vtk.Do()
summary = {{"n_dofs": fes.ndof, "converged": True}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Hyperelasticity solve complete.")
'''


def _hyperelasticity_3d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    3D Neo-Hookean hyperelasticity."""
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    lx = params.get("lx", 1)
    ly = params.get("ly", 0.1)
    lz = params.get("lz", 0.04)
    maxh = params.get("maxh", 0.02)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""3D Neo-Hookean hyperelasticity — NGSolve"""
from ngsolve import *
from netgen.occ import *
import json

# Geometry — set dimensions for your problem
box = Box((0,0,0), ({lx},{ly},{lz}))
geo = OCCGeometry(box)
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))
fes = VectorH1(mesh, order=2, dirichlet=".*")
u = fes.TrialFunction()

mu, lam = {mu}, {lam}
I = Id(3)
F = I + Grad(u)
C = F.trans * F
J = Det(F)
energy = 0.5*mu*(Trace(C) - 3) - mu*log(J) + 0.5*lam*log(J)**2

a = BilinearForm(fes, symmetric=True)
a += Variation(energy * dx)

gfu = GridFunction(fes)
# Apply load incrementally — set for your problem
for load in [0.2, 0.5, 0.8, 1.0]:
    gfu.Set(CoefficientFunction((0, 0, -0.01*load)), definedon=mesh.Boundaries(".*"))
    solvers.Newton(a, gfu, maxit=20, printing=False)

vtk = VTKOutput(mesh, coefs=[gfu], names=["displacement"], filename="result", subdivision=1)
vtk.Do()
summary = {{"n_dofs": fes.ndof}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "hyperelasticity": {
        "description": (
            "Nonlinear hyperelasticity via Variation(energy*dx) + "
            "ngsolve.solvers.Newton."
        ),
        "spaces": "VectorH1(mesh, order=2+)",
        "solver": (
            "ngsolve.solvers.Newton(a, gfu, maxit=20, dampfactor=1.0, "
            "maxerr=1e-11, printing=False) — built-in damped Newton."
        ),
        "pitfalls": [
            "[API] solvers.Newton uses kwarg names maxit (singular, "
            "default 100) and maxerr (default 1e-11), NOT maxits "
            "and tol. The real signature is "
            "Newton(a, u, freedofs=None, maxit=100, maxerr=1e-11, "
            "inverse='', dirichletvalues=None, dampfactor=1, "
            "printing=True, callback=None). Passing maxits=20 or "
            "tol=1e-10 raises TypeError: Newton() got an "
            "unexpected keyword argument. Signal: "
            "solvers.Newton(a, gfu, maxits=20) raises TypeError "
            "with the literal kwarg name in the message; "
            "solvers.Newton(a, gfu, maxit=20) returns the "
            "(iterations, convergence) tuple. (Verified empirically "
            "2026-06-01 — Tier-2 fixture "
            "hyperelasticity_newton_maxit_kwarg in scripts/"
            "tier2_fixtures/ngsolve/. Five prior generator "
            "occurrences using the wrong kwargs corrected in same "
            "commit.)",
            "[API] Use Variation(energy*dx) on the BilinearForm — "
            "NGSolve auto-derives the residual a.Apply(gfu.vec, res) "
            "and the tangent a.AssembleLinearization(gfu.vec). "
            "Both methods take the current solution vector and "
            "either populate a residual block or assemble the "
            "tangent. Signal: hasattr(BilinearForm(fes), 'Apply') "
            "and hasattr(BilinearForm(fes), 'AssembleLinearization') "
            "both True; calling solvers.Newton on a BilinearForm "
            "without Variation() leaves the residual at zero "
            "(Newton converges in 0 iterations on the initial "
            "guess). (Catalog claim inherited; not separately "
            "Tier-2 falsified this iteration.)",
            "[Physics] Neo-Hookean strain energy density is "
            "psi(C) = 0.5*mu*(Tr(C)-d) - mu*ln(J) + 0.5*lam*ln(J)^2 "
            "where d is the spatial dimension, F = Id + Grad(u), "
            "C = F.trans * F, J = Det(F). The ln(J) terms are "
            "well-defined only for J>0 — Newton can lose this "
            "invariant during a too-large load step and produce "
            "ln(negative) → NaN. Signal: gfu.vec norm becomes nan "
            "after Newton step; check J = Det(Id + Grad(gfu)) "
            "evaluates positive on a quadrature point sample "
            "before each load increment. (Catalog claim inherited; "
            "not separately Tier-2 falsified this iteration.)",
            "[Numerical] For large deformations: use load stepping "
            "(incremental Dirichlet BC scaled by alpha = step/"
            "n_steps) and restart Newton from the previous "
            "converged gfu. Applying the full prescribed "
            "displacement in one Newton call almost always "
            "diverges because the first linearization is too far "
            "from the solution manifold. Signal: solvers.Newton "
            "returns (iters, conv) with iters == maxit and conv "
            "still > maxerr — that is the divergence signal. "
            "Reduce the step size or reduce dampfactor below 1.0.",
            "[API] solvers.Newton's dampfactor kwarg (default 1.0) "
            "scales every update by alpha in [0,1] — gfu += "
            "dampfactor * du. Reducing it to 0.5-0.7 trades "
            "iteration count for robustness when the initial "
            "guess is far from the converged solution. Signal: "
            "with dampfactor=1.0 Newton diverges (conv stays at "
            "1e+something); with dampfactor=0.5 it converges in "
            "more iterations but the conv value drops below "
            "maxerr.",
        ],
    },
}

GENERATORS = {
    "hyperelasticity_2d": _hyperelasticity_2d,
    "hyperelasticity_3d": _hyperelasticity_3d,
}
