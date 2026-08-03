"""DUNE-fem nonlinear PDE generators and knowledge."""


def _nonlinear_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Nonlinear PDE via Newton iteration — DUNE-fem."""
    nx = params.get("nx", 32)
    return f'''\
"""Nonlinear PDE: -div((1+u^2)*grad(u)) = 1 — Newton — DUNE-fem"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import TrialFunction, TestFunction, dot, grad, dx
import numpy as np
import json

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
space = lagrange(gridView, order=2)
u = TrialFunction(space)
v = TestFunction(space)

# Nonlinear form: -div((1+u^2)*grad(u)) = f
# Write the NONLINEARITY IN THE TRIAL FUNCTION. dune-fem differentiates
# the form symbolically to build the Jacobian and runs Newton inside
# scheme.solve(); a form written with the discrete function instead has
# only ONE argument and is rejected with
#   ValueError: Integrands model requires form with at least two arguments.
uh = space.interpolate(0, name="solution")
a = (1 + u**2) * dot(grad(u), grad(v)) * dx
b = 1.0 * v * dx

dbc = DirichletBC(space, 0)
scheme = galerkin([a == b, dbc], solver="cg")

# Newton iteration is internal to galerkin scheme
info = scheme.solve(target=uh)
vals = np.array(uh.as_numpy)
print(f"Nonlinear PDE: max={{vals.max():.6f}}")
gridView.writeVTK("result", pointdata={{"solution": uh}})
summary = {{"max_value": float(vals.max()), "n_dofs": len(vals)}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
'''


KNOWLEDGE = {
    "nonlinear": {
        "description": "Nonlinear PDEs solved via built-in Newton iteration",
        "solver": (
            "galerkin scheme handles Newton automatically when the "
            "form is NONLINEAR IN THE TRIAL FUNCTION; scheme.solve() "
            "returns info['iterations'] = the Newton count"),
        "pitfalls": [
            (
                "[API] DUNE-fem LINEARIZES AND APPLIES "
                "NEWTON INTERNALLY — no manual Newton loop "
                "needed. Signal: writing a manual "
                "while-not-converged loop with explicit "
                "Jacobian assembly works but is "
                "redundant; scheme.solve() handles "
                "Newton-Krylov natively, and the returned "
                "info dict carries 'iterations' (Newton) "
                "alongside 'linear_iterations'. (Audit "
                "2026-06-02; confirmed by execution "
                "2026-08-03 — a 2D "
                "-div((1+u^2) grad u) = 1 problem "
                "converged with iterations=2, "
                "linear_iterations=19.)"
            ),
            (
                "[API] Write the nonlinearity in the "
                "TRIAL FUNCTION, not in the discrete "
                "solution function. dune-fem differentiates "
                "the UFL form symbolically to build the "
                "Jacobian, so it needs a form with TWO "
                "arguments (trial and test). Signal: "
                "a = (1 + uh**2)*dot(grad(uh), grad(v))*dx "
                "— the natural 'residual written with the "
                "current iterate' spelling — is rejected "
                "at scheme construction with ValueError: "
                "'Integrands model requires form with at "
                "least two arguments.' The working forms "
                "are a = (1 + u**2)*dot(grad(u), "
                "grad(v))*dx with a == b, or the residual "
                "F = ((1 + u**2)*dot(grad(u), grad(v)) - "
                "f*v)*dx with F == 0; both use u = "
                "TrialFunction(space) and both converged "
                "to the identical solution (max 0.07446243 "
                "on an 8x8 grid, 2 Newton iterations). "
                "(Executed 2026-08-03 on dune-fem "
                "2.12.0.2; this FALSIFIES the earlier "
                "guidance that the form must be written "
                "with u_h rather than TrialFunction. The "
                "shipped nonlinear_2d template was fixed "
                "in the same pass; hyperelasticity_2d "
                "carried the IDENTICAL bug and was found "
                "still broken by the adversarial audit "
                "2026-08-03 — it now runs, 4 Newton "
                "iterations, converged. Note its "
                "finite-strain tangent is not symmetric, "
                "so cg is the wrong Krylov method for it; "
                "bicgstab or gmres.)"
            ),
            (
                "[Numerical] For DIFFICULT nonlinear "
                "problems: use load stepping or "
                "continuation. Signal: scheme.solve() "
                "returning 'Newton did not converge' "
                "(max iterations hit at residual O(1)) "
                "on a problem with strong nonlinearity "
                "(D(u) = u^3, large deformation); solving "
                "a sequence of problems with continuation "
                "parameter from easy to hard with each "
                "previous solution as initial guess "
                "succeeds. (Audit 2026-06-02.)"
            ),
            (
                "[Input] Convergence is controlled by "
                "scheme parameters, and the key PREFIX "
                "changed: use 'nonlinear.*', not "
                "'newton.*'. Signal: scheme = galerkin("
                "[...], solver='gmres', parameters={"
                "'nonlinear.tolerance': 1e-8, "
                "'nonlinear.maxiterations': 50, "
                "'linear.tolerance': 1e-10}) tunes Newton "
                "and its inner Krylov solve. The old "
                "'newton.tolerance' spelling still runs "
                "but emits UserWarning \"the parameter key "
                "'newton' is deprecated. Replace with "
                "'nonlinear'\" and is silently rewritten "
                "to 'nonlinear.tolerance'; the nested "
                "'newton.linear.*' form is deprecated in "
                "favour of plain 'linear.*'. Note also "
                "that the Newton iteration cap is "
                "'maxiterations', not 'maxiter': an "
                "unrecognised key is accepted with no "
                "exception and no warning, so a typo "
                "silently leaves the default in place "
                "(measured — parameters={'nonlinear."
                "maxiter': 3, 'totally.bogus.key': 42} "
                "ran to the normal answer with an empty "
                "warning list). (Audit 2026-06-02; the "
                "newton->nonlinear rewrite and its warning "
                "verified by execution 2026-08-03 on "
                "dune-fem 2.12.0.2, where scheme.parameters "
                "came back as {'nonlinear.tolerance': "
                "1e-10, 'linear.method': 'cg'}.)"
            ),
        ],
    },
}

GENERATORS = {
    "nonlinear_2d": _nonlinear_2d,
}
