"""NGSolve coupled thermal-structural generators and knowledge."""


def _thermal_structural_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Coupled thermal-structural: heat -> elasticity with thermal strain."""
    E = params.get("E", 200e3)
    nu = params.get("nu", 0.3)
    alpha = params.get("alpha", 12e-6)
    T_hot = params.get("T_hot", 100.0)
    T_cold = params.get("T_cold", 0.0)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Coupled thermal-structural: heat -> thermal expansion — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh=0.03))

# Step 1: Heat conduction
V_T = H1(mesh, order=2, dirichlet="left|right")
uT, vT = V_T.TnT()
aT = BilinearForm(grad(uT)*grad(vT)*dx).Assemble()
# LinearForm(0*vT*dx) collapses symbolically before
# construction, so the resulting form has no TestFunction
# and Assemble() raises NgException 'Linearform must have
# TestFunction'. Build with no integrand and Assemble()
# directly — matches the fix shipped for ngsolve::heat.
fT = LinearForm(V_T); fT.Assemble()
gfT = GridFunction(V_T)
gfT.Set(CoefficientFunction({T_hot}), definedon=mesh.Boundaries("left"))
gfT.Set(CoefficientFunction({T_cold}), definedon=mesh.Boundaries("right"))
fT.vec.data -= aT.mat * gfT.vec
gfT.vec.data += aT.mat.Inverse(V_T.FreeDofs()) * fT.vec
# gfT.name is read-only in this NGSolve build (property
# has no setter). Pass the name string to VTKOutput or
# similar consumers instead of attempting to assign.
print(f"Temperature: [{{min(gfT.vec):.2f}}, {{max(gfT.vec):.2f}}]")

# Step 2: Elasticity with thermal strain
V_u = VectorH1(mesh, order=2, dirichlet="left")
u, v = V_u.TnT()
mu, lam, alpha = {mu}, {lam}, {alpha}
def Strain(u): return 0.5*(Grad(u) + Grad(u).trans)
# Thermal strain
eps_th = alpha * gfT * Id(2)
a_u = BilinearForm(InnerProduct(2*mu*Strain(u) + lam*Trace(Strain(u))*Id(2), Strain(v))*dx).Assemble()
f_u = LinearForm(InnerProduct((3*lam+2*mu)*alpha*gfT*Id(2), Strain(v))*dx).Assemble()

gfu = GridFunction(V_u)
gfu.vec.data = a_u.mat.Inverse(V_u.FreeDofs()) * f_u.vec
# gfu.name property has no setter — see pitfall in the
# heat template; supply the name via the consumer (e.g.
# VTKOutput(names=[...])) instead.

disp_arr = [gfu.components[0](mesh(1,0.5)), gfu.components[1](mesh(1,0.5))]
print(f"Displacement at (1,0.5): u_x={{disp_arr[0]:.6e}}, u_y={{disp_arr[1]:.6e}}")

vtk = VTKOutput(mesh, coefs=[gfT, gfu], names=["temperature", "displacement"],
                filename="result", subdivision=1)
vtk.Do()
summary = {{
    "T_min": float(min(gfT.vec)), "T_max": float(max(gfT.vec)),
    "disp_x_at_tip": disp_arr[0], "disp_y_at_tip": disp_arr[1],
    "n_dofs_thermal": V_T.ndof, "n_dofs_structural": V_u.ndof,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Coupled thermal-structural analysis complete.")
'''


KNOWLEDGE = {
    "thermal_structural": {
        "description": "Coupled thermal-structural: sequential heat -> elasticity with thermal strain",
        "spaces": "H1 (thermal) + VectorH1 (structural)",
        "solver": "Two sequential solves (one-way coupling)",
        "pitfalls": [
            "Thermal strain: eps_th = alpha * T * Id(dim)",
            "RHS for elasticity: (3*lam+2*mu)*alpha*T*Id(dim) contracted with Strain(v)",
            "For two-way coupling: iterate until temperature/displacement converge",
            "[Syntax] Symbolic-zero RHS: LinearForm(0*v*dx) "
            "collapses before construction, leaving a form with "
            "no TestFunction. Assemble() then raises NgException "
            "'Linearform must have TestFunction'. Use the no-"
            "integrand constructor LinearForm(V); f.Assemble() "
            "to build an empty RHS — matches the fix shipped for "
            "ngsolve::heat. Signal: NgException text 'Linearform "
            "must have TestFunction' emitted from LinearForm."
            "Assemble. (Verified empirically 2026-06-01 — "
            "Layer F catch.)",
            "[API] GridFunction.name is a read-only property in "
            "current NGSolve builds — there is no name setter. "
            "Code like gfu.name = 'displacement' raises "
            "AttributeError 'property of GridFunction object "
            "has no setter'. Pass the name string to the "
            "consumer instead (e.g. VTKOutput(names=['u'])). "
            "Signal: AttributeError with the literal text "
            "'property of \\'GridFunction\\' object has no "
            "setter' from a direct gfu.name = '...' assignment. "
            "(Verified empirically 2026-06-01 — Layer F catch.)",
        ],
    },
}

GENERATORS = {
    "thermal_structural_2d": _thermal_structural_2d,
}
