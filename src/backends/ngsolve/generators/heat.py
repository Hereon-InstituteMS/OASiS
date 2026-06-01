"""NGSolve heat conduction generators and knowledge."""


def _heat_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Heat conduction on [0,1]², T_left -> T_right."""
    nx = params.get("nx", 32)
    T_left = params.get("T_left", 100.0)
    T_right = params.get("T_right", 0.0)
    maxh = 1.0 / nx
    return f'''\
"""Heat conduction on [0,1]² — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))

fes = H1(mesh, order=1, dirichlet="left|right")
u, v = fes.TnT()

a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
f = LinearForm(0*v*dx).Assemble()

gfu = GridFunction(fes)

# Apply Dirichlet BCs
gfu.Set(CoefficientFunction({T_left}), definedon=mesh.Boundaries("left"))
gfu.Set(CoefficientFunction({T_right}), definedon=mesh.Boundaries("right"))

# Modify RHS for Dirichlet
f.vec.data -= a.mat * gfu.vec
gfu.vec.data += a.mat.Inverse(fes.FreeDofs()) * f.vec

T_max = max(gfu.vec)
print(f"Temperature: max={{T_max:.6f}}")

vtk = VTKOutput(mesh, coefs=[gfu], names=["temperature"],
                filename="result", subdivision=0)
vtk.Do()

summary = {{
    "max_value": float(T_max),
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Heat solve complete.")
'''


def _heat_transient_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Transient heat equation with implicit Euler time-stepping."""
    nx = params.get("nx", 32)
    dt = params.get("dt", 0.001)
    T_end = params.get("T_end", 0.1)
    maxh = 1.0 / nx
    return f'''\
"""Transient heat equation on [0,1]² — implicit Euler — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))
fes = H1(mesh, order=3, dirichlet="bottom|right|top|left")
u, v = fes.TnT()

# Mass and stiffness matrices
# (Previously had nonsym=True kwarg; current NGSolve flags that as
# undocumented + has no observable effect. Drop it.)
a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
m = BilinearForm(u*v*dx).Assemble()

# Combined: M + dt*A
dt = {dt}
mstar = m.mat.CreateMatrix()
mstar.AsVector().data = m.mat.AsVector() + dt * a.mat.AsVector()
inv_mstar = mstar.Inverse(fes.FreeDofs())

# Source term
f = LinearForm(1*v*dx).Assemble()

# Initial condition: u=0
gfu = GridFunction(fes)
gfu.Set(0)

# Time stepping
vtk = VTKOutput(mesh, coefs=[gfu], names=["temperature"], filename="result", subdivision=0)
t = 0.0
step = 0
while t < {T_end} - 1e-12:
    res = dt * f.vec - dt * a.mat * gfu.vec
    gfu.vec.data += inv_mstar * res
    t += dt
    step += 1

vtk.Do()
max_val = max(gfu.vec)
print(f"t={{t:.4f}}, max(T) = {{max_val:.10f}}, steps={{step}}")
summary = {{"max_value": float(max_val), "n_dofs": fes.ndof, "time": t, "steps": step}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "heat": {
        "description": "Heat conduction: steady and transient (implicit Euler, Crank-Nicolson)",
        "spaces": "H1 (any order)",
        "solver": "Steady: direct. Transient: M+dt*A factored once, reused each step",
        "pitfalls": [
            "[API] The prior catalog wording said 'Transient: "
            "need nonsym=True for mass matrix to get compatible "
            "sparsity pattern' — that is WRONG in current NGSolve. "
            "Passing nonsym=True to the BilinearForm constructor "
            "emits a warning: 'kwarg \"nonsym\" is an undocumented "
            "flags option for class BilinearForm, maybe there is "
            "a typo?', the kwarg is silently dropped, and the "
            "resulting matrix sparsity is identical to the "
            "default. The compatible-sparsity claim never had any "
            "empirical basis. Signal: BilinearForm(..., "
            "nonsym=True).Assemble() emits the warning text on "
            "stderr; m.mat.AsVector().size equals the default "
            "build. (Verified empirically 2026-06-01 — drift "
            "correction; the generator template above also "
            "dropped the kwarg.)",
            "[API] Matrix addition for the implicit-Euler "
            "operator uses BaseMatrix.AsVector() concatenation: "
            "mstar.AsVector().data = m.mat.AsVector() + dt * "
            "a.mat.AsVector(). The .data assignment is needed "
            "because AsVector returns a view, not a copy. "
            "Signal: omitting .data and writing mstar = "
            "m.mat + dt*a.mat does not produce a usable matrix "
            "(the result is a BaseMatrix expression that has "
            "no .Inverse method); attempting mstar.Inverse "
            "raises AttributeError. (Claim inherited — not "
            "yet empirically separated.)",
            "[API] After mesh.Refine(), call fes.Update() and "
            "gfu.Update() on every dependent FESpace + "
            "GridFunction. Current NGSolve also auto-updates "
            "fes.ndof on access after a Refine, but explicit "
            "Update is needed for the underlying DoF tables and "
            "for GridFunction storage to grow. Signal: "
            "fes.Update() and gfu.Update() are both available "
            "as no-arg methods; gfu.vec length after a refine "
            "matches the new ndof only after gfu.Update() is "
            "called. (Verified empirically 2026-06-01 — "
            "fes.ndof DOES auto-update on read but the "
            "GridFunction storage and DoF table do not.)",
            "[Syntax] Non-homogeneous Dirichlet on NGSolve: "
            "two-step pattern. First gfu.Set(boundary_cf, "
            "definedon=mesh.Boundaries(name)) to populate the "
            "boundary DOFs; then solve the residual system on "
            "FreeDofs: gfu.vec.data += a.mat.Inverse("
            "fes.FreeDofs()) * (f.vec - a.mat * gfu.vec). "
            "Skipping the residual step leaves the interior "
            "uncorrected for the non-zero BC. Signal: gfu trace "
            "on the boundary matches boundary_cf but the "
            "interior solution shows discontinuity at the "
            "boundary. (Claim inherited — pattern matches "
            "NGSolve docs.)",
        ],
    },
}

GENERATORS = {
    "heat_2d": _heat_2d,
    "heat_2d_steady": _heat_2d,
    "heat_2d_transient": _heat_transient_2d,
}
