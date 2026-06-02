"""NGSolve Navier-Stokes generators and knowledge."""


def _navier_stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Incompressible Navier-Stokes with IMEX time-stepping."""
    Re = params.get("Re", 100)
    nx = params.get("nx", 32)
    dt = params.get("dt", 0.001)
    T_end = params.get("T_end", 1.0)
    nu = 1.0 / Re
    maxh = 1.0 / nx
    return f'''\
"""Navier-Stokes — IMEX time-stepping — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))
V = VectorH1(mesh, order=2, dirichlet="bottom|right|top|left")
Q = H1(mesh, order=1)
X = V * Q
(u, p), (v, q) = X.TnT()

nu = {nu}
dt = {dt}
# Stokes part (implicit)
stokes = nu*InnerProduct(Grad(u), Grad(v))*dx + div(u)*q*dx + div(v)*p*dx
# Mass
mass = InnerProduct(u, v)*dx
mstar = BilinearForm(X)
mstar += mass + dt*stokes
mstar.Assemble()

gfu = GridFunction(X)
gfu.components[0].Set(CoefficientFunction((1, 0)), definedon=mesh.Boundaries("top"))

velocity = gfu.components[0]

inv = mstar.mat.Inverse(X.FreeDofs(), "umfpack")

t = 0.0
for step in range(int({T_end}/{dt})):
    # Convection (explicit)
    conv = LinearForm(X)
    conv += InnerProduct(Grad(velocity)*velocity, v)*dx
    conv.Assemble()

    rhs = LinearForm(X)
    rhs.Assemble()
    rhs.vec.data = mstar.mat * gfu.vec - dt*conv.vec
    gfu.vec.data = inv * rhs.vec
    t += dt

print(f"t={{t:.4f}}, Re={Re}")
vtk = VTKOutput(mesh, coefs=[gfu.components[0], gfu.components[1]],
                names=["velocity", "pressure"], filename="result", subdivision=1)
vtk.Do()
summary = {{"Re": {Re}, "time": t, "n_dofs": X.ndof}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Navier-Stokes solve complete.")
'''


KNOWLEDGE = {
    "navier_stokes": {
        "description": "Incompressible Navier-Stokes with IMEX (convection explicit, Stokes implicit)",
        "spaces": "VectorH1(order=2) * H1(order=1)",
        "solver": "IMEX: factor Stokes operator once, explicit convection each step",
        "pitfalls": [
            (
                "[Numerical] CFL condition for explicit "
                "convection: dt < h / max(velocity). Signal: "
                "velocity field reaches NaN within the first "
                "10 steps when dt*max|u|/h > ~0.5; per-step "
                "max(|u|) diverges geometrically. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Convection term: Grad(velocity)*"
                "velocity for standard, or skew-symmetric "
                "form. Signal: in a closed periodic box the "
                "non-conservative form drifts the total "
                "kinetic energy by ~1% over 1000 steps; the "
                "skew-symmetric variant preserves it to "
                "machine precision. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Re > ~500 needs finer mesh or "
                "stabilization (SUPG). Signal: spurious "
                "wiggles in the GridFunction velocity "
                "upstream of obstacles; energy spectrum has "
                "high-frequency content not present in a "
                "reference DNS; on Schafer-Turek drag "
                "coefficient computed via BilinearForm "
                "boundary integrals differs > 10% from the "
                "5.57 (Re=20) / 3.22 (Re=100) reference. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Validation] Benchmark: Schafer-Turek DFG "
                "channel with cylinder at Re=20 (steady) and "
                "Re=100 (periodic vortex shedding). Signal: a "
                "Taylor-Hood Re=20 implementation should "
                "produce drag coefficient Cd around 5.57 and "
                "lift Cl in [0.0104, 0.0110] (Schafer-Turek "
                "published bounds); values outside this "
                "envelope expose either pressure-pin / "
                "boundary-condition errors or insufficient "
                "mesh refinement around the cylinder. (Audit "
                "2026-06-02.)"
            ),
        ],
    },
}

GENERATORS = {
    "navier_stokes_2d": _navier_stokes_2d,
}
