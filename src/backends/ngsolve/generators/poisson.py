"""NGSolve Poisson equation generators and knowledge."""


def _poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Poisson -Δu = f on [0,1]², u=0 on ∂Ω."""
    nx = params.get("nx", 32)
    f_val = params.get("f", 1.0)
    order = params.get("order", 1)
    maxh = 1.0 / nx
    return f'''\
"""Poisson -Δu = {f_val} on [0,1]², u=0 on boundary — NGSolve"""
from ngsolve import *
from ngsolve.webgui import Draw  # type: ignore
import json

# Mesh
mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))
print(f"Mesh: {{mesh.ne}} elements, {{mesh.nv}} vertices")

# FE space
fes = H1(mesh, order={order}, dirichlet="bottom|right|top|left")
u, v = fes.TnT()

# Bilinear form and linear form
a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
f = LinearForm({f_val}*v*dx).Assemble()

# Solve
gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

# Output
max_val = max(gfu.vec)
print(f"max(u) = {{max_val:.10f}}")
print(f"DOFs: {{fes.ndof}}")

# VTK output
vtk = VTKOutput(mesh, coefs=[gfu], names=["phi"],
                filename="result", subdivision=0)
vtk.Do()

# Summary
summary = {{
    "max_value": float(max_val),
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
    "h": {maxh},
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Poisson solve complete.")
'''


def _poisson_3d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Poisson -Δu = f on [0,1]³, u=0 on ∂Ω."""
    nx = params.get("nx", 8)
    f_val = params.get("f", 1.0)
    order = params.get("order", 1)
    maxh = 1.0 / nx
    return f'''\
"""Poisson -Δu = {f_val} on [0,1]³, u=0 on boundary — NGSolve"""
from ngsolve import *
import json
from netgen.csg import unit_cube

# Mesh
geo = unit_cube
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))
print(f"Mesh: {{mesh.ne}} elements, {{mesh.nv}} vertices")

# FE space
fes = H1(mesh, order={order}, dirichlet=".*")
u, v = fes.TnT()

a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
f = LinearForm({f_val}*v*dx).Assemble()

gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

max_val = max(gfu.vec)
print(f"max(u) = {{max_val:.10f}}")

vtk = VTKOutput(mesh, coefs=[gfu], names=["phi"],
                filename="result", subdivision=0)
vtk.Do()

summary = {{
    "max_value": float(max_val),
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("3D Poisson solve complete.")
'''


KNOWLEDGE = {
    "poisson": {
        "description": "Poisson equation -Δu = f with NGSolve (arbitrary-order H1)",
        "spaces": "H1 (Lagrange, order 1-10+)",
        "solver": "Direct: sparsecholesky, umfpack, pardiso. Iterative: CG + h1amg/multigrid/bddc",
        "mesh": "unit_square, unit_cube, SplineGeometry (2D), CSG (3D), OCC (CAD import)",
        "pitfalls": [
            "[Syntax] Boundary names in the `dirichlet=` "
            "argument of H1/HCurl/etc. must match the mesh's "
            "boundary labels EXACTLY (case-sensitive). The "
            "unit_square / unit_cube netgen meshes use lowercase "
            "'left', 'right', 'top', 'bottom' (and 'front', 'back' "
            "in 3D). Failure mode is SILENT: a wrong-case "
            "'Left|Right|Top|Bottom' does NOT raise — it produces "
            "an FESpace where the catalog-expected Dirichlet DoFs "
            "are still FREE. Signal: H1(mesh, ..., dirichlet="
            "'Left|...').FreeDofs() reports 0 fixed DoFs instead "
            "of the boundary count; sum(bool(f) for f in "
            "fes.FreeDofs()) equals fes.ndof. (Verified "
            "empirically 2026-06-01 with unit_square + wrong "
            "capitalisation.)",
            "[API] max(gfu.vec) returns the maximum over the "
            "underlying FlatVector of DOF VALUES, not the "
            "pointwise maximum of the FE function over the "
            "domain. For P1 / nodal interpolations on rectilinear "
            "geometries the two coincide; for hierarchical P2+ or "
            "non-nodal bases (HCurl, HDiv, DG) the DOF max may "
            "be very different from the field max. Signal: "
            "comparing max(gfu.vec) against an L_inf reference "
            "obtained by sampling gfu at a fine grid disagrees "
            "by a factor that depends on element order and basis "
            "type. (Claim inherited — verified that DOF-max and "
            "sampled-max coincide for u=x*y on P2; a stronger "
            "counterexample with a polynomial function whose "
            "maximum is mid-element is needed to fully separate "
            "the two.)",
            "[API] Dirichlet inhomogeneous values on NGSolve: "
            "construct gfu = GridFunction(fes); call gfu.Set("
            "boundary_cf, definedon=mesh.Boundaries(name)) to "
            "set the boundary values, then modify the RHS as "
            "f.vec -= a.mat * gfu.vec before calling Inverse on "
            "FreeDofs. Skipping the RHS modification leaves the "
            "system inconsistent. Signal: post-solve, the trace "
            "of gfu on the boundary differs from the prescribed "
            "boundary_cf by O(1) (rather than O(eps)). (Claim "
            "inherited from prose — empirically plausible "
            "pattern matching NGSolve docs, not yet probed.)",
            "[API] VTKOutput writes .vtu natively — no XDMF "
            "conversion required. The constructor expects a "
            "(mesh, coefs, names, filename) tuple; VTKOutput.Do() "
            "emits the file. For higher-order fields, pass "
            "subdivision=N to refine the visual mesh by 2^N. "
            "Signal: a P2+ GridFunction written with VTKOutput("
            "..., subdivision=0).Do() yields a .vtu whose cells "
            "are linear (the inter-vertex P2 enrichment is "
            "absent from the output); switching to "
            "VTKOutput(..., subdivision=2).Do() recovers the "
            "curvature in the file. (Claim inherited — not yet "
            "empirically compared against a downstream render.)",
            "[API] subdivision=2 on VTKOutput is the recommended "
            "default for any order >= 2 FESpace. Higher subdivision "
            "linearly multiplies output file size (factor 4^N in "
            "2D, 8^N in 3D). Signal: a unit-square P2 solve "
            "written with subdivision=0 produces a ~10 KB .vtu "
            "with N_elements*3 cells; subdivision=2 produces "
            "~150 KB with N_elements*48 cells. (Claim inherited "
            "— not yet empirically file-size verified.)",
        ],
    },
}

GENERATORS = {
    "poisson_2d": _poisson_2d,
    "poisson_3d": _poisson_3d,
}
