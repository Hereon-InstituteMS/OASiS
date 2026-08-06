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
            "domain. NGSolve's H1 basis is HIERARCHICAL, so from "
            "order 2 upward the vertex/edge/bubble coefficients "
            "are NOT function samples and max(gfu.vec) is simply "
            "a different number. Signal: on unit_square maxh=0.5, "
            "gfu.Set(x*(1-x)*y*(1-y)) on H1(order=3) gives "
            "max(gfu.vec)=0.098666 against a sampled field max of "
            "0.063096 (ratio 1.56); gfu.Set(sin(pi*x)*sin(pi*y)) "
            "on H1(order=4) gives max(gfu.vec)=1.469994 vs "
            "sampled 0.999293 (ratio 1.47) AND min(gfu.vec)="
            "-7.609785 for a function that is non-negative "
            "everywhere. The divergence starts at order 2, not "
            "order 3: on the SAME mesh, H1(order=2) with "
            "gfu.Set(x*(1-x)*y*(1-y)) gives max(gfu.vec) = "
            "-0.001019 against a sampled max of +0.061844 — the "
            "DOF maximum is NEGATIVE for a strictly non-negative "
            "function, the worst disagreement of any order — and "
            "H1(order=2) with sin(pi*x)*sin(pi*y) gives "
            "max(gfu.vec) = -0.010028 against sampled +1.025495. "
            "Only order 1 coincides (0.048828 vs 0.048752 and "
            "0.750000 vs 0.748606). Use Integrate/gfu(mesh(x,y)) "
            "sampling for a true L_inf. (Verified empirically "
            "2026-08-03 on NGSolve 6.2.2604 — counterexample "
            "supplied; the prior 'stronger counterexample needed' "
            "caveat is resolved. Order-2 numbers and the "
            "'from order 2 upward' bound re-measured in the "
            "2026-08-03 adversarial re-audit, which falsified the "
            "'on order 1-2 the two still coincide' claim.)",
            "[API] Dirichlet inhomogeneous values on NGSolve: "
            "construct gfu = GridFunction(fes); call gfu.Set("
            "boundary_cf, definedon=mesh.Boundaries(name)) to "
            "set the boundary values, then modify the RHS as "
            "f.vec -= a.mat * gfu.vec before calling Inverse on "
            "FreeDofs. Skipping the RHS modification leaves the "
            "system inconsistent. Signal: on unit_square maxh=0.3, "
            "H1(order=2) with the harmonic exact solution "
            "u = x^2 - y^2 imposed on all four edges, the "
            "two-step pattern (gfu.Set then "
            "gfu.vec.data += a.mat.Inverse(fes.FreeDofs()) * "
            "(f.vec - a.mat * gfu.vec)) gives L2 error 1.75e-16, "
            "while overwriting with "
            "gfu.vec.data = a.mat.Inverse(...) * f.vec gives L2 "
            "error 4.22e-01 — i.e. the boundary data is simply "
            "erased, no exception. (Verified empirically "
            "2026-08-03 on NGSolve 6.2.2604.)",
            "[API] VTKOutput writes .vtu natively — no XDMF "
            "conversion required. The constructor expects a "
            "(mesh, coefs, names, filename) tuple; VTKOutput.Do() "
            "emits the file. For higher-order fields, pass "
            "subdivision=N: every exported cell is LINEAR, so "
            "subdivision is the only way the P2+ enrichment "
            "reaches the file. Note 'all linear TRIANGLES' holds "
            "only at subdivision=0 — from subdivision=1 upward "
            "NGSolve also emits linear QUADS, so the cell-type "
            "array carries both VTK type 5 and type 9 and a "
            "reader that assumes a single cell type mis-parses "
            "the file. Signal: on unit_square maxh=0.3 (24 "
            "triangles), a P2 GridFunction written with "
            "subdivision=0 yields a .vtu with NumberOfCells=24 — "
            "exactly one linear cell per element, so the "
            "inter-vertex P2 enrichment is absent; subdivision=2 "
            "yields NumberOfCells=240 and a majority of the "
            "exported VALUES are ones the subdivision-0 file did "
            "not contain, which is the check that subdivision "
            "actually added samples rather than just cells. "
            "Reading the file back needs care: NGSolve writes "
            "the arrays as RAW APPENDED BINARY with 4-byte "
            "length prefixes, so follow the XML offsets — "
            "reading the element text gives an empty array and "
            "assuming an 8-byte prefix gives garbage. (Verified "
            "empirically 2026-08-03, extended 2026-08-06 on "
            "NGSolve 6.2.2604 — the 'all linear triangles' "
            "wording holds only at subdivision 0.)",
            "[API] subdivision=2 on VTKOutput is the recommended "
            "default for any order >= 2 FESpace. Cells per element "
            "grow as T(2^N) = 2^N*(2^N+1)/2, NOT as 4^N: measured "
            "1, 3, 10, 36 cells per triangle for N = 0, 1, 2, 3. "
            "Signal: unit_square maxh=0.3 (24 elements), P2 "
            "GridFunction — subdivision=0 -> NumberOfCells=24, "
            "3514 bytes; subdivision=1 -> 72 cells, 6731 bytes; "
            "subdivision=2 -> 240 cells, 16983 bytes; "
            "subdivision=3 -> 864 cells, 52745 bytes. The prior "
            "catalog numbers (N_elements*3 cells / ~10 KB at "
            "subdivision=0; N_elements*48 / ~150 KB at "
            "subdivision=2) were both WRONG. Note the .vtu is "
            "BINARY-appended — open it 'rb' if you want to grep "
            "the header. (Verified empirically 2026-08-03 on "
            "NGSolve 6.2.2604 — catalog-drift correction.)",
            "[Numerical] GAP FILLED 2026-08-03 — the H1 order/"
            "convergence promise in `spaces` above is now "
            "measured, not asserted. MMS on unit_square with "
            "u = sin(pi x) sin(pi y), f = 2 pi^2 sin sin, full "
            "Dirichlet, maxh = 0.4 / 0.2 / 0.1 / 0.05, "
            "Integrate(..., order=2k+4): order=1 L2 errors "
            "1.669e-01 / 3.923e-02 / 8.013e-03 / 1.763e-03 "
            "(rates 2.09, 2.29, 2.18); order=2 1.324e-02 / "
            "1.559e-03 / 1.774e-04 / 1.983e-05 (rates 3.09, "
            "3.14, 3.16); order=3 1.717e-03 / 1.137e-04 / "
            "4.833e-06 / 2.324e-07 (rates 3.92, 4.56, 4.38). "
            "L2 order k+1 confirmed for k = 1, 2, 3. Signal: if "
            "your own MMS rate comes out one order low, suspect "
            "under-integration — pass an explicit `order=` to "
            "Integrate when measuring the error, because the "
            "default quadrature is chosen for the FORM, not for "
            "an error functional against a transcendental exact "
            "solution. (Verified empirically 2026-08-03 on "
            "NGSolve 6.2.2604.)",
        ],
    },
}

GENERATORS = {
    "poisson_2d": _poisson_2d,
    "poisson_3d": _poisson_3d,
}
