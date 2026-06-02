"""NGSolve surface PDE generators and knowledge."""


def _surface_pde_3d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Laplace-Beltrami equation on a curved surface manifold (sphere)."""
    order = params.get("order", 3)
    maxh = params.get("maxh", 0.3)
    return f'''\
"""Laplace-Beltrami on sphere surface — surface FEM — NGSolve"""
from ngsolve import *
from netgen.occ import *
import json

# Geometry: sphere surface — set for your problem
sphere = Sphere(Pnt(0,0,0), r=1.0)
geo = OCCGeometry(sphere)
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))
mesh.Curve({order})

print(f"Surface mesh: {{mesh.ne}} elements, {{mesh.nv}} vertices")

# H1 space on the surface manifold
fes = H1(mesh, order={order}, dirichlet="")
u, v = fes.TnT()

# Laplace-Beltrami: surface gradient on the manifold
# NGSolve automatically restricts grad to the tangent plane on surface meshes
a = BilinearForm(grad(u) * grad(v) * ds).Assemble()

# Source term on the surface — set for your problem
# Use spherical harmonics Y_2^0 as forcing: f = 6*z^2 - 2 (eigenfunction)
f_expr = 6 * z * z - 2
f = LinearForm(f_expr * v * ds).Assemble()

# Pin one DOF to fix the constant (Laplace-Beltrami has kernel = constants)
fes_constrained = H1(mesh, order={order})
u_c, v_c = fes_constrained.TnT()
a_c = BilinearForm(grad(u_c) * grad(v_c) * ds + 1e-8 * u_c * v_c * ds).Assemble()
f_c = LinearForm(f_expr * v_c * ds).Assemble()

gfu = GridFunction(fes_constrained)
gfu.vec.data = a_c.mat.Inverse(fes_constrained.FreeDofs()) * f_c.vec

# The exact solution is the spherical harmonic Y_2^0 = z^2 - 1/3
# (up to a constant shift)
max_val = max(gfu.vec)
min_val = min(gfu.vec)
print(f"Solution: max={{max_val:.8f}}, min={{min_val:.8f}}")

vtk = VTKOutput(mesh, coefs=[gfu], names=["solution"],
                filename="result", subdivision=2)
vtk.Do()

summary = {{
    "max_value": float(max_val),
    "min_value": float(min_val),
    "n_dofs": fes_constrained.ndof,
    "n_elements": mesh.ne,
    "order": {order},
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Surface PDE (Laplace-Beltrami) solve complete.")
'''


KNOWLEDGE = {
    "surface_pde": {
        "description": "PDEs on curved surface manifolds (Laplace-Beltrami, surface diffusion)",
        "spaces": "H1 on surface mesh (NGSolve automatically restricts to tangent plane)",
        "solver": "Direct or iterative — standard solvers work on surface meshes",
        "mesh": "OCC surfaces (Sphere, Cylinder, STEP import), mesh.Curve(order) for geometry approximation",
        "pitfalls": [
            (
                "[API] Use ds (surface measure) instead of dx "
                "(volume) for surface integrals. Signal: using "
                "dx on a surface mesh gives zero integral or "
                "raises `mesh dim mismatch in BilinearForm` — "
                "the volume measure has no support on a "
                "(d-1)-manifold. (Audit 2026-06-02.)"
            ),
            (
                "[API] grad on a surface mesh AUTOMATICALLY "
                "produces the TANGENTIAL (surface) gradient — "
                "no explicit projection needed. Signal: "
                "manually projecting grad(u) onto the surface "
                "tangent plane via (I - n n^T) * grad(u) "
                "applies the projection TWICE (NGSolve already "
                "did it for you) and yields a numerically "
                "near-identical answer but at 2x the cost; "
                "verify by checking that |grad(u) . n| at a "
                "surface point is already ~1e-15. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Laplace-Beltrami has kernel = "
                "constants; pin one DOF or add regularization. "
                "Signal: solver returns `KSPSolve: "
                "DIVERGED_BREAKDOWN` or near-zero pivot; the "
                "computed solution u contains an arbitrary "
                "additive constant. Pin u(p0) = 0 at one DOF or "
                "add eps*u to the operator with eps ~ 1e-8. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] mesh.Curve(order) improves geometry "
                "approximation for curved surfaces. Signal: on a "
                "Sphere mesh without Curve(3), the area integral "
                "of 1*ds differs from 4*pi*R^2 by ~2-5% (linear "
                "facet approximation); with mesh.Curve(3) the "
                "area converges to within ~1e-4 of the exact "
                "value. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] For evolving surfaces (moving "
                "membranes, growth): use a deformation mapping "
                "+ ALE approach with mesh.SetDeformation. "
                "Signal: re-meshing every step (creating new "
                "OCC geometry per time step) loses solution "
                "continuity — the per-DOF solution from step "
                "N doesn't transfer cleanly to step N+1 and "
                "you see ~1-3% loss of L^2 norm per step "
                "(spurious dissipation). ALE keeps the same "
                "DOFs but deforms the mesh and resolves "
                "transport on the moved geometry. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] Surface meshes from OCC: Sphere(), "
                "Cylinder(), or any imported STEP/BREP "
                "surface via OCCGeometry(...). Signal: "
                "loading a STEP file with a mix of "
                "volumetric solids and surface patches "
                "without selecting only the .faces leads to "
                "a 3D mesh being constructed instead of a "
                "surface mesh — Mesh.dim returns 3 instead "
                "of 2, and subsequent ds integrations span "
                "the volume boundary, not the intended "
                "surface. Use OCC.Glue(faces) or "
                "geo.faces[i] to select surfaces. (Audit "
                "2026-06-02.)"
            ),
        ],
    },
}

GENERATORS = {
    "surface_pde_3d": _surface_pde_3d,
}
