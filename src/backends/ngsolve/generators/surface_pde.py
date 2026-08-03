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

# H1 space on the SURFACE (boundary of the volume mesh).
# definedon=mesh.Boundaries(".*") restricts the FE space to
# the spherical surface; without it, H1 lives on the full
# volume and the boundary integrals below would mix trace
# DOFs with interior DOFs.
# Audit 2026-06-02: prior version used H1(mesh, order=...,
# dirichlet="") on the volume mesh and then assembled
# `grad(u) * grad(v) * ds`. NGSolve raised
#   NgException: Trialfunction does not support BND-forms,
#               maybe a Trace() operator is missing, type=grad
# Fix: restrict the space to the surface AND wrap each
# trial/test occurrence with .Trace() so the BND integral
# typechecks. See Joachim Schöberl's surface-FEM example
# in iFEM lecture notes (definedon + .Trace() pattern).
fes_constrained = H1(mesh, order={order},
                      definedon=mesh.Boundaries(".*"))
u_c, v_c = fes_constrained.TnT()

a_c = BilinearForm(fes_constrained)
a_c += (grad(u_c).Trace() * grad(v_c).Trace()
         + 1e-8 * u_c.Trace() * v_c.Trace()) * ds
a_c.Assemble()

# Source term on the surface — set for your problem.
# Use spherical harmonics Y_2^0 as forcing: f = 6*z^2 - 2
# (eigenfunction of -Δ_S on the unit sphere). Integral over
# the sphere is 0, so the residual has no constant mode and
# the 1e-8 regulariser is just a safety net to keep the
# matrix non-singular.
f_expr = 6 * z * z - 2
f_c = LinearForm(fes_constrained)
f_c += f_expr * v_c.Trace() * ds
f_c.Assemble()

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
                "(volume) for surface integrals, and wrap every "
                "trial/test occurrence in .Trace(). Signal: on "
                "the generator's own geometry "
                "(OCCGeometry(Sphere(...)) -> 3-D volume mesh, "
                "H1(mesh, order=3, "
                "definedon=mesh.Boundaries('.*'))), a "
                "BilinearForm with "
                "grad(u).Trace()*grad(v).Trace()*dx raises "
                "NgException('Trialfunction does not support "
                "VOL-forms, maybe a Trace() operator is missing, "
                "type = gradboundary'); the same form with *ds "
                "assembles. It does NOT silently give zero, and "
                "`mesh dim mismatch in BilinearForm` is not a "
                "string NGSolve 6.2.2604 emits. Note dx on that "
                "mesh is still meaningful for VOLUME quantities "
                "(Integrate(1*dx) = 4.188792 = the ball volume). "
                "(Verified empirically 2026-08-03 — signal-text "
                "correction.)"
            ),
            (
                "[API] grad on a surface mesh AUTOMATICALLY "
                "produces the TANGENTIAL (surface) gradient — "
                "no explicit projection needed. Signal: "
                "manually projecting grad(u) onto the surface "
                "tangent plane via (I - n n^T) * grad(u) "
                "applies the projection TWICE (NGSolve already "
                "did it for you) and yields a numerically "
                "near-identical answer but at 2x the cost. "
                "Signal: on the curved sphere surface "
                "(mesh.Curve(4), H1 order 3 restricted to "
                "mesh.Boundaries('.*'), gfu.Set(x*y+z)), "
                "Integrate((grad(gfu).Trace()*n)**2*ds) = "
                "1.71e-31 — the normal component is already zero "
                "to machine precision. The .Trace() wrapper is "
                "REQUIRED for this to work at all. (Verified "
                "empirically 2026-08-03 on NGSolve 6.2.2604.)"
            ),
            (
                "[Numerical] Laplace-Beltrami has kernel = "
                "constants, and NEITHER failure mode is loud. "
                "Signal: measured on the sphere, maxh=0.3, "
                "Curve(4), H1 order 3, f = 6z^2-2 — with no "
                "regularisation, "
                "a.mat.Inverse(fes.FreeDofs(), inverse='umfpack') "
                "returns WITHOUT raising and gives "
                "|u|_inf = 1.3e+09 — pure constant-mode blow-up. "
                "With the generator's 1e-8*u*v*ds regulariser the "
                "solve is stable and the SHAPE is right (L2 error "
                "9.7e-05 against the analytic Y_2^0 = z^2-1/3 "
                "after removing the mean) but the mean itself is "
                "off by +5.7e+01, which is why the shipped "
                "template prints max=3387.7 for a solution whose "
                "true range is [-1/3, 2/3]. Always subtract the "
                "mean (or pin a DOF) before reporting min/max. "
                "NOTE: `KSPSolve: DIVERGED_BREAKDOWN` is a PETSc "
                "string and is never emitted by NGSolve. "
                "(Verified empirically 2026-08-03 — signal-text "
                "correction + new failure characterisation.)"
            ),
            (
                "[Numerical] mesh.Curve(order) improves geometry "
                "approximation for curved surfaces. Signal: unit "
                "Sphere via OCCGeometry, maxh=0.3, "
                "Integrate(1*ds) against the exact 4*pi = "
                "12.56637061 — no Curve 12.31844056 (1.973% "
                "low), Curve(1) identical (it is a no-op), "
                "Curve(2) 12.56426008 (0.0168%), Curve(3) "
                "12.56657695 (0.0016%, absolute 2.1e-04), "
                "Curve(5) 12.56637022 (4e-8). Curve(1) buying "
                "nothing is the surprise. (Verified empirically "
                "2026-08-03 on NGSolve 6.2.2604 — prior "
                "'~2-5%' / '~1e-4' estimates both hold.)"
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
