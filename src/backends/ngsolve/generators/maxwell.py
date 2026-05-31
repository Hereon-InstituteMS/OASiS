"""NGSolve Maxwell equations generators and knowledge."""


def _maxwell_3d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    3D magnetostatics with HCurl (Nedelec) elements."""
    order = params.get("order", 2)
    maxh = params.get("maxh", 0.3)
    return f'''\
"""Magnetostatics: curl-curl equation — HCurl (Nedelec) — NGSolve"""
from ngsolve import *
from netgen.csg import *
import json, math

# Geometry with material region — set for your problem
geo = CSGeometry()
outer = OrthoBrick(Pnt(-1,-1,-1), Pnt(1,1,1)).bc("outer")
inner = OrthoBrick(Pnt(-0.3,-0.3,-0.3), Pnt(0.3,0.3,0.3))
geo.Add(outer - inner)
geo.Add(inner, mat="source")
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))

fes = HCurl(mesh, order={order}, dirichlet="outer", nograds=True)
u, v = fes.TnT()

mu0 = 4*math.pi*1e-7
# curl-curl + regularization
a = BilinearForm(fes)
a += 1/mu0 * curl(u)*curl(v)*dx + 1e-8/mu0 * u*v*dx
a.Assemble()

# Source current — set for your problem
J = mesh.MaterialCF({{"source": (0, 0, 1)}}, default=(0, 0, 0))
f = LinearForm(J*v*dx).Assemble()

gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

# Magnetic field B = curl(A)
print(f"DOFs: {{fes.ndof}}, Elements: {{mesh.ne}}")
vtk = VTKOutput(mesh, coefs=[gfu, curl(gfu)], names=["A_field", "B_field"],
                filename="result", subdivision=0)
vtk.Do()
summary = {{"n_dofs": fes.ndof, "n_elements": mesh.ne}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Maxwell magnetostatics solve complete.")
'''


KNOWLEDGE = {
    "maxwell": {
        "description": "Maxwell equations with HCurl (Nedelec) edge elements",
        "spaces": "HCurl(mesh, order=k, nograds=True) — tangential continuity",
        "solver": "Direct for small. HCurlAMG preconditioner for large systems",
        "pitfalls": [
            "[Numerical] For SOURCE problems (magnetostatics): use "
            "HCurl(..., nograds=True) to remove the gradient kernel, "
            "plus a 1e-8*u*v*dx regularisation. Signal: without "
            "nograds, BilinearForm.Assemble() succeeds but the "
            "direct factorisation reports a near-singular matrix "
            "(infinite condition number); ArnoldiSolver / Inverse "
            "raises a 'matrix is singular' or 'pivot too small' "
            "NgException because the gradient kernel of HCurl is "
            "in the null space of curl-curl.",
            "[Numerical] For EIGENVALUE problems: do NOT use "
            "nograds=True — it degrades accuracy by 1-3% and "
            "causes eigenvalues to converge from below. Use the "
            "full HCurl space with ArnoldiSolver(shift=<near "
            "expected eigenvalue>). Signal: eigenvalues computed "
            "with nograds=True differ from analytic cavity "
            "eigenvalues (k^2 = (m*pi/Lx)^2 + (n*pi/Ly)^2 + "
            "(p*pi/Lz)^2) by 1-3% AND the sequence approaches "
            "from below; without nograds the same eigenvalues "
            "converge from above and within 0.1%.",
            "[Physics] B = curl(A) — magnetic field is the curl of "
            "the vector potential. Forgetting the curl gives B == A "
            "(vector potential treated as field) and Tesla units "
            "off by order(curl) ~ 1/L. Signal: post-processed "
            "max(|B|) is on the order of the prescribed Dirichlet "
            "value of A directly (no spatial derivative taken).",
            "[Syntax] Complex-valued for time-harmonic: HCurl("
            "mesh, complex=True). On a real HCurl (complex=False) "
            "space, adding a BilinearForm integrator with an "
            "explicit complex coefficient (e.g. 1j*curl(u)*"
            "curl(v)*dx) raises NgException at BilinearForm."
            "Assemble. Signal: NgException with text 'real "
            "Evaluate called for complex ScaleCF' from 'Assemble "
            "BilinearForm'. (Verified empirically 2026-06-01 — "
            "prior catalog wording 'complex values cannot be "
            "assigned to a real FESpace' does not appear in "
            "NGSolve 6.2; the actual emitted string is the "
            "ScaleCF one above.)",
            "[Physics] 3D only — 2D Maxwell reduces to scalar "
            "Helmholtz, NOT to vector HCurl. Defining HCurl(mesh) "
            "on a 2D mesh produces a 1-component space and "
            "curl(u) is a scalar, not a vector. Signal: in 2D, "
            "fes.dim == 1 (scalar) instead of 2 (vector) on an "
            "HCurl space, and BilinearForm += curl(u)*curl(v)*dx "
            "assembles a scalar Helmholtz operator without "
            "warning.",
            "[Numerical] ArnoldiSolver shift: set near expected "
            "eigenvalue range, not near zero. Estimate the lowest "
            "eigenvalue analytically first (k^2 ~ (pi/L)^2 for "
            "cavity); shift=0.5*k^2_expected works well. Signal: "
            "ArnoldiSolver with shift=0.0 returns eigenvalues "
            "near 0 (gradient kernel) and misses the physical "
            "spectrum; eigenvalue residuals are O(1) instead of "
            "1e-8 typical convergence.",
            "[API] Eigenvalue solvers return complex values even "
            "for real-symmetric problems — take .real before "
            "comparison. Signal: numpy.array(ArnoldiSolver result) "
            "has dtype complex128; comparing to analytic real "
            "eigenvalues without .real raises TypeError or "
            "produces nan from complex>real.",
            "[Syntax] HCurl is for vector fields with tangential "
            "continuity. Calling curl(u) on an H1 scalar space "
            "raises NgException('Operator \"curl\" does not exist "
            "for H1HighOrderFESpace'). Signal: NgException with "
            "literal text 'curl' and 'H1HighOrderFESpace' is the "
            "BFI compile-time error a beginner hits when copying "
            "code between scalar (Poisson) and vector (Maxwell) "
            "formulations.",
            "[Syntax] BilinearForm composing grad(u)*grad(v)*dx "
            "(scalar Poisson form) on an HCurl space raises "
            "NgException because grad() is not defined on HCurl. "
            "Signal: NgException emitted by SymbolicBFI about "
            "'grad' / 'HCurl' / 'scalar-valued'; the user gets "
            "an immediate compile-time error when trying to "
            "reuse the Poisson assembly on a Maxwell space.",
            "[API] LinearForm += f*v*dx where f is a 3-vector "
            "and v is a scalar (H1) test function is a vector-"
            "source on a scalar form. Signal: NgException about "
            "SymbolicLFI requiring 'scalar-valued' integrand; "
            "the same JJ vector that works on HCurl test "
            "functions raises immediately on H1 test functions.",
        ],
    },
}

GENERATORS = {
    "maxwell_3d_magnetostatics": _maxwell_3d,
}
