"""NGSolve Helmholtz equation generators and knowledge."""


def _helmholtz_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Helmholtz equation with PML absorbing layer."""
    k = params.get("k", 10)
    order = params.get("order", 4)
    maxh = params.get("maxh", 0.05)
    return f'''\
"""Helmholtz: -Δu - k²u = f with PML — NGSolve (complex-valued)"""
from ngsolve import *
from netgen.geom2d import SplineGeometry
import json

geo = SplineGeometry()
geo.AddCircle((0,0), r=1.0, bc="outer")
geo.AddCircle((0,0), r=0.7, leftdomain=2, rightdomain=1)
geo.SetMaterial(1, "pml")
geo.SetMaterial(2, "inner")
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))

mesh.SetPML(pml.Radial(rad=0.7, alpha=2j), definedon="pml")

fes = H1(mesh, order={order}, complex=True, dirichlet="outer")
u, v = fes.TnT()
k = {k}
a = BilinearForm(grad(u)*grad(v)*dx - k**2*u*v*dx).Assemble()

# Point source at origin
f = LinearForm(exp(-100*(x**2+y**2))*v*dx).Assemble()

gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

print(f"Helmholtz k={k}, DOFs: {{fes.ndof}}")
vtk = VTKOutput(mesh, coefs=[gfu.real, gfu.imag],
                names=["Re_u", "Im_u"], filename="result", subdivision=2)
vtk.Do()
summary = {{"k": k, "n_dofs": fes.ndof}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Helmholtz solve complete.")
'''


KNOWLEDGE = {
    "helmholtz": {
        "description": "Helmholtz equation with PML (perfectly matched layer)",
        "spaces": "H1(mesh, order=k, complex=True) — MUST use complex=True",
        "solver": "Direct for moderate k. For large k, use GMRES with multigrid",
        "pitfalls": [
            "[Syntax] complex=True flag required on FESpace for any "
            "Helmholtz form that carries a complex coefficient "
            "(absorbing BC, PML, time-harmonic source). Without it, "
            "the BilinearForm with a 1j*k*u*v*dx term hits the "
            "same ScaleCF check that maxwell#3 trips on. Signal: "
            "NgException 'real Evaluate called for complex ScaleCF "
            "in Assemble BilinearForm \"biform_from_py\"' from "
            "BilinearForm.Assemble. (Verified empirically "
            "2026-06-01 — identical wording to the maxwell#3 case; "
            "the catch site is the BFI assembler.)",
            "[Numerical] PML setup uses mesh.SetPML(pml.Radial("
            "rad=r, alpha=a_j)) where alpha is IMAGINARY "
            "(complex-valued). A real-only alpha is not a PML — it "
            "becomes a lossy real boundary that reflects the "
            "outgoing wave. Signal: post-processed outgoing wave "
            "amplitude in the bulk shows standing-wave fringes "
            "(reflections) rather than monotonic decay into the "
            "PML region; the L2 norm in the bulk does NOT decrease "
            "as PML thickness is increased. (Catalog claim "
            "inherited — not yet empirically verified on a "
            "running PML simulation.)",
            "[Numerical] PML alpha magnitude must be tuned: too "
            "small → outgoing wave is partially reflected from "
            "the PML/bulk interface; too large → high local "
            "wavenumber inside PML makes the linear solve "
            "ill-conditioned. Practical starting range "
            "alpha = 1.0..5.0 (imaginary). Signal: reflection "
            "amplitude vs alpha forms a U-curve with the minimum "
            "in the 1.0..5.0 range; below or above that the "
            "post-processed |u_outgoing| at the inner PML "
            "boundary is O(0.1)|u_source|. (Catalog claim "
            "inherited — not yet empirically verified.)",
            "[Numerical] Resolution rule of thumb: ~10 DOFs per "
            "wavelength minimum, i.e. order p, h < lambda/(2p). "
            "Insufficient resolution makes the Helmholtz pollution "
            "effect dominate. Signal: phase error of the "
            "post-processed solution grows as O(k^(p+1) h^(2p+1)); "
            "for k > 100, h < lambda/10 is not enough at p=1, and "
            "the L2 error against an analytic plane wave plateaus "
            "around 10% regardless of further refinement. (Catalog "
            "claim inherited — not yet empirically verified at this "
            "magnitude.)",
            "[API] For eigenvalues / cavity resonances: use "
            "ArnoldiSolver with shift-invert; the shift should be "
            "near the expected eigenvalue (k^2_estimate from "
            "analytic cavity formula). shift=0 raises NgException "
            "'UmfpackInverse: Numeric factorization failed. UMFPACK "
            "... WARNING: matrix is singular' on operators with a "
            "non-empty null space (same family as maxwell#5). "
            "Signal: NgException with 'UmfpackInverse' + 'matrix "
            "is singular' from ArnoldiSolver when shift=0. "
            "(Verified empirically 2026-06-01 — same pattern as "
            "maxwell#5.)",
        ],
    },
}

GENERATORS = {
    "helmholtz_2d": _helmholtz_2d,
}
