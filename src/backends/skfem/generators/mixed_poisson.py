"""scikit-fem mixed Poisson generators and knowledge."""


def _mixed_poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Mixed Poisson with Raviart-Thomas + piecewise constant."""
    nx = params.get("nx", 16)
    refine_level = params.get("refine_level", 4)
    return f'''\
"""Mixed Poisson: RT0 + P0 — scikit-fem"""
from skfem import *
# skfem ships the standard differential helpers (grad, div,
# d/dn, etc.) under skfem.helpers — importing div() from
# this module is the supported way to take the divergence
# of a vector field inside a BilinearForm. The legacy
# attribute-access pattern sigma[0].grad[0] does NOT exist
# on the underlying numpy array (raises AttributeError
# 'numpy.ndarray' object has no attribute 'grad').
from skfem.helpers import div
import numpy as np
import json

m = MeshTri.init_symmetric().refined({refine_level})
e_rt = ElementTriRT0()
e_dg = ElementTriP0()

ib_rt = Basis(m, e_rt)
ib_dg = Basis(m, e_dg)

@BilinearForm
def mass_rt(sigma, tau, w):
    return sigma[0]*tau[0] + sigma[1]*tau[1]

@BilinearForm
def div_form(sigma, v, w):
    return div(sigma) * v

A = asm(mass_rt, ib_rt)
B = asm(div_form, ib_rt, ib_dg)

from scipy.sparse import bmat
K = bmat([[A, B.T], [B, None]], format='csr')
f = np.zeros(K.shape[0])
# Source in scalar part — LinearForm decorator wraps a
# callable that returns the integrand. The 'w' argument
# carries quadrature-point metadata (w.x, w.h, ...).
@LinearForm
def source(v, w):
    return 1.0 * v

f[A.shape[0]:] = -1.0 * asm(source, ib_dg)

u = np.linalg.lstsq(K.toarray(), f, rcond=None)[0]
print(f"Mixed Poisson: {{K.shape[0]}} DOFs")

summary = {{"n_dofs": K.shape[0], "n_elements": m.nelements}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Mixed Poisson solve complete.")
'''


KNOWLEDGE = {
    "mixed_poisson": {
        "description": "Mixed Poisson with Raviart-Thomas (example 37)",
        "solver": "Direct (saddle-point system) or iterative with Schur complement",
        "elements": "ElementTriRT0 (flux) + ElementTriP0 (scalar)",
        "pitfalls": [
            "[Numerical] Mixed Poisson assembles a SADDLE-POINT "
            "block system [[A, B^T], [B, 0]] where A = mass("
            "sigma, tau) (mass form on the flux space) and B = "
            "div(sigma) * v (divergence coupling against the "
            "scalar space). The full block matrix is INDEFINITE — "
            "direct solve via scipy.sparse.linalg.spsolve works "
            "for moderate sizes; iterative solvers need Schur "
            "complement preconditioning. Signal: scipy.sparse."
            "linalg.cg on the block matrix diverges immediately "
            "(indefinite system); spsolve succeeds. (Claim "
            "inherited — not yet empirically separated.)",
            "[API] skfem ships Raviart-Thomas elements as "
            "ElementTriRT0 (3 DOFs per triangle, one normal-flux "
            "DOF per edge), ElementTriRT1, ElementTetRT0. "
            "Catalog name 'ElementTriRaviartThomas' (full spelling) "
            "does NOT exist in skfem; use the RT0/RT1 abbreviated "
            "names. Signal: hasattr(skfem, 'ElementTriRT0') is "
            "True; Basis(MeshTri(), ElementTriRT0()).Nbfun == 3 "
            "(matches the 3-edge count of a triangle); "
            "hasattr(skfem, 'ElementTriRaviartThomas') is False. "
            "(Verified empirically 2026-06-01.)",
            "[API] Use skfem.helpers.div(sigma) inside a "
            "BilinearForm to take the divergence of an RT vector "
            "field. The legacy element-wise pattern "
            "sigma[0].grad[0] + sigma[1].grad[1] does NOT work on "
            "the underlying numpy array — sigma[0] is just a "
            "scalar ndarray with no .grad attribute (raises "
            "AttributeError: 'numpy.ndarray' object has no "
            "attribute 'grad'). div(sigma) is the supported helper "
            "and operates correctly on RT0/RT1/RT2 fields. "
            "(Verified empirically 2026-06-01 — Layer F catch.)",
            "[API] Wrap source/RHS callables via the @LinearForm "
            "decorator on a plain Python function, not via "
            "LinearForm(lambda v, w: ...). The lambda form mostly "
            "works but mis-resolves the kwargs adapter inside asm; "
            "the decorator form is the canonical skfem pattern. "
            "Signal: asm(LinearForm(lambda v, w: 1.0 * v), basis) "
            "may raise opaque shape errors deep inside "
            "skfem.assembly.form.linear_form; switching to a "
            "decorator-wrapped function resolves them. (Inherited "
            "claim — confirmed during Layer F mixed_poisson fix.)",
            "[Numerical] Neumann BC for the mixed (flux-pressure) "
            "formulation requires adding a boundary integral to "
            "the RHS — the flux trace is the natural BC and is "
            "imposed by integrating g * normal_v over the "
            "Neumann boundary on the test side. Forgetting this "
            "leaves the boundary flux unconstrained. Signal: "
            "post-processed sigma at the Neumann boundary "
            "deviates from the prescribed value by O(1); adding "
            "the boundary integral via skfem.FacetBasis + asm() "
            "recovers it. (Claim inherited — not yet empirically "
            "separated.)",
        ],
    },
}

GENERATORS = {
    "mixed_poisson_2d": _mixed_poisson_2d,
}
