"""scikit-fem adaptive Poisson (h-adaptive) generator and knowledge.

Mirrors scikit-fem upstream ex11 (adaptive Poisson) and ex22 (residual
estimator). The backend previously had `poisson` (uniform mesh) but no
adaptive h-refinement loop, leaving a clear gap relative to upstream.

The residual error estimator (Babuška-Rheinboldt) for -Δu = f on P1 is

    η_K² = h_K² ∫_K f² dx + (1/2) Σ_{e ⊂ ∂K} h_e ∫_e [∇u_h · n_e]² ds

(volume residual `f + Δu_h` reduces to `f` for P1 since Δu_h = 0 in the
element interior).  Mark elements with η_K ≥ θ · max_K(η_K), refine, and
repeat until a target DOF budget is reached.
"""


def _adaptive_poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate
    values for your specific problem.

    h-adaptive Poisson with Babuška-Rheinboldt residual estimator,
    L-shaped re-entrant-corner mesh as a canonical test (the
    singular gradient at the corner drives refinement)."""
    n_iters = params.get("n_iters", 5)
    theta = params.get("theta", 0.5)
    dof_budget = params.get("dof_budget", 20000)
    return f'''\
"""h-adaptive Poisson with residual estimator — scikit-fem"""
from skfem import (MeshTri, Basis, ElementTriP1, solve, condense,
                   FacetBasis, InteriorFacetBasis, Functional)
from skfem.models.poisson import laplace, unit_load
from skfem.helpers import dot, grad
from skfem.assembly import asm
import numpy as np
import json

# L-shaped domain by removing the (0,1)x(0,1) quadrant from
# (-1,1)x(-1,1). The re-entrant corner at (0,0) drives the
# refinement: solution gradient is unbounded there for f=1.
def _build_lshape():
    # Build by manually specifying nodes + triangles.  Six
    # quadrants of the [-1,1]^2 square, minus the upper-right one,
    # cut into right triangles.
    p = np.array([
        [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
        [-1.0,  0.0], [0.0,  0.0], [1.0,  0.0],
        [-1.0,  1.0], [0.0,  1.0],
    ]).T
    t = np.array([
        [0, 1, 4], [0, 4, 3],
        [1, 2, 5], [1, 5, 4],
        [3, 4, 7], [3, 7, 6],
    ]).T
    return MeshTri(p, t)

m = _build_lshape()
# Pre-refine to give the adaptive loop something to bite into.
m = m.refined()
m = m.refined()

theta = {theta}
n_iters = {n_iters}
dof_budget = {dof_budget}

history = []  # list of (iter, n_dof, max_eta, sum_eta)

for k in range(n_iters):
    e = ElementTriP1()
    ib = Basis(m, e)
    if ib.N > dof_budget:
        print(f"DOF budget {{dof_budget}} reached at iter {{k}} (N={{ib.N}})")
        break

    K = laplace.assemble(ib)
    f = unit_load.assemble(ib)
    # Homogeneous Dirichlet on the whole boundary.
    D = ib.get_dofs().flatten()
    u = solve(*condense(K, f, D=D))

    # ── Residual error estimator (Babuska-Rheinboldt) ─────────
    # eta_K^2 = h_K^2 * int_K f^2 dx
    #          + 0.5 * sum_{{e in dK \\ bnd}} h_e * int_e [du/dn]^2 ds
    # For P1, Delta(u_h) = 0 inside elements, so the volume term
    # reduces to h_K^2 * ||f||^2_K with f = 1.

    # Element diameter h_K ~ max edge length.
    p_t = m.p[:, m.t]                              # (2, 3, nE)
    e01 = p_t[:, 1, :] - p_t[:, 0, :]
    e12 = p_t[:, 2, :] - p_t[:, 1, :]
    e20 = p_t[:, 0, :] - p_t[:, 2, :]
    len01 = np.linalg.norm(e01, axis=0)
    len12 = np.linalg.norm(e12, axis=0)
    len20 = np.linalg.norm(e20, axis=0)
    h_K = np.maximum.reduce([len01, len12, len20])

    # Element area via 2D cross-product (oriented).
    area = 0.5 * np.abs(e01[0] * e20[1] - e01[1] * e20[0])

    # Volume residual term (f=1 here): eta_vol^2 = h_K^2 * area.
    eta_vol2 = (h_K ** 2) * area

    # Jump term across interior facets:
    # For P1, grad(u) is constant on each element.  On each
    # interior facet, jump = (grad_in - grad_out) . n.
    grad_u = np.zeros((2, m.t.shape[1]))
    # Per-element gradient from nodal values via the inverse
    # element Jacobian.  P1 shape functions span affine maps;
    # use the closed-form formula:
    #   grad(u)|_K = sum_i u_i * (1/(2A_K)) * R(p_{{i+1}} - p_{{i+2}})
    # where R rotates by 90 deg.  Cleaner: compute via skfem.
    grad_u_field = ib.interpolate(u).grad         # (2, nE, nq)
    # The Cell quadrature is constant-on-element for P1, so grad
    # is identical across quadrature points; take the first.
    grad_u = grad_u_field[:, :, 0]

    eta_jmp2 = np.zeros(m.t.shape[1])
    # Iterate over interior facets only.  m.f2t has shape
    # (2, n_facets) — second row is -1 for boundary facets.
    f2t = m.f2t
    interior_facets = np.where(f2t[1] >= 0)[0]
    for fi in interior_facets:
        e_in, e_out = f2t[0, fi], f2t[1, fi]
        # Facet endpoints.
        vi0, vi1 = m.facets[:, fi]
        edge_vec = m.p[:, vi1] - m.p[:, vi0]
        h_e = np.linalg.norm(edge_vec)
        n = np.array([edge_vec[1], -edge_vec[0]]) / h_e
        jump = (grad_u[:, e_in] - grad_u[:, e_out]) @ n
        # Contribution: 0.5 * h_e * jump^2 * h_e  (line integral
        # of squared constant jump over facet of length h_e).
        contrib = 0.5 * h_e * (jump ** 2) * h_e
        eta_jmp2[e_in] += contrib
        eta_jmp2[e_out] += contrib

    eta2 = eta_vol2 + eta_jmp2
    eta = np.sqrt(eta2)

    max_eta = float(eta.max())
    sum_eta = float(eta.sum())
    history.append((int(k), int(ib.N), max_eta, sum_eta))
    print(f"iter={{k}} N={{ib.N:6d}}  max_eta={{max_eta:.4e}}  "
          f"sum_eta={{sum_eta:.4e}}")

    # Mark elements with eta_K >= theta * max(eta) — Dorfler-style
    # but with absolute threshold for simplicity.
    if max_eta < 1e-12:
        print("estimator below threshold — stopping")
        break
    mark = np.where(eta >= theta * max_eta)[0]
    if len(mark) == 0:
        break
    m = m.refined(mark)

# Final solve on the adapted mesh.
e = ElementTriP1()
ib = Basis(m, e)
K = laplace.assemble(ib)
f = unit_load.assemble(ib)
D = ib.get_dofs().flatten()
u = solve(*condense(K, f, D=D))

import meshio
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
mio = meshio.Mesh(points, [("triangle", m.t.T)],
                  point_data={{"u": u}})
mio.write("result.vtu")

summary = {{
    "n_iters_run": len(history),
    "final_n_dofs": int(ib.N),
    "max_u": float(u.max()),
    "history": history,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


GENERATORS: dict = {
    "adaptive_poisson_2d": _adaptive_poisson_2d,
}


KNOWLEDGE: dict = {
    "adaptive_poisson": {
        "description": (
            "h-adaptive Poisson with the Babuška-Rheinboldt "
            "residual estimator on triangular P1, driven by an "
            "L-shaped re-entrant-corner test case. Matches "
            "scikit-fem upstream ex11 (adaptive Poisson) + ex22 "
            "(residual estimator) — the backend previously had "
            "only uniform Poisson, so this fills the canonical "
            "h-adaptive gap."
        ),
        "weak_form": (
            "Solve -Δu = f, u=0 on ∂Ω. Estimator: "
            "η_K² = h_K² ∫_K f² dx + 0.5 Σ_{e⊂∂K} h_e ∫_e [∇u_h · n]² ds. "
            "Mark with η_K ≥ θ · max_K η_K and refine via "
            "MeshTri.refined(indices)."
        ),
        "elements": ["ElementTriP1"],
        "variants": ["2d"],
        "pitfalls": [
            "[Numerical] For P1 elements the volume residual "
            "Δu_h vanishes inside elements (u_h is piecewise "
            "affine), so the η_K² volume term is just "
            "h_K² ∫_K f² dx — NOT h_K² ∫_K (f + Δu_h)² dx. "
            "Including the (already-zero) Δu_h term harmlessly, "
            "but for P2 or higher you must reinstate "
            "`laplace.assemble` of the test field to get the "
            "second-derivative residual right. "
            "Signal: error estimator converges with order h "
            "instead of h^2 on smooth solutions; refinement "
            "stalls at a non-zero plateau because the dominant "
            "residual term is missing. Concretely: "
            "`laplace.assemble(ib)` and `unit_load.assemble(ib)` "
            "still return the right system, but the post-solve "
            "η_K computed via `Basis.interpolate(u).grad` will "
            "under-estimate the true error for P2+ bases.",
            "[API] scikit-fem ≥ 8 expects "
            "`MeshTri.refined(element_indices)` for adaptive "
            "refinement; passing a boolean mask of length "
            "n_elements works (it's converted via "
            "`np.where(mask)`), but passing facet indices "
            "raises IndexError because `refined` interprets its "
            "argument as element indices. "
            "Signal: IndexError 'index N is out of bounds for "
            "axis 0 with size M' from `MeshTri.refined`; or "
            "the resulting mesh has fewer elements than "
            "expected because the mask was element-indexed but "
            "the call shape was per-facet.",
            "[API] `Basis.interpolate(u).grad` returns shape "
            "`(spatial_dim, n_elements, n_qpoints)`. For P1 the "
            "gradient is constant per element so taking "
            "`grad_u[:, :, 0]` extracts a per-element gradient "
            "vector cheaply. Confusing the axis order (e.g. "
            "`grad_u[:, 0, :]` for 'gradient of element 0') "
            "produces silently-wrong jump terms. "
            "Signal: η_K values are O(1) where they should be "
            "O(h); refinement targets random elements rather "
            "than the re-entrant corner; "
            "ValueError 'operands could not be broadcast "
            "together' if you try to dot the wrong-shape "
            "tensor with the facet normal.",
            "[Mesh] `m.f2t` has shape (2, n_facets); the second "
            "row is -1 for boundary facets. Iterating over all "
            "facets without filtering `f2t[1] >= 0` adds "
            "spurious boundary-jump contributions that double-"
            "count Dirichlet BC residuals. "
            "Signal: η_K elevated along the boundary even on a "
            "uniform mesh with smooth solution; refinement "
            "preferentially refines boundary elements; "
            "IndexError 'index -1 is out of bounds' if you "
            "dereference `f2t[1, fi]` as an element id without "
            "the >= 0 check.",
            "[Numerical] Dorfler / max-strategy marking with "
            "θ=0.5 is the standard sweet spot — θ → 1 gives "
            "near-uniform refinement (slow convergence per "
            "DOF), θ → 0 gives 1-element-at-a-time refinement "
            "(many iterations needed). For the L-shape "
            "problem, θ ∈ [0.4, 0.6] typically converges in "
            "5-10 iterations. "
            "Signal: refinement runs many iterations without "
            "significant DOF growth and `max_eta` plateaus "
            "instead of shrinking — θ is too small; conversely, "
            "`ib.N` (the Basis dof count) doubles every "
            "iteration but the estimator barely budges — "
            "θ is too large. Practical check: print "
            "`(ib.N, max_eta)` per iteration; healthy adaptive "
            "convergence shows N growing ~1.3-2× per step while "
            "max_eta drops ~30-50%.",
            "[Numerical] L-shape re-entrant corner gives a "
            "u ~ r^(2/3) singularity at (0,0); uniform "
            "refinement converges at H^1 rate 2/3 (sub-optimal), "
            "while h-adaptive refinement recovers the optimal "
            "rate 1. Run `m = m.refined()` (uniform) once "
            "before the adaptive loop to give the estimator "
            "enough elements to discriminate the singular zone. "
            "Signal: H^1-norm convergence rate observed via "
            "`numpy.linalg.norm(grad(u_h) - grad(u_exact))` "
            "stalls at 0.67 instead of approaching 1.0; the "
            "refined-mesh region does not cluster around (0,0).",
        ],
        "references": [
            "scikit-fem examples: ex11 (adaptive Poisson), "
            "ex22 (residual estimator)",
            "Babuška, I. & Rheinboldt, W. (1978) — "
            "'Error estimates for adaptive finite element "
            "computations'",
            "Dörfler, W. (1996) — 'A convergent adaptive "
            "algorithm for Poisson's equation'",
        ],
    },
}
