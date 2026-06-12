"""DUNE-fem linear elasticity generators and knowledge."""

from .poisson import _poisson_2d


def _elasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Linear elasticity — DUNE-fem (placeholder)."""
    return _poisson_2d(params)  # Placeholder


KNOWLEDGE = {
    "linear_elasticity": {
        "description": "Linear elasticity with UFL vector spaces",
        "solver": "galerkin scheme with vector Lagrange space",
        "spaces": "lagrange(gridView, dimRange=2, order=k) for 2D vector",
        "pitfalls": [
            (
                "[API] Use dimRange=2 (or 3) for vector-"
                "valued spaces in DUNE-fem. Signal: "
                "lagrange(gridView, order=k) without "
                "dimRange creates a SCALAR space; "
                "assembling a vector elasticity form into "
                "it raises 'TestFunction has wrong rank' "
                "from UFL. The correct pattern is "
                "lagrange(gridView, order=k, dimRange=2) "
                "for 2D elasticity (or 3 for 3D). "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Lame parameters computed from "
                "E and nu: mu = E / (2*(1+nu)), lam = "
                "E*nu / ((1+nu)*(1-2*nu)). Signal: "
                "swapping the formulae for mu and lam in "
                "the dune.fem galerkin scheme + lagrange "
                "space form gives Poisson ratio inverted "
                "in the discrete response — a tension "
                "test produces wrong-sign lateral "
                "contraction. Sanity check: pure shear "
                "should give nu-independent response. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Strain: 0.5*(grad(u) + "
                "grad(u).T); Stress: lam*tr(eps)*I + 2*mu*"
                "eps. Signal: writing strain as grad(u) "
                "(unsymmetrised) couples rotations into "
                "stress — a pure rigid-body rotation "
                "produces non-zero stress. Use "
                "sym(grad(u)) or 0.5*(grad(u) + "
                "grad(u).T) explicitly. (Audit "
                "2026-06-02.)"
            ),
        ],
    },
}

GENERATORS = {
    "linear_elasticity_2d": _elasticity_2d,
}
