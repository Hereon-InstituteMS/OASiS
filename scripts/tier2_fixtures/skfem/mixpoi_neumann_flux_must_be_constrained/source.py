"""Tier-2: an unconstrained boundary flux in mixed Poisson is O(1) wrong.

Claim: skfem mixed_poisson#4 -- the flux boundary condition must be imposed;
forgetting it leaves the boundary flux unconstrained and the post-processed sigma
at that boundary deviates from the prescribed value by O(1).

Wrong variant: assemble the RT0/P0 dual mixed system and solve without touching
the outflow facet DOFs. Right variant: constrain them.

(The claim calls the flux trace the natural condition imposed by a test-side
integral. In this dual mixed form it is the essential one -- the RT0 facet DOF
IS the flux through the facet, so prescribing it is a Dirichlet constraint on
sigma. Either way the observable the Signal names is what is checked here.)

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology site
-- the wrong-variant solve, which leaves the outflow RT0 facet DOFs free, is
replaced by the same condense(K, rhs, x=x, D=outflow) the right variant uses.
The two boundary fluxes then coincide, the O(1) gap collapses to zero and
unconstrained_flux_deviates_by_order_one=True disappears.  Re-run:
T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP0,
    ElementTriRT0,
    FacetBasis,
    LinearForm,
    MeshTri,
    condense,
    solve,
)
from skfem.helpers import div, dot

G_PRESCRIBED = 0.7

MUTATE = os.environ.get("T2_MUTATE") == "1"


@BilinearForm
def flux_mass(s, t, w):
    return dot(s, t)


@BilinearForm
def divergence(s, v, w):
    return div(s) * v


@LinearForm
def source(v, w):
    return -1.0 * v


def facet_flux(fb, sigma) -> float:
    """Mean sigma . n over the facets of fb."""
    val = np.asarray(fb.interpolate(sigma))
    nrm = np.asarray(fb.normals)
    integral = sum((val[i] * nrm[i] * fb.dx).sum() for i in range(val.shape[0]))
    return float(integral / fb.dx.sum())


def main() -> int:
    ok = True
    m = MeshTri().refined(3).with_boundaries({
        "left": lambda x: x[0] < 1e-10,
        "right": lambda x: x[0] > 1.0 - 1e-10,
    })
    bs = Basis(m, ElementTriRT0())
    bu = Basis(m, ElementTriP0())
    A = flux_mass.assemble(bs)
    B = divergence.assemble(bs, bu)
    print(f"rt0_N={bs.N}")
    print(f"p0_N={bu.N}")
    shapes_ok = A.shape == (bs.N, bs.N) and B.shape == (bu.N, bs.N)
    print(f"rt0_and_p0_block_shapes_consistent={shapes_ok}")
    if not shapes_ok:
        print(f"FAIL: block shapes {A.shape!r} / {B.shape!r}", file=sys.stderr)
        ok = False

    K = sp.bmat([[A, B.T], [B, None]], format="csr")
    rhs = np.concatenate([np.zeros(bs.N), source.assemble(bu)])
    fb = FacetBasis(m, ElementTriRT0(), facets=m.boundaries["right"])
    outflow = bs.get_dofs("right").flatten()
    print(f"facet_dof_count_on_right={len(outflow)}")

    # --- RIGHT variant: constrain the outflow facet DOFs ----------------
    x = np.zeros(K.shape[0])
    x[outflow] = G_PRESCRIBED
    u = solve(*condense(K, rhs, x=x, D=outflow))
    recovered = bool(np.allclose(u[outflow], G_PRESCRIBED, atol=1e-10))
    flux_held = facet_flux(fb, u[:bs.N])
    print(f"constrained_dof_values={np.round(u[outflow][:4], 6).tolist()}")
    print(f"constrained_flux_matches_prescribed={recovered}")
    print(f"constrained_mean_flux={flux_held:.6f}")
    if not recovered:
        print(f"FAIL: constrained DOFs came back {u[outflow][:4]!r}, expected "
              f"{G_PRESCRIBED!r}", file=sys.stderr)
        ok = False

    # --- WRONG variant: never constrain the outflow flux ----------------
    # The two runs are compared to each other, not to the DOF value: the RT0
    # facet DOF is the flux THROUGH the facet, so it is not numerically the
    # mean normal component and comparing them would not test anything.
    # PATHOLOGY: solving with the outflow facet DOFs left free.  T2_MUTATE=1
    # applies the documented fix here -- impose the flux BC on those DOFs.
    free = (solve(*condense(K, rhs, x=x, D=outflow)) if MUTATE
            else np.linalg.lstsq(K.toarray(), rhs, rcond=None)[0])
    flux_free = facet_flux(fb, free[:bs.N])
    print(f"unconstrained_mean_flux={flux_free:.6f}")
    gap = abs(flux_free - flux_held)
    deviates = gap > 0.5
    print(f"free_minus_constrained_flux_gap={gap:.6f}")
    print(f"unconstrained_flux_deviates_by_order_one={deviates}")
    if not deviates:
        print(f"FAIL: leaving the outflow DOFs free gave the same boundary flux "
              f"({flux_free!r}) as constraining them ({flux_held!r})",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
