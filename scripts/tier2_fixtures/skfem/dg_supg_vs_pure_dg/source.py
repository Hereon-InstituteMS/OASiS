"""Tier-2: at Pe_h > 5 pure upwind DG rings and does not clear; SUPG-CG damps.

Claim: skfem dg_methods#10 -- SUPG (continuous Galerkin with stabilisation) is
often more stable than pure DG for steady-state advection on a P1 mesh.
"Signal: a pure-ElementTriDG run at Pe_h > 5 on a coarse MeshTri shows ringing
across element faces (amplitude ~10-30% of nominal) that takes ~3 levels of
MeshTri refinement to clear; an equivalent SUPG-CG run with ElementTriP1
BilinearForm damps the oscillation monotonically as h decreases."

Wrong variant: solve the interior-layer advection-diffusion problem with pure
ElementDG(ElementTriP1) upwind DG at Pe_h = 10, then refine three times and
watch the ringing.

Test problem: -eps*lap(u) + b.grad(u) = 0 on the unit square, b at 30 degrees,
inflow datum g = 1 for y > 0.5 on x = 0 and 0 elsewhere, eps = 1/160, so the
exact solution lies in [0, 1] and any excursion outside it is ringing.
Pe_h = |b| h / (2 eps) = 10, 5, 2.5 for n = 8, 16, 32.

Observed on skfem 12.0.1:
  * pure upwind DG  -> excursion 0.083, 0.090, 0.100 -- it does NOT decrease,
    it creeps UP through three levels of refinement
  * SUPG-CG P1      -> 0.041, 0.011, 0.0001 -- monotonically damped, and
    smaller than DG at every single level, using 1/6 of the DOFs
  * unstabilised CG -> 0.063, 0.023, 0.002, i.e. the SUPG term is doing real
    work but is not what makes CG beat DG here.

PARTIAL FALSIFICATION of the numbers: the DG ringing measures 8-10% of the
nominal value, just under the catalog's "~10-30%" band, and it does NOT clear
after ~3 levels of refinement -- the exact solution carries a genuine
discontinuity, so the DG overshoot is h-independent.  The claim's ORDERING
(SUPG-CG damps monotonically, pure DG does not) reproduces exactly.

Mutation control: ``T2_MUTATE=1 python source.py`` applies the documented fix at
the pathology site -- the pure-upwind-DG solve that fills the ``dg`` excursion
list is replaced by the SUPG-CG solve ``cg_run(n, True)``, i.e. the very
substitution the claim tells the user to make.  The ringing then is not there to
be measured and the DG-vs-SUPG contrast collapses, so the fixture goes red.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from skfem import (
    Basis,
    BilinearForm,
    ElementDG,
    ElementTriP1,
    FacetBasis,
    InteriorFacetBasis,
    LinearForm,
    MeshTri,
    asm,
    condense,
    solve,
)
from skfem.helpers import dot, grad, jump
from scipy.sparse.linalg import spsolve

MUTATE = os.environ.get("T2_MUTATE") == "1"

THETA = np.deg2rad(30.0)
B = np.array([np.cos(THETA), np.sin(THETA)])
BMAG = 1.0
EPS = 1.0 / 160.0
SIGMA = 8.0


def mesh(n):
    return MeshTri.init_tensor(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))


def inflow(p):
    return (p[0] < 1e-10) | (p[1] < 1e-10)


def g_inflow(x):
    return np.where((x[0] < 1e-9) & (x[1] > 0.5), 1.0, 0.0)


def dg_run(n):
    m = mesh(n)
    e = ElementDG(ElementTriP1())
    ib = Basis(m, e)
    i0 = InteriorFacetBasis(m, e, side=0)
    i1 = InteriorFacetBasis(m, e, side=1)
    fin = FacetBasis(m, e,
                     facets=m.facets_satisfying(inflow, boundaries_only=True))

    @BilinearForm
    def vol(u, v, w):
        return EPS * dot(grad(u), grad(v)) + (B[0] * u.grad[0] + B[1] * u.grad[1]) * v

    @BilinearForm
    def facet_interior(u, v, w):
        bn = B[0] * w.n[0] + B[1] * w.n[1]
        sv = (-1.0) ** w.idx[1]
        ju, jv = jump(w, u, v)
        upwind = np.minimum(sv * bn, 0.0) * (-sv) * ju * v
        gun = 0.5 * (u.grad[0] * w.n[0] + u.grad[1] * w.n[1])
        gvn = 0.5 * (v.grad[0] * w.n[0] + v.grad[1] * w.n[1])
        return upwind + EPS * (-gun * jv - gvn * ju) + EPS * SIGMA / w.h * ju * jv

    @BilinearForm
    def facet_boundary(u, v, w):
        bn = B[0] * w.n[0] + B[1] * w.n[1]
        gun = u.grad[0] * w.n[0] + u.grad[1] * w.n[1]
        gvn = v.grad[0] * w.n[0] + v.grad[1] * w.n[1]
        return (-np.minimum(bn, 0.0) * u * v
                + EPS * (-gun * v - gvn * u) + EPS * SIGMA / w.h * u * v)

    @LinearForm
    def rhs(v, w):
        bn = B[0] * w.n[0] + B[1] * w.n[1]
        g = g_inflow(w.x)
        gvn = v.grad[0] * w.n[0] + v.grad[1] * w.n[1]
        return (-np.minimum(bn, 0.0) * g * v
                + EPS * (-gvn * g) + EPS * SIGMA / w.h * g * v)

    A = (asm(vol, ib)
         + asm(facet_interior, [i0, i1], [i0, i1])
         + asm(facet_boundary, fin))
    return spsolve(A.tocsr(), asm(rhs, fin)), ib


def cg_run(n, supg):
    m = mesh(n)
    ib = Basis(m, ElementTriP1())
    h = 1.0 / n
    pe = BMAG * h / (2.0 * EPS)
    tau = (h / (2.0 * BMAG)) * (1.0 / np.tanh(pe) - 1.0 / pe) if supg else 0.0

    @BilinearForm
    def a(u, v, w):
        bu = B[0] * u.grad[0] + B[1] * u.grad[1]
        bv = B[0] * v.grad[0] + B[1] * v.grad[1]
        r = EPS * dot(grad(u), grad(v)) + bu * v
        if tau > 0:
            r = r + tau * bu * bv
        return r

    A = a.assemble(ib)
    D = ib.get_dofs(inflow).flatten()
    x0 = np.zeros(A.shape[0])
    x0[D] = g_inflow(ib.doflocs[:, D])
    return solve(*condense(A, np.zeros(A.shape[0]), x=x0, D=D)), ib, pe


def excursion(u):
    """How far outside the physical range [0, 1] the solution goes."""
    return max(float(u.max() - 1.0), float(-u.min()), 0.0)


def main() -> int:
    ok = True
    levels = (8, 16, 32)
    dg, supg, plain, pes, ndof = [], [], [], [], []
    for n in levels:
        # The pathological run.  Under T2_MUTATE the documented fix is applied
        # here: the pure-upwind-DG discretisation is swapped for SUPG-CG.
        if MUTATE:
            _u, _ib, _ = cg_run(n, True)
            u_dg, ib_dg = _u, _ib
        else:
            u_dg, ib_dg = dg_run(n)
        u_su, ib_su, pe = cg_run(n, True)
        u_pl, _, _ = cg_run(n, False)
        dg.append(excursion(u_dg))
        supg.append(excursion(u_su))
        plain.append(excursion(u_pl))
        pes.append(pe)
        ndof.append((ib_dg.N, ib_su.N))

    print(f"peclet_numbers={[round(p, 2) for p in pes]}")
    print(f"coarse_level_pe_h_gt_5={pes[0] > 5.0}")
    print(f"n_refinement_levels={len(levels)}")
    print(f"dg_excursions={[round(x, 4) for x in dg]}")
    print(f"supg_excursions={[round(x, 4) for x in supg]}")
    print(f"plain_cg_excursions={[round(x, 4) for x in plain]}")
    print(f"dofs_dg={[d[0] for d in ndof]} dofs_cg={[d[1] for d in ndof]}")

    # --- WRONG variant: pure DG at Pe_h > 5 -------------------------------
    dg_rings = dg[0] > 0.05
    dg_monotone = all(dg[i + 1] < dg[i] for i in range(len(dg) - 1))
    dg_cleared = dg[-1] < 0.1 * dg[0]
    print(f"dg_rings_at_coarse_level={dg_rings}")
    print(f"dg_damps_monotonically={dg_monotone}")
    print(f"dg_ringing_cleared_after_3_levels={dg_cleared}")
    print(f"dg_ring_amplitude_in_10_to_30_percent_band="
          f"{0.10 <= dg[0] <= 0.30}")
    if not dg_rings:
        print("FAIL: pure DG did not ring at Pe_h > 5", file=sys.stderr)
        ok = False
    if dg_monotone:
        print("FAIL: pure DG damped monotonically -- the contrast is gone",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant: SUPG-CG damps monotonically -----------------------
    supg_monotone = all(supg[i + 1] < supg[i] for i in range(len(supg) - 1))
    supg_better = all(s < d for s, d in zip(supg, dg))
    print(f"supg_damps_monotonically={supg_monotone}")
    print(f"supg_beats_dg_at_every_level={supg_better}")
    print(f"supg_uses_fewer_dofs_than_dg={all(c < g for g, c in ndof)}")
    if not supg_monotone:
        print("FAIL: SUPG-CG did not damp monotonically under refinement",
              file=sys.stderr)
        ok = False
    if not supg_better:
        print("FAIL: SUPG-CG was not more stable than pure DG", file=sys.stderr)
        ok = False

    # --- the SUPG term itself is load bearing -----------------------------
    print(f"supg_beats_unstabilised_cg_at_every_level="
          f"{all(s < p for s, p in zip(supg, plain))}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
