"""Tier-2: a central flux on pure-advection DG strips the interior dissipation
out of the operator and collapses the explicit stability limit -- and the
collapse gets worse under refinement, not better.

Claim: ngsolve dg_methods#4 -- "IfPos(b*n, u, u.Other()) selects the upwind side
for convection.  Signal: using 0.5*(u + u.Other()) (central flux) on a
pure-advection DG problem produces unconditional instability -- the solution
amplitude grows exponentially in time regardless of mesh; upwind via IfPos
restores stability."

Wrong variant: the numerical flux replaced by the central average.

CORRECTION this fixture records.  "Unconditional instability" overstates it, and
an agent that tests for it by running and watching for a blow-up will conclude
the central flux is fine.  On this problem the outflow boundary term leaves a
sliver of dissipation, so the central-flux operator still has all its eigenvalues
in the left half-plane and forward Euler at a small enough step is stable.  What
is really lost is the INTERIOR dissipation: the spectral damping rate collapses
towards zero as h shrinks, and with it the explicit step limit -- a fixed dt that
is stable for upwind on the fine mesh is unstable for central by a growing
factor.  That is the effect worth guarding, and it is a fixed-dt failure, not an
unconditional one.

What this fixture pins, all re-measured on this run:
  * both fluxes give a strictly stable semi-discrete operator -- max Re(lambda)
    is negative for BOTH, so the claim's "unconditional instability" does not
    occur;
  * upwind's damping rate is larger in magnitude than central's on every mesh;
  * the forward-Euler step limit, computed by bisection on
    max|1 + dt*lambda| <= 1, is smaller for central on every mesh;
  * the ratio of those limits GROWS as the mesh is refined -- the discrepancy is
    not a fixed constant, it diverges;
  * marched in time at a step the upwind operator is comfortably stable at, the
    central-flux run's amplitude grows while the upwind run's decays.

Mutation control:  T2_MUTATE=1 applies the documented fix at the pathology site
-- the "central" branch of build() also uses the upwind numerical flux
IfPos(b*n, u, u.Other()) instead of the average 0.5*(u + u.Other()), so both
discretisations are upwind and the interior dissipation is back.  The
expectations upwind_more_damped_on_every_mesh=True,
upwind_admits_larger_dt_on_every_mesh=True,
dt_penalty_grows_under_refinement=True and central_grows_at_this_dt=True then
disappear.  Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy
import scipy.sparse
from netgen.geom2d import unit_square
from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    GridFunction,
    IfPos,
    L2,
    Mesh,
    ds,
    dx,
    exp,
    grad,
    specialcf,
    x,
    y,
)

ORDER = 1
BVEC = (1.0, 0.0)

# Mutation control: under T2_MUTATE=1 the "central" branch uses the upwind flux
# too, so the pathology (the central average) is gone from the run.
MUTATE = os.environ.get("T2_MUTATE") == "1"


def _dense(form, ndof):
    rows, cols, vals = form.mat.COO()
    return scipy.sparse.coo_matrix(
        (numpy.asarray(vals), (numpy.asarray(rows), numpy.asarray(cols))),
        shape=(ndof, ndof)).toarray()


def build(mesh, flux):
    fes = L2(mesh, order=ORDER, dgjumps=True)
    u, v = fes.TnT()
    b = CoefficientFunction(BVEC)
    n = specialcf.normal(2)
    uhat = IfPos(b * n, u, u.Other()) if (flux == "upwind" or MUTATE) \
        else 0.5 * (u + u.Other())
    a = BilinearForm(fes)
    a += -u * (b * grad(v)) * dx
    a += uhat * (b * n) * (v - v.Other()) * dx(skeleton=True)
    a += IfPos(b * n, u, 0) * (b * n) * v * ds(skeleton=True)
    a.Assemble()
    m = BilinearForm(fes)
    m += u * v * dx
    m.Assemble()
    return fes, a, m


def spectrum(mesh, flux):
    fes, a, m = build(mesh, flux)
    L = -numpy.linalg.solve(_dense(m, fes.ndof), _dense(a, fes.ndof))
    return fes.ndof, numpy.linalg.eigvals(L)


def dt_limit(w):
    lo, hi = 1e-9, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if numpy.abs(1.0 + mid * w).max() <= 1.0:
            lo = mid
        else:
            hi = mid
    return lo


def march(mesh, flux, dt, nsteps):
    fes, a, m = build(mesh, flux)
    minv = m.mat.Inverse()
    gfu = GridFunction(fes)
    gfu.Set(exp(-100 * ((x - 0.3) ** 2 + (y - 0.5) ** 2)))
    w = gfu.vec.CreateVector()
    a0 = float(numpy.abs(gfu.vec.FV().NumPy()).max())
    for _ in range(nsteps):
        w.data = a.mat * gfu.vec
        gfu.vec.data -= dt * (minv * w)
        cur = float(numpy.abs(gfu.vec.FV().NumPy()).max())
        if not numpy.isfinite(cur) or cur > 1e30:
            return a0, cur
    return a0, float(numpy.abs(gfu.vec.FV().NumPy()).max())


def main() -> int:
    hs = [0.5, 0.3, 0.2]
    ratios = []
    both_stable = True
    upwind_more_damped = True
    upwind_bigger_dt = True

    for hh in hs:
        mesh = Mesh(unit_square.GenerateMesh(maxh=hh))
        nd_u, w_u = spectrum(mesh, "upwind")
        nd_c, w_c = spectrum(mesh, "central")
        re_u, re_c = float(w_u.real.max()), float(w_c.real.max())
        d_u, d_c = dt_limit(w_u), dt_limit(w_c)
        ratios.append(d_u / d_c)
        both_stable = both_stable and re_u < 0 and re_c < 0
        upwind_more_damped = upwind_more_damped and re_u < re_c
        upwind_bigger_dt = upwind_bigger_dt and d_u > d_c
        print(f"maxh={hh} ndof={nd_u} upwind_maxRe={re_u:+.4e} "
              f"central_maxRe={re_c:+.4e} upwind_dtmax={d_u:.4e} "
              f"central_dtmax={d_c:.4e} ratio={d_u / d_c:.2f}")

    print(f"both_fluxes_have_stable_spectrum={both_stable}")
    print(f"claimed_unconditional_instability_not_observed={both_stable}")
    print(f"upwind_more_damped_on_every_mesh={upwind_more_damped}")
    print(f"upwind_admits_larger_dt_on_every_mesh={upwind_bigger_dt}")
    growing = all(b > a for a, b in zip(ratios, ratios[1:]))
    print(f"dt_limit_ratios={[round(r, 2) for r in ratios]}")
    print(f"dt_penalty_grows_under_refinement={growing}")

    # Fixed-dt march on the finest mesh, at a step the upwind operator handles.
    mesh = Mesh(unit_square.GenerateMesh(maxh=hs[-1]))
    _, w_u = spectrum(mesh, "upwind")
    dt = 0.5 * dt_limit(w_u)
    a0u, a1u = march(mesh, "upwind", dt, 400)
    a0c, a1c = march(mesh, "central", dt, 400)
    print(f"march_dt={dt:.4e}")
    print(f"upwind_amp_start={a0u:.6e} end={a1u:.6e}")
    print(f"central_amp_start={a0c:.6e} end={a1c:.6e}")
    print(f"upwind_decays_at_this_dt={a1u < a0u}")
    print(f"central_grows_at_this_dt={a1c > a0c}")

    ok = (
        both_stable and upwind_more_damped and upwind_bigger_dt
        and growing and a1u < a0u and a1c > a0c
    )
    if ok:
        return 0
    print("FAIL: upwind/central stability invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
