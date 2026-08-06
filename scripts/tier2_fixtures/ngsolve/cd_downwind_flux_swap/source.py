"""Tier-2: swapping the IfPos arguments gives a downwind flux whose operator has
eigenvalues in the RIGHT half-plane -- growth no time step can cure.

Claim: ngsolve convection_diffusion#3 -- "IfPos(b*n, u, u.Other()) for upwind
flux selection.  Signal: swapping the arguments -- IfPos(b*n, u.Other(), u) --
gives DOWNWIND advection that is unconditionally unstable; concentration field
develops oscillations growing geometrically in the advection direction."

Wrong variant: IfPos(b*n, u.Other(), u).

Setup: pure advection with b = (1, 0) on unit_square, L2 order 1 with
dgjumps=True, the semi-discrete operator L = -M^-1 A formed densely so its
spectrum can be read directly.  "Unconditionally unstable" is a statement about
that spectrum, not about one time step: if any eigenvalue has a positive real
part, every explicit AND implicit scheme grows, because the exact solution of
the semi-discrete system already grows.  That is checked here, on three meshes,
so the verdict does not depend on a step size.

What this fixture pins, all re-measured on this run:
  * the upwind operator has every eigenvalue strictly in the left half-plane on
    every mesh -- the reference behaviour;
  * the swapped operator has eigenvalues with POSITIVE real part on every mesh,
    which is what makes it unconditionally unstable rather than merely
    step-limited;
  * the growth rate of the swapped operator is large and grows as the mesh is
    refined, so refining makes it worse;
  * marched in time at a step the upwind operator handles comfortably, the
    upwind amplitude decays and the swapped one grows without bound;
  * the two forms differ only in the order of the IfPos arguments -- the same
    expression is built by a shared helper, so nothing else can account for it.
"""
from __future__ import annotations

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


def _dense(form, n):
    rows, cols, vals = form.mat.COO()
    return scipy.sparse.coo_matrix(
        (numpy.asarray(vals), (numpy.asarray(rows), numpy.asarray(cols))),
        shape=(n, n)).toarray()


def build(mesh, swapped):
    fes = L2(mesh, order=ORDER, dgjumps=True)
    u, v = fes.TnT()
    b = CoefficientFunction(BVEC)
    n = specialcf.normal(2)
    # The ONLY difference between the two variants:
    uhat = IfPos(b * n, u.Other(), u) if swapped else IfPos(b * n, u, u.Other())
    a = BilinearForm(fes)
    a += -u * (b * grad(v)) * dx
    a += uhat * (b * n) * (v - v.Other()) * dx(skeleton=True)
    a += IfPos(b * n, u, 0) * (b * n) * v * ds(skeleton=True)
    a.Assemble()
    m = BilinearForm(fes)
    m += u * v * dx
    m.Assemble()
    return fes, a, m


def spectrum(mesh, swapped):
    fes, a, m = build(mesh, swapped)
    L = -numpy.linalg.solve(_dense(m, fes.ndof), _dense(a, fes.ndof))
    return fes.ndof, numpy.linalg.eigvals(L)


def march(mesh, swapped, dt, nsteps):
    fes, a, m = build(mesh, swapped)
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
    up_ok, sw_bad, rates = True, True, []
    for hh in hs:
        mesh = Mesh(unit_square.GenerateMesh(maxh=hh))
        nd, wu = spectrum(mesh, False)
        _, ws = spectrum(mesh, True)
        ru, rs = float(wu.real.max()), float(ws.real.max())
        rates.append(rs)
        up_ok = up_ok and ru < 0
        sw_bad = sw_bad and rs > 0
        print(f"maxh={hh} ndof={nd} upwind_maxRe={ru:+.4e} "
              f"swapped_maxRe={rs:+.4e} upwind_stable={ru < 0} "
              f"swapped_grows={rs > 0}")

    print(f"upwind_spectrum_in_left_half_plane_everywhere={up_ok}")
    print(f"swapped_has_positive_growth_everywhere={sw_bad}")
    print(f"swapped_growth_rates={[round(r, 3) for r in rates]}")
    worse = all(b > a for a, b in zip(rates, rates[1:]))
    print(f"swapped_growth_rate_increases_under_refinement={worse}")
    print(f"unconditional_because_the_continuous_solution_grows={sw_bad}")

    mesh = Mesh(unit_square.GenerateMesh(maxh=hs[-1]))
    _, wu = spectrum(mesh, False)
    lo, hi = 1e-9, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if numpy.abs(1.0 + mid * wu).max() <= 1.0:
            lo = mid
        else:
            hi = mid
    dt = 0.5 * lo
    a0u, a1u = march(mesh, False, dt, 300)
    a0s, a1s = march(mesh, True, dt, 300)
    print(f"march_dt={dt:.4e}")
    print(f"upwind_amp_start={a0u:.4e} end={a1u:.4e}")
    print(f"swapped_amp_start={a0s:.4e} end={a1s:.4e}")
    print(f"upwind_decays={a1u < a0u}")
    print(f"swapped_grows_without_bound={a1s > 1e3 * a0s}")

    ok = up_ok and sw_bad and worse and a1u < a0u and a1s > 1e3 * a0s
    if ok:
        return 0
    print("FAIL: upwind/downwind invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
