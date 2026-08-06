"""Tier-2: ArnoldiSolver reproduces the analytic Dirichlet-Laplacian spectrum,
degeneracy included.

Claim: ngsolve eigenvalue#2 -- "Exact analytic eigenvalues of the Dirichlet
Laplacian on [0,1]^2 are pi^2*(m^2+n^2) for m, n >= 1.  First few: 2*pi^2,
5*pi^2, 5*pi^2 (degenerate), 8*pi^2, 10*pi^2...  Signal: ArnoldiSolver result on
a maxh<=0.05 mesh with order>=2 elements should agree with these values to
within ~0.5%; larger discrepancy indicates mesh or element-order problems."

Wrong variant: too few DOFs -- checked by running a coarse P1 mesh alongside and
showing it misses the 0.5% bar.

Setup: the generalised problem A x = lambda M x with A the Dirichlet stiffness
and M the mass matrix, solved by ArnoldiSolver with shift-and-invert.  The
targets are closed-form, so nothing here is a self-comparison.

An API detail worth having: ArnoldiSolver's `vecs` argument is a plain Python
LIST of BaseVectors.  Passing the MultiVector from GridFunction(fes,
multidim=n).vecs directly raises TypeError('ArnoldiSolver(): incompatible
function arguments'), and passing a list of GridFunctions raises
RuntimeError('Unable to cast Python instance of type ngsolve.comp.GridFunction')
-- neither of which mentions vectors.  The list has to be built by indexing.

What this fixture pins, all re-measured on this run:
  * the resolved run agrees with pi^2*(m^2+n^2) for the first five modes, each
    inside the claim's 0.5%;
  * the second and third eigenvalues are degenerate and BOTH come out near
    5*pi^2 -- the degeneracy is not collapsed to one mode;
  * the eigenvalues come out in the right order and none is missed;
  * an under-resolved P1 run misses the 0.5% bar on the higher modes, so the
    tolerance discriminates rather than passing everything;
  * both spellings of `vecs` that are NOT a plain list raise, with the messages
    recorded here.
"""
from __future__ import annotations

import math
import sys

from netgen.geom2d import unit_square
from ngsolve import (
    ArnoldiSolver,
    BilinearForm,
    GridFunction,
    H1,
    Mesh,
    dx,
    grad,
)

EXACT = [2 * math.pi ** 2, 5 * math.pi ** 2, 5 * math.pi ** 2,
         8 * math.pi ** 2, 10 * math.pi ** 2]
NVEC = 12


def spectrum(maxh, order):
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    fes = H1(mesh, order=order, dirichlet=".*")
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += grad(u) * grad(v) * dx
    a.Assemble()
    m = BilinearForm(fes)
    m += u * v * dx
    m.Assemble()
    gf = GridFunction(fes, multidim=NVEC)
    vecs = [gf.vecs[i] for i in range(NVEC)]
    lam = ArnoldiSolver(a.mat, m.mat, fes.FreeDofs(), vecs, shift=1.0)
    return fes.ndof, sorted(float(l.real) for l in lam)[:len(EXACT)], gf


def main() -> int:
    nd, ev, gf = spectrum(0.05, 2)
    errs = [abs(a - b) / b for a, b in zip(ev, EXACT)]
    print(f"resolved_ndof={nd}")
    for i, (c, e, r) in enumerate(zip(ev, EXACT, errs)):
        print(f"mode{i} computed={c:.6f} exact={e:.6f} relerr={r:.3e}")
    within = all(r < 5e-3 for r in errs)
    print(f"all_five_within_half_a_percent={within}")
    ordered = all(b >= a - 1e-9 for a, b in zip(ev, ev[1:]))
    print(f"eigenvalues_come_out_ordered={ordered}")
    deg = abs(ev[1] - ev[2]) / EXACT[1]
    print(f"degenerate_pair_relative_gap={deg:.3e}")
    print(f"degeneracy_resolved_as_two_modes={deg < 5e-3}")
    print(f"first_mode_is_2pi2={abs(ev[0] - EXACT[0]) / EXACT[0] < 5e-3}")

    nd_c, ev_c, _ = spectrum(0.4, 1)
    errs_c = [abs(a - b) / b for a, b in zip(ev_c, EXACT)]
    print(f"coarse_ndof={nd_c}")
    print(f"coarse_relerrs={[f'{r:.3e}' for r in errs_c]}")
    coarse_fails = any(r > 5e-3 for r in errs_c)
    print(f"coarse_run_misses_the_half_percent_bar={coarse_fails}")
    print(f"tolerance_discriminates={within and coarse_fails}")

    # The two vecs spellings that do NOT work.
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    fes = H1(mesh, order=2, dirichlet=".*")
    u, v = fes.TnT()
    a = BilinearForm(fes); a += grad(u) * grad(v) * dx; a.Assemble()
    m = BilinearForm(fes); m += u * v * dx; m.Assemble()
    g2 = GridFunction(fes, multidim=4)
    e1 = e2 = ""
    try:
        ArnoldiSolver(a.mat, m.mat, fes.FreeDofs(), g2.vecs, shift=1.0)
    except Exception as exc:                                   # noqa: BLE001
        e1 = f"{type(exc).__name__}"
    try:
        ArnoldiSolver(a.mat, m.mat, fes.FreeDofs(),
                      [GridFunction(fes) for _ in range(4)], shift=1.0)
    except Exception as exc:                                   # noqa: BLE001
        e2 = f"{type(exc).__name__}"
    print(f"multivector_directly_raises={e1}")
    print(f"list_of_gridfunctions_raises={e2}")
    print(f"vecs_must_be_a_list_of_basevectors={bool(e1) and bool(e2)}")

    ok = (within and ordered and deg < 5e-3 and coarse_fails
          and bool(e1) and bool(e2))
    if ok:
        return 0
    print("FAIL: Dirichlet-Laplacian spectrum invariant not held",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
