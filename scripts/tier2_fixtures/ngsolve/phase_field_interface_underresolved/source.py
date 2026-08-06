"""Tier-2: an unresolved interface width gives 10-30% over/undershoot in c,
and NGSolve's Newton says nothing about it.

Claim: ngsolve phase_field#1 - eps must be resolved by the mesh (several
elements across the interface).  Signal: c develops over/undershoot of
10-30%; or NewtonMinimization diverges with `Newton did not converge after
N iterations`.

Wrong variant: keep the mesh at maxh=0.1 and shrink eps to 0.002, so the
  tanh interface (width ~4*eps = 0.008) is a twelfth of one element.
Right variant : eps=0.05 on the same mesh - the interface spans several
  elements and c stays inside [0, 1].

Observed on NGSolve 6.2.2604, unit_square maxh=0.1 (ne=230), P1,
semi-implicit Allen-Cahn, M=1, dt=1e-3, 20 steps (2026-08-06), min/max of c:
  eps=0.002  h/eps = 50   (-0.2954, +1.2805)   ~28% over/undershoot
  eps=0.010  h/eps = 10   (-0.1586, +1.1273)   ~13-16%
  eps=0.050  h/eps =  2   (-0.0000, +1.0000)   clean

FALSIFIED half of the claim: the Newton escape hatch does not exist.
`Newton did not converge after N iterations` appears nowhere in NGSolve
6.2.2604 (its wording is `Warning: Newton might not converge! Error = `),
and on this deliberately unresolved problem the fully implicit Newton
CONVERGES - flag 0 in 3 iterations - while c overshoots by 28%.  Newton is
silent about under-resolution; only the range of c reveals it.

Mutation control.  T2_MUTATE=1 runs the "underresolved" case at EPS_FINE
instead of EPS_COARSE, i.e. the documented fix of resolving the interface on
the mesh you have (the fixture's own right variant, eps=0.05 at maxh=0.1).  c
then stays inside [0, 1], so `underresolved_overshoot_gt_10pct`,
`underresolved_undershoot_gt_10pct` and `overshoot_grows_with_h_over_eps` print
False and the fixture goes red.  Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys
import tempfile

from netgen.geom2d import unit_square
from ngsolve import (BilinearForm, GridFunction, H1, LinearForm, Mesh, dx,
                     exp, grad, x)
from ngsolve.solvers import Newton

MAXH, MOB, DT, NSTEPS = 0.1, 1.0, 1e-3, 20
EPS_COARSE, EPS_MID, EPS_FINE = 0.002, 0.010, 0.050

# Mutation control: resolve the interface (the documented fix) in the run whose
# over/undershoot is measured.
MUTATE = os.environ.get("T2_MUTATE") == "1"


def w_prime(c):
    return 2 * c * (1 - c) * (1 - 2 * c)


def profile(eps):
    e2 = exp(2 * (x - 0.5) / (2 * eps))
    return 0.5 + 0.5 * (e2 - 1) / (e2 + 1)


def semi_implicit(mesh, fes, eps):
    c, w = fes.TnT()
    lhs = BilinearForm(((1 / DT) * c * w
                        + eps ** 2 * MOB * grad(c) * grad(w)) * dx).Assemble()
    inv = lhs.mat.Inverse(fes.FreeDofs())
    gfc = GridFunction(fes)
    gfc.Set(profile(eps))
    for _ in range(NSTEPS):
        rhs = LinearForm(((1 / DT) * gfc - MOB * w_prime(gfc)) * w * dx)
        rhs.Assemble()
        gfc.vec.data = inv * rhs.vec
    return min(gfc.vec), max(gfc.vec)


def capture_fds(fn):
    with tempfile.TemporaryFile(mode="w+") as tmp:
        so, se = os.dup(1), os.dup(2)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(tmp.fileno(), 1)
            os.dup2(tmp.fileno(), 2)
            value = fn()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(so, 1)
            os.dup2(se, 2)
            os.close(so)
            os.close(se)
        tmp.seek(0)
        return value, tmp.read()


def main() -> int:
    ok = True
    mesh = Mesh(unit_square.GenerateMesh(maxh=MAXH))
    fes = H1(mesh, order=1)
    print(f"mesh_ne={mesh.ne} ndof={fes.ndof}")

    lo_c, hi_c = semi_implicit(mesh, fes, EPS_FINE if MUTATE else EPS_COARSE)
    lo_m, hi_m = semi_implicit(mesh, fes, EPS_MID)
    lo_f, hi_f = semi_implicit(mesh, fes, EPS_FINE)
    print(f"h_over_eps_coarse={MAXH / EPS_COARSE:.0f} "
          f"range=({lo_c:.4f},{hi_c:.4f})")
    print(f"h_over_eps_mid={MAXH / EPS_MID:.0f} range=({lo_m:.4f},{hi_m:.4f})")
    print(f"h_over_eps_fine={MAXH / EPS_FINE:.0f} "
          f"range=({lo_f:.4f},{hi_f:.4f})")

    print(f"underresolved_overshoot_gt_10pct={hi_c - 1.0 > 0.10}")
    print(f"underresolved_undershoot_gt_10pct={-lo_c > 0.10}")
    print(f"underresolved_overshoot_lt_40pct={hi_c - 1.0 < 0.40}")
    if not (hi_c - 1.0 > 0.10 and -lo_c > 0.10):
        print(f"FAIL: the unresolved interface did not over/undershoot "
              f"({lo_c:.4f}, {hi_c:.4f})", file=sys.stderr)
        ok = False

    print("resolved_stays_in_unit_interval="
          f"{lo_f > -1e-3 and hi_f < 1.0 + 1e-3}")
    if not (lo_f > -1e-3 and hi_f < 1.0 + 1e-3):
        print(f"FAIL: the resolved interface also left [0,1] "
              f"({lo_f:.4f}, {hi_f:.4f})", file=sys.stderr)
        ok = False

    mono = (hi_c - 1.0) > (hi_m - 1.0) > (hi_f - 1.0)
    print(f"overshoot_grows_with_h_over_eps={mono}")
    if not mono:
        print("FAIL: the overshoot is not monotone in h/eps",
              file=sys.stderr)
        ok = False

    # ---- the Newton escape hatch the claim describes does not exist -----
    c, w = fes.TnT()
    cold = GridFunction(fes)
    gfn = GridFunction(fes)
    gfn.Set(profile(EPS_COARSE))
    a = BilinearForm(fes)
    a += ((1 / DT) * (c - cold) * w
          + EPS_COARSE ** 2 * MOB * grad(c) * grad(w)
          + MOB * w_prime(c) * w) * dx

    def run_newton():
        out = []
        for _ in range(3):
            cold.vec.data = gfn.vec
            out.append(Newton(a, gfn, printing=True, maxit=25))
        return out

    flags, chatter = capture_fds(run_newton)
    print(f"newton_flags={flags}")
    print(f"newton_on_underresolved_converged={all(f[0] == 0 for f in flags)}")
    print("newton_printed_iteration_banner="
          f"{'Newton iteration' in chatter}")
    print("claimed_string_did_not_converge_after_emitted="
          f"{'did not converge after' in chatter}")
    print("ngsolve_actual_nonconvergence_wording_emitted="
          f"{'Newton might not converge' in chatter}")
    print("newton_chatter_first_two_lines="
          f"{chatter.splitlines()[:2]}")
    if "Newton iteration" not in chatter:
        print("FAIL: NGSolve's Newton printed no iteration banner; the "
              "capture did not work", file=sys.stderr)
        ok = False
    if "did not converge after" in chatter:
        print("FAIL: NGSolve emitted the string the catalog claims; the "
              "falsification no longer holds", file=sys.stderr)
        ok = False
    if not all(f[0] == 0 for f in flags):
        print(f"FAIL: Newton did NOT converge on the unresolved problem "
              f"({flags}); the claim's escape hatch may be real after all",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
