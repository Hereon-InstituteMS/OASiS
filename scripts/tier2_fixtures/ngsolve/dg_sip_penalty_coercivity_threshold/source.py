"""Tier-2: the SIP penalty has a coercivity threshold -- below it the DG
stiffness matrix stops being positive definite -- and a large penalty costs
conditioning only linearly, nowhere near the ceiling the claim names.

Claim: ngsolve dg_methods#3 -- "Penalty parameter: alpha * order^2 / h.  Signal:
alpha too small (< order^2) gives coercivity loss -- discrete solution norm grows
under refinement instead of converging; alpha too large (> 100 * order^2) gives
cond(K) > 1e14 and CG/GMRES stagnate.  Rule of thumb: alpha = 4 * (order + 1)^2
for SIP DG."

Wrong variant: assemble the same SIP form with the penalty scaled down until
coercivity is lost.

TWO CORRECTIONS this fixture records.

  (a) The stated OBSERVABLE for small alpha is wrong.  The discrete solution
      norm does NOT grow under refinement; on this problem it stays put near the
      correct value while the error wanders erratically.  What actually changes,
      cleanly and mesh-robustly, is the sign of the smallest eigenvalue of the
      symmetric part: the form stops being coercive.  An agent watching the
      solution norm as the claim instructs sees nothing.

  (b) The stated CEILING for large alpha is wrong by orders of magnitude.
      cond(K) grows only LINEARLY in alpha here, so alpha = 100*order^2 -- the
      value the claim says produces cond > 1e14 -- leaves the matrix
      comfortably solvable, and even 1000x that stays far below 1e14.  The
      fixture measures the growth exponent instead of pinning a threshold.

What this fixture pins, all re-measured on this run:
  * there IS a threshold: some alpha in the swept range has lambda_min < 0 and
    some larger alpha has lambda_min > 0, on two independent meshes;
  * the threshold is monotone -- once coercive, larger alpha stays coercive;
  * the claim's own rule of thumb 4*(order+1)^2 lands on the coercive side;
  * cond(K) at alpha = 100*order^2 is nowhere near 1e14, and the log-log slope
    of cond against alpha in the large-alpha regime is ~1, i.e. linear.
"""
from __future__ import annotations

import sys

import numpy
import scipy.sparse
from netgen.geom2d import unit_square
from ngsolve import (
    BilinearForm,
    L2,
    Mesh,
    ds,
    dx,
    grad,
    specialcf,
)

ORDER = 2
RULE_OF_THUMB = 4 * (ORDER + 1) ** 2


def sip_matrix(mesh, alpha):
    fes = L2(mesh, order=ORDER, dgjumps=True)
    u, v = fes.TnT()
    n = specialcf.normal(2)
    h = specialcf.mesh_size
    ju, jv = u - u.Other(), v - v.Other()
    mdu = 0.5 * (grad(u) + grad(u.Other()))
    mdv = 0.5 * (grad(v) + grad(v.Other()))
    a = BilinearForm(fes)
    a += grad(u) * grad(v) * dx
    a += alpha * ORDER ** 2 / h * ju * jv * dx(skeleton=True)
    a += (-mdu * n * jv - mdv * n * ju) * dx(skeleton=True)
    a += alpha * ORDER ** 2 / h * u * v * ds(skeleton=True)
    a += (-grad(u) * n * v - grad(v) * n * u) * ds(skeleton=True)
    a.Assemble()
    rows, cols, vals = a.mat.COO()
    A = scipy.sparse.coo_matrix(
        (numpy.asarray(vals), (numpy.asarray(rows), numpy.asarray(cols))),
        shape=(fes.ndof, fes.ndof)).toarray()
    return fes.ndof, A


def main() -> int:
    meshes = [Mesh(unit_square.GenerateMesh(maxh=hh)) for hh in (0.5, 0.3)]
    alphas = [0.05, 0.5, 1.0, 2.0, 4.0, RULE_OF_THUMB, 100.0 * ORDER ** 2,
              100000.0 * ORDER ** 2]

    lmin = {}
    cond = {}
    ndofs = []
    for mi, mesh in enumerate(meshes):
        for alpha in alphas:
            nd, A = sip_matrix(mesh, alpha)
            if alpha == alphas[0]:
                ndofs.append(nd)
            w = numpy.linalg.eigvalsh(0.5 * (A + A.T))
            lmin[(mi, alpha)] = float(w[0])
            cond[(mi, alpha)] = float(numpy.linalg.cond(A))
        row = " ".join(
            f"a={a:g}:lmin={lmin[(mi, a)]:+.3e}" for a in alphas)
        print(f"mesh{mi}_ndof={ndofs[mi]} {row}")

    # (1) A threshold exists on BOTH meshes: some alpha is non-coercive, a
    #     larger one is coercive.
    thresholds = []
    monotone = True
    for mi in range(len(meshes)):
        signs = [lmin[(mi, a)] > 0 for a in alphas]
        has_neg = not all(signs)
        has_pos = any(signs)
        # monotone means: once True, never False again
        first_true = signs.index(True) if has_pos else len(signs)
        monotone = monotone and all(signs[first_true:])
        thresholds.append(alphas[first_true] if has_pos else None)
        print(f"mesh{mi}_loses_coercivity_for_small_alpha={has_neg}")
        print(f"mesh{mi}_regains_coercivity_for_large_alpha={has_pos}")
        print(f"mesh{mi}_smallest_coercive_alpha_in_sweep={thresholds[mi]}")
    print(f"coercivity_is_monotone_in_alpha={monotone}")
    threshold_on_both = all(t is not None for t in thresholds)
    print(f"threshold_found_on_both_meshes={threshold_on_both}")

    # (2) The claim's own rule of thumb is on the safe side, on both meshes.
    rot_ok = all(lmin[(mi, RULE_OF_THUMB)] > 0 for mi in range(len(meshes)))
    print(f"rule_of_thumb_alpha={RULE_OF_THUMB}")
    print(f"rule_of_thumb_is_coercive={rot_ok}")

    # (3) The large-alpha conditioning ceiling in the claim is not reached.
    a_hi = 100.0 * ORDER ** 2
    a_hier = 100000.0 * ORDER ** 2
    c_hi = cond[(1, a_hi)]
    c_hier = cond[(1, a_hier)]
    print(f"cond_at_100_order2={c_hi:.4e}")
    print(f"cond_at_1e5_order2={c_hier:.4e}")
    print(f"cond_at_100_order2_far_below_1e14={c_hi < 1e12}")
    slope = (numpy.log10(c_hier) - numpy.log10(c_hi)) \
        / (numpy.log10(a_hier) - numpy.log10(a_hi))
    print(f"cond_vs_alpha_loglog_slope={slope:.4f}")
    print(f"cond_grows_linearly_in_alpha={abs(slope - 1.0) < 0.15}")

    ok = (
        threshold_on_both and monotone and rot_ok
        and c_hi < 1e12 and abs(slope - 1.0) < 0.15
    )
    if ok:
        return 0
    print("FAIL: SIP penalty coercivity invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
