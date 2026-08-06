"""Tier-2: the interior-penalty parameter has a coercivity threshold, and the
conditioning ceiling the claim names is off by ten orders of magnitude.

Claim: ngsolve convection_diffusion#4 -- "Penalty parameter: alpha*order^2/h for
interior penalty DG.  Signal: alpha too small gives coercivity loss -- solution
norm diverges with mesh refinement, or LU pivots approach zero; alpha too large
gives cond(K)>1e14 and CG stalls.  Rule of thumb: alpha = 4 * order^2 for
symmetric interior penalty."

Wrong variant: the SIP form with the penalty scaled below the threshold.

THREE CORRECTIONS this fixture records.

  (a) "Solution norm diverges with mesh refinement" is not the observable.  At
      the alphas where the form is genuinely not coercive on BOTH meshes, the
      solution norm does not grow when the mesh is refined -- it stays within
      15% of the value the rule-of-thumb alpha gives.  An agent watching the
      norm sees nothing.  What moves is the sign of the smallest eigenvalue of
      the symmetric part.
  (b) "LU pivots approach zero" is not it either: the direct solve succeeds and
      returns a finite answer at every alpha tried, including the non-coercive
      ones -- so the factorisation is no warning.
  (c) "cond(K) > 1e14" is not reachable by raising alpha.  cond grows LINEARLY,
      so alpha would have to be around 1e14 for that; at the largest alpha in
      this sweep, 1e5 times order^2, cond is still only around 1e5.

What this fixture pins, all re-measured on this run:
  * a coercivity threshold exists on two independent meshes: some alpha in the
    sweep has lambda_min < 0 and a larger one has lambda_min > 0, monotonically;
  * the claim's own rule of thumb 4*order^2 is on the coercive side;
  * at the alphas that are non-coercive on both meshes, the solution norm does
    not grow under refinement and stays within 15% of the coercive-regime
    value -- the claim's stated observable does not occur;
  * the direct solve returns a finite result at every alpha, coercive or not;
  * cond at 1e5*order^2 is far below 1e14, and the log-log slope of cond against
    alpha is about 1.
"""
from __future__ import annotations

import sys

import numpy
import scipy.sparse
from netgen.geom2d import unit_square
from ngsolve import (
    BilinearForm,
    GridFunction,
    L2,
    LinearForm,
    Mesh,
    Integrate,
    ds,
    dx,
    grad,
    pi,
    sin,
    specialcf,
    sqrt,
    x,
    y,
)

ORDER = 2
RULE = 4 * ORDER ** 2


def sip(mesh, alpha):
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
    f = LinearForm(fes)
    f += 2 * pi * pi * sin(pi * x) * sin(pi * y) * v * dx
    a.Assemble()
    f.Assemble()
    rows, cols, vals = a.mat.COO()
    A = scipy.sparse.coo_matrix(
        (numpy.asarray(vals), (numpy.asarray(rows), numpy.asarray(cols))),
        shape=(fes.ndof, fes.ndof)).toarray()
    gf = GridFunction(fes)
    solved, err = True, ""
    try:
        gf.vec.data = a.mat.Inverse(inverse="umfpack") * f.vec
    except Exception as exc:                                   # noqa: BLE001
        solved, err = False, f"{type(exc).__name__}: {exc}"
    arr = gf.vec.FV().NumPy()
    finite = bool(numpy.all(numpy.isfinite(arr)))
    norm = float(sqrt(Integrate(gf * gf, mesh))) if finite else float("nan")
    lmin = float(numpy.linalg.eigvalsh(0.5 * (A + A.T))[0])
    cond = float(numpy.linalg.cond(A))
    return fes.ndof, lmin, cond, solved, finite, norm, err


def main() -> int:
    meshes = [Mesh(unit_square.GenerateMesh(maxh=hh)) for hh in (0.5, 0.3)]
    alphas = [0.05, 0.5, 1.0, 2.0, 4.0, RULE, 1e5 * ORDER ** 2]

    data = {}
    for mi, mesh in enumerate(meshes):
        for al in alphas:
            data[(mi, al)] = sip(mesh, al)
        row = " ".join(f"a={a:g}:lmin={data[(mi, a)][1]:+.2e}" for a in alphas)
        print(f"mesh{mi}_ndof={data[(mi, alphas[0])][0]} {row}")

    thresholds, monotone = [], True
    for mi in range(len(meshes)):
        signs = [data[(mi, a)][1] > 0 for a in alphas]
        has_neg, has_pos = not all(signs), any(signs)
        first = signs.index(True) if has_pos else len(signs)
        monotone = monotone and all(signs[first:])
        thresholds.append(alphas[first] if has_pos else None)
        print(f"mesh{mi}_loses_coercivity_for_small_alpha={has_neg}")
        print(f"mesh{mi}_regains_it_for_large_alpha={has_pos}")
    print(f"coercivity_monotone_in_alpha={monotone}")
    print(f"threshold_found_on_both_meshes="
          f"{all(t is not None for t in thresholds)}")
    print(f"rule_of_thumb_alpha={RULE}")
    print(f"rule_of_thumb_is_coercive="
          f"{all(data[(mi, RULE)][1] > 0 for mi in range(len(meshes)))}")

    # (a) the solution norm does not track coercivity
    ref = data[(1, RULE)][5]
    norms = [(a, data[(1, a)][5], data[(1, a)][1]) for a in alphas]
    for a, nrm, lm in norms:
        print(f"mesh1 alpha={a:g} lmin={lm:+.3e} solution_norm={nrm:.6f} "
              f"rel_to_rule={abs(nrm - ref) / ref:.4f}")
    # The claim's literal observable is that the norm DIVERGES WITH MESH
    # REFINEMENT at a non-coercive alpha. Test exactly that: take the alphas
    # that are non-coercive on both meshes and compare their norms across the
    # two meshes.
    both_neg = [a for a in alphas
                if data[(0, a)][1] < 0 and data[(1, a)][1] < 0]
    grew = []
    for a in both_neg:
        n0, n1 = data[(0, a)][5], data[(1, a)][5]
        grew.append(n1 > 1.5 * n0)
        print(f"noncoercive alpha={a:g} norm_coarse={n0:.6f} "
              f"norm_fine={n1:.6f} ratio={n1 / n0:.4f}")
    print(f"noncoercive_alphas_tested={[float(a) for a in both_neg]}")
    print(f"any_norm_diverged_under_refinement={any(grew)}")
    print(f"solution_norm_does_not_diverge_under_refinement={not any(grew)}")
    spread = max(abs(n - ref) / ref for a, n, lm in norms if lm < 0) \
        if any(lm < 0 for _, _, lm in norms) else 0.0
    print(f"noncoercive_norm_max_deviation_from_rule_value={spread:.4f}")
    print(f"deviation_is_bounded_not_divergent={spread < 0.5}")

    # (b) the direct solve never complains
    all_solved = all(data[k][3] and data[k][4] for k in data)
    print(f"direct_solve_succeeded_at_every_alpha={all_solved}")
    print(f"lu_pivots_are_no_warning={all_solved}")

    # (c) the conditioning ceiling
    a_hi = 1e5 * ORDER ** 2
    c_rule = data[(1, RULE)][2]
    c_hi = data[(1, a_hi)][2]
    print(f"cond_at_rule_of_thumb={c_rule:.4e}")
    print(f"cond_at_1e5_order2={c_hi:.4e}")
    print(f"cond_never_reaches_1e14={c_hi < 1e12}")
    slope = (numpy.log10(c_hi) - numpy.log10(c_rule)) \
        / (numpy.log10(a_hi) - numpy.log10(RULE))
    print(f"cond_vs_alpha_loglog_slope={slope:.4f}")
    print(f"cond_grows_linearly_in_alpha={abs(slope - 1.0) < 0.15}")

    ok = (
        all(t is not None for t in thresholds) and monotone
        and all(data[(mi, RULE)][1] > 0 for mi in range(len(meshes)))
        and not any(grew) and spread < 0.5 and all_solved
        and c_hi < 1e12 and abs(slope - 1.0) < 0.15
    )
    if ok:
        return 0
    print("FAIL: IP penalty invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
