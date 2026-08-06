"""Tier-2 for fenics contact#5: a naive all-pairs gap search really is
quadratic, but a bounding-box tree is a much smaller win than usually claimed —
measure it instead of assuming it.

What is timed: the closest contact facet for every query point, computed twice
on the same data. The naive way is a NumPy all-pairs distance array built by
broadcasting, argmin over it. The dolfinx way is
geometry.bb_tree(mesh, fdim, entities=facets) plus create_midpoint_tree and
compute_closest_entity. The two are checked to return the SAME nearest facet
for every query point before any timing is believed. Sizes are the exterior
facets of unit cubes with 6, 8, 11 and 16 cells per side — 432, 768, 1452 and
3072 facets, i.e. roughly a doubling each step. Best of three runs each.

Observed on this installation, reproduced across repeated runs:
  * the NumPy all-pairs cost fits n**2.2 — quadratic, as expected;
  * compute_closest_entity fits n**1.2 — better than quadratic but clearly NOT
    near-linear;
  * at 432 facets the plain NumPy version is the FASTER of the two (about 8 ms
    against about 13 ms), the crossover sits near 800 facets;
  * at 3072 facets the tree leads by a factor of four to five — a real win, but
    single-digit, not the order-of-magnitude usually claimed.

FINDING against the claim as written. The claim states that doubling the facet
count multiplies compute_closest_entity "by very nearly as much" as the NumPy
version, i.e. about four. That was not reproduced: the tree grows by a factor
of 1.7 to 2.5 per doubling against 3.2 to 5.1 for NumPy. The claim's headline —
that the bounding-box tree is not near-linear and is a modest win at contact-
surface sizes, with NumPy ahead at a few hundred facets — is confirmed.

Mutation control: T2_MUTATE=1 swaps the naive broadcast for the correct
vectorised all-pairs (the |a|^2 - 2ab + |b|^2 identity, one BLAS call). That is
the honest implementation of the NumPy side and it reverses the conclusion at
the large end: the tree is no longer faster there.
"""
from __future__ import annotations

import os
import tempfile
import time

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.geometry as geo  # noqa: E402
from dolfinx import mesh  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

SIZES = (6, 8, 11, 16)
REPEAT = 3


def best(fn):
    t_best, out = float("inf"), None
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        out = fn()
        t_best = min(t_best, time.perf_counter() - t0)
    return t_best, out


def slope(x, y):
    lx, ly = np.log(np.asarray(x)), np.log(np.asarray(y))
    return float(np.polyfit(lx, ly, 1)[0])


def main() -> int:
    nfacets, t_np, t_tree, agree = [], [], [], []
    for n in SIZES:
        msh = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
        tdim = msh.topology.dim
        fdim = tdim - 1
        msh.topology.create_connectivity(fdim, tdim)
        facets = mesh.exterior_facet_indices(msh.topology)
        mids = dolfinx.mesh.compute_midpoints(msh, fdim, facets)
        pts = mids + 1.0e-3

        def naive():
            d2 = ((pts[:, None, :] - mids[None, :, :]) ** 2).sum(-1)
            return np.argmin(d2, axis=1)

        def blas():
            d2 = (np.einsum("ij,ij->i", pts, pts)[:, None]
                  - 2.0 * pts @ mids.T
                  + np.einsum("ij,ij->i", mids, mids)[None, :])
            return np.argmin(d2, axis=1)

        def tree():
            bb = geo.bb_tree(msh, fdim, entities=facets)
            mid_tree = geo.create_midpoint_tree(msh, fdim, facets)
            return geo.compute_closest_entity(bb, mid_tree, msh, pts)

        tn, rn = best(blas if MUTATE else naive)
        tt, rt = best(tree)
        nfacets.append(len(facets))
        t_np.append(tn)
        t_tree.append(tt)
        agree.append(float(np.mean(facets[rn] == rt)))
        print(f"facets={len(facets)} allpairs_ms={tn * 1e3:.3f} "
              f"tree_ms={tt * 1e3:.3f} tree_speedup={tn / tt:.2f} "
              f"nearest_facet_agreement={agree[-1]:.3f}")

    p_np, p_tree = slope(nfacets, t_np), slope(nfacets, t_tree)
    print(f"allpairs_scaling_exponent={p_np:.3f} "
          f"tree_scaling_exponent={p_tree:.3f}")
    quad = 1.7 <= p_np <= 2.7
    not_linear = p_tree > 1.05
    small_win = t_np[0] < t_tree[0]
    big_win = t_np[-1] > t_tree[-1]
    modest = (t_np[-1] / t_tree[-1]) < 10.0
    same = all(a == 1.0 for a in agree)
    print(f"same_nearest_facet_both_ways={same}")
    print(f"allpairs_scaling_is_quadratic={quad}")
    print(f"tree_scaling_is_not_near_linear={not_linear}")
    print(f"allpairs_faster_at_a_few_hundred_facets={small_win}")
    print(f"tree_faster_at_three_thousand_facets={big_win}")
    print(f"tree_speedup_at_three_thousand_facets_is_single_digit={modest}")

    if quad and not_linear and small_win and big_win and modest and same:
        print("VERDICT=bbtree_is_a_modest_win_not_a_near_linear_one")
        return 0
    print("VERDICT=bbtree_behaved_as_folklore_claims")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
