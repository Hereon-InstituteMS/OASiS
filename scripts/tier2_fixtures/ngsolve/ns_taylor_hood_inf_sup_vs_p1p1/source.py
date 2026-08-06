"""Tier-2: the discrete inf-sup constant separates Taylor-Hood from equal-order
P1/P1, and P1/P1's spurious pressure kernel is bigger than the constant mode.

Claim: ngsolve time_dependent_ns#5 -- "Taylor-Hood P2/P1 satisfies inf-sup;
P1/P1 does not (needs stabilization like MINI).  Signal: P1/P1 H1 spaces without
stabilisation produce checkerboard pressure pattern visible in the GridFunction
output, with magnitude that does not converge under refinement; switching to
P2/P1 H1 spaces or adding the cubic bubble (MINI) inside the BilinearForm
removes the checkerboard."

Wrong variant: equal-order VectorH1 order 1 velocity with H1 order 1 pressure.

The observable the claim names -- "checkerboard pattern visible in the output"
-- is an eyeball test, and "magnitude that does not converge under refinement"
needs a magnitude to compare.  This fixture measures the thing that causes both:
the discrete inf-sup constant

    beta_h = min over pressures q of  sup over velocities v of
             (div v, q) / (|v|_1 |q|_0)

computed as the square root of the smallest nonzero eigenvalue of
Mq^-1 B K^-1 B^T, where B is the divergence block and K the velocity stiffness.
An inf-sup stable pair has beta_h bounded away from zero as h shrinks; an
unstable pair does not.  Both are read off the same assembled matrices, so the
comparison holds on any host.

What this fixture pins, all re-measured on this run:
  * on three meshes, Taylor-Hood's kernel of B^T is exactly one-dimensional --
    the constant pressure, which every enclosed-flow discretisation has;
  * P1/P1's kernel is strictly larger on every mesh: there are spurious
    pressure modes beyond the constant, and they are what the checkerboard is;
  * Taylor-Hood's inf-sup constant is essentially the same on all three meshes,
    i.e. bounded away from zero;
  * P1/P1's is smaller than Taylor-Hood's by more than a factor of five on
    every mesh, and unlike Taylor-Hood's it does not settle.
"""
from __future__ import annotations

import sys

import numpy
import scipy.sparse
from netgen.geom2d import SplineGeometry
from ngsolve import (
    BilinearForm,
    Grad,
    H1,
    InnerProduct,
    Mesh,
    VectorH1,
    div,
    dx,
)

MESHES = (0.3, 0.2, 0.15)


def _dense(form, n):
    rows, cols, vals = form.mat.COO()
    return scipy.sparse.coo_matrix(
        (numpy.asarray(vals), (numpy.asarray(rows), numpy.asarray(cols))),
        shape=(n, n)).toarray()


def analyse(mesh, vorder):
    V = VectorH1(mesh, order=vorder, dirichlet=".*")
    Q = H1(mesh, order=1)
    X = V * Q
    (u, p), (v, q) = X.TnT()
    a = BilinearForm(X)
    a += (InnerProduct(Grad(u), Grad(v)) - div(u) * q - div(v) * p) * dx
    a.Assemble()
    m = BilinearForm(X)
    m += p * q * dx
    m.Assemble()

    A = _dense(a, X.ndof)
    Mfull = _dense(m, X.ndof)
    fd = X.FreeDofs()
    off = X.Range(1).start
    vi = [i for i in range(off) if fd[i]]
    pi = [i for i in range(off, X.ndof) if fd[i]]
    B = -A[numpy.ix_(pi, vi)]
    K = A[numpy.ix_(vi, vi)]
    Mq = Mfull[numpy.ix_(pi, pi)]

    S = B @ numpy.linalg.solve(K, B.T)
    L = numpy.linalg.cholesky(Mq)
    T = numpy.linalg.solve(L, numpy.linalg.solve(L, S).T).T
    w = numpy.sort(numpy.linalg.eigvalsh(0.5 * (T + T.T)))
    tol = 1e-9 * w.max()
    kernel = int((w < tol).sum())
    nz = w[w >= tol]
    return len(vi), len(pi), kernel, float(numpy.sqrt(max(nz.min(), 0.0)))


def main() -> int:
    th, p1 = [], []
    for hh in MESHES:
        mesh = Mesh(SplineGeometry_rect(hh))
        nv2, npq2, k2, b2 = analyse(mesh, 2)
        nv1, npq1, k1, b1 = analyse(mesh, 1)
        th.append((hh, nv2, npq2, k2, b2))
        p1.append((hh, nv1, npq1, k1, b1))
        print(f"maxh={hh} TH   vel_dofs={nv2} p_dofs={npq2} "
              f"kernel_dim={k2} inf_sup_beta={b2:.6f}")
        print(f"maxh={hh} P1P1 vel_dofs={nv1} p_dofs={npq1} "
              f"kernel_dim={k1} inf_sup_beta={b1:.6f}")

    th_kernel_one = all(r[3] == 1 for r in th)
    p1_kernel_bigger = all(b[3] > a[3] for a, b in zip(th, p1))
    print(f"taylor_hood_kernel_is_only_the_constant={th_kernel_one}")
    print(f"p1p1_has_spurious_pressure_modes={p1_kernel_bigger}")
    print(f"p1p1_extra_modes={[b[3] - a[3] for a, b in zip(th, p1)]}")

    betas_th = [r[4] for r in th]
    betas_p1 = [r[4] for r in p1]
    th_spread = (max(betas_th) - min(betas_th)) / max(betas_th)
    print(f"taylor_hood_betas={[round(b, 6) for b in betas_th]}")
    print(f"p1p1_betas={[round(b, 6) for b in betas_p1]}")
    print(f"taylor_hood_beta_relative_spread={th_spread:.6f}")
    print(f"taylor_hood_beta_stable_under_refinement={th_spread < 0.05}")
    ratio_ok = all(a > 5.0 * b for a, b in zip(betas_th, betas_p1))
    print(f"taylor_hood_beta_over_p1p1={[round(a / b, 2) for a, b in zip(betas_th, betas_p1)]}")
    print(f"taylor_hood_beta_beats_p1p1_by_5x_everywhere={ratio_ok}")
    p1_settles = (max(betas_p1) - min(betas_p1)) / max(betas_p1) < 0.05
    print(f"p1p1_beta_relative_spread="
          f"{(max(betas_p1) - min(betas_p1)) / max(betas_p1):.6f}")
    print(f"p1p1_beta_does_not_settle={not p1_settles}")

    ok = (th_kernel_one and p1_kernel_bigger and th_spread < 0.05
          and ratio_ok and not p1_settles)
    if ok:
        return 0
    print("FAIL: inf-sup invariant not held", file=sys.stderr)
    return 2


def SplineGeometry_rect(maxh):
    g = SplineGeometry()
    g.AddRectangle((0, 0), (1, 1), bcs=["bot", "right", "top", "left"])
    return g.GenerateMesh(maxh=maxh)


if __name__ == "__main__":
    sys.exit(main())
