"""Tier-2: the Stokes block matrix really has eigenvalues of both signs, and CG
on it fails without saying so.

Claim: ngsolve stokes#0 -- "Stokes block system is INDEFINITE -- use MinRes or
GMRES, never CG.  CG on the full block matrix has eigenvalues of BOTH signs, so
CG carries no convergence guarantee and can break down; MinRes / GMRES are the
safe choice.  Do NOT expect a loud failure from CG, though."

Wrong variant: CGSolver on the assembled block matrix.

Setup: Stokes on unit_square maxh 0.25, Taylor-Hood VectorH1 order 2 * H1 order
1, no-slip everywhere, pressure pinned at one DOF so the system is nonsingular.
The eigenvalues of the free-DOF block are computed densely, and the same matrix
is then handed to CG with the Jacobi smoother NGSolve's own CreateSmoother
returns.

A nuance the claim's wording invites getting wrong, recorded here: CG is
guarantee-FREE on this matrix, not guaranteed to fail.  With a strong enough
preconditioner it does converge -- measured below with a block preconditioner
built from the velocity Laplacian and the pressure mass matrix.  What is
reliably true is the indefiniteness and the silence of the failure.

What this fixture pins, all re-measured on this run:
  * the free-DOF block has both negative and positive eigenvalues, counted --
    so "indefinite" is arithmetic here, not a description;
  * the negative eigenvalues are not roundoff: the most negative is far from
    zero relative to the largest;
  * CG with the Jacobi smoother exhausts its iteration budget and returns a
    vector containing non-finite entries -- and raises nothing and prints
    nothing, checked at the file-descriptor level;
  * with a block preconditioner the SAME CG converges and matches the direct
    solve, so the matrix is not the whole story and "never CG" overstates it;
  * the direct solve is available throughout as the reference, so "converged"
    is measured against a known answer rather than against a residual.

Mutation control: the pathology is CG driven with nothing but the Jacobi
smoother on the indefinite block matrix.  T2_MUTATE=1 hands that same CG run the
block preconditioner instead (the `prec` argument to CGSolver inside
_cg_jacobi), which is the documented fix, so the run converges in ~47 iterations
to a finite vector and the fixture loses cg_exhausted_its_budget=True and
cg_failure_is_silent=True.  Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy
import scipy.sparse
from netgen.geom2d import unit_square
from ngsolve import (
    BilinearForm,
    BitArray,
    CoefficientFunction,
    GridFunction,
    Grad,
    H1,
    InnerProduct,
    LinearForm,
    Mesh,
    VectorH1,
    div,
    dx,
)
from ngsolve.krylovspace import CGSolver

MAXITER = 400
MUTATE = os.environ.get("T2_MUTATE") == "1"


def _capture_fds(fn):
    out_fd, err_fd = sys.stdout.fileno(), sys.stderr.fileno()
    sys.stdout.flush(); sys.stderr.flush()
    so, se = os.dup(out_fd), os.dup(err_fd)
    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(tmp.fileno(), out_fd); os.dup2(tmp.fileno(), err_fd)
        try:
            res, exc = fn(), None
        except Exception as e:                                 # noqa: BLE001
            res, exc = None, f"{type(e).__name__}: {e}"
        sys.stdout.flush(); sys.stderr.flush()
    finally:
        os.dup2(so, out_fd); os.dup2(se, err_fd); os.close(so); os.close(se)
    tmp.seek(0); text = tmp.read().decode("utf-8", "replace"); tmp.close()
    return res, exc, text


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.25))
    V = VectorH1(mesh, order=2, dirichlet=".*")
    Q = H1(mesh, order=1)
    X = V * Q
    (u, p), (v, q) = X.TnT()
    a = BilinearForm(X)
    a += (InnerProduct(Grad(u), Grad(v)) - div(u) * q - div(v) * p) * dx
    a.Assemble()
    f = LinearForm(X)
    f += CoefficientFunction((0, 1)) * v * dx
    f.Assemble()
    fr = BitArray(X.FreeDofs())
    fr[X.Range(1).start] = False

    rows, cols, vals = a.mat.COO()
    A = scipy.sparse.coo_matrix(
        (numpy.asarray(vals), (numpy.asarray(rows), numpy.asarray(cols))),
        shape=(X.ndof, X.ndof)).toarray()
    idx = [i for i in range(X.ndof) if fr[i]]
    S = 0.5 * (A + A.T)[numpy.ix_(idx, idx)]
    w = numpy.linalg.eigvalsh(S)
    n_neg = int((w < 0).sum())
    n_pos = int((w > 0).sum())
    print(f"free_dofs={len(idx)}")
    print(f"eig_min={w[0]:+.6e} eig_max={w[-1]:+.6e}")
    print(f"negative_eigenvalues={n_neg} positive_eigenvalues={n_pos}")
    print(f"block_matrix_is_indefinite={n_neg > 0 and n_pos > 0}")
    ratio = abs(w[0]) / abs(w[-1])
    print(f"most_negative_over_largest={ratio:.3e}")
    print(f"negative_part_is_not_roundoff={ratio > 1e-6}")

    ref = f.vec.CreateVector()
    ref.data = a.mat.Inverse(fr, inverse="umfpack") * f.vec
    exact = ref.FV().NumPy().copy()

    # The block preconditioner (velocity Laplacian + pressure mass).  Built
    # here rather than further down so the mutation branch can hand it to the
    # Jacobi run below; the unmutated path is unaffected.
    pre_a = BilinearForm(X)
    pre_a += (InnerProduct(Grad(u), Grad(v)) + p * q) * dx
    pre_a.Assemble()
    pre = pre_a.mat.Inverse(fr, inverse="umfpack")

    def _cg_jacobi():
        g = GridFunction(X)
        # MUTATION SITE: the pathology is CG preconditioned by nothing but the
        # Jacobi smoother.  T2_MUTATE=1 applies the documented fix here.
        prec = pre if MUTATE else a.mat.CreateSmoother(fr)
        inv = CGSolver(a.mat, prec, maxiter=MAXITER,
                       tol=1e-10, printrates=False)
        g.vec.data = inv * f.vec
        return inv.iterations, g.vec.FV().NumPy().copy()

    res, exc, out = _capture_fds(_cg_jacobi)
    it_j, arr_j = res if res is not None else (None, None)
    finite_j = bool(numpy.all(numpy.isfinite(arr_j)))
    print(f"cg_jacobi_iterations={it_j}")
    print(f"cg_jacobi_result_finite={finite_j}")
    print(f"cg_jacobi_raised={exc!r}")
    print(f"cg_jacobi_printed={out.strip()!r}")
    print(f"cg_exhausted_its_budget={it_j == MAXITER}")
    print(f"cg_failure_is_silent="
          f"{exc is None and out.strip() == '' and not finite_j}")

    g2 = GridFunction(X)
    inv2 = CGSolver(a.mat, pre, maxiter=MAXITER, tol=1e-10, printrates=False)
    g2.vec.data = inv2 * f.vec
    arr2 = g2.vec.FV().NumPy()
    rel2 = float(numpy.abs(arr2 - exact).max()
                 / max(1e-30, numpy.abs(exact).max()))
    print(f"cg_block_preconditioned_iterations={inv2.iterations}")
    print(f"cg_block_preconditioned_relerr={rel2:.3e}")
    print(f"cg_can_converge_with_a_good_preconditioner="
          f"{inv2.iterations < MAXITER and rel2 < 1e-8}")
    print(f"never_cg_overstates_it="
          f"{inv2.iterations < MAXITER and rel2 < 1e-8}")

    ok = (
        n_neg > 0 and n_pos > 0 and ratio > 1e-6
        and it_j == MAXITER and not finite_j
        and exc is None and out.strip() == ""
        and inv2.iterations < MAXITER and rel2 < 1e-8
    )
    if ok:
        return 0
    print("FAIL: Stokes indefiniteness invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
