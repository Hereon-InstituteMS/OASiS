"""Tier-2: the enclosed-flow Stokes system has exactly one zero singular value,
and NGSolve's direct solver says nothing at all about it.

Claim: ngsolve time_dependent_ns#4 -- "Pressure uniqueness: fix pressure at one
point or use mean-zero constraint (NumberSpace).  Signal: PETSc reports
`KSPSolve: DIVERGED_BREAKDOWN` or near-zero pivot; the pressure field shows a
uniform drift unrelated to the source.  Pin a single DOF or attach a NumberSpace
mean-zero constraint."

Wrong variant: the enclosed-flow Stokes system with no pressure constraint.

CORRECTION this fixture records.  The Signal names a PETSc message.  NGSolve
6.2.2604 does not go through PETSc for this solve and the default umfpack path
emits nothing: no exception, no warning, no diagnostic line on stdout or stderr,
captured at the file-descriptor level so C++-side output cannot escape the
check.  It returns a finite vector.  An agent watching for DIVERGED_BREAKDOWN
sees a clean solve on a singular system.  The defect is visible only in the
matrix, or in the answer -- which is what this fixture measures.

Setup: Stokes on the unit square, Taylor-Hood VectorH1 order 2 * H1 order 1,
no-slip on the whole boundary, so the pressure is determined only up to a
constant.  The singular values of the free-DOF block are computed densely.

What this fixture pins, all re-measured on this run:
  * the unconstrained system has exactly ONE singular value at zero to
    roundoff, and the next one is many orders larger -- a one-dimensional
    nullspace, as the constant-pressure mode should be;
  * that nullspace vector lives entirely in the pressure block and is
    constant there, so it IS the constant-pressure mode and not something else;
  * pinning one pressure DOF removes it and leaves a finite condition number;
  * solving the singular system through the default direct solver produces no
    output at all and no exception -- the claim's PETSc wording appears nowhere;
  * the two solutions differ by a constant in the pressure and agree in the
    velocity, which is the "uniform drift unrelated to the source" the claim
    describes, and the only thing an agent could have noticed.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy
import scipy.sparse
from netgen.geom2d import SplineGeometry
from ngsolve import (
    BilinearForm,
    BitArray,
    CoefficientFunction,
    GridFunction,
    Grad,
    H1,
    InnerProduct,
    Integrate,
    LinearForm,
    Mesh,
    VectorH1,
    div,
    dx,
    y,
)

PETSC_WORDS = ("DIVERGED_BREAKDOWN", "KSPSolve", "zero pivot", "singular")


def _capture_fds(fn):
    out_fd, err_fd = sys.stdout.fileno(), sys.stderr.fileno()
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(out_fd), os.dup(err_fd)
    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(tmp.fileno(), out_fd)
        os.dup2(tmp.fileno(), err_fd)
        try:
            result, exc = fn(), None
        except Exception as e:                                 # noqa: BLE001
            result, exc = None, f"{type(e).__name__}: {e}"
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os.dup2(saved_out, out_fd)
        os.dup2(saved_err, err_fd)
        os.close(saved_out)
        os.close(saved_err)
    tmp.seek(0)
    text = tmp.read().decode("utf-8", "replace")
    tmp.close()
    return result, exc, text


def build(mesh):
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
    return X, a, f


def free_block(X, a, pin):
    fr = BitArray(X.FreeDofs())
    if pin:
        fr[X.Range(1).start] = False
    rows, cols, vals = a.mat.COO()
    A = scipy.sparse.coo_matrix(
        (numpy.asarray(vals), (numpy.asarray(rows), numpy.asarray(cols))),
        shape=(X.ndof, X.ndof)).toarray()
    idx = [i for i in range(X.ndof) if fr[i]]
    return fr, idx, A[numpy.ix_(idx, idx)]


def main() -> int:
    g = SplineGeometry()
    g.AddRectangle((0, 0), (1, 1), bcs=["bot", "right", "top", "left"])
    mesh = Mesh(g.GenerateMesh(maxh=0.25))
    X, a, f = build(mesh)
    pstart = X.Range(1).start

    fr0, idx0, S0 = free_block(X, a, pin=False)
    U, sv0, Vt = numpy.linalg.svd(S0)
    print(f"unconstrained_free_dofs={len(idx0)}")
    print(f"smallest_singular_value={sv0[-1]:.6e}")
    print(f"second_smallest_singular_value={sv0[-2]:.6e}")
    tol = 1e-10 * sv0[0]
    nk = int((sv0 < tol).sum())
    print(f"nullspace_dimension={nk}")
    print(f"exactly_one_zero_mode={nk == 1}")

    # Is that mode the constant pressure?
    vec = Vt[-1]
    vpart = numpy.array([vec[k] for k, i in enumerate(idx0) if i < pstart])
    ppart = numpy.array([vec[k] for k, i in enumerate(idx0) if i >= pstart])
    p_frac = float(ppart @ ppart) / float(vec @ vec)
    spread = float(ppart.std() / max(abs(ppart.mean()), 1e-30))
    print(f"nullmode_pressure_fraction={p_frac:.8f}")
    print(f"nullmode_pressure_relative_spread={spread:.3e}")
    print(f"nullmode_is_the_constant_pressure="
          f"{p_frac > 1 - 1e-8 and spread < 1e-8}")

    fr1, idx1, S1 = free_block(X, a, pin=True)
    sv1 = numpy.linalg.svd(S1, compute_uv=False)
    print(f"pinned_free_dofs={len(idx1)}")
    print(f"pinned_smallest_singular_value={sv1[-1]:.6e}")
    print(f"pinning_one_dof_removes_the_nullspace={sv1[-1] > 1e-10 * sv1[0]}")
    print(f"pinning_removes_exactly_one_dof={len(idx0) - len(idx1) == 1}")

    # --- what does the solver say about the singular system? -----------
    gsing = GridFunction(X)

    def _solve_singular():
        gsing.vec.data = a.mat.Inverse(fr0, inverse="umfpack") * f.vec
        return True

    _, exc, captured = _capture_fds(_solve_singular)
    arr = gsing.vec.FV().NumPy()
    print(f"singular_solve_raised={exc!r}")
    print(f"singular_solve_output_length={len(captured.strip())}")
    hits = [w for w in PETSC_WORDS if w.lower() in captured.lower()]
    print(f"petsc_style_words_found={hits}")
    print(f"solver_said_nothing={exc is None and captured.strip() == ''}")
    print(f"singular_solve_returned_finite={bool(numpy.all(numpy.isfinite(arr)))}")

    gpin = GridFunction(X)
    gpin.vec.data = a.mat.Inverse(fr1, inverse="umfpack") * f.vec

    du = float(numpy.sqrt(Integrate(
        InnerProduct(gsing.components[0] - gpin.components[0],
                     gsing.components[0] - gpin.components[0]), mesh)))
    dp = gsing.components[1] - gpin.components[1]
    mean_dp = float(Integrate(dp, mesh))
    var_dp = float(Integrate((dp - mean_dp) ** 2, mesh))
    print(f"velocity_difference_L2={du:.6e}")
    print(f"pressure_difference_mean={mean_dp:.6e}")
    print(f"pressure_difference_variance={var_dp:.6e}")
    print(f"velocities_agree={du < 1e-10}")
    print(f"pressures_differ_by_a_constant="
          f"{abs(mean_dp) > 1e-8 and var_dp < 1e-16}")

    ok = (
        nk == 1
        and p_frac > 1 - 1e-8 and spread < 1e-8
        and sv1[-1] > 1e-10 * sv1[0]
        and len(idx0) - len(idx1) == 1
        and exc is None and captured.strip() == ""
        and bool(numpy.all(numpy.isfinite(arr)))
        and du < 1e-10
        and abs(mean_dp) > 1e-8 and var_dp < 1e-16
    )
    if ok:
        return 0
    print("FAIL: pressure-nullspace invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
