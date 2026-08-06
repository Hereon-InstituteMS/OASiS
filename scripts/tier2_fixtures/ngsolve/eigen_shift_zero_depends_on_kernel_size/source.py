"""Tier-2: shift=0 is not decided by "has a null space" -- a one-dimensional
constant kernel goes straight through, the curl-curl gradient kernel does not.

Claim: ngsolve eigenvalue#0 -- "ArnoldiSolver(a.mat, m.mat, fes.FreeDofs(),
vecs, shift=target) uses shift-and-invert: eigenvalues near 'shift' converge
fastest.  shift=0 fails for operators with the gradient kernel as null space --
raises NgException UmfpackInverse 'matrix is singular'.  For Laplace
eigenproblems with Dirichlet BCs the matrix is positive-definite so shift=0 is
safe."

Wrong variant: shift=0 on the curl-curl operator, whose kernel is the gradients.

REFINEMENT this fixture records.  Both halves of the claim hold as written, but
the boundary between them is finer than "the operator has a null space", which
is how it is easy to read.  The Neumann Laplacian has a null space too -- the
constants, one dimension -- and shift=0 goes through it without complaint,
returning the zero eigenvalue as an ordinary result.  It is the curl-curl
operator's gradient kernel, which grows with the mesh, that makes the shifted
matrix numerically singular.  An agent that generalises "kernel => shift=0
fails" will avoid a shift that is perfectly usable.

What this fixture pins, all re-measured on this run:
  * the Dirichlet Laplacian at shift=0 completes and its lowest eigenvalue
    matches the analytic 2*pi^2 -- the claim's "safe" half, against a closed
    form rather than a reference run;
  * the Neumann Laplacian, which DOES have a null space, also completes at
    shift=0 and returns an eigenvalue at machine zero -- the kernel mode itself;
  * the HCurl curl-curl operator at shift=0 raises NgException carrying
    'UmfpackInverse';
  * the 'matrix is singular' line the claim also quotes does NOT arrive while
    the call is in flight -- UMFPACK writes it through C stdio, flushed at
    process exit, so a capture placed around the call comes back empty. The
    exception is the reliable signal; the warning text is not one to guard on;
  * the same operator with a nonzero shift completes, so the failure is the
    shift and not the operator;
  * the kernel dimensions are counted, showing why the two differ: the Neumann
    kernel is one-dimensional, the curl-curl kernel is a large fraction of the
    space.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy
import scipy.sparse
from netgen.csg import unit_cube
from netgen.geom2d import unit_square
from ngsolve import (
    ArnoldiSolver,
    BilinearForm,
    GridFunction,
    H1,
    HCurl,
    Mesh,
    curl,
    dx,
    grad,
)

NVEC = 8


def _capture_fds(fn):
    o, e = sys.stdout.fileno(), sys.stderr.fileno()
    sys.stdout.flush(); sys.stderr.flush()
    so, se = os.dup(o), os.dup(e)
    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(tmp.fileno(), o); os.dup2(tmp.fileno(), e)
        try:
            r, exc = fn(), None
        except Exception as ex:                                # noqa: BLE001
            r, exc = None, f"{type(ex).__name__}: {ex}"
        sys.stdout.flush(); sys.stderr.flush()
    finally:
        os.dup2(so, o); os.dup2(se, e); os.close(so); os.close(se)
    tmp.seek(0); t = tmp.read().decode("utf-8", "replace"); tmp.close()
    return r, exc, t


def kernel_dim(a, fes):
    rows, cols, vals = a.mat.COO()
    A = scipy.sparse.coo_matrix(
        (numpy.asarray(vals), (numpy.asarray(rows), numpy.asarray(cols))),
        shape=(fes.ndof, fes.ndof)).toarray()
    fd = fes.FreeDofs()
    idx = [i for i in range(fes.ndof) if fd[i]]
    sv = numpy.linalg.svd(A[numpy.ix_(idx, idx)], compute_uv=False)
    return len(idx), int((sv < 1e-10 * sv[0]).sum())


def run(a, m, fes, shift):
    gf = GridFunction(fes, multidim=NVEC)
    vecs = [gf.vecs[i] for i in range(NVEC)]

    def _go():
        return sorted(float(l.real) for l in ArnoldiSolver(
            a.mat, m.mat, fes.FreeDofs(), vecs, shift=shift))

    return _capture_fds(_go)


def laplace(mesh, dirichlet):
    fes = H1(mesh, order=2, dirichlet=dirichlet) if dirichlet \
        else H1(mesh, order=2)
    u, v = fes.TnT()
    a = BilinearForm(fes); a += grad(u) * grad(v) * dx; a.Assemble()
    m = BilinearForm(fes); m += u * v * dx; m.Assemble()
    return fes, a, m


def main() -> int:
    mesh2 = Mesh(unit_square.GenerateMesh(maxh=0.2))

    # ---- Dirichlet Laplacian: no kernel, shift=0 safe -------------------
    fd, ad, md = laplace(mesh2, ".*")
    nd, kd = kernel_dim(ad, fd)
    ev_d, exc_d, out_d = run(ad, md, fd, 0.0)
    exact = 2 * math.pi ** 2
    print(f"dirichlet_free_dofs={nd} kernel_dim={kd}")
    print(f"dirichlet_shift0_raised={exc_d!r}")
    print(f"dirichlet_lowest={ev_d[0] if ev_d else None} exact={exact:.6f}")
    d_ok = exc_d is None and abs(ev_d[0] - exact) / exact < 5e-3
    print(f"dirichlet_shift0_is_safe={exc_d is None}")
    print(f"dirichlet_lowest_matches_2pi2={d_ok}")

    # ---- Neumann Laplacian: 1-D kernel, shift=0 STILL safe --------------
    fn, an, mn = laplace(mesh2, None)
    nn, kn = kernel_dim(an, fn)
    ev_n, exc_n, out_n = run(an, mn, fn, 0.0)
    print(f"neumann_free_dofs={nn} kernel_dim={kn}")
    print(f"neumann_has_a_null_space={kn >= 1}")
    print(f"neumann_shift0_raised={exc_n!r}")
    zero_mode = min(abs(e) for e in ev_n) if ev_n else None
    print(f"neumann_smallest_abs_eigenvalue={zero_mode:.3e}")
    print(f"neumann_shift0_completed_anyway={exc_n is None}")
    print(f"neumann_returned_the_kernel_mode={zero_mode < 1e-10}")

    # ---- HCurl curl-curl: gradient kernel, shift=0 fails ----------------
    mesh3 = Mesh(unit_cube.GenerateMesh(maxh=0.5))
    fc = HCurl(mesh3, order=1, dirichlet=".*")
    u, v = fc.TnT()
    ac = BilinearForm(fc); ac += curl(u) * curl(v) * dx; ac.Assemble()
    mc = BilinearForm(fc); mc += u * v * dx; mc.Assemble()
    nc, kc = kernel_dim(ac, fc)
    print(f"hcurl_free_dofs={nc} kernel_dim={kc}")
    print(f"hcurl_kernel_fraction={kc / nc:.4f}")
    print(f"hcurl_kernel_is_large={kc > 0.2 * nc}")

    ev_c, exc_c, out_c = run(ac, mc, fc, 0.0)
    print(f"hcurl_shift0_raised={exc_c!r}")
    print(f"hcurl_shift0_names_umfpackinverse="
          f"{'UmfpackInverse' in (exc_c or '')}")
    # The UMFPACK "matrix is singular" line does NOT arrive while the call is
    # in flight: UMFPACK writes it through C stdio, which is flushed at process
    # exit, so a capture placed around the call comes back empty. It does reach
    # the process's own output, which is what a runner sees. Recorded because a
    # guard that captures around the call site will miss it.
    print(f"hcurl_shift0_captured_output={out_c.strip()!r}")
    print(f"umfpack_warning_not_captured_around_the_call="
          f"{'matrix is singular' not in out_c.lower()}")
    print(f"exception_is_the_reliable_signal_here="
          f"{'UmfpackInverse' in (exc_c or '')}")

    ev_c2, exc_c2, out_c2 = run(ac, mc, fc, 10.0)
    print(f"hcurl_shift10_raised={exc_c2!r}")
    print(f"hcurl_nonzero_shift_completes={exc_c2 is None}")
    print(f"failure_is_the_shift_not_the_operator="
          f"{exc_c is not None and exc_c2 is None}")

    print(f"kernel_alone_does_not_predict_the_failure="
          f"{kn >= 1 and exc_n is None and exc_c is not None}")

    ok = (
        exc_d is None and d_ok
        and kn >= 1 and exc_n is None and zero_mode < 1e-10
        and kc > 0.2 * nc
        and exc_c is not None and "UmfpackInverse" in exc_c
        and "matrix is singular" not in out_c.lower()
        and exc_c2 is None
    )
    if ok:
        return 0
    print("FAIL: shift=0 kernel invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
