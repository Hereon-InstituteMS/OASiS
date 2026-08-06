"""Tier-2: skipping f.vec -= a.mat * gfu.vec leaves the boundary values in place
and the interior wrong -- and nothing complains.

Claim: ngsolve poisson#2 -- "Dirichlet inhomogeneous values on NGSolve:
construct gfu = GridFunction(fes); call gfu.Set(boundary_cf,
definedon=mesh.Boundaries(name)) to set the boundary values, then modify the RHS
as f.vec -= a.mat * gfu.vec before calling Inverse on FreeDofs.  Skipping the
RHS modification leaves the system inconsistent."

Wrong variant: solving on FreeDofs with the unmodified right-hand side.

Setup: Laplace on the unit square, H1 order 2, inhomogeneous Dirichlet data
g = x^2 - y on the whole boundary, zero source.  The lifted and unlifted routes
differ only in whether a.mat * gfu.vec is subtracted before the solve.

What this fixture pins, all re-measured on this run:
  * BOTH routes reproduce the boundary data exactly -- 6.5e-16 -- so a boundary
    check cannot tell them apart, which is what makes the mistake survivable;
  * the lifted route's residual on the free DOFs is at roundoff;
  * the unlifted route's residual on the same DOFs is O(1) -- the system it
    solved is not the one that was posed;
  * the two interior solutions differ by far more than any tolerance, so it is
    not a small perturbation;
  * neither route raises or prints anything, checked at the file-descriptor
    level so C++ output cannot escape.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy
from netgen.geom2d import unit_square
from ngsolve import (
    BilinearForm,
    GridFunction,
    H1,
    Integrate,
    LinearForm,
    Mesh,
    ds,
    dx,
    grad,
    sqrt,
    x,
    y,
)


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


def solve(mesh, lift):
    fes = H1(mesh, order=2, dirichlet=".*")
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += grad(u) * grad(v) * dx
    a.Assemble()
    f = LinearForm(fes)
    f += 0 * v * dx
    f.Assemble()
    gD = x * x - y
    gf = GridFunction(fes)
    gf.Set(gD, definedon=mesh.Boundaries(".*"))
    r = f.vec.CreateVector()
    r.data = f.vec - a.mat * gf.vec if lift else f.vec
    gf.vec.data += a.mat.Inverse(fes.FreeDofs(), inverse="umfpack") * r

    berr = float(sqrt(Integrate((gf - gD) ** 2 * ds, mesh)))
    res = gf.vec.CreateVector()
    res.data = a.mat * gf.vec - f.vec
    free = fes.FreeDofs()
    rn = float(numpy.linalg.norm(
        [res[i] for i in range(fes.ndof) if free[i]]))
    return berr, rn, gf.vec.FV().NumPy().copy()


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
    (good, exc_g, out_g) = _capture_fds(lambda: solve(mesh, True))
    (bad, exc_b, out_b) = _capture_fds(lambda: solve(mesh, False))
    b_g, r_g, v_g = good
    b_b, r_b, v_b = bad

    print(f"lifted_boundary_error={b_g:.4e}")
    print(f"unlifted_boundary_error={b_b:.4e}")
    print(f"both_match_the_boundary_data={b_g < 1e-12 and b_b < 1e-12}")
    print(f"boundary_check_cannot_tell_them_apart="
          f"{abs(b_g - b_b) < 1e-12}")

    print(f"lifted_free_dof_residual={r_g:.6e}")
    print(f"unlifted_free_dof_residual={r_b:.6e}")
    print(f"lifted_residual_at_roundoff={r_g < 1e-10}")
    print(f"unlifted_residual_is_order_one={r_b > 1e-2}")

    diff = float(numpy.abs(v_g - v_b).max())
    scale = float(numpy.abs(v_g).max())
    print(f"solutions_differ_by={diff:.6e} on a scale of {scale:.6e}")
    print(f"interior_solutions_differ={diff > 0.01 * scale}")

    print(f"lifted_raised={exc_g!r}")
    print(f"unlifted_raised={exc_b!r}")
    print(f"neither_printed_anything="
          f"{out_g.strip() == '' and out_b.strip() == ''}")
    print(f"failure_is_silent={exc_b is None and out_b.strip() == ''}")

    ok = (b_g < 1e-12 and b_b < 1e-12 and r_g < 1e-10 and r_b > 1e-2
          and diff > 0.01 * scale and exc_b is None and out_b.strip() == "")
    if ok:
        return 0
    print("FAIL: Dirichlet lifting invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
