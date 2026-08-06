"""Tier-2: writing the solve straight into the whole vector does not let the
boundary velocity "drift" -- it deletes it on the very first step, in silence.

Claim: ngsolve time_dependent_ns#2 -- "Re-impose Dirichlet BCs after each solve
to fix boundary nodes.  Signal: boundary velocity drifts away from the
prescribed value across time steps; for a lid-driven cavity benchmark the
top-wall velocity slowly diverges from u_lid (the factor-of-2 norm of the BC
step is added back each step instead of overwriting)."

Wrong variant: gfu.vec.data = inv * rhs, the spelling the catalog's own IMEX
template uses -- an Inverse built on FreeDofs writes zero into every constrained
row, so the assignment overwrites the boundary values with zero.

CORRECTION this fixture records.  The advice is right and the consequence is
worse than stated.  This is not a drift: after ONE step the prescribed lid
velocity is entirely gone, and it stays gone.  The boundary error does not creep
up over many steps -- it is at its full value immediately and is flat
thereafter.  An agent told to watch for slow divergence, and sampling every
hundred steps, sees a constant error and may read it as a steady state.

Setup: lid-driven cavity, Taylor-Hood VectorH1 order 2 * H1 order 1, nu = 1e-3,
IMEX with the Stokes part implicit.  Two runs differing only in how the
Dirichlet rows are handled: the naive assignment, and writing the constrained
DOFs back before solving for the free correction.  The boundary error is the L2
norm of (u - u_lid) over the top edge.

What this fixture pins, all re-measured on this run:
  * the correct handling keeps the boundary error at roundoff for every step;
  * the naive assignment has its FULL final error already after one step -- the
    step-1 error and the step-40 error agree to better than a part in a
    thousand, so there is nothing gradual about it;
  * the naive error is not small: it is the same size as the prescribed lid
    velocity's own norm over that edge, i.e. the boundary condition is simply
    absent;
  * nothing is raised or printed by either run -- captured at the
    file-descriptor level, so C++-side output cannot escape the check.

Mutation control.  T2_MUTATE=1 makes the naive branch of march() re-impose as
well -- the `else: gfu.vec.data = inv * rhs` path takes the same
write-the-constrained-DOFs-back-and-solve-the-free-correction route as the good
run, which is the documented fix.  The second run then keeps the lid velocity,
so `naive_loses_the_bc` and `boundary_condition_effectively_absent` both print
False and the fixture goes red.  Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy
from netgen.geom2d import SplineGeometry
from ngsolve import (
    BilinearForm,
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
    ds,
    dx,
    sqrt,
    x,
)

MAXH = 0.15
NU = 1e-3
DT = 0.02
NSTEPS = 40

# Mutation control: apply the documented fix (re-impose the Dirichlet rows) to
# the naive branch too, so the pathology this fixture measures is removed.
MUTATE = os.environ.get("T2_MUTATE") == "1"


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


def march(mesh, reimpose):
    V = VectorH1(mesh, order=2, dirichlet=".*")
    Q = H1(mesh, order=1)
    X = V * Q
    (u, p), (v, q) = X.TnT()
    stokes = (NU * InnerProduct(Grad(u), Grad(v))
              + div(u) * q + div(v) * p) * dx
    mstar = BilinearForm(X)
    mstar += InnerProduct(u, v) * dx + DT * stokes
    mstar.Assemble()

    lid = CoefficientFunction((4 * x * (1 - x), 0))
    gbc = GridFunction(X)
    gbc.components[0].Set(lid, definedon=mesh.Boundaries("top"))
    gfu = GridFunction(X)
    gfu.vec.data = gbc.vec
    w = gfu.components[0]

    inv = mstar.mat.Inverse(X.FreeDofs(), "umfpack")
    free = X.FreeDofs()
    rhs = gfu.vec.CreateVector()
    r = gfu.vec.CreateVector()
    top = ds(definedon=mesh.Boundaries("top"))
    errs = []
    for _ in range(NSTEPS):
        conv = LinearForm(X)
        conv += InnerProduct(Grad(w) * w, v) * dx
        conv.Assemble()
        rhs.data = mstar.mat * gfu.vec - DT * conv.vec
        if reimpose or MUTATE:
            for i in range(X.ndof):
                if not free[i]:
                    gfu.vec[i] = gbc.vec[i]
            r.data = rhs - mstar.mat * gfu.vec
            gfu.vec.data += inv * r
        else:
            gfu.vec.data = inv * rhs
        errs.append(float(sqrt(Integrate((w - lid) * (w - lid) * top, mesh))))
    lid_norm = float(sqrt(Integrate(lid * lid * top, mesh)))
    return errs, lid_norm


def main() -> int:
    g = SplineGeometry()
    g.AddRectangle((0, 0), (1, 1), bcs=["bot", "right", "top", "left"])
    mesh = Mesh(g.GenerateMesh(maxh=MAXH))

    (good, exc_g, out_g) = _capture_fds(lambda: march(mesh, True))
    (bad, exc_b, out_b) = _capture_fds(lambda: march(mesh, False))
    errs_g, lid_norm = good
    errs_b, _ = bad

    print(f"lid_profile_L2_norm_on_top_edge={lid_norm:.6f}")
    print(f"reimposed_error_step1={errs_g[0]:.6e}")
    print(f"reimposed_error_last={errs_g[-1]:.6e}")
    print(f"reimposed_error_max={max(errs_g):.6e}")
    print(f"reimposing_keeps_the_bc={max(errs_g) < 1e-12}")

    print(f"naive_error_step1={errs_b[0]:.6e}")
    print(f"naive_error_last={errs_b[-1]:.6e}")
    print(f"naive_error_max={max(errs_b):.6e}")
    print(f"naive_loses_the_bc={errs_b[-1] > 1e-3}")

    rel_change = abs(errs_b[-1] - errs_b[0]) / max(errs_b[-1], 1e-30)
    print(f"naive_error_relative_change_over_the_run={rel_change:.6e}")
    print(f"naive_error_is_full_after_one_step={rel_change < 1e-3}")
    print(f"naive_failure_is_immediate_not_gradual={rel_change < 1e-3}")

    ratio = errs_b[-1] / lid_norm
    print(f"naive_error_over_lid_norm={ratio:.6f}")
    print(f"boundary_condition_effectively_absent={ratio > 0.5}")

    print(f"good_run_raised={exc_g!r}")
    print(f"bad_run_raised={exc_b!r}")
    print(f"neither_run_printed_anything="
          f"{out_g.strip() == '' and out_b.strip() == ''}")
    print(f"failure_is_silent={exc_b is None and out_b.strip() == ''}")

    ok = (
        max(errs_g) < 1e-12
        and errs_b[-1] > 1e-3
        and rel_change < 1e-3
        and ratio > 0.5
        and exc_b is None and out_b.strip() == ""
    )
    if ok:
        return 0
    print("FAIL: Dirichlet re-imposition invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
