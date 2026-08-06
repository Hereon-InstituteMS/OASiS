"""Tier-2: the enclosed-flow Stokes pressure nullspace does not announce itself,
and an open flow with a traction-free outlet has no nullspace at all.

Claim: ngsolve stokes#2 -- "Enclosed-flow Stokes admits the constant pressure
null space -- pin pressure at one node or add a NumberSpace Lagrange multiplier
enforcing mean(p) = 0.  Open flows with a do-nothing (traction-free) outlet
determine pressure uniquely.  The singular system does NOT announce itself."

Wrong variant: the enclosed problem with no pressure constraint.

Setup: Taylor-Hood VectorH1 order 2 * H1 order 1 on two problems that differ
ONLY in the boundary conditions -- an enclosed cavity with velocity Dirichlet on
the whole boundary, and the same square with the right edge left traction-free.
Singular values of the free-DOF block are computed densely for both.

What this fixture pins, all re-measured on this run:
  * the enclosed system has exactly one zero singular value and the open one has
    none, so the difference is the boundary condition and nothing else;
  * the enclosed nullspace vector lives entirely in the pressure block and is
    constant there -- it IS the constant-pressure mode;
  * the singular solve is silent: no exception, no output on stdout or stderr,
    checked at the file-descriptor level, and it returns a finite vector;
  * a NumberSpace mean-zero multiplier removes the nullspace, as an alternative
    to pinning -- both routes are exercised and both give the same velocity;
  * the mean-zero route additionally fixes the pressure's constant, which
    pinning a node does not.

Mutation control: the claim's second sentence rests on the open problem leaving
the right edge traction-free.  T2_MUTATE=1 changes the dirichlet string handed
to build() for the open problem from "bot|top|left" to ".*", which closes the
flow, so it acquires the same constant-pressure nullspace and the fixture loses
open_flow_pressure_is_unique=True.  Re-run: T2_MUTATE=1 python source.py
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
    NumberSpace,
    VectorH1,
    div,
    dx,
)

MUTATE = os.environ.get("T2_MUTATE") == "1"


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


def mesh_():
    g = SplineGeometry()
    g.AddRectangle((0, 0), (1, 1), bcs=["bot", "right", "top", "left"])
    return Mesh(g.GenerateMesh(maxh=0.25))


def build(mesh, dirichlet):
    V = VectorH1(mesh, order=2, dirichlet=dirichlet)
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


def singulars(X, a, fr):
    rows, cols, vals = a.mat.COO()
    A = scipy.sparse.coo_matrix(
        (numpy.asarray(vals), (numpy.asarray(rows), numpy.asarray(cols))),
        shape=(X.ndof, X.ndof)).toarray()
    idx = [i for i in range(X.ndof) if fr[i]]
    U, sv, Vt = numpy.linalg.svd(A[numpy.ix_(idx, idx)])
    return sv, Vt, idx


def main() -> int:
    mesh = mesh_()

    Xe, ae, fe = build(mesh, ".*")                   # enclosed
    sve, Vte, idxe = singulars(Xe, ae, Xe.FreeDofs())
    ke = int((sve < 1e-10 * sve[0]).sum())
    print(f"enclosed_smallest_sv={sve[-1]:.3e} next={sve[-2]:.3e}")
    print(f"enclosed_nullspace_dimension={ke}")
    print(f"enclosed_has_exactly_one_zero_mode={ke == 1}")

    pstart = Xe.Range(1).start
    vec = Vte[-1]
    pp = numpy.array([vec[k] for k, i in enumerate(idxe) if i >= pstart])
    frac = float(pp @ pp) / float(vec @ vec)
    spread = float(pp.std() / max(abs(pp.mean()), 1e-30))
    print(f"nullmode_pressure_fraction={frac:.8f} spread={spread:.3e}")
    print(f"nullmode_is_the_constant_pressure={frac > 1 - 1e-8 and spread < 1e-8}")

    # MUTATION SITE: the traction-free right edge is the only thing separating
    # the open problem from the enclosed one.  T2_MUTATE=1 closes it.
    Xo, ao, fo = build(mesh, ".*" if MUTATE else "bot|top|left")
    svo, _, _ = singulars(Xo, ao, Xo.FreeDofs())
    ko = int((svo < 1e-10 * svo[0]).sum())
    print(f"open_smallest_sv={svo[-1]:.3e}")
    print(f"open_nullspace_dimension={ko}")
    print(f"open_flow_pressure_is_unique={ko == 0}")

    gs = GridFunction(Xe)

    def _solve():
        gs.vec.data = ae.mat.Inverse(Xe.FreeDofs(), inverse="umfpack") * fe.vec
        return True

    _, exc, out = _capture_fds(_solve)
    arr = gs.vec.FV().NumPy()
    print(f"singular_solve_raised={exc!r}")
    print(f"singular_solve_output={out.strip()!r}")
    print(f"singular_system_does_not_announce_itself="
          f"{exc is None and out.strip() == ''}")
    print(f"singular_solve_returned_finite="
          f"{bool(numpy.all(numpy.isfinite(arr)))}")

    # Route 1: pin one pressure DOF.
    frp = BitArray(Xe.FreeDofs())
    frp[pstart] = False
    gp = GridFunction(Xe)
    gp.vec.data = ae.mat.Inverse(frp, inverse="umfpack") * fe.vec

    # Route 2: NumberSpace mean-zero multiplier.
    V = VectorH1(mesh, order=2, dirichlet=".*")
    Q = H1(mesh, order=1)
    N = NumberSpace(mesh)
    Y = V * Q * N
    (u, p, lam), (v, q, mu) = Y.TnT()
    an = BilinearForm(Y)
    an += (InnerProduct(Grad(u), Grad(v)) - div(u) * q - div(v) * p
           + p * mu + q * lam) * dx
    an.Assemble()
    fn = LinearForm(Y)
    fn += CoefficientFunction((0, 1)) * v * dx
    fn.Assemble()
    gn = GridFunction(Y)
    gn.vec.data = an.mat.Inverse(Y.FreeDofs(), inverse="umfpack") * fn.vec

    duv = float(numpy.sqrt(Integrate(
        InnerProduct(gp.components[0] - gn.components[0],
                     gp.components[0] - gn.components[0]), mesh)))
    print(f"pinned_vs_numberspace_velocity_difference={duv:.3e}")
    print(f"both_routes_give_the_same_velocity={duv < 1e-10}")

    mean_pin = float(Integrate(gp.components[1], mesh))
    mean_num = float(Integrate(gn.components[1], mesh))
    print(f"pinned_pressure_mean={mean_pin:.6e}")
    print(f"numberspace_pressure_mean={mean_num:.6e}")
    print(f"numberspace_enforces_mean_zero={abs(mean_num) < 1e-10}")
    print(f"pinning_does_not_fix_the_mean={abs(mean_pin) > 1e-6}")

    ok = (
        ke == 1 and ko == 0
        and frac > 1 - 1e-8 and spread < 1e-8
        and exc is None and out.strip() == ""
        and bool(numpy.all(numpy.isfinite(arr)))
        and duv < 1e-10
        and abs(mean_num) < 1e-10 and abs(mean_pin) > 1e-6
    )
    if ok:
        return 0
    print("FAIL: Stokes pressure-nullspace invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
