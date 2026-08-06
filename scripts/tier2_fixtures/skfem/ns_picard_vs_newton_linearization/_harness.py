"""Shared Navier-Stokes Newton harness for this fixture."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import scipy.sparse as sp
from skfem import (
    Basis, BilinearForm, ElementTriP1, ElementTriP2, ElementVector,
    LinearForm, MeshTri, condense, solve,
)
from skfem.helpers import ddot, div, dot, grad


def build(re, refine):
    """Lid-driven cavity Navier-Stokes, Taylor-Hood, one pressure DOF pinned."""
    nu = 1.0 / re
    m = MeshTri().refined(refine)
    ev = ElementVector(ElementTriP2())
    bu = Basis(m, ev, intorder=4)
    bp = Basis(m, ElementTriP1(), intorder=4)

    @BilinearForm
    def viscous(u, v, w):
        return nu * ddot(grad(u), grad(v))

    @BilinearForm
    def neg_div(u, q, w):
        return -div(u) * q

    A = viscous.assemble(bu)
    B = neg_div.assemble(bu, bp)
    lid = bu.get_dofs(lambda x: x[1] > 1.0 - 1e-10)
    D = np.concatenate([bu.get_dofs().all(), [bu.N]])
    xb = np.zeros(bu.N + bp.N)
    xb[lid.nodal["u^1"]] = 1.0
    return bu, bp, A, B, D, xb


@BilinearForm
def conv_transport(u, v, w):
    """(u_prev . grad) du . v  -- the Picard/transport term."""
    return dot(dot(grad(u), w["up"]), v)


@BilinearForm
def conv_reaction(u, v, w):
    """(du . grad) u_prev . v  -- the term Picard drops."""
    return dot(dot(grad(w["up"]), u), v)


@LinearForm
def conv_residual(v, w):
    return dot(dot(grad(w["up"]), w["up"]), v)


def iterate(re, variant, nit=14, refine=3, u0=None, picard_first=0,
            tol=1e-12):
    """variant: 'newton' | 'picard' | 'stokes'.  Returns (x, residual history).

    The residual is ALWAYS that of the full Navier-Stokes equations, so the
    variants are compared on the same measure and only the Jacobian differs.
    """
    bu, bp, A, B, D, xb = build(re, refine)
    x = np.zeros(bu.N + bp.N) if u0 is None else u0.copy()
    x[D] = xb[D]
    free = np.setdiff1d(np.arange(len(x)), D)
    hist = []
    for it in range(nit):
        up = bu.interpolate(x[:bu.N])
        C1 = conv_transport.assemble(bu, up=up)
        C2 = conv_reaction.assemble(bu, up=up)
        if variant == "newton":
            J = A + C1 + C2
        elif variant == "picard":
            J = A + C1
        elif variant == "stokes":
            J = A
        else:
            raise AssertionError(variant)
        if it < picard_first:
            J = A + C1
        K = sp.bmat([[J, B.T], [B, None]], format="csr")
        R = np.concatenate([A @ x[:bu.N] + conv_residual.assemble(bu, up=up)
                            + B.T @ x[bu.N:], B @ x[:bu.N]])
        try:
            x = x + solve(*condense(K, -R, x=np.zeros_like(x), D=D))
        except Exception:                             # noqa: BLE001
            hist.append(float("nan"))
            break
        up2 = bu.interpolate(x[:bu.N])
        R2 = np.concatenate([A @ x[:bu.N] + conv_residual.assemble(bu, up=up2)
                             + B.T @ x[bu.N:], B @ x[:bu.N]])
        hist.append(float(np.linalg.norm(R2[free])))
        if not np.isfinite(hist[-1]) or hist[-1] < tol:
            break
    return x, hist


def converged(hist, tol=1e-10):
    return bool(hist and np.isfinite(hist[-1]) and hist[-1] < tol)
