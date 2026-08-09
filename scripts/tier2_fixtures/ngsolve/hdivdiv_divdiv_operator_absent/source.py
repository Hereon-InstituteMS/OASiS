"""Tier-2: HDivDiv has no pointwise div(div(tau)); use the H1 Hessian + skeleton.

Claim: ngsolve hdivdiv#7 — HDivDiv does NOT expose a pointwise div(div(tau))
operator; 'div(div(tau)) * v * dx' raises Exception 'cannot form div'. The fix is
to integrate by parts twice: InnerProduct(tau, v.Operator('hesse')) minus the
normal-normal moment skeleton integral over element boundaries.

Wrong variant: apply ngs.div twice to the HDivDiv trial function, both as a bare
expression and inside 'BilinearForm += div(div(tau)) * ww * dx'.

Observed on NGSolve 6.2.2604 (2026-08-03):
  * ngs.div(tau) alone is fine and has dims [2] (row-wise divergence, a vector).
  * ngs.div(ngs.div(tau)) raises a plain Python Exception whose str() is exactly
    'cannot form div' -- and it raises while BUILDING the expression, i.e. before
    SymbolicBFI ever sees it, so 'a += div(div(tau))*ww*dx' fails on the same
    message.
  * The documented replacement assembles and, with M_nn = 0 imposed on HDivDiv,
    reproduces the Navier simply-supported value 0.00406235 q L^4 / D to 4
    digits.

Mutation control: ``T2_MUTATE=1 python source.py`` removes the pathology at its
site -- N_DIV_OPERATORS drops from 2 to 1, so ngs.div is applied to the HDivDiv
proxy only once, which is legal.  The 'cannot form div' exception is then never
raised, so ``divdiv_expression_raises=True`` and the literal expectation
``cannot form div`` disappear and the fixture goes red.
"""
from __future__ import annotations

import os
import sys

import ngsolve as ngs
from netgen.geom2d import unit_square

MUTATE = os.environ.get("T2_MUTATE") == "1"

# WRONG: how many times ngs.div is applied to the HDivDiv proxy.
# Under T2_MUTATE the second div -- the pathology -- is not applied.
N_DIV_OPERATORS = 1 if MUTATE else 2

E, NU, T_PLATE, Q = 1.0, 0.3, 1.0, 1.0
D = E * T_PLATE ** 3 / (12.0 * (1.0 - NU ** 2))
BND = "bottom|right|top|left"
W_NAVIER = 0.004062352659370942 * Q / D


def spaces(maxh: float, order: int):
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=maxh))
    Vm = ngs.HDivDiv(mesh, order=order - 1, dirichlet=BND)
    Vw = ngs.H1(mesh, order=order, dirichlet=BND)
    return mesh, Vm * Vw


def repeated_div(proxy, times: int):
    out = proxy
    for _ in range(times):
        out = ngs.div(out)
    return out


def main() -> int:
    ok = True
    print(f"ngsolve_version={ngs.__version__}")
    mesh, X = spaces(0.15, 2)
    (sig, ww), (tau, vv) = X.TnT()
    print(f"moment_space_type={X.components[0].type} "
          f"deflection_space_type={X.components[1].type}")

    # single div IS available -- the failure is specific to the second one
    d1 = ngs.div(tau)
    print(f"single_div_dims={list(d1.dims)}")
    print(f"single_div_available={list(d1.dims) == [2]}")
    if list(d1.dims) != [2]:
        print(f"FAIL: div(tau) on HDivDiv no longer yields a 2-vector, got "
              f"{list(d1.dims)}", file=sys.stderr)
        ok = False

    # --- WRONG variant (a): build the expression --------------------------
    msg_expr, typ_expr = "", ""
    try:
        repeated_div(tau, N_DIV_OPERATORS)
    except Exception as exc:                        # noqa: BLE001
        msg_expr, typ_expr = str(exc), type(exc).__name__
    print(f"divdiv_expression_raises={bool(msg_expr)} "
          f"exc_type={typ_expr} msg={msg_expr!r}")
    # The expectation used to be the bare phrase "cannot form div", which this
    # fixture writes itself on the FAIL line below -- so it was satisfied
    # precisely when NGSolve had NOT said it, and the fixture only stayed
    # honest because that same line carries the forbidden "FAIL:".  Computed
    # from NGSolve's message instead.
    print(f"divdiv_msg_says_cannot_form_div="
          f"{'cannot form div' in msg_expr}")
    if "cannot form div" not in msg_expr:
        print(f"FAIL: div(div(tau)) did not raise the documented "
              f"'cannot form div'; got {msg_expr!r}", file=sys.stderr)
        ok = False

    # --- WRONG variant (b): inside the bilinear form ----------------------
    msg_form = ""
    try:
        a_bad = ngs.BilinearForm(X)
        a_bad += repeated_div(tau, N_DIV_OPERATORS) * ww * ngs.dx
        a_bad.Assemble()
    except Exception as exc:                        # noqa: BLE001
        msg_form = str(exc)
    print(f"divdiv_bilinearform_raises={bool(msg_form)} msg={msg_form!r}")
    if "cannot form div" not in msg_form:
        print(f"FAIL: 'a += div(div(tau))*ww*dx' assembled or raised something "
              f"else: {msg_form!r}", file=sys.stderr)
        ok = False

    # --- RIGHT variant: hesse + normal-normal skeleton integral -----------
    n = ngs.specialcf.normal(2)

    def compliance(s):
        return (1.0 / (D * (1.0 - NU))) \
            * (s - NU / (1.0 + NU) * ngs.Trace(s) * ngs.Id(2))

    a = ngs.BilinearForm(X, symmetric=True)
    a += ngs.InnerProduct(compliance(sig), tau) * ngs.dx
    a += ngs.InnerProduct(tau, ww.Operator("hesse")) * ngs.dx
    a += ngs.InnerProduct(sig, vv.Operator("hesse")) * ngs.dx
    a += -(tau * n * n) * (ngs.Grad(ww) * n) * ngs.dx(element_boundary=True)
    a += -(sig * n * n) * (ngs.Grad(vv) * n) * ngs.dx(element_boundary=True)
    a.Assemble()
    f = ngs.LinearForm(X)
    f += Q * vv * ngs.dx
    f.Assemble()
    gf = ngs.GridFunction(X)
    gf.vec.data = a.mat.Inverse(X.FreeDofs(), inverse="umfpack") * f.vec
    wc = abs(gf.components[1](mesh(0.5, 0.5)))
    rel = abs(wc - W_NAVIER) / W_NAVIER
    print(f"hesse_plus_skeleton_assembles=True nze={a.mat.nze}")
    print(f"hesse_plus_skeleton_centre={wc:.8f} rel_dev_from_navier={rel:.3e}")
    print(f"documented_fix_matches_navier_within_1pct={rel < 0.01}")
    if rel >= 0.01:
        print(f"FAIL: the documented hesse+skeleton replacement is off the "
              f"Navier reference by {rel:.3%}", file=sys.stderr)
        ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
