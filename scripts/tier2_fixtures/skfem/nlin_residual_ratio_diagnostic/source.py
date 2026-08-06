"""Tier-2: a constant residual ratio is the signature of an inexact Jacobian.

Claim: skfem nonlinear#4 -- if the ratio r_{k+1}/r_k stays roughly constant
across iterations, the assembled BilinearForm Jacobian is wrong or inexact.

This fixture runs both operators and checks the DIAGNOSTIC discriminates: the
exact linearisation's successive ratios collapse by more than two orders of
magnitude, the inexact one's stay within a few percent of each other.

Wrong variant: jacobian_frozen (the exact linearisation minus one term).
"""
from __future__ import annotations

import sys

import numpy as np
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    LinearForm,
    MeshTri,
    condense,
    solve,
)
from skfem.helpers import dot, grad

# -div((1 + u^2) grad u) = 10 on the unit square, u = 0 on the boundary.
# Smooth, coercive, and its exact linearisation has a term a lazy
# implementation drops -- which is what several of these claims are about.


@LinearForm
def residual(v, w):
    u = w["u"]
    return (1.0 + u.value ** 2) * dot(u.grad, grad(v)) - 10.0 * v


@BilinearForm
def jacobian_exact(du, v, w):
    u = w["u"]
    return ((1.0 + u.value ** 2) * dot(grad(du), grad(v))
            + 2.0 * u.value * du * dot(u.grad, grad(v)))


@BilinearForm
def jacobian_frozen(du, v, w):
    """The exact linearisation MINUS the 2 u du term -- a Picard/quasi-Newton
    operator that is still a descent direction but not the derivative."""
    u = w["u"]
    return (1.0 + u.value ** 2) * dot(grad(du), grad(v))


def newton(basis, jac, n_steps: int = 6, alpha: float = 1.0):
    """Return the residual norm history over the free DOFs."""
    D = basis.get_dofs()
    free = basis.complement_dofs(D)
    u = basis.zeros()
    history = []
    for _ in range(n_steps):
        w = basis.interpolate(u)
        r = residual.assemble(basis, u=w)
        history.append(float(np.linalg.norm(r[free])))
        J = jac.assemble(basis, u=w)
        u = u + alpha * solve(*condense(J, -r, D=D))
    return history


def ratios(history):
    return np.array([b / a for a, b in zip(history, history[1:]) if a > 0])


def main() -> int:
    ok = True
    basis = Basis(MeshTri().refined(3), ElementTriP1())
    exact = ratios(newton(basis, jacobian_exact))
    inexact = ratios(newton(basis, jacobian_frozen))
    print(f"exact_ratios={np.round(exact, 8).tolist()}")
    print(f"inexact_ratios={np.round(inexact, 8).tolist()}")

    collapse = exact[0] / exact[-1] > 100.0
    print(f"exact_first_over_last_ratio_gt_100={collapse}")
    print(f"exact_successive_ratios_fall_by_over_100x={collapse}")
    if not collapse:
        print(f"FAIL: the exact Jacobian's ratios did not collapse "
              f"({np.round(exact, 8)})", file=sys.stderr)
        ok = False

    span = float(inexact.max() / inexact.min())
    flat = span < 2.5
    print(f"inexact_ratio_span={span:.4f}")
    print(f"inexact_ratio_stays_within_2p5x={flat}")
    if not flat:
        print(f"FAIL: the inexact Jacobian's ratios are not roughly constant "
              f"(span {span:.4f}, {np.round(inexact, 8)})", file=sys.stderr)
        ok = False

    separates = collapse and flat
    print(f"diagnostic_separates_the_two={separates}")
    if not separates:
        print("FAIL: the ratio diagnostic does not separate exact from inexact",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
