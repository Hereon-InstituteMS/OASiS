"""Tier-2: only the exact linearisation gives quadratic Newton convergence.

Claim: skfem nonlinear#1 -- the Jacobian must be the linearisation of the weak
form about the previous iterate, assembled fresh each iteration with
basis.interpolate(u_prev) inside the @BilinearForm body. A quasi-Newton
substitute collapses to linear convergence.

Wrong variant: jacobian_frozen, the exact linearisation with the 2 u du term
dropped. Right variant: jacobian_exact.
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
    exact = newton(basis, jacobian_exact)
    frozen = newton(basis, jacobian_frozen)
    print(f"exact_history={[f'{v:.3e}' for v in exact]}")
    print(f"frozen_history={[f'{v:.3e}' for v in frozen]}")

    re, rf = ratios(exact), ratios(frozen)
    print(f"exact_ratios={np.round(re, 8).tolist()}")
    print(f"frozen_ratios={np.round(rf, 8).tolist()}")

    # Once the residual reaches roundoff (~1e-15) the ratio stops being
    # informative and can tick back up, so only the pre-roundoff window is
    # checked for monotonicity.
    pre = [b / a for a, b in zip(exact, exact[1:]) if b > 1e-12]
    decreasing = len(pre) >= 3 and all(y < x for x, y in zip(pre, pre[1:]))
    print(f"n_pre_roundoff_ratios={len(pre)}")
    print(f"exact_jacobian_ratio_decreases_before_roundoff={decreasing}")
    deep = exact[-1] < 1e-9
    print(f"exact_jacobian_reaches_below_1e_9={deep}")
    if not (decreasing and deep):
        print(f"FAIL: the exact Jacobian did not converge quadratically "
              f"(pre-roundoff ratios {np.round(pre, 8)}, final {exact[-1]!r})",
              file=sys.stderr)
        ok = False

    # Linear convergence: the ratio barely moves (a factor under 2 across the
    # whole run) and it never gets deep. The exact one moves by >1e4.
    span_frozen = float(rf.max() / rf.min())
    span_exact = float(re.max() / re.min())
    print(f"frozen_ratio_span={span_frozen:.3f}")
    print(f"exact_ratio_span={span_exact:.3e}")
    constant = span_frozen < 2.5
    print(f"frozen_jacobian_ratio_stays_within_2p5x={constant}")
    print(f"exact_ratio_span_over_1000x={span_exact > 1e3}")
    stalls = frozen[-1] > 1e-9
    print(f"frozen_jacobian_stalls_above_1e_9={stalls}")
    if not (constant and stalls and span_exact > 1e3):
        print(f"FAIL: the frozen Jacobian did not show linear convergence "
              f"(span {span_frozen:.3f}, final {frozen[-1]!r}; exact span "
              f"{span_exact:.3e})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
