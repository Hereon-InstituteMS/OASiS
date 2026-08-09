"""Tier-2: a full Newton step from a poor guess grows the residual.

Claim: skfem nonlinear#5 -- for large initial residual / poor initial guess, add
damping or a backtracking line search: with alpha=1 the residual norm grows
across iterations; halving alpha until the residual decreases makes it monotone.

The problem is deliberately stiffer than the other nonlinear fixtures':
-lap u + exp(u) = 200 from u = 0. The exponential makes the first full Newton
step overshoot badly.

Wrong variant: alpha = 1 throughout. Right variant: halve alpha until the
residual at the trial point is below the current one.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology site
-- the alpha = 1 branch inside run() also backtracks -- so the undamped run
becomes a damped one. The residual then no longer grows and does reach machine
precision, which removes 'full_step_residual_grows=True' and
'full_step_never_reaches_1e_9=True' from the output. Re-run: T2_MUTATE=1
python source.py
"""
from __future__ import annotations

import os
import sys

MUTATE = os.environ.get("T2_MUTATE") == "1"

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


@LinearForm
def residual(v, w):
    u = w["u"]
    return dot(u.grad, grad(v)) + np.exp(u.value) * v - 200.0 * v


@BilinearForm
def jacobian(du, v, w):
    u = w["u"]
    return dot(grad(du), grad(v)) + np.exp(u.value) * du * v


def run(basis, backtrack: bool, n_steps: int = 8):
    D = basis.get_dofs()
    free = basis.complement_dofs(D)
    u = basis.zeros()
    history = []
    for _ in range(n_steps):
        w = basis.interpolate(u)
        r = residual.assemble(basis, u=w)
        rn = float(np.linalg.norm(r[free]))
        history.append(rn)
        if not np.isfinite(rn):
            break
        du = solve(*condense(jacobian.assemble(basis, u=w), -r, D=D))
        # THE PATHOLOGY: the undamped full step, alpha = 1.  Under T2_MUTATE
        # this branch is skipped, i.e. the documented fix (backtracking) is
        # applied to the "full step" run as well.
        if not backtrack and not MUTATE:
            u = u + du
            continue
        alpha = 1.0
        for _ in range(40):
            trial = u + alpha * du
            rt = float(np.linalg.norm(
                residual.assemble(basis, u=basis.interpolate(trial))[free]))
            if np.isfinite(rt) and rt < rn:
                break
            alpha *= 0.5
        u = u + alpha * du
    return history


def main() -> int:
    ok = True
    basis = Basis(MeshTri().refined(3), ElementTriP1())
    full = run(basis, backtrack=False)
    damped = run(basis, backtrack=True)
    print(f"full_step_history={[f'{v:.3e}' for v in full]}")
    print(f"backtracking_history={[f'{v:.3e}' for v in damped]}")

    grows = len(full) > 1 and full[1] > 10.0 * full[0]
    print(f"full_step_first_ratio_gt_10={grows}")
    print(f"full_step_residual_grows={grows}")
    never = min(full) > 1e-9
    print(f"full_step_never_reaches_1e_9={never}")
    if not (grows and never):
        print(f"FAIL: the full-step run did not misbehave (history {full})",
              file=sys.stderr)
        ok = False

    monotone = all(b < a for a, b in zip(damped, damped[1:]))
    print(f"backtracking_residual_monotone={monotone}")
    deep = damped[-1] < 1e-9
    print(f"backtracking_reaches_below_1e_9={deep}")
    if not (monotone and deep):
        print(f"FAIL: backtracking was not monotone to machine precision "
              f"(history {damped})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
