"""Tier-2: backward Euler needs M@u_old on the RHS; K@u_old blows up.

Claim: skfem heat#1 -- M du/dt + K u = f, so backward Euler is
(M + dt K) u_new = M u_old + dt f. Forgetting the mass matrix on the RHS is the
classic theta-method confusion, and the corrected signal is that the solution
BLOWS UP rather than decaying to zero.

Wrong variant (a): K @ u_old on the RHS.
Wrong variant (b): the bare u_old vector with no matrix at all.
Right variant: M @ u_old, whose per-step decay tracks the analytic factor
exp(-2 pi^2 dt) for the first eigenmode initial condition.

Mutation control: T2_MUTATE=1 applies the documented fix to both wrong
variants -- the RHS callables become M @ u instead of K @ u and instead of the
bare u.  The blow-up then does not happen, so 'K_u_old_blows_up=True',
'bare_u_old_blows_up=True' and 'neither_wrong_variant_decays_to_zero=True'
disappear from the output.  Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from skfem import Basis, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace, mass

MUTATE = os.environ.get("T2_MUTATE") == "1"


def march(A, rhs, u0, D, n_steps: int) -> list[float]:
    u = u0.copy()
    out = []
    for _ in range(n_steps):
        u = solve(*condense(A, rhs(u), D=D))
        out.append(float(np.abs(u).max()))
    return out


def main() -> int:
    ok = True
    basis = Basis(MeshTri().refined(3), ElementTriP1())
    K = laplace.assemble(basis)
    M = mass.assemble(basis)
    u0 = basis.project(
        lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    D = basis.get_dofs()
    dt = 0.005
    A = (M + dt * K).tocsr()
    n = 6

    # Under mutation both wrong RHS builders are replaced by the documented
    # correct one, M @ u_old.
    rhs_k = (lambda u: K @ u) if not MUTATE else (lambda u: M @ u)
    rhs_bare = (lambda u: u) if not MUTATE else (lambda u: M @ u)

    correct = march(A, lambda u: M @ u, u0, D, n)
    wrong_k = march(A, rhs_k, u0, D, n)
    wrong_bare = march(A, rhs_bare, u0, D, n)
    print(f"correct_history={[f'{v:.3e}' for v in correct]}")
    print(f"K_u_old_history={[f'{v:.3e}' for v in wrong_k]}")
    print(f"bare_u_old_history={[f'{v:.3e}' for v in wrong_bare]}")

    decays = all(b < a for a, b in zip(correct, correct[1:]))
    print(f"correct_decays_monotonically={decays}")
    analytic = float(np.exp(-2.0 * np.pi ** 2 * dt))
    ratios = np.array([b / a for a, b in zip(correct, correct[1:])])
    tracks = bool((np.abs(ratios - analytic) < 0.02).all())
    print(f"correct_tracks_analytic_decay_factor={tracks}")
    if not (decays and tracks):
        print(f"FAIL: the correct scheme does not decay at the analytic factor "
              f"{analytic!r}; ratios {np.round(ratios, 4)}", file=sys.stderr)
        ok = False

    # --- WRONG variants -------------------------------------------------
    k_blows = wrong_k[-1] > 1e6 and all(
        b > a for a, b in zip(wrong_k, wrong_k[1:]))
    bare_blows = wrong_bare[-1] > 1e6 and all(
        b > a for a, b in zip(wrong_bare, wrong_bare[1:]))
    print(f"K_u_old_blows_up={k_blows}")
    print(f"bare_u_old_blows_up={bare_blows}")
    print(f"neither_wrong_variant_decays_to_zero="
          f"{wrong_k[-1] > correct[0] and wrong_bare[-1] > correct[0]}")
    if not (k_blows and bare_blows):
        print(f"FAIL: a wrong RHS did not blow up: K@u_old {wrong_k[-1]!r}, "
              f"bare {wrong_bare[-1]!r}", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
