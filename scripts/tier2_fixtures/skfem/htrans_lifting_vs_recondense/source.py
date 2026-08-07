"""Tier-2: the pre-computed lifting for a time-varying Dirichlet value.

Claim: skfem heat_transient#4 -- "For non-homogeneous BCs that change in time:
re-condense at each step or pre-compute the lifting once.  Signal:
time-evolving boundary temperature does not appear in the solution --
max(T - T_D) at boundary DOFs is O(1) instead of 0 because the same x=
argument was reused frozen."

Both options in the claim are exercised here, because only one of them is
actually safe when the boundary DATA changes shape rather than amplitude.

Measured on skfem 12.0.1, heat equation on a 16x16 MeshTri.init_tensor with
ElementTriP1 (289 DOFs, 64 boundary DOFs), backward Euler, dt = 2e-3, 25
steps:

  * FROZEN x: reusing the x= argument from step 0 pins every boundary DOF at
    its initial value; max|T - T_D| at the boundary ends at O(1), exactly as
    claimed, and nothing warns.
  * PRE-COMPUTED LIFTING, scaled by the time factor, reproduces the
    re-condensed answer to machine precision -- but only because this
    boundary datum is a fixed spatial profile times a scalar in time.  The
    fixture also runs a datum whose SHAPE changes with time, and there the
    scaled lifting is wrong by an O(1) amount while re-condensing stays
    exact.  "Pre-compute the lifting once" is safe for separable data only.
  * every wrong variant is silent: finite, smooth, no warning, no exception.

Mutation control: T2_MUTATE=1 applies the documented fix -- "re-condense at each
step" -- at both wrong-variant sites: the frozen x and the scaled lifting are
each replaced by the current datum g in the condense call.  Both shortcuts then
honour the boundary data exactly, so
'sep_frozen_boundary_error_is_order_one=True',
'nonsep_scaled_lifting_is_order_one=True' and
'lifting_shortcut_is_only_safe_for_separable_data=True' disappear from the
output.  Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from skfem import Basis, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace, mass

MUTATE = os.environ.get("T2_MUTATE") == "1"

NX = 16
DT = 2e-3
NSTEPS = 25


def separable(x, t):
    """A fixed spatial profile times a scalar in time, ramped to O(1)."""
    return (t / (NSTEPS * DT)) * np.sin(np.pi * x[0])


def non_separable(x, t):
    """A profile whose SHAPE moves with time."""
    return np.sin(np.pi * (x[0] + t))


def run(mode, datum):
    m = MeshTri.init_tensor(np.linspace(0.0, 1.0, NX + 1),
                            np.linspace(0.0, 1.0, NX + 1))
    ib = Basis(m, ElementTriP1())
    K = laplace.assemble(ib)
    M = mass.assemble(ib)
    D = ib.get_dofs().all()
    A = (M + DT * K).tocsr()
    u = ib.zeros()
    lifting = None
    frozen = None
    msgs = []
    for step in range(1, NSTEPS + 1):
        t = step * DT
        g = ib.zeros()
        g[D] = datum(ib.doflocs[:, D], t)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if mode == "recondense":
                u = solve(*condense(A, M @ u, x=g, D=D))
            elif mode == "frozen":
                if frozen is None:
                    frozen = ib.zeros()
                    frozen[D] = datum(ib.doflocs[:, D], DT)
                # Under mutation the documented fix -- re-condense at each step
                # against the current datum -- is applied here.
                u = solve(*condense(A, M @ u,
                                    x=(frozen if not MUTATE else g), D=D))
            elif mode == "scaled_lifting":
                if lifting is None:
                    lifting = ib.zeros()
                    lifting[D] = datum(ib.doflocs[:, D], DT)
                scaled = (t / DT) * lifting
                u = solve(*condense(A, M @ u,
                                    x=(scaled if not MUTATE else g), D=D))
            else:
                raise AssertionError(mode)
            msgs += [str(c.message) for c in caught]
    g_final = ib.zeros()
    g_final[D] = datum(ib.doflocs[:, D], NSTEPS * DT)
    return u, ib, D, g_final, sorted(set(msgs))


def main() -> int:
    ok = True
    # --- separable datum --------------------------------------------------
    ref, ib, D, g, _ = run("recondense", separable)
    froz, _, _, _, m_fro = run("frozen", separable)
    lift, _, _, _, m_lif = run("scaled_lifting", separable)
    print(f"dofs_N={ib.N} boundary_dofs={len(D)} steps={NSTEPS}")
    err_ref = float(np.abs(ref[D] - g[D]).max())
    err_fro = float(np.abs(froz[D] - g[D]).max())
    err_lif = float(np.abs(lift[D] - g[D]).max())
    print(f"sep_recondense_boundary_error={err_ref:.3e}")
    print(f"sep_frozen_boundary_error={err_fro:.3e}")
    print(f"sep_scaled_lifting_boundary_error={err_lif:.3e}")
    print(f"sep_recondense_exact={err_ref < 1e-14}")
    print(f"sep_frozen_boundary_error_is_order_one={err_fro > 0.1}")
    print(f"sep_scaled_lifting_exact={err_lif < 1e-14}")
    print(f"sep_lifting_matches_recondense="
          f"{float(np.abs(lift - ref).max()) < 1e-12}")
    print(f"sep_frozen_warnings={m_fro!r}")
    print(f"sep_lifting_warnings={m_lif!r}")
    print(f"sep_frozen_is_silent={not m_fro}")
    if err_ref >= 1e-14:
        print("FAIL: re-condensing did not honour the datum", file=sys.stderr)
        ok = False
    if err_fro <= 0.1:
        print("FAIL: the frozen x did NOT leave an O(1) boundary error",
              file=sys.stderr)
        ok = False
    if err_lif >= 1e-14:
        print("FAIL: the scaled lifting was not exact on separable data",
              file=sys.stderr)
        ok = False
    if m_fro or m_lif:
        print(f"FAIL: a variant warned {sorted(set(m_fro + m_lif))!r}",
              file=sys.stderr)
        ok = False

    # --- non-separable datum ----------------------------------------------
    ref2, _, _, g2, _ = run("recondense", non_separable)
    lift2, _, _, _, m_lif2 = run("scaled_lifting", non_separable)
    e_ref2 = float(np.abs(ref2[D] - g2[D]).max())
    e_lif2 = float(np.abs(lift2[D] - g2[D]).max())
    print(f"nonsep_recondense_boundary_error={e_ref2:.3e}")
    print(f"nonsep_scaled_lifting_boundary_error={e_lif2:.3e}")
    print(f"nonsep_recondense_still_exact={e_ref2 < 1e-14}")
    print(f"nonsep_scaled_lifting_is_order_one={e_lif2 > 0.1}")
    print(f"nonsep_lifting_warnings={m_lif2!r}")
    print(f"nonsep_lifting_is_silent={not m_lif2}")
    print(f"nonsep_lifting_finite={bool(np.isfinite(lift2).all())}")
    print(f"lifting_shortcut_is_only_safe_for_separable_data="
          f"{err_lif < 1e-14 and e_lif2 > 0.1}")
    if e_ref2 >= 1e-14:
        print("FAIL: re-condensing failed on the non-separable datum",
              file=sys.stderr)
        ok = False
    if e_lif2 <= 0.1:
        print("FAIL: the scaled lifting survived the non-separable datum, so "
              "the shortcut is safe after all", file=sys.stderr)
        ok = False
    if m_lif2:
        print(f"FAIL: the non-separable lifting warned {m_lif2!r}",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
