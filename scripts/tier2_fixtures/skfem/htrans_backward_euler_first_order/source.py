"""Tier-2: backward Euler is first order in the L-infinity norm too.

Claim: skfem heat_transient#0 -- backward Euler is unconditionally stable but
only first-order accurate in time.  "Signal: a manufactured-solution study
iterating scipy.sparse.linalg.spsolve with halved dt shows the
linfty_norm(u_h - u_exact) interpolated onto an InteriorBasis decreasing by a
factor ~2 per halving instead of ~4 (slope 1 vs 2 on a log-log plot)."

This is the L-infinity form of the claim, and it needs the same care as the
L^2 form: measured against the ANALYTIC solution the spatial error saturates,
so the study has to use a same-mesh, small-dt reference.

Measured on skfem 12.0.1, u_t = Laplace u on the unit square,
MeshTri.init_tensor 20x20 with ElementTriP1, homogeneous Dirichlet, IC
sin(pi x) sin(pi y), integrated to T = 0.02, dt halved four times, error taken
as max|u_h - u_ref| over the interior DOFs:

  * the factor-of-two-per-halving behaviour reproduces: the L-infinity error
    falls by close to 2 at each halving, giving a measured slope of about 1.
  * the entry's discriminator works -- Crank-Nicolson on the same mesh gives
    a factor of about 4 per halving, i.e. slope 2 -- so the two schemes are
    told apart by the ratio.
  * unconditional stability is checked separately: a dt far beyond any
    explicit limit still yields a bounded, decaying, finite solution.

Mutation control: T2_MUTATE=1 applies the fix this entry points at -- the theta
of the "be" sweep is raised from 1.0 (backward Euler) to 0.5 (Crank-Nicolson),
so the first-order behaviour is gone.  'be_factor_is_about_two=True',
'be_slope_is_one=True', 'be_factor_is_about_four=False' and
'the_two_schemes_are_distinguishable=True' all disappear from the output.  The
unconditional-stability probe is left at theta = 1.0: it is a property of
backward Euler, not the pathology.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from skfem import Basis, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace, mass

MUTATE = os.environ.get("T2_MUTATE") == "1"

T_END = 0.02
NX = 20


def integrate(theta, nsteps):
    m = MeshTri.init_tensor(np.linspace(0.0, 1.0, NX + 1),
                            np.linspace(0.0, 1.0, NX + 1))
    ib = Basis(m, ElementTriP1())
    K = laplace.assemble(ib)
    M = mass.assemble(ib)
    D = ib.get_dofs().all()
    u = ib.project(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    u[D] = 0.0
    dt = T_END / nsteps
    A = (M + theta * dt * K).tocsr()
    B = (M - (1.0 - theta) * dt * K).tocsr()
    for _ in range(nsteps):
        u = solve(*condense(A, B @ u, D=D))
    return u, ib, D


def main() -> int:
    ok = True
    counts = [4, 8, 16, 32, 64]
    results = {}
    # Under mutation the first-order scheme is replaced by the second-order one
    # at the same mesh and the same dt sequence.
    be_theta = 1.0 if not MUTATE else 0.5
    for theta, tag in ((be_theta, "be"), (0.5, "cn")):
        uref, ib, D = integrate(theta, 1024)
        free = np.setdiff1d(np.arange(ib.N), D)
        errs = []
        for n in counts:
            u, _, _ = integrate(theta, n)
            errs.append(float(np.abs(u[free] - uref[free]).max()))
        factors = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
        slopes = [float(np.log2(f)) for f in factors]
        results[tag] = (errs, factors, slopes)
        print(f"{tag}_dofs_N={ib.N} interior_dofs={len(free)}")
        print(f"{tag}_linfty_errors={[f'{e:.3e}' for e in errs]}")
        print(f"{tag}_reduction_factors={[f'{f:.3f}' for f in factors]}")
        print(f"{tag}_slopes={[f'{s:.3f}' for s in slopes]}")

    be_e, be_f, be_s = results["be"]
    cn_e, cn_f, cn_s = results["cn"]
    print(f"be_factor_is_about_two={all(1.8 < f < 2.3 for f in be_f)}")
    print(f"be_slope_is_one={all(0.85 < s < 1.2 for s in be_s)}")
    print(f"be_factor_is_about_four={all(3.6 < f < 4.4 for f in be_f)}")
    print(f"cn_factor_is_about_four={all(3.6 < f < 4.4 for f in cn_f)}")
    print(f"cn_slope_is_two={all(1.85 < s < 2.15 for s in cn_s)}")
    print(f"the_two_schemes_are_distinguishable={max(be_f) < min(cn_f)}")
    if not all(1.8 < f < 2.3 for f in be_f):
        print("FAIL: backward Euler did not halve its L-infinity error",
              file=sys.stderr)
        ok = False
    if all(3.6 < f < 4.4 for f in be_f):
        print("FAIL: backward Euler quartered its error, so it measured "
              "second order", file=sys.stderr)
        ok = False
    if not all(3.6 < f < 4.4 for f in cn_f):
        print("FAIL: Crank-Nicolson did not quarter its error",
              file=sys.stderr)
        ok = False

    # --- unconditional stability -----------------------------------------
    u_big, ib, D = integrate(1.0, 1)
    print(f"single_giant_step_finite={bool(np.isfinite(u_big).all())}")
    print(f"single_giant_step_max={float(np.abs(u_big).max()):.6f}")
    print(f"single_giant_step_decayed={float(np.abs(u_big).max()) < 1.0}")
    if not (np.isfinite(u_big).all() and np.abs(u_big).max() < 1.0):
        print("FAIL: a single giant backward-Euler step was not stable",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
