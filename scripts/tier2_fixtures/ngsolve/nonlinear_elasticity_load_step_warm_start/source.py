"""Tier-2: load stepping only works if the previous converged GridFunction really
survives into the next step.

Claim: ngsolve nonlinear_elasticity#1 -- apply the load in INCREMENTS, using the
previous converged GridFunction as the initial guess.

Wrong variant: the pattern shipped in the catalog's nonlinear_elasticity_2d /
_3d templates, i.e. re-prescribing the Dirichlet data every step with

    gfu.Set(CoefficientFunction((0, disp_now)), definedon=mesh.Boundaries("top"))

`GridFunction.Set` with `definedon=` ZEROES every dof outside the region -- all
free/interior dofs included.  So the loop is not warm-started at all: every load
step is a cold start from u = 0 with a larger Dirichlet jump, which is exactly
what the claim forbids.

Observed on NGSolve 6.2.2604, at the catalog's own 2-D defaults
(unit_square maxh=0.1, order 2, dirichlet="left|top", total disp 0.5, 10 steps):
  * cold restart  -- Newton iteration count grows 6, 8, 9 and step 4 aborts with
    NgException: UmfpackInverse: Numeric factorization failed.
  * warm start (overwrite ONLY the constrained dofs via
    Projector(fes.FreeDofs(), False)) -- all 10 steps converge in a constant 6
    iterations and max|u_y| equals the prescribed 0.5 exactly.

Also pinned here: `solvers.Newton` returns (status, numit), NOT (iters, err).
The templates unpack it as `(iters, conv)`, which is why they print
"Newton iters=0" at every step -- 0 is the CONVERGED STATUS FLAG, not an
iteration count.
"""

from __future__ import annotations

import sys

import numpy as np
from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    Det,
    GridFunction,
    Grad,
    Id,
    Mesh,
    Projector,
    Trace,
    Variation,
    VectorH1,
    dx,
    log,
    solvers,
    unit_square,
)

E, NU = 200.0, 0.3
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))

MAXH = 0.1  # catalog default for nonlinear_elasticity_2d
ORDER = 2
TOTAL_DISP = 0.5
N_STEPS = 10

# The wrong variant is selected by this flag; flip it to True and the pathology
# disappears (that is the mutation check).
WARM_WRONG = False


def build():
    mesh = Mesh(unit_square.GenerateMesh(maxh=MAXH))
    fes = VectorH1(mesh, order=ORDER, dirichlet="left|top")
    u = fes.TrialFunction()
    ident = Id(2)
    F = ident + Grad(u)
    C = F.trans * F
    J = Det(F)
    energy = 0.5 * MU * (Trace(C) - 2) - MU * log(J) + 0.5 * LAM * log(J) ** 2
    a = BilinearForm(fes, symmetric=True)
    a += Variation(energy * dx)
    return mesh, fes, a


def load_step(warm: bool):
    """Return (per_step_records, max_abs_uy, exception_text)."""
    mesh, fes, a = build()
    gfu = GridFunction(fes)
    scratch = GridFunction(fes)
    # Projector onto the NON-free (= constrained) dofs.
    proj_dir = Projector(fes.FreeDofs(), False)
    records: list[tuple[int, int, int]] = []
    exc_text = ""
    for step in range(1, N_STEPS + 1):
        disp_now = step / N_STEPS * TOTAL_DISP
        if warm:
            # Touch ONLY the constrained dofs; the converged interior survives.
            scratch.Set(CoefficientFunction((0.0, disp_now)),
                        definedon=mesh.Boundaries("top"))
            gfu.vec.data += proj_dir * (scratch.vec - gfu.vec)
        else:
            # Catalog template pattern -- wipes every free dof back to zero.
            gfu.Set(CoefficientFunction((0.0, disp_now)),
                    definedon=mesh.Boundaries("top"))
        try:
            status, numit = solvers.Newton(a, gfu, maxit=25, dampfactor=1.0,
                                           printing=False, maxerr=1e-10)
        except Exception as exc:  # noqa: BLE001 - we want the library's wording
            exc_text = f"{type(exc).__name__}: {exc}"
            break
        records.append((step, status, numit))
    max_uy = float(np.abs(gfu.components[1].vec.FV().NumPy()).max())
    return records, max_uy, exc_text


def main() -> int:
    ok = True

    # --- the API fact behind the bogus "Newton iters=0" print ---------------
    doc = solvers.Newton.__doc__ or ""
    doc_says_status_first = "0 if Newton's method did converge" in doc
    print(f"newton_doc_status_is_first_return={doc_says_status_first}")
    if not doc_says_status_first:
        print("FAIL: solvers.Newton docstring no longer documents (status, numit)",
              file=sys.stderr)
        ok = False

    # --- Set(definedon=...) really does zero the free dofs -----------------
    mesh, fes, _ = build()
    probe = GridFunction(fes)
    from ngsolve import x as _x, y as _y  # noqa: PLC0415 - local, keeps top clean
    probe.Set(CoefficientFunction((0.1 * _x, 0.2 * _y)))
    free = fes.FreeDofs()
    idx_free = np.array([bool(free[i]) for i in range(fes.ndof)])
    before = float(np.abs(probe.vec.FV().NumPy()[idx_free]).max())
    probe.Set(CoefficientFunction((0.0, 0.05)), definedon=mesh.Boundaries("top"))
    after = float(np.abs(probe.vec.FV().NumPy()[idx_free]).max())
    zeroes_free = before > 1e-3 and after == 0.0
    print(f"set_definedon_zeroes_all_free_dofs={zeroes_free}")
    if not zeroes_free:
        print("FAIL: Set(definedon=...) no longer wipes the free dofs -- the "
              "template's cold-restart mechanism is gone", file=sys.stderr)
        ok = False

    # --- WRONG variant: catalog cold-restart load stepping -----------------
    cold, cold_uy, cold_exc = load_step(warm=WARM_WRONG)
    cold_iters = [n for (_s, _st, n) in cold]
    print(f"cold_steps_completed={len(cold)}")
    print(f"cold_iters={cold_iters}")
    print(f"cold_iteration_count_grows={len(cold_iters) > 1 and cold_iters[-1] > cold_iters[0]}")
    print(f"cold_exception={cold_exc!r}")
    cold_broke = len(cold) < N_STEPS and "Numeric factorization failed" in cold_exc
    print(f"cold_restart_broke_before_full_load={cold_broke}")
    if not cold_broke:
        print("FAIL: cold-restart load stepping completed all steps -- the "
              "pitfall did not reproduce", file=sys.stderr)
        ok = False
    # displacement left behind is NOT the prescribed value
    print(f"cold_max_uy_is_wrong={abs(cold_uy - TOTAL_DISP) > 1e-3}")

    # --- RIGHT variant: genuine warm start ---------------------------------
    warm, warm_uy, warm_exc = load_step(warm=True)
    warm_iters = [n for (_s, _st, n) in warm]
    warm_status = [st for (_s, st, _n) in warm]
    print(f"warm_steps_completed={len(warm)}")
    print(f"warm_iters={warm_iters}")
    print(f"warm_all_status_zero={all(s == 0 for s in warm_status)}")
    print(f"warm_iters_constant={len(set(warm_iters)) == 1}")
    print(f"warm_max_iters_le_cold_max_iters={max(warm_iters) <= max(cold_iters)}")
    print(f"warm_max_uy_equals_prescribed={abs(warm_uy - TOTAL_DISP) < 1e-6}")
    warm_good = (len(warm) == N_STEPS and not warm_exc
                 and all(s == 0 for s in warm_status)
                 and abs(warm_uy - TOTAL_DISP) < 1e-6)
    if not warm_good:
        print(f"FAIL: warm-started load stepping did not solve cleanly "
              f"(exc={warm_exc!r}, uy={warm_uy})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
