"""Tier-2: solvers.Newton returns (status, numit) -- status FIRST -- and reading
it the other way round inverts the failure test.

Claim: ngsolve nonlinear_elasticity#8 -- "ngsolve.solvers.Newton
(ngsolve.nonlinearsolvers.Newton) returns (status, numit) -- an INTEGER STATUS
FIRST, where 0 means converged and -1 means it gave up, then the iteration
count.  It is NOT (iters, convergence_error): there is no error value in the
return at all.  ... The damage is that the failure test inverts: a converged
solve reports 'iters=0' (which reads like nothing happened) and a diverged solve
reports 'iters=-1' ... Signal: every load step prints 'Newton iters=0' no matter
how hard the step was, and a genuinely failing step prints -1 together with
NGSolve's own line 'Warning: Newton might not converge! Error = ' -- that
warning, not the return value, is where the error norm appears."

Wrong variant: (iters, conv) = solvers.Newton(...).

Setup: a Neo-Hookean cantilever on VectorH1, solved twice at the SAME load --
once with an iteration budget that is enough and once with a budget that is not
-- so both the converged and the diverged return come out of real solves and
differ in one parameter, with nothing else changed.

What this fixture pins, all re-measured on this run:
  * the return is a 2-tuple of two ints, with no float anywhere in it, so
    "there is no error value in the return at all" is checked, not assumed;
  * the docstring itself says the first entry is 0 on convergence;
  * on the easy load the return's FIRST entry is 0 and its second is a plausible
    iteration count strictly greater than 1 -- so the pair is (status, numit)
    and not (numit, status);
  * misreading it as (iters, conv) makes the easy solve report iters=0, which is
    the claim's "reads like nothing happened" -- shown by unpacking both ways
    from the same return value;
  * on the hard load the return's first entry is -1, which the misreading turns
    into iters=-1;
  * the error norm appears only in NGSolve's own warning line, captured at the
    file-descriptor level, and 'Warning: Newton might not converge! Error = ' is
    the wording -- the string 'Newton did not converge after' appears nowhere.

Mutation control: T2_MUTATE=1 raises HARD_MAXIT from 2 to 12, so the "hard" solve
gets the same adequate iteration budget as the easy one and converges -- the
mutation this fixture's own _comment records.  The -1 status then never occurs, so
'hard_first_entry_is_minus_one=True',
'misreading_reports_iters_minus_1_on_a_failed_solve=True' and
'real_warning_present_on_the_hard_solve=True' go missing and the fixture goes red.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys
import tempfile

from netgen.geom2d import SplineGeometry
from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    Det,
    GridFunction,
    Grad,
    Id,
    InnerProduct,
    Mesh,
    Trace,
    Variation,
    VectorH1,
    dx,
    log,
    solvers,
)

E, NU = 200.0, 0.3
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
FABRICATED = "Newton did not converge after"
REAL = "Warning: Newton might not converge! Error = "

# Mutation control: the iteration budget of the deliberately-starved solve.
# 2 is not enough (status -1); T2_MUTATE=1 gives it the easy run's 12, which is
# enough, and the failed-solve half of the demonstration disappears.
MUTATE = os.environ.get("T2_MUTATE") == "1"
HARD_MAXIT = 12 if MUTATE else 2


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


def solve(load, maxit=12):
    g = SplineGeometry()
    g.AddRectangle((0, 0), (4, 1), bcs=["bot", "right", "top", "left"])
    mesh = Mesh(g.GenerateMesh(maxh=0.4))
    fes = VectorH1(mesh, order=2, dirichlet="left")
    u = fes.TrialFunction()
    F = Id(2) + Grad(u)
    C = F.trans * F
    J = Det(F)
    psi = 0.5 * MU * (Trace(C) - 2) - MU * log(J) + 0.5 * LAM * log(J) ** 2
    a = BilinearForm(fes, symmetric=False)
    a += Variation(psi * dx)
    a += Variation(load * u[1] * dx)
    gfu = GridFunction(fes)
    gfu.vec[:] = 0
    return solvers.Newton(a, gfu, maxit=maxit, printing=False)


def main() -> int:
    doc = (solvers.Newton.__doc__ or "").replace("\n", " ")
    print(f"docstring_says_first_entry_is_the_status="
          f"{'first one is 0 if' in doc.lower()}")

    easy, exc_e, out_e = _capture_fds(lambda: solve(2.0))
    hard, exc_h, out_h = _capture_fds(lambda: solve(2.0, maxit=HARD_MAXIT))
    print(f"easy_return={easy}")
    print(f"hard_return={hard}")

    both = [easy, hard]
    print(f"return_is_a_two_tuple="
          f"{all(isinstance(r, tuple) and len(r) == 2 for r in both)}")
    print(f"both_entries_are_ints="
          f"{all(all(isinstance(e, int) for e in r) for r in both)}")
    print(f"no_float_in_the_return="
          f"{not any(isinstance(e, float) for r in both for e in r)}")

    status_e, numit_e = easy
    status_h, numit_h = hard
    print(f"easy_status={status_e} easy_numit={numit_e}")
    print(f"easy_first_entry_is_zero={status_e == 0}")
    print(f"easy_second_entry_is_a_real_iteration_count={numit_e > 1}")
    print(f"pair_is_status_then_numit={status_e == 0 and numit_e > 1}")

    # The misreading, from the SAME return value.
    iters_wrong, conv_wrong = easy
    print(f"misread_easy_iters={iters_wrong} misread_easy_conv={conv_wrong}")
    print(f"misreading_reports_iters_0_on_a_converged_solve="
          f"{iters_wrong == 0}")

    print(f"hard_status={status_h} hard_numit={numit_h}")
    print(f"hard_first_entry_is_minus_one={status_h == -1}")
    print(f"hard_used_its_whole_budget={numit_h >= 2}")
    iters_wrong_h, _ = hard
    print(f"misreading_reports_iters_minus_1_on_a_failed_solve="
          f"{iters_wrong_h == -1}")

    print(f"real_warning_present_on_the_hard_solve={REAL in out_h}")
    print(f"fabricated_wording_absent="
          f"{FABRICATED not in out_h and FABRICATED not in out_e}")
    print(f"error_norm_only_in_the_warning="
          f"{REAL in out_h and not any(isinstance(e, float) for e in hard)}")
    print(f"easy_solve_printed_no_warning={REAL not in out_e}")

    ok = (
        "first one is 0 if" in doc.lower()
        and all(isinstance(r, tuple) and len(r) == 2 for r in both)
        and all(all(isinstance(e, int) for e in r) for r in both)
        and status_e == 0 and numit_e > 1
        and status_h == -1 and numit_h >= 2
        and REAL in out_h and FABRICATED not in out_h
        and REAL not in out_e
    )
    if ok:
        return 0
    print("FAIL: Newton return-order invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
