"""Tier-2: the cavity resonance comes out right when the shift is put near the
analytic k^2, and shift=0 raises on the curl-curl operator.

Claim: ngsolve helmholtz#5 -- "For eigenvalues / cavity resonances: use
ArnoldiSolver with shift-invert; the shift should be near the expected
eigenvalue (k^2_estimate from analytic cavity formula).  shift=0 raises
NgException 'UmfpackInverse: Numeric factorization failed. UMFPACK ... WARNING:
matrix is singular' on operators with a non-empty null space (same family as
maxwell#5).  Signal: NgException with 'UmfpackInverse' + 'matrix is singular'
from ArnoldiSolver when shift=0."

Wrong variant: shift=0 on the cavity's curl-curl operator.

Setup: the PEC cubic cavity of side 1.  Its resonances are k^2 =
pi^2 (m^2 + n^2 + p^2) with at most one index zero, so the lowest is 2*pi^2,
doubly degenerate.  Solved as the curl-curl / mass pencil on HCurl with
tangential Dirichlet, which is the operator the claim is about -- its kernel is
the gradients, and that kernel is what shift=0 runs into.

CORRECTION this fixture records.  The claim quotes two strings and only one of
them is usable.  The NgException does carry 'UmfpackInverse'.  The 'matrix is
singular' line is written by UMFPACK through C stdio and flushed at process
exit, NOT while the call is in flight, so a guard that captures around the
ArnoldiSolver call comes back with an empty buffer and concludes nothing
happened.  Guard on the exception.

What this fixture pins, all re-measured on this run:
  * shift=0 raises NgException naming UmfpackInverse;
  * a capture placed around that same call sees no 'matrix is singular' text --
    the claim's second string is not available at the call site;
  * putting the shift near the analytic k^2 = 2*pi^2 completes and returns a
    lowest nonzero resonance within a few percent of it, and the mode is
    doubly degenerate as the cavity formula says;
  * the zero-eigenvalue gradient modes are present in the spectrum, so the
    operator genuinely has the null space the claim describes;
  * a shift far from any resonance still completes -- the failure is shift=0
    specifically, not "a bad shift".

Mutation control: T2_MUTATE=1 moves the first solve's shift off zero (0.0 ->
1.0), which is the fix a user applies to this pitfall -- the shift is what the
claim is about. UMFPACK then factorises, no NgException is raised, and
`shift0_names_umfpackinverse=True`, `exception_is_the_usable_signal=True` and
`only_shift_zero_fails=True` disappear from the output.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy
from netgen.csg import unit_cube
from ngsolve import (
    ArnoldiSolver,
    BilinearForm,
    GridFunction,
    HCurl,
    Mesh,
    curl,
    dx,
)

NVEC = 24
K2_LOWEST = 2 * math.pi ** 2
MUTATE = os.environ.get("T2_MUTATE") == "1"
# The pathology is the zero shift itself; T2_MUTATE=1 moves it off zero.
SHIFT_BAD = 1.0 if MUTATE else 0.0


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


def main() -> int:
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.35))
    fes = HCurl(mesh, order=2, dirichlet=".*")
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += curl(u) * curl(v) * dx
    a.Assemble()
    m = BilinearForm(fes)
    m += u * v * dx
    m.Assemble()
    print(f"cavity_ndof={fes.ndof}")
    print(f"analytic_lowest_k2={K2_LOWEST:.6f}")

    def solve(shift):
        gf = GridFunction(fes, multidim=NVEC)
        vecs = [gf.vecs[i] for i in range(NVEC)]

        def _go():
            return sorted(float(l.real) for l in ArnoldiSolver(
                a.mat, m.mat, fes.FreeDofs(), vecs, shift=shift))

        return _capture_fds(_go)

    ev0, exc0, out0 = solve(SHIFT_BAD)
    print(f"shift0_raised={exc0!r}")
    print(f"shift0_names_umfpackinverse={'UmfpackInverse' in (exc0 or '')}")
    print(f"shift0_captured_output={out0.strip()!r}")
    print(f"matrix_is_singular_not_available_at_the_call_site="
          f"{'matrix is singular' not in out0.lower()}")
    print(f"exception_is_the_usable_signal="
          f"{exc0 is not None and 'UmfpackInverse' in exc0}")

    evk, exck, outk = solve(K2_LOWEST)
    print(f"shift_at_analytic_k2_raised={exck!r}")
    print(f"shift_near_analytic_completes={exck is None}")
    if evk is None:
        print("FAIL: the analytic-shift run did not complete", file=sys.stderr)
        return 2
    nonzero = [e for e in evk if e > 1e-6 * max(evk)]
    zeros = [e for e in evk if abs(e) <= 1e-6 * max(evk)]
    print(f"zero_modes_returned={len(zeros)}")
    print(f"operator_has_the_gradient_kernel={len(zeros) > 0}")
    lowest = min(nonzero)
    rel = abs(lowest - K2_LOWEST) / K2_LOWEST
    print(f"lowest_resonance={lowest:.6f} analytic={K2_LOWEST:.6f} "
          f"relerr={rel:.4e}")
    print(f"lowest_resonance_matches_the_cavity_formula={rel < 0.05}")
    second = sorted(nonzero)[1]
    gap = abs(second - lowest) / K2_LOWEST
    print(f"second_nonzero={second:.6f} relative_gap={gap:.4e}")
    print(f"lowest_mode_is_doubly_degenerate={gap < 0.05}")

    evf, excf, outf = solve(500.0)
    print(f"far_shift_raised={excf!r}")
    print(f"far_shift_still_completes={excf is None}")
    print(f"only_shift_zero_fails={exc0 is not None and excf is None}")

    ok = (
        exc0 is not None and "UmfpackInverse" in exc0
        and "matrix is singular" not in out0.lower()
        and exck is None
        and len(zeros) > 0
        and rel < 0.05 and gap < 0.05
        and excf is None
    )
    if ok:
        return 0
    print("FAIL: cavity-resonance shift invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
