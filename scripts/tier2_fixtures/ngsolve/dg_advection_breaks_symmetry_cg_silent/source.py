"""Tier-2: advection makes the DG matrix unsymmetric, and CG on it does NOT
raise -- it runs to the iteration cap and hands back a field two orders of
magnitude wrong, with no error and no warning.

Claim: ngsolve dg_methods#6 -- "DG bilinear form is NOT symmetric when advection
is present (upwind term is one-sided).  Signal: feeding the assembled matrix to
a CG solver raises a 'matrix not positive definite' error or returns wildly
wrong iterates; switch to GMRES (or BiCGStab) for the unsymmetric system.  Pure
diffusion SIP DG IS symmetric -- advection breaks symmetry."

Wrong variant: CGSolver on the advective (unsymmetric) DG matrix.

CORRECTION this fixture records.  The claim offers two alternative signals; only
one of them happens.  NGSolve 6.2.2604's CGSolver raises NOTHING on this matrix.
There is no 'matrix not positive definite' message anywhere in the run -- the
solve returns normally, having hit its iteration cap, with a result whose
relative error against the direct solve is O(100).  An agent guarding on an
exception, which is the first branch the claim names, sees a clean solve and
ships the wrong field.  That is the entry worth having.

What this fixture pins, all re-measured on this run:
  * the pure-diffusion SIP matrix is symmetric to roundoff, and the same form
    plus advection is not -- measured as max|A - A^T| / max|A|;
  * CG on the symmetric matrix converges below its cap and matches the direct
    solve;
  * CG on the unsymmetric matrix raises no exception, emits no
    positive-definiteness message, consumes its full iteration budget, and
    returns a vector whose relative error against the direct solve is larger
    than the solution itself;
  * GMRES on the SAME matrix and the SAME preconditioner converges in fewer
    iterations than the cap and matches the direct solve.

Mutation control:  T2_MUTATE=1 drops the three advection integrals from build()
so the "with_advection" system is the pure SIP diffusion operator again -- the
symmetry-breaking terms, i.e. the pathology, are removed.  CG then converges and
the expectations advection_breaks_symmetry=True,
cg_exhausted_its_iteration_cap=True,
cg_relative_error_exceeds_the_solution=True and
cg_fails_silently_not_loudly=True all disappear.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import contextlib
import os
import sys
import tempfile

import numpy
import scipy.sparse
from netgen.geom2d import unit_square
from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    GridFunction,
    IfPos,
    L2,
    LinearForm,
    Mesh,
    ds,
    dx,
    grad,
    specialcf,
)
from ngsolve.krylovspace import CGSolver, GMResSolver

ORDER = 1
MAXH = 0.2
MAXITER = 500
NOT_PD_PHRASES = ("not positive definite", "positive definite",
                  "indefinite", "breakdown")

# Mutation control: under T2_MUTATE=1 the advection integrals are never added,
# so the "with_advection" matrix is symmetric -- the pathology is removed.
MUTATE = os.environ.get("T2_MUTATE") == "1"


def build(with_advection):
    mesh = Mesh(unit_square.GenerateMesh(maxh=MAXH))
    fes = L2(mesh, order=ORDER, dgjumps=True)
    u, v = fes.TnT()
    n = specialcf.normal(2)
    h = specialcf.mesh_size
    ju, jv = u - u.Other(), v - v.Other()
    mdu = 0.5 * (grad(u) + grad(u.Other()))
    mdv = 0.5 * (grad(v) + grad(v.Other()))
    alpha = 4 * (ORDER + 1) ** 2

    a = BilinearForm(fes)
    a += grad(u) * grad(v) * dx
    a += alpha / h * ju * jv * dx(skeleton=True)
    a += (-mdu * n * jv - mdv * n * ju) * dx(skeleton=True)
    a += alpha / h * u * v * ds(skeleton=True)
    a += (-grad(u) * n * v - grad(v) * n * u) * ds(skeleton=True)
    if with_advection and not MUTATE:
        b = CoefficientFunction((20.0, 0.0))
        a += -u * (b * grad(v)) * dx
        a += IfPos(b * n, u, u.Other()) * (b * n) * jv * dx(skeleton=True)
        a += IfPos(b * n, u, 0) * (b * n) * v * ds(skeleton=True)
    f = LinearForm(fes)
    f += 1 * v * dx
    a.Assemble()
    f.Assemble()
    return fes, a, f


def relative_asymmetry(a, ndof):
    rows, cols, vals = a.mat.COO()
    A = scipy.sparse.coo_matrix(
        (numpy.asarray(vals), (numpy.asarray(rows), numpy.asarray(cols))),
        shape=(ndof, ndof)).toarray()
    return float(numpy.abs(A - A.T).max() / numpy.abs(A).max())


@contextlib.contextmanager
def captured_streams():
    """Everything the solve writes to fd 1 and fd 2, C++ level included.

    NGSolve's solvers print from compiled code, so redirecting sys.stdout would
    not see them; the file descriptors have to be redirected.  This exists
    because `cg_emitted_positive_definite_message` used to be computed from the
    caught exception's text alone, and that text is "" exactly when the
    co-asserted `cg_raised_on_unsymmetric_matrix=False` holds -- the two
    expectations were one measurement, and neither looked at what the solver
    actually WROTE.  The claim being corrected says CG "raises a 'matrix not
    positive definite' error", and a library can emit that message without
    raising, so both channels have to be read for the correction to mean
    anything.
    """
    sink = tempfile.TemporaryFile(mode="w+b")
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    try:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        yield sink
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        sink.seek(0)


def solve_krylov(kind, fes, a, f, reference):
    gfu = GridFunction(fes)
    cls = CGSolver if kind == "cg" else GMResSolver
    raised = ""
    with captured_streams() as sink:
        try:
            inv = cls(a.mat, a.mat.CreateSmoother(), maxiter=MAXITER, tol=1e-10,
                      printrates=False)
            gfu.vec.data = inv * f.vec
            iters = inv.iterations
        except Exception as exc:                               # noqa: BLE001
            raised = f"{type(exc).__name__}: {exc}"
    printed = sink.read().decode("utf-8", "replace")
    if raised:
        return None, None, raised, printed
    got = gfu.vec.FV().NumPy()
    rel = float(numpy.abs(got - reference).max()
                / max(1e-30, numpy.abs(reference).max()))
    return iters, rel, raised, printed


def main() -> int:
    results = {}
    for label, with_adv in (("diffusion_only", False), ("with_advection", True)):
        fes, a, f = build(with_adv)
        asym = relative_asymmetry(a, fes.ndof)

        ref_vec = f.vec.CreateVector()
        ref_vec.data = a.mat.Inverse(inverse="umfpack") * f.vec
        reference = ref_vec.FV().NumPy().copy()

        it_cg, rel_cg, exc_cg, out_cg = solve_krylov("cg", fes, a, f, reference)
        it_gm, rel_gm, exc_gm, _ = solve_krylov("gmres", fes, a, f, reference)
        results[label] = (fes.ndof, asym, it_cg, rel_cg, exc_cg,
                          it_gm, rel_gm, exc_gm, out_cg)
        print(f"{label}_ndof={fes.ndof}")
        print(f"{label}_relative_asymmetry={asym:.4e}")
        print(f"{label}_cg_iterations={it_cg} cg_relerr={rel_cg}")
        print(f"{label}_cg_exception={exc_cg!r}")
        print(f"{label}_cg_printed_bytes={len(out_cg)}")
        print(f"{label}_gmres_iterations={it_gm} gmres_relerr={rel_gm}")

    _, asym_d, it_cg_d, rel_cg_d, exc_cg_d, _, _, _, _ = \
        results["diffusion_only"]
    _, asym_a, it_cg_a, rel_cg_a, exc_cg_a, it_gm_a, rel_gm_a, _, out_cg_a = \
        results["with_advection"]

    print(f"pure_diffusion_is_symmetric={asym_d < 1e-12}")
    print(f"advection_breaks_symmetry={asym_a > 1e-3}")
    print(f"cg_ok_on_symmetric_matrix={it_cg_d < MAXITER and rel_cg_d < 1e-8}")

    # The correction: no exception, no message, just a wrong answer.
    #
    # "no message" has to be read off the SOLVER'S OUTPUT, not off the
    # exception.  Computing it from `exc_cg_a` alone made it False by
    # construction on any run where `cg_raised_on_unsymmetric_matrix` is False,
    # which is the other half of the same expectation pair -- one measurement
    # printed twice, and neither half ever saw what CGSolver wrote.
    cg_raised = bool(exc_cg_a)
    channels = {"exception": (exc_cg_a or "").lower(),
                "stdout_stderr": out_cg_a.lower()}
    emitted_in = sorted(ch for ch, text in channels.items()
                        if any(p in text for p in NOT_PD_PHRASES))
    pd_message = bool(emitted_in)
    print(f"cg_raised_on_unsymmetric_matrix={cg_raised}")
    print(f"cg_printed_during_solve={out_cg_a.strip()[:200]!r}")
    print(f"cg_positive_definite_message_channels={emitted_in}")
    print(f"cg_emitted_positive_definite_message={pd_message}")
    print(f"cg_exhausted_its_iteration_cap={it_cg_a == MAXITER}")
    print(f"cg_relative_error_exceeds_the_solution={rel_cg_a > 1.0}")
    print(f"cg_fails_silently_not_loudly={not cg_raised and rel_cg_a > 1.0}")
    print(f"gmres_converges_on_the_same_matrix="
          f"{it_gm_a < MAXITER and rel_gm_a < 1e-8}")

    ok = (
        asym_d < 1e-12 and asym_a > 1e-3
        and it_cg_d < MAXITER and rel_cg_d < 1e-8
        and not cg_raised and not pd_message
        and it_cg_a == MAXITER and rel_cg_a > 1.0
        and it_gm_a < MAXITER and rel_gm_a < 1e-8
    )
    if ok:
        return 0
    print("FAIL: DG symmetry / CG invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
