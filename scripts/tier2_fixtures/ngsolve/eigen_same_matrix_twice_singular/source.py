"""Tier-2: handing ArnoldiSolver the same matrix twice does not fall back to an
identity mass -- it forms a pencil that is exactly singular at the shift.

Claim: ngsolve eigenvalue#3 -- "For the generalized eigenvalue problem
A*x = lambda*M*x, pass BOTH matrices to ArnoldiSolver as a.mat and m.mat -- the
signature requires two, so 'passing only A' is not a reachable mistake.  Passing
the SAME matrix twice does NOT fall back to an identity mass (the prior catalog
mechanism was wrong): it forms the pencil (A, A)."

Wrong variant: ArnoldiSolver(a.mat, a.mat, ...).

What this fixture pins, all re-measured on this run:
  * the signature really does require two matrices: calling with one raises
    TypeError, so "passing only A" is not something an agent can do by accident;
  * passing the same matrix twice does NOT silently return the ordinary
    spectrum -- it raises, from the shift-and-invert factorisation, and the
    message says the matrix is singular;
  * the reason is arithmetic and is checked here: the pencil A - shift*A is
    (1 - shift)*A, which at shift = 1 is exactly the zero matrix.  The fixture
    verifies that by assembling A - shift*A itself and measuring its norm;
  * an identity-mass fallback would have produced the spectrum of A alone, and
    that spectrum is computed separately here to show it is a perfectly
    well-defined set of numbers -- so the failure is not "there was nothing to
    return";
  * the correct call with the true mass matrix returns the generalised
    eigenvalues, which differ from A's own spectrum, so the mass matrix is not
    cosmetic.
"""
from __future__ import annotations

import sys

import numpy
import scipy.sparse
from netgen.geom2d import unit_square
from ngsolve import (
    ArnoldiSolver,
    BilinearForm,
    GridFunction,
    H1,
    Mesh,
    dx,
    grad,
)

NVEC = 8
SHIFT = 1.0


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes = H1(mesh, order=2, dirichlet=".*")
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += grad(u) * grad(v) * dx
    a.Assemble()
    m = BilinearForm(fes)
    m += u * v * dx
    m.Assemble()
    gf = GridFunction(fes, multidim=NVEC)
    vecs = [gf.vecs[i] for i in range(NVEC)]

    one_arg = ""
    try:
        ArnoldiSolver(a.mat, fes.FreeDofs(), vecs, shift=SHIFT)
    except Exception as exc:                                   # noqa: BLE001
        one_arg = type(exc).__name__
    print(f"single_matrix_call_raises={one_arg}")
    print(f"passing_only_A_is_not_reachable={one_arg == 'TypeError'}")

    correct = sorted(float(l.real) for l in ArnoldiSolver(
        a.mat, m.mat, fes.FreeDofs(), vecs, shift=SHIFT))[:4]
    print(f"generalised_eigenvalues={[round(c, 6) for c in correct]}")

    # A true identity mass, assembled as a matrix of the same class so
    # ArnoldiSolver accepts it (an IdentityMatrix raises NgException
    # 'BaseMatrix::AsVector not overloaded').
    im = BilinearForm(fes)
    im += u * v * dx
    im.Assemble()
    rows_i, cols_i, _ = im.mat.COO()
    im.mat.AsVector()[:] = 0.0
    for i in range(fes.ndof):
        im.mat[i, i] = 1.0
    ident = sorted(float(l.real) for l in ArnoldiSolver(
        a.mat, im.mat, fes.FreeDofs(), vecs, shift=SHIFT))[:4]
    print(f"identity_mass_eigenvalues={[round(c, 6) for c in ident]}")
    differ = max(abs(c - i) / max(abs(c), 1e-30)
                 for c, i in zip(correct, ident))
    print(f"identity_vs_true_mass_max_relative_difference={differ:.3e}")
    print(f"an_identity_fallback_would_have_been_visible={differ > 1e-3}")

    same = ""
    msg = ""
    try:
        ArnoldiSolver(a.mat, a.mat, fes.FreeDofs(), vecs, shift=SHIFT)
    except Exception as exc:                                   # noqa: BLE001
        same = type(exc).__name__
        msg = str(exc)
    print(f"same_matrix_twice_raises={same}")
    print(f"same_matrix_message_mentions_factorization="
          f"{'factorization failed' in msg.lower()}")
    print(f"same_matrix_did_not_silently_return={bool(same)}")
    print(f"no_identity_fallback_happened={bool(same)}")

    # Why: A - shift*A is exactly zero at shift = 1.
    rows, cols, vals = a.mat.COO()
    A = scipy.sparse.coo_matrix(
        (numpy.asarray(vals), (numpy.asarray(rows), numpy.asarray(cols))),
        shape=(fes.ndof, fes.ndof)).toarray()
    pencil = A - SHIFT * A
    print(f"pencil_A_minus_shift_A_max_abs={numpy.abs(pencil).max():.3e}")
    print(f"pencil_is_exactly_zero_at_shift_one="
          f"{float(numpy.abs(pencil).max()) == 0.0}")
    print(f"A_itself_is_not_zero={float(numpy.abs(A).max()) > 1.0}")

    ok = (
        one_arg == "TypeError"
        and differ > 1e-3
        and bool(same)
        and "factorization failed" in msg.lower()
        and float(numpy.abs(pencil).max()) == 0.0
        and float(numpy.abs(A).max()) > 1.0
    )
    if ok:
        return 0
    print("FAIL: ArnoldiSolver pencil invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
