"""Tier-2: csr.sum(axis=1) returns np.matrix, and dividing by it is silent.

Claim: skfem wave#3 -- mass.assemble(basis) returns a scipy.sparse.csr_matrix;
summing along axis=1 gives a numpy.matrix in NumPy < 2.0 and an ndarray in
NumPy >= 2.0.  "Use np.asarray(...).ravel() to coerce to 1-D regardless.
Signal: TypeError 'unsupported operand type(s) for /' with ndarray / matrix on
the M_lumped division line, or DeprecationWarning 'matrix subclass is
deprecated' from numpy."

Measured on skfem 12.0.1 / numpy 1.26.4:

  * the TYPE half is right: M.sum(axis=1) is a numpy.matrix of shape (N, 1).
  * BOTH QUOTED SIGNALS ARE ABSENT.  ndarray / matrix does NOT raise a
    TypeError, and no DeprecationWarning about the matrix subclass is
    emitted.  What happens instead is the dangerous thing: the (N,)
    ndarray broadcasts against the (N, 1) matrix and the result is an
    (N, N) numpy.matrix -- an outer-product-shaped object where a length-N
    vector was intended.
  * the damage shows up further downstream and is invisible by shape.  Using
    that (N, N) object as the inverse lumped mass in a wave acceleration
    still returns a length-N vector -- so a shape check passes -- but the
    values are O(10) away from the correct elementwise division, with no
    warning at all.
  * np.asarray(...).ravel() gives the (N,) array, and the lumped diagonal it
    produces sums to the domain area.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
from skfem import Basis, ElementQuad1, MeshQuad
from skfem.models.poisson import mass


def main() -> int:
    ok = True
    print(f"numpy_version={np.__version__}")
    m = MeshQuad.init_tensor(np.linspace(0.0, 1.0, 5),
                             np.linspace(0.0, 1.0, 5))
    ib = Basis(m, ElementQuad1())
    M = mass.assemble(ib)
    print(f"dofs_N={ib.N}")
    print(f"mass_type={type(M).__name__}")
    print(f"mass_is_csr={type(M).__name__ == 'csr_matrix'}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = M.sum(axis=1)
        msgs = sorted({str(c.message) for c in caught})
    print(f"rowsum_type={type(s).__name__}")
    print(f"rowsum_is_np_matrix={isinstance(s, np.matrix)}")
    print(f"rowsum_shape={np.shape(s)}")
    print(f"rowsum_warnings={msgs!r}")
    print(f"deprecation_warning_emitted="
          f"{any('matrix subclass' in x for x in msgs)}")
    if not isinstance(s, np.matrix):
        print("FAIL: sum(axis=1) is not a numpy.matrix on this numpy",
              file=sys.stderr)
        ok = False
    if any("matrix subclass" in x for x in msgs):
        print("FAIL: a matrix-subclass DeprecationWarning WAS emitted, so "
              "the quoted signal fires after all", file=sys.stderr)
        ok = False

    # --- the division the claim says raises -----------------------------
    err = None
    result = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = np.ones(ib.N) / s
        except TypeError as e:
            err = e
        div_msgs = sorted({str(c.message) for c in caught})
    print(f"division_raises_TypeError={isinstance(err, TypeError)}")
    print(f"division_warnings={div_msgs!r}")
    if err is not None:
        print(f"FAIL: ndarray / matrix raised {err}", file=sys.stderr)
        ok = False
    else:
        print(f"division_result_type={type(result).__name__}")
        print(f"division_result_shape={np.shape(result)}")
        print(f"division_result_is_N_by_N="
              f"{np.shape(result) == (ib.N, ib.N)}")
        print(f"division_silently_broadcasts_to_a_matrix="
              f"{isinstance(result, np.matrix) and np.shape(result)[0] > 1}")
        if np.shape(result) != (ib.N, ib.N):
            print("FAIL: the division did not silently broadcast to (N, N)",
                  file=sys.stderr)
            ok = False

    # --- and it produces a wrong acceleration, silently ------------------
    from skfem.models.poisson import laplace
    K = laplace.assemble(ib).tocsr()
    u = ib.project(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    force = -(K @ u)
    correct = force / np.asarray(s).ravel()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrong = np.asarray(np.asarray(result) @ force).ravel()
        acc_msgs = sorted({str(c.message) for c in caught})
    print(f"correct_acceleration_shape={correct.shape}")
    print(f"wrong_acceleration_shape={wrong.shape}")
    print(f"wrong_acceleration_still_length_N={wrong.shape == (ib.N,)}")
    dev = float(np.abs(wrong - correct).max())
    print(f"acceleration_max_abs_deviation={dev:.4e}")
    print(f"acceleration_is_silently_wrong={dev > 1e-6}")
    print(f"acceleration_warnings={acc_msgs!r}")
    if dev <= 1e-6:
        print("FAIL: the broadcast object gave the correct acceleration",
              file=sys.stderr)
        ok = False
    if acc_msgs:
        print(f"FAIL: the wrong acceleration warned {acc_msgs!r}",
              file=sys.stderr)
        ok = False

    # --- the documented fix ----------------------------------------------
    flat = np.asarray(s).ravel()
    print(f"asarray_ravel_shape={flat.shape}")
    print(f"asarray_ravel_is_length_N={flat.shape == (ib.N,)}")
    print(f"lumped_diagonal_sum={float(flat.sum()):.6f}")
    print(f"lumped_diagonal_sums_to_domain_area="
          f"{abs(float(flat.sum()) - 1.0) < 1e-12}")
    good = np.ones(ib.N) / flat
    print(f"fixed_division_shape={good.shape}")
    print(f"fixed_division_is_length_N={good.shape == (ib.N,)}")
    if flat.shape != (ib.N,) or good.shape != (ib.N,):
        print("FAIL: np.asarray(...).ravel() did not fix the shape",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
