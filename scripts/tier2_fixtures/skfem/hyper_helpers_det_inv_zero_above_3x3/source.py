"""Tier-2: skfem.helpers.det/inv return all-zeros above 3x3, silently.

Claim: skfem hyperelasticity#11 -- skfem.helpers.det() and inv() SILENTLY
return all-zeros for ANY square matrix whose leading dimension is NOT 2 or 3,
because helpers.py branches on A.shape[0] == 2 / == 3 with no else and the
initial `detA = zeros_like(A[0, 0])` sticks.  No NotImplementedError, no
warning.  Signal: det(A).any() is False for A.shape[0] >= 4.

Measured on skfem 12.0.1.  Confirmed, and the downstream consequence is
reproduced: a hyperelastic-style kernel that computes log(det(F)) on a 4x4
augmented tensor gets log(0) = -inf and an inf/NaN residual, with the only
warning coming from numpy's own log, not from skfem.

The fixture checks the source text as well as the behaviour, so it stays
honest if a future release adds the missing else-branch.
"""
from __future__ import annotations

import inspect
import sys
import warnings

import numpy as np
import skfem
import skfem.helpers as helpers
from skfem.helpers import det, inv


def diag_field(d, value=2.0, nelem=4, nqp=3):
    A = np.zeros((d, d, nelem, nqp))
    for i in range(d):
        A[i, i] = value
    return A


def main() -> int:
    ok = True
    print(f"skfem_version={skfem.__version__}")

    results = {}
    for d in (2, 3, 4, 5, 6):
        A = diag_field(d)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dA = np.asarray(det(A))
            iA = np.asarray(inv(A))
            msgs = [str(c.message) for c in caught]
        analytic = 2.0 ** d
        nonzero = bool(dA.any())
        results[d] = nonzero
        print(f"d{d}_det_any_nonzero={nonzero}")
        print(f"d{d}_det_first={float(dA.ravel()[0])} analytic={analytic}")
        print(f"d{d}_inv_any_nonzero={bool(iA.any())}")
        print(f"d{d}_warnings={msgs!r}")

    print(f"det_correct_for_2x2_and_3x3="
          f"{results[2] and results[3]}")
    print(f"det_zero_for_4x4_and_above="
          f"{not results[4] and not results[5] and not results[6]}")
    if not (results[2] and results[3]):
        print("FAIL: det is wrong even at 2x2/3x3", file=sys.stderr)
        ok = False
    if results[4] or results[5] or results[6]:
        print("FAIL: det no longer returns zeros above 3x3 -- the silent "
              "branch has been fixed upstream", file=sys.stderr)
        ok = False

    # --- nothing is raised, which is the whole point ------------------
    raised = None
    try:
        det(diag_field(4))
        inv(diag_field(4))
    except Exception as e:            # noqa: BLE001 -- we want ANY exception
        raised = e
    print(f"det_inv_raise_on_4x4={raised is not None}")
    print(f"silent_no_exception={raised is None}")
    if raised is not None:
        print(f"FAIL: a {type(raised).__name__} was raised, so the failure is "
              f"not silent", file=sys.stderr)
        ok = False

    # --- the source really has no else-branch -------------------------
    src = inspect.getsource(helpers.det)
    print(f"det_source_has_shape0_eq_3={'A.shape[0] == 3' in src}")
    print(f"det_source_has_shape0_eq_2={'A.shape[0] == 2' in src}")
    print(f"det_source_has_else={'else:' in src}")
    print(f"det_source_seeds_zeros="
          f"{'zeros_like(A[0, 0])' in src}")
    if "else:" in src:
        print("FAIL: helpers.det now has an else-branch", file=sys.stderr)
        ok = False

    # --- the downstream damage: log(det(F)) on an augmented tensor ----
    F4 = diag_field(4, value=1.1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        J4 = np.asarray(det(F4))
        lnJ4 = np.log(J4)
        log_msgs = [str(c.message) for c in caught]
    print(f"augmented_J_is_zero={bool((J4 == 0).all())}")
    print(f"augmented_lnJ_is_neg_inf={bool(np.isneginf(lnJ4).all())}")
    print(f"only_numpy_warned={log_msgs!r}")
    if not np.isneginf(lnJ4).all():
        print("FAIL: log(det(F)) on a 4x4 tensor was not -inf",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
