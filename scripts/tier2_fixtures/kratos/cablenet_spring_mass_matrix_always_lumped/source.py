"""Tier-2: the spring's mass matrix is diagonal whatever the flag says.

A consistent mass matrix has non-zero off-diagonal coupling.
This one has none, and the request for one is silently
discarded — so an explicit-dynamics or modal run gets the
lumped answer while believing it asked otherwise.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import KratosMultiphysics.CableNetApplication as CN



def _spring(coords=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))):
    model = KM.Model()
    mp = model.CreateModelPart("sp")
    for v in (KM.DISPLACEMENT, KM.REACTION):
        mp.AddNodalSolutionStepVariable(v)
    mp.SetBufferSize(2)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    for i, (x, y, z) in enumerate(coords):
        mp.CreateNewNode(i + 1, x, y, z)
    prop = mp.CreateNewProperties(1)
    prop.SetValue(KM.YOUNG_MODULUS, 2.1e11)
    prop.SetValue(KM.DENSITY, 7850.0)
    prop.SetValue(SMA.CROSS_AREA, 1e-4)
    prop.SetValue(CN.SPRING_DEFORMATION_EMPIRICAL_POLYNOMIAL,
                  KM.Vector([1000.0, 0.0]))
    el = mp.CreateNewElement("EmpiricalSpringElement3D2N", 1, [1, 2], prop)
    el.Initialize(mp.ProcessInfo)
    return mp, el


def _max_offdiag(lumped_flag):
    mp, el = _spring()
    if lumped_flag is not None:
        mp.ProcessInfo.SetValue(KM.COMPUTE_LUMPED_MASS_MATRIX, lumped_flag)
    M = KM.Matrix(6, 6)
    el.CalculateMassMatrix(M, mp.ProcessInfo)
    return max(abs(M[i, j]) for i in range(6) for j in range(6) if i != j)


def _max_diag():
    mp, el = _spring()
    M = KM.Matrix(6, 6)
    el.CalculateMassMatrix(M, mp.ProcessInfo)
    return max(abs(M[i, i]) for i in range(6))


# (label, callable-returning-value, must_succeed)
def _probe(label, fn, must_succeed):
    """A probe FAILS if it raises, and equally if it returns False.

    Both matter: some claims are 'this attribute does not resolve'
    (an exception) and some are 'this predicate is false' (a bool).
    Treating a returned False as success would let a fixture report a
    pass while observing the opposite of what it claims.
    """
    try:
        val = fn()
        ok = val is not False
        err = "" if ok else "returned False"
    except Exception as exc:
        ok = False
        err = f"{type(exc).__name__}: {str(exc).strip().splitlines()[0] if str(exc).strip() else ''}"
    print(f"probe[{label}]={ok}_expected={must_succeed}")
    if not ok:
        print(f"  message: {err[:150]}")
    return ok == must_succeed


def main() -> int:
    bad = 0
    if not _probe('mass_matrix_has_nonzero_diagonal', lambda: (_max_diag() > 0.0), True):
        bad += 1
    if not _probe('default_offdiag_is_zero', lambda: (_max_offdiag(None) == 0.0), True):
        bad += 1
    if not _probe('offdiag_still_zero_with_flag_False', lambda: (_max_offdiag(False) == 0.0), True):
        bad += 1
    if not _probe('offdiag_nonzero_with_flag_False', lambda: (_max_offdiag(False) > 0.0), False):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
