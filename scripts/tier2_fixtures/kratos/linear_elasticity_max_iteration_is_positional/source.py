"""Tier-2: max_iteration is a strategy ctor argument, absent from the criterion.

The claim's advice is to raise max_iteration and to know where
it lives. Both halves are checkable: the strategy demands it
positionally, and the criterion does not carry it.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA



def _nr(max_iteration):
    """Build the strategy with, or without, the positional max_iteration."""
    model = KM.Model()
    mp = model.CreateModelPart("s")
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    mp.AddNodalSolutionStepVariable(KM.REACTION)
    mp.SetBufferSize(2)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    mp.CreateNewNode(2, 1.0, 0.0, 0.0)
    for n in mp.Nodes:
        n.AddDof(KM.DISPLACEMENT_X, KM.REACTION_X)
        n.AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y)
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    lin = KM.LinearSolverFactory().Create(
        KM.Parameters('{"solver_type":"skyline_lu_factorization"}'))
    bs = KM.ResidualBasedBlockBuilderAndSolver(lin)
    conv = KM.DisplacementCriteria(1e-6, 1e-9)
    if max_iteration is None:
        return KM.ResidualBasedNewtonRaphsonStrategy(mp, scheme, conv, bs)
    return KM.ResidualBasedNewtonRaphsonStrategy(
        mp, scheme, conv, bs, max_iteration, False, False, False)


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
    if not _probe('strategy_ctor_with_max_iteration', lambda: ((lambda: _nr(30))()), True):
        bad += 1
    if not _probe('strategy_ctor_without_max_iteration', lambda: ((lambda: _nr(None))()), False):
        bad += 1
    if not _probe('ResidualCriteria_has_max_iteration', lambda: (hasattr(KM.ResidualCriteria(1e-6, 1e-9), 'max_iteration')), False):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
