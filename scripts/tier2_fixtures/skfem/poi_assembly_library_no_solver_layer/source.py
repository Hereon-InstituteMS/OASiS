"""Tier-2: scikit-fem is an assembly library; there is no solver layer.

Claim: skfem poisson#1 -- laplace/unit_load build K and f, then you call
solve(*condense(...)) yourself. No SolverInterface, no LinearProblem, no KSP
wrapper. K is a scipy.sparse.csr_matrix and swapping the solver is a one-line
change.

Wrong variant: reaching for the object-oriented solver API that other FEM
libraries have. `skfem.LinearProblem` raises AttributeError -- there is nothing
to configure, which is the point of the claim.

Mutation control: T2_MUTATE=1 changes the probed attribute name from
"LinearProblem" to "solve", i.e. it applies the documented fix (call
skfem.solve yourself instead of reaching for a solver object). The lookup then
succeeds, no AttributeError is raised, and the expectation
"no attribute 'LinearProblem'" is no longer in the output, so the fixture goes
red. Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import skfem
from skfem import Basis, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace, unit_load

MUTATE = os.environ.get("T2_MUTATE") == "1"

# THE PATHOLOGY: there is no solver-object API to reach for.  The documented
# fix is to call skfem.solve (+ condense) yourself instead.
PROBE = "LinearProblem" if not MUTATE else "solve"


def main() -> int:
    ok = True

    # --- WRONG variant: look for a solver-object API --------------------
    for name in ("LinearProblem", "SolverInterface", "KSP", "NonlinearProblem"):
        present = hasattr(skfem, name)
        print(f"skfem_has_{name}={present}")
        if present:
            print(f"FAIL: skfem unexpectedly exposes {name}", file=sys.stderr)
            ok = False
    raised = ""
    try:
        getattr(skfem, PROBE)          # deliberate probe
    except AttributeError as exc:
        raised = str(exc)
    print(f"attributeerror_text={raised!r}")
    if "no attribute 'LinearProblem'" not in raised:
        print(f"FAIL: expected the module AttributeError, got {raised!r}",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant: assemble, condense, pick any scipy solver -------
    basis = Basis(MeshTri().refined(3), ElementTriP1())
    K = laplace.assemble(basis)
    f = unit_load.assemble(basis)
    print(f"K_type_name={type(K).__name__}")
    print(f"K_is_scipy_sparse={sp.issparse(K)}")
    print(f"K_shape_is_N_by_N={K.shape == (basis.N, basis.N)}")
    if type(K).__name__ != "csr_matrix" or K.shape != (basis.N, basis.N):
        print(f"FAIL: K is {type(K)!r} of shape {K.shape!r}", file=sys.stderr)
        ok = False

    Kc, fc, u_tmpl, I = condense(K, f, D=basis.get_dofs())
    direct = sla.spsolve(Kc, fc)
    lu = sla.factorized(Kc.tocsc())(fc)
    itr, info = sla.cg(Kc, fc, rtol=1e-12, maxiter=5000)
    agree = (np.allclose(direct, lu, rtol=1e-10)
             and np.allclose(direct, itr, rtol=1e-6) and info == 0)
    print(f"cg_info={info}")
    print(f"spsolve_factorized_cg_all_agree={agree}")
    if not agree:
        print("FAIL: the three scipy solvers disagree on the condensed system",
              file=sys.stderr)
        ok = False

    u = solve(*condense(K, f, D=basis.get_dofs()))
    print(f"solve_returns_full_length={len(u) == basis.N}")
    if len(u) != basis.N:
        print(f"FAIL: solve returned {len(u)} entries, basis.N={basis.N}",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
