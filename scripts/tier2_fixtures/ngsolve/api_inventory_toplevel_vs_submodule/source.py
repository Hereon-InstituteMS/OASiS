"""Tier-2: NGSolve top-level vs submodule API inventory.

Several APIs have NON-OBVIOUS module locations:

  NewtonCF / MinimizationCF  → ngsolve.fem (NOT top level)
  GMResSolver / MinResSolver → ngsolve.krylovspace
  BramblePasciakCG           → ngsolve.krylovspace
  Curl / Div (capital)       → DO NOT EXIST as top-level
                                (only ngs.curl / ngs.div)

This fixture asserts:
  * The 4 'phantom' top-level names stay absent (catching
    any future alias addition that would silently mask
    catalog drift).
  * The real submodule paths resolve.
  * NewtonSolver kwarg name `a` (not `bf` or `bilinear`)
    is the contract.
"""
from __future__ import annotations

import inspect
import sys

import ngsolve as ngs
import ngsolve.fem as fem
import ngsolve.krylovspace as kry
import ngsolve.solvers as sol


PHANTOM_AT_TOPLEVEL = (
    "NewtonCF", "MinimizationCF",
    "Curl", "Div",
    "GMResSolver", "MinResSolver",
)
REQUIRED_FEM = ("NewtonCF", "MinimizationCF", "IfPos")
REQUIRED_KRYLOV = (
    "CGSolver", "GMResSolver", "MinResSolver",
    "BramblePasciakCG",
)
REQUIRED_SOLVERS = (
    "Newton", "CG", "BVP", "PreconditionedRichardson",
)


def main() -> int:
    print(f"ngsolve_version={ngs.__version__}")

    phantom_present = [n for n in PHANTOM_AT_TOPLEVEL
                       if hasattr(ngs, n)]
    print(f"phantom_at_toplevel={phantom_present}")

    fem_missing = [n for n in REQUIRED_FEM
                   if not hasattr(fem, n)]
    krylov_missing = [n for n in REQUIRED_KRYLOV
                      if not hasattr(kry, n)]
    solvers_missing = [n for n in REQUIRED_SOLVERS
                       if not hasattr(sol, n)]
    print(f"fem_missing={fem_missing}")
    print(f"krylov_missing={krylov_missing}")
    print(f"solvers_missing={solvers_missing}")

    # Lowercase differential ops MUST exist
    has_curl_lower = hasattr(ngs, "curl")
    has_div_lower = hasattr(ngs, "div")
    has_grad_lower = hasattr(ngs, "grad")
    print(f"lowercase_diff_ops={dict(curl=has_curl_lower, div=has_div_lower, grad=has_grad_lower)}")

    # Newton kwarg name contract
    sig = inspect.signature(sol.Newton)
    params = list(sig.parameters)
    newton_first_two = params[:2]
    print(f"newton_first_two_params={newton_first_two}")
    has_a_kwarg = "a" in sig.parameters
    has_u_kwarg = "u" in sig.parameters

    ok = (
        not phantom_present
        and not fem_missing
        and not krylov_missing
        and not solvers_missing
        and has_curl_lower
        and has_div_lower
        and has_grad_lower
        and has_a_kwarg
        and has_u_kwarg
        and newton_first_two == ["a", "u"]
    )
    if ok:
        return 0
    print("FAIL: ngsolve API inventory regression",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
