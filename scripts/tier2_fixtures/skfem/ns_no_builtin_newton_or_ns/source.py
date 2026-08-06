"""Tier-2: skfem ships no Newton solver and no Navier-Stokes assembly.

Claim: skfem navier_stokes#0 -- scikit-fem has NO built-in Newton solver or NS
assembly, so it must be built by hand.  Signal: searching skfem.utils for
`NewtonSolver` or `NavierStokes` returns no match; there is no single-call NS
API.

Measured on skfem 12.0.1 by enumerating the namespace rather than trusting two
spellings: every plausible name for a nonlinear solver, a Navier-Stokes
assembly and a time integrator is probed against both `skfem` and
`skfem.utils`, and the whole public surface of skfem.utils is printed.

  * confirmed.  None of the probed names exists in either namespace.
  * the positive half is the useful part: skfem.utils exposes LINEAR and
    EIGEN solvers only -- solve, solve_linear, solve_eigen, the
    solver_iter_* family, condense/enforce/penalize -- and nothing that
    iterates on a nonlinear residual.  So the absence is structural, not a
    naming accident.
  * the pieces a hand-rolled Newton needs ARE all present (BilinearForm,
    LinearForm, asm, condense, solve), which is why the catalog ships
    snippets rather than a call.
"""
from __future__ import annotations

import sys

import skfem
import skfem.utils

NONLINEAR_NAMES = [
    "NewtonSolver", "Newton", "newton", "newton_solve", "solve_newton",
    "NavierStokes", "navier_stokes", "NonlinearProblem", "nonlinear_solve",
    "solve_nonlinear", "NonlinearSolver", "picard", "Picard",
    "TimeStepper", "solve_transient",
]
LINEAR_PIECES = ["solve", "solve_linear", "condense", "enforce", "penalize",
                 "asm", "solve_eigen"]
BUILDING_BLOCKS = ["BilinearForm", "LinearForm", "Basis", "asm", "condense",
                   "solve"]


def main() -> int:
    ok = True
    print(f"skfem_version={skfem.__version__}")

    found = [n for n in NONLINEAR_NAMES
             if hasattr(skfem, n) or hasattr(skfem.utils, n)]
    print(f"nonlinear_names_probed={len(NONLINEAR_NAMES)}")
    print(f"nonlinear_names_found={found}")
    print(f"no_nonlinear_solver_anywhere={not found}")
    if found:
        print(f"FAIL: skfem does ship a nonlinear solver: {found}",
              file=sys.stderr)
        ok = False

    public = sorted(n for n in dir(skfem.utils) if not n.startswith("_"))
    print(f"skfem_utils_public_count={len(public)}")
    print(f"skfem_utils_public={public}")
    present = [n for n in LINEAR_PIECES if hasattr(skfem.utils, n)]
    print(f"linear_and_eigen_solvers_present={present}")
    print(f"utils_offers_only_linear_and_eigen_solvers="
          f"{len(present) >= 5 and not found}")
    if len(present) < 5:
        print("FAIL: the linear solver surface is not what was expected",
              file=sys.stderr)
        ok = False

    blocks = [n for n in BUILDING_BLOCKS if hasattr(skfem, n)]
    print(f"hand_rolled_newton_building_blocks={blocks}")
    print(f"all_building_blocks_present="
          f"{len(blocks) == len(BUILDING_BLOCKS)}")
    if len(blocks) != len(BUILDING_BLOCKS):
        print("FAIL: a piece a hand-rolled Newton needs is missing",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
