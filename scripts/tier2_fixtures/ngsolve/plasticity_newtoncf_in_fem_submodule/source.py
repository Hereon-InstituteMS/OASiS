"""Tier-2: NewtonCF/MinimizationCF live in ngsolve.fem, not top-level.

Catalog claim under audit (from the NGSolve plasticity pitfalls
list): 'NewtonCF/MinimizationCF in NGSolve can handle nonlinear
material at integration point level'.

The wording implies these symbols are top-level NGSolve names.
In NGSolve 6.2.2604 they are NOT — they live in the
ngsolve.fem submodule and are NOT re-exported by
`from ngsolve import *`. An LLM agent that pastes the catalog
hint into
  from ngsolve import NewtonCF
or
  ngsolve.NewtonCF(...)
hits ImportError / AttributeError respectively.

The correct access is
  from ngsolve.fem import NewtonCF, MinimizationCF

This fixture asserts:
  * hasattr(ngsolve, 'NewtonCF') is False
  * hasattr(ngsolve, 'MinimizationCF') is False
  * `from ngsolve import *` does NOT add NewtonCF /
    MinimizationCF to the importing namespace
  * `from ngsolve.fem import NewtonCF, MinimizationCF` works
  * Both are callable

Mutation control (re-runnable, INVERTED POLARITY): this fixture only ever
executes the FIX, so the discriminating edit is to COMMIT the documented
mistake.  T2_MUTATE=1 replaces the correct
`from ngsolve.fem import MinimizationCF, NewtonCF` with the catalog-implied
`from ngsolve import MinimizationCF, NewtonCF`, which raises ImportError (caught
so the report survives) and leaves both names unbound.  The fixture then goes red
on submodule_NewtonCF_callable=True and submodule_MinimizationCF_callable=True.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import ngsolve

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    top_newton = hasattr(ngsolve, "NewtonCF")
    top_minim = hasattr(ngsolve, "MinimizationCF")
    print(f"top_level_NewtonCF={top_newton}")
    print(f"top_level_MinimizationCF={top_minim}")

    # `from ngsolve import *` exposure check
    ns: dict = {}
    exec("from ngsolve import *", ns)  # noqa: S102
    star_newton = "NewtonCF" in ns
    star_minim = "MinimizationCF" in ns
    print(f"star_import_NewtonCF={star_newton}")
    print(f"star_import_MinimizationCF={star_minim}")

    if top_newton or top_minim or star_newton or star_minim:
        print("FAIL: NewtonCF/MinimizationCF reachable from "
              "top-level — catalog wording aligned, fixture "
              "should be retired.", file=sys.stderr)
        return 2

    # Correct path
    if MUTATE:
        # the documented mistake: reach for the names at top level
        try:
            from ngsolve import MinimizationCF, NewtonCF
        except ImportError as exc:
            print(f"top_level_import_raised={type(exc).__name__}: {exc}")
            MinimizationCF = NewtonCF = None
    else:
        from ngsolve.fem import MinimizationCF, NewtonCF
    print(f"submodule_NewtonCF_callable={callable(NewtonCF)}")
    print(f"submodule_MinimizationCF_callable={callable(MinimizationCF)}")
    print(f"submodule_NewtonCF_kind="
          f"{type(NewtonCF).__name__}")

    if callable(NewtonCF) and callable(MinimizationCF):
        return 0
    print("FAIL: submodule symbols not callable", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
