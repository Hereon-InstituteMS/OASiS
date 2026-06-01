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
"""
from __future__ import annotations

import sys

import ngsolve


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
