"""Tier-2: NGSolve Dirichlet boundary name with wrong capitalisation
silently produces NO Dirichlet constraints.

Pitfall (NGSolve poisson#0): The `dirichlet=` argument of
H1/HCurl/etc. must match the mesh's boundary labels EXACTLY
(case-sensitive). netgen's unit_square uses lowercase 'left',
'right', 'top', 'bottom'. Passing 'Left|Right|Top|Bottom'
(wrong case) silently leaves ALL DoFs free — no exception
fires. The pitfall is observable only via FreeDofs() count.

Mutation control: T2_MUTATE=1 applies the documented fix at the
pathology site — the `wrong` space is built with the mesh's actual
lowercase labels instead of the capitalised ones — so its dofs get
constrained and the pathological counts vanish. Re-run with
`T2_MUTATE=1 python source.py`.
"""
from __future__ import annotations

import os
import sys

from netgen.geom2d import unit_square
from ngsolve import H1, Mesh

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    correct = H1(mesh, order=1, dirichlet="left|right|top|bottom")
    # The pathology: capitalised labels that no boundary carries.
    # T2_MUTATE=1 corrects the capitalisation at this call site.
    wrong_labels = ("left|right|top|bottom" if MUTATE
                    else "Left|Right|Top|Bottom")
    wrong = H1(mesh, order=1, dirichlet=wrong_labels)

    correct_n_free = sum(bool(f) for f in correct.FreeDofs())
    wrong_n_free = sum(bool(f) for f in wrong.FreeDofs())
    wrong_n_dirichlet = wrong.ndof - wrong_n_free

    print(f"correct_case_n_free={correct_n_free}")
    print(f"wrong_case_n_free={wrong_n_free}")
    print(f"wrong_case_n_dirichlet={wrong_n_dirichlet}")
    print(f"ndof={correct.ndof}")

    # Behaviour we expect under the pitfall:
    #   - correct case: all DoFs constrained → free=0
    #   - wrong case:   no DoFs constrained → free=ndof
    if correct_n_free == 0 and wrong_n_free == correct.ndof:
        return 0
    print("ERROR: did not observe the case-mismatch silent-failure "
          "pattern. catalog claim may be wrong or backend changed.",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
