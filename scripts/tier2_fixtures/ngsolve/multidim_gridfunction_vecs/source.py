"""Tier-2: NGSolve GridFunction multidim access via gfu.vecs.

Pitfall (NGSolve eigenvalue#1): A GridFunction declared with
multidim=n holds n independent vectors. The correct access
path is gfu.vecs (a list-like sequence with length n). There
is NO gfu.mdcomponents attribute — that name does not exist
on the binding.

Mutation control: T2_MUTATE=1 drops the `multidim=4` argument from
the GridFunction constructor — the omission a user makes when they
expect a plain GridFunction to hold several vectors — so only one
vector is allocated and gfu.vecs has length 1. Re-run with
`T2_MUTATE=1 python source.py`.
"""
from __future__ import annotations

import os
import sys

from netgen.geom2d import unit_square
from ngsolve import H1, Mesh, GridFunction

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    fes = H1(mesh, order=1, dirichlet="left|right|top|bottom")
    # multidim=n is what allocates the n vectors gfu.vecs exposes.
    # T2_MUTATE=1 removes that argument at this call site.
    gfu = GridFunction(fes) if MUTATE else GridFunction(fes, multidim=4)

    has_vecs = hasattr(gfu, "vecs")
    has_md = hasattr(gfu, "mdcomponents")
    n_vecs = len(list(gfu.vecs)) if has_vecs else None

    print(f"vecs_present={has_vecs}")
    print(f"n_vecs={n_vecs}")
    print(f"mdcomponents_present={has_md}")

    if has_vecs and n_vecs == 4 and not has_md:
        return 0
    print("ERROR: did not observe expected vecs/mdcomponents pattern",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
