"""Tier-2: NGSolve compound FESpace TnT() unpack structure.

Pitfall (NGSolve stokes#1): For a Taylor-Hood-style mixed space
X = FESpace([V, Q]) (e.g. VectorH1 * H1), X.TnT() returns a
2-tuple where each element is a LIST of ProxyFunctions. The
recommended unpack is '(u, p), (v, q) = X.TnT()'.

Verifies the exact result types so a future NGSolve API change
that swaps lists for tuples (or flattens the structure) is
caught.

Mutation control (INVERTED POLARITY): this fixture only ever
demonstrates the correct unpack, so there is no pathology to
remove -- the control commits the documented mistake instead.
T2_MUTATE=1 unpacks the compound TnT() with the single-space
idiom `u, v = X.TnT()` rather than `(u, p), (v, q) = X.TnT()`,
so `u` binds the whole trial LIST instead of a proxy and the
fixture loses u_type=ProxyFunction and goes red.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

from netgen.geom2d import unit_square
from ngsolve import H1, FESpace, Mesh

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    V = H1(mesh, order=2, dirichlet="left|right|top|bottom")
    Q = H1(mesh, order=1)
    X = FESpace([V, Q])
    result = X.TnT()
    top_t = type(result).__name__
    inner_t = type(result[0]).__name__
    inner_len = len(result[0])
    # MUTATION SITE: the unpack idiom itself.  T2_MUTATE=1 commits the
    # documented mistake -- the single-space unpack on a compound space.
    if MUTATE:
        u, v = result
    else:
        (u, p), (v, q) = result
    u_t = type(u).__name__

    print(f"top_type={top_t}")
    print(f"inner_type={inner_t}")
    print(f"inner_len={inner_len}")
    print(f"u_type={u_t}")

    ok = (top_t == "tuple" and inner_t == "list" and inner_len == 2
          and u_t == "ProxyFunction")
    if ok:
        return 0
    print("ERROR: unexpected TnT structure", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
