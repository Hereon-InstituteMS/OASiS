"""Tier-2: VTKOutput's cells-per-element grow as the triangular number
T(2^N) = 2^N (2^N + 1) / 2, not as 4^N.

Claim: ngsolve poisson#4 -- "subdivision=2 on VTKOutput is the recommended
default for any order >= 2 FESpace.  Cells per element grow as
T(2^N) = 2^N*(2^N+1)/2, NOT as 4^N: measured 1, 3, 10, 36 cells per triangle for
N = 0, 1, 2, 3."

Wrong variant: sizing an export by 4^N, which over-counts from N = 1 upward and
by a factor of 7.1 at N = 3.

What this fixture pins, all re-measured on this run:
  * the cells-per-element sequence for N = 0..3 is exactly 1, 3, 10, 36;
  * each entry equals T(2^N) = 2^N (2^N + 1) / 2 exactly, as integers;
  * 4^N gives 1, 4, 16, 64 -- it agrees only at N = 0 and is wrong at every
    other level, so the two formulas are genuinely distinguishable here;
  * the file size grows with N as well, so the cost the formula predicts is
    real and not just a header count;
  * the point count grows too, which is what makes the extra cells useful.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile

from netgen.geom2d import unit_square
from ngsolve import GridFunction, H1, Mesh, VTKOutput, pi, sin, x, y


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    fes = H1(mesh, order=2)
    gf = GridFunction(fes)
    gf.Set(sin(pi * x) * sin(pi * y))
    tmpdir = tempfile.mkdtemp()

    rows = []
    for N in (0, 1, 2, 3):
        base = os.path.join(tmpdir, f"sub{N}")
        VTKOutput(mesh, coefs=[gf], names=["u"], filename=base,
                  subdivision=N).Do()
        path = base + ".vtu"
        blob = open(path, "rb").read()
        header = blob[:blob.find(b"<AppendedData")].decode("ascii", "replace")
        ncell = int(re.search(r'NumberOfCells="(\d+)"', header).group(1))
        npt = int(re.search(r'NumberOfPoints="(\d+)"', header).group(1))
        per = ncell // mesh.ne
        rows.append((N, ncell, npt, per, len(blob)))
        print(f"subdivision={N} cells={ncell} points={npt} "
              f"cells_per_element={per} bytes={len(blob)}")

    per = [r[3] for r in rows]
    triangular = [(2 ** N) * (2 ** N + 1) // 2 for N in (0, 1, 2, 3)]
    powers4 = [4 ** N for N in (0, 1, 2, 3)]
    print(f"cells_per_element={per}")
    print(f"triangular_T_of_2N={triangular}")
    print(f"four_to_the_N={powers4}")
    print(f"matches_triangular_formula={per == triangular}")
    print(f"is_1_3_10_36={per == [1, 3, 10, 36]}")
    print(f"does_not_match_4_to_the_N={per != powers4}")
    disagree = [i for i in range(4) if triangular[i] != powers4[i]]
    print(f"levels_where_the_two_formulas_disagree={disagree}")
    print(f"formulas_are_distinguishable_here={len(disagree) >= 3}")
    print(f"four_to_the_N_overcount_at_N3={powers4[3] / triangular[3]:.4f}")

    sizes = [r[4] for r in rows]
    pts = [r[2] for r in rows]
    print(f"file_sizes={sizes}")
    print(f"file_size_grows_with_subdivision="
          f"{all(b > a for a, b in zip(sizes, sizes[1:]))}")
    print(f"point_count_grows_with_subdivision="
          f"{all(b > a for a, b in zip(pts, pts[1:]))}")

    ok = (
        per == triangular and per == [1, 3, 10, 36] and per != powers4
        and len(disagree) >= 3
        and all(b > a for a, b in zip(sizes, sizes[1:]))
        and all(b > a for a, b in zip(pts, pts[1:]))
    )
    if ok:
        return 0
    print("FAIL: VTK subdivision cell-count invariant not held",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
