"""Tier-2: writeVTK with the same name every step leaves one frame.

Covers _general.vtk_time_series_measured.Signal: a 100-step run that
calls gridView.writeVTK('name', ...) each step ends with a SINGLE .vtu,
because each call overwrites the last. The fix is number=step, which
appends a zero-padded five-digit index. Also checks the two secondary
statements of that section: no .pvd collection file is produced, and
OutputType exposes exactly ascii / base64 / appendedraw /
appendedbase64.

Nothing here builds a weak form; the interpolant is the only compiled
artefact and it is shared with the other fixtures.

Verified by execution against dune-fem 2.12.0.2.

MUTATION CONTROL. T2_MUTATE=1 passes number=step in the trap loop —
the pathology removed. The ten calls then leave ten files, so
'same_name_files=1', "same_name_file_list=['overwritten.vtu']" and
'same_name_leaves_one_frame=True' are no longer printed and a FAIL:
line appears.
"""
from __future__ import annotations

import glob
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dune.grid import structuredGrid, OutputType                # noqa: E402
from dune.fem.space import lagrange                             # noqa: E402
from ufl import SpatialCoordinate                               # noqa: E402


def main() -> int:
    fail: list[str] = []
    workdir = tempfile.mkdtemp(prefix="dune_vtk_series_")
    os.chdir(workdir)

    gridView = structuredGrid([0, 0], [1, 1], [4, 4])
    space = lagrange(gridView, order=1)
    x = SpatialCoordinate(space)
    uh = space.interpolate(x[0] + 2 * x[1], name="uh")

    steps = 10

    # ── the trap: same name every step ──────────────────────────────
    if MUTATE:
        print("mutation=the_trap_loop_passes_number_step")
    for step in range(steps):
        if MUTATE:
            gridView.writeVTK("overwritten", pointdata={"u": uh},
                              number=step)
        else:
            gridView.writeVTK("overwritten", pointdata={"u": uh})
    same_name = sorted(os.path.basename(p)
                       for p in glob.glob("overwritten*.vtu"))
    print(f"steps_written={steps}")
    print(f"same_name_files={len(same_name)}")
    print(f"same_name_file_list={same_name}")
    print(f"same_name_leaves_one_frame={len(same_name) == 1}")
    if len(same_name) != 1:
        fail.append(f"writing {steps} steps under one name produced "
                    f"{len(same_name)} files {same_name}; the claim is "
                    f"that each call overwrites the last")

    # ── the fix: number=step ───────────────────────────────────────
    for step in range(steps):
        gridView.writeVTK("series", pointdata={"u": uh}, number=step)
    numbered = sorted(os.path.basename(p)
                      for p in glob.glob("series*.vtu"))
    print(f"numbered_files={len(numbered)}")
    print(f"numbered_first_three={numbered[:3]}")
    print(f"number_kwarg_gives_one_file_per_step="
          f"{len(numbered) == steps}")
    if len(numbered) != steps:
        fail.append(f"number=step produced {len(numbered)} files for "
                    f"{steps} steps")
    if numbered[:3] != ["series00000.vtu", "series00001.vtu",
                        "series00002.vtu"]:
        fail.append(f"the index is not a zero-padded five-digit "
                    f"suffix: {numbered[:3]}")

    # no collection file is written
    pvd = sorted(glob.glob("*.pvd"))
    print(f"pvd_files={pvd}")
    print(f"no_pvd_collection_written={pvd == []}")
    if pvd:
        fail.append(f"a collection file appeared ({pvd}); the claim is "
                    f"that ParaView has to open the sequence by pattern")

    # ── the encodings ──────────────────────────────────────────────
    members = sorted(m for m in dir(OutputType)
                     if not m.startswith("_") and m not in
                     ("name", "value"))
    print(f"output_types={members}")
    expected = ["appendedbase64", "appendedraw", "ascii", "base64"]
    if members != expected:
        fail.append(f"OutputType members are {members}, expected "
                    f"{expected}")
    gridView.writeVTK("ascii_probe", pointdata={"u": uh},
                      outputType=OutputType.ascii)
    ascii_file = glob.glob("ascii_probe*.vtu")
    readable = False
    if ascii_file:
        head = open(ascii_file[0], "r", errors="ignore").read(400)
        readable = "<VTKFile" in head
    print(f"ascii_outputtype_wrote_readable_vtu={readable}")
    if not readable:
        fail.append("outputType=OutputType.ascii did not produce a "
                    "readable .vtu")

    if not fail:
        print("dune_vtk_time_series_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
