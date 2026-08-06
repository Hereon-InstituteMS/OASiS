"""Tier-2: the per-surf shape rules, the watertight test, and the fact that
'data.circle' names two different bodies in the SPARTA distribution.

  surface_interaction:4      'compute surf' is ALWAYS a per-surf ARRAY, so a
                             bare c_ID is an error in fix ave/surf and in
                             dump surf.
  conjugate_heat_transfer:3  a fix ave/surf fed ONE value is the opposite — a
                             per-surf VECTOR, read as f_ID with NO bracket;
                             indexing it aborts.
  surface_interaction:7      an OPEN curve fails the watertight test.
  conjugate_heat_transfer:6  the same claim for a half-body on a symmetry plane.
  surface_interaction:8      several example directories ship a file called
                             'data.circle' and they are not the same body, so
                             staging by basename can hand a deck the wrong
                             geometry.

WHAT THIS SHARPENS. Both watertight entries prescribe: "Place the open endpoints
exactly on a simulation-box face and read it with 'read_surf <file> clip'."
Executed, the 'clip' keyword is not what saves it. An open two-segment curve
whose endpoints sit exactly on the ylo face is accepted with or WITHOUT 'clip';
the same curve moved into the interior aborts either way. It is the endpoints
being on the face that matters, and an agent that reads the entry as "add clip
and it will be fine" will keep failing. Both entries now say so.

Every message below came out of a run. No count from any run is pinned.

Mutation control: T2_MUTATE=1 moves the open curve that sat in the INTERIOR so
its two endpoints lie exactly on the ylo face — the geometry the accepted case
already uses — which removes the failing-watertight pathology and nothing else.
Both interior decks then load, so `an_open_curve_in_the_interior_fails_watertight`,
`adding_clip_does_NOT_rescue_an_interior_open_curve` and
`clip_is_not_what_makes_the_open_curve_legal` go False. The per-surf shape cases
and the data.circle collision run on decks this control does not touch.
"""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import (  # noqa: E402
    errors, find_example, require, run, skip_if_unavailable, stats_rows,
)

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("ar.species", "ar.vss")

# The two DIFFERENT bodies that both answer to the name 'data.circle'.
CIRCLES = {}
for sub in ("adapt", "ambi"):
    p = find_example(sub, "data.circle")
    if p is None:
        print(f"SKIP: SPARTA examples/{sub}/data.circle not found "
              f"(set SPARTA_ROOT)")
        sys.exit(0)
    CIRCLES[sub] = p

SCRATCH = Path(tempfile.mkdtemp(prefix="spa_surf_"))
atexit.register(shutil.rmtree, SCRATCH, True)


def surf_file(name: str, pts: list[tuple[float, float]]) -> Path:
    lines = "\n".join(f"{i + 1} {i + 1} {i + 2}" for i in range(len(pts) - 1))
    body = (f"open polyline\n\n{len(pts)} points\n{len(pts) - 1} lines\n\n"
            "Points\n\n"
            + "\n".join(f"{i + 1} {x} {y}" for i, (x, y) in enumerate(pts))
            + "\n\nLines\n\n" + lines + "\n")
    p = SCRATCH / name
    p.write_text(body)
    return p


if MUTATE:
    print("mutation=interior_open_curve_endpoints_moved_onto_the_ylo_face")

# Under mutation the pathology is REMOVED: the interior curve's two endpoints
# are moved onto the ylo face, the geometry ON_FACE already uses. The curve is
# still open and still two segments; only where its ends sit changes.
INTERIOR = surf_file("data.openinterior",
                     [(4.0, 0.0), (5.0, 3.0), (6.0, 0.0)] if MUTATE
                     else [(4.0, 4.0), (5.0, 6.0), (6.0, 4.0)])
ON_FACE = surf_file("data.openface",
                    [(4.0, 0.0), (5.0, 3.0), (6.0, 0.0)])

HEAD = """seed 12345
dimension 2
global gridcut 0.0 comm/sort yes
boundary o o p
create_box {box}
create_grid 10 10 1
global nrho 1e20 fnum 1e17
species ar.species Ar
mixture gas Ar vstream 100 0 0 temp 273.15
"""
SURF = """read_surf {surf}
surf_collide cw diffuse 300 1.0
surf_modify all collide cw
collide vss gas ar.vss
create_particles gas n 0
compute cs surf all all n press
timestep 1e-6
"""
BIG_BOX = "0 10 0 10 -0.5 0.5"
SMALL_BOX = "-2 2 -2 2 -0.5 0.5"


def probe(deck: str, extra: dict | None = None) -> dict:
    files = dict(DATA)
    files.update(extra or {})
    rc, txt = run(deck, files)
    header, rows = stats_rows(txt)
    return {"rc": rc, "errors": errors(txt), "n_stats_rows": len(rows)}


def shape_case(tail: str) -> dict:
    return probe(HEAD.format(box=BIG_BOX)
                 + SURF.format(surf="data.circle") + tail,
                 {"data.circle": CIRCLES["adapt"]})


CASES = {
    # surface_interaction:4 — compute surf is an ARRAY
    "fix_ave_surf_fed_a_bare_compute_id":
        shape_case("fix fs ave/surf all 1 10 10 c_cs\n"
                   "stats 10\nstats_style step np\nrun 20\n"),
    "dump_surf_fed_a_bare_compute_id":
        shape_case("dump 1 surf all 10 dmp.* id c_cs\n"
                   "stats 10\nstats_style step np\nrun 20\n"),
    "fix_ave_surf_fed_an_indexed_compute":
        shape_case("fix fs ave/surf all 1 10 10 c_cs[1]\n"
                   "stats 10\nstats_style step np\nrun 20\n"),
    # conjugate_heat_transfer:3 — a one-value fix ave/surf is a VECTOR
    "compute_reduce_indexes_a_single_value_fix":
        shape_case("fix fs ave/surf all 1 10 10 c_cs[1]\n"
                   "compute r reduce ave f_fs[1]\n"
                   "stats 10\nstats_style step np c_r\nrun 20\n"),
    "compute_reduce_reads_a_single_value_fix_bare":
        shape_case("fix fs ave/surf all 1 10 10 c_cs[1]\n"
                   "compute r reduce ave f_fs\n"
                   "stats 10\nstats_style step np c_r\nrun 20\n"),
    # surface_interaction:7 / conjugate_heat_transfer:6 — watertight
    "open_curve_in_the_interior":
        probe(HEAD.format(box=BIG_BOX) + "read_surf data.openinterior\n",
              {"data.openinterior": INTERIOR}),
    "open_curve_in_the_interior_with_clip":
        probe(HEAD.format(box=BIG_BOX) + "read_surf data.openinterior clip\n",
              {"data.openinterior": INTERIOR}),
    "open_curve_with_endpoints_on_a_box_face":
        probe(HEAD.format(box=BIG_BOX) + "read_surf data.openface\n",
              {"data.openface": ON_FACE}),
    "open_curve_with_endpoints_on_a_box_face_and_clip":
        probe(HEAD.format(box=BIG_BOX) + "read_surf data.openface clip\n",
              {"data.openface": ON_FACE}),
    # surface_interaction:8 — same name, different body
    "adapt_circle_in_the_box_it_was_drawn_for":
        probe(HEAD.format(box=BIG_BOX) + "read_surf data.circle\n",
              {"data.circle": CIRCLES["adapt"]}),
    "ambi_circle_in_the_adapt_box":
        probe(HEAD.format(box=BIG_BOX) + "read_surf data.circle\n",
              {"data.circle": CIRCLES["ambi"]}),
    "ambi_circle_in_the_box_it_was_drawn_for":
        probe(HEAD.format(box=SMALL_BOX) + "read_surf data.circle\n",
              {"data.circle": CIRCLES["ambi"]}),
    "adapt_circle_in_the_ambi_box":
        probe(HEAD.format(box=SMALL_BOX) + "read_surf data.circle\n",
              {"data.circle": CIRCLES["adapt"]}),
}

QUOTED = {
    "fix_ave_surf_fed_a_bare_compute_id":
        "ERROR: Fix ave/surf compute does not calculate a per-surf vector "
        "(../fix_ave_surf.cpp:150)",
    "dump_surf_fed_a_bare_compute_id":
        "ERROR: Dump surf compute does not compute per-surf vector "
        "(../dump_surf.cpp:569)",
    "compute_reduce_indexes_a_single_value_fix":
        "ERROR: Compute reduce fix does not calculate a per-surf array "
        "(../compute_reduce.cpp:293)",
}

for name, r in CASES.items():
    print(f"{name}_rc={r['rc']} n_errors={len(r['errors'])}")
    if r["errors"]:
        print(f"{name}_message={r['errors'][0]}")

quoted_ok = True
for name, msg in QUOTED.items():
    if msg not in CASES[name]["errors"]:
        quoted_ok = False
        print(f"UNEXPECTED: {name} printed {CASES[name]['errors'][:1]} "
              f"not {msg!r}")


def aborts(name):
    return CASES[name]["rc"] != 0


def runs(name):
    return CASES[name]["rc"] == 0


def watertight_failed(name):
    return any("Watertight check failed" in e and "unmatched points" in e
               for e in CASES[name]["errors"])


def outside_box(name):
    return any("surface points are not inside simulation box" in e
               for e in CASES[name]["errors"])


indexed_forms_run = (runs("fix_ave_surf_fed_an_indexed_compute")
                     and runs("compute_reduce_reads_a_single_value_fix_bare"))
clip_is_not_the_remedy = (
    watertight_failed("open_curve_in_the_interior")
    and watertight_failed("open_curve_in_the_interior_with_clip")
    and runs("open_curve_with_endpoints_on_a_box_face")
    and runs("open_curve_with_endpoints_on_a_box_face_and_clip"))
same_name_different_body = (
    runs("adapt_circle_in_the_box_it_was_drawn_for")
    and outside_box("ambi_circle_in_the_adapt_box")
    and runs("ambi_circle_in_the_box_it_was_drawn_for")
    and outside_box("adapt_circle_in_the_ambi_box"))

print(f"every_quoted_per_surf_message_is_verbatim={quoted_ok}")
print(f"the_indexed_and_bare_forms_that_should_work_do={indexed_forms_run}")
print(f"an_open_curve_in_the_interior_fails_watertight="
      f"{watertight_failed('open_curve_in_the_interior')}")
print(f"adding_clip_does_NOT_rescue_an_interior_open_curve="
      f"{watertight_failed('open_curve_in_the_interior_with_clip')}")
print(f"endpoints_on_a_box_face_are_accepted_without_clip="
      f"{runs('open_curve_with_endpoints_on_a_box_face')}")
print(f"clip_is_not_what_makes_the_open_curve_legal={clip_is_not_the_remedy}")
print(f"two_files_named_data_circle_are_not_interchangeable="
      f"{same_name_different_body}")
print(f"the_wrong_data_circle_reports_points_outside_the_box="
      f"{outside_box('ambi_circle_in_the_adapt_box')}")

ok = (quoted_ok and indexed_forms_run and clip_is_not_the_remedy
      and same_name_different_body)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
