"""Tier-2: a raw 'compute surf' written straight to 'dump surf' — what is
actually in the file.

  surface_interaction:6  route a compute surf through 'fix ave/surf' before
                         dumping it, or a large fraction of the elements read
                         exactly zero.

WHAT THIS FALSIFIED. The entry said the dumped value "is the tally accumulated
since the compute was last invoked, so its magnitude depends on the DUMP
FREQUENCY", and told the reader to look for "nonzero values scaling with the
dump interval". They do not scale, and there is nothing to see.

Two decks were run whose ONLY difference is 'dump ... all 50' against
'dump ... all 200' — a four-fold change in dump interval, with the dump as the
sole consumer of the compute so nothing else can be resetting it. At the same
timestep the two files agree element for element: same count of exact zeros,
same mean, same maximum. update.cpp:1592 says why — 'slist_compute[i]->clear()'
runs on every step the compute is active, so what lands in the dump is the tally
for THAT ONE TIMESTEP, never an accumulation across the interval. It is the same
per-step semantics as the Ncoll family.

So an agent following the published test would change the dump interval, see the
magnitude not move, and conclude its deck was fine. What IS true, and what this
fixture asserts instead, is the zero fraction: about a quarter of the elements
in the raw field are exactly 0 at any snapshot, and the fix-averaged field of
the same run has none. That is the discriminator, and it does not depend on the
dump interval either.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import (  # noqa: E402
    dump_column, errors, find_example, require, run_keep, skip_if_unavailable,
)

DATA = skip_if_unavailable("ar.species", "ar.vss")
CIRCLE = find_example("circle", "data.circle")
if CIRCLE is None:                                          # pragma: no cover
    print("SKIP: SPARTA examples/circle/data.circle not found (set SPARTA_ROOT)")
    sys.exit(0)
FILES = dict(DATA, **{"data.circle": CIRCLE})

DECK = """seed 12345
dimension 2
global gridcut 0.0 comm/sort yes
boundary o r p
create_box 0 10 0 10 -0.5 0.5
create_grid 20 20 1
global nrho 1e20 fnum 1e17
species ar.species Ar
mixture gas Ar vstream 300 0 0 temp 273.15
read_surf data.circle
surf_collide cw diffuse 300 1.0
surf_modify all collide cw
collide vss gas ar.vss
fix in emit/face gas xlo
compute cs surf all all n
{extra}timestep 1e-4
dump raw surf all {n} raw.*
dump_modify raw pad 6
stats 200
stats_style step np nscoll
run 600
"""
# The dump field list is appended here so the two variants differ ONLY in the
# interval and in whether a fix is present.
DECK = DECK.replace("dump raw surf all {n} raw.*",
                    "dump raw surf all {n} raw.* id c_cs[1]")
AVG = ("fix fs ave/surf all 1 {n} {n} c_cs[1]\n"
       "dump avg surf all {n} avg.* id f_fs\n"
       "dump_modify avg pad 6\n")


def probe(n: int, averaged: bool) -> dict:
    extra = AVG.format(n=n) if averaged else ""
    rc, txt, work = run_keep(DECK.format(n=n, extra=extra), FILES, timeout=600)
    result = {"rc": rc, "errors": errors(txt), "raw": [], "avg": []}
    try:
        raw = work / f"raw.{600:06d}"
        if raw.is_file():
            result["raw"] = dump_column(raw, 1)
        avg = work / f"avg.{600:06d}"
        if avg.is_file():
            result["avg"] = dump_column(avg, 1)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return result


# Sole consumer of the compute, two dump intervals.
solo_fast = probe(50, averaged=False)
solo_slow = probe(200, averaged=False)
# Same run, raw field and fix-averaged field side by side.
paired = probe(50, averaged=True)


def zeros(v):
    return sum(1 for x in v if x == 0.0)


def summary(tag, v):
    if not v:
        print(f"{tag}_n_elements=0")
        return
    print(f"{tag}_n_elements={len(v)} zeros={zeros(v)} "
          f"mean={sum(v) / len(v):.6g} max={max(v):.6g}")


for tag, r in (("dump_interval_50", solo_fast), ("dump_interval_200", solo_slow),
               ("paired_run", paired)):
    print(f"{tag}_rc={r['rc']} n_errors={len(r['errors'])}")
summary("raw_at_interval_50", solo_fast["raw"])
summary("raw_at_interval_200", solo_slow["raw"])
summary("raw_in_the_paired_run", paired["raw"])
summary("fix_averaged_in_the_paired_run", paired["avg"])

all_clean = all(r["rc"] == 0 and not r["errors"]
                for r in (solo_fast, solo_slow, paired))
have_fields = bool(solo_fast["raw"]) and bool(solo_slow["raw"]) \
    and bool(paired["raw"]) and bool(paired["avg"])
# The falsification: a 4x change in dump interval leaves the field identical.
identical_across_intervals = solo_fast["raw"] == solo_slow["raw"]
raw_zero_fraction = zeros(paired["raw"]) / len(paired["raw"]) if paired["raw"] else 0.0
avg_zero_fraction = zeros(paired["avg"]) / len(paired["avg"]) if paired["avg"] else 1.0
raw_has_many_zeros = raw_zero_fraction > 0.1
averaging_removes_the_zeros = avg_zero_fraction < 0.5 * raw_zero_fraction

print(f"raw_zero_fraction={raw_zero_fraction:.4g}")
print(f"fix_averaged_zero_fraction={avg_zero_fraction:.4g}")
print(f"every_run_exits_zero_with_no_error={all_clean}")
print(f"both_dumps_and_both_fields_were_written={have_fields}")
print(f"a_4x_dump_interval_change_leaves_the_field_IDENTICAL="
      f"{identical_across_intervals}")
print(f"so_magnitude_does_NOT_scale_with_the_dump_interval="
      f"{identical_across_intervals}")
print(f"the_raw_field_has_a_large_fraction_of_exact_zeros={raw_has_many_zeros}")
print(f"routing_through_fix_ave_surf_removes_them={averaging_removes_the_zeros}")

ok = (all_clean and have_fields and identical_across_intervals
      and raw_has_many_zeros and averaging_removes_the_zeros)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
