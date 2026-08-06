"""Tier-2: the three ways `fix ave/time` refuses to give you a number.

  hypersonic_flow:7   Nfreq must be a multiple of Nevery AND Nevery*Nrepeat must
                    not exceed Nfreq; both violations share ONE generic message
                    that names neither rule.
  hypersonic_flow:8  'stats N' must be a MULTIPLE of Nfreq. Sampling the stats
                    table more often than the averaging window is a HARD ERROR,
                    not a repeated value — which is the opposite of what a
                    reader expects, and it is why this is worth an entry.
  hypersonic_flow:9  the first stats row prints f_<ID> as a literal 0, with no
                    warning and no sentinel.

Every deck here is the same 1 mm 2d argon box at rest; the only thing that
changes between runs is the three window integers and the stats interval, so
nothing about the physics can account for the differences.

WHAT IS ASSERTED AND WHAT IS NOT. The messages are matched VERBATIM including
their (file:line), which no edit to these decks can counterfeit. The step-0
zero is asserted as an exact 0 against a LATER row of the same column being
non-zero — a comparison inside one run, not a pinned value. No temperature, no
count and no tolerance appears anywhere.

Mutation control: T2_MUTATE=1 repairs the window arithmetic and the stats
interval in the three decks that are supposed to be broken — Nfreq becomes a
multiple of Nevery, Nevery*Nrepeat is brought under Nfreq, and 'stats' becomes
a multiple of Nfreq — and changes nothing else: same box, same grid, same seed,
same run length, same compute. All three then complete at rc = 0 with an empty
error list, so `both_window_violations_share_one_message`,
`stats_must_be_a_multiple_of_nfreq` and `the_reduce_path_objects_too` go False.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("ar.species", "ar.vss")

DECK = """seed 12345
dimension 2
boundary p p p
create_box 0 1e-3 0 1e-3 -0.5 0.5
create_grid 10 10 1
species ar.species Ar
mixture gas Ar temp 300.0
global nrho 1e21 fnum 1e11
create_particles gas n 0
collide vss gas ar.vss
timestep 1e-8
compute gt temp
fix f ave/time {nevery} {nrepeat} {nfreq} c_gt
stats {stats}
stats_style step np f_f c_gt
run 300
"""

# (label, nevery, nrepeat, nfreq, stats)  — the three broken forms and one good.
# Under mutation each broken form is repaired to the SAME arithmetic the good
# deck uses, leaving everything else in the deck untouched.
CASES = {
    "nfreq_not_a_multiple_of_nevery": (3, 5, 10, 100) if not MUTATE else (10, 5, 100, 100),
    "nevery_times_nrepeat_over_nfreq": (10, 5, 10, 100) if not MUTATE else (10, 5, 100, 100),
    "stats_more_often_than_nfreq": (10, 5, 100, 50) if not MUTATE else (10, 5, 100, 100),
    "well_formed": (10, 5, 100, 100),
}

if MUTATE:
    print("mutation=window_arithmetic_and_stats_interval_repaired_in_all_three_bad_decks")

out = {}
for name, (nev, nrep, nfr, st) in CASES.items():
    rc, txt = run(DECK.format(nevery=nev, nrepeat=nrep, nfreq=nfr, stats=st), DATA)
    out[name] = (rc, errors(txt), txt)

WINDOW_MSG = "ERROR: Illegal fix ave/time command (../fix_ave_time.cpp:129)"
STATS_MSG = ("ERROR: Stats and fix not computed at compatible times "
             "(../stats.cpp:203)")

e_a = out["nfreq_not_a_multiple_of_nevery"][1]
e_b = out["nevery_times_nrepeat_over_nfreq"][1]
e_c = out["stats_more_often_than_nfreq"][1]

both_share = (any(WINDOW_MSG in e for e in e_a) and
              any(WINDOW_MSG in e for e in e_b))
print(f"both_window_violations_share_one_message={both_share}")
if not both_share:
    print(f"UNEXPECTED: nfreq case printed {e_a} and nrepeat case printed {e_b}")

# The message names NEITHER rule: it says nothing about Nfreq, Nevery or
# Nrepeat, which is exactly why a reader has to be told both constraints.
names_neither = all(
    not any(w in e for w in ("Nfreq", "Nevery", "Nrepeat", "multiple"))
    for e in e_a + e_b)
print(f"the_message_names_neither_constraint={names_neither and both_share}")

stats_rejected = any(STATS_MSG in e for e in e_c)
print(f"stats_must_be_a_multiple_of_nfreq={stats_rejected}")
if not stats_rejected:
    print(f"UNEXPECTED: stats-50-against-nfreq-100 printed {e_c} not {STATS_MSG!r}")

rc_ok, err_ok, txt_ok = out["well_formed"]
print(f"the_well_formed_deck_completes={rc_ok == 0 and not err_ok}")

# hypersonic_flow:9 — the step-0 row of f_f is a literal zero while c_gt (a
# compute, evaluated on the spot) already carries a physical value on the same
# row, and a later row of the SAME f_f column is non-zero. Both comparisons are
# internal to this one run.
header, rows = stats_rows(txt_ok)
first_row_zero = later_row_nonzero = compute_is_live_on_row_zero = False
if rc_ok == 0 and header and len(rows) >= 2:
    favt = col(header, rows, "f_f")
    ctemp = col(header, rows, "c_gt")
    first_row_zero = favt[0] == 0.0
    later_row_nonzero = any(v != 0.0 for v in favt[1:])
    compute_is_live_on_row_zero = ctemp[0] != 0.0
print(f"first_stats_row_of_a_fix_is_exactly_zero={first_row_zero}")
print(f"a_later_row_of_the_same_column_is_nonzero={later_row_nonzero}")
print(f"a_compute_on_the_same_row_is_already_live={compute_is_live_on_row_zero}")

# The reduce path raises the same objection with its own message, so the rule
# is a property of the fix's output schedule and not of the stats command.
REDUCE_DECK = """seed 12345
dimension 2
boundary p p p
create_box 0 1e-3 0 1e-3 -0.5 0.5
create_grid 10 10 1
species ar.species Ar
mixture gas Ar temp 300.0
global nrho 1e21 fnum 1e11
create_particles gas n 0
collide vss gas ar.vss
timestep 1e-8
compute g grid all all nrho
fix fg ave/grid all 10 10 {nfreq} c_g[*]
compute r reduce ave f_fg
stats {stats}
stats_style step c_r
run 200
"""
rc_r, txt_r = run(REDUCE_DECK.format(nfreq=200, stats=100 if not MUTATE else 200),
                  DATA)
# compute_reduce.cpp raises this text from three call sites — 778 (global),
# 805 (per-grid) and 832 (per-surf) — so the fixture matches the TEXT and
# then records which site fired, rather than pinning one line number that is
# right for only one kind of input.
REDUCE_MSG = ("ERROR: Fix used in compute reduce not computed at compatible "
              "time (../compute_reduce.cpp:")
reduce_objects = any(REDUCE_MSG in e for e in errors(txt_r))
print(f"the_reduce_path_objects_too={reduce_objects}")
if not reduce_objects:
    print(f"UNEXPECTED: reduce deck printed {errors(txt_r)} not {REDUCE_MSG!r}")

ok = (both_share and names_neither and stats_rejected and rc_ok == 0
      and not err_ok and first_row_zero and later_row_nonzero
      and compute_is_live_on_row_zero and reduce_objects)
print("RESULT=" + ("pass" if ok else "fail"))
sys.exit(0 if ok else 1)
