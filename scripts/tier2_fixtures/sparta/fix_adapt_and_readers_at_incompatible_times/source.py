"""Tier-2: the three "not computed at compatible time" aborts, plus the fourth
message that gets mistaken for them.

  adaptive_grid:3     'fix adapt' Nevery must be a multiple of the Nfreq of the
                      fix it reads, and so must the stats interval of anything
                      that prints a dependent quantity. Three different
                      messages depending on WHO reads the fix.
  hypersonic_flow:4   the same rule stated for the hypersonic row, whose point
                      is that a per-grid COMPUTE (c_ID[i]) is re-evaluated every
                      adaptation step while a FIX (f_ID) is not.

Every message below was produced by running the deck, not read out of the
source. The decks differ from each other in one or two lines of a single base
deck.

WHAT THIS SHARPENS. Both entries attach 'Stats and fix not computed at
compatible times (../stats.cpp:203)' to the case "stats_style prints f_<ID>
directly". Executed, printing a PER-GRID fix that way does not give that message
at all — it gives 'Stats fix does not compute scalar (../stats.cpp:705)', which
is a shape complaint raised while the stats_style line is parsed, long before
any frequency is compared. stats.cpp:203 needs a fix that produces a global
scalar, e.g. 'fix ave/time'. An agent that grep'd for :203 after making the
per-grid mistake would not find it and would conclude the entry was wrong. Both
entries now name the condition; this fixture produces all four messages so the
distinction cannot silently revert.

Mutation control: T2_MUTATE=1 repairs each of the four aborting decks by the one
line that makes it abort, and nothing else — the adapt Nevery goes 30 -> 50, the
two stats intervals go 30 -> 50 so they are multiples of the fix's Nfreq 50, and
the per-grid fix is routed through a 'compute reduce' instead of being printed
raw by stats_style. All four then run, so all four quoted-message lines vanish
together with `every_quoted_compatible_time_message_is_verbatim`,
`the_compute_reduce_abort_is_MID_RUN_after_a_stats_line`,
`the_adapt_abort_is_at_setup_before_any_stats_line` and
`a_per_grid_fix_in_stats_style_gives_a_DIFFERENT_message`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import errors, run, skip_if_unavailable, stats_rows  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("ar.species", "ar.vss")

BASE = """seed 12345
dimension 2
boundary rr rr p
create_box 0 1e-4 0 1e-4 -0.5 0.5
create_grid 10 10 1
species ar.species Ar
mixture gas Ar vstream 0 0 0 temp 273.15
global nrho 7.07043e22 fnum 7.07043e11
collide vss gas ar.vss
create_particles gas n 0
compute npc grid all all n
compute tt temp
fix fnp ave/grid all 1 50 50 c_npc[1]
timestep 1e-9
{body}"""


def probe(body: str) -> dict:
    rc, txt = run(BASE.format(body=body), DATA)
    header, rows = stats_rows(txt)
    return {"rc": rc, "errors": errors(txt), "n_stats_rows": len(rows),
            "output": txt}


if MUTATE:
    print("mutation=each_incompatible_reader_repaired_to_a_multiple_of_nfreq_50")

# Under mutation every aborting deck is repaired by the one line that is the
# pathology: the offending interval becomes a multiple of the fix's Nfreq 50,
# and the per-grid fix is reduced to a scalar before stats_style sees it.
NEVERY = "50" if MUTATE else "30"
INTERVAL = "50" if MUTATE else "30"

CASES = {
    # the adapt's OWN complaint: Nevery 30 is not a multiple of Nfreq 50
    "adapt_nevery_not_a_multiple_of_the_fix_nfreq": probe(
        "stats 50\nstats_style step np ngrid\n"
        f"fix ad adapt {NEVERY} all refine value f_fnp 1.0 0.1\nrun 100\n"),
    # the same deck with Nevery 50 — one number changed
    "adapt_nevery_is_a_multiple": probe(
        "stats 50\nstats_style step np ngrid\n"
        "fix ad adapt 50 all refine value f_fnp 1.0 0.1\nrun 100\n"),
    # a compute reduce reading the fix at an incompatible stats interval
    "compute_reduce_reads_the_fix_at_an_incompatible_time": probe(
        f"compute r reduce min f_fnp\nstats {INTERVAL}\n"
        "stats_style step np c_r\nrun 100\n"),
    # stats printing a GLOBAL-SCALAR fix at an incompatible interval
    "stats_prints_a_global_fix_at_an_incompatible_time": probe(
        f"fix ft ave/time 1 50 50 c_tt\nstats {INTERVAL}\n"
        "stats_style step np f_ft\nrun 100\n"),
    # the message that is NOT one of the three: a per-grid fix in stats_style
    "stats_prints_a_per_grid_fix": probe(
        "stats 50\ncompute rr reduce ave f_fnp\n"
        "stats_style step np c_rr\nrun 100\n" if MUTATE
        else "stats 30\nstats_style step np f_fnp\nrun 100\n"),
    # hypersonic_flow:4's actual point: a per-grid COMPUTE has no such
    # restriction, so the identical adapt Nevery is accepted
    "adapt_driven_by_a_compute_has_no_frequency_restriction": probe(
        "stats 50\nstats_style step np ngrid\n"
        "fix ad adapt 30 all refine value c_npc[1] 1.0 0.1\nrun 100\n"),
}

QUOTED = {
    "adapt_nevery_not_a_multiple_of_the_fix_nfreq":
        "ERROR: Fix for adapt not computed at compatible time "
        "(../adapt_grid.cpp:502)",
    "compute_reduce_reads_the_fix_at_an_incompatible_time":
        "ERROR: Fix used in compute reduce not computed at compatible time "
        "(../compute_reduce.cpp:805)",
    "stats_prints_a_global_fix_at_an_incompatible_time":
        "ERROR: Stats and fix not computed at compatible times "
        "(../stats.cpp:203)",
    "stats_prints_a_per_grid_fix":
        "ERROR: Stats fix does not compute scalar (../stats.cpp:705)",
}

for name, r in CASES.items():
    print(f"{name}_rc={r['rc']} n_errors={len(r['errors'])} "
          f"n_stats_rows={r['n_stats_rows']}")
    if r["errors"]:
        print(f"{name}_message={r['errors'][0]}")

quoted_ok = True
for name, msg in QUOTED.items():
    if msg not in CASES[name]["errors"]:
        quoted_ok = False
        print(f"UNEXPECTED: {name} printed {CASES[name]['errors'][:1]} "
              f"not {msg!r}")

good = CASES["adapt_nevery_is_a_multiple"]
reduce_case = CASES["compute_reduce_reads_the_fix_at_an_incompatible_time"]
compute_driven = CASES["adapt_driven_by_a_compute_has_no_frequency_restriction"]
pergrid = CASES["stats_prints_a_per_grid_fix"]

print(f"every_quoted_compatible_time_message_is_verbatim={quoted_ok}")
print(f"the_same_adapt_with_a_multiple_nevery_runs={good['rc'] == 0}")
print(f"the_compute_reduce_abort_is_MID_RUN_after_a_stats_line="
      f"{reduce_case['rc'] != 0 and reduce_case['n_stats_rows'] >= 1}")
adapt_case = CASES["adapt_nevery_not_a_multiple_of_the_fix_nfreq"]
adapt_aborts_before_stats = (adapt_case["rc"] != 0
                             and adapt_case["n_stats_rows"] == 0)
print(f"the_adapt_abort_is_at_setup_before_any_stats_line="
      f"{adapt_aborts_before_stats}")
print(f"the_same_nevery_driven_by_a_per_grid_COMPUTE_is_accepted="
      f"{compute_driven['rc'] == 0}")
print(f"a_per_grid_fix_in_stats_style_gives_a_DIFFERENT_message="
      f"{pergrid['rc'] != 0 and not any('compatible' in e for e in pergrid['errors'])}")

ok = (quoted_ok and good["rc"] == 0
      and reduce_case["rc"] != 0 and reduce_case["n_stats_rows"] >= 1
      and compute_driven["rc"] == 0
      and pergrid["rc"] != 0
      and not any("compatible" in e for e in pergrid["errors"]))
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
