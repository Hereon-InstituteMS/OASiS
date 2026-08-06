"""Tier-2: 'global nrho ... fnum ...' after create_particles, or missing, gives
an EMPTY domain and says nothing about it.

The universal [Setup] claim in _common.py, and the one an agent is most likely
to trip over, because the failure has no diagnostic of any kind. It is executed
here in full — the entry's Signal names two observables and both are checked:

  * the setup line 'Created 0 particles';
  * an Np column that is 0 on EVERY stats line.

Plus the three things that make it dangerous rather than merely wrong: exit code
0, no ERROR line, and no WARNING line. SPARTA's defaults are nrho = 1.0 and
fnum = 1.0, so on a 1e-4 m box the requested count is nrho*V/fnum = 1e-8, which
truncates to nothing.

The control run — the identical deck with the global line moved BEFORE
create_particles — is what makes the zero meaningful. No number here is pinned:
the assertions are "the same deck gives a nonzero count one way and exactly zero
the other", and 'exactly zero' is the pathology, not a measurement.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

DATA = skip_if_unavailable("ar.species")

DECK = """seed 12345
dimension 2
boundary rr rr p
create_box 0 1e-4 0 1e-4 -0.5 0.5
create_grid 10 10 1
species ar.species Ar
mixture gas Ar vstream 0 0 0 temp 273.15
{pre}create_particles gas n 0
{post}timestep 1e-9
stats 50
stats_style step np ncoll
run 100
"""

GLOBAL = "global nrho 7.07043e22 fnum 7.07043e11\n"


def probe(pre: str, post: str) -> dict:
    rc, txt = run(DECK.format(pre=pre, post=post), DATA)
    created = None
    for line in txt.splitlines():
        if line.startswith("Created ") and line.endswith("particles"):
            created = int(line.split()[1])
    header, rows = stats_rows(txt)
    np_col = col(header, rows, "Np") if (header and rows and "Np" in header) else []
    return {
        "rc": rc,
        "created": created,
        "np": np_col,
        "errors": errors(txt),
        "warnings": [l for l in txt.splitlines() if l.startswith("WARNING")],
    }


before = probe(GLOBAL, "")
after = probe("", GLOBAL)
absent = probe("", "")

for tag, r in (("global_before_create_particles", before),
               ("global_after_create_particles", after),
               ("global_omitted_entirely", absent)):
    print(f"{tag}_rc={r['rc']}")
    print(f"{tag}_created={r['created']}")
    print(f"{tag}_np_column={[int(v) for v in r['np']]}")
    print(f"{tag}_n_errors={len(r['errors'])} n_warnings={len(r['warnings'])}")

control_nonzero = (before["created"] or 0) > 0 and all(v > 0 for v in before["np"])
zero_cases = (after, absent)
created_zero = all(r["created"] == 0 for r in zero_cases)
np_all_zero = all(r["np"] and all(v == 0 for v in r["np"]) for r in zero_cases)
silent = all(r["rc"] == 0 and not r["errors"] and not r["warnings"]
             for r in zero_cases)
stats_lines_present = all(len(r["np"]) >= 3 for r in zero_cases)

print(f"control_run_with_global_first_is_populated={control_nonzero}")
print(f"created_line_reads_exactly_zero_both_ways={created_zero}")
print(f"np_is_zero_on_every_stats_line={np_all_zero}")
print(f"a_full_stats_table_is_still_printed={stats_lines_present}")
print(f"failure_is_completely_silent_rc0_no_error_no_warning={silent}")

if not (control_nonzero and created_zero and np_all_zero and silent
        and stats_lines_present):
    print("FAIL: fixture expectations not met")
    sys.exit(1)
