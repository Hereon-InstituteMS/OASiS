"""Tier-2: adapt_grid on a fix that has not yet produced output.

Before any run, a `fix ave/grid` reads as literal 0 in every cell. With the
default `thresh more less` no cell qualifies and SPARTA prints
`  no grid adaptation performed` instead of the `<N> cells refined` line —
rc = 0, no warning. That looks like a harmless no-op, but the zeros are not
harmless: flipping the action to `coarsen` makes the SAME command coarsen the
whole grid. After a run the fix has data and the behaviour reverses.

Also checks the bracket rule that is easy to confuse with this trap: a fix
ave/grid with one input value is a per-grid VECTOR, so `value f_ID[1]` aborts
both before AND after a run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import errors, run, skip_if_unavailable  # noqa: E402

DATA = skip_if_unavailable("ar.species", "ar.vss")

# a two-level grid so 'coarsen' has something to act on
BASE = """seed 12345
dimension 2
boundary rr rr p
create_box 0 1e-4 0 1e-4 -0.5 0.5
create_grid 10 10 1 levels 2 subset 2 * * * 2 2 1
species ar.species Ar
mixture gas Ar vstream 0 0 0 temp 273.15
global nrho 7.07043e22 fnum 7.07043e11
collide vss gas ar.vss
create_particles gas n 0
compute npc grid all all n
fix fnp ave/grid all 1 50 50 c_npc[1]
timestep 1e-9
stats 50
stats_style step np ngrid
{body}"""


def probe(body: str):
    rc, txt = run(BASE.format(body=body), DATA)
    banner = "Adapting grid ..." in txt
    noop = "no grid adaptation performed" in txt
    refined = coarsened = None
    for line in txt.splitlines():
        if "cells refined" in line:
            parts = line.split()
            refined, coarsened = int(parts[0]), int(parts[3])
    return rc, banner, noop, refined, coarsened, errors(txt)


# 1. refine on an unfed fix, BEFORE any run
rc_b, ban_b, noop_b, ref_b, _, _ = probe(
    "adapt_grid all refine value f_fnp 1.0 0.1\nrun 50\n")
# 2. same command AFTER a run that fed the fix
rc_a, ban_a, noop_a, ref_a, _, _ = probe(
    "run 50\nadapt_grid all refine value f_fnp 1.0 0.1\nrun 50\n")
# 3. COARSEN on the same unfed fix -- the zeros now match the criterion
rc_c, ban_c, noop_c, _, coa_c, _ = probe(
    "adapt_grid all coarsen value f_fnp 1.0 0.1\nrun 50\n")
# 4. bracket rule: a single-value fix ave/grid is a VECTOR, f_ID[1] aborts
rc_x, _, _, _, _, err_x = probe(
    "adapt_grid all refine value f_fnp[1] 1.0 0.1\nrun 50\n")

print(f"before_run_rc={rc_b} banner={ban_b} noop_line={noop_b} refined={ref_b}")
print(f"after_run_rc={rc_a} banner={ban_a} noop_line={noop_a} refined={ref_a}")
print(f"coarsen_before_run_rc={rc_c} cells_coarsened={coa_c}")
print(f"bracketed_fix_rc={rc_x} err={err_x[:1]}")

print(f"unfed_fix_prints_no_adaptation_line={noop_b and ref_b is None}")
print(f"unfed_fix_run_still_exits_zero={rc_b == 0}")
print(f"fed_fix_actually_refines={ref_a is not None and ref_a > 0}")
print(f"coarsen_on_unfed_fix_is_not_a_noop={rc_c == 0 and (coa_c or 0) > 0}")
print("bracketed_single_value_fix_aborts="
      f"{rc_x != 0 and any('Adapt fix does not calculate a per-grid array' in e for e in err_x)}")

ok = (noop_b and ref_b is None and rc_b == 0
      and ref_a is not None and ref_a > 0
      and rc_c == 0 and (coa_c or 0) > 0
      and rc_x != 0
      and any("Adapt fix does not calculate a per-grid array" in e for e in err_x))
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
