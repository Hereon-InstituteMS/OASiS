"""Tier-2: `react tce/qk` is a different model from `react tce`, and on the
distribution's own air.tce it fires nothing at all.

  chemistry:4  tce/qk dispatches per reaction on the style letter in column 2 of
               the coefficient line, and its Arrhenius branch is a DIFFERENT
               expression from react tce's — no gamma-function prefactor, plain
               exponents. On an all-'A' file it produces zero reactions while
               react tce and react qk on the same file both produce many, and
               it says so nowhere: rc = 0, no warning, and the parse count line
               reports the reactions WERE loaded.
  chemistry:5  it refuses recombination reactions and 'react_modify
               compute_chem_rates', both checked at the start of the first run.

The comparison is three styles on ONE file, one seed, one box, one temperature.
Nothing is asserted about how many reactions tce fires; only that tce and qk
each fire SOME and tce/qk fires NONE, and that the particle count moves in the
first two and not in the third. Those are presence/absence statements, which is
what makes them safe on a Monte-Carlo code: a stochastic run can move a count
around, it cannot turn a hundred tallies into zero.

Mutation control: T2_MUTATE=1 swaps the tce/qk deck's STYLE WORD to 'tce',
leaving the file, the species list, the mixture, the temperature, the seed, the
timestep and the run length identical. It then behaves like the tce deck — its
tally block fills and Np moves — so `tce_qk_fires_no_reaction_at_all`,
`tce_qk_leaves_the_particle_count_untouched` and
`the_three_styles_do_not_agree` go False.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

AIR = skip_if_unavailable("air.species", "air.vss", "air.tce")
MARS = skip_if_unavailable("mars.species", "mars.vss", "mars.tce")

HOT_AIR = """seed 12345
dimension 3
boundary rr rr rr
create_box 0 1e-4 0 1e-4 0 1e-4
create_grid 10 10 10
species air.species N O N2 O2 NO
mixture air N O N2 O2 NO
mixture air N2 frac 0.8
mixture air O2 frac 0.2
mixture air vstream 0 0 0 temp 20000.0
global nrho 7.07043e22 fnum 7.07043e7
collide vss air air.vss
react {style} air.tce
{modify}
create_particles air n 0
timestep 1e-9
stats 200
stats_style step np ncoll nreact
run 400
"""

TALLY = re.compile(r"Gas reaction tallies:(.*?)(?:\n\s*\n|\Z)", re.S)


def go(style: str, modify: str = ""):
    rc, txt = run(HOT_AIR.format(style=style, modify=modify), AIR)
    m = TALLY.search(txt)
    body = m.group(1).strip().splitlines() if m else []
    parsed = [l for l in body if "#-of-reactions" in l]
    per_reaction = [l for l in body if l.strip().startswith("reaction ")]
    h, r = stats_rows(txt)
    np_col = col(h, r, "Np") if (h and r) else []
    return rc, errors(txt), parsed, per_reaction, np_col


if MUTATE:
    print("mutation=tce_qk_deck_style_word_swapped_to_tce")

rc_t, e_t, par_t, rx_t, np_t = go("tce")
rc_q, e_q, par_q, rx_q, np_q = go("qk")
rc_x, e_x, par_x, rx_x, np_x = go("tce" if MUTATE else "tce/qk")

all_ran = all(rc == 0 for rc in (rc_t, rc_q, rc_x)) and not (e_t + e_q + e_x)
print(f"all_three_styles_run_cleanly={all_ran}")

# All three PARSE the same file: the count line is present in every run, which
# is why the count line cannot be used to tell them apart.
parsed_everywhere = bool(par_t) and bool(par_q) and bool(par_x)
print(f"all_three_report_the_reactions_as_parsed={parsed_everywhere}")

tce_fires = len(rx_t) > 0
qk_fires = len(rx_q) > 0
x_fires = len(rx_x) > 0
print(f"react_tce_fires_reactions={tce_fires}")
print(f"react_qk_fires_reactions={qk_fires}")
print(f"tce_qk_fires_no_reaction_at_all={not x_fires}")
if x_fires and not MUTATE:
    print(f"UNEXPECTED: tce/qk tallied {len(rx_x)} reactions: {rx_x[:2]}")

# Products are created under tce and qk (the reflecting box conserves particles
# except through dissociation), and not under tce/qk.
moved_t = len(np_t) > 1 and any(v != np_t[0] for v in np_t)
moved_q = len(np_q) > 1 and any(v != np_q[0] for v in np_q)
moved_x = len(np_x) > 1 and any(v != np_x[0] for v in np_x)
print(f"tce_and_qk_change_the_particle_count={moved_t and moved_q}")
print(f"tce_qk_leaves_the_particle_count_untouched={not moved_x}")

print(f"the_three_styles_do_not_agree={tce_fires and qk_fires and not x_fires}")

# chemistry:5 — the two refusals, and the escape for the first.
MARS_DECK = """seed 12345
dimension 3
boundary rr rr rr
create_box 0 1e-4 0 1e-4 0 1e-4
create_grid 8 8 8
species mars.species O2 N2 O N NO CO2 CO C CN C2 O2+ O+ N+ NO+ CO+ C+ e
mixture m O2 N2 O N NO CO2 CO C CN C2 O2+ O+ N+ NO+ CO+ C+ e
mixture m CO2 frac 0.95
mixture m N2 frac 0.05
mixture m vstream 0 0 0 temp 20000.0
global nrho 7.07043e22 fnum 7.07043e8
collide vss m mars.vss
react tce/qk mars.tce
{modify}
create_particles m n 0
timestep 1e-9
stats 100
run 100
"""
rc_r, txt_r = run(MARS_DECK.format(modify=""), MARS)
RECOMB_MSG = ("ERROR: React tce/qk does not currently support recombination "
              "reactions (../react_tce_qk.cpp:48)")
recomb_refused = any(RECOMB_MSG in e for e in errors(txt_r))
print(f"tce_qk_refuses_a_file_with_recombination={recomb_refused}")

rc_rn, txt_rn = run(MARS_DECK.format(modify="react_modify recomb no"), MARS)
print(f"react_modify_recomb_no_is_the_escape="
      f"{rc_rn == 0 and not errors(txt_rn)}")

rc_c, e_c, _, _, _ = go("tce/qk", "react_modify compute_chem_rates yes")
RATES_MSG = ("ERROR: React tce/qk does not currently support the "
             "'react_modify compute_chem_rates' option "
             "(../react_tce_qk.cpp:52)")
rates_refused = any(RATES_MSG in e for e in e_c)
print(f"tce_qk_refuses_compute_chem_rates={rates_refused}")

ok = (all_ran and parsed_everywhere and tce_fires and qk_fires and not x_fires
      and moved_t and moved_q and not moved_x
      and recomb_refused and rc_rn == 0 and not errors(txt_rn) and rates_refused)
print("RESULT=" + ("pass" if ok else "fail"))
sys.exit(0 if ok else 1)
