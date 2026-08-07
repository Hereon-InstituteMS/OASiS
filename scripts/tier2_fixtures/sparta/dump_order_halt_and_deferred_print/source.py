"""Tier-2: the dump argument order, and the two run-control fixes that fail
where a driver cannot see it.

  universal:10  'dump <ID> <style> <group> <Nevery> <file> <attrs>' — the style
                is the SECOND token, and getting it wrong is rejected by a name
                you did not type. A '*' in the filename gives one file per
                snapshot and its absence gives one file for all of them. A
                per-grid compute in a dump obeys the bracket rule. 'movie' is a
                registered style that cannot run on a build without the encoder.
  universal:11  'fix halt' defaults to a SOFT halt that ends the run early and
                exits 0; its attribute may only be 'tlimit' or an equal-style
                variable. 'fix print' substitutes ${var} at EXECUTION time, so
                an undefined variable kills the run after the first stats row
                rather than at parse time.

The soft-halt claim is asserted on the EXIT STATUS and on the last Step of the
table against the argument of 'run' — a comparison between what the deck asked
for and what it got, with no number stored. The deferred-substitution claim is
asserted on the ORDER of two lines in one log.

Mutation control: T2_MUTATE=1 gives the halt deck 'error hard', the single
keyword that turns the silent early stop into a nonzero exit, and defines the
variable the print deck was missing. `a_soft_halt_exits_zero` and
`an_undefined_variable_survives_setup` both go False. Nothing else changes:
same box, same condition, same seed, same run length.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import (  # noqa: E402
    col, errors, run, run_keep, skip_if_unavailable, stats_rows,
)

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("ar.species", "ar.vss")

BOX = """seed 12345
dimension 2
boundary p p p
create_box 0 1e-3 0 1e-3 -0.5 0.5
create_grid 10 10 1
species ar.species Ar
mixture gas Ar temp 300.0
global nrho 1e21 fnum 1e12
create_particles gas n 0
collide vss gas ar.vss
timestep 1e-8
compute gt temp
{body}
stats {stats}
stats_style step np
run {nsteps}
"""


def go(body: str, stats: int = 50, nsteps: int = 100, keep: bool = False):
    deck = BOX.format(body=body, stats=stats, nsteps=nsteps)
    if keep:
        rc, txt, work = run_keep(deck, DATA)
        return rc, errors(txt), txt, work
    rc, txt = run(deck, DATA)
    return rc, errors(txt), txt, None


if MUTATE:
    print("mutation=halt_made_hard_and_the_missing_print_variable_defined")

# universal:10 — the argument order.
rc_bad, e_bad, _, _ = go("dump im all image 50 img.*.ppm type type")
ORDER_MSG = "ERROR: Unrecognized dump style (../output.cpp:538)"
order_rejected = any(ORDER_MSG in e for e in e_bad)
print(f"style_before_group_is_rejected={order_rejected}")
# and the message quotes no style name at all, which is why it misleads
print(f"the_message_names_no_style="
      f"{order_rejected and all('all' not in e.split('style')[-1] for e in e_bad)}")

rc_ok, e_ok, _, work = go("dump im image all 50 img.*.ppm type type", keep=True)
imgs = sorted(p.name for p in work.glob("img.*.ppm")) if work else []
if work:
    shutil.rmtree(work, ignore_errors=True)
print(f"the_correct_order_runs_and_writes={rc_ok == 0 and not e_ok and len(imgs) > 1}")

rc_w, e_w, _, work_w = go("dump dg grid all 50 dg.* id xc yc", keep=True)
many = sorted(p.name for p in work_w.glob("dg.*")) if work_w else []
if work_w:
    shutil.rmtree(work_w, ignore_errors=True)
rc_o, e_o, _, work_o = go("dump dg grid all 50 dg.out id xc yc", keep=True)
one = sorted(p.name for p in work_o.glob("dg*")) if work_o else []
if work_o:
    shutil.rmtree(work_o, ignore_errors=True)
print(f"a_star_gives_one_file_per_snapshot={len(many) > 1}")
print(f"no_star_gives_one_file_for_all={one == ['dg.out']}")

rc_nb, e_nb, _, _ = go("compute g grid all all nrho\n"
                       "dump dg grid all 50 dg.* id c_g")
DUMPGRID_MSG = ("ERROR: Dump grid compute does not calculate per-grid vector "
                "(../dump_grid.cpp:515)")
print(f"a_per_grid_compute_needs_its_bracket="
      f"{any(DUMPGRID_MSG in e for e in e_nb)}")
rc_b, e_b, _, _ = go("compute g grid all all nrho\n"
                     "dump dg grid all 50 dg.* id c_g[1]")
print(f"with_the_bracket_it_runs={rc_b == 0 and not e_b}")

rc_mv, e_mv, _, _ = go("dump mv movie all 50 mov.mp4 type type")
MOVIE_MSG = ("ERROR on proc 0: Support for writing movies not included "
             "(../dump_movie.cpp:52)")
movie_inert = any(MOVIE_MSG in e for e in e_mv)
print(f"movie_parses_but_cannot_run={movie_inert}")

# universal:11 — fix halt.
NSTEPS = 200
HARD = " error hard" if MUTATE else ""
rc_h, e_h, txt_h, _ = go(f"variable n equal np\n"
                         f"fix hl halt 20 v_n < 100000000{HARD}",
                         stats=20, nsteps=NSTEPS)
h_h, r_h = stats_rows(txt_h)
last_step = col(h_h, r_h, "Step")[-1] if (h_h and r_h) else None
stopped_early = last_step is not None and last_step < NSTEPS
print(f"the_halt_condition_stops_the_run_early={stopped_early}")
soft_exits_zero = stopped_early and rc_h == 0 and not e_h
print(f"a_soft_halt_exits_zero={soft_exits_zero}")
if not soft_exits_zero and not MUTATE:
    print(f"UNEXPECTED: soft halt gave rc={rc_h} errors={e_h}")
HALT_LINE = "Fix halt condition for fix-id hl met on step"
print(f"and_says_so_in_one_line={HALT_LINE in txt_h}")

rc_hh, e_hh, txt_hh, _ = go("variable n equal np\n"
                            "fix hl halt 20 v_n < 100000000 error hard",
                            stats=20, nsteps=NSTEPS)
print(f"error_hard_makes_it_a_nonzero_exit="
      f"{rc_hh != 0 and any(HALT_LINE in e for e in e_hh)}")

rc_at, e_at, _, _ = go("fix hl halt 20 np < 100000000", stats=20)
ATTR_MSG = "ERROR: Invalid fix halt attribute np (../fix_halt.cpp:62)"
print(f"a_bare_stats_keyword_is_not_a_halt_attribute="
      f"{any(ATTR_MSG in e for e in e_at)}")

# universal:11 — fix print substitutes at execution time.
PRE = "variable x equal c_gt\n" if MUTATE else ""
rc_p, e_p, txt_p, _ = go(PRE + 'fix pr print 25 "tempnow ${x}"', stats=50)
SUBST_MSG = ("ERROR on proc 0: Substitution for illegal variable "
             "(../input.cpp:531)")
died_late = any(SUBST_MSG in e for e in e_p)
survived_setup = died_late and "Step Np" in txt_p and txt_p.index(
    "Step Np") < txt_p.index("Substitution for illegal variable")
print(f"an_undefined_variable_survives_setup={survived_setup}")
if not survived_setup and not MUTATE:
    print(f"UNEXPECTED: fix print deck gave rc={rc_p} errors={e_p}")

rc_pd, e_pd, txt_pd, _ = go('fix pr print 25 "tempnow ${x}"\n'
                            "variable x equal c_gt", stats=50)
print(f"a_variable_defined_after_the_fix_still_works="
      f"{rc_pd == 0 and not e_pd and 'tempnow' in txt_pd}")

ok = (order_rejected and rc_ok == 0 and not e_ok and len(imgs) > 1
      and len(many) > 1 and one == ["dg.out"]
      and any(DUMPGRID_MSG in e for e in e_nb) and rc_b == 0 and not e_b
      and movie_inert and stopped_early and soft_exits_zero
      and HALT_LINE in txt_h and rc_hh != 0
      and any(ATTR_MSG in e for e in e_at) and survived_setup
      and rc_pd == 0 and not e_pd)
print("RESULT=" + ("pass" if ok else "fail"))
sys.exit(0 if ok else 1)
