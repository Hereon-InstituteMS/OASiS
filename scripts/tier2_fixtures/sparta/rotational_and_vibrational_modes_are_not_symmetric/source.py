"""Tier-2: rotational relaxation is on by default, vibrational is not, and the
two vibrational estimators do not agree.

  collision_relaxation:2  with no 'collide' command at all, create_particles
                          gives every particle zero internal energy, so Trot
                          and Tvib read exactly 0 for the whole run even though
                          the mixture asked for trot and tvib.
  collision_relaxation:4  with a collide command but no collide_modify, Trot
                          relaxes visibly while Tvib is exactly 0.0 on every
                          stats line — rc = 0, no warning.
  collision_relaxation:5  'compute grid <grp> <mix> tvib' and
                          'compute tvib/grid <grp> <mix>' are different
                          estimators whose ratio is roughly constant and far
                          from 1.

WHAT THIS ADDS, and it is a silent failure the entry did not have. The entry
says 'collide_modify vibrate smooth' "seeds the mode from the mixture's tvib at
creation, so Tvib is nonzero from step 0". That is true ONLY when the
collide_modify line precedes create_particles. Move create_particles above it —
which reads perfectly naturally, and is the order half the upstream decks use
for other commands — and Tvib is exactly 0 at step 0 and stays at 0 for hundreds
of steps, with no error and no warning. The two orderings are executed here side
by side and the entry now states the dependency.

Every assertion is a comparison between decks in this script or a within-run
shape (a column that is exactly zero, a column that moves). No temperature is
pinned.

Mutation control: T2_MUTATE=1 gives the deck that had 'collide' but no
'collide_modify' the line 'collide_modify vibrate smooth' ahead of
create_particles — the ordering the same script already shows to be the working
one — so vibration is no longer silently off and nothing else changes. Tvib is
then nonzero on every stats line and
`with_collide_but_no_collide_modify_only_rotation_relaxes` goes False. The
no-collide deck, the two ordering decks and the estimator-ratio comparison are
untouched.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("air.species", "air.vss")

DECK = """seed 12345
dimension 2
boundary p p p
global gridcut 0.0
create_box 0 1e-3 0 1e-3 -0.5 0.5
create_grid 10 10 1
global nrho 1e22 fnum 5e12
species air.species N2
mixture gas N2 temp 273.15 trot 1000.0 tvib 1000.0
{body}compute tr grid all all trot
compute tv grid all all tvib
compute tv2 tvib/grid all all
compute mtr reduce ave c_tr[1]
compute mtv reduce ave c_tv[1]
compute mtv2 reduce ave c_tv2[1]
timestep 1e-7
stats 300
stats_style step np ncoll c_mtr c_mtv c_mtv2
run 1500
"""

COLLIDE = "collide vss gas air.vss\n"
CREATE = "create_particles gas n 0\n"
SMOOTH = "collide_modify vibrate smooth\n"


def probe(body: str) -> dict:
    rc, txt = run(DECK.format(body=body), DATA)
    header, rows = stats_rows(txt)

    def column(name):
        return (col(header, rows, name)
                if (header and rows and name in header) else [])
    return {"rc": rc, "errors": errors(txt),
            "warnings": [l for l in txt.splitlines() if l.startswith("WARNING")],
            "ncoll": column("Ncoll"), "trot": column("c_mtr"),
            "tvib": column("c_mtv"), "tvib_grid": column("c_mtv2")}


if MUTATE:
    print("mutation=default_deck_given_collide_modify_vibrate_smooth")

# Under mutation the pathology is REMOVED: the deck without a collide_modify
# gets one, in the ordering this script shows to work. Nothing else changes.
no_collide = probe(CREATE)
default = probe(COLLIDE + SMOOTH + CREATE if MUTATE else COLLIDE + CREATE)
smooth_first = probe(COLLIDE + SMOOTH + CREATE)
create_first = probe(COLLIDE + CREATE + SMOOTH)

for tag, r in (("no_collide_command", no_collide),
               ("collide_but_no_collide_modify", default),
               ("collide_modify_before_create_particles", smooth_first),
               ("create_particles_before_collide_modify", create_first)):
    print(f"{tag}_rc={r['rc']} n_errors={len(r['errors'])} "
          f"n_warnings={len(r['warnings'])}")
    print(f"{tag}_trot_column={[round(v, 3) for v in r['trot']]}")
    print(f"{tag}_tvib_column={[round(v, 3) for v in r['tvib']]}")


def all_exactly_zero(c):
    return bool(c) and all(v == 0.0 for v in c)


def moves(c):
    return bool(c) and max(c) > 0.0 and max(c) != min(c)


all_clean = all(r["rc"] == 0 and not r["errors"] and not r["warnings"]
                for r in (no_collide, default, smooth_first, create_first))
no_collide_kills_both = (all_exactly_zero(no_collide["trot"])
                         and all_exactly_zero(no_collide["tvib"])
                         and all_exactly_zero(no_collide["ncoll"]))
default_relaxes_rot_only = (moves(default["trot"])
                            and all_exactly_zero(default["tvib"])
                            and all_exactly_zero(default["tvib_grid"]))
smooth_first_seeds_at_step_zero = (bool(smooth_first["tvib"])
                                   and smooth_first["tvib"][0] > 0.0)
create_first_does_not = (bool(create_first["tvib"])
                         and create_first["tvib"][0] == 0.0)
ordering_is_silent = (create_first["rc"] == 0 and not create_first["errors"]
                      and not create_first["warnings"])

ratios = [b / a for a, b in zip(smooth_first["tvib"], smooth_first["tvib_grid"])
          if a > 0.0]
ratio_far_from_one = bool(ratios) and min(ratios) > 1.5
ratio_roughly_constant = (bool(ratios)
                          and (max(ratios) - min(ratios)) / max(ratios) < 0.1)

print(f"every_run_exits_zero_with_no_error_and_no_warning={all_clean}")
print(f"without_a_collide_command_both_internal_modes_are_exactly_zero="
      f"{no_collide_kills_both}")
print(f"with_collide_but_no_collide_modify_only_rotation_relaxes="
      f"{default_relaxes_rot_only}")
print(f"collide_modify_BEFORE_create_particles_seeds_tvib_at_step_0="
      f"{smooth_first_seeds_at_step_zero}")
print(f"create_particles_FIRST_leaves_tvib_at_exactly_zero="
      f"{create_first_does_not}")
print(f"the_ordering_mistake_is_completely_silent={ordering_is_silent}")
print(f"tvib_estimator_ratio_min={min(ratios) if ratios else 0:.4g} "
      f"max={max(ratios) if ratios else 0:.4g}")
print(f"the_two_vibrational_estimators_disagree_by_far_more_than_noise="
      f"{ratio_far_from_one}")
print(f"and_their_ratio_is_roughly_constant={ratio_roughly_constant}")

ok = (all_clean and no_collide_kills_both and default_relaxes_rot_only
      and smooth_first_seeds_at_step_zero and create_first_does_not
      and ordering_is_silent and ratio_far_from_one and ratio_roughly_constant)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
