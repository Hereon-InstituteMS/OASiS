"""Tier-2: per-surf/per-grid computes are ARRAYS, and mixtures have ONE group.

Two rules that together decide whether a SPARTA diagnostic means what you
think:

  1. `compute surf` and `compute grid` always produce a per-surf/per-grid
     ARRAY, even with one value and one group, so the bare compute ID is never
     a valid input to fix ave/* or dump.
  2. The column count is ngroup*nvalue, and ngroup is 1 for the built-in `all`
     mixture, for a mixture defined WITHOUT the `group` keyword, and even for
     one defined with `group <name>`. Only `group SELF` (or the built-in
     `species` mixture) gives one group per species — so a two-species mixture
     yields ONE column unless you ask for SELF, and reading it as "the nitrogen
     flux" is silently wrong.

A fix ave/* fed one value is the opposite: a VECTOR, read bare as f_ID.

Mutation control: T2_MUTATE=1 gives the two mixtures that read c_g[2] the
`group SELF` keyword — the one defined without any `group` and the one defined
with the named `group lite` — so both acquire one group per species and the
second column they ask for exists. That removes the pathology (asking for a
per-species column from a mixture that has one group) and nothing else: same
two species, same compute, same reduce, same run. Both decks then exit 0, so
`default_mixture_has_one_group` and `named_group_is_still_one_group` go False.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("air.species", "air.vss")

BASE = """seed 12345
dimension 2
boundary rr rr p
create_box 0 1e-4 0 1e-4 -0.5 0.5
create_grid 10 10 1
species air.species N O
mixture m2 N O vstream 0 0 0 temp 273.15{grp}
global nrho 7.07043e22 fnum 7.07043e11
collide vss m2 air.vss
create_particles m2 n 0
compute g grid all {mix} n
{extra}timestep 1e-9
stats 50
stats_style step np {cols}
run 50
"""


def probe(mix="m2", grp="", extra="", cols="c_r1"):
    rc, txt = run(BASE.format(mix=mix, grp=grp, extra=extra, cols=cols), DATA)
    hdr, rows = stats_rows(txt)
    return rc, hdr, rows, errors(txt)


# 1. bare compute ID into compute reduce -> not a vector
rc_bare, _, _, e_bare = probe(extra="compute r1 reduce ave c_g\n")
# 2. indexed column 1 -> fine
rc_c1, h1, r1, _ = probe(extra="compute r1 reduce ave c_g[1]\n")
# Under mutation the pathology is REMOVED: the two mixtures whose second column
# is asked for are given per-species groups, so that column exists.
SELF = " group SELF"
if MUTATE:
    print("mutation=both_c_g2_mixtures_given_group_SELF")
# 3. default mixture, column 2 -> out of range (only ONE group)
rc_c2, _, _, e_c2 = probe(grp=SELF if MUTATE else "",
                          extra="compute r1 reduce ave c_g[2]\n")
# 4. mixture with a NAMED group -> still one group
rc_named, _, _, e_named = probe(grp=SELF if MUTATE else " group lite",
                                extra="compute r1 reduce ave c_g[2]\n")
# 5. group SELF -> two columns
rc_self, h_self, r_self, _ = probe(
    grp=" group SELF",
    extra="compute r1 reduce ave c_g[1]\ncompute r2 reduce ave c_g[2]\n",
    cols="c_r1 c_r2")
# 6. built-in 'species' mixture -> two columns
rc_sp, h_sp, r_sp, _ = probe(
    mix="species",
    extra="compute r1 reduce ave c_g[1]\ncompute r2 reduce ave c_g[2]\n",
    cols="c_r1 c_r2")
# 7. fix ave/grid with ONE value -> per-grid VECTOR, read bare
rc_vec, h_v, r_v, _ = probe(
    extra="fix fg ave/grid all 1 50 50 c_g[1]\ncompute r1 reduce ave f_fg\n")
# 8. same fix indexed -> error
rc_vec_idx, _, _, e_vidx = probe(
    extra="fix fg ave/grid all 1 50 50 c_g[1]\ncompute r1 reduce ave f_fg[1]\n")

print(f"bare_compute_rc={rc_bare} err={e_bare[:1]}")
print(f"indexed_compute_rc={rc_c1} value={col(h1, r1, 'c_r1')[-1] if r1 else None}")
print(f"default_mixture_col2_rc={rc_c2} err={e_c2[:1]}")
print(f"named_group_col2_rc={rc_named} err={e_named[:1]}")
if r_self:
    print(f"group_self_cols={col(h_self, r_self, 'c_r1')[-1]:.3f} "
          f"{col(h_self, r_self, 'c_r2')[-1]:.3f}")
if r_sp:
    print(f"species_mixture_cols={col(h_sp, r_sp, 'c_r1')[-1]:.3f} "
          f"{col(h_sp, r_sp, 'c_r2')[-1]:.3f}")
print(f"fix_single_value_bare_rc={rc_vec}")
print(f"fix_single_value_indexed_rc={rc_vec_idx} err={e_vidx[:1]}")

checks = {
    "bare_compute_grid_is_not_a_vector":
        rc_bare != 0 and any("does not calculate a per-grid vector" in e
                             for e in e_bare),
    "indexed_compute_grid_works": rc_c1 == 0,
    "default_mixture_has_one_group":
        rc_c2 != 0 and any("array is accessed out-of-range" in e for e in e_c2),
    "named_group_is_still_one_group":
        rc_named != 0 and any("array is accessed out-of-range" in e
                              for e in e_named),
    "group_self_gives_per_species_columns": rc_self == 0 and bool(r_self),
    "builtin_species_mixture_gives_per_species_columns":
        rc_sp == 0 and bool(r_sp),
    "single_value_fix_is_a_vector": rc_vec == 0,
    "single_value_fix_indexed_aborts": rc_vec_idx != 0,
}
for k, v in checks.items():
    print(f"{k}={v}")
if not all(checks.values()):
    print("FAIL: fixture expectations not met")
    sys.exit(1)
