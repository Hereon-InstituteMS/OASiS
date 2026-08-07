"""Tier-2: the three silent ways a SPARTA deck ends up at the wrong density.

  * create_particles n <N>0> ignores global nrho and makes exactly N;
  * a nrho keyword on the MIXTURE overrides global nrho;
  * global nrho/fnum issued after create_particles is too late -> 0 particles;
  * a region argument scales the requested count by the volume fraction and
    suppresses SPARTA's own 'Created unexpected # of particles' warning.

All four exit 0 with no warning.

Mutation control: T2_MUTATE=1 writes each of the four wrong decks the way the
documentation prescribes, one token per deck and nothing else — 'n 500' becomes
'n 0', the 'nrho 1e21' keyword comes off the mixture line, the global
nrho/fnum line moves back in front of create_particles, and the 'region half'
restriction comes off. Every deck then lands on the density 'global nrho' asks
for, so `n_nonzero_overrides_nrho`, `mixture_nrho_overrides_global`,
`global_after_create_particles_gives_zero` and `region_scales_the_requested_count`
all go False.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import (  # noqa: E402
    col, errors, run, skip_if_unavailable, stats_rows,
)

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("ar.species")

BASE = """seed 12345
dimension 2
boundary rr rr p
create_box 0 1e-4 0 1e-4 -0.5 0.5
create_grid 10 10 1
species ar.species Ar
mixture gas Ar vstream 0 0 0 temp 273.15{mixnrho}
{globals}{region}create_particles gas n {n}{regarg}
{late}timestep 1e-9
stats 100
stats_style step np
run 100
"""

GLOB = "global nrho 7.07043e22 fnum 7.07043e11\n"


def created(deck: str):
    rc, txt = run(deck, DATA)
    n = None
    for line in txt.splitlines():
        if line.startswith("Created ") and line.endswith("particles"):
            n = int(line.split()[1])
    return rc, n, errors(txt)


def build(n="0", mixnrho="", globals_=GLOB, late="", region="", regarg=""):
    return BASE.format(n=n, mixnrho=mixnrho, globals=globals_, late=late,
                       region=region, regarg=regarg)


# Under mutation each wrong deck loses the ONE token that is the pathology and
# nothing else: the explicit count, the mixture nrho keyword, the late globals,
# the region restriction.
rc0, n0, _ = created(build(n="0"))
rc1, n1, _ = created(build(n="0" if MUTATE else "500"))
rc2, n2, _ = created(build(n="0", mixnrho="" if MUTATE else " nrho 1e21"))
rc3, n3, _ = created(build(n="0",
                           globals_=GLOB if MUTATE else "",
                           late="" if MUTATE else GLOB))
rc4, n4, _ = created(build(
    n="500",
    region="" if MUTATE else "region half block 0 5e-5 INF INF INF INF\n",
    regarg="" if MUTATE else " region half"))
if MUTATE:
    print("mutation=every_wrong_density_deck_written_the_prescribed_way")

print(f"created_n0={n0}")
print(f"created_n500={n1}")
print(f"created_mixture_nrho_1e21={n2}")
print(f"created_global_after={n3}")
print(f"created_n500_in_half_region={n4}")
print(f"all_runs_exit_zero={all(r == 0 for r in (rc0, rc1, rc2, rc3, rc4))}")
print(f"n_zero_honours_nrho={n0 is not None and 990 <= n0 <= 1010}")
print(f"n_nonzero_overrides_nrho={n1 == 500}")
print(f"mixture_nrho_overrides_global={n2 is not None and n2 < n0 / 50}")
print(f"global_after_create_particles_gives_zero={n3 == 0}")
print(f"region_scales_the_requested_count={n4 is not None and 200 <= n4 <= 300}")

ok = (all(r == 0 for r in (rc0, rc1, rc2, rc3, rc4))
      and n0 is not None and 990 <= n0 <= 1010 and n1 == 500
      and n2 is not None and n2 < n0 / 50 and n3 == 0
      and n4 is not None and 200 <= n4 <= 300)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
