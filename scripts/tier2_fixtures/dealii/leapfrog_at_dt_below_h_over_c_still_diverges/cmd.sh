#!/bin/bash
# Tier-2 for dealii time_dependent_wave#1 -- probe "explicit_newmark_cfl" of the
# shared wave translation unit _shared/wave_family.cc, which is compiled once and
# cached, so this fixture adds no build of its own.
#
# Newmark with beta = 0 and gamma = 1/2 IS the leapfrog / central-difference
# scheme the entry names. On the unit square with c = 1 and h = 1/16 the stability
# limit is found by scanning dt, and the entry's rule is wrong in the unsafe
# direction: the claimed bound h/c is about 2.2 times the measured limit, and a
# run at 0.8 * h/c -- which SATISFIES "dt < h/c" -- diverged at step 10. The
# claimed conservative safety factor 0.5 * h/c is also above the measured limit.
# The claimed growth of "~2 per step" is not the size of it either: at 4 * h/c the
# energy grew by a factor of ~60 per step. An implicit run at the same dt stays
# bounded, in the same program.
#
# T2_MUTATE=1 runs the explicit scheme inside the measured stable range, nothing
# diverges, and the fixture fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" wave_family release explicit_newmark_cfl
