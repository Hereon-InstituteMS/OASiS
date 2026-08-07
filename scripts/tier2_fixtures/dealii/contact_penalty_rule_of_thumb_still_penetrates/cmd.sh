#!/bin/bash
# Tier-2 for dealii contact#1 -- probe "contact_penalty" of the shared contact
# translation unit _shared/contact_family.cc, compiled once and cached so three
# fixture directories share ONE C++ build.
#
# The same indenter problem solved by PENALTY regularisation at five penalties
# (1, 1e2, the entry's rule 1e3*E/h = 1.6e4, 1e8, 1e14), each with 25 fixed-point
# sweeps, reporting the worst penetration as a fraction of an element edge, the
# unpreconditioned CG count, and the condition number from the assembled
# spectrum.
#
# The "too small" half of the entry reproduces: penalty 1 leaves 10.4 element
# edges of penetration.
#
# THE OTHER TWO NUMBERS DO NOT. The entry's own rule of thumb, 1e3*E/h, still
# leaves 0.104 of an element edge -- twice the 5% the entry itself sets as the
# failure threshold; it takes about 1e8 here to get under it. And at penalty 1e14
# the condition number is 1.3e13, an order of magnitude BELOW the quoted 1e14, and
# CG does not stagnate -- it converges in ONE iteration, because at that penalty
# the operator is dominated by its own diagonal.
#
# Mutation control: T2_MUTATE=1 puts a penalty of 1e8 under test, the penetration
# drops below the threshold, and the fixture fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" contact_family release contact_penalty
