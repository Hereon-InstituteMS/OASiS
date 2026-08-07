#!/bin/bash
# Tier-2 for dealii wave#3 — probe "zero_initial_acceleration" of the shared
# wave translation unit _shared/wave_family.cc, compiled once and cached so six
# fixtures share one build.
#
# Two integrations of the same problem, same mesh, same dt, in one process: one
# starts from a_0 = 0, the other from the a_0 that solves M a_0 = -c^2 K u_0.
# a_0 is the only difference, so the deviation between them IS the pitfall.
#
# The entry's magnitudes are measured, not repeated: it promises an O(1) error
# in the first ten steps decaying to O(h^p) later, and a superimposed mode of
# magnitude 0.1-0.3. What this run shows is a deviation of a few percent that
# does NOT decay — the fixture pins both of those as false.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant (solve
# for a_0), the two integrations become identical, and the fixture then fails
# its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" wave_family release zero_initial_acceleration
