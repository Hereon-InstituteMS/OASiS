#!/bin/bash
# Tier-2 for dealii navier_stokes#4 -- probe "pressure_level_undetermined" of the shared
# translation unit _shared/ns_family.cc, compiled once and cached so every
# fixture naming a probe of that unit shares ONE C++ build. deal.II fixtures
# cost a C++ build, so the build count must not grow with the claim count.
#
# Two Newton solves of the SAME closed-cavity problem that differ only in the
# pressure level of the starting vector, each inner solve by unpreconditioned
# GMRES so the iteration counts per outer step are visible.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" ns_family release pressure_level_undetermined
