#!/bin/bash
# Tier-2 for dealii navier_stokes#0 -- probe "linear_solve_is_stokes" of the shared
# translation unit _shared/ns_family.cc, compiled once and cached so every
# fixture naming a probe of that unit shares ONE C++ build. deal.II fixtures
# cost a C++ build, so the build count must not grow with the claim count.
#
# Lid-driven cavity at nu = 1/200. One linear solve with the convective term
# never assembled is compared, in the SAME run, against a Stokes reference and
# against a Newton-converged Navier-Stokes reference reached by continuation.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" ns_family release linear_solve_is_stokes
