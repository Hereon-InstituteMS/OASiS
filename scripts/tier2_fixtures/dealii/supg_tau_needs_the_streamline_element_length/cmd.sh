#!/bin/bash
# Tier-2 for dealii navier_stokes#5 -- probe "supg_tau_dimension" of the shared
# translation unit _shared/ns_family.cc, compiled once and cached so every
# fixture naming a probe of that unit shares ONE C++ build. deal.II fixtures
# cost a C++ build, so the build count must not grow with the claim count.
#
# Skew advection at 45 degrees with an outflow boundary layer, four tau choices
# in one run. The exact interior solution is the arclength from the inflow, so
# the overshoot is measured against an analytic maximum, not a reference run.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" ns_family release supg_tau_dimension
