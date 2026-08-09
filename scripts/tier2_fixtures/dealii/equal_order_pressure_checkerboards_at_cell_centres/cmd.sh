#!/bin/bash
# Tier-2 for dealii navier_stokes#1 -- probe "equal_order_checkerboard" of the shared
# translation unit _shared/ns_family.cc, compiled once and cached so every
# fixture naming a probe of that unit shares ONE C++ build. deal.II fixtures
# cost a C++ build, so the build count must not grow with the claim count.
#
# The claim's own signal, measured where the claim puts it: the pressure at
# adjacent cell centres. Q1/Q1 against Taylor-Hood Q2/Q1 on the same cavity.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" ns_family release equal_order_checkerboard
