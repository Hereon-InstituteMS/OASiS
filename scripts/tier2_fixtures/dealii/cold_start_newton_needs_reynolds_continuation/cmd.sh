#!/bin/bash
# Tier-2 for dealii navier_stokes#2 -- probe "reynolds_continuation" of the shared
# translation unit _shared/ns_family.cc, compiled once and cached so every
# fixture naming a probe of that unit shares ONE C++ build. deal.II fixtures
# cost a C++ build, so the build count must not grow with the claim count.
#
# 32x32 Taylor-Hood cavity at Re = 700. A cold start from the zero vector
# against continuation through Re = 10, 50, 100, 200, 400.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" ns_family release reynolds_continuation
