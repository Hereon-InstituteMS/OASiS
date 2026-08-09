#!/bin/bash
# Tier-2 for dealii linear_elasticity#3 — probe "shape_value_vs_extractor" of the shared elasticity translation
# unit. One compile serves every fixture that names a probe of
# _shared/elasticity_family.cc; deal.II fixtures cost a C++ build, so the build
# count must not grow with the claim count.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" elasticity_family release shape_value_vs_extractor
