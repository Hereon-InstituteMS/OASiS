#!/bin/bash
# Tier-2 for dealii nonlinear_elasticity#2 -- probe "missing_tangent_term_rate" of the shared
# translation unit _shared/hyperelastic_family.cc, compiled once and cached so
# every fixture naming a probe of that unit shares ONE C++ build.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" hyperelastic_family release missing_tangent_term_rate
