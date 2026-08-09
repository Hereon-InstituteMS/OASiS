#!/bin/bash
# Tier-2 for dealii hyperelasticity#3 -- probe "geometric_term" of the shared
# translation unit _shared/hyperelastic_family.cc, compiled once and cached so
# every fixture that names a probe of that unit shares ONE C++ build.
#
# A Neo-Hookean cantilever is driven to a tip deflection of 0.90 of its length
# with the consistent tangent, and at THAT state both tangents are assembled:
# K_mat + K_geo and K_mat alone. The probe runs exactly the measurement the
# claim asks for (max|A_ij - A_ji| over the assembled matrix) and, in addition,
# a central-difference directional-derivative check of K against the residual.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" hyperelastic_family release geometric_term
