#!/bin/bash
# Tier-2 for dealii hyperelasticity#4 -- probe "volumetric_locking" of the shared
# translation unit _shared/hyperelastic_family.cc, compiled once and cached so
# every fixture that names a probe of that unit shares ONE C++ build.
#
# Bending-dominated Neo-Hookean cantilever (1 x 0.1, 16x2 Q1, end shear
# traction), plane strain, nu swept 0.3 / 0.45 / 0.49 / 0.4999. Every nu is
# solved twice: single-field Q1 with full integration, and Q1 with the
# volumetric term integrated at the cell centre only, which is the
# Malkus-Hughes equivalent of the Q1/P0 mixed element. A 64x8 Q2 run at
# nu = 0.4999 is the independent reference. The step-44 three-field element
# itself is NOT built here.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" hyperelastic_family release volumetric_locking
