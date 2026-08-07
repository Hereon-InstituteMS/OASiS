#!/bin/bash
# Tier-2 for dealii hyperelasticity#1 -- probe "det_f_guard" of the shared
# translation unit _shared/hyperelastic_family.cc, compiled once and cached so
# every fixture that names a probe of that unit shares ONE C++ build.
#
# Load control (body force 1600 in -y on a Neo-Hookean cantilever, 12x3 Q1), so
# the constraints stay homogeneous and the step length is free on every
# iteration: the ONLY difference between the two variants is whether the Newton
# step is backtracked until every quadrature point still has det F > 0.
# The probe also runs one assembly with the AssertThrow(J > 0, ...) guard the
# claim tells the user to write, to show that the message can only come from
# the user's own code.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" hyperelastic_family release det_f_guard
