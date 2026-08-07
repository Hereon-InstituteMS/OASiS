#!/bin/bash
# Tier-2 for dealii convection_diffusion#2 -- probe "upwind_direction" of the shared translation unit
# _shared/convdiff_family.cc, compiled once and cached so every fixture that names a
# probe of that unit shares ONE C++ build.
#
# The same DG transport with the numerical flux taking the DOWNSTREAM cell value. Two operators are run: pure advection, where the flipped flux gives a singular matrix, and the same transport with a reaction term sigma*u, which is non-singular for both flux choices so the question 'where does the inflow datum end up' has an answer. It does not end up at the outflow: it produces an O(1e3) oscillation instead.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" convdiff_family release upwind_direction
