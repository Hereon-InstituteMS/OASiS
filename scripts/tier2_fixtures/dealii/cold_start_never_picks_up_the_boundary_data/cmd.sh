#!/bin/bash
# Tier-2 for dealii nonlinear#0 -- probe "cold_start" of the shared translation unit
# _shared/nonlinear_family.cc, compiled once and cached so every fixture that names a
# probe of that unit shares ONE C++ build.
#
# Minimal surface equation (step-15's problem) on the unit square with boundary data of amplitude 8. Three starting points are run in every invocation: the literal all-zero vector, the same zero interior WITH the boundary values interpolated onto it, and a five-step continuation. Every Newton loop here uses a backtracking line search, so what is being isolated is the initial guess and nothing else.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" nonlinear_family release cold_start
