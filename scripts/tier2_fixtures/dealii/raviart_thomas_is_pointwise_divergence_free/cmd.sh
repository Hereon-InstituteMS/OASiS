#!/bin/bash
# Tier-2 for dealii stokes#5 -- probe "hdiv_divergence" of the shared translation unit
# _shared/stokes_family.cc, compiled once and cached so every fixture that names a
# probe of that unit shares ONE C++ build.
#
# Every run prints all three numbers the claim compares: the RT_1/DGQ_1 mixed saddle point, the same mixed form with Q2/Q1, and a genuine Taylor-Hood Q2/Q1 STOKES solve, all for the same manufactured, exactly divergence-free velocity. Only the space under test drives the verdict.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" stokes_family release hdiv_divergence
