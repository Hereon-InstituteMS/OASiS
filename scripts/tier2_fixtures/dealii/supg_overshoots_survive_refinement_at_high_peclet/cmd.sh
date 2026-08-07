#!/bin/bash
# Tier-2 for dealii convection_diffusion#3 -- probe "supg_high_pe" of the shared translation unit
# _shared/convdiff_family.cc, compiled once and cached so every fixture that names a
# probe of that unit shares ONE C++ build.
#
# Skew advection (30 degrees to the mesh) of a discontinuous inflow datum with eps = 1e-6 -- the classic internal-layer test. The exact solution is bounded by its own inflow data, so any excursion outside [0, 1] is the stabilisation failing. Measured at three refinement levels, so 'does it go away when you refine' is answered rather than assumed. T2_MUTATE=1 runs the upwind DG(0) alternative the claim names.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" convdiff_family release supg_high_pe
