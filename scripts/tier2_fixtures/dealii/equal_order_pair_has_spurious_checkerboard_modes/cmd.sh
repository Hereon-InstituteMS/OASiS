#!/bin/bash
# Tier-2 for dealii stokes#4 -- probe "equal_order_infsup" of the shared translation unit
# _shared/stokes_family.cc, compiled once and cached so every fixture that names a
# probe of that unit shares ONE C++ build.
#
# The inf-sup constant is measured, not guessed: beta^2 is the smallest eigenvalue of the pressure Schur complement B A^{-1} B^T with respect to the pressure mass matrix, above the numerically null cluster. The checkerboard vector from the claim's own Signal is built explicitly from the pressure support points and its Rayleigh quotient in that Schur complement is reported.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" stokes_family release equal_order_infsup
