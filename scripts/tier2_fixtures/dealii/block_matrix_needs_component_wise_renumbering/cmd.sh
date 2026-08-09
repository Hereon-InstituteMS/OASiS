#!/bin/bash
# Tier-2 for dealii stokes#1 -- probe "block_renumbering" of the shared translation unit
# _shared/stokes_family.cc, compiled once and cached so every fixture that names a
# probe of that unit shares ONE C++ build.
#
# Taylor-Hood Q2/Q1 on a 2-refinement hyper_cube. The probe prints the block SIZES before and after DoFRenumbering::component_wise -- they are identical, so the sizes are NOT the diagnostic -- and then assembles the BlockSparseMatrix and reports the (1,1) block's Frobenius norm, which is the diagnostic.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" stokes_family release block_renumbering
