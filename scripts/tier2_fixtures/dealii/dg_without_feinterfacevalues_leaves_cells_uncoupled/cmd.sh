#!/bin/bash
# Tier-2 for dealii convection_diffusion#1 -- probe "dg_face_terms" of the shared translation unit
# _shared/convdiff_family.cc, compiled once and cached so every fixture that names a
# probe of that unit shares ONE C++ build.
#
# Upwind DG for pure advection on a 16x16 mesh with FE_DGQ(1). The probe counts the nonzero matrix entries that connect dofs of DIFFERENT cells, which is zero without the FEInterfaceValues face loop, and then hands the matrix to SparseDirectUMFPACK. The claim's own Signal -- a DG solution whose face jumps drop below 1e-8 -- does NOT happen: there is no solution to look at, because the cell-only operator is singular.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" convdiff_family release dg_face_terms
