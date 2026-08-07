#!/bin/bash
# Tier-2 for dealii dg_advection_reaction#0 -- probe "central_flux" of the shared DG-transport
# translation unit _shared/dgtransport_family.cc, compiled once and cached so
# eleven fixture directories share ONE C++ build.
#
# The central flux 0.5*(u^+ + u^-) against upwind on the same DG transport
# operator. The growth rate of du/dt = -M^{-1} A u is the largest real part of an
# eigenvalue of that operator, so it is computed with LAPACK at two mesh sizes,
# alongside an RK4 integration of a pulse that the field carries out of the box
# before t = 1.
#
# The entry is WRONG about the direction: the central-flux operator has NO
# eigenvalue with positive real part and the amplitude does not grow. What is
# true is that its damping vanishes under refinement -- the rightmost real part
# halves from one level to the next, while upwind's grows -- so the pulse leaves
# a residue in the box more than ten times the upwind one.
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" dgtransport_family release central_flux
