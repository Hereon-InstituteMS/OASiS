#!/bin/bash
# Tier-2 for dealii phase_field#0 -- probe "galerkin_wiggles" of the shared
# stabilisation translation unit _shared/stabilisation_family.cc, compiled once
# and cached so three fixture directories share ONE C++ build.
#
# -eps u'' + u' = 0 on (0,1) with eps = 1e-3, u(0) = 0 and u(1) = 1, discretised
# with FE_Q(1) on a strip two cells deep so the problem is exactly 1D and the
# layer solution is known in closed form. The exact solution lives in [0,1], so
# any nodal value outside that range is the wiggle the entry is about, and it is
# measured at six mesh sizes spanning cell Peclet 62.5 down to 0.49.
#
# The entry says the wiggles "do not damp with refinement". They do. The
# undershoot falls monotonically (-7.7, -0.91, -0.59, -0.32) while the cell
# Peclet is above one, and at the first mesh with cell Peclet below one the
# minimum is exactly 0 -- the oscillation is gone, with no stabilisation added.
# What refinement has to reach is not "fine enough" in the abstract, it is cell
# Peclet < 1.
#
# Mutation control: T2_MUTATE=1 adds the SUPG term with the doubly-asymptotic
# tau, the solution stays in [0,1] at EVERY mesh size, and the fixture fails its
# own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" stabilisation_family release galerkin_wiggles
