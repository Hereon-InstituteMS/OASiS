#!/bin/bash
# Tier-2 for dealii multigrid#2 -- probe "mg_smoother_on_indefinite" of the
# shared multigrid translation unit _shared/multigrid_family.cc, compiled once
# and cached so five fixture directories share ONE C++ build.
#
# The same step-16 hierarchy and the same SOR relaxation smoother, applied to the
# INDEFINITE operator K - 300 M instead of the SPD Laplace -- the shift puts
# several eigenvalues below zero while changing nothing else in the setup. The
# V-cycle is run as a STATIONARY iteration, so the residual of each cycle is
# visible instead of being hidden inside a Krylov method, exactly the
# cycle-to-cycle quantity the entry names.
#
# It does not shrink. It grows by a factor of about 1.9e5 per cycle, from 3.4e-2
# to 1.9e+30 in six cycles.
#
# One thing the entry does not predict: the OUTER CG still reports convergence
# (192 iterations) with this diverging V-cycle as its preconditioner, so watching
# the solver's own success flag would not have caught it -- the per-cycle residual
# is what shows it.
#
# Mutation control: T2_MUTATE=1 removes the shift, the same smoother on the SPD
# Laplace contracts every cycle, and the fixture fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" multigrid_family release mg_smoother_on_indefinite
