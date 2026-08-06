#!/bin/bash
# Tier-2 for dealii advection_dg#1 -- probe "sipg_penalty" of the shared
# stabilisation translation unit _shared/stabilisation_family.cc, compiled once
# and cached so three fixture directories share ONE C++ build.
#
# SIPG for -laplace(u) = f with FE_DGQ(1) and the manufactured solution
# sin(pi x) sin(pi y), penalty sigma_F = alpha * 0.5 * (1/h1 + 1/h2), which is
# deal.II's own step-74 shape with the constant pulled out so that the entry's
# rule alpha = 4 (p+1)^2 = 16 can be dialled in directly.
#
# alpha is swept over 0.01, 0.1, 0.5, 2 and 16, and for each one the SMALLEST
# EIGENVALUE of the assembled matrix is computed with LAPACK next to the L2 error
# at three refinement levels. Coercivity is genuinely lost below alpha ~ 2: the
# matrix is indefinite. But the rest of the entry does NOT follow from it:
#   the L2 error does NOT diverge with refinement, it keeps falling;
#   the solution norm does not run away either, it settles on the exact one;
#   the observed L2 RATE does not degrade -- it is above two at alpha = 0.01;
#   and SolverCG converges on the indefinite matrix anyway, in more steps.
# So the only thing an under-penalised SIPG form loses here is definiteness.
#
# At the other end, alpha = 1e12: GMRES does stagnate (5000 steps, no
# convergence), but the condition number measured from the spectrum is ~7e12, an
# order of magnitude BELOW the 1e14 the entry quotes.
#
# Mutation control: T2_MUTATE=1 uses the rule alpha, the matrix is positive
# definite, and the fixture fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" stabilisation_family release sipg_penalty
