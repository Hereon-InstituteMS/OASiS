#!/bin/bash
# Tier-2 for dealii advection_dg#1 -- probe "sipg_penalty" of the shared DG
# translation unit _shared/dgip_family.cc, compiled once and cached so every
# fixture that names a probe of that unit shares ONE C++ build.
#
# SIPG Poisson, FE_DGQ(1), exact solution sin(pi x) sin(pi y) with Nitsche
# boundary data, penalty alpha/h on every face. The entry's rule for p = 1 is
# alpha = 4*(p+1)^2 = 16; the run under test uses alpha = 0.1.
#
# Coercivity is measured, not argued: on a 256-dof mesh the whole matrix goes to
# LAPACK and the smallest eigenvalue is computed exactly (the form is symmetric,
# relative asymmetry 0 to roundoff).
#   alpha = 0.01 / 0.1 / 1    min eigenvalue -1.03 / -0.97 / -0.44  NOT definite
#   alpha = 16 (the rule)     min eigenvalue +0.074                 definite
# So the "coercivity loss" half of the entry reproduces, and this is the sharp
# way to see it.
#
# Two quantitative clauses do NOT reproduce and the fixture pins them as
# measured:
#   "the L2 norm from integrate_difference diverges with mesh refinement" -- at
#       alpha = 0.1 the L2 error still falls 0.0478 -> 0.0085 -> 0.0020 over three
#       refinements, a rate of 2.11, slightly WORSE in magnitude than the rule's
#       (0.0294 -> 0.0075 -> 0.0019, rate 1.99) but converging at full order. Loss
#       of positive definiteness is not visible in the error on this problem.
#   "alpha too large -> condition number > 1e14 and SolverGMRES stagnates" -- the
#       stagnation reproduces (unpreconditioned GMRES exhausts a 2000-iteration
#       budget at alpha = 1e8 and 1e12) but the condition number at alpha = 1e12
#       is 1.9e13, still below the quoted 1e14. The threshold is quoted about an
#       order of magnitude too high for this mesh.
# GMRES here runs UNPRECONDITIONED on purpose: with a Jacobi preconditioner the
# huge penalty scales the diagonal, the stopping criterion is met at step 0 and
# the stagnation is hidden.
#
# Mutation control: T2_MUTATE=1 makes the probe use the entry's rule
# alpha = 4*(p+1)^2, and the fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" dgip_family release sipg_penalty
