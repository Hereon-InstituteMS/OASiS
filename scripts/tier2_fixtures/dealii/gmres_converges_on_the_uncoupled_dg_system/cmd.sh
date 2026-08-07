#!/bin/bash
# Tier-2 for dealii dg_transport#1 -- probe "gmres_without_face_terms" of the
# shared DG translation unit _shared/dg_family.cc, compiled once and cached so
# every fixture that names a probe of that unit shares ONE C++ build.
#
# Upwind DG advection-reaction (FE_DGQ(1), b = (1, 0.3), sigma = 1, inflow datum
# 1 below y = 0.5 and 0 above) assembled with FEValues ONLY: the cell-interior
# and boundary-face terms are there, the FEInterfaceValues interior-face terms
# are not. A reaction term is present so the cell-local blocks stay invertible
# and the broken operator can actually be handed to a Krylov solver, which is
# what the entry's Signal presupposes.
#
# The entry's Signal has two halves and they do not agree with each other:
#   "SolverGMRES converges"  -- TRUE. Jacobi-preconditioned GMRES reports
#       convergence in 9 steps on the uncoupled operator, against 231 steps for
#       the correctly assembled one. Nothing is raised in any build: the missing
#       face terms are entries never written, not entries written wrongly.
#   "jump-across-face values are 1e-8 (effectively zero) ... where they should
#    be O(1)"  -- FALSE, and backwards. Omitting the interface terms decouples
#       the cells (0 inter-cell matrix couplings against 1920), so nothing
#       smooths anything: the answer is not a continuous-looking field but a
#       per-cell blow-up of amplitude 1.1e5 where the inflow datum is bounded by
#       1, with a maximum interior jump of 1.6e5. The reference assembly in the
#       same invocation is the one that shows the O(1) jump (0.80) the entry
#       expects to be missing.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant
# (interior faces via FEInterfaceValues), and the fixture then fails its own
# expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" dg_family release gmres_without_face_terms
