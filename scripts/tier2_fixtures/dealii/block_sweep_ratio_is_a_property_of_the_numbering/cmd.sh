#!/bin/bash
# Tier-2 for dealii dg_transport#3 -- probe "block_sweep_regimes" of the shared DG
# translation unit _shared/dg_family.cc, compiled once and cached so every fixture
# that names a probe of that unit shares ONE C++ build.
#
# Pure upwind DG transport (FE_DGQ(1), no reaction, step inflow datum) on three
# globally refined meshes, 256 / 1024 / 4096 dofs, identical operator, rhs and
# relative tolerance. For each mesh the probe measures the entry's own diagnostic
# -- the ONE-APPLICATION residual ||b - A*P(b)||/||b|| -- and the GMRES counts for
# PreconditionBlockSSOR (block_size = fe.dofs_per_cell), point SSOR and point
# Jacobi. deal.II's default cell ordering is what both runs use; no renumbering
# is applied.
#
# The measurement is the entry's regime (a), and it is the reason a single
# speed-up figure is meaningless:
#   constant beta = (1,1):  one-application residual 1.8e-15 / 2.5e-15 / 3.6e-15
#       -- the block sweep is not a preconditioner but an exact block-triangular
#       DIRECT SOLVE. GMRES then takes 2 iterations at EVERY mesh size, while
#       point Jacobi takes 82 / 239 / 389, so the "speed-up" grows 41x -> 195x
#       across three refinements. Quoting one ratio quotes one mesh.
#   rotation about the corner (T2_MUTATE=1, the entry's regime (c)): residual
#       0.69 / 0.65 / 0.61 -- an ordinary preconditioner -- block counts grow
#       8 -> 25 with the mesh, and the advantage settles at a roughly mesh-stable
#       11x-18x over point Jacobi and 3.4x-3.6x over point SSOR.
# The entry's clause (e) is also measured. On THIS downstream-compatible problem
# point SSOR is not within a small factor of the block version (32 against 2 on
# the finest mesh); it is in the rotation regime, where both are real
# preconditioners.
#
# Mutation control: T2_MUTATE=1 makes the probe run the corner-rotation field,
# for which no downstream ordering exists, and the fixture then fails its own
# expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" dg_family release block_sweep_regimes
