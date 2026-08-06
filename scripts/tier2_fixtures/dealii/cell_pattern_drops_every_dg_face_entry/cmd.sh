#!/bin/bash
# Tier-2 for dealii advection_dg#0 -- probe "flux_sparsity_pattern" of the shared
# DG-transport translation unit _shared/dgtransport_family.cc, compiled once and
# cached so eleven fixture directories share ONE C++ build.
#
# DoFTools::make_sparsity_pattern instead of make_flux_sparsity_pattern for an
# upwind DG assembly. The two patterns are built for the same DoFHandler and
# counted: the flux pattern holds 4.5x the non-zeros of the cell-only one on a
# globally refined FE_DGQ(1) mesh.
#
# WHAT HAPPENS NEXT DEPENDS ON THE BUILD TYPE, and the two answers are completely
# different:
#   Release  the Assert in lac/sparse_matrix.h is compiled out, SparseMatrix::add
#            takes the invalid_entry branch, returns normally and SILENTLY DROPS
#            every face contribution. rc=0, and the run even produces a
#            plausible-looking solution with an L2 error two orders too large.
#   Debug    the same call ABORTS (rc=134) with the violated condition
#            (index != SparsityPattern::invalid_entry) || (value == number())
#            and the message "You are trying to access the matrix entry with
#            index <4,1>, but this entry does not exist in the sparsity pattern
#            of this matrix."
# So this fixture runs BOTH libraries and pins the pair; that is why it sets
# requires_debug.
#
# The grep checks the entry's quoted error text. The message it promises,
# "SparseMatrix::add() requires row/col to be in pattern", is in no header and no
# source file of this tree.
#
# Mutation control: T2_MUTATE=1 builds the flux pattern, nothing is dropped, and
# both builds return rc=0.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SHARED="$HERE/../_shared"

echo -n "phantom_add_message_files="
grep -rl "requires row/col to be in pattern" /home/alexander/dealii/include \
  /home/alexander/dealii/source 2>/dev/null | wc -l

for variant in release debug; do
  echo "=== variant=$variant"
  out="$(bash "$SHARED/run.sh" dgtransport_family "$variant" flux_sparsity_pattern 2>&1 \
         | grep -vE '^(/media/|/lib/|/usr/lib|\[0x|#[0-9])')"
  echo "$out"
  rc="$(printf '%s\n' "$out" | sed -n 's/^exit_code=//p' | tail -1)"
  echo "summary_${variant}_rc=${rc}"
  printf '%s\n' "$out" | grep -q "^after_assembly" \
    && echo "summary_${variant}_assembly_returned=true" \
    || echo "summary_${variant}_assembly_returned=false"
done
