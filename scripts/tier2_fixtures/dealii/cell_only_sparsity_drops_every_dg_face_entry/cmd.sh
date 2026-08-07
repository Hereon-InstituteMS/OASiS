#!/bin/bash
# Tier-2 for dealii advection_dg#0 -- probe "cell_only_sparsity" of the shared DG
# translation unit _shared/dgip_family.cc, compiled once per build type and cached
# so every fixture that names a probe of that unit shares ONE C++ build.
#
# Upwind FE_DGQ(1) advection on a globally refined unit square, assembled into a
# matrix whose pattern came from DoFTools::make_sparsity_pattern instead of
# make_flux_sparsity_pattern. Both patterns are built for the same DoFHandler in
# every run, so the entry's cheap positive check is measured rather than
# asserted: 4608 against 1024 non-zeros, a ratio of 4.5. A dry run over the
# interior faces then counts 3584 entries the assembly wants that the cell-only
# pattern does not have.
#
# WHAT HAPPENS NEXT DEPENDS ON THE BUILD TYPE, so both run here:
#   Release  rc=0. The Assert in SparseMatrix::add is compiled out, the face
#            contributions are SILENTLY DROPPED, "after_face_assembly" is
#            reached, and the answer is O(1) wrong (max difference 1.18 on a
#            field of amplitude 1.18).
#   Debug    rc=134, an ABORT from sparse_matrix.templates.h with the violated
#            condition "value == number()" and the message "You are trying to
#            access the matrix entry with index <4,1>, but this entry does not
#            exist in the sparsity pattern of this matrix" --
#            "after_face_assembly" is never reached.
# Neither build raises anything a try/catch could see; the Debug path is an
# abort, which is why this fixture asserts on the exit code.
#
# The two greps check the two exception strings this entry and dg_transport::0
# used to promise. Neither string is in any header or source of this tree.
#
# Mutation control: T2_MUTATE=1 makes the probe use
# DoFTools::make_flux_sparsity_pattern, and the fixture then fails its own
# expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SHARED="$HERE/../_shared"

echo -n "phantom_add_message_files="
grep -rl "requires row/col to be in pattern" \
  /home/alexander/dealii/include /home/alexander/dealii/source 2>/dev/null | wc -l
echo -n "phantom_excmessage_files="
grep -rl "matrix entry at i,j does not exist in sparsity pattern" \
  /home/alexander/dealii/include /home/alexander/dealii/source 2>/dev/null | wc -l

for variant in release debug; do
  echo "=== variant=$variant"
  out="$(bash "$SHARED/run.sh" dgip_family "$variant" cell_only_sparsity 2>&1 \
         | grep -vE '^(/media/|/lib/|/usr/lib|\[0x|#[0-9])')"
  echo "$out"
  rc="$(printf '%s\n' "$out" | sed -n 's/^exit_code=//p' | tail -1)"
  echo "summary_${variant}_rc=${rc}"
  if printf '%s\n' "$out" | grep -q "^after_face_assembly$"; then
    echo "summary_${variant}_reached_after_face_assembly=true"
  else
    echo "summary_${variant}_reached_after_face_assembly=false"
  fi
done
