#!/bin/bash
# Tier-2 for dealii hp_adaptive#1 — probe "fe_index_past_collection" of the
# shared hp translation unit _shared/hp_family.cc, compiled once and cached so
# seven fixtures share one build.
#
# The entry's own shape: a 2-entry hp::FECollection with one cell set to
# active_fe_index 5, then distribute_dofs(). The two build types answer
# completely differently, and the Release answer is the dangerous one:
#   Release  SIGSEGV, rc=139, no output after "before_distribute_dofs" —
#            no exception, and no chance to read n_dofs() == 0 either
#   Debug    Assert ABORT, rc=134, from source/dofs/dof_handler.cc with the
#            violated condition cell->active_fe_index() < ff.size() and the
#            message "The mesh contains a cell with an active FE index of 5, but
#            the finite element collection only has 2 elements"
# That Assert is in the COMPILED library, so -DDEBUG on the consumer cannot
# bring it back — only the Debug library can, which is why this fixture sets
# requires_debug and runs both.
#
# The grep below checks the entry's other statement: the ExcMessage it used to
# promise, "Index in FECollection out of range", is in no header of this tree.
#
# Mutation control: T2_MUTATE=1 uses index 1, inside the collection, and both
# builds then return normally with rc=0, so the fixture fails its expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SHARED="$HERE/../_shared"

echo -n "phantom_exception_string_files="
grep -rl "Index in FECollection out of range" \
  /home/alexander/dealii/include /home/alexander/dealii/source 2>/dev/null | wc -l

for variant in release debug; do
  echo "=== variant=$variant"
  out="$(bash "$SHARED/run.sh" hp_family "$variant" fe_index_past_collection 2>&1 \
         | grep -vE '^(/media/|/lib/|/usr/lib|\[0x|#[0-9])')"
  echo "$out"
  rc="$(printf '%s\n' "$out" | sed -n 's/^exit_code=//p' | tail -1)"
  echo "summary_${variant}_rc=${rc}"
  printf '%s\n' "$out" | grep -q "^after_distribute_dofs$" \
    && echo "summary_${variant}_distribute_dofs_returned=true" \
    || echo "summary_${variant}_distribute_dofs_returned=false"
done
