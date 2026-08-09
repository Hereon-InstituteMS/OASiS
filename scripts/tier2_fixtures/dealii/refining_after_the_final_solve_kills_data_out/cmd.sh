#!/bin/bash
# Tier-2 for dealii hp_adaptive#0 — probe "refine_after_final_solve" of the
# shared hp translation unit _shared/hp_family.cc, compiled once and cached so
# seven fixtures share one build.
#
# Three adaptive cycles; the loop body ends with
# execute_coarsening_and_refinement() on EVERY cycle, including the last, and
# the closing DataOut then reads the (dof_handler, solution) pair the refinement
# has just invalidated. The probe prints the stale pairing first
# (solution_vector_size and dof_handler_n_dofs still at the old count while the
# triangulation already has the new cells), then "before_data_out".
#
# Run against BOTH libraries on this host, because the two answers are different
# and only one of them is a diagnosis:
#   Release  SIGSEGV, rc=139, no message at all, right after the last "Cycle N:"
#   Debug    Assert ABORT, rc=134, inside DataOut::build_patches ->
#            get_interpolated_dof_values -> get_dof_values, naming the stale
#            index: "Index 100 is not in the half-open range [0,70)."
# Assert aborts; it does not throw, so the exit code is the observable and a
# try/catch would see nothing.
#
# NOT REPRODUCED on 9.8.0-pre: the entry's alternative symptom, an ExcOutOfMemory
# from posix_memalign asking for ~8.5e18 bytes. That was a 9.3.2 observation.
#
# Mutation control: T2_MUTATE=1 breaks out of the loop before marking on the
# final cycle — the fix every tutorial uses — and both builds then run to
# completion with rc=0, so the fixture fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SHARED="$HERE/../_shared"

for variant in release debug; do
  echo "=== variant=$variant"
  out="$(bash "$SHARED/run.sh" hp_family "$variant" refine_after_final_solve 2>&1 \
         | grep -vE '^(/media/|/lib/|/usr/lib|\[0x|#[0-9])')"
  echo "$out"
  rc="$(printf '%s\n' "$out" | sed -n 's/^exit_code=//p' | tail -1)"
  echo "summary_${variant}_rc=${rc}"
  printf '%s\n' "$out" | grep -q "^after_data_out$" \
    && echo "summary_${variant}_data_out_completed=true" \
    || echo "summary_${variant}_data_out_completed=false"
done
