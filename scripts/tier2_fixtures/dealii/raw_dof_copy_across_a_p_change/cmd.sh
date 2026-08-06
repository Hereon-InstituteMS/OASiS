#!/bin/bash
# Tier-2 for dealii hp_adaptive#6 — probe "raw_copy_across_p_change" of the
# shared hp translation unit _shared/hp_family.cc, compiled once and cached so
# seven fixtures share one build.
#
# A field with real content (sin(3 pi x) sin(3 pi y)) interpolated onto an
# all-FE_Q(3) hp DoFHandler, then p-refined to FE_Q(4) on every cell via
# set_future_fe_index + execute_coarsening_and_refinement. The correct path is
# SolutionTransfer; the mistake is writing the old dof values into the new
# vector by index. The L2 error against the original field, measured on the new
# space, is the observable: it goes from O(1e-4) to O(1e-1).
#
# The entry's stated Signal does NOT reproduce: it promises linfty_norm()
# dropping by 10-50%, and here the raw copy leaves linfty EXACTLY unchanged —
# the same values, attached to the wrong degrees of freedom. Reading the norm
# would have shown nothing at all. The fixture pins that as false.
#
# Mutation control: T2_MUTATE=1 uses SolutionTransfer, and the fixture then
# fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" hp_family release raw_copy_across_p_change
