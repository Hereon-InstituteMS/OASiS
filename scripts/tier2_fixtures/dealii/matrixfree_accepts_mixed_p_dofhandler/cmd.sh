#!/bin/bash
# Tier-2 for dealii hp_adaptive#5 — probe "matrixfree_hp_support" of the shared
# hp translation unit _shared/hp_family.cc, compiled once and cached so seven
# fixtures share one build.
#
# This entry is a POSITIVE claim — matrix-free DOES support hp on 9.x — so the
# fixture pins the property and the mutation is a NEGATIVE CONTROL that removes
# it, the same way the eigenvalue template fixtures do. The probe builds two
# DoFHandlers in one process: one whose cells alternate FE_Q(1)/FE_Q(2) with
# set_active_fe_index called BEFORE distribute_dofs, and one where it was never
# called. MatrixFree::reinit(mapping, dof, constraints, hp::QCollection<1>,
# additional_data) returns normally on both; on the mixed one the cell batches
# carry two different active_fe_indices.
#
# The entry's Signal is re-tested and comes out WRONG on 9.8.0-pre:
# n_active_fe_indices() returns 2 for BOTH configurations, because it reports
# shape_info.size(2) — the size of the FECollection — not how many indices the
# mesh uses. The read-back that does distinguish them is
# get_cell_active_fe_index() per cell batch.
#
# The greps confirm the entry's other statement: neither ExcMessage it warns
# about exists anywhere in this tree.
#
# Mutation control: T2_MUTATE=1 makes the run under test the uniform-p one, and
# the fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

echo -n "files_with_all_cells_must_have_same_active_fe_index="
grep -rl "all cells must have same active_fe_index" \
  /home/alexander/dealii/include /home/alexander/dealii/source 2>/dev/null | wc -l
echo -n "files_with_hp_fevalues_requires_hp_mappingcollection="
grep -rl "hp-FEValues requires hp::MappingCollection" \
  /home/alexander/dealii/include /home/alexander/dealii/source 2>/dev/null | wc -l

exec bash "$HERE/../_shared/run.sh" hp_family release matrixfree_hp_support
