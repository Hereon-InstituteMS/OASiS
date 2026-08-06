#!/bin/bash
# Tier-2 for dealii mixed_laplacian#0 -- probes "rt_dof_structure",
# "rt_map_support_points" and "rt_vertex_dof_index" of the shared mixed-Laplacian
# translation unit _shared/mixed_family.cc, compiled once and cached so both
# fixtures of this topic share ONE C++ build.
#
# Where an H(div) dof lives is COUNTED, not asserted: FE_RaviartThomas(0) and (1)
# report n_dofs_per_vertex() == 0 and one resp. two dofs per FACE, while
# FESystem(FE_Q(1), dim) reports two per vertex. A known flux field (x, 2y) is
# interpolated, and the value of the face dof turns out to be EXACTLY the face
# flux integral, to the last bit -- so the dofs are face functionals, and a
# post-processor that gathers dofs at the vertices of a face collects none and
# returns zero.
#
# The two nodal calls a user reaches for are each fatal, so they run in their own
# processes here and the exit codes are the observable:
#   DoFTools::map_dofs_to_support_points  Release 139 (SIGSEGV),
#     Debug 134 with "You are trying to access the support points of a finite
#     element that either has no support points at all, or for which the
#     corresponding tables have not been implemented."
#   cell->vertex_dof_index(0, 0)          139 in BOTH builds -- there is no
#     Assert for this one, so the Debug library does not help.
#
# Neither of the entry's own promises holds: DataOutBase::vertex_data is not an
# enumerator anywhere in this tree (the DataVectorType values are type_dof_data,
# type_cell_data and type_automatic), and DataOut::add_data_vector with
# type_dof_data on an FE_RaviartThomas field does not raise ExcInternalError --
# it returns normally and writes a valid vtu.
#
# Mutation control: T2_MUTATE=1 reads the FACE dof instead of gathering at the
# vertices, the flux integral is reproduced exactly, and the fixture fails its
# own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SHARED="$HERE/../_shared"

echo -n "phantom_vertex_data_enumerator_files="
grep -rl "DataOutBase::vertex_data" /home/alexander/dealii/include \
  /home/alexander/dealii/source 2>/dev/null | wc -l
echo -n "real_datavectortype_enumerators="
grep -c "type_dof_data,\|type_cell_data,\|type_automatic" \
  /home/alexander/dealii/include/deal.II/numerics/data_out_dof_data.h

for spec in "release rt_map_support_points" "debug rt_map_support_points" \
            "release rt_vertex_dof_index" "debug rt_vertex_dof_index"; do
  set -- $spec
  variant="$1"; probe="$2"
  echo "=== variant=$variant probe=$probe"
  out="$(bash "$SHARED/run.sh" mixed_family "$variant" "$probe" 2>&1 \
         | grep -vE '^(/media/|/lib/|/usr/lib|\[0x|#[0-9])')"
  echo "$out"
  rc="$(printf '%s\n' "$out" | sed -n 's/^exit_code=//p' | tail -1)"
  echo "summary_${variant}_${probe}_rc=${rc}"
done

echo "=== probe=rt_dof_structure"
exec bash "$SHARED/run.sh" mixed_family release rt_dof_structure
