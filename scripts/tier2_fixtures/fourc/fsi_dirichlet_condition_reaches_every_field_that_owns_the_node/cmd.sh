#!/bin/bash
# Tier-2 for fourc::fsi#8 — a DESIGN DIRICH condition really is handed to every
# discretisation that owns one of its nodes, and the DOF-count template really is
# '{} DOFs given but {} expected in {}'.  But the rejection is ASYMMETRIC, so
# only one of the two ways to share a node is caught.
#
# Setup: upstream fsi_fp_mono_fs_ga_ga.4C.yaml keeps its fields on disjoint node
# ids — DNODE 1 = fluid nodes 13..16 with a NUMDOF 4 Dirichlet, DNODE 2 =
# structure nodes 17..20 with a NUMDOF 3 Dirichlet.  Two one-line topology edits
# make a single node belong to a condition written for the other field:
#
#   FLUID_INTO_STRUCT   add "NODE 13 DNODE 2" -> a fluid node (4 dofs) inside the
#                       structure's 3-entry Dirichlet.  ABORTS with
#                         "3 DOFs given but 4 expected in Point Dirichlet
#                          boundary condition"
#                       from core/fem/src/discretization/4C_fem_discretization_
#                       utils_dbc.cpp line 292.  That is proof the structure's
#                       condition was evaluated against the FLUID discretisation.
#   STRUCT_INTO_FLUID   add "NODE 17 DNODE 1" -> a structure node (3 dofs) inside
#                       the fluid's 4-entry Dirichlet.  ACCEPTED: exit 0,
#                       OK (6), no warning.  The extra entry is dropped.
#
# So the failure is loud only when the shared node needs MORE dofs than the
# condition declares.  The other direction quietly imposes a fluid Dirichlet on
# structural dofs.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_fp_mono_fs_ga_ga.4C.yaml) || exit 3
grep -q '  - "NODE 13 DNODE 1"' "$BASE" || { echo "FIXTURE_ABORT=upstream_dnode1_topology_changed"; exit 3; }
grep -q '  - "NODE 17 DNODE 2"' "$BASE" || { echo "FIXTURE_ABORT=upstream_dnode2_topology_changed"; exit 3; }

# The pathology: let one node belong to a Dirichlet written for the other field.
SHARE_A_NODE=yes

cp "$BASE" "$TMP/disjoint.yaml"
python3 - "$BASE" "$TMP" "$SHARE_A_NODE" <<'PY'
import sys
src, tmp, do = sys.argv[1:4]
t = open(src).read()
a, b = '  - "NODE 13 DNODE 1"\n', '  - "NODE 17 DNODE 2"\n'
open(tmp + "/fluid_into_struct.yaml", "w").write(
    t.replace(a, a + '  - "NODE 13 DNODE 2"\n', 1) if do == "yes" else t)
open(tmp + "/struct_into_fluid.yaml", "w").write(
    t.replace(b, b + '  - "NODE 17 DNODE 1"\n', 1) if do == "yes" else t)
PY
echo "FLUIDINTOSTRUCT_SHARED_ROWS=$(grep -c '"NODE 13 DNODE 2"' "$TMP/fluid_into_struct.yaml")"
echo "STRUCTINTOFLUID_SHARED_ROWS=$(grep -c '"NODE 17 DNODE 1"' "$TMP/struct_into_fluid.yaml")"

probe DISJOINT          "$TMP/disjoint.yaml"
probe FLUIDINTOSTRUCT   "$TMP/fluid_into_struct.yaml"
probe STRUCTINTOFLUID   "$TMP/struct_into_fluid.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/DISJOINT.log"
grep -m1 -F "OK (6)" "$TMP/DISJOINT.log"

# The structure's own condition is checked against the fluid's DOF count.
grep -m1 -F "3 DOFs given but 4 expected in Point Dirichlet boundary condition" "$TMP/FLUIDINTOSTRUCT.log"
grep -m1 -F "4C_fem_discretization_utils_dbc.cpp" "$TMP/FLUIDINTOSTRUCT.log"
grep -m1 -F "read_dirichlet_condition" "$TMP/FLUIDINTOSTRUCT.log"
echo "FLUIDINTOSTRUCT_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/FLUIDINTOSTRUCT.log")"

# The mirror image is not caught at all.
grep -m1 -F "OK (6)" "$TMP/STRUCTINTOFLUID.log"
echo "STRUCTINTOFLUID_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/STRUCTINTOFLUID.log")"
echo "STRUCTINTOFLUID_ANY_DOF_COMPLAINT=$(grep -ci 'DOFs given but' "$TMP/STRUCTINTOFLUID.log")"

# The retired wording is in neither arm.
echo "CLAIMED_NUMDOF_MISMATCH_TEXT=$(cat "$TMP"/FLUIDINTOSTRUCT.log "$TMP"/STRUCTINTOFLUID.log \
      | grep -ciE 'NUMDOF mismatch|4C_dofset.cpp')"
exit 0
