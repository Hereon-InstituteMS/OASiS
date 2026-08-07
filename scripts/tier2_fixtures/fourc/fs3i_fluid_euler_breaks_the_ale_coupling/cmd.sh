#!/bin/bash
# Tier-2 for fourc::fs3i#4 — FS3I fluid elements really do need `NA ALE`, but the
# failure is not the kinematic-type message the entry quoted.
#
# Claimed:  "setting NA: Euler on FS3I FLUID3 elements raises 'fluid kinematic
#            type incompatible with moving-mesh scalar transport' at setup".
# Observed: on upstream fs3i_part_1wc_infperm.4C.yaml, rewriting every FLUID
#           element's `NA ALE` to `NA Euler` aborts in
#           coupling/src/adapter/4C_coupling_adapter.cpp line 69 with
#             "got 42 master nodes but 0 slave nodes for coupling"
#           — the ALE discretisation is simply never built from Eulerian fluid
#           elements, so the FSI interface has nothing to couple to.  Nothing is
#           said about kinematics, scalar transport or a moving mesh, and the
#           number in the message is a node count on the FSI interface.
#
# Note that the deck writes the flag as an element-line token (`NA ALE`), not as
# a YAML key `NA: Euler` — that spelling in the entry does not correspond to
# anything in the deck format either.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fs3i_part_1wc_infperm.4C.yaml) || exit 3
grep -q 'MAT 1 NA ALE"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_fluid_element_flag_changed"; exit 3; }
cp -r "$(dirname "$BASE")/xml" "$TMP/" \
  || { echo "FIXTURE_ABORT=missing_teko_xml"; exit 3; }
cd "$TMP" || exit 3

# The pathology: the kinematic flag on every fluid element.
FLUID_KINEMATICS=Euler

cp "$BASE" "$TMP/ale.yaml"
sed "s/ MAT 1 NA ALE\"/ MAT 1 NA $FLUID_KINEMATICS\"/" "$BASE" > "$TMP/euler.yaml"
echo "EULER_ARM_ALE_ELEMENTS=$(grep -c 'MAT 1 NA ALE"' "$TMP/euler.yaml")"
echo "EULER_ARM_EULER_ELEMENTS=$(grep -c 'MAT 1 NA Euler"' "$TMP/euler.yaml")"

probe ALE   "$TMP/ale.yaml"
probe EULER "$TMP/euler.yaml"

grep -m1 -F "OK (3)" "$TMP/ALE.log"
grep -m1 -F "processor 0 finished normally" "$TMP/ALE.log"
grep -m1 -F "got 42 master nodes but 0 slave nodes for coupling" "$TMP/EULER.log"
grep -m1 -F "4C_coupling_adapter.cpp" "$TMP/EULER.log"

# The quoted diagnostic does not exist, and nothing mentions kinematics at all.
echo "CLAIMED_KINEMATIC_TYPE_TEXT=$(grep -ci 'kinematic type incompatible' "$TMP/EULER.log")"
echo "EULER_KINEMATICS_MENTIONS=$(grep -ciE 'kinematic|moving.?mesh' "$TMP/EULER.log")"
# It aborts at coupling setup, before any time step.
echo "EULER_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/EULER.log")"
echo "EULER_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/EULER.log")"
exit 0
