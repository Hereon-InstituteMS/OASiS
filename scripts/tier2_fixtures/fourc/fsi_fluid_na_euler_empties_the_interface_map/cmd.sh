#!/bin/bash
# Tier-2 for fourc::fsi#1 — FSI fluid elements really do need `NA ALE`, but
# neither half of the claimed Signal happens.
#
# Claimed: "leaving NA: Euler triggers 'fluid element type incompatible with ALE
#           mesh motion' at setup, OR (worse) the simulation runs but the fluid
#           mesh does NOT move with the structure — interface velocities
#           mismatch and Newton diverges within ~10 steps."
# Observed: neither.  The string does not exist in 4C, there is no
#           "runs-but-drifts" mode, and NOTHING mentions the element or the NA
#           keyword.  Swapping `NA ALE` for `NA Euler` in the FLUID ELEMENTS
#           lines of upstream fsi_fp_mono_fs_ga_ga.4C.yaml aborts before time
#           step 1 with
#             "got 4 master nodes but 0 slave nodes for coupling"
#           from coupling/src/adapter/4C_coupling_adapter.cpp line 69, raised by
#           FSI::Monolithic::setup_system while wiring the structure->fluid
#           condition coupling.  The 4 master nodes are the structure side of
#           the FSI interface; the 0 slave nodes are the fluid side, whose
#           fsi_cond_map is empty because a non-ALE fluid field has no FSI
#           interface at all.  A reader has to know all of that to get back to
#           the two-character edit that caused it.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_fp_mono_fs_ga_ga.4C.yaml) || exit 3
grep -q 'FLUID HEX8 .* MAT 2 NA ALE' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_no_longer_declares_NA_ALE"; exit 3; }

# The pathology: declare the FSI fluid elements as Eulerian.
FLUID_NA=Euler

cp "$BASE" "$TMP/ale.yaml"
sed "s/ NA ALE\"/ NA $FLUID_NA\"/" "$BASE" > "$TMP/euler.yaml"
echo "EULER_ARM_FLUID_ELEMENTS_WITH_NA_ALE=$(grep -c 'NA ALE"' "$TMP/euler.yaml")"
echo "EULER_ARM_FLUID_ELEMENTS_WITH_NA_EULER=$(grep -c "NA $FLUID_NA\"" "$TMP/euler.yaml")"

probe ALE   "$TMP/ale.yaml"
probe EULER "$TMP/euler.yaml"

# Control: the unmodified deck runs and matches all six pinned results.
grep -m1 -F "processor 0 finished normally" "$TMP/ALE.log"
grep -m1 -F "OK (6)" "$TMP/ALE.log"

# The real diagnostic, and where it comes from.
grep -m1 -F "got 4 master nodes but 0 slave nodes for coupling" "$TMP/EULER.log"
grep -m1 -F "4C_coupling_adapter.cpp"                           "$TMP/EULER.log"
grep -m1 -F "FSI::Monolithic::setup_system"                     "$TMP/EULER.log"

# It aborts at setup, so there is no "runs but the mesh does not move" mode and
# no Newton divergence to watch for.
echo "EULER_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/EULER.log")"
echo "EULER_NEWTON_STEPS=$(grep -c 'Nonlinear Solver Step' "$TMP/EULER.log")"
# And nothing in the log points at the element declaration.
echo "EULER_CLAIMED_TEXT=$(grep -ciE 'incompatible with ALE mesh motion' "$TMP/EULER.log")"
echo "EULER_MENTIONS_NA_KEYWORD=$(grep -ciE '\bNA\b.*(ALE|Euler)' "$TMP/EULER.log")"
exit 0
