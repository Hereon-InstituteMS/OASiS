#!/bin/bash
# Tier-2 for fourc::fsi_xfem#6 -- NA: ALE on an FSI-XFEM fluid block really does
# break the run, but not with the message the entry quoted.
#
# Claimed: aborts with 'fluid element kinematic type incompatible with XFEM'
#          from 4C_fluid_xfem_factory.cpp.
# Observed: neither the string nor the file exists in 4C.  The deck parses, the
#          discretisations are built, the first Newton step starts, and only then
#          FluidBoundaryImpl::evaluate_neumann asks the fluid discretisation for
#          a state it never had: "Cannot find state dispnp in discretization
#          fluid" from 4C_fem_discretization.hpp.  Nothing in that message says
#          XFEM, ALE, or the NA keyword the user actually mis-set.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream xfsi_3D_boxes.4C.yaml) || exit 3
grep -q "        NA: Euler" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/euler.yaml"
sed 's/        NA: Euler/        NA: ALE/' "$BASE" > "$TMP/naale.yaml"

probe EULER "$TMP/euler.yaml"
probe NAALE "$TMP/naale.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/EULER.log"
echo "EULER_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/EULER.log")"
grep -m1 -F "Cannot find state dispnp in discretization fluid" "$TMP/NAALE.log"
grep -m1 -F "4C_fem_discretization.hpp" "$TMP/NAALE.log"
grep -m1 -F "FluidBoundaryImpl" "$TMP/NAALE.log"
# the claimed diagnostic and its claimed source file appear nowhere
echo "CLAIMED_KINEMATIC_TEXT=$(grep -ci 'kinematic type incompatible' "$TMP/NAALE.log")"
echo "CLAIMED_FACTORY_FILE=$(grep -ci '4C_fluid_xfem_factory' "$TMP/NAALE.log")"
# and the message never names what the user actually typed
echo "MESSAGE_NAMES_ALE_KEYWORD=$(grep -c 'Cannot find state dispnp in discretization fluid' "$TMP/NAALE.log")"
exit 0
