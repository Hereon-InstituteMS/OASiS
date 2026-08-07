#!/bin/bash
# Tier-2 for fourc::xfem_fluid#0 -- an ALE DYNAMIC section in a Fluid_XFEM deck
# is neither honoured nor rejected: it is read, ignored, and never mentioned.
#
# Claimed: parser warns `ALE DYNAMIC ignored under XFEM` or runtime
#          `incompatible ALE+XFEM combination`.
# Observed: neither string exists.  The deck with ALE DYNAMIC bolted on runs to
#          "processor 0 finished normally" and reproduces every result test of
#          the untouched deck.  What DOES break XFEM is asking the fluid
#          ELEMENTS for ALE kinematics (NA: ALE) -- and even then the message is
#          about a missing state vector, not about ALE and XFEM being
#          incompatible.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream xfluid_ls_neumann_inflow_stab.4C.yaml) || exit 3
grep -q "NA: Euler" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "^XFEM GENERAL:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/base.yaml"
python3 - "$BASE" "$TMP/aledyn.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
t = t.replace("XFEM GENERAL:", """ALE DYNAMIC:
  TIMESTEP: 0.1
  NUMSTEP: 3
  MAXTIME: 0.3
  LINEAR_SOLVER: 1
XFEM GENERAL:""", 1)
open(sys.argv[2], "w").write(t)
PY
sed 's/NA: Euler/NA: ALE/' "$BASE" > "$TMP/naale.yaml"

probe BASE   "$TMP/base.yaml"
probe ALEDYN "$TMP/aledyn.yaml"
probe NAALE  "$TMP/naale.yaml"

# the mutant deck really does carry the section 4C is supposed to object to
echo "ALEDYN_HAS_ALE_SECTION=$(grep -c '^ALE DYNAMIC:' "$TMP/aledyn.yaml")"
# ...and 4C ran it to completion with every result test still matching
grep -m1 -F "processor 0 finished normally" "$TMP/ALEDYN.log"
echo "ALEDYN_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/ALEDYN.log")"
# ...and said nothing at all about ALE
echo "CLAIMED_ALE_WARNING=$(grep -ciE 'ALE DYNAMIC ignored under XFEM|incompatible ALE\+XFEM' "$TMP/ALEDYN.log")"
# The element-level ALE request is what actually fails, with an unrelated message.
grep -m1 -F "Cannot find state dispnp in discretization fluid" "$TMP/NAALE.log"
grep -m1 -F "4C_fem_discretization.hpp" "$TMP/NAALE.log"
exit 0
