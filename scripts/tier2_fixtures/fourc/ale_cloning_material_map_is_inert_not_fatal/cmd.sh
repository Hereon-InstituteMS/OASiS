#!/bin/bash
# Tier-2 for fourc::ale#4 — and a FALSIFICATION of the claim.
#
# Claimed: "using CLONING MATERIAL MAP in a stand-alone ALE input has no fluid
#          field to clone from — setup fails with 'source field not found'".
# Observed: it does not fail. The section is INERT in a single-field problem.
#          Adding a fluid->ale map to the upstream ale2d_solid deck changes
#          nothing: exit 0, the result test still passes to 1.4e-17, and the word
#          "clone" never appears in the output. Worse, a map naming fields that do
#          not exist ("frobnicate" -> "nonexistent") and material IDs that are not
#          defined (99, 42) is ALSO accepted silently and the run still completes.
#
# So the corrected claim is the opposite failure mode: a stray or wrong CLONING
# MATERIAL MAP gives you no diagnostic at all, which means it cannot be used as
# evidence that field cloning is wired up. There is no 'source field not found'
# string in 4C.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ale2d_solid.4C.yaml) || exit 3
grep -q "^MATERIALS:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$BASE" "$TMP/plausible.yaml" "$TMP/bogus.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
open(sys.argv[2], "w").write(t.replace("MATERIALS:", """CLONING MATERIAL MAP:
  - SRC_FIELD: "fluid"
    SRC_MAT: 1
    TAR_FIELD: "ale"
    TAR_MAT: 1
MATERIALS:""", 1))
open(sys.argv[3], "w").write(t.replace("MATERIALS:", """CLONING MATERIAL MAP:
  - SRC_FIELD: "frobnicate"
    SRC_MAT: 99
    TAR_FIELD: "nonexistent"
    TAR_MAT: 42
MATERIALS:""", 1))
PY

cp "$BASE" "$TMP/nomap.yaml"
probe NOMAP     "$TMP/nomap.yaml"
probe PLAUSIBLE "$TMP/plausible.yaml"
probe BOGUS     "$TMP/bogus.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/BOGUS.log"
grep -m1 -F "is CORRECT, abs(diff)=" "$TMP/BOGUS.log"
# Nothing at all is said about cloning, in either arm.
echo "PLAUSIBLE_MENTIONS_CLONE=$(grep -ci clon "$TMP/PLAUSIBLE.log")"
echo "BOGUS_MENTIONS_CLONE=$(grep -ci clon "$TMP/BOGUS.log")"
# The undefined field names and material IDs are never echoed or rejected.
echo "BOGUS_MENTIONS_FROBNICATE=$(grep -ci frobnicate "$TMP/BOGUS.log")"
echo "CLAIMED_SOURCE_FIELD_NOT_FOUND_TEXT=$(grep -ci 'source field not found' "$TMP/BOGUS.log")"
# ...and the answer is bit-identical to the run without any map.
a=$(grep -m1 -o 'dispx.*abs(diff)= [0-9.e+-]*' "$TMP/NOMAP.log")
b=$(grep -m1 -o 'dispx.*abs(diff)= [0-9.e+-]*' "$TMP/BOGUS.log")
[ "$a" = "$b" ] && echo "BOGUS_MAP_CHANGES_ANSWER=no" || echo "BOGUS_MAP_CHANGES_ANSWER=yes"
exit 0
