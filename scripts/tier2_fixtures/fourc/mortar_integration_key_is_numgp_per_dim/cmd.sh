#!/bin/bash
# Tier-2 for fourc::constraint#2 — mortar coupling does need its interface
# integrals evaluated, and 4C refuses to guess how many Gauss points to use. The
# entry named the wrong key.
#
# Claimed:  "Use INTPOINTS_MORTAR appropriate to element order."
# Observed: there is no INTPOINTS_MORTAR anywhere in 4C; writing it is a parse
#           error. The key is NUMGP_PER_DIM in the MORTAR COUPLING section, and
#           it only applies when INTTYPE is Elements or Elements_BS.
#
# Upstream deck: meshtying3D_elebased — a 3D mortar meshtying problem with
# INTTYPE Elements, NUMGP_PER_DIM 2 and a result test at 1e-08. Five arms:
#
#   as shipped        -> converges, result test passes
#   NUMGP_PER_DIM cut -> the default is 0, and 0 is rejected outright
#   NUMGP_PER_DIM 1   -> also rejected; element-based integration needs > 1
#   NUMGP_PER_DIM 7   -> fine
#   INTTYPE Segments  -> the same number is now read as a triangle rule and is
#                        not one, so the run dies deeper in, in the integrator
#
# The last arm is the one worth remembering: NUMGP_PER_DIM is not ignored when
# you switch integration type, it is reinterpreted.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream meshtying3D_elebased.4C.yaml) || exit 3
grep -q "  NUMGP_PER_DIM: 2" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '  INTTYPE: "Elements"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/shipped.yaml"
sed '/  NUMGP_PER_DIM: 2/d'                          "$BASE" > "$TMP/nogp.yaml"
sed 's/  NUMGP_PER_DIM: 2/  NUMGP_PER_DIM: 1/'       "$BASE" > "$TMP/gp1.yaml"
sed 's/  NUMGP_PER_DIM: 2/  NUMGP_PER_DIM: 7/'       "$BASE" > "$TMP/gp7.yaml"
sed 's/  INTTYPE: "Elements"/  INTTYPE: "Segments"/' "$BASE" > "$TMP/segments.yaml"
sed 's/  NUMGP_PER_DIM: 2/  INTPOINTS_MORTAR: 2/'    "$BASE" > "$TMP/claimedkey.yaml"

probe SHIPPED    "$TMP/shipped.yaml"
probe NOGP       "$TMP/nogp.yaml"
probe GP1        "$TMP/gp1.yaml"
probe GP7        "$TMP/gp7.yaml"
probe SEGMENTS   "$TMP/segments.yaml"
probe CLAIMEDKEY "$TMP/claimedkey.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/SHIPPED.log"
echo "SHIPPED_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/SHIPPED.log")"
echo "GP7_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/GP7.log")"

# Omitting the key leaves it at its default of 0, which is refused.
grep -m1 -F "Invalid Gauss point number NUMGP_PER_DIM for element-based integration." "$TMP/NOGP.log"
grep -m1 -F "4C_contact_meshtying_strategy_factory.cpp" "$TMP/NOGP.log"
# One point per dimension is refused too.
grep -m1 -F "Invalid Gauss point number NUMGP_PER_DIM for element-based integration." "$TMP/GP1.log"
# Switching INTTYPE reinterprets the same number as a triangle rule.
grep -m1 -F "unknown tri gauss rule" "$TMP/SEGMENTS.log"
grep -m1 -F "4C_mortar_integrator.cpp" "$TMP/SEGMENTS.log"
# And the key the entry named is not a key.
grep -m1 -F "Could not match this input" "$TMP/CLAIMEDKEY.log"

python3 - "$TMP/NOGP.log" "$TMP/CLAIMEDKEY.log" <<'PY'
import sys
n = 0
for p in sys.argv[1:]:
    t = open(p, "rb").read().decode("utf-8", "replace").lower()
    n += t.count("intpoints_mortar is") + t.count("did you mean")
print("PARSER_SUGGESTS_THE_RIGHT_KEY=%d" % n)
PY
exit 0
