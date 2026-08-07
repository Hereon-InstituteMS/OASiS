#!/bin/bash
# Tier-2 for fourc::particle_dem#10 — a contact parameter belonging to a law you
# are not using is read, accepted and ignored, with no unused-parameter warning.
#
# Claimed: COEFF_RESTITUTION on the plain NormalLinearSpring is inert; the deck
#          runs bit-identically with and without it.
#
# Measured as an IDENTITY on the verdict lines, which is the only thing that
# settles inertness. A positive control is included so the identity is known to
# be informative: NORMAL_STIFF, a parameter the law DOES use, moves the same
# lines under the same comparison.
#
# T2_MUTATE=1 removes the pathology: COEFF_RESTITUTION is not added at all, so
# the two decks compared are literally the same file, the identity holds for a
# trivial reason and IDENTITY_IS_INFORMATIVE flips to no.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

BASE=$(upstream particle_dem_1d_normalcontact_linspring_stiffset.4C.yaml) || exit 3
grep -q 'NORMALCONTACTLAW' "$BASE" && { echo "FIXTURE_ABORT=deck_no_longer_uses_the_default_law"; exit 3; }
grep -q "NORMAL_STIFF: 3.5e-05" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$BASE" "$TMP" "$MUTATE" <<'PY'
import sys
src, tmp, mutate = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3] == "1"
k = "  NORMAL_STIFF: 3.5e-05"
# the claim: adding a damping-law parameter to a law with no damping
open(tmp + "/withrest.yaml", "w").write(
    src if mutate else src.replace(k, k + "\n  COEFF_RESTITUTION: 0.8"))
open(tmp + "/plain.yaml", "w").write(src)
# positive control: a parameter this law DOES use
open(tmp + "/stiffer.yaml", "w").write(src.replace(k, "  NORMAL_STIFF: 7.0e-05"))
PY

probe PLAIN    "$TMP/plain.yaml"
probe WITHREST "$TMP/withrest.yaml"
probe STIFFER  "$TMP/stiffer.yaml"

# GUARD, and it is the part the mutation actually kills. The equality below is
# satisfied a fortiori by two identical decks, so on its own it can neither be
# killed by a mutation nor detect an edit that silently failed to apply. Check
# that the deck under test really carries the key before believing the identity.
echo "PATHOLOGY_PRESENT=$(grep -c 'COEFF_RESTITUTION' "$TMP/withrest.yaml" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
echo "CONTROL_DECK_IS_CLEAN=$(grep -c 'COEFF_RESTITUTION' "$TMP/plain.yaml" | sed 's/^0$/yes/;s/^[1-9].*/no/')"

verdicts() { grep -E "is CORRECT|is WRONG" "$1"; }
verdicts "$TMP/PLAIN.log"    > "$TMP/v_plain"
verdicts "$TMP/WITHREST.log" > "$TMP/v_rest"
verdicts "$TMP/STIFFER.log"  > "$TMP/v_stiff"

echo "VERDICT_LINES=$(wc -l < "$TMP/v_plain")"
cmp -s "$TMP/v_plain" "$TMP/v_rest"  && echo "RESTITUTION_IS_INERT=yes" || echo "RESTITUTION_IS_INERT=no"
cmp -s "$TMP/v_plain" "$TMP/v_stiff" && echo "IDENTITY_IS_INFORMATIVE=no" || echo "IDENTITY_IS_INFORMATIVE=yes"
# No unused-parameter warning anywhere in the DEM stack.
echo "UNUSED_PARAM_WARNINGS=$(grep -ciE 'unused|ignored|has no effect|not used' "$TMP/WITHREST.log")"
grep -m1 -E "^OK \(" "$TMP/WITHREST.log" && echo "WITHREST_STILL_PASSES=yes"
exit 0
