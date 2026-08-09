#!/bin/bash
# Tier-2 for fourc::particle_pd#9 — both halves of the rule hold, and both of
# the failures are worse than advertised.
#
# Claimed:  writing (x, y) with no z "raises a parser error 'expected 3
#           coordinates'"; setting the z bounds of DOMAINBOUNDINGBOX to 0
#           "produces a degenerate bounding box that conflicts with binning".
# Observed: (a) POS takes a fixed-size 3-vector, so dropping z makes the parser
#           swallow the NEXT token as the missing component and then fail on it:
#             Could not parse 'PDBODYID' as a double value.
#           from core/io/src/4C_io_value_parser.cpp.  The message blames the
#           optional state that follows POS, not the coordinate count, and there
#           is no 'expected 3 coordinates' string in 4C.
#           (b) a zero z-extent does not "conflict with binning" politely — 4C
#           divides the domain length by the bin size and takes a
#             Floating point exception (8) / divide-by-zero (3)
#           in ParticleEngine::init_binning_strategy.  There is no PROC 0 ERROR
#           block, no message and no MPI_Abort banner: the process dies with a
#           core dump, and the exit status is a signal, not 1.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3
FIRST='  - "TYPE pdphase POS -4.50E+00 -4.50E+00 0.000000000000000e+00 PDBODYID 0"'
grep -qF "$FIRST" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -qF 'DOMAINBOUNDINGBOX: "-20.0 -20.0 -0.01 20.0 20.0 0.01"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/thinslab.yaml"
sed 's|POS -4.50E+00 -4.50E+00 0.000000000000000e+00 PDBODYID 0|POS -4.50E+00 -4.50E+00 PDBODYID 0|' "$BASE" > "$TMP/twocoord.yaml"
sed 's|DOMAINBOUNDINGBOX: "-20.0 -20.0 -0.01 20.0 20.0 0.01"|DOMAINBOUNDINGBOX: "-20.0 -20.0 0.0 20.0 20.0 0.0"|' "$BASE" > "$TMP/flatbox.yaml"

probe THINSLAB "$TMP/thinslab.yaml"
probe TWOCOORD "$TMP/twocoord.yaml"
# the flat-box arm dies on a signal, so keep its raw status
run4c "$TMP/flatbox.yaml" "$TMP/o_FLATBOX" > "$TMP/FLATBOX.log" 2>&1
FLATBOX_STATUS=$?
echo "EXIT_FLATBOX=$FLATBOX_STATUS"

grep -m1 -F "processor 0 finished normally" "$TMP/THINSLAB.log"
echo "THINSLAB_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/THINSLAB.log")"

# (a) the two-coordinate case blames the wrong token.
grep -m1 -F "Could not parse 'PDBODYID' as a double value." "$TMP/TWOCOORD.log"
grep -m1 -oE "4C_io_value_parser\.cpp, line [0-9]+" "$TMP/TWOCOORD.log"
echo "CLAIMED_EXPECTED_3_COORDINATES_TEXT=$(grep -ci 'expected 3 coordinates' "$TMP/TWOCOORD.log")"
if grep -qiE "coordinate|POS" "$TMP/TWOCOORD.log"; then
  echo "TWOCOORD_DIAGNOSTIC_MENTIONS_COORDINATES=yes"
else
  echo "TWOCOORD_DIAGNOSTIC_MENTIONS_COORDINATES=no"
fi

# (b) the flat box does not report anything: it takes a signal.
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/FLATBOX.log"
grep -m1 -F "Floating point divide-by-zero (3)" "$TMP/FLATBOX.log"
grep -m1 -oE "init_binning_strategy" "$TMP/FLATBOX.log"
echo "FLATBOX_HAS_PROC_ERROR_BLOCK=$(grep -c 'PROC 0 ERROR' "$TMP/FLATBOX.log")"
if [ "$FLATBOX_STATUS" -gt 128 ]; then
  echo "FLATBOX_DIED_ON_A_SIGNAL=yes"
else
  echo "FLATBOX_DIED_ON_A_SIGNAL=no"
fi
exit 0
