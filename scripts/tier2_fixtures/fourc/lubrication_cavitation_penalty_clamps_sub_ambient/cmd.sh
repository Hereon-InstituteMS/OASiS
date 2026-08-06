#!/bin/bash
# Tier-2 for fourc::lubrication#3 — a Reynolds solve DOES return sub-ambient
# pressure in a diverging gap, and 4C's remedy is a key that already exists:
# LUBRICATION DYNAMIC/PENALTY_CAVITATION (default 0, i.e. OFF).  Not Elrod-Adams,
# not post-processing.
#
# Upstream lubrication_sb_2d slides the surface in +x through a converging gap
# and builds +73.94 Pa at node 10, with PENALTY_CAVITATION: 1e+08 set but inert
# (the pressure never goes negative, so the penalty never activates — proved here
# by the NOPEN arm, which turns the penalty off and reproduces the reference
# result to 1e-10).
#
# Reverse the surface velocity and the same gap is now DIVERGING:
#   PENALTY_CAVITATION: 0     -> p = -73.94, exactly minus the baseline.
#                                A liquid film cannot sustain that.
#   PENALTY_CAVITATION: 1e+08 -> p = -9.4e-07, driven to ambient.
# So: the penalty is what stops it, it is off by default, and with it off you get
# a large negative pressure with no diagnostic whatsoever.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream lubrication_sb_2d.4C.yaml) || exit 3
grep -q 'PENALTY_CAVITATION: 1e+08' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_penalty_key_changed"; exit 3; }
grep -q 'SYMBOLIC_FUNCTION_OF_SPACE_TIME: "20000"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_velocity_function_changed"; exit 3; }

# The pathology: the sliding direction that makes the gap diverge.
BAD_VELOCITY='-20000'

# converging gap, penalty on (the upstream reference) and penalty off
cp "$BASE" "$TMP/conv_pen.yaml"
sed 's/PENALTY_CAVITATION: 1e+08/PENALTY_CAVITATION: 0.0/' "$BASE" > "$TMP/conv_nopen.yaml"
# diverging gap, penalty off and penalty on
sed "s/SYMBOLIC_FUNCTION_OF_SPACE_TIME: \"20000\"/SYMBOLIC_FUNCTION_OF_SPACE_TIME: \"$BAD_VELOCITY\"/" \
    "$BASE" > "$TMP/div_pen.yaml"
sed 's/PENALTY_CAVITATION: 1e+08/PENALTY_CAVITATION: 0.0/' "$TMP/div_pen.yaml" > "$TMP/div_nopen.yaml"

probe CONV_PEN   "$TMP/conv_pen.yaml"
probe CONV_NOPEN "$TMP/conv_nopen.yaml"
probe DIV_NOPEN  "$TMP/div_nopen.yaml"
probe DIV_PEN    "$TMP/div_pen.yaml"

grep -m1 -F "is CORRECT, abs(diff)=" "$TMP/CONV_PEN.log"
grep -m1 -F "processor 0 finished normally" "$TMP/CONV_NOPEN.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/DIV_NOPEN.log"

val() { grep -m1 -oE 'actresult=[ ]*[-0-9.eE+]+' "$1" | tr -d ' ' | cut -d= -f2; }
PN=$(val "$TMP/DIV_NOPEN.log")
PP=$(val "$TMP/DIV_PEN.log")
echo "DIVERGING_GAP_PRESSURE_PENALTY_OFF=$PN"
echo "DIVERGING_GAP_PRESSURE_PENALTY_ON=$PP"
python3 - "$PN" "$PP" <<'PY'
import sys
pn, pp = float(sys.argv[1]), float(sys.argv[2])
print("SUB_AMBIENT_WITHOUT_PENALTY=%s" % ("yes" if pn < -1.0 else "no"))
print("SUB_AMBIENT_WITH_PENALTY=%s" % ("yes" if pp < -1.0 else "no"))
print("PENALTY_SUPPRESSION_FACTOR=%.1e" % (abs(pp) / abs(pn)))
PY
# 4C never mentions cavitation, vaporisation or a negative pressure.
echo "CAVITATION_WARNINGS_PENALTY_OFF=$(grep -ciE 'cavitat|vapor|negative pressure|sub-ambient' "$TMP/DIV_NOPEN.log")"
# And 'Elrod' / 'Elrod-Adams' is not a thing in 4C's output.
echo "CLAIMED_ELROD_ADAMS=$(grep -ci 'elrod' "$TMP/DIV_NOPEN.log")"
exit 0
