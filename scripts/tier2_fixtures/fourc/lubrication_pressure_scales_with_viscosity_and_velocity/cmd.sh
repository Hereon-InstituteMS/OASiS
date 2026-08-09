#!/bin/bash
# Tier-2 for fourc::lubrication#5 — the two unit slips the entry names, measured.
#
# Reynolds gives p ~ 6 mu U L / h^2, so on upstream lubrication_sb_2d.4C.yaml:
#   * viscosity 5e-07 -> 5e-06 (the poise-vs-Pa.s factor of 10) multiplies the
#     node-10 pressure by EXACTLY 10.  The entry's "wrong by exactly 10x" is
#     confirmed to 15 digits.
#   * surface velocity 20000 -> 20 (an m/s-vs-mm/s slip of 1000) divides it by
#     EXACTLY 1000.
# and 4C accepts both without a word, because nothing in the input carries units.
#
# The entry also claimed "h in mm vs m gives factor 1e9 off".  That exponent is
# wrong: pressure goes as 1/h^2, not 1/h^3 — the h^3 sits inside the divergence
# and one power is spent on the two gradients — so a 1000-fold slip in h moves
# the pressure by 1e6, measured here as the third arm.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream lubrication_sb_2d.4C.yaml) || exit 3
grep -q 'VISCOSITY: 5e-07' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_viscosity_changed"; exit 3; }
grep -q 'SYMBOLIC_FUNCTION_OF_SPACE_TIME: "20000"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_velocity_function_changed"; exit 3; }
grep -q '"(0.045)-(x\*0.5e-3)"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_height_function_changed"; exit 3; }

# The three unit slips.
BAD_VISCOSITY='5e-06'
BAD_VELOCITY='20.0'
BAD_HEIGHT='(45.0)-(x*0.5)'

REF=73.94356207110268
cp "$BASE" "$TMP/si.yaml"
sed "s/VISCOSITY: 5e-07/VISCOSITY: $BAD_VISCOSITY/" "$BASE" > "$TMP/poise.yaml"
sed "s/SYMBOLIC_FUNCTION_OF_SPACE_TIME: \"20000\"/SYMBOLIC_FUNCTION_OF_SPACE_TIME: \"$BAD_VELOCITY\"/" \
    "$BASE" > "$TMP/mmps.yaml"
sed "s|(0.045)-(x\*0.5e-3)|$BAD_HEIGHT|" "$BASE" > "$TMP/hmm.yaml"

probe SI    "$TMP/si.yaml"
probe POISE "$TMP/poise.yaml"
probe MMPS  "$TMP/mmps.yaml"
probe HMM   "$TMP/hmm.yaml"

grep -m1 -F "is CORRECT, abs(diff)=" "$TMP/SI.log"
grep -m1 -F "processor 0 finished normally" "$TMP/SI.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/POISE.log"

val() { grep -m1 -oE 'actresult=[ ]*[-0-9.eE+]+' "$1" | tr -d ' ' | cut -d= -f2; }
python3 - "$(val "$TMP/POISE.log")" "$(val "$TMP/MMPS.log")" "$(val "$TMP/HMM.log")" "$REF" <<'PY'
import sys
p, v, h, ref = (float(x) for x in sys.argv[1:5])
print("VISCOSITY_x10_PRESSURE_RATIO=%.6f" % (p / ref))
print("VELOCITY_div1000_PRESSURE_RATIO=%.4e" % (v / ref))
print("HEIGHT_x1000_PRESSURE_RATIO=%.4e" % (h / ref))
print("PRESSURE_IS_EXACTLY_LINEAR_IN_VISCOSITY=%s"
      % ("yes" if abs(p / ref - 10.0) < 1e-12 else "no"))
print("HEIGHT_EXPONENT_IS_MINUS_TWO=%s"
      % ("yes" if abs(h / ref - 1e-6) < 1e-12 else "no"))
PY
# No unit or scaling diagnostic of any kind.
echo "UNIT_WARNINGS=$(grep -ciE 'unit|dimension|scal(e|ing) (error|mismatch)' "$TMP/POISE.log")"
exit 0
