#!/bin/bash
# Tier-2 for fourc::thermal#8 — in a coupled TSI run the thermal field is
# SOLVED, not prescribed.  Pinning the temperature on every node with a
# Dirichlet condition destroys the structure->thermo feedback, and 4C does not
# complain: the coupling iteration converges, the run finishes, and it has
# answered a different question.
#
# Upstream tsi_lincompression_iterstaggdisp (two-way, tsi_iterstagg) is the
# baseline: mechanical work heats the bar from its initial 300 to a converged
# 347.234 at all three tested nodes, and the deck's own result tests pass.
# The mutant adds ONE section — DESIGN VOL THERMO DIRICH CONDITIONS at 300 on
# the DVOL that already covers all 12 nodes — and nothing else.  The
# temperature then reads exactly 3.00000000000000000e+02 everywhere: identical
# to the prescribed value at every step, which is the detector.  The
# displacement moves too, because the feedback it lost was real.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream tsi_lincompression_iterstaggdisp.4C.yaml) || exit 3
grep -q "^DVOL-NODE TOPOLOGY:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'COUPALGO: "tsi_iterstagg"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
cp "$BASE" "$TMP/free.yaml"

python3 - "$BASE" "$TMP/pinned.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
anchor = "DVOL-NODE TOPOLOGY:"
if t.count(anchor) != 1:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
blk = """DESIGN VOL THERMO DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [300.0]
    FUNCT: [0]
DVOL-NODE TOPOLOGY:"""
open(sys.argv[2], "w").write(t.replace(anchor, blk, 1))
PY
[ -s "$TMP/pinned.yaml" ] || exit 3

probe FREE   "$TMP/free.yaml"
probe PINNED "$TMP/pinned.yaml"

# The solved field: mechanical heating carries the bar to 347.234.
grep -m1 -F "temp    (T(x=0)) at node   1	 is CORRECT" "$TMP/FREE.log"
grep -m1 -F "processor 0 finished normally" "$TMP/FREE.log"
# The pinned field is exactly what was prescribed, at every tested node.
echo "PINNED_NODES_AT_PRESCRIBED_300=$(grep -c 'temp.*is WRONG --> actresult= 3.00000000000000000e+02' "$TMP/PINNED.log")"
grep -m1 -F "temp    (T(x=0)) at node   1	 is WRONG --> actresult= 3.00000000000000000e+02, givenresult= 3.47234354640127151e+02" "$TMP/PINNED.log"
# Losing the feedback also moves the structure.
grep -m1 -F "dispx   (ux(x=4)) at node   9	 is WRONG --> actresult=-7.61904761904762196e-01" "$TMP/PINNED.log"
# And 4C never says a word: the coupling loop converges just as happily.
echo "PINNED_CONVERGENCE_COMPLAINTS=$(grep -ciE 'not converged|unconverged' "$TMP/PINNED.log")"
echo "FREE_CONVERGENCE_COMPLAINTS=$(grep -ciE 'not converged|unconverged' "$TMP/FREE.log")"
exit 0
