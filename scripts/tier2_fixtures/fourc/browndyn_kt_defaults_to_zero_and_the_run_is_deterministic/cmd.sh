#!/bin/bash
# Tier-2 for fourc::brownian_dynamics#1 — KT is the fluctuation magnitude, and
# leaving it out silently turns Brownian dynamics off.
#
# BROWNIAN DYNAMICS/KT has default_value = 0.0 in 4C's input spec, and
# BROWNDYNPROB: true does NOT imply a temperature.  Upstream's periodic-RVE
# filament deck is exactly that configuration — BROWNDYNPROB true, VISCOSITY
# set, no KT — and it is a purely deterministic run: node 2's transverse
# displacements come out at 1.79e-16 and 1.19e-16, machine zero, and the deck
# result-tests them as zero to 1e-8.
#
# Add KT and the thermal kicks appear at once: dispx 1.97729908169594938e-02,
# dispy -1.19525609708355449e-02, and the prescribed axial component is
# disturbed too.  All three result tests fail.
#
# So a deck with BROWNDYNPROB: true and no KT looks like a Brownian simulation,
# runs like one, and contains no thermal physics whatsoever.  4C never says so.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream beam3r_line2_backweuler_browndyn_periodic_rve_dirich_element.4C.yaml) || exit 3
BLK='BROWNIAN DYNAMICS:
  BROWNDYNPROB: true
  VISCOSITY: 0.001
'
grep -qF "$BLK" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
echo "UPSTREAM_DECK_SETS_KT=$(grep -c '^  KT:' "$BASE")"

cp "$BASE" "$TMP/nokt.yaml"
python3 - "$BASE" "$TMP/withkt.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = "BROWNIAN DYNAMICS:\n  BROWNDYNPROB: true\n  VISCOSITY: 0.001\n"
assert blk in t
open(sys.argv[2], "w").write(t.replace(blk, blk + "  KT: 4.14e-06\n"))
PY

probe NOKT   "$TMP/nokt.yaml"
probe WITHKT "$TMP/withkt.yaml"

# BROWNDYNPROB true and no KT: deterministic to machine precision.
grep -m1 -F "processor 0 finished normally" "$TMP/NOKT.log"
echo "NOKT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/NOKT.log")"
grep -m1 -F "dispx    at node   2" "$TMP/NOKT.log"
python3 - "$TMP/NOKT.log" <<'PY'
import re, sys
d = {}
for l in open(sys.argv[1]):
    m = re.search(r"disp([xyz])\s+at node\s+2\s+is CORRECT, abs\(diff\)=\s*([0-9.e+-]+)", l)
    if m:
        d[m.group(1)] = float(m.group(2))
print("NOKT_TRANSVERSE_MOTION_IS_MACHINE_ZERO=%s"
      % ("yes" if d and max(d.get('x', 1), d.get('y', 1)) < 1e-14 else "no"))
PY
# 4C never warns that the thermal forcing is off.
echo "NOKT_WARNINGS=$(grep -ciE '\bKT\b|thermal energy|temperature' "$TMP/NOKT.log")"

# With KT the fluctuations appear.
echo "WITHKT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/WITHKT.log")"
grep -m1 -F "is WRONG --> actresult= 1.97729908169594938e-02" "$TMP/WITHKT.log"
grep -m1 -F "is WRONG --> actresult=-1.19525609708355449e-02" "$TMP/WITHKT.log"
exit 0
